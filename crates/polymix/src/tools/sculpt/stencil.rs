//! Brush alphas and screen stencils for the sculpt brushes: the built-in patterns and grey
//! images (PNG or JPEG). The user's own images live in the `stencils` folder of the data
//! directory, next to the Stamp tool's outlines.

use mm_edit::brush_alpha::{AlphaImage, Pattern, ScreenStencil};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Largest side kept of a loaded image (pixels).
const MAX_SIDE: usize = 1024;
/// Size the built-in patterns are drawn at.
const PATTERN_SIZE: usize = 256;
/// Image files the stencils folder lists.
const EXTENSIONS: [&str; 3] = ["png", "jpg", "jpeg"];

/// Where an alpha comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Pattern(Pattern),
    File(PathBuf),
}

/// A loaded alpha, with its texture for drawing it over the view.
pub struct Alpha {
    pub source: Source,
    pub image: Arc<AlphaImage>,
    /// Texture and whether it shows the image inverted.
    texture: Option<(egui::TextureHandle, bool)>,
}

impl Alpha {
    pub fn pattern(p: Pattern) -> Self {
        Self { source: Source::Pattern(p), image: Arc::new(p.image(PATTERN_SIZE)), texture: None }
    }

    pub fn open(path: &Path) -> Result<Self, String> {
        Ok(Self { source: Source::File(path.to_path_buf()), image: Arc::new(load_grey(path)?), texture: None })
    }

    /// Name shown in the panel.
    pub fn name(&self) -> String {
        match &self.source {
            Source::Pattern(p) => p.label().into(),
            Source::File(f) => f.file_stem().map_or_else(|| f.display().to_string(), |s| s.to_string_lossy().into_owned()),
        }
    }

    /// Name for settings: the pattern's key, a library image's file name, else its path.
    pub fn key(&self) -> String {
        match &self.source {
            Source::Pattern(p) => p.key().into(),
            Source::File(f) if self.in_library() => f.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            Source::File(f) => f.display().to_string(),
        }
    }

    /// The image file is in the stencils folder.
    pub fn in_library(&self) -> bool {
        match &self.source {
            Source::File(f) => stencils_dir().is_some_and(|d| f.parent() == Some(d.as_path())),
            Source::Pattern(_) => true,
        }
    }

    /// From a settings name: `none`, a pattern (`noise`, `rings` …), an image in the stencils
    /// folder (with or without its extension), or a path to a PNG / JPEG.
    pub fn find(key: &str) -> Result<Option<Alpha>, String> {
        Self::find_in(key, stencils_dir().as_deref())
    }

    fn find_in(key: &str, dir: Option<&Path>) -> Result<Option<Alpha>, String> {
        let key = key.trim();
        if key.is_empty() || key.eq_ignore_ascii_case("none") {
            return Ok(None);
        }
        if let Some(p) = Pattern::from_key(key) {
            return Ok(Some(Alpha::pattern(p)));
        }
        let lib = library_in(dir).into_iter().find(|f| f.file_name().is_some_and(|n| n.to_string_lossy() == key) || f.file_stem().is_some_and(|n| n.to_string_lossy() == key));
        let path = lib.unwrap_or_else(|| PathBuf::from(key));
        if !path.is_file() {
            let names: Vec<&str> = Pattern::ALL.iter().map(|p| p.key()).collect();
            return Err(format!("no stencil \"{key}\": use none, {}, an image in the stencils folder, or a path to a PNG / JPEG", names.join(", ")));
        }
        Alpha::open(&path).map(Some)
    }

    /// The image as a texture: white where the brush acts, transparent where it does not.
    pub fn texture(&mut self, ctx: &egui::Context, invert: bool) -> egui::TextureId {
        if let Some((t, inv)) = &self.texture {
            if *inv == invert {
                return t.id();
            }
        }
        let img = &self.image;
        let rgba: Vec<u8> = img.data.iter().flat_map(|&v| [255, 255, 255, ((if invert { 1.0 - v } else { v }) * 255.0).round() as u8]).collect();
        let color = egui::ColorImage::from_rgba_unmultiplied([img.width, img.height], &rgba);
        let t = ctx.load_texture(format!("brush alpha {}", self.key()), color, egui::TextureOptions::LINEAR_REPEAT);
        let id = t.id();
        self.texture = Some((t, invert));
        id
    }

    /// Copy the image into the stencils folder; returns its name there.
    pub fn save_to_library(&mut self) -> Result<String, String> {
        let Source::File(f) = &self.source else { return Err("built-in patterns need no saving".into()) };
        let dir = stencils_dir().ok_or("no data directory")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let name = f.file_name().ok_or("no file name")?.to_string_lossy().into_owned();
        let to = dir.join(&name);
        if to.exists() {
            return Err(format!("the stencils folder already has {name}"));
        }
        std::fs::copy(f, &to).map_err(|e| e.to_string())?;
        self.source = Source::File(to);
        Ok(name)
    }
}

pub fn stencils_dir() -> Option<PathBuf> {
    crate::app::data_dir().map(|d| d.join("stencils"))
}

/// The images in the stencils folder, by name.
pub fn library() -> Vec<PathBuf> {
    library_in(stencils_dir().as_deref())
}

fn library_in(dir: Option<&Path>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = dir
        .and_then(|d| std::fs::read_dir(d).ok())
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| EXTENSIONS.iter().any(|e| x.eq_ignore_ascii_case(e))))
        .collect();
    out.sort();
    out
}

/// A PNG or JPEG file as grey values (colour by its brightness; transparent counts as black),
/// shrunk to at most [`MAX_SIDE`] pixels across.
pub fn load_grey(path: &Path) -> Result<AlphaImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    decode_grey(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn decode_grey(bytes: &[u8]) -> Result<AlphaImage, String> {
    let img = if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        decode_png(bytes)?
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        let grey = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg).map_err(|e| e.to_string())?.to_luma8();
        let (w, h) = (grey.width() as usize, grey.height() as usize);
        AlphaImage::new(w, h, grey.into_raw().into_iter().map(|v| v as f32 / 255.0).collect())
    } else {
        return Err("not a PNG or JPEG image".into());
    };
    Ok(img.fit(MAX_SIDE))
}

fn decode_png(bytes: &[u8]) -> Result<AlphaImage, String> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0; reader.output_buffer_size().ok_or("the PNG is too large")?];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let px = &buf[..info.buffer_size()];
    let luma = |r: u8, g: u8, b: u8| (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) / 255.0;
    let data: Vec<f32> = match info.color_type {
        png::ColorType::Grayscale => px.iter().map(|&v| v as f32 / 255.0).collect(),
        png::ColorType::GrayscaleAlpha => px.as_chunks::<2>().0.iter().map(|p| p[0] as f32 / 255.0 * p[1] as f32 / 255.0).collect(),
        png::ColorType::Rgb => px.as_chunks::<3>().0.iter().map(|p| luma(p[0], p[1], p[2])).collect(),
        png::ColorType::Rgba => px.as_chunks::<4>().0.iter().map(|p| luma(p[0], p[1], p[2]) * p[3] as f32 / 255.0).collect(),
        png::ColorType::Indexed => return Err("unexpected indexed PNG".into()),
    };
    if data.len() != w * h {
        return Err("damaged PNG".into());
    }
    Ok(AlphaImage::new(w, h, data))
}

/// Combo box choosing an alpha (None, the built-in patterns, the stencils folder, or an image
/// opened from anywhere), with a thumbnail (shown `invert`ed or not) and Save to library for
/// an image from elsewhere. Returns whether the choice changed.
pub fn picker(ui: &mut egui::Ui, id: &str, current: &mut Option<Alpha>, invert: bool, status: &mut String) -> bool {
    let mut choice: Option<Option<Source>> = None;
    let mut open = false;
    let label = current.as_ref().map_or_else(|| "None".to_string(), Alpha::name);
    ui.horizontal(|ui| {
        if let Some(a) = current.as_mut() {
            let tex = a.texture(ui.ctx(), invert);
            let ink = ui.visuals().strong_text_color();
            ui.add(egui::Image::new((tex, egui::vec2(36.0, 36.0))).tint(ink).bg_fill(ui.visuals().extreme_bg_color));
        }
        egui::ComboBox::from_id_salt(id).selected_text(label).width(130.0).show_ui(ui, |ui| {
            if ui.selectable_label(current.is_none(), "None").clicked() {
                choice = Some(None);
            }
            let now = current.as_ref().map(|a| &a.source);
            for p in Pattern::ALL {
                if ui.selectable_label(now == Some(&Source::Pattern(p)), p.label()).on_hover_text(p.about()).clicked() {
                    choice = Some(Some(Source::Pattern(p)));
                }
            }
            let lib = library();
            if !lib.is_empty() {
                ui.separator();
                for f in lib {
                    let name = f.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                    if ui.selectable_label(now == Some(&Source::File(f.clone())), name).clicked() {
                        choice = Some(Some(Source::File(f)));
                    }
                }
            }
            ui.separator();
            if ui.selectable_label(false, "Open image…").on_hover_text("A grey PNG or JPEG: white is full strength, black none").clicked() {
                open = true;
            }
        });
        if let Some(a) = current.as_mut().filter(|a| !a.in_library()) {
            if ui.small_button("Save to library").on_hover_text("Copy the image into the stencils folder so it is listed next time").clicked() {
                *status = match a.save_to_library() {
                    Ok(name) => format!("Added {name} to the stencils folder"),
                    Err(e) => format!("Save to library: {e}"),
                };
            }
        }
    });
    if open {
        let mut dlg = rfd::FileDialog::new().add_filter("Image", &EXTENSIONS);
        if let Some(d) = stencils_dir().filter(|d| d.is_dir()) {
            dlg = dlg.set_directory(d);
        }
        if let Some(path) = dlg.pick_file() {
            choice = Some(Some(Source::File(path)));
        }
    }
    let Some(choice) = choice else { return false };
    match choice {
        None => *current = None,
        Some(Source::Pattern(p)) => *current = Some(Alpha::pattern(p)),
        Some(Source::File(f)) => match Alpha::open(&f) {
            Ok(a) => *current = Some(a),
            Err(e) => {
                *status = e;
                return false;
            }
        },
    }
    true
}

/// Draw a screen stencil over the viewport `rect` (once, or repeated over the whole view
/// when it tiles), with its outline.
pub fn paint_stencil(painter: &egui::Painter, rect: egui::Rect, alpha: &mut Alpha, st: &ScreenStencil, opacity: f32, ctx: &egui::Context) {
    let tex = alpha.texture(ctx, st.map.invert);
    let size = glam::DVec2::new(rect.width() as f64, rect.height() as f64);
    let to_pos = |p: glam::DVec2| rect.min + egui::vec2(p.x as f32, p.y as f32);
    let tint = egui::Color32::from_rgb(150, 210, 255).gamma_multiply(opacity.clamp(0.0, 1.0));
    let corners_uv = [glam::DVec2::ZERO, glam::DVec2::X, glam::DVec2::ONE, glam::DVec2::Y];
    let outline: Vec<egui::Pos2> = corners_uv.iter().map(|&uv| to_pos(st.screen_of(uv, size))).collect();
    let quad: [(egui::Pos2, glam::DVec2); 4] = if st.map.tile {
        // The whole view, with the image coordinates found under each corner.
        [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom()].map(|p| (p, st.uv(glam::DVec2::new((p.x - rect.min.x) as f64, (p.y - rect.min.y) as f64), size)))
    } else {
        [0, 1, 2, 3].map(|i| (outline[i], corners_uv[i]))
    };
    let mut mesh = egui::Mesh::with_texture(tex);
    for (pos, uv) in quad {
        mesh.vertices.push(egui::epaint::Vertex { pos, uv: egui::pos2(uv.x as f32, uv.y as f32), color: tint });
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.with_clip_rect(rect).add(egui::Shape::mesh(mesh));
    let mut closed = outline;
    closed.push(closed[0]);
    painter.with_clip_rect(rect).add(egui::Shape::dashed_line(&closed, egui::Stroke::new(1.0, tint.gamma_multiply(1.5)), 6.0, 4.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_and_jpeg_load_as_grey() {
        // A grey + alpha PNG: transparent counts as black.
        let (w, h) = (6u32, 4u32);
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, w, h);
            enc.set_color(png::ColorType::GrayscaleAlpha);
            enc.set_depth(png::BitDepth::Eight);
            let px: Vec<u8> = (0..w * h).flat_map(|i| [(i * 10) as u8, if i % w == 0 { 0 } else { 255 }]).collect();
            enc.write_header().unwrap().write_image_data(&px).unwrap();
        }
        let img = decode_grey(&bytes).unwrap();
        assert_eq!((img.width, img.height), (6, 4));
        assert!((img.data[1] - 10.0 / 255.0).abs() < 1e-6 && img.data[6] == 0.0);
        // An RGB PNG by brightness.
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, 2, 1);
            enc.set_color(png::ColorType::Rgb);
            enc.write_header().unwrap().write_image_data(&[255, 255, 255, 0, 0, 255]).unwrap();
        }
        let img = decode_grey(&bytes).unwrap();
        assert!((img.data[0] - 1.0).abs() < 1e-6 && (img.data[1] - 0.114).abs() < 1e-3);
        // JPEG, and the image is shrunk to a sensible size.
        let img = decode_grey(include_bytes!("testdata/colour.jpg")).unwrap();
        assert_eq!((img.width, img.height), (37, 29));
        assert!(decode_grey(b"GIF89a").is_err());
        let big = AlphaImage::new(3000, 10, vec![0.5; 30_000]).fit(MAX_SIDE);
        assert!(big.width <= MAX_SIDE);
    }

    #[test]
    fn stencils_come_from_the_stencils_folder() {
        let dir = std::env::temp_dir().join(format!("polymix-stencils-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, 2, 2);
            enc.set_color(png::ColorType::Grayscale);
            enc.write_header().unwrap().write_image_data(&[0, 255, 255, 0]).unwrap();
        }
        std::fs::write(dir.join("checker.png"), &bytes).unwrap();
        std::fs::write(dir.join("My stencil 1.txt"), "0 0\n1 0\n0 1\n").unwrap();
        let lib = library_in(Some(&dir));
        assert_eq!(lib.len(), 1, "images only, not the Stamp outlines: {lib:?}");
        let by_name = Alpha::find_in("checker", Some(&dir)).unwrap().unwrap();
        assert_eq!(by_name.source, Source::File(dir.join("checker.png")));
        assert_eq!(Alpha::find_in("checker.png", Some(&dir)).unwrap().unwrap().image.data, vec![0.0, 1.0, 1.0, 0.0]);
        let by_path = Alpha::find_in(&dir.join("checker.png").display().to_string(), None).unwrap().unwrap();
        assert_eq!(by_path.name(), "checker");
        assert!(Alpha::find_in("none", Some(&dir)).unwrap().is_none());
        assert_eq!(Alpha::find_in("dots", Some(&dir)).unwrap().unwrap().key(), "dots");
        assert!(Alpha::find_in("no such stencil", Some(&dir)).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
