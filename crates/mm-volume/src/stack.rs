//! Volumes made of 2D images: folders of numbered slices (JPEG, PNG, BMP, TIFF exported from
//! a scanner, microscope or another program) and multi-page TIFF files. Slices are ordered by
//! name with numbers compared by value ("slice2" before "slice10"). Voxel sizes come from
//! ImageJ or OME metadata in TIFF files; otherwise 1 mm is assumed.

use crate::geometry::Geometry;
use crate::model::{DataType, Field, Format, Kind, Volume, Voxels};
use anyhow::{bail, Context, Result};
use glam::DVec3;
use std::path::{Path, PathBuf};

/// Name order with digit runs compared as numbers.
pub fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let (mut x, mut y) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (x.peek().copied(), y.peek().copied()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(c), Some(d)) if c.is_ascii_digit() && d.is_ascii_digit() => {
                let take = |it: &mut std::iter::Peekable<std::str::Chars>| {
                    let mut s = String::new();
                    while let Some(c) = it.peek().copied().filter(char::is_ascii_digit) {
                        s.push(c);
                        it.next();
                    }
                    s
                };
                let (p, q) = (take(&mut x), take(&mut y));
                let (p0, q0) = (p.trim_start_matches('0'), q.trim_start_matches('0'));
                let o = p0.len().cmp(&q0.len()).then_with(|| p0.cmp(q0)).then_with(|| p.len().cmp(&q.len()));
                if o.is_ne() {
                    return o;
                }
            }
            (Some(c), Some(d)) => {
                let o = c.to_ascii_lowercase().cmp(&d.to_ascii_lowercase());
                if o.is_ne() {
                    return o;
                }
                x.next();
                y.next();
            }
        }
    }
}

pub(crate) fn is_tiff(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
}

/// Size, data type and voxel size (when the file says) of one image or TIFF page.
struct ImageInfo {
    width: usize,
    height: usize,
    dtype: DataType,
    /// Millimetres per pixel across, down and between slices.
    spacing: Option<[f64; 3]>,
    description: Option<String>,
}

fn tiff_decoder(path: &Path) -> Result<tiff::decoder::Decoder<std::io::BufReader<std::fs::File>>> {
    let f = std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    tiff::decoder::Decoder::new(std::io::BufReader::new(f)).map_err(|e| anyhow::anyhow!("not a readable TIFF: {e}"))
}

fn tiff_dtype(c: tiff::ColorType, sample_format: u16) -> Option<DataType> {
    use tiff::ColorType::*;
    let float = sample_format == 3;
    let signed = sample_format == 2;
    Some(match c {
        Gray(8) | Palette(8) => if signed { DataType::I8 } else { DataType::U8 },
        Gray(16) => if signed { DataType::I16 } else { DataType::U16 },
        Gray(32) if float => DataType::F32,
        Gray(32) => if signed { DataType::I32 } else { DataType::U32 },
        Gray(64) if float => DataType::F64,
        RGB(8) => DataType::Rgb24,
        RGBA(8) => DataType::Rgba32,
        _ => return None,
    })
}

fn rational(v: tiff::decoder::ifd::Value) -> Option<f64> {
    use tiff::decoder::ifd::Value::*;
    match v {
        Rational(n, d) if d != 0 => Some(n as f64 / d as f64),
        Float(f) => Some(f as f64),
        Double(f) => Some(f),
        Unsigned(u) => Some(u as f64),
        _ => None,
    }
}

/// A unit name → millimetres.
fn unit_mm(u: &str) -> Option<f64> {
    Some(match u.trim().to_lowercase().as_str() {
        "mm" | "millimeter" | "millimetre" => 1.0,
        "cm" | "centimeter" | "centimetre" => 10.0,
        "m" | "meter" | "metre" => 1000.0,
        "micron" | "microns" | "um" | "µm" | "\\u00b5m" | "micrometer" | "micrometre" => 1e-3,
        "nm" | "nanometer" | "nanometre" => 1e-6,
        "inch" | "in" => 25.4,
        _ => return None,
    })
}

/// `key="value"` in an XML tag.
fn xml_attr<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let at = s.find(&format!(" {key}=\""))? + key.len() + 3;
    s[at..].split('"').next()
}

/// Voxel size from ImageJ (`spacing=`, `unit=`, X/Y resolution) or OME-XML metadata.
fn tiff_spacing(d: &mut tiff::decoder::Decoder<std::io::BufReader<std::fs::File>>, description: Option<&str>) -> Option<[f64; 3]> {
    let desc = description.unwrap_or("");
    if desc.contains("<OME") {
        let pixels = &desc[desc.find("<Pixels")?..];
        let size = |k: &str| xml_attr(pixels, &format!("PhysicalSize{k}")).and_then(|v| v.parse::<f64>().ok());
        let unit = |k: &str| xml_attr(pixels, &format!("PhysicalSize{k}Unit")).and_then(unit_mm).unwrap_or(1e-3);
        let (x, y) = (size("X")?, size("Y").or(size("X"))?);
        let z = size("Z").unwrap_or(x);
        return Some([x * unit("X"), y * unit("Y"), z * unit("Z")]);
    }
    if desc.starts_with("ImageJ=") {
        let get = |k: &str| desc.lines().find_map(|l| l.strip_prefix(&format!("{k}=")));
        let unit = get("unit").and_then(unit_mm).unwrap_or(1.0);
        let xr = d.find_tag(tiff::tags::Tag::XResolution).ok().flatten().and_then(rational).filter(|r| *r > 0.0)?;
        let yr = d.find_tag(tiff::tags::Tag::YResolution).ok().flatten().and_then(rational).filter(|r| *r > 0.0).unwrap_or(xr);
        let z = get("spacing").and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0 / xr);
        return Some([unit / xr, unit / yr, z * unit]);
    }
    None
}

fn image_info(path: &Path) -> Result<ImageInfo> {
    if is_tiff(path) {
        let mut d = tiff_decoder(path)?;
        let (w, h) = d.dimensions().map_err(|e| anyhow::anyhow!("{e}"))?;
        let c = d.colortype().map_err(|e| anyhow::anyhow!("{e}"))?;
        let sf = d.find_tag_unsigned::<u16>(tiff::tags::Tag::SampleFormat).ok().flatten().unwrap_or(1);
        let dtype = tiff_dtype(c, sf).with_context(|| format!("unsupported TIFF pixels ({c:?})"))?;
        let description = d.find_tag(tiff::tags::Tag::ImageDescription).ok().flatten().and_then(|v| v.into_string().ok());
        let spacing = tiff_spacing(&mut d, description.as_deref());
        return Ok(ImageInfo { width: w as usize, height: h as usize, dtype, spacing, description });
    }
    let r = image::ImageReader::open(path)?.with_guessed_format()?;
    let dec = r.into_decoder().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (w, h) = image::ImageDecoder::dimensions(&dec);
    let dtype = match image::ImageDecoder::color_type(&dec) {
        image::ColorType::L8 | image::ColorType::La8 => DataType::U8,
        image::ColorType::L16 | image::ColorType::La16 => DataType::U16,
        image::ColorType::Rgb8 | image::ColorType::Rgba8 => DataType::Rgb24,
        image::ColorType::Rgb16 | image::ColorType::Rgba16 => DataType::U16,
        _ => DataType::F32,
    };
    Ok(ImageInfo { width: w as usize, height: h as usize, dtype, spacing: None, description: None })
}

/// Pages in a TIFF file.
fn tiff_pages(path: &Path) -> Result<usize> {
    let mut d = tiff_decoder(path)?;
    let mut n = 1;
    while d.more_images() {
        d.next_image().map_err(|e| anyhow::anyhow!("{e}"))?;
        n += 1;
        if n > 100_000 {
            break;
        }
    }
    Ok(n)
}

fn build(name: String, format: Format, slices: Vec<(PathBuf, u32)>, files: Vec<PathBuf>, info: &ImageInfo, mut warnings: Vec<String>) -> Volume {
    let n = slices.len();
    let dims = [info.width, info.height, n, 1];
    let spacing = info.spacing.unwrap_or_else(|| {
        warnings.push("Voxel size unknown: 1 mm assumed".into());
        [1.0; 3]
    });
    // Shown as stored: the first row at the top, the first column on the left.
    let geometry = Geometry::from_axes(dims, spacing, [DVec3::NEG_X, DVec3::NEG_Y, DVec3::Z], DVec3::ZERO);
    warnings.push("No orientation in 2D images".into());
    let mut fields = vec![
        Field::new("File", "Format", format.label()),
        Field::new("Geometry", "Dimensions", format!("{} × {} × {n}", info.width, info.height)),
        Field::new("Geometry", "Voxel size", format!("{:.4} × {:.4} × {:.4} mm", spacing[0], spacing[1], spacing[2])),
        Field::new("Pixel", "Data type", info.dtype.label()),
    ];
    if let Some(d) = &info.description {
        let short: String = d.lines().next().unwrap_or("").chars().take(120).collect();
        fields.push(Field::new("File", "Description", short));
    }
    let bytes = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    Volume { format, kind: Kind::Image, name, files, bytes, geometry: Some(geometry), data_type: Some(info.dtype), fields, warnings, voxels: Voxels::Images { slices }, window: None, photometric: None, patient: None }
}

/// A folder's images of one type as a volume, in natural name order.
pub fn stack_volume(dir: &Path, mut files: Vec<PathBuf>) -> Result<Volume> {
    if files.is_empty() {
        bail!("no images");
    }
    files.sort_by(|a, b| natural_cmp(&a.file_name().unwrap_or_default().to_string_lossy(), &b.file_name().unwrap_or_default().to_string_lossy()));
    let info = image_info(&files[0])?;
    let mut warnings = Vec::new();
    if let Ok(last) = image_info(files.last().unwrap()) {
        if (last.width, last.height) != (info.width, info.height) {
            warnings.push(format!("Images differ in size ({}×{} and {}×{})", info.width, info.height, last.width, last.height));
        }
    }
    let name = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let slices = files.iter().map(|f| (f.clone(), 0)).collect();
    Ok(build(name, Format::ImageStack, slices, files, &info, warnings))
}

/// A multi-page TIFF as a volume; `None` for a single image.
pub fn tiff_volume(path: &Path) -> Result<Option<Volume>> {
    let pages = tiff_pages(path)?;
    if pages < 2 {
        return Ok(None);
    }
    let info = image_info(path)?;
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let slices = (0..pages as u32).map(|k| (path.to_path_buf(), k)).collect();
    Ok(Some(build(name, Format::Tiff, slices, vec![path.to_path_buf()], &info, vec![])))
}

/// Decode one image (or TIFF page) to grey values, x fastest from the top left.
pub fn read_plane(path: &Path, page: u32) -> Result<(usize, usize, Vec<f32>)> {
    if is_tiff(path) {
        let mut d = tiff_decoder(path)?;
        if page > 0 {
            d.seek_to_image(page as usize).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        return decode_current(&mut d);
    }
    let img = image::ImageReader::open(path)?.with_guessed_format()?.decode().map_err(|e| anyhow::anyhow!("cannot decode {}: {e}", path.display()))?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let v: Vec<f32> = match img.color() {
        image::ColorType::L16 | image::ColorType::La16 | image::ColorType::Rgb16 | image::ColorType::Rgba16 => img.to_luma16().into_raw().into_iter().map(f32::from).collect(),
        image::ColorType::Rgb32F | image::ColorType::Rgba32F => img.to_luma32f().into_raw(),
        _ => img.to_luma8().into_raw().into_iter().map(f32::from).collect(),
    };
    Ok((w, h, v))
}

/// Decode pages `first` to `first + count - 1` of a multi-page TIFF with one decoder, walking
/// the pages once (rather than from the first page for each), and hand each page's grey values
/// to `put` with its place in the run. Stops at the first error, from `put` too.
pub(crate) fn read_pages(path: &Path, first: u32, count: usize, put: &mut dyn FnMut(usize, Vec<f32>) -> Result<()>) -> Result<()> {
    let mut d = tiff_decoder(path)?;
    if first > 0 {
        d.seek_to_image(first as usize).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    for i in 0..count {
        if i > 0 {
            d.next_image().map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        put(i, decode_current(&mut d)?.2)?;
    }
    Ok(())
}

/// Grey values of the TIFF page `d` is at, x fastest from the top left, with its size.
fn decode_current(d: &mut tiff::decoder::Decoder<std::io::BufReader<std::fs::File>>) -> Result<(usize, usize, Vec<f32>)> {
    use tiff::decoder::DecodingResult as R;
    let (w, h) = d.dimensions().map_err(|e| anyhow::anyhow!("{e}"))?;
    let samples = match d.colortype().map_err(|e| anyhow::anyhow!("{e}"))? {
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        tiff::ColorType::GrayA(_) => 2,
        _ => 1,
    };
    let v: Vec<f32> = match d.read_image().map_err(|e| anyhow::anyhow!("cannot decode the TIFF page: {e}"))? {
        R::U8(v) => v.into_iter().map(f32::from).collect(),
        R::U16(v) => v.into_iter().map(f32::from).collect(),
        R::U32(v) => v.into_iter().map(|x| x as f32).collect(),
        R::U64(v) => v.into_iter().map(|x| x as f32).collect(),
        R::F16(v) => v.into_iter().map(f32::from).collect(),
        R::F32(v) => v,
        R::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        R::I8(v) => v.into_iter().map(f32::from).collect(),
        R::I16(v) => v.into_iter().map(f32::from).collect(),
        R::I32(v) => v.into_iter().map(|x| x as f32).collect(),
        R::I64(v) => v.into_iter().map(|x| x as f32).collect(),
    };
    let grey = if samples >= 3 { v.chunks(samples).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect() } else if samples == 2 { v.chunks(2).map(|p| p[0]).collect() } else { v };
    Ok((w as usize, h as usize, grey))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_order() {
        let mut v = vec!["s10.png", "s2.png", "s1.png", "S03.png", "s003b.png"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, vec!["s1.png", "s2.png", "S03.png", "s003b.png", "s10.png"]);
    }

    #[test]
    fn png_stack_and_multipage_tiff() {
        let dir = std::env::temp_dir().join(format!("mmv-stack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Three 16-bit PNG slices, written out of order.
        for k in [2u16, 0, 10] {
            let img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(4, 3, |x, y| image::Luma([k * 100 + (y * 4 + x) as u16]));
            img.save(dir.join(format!("slice{k}.png"))).unwrap();
        }
        let files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        let v = stack_volume(&dir, files).unwrap();
        assert_eq!(v.geometry.as_ref().unwrap().dims, [4, 3, 3, 1]);
        assert_eq!(v.data_type, Some(DataType::U16));
        let Voxels::Images { slices } = &v.voxels else { panic!() };
        assert!(slices[2].0.ends_with("slice10.png"));
        let (w, h, px) = read_plane(&slices[1].0, 0).unwrap();
        assert_eq!((w, h, px[5]), (4, 3, 205.0));
        // A 3-page 8-bit TIFF with ImageJ metadata (0.5 µm pixels, 2 µm slices).
        let p = dir.join("stack.tif");
        {
            let f = std::fs::File::create(&p).unwrap();
            let mut enc = tiff::encoder::TiffEncoder::new(f).unwrap();
            for k in 0..3u8 {
                let mut im = enc.new_image::<tiff::encoder::colortype::Gray8>(4, 3).unwrap();
                im.encoder().write_tag(tiff::tags::Tag::ImageDescription, "ImageJ=1.54f\nimages=3\nslices=3\nunit=micron\nspacing=2\n").unwrap();
                im.x_resolution(tiff::encoder::Rational { n: 2, d: 1 });
                im.y_resolution(tiff::encoder::Rational { n: 2, d: 1 });
                im.write_data(&[k * 10; 12]).unwrap();
            }
        }
        let v = tiff_volume(&p).unwrap().unwrap();
        let g = v.geometry.as_ref().unwrap();
        assert_eq!(g.dims, [4, 3, 3, 1]);
        assert!((g.spacing[0] - 0.0005).abs() < 1e-12 && (g.spacing[2] - 0.002).abs() < 1e-12, "{:?}", g.spacing);
        assert_eq!(read_plane(&p, 2).unwrap().2[0], 20.0);
        let single = dir.join("one.tif");
        let f = std::fs::File::create(&single).unwrap();
        tiff::encoder::TiffEncoder::new(f).unwrap().write_image::<tiff::encoder::colortype::Gray8>(2, 2, &[0; 4]).unwrap();
        assert!(tiff_volume(&single).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
