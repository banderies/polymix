//! Image slices in the 3D view: the image open in the Segmentation tab as axial, sagittal and
//! coronal planes among the meshes, to check a surface against the image it came from.

use crate::image_planes::{PlaneSettings, PlanesFrame, PLANES};
use mm_volume::load::LoadedVolume;
use std::sync::Arc;

/// The image the Segmentation tab shows, with its crosshair and window.
#[derive(Clone)]
pub struct SliceSource {
    pub volume: Arc<LoadedVolume>,
    pub name: String,
    pub cursor: [f64; 3],
    pub window: (f64, f64),
}

#[derive(Default)]
pub struct ImageSlices {
    pub show: bool,
    pub source: Option<SliceSource>,
    /// Where the planes are and how they look (`None` until there is an image).
    pub settings: Option<PlaneSettings>,
    /// The planes go through the Segmentation tab's crosshair, with its window.
    pub follow: bool,
}

impl ImageSlices {
    pub fn new() -> Self {
        Self { follow: true, ..Default::default() }
    }

    /// The image open in the Segmentation tab (`None`: none is).
    pub fn set_source(&mut self, source: Option<SliceSource>) {
        let same = match (&self.source, &source) {
            (Some(a), Some(b)) => Arc::ptr_eq(&a.volume, &b.volume),
            (None, None) => true,
            _ => false,
        };
        if !same {
            self.settings = source.as_ref().map(|s| PlaneSettings::new(&s.volume));
        }
        if let (Some(s), Some(src), true) = (&mut self.settings, &source, self.follow) {
            s.cursor = src.cursor;
            s.window = src.window;
        }
        self.source = source;
    }

    /// What the renderer draws this frame.
    pub fn frame(&self) -> Option<PlanesFrame> {
        let (true, Some(src), Some(s)) = (self.show, &self.source, &self.settings) else { return None };
        Some(PlanesFrame { volume: src.volume.clone(), settings: *s })
    }

    /// The floating panel of the planes' settings, while they are shown.
    pub fn window(&mut self, ctx: &egui::Context) {
        if !self.show {
            return;
        }
        let mut open = true;
        egui::Window::new("Image slices").open(&mut open).resizable(false).default_width(260.0).show(ctx, |ui| {
            let (Some(src), Some(s)) = (&self.source, &mut self.settings) else {
                ui.weak("Open an image in the Segmentation tab (⌘2): its slices are shown here among the meshes.");
                return;
            };
            ui.label(egui::RichText::new(&src.name).strong());
            ui.checkbox(&mut self.follow, "Follow the Segmentation crosshair").on_hover_text("The planes go through the crosshair of the Segmentation tab's views, with their window");
            let vol = &src.volume;
            for (k, plane) in PLANES.iter().enumerate() {
                let axis = mm_volume::slice::plane_axes(&vol.geometry, *plane).slice;
                ui.horizontal(|ui| {
                    ui.checkbox(&mut s.show[k], plane.label());
                    let n = vol.dims[axis].max(1) - 1;
                    let mut at = s.cursor[axis].round() as usize;
                    if ui.add_enabled(s.show[k], egui::Slider::new(&mut at, 0..=n).show_value(true)).changed() {
                        s.cursor[axis] = at as f64;
                        self.follow = false;
                    }
                });
            }
            ui.add(egui::Slider::new(&mut s.opacity, 0.0..=1.0).text("opacity"));
            ui.checkbox(&mut s.outline, "Outlines in the planes' colours");
            ui.horizontal(|ui| {
                ui.label("Meshes in");
                ui.selectable_value(&mut s.lps, true, "LPS").on_hover_text("As 3D Slicer saves models, and as Polymix makes meshes from segments");
                ui.selectable_value(&mut s.lps, false, "RAS");
            });
            if !self.follow {
                ui.horizontal(|ui| {
                    ui.label("Window");
                    let speed = ((vol.range.1 - vol.range.0) as f64 / 500.0).max(1e-3);
                    ui.add(egui::DragValue::new(&mut s.window.1).speed(speed).range(1e-6..=f64::MAX));
                    ui.label("level");
                    ui.add(egui::DragValue::new(&mut s.window.0).speed(speed));
                    if ui.small_button("Auto").clicked() {
                        s.window = vol.auto_window;
                    }
                });
            }
        });
        if !open {
            self.show = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    fn source(cursor: [f64; 3]) -> SliceSource {
        let g = Geometry::from_axes([4, 4, 4, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let volume = Arc::new(LoadedVolume::from_values("t", [4, 4, 4], vec![0.0; 64], g));
        SliceSource { volume, name: "t".into(), cursor, window: (1.0, 2.0) }
    }

    #[test]
    fn slices_follow_the_segmentation_crosshair_until_moved() {
        let mut s = ImageSlices::new();
        assert!(s.frame().is_none());
        let src = source([1.0, 2.0, 3.0]);
        s.set_source(Some(src.clone()));
        s.show = true;
        let f = s.frame().unwrap();
        assert_eq!((f.settings.cursor, f.settings.window), ([1.0, 2.0, 3.0], (1.0, 2.0)));
        // The crosshair moves: the planes follow; not once the user placed them.
        s.set_source(Some(SliceSource { cursor: [0.0, 0.0, 0.0], ..src.clone() }));
        assert_eq!(s.frame().unwrap().settings.cursor, [0.0; 3]);
        s.follow = false;
        s.set_source(Some(SliceSource { cursor: [3.0, 3.0, 3.0], ..src.clone() }));
        assert_eq!(s.frame().unwrap().settings.cursor, [0.0; 3]);
        // No image: nothing drawn.
        s.set_source(None);
        assert!(s.frame().is_none());
    }
}
