//! Render a synthetic volume (a ball, a curved tube, label surfaces) in the three 3D modes
//! and save a screenshot: `cargo run -p polymix-imaging --example view3d -- out.png`.

use eframe::egui;
use glam::DVec3;
use mm_volume::geometry::Geometry;
use mm_volume::load::LoadedVolume;
use polymix_imaging::{LabelOverlay, Mode3d, ViewState, VolumeView3d};
use std::sync::Arc;

struct App {
    vol: Arc<LoadedVolume>,
    labels: Vec<u8>,
    state: ViewState,
    views: [VolumeView3d; 3],
    render_state: Option<eframe::egui_wgpu::RenderState>,
    frame: u32,
    out: std::path::PathBuf,
}

fn synthetic() -> (LoadedVolume, Vec<u8>) {
    let dims = [96, 96, 80];
    let mut values = vec![0f32; dims[0] * dims[1] * dims[2]];
    let mut labels = vec![0u8; values.len()];
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = DVec3::new(i as f64, j as f64, k as f64 * 1.5);
                let n = (k * dims[1] + j) * dims[0] + i;
                let ball = (p - DVec3::new(40.0, 48.0, 60.0)).length();
                // A tube curving through the volume.
                let t = k as f64 / dims[2] as f64;
                let c = DVec3::new(70.0 - 20.0 * (t * 3.0).sin(), 30.0 + 30.0 * t, p.z);
                let tube = ((p - c) * DVec3::new(1.0, 1.0, 0.0)).length();
                if ball < 25.0 {
                    values[n] = (600.0 + 200.0 * (ball / 25.0)) as f32;
                    if p.x < 40.0 {
                        labels[n] = 2;
                    }
                }
                if tube < 6.0 {
                    values[n] = 1200.0;
                    labels[n] = 1;
                }
            }
        }
    }
    let g = Geometry::from_axes([96, 96, 80, 1], [1.0, 1.0, 1.5], [-DVec3::X, -DVec3::Y, DVec3::Z], DVec3::ZERO);
    (LoadedVolume::from_values("synthetic", dims, values, g), labels)
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let colours = [egui::Color32::TRANSPARENT, egui::Color32::from_rgb(220, 60, 50), egui::Color32::from_rgb(80, 160, 230)];
        let overlay = LabelOverlay { labels: &self.labels, colours: &colours, opacity: 1.0, version: 1 };
        egui::CentralPanel::default().show(ui, |ui| {
            ui.columns(3, |cols| {
                for (k, col) in cols.iter_mut().enumerate() {
                    self.views[k].ui(col, self.render_state.as_ref(), &self.vol, &self.state, if k == 2 { Some(&overlay) } else { None });
                }
            });
        });
        self.frame += 1;
        let ctx = ui.ctx().clone();
        if self.frame == 10 {
            let out = self.out.clone();
            let ctx2 = ctx.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::ScreenshotCallback::new(move |img| {
                let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                image::RgbaImage::from_raw(img.size[0] as u32, img.size[1] as u32, bytes).unwrap().save(&out).unwrap();
                ctx2.send_viewport_cmd(egui::ViewportCommand::Close);
            })));
        }
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result {
    let out = std::env::args().nth(1).unwrap_or_else(|| "view3d.png".into()).into();
    let (vol, labels) = synthetic();
    let state = ViewState::new(&vol);
    let options = eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 420.0]), renderer: eframe::Renderer::Wgpu, ..Default::default() };
    eframe::run_native("view3d", options, Box::new(move |cc| {
        let mut views = [VolumeView3d::new(), VolumeView3d::new(), VolumeView3d::new()];
        views[0].mode = Mode3d::Mip;
        views[1].mode = Mode3d::Volume;
        views[2].mode = Mode3d::Surface;
        views[2].threshold = Some(500.0);
        Ok(Box::new(App { vol: Arc::new(vol), labels, state, views, render_state: cc.wgpu_render_state.clone(), frame: 0, out }))
    }))
}
