//! Polymix — an open-source mesh editor in the spirit of Autodesk Meshmixer.

mod app;
mod camera;
mod centerline;
mod commands;
mod jobs;
mod renderer;
mod scene;
mod tools;
mod ui_util;

#[cfg(test)]
mod pipeline_tests;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let files: Vec<std::path::PathBuf> = std::env::args_os().skip(1).map(Into::into).collect();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Polymix")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true)
            .with_icon(app_icon()),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: wgpu_options(),
        ..Default::default()
    };
    eframe::run_native("Polymix", options, Box::new(move |cc| Ok(Box::new(app::PolymixApp::new(cc, files)))))
}

/// Window / dock icon decoded from the embedded PNG.
fn app_icon() -> egui::IconData {
    let decoder = png::Decoder::new(std::io::Cursor::new(include_bytes!("../../../assets/icon_1024.png").as_slice()));
    let decoded = decoder.read_info().and_then(|mut r| {
        let mut buf = vec![0; r.output_buffer_size().unwrap_or(0)];
        let info = r.next_frame(&mut buf)?;
        buf.truncate(info.buffer_size());
        Ok((buf, info.width, info.height))
    });
    match decoded {
        Ok((rgba, width, height)) => egui::IconData { rgba, width, height },
        Err(_) => egui::IconData::default(),
    }
}

/// Request the adapter's full limits: the defaults cap buffers at 256 MiB, which a
/// 3M-triangle mesh exceeds.
fn wgpu_options() -> eframe::egui_wgpu::WgpuConfiguration {
    use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup};
    use eframe::wgpu;
    let mut cfg = WgpuConfiguration::default();
    if let WgpuSetup::CreateNew(create) = &mut cfg.wgpu_setup {
        create.device_descriptor = std::sync::Arc::new(|adapter: &wgpu::Adapter| {
            let limits = if adapter.get_info().backend == wgpu::Backend::Gl { wgpu::Limits::downlevel_webgl2_defaults() } else { adapter.limits() };
            wgpu::DeviceDescriptor { label: Some("polymix device"), required_limits: limits, ..Default::default() }
        });
    }
    cfg
}
