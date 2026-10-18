//! Polymix — an open-source mesh editor in the spirit of Autodesk Meshmixer.

mod app;
mod camera;
mod centerline;
mod commands;
mod icons;
mod jobs;
mod materials;
mod navigation;
mod project;
mod renderer;
mod scene;
mod spacemouse;
mod tools;
mod ui_util;
mod view_cube;

#[cfg(test)]
mod pipeline_tests;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    if std::env::args().any(|a| a == "--probe-3dmouse") {
        probe_3d_mouse();
        return Ok(());
    }
    let files: Vec<std::path::PathBuf> = std::env::args_os().skip(1).map(Into::into).collect();
    let scripted = std::env::var_os("POLYMIX_SCRIPT").is_some();
    #[allow(unused_mut)]
    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Polymix")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true)
            .with_icon(app_icon())
            .with_active(!scripted),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: wgpu_options(),
        ..Default::default()
    };
    // Scripted test runs must not steal focus (or keystrokes) from whatever the user is doing.
    #[cfg(target_os = "macos")]
    if scripted {
        options.event_loop_builder = Some(Box::new(|b| {
            use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
            b.with_activate_ignoring_other_apps(false).with_activation_policy(ActivationPolicy::Accessory);
        }));
    }
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

/// `polymix --probe-3dmouse`: connect to the 3D mouse driver without opening a window, then
/// record up to six gestures (cap motions separated by letting go) and report the dominant
/// raw axis of each, to check the axis mapping on a given device.
fn probe_3d_mouse() {
    let mut sm = spacemouse::SpaceMouse::start_with(&egui::Context::default(), true);
    println!("3D mouse: {:?}", sm.status);
    println!("Within 40 s, make these six moves, letting go between them:");
    println!("  1 push right   2 pull up   3 pull toward you");
    println!("  4 tilt the top away from you   5 twist clockwise   6 tilt the top to the right");
    let names = ["pan x", "pan y", "zoom", "tilt", "spin", "roll"];
    let start = std::time::Instant::now();
    let mut gesture: Option<[f32; 6]> = None;
    let mut buttons_seen = Vec::new();
    let mut n = 0;
    while start.elapsed() < std::time::Duration::from_secs(40) && n < 6 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        let (m, pressed) = sm.poll();
        buttons_seen.extend(pressed);
        let v: Vec<f32> = m.t.iter().chain(m.r.iter()).copied().collect();
        if m.is_zero() {
            if let Some(peak) = gesture.take() {
                n += 1;
                let (i, p) = peak.iter().enumerate().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap();
                println!("gesture {n}: {} {:+.2}   (all: {})", names[i], p, peak.iter().map(|x| format!("{x:+.2}")).collect::<Vec<_>>().join(" "));
            }
        } else {
            let g = gesture.get_or_insert([0.0; 6]);
            for k in 0..6 {
                if v[k].abs() > g[k].abs() {
                    g[k] = v[k];
                }
            }
        }
    }
    if !buttons_seen.is_empty() {
        println!("buttons pressed: {buttons_seen:?}");
    }
    println!("expected: 1 pan x +, 2 pan y +, 3 zoom +, 4 tilt -, 5 spin -, 6 roll -");
}
