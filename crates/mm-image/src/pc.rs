//! Phase-contrast MRI (2D / 4D flow): phase images to blood velocities. Each of the three
//! flow-encoded phase images maps its value range onto −π…π, and a phase φ means a velocity
//! of VENC · φ / π along that encoding direction. Averaged over the cardiac phases this gives
//! velocity components, the speed, and a phase-contrast angiogram (speed weighted by the
//! magnitude image), the start of CFD boundary conditions and validation.
//!
//! Vendors store phase differently (Siemens: 0…4095 or −4096…4095; others as velocities
//! already), and the sign of each direction depends on the scanner set-up: check a known flow
//! direction (e.g. the aorta) and flip a component if needed.

use crate::image::{check, Progress};
use anyhow::{bail, Result};
use mm_volume::load::{load_volume_at, volume_count, LoadedVolume};
use mm_volume::model::Volume;

/// How a phase image's values relate to the phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhaseValues {
    /// Values from `lo` to `hi` span −π to π.
    Range(f64, f64),
    /// The values are already velocities (cm/s, times `scale`).
    Velocity { scale: f64 },
}

/// The usual phase value range, from the values themselves: Siemens 12-bit stored unsigned
/// (0…4095) or signed (−4096…4095), else the image's own minimum and maximum.
pub fn guess_range(vol: &LoadedVolume) -> (f64, f64) {
    let (lo, hi) = (vol.range.0 as f64, vol.range.1 as f64);
    if lo >= -4096.5 && hi <= 4096.5 && lo < -100.0 {
        (-4096.0, 4096.0)
    } else if lo >= -0.5 && hi <= 4096.5 && hi > 100.0 {
        (0.0, 4096.0)
    } else if lo >= -std::f64::consts::PI - 1e-3 && hi <= std::f64::consts::PI + 1e-3 {
        (-std::f64::consts::PI, std::f64::consts::PI)
    } else {
        (lo, hi)
    }
}

/// Velocities (cm/s) from a phase image with encoding velocity `venc` (cm/s).
pub fn velocity(phase: &LoadedVolume, values: PhaseValues, venc: f64) -> Vec<f32> {
    match values {
        PhaseValues::Velocity { scale } => phase.values.iter().map(|&x| (x as f64 * scale) as f32).collect(),
        PhaseValues::Range(lo, hi) => {
            let mid = 0.5 * (lo + hi);
            let half = (0.5 * (hi - lo)).max(1e-12);
            phase.values.iter().map(|&x| (venc * (x as f64 - mid) / half) as f32).collect()
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PcParams {
    /// Encoding velocity of each direction (cm/s).
    pub venc: [f64; 3],
    /// How each phase image stores phase (`None`: guessed from its values).
    pub values: [Option<PhaseValues>; 3],
    /// Flip a direction (scanner set-ups differ).
    pub flip: [bool; 3],
}

impl Default for PcParams {
    fn default() -> Self {
        Self { venc: [100.0; 3], values: [None; 3], flip: [false; 3] }
    }
}

/// Time-averaged results on the phase images' grid.
pub struct PcOutput {
    /// Mean velocity components (cm/s).
    pub velocity: [Vec<f32>; 3],
    /// Mean and peak speed (cm/s).
    pub mean_speed: Vec<f32>,
    pub peak_speed: Vec<f32>,
    /// Mean speed × normalised magnitude (when a magnitude image is given): an angiogram.
    pub angiogram: Option<Vec<f32>>,
    /// The first phase image's first volume (its grid for saving).
    pub grid: LoadedVolume,
    pub time_points: usize,
}

/// Velocities from three phase images (4D: one volume per cardiac phase), optionally with a
/// magnitude image on the same grid.
pub fn phase_contrast(phases: [&Volume; 3], magnitude: Option<&Volume>, p: &PcParams, max_voxels: usize, progress: Progress) -> Result<PcOutput> {
    let t = volume_count(phases[0]);
    if phases.iter().any(|v| volume_count(v) != t) {
        bail!("the three phase images have different numbers of time points");
    }
    let mut sum = [Vec::new(), Vec::new(), Vec::new()];
    let (mut speed_sum, mut peak, mut mag_sum): (Vec<f32>, Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
    let mut grid: Option<LoadedVolume> = None;
    let mut ranges: [Option<PhaseValues>; 3] = p.values;
    for k in 0..t {
        let mut v3: Vec<Vec<f32>> = Vec::new();
        for (a, ph) in phases.iter().enumerate() {
            let f0 = (k * 3 + a) as f32 / (t * 3) as f32;
            let vol = load_volume_at(ph, k, max_voxels, &|f| progress(f0 + f / (t * 3) as f32))?;
            if let Some(g) = &grid {
                if g.dims != vol.dims {
                    bail!("the phase images are on different grids");
                }
            }
            let values = *ranges[a].get_or_insert_with(|| {
                let (lo, hi) = guess_range(&vol);
                PhaseValues::Range(lo, hi)
            });
            let mut v = velocity(&vol, values, p.venc[a]);
            if p.flip[a] {
                v.iter_mut().for_each(|x| *x = -*x);
            }
            v3.push(v);
            if grid.is_none() {
                grid = Some(vol);
            }
        }
        let n = v3[0].len();
        if k == 0 {
            sum = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            speed_sum = vec![0.0; n];
            peak = vec![0.0; n];
        }
        for i in 0..n {
            let s = (v3[0][i] * v3[0][i] + v3[1][i] * v3[1][i] + v3[2][i] * v3[2][i]).sqrt();
            speed_sum[i] += s;
            peak[i] = peak[i].max(s);
            for a in 0..3 {
                sum[a][i] += v3[a][i];
            }
        }
        if let Some(m) = magnitude {
            let mv = load_volume_at(m, k.min(volume_count(m) - 1), max_voxels, &|_| true)?;
            if mv.values.len() != n {
                bail!("the magnitude image is on a different grid");
            }
            if mag_sum.is_empty() {
                mag_sum = vec![0.0; n];
            }
            mag_sum.iter_mut().zip(&mv.values).for_each(|(a, b)| *a += b);
        }
        check(progress, (k + 1) as f32 / t as f32)?;
    }
    let tf = t as f32;
    let mean_speed: Vec<f32> = speed_sum.iter().map(|s| s / tf).collect();
    let angiogram = (!mag_sum.is_empty()).then(|| {
        let mmax = mag_sum.iter().copied().fold(0.0f32, f32::max).max(1e-12);
        mean_speed.iter().zip(&mag_sum).map(|(s, m)| s * (m / mmax)).collect()
    });
    let velocity = sum.map(|v| v.iter().map(|x| x / tf).collect());
    Ok(PcOutput { velocity, mean_speed, peak_speed: peak, angiogram, grid: grid.unwrap(), time_points: t })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;
    use mm_volume::model::{DataType, Format, Kind, Voxels};

    fn raw_volume(dir: &std::path::Path, name: &str, values: &[f32], dims4: [usize; 4]) -> Volume {
        let p = dir.join(name);
        std::fs::write(&p, values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        Volume {
            format: Format::Nifti1,
            kind: Kind::TimeSeries,
            name: name.into(),
            files: vec![p.clone()],
            bytes: 0,
            geometry: Some(Geometry::from_axes(dims4, [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO)),
            data_type: Some(DataType::F32),
            fields: vec![],
            warnings: vec![],
            voxels: Voxels::Raw { file: p, file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype: DataType::F32, big_endian: false, slope: 1.0, intercept: 0.0 },
            window: None,
            photometric: None,
            patient: None,
        }
    }

    #[test]
    fn velocities_from_siemens_style_phases() {
        let dir = std::env::temp_dir().join(format!("mmi-pc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Two voxels, two cardiac phases. Unsigned 12-bit phase: 2048 = 0, 3072 = +π/2.
        let x = raw_volume(&dir, "x", &[3072.0, 2048.0, 1024.0, 2048.0], [2, 1, 1, 2]);
        let y = raw_volume(&dir, "y", &[2048.0, 4095.0, 2048.0, 4095.0], [2, 1, 1, 2]);
        let z = raw_volume(&dir, "z", &[2048.0, 2048.0, 2048.0, 0.0], [2, 1, 1, 2]);
        let mag = raw_volume(&dir, "m", &[10.0, 5.0, 10.0, 5.0], [2, 1, 1, 2]);
        let p = PcParams { venc: [100.0, 50.0, 80.0], ..Default::default() };
        let out = phase_contrast([&x, &y, &z], Some(&mag), &p, 100, &|_| true).unwrap();
        assert_eq!(out.time_points, 2);
        // Voxel 0: vx +50 then −50 → mean 0, speed 50 each phase.
        assert!(out.velocity[0][0].abs() < 1e-3 && (out.mean_speed[0] - 50.0).abs() < 1e-3);
        // Voxel 1: vy ≈ +50 both phases, vz 0 then −80.
        assert!((out.velocity[1][1] - 49.98).abs() < 0.05, "{}", out.velocity[1][1]);
        assert!((out.peak_speed[1] - (49.98f32.powi(2) + 80.0f32.powi(2)).sqrt()).abs() < 0.1);
        let a = out.angiogram.unwrap();
        assert!((a[0] - 50.0).abs() < 1e-3 && (a[1] - out.mean_speed[1] * 0.5).abs() < 1e-3);
        // Flipping a direction negates it.
        let f = phase_contrast([&x, &y, &z], None, &PcParams { flip: [false, true, false], ..p.clone() }, 100, &|_| true).unwrap();
        assert!((f.velocity[1][1] + out.velocity[1][1]).abs() < 1e-3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
