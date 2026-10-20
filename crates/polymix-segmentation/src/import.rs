//! Segments from elsewhere: a label map file (3D Slicer `.seg.nrrd`, or a NIfTI label map)
//! read onto the volume's grid, and a closed mesh from the Mesh workspace voxelised.

use egui::Color32;
use glam::DVec3;
use mm_core::Mesh;
use mm_volume::load::LoadedVolume;
use mm_volume::scan::{classify, Found};
use rayon::prelude::*;
use std::path::Path;

/// One segment found in a file: its label there, name, colour and Slicer status key.
#[derive(Clone, Debug, PartialEq)]
pub struct Found3d {
    pub label: u8,
    pub name: String,
    pub colour: Option<Color32>,
    pub status: String,
}

/// Segments read into the volume's grid: a label per voxel (the file's labels), what is
/// known of each, and anything worth saying (resampled, layers left out).
pub struct Imported {
    pub labels: Vec<u8>,
    pub segments: Vec<Found3d>,
    pub notes: Vec<String>,
}

/// Read the label map at `path` onto `vol`'s grid (nearest voxel by position when the grids
/// differ).
pub fn read_label_map(path: &Path, vol: &LoadedVolume) -> anyhow::Result<Imported> {
    let size = std::fs::metadata(path)?.len();
    let v = match classify(path, size) {
        Found::Volume(v) => v,
        Found::Problem(p) => anyhow::bail!("cannot read {}: {p}", path.display()),
        _ => anyhow::bail!("{} is not a label map Polymix reads (.seg.nrrd, .nrrd, .nii, .nii.gz)", path.display()),
    };
    let map = mm_volume::load::load_volume(&v, 2_000_000_000, &|_| true)?;
    let mut notes = Vec::new();
    let to_label = |x: f32| if x.is_finite() { x.round().clamp(0.0, 255.0) as u8 } else { 0 };
    let same = map.dims == vol.dims && (map.geometry.matrix() - vol.geometry.matrix()).to_cols_array().iter().all(|d| d.abs() < 1e-3);
    let labels: Vec<u8> = if same {
        map.values.par_iter().map(|&x| to_label(x)).collect()
    } else {
        notes.push(format!("resampled from its {}×{}×{} grid onto the volume's", map.dims[0], map.dims[1], map.dims[2]));
        let to_map = map.geometry.matrix().inverse() * vol.geometry.matrix();
        let [nx, ny, _] = vol.dims;
        (0..vol.voxel_count())
            .into_par_iter()
            .map(|n| {
                let p = to_map.transform_point3(DVec3::new((n % nx) as f64, ((n / nx) % ny) as f64, (n / nx / ny) as f64)).round();
                let inside = (0..3).all(|a| p[a] >= 0.0 && (p[a] as usize) < map.dims[a]);
                if inside { to_label(map.get(p.x as usize, p.y as usize, p.z as usize)) } else { 0 }
            })
            .collect()
    };
    let mut present = [false; 256];
    for &l in &labels {
        present[l as usize] = true;
    }
    let mut segments = Vec::new();
    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    if name.ends_with(".nrrd") || name.ends_with(".nhdr") {
        // 3D Slicer's segment fields.
        let h = mm_volume::nrrd::read(path, 1 << 22)?;
        for n in 0..256 {
            let key = |f: &str| h.keys.get(&format!("Segment{n}_{f}")).cloned();
            let Some(seg_name) = key("Name").or_else(|| key("ID")) else { continue };
            if key("Layer").is_some_and(|l| l.trim() != "0") {
                notes.push(format!("{seg_name} lies in another layer (overlapping segments) and was left out"));
                continue;
            }
            let label = key("LabelValue").and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(n as u8 + 1);
            let colour = key("Color").and_then(|c| {
                let v: Vec<f64> = c.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                (v.len() == 3).then(|| Color32::from_rgb((v[0] * 255.0).round() as u8, (v[1] * 255.0).round() as u8, (v[2] * 255.0).round() as u8))
            });
            let status = key("Tags").and_then(|t| t.split('|').find_map(|kv| kv.strip_prefix("Segmentation.Status:").map(str::to_string))).unwrap_or_default();
            segments.push(Found3d { label, name: seg_name, colour, status });
        }
    } else if let Some(d) = v.fields.iter().find(|f| f.name.eq_ignore_ascii_case("descrip") || f.name.to_lowercase().contains("description")) {
        // Names as Polymix writes them into NIfTI: "1=Aorta 2=Branch".
        let text = d.value.clone();
        let starts: Vec<(usize, u8)> = text
            .match_indices('=')
            .filter_map(|(i, _)| {
                let head = &text[..i];
                let digits = head.chars().rev().take_while(|c| c.is_ascii_digit()).count();
                let at = i - digits;
                (digits > 0 && (at == 0 || text[..at].ends_with(' '))).then(|| text[at..i].parse::<u8>().ok().map(|l| (at, l))).flatten()
            })
            .collect();
        for (k, &(at, l)) in starts.iter().enumerate() {
            let from = at + l.to_string().len() + 1;
            let to = starts.get(k + 1).map_or(text.len(), |s| s.0);
            let seg_name = text[from..to].trim().to_string();
            if !seg_name.is_empty() {
                segments.push(Found3d { label: l, name: seg_name, colour: None, status: String::new() });
            }
        }
    }
    // Labels in the map without a description, by number.
    for l in 1..=255u8 {
        if present[l as usize] && !segments.iter().any(|s| s.label == l) {
            segments.push(Found3d { label: l, name: format!("Label {l}"), colour: None, status: String::new() });
        }
    }
    segments.retain(|s| present[s.label as usize]);
    if segments.is_empty() {
        anyhow::bail!("{} has no labelled voxels on the volume's grid", path.display());
    }
    Ok(Imported { labels, segments, notes })
}

/// The voxels of `vol`'s grid inside the closed `mesh` (by winding number), its coordinates
/// in LPS millimetres (as 3D Slicer's models and Polymix's Make mesh have them) or RAS:
/// whichever overlaps the volume more, said in the note.
pub fn voxelise(mesh: &Mesh, name: &str, vol: &LoadedVolume) -> anyhow::Result<Imported> {
    if mesh.triangle_count() == 0 {
        anyhow::bail!("the mesh is empty");
    }
    let b = mesh.bounds();
    let ras_box = |lps: bool| {
        let f = |p: DVec3| if lps { DVec3::new(-p.x, -p.y, p.z) } else { p };
        let (a, c) = (f(b.min), f(b.max));
        (a.min(c), a.max(c))
    };
    // The volume's box in RAS.
    let corners: Vec<DVec3> = (0..8).map(|c| vol.geometry.world(DVec3::new(if c & 1 == 0 { 0.0 } else { vol.dims[0] as f64 - 1.0 }, if c & 2 == 0 { 0.0 } else { vol.dims[1] as f64 - 1.0 }, if c & 4 == 0 { 0.0 } else { vol.dims[2] as f64 - 1.0 }))).collect();
    let (vlo, vhi) = corners.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(a, c), &p| (a.min(p), c.max(p)));
    let overlap = |(lo, hi): (DVec3, DVec3)| {
        let d = (hi.min(vhi) - lo.max(vlo)).max(DVec3::ZERO);
        d.x * d.y * d.z
    };
    let lps = overlap(ras_box(true)) >= overlap(ras_box(false));
    if overlap(ras_box(lps)) <= 0.0 {
        anyhow::bail!("the mesh lies outside the volume (in LPS and in RAS coordinates)");
    }
    let wn = mm_solid::WindingNumber::new(mesh);
    // Only voxels within the mesh's box.
    let (lo, hi) = ras_box(lps);
    let inv = vol.geometry.matrix().inverse();
    let box_corners: Vec<DVec3> = (0..8).map(|c| inv.transform_point3(DVec3::new(if c & 1 == 0 { lo.x } else { hi.x }, if c & 2 == 0 { lo.y } else { hi.y }, if c & 4 == 0 { lo.z } else { hi.z }))).collect();
    let (ilo, ihi) = box_corners.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(a, c), &p| (a.min(p), c.max(p)));
    let first = [0, 1, 2].map(|a| ilo[a].floor().max(0.0) as usize);
    let last = [0, 1, 2].map(|a| (ihi[a].ceil().max(0.0) as usize).min(vol.dims[a] - 1));
    let [nx, ny, _] = vol.dims;
    let mut labels = vec![0u8; vol.voxel_count()];
    labels.par_chunks_mut(nx * ny).enumerate().filter(|(k, _)| (first[2]..=last[2]).contains(k)).for_each(|(k, plane)| {
        for j in first[1]..=last[1] {
            for i in first[0]..=last[0] {
                let p = vol.geometry.world(DVec3::new(i as f64, j as f64, k as f64));
                let q = if lps { DVec3::new(-p.x, -p.y, p.z) } else { p };
                if wn.eval(q).abs() >= 0.5 {
                    plane[j * nx + i] = 1;
                }
            }
        }
    });
    if !labels.iter().any(|&l| l != 0) {
        anyhow::bail!("no voxel of the volume lies inside the mesh (is it closed?)");
    }
    let notes = vec![format!("as {} coordinates", if lps { "LPS (3D Slicer's models)" } else { "RAS" })];
    Ok(Imported { labels, segments: vec![Found3d { label: 1, name: name.to_string(), colour: None, status: String::new() }], notes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::geometry::Geometry;

    fn volume() -> LoadedVolume {
        let g = Geometry::from_axes([20, 20, 20, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        LoadedVolume::from_values("t", [20, 20, 20], vec![0.0; 8000], g)
    }

    #[test]
    fn label_maps_read_back_with_their_names() {
        let vol = volume();
        let dir = std::env::temp_dir().join(format!("polymix-import-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut labels = vec![0u8; 8000];
        for (n, l) in labels.iter_mut().enumerate() {
            let [i, j, k] = [n % 20, (n / 20) % 20, n / 400];
            if (2..6).contains(&i) && (2..6).contains(&j) && (2..6).contains(&k) {
                *l = 3;
            } else if (10..12).contains(&i) && (10..12).contains(&j) && (10..12).contains(&k) {
                *l = 7;
            }
        }
        // Slicer's .seg.nrrd: names, colours, statuses.
        let infos = [mm_segment::SegmentInfo { name: "Aorta".into(), colour: [200, 40, 40], label: 3, status: "completed".into() }, mm_segment::SegmentInfo { name: "Branch".into(), colour: [40, 40, 200], label: 7, status: String::new() }];
        let seg = dir.join("t.seg.nrrd");
        mm_segment::write_seg_nrrd(&seg, &labels, [20, 20, 20], &vol.geometry, &infos).unwrap();
        let r = read_label_map(&seg, &vol).unwrap();
        assert_eq!(r.labels, labels);
        assert_eq!(r.segments.len(), 2);
        assert_eq!((r.segments[0].label, r.segments[0].name.as_str(), r.segments[0].status.as_str()), (3, "Aorta", "completed"));
        assert_eq!(r.segments[1].colour, Some(Color32::from_rgb(40, 40, 200)));
        assert!(r.notes.is_empty());
        // NIfTI: names from the description.
        let nii = dir.join("t.nii.gz");
        mm_segment::write_nifti(&nii, &labels, [20, 20, 20], &vol.geometry, "3=Left sinus 7=Branch").unwrap();
        let r = read_label_map(&nii, &vol).unwrap();
        assert_eq!(r.segments.iter().map(|s| (s.label, s.name.clone())).collect::<Vec<_>>(), vec![(3, "Left sinus".to_string()), (7, "Branch".to_string())]);
        // Another grid (2 mm voxels over the same space): resampled by position.
        let g2 = Geometry::from_axes([10, 10, 10, 1], [2.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let coarse: Vec<u8> = (0..1000).map(|n| ((n % 10) < 3 && ((n / 10) % 10) < 3 && n / 100 < 3) as u8 * 5).collect();
        let nii2 = dir.join("coarse.nii");
        mm_segment::write_nifti(&nii2, &coarse, [10, 10, 10], &g2, "").unwrap();
        let r = read_label_map(&nii2, &vol).unwrap();
        assert!(!r.notes.is_empty() && r.segments[0].name == "Label 5");
        assert!(r.labels[0] == 5 && r.labels[(4 * 20 + 4) * 20 + 4] == 5 && r.labels[(6 * 20 + 6) * 20 + 6] == 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn meshes_voxelise_in_slicer_coordinates() {
        let vol = volume();
        // A cube at RAS (5–9, 5–9, 5–9), given in LPS as Slicer models are.
        let mut cube = mm_core::primitives::cube(4.0, 4);
        let lps = DVec3::new(-7.0, -7.0, 7.0);
        for v in cube.vertex_ids().collect::<Vec<_>>() {
            cube.set_vertex(v, cube.vertex(v) + lps);
        }
        let r = voxelise(&cube, "cube", &vol).unwrap();
        assert!(r.notes[0].contains("LPS"), "{:?}", r.notes);
        let n = r.labels.iter().filter(|&&l| l != 0).count();
        assert!((60..=130).contains(&n), "about 4³ voxels: {n}");
        assert!(r.labels[(7 * 20 + 7) * 20 + 7] == 1 && r.labels[0] == 0);
        // Far outside: refused.
        for v in cube.vertex_ids().collect::<Vec<_>>() {
            cube.set_vertex(v, cube.vertex(v) + DVec3::splat(500.0));
        }
        assert!(voxelise(&cube, "cube", &vol).is_err());
    }
}
