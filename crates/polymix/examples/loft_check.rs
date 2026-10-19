//! Loft the two closest openings of a mesh and measure the tube (for tuning `mm_edit::loft`).
//!
//! `cargo run --release --example loft_check -- <mesh> [key=value …]`. Keys: `context`,
//! `fair`, `round`, `blend` (0..1), `end=g0|g1|g2`, `tension`, `dir=wall|normal`,
//! `helix=R,rise` (the true vessel axis of a test mesh: an arc of radius R about z rising
//! `rise` per radian, for the distance column), `out=<file>` to save the result.
//!
//! Prints the tube's self-crossings, the sharpest bends between neighbouring triangles on
//! the tube and across its seams (degrees), and with `helix` how far the tube strays from
//! the vessel's true wall.

use mm_core::{DVec3, Mesh};
use std::collections::HashSet;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| anyhow::anyhow!("usage: loft_check <mesh> [key=value …]"))?;
    let mut mesh = mm_io::read_mesh(std::path::Path::new(path))?.mesh;
    let mut params = mm_edit::LoftParams::default();
    let mut helix = None;
    let mut out = None;
    for a in &args[1..] {
        let (k, v) = a.split_once('=').unwrap_or((a, ""));
        let x = v.parse::<f64>().ok();
        match k {
            "fair" => params.fair = x.unwrap_or(0.0),
            "context" => params.context = x.unwrap_or(0.0),
            "blend" => params.blend = x.unwrap_or(0.0),
            "remesh" => params.remesh = v != "0",
            "round" => params.round = x.unwrap_or(0.0),
            "tension" => params.ends.iter_mut().for_each(|e| e.tension = x.unwrap_or(1.0)),
            "end" => {
                let c = match v {
                    "g0" => mm_edit::EndContinuity::G0,
                    "g2" => mm_edit::EndContinuity::G2,
                    _ => mm_edit::EndContinuity::G1,
                };
                params.ends.iter_mut().for_each(|e| e.continuity = c);
            }
            "dir" => params.ends.iter_mut().for_each(|e| e.direction = if v == "normal" { mm_edit::EndDirection::Normal } else { mm_edit::EndDirection::Wall }),
            "helix" => {
                let n: Vec<f64> = v.split(',').filter_map(|s| s.parse().ok()).collect();
                helix = Some((n[0], n[1]));
            }
            "out" => out = Some(v.to_string()),
            _ => anyhow::bail!("unknown key {k}"),
        }
    }
    // The two openings closest to each other.
    let loops = mm_core::topology::boundary_loops(&mesh);
    let centre = |l: &mm_core::topology::EdgeLoop| l.vertices.iter().map(|&v| mesh.vertex(v)).sum::<DVec3>() / l.vertices.len() as f64;
    let mut best = (f64::MAX, 0, 0);
    for i in 0..loops.len() {
        for j in i + 1..loops.len() {
            let d = centre(&loops[i]).distance(centre(&loops[j]));
            if d < best.0 {
                best = (d, i, j);
            }
        }
    }
    let (a, b) = (loops[best.1].vertices.clone(), loops[best.2].vertices.clone());
    let started = std::time::Instant::now();
    let r = mm_edit::loft(&mut mesh, &a, &b, &params)?;
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let tube: HashSet<u32> = r.triangles.iter().copied().collect();
    // Seams: the original triangles touching the tube.
    let tube_verts: HashSet<u32> = r.triangles.iter().flat_map(|&t| mesh.triangle(t)).collect();
    let seam: HashSet<u32> = tube_verts.iter().flat_map(|&v| mesh.vertex_triangles(v)).filter(|t| !tube.contains(t)).collect();
    let bends = |set: &dyn Fn(u32) -> bool, both: bool| {
        let mut angles: Vec<f64> = mesh
            .edge_ids()
            .filter_map(|e| {
                let ts = mesh.edge_triangles(e);
                if ts.len() != 2 || ts[1] == mm_core::INVALID {
                    return None;
                }
                let (x, y) = (set(ts[0]), set(ts[1]));
                (if both { x && y } else { x != y }).then(|| mesh.tri_normal(ts[0]).angle_between(mesh.tri_normal(ts[1])).to_degrees())
            })
            .collect();
        angles.sort_by(f64::total_cmp);
        let q = |f: f64| angles.get(((angles.len() as f64 - 1.0) * f) as usize).copied().unwrap_or(0.0);
        (q(0.95), q(1.0))
    };
    let (t95, tmax) = bends(&|t| tube.contains(&t), true);
    // The tube's first and last bands (triangles touching an opening's vertices) apart.
    let ends: HashSet<u32> = seam.iter().flat_map(|&t| mesh.triangle(t)).collect();
    let band = |t: u32| mesh.triangle(t).iter().any(|v| ends.contains(v));
    let (b95, bmax) = bends(&|t| tube.contains(&t) && band(t), true);
    let (m95, mmax) = bends(&|t| tube.contains(&t) && !band(t), true);
    print!("[end bands p95={b95:.1}° max={bmax:.1}° | middle p95={m95:.1}° max={mmax:.1}°] ");
    // Folds (> 90°) by where they are: inside the tube, tube to vessel, inside the vessel near it.
    let near: HashSet<u32> = seam.iter().flat_map(|&t| mesh.triangle(t)).flat_map(|v| mesh.vertex_triangles(v)).collect();
    let (mut in_tube, mut across, mut in_vessel) = (0, 0, 0);
    for e in mesh.edge_ids() {
        let ts = mesh.edge_triangles(e);
        if ts[1] == mm_core::INVALID || mesh.tri_normal(ts[0]).angle_between(mesh.tri_normal(ts[1])).to_degrees() < 90.0 {
            continue;
        }
        match (tube.contains(&ts[0]), tube.contains(&ts[1])) {
            (true, true) => in_tube += 1,
            (true, false) | (false, true) => across += 1,
            _ if near.contains(&ts[0]) => in_vessel += 1,
            _ => {}
        }
    }
    print!("folds: tube {in_tube} seam {across} vessel {in_vessel}  ");
    let (s95, smax) = bends(&|t| tube.contains(&t) || seam.contains(&t), false);
    print!("tris={} rows≈{} {ms:.0} ms  crossings={}  tube bend p95={t95:.1}° max={tmax:.1}°  seam p95={s95:.1}° max={smax:.1}°", r.triangles.len(), r.new_vertices.len() / a.len().max(1), crossings(&mesh, &r.triangles));
    if let Some((rr, rise)) = helix {
        // Distance from the true axis, against the wall's mean radius around the gap.
        let axis = |p: DVec3| {
            let mut best = f64::MAX;
            for k in 0..=4000 {
                let t = k as f64 / 4000.0 * 2.2;
                best = best.min(p.distance(DVec3::new(rr * t.cos(), rr * t.sin(), rise * t)));
            }
            best
        };
        let d: Vec<f64> = r.new_vertices.iter().map(|&v| axis(mesh.vertex(v))).collect();
        let ring: Vec<f64> = ends.iter().map(|&v| axis(mesh.vertex(v))).collect();
        let r0 = ring.iter().sum::<f64>() / ring.len() as f64;
        let dev: Vec<f64> = d.iter().map(|x| (x - r0).abs()).collect();
        let mean = dev.iter().sum::<f64>() / dev.len().max(1) as f64;
        print!("  wall deviation mean={mean:.2} max={:.2} (radius {r0:.2})", dev.iter().copied().fold(0.0, f64::max));
    }
    let mut holes: Vec<usize> = mm_core::topology::boundary_loops(&mesh).iter().map(|l| l.vertices.len()).collect();
    holes.sort_unstable();
    print!("  open loops {holes:?} valid={}", mesh.check_validity().is_ok());
    println!();
    if let Some(o) = out {
        mm_io::write_mesh(std::path::Path::new(&o), &mesh, &Default::default())?;
    }
    Ok(())
}

/// Pairs of tube triangles, or tube and other triangles, that cut through each other.
fn crossings(m: &Mesh, part: &[u32]) -> usize {
    let bvh = mm_core::Bvh::build_subset(m, part);
    let inside: HashSet<u32> = part.iter().copied().collect();
    let bounds = mm_core::Aabb::from_points(part.iter().flat_map(|&t| m.tri_positions(t)));
    m.triangle_ids()
        .filter(|&t| mm_core::Aabb::from_points(m.tri_positions(t)).intersects(&bounds))
        .map(|t| {
            let (tv, tp) = (m.triangle(t), m.tri_positions(t));
            let mut n = 0;
            bvh.query_aabb(&mm_core::Aabb::from_points(tp), |o| {
                if (inside.contains(&t) && o <= t) || m.triangle(o).iter().any(|v| tv.contains(v)) {
                    return;
                }
                if mm_core::intersect::triangles_intersect(tp, m.tri_positions(o)) {
                    n += 1;
                }
            });
            n
        })
        .sum()
}
