// SPDX-License-Identifier: AGPL-3.0-only
//! Fit all chains on a section through a mesh file, and print what came out — for checking
//! the segmentation on real scans: `cargo run -p mm-reverse --example fit_slice -- part.stl z 0`.

use glam::DVec3;
use mm_reverse::frame::{world_cut_axis, SectionFrame, WorldAxis};
use mm_reverse::section::{fit_all_chains, section_chains, FitOptions};
use mm_reverse::sketch::{Shape, Sketch};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let axis = match args.get(2).map(String::as_str) {
        Some("x") => WorldAxis::X,
        Some("y") => WorldAxis::Y,
        _ => WorldAxis::Z,
    };
    let offset: f64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let mesh = mm_io::read_mesh(std::path::Path::new(path)).expect("readable mesh").mesh;
    let (pos, tris, _) = mesh.to_arrays();
    let b = mesh.bounds();
    let centre = (b.min + b.max) / 2.0;
    println!("bounds {:?} .. {:?}", b.min, b.max);
    let frame = SectionFrame::along(world_cut_axis(axis, DVec3::ZERO), centre.dot(axis.dir()) + offset, None).unwrap();
    let chains = section_chains(&pos, &tris, &frame);
    for (i, c) in chains.iter().enumerate() {
        println!("chain {i}: {} points", c.len());
    }
    if let Some(range) = std::env::var("POINTS").ok() {
        let v: Vec<usize> = range.split(',').filter_map(|x| x.parse().ok()).collect();
        let view = mm_reverse::chain::view_chain(&chains[v[0]]);
        for i in v[1]..=v[2] {
            println!("  pt {i}: {:.3},{:.3}", view.points[i].x, view.points[i].y);
        }
        let keep = mm_reverse::simplify::simplify_run(&view.points, v[1], v[2] + 1, 0.1842);
        println!("  rdp keeps {keep:?}");
    }
    if std::env::var_os("SEGMENTS").is_some() {
        for (i, c) in chains.iter().enumerate() {
            let Ok(cut) = mm_reverse::segment::segment_chain(c, Default::default()) else { println!("chain {i}: no fit"); continue };
            println!("chain {i}: closed {} scatter {:.4} spacing {:.4} tol {:.4}", cut.closed, cut.scatter, cut.spacing, cut.tolerance);
            for seg in &cut.segments {
                let (a, b) = (seg.points[0], seg.points[seg.points.len() - 1]);
                println!("  {:?} [{}..{}] {} pts {:.2},{:.2} -> {:.2},{:.2} corner_after {}", seg.piece.kind(), seg.lo, seg.hi, seg.points.len(), a.x, a.y, b.x, b.y, seg.corner_after);
            }
        }
    }
    let mut s = Sketch::default();
    let r = fit_all_chains(&mut s, &chains, &frame, FitOptions::default(), &[]);
    // How far each line runs past its own slice points: a piece carried off by a corner.
    let overhang = s.entities.iter().filter(|e| !e.points.is_empty()).filter_map(|e| match e.shape {
        Shape::Line { a, b } => {
            let d = (b - a).normalize_or_zero();
            let ts: Vec<f64> = e.points.iter().map(|p| (*p - a).dot(d)).collect();
            let lo = ts.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = ts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Some((-lo).max(hi - (b - a).length()).max(0.0))
        }
        _ => None,
    }).fold(0.0f64, f64::max);
    let off = s.entities.iter().map(|e| e.points.iter().fold(0.0f64, |m, p| m.max(mm_reverse::sketch::nearest_on(&e.shape, *p).1))).fold(0.0f64, f64::max);
    println!("SUMMARY axis {:?} offset {offset} chains {} lines {} arcs {} open {} ok {} rms {:.4} max-overhang {:.2} worst-off {:.2}", axis, chains.len(), r.lines, r.arcs, r.open, r.solve.ok, r.solve.soft_rms, overhang, off);
    println!("lines {} arcs {} joined {} tangent {} open {} solve ok {} rms {:.4} unmet {:?}", r.lines, r.arcs, r.joined, r.tangent, r.open, r.solve.ok, r.solve.soft_rms, r.solve.unmet);
    for (id, res) in &r.solve.unmet {
        if *res > 0.01 {
            let c = s.constraint(*id).unwrap();
            println!("UNMET {id} {:?} by {res:.3}", c.rule);
        }
    }
    for e in &s.entities {
        let far = e.points.iter().fold(0.0f64, |m, p| m.max(mm_reverse::sketch::nearest_on(&e.shape, *p).1));
        match e.shape.clone() {
            Shape::Line { a, b } if std::env::var_os("OVERHANG").is_some() => {
                let d = (b - a).normalize_or_zero();
                let ts: Vec<f64> = e.points.iter().map(|p| (*p - a).dot(d)).collect();
                let lo = ts.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = ts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let o = (-lo).max(hi - (b - a).length()).max(0.0);
                if o > 2.0 { println!("OVERHANG #{} line {:.2},{:.2} -> {:.2},{:.2} len {:.2} points span {:.2}..{:.2} overhang {o:.2}", e.id, a.x, a.y, b.x, b.y, (b - a).length(), lo, hi); }
            }
            Shape::Line { a, b } => println!("#{} line {:.2},{:.2} -> {:.2},{:.2} len {:.2} pts {} worst {:.3}", e.id, a.x, a.y, b.x, b.y, (b - a).length(), e.points.len(), far),
            Shape::Arc { c, r, sweep, .. } => println!("#{} arc c {:.2},{:.2} r {:.2} sweep {:.1}° pts {} worst {:.3}", e.id, c.x, c.y, r, sweep.to_degrees(), e.points.len(), far),
            s => println!("#{} {s:?}", e.id),
        }
    }
}

#[allow(dead_code)]
fn unused() {}
