//! Region reduce must leave every triangle outside its region untouched, also where the
//! collapses cost nothing (flat areas, as in the AdaptiveReduce sculpt brush).

mod common;

use common::*;
use mm_core::{primitives, DVec3};
use mm_remesh::*;

#[test]
fn flat_grid_region_reduce_leaves_outside_unchanged() {
    let grid = primitives::plane_grid(4.0, 4.0, 40, 40);
    // A curved copy too: only the flat one has zero-cost collapses right at the border.
    let mut bumpy = grid.clone();
    for v in bumpy.vertex_ids().collect::<Vec<_>>() {
        let p = bumpy.vertex(v);
        bumpy.set_vertex(v, p + DVec3::Z * (0.2 * (3.0 * p.x).sin() * (2.0 * p.y).cos()));
    }
    for mesh in [grid, bumpy] {
        let disc: Vec<u32> = mesh.triangle_ids().filter(|&t| mesh.tri_centroid(t).truncate().length() < 1.2).collect();
        let square: Vec<u32> = mesh.triangle_ids().filter(|&t| mesh.tri_centroid(t).x.abs() < 0.9 && mesh.tri_centroid(t).y > 0.0).collect();
        for region in [disc, square] {
            for params in [
                // The AdaptiveReduce brush.
                ReduceParams { target: ReduceTarget::Percentage(60.0), max_normal_change_deg: 6.0, ..Default::default() },
                ReduceParams::with_target(ReduceTarget::Percentage(20.0)),
                ReduceParams { uniform: true, ..ReduceParams::with_target(ReduceTarget::Percentage(30.0)) },
                ReduceParams { preserve_boundary: false, ..ReduceParams::with_target(ReduceTarget::TriangleCount(10)) },
                ReduceParams::with_target(ReduceTarget::MaxDeviation(0.05)),
            ] {
                let mut m = mesh.clone();
                let before = snapshot_outside(&m, &region);
                let rep = reduce(&mut m, &ReduceParams { region: Some(region.clone()), ..params.clone() }, None).unwrap();
                assert_valid(&m);
                assert!(rep.collapses > 0, "{params:?}: nothing reduced");
                let new_region = rep.region_triangles.clone().unwrap();
                tri_set(&m, &new_region);
                assert_eq!(rep.final_triangles, new_region.len());
                assert_eq!(snapshot_outside(&m, &new_region), before, "{params:?}: triangles outside the region changed");
                assert_eq!(m.triangle_count() - new_region.len(), mesh.triangle_count() - region.len());
            }
        }
    }
}
