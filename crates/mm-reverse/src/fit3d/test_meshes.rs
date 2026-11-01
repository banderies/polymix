// SPDX-License-Identifier: AGPL-3.0-only
//! Meshes the seeded fits are tried on beyond the plain shapes of `testkit`: the sample
//! L-shaped bracket (unposed), a hollow cup, a plate with a hole and a block with a stepped
//! bore. Test-only.
//!
//! Ported from ScanRuler (`src/core/geometry/samplePart.ts` and `tests/guideMeshes.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only. The sample part's skew pose is not ported:
//! the tests take it flat in its own frame.

use std::collections::HashMap;

use glam::DVec3;

use super::testkit::p32;
use crate::fit2d::Mulberry32;

type P3 = [f64; 3];

/// The L's profile, millimetres: a 40 × 30 outline with legs 12 thick.
pub const SAMPLE_WIDTH: f64 = 40.0;
pub const SAMPLE_HEIGHT: f64 = 30.0;
pub const SAMPLE_LEG: f64 = 12.0;
/// How far the profile is extruded.
pub const SAMPLE_DEPTH: f64 = 20.0;

fn inside(x: i64, y: i64, z: i64) -> bool {
    if x < 0 || y < 0 || z < 0 || x >= SAMPLE_WIDTH as i64 || y >= SAMPLE_HEIGHT as i64 || z >= SAMPLE_DEPTH as i64 {
        return false;
    }
    y < SAMPLE_LEG as i64 || x < SAMPLE_LEG as i64
}

/// The sample L-shaped bracket as a mesh, in its own frame: millimetre cells, the boundary
/// faces of the cells emitted as quads over vertices shared by their integer corner, which
/// makes the mesh watertight by construction.
pub fn sample_part() -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let nx = SAMPLE_WIDTH as i64 + 1;
    let ny = SAMPLE_HEIGHT as i64 + 1;
    let mut ids: HashMap<i64, u32> = HashMap::new();
    let mut xyz: Vec<DVec3> = Vec::new();
    let mut tri: Vec<[u32; 3]> = Vec::new();
    let mut vertex = |x: i64, y: i64, z: i64, xyz: &mut Vec<DVec3>| -> u32 {
        let key = (z * ny + y) * nx + x;
        *ids.entry(key).or_insert_with(|| {
            xyz.push(DVec3::new(x as f64, y as f64, z as f64));
            xyz.len() as u32 - 1
        })
    };
    // A unit quad at a cell's face, its corners wound to face along +axis when `positive`,
    // the other way otherwise.
    let mut quad = |o: [i64; 3], u: [i64; 3], v: [i64; 3], positive: bool, xyz: &mut Vec<DVec3>, tri: &mut Vec<[u32; 3]>| {
        let a = vertex(o[0], o[1], o[2], xyz);
        let b = vertex(o[0] + u[0], o[1] + u[1], o[2] + u[2], xyz);
        let c = vertex(o[0] + u[0] + v[0], o[1] + u[1] + v[1], o[2] + u[2] + v[2], xyz);
        let d = vertex(o[0] + v[0], o[1] + v[1], o[2] + v[2], xyz);
        if positive {
            tri.extend([[a, b, c], [a, c, d]]);
        } else {
            tri.extend([[a, c, b], [a, d, c]]);
        }
    };
    let (ex, ey, ez) = ([1, 0, 0], [0, 1, 0], [0, 0, 1]);
    for z in 0..SAMPLE_DEPTH as i64 {
        for y in 0..SAMPLE_HEIGHT as i64 {
            for x in 0..SAMPLE_WIDTH as i64 {
                if !inside(x, y, z) {
                    continue;
                }
                // u × v is the +axis in each case, so `positive` is the outward side.
                if !inside(x + 1, y, z) {
                    quad([x + 1, y, z], ey, ez, true, &mut xyz, &mut tri);
                }
                if !inside(x - 1, y, z) {
                    quad([x, y, z], ey, ez, false, &mut xyz, &mut tri);
                }
                if !inside(x, y + 1, z) {
                    quad([x, y + 1, z], ez, ex, true, &mut xyz, &mut tri);
                }
                if !inside(x, y - 1, z) {
                    quad([x, y, z], ez, ex, false, &mut xyz, &mut tri);
                }
                if !inside(x, y, z + 1) {
                    quad([x, y, z + 1], ex, ey, true, &mut xyz, &mut tri);
                }
                if !inside(x, y, z - 1) {
                    quad([x, y, z], ex, ey, false, &mut xyz, &mut tri);
                }
            }
        }
    }
    (xyz, tri)
}

fn push(out: &mut Vec<DVec3>, ps: &[P3]) {
    for p in ps {
        out.push(p32(p[0], p[1], p[2]));
    }
}

/// A grid of points as a triangle soup, each quad split in two; `flip` reverses the winding.
fn grid_soup(out: &mut Vec<DVec3>, grid: &[Vec<P3>], wrap: bool, flip: bool) {
    for i in 0..grid.len().saturating_sub(1) {
        let n = grid[i].len();
        let cols = if wrap { n } else { n - 1 };
        for j in 0..cols {
            let j2 = (j + 1) % n;
            let (a, b, c, d) = (grid[i][j], grid[i + 1][j], grid[i + 1][j2], grid[i][j2]);
            if flip {
                push(out, &[a, c, b, a, d, c]);
            } else {
                push(out, &[a, b, c, a, c, d]);
            }
        }
    }
}

const TAU: f64 = 2.0 * std::f64::consts::PI;

/// A hollow hemispherical cup, open at the top: outside R `outer`, inside R `inner`, a flat
/// rim at z = 0 between them. Radial noise of `noise` (ScanRuler's default 0.004) off the
/// rim.
pub fn cup_mesh(inner: f64, outer: f64, noise: f64) -> Vec<DVec3> {
    let mut rand = Mulberry32::new(5);
    const PHI: usize = 72;
    const TH: usize = 28;
    let mut shell = |r_shell: f64| -> Vec<Vec<P3>> {
        (0..=TH)
            .map(|i| {
                let th = (i as f64 / TH as f64) * (std::f64::consts::PI / 2.0);
                (0..PHI)
                    .map(|j| {
                        let ph = (j as f64 / PHI as f64) * TAU;
                        let r = if i == TH { r_shell } else { r_shell + rand.gaussian() * noise };
                        [r * th.sin() * ph.cos(), r * th.sin() * ph.sin(), if i == TH { 0.0 } else { -r * th.cos() }]
                    })
                    .collect()
            })
            .collect()
    };
    let outer_shell = shell(outer);
    let inner_shell = shell(inner);
    let mut tris = Vec::new();
    grid_soup(&mut tris, &outer_shell, true, true);
    grid_soup(&mut tris, &inner_shell, true, false);
    let rim: Vec<Vec<P3>> = (0..=6)
        .map(|i| {
            let rho = inner + ((outer - inner) * i as f64) / 6.0;
            (0..PHI).map(|j| [rho * ((j as f64 / PHI as f64) * TAU).cos(), rho * ((j as f64 / PHI as f64) * TAU).sin(), 0.0]).collect()
        })
        .collect();
    grid_soup(&mut tris, &rim, true, false);
    tris
}

/// A square plate, side `side`, `t` thick about z = 0, with a round hole of radius `r`
/// through its middle along z. Radial noise of `noise` (ScanRuler's default 0.004) on the
/// bore's wall.
pub fn plated_hole(r: f64, side: f64, t: f64, noise: f64) -> Vec<DVec3> {
    let mut rand = Mulberry32::new(9);
    const PHI: usize = 64;
    const RINGS: usize = 10;
    let half = side / 2.0;
    let edge = |ph: f64| half / ph.cos().abs().max(ph.sin().abs());
    let face = |z: f64| -> Vec<Vec<P3>> {
        (0..=RINGS)
            .map(|i| {
                (0..PHI)
                    .map(|j| {
                        let ph = (j as f64 / PHI as f64) * TAU;
                        let rho = r + ((edge(ph) - r) * i as f64) / RINGS as f64;
                        [rho * ph.cos(), rho * ph.sin(), z]
                    })
                    .collect()
            })
            .collect()
    };
    let mut wall = |rad: &dyn Fn(f64) -> f64, wobble: bool| -> Vec<Vec<P3>> {
        const STEPS: usize = 8;
        (0..=STEPS)
            .map(|i| {
                let z = -t / 2.0 + (t * i as f64) / STEPS as f64;
                let end = i == 0 || i == STEPS;
                (0..PHI)
                    .map(|j| {
                        let ph = (j as f64 / PHI as f64) * TAU;
                        let rho = rad(ph) + if wobble && !end { rand.gaussian() * noise } else { 0.0 };
                        [rho * ph.cos(), rho * ph.sin(), z]
                    })
                    .collect()
            })
            .collect()
    };
    let mut tris = Vec::new();
    grid_soup(&mut tris, &face(t / 2.0), true, false);
    grid_soup(&mut tris, &face(-t / 2.0), true, true);
    let bore = wall(&|_| r, true);
    grid_soup(&mut tris, &bore, true, false);
    let outside = wall(&edge, false);
    grid_soup(&mut tris, &outside, true, true);
    tris
}

/// The stepped bore's shape: see `stepped_bore`.
#[derive(Debug, Clone, Copy)]
pub struct SteppedBore {
    pub r1: f64,
    pub l1: f64,
    pub r2: f64,
    pub l2: f64,
    pub noise: f64,
    pub offset: f64,
    pub plug: f64,
    pub taper: f64,
    pub side: f64,
}

impl SteppedBore {
    /// A bore of radius `r1` for `l1`, then `r2` for `l2`, with ScanRuler's defaults for the
    /// rest: noise 0.004, no offset, plug or taper, a 30 mm block.
    pub fn new(r1: f64, l1: f64, r2: f64, l2: f64) -> Self {
        Self { r1, l1, r2, l2, noise: 0.004, offset: 0.0, plug: 0.0, taper: 0.0, side: 30.0 }
    }
}

/// A block `side` square with a bore down z from its top face at z = 0: radius `r1` for
/// `l1`, then radius `r2` for `l2` — a bore stepping wider below a flange, `offset` moving
/// the wider one's axis along x. With `plug`, a solid that thick closes the bore at the step
/// instead, capped flat on both sides, as a wrapped scan leaves a hole it did not see
/// through. `taper` runs the step as a cone that long down the bore rather than a flat
/// shoulder — a cast step's draft, which a region grows into. Radial noise on the bore walls.
pub fn stepped_bore(o: SteppedBore) -> Vec<DVec3> {
    let mut rand = Mulberry32::new(13);
    const PHI: usize = 144;
    const RINGS: usize = 12;
    let noise = o.noise;
    let off = o.offset;
    let plug = o.plug;
    let half = o.side / 2.0;
    let z1 = -o.l1;
    let z2 = z1 - if plug != 0.0 { plug } else { o.taper };
    let z3 = z2 - o.l2;
    let ang = |j: usize| (j as f64 / PHI as f64) * TAU;
    let edge = |j: usize| -> [f64; 2] {
        let ph = ang(j);
        let rho = half / ph.cos().abs().max(ph.sin().abs());
        [rho * ph.cos(), rho * ph.sin()]
    };
    let circle = |cx: f64, r: f64| move |j: usize| -> [f64; 2] { [cx + r * ang(j).cos(), r * ang(j).sin()] };
    // Rings from one loop to another at one height.
    let between = |from: &dyn Fn(usize) -> [f64; 2], to: &dyn Fn(usize) -> [f64; 2], z: f64, rings: usize| -> Vec<Vec<P3>> {
        (0..=rings)
            .map(|i| {
                (0..PHI)
                    .map(|j| {
                        let (a, b) = (from(j), to(j));
                        [a[0] + ((b[0] - a[0]) * i as f64) / rings as f64, a[1] + ((b[1] - a[1]) * i as f64) / rings as f64, z]
                    })
                    .collect()
            })
            .collect()
    };
    // A wall down a loop from one height to another, rows a quarter millimetre apart.
    let wall = |rand: &mut Mulberry32, lp: &dyn Fn(usize) -> [f64; 2], cx: f64, top: f64, bottom: f64, wobble: bool| -> Vec<Vec<P3>> {
        let rows = (((top - bottom) / 0.25).round() as usize).max(1);
        (0..=rows)
            .map(|i| {
                let z = top + ((bottom - top) * i as f64) / rows as f64;
                let end = i == 0 || i == rows;
                (0..PHI)
                    .map(|j| {
                        let [x, y] = lp(j);
                        let k = if wobble && !end { 1.0 + (rand.gaussian() * noise) / (x - cx).hypot(y) } else { 1.0 };
                        [cx + (x - cx) * k, y * k, z]
                    })
                    .collect()
            })
            .collect()
    };
    let upper = circle(0.0, o.r1);
    let lower = circle(off, o.r2);
    let centre = |cx: f64| move |_: usize| -> [f64; 2] { [cx, 0.0] };
    let mut tris = Vec::new();
    grid_soup(&mut tris, &between(&upper, &edge, 0.0, RINGS), true, false);
    let w = wall(&mut rand, &upper, 0.0, 0.0, z1, true);
    grid_soup(&mut tris, &w, true, false);
    if plug > 0.0 {
        grid_soup(&mut tris, &between(&upper, &centre(0.0), z1, 8), true, true);
        grid_soup(&mut tris, &between(&lower, &centre(off), z2, 8), true, false);
    } else if o.taper != 0.0 {
        // Down the cone, the loop moving from the narrow one to the wide one.
        let rows = ((o.taper / 0.25).round() as usize).max(2);
        let cone: Vec<Vec<P3>> = (0..=rows)
            .map(|i| {
                (0..PHI)
                    .map(|j| {
                        let (a, b) = (upper(j), lower(j));
                        let f = i as f64 / rows as f64;
                        let end = i == 0 || i == rows;
                        let (x, y) = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f);
                        let cx = off * f;
                        let k = if end { 1.0 } else { 1.0 + (rand.gaussian() * noise) / (x - cx).hypot(y) };
                        [cx + (x - cx) * k, y * k, z1 + (z2 - z1) * f]
                    })
                    .collect()
            })
            .collect();
        grid_soup(&mut tris, &cone, true, false);
    } else {
        grid_soup(&mut tris, &between(&upper, &lower, z1, 2), true, true);
    }
    let w = wall(&mut rand, &lower, off, z2, z3, true);
    grid_soup(&mut tris, &w, true, false);
    grid_soup(&mut tris, &between(&lower, &edge, z3, RINGS), true, true);
    let w = wall(&mut rand, &edge, 0.0, 0.0, z3, false);
    grid_soup(&mut tris, &w, true, true);
    tris
}
