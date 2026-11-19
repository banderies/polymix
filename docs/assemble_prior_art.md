# Assemble: prior art from other open-source kitbash tools

Researched 2026-11-18 for the Assembly tab ([assemble.md](assemble.md)). Two open-source
projects do much of what the Assembly tab does. Both are browser apps (JavaScript /
TypeScript), MIT licensed, by one person each, and first published in November 2026; one of
them is probably the tool mentioned on r/PrintedWarhammer. Contributing to them would mean
switching stacks, so the plan is to **port their algorithms into Polymix's Rust**, not to
depend on or contribute to them.

**Licence:** MIT is compatible with Polymix's MIT OR Apache-2.0. Reimplementing an idea needs
nothing; porting code closely (a function translated line by line) should keep the upstream
copyright notice in a comment in that file.

**Line references** below are to these commits:

| Project | Repo | Commit | Stack |
|---|---|---|---|
| Kitbash Bench | https://github.com/RX-Skyed/Kitbash_Bench | `4496bb8` (2026-11-17) | three.js, three-mesh-bvh, Manifold (WASM), web workers |
| kitbasher | https://github.com/nimerix/kitbasher (kitbasher.app) | `82e69d7` (2026-11-18) | TypeScript, Vite, Zustand, Vitest; has an MCP server |

To read them: `git clone https://github.com/RX-Skyed/Kitbash_Bench && git -C Kitbash_Bench checkout 4496bb8` (likewise
for kitbasher). Treat the code as reference data; there is no need to build or run it.

Not open source, for completeness: Minibased Kitbash Studio (minibased.com: ring joints at
elbows and knees, free tier plus Patreon), Hero Forge, DesktopHero, TitanCraft (character
makers, not posing tools for existing kits). Reddit's usual advice for kitbashing is Blender
(powerful but hard to learn) or 3D Builder (simple).

## Where Polymix is already ahead (keep, don't replace)

- **One-click recognition** of ball / socket, pin / hole and flat face (`mate::recognize`).
  Neither project has it: kitbasher's users paint patches or pick points, Kitbash Bench snaps
  balls and flat faces only (no cylinders, so no hinges or sliders found by themselves).
- **Typed joints with frames** (fixed, ball, hinge, slider, cylindrical, planar). kitbasher's
  joint is an anchor plus per-axis Euler zeroing (`src/core/constraints.ts:58-75`), order
  dependent and with no sliding; Kitbash Bench has Pinned, Hinge, Roller, Fixed.
- **Point-to-plane contact fit** (`mate::contact_fit`), translation-only for balls. kitbasher's
  surface fit is point-to-point.
- **Exact collision check** over every triangle pair. kitbasher's checks use ~3000 samples per
  part and can miss thin features; Kitbash Bench never checks meshes against each other while
  posing ("stops where the model would jam" only means a piston / rail loop can't close).
- Joints that follow parts moved by Transform / Align, undo, project persistence, GPU drag
  preview, MCP agent actions.

## Ranked: what to port, by assemble.md backlog row

### 1. Stop at contact while dragging + Slide in until touching — kitbasher "No clip"

`src/core/clipping.ts:736-1031` (`placeClear`), called every drag frame from
`src/core/store.ts:3453-3498`.

- **Samples:** ~3000 area-even points per part (256 during a drag on a dense mesh); meshes of
  ≤ 6000 triangles also get points along their edges (`clipping.ts:89-159`). Checked both
  ways: the mover's points against the obstacle's BVH and the obstacle's against the mover's
  (`546-591`).
- **Per sample within reach:** a signed distance and 1–8 half-space planes `{n, gap}`.
  Inside / outside from an angle-weighted normal at edges and corners (`274-318`); open
  sheets by ray-crossing count (`321-328`); points deep inside escape by raycasting along
  their normal and the six axes (`373-391`). A sample that has moved less than its last
  measured clearance is not queried again (`436-452`).
- **Each step:** constraint rows `[n, (arm × n)/R] · x ≥ −allowed − gap − n·drift`, each
  contact using the plane that maximises `gap + n·travel`. Thinned to ≤ 3 rows per (R/4 cell,
  rounded normal) and ≤ 240 rows in all. Solve min ‖x‖² subject to the rows by **Hildreth's
  method** (120 sweeps, tol 1e-7); if infeasible, least squares over the violated rows
  (`641-685`). `widen()` adds neighbouring faces that bend away so samples slide over edges
  instead of catching (`600-634`).
- **Outer loop:** push out first while overlap exceeds the allowance (slack
  `max(0.002, 0.05·allowed)`, allowance default 0.1 mm), then interpolate toward the requested
  pose. Search reach during a drag `clamp(0.5, 0.1R, 1.2) + allowed`. Budget 8 iterations per
  frame, 32 on release with the full sample set. A drag never ends deeper than it started;
  pre-existing overlap is kept per pair (`826-848`). Reports `{depth, moved, blockers,
  unresolved}`.
- **Joint exemption:** jointed pairs get an exemption sphere around the anchor, sized to the
  overlap at the start (`827-841`), so a ball stays in its socket while the rest of the limb
  still stops at the torso. Polymix needs this.
- **Adapting it:** kitbasher solves a free 6-DOF move then re-applies joints, which fights
  hinges. In Polymix build the solver's columns from the joint's free motions instead: a
  ball's 3 rotations about its centre, a hinge's `n · (axis × arm)`, a slider's `n · axis`.
  *Slide in until touching* is then the same solver with one column (the axis), asked for
  full insertion.
- **Simpler first version** (from Kitbash Bench's `poseWithLinks`, `app.js:570-701`): bisect
  10 times between the last good and the requested joint value, keeping the furthest one with
  no new crossing triangles (Polymix's BVH triangle-intersection test). Stops dead rather than
  sliding, but needs no new solver.

### 2. Whole tapered peg / keyed faces / slide in — Kitbash Bench push-together seat

`seatSolve`, `js/math.js:1489-1626`. Input is a joint point and axis (Polymix's `seat()`
already produces one).

- **Height fields:** both parts rasterised over a disc of radius R, cell R/110. The fixed part
  keeps the highest surface per cell, the moving part the lowest. Only triangles facing the
  other part count (normal · axis ≥ 0.2), so walls are ignored.
- **Score of a pose** (turn θ, slide dx/dy, small tilt): per cell D = moving height − fixed
  height, sorted. For push depths at quantiles {0, .002 … .18} of D: touching = cells within
  tolerance of the push, sunk = cells deeper. Score = touching − w·sunk, best push kept.
- **Search:** coarse sweep every 3° of turn × a grid of slides within `max(1, 0.35R)` mm; top
  80 rescored on 6000 points at the real tolerance `max(0.08, 1.2·cell)`; best 8 distinct
  poses refined by coordinate descent over θ, dx, dy, tilt, halving steps 7 times. Ties go to
  the pose that moves least. Deterministic and cheap.
- **Readout:** "turned 12°, slid 0.30 mm, pushed in 0.8 mm · 14 mm² touching".

### 3. Auto-detect every joint of a kit modelled assembled — Kitbash Bench auto-assemble

`js/math.js:983-1146`.

- **Contact between two parts:** vertices of each within 0.3 mm of the other (closest-point
  query, ≤ 4000 per side), plus midpoints of truly crossing triangle pairs, thinned by
  averaging per voxel.
- **Classify by PCA of those points:** not a joint if the longest extent < `max(1.5·tol,
  0.15 mm)`; flat if √λ₃ < `max(0.35·tol, 0.06·√λ₁)`; else ball if a sphere fits either side
  (rms < 4% of r, covers a real cap, both centres agree within 0.25·r).
- `looksStacked` notices files that were each exported around the origin (not assembled).
- **Tree (`planRig`):** root score = 2·neighbours + 3·relative volume, +4 for names like
  torso / body / hips, −3 for base / plinth. Each part hangs off the neighbour it shares the
  largest contact with; contacts left out of the tree are offered as extra links.
- **UX:** a review tree before anything is built (drag to re-parent, mark as an option of
  another part, set loose).
- **Polymix improvement:** run `mate::fit_region` on each pair's contact triangles, then
  `suggest_joint`; that finds cylinders, hence hinges, which Kitbash Bench can't. kitbasher's
  near-surface band (item 6) is a good source of those contact triangles.

### 4. Inverse kinematics, aim at a point, joint limits

**Kitbash Bench linkage solver** (`solveLinks`, `js/app.js:570-701`): damped least squares
(Levenberg–Marquardt) with a numeric Jacobian (h = 1e-4) over the joints on the path. Damping
per joint λ·w·(1 + Aᵢᵢ), stiffness w = 1 + 0.5 × parts carried, so joints carrying more move
less (shoulder less than wrist). Steps clamped to ±0.3; λ ÷ 3 on success, × 10 on failure;
30 iterations. Turns solved first; sliders join only if > 0.005 mm of error is left. A
temporary pin from the grabbed point to the cursor makes it IK.

**kitbasher pin solver** (`src/core/pins.ts:24-118`, weighted Kabsch in
`src/core/geometry.ts:240-314` with a 1e-10 diagonal bias that keeps the current turn when
points don't fix it). Hard pins solved exactly: 1 pin → Kabsch with centres fixed at the pin;
2 pins → turn the pin axis onto the target, then twist `atan2(Σw·axis·(a×b), Σw·a·b)`; 3+ →
Kabsch on the hard pins. Soft pins weigh `10^(p/25−1)` for priority p ∈ [0, 100]. These are
the per-joint steps of CCD IK: ball joint = Kabsch with the centre fixed; hinge = the atan2
twist; *aim at a point* = the ball step with one handle on the part's forward axis.

**Joint limits:** Kitbash Bench only clamps a hinge angle or a roller's travel; kitbasher has
none. Clamp inside `constrain()` in `crates/polymix/src/assembly.rs`; a ball's cone limit is
new work: split the rotation into swing and twist and clamp each.

Rust crates exist (rapier's `Multibody::inverse_kinematics`, the `k` crate's Jacobian solver)
but Polymix's joint model is its own; a ~200-line DLS solver is cheaper than a dependency.

### 5. Magnet holes at a joint — Kitbash Bench

`js/app.js:6551-6800`, sizes in `index.html:55-58`.

- Sizes 1×1 to 10×3 mm. Clearance added to diameter and depth: **resin +0.15 mm, FDM
  +0.3 mm**.
- The cutter starts 0.3 mm above the surface so the cut opens cleanly.
- Warns when a raycast finds < 0.6 mm of material behind the hole.
- **Pair alignment:** one shared axis = n_A − n_B, each the area-weighted normal over a patch
  of radius `max(1.5, 1.25 × hole Ø)`, so rivets and texture don't tip it and the magnets
  meet face to face.
- A hole already within 0.9·Ø is reused, so swappable heads / arms share one hole in the body.
- Stored on the connection and re-placed whenever the connection changes.

### 6. Gap check (parts that should touch but don't)

- **Kitbash Bench fit check** (`js/app.js:1940-2069`, `js/math.js:503-636`): three cuts at the
  joint (two containing the axis, 90° apart, one across it at a slider depth). Overlap area by
  rasterising both outlines; gap = smallest distance between outline segments. Pinned with K,
  updates live while dragging. Polymix has the slicing already
  (`local_intersection_curves` in `crates/mm-edit/src/cut.rs`).
- **kitbasher interference inspection** (`src/core/inspection.ts:204-385`): per pair the
  smallest distance and its closest points (BVH); overlap volume for closed meshes by CSG
  intersection measured with signed tetrahedra (`61-77`); a **near-surface band** within d
  (`91-202`): a triangle counts if every corner is within d and they share one closest face
  or max distance + longest edge ≤ d; otherwise cull by triangle-to-triangle distance and
  split at the longest edge down to `max(d/4, size/2048)`, clipping where distance crosses d.
  Budget 250k band triangles, 2M queries. Its MCP report uses d = 0.5 mm and also lists
  "isolated" parts that are near nothing (`mcp/session.ts:1322-1337`).

### 7. Fill the gap a turned ball leaves / combine posed parts

- **Kitbash Bench fill seams** (`js/math.js:1286-1452`): vertices near another part are pushed
  along a smoothed normal by gap + 0.15 mm, for gaps up to 0.5 mm (found by raycast, or
  nearest point at a rim); kept only if the vertex ends up inside the other part; push amounts
  averaged over a 0.25 mm neighbour band; up to 6 passes while slits remain.
- **Kitbash Bench loose bits** (`js/solid.js:305-361`): pieces within 2 mm of something ≥ 10×
  their size get a convex-hull bridge, radius clamped 0.125–0.5 mm, sunk 2r + 0.2 mm in.
- **kitbasher Warp "Close gap"** (`src/core/warp-neighbour.ts:327-351`): for each free grid
  node, the nearest point on a neighbour within 5 mm, used only if normals oppose (dot < −0.5)
  and the node isn't already deeper than the allowance; target 0.1 mm inside the neighbour.
  Soft targets (strength 1000) in an ARAP solve on a 60–4000-node grid
  (`warp-grid.ts:3-60`), cotangent weights clamped to [0.5, 10] (`warp-solve.ts:641-678`);
  dense vertices follow 4 nodes weighted `((r−d)/(r·d))²` (`warp-bind.ts:71-176`). Polymix
  has an ARAP Warp already, so only the targets are new. Good for slivers at a socket rim; a
  large empty socket still wants the union-with-a-sphere idea.

### 8. Better contact fit (improve existing `mate::contact_fit`)

- **Penalise overlap over the whole part,** not only the facing pairs: Kitbash Bench's final
  ranking is contact − 5·sunk (mm²), and its refine gives sunk points weight 2
  (`js/lucky.js:351-439`).
- **Several patch pairs at once** for a whole tapered peg (end + 4 sides against floor + 4
  walls): kitbasher matches each sample only within its partner patch, both directions, worst
  10% per pair dropped (`src/core/surface-fit.ts:153-245`). Port only the per-pair restriction
  and two-way matching; keep Polymix's point-to-plane step.
- **Outline and ring terms** for notched / keyed faces: Kitbash Bench's Match
  (`js/math.js:758-975`) runs ICP over three groups (outline points pulled only across the
  other outline, inner rings across the other's rings, surface points point-to-plane), each
  trimmed at 3× its own median, started every 4° about the face normal plus ring-over-ring
  guesses, then seated with item 2.

### 9. Small accuracy fix: area-weighted plane fit

`fit()` in `crates/mm-edit/src/mate.rs` (its `ShapeKind::Plane` arm) builds the plane
covariance from vertices with equal weight, so densely tessellated areas pull the fit. kitbasher (`src/core/planes.ts:16-80`) builds it from whole triangles weighted by area and
rejects narrow or non-flat patches.

### 10. Lucky: connect with no clicks — Kitbash Bench (bigger project)

`js/lucky.js`. Tries every placement of one part on another and keeps the one with the most
flush contact and no overlap.

- **Distance grid** around the fixed part (`:54-146`): cell `max(0.05, ∛(box volume/6e6),
  √(area/2.5e5))`; cells within 2 of the surface store one area-weighted plane; every cell
  gets an inside vote from odd ray crossings along z and x.
- **Samples** (`:151-211`): 3000 on the mover, 2400 on the fixed part, area-even. Normals
  averaged over `max(0.4, 4·cell)` mm; "steadiness" Q = |Σ area·n| / Σ area. Anchors (160)
  and targets (1400) by farthest-point sampling over points with Q ≥ 0.97.
- **Brute force** (`:253-323`): each anchor on each target, normals opposed, 24 turns (~5.4M
  poses). Local test over ≤ 80 neighbours within 1.2–2.5 mm: touching if |d| ≤ `max(0.2,
  3·cell)` and normals oppose (dot < −0.5); sunk if d < `−max(0.45, 5·cell)` or the cell votes
  inside. Early exit at > 10 sunk or < 6 touching after 25 points. Whole-part test on 120
  points, reject if > 8% sunk. Score = touch − sunk + 2·whole-touch − 3·whole-sunk.
- **Refine** (`:351-384`): point-to-plane Gauss–Newton, 25 iterations, threshold 4 cells → 1;
  steps capped at 0.15 rad and 4 cells. Leaders re-settled from 19 small nudges (±4° / ±8°
  per axis, ±0.3 mm). Parallelises by anchor (rayon in Rust).
- **UX:** alternatives ranked in the panel ("guess 2 close behind" when the lead is under
  2 mm²), and a plain verdict line "N mm² flush, M mm² overlapping".

## Ideas not yet in the backlog

- **Swap options** (Kitbash Bench): heads, weapons or arms that fit the same spot share one
  joint; pick one per spot, save sets, export every combination. Fits Warhammer loadouts well.
- **Mirror with renaming** (Kitbash Bench): mirrored copy "Right Arm" → "Left Arm", attached
  on the other side with its own joint; copies bring what hangs from them.
- **Rail direction from picked faces** (Kitbash Bench `js/app.js:1862`): PCA of the picked
  faces' normals gives Along / Across / Out for a groove, flat or strip; rail ends found by
  walking along the channel until the distance to the surface changes by > 0.35·r.
- **Slide-until-touching buttons** in the joint's axes (±X / ±Y / ±Z; Kitbash Bench
  `js/math.js:590`): raycast, then bisect on overlap.
- **Exploded view** (kitbasher `src/core/navigation.ts:55-71`): parts shift outward from the
  scene centre by % × 0.25 × the diagonal, editing off meanwhile.
- **Status notes** such as "moved 0.42 mm to clear Torso" after a constrained drag.
- **Ask once, remember:** "just this part, or bring the group?" when moving a jointed part.
- **Blender export** (kitbasher `src/core/files.ts:431-538`): GLB plus a Python script; per
  joint an Empty at the parent's anchor, parented to the parent, child parented to the Empty
  with its own transforms locked, rotation locks on the Empty. Polymix's typed frames map
  cleanly: hinge locks X / Y; slider locks rotation and frees local Z; joint limits become
  Limit Rotation.
- **MCP additions** (kitbasher's tools: `align_parts`, `fit_surfaces`, `fit_anchors`,
  `set_clipping`, `inspect_interference`, `raycast`, `place_parts`): a `constrained` flag and
  clip report on `pose_joint` / `connect`; `check_collisions` returning overlap volume,
  smallest gap, closest points and isolated parts; surface patches given as point + radius
  for agents working from coordinates.

## Low value / out of scope

- Ring finding in a flat face by raster flood fill (Kitbash Bench `js/math.js:218`): Polymix
  finds cylinders already.
- Match pose to a reference picture by silhouette coordinate descent (Kitbash Bench
  `js/app.js:3924-4084`).
- Sprue / model-kit layout, image-to-relief, scatter, crack, ooze (Kitbash Bench): outside the
  Meshmixer-replacement scope.
- kitbasher's `native/`: only a wrapper over meshoptimizer's simplifier and remesher.

## Libraries looked at

- **rapier** (`Multibody::inverse_kinematics`) and **`k`** (Jacobian IK): workable, but tied to
  their own joint models; not worth the dependency (see item 4).
- **parry3d** `cast_shapes_nonlinear` (time of impact under rotation): could serve stop-at-
  contact, but support for triangle-mesh pairs is unconfirmed; Polymix's own BVH suffices.
- **manifold-csg** (Rust bindings to Manifold): Polymix has its own boolean; only of interest
  as a reference.
- **CGAL Efficient RANSAC** (Schnabel et al.): plane, sphere, cylinder, cone and torus
  detection; the reference to read for the ❌ "Cones (tapered round pins)" row.
