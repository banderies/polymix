# Virtual stenting

**Vessel → Virtual stent** places a stent between two cross-sections of a vessel and reshapes
the vessel around it. The result is a treated-anatomy variant for flow studies (CFD). The
first use is dural venous sinus stenosis, treated with bare (uncovered) self-expanding stents.
Covered stents (stent grafts) are modelled as well.

The geometry is in `mm_edit::stent` ([`place_stent`], [`apply_stent`]); the tool is
`crates/polymix/src/tools/stent.rs`.

## Using it

- **Ends.** Click the vessel wall at both ends of the stretch to treat. Each click makes a
  cross-section, a green ring like *Measure → Vessel diameter*. Clicking again moves the
  nearer end; **Clear** starts over.
  - From **Measure**: make two *Vessel diameter* measurements, then press *Place stent
    between the last two*.
- **Stent.**
  - **Diameter**: the nominal, fully expanded diameter.
  - **Length**: measured along the stent's own axis.
  - Defaults: the wider end's diameter rounded to 0.5 mm, and the centerline length between
    the ends rounded up to the next 5 mm. Typing a value keeps it.
  - Mesh units are taken as millimetres.
- **Placement.** The stent is centred on the two ends. For example, ends 35 mm apart along
  the centerline with a 40 mm stent: it reaches 2.5 mm past each end. A shorter stent stops
  equally short of each.
- **Settings.**

  | Setting | What it does |
  |---|---|
  | Straightness | 0 %: the stent follows the vessel's centerline. 100 %: it takes the smoothest path between the two ends (where a free stent held at both would settle). In between, a blend. Where the stent grips the wall, the vessel moves with it. |
  | Smoothest path G0 / G1 / G2 | How that path meets the ends. **G0**: a straight line between their centres. **G1**: leaves each end square to its cross-section. **G2**: also continues the vessel's bend there. |
  | Stiffness | Radial force. Along the stent, in each direction across it, the lumen lies this share of the way from the vessel's own wall to the stent's radius: a 2 mm vessel, a 6 mm stent and 50 % give 4 mm. So the vessel's shape eases into the round stent: 100 %, a perfect tube of the stent's diameter; 0 %, the vessel as it is. Wall wider than the stent is never pulled in (unless covered). |
  | End blend | How far past each end of the stent the wall eases back to its own shape and course (0–40 mm). **Nothing beyond it changes.** A stent straightened across a bend needs one long enough for the vessel to turn back onto its course (the tool warns when it is too short). |
  | Covered (stent graft) | The lumen along the stent becomes the stent's tube; see below. |
  | Remesh the reshaped wall | Even triangles where the wall was stretched or squeezed (recommended for CFD). |
  | Edit in place | Reshape the object itself. Otherwise the result is a new object named after the stent, and the original is hidden. |

- **Preview.** The result is previewed as the settings change, with the stent drawn as a
  wireframe and its ends in orange. **Apply** (or Enter) keeps it.
- **Readouts:**
  - the centerline length between the ends, and the stent axis' length between them;
  - the minimal lumen diameter between the ends: the largest sphere inside the lumen at its
    narrowest (the standard measure of a stenosis);
  - how far the stent reaches past (or stops short of) each end;
  - the narrowest lumen diameter before → after;
  - for a covered stent, the side-branch ends it sealed off.
- **Warnings:**
  - the stent's axis leaves the vessel;
  - the reshaped wall runs into itself (new crossings only);
  - the vessel must bend back sharply within the end blend past an end (lengthen the end
    blend or lower the straightness);
  - the reshaping moves an open end (inlet or outlet) of the mesh.

## How it works

### The vessel's centerline

- The lumen's centerline tree, as in the Centerline tool (`centerline::compute_lumen`:
  through the centres of the largest spheres inside the lumen, every point and segment
  inside it): from the model's widest open end to the others, blind side branches too,
  traced once per shape of the mesh and kept for the next pair of ends. Where an end's
  cross-section lies off that tree (or the model has no open ends), it is traced from end 1's
  centre instead. The path runs along it from end 1 through end 2 and on past both ends,
  taking the straightest way at junctions.
- Each vertex is wall of the vessel whose centerline tube (its spheres swept along it) is
  nearest, as VMTK splits branches; those of the path are the stented vessel's own wall, even
  at a junction or either side of a septum. Side branches and anything else joined near it
  follow with distance along the surface.
- The path's radii are the vessel's mean half-width across its cross-section (sixteen rays,
  `centerline::section_radii`), which bound the wall the stent carries; a flat sinus's
  inscribed sphere is much narrower than its wings. The path is smoothed over half that
  radius.
- **Through the centres of inscribed spheres between the ends**, on a finer grid. Between the
  ends the path is the lumen's centerline through the centres of the largest spheres inside
  it, as VMTK's Voronoi centerlines run (`centerline::inscribed_path`):
  - The lumen is voxelized in a tube a few radii around the traced path (inside by the parity
    of ray crossings, the mesh's open ends capped), and each voxel gets its distance to the
    wall (an exact distance transform).
  - The path is the cheapest between the deepest voxels near the two ends, each step costing
    1 / distance², so it keeps to the middle and cannot wander into another vessel.
  - It is smoothed, and each point climbs within its cross-section to where the wall is
    farthest (exact distances, finer than the grid). Its radii are the spheres' radii, which
    also give the minimal lumen diameter.
  - Where the lumen does not connect the ends at the grid's resolution, the path is centred
    on cross-sections instead (their centroids, or their largest inscribed circles).
  - Where this centerline meets the traced one at each end (the deepest point of the lumen
    near the end can lie a little aside), the two are eased together over one and a half of
    the vessel's radii there, so the path has no corner.
- The stent's ends are where this path crosses each end's plane: the lumen's centre there.

### Placement (`place_stent`)

- **The two paths.** The stent's axis blends the centerline between the ends with the
  smoothest path between them, at equal fractions of their length.
- **The smoothest path:**
  - G0: a line.
  - G1 / G2: a cubic / quintic Hermite spline along the ends' normals (G2 also matches the
    vessel's curvature there).
  - Its two tangent lengths minimise *length × ∫κ² ds*. This measure has no preferred scale,
    and a circular arc does best.
  - So on a uniformly curved stretch with the ends square to it, G1 reproduces the curve
    itself, and only G0 (ends free to angle) straightens it.
- **Past the sections.** The paths continue along the vessel and along the smooth path's end
  direction (for G2, its bend fades out over two radii).
- **No kinks.** The axis is eased to bend no tighter than the stent's diameter (its radius
  × curvature at most 0.5). Where the vessel's course jogs, as through an eccentric
  narrowing, a stent spreads the bend; a tube bent tighter than its radius would fold
  through itself.
- **Centring.** The stent keeps its length along its own axis. A straighter axis is shorter
  between the ends, so a straightened stent reaches further past each.
- **Return past each end.** The vessel returns from the stent's end onto its own centerline
  along the least-bending path that leaves the stent along its direction and meets the
  centerline at the end of the end blend. `return_bend` reports how sharply it turns (radius ×
  curvature): from 0.4 the wall starts to crease on the inside of the turn, and above 0.7 the
  tool warns.

### Bare stent (`apply_stent`)

1. **Grip.** The stent holds the vessel where its wall is no wider than the stent (30th
   percentile of each slice's radius), letting go by 1.6 × the stent's radius, smoothed along
   the vessel. Part of a stent floating in a wider lumen (an "elephant trunk") moves nothing.
2. **Carry.** Each slice of gripped vessel is carried from the centerline onto the stent's
   axis (and past the ends onto the return path). It turns to face along the new path, so
   the vessel straightens with the stent without shearing. Past each end the carry and the
   turn fade out over the end blend, to nothing at its end.
   - A slice carries only the stented vessel's own wall: wall within three radii of its
     centerline (the vessel's or the stent's, whichever is wider) and joined, through such
     wall, to the wall around the stent. Wall that only passes near, such as the vessel
     turning back alongside further on or another vessel, stays where it is. Side branches
     follow with distance along the surface, fading out before three radii.
   - Where the centerline turns tighter than the wall's distance from it (it jogs through an
     eccentric narrowing, or bends sharply), wall a few millimetres off it can be nearer one
     slice or another from one vertex to the next, and the two would be carried apart: a
     step in the wall. So where the carry would more than double (or halve) an edge, a zone
     around it takes moves interpolated smoothly from the zone's rim (a harmonic fill),
     widened until no edge there is torn. The step becomes a smooth change a few
     millimetres wide; the rest of the carry is untouched.
3. **Remesh the carry.** Triangles stretched or squeezed by more than 10 % by the carry,
   plus two rings around them, are remeshed to the region's mean edge length.
4. **Join the stent's tube.** The new lumen is the old one joined with the stent's tube (an
   exact mesh union), so the vessel takes the stent's own surface where the stent opens it:
   - The tube is lofted from rings square to the stent's axis, about one edge length apart.
     In each direction across, a ring lies *stiffness*'s share of the way from the native wall
     to the stent's radius, never beyond it: at 100 % a round tube, at 0 % the native lumen,
     and in between the native shape easing into the round stent (shape-keeping). The native
     outline is the straightened wall seen from the axis, smoothed a little around the ring
     (±7°) and along the stent (a quarter of its radius) against the mesh's roughness, but
     never beyond the wall, so at 0 % nothing opens.
   - Where the wall lies about at a ring (within half a percent of the radius, or twice the
     sag of the wall's own facets), the ring keeps just inside it there, so the two surfaces
     never run along one another (a union cannot join them). A ring opening the lumen by less
     than that keeps just inside the wall altogether.
   - Where the wall is wider than a ring, or there is none (a side branch's opening), the ring
     lies inside the lumen and drops out of the union.
   - Past each end the rings ease over *end blend* to a round ring just inside the nearest
     wall, and the tube is closed there.
   - Zero-area and microscopic triangles near the stent are fixed first.
   - The mesh's open ends are capped for the union and opened again after; the tube must
     not reach them.
   - Only the part the union changed is replaced: elsewhere vertices and triangles keep their
     ids. The new triangles and two rings around them are remeshed.
   - Wall inside the tube gives way to the tube's surface however it was folded or nearly
     closed, and wall outside it (a wider vessel, side branches) stays as it was.
5. **Push (fallback).** Where the union cannot be made, wall inside the stent's tube is pushed
   out radially instead, and the tool says so. *Stiffness* sets how far:
   ρ → ρ + stiffness × (max(ρ, R) − ρ), with a 2 % soft corner where contact starts, fading
   out over *end blend* past each end about the vessel's own centre. At full stiffness the
   wall keeps its order away from the axis within a band half a percent of the radius wide
   rather than all landing on the radius, where a wall doubling back in a slice would land
   on itself. The covered stent pushes the same way before it builds its tube.

### Covered stent

After the grip, carry and push:

- The vessel is cut across at both ends of the stent.
- Everything joined to the inside of the cuts is removed: the stented wall, and any aneurysm
  or side branch opening along it. Their open ends are counted as sealed off.
- The stent's tube is built in its place from the same rings as the bare stent's, measured on
  the straightened vessel before the push. Wider wall is narrowed to the stent, narrower wall
  is open as far as Stiffness allows, keeping its shape in between.
- A graft whose end lands in a lumen far wider than itself is flared to the wall there over a
  short distance, which can fold; with crossings, the tool says which end and suggests
  shortening the stent or moving that end (see Future work: landing zones).
- The tube flares into each cut over half a radius and is blended and remeshed into the
  vessel.
- Both ends of a covered stent must lie inside the vessel. The covered stretch must not be
  joined to the rest of the vessel another way round (a loop in the vessel tree).

## Limits

- **Geometric, not mechanical.** Straightness and Stiffness are proxies for bending and
  radial stiffness. There are no contact forces, wall mechanics or deployment simulation.
- **Not modelled yet:** braided-stent foreshortening and stent struts.
- **Nothing past the end blend changes:** no vertex moves and no triangle is rebuilt there,
  by the carry, the push, the tube's union, the covered stent's cut and join, or the remesh.
  A stent straightened across a bend therefore needs an end blend long enough for the vessel
  to turn back onto its course; with a short one the wall creases there (the tool warns).
- **Overlapping shells.** Pieces of the mesh that overlap the vessel without being joined to
  it (vessels merged without a union) are not treated as its wall.
- **Speed.** About 1 s per preview on an 800 000-triangle vessel (the union takes half).
  Straightening the whole vessel or covering takes longer.

## What the tests check

- **`mm-edit` (`tests/stent_tests.rs`, `tests/cut_tests.rs`):**
  - centring and overhangs, in either order of ends;
  - a narrowing opened to a round lumen of the stent's radius (half way at 50 % stiffness);
    nothing moves at 0 % stiffness and 0 % straightness, and nothing past the end blend;
  - the G1 / G2 smoothest path on an arc is the arc, and G0 the chord;
  - straightening moves a bent vessel onto the stent with its rings kept round;
  - a stent end floating in a wide lumen leaves it alone;
  - a covered stent lines a dilated stretch and seals off a side branch;
  - a bare stent keeps a side branch open;
  - no self-crossings, valid, consistently oriented meshes;
  - a narrowing entered by a lip that leans back opens with no crossings;
  - a vessel whose course jogs gets a stent axis bending no tighter than its diameter;
  - at 50 % an elliptical narrowing opens half way each way across and stays elliptical;
    at 100 % it is round;
  - a carry torn into a step is evened out (no edge more than doubles), the rest unchanged;
  - outside the end blend, vertices keep their ids and positions;
  - a stent straightened across a bend changes nothing past the end blend (no vertex moved,
    no neighbourhood rebuilt), with the stretch itself straightened and opened;
  - a hairpin whose returning leg runs beside the stented stretch, every vertex marked as the
    vessel's own: the returning leg stays exactly as it was;
  - local cross-sections: a point inside a lumen, and the widest point inside a crescent.
- **App (`tools/stent.rs`):**
  - a bent, eccentrically narrowed vessel, traced from its first end: bare, straightened and
    covered stents, centred, opened, no crossings, sharpest crease under 30° (straightened
    with a longer end blend), nothing changed past the end blend; with too short an end blend
    for the straightening, still nothing past it changes, and the tool says to lengthen it;
  - Measure's vessel diameters carry the stent's ends.
  - a trace that passes outside an eccentric narrowing is moved to the lumen's centre, stays
    smooth, and is unchanged past the ends;
  - the inscribed-sphere centerline finds an eccentric narrowing's centre and its radius.
- **Ignored tests on local data**, printing counts and timings only:
  - `POLYMIX_STENT_MESH=<file>`: stents along the longest branch;
  - `POLYMIX_STENT_PROJECT=<.polymix>` with `POLYMIX_STENT_ENDS` (and optionally
    `POLYMIX_STENT_AXES`, `POLYMIX_STENT_LENGTH`): a stent between given ends, as
    `mesh_tool_state` reports them.

## Future work

### Covered stents: what the basic model leaves out

- **Landing zones wider than the stent.** Today the fabric flares to the wall. A real graft
  leaves a gap there, and flow can run around its edge (a type I endoleak). Model the fabric
  as a thin wall with the gap open, or report the gap.
- **Fenestrated, branched and chimney grafts.** Keep chosen side branches open through the
  fabric (click the branches to spare).
- **Partial and combined devices:**
  - a covered segment within a bare stent;
  - overlapping and telescoping stents;
  - a stent placed inside an earlier stent.
- **Fabric permeability.** Treat the fabric as a porous zone for CFD instead of a solid wall.
  This also applies to flow diverters.
- **Excluded volume.** The sealed-off sac or branch could be kept as a separate object, for
  thrombosis or volume studies, instead of being deleted.

### Stents in general

- **Braided-stent foreshortening.** A braided stent gets longer when it is held narrower than
  its labelled diameter (as in venous Wallstent-type devices). Use the length–diameter
  relation from manufacturers' tables, applied along the deployed radius.
- **Struts.** Open- and closed-cell patterns and flow diverters (metal coverage, porosity),
  as explicit struts or as a porous layer.
- **Mechanics.** Deployment by finite elements (crimp, release, contact) to set straightening
  and opening from material properties instead of sliders. Wall stiffness could also vary
  along the vessel: dural sinus walls are stiff, arachnoid granulations compliant.
- **Device catalogue.** Diameter × length options of common venous and arterial stents, and
  oversizing guidance from the end diameters.
- **Non-round lumens.** Dural sinuses are triangular in cross-section. A stent at 100 %
  stiffness rounds them; an ovality allowance could keep some of their shape.
- **Several stents** in one session, and a stent that crosses a bifurcation into one branch.
  Today the path runs through the junction and the other branch's ostium is pushed open.
- **Export** the stent itself (axis and surface) as a separate object for post-processing and
  figures.

[`place_stent`]: ../crates/mm-edit/src/stent.rs
[`apply_stent`]: ../crates/mm-edit/src/stent.rs
