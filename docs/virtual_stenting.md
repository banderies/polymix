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
  | Stiffness | Radial force. Along the stent the lumen becomes a round tube whose diameter lies this share of the way from the vessel's own to the stent's: a 2 mm vessel, a 6 mm stent and 50 % give 4 mm. 100 %: a perfect tube of the stent's diameter. 0 %: the vessel keeps its shape. Wall wider than the stent is never pulled in (unless covered). |
  | End blend | How far past each end of the stent the opened wall eases back to its own shape. |
  | Covered (stent graft) | The lumen along the stent becomes the stent's tube; see below. |
  | Remesh the reshaped wall | Even triangles where the wall was stretched or squeezed (recommended for CFD). |
  | Edit in place | Reshape the object itself. Otherwise the result is a new object named after the stent, and the original is hidden. |

- **Preview.** The result is previewed as the settings change, with the stent drawn as a
  wireframe and its ends in orange. **Apply** (or Enter) keeps it.
- **Readouts:**
  - the centerline length between the ends, and the stent axis' length between them;
  - how far the stent reaches past (or stops short of) each end;
  - the narrowest lumen diameter before → after;
  - for a covered stent, the side-branch ends it sealed off.
- **Warnings:**
  - the stent's axis leaves the vessel;
  - the reshaped wall runs into itself (new crossings only);
  - the vessel must bend back sharply past an end;
  - the reshaping moves an open end (inlet or outlet) of the mesh.

## How it works

### The vessel's centerline

- Traced from end 1's cross-section: geodesic distance rings, as in the Centerline tool,
  through end 2 and on past both ends, taking the straightest way at junctions.
- Vertices whose ring lies on this path are the stented vessel's own wall. Side branches and
  anything else joined near it follow with distance along the surface.
- The path is smoothed over half the vessel's radius. Rings kink by open ends and at
  junctions.
- **Centred in the lumen between the ends.** Ring centroids can lie millimetres off the
  lumen's centre, even outside it, through a stenosis that bends or narrows eccentrically.
  The stent would then push the wall open about a point outside the vessel and fold it onto
  itself. So between the ends the path moves to the lumen's centre:
  - it is resampled densely enough to follow a narrow channel;
  - each point goes to the centre of the cross-section square to the path there: its
    centroid if well inside, else the centre of the largest circle inside (a crescent's
    centroid is outside it). About that centre, cross-sections are star-shaped, as the push
    needs;
  - where no outline is found, or the nearest belongs to another vessel, the move is
    interpolated from the neighbours';
  - only lumens about as narrow as the ends move it (a wide chamber's centre says nothing
    about the vessel's course);
  - the moves are smoothed over the lumen's radius and ease out just past each end.
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
  along the least-bending path that leaves the stent along its direction. Its length is the
  shortest for which the bend stays gentle for the vessel's radius (radius × curvature ≤ 0.4),
  up to 12 stent radii. Turning back more sharply would fold the wall on the inside of the
  turn. `return_bend` reports what was reached.

### Bare stent (`apply_stent`)

1. **Grip.** The stent holds the vessel where its wall is no wider than the stent (30th
   percentile of each slice's radius), letting go by 1.6 × the stent's radius, smoothed along
   the vessel. Part of a stent floating in a wider lumen (an "elephant trunk") moves nothing.
2. **Carry.** Each slice of gripped vessel is carried from the centerline onto the stent's
   axis (and past the ends onto the return path). It turns to face along the new path, so
   the vessel straightens with the stent without shearing.
3. **Remesh the carry.** Triangles stretched or squeezed by more than 10 % by the carry,
   plus two rings around them, are remeshed to the region's mean edge length.
4. **Join the stent's tube.** The new lumen is the old one joined with the stent's tube (an
   exact mesh union), so the vessel takes the stent's own surface where the stent opens it:
   - The tube is lofted from round rings square to the stent's axis, about one edge length
     apart. Each ring's diameter lies *stiffness*'s share of the way from the native lumen's to
     the stent's, never wider than the stent. The native lumen there is the widest circle about
     the axis inside the (straightened) wall, so at 0 % nothing opens. Below 100 % the rings
     are smoothed a little along the stent, against measuring jitter.
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
- The stent's tube is built in its place from the same round rings as the bare stent's,
  measured on the straightened vessel before the push. Wider wall is narrowed to the stent,
  narrower wall is open as far as Stiffness allows.
- The tube flares into each cut over half a radius and is blended and remeshed into the
  vessel.
- Both ends of a covered stent must lie inside the vessel. The covered stretch must not be
  joined to the rest of the vessel another way round (a loop in the vessel tree).

## Limits

- **Geometric, not mechanical.** Straightness and Stiffness are proxies for bending and
  radial stiffness. There are no contact forces, wall mechanics or deployment simulation.
- **Not modelled yet:** braided-stent foreshortening and stent struts.
- **Straightening reaches far.** A long stent straightened across a tight bend moves the
  vessel well past its ends. If an inlet or outlet is that close, it moves too (the tool
  warns).
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
  - outside the end blend, vertices keep their ids and positions;
  - local cross-sections: a point inside a lumen, and the widest point inside a crescent.
- **App (`tools/stent.rs`):**
  - a bent, eccentrically narrowed vessel, traced from its first end: bare, straightened and
    covered stents, centred, opened, no crossings, sharpest crease under 30°;
  - Measure's vessel diameters carry the stent's ends.
  - a trace that passes outside an eccentric narrowing is moved to the lumen's centre, stays
    smooth, and is unchanged past the ends.
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
