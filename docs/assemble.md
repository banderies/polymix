# Assemble: posing kit parts by their joints

Miniature kits (tabletop wargaming proxies and the like) ship as separate parts (torso, head,
arms, legs, weapons) that meet at a few kinds of joint: a **ball in a socket**, a **round pin
in a hole**, or a **flat-ended peg in a recess**, often tapered and rectangular so it fits only
one way. The **Assembly** tab (⌘4) joins such parts and poses them before printing: one part is
grounded, the others hang from it by joints, and dragging a limb turns it about its joint while
what hangs from it follows.

The Assembly tab has a scene of its own (its own parts, camera and undo; saved in the same
project) with only the assembly tools in its toolbar: Connect, Pose, Add Joint, Select (for joint
surfaces), Transform, Align, Collisions, Clearance and Send posed to Mesh. Parts get there from the
Mesh tab: right-click an object, **Send to Assembly** (or **Send visible to Assembly**). A part is
a copy of its Mesh object linked by its mesh only: edits of the object's shape in the Mesh tab
(brushes, remeshing, booleans, cuts) reach the part, which keeps its pose; moving, turning or
scaling either one leaves the other where it is. Nothing there edits the meshes: its Select only
selects (no Discard, Erase & Fill, Extrude…, and their keys do nothing), and the sculpt and
Inspector keys stay in the Mesh tab. The Mesh tab is as it was before the Assembly tab: a flat
object list, no joints, no Assemble.

**Status:** ✅ done · 🟡 partial (see notes) · ❌ not yet · ⚪ low priority.

**Code:** recognition, seating and the touching fit are GUI-free in `crates/mm-edit/src/mate.rs`;
the joint model (grounding, joints, kinematics, following parts other tools move, project text)
is `crates/polymix/src/assembly.rs`; the tool is `crates/polymix/src/tools/assemble.rs`; parts and
the Mesh objects they came from (telling a move from an edit, sending both ways) are
`crates/polymix/src/link.rs`; the tab is `Tab::Assembly` in `crates/polymix/src/shell.rs`, shown
by the Mesh workspace with its other `Desk` (scene, history, camera, tools) and
`MeshWorkspace::assembly_palette`.

**Prior art:** [assemble_prior_art.md](assemble_prior_art.md) maps two MIT-licensed kitbash
tools (Kitbash Bench, kitbasher) onto the ❌ rows below, with the algorithms worth porting
("PA #n" below is its item n). Reimplement from those descriptions; a function ported closely
keeps the upstream MIT notice in a comment.

## Roadmap

Each phase builds on the one before; the ❌ rows in the tables below say which phase takes them.

**Phase 0 (done, 2026-11-18).** Assembly tab, one-click recognition, connect (centres or
surfaces touching), typed joints, posing, offsets, collisions, the object-browser tree, see-through
parts, and the fixes from a review of all of it (switching tools over Assemble never connects,
the tab's mode follows agents, offsets keep their meaning when a group is turned round, pivot and
seat edits keep the rest pose, reshaped parts keep their joint frames, grounding stays out of
undo, a flat face rests on a ball, a cylinder's axis points out of its part, area-weighted plane
fit (PA #9)).

**Separate scenes (2026-11-19).** The Mesh tab back to how it was before Assembly; the Assembly
tab a scene of its own, parts sent to it from Mesh objects and linked by their meshes only (see
above); Send posed to Mesh in place of Combine visible / Boolean / Make Solid / Separate Shells
there.

**Phase 1: joints owned by parts, connections between them** (approved 2026-11-18; ✅ built
the same day).
- A **joint** belongs to one part and is listed under it in the object tree: ball, socket, pin,
  hole, flat face, or a bare point / frame. Made from a selection (fitted) or placed on a surface,
  then edited: radius, depth (proud of the surface or inside it), position and turn. Drawn on
  the part; its frame rides with the part as today's joint frames do.
- A **connection** joins a joint on one part to a joint on another with an interaction:
  concentric (ball / socket ↔ ball / socket), tangent (ball ↔ flat: the face rests on the ball
  and turns about its centre; the radius sets how far out), coaxial (pin / hole), flush (flat ↔
  flat: fixed or sliding), coincident (point). The motion (ball, hinge, fixed, …) is the
  connection's.
- One connection per joint at a time (connecting another part offers to replace: weapon swaps).
  Editing a joint re-seats what is connected to it, keeping its pose.
- Tree: part → its joints → the part connected there → its joints … The two-click Connect stays
  as a shortcut making both joints and the connection. Today's joints load as connections
  between point joints.
- Made to carry what comes later: the connection holds its limits (Phase 3), magnet holes
  (Phase 5) and swap options; each motion type gives the solver its free motions as columns
  (Phase 3: a ball's three turns about its centre, a hinge's turn, a slider's slide).

**Phase 2: seating that finds the fit by itself.**
- Push-together seat search (PA #2, Kitbash Bench `seatSolve`): height fields of both parts
  over the joint, search turn × slide × small tilt for the most touching and least sunk area.
  Gives whole tapered pegs, keyed faces and *slide in until touching* in one, with a readout
  ("turned 12°, slid 0.30 mm, pushed in 0.8 mm · 14 mm² touching").
- `contact_fit` (PA #8): penalise overlap over the whole part, not only facing pairs; several
  patch pairs at once (peg end and sides), matched two ways; outline and ring terms for notched
  faces.
- Cones (tapered round pins): CGAL Efficient RANSAC as the reference.

**Phase 3: posing that respects the model.**
- Stop at contact while dragging (PA #1). First the simple version: bisect between the last good
  and the asked-for joint value, keeping the furthest without new crossing triangles. Then
  kitbasher's "No clip" solver (Hildreth's method over contact half-spaces), its columns the
  joint's free motions, with an exemption sphere at each joint so a ball stays in its socket
  while the limb stops at the torso; status notes like "moved 0.42 mm to clear Torso".
- Joint limits (PA #4): clamp in `constrain()`; a ball's cone by swing–twist.
- IK and aim (PA #4): damped least squares over the joints to the grounded part, stiffer joints
  carrying more parts; aim = a ball turned to point a part's forward axis at a click.
- Pose library, mirror a pose.

**Phase 4: kits that assemble themselves.**
- Auto-detect the joints of a kit modelled assembled (PA #3): contact points between each pair →
  `fit_region` → `suggest_joint` (finds hinges too, which Kitbash Bench cannot); a root chosen by
  neighbours, volume and names (torso / body / hips; not base / plinth); a review tree before
  anything is built (re-parent, mark as an option, set loose); notice files exported each around
  the origin.
- Lucky placement (PA #10): try every placement of one part on another, keep the most flush
  with no overlap; ranked alternatives in the panel.

**Phase 5: finishing for the print.**
- Gap check (PA #6): slices through the joint (overlap area, smallest gap), live while dragging;
  near-surface band and isolated parts.
- Fill seams / close gaps (PA #7): push vertices across small gaps, or ARAP Warp targets (Polymix
  has Warp); bridges for loose bits.
- Magnet holes (PA #5): standard sizes with resin / FDM clearance, pair alignment, a warning
  when too little material is left, one hole shared by swappable parts.
- Round bases, print-tolerance offsets of a joint's surface, collision report with overlap
  volume and closest points.

**Unscheduled ideas** (from the prior art): swap options and loadouts (one joint, several parts
to choose from, export every combination); mirror with renaming ("Right Arm" → "Left Arm");
rail direction from picked faces; slide-until-touching buttons in the joint's axes; exploded
view; "just this part, or the group?" asked once; Blender export (GLB plus a rig script); more
MCP actions (constrained flag and clip report, inspect interference, raycast).

## Using it

1. Import the parts into the **Mesh tab** (one file each, or one file split with **Separate
   Shells**), and fix them there as needed.
2. Right-click a part in the object browser: **Send to Assembly** (or **Send visible to
   Assembly** for the whole kit). The **Assembly tab** (⌘4) shows them, where they were. Edits
   of their shapes made later in the Mesh tab reach them there, where they are posed; moving
   them in either tab leaves the other tab's copy where it is. The largest part is grounded; the ground symbol on each row of the
   object browser changes that. The browser shows the parts as a tree: under each part its
   joints, and under each joint the part connected there (click a joint or a connection to edit
   it). The eye beside a part makes it see-through, so clicks and brushes reach a surface behind
   it.
3. **Joints belong to parts.** **Add Joint**, then click a part where it should have one: a ball,
   socket, pin, hole or flat face is recognized there (or paint the surface with Select, S, and
   use it). In the joint's panel change its shape, radius, length, depth (proud of the surface,
   or into the part below zero) and nudge it across. Another kind of shape is fitted to the
   joint's own surface again (a pin's triangles fitted as a ball); where none fits (a ball on a
   flat face) it keeps its size and place, and says so. With a joint selected, **Look for** fits
   it again as the shape chosen there, as does clicking its surface again. Clicking a joint or a
   connection in the object browser brings its editor back (closing Select over Assemble).
   Joints are drawn on their parts; they stay with the part whatever is connected to it.
4. **Connect** a joint on one part to a joint on another: click each (a joint's marker or its
   surface, or a new surface, which becomes a joint when connecting). The part that is not held
   by a grounded part is previewed seated, its limb coming out of the socket's opening. Check the
   interaction (concentric, tangent, coaxial, flush, coincident) and the motion (ball, hinge,
   fixed, …), then **Connect** (Enter). A joint holds one connection: connecting it again
   replaces the old one (the panel says so).
   - **Tangent** (a ball and a flat face): the face rests on the ball and turns about its
     centre; the ball's radius sets how far out it sits.
   - **Seat: Surfaces touch** (default) fits the joints' surfaces onto each other (least
     squares over the facing points; the panel shows the gap and the deepest overlap). It uses a
     joint's surface only while the joint is as it was found: once its shape, radius or place is
     edited (a flat face made a ball, a depth set), and for tangent connections, the joints'
     anchors meet instead. **Joints meet** puts the anchors together, leaving any clearance all
     round. **Seat offset** moves the part from its seat (in the axes of the joint it is
     connected to).
5. **Pose:** drag a part. Ball: the grabbed point follows the cursor; Shift+drag twists the
   limb. Hinge: turns about its axis. Slider: slides. ⌘ snaps to 15° steps. The selected
   connection has **handles** for what its motion allows, one turn or slide at a time (a ball's
   rings for x, y and twist; a hinge's ring; a slider's arrow); ⌘ steps them by 15° or 1 mm,
   Shift by 1° or 0.1 mm, with the angle shown as it turns. Select a connection
   to type its angles, change its interaction or motion, move its seat, fit its surfaces
   touching, put it back to rest or disconnect it. **Editing a joint re-seats what is connected
   to it, keeping its pose** (a bigger ball pushes a resting face out; moving a socket's joint
   moves the pivot, and the limb with it).
6. **Check collisions**, then export the parts as posed, or **Send posed to Mesh**: copies of the
   visible parts, as posed, become new objects in the Mesh tab, to join there (**Boolean**
   union, **Make Solid**) and print in one piece.

The Assembly tab's parts, joints and connections are saved in `.polymix` projects (`assembly/`:
its manifest, a mesh per part unless the part is its Mesh object moved, turned or scaled, which is
stored as that map, and `joints.txt`, version 2) and are part of its undo. Projects of Polymix 0.6
(`assembly.txt` on the Mesh objects) open with every object sent to the Assembly tab, joints and
grounding on the parts; version 1 files from before joints belonged to parts load as point
joints.

## Features

### Parts and grounding

| Feature | Status | Notes |
|---|---|---|
| Assembly tab (⌘4): a scene of its own with only assembly tools | ✅ | Its own parts, camera, undo and open tools (each tab finds them as it left them); saved in the same project; Assemble opens with the tab when nothing is open there |
| Parts sent from the Mesh tab, linked by their meshes | ✅ | Send to Assembly / Send visible to Assembly (object browser's right-click); shape edits in Mesh reach the part where it is posed; moves, turns and scales in either tab stay there (a mirror counts as an edit); a part whose object is deleted keeps its last shape; undo in either tab keeps the other's work |
| Send posed to Mesh | ✅ | Copies of the visible parts, as posed, as new (unlinked) Mesh objects, to join with Boolean / Make Solid |
| No accidental mesh edits in the Assembly tab | ✅ | Select there only selects; edit keys (X, Delete, F, D, R, B, P, J, Y, T, ⌘G) do nothing; sculpt / Inspector keys only in Mesh |
| Object browser as a tree (Assembly tab) | ✅ | Grounded parts as roots, listed first; under each part its joints (shape, name; click: select; ×: delete), under each joint the part connected there with a connection chip (click: select; right-click: disconnect; red when it came apart) |
| Joints owned by parts | ✅ | Ball, socket, pin, hole, flat face, point; made from a click (recognized) or a selection; edit shape, radius, length, depth, nudge, name; fit again from a selection; drawn on the part |
| Connections between joints | ✅ | Concentric, tangent (flat on ball), coaxial, flush, coincident × ball, hinge, fixed, slider, cylindrical, planar; one per joint (connecting again replaces); editing a joint re-seats keeping the pose |
| See-through parts (the browser's eye) | ✅ | Assembly tab: shown translucent, every pick (clicks, brushes, lassos, posing) goes through it to what is behind; not in undo, exported as usual |
| Ground / free a part | ✅ | The ground symbol on each row of the object browser; several grounded parts allowed (a joint between two of them cannot move) |
| Ground the largest part when nothing is grounded | ✅ | By bounding-box volume, when the tool opens |
| Parts as separate files in one go (folder import) | 🟡 | Existing multi-file import |
| Isolate / hide a part from the list | ✅ | The object browser's visibility box, and its eye for see-through |
| Auto-colour parts distinctly | ⚪ | Objects already get a palette colour each |

### Recognizing joint surfaces

| Feature | Status | Notes |
|---|---|---|
| One click finds a ball / socket (sphere) | ✅ | Fit to a patch around the click, grown over the triangles on the sphere, refitted as it grows; stops at necks and rims |
| One click finds a pin / hole (cylinder) | ✅ | Axis from the normals, radius by circle fit |
| One click finds a flat face, with its outline rectangle | ✅ | Minimum-area rectangle; round faces told from rectangles |
| Best shape chosen automatically, or forced in the panel | ✅ | The shape that explains the most surface (a plane unless a curve explains clearly more) |
| Tolerance for sculpted, lumpy surfaces | ✅ | Slider, % of the radius |
| Fit to a selection instead (click on selected triangles) | ✅ | For surfaces too irregular to grow |
| Select a joint's surface without leaving the tool | ✅ | Refine… / Select / S: a Select that only selects, starting from the surface found; Use this surface fits the joint to it |
| Which way the part's body lies (limb, into the socket) | ✅ | Seats limbs out of the socket opening, pins into their holes |
| Cones (tapered round pins) | ❌ | Phase 2 (RANSAC reference). A mild taper passes as a cylinder within tolerance |
| Whole tapered peg (four sides + end) as one feature | ❌ | Phase 2: push-together seat search (PA #2), multi-patch fit (PA #8). Today: click the peg's end face and the recess floor |
| Ball / socket clearance readout | ✅ | Socket radius − ball radius (interference shown red) |

### Connecting

| Feature | Status | Notes |
|---|---|---|
| Seat one part on another from two clicks | ✅ | Ball centres meet; pin axes meet; flat faces meet face to face, outlines lined up |
| Surfaces touch: fit one surface onto the other | ✅ | Iterative closest points over facing pairs (worst 20% dropped); balls only move, pins and pegs turn too; gap and deepest overlap shown |
| Seat and pivot offsets (3D, in the joint's axes), before and after connecting | ✅ | Moving the pivot leaves the parts; moving the seat takes the pivot along |
| Fit an existing joint's surfaces touching again | ✅ | Kept while the parts' triangles are unchanged (not saved in projects) |
| The free part moves, whichever was clicked first | ✅ | The part not held by a grounded part; everything already joined to it comes along |
| Joint type suggested, overridable | ✅ | Ball → ball joint, pin → hinge, flat faces → fixed (round faces → hinge) |
| Flip / keep the turn / next way round for keyed faces | ✅ | Rectangles fit two ways, squares four |
| Join where they are (kits modelled assembled) | ✅ | "Move into place" off |
| Refuse joints that would close a loop | ✅ | |
| Slide in until touching (pin shoulders, tapered pegs) | ❌ | Phase 2 (PA #2), or the contact solver with the axis as its one column (PA #1) |
| Auto-detect every joint of a kit modelled assembled | ❌ | Phase 4 (PA #3) |

### Joints and posing

| Feature | Status | Notes |
|---|---|---|
| Joint types: fixed, ball, hinge, slider, cylindrical, planar | ✅ | |
| Drag a part to pose it; what hangs from it follows | ✅ | Live preview drawn on the GPU (no mesh uploads while dragging); one undo step per drag |
| Ball: grabbed point follows the cursor; Shift twists | ✅ | |
| Hinge / slider / planar drags | ✅ | Axes seen side-on fall back to screen-space motion |
| Fixed joints weld parts (dragging a sword turns the wrist) | ✅ | |
| Drag from either side (grounding decides which side moves) | ✅ | |
| ⌘ snaps angles to 15° | ✅ | |
| Handles on the selected connection | ✅ | Only what its motion allows: ball three rings (x, y, twist; they turn with the part), hinge one ring, slider an arrow, cylindrical a ring and an arrow, planar a ring and two arrows, fixed none. ⌘ steps of 15° / 1 mm, Shift 1° / 0.1 mm (a one-axis turn or slide steps the connection's own value); the angle or slide is shown while dragging |
| Type a joint's angles / slide in the panel | ✅ | Ball: X, Y, twist; hinge: angle; slider: slide |
| Rest pose (one joint or all) | ✅ | |
| Change a joint's type (pose kept as far as it allows) | ✅ | |
| Joints follow parts moved by other tools (Transform, Align) | ✅ | Anchor vertices + rigid fit; a joint left apart shows red with Snap together |
| Saved in projects, part of undo | ✅ | |
| Agent (MCP) actions | ✅ | pick_surface, connect, ground, pose_joint, set_joint_type, rest_pose, delete_joint, check_collisions |
| Joint limits (cone for a ball, range for a hinge) | ❌ | Phase 3 (PA #4): clamp in `constrain()`, swing–twist for a ball's cone |
| Stop at contact while dragging | ❌ | Phase 3 (PA #1): bisect first, then the constraint solver over the joint's free motions |
| Inverse kinematics (drag a hand, the shoulder and elbow follow) | ❌ | Phase 3 (PA #4): damped least squares, stiffness by parts carried |
| Pose library (save / recall named poses) | ❌ | Phase 3: joint values per pose |
| Mirror a pose left ↔ right | ❌ | Phase 3; mirror with renaming from the prior art |
| Aim a part at a point (head / gun looks at a click) | ❌ | Phase 3 (PA #4) |
| Rotation gizmo on the selected joint | ❌ | Phase 3: rings for the joint's free axes only |

### Checking and finishing

| Feature | Status | Notes |
|---|---|---|
| Check collisions between parts in the pose | ✅ | Crossing triangles drawn red; pairs listed, joined pairs greyed |
| Gap check (parts that should touch but don't) | ❌ | Phase 5 (PA #6). The Clearance tool covers one pair |
| Fill the gap a turned ball leaves in its socket | ❌ | Phase 5 (PA #7): seam fill or Warp targets; a large empty socket wants a union with the sphere |
| Combine posed parts into one print | 🟡 | Send posed to Mesh, then Boolean / Make Solid there |
| Export each posed part separately | 🟡 | Export already writes visible objects; no per-part batch yet |
| Magnet holes / pins at a joint | ❌ | Phase 5 (PA #5): held by the connection (Phase 1) |
| Round bases (25 / 32 / 40 / 50 mm) and drop the model on one | ❌ | Phase 5 |
| Print tolerance: shrink a ball / grow a socket | ❌ | Phase 5; with part-owned joints (Phase 1) a joint's radius is already editable |
