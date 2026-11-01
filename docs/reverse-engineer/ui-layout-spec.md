<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# ScanRuler "Reverse engineer" workspace — UI layout specification

For the Rust/egui port: how Polymix's **Reverse Engineer** tab should be laid out so it
matches ScanRuler's `plugins/reverse` workspace control for control.

Sources read (read-only, never run): `plugins/reverse/ui/*.tsx` (+ `reverse.css`),
`plugins/reverse/viewer/*.ts`, `plugins/reverse/app/keyboard.ts`,
`plugins/reverse/app/useReversePlugin.tsx`, `plugins/reverse/README.md`, and the shared
shell in `src/ui` (`ViewBar`, `PartKey`, `MapLegend`, `ElementRow`, `RowTools`,
`ModelSlot`, `InfoDot`, `RefPicker`, `ShowAllButton`, `StatusStrip`, `MarkTools`,
`NameField`) plus the bits of `useSketchSceneSync.ts`, `useReverseWorkspace.ts`,
`extendGrips.ts`, `core/preview.ts`, `reFeatureSlice.ts`, `rePrefs.ts` that fix
defaults, colours and grips.

## 0. Conventions used in this document

* **Labels are quoted exactly** as the source writes them, including typographic
  apostrophes (`’`), ellipses (`…`), arrows and symbols. `{x}` marks a substituted value.
* **Tooltip** = the HTML `title` attribute (hover text). In egui use
  `.on_hover_text()`. Where a control has no tooltip the entry says *no tooltip*.
* **ⓘ** = an `InfoDot`: a small round "i" key after a label. A click opens a card
  (heading + paragraphs) **outside the panel**, horizontally clear of it, vertically level
  with the dot; a second click, a click anywhere else, or `Esc` closes it. The card text
  is given in full where it carries rules the user needs.
* **Enabled when** = the control is drawn but greyed out otherwise. **Shown when** = the
  control is not drawn at all otherwise.
* Lengths are held in millimetres and shown/typed in the display units (mm or in); the
  unit word written after a field is the display unit's (`mm` / `in`). Angles are degrees.
* "Box" = an editor (`draftbox`). Only **one** box is open at a time (sketch, extrude,
  revolve or one feature box); opening one closes the others.
* "Ready" = the box's create condition (`extrudeDraftReady`, `revolveDraftReady`,
  `featureDraftReady`); Create / Enter / middle-click do nothing until it holds.

---

## 1. The screen

```
┌──────────────────────────────── TopBar ─────────────────────────────────────────────┐
│ brand · workspace tabs (… "Reverse engineer") · part chip · project name · save     │
│ status · New / Save / Open project · Settings                                        │
├─────────────── RecoveryBar / ConnectorBar (only when they have something) ──────────┤
│┌── left panel ──┐┌──────────────────────── stage ────────────────────────────────┐ │
││ 336 px wide     ││ [playback caption]     [hint chip, top centre]   [map legend ]│ │
││ scrolls (only   ││ top-left 14/14                                    [ column   ]│ │
││ scroller on the ││                                                   [ top 14,  ]│ │
││ page)           ││            3D viewport (or the 2D sketch sheet    [ right 14,]│ │
││                 ││            laid over it while a sketch is open)   [ bottom 14]│ │
││ Model           ││                                                               │ │
││ Parameters      ││   chip bar / note card: fixed, 12 px right+below the click    │ │
││ Create feature  ││                                                               │ │
││ (editor box)    ││ ┌ stage corner, bottom-left 12/12 ┐          orientation gizmo │ │
││ Features        ││ │ support card                    │              bottom-right │ │
││ Bodies          ││ │ VIEW BAR                         │                           │ │
││ Notes…          ││ └─────────────────────────────────┘                           │ │
││ Export / Compare││                                                               │ │
│└─────────────────┘└───────────────────────────────────────────────────────────────┘ │
├──────────────────────────────── StatusStrip (30 px) ─────────────────────────────────┤
│ ● READY | status text / ⚠ error (grows) | § IMPRINT | LOCAL ONLY · FILES NEVER LEAVE… │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

* **Workspace tab**: label `Reverse engineer`, icon `extract`, tooltip
  `Reverse engineer` + newline + `Sketch on sections, extrude and revolve, and export a solid STEP body built over the scan`.
  Order 10 among the workspace tabs.
* **Left panel** (`aside.panel`): width **336 px** (min 336), padding 15 / 15 / 22 px,
  background `--panel`, 1 px right border `--line`, vertical scroll only, flex column. On
  narrow screens: 282 px, padding 12 / 12 / 18. When an editor box opens, the panel
  scrolls smoothly so the box's top is in view.
* **Stage**: the 3D viewport. While a sketch is open the 2D sheet (`.viewslot.sketch`,
  z-index 2) lies over it with a veil of the stage colour at **58 %**
  (`color-mix(stage 58%, transparent)`), so the part reads faded and the sketch at full
  strength; the 3D camera follows the sheet's pan/zoom (§7).
* **No start pane**: with no scan the workspace opens on a blank stage (a hand-sized box,
  half-diagonal 50 mm) with the XY, XZ, YZ planes shown.
* **Status strip** (bottom, 30 px, dark `--strip`): lamp + `READY` / `LOADING` / `ERROR`
  / `AGENT` / `FITTING`; message area (grows) shows `⚠ {error}` in warn colour, else the
  agent command running, else the status text; `§ IMPRINT` key; link
  `LOCAL ONLY · FILES NEVER LEAVE YOUR BROWSER` (or `AGENT CONNECTED · WHAT IT READS GOES TO ITS AI MODEL`).
  Every `setStatus(...)`/`setError(...)` string quoted in this spec lands here.

### 1.1 Stage overlays and their positions

| Overlay | Position | Shown when |
|---|---|---|
| Map legend (live map or curvature) | absolute, top 14, right 14, bottom 14; column, right-aligned; pointer-events none except its editable end figures | a live-map report exists, else the curvature map is on (§5.3) |
| Playback caption card | absolute, top 14, left 14; max-width min(360 px, 100 % − 28); z 4; 3 px accent left border; no pointer events | playback running and **Caption** ticked; not while a sketch is open |
| Hint chip `Part loaded — choose a plane in the panel to sketch on` | absolute, top 10, centred | a scan is loaded, no feature yet, no sketch open |
| Build chip (spinner + `Loading the CAD kernel…` / `Building the part…`) | same hint-chip slot, top centre | kernel loading with features, or building (not during playback); only after **300 ms**, so quick builds never flash |
| Stage corner (support card above, **view bar** at the very bottom) | absolute, bottom 12, left 12, z 5, column, gap 10 | always (view bar content changes in sketch mode) |
| Context chip bar / Note card | `position: fixed`, at click + (12, 12) px; shifted left to stay ≥ 8 px from the right edge; flipped above the click (−(height + 24)) if it would pass the bottom | a selection exists (§3.4) / a note is being edited (§3.8) |
| Grip number field | CSS2D label 6 px beyond the arrow tip, in the tip's screen direction (ring: on the ring's centre) | after a grip has been dragged (§6.6) |
| Fit options window | fixed, left 352, top 96, width 400, z 40, no backdrop, draggable by its head | opened from the sketch box |
| Right-click menus (feature row, body row) | fixed at the pointer, kept on screen (opens leftward/upward near edges), z 50, min-width 200, max 260 | right click on a row |
| Shift-drag body selection window | absolute in the viewport, z 5 | while dragging |

---

## 2. Visual tokens and shared primitives

### 2.1 Colour tokens (light / dark)

| Token | Light | Dark | Use |
|---|---|---|---|
| `--chassis` | `#e7e6e1` | `#1b1d20` | app housing |
| `--panel` | `#fcfcfa` | `#25282c` | panel, menus |
| `--well` | `#efeee9` | `#1e2124` | inputs, keys, editor box background |
| `--line` | `#cfcec7` | `#383c41` | borders |
| `--line-strong` | `#b5b4ad` | `#4b5057` | menu border, drop strip |
| `--ink` | `#26282a` | `#e4e6e8` | text |
| `--dim` | `#75787b` | `#a0a4a8` | labels, hints |
| `--faint` | `#9a9d9f` | `#767a7f` | disabled, struck-through |
| `--accent` | `#1877c0` | `#3f93d6` | selection, primary, rollback bar |
| `--accent-d` | `#12629f` | `#62aae4` | pressed/hover text |
| `--accent-soft` | `#d9e8f5` | `#223c55` | pressed key fill, selected row |
| `--amber` | `#c97b10` | `#e2a43b` | warnings |
| `--red` | `#b3361c` | `#e26b4f` | errors, alarm DRO text |
| `--stage` | `#dedcd6` | `#2e3135` | viewport background |
| radius `--r` | 3 px | | |

Fonts: UI `Barlow`; condensed `Barlow Semi Condensed` (section heads, key labels);
mono `B612 Mono` (numbers, expressions, chips).

Fixed colours used across the workspace:

| Meaning | Colour |
|---|---|
| Axis X / Y / Z (gizmo, coordinate planes, Move fields) | `#e5534b` / `#2e7d46` / `#1877c0` |
| Body palette (auto colour by feature id: `PALETTE[(maker + 5 + 3·piece) % 8]`) | `#1877c0 #e8590c #2e7d46 #b5179e #c99a0a #0f9b9b #b3361c #6d4bbd` |
| Sketch colour on the part and in a sketch row | `#3d8fd6` |
| Cut ghost / cut grips, overlap hatch, trim open edges | `#e0564f` |
| Picked (faces, edges, regions, window box) | `#3cb371` |
| Hovered on bodies / sketch lines / grips | `#ffffff` |
| Tangent-chain edges | `#8fd6ae` |
| Body lit from its list row | `#3a9be8` |
| Picker hint (hovered plane/axis in a list) | `#ffb020` |
| Live-map "deep inside" | violet `rgb(186, 50, 196)` |
| Note badge: open / agent question / handled | `#f2a516` / `#1877c0` / `#9aa1ab` |

Body swatches (colour menu of a body row), in order, by name: black `#1e1f22`, white
`#f1f1ee`, grey `#8b9096`, aluminium `#c5c9cd`, steel `#707880`, brass `#c9a646`, copper
`#b87333`, pcb green `#1f5e33`, pcb blue `#1d3f7a`, red `#c62f2f`, orange `#e57a2e`,
yellow `#efc33a`, green `#3f9b4a`, blue `#2f73c4`, purple `#7c52b3`, then `Auto`.

### 2.2 Primitives (build these once, reuse everywhere)

| Primitive | Look | Behaviour |
|---|---|---|
| **Group** | a block of the panel; groups follow each other with the panel's gap | — |
| **Section head** (`sec-head`) | condensed, 11.5 px, 700, UPPERCASE, letter-spacing .14 em, ink; row with 7 px gap; ⓘ after the words | — |
| **Group label** (`g-label`) | condensed 10 px, 600, UPPERCASE, .12 em, dim; flex space-between, baseline; a bold count/figure at the right end | small keys may sit in it |
| **Field** (`field`) | one row: label (12 px, dim) at left, control at right, `space-between`, gap 8 | — |
| **Key/value row** (`kv`) | flex row space-between, 12 px, padding 3 px 1 px, dim; `.ghost` = faded (hidden thing); `.sel` = selected | — |
| **Check row** (`checkrow`) | checkbox + label (12 px dim), gap 7 | the label toggles |
| **Tool row** (`toolrow`) | wrapping flex row, gap 6 | — |
| **Hint** (`hint`) | small dim paragraph; `.warn` amber/red; `.rowerror` indented 17 px under a list row, wraps anywhere | — |
| **Editor box** (`draftbox`) | column, gap 9, padding 11, `--well` bg, 1 px `--line` border, **3 px accent left border**, radius 3, inset shadow | — |
| **Primary block button** | full-width accent button | the box's Create / Finish |
| **Icon row** (`iconrow`) | grid, auto-fill columns min 36 px, gap 6; keys are icon-only, padding 6/0, `--well`; hover and `.on` = `--accent-soft` fill, `--accent` border, `--accent-d` icon | each key's name is its tooltip |
| `iconrow.ten` | exactly 10 equal columns, gap 2 (Dimension + 9 relations) | — |
| **Feature key grid** (`kindrow featurekeys`) | grid of **4** equal columns, gap 6; each key: icon above word, condensed 11.5 px, 600, padding 8/2/6, `--well`, inset shadow; word ellipsised | — |
| **Nominal chip** (`chip nominal`) | pill (radius 10), mono 10.5 px, 1 px accent border, accent text, transparent; hover: accent fill, white text; never takes Tab focus | one click takes the offered value |
| **Relation chip** (`relchip`) | pill, 11 px, **dashed** `--line` border, dim; `.sel` solid accent border, ink; `.warn` amber; inner `×` | — |
| **DRO** (`dro` + `dro-window`) | label row above; window: well bg, 1 px line border, radius 2, inset shadow, padding 5/10; `.alarm`: light `#f8ece8`/border `#ddb4a8` (dark `#3b241d`/`#6e3b2c`), text `--red` 12 px | — |
| **InfoDot** ⓘ | see §0 | — |
| **RefPicker** (plane/axis list) | a button face showing the chosen label (or the placeholder, dimmed) and `▾`; opens a list with optional group headings; opens upward if no room below; empty → `Nothing to choose from` | hovering an option reports it → the workspace draws that plane/axis on the part (§6.9); ↑/↓ move, Enter takes, Esc closes; click outside closes; closing clears the hover hint |
| **Expression field** (`ExprInput`) | text input (width 172, min 108; in tight rows narrower), then an italic unit word; with an expression set, the input shows the expression and the italic tail reads `= {value}` (the field gets class `expr`), tooltip `{expr} = {value}` | commits on blur or Enter; Escape puts the old text back and stops there; invalid text stays with the reason written under the field (`hint warn`) until fixed; autocompletes parameter names; accepts `+ - * / ^`, parentheses, `sin cos tan asin acos atan atan2 sqrt abs min max floor ceil round pow`, `pi`, `e`, unit suffixes `in cm mm deg rad` |
| **Num** (labelled number) | `field`: label (+ⓘ with the hint text) · optional nominal chip `→ {rounded}` (not when an expression drives it) · expression field | value written to the box; its expression kept per slot |
| **NumberField** (`Range ±`) | app's plain number field with label | — |
| **Name field** | `field`: `Name` + text input | shown only when editing an existing feature |
| **Select** | native drop-down | — |

Nominal rounding offered by chips: halves under 10 mm and whole millimetres above;
radii to the half; angles to 5°.

---

## 3. The panel, top to bottom

The panel is a column of these blocks, in this order:

1. **Model** group (§3.1)
2. **Parameters** group (§3.2)
3. **Create feature** group (§3.3) — *shown only while no box is open*
4. Context chip bar (§3.4) — rendered from here but floats at the click
5. **Editor slot** (§4) — the one open box: Sketch, Extrude, Revolve or Feature
6. **Features** group (§3.5) — *shown when there is at least one feature (incl. rolled back)*
7. Feature-row context menu (floating)
8. **Bodies** group (§3.6) — coordinate-plane rows first, then components and bodies
9. **Notes for the agent** group (§3.7) — *shown only while the agent connection is on*
10. Divider + **export row** (§3.8) — *shown when a scan is loaded or a body exists*

### 3.1 Model group

| # | Control | Exact text | Details |
|---|---|---|---|
| 1 | Section head | `Model` + ⓘ (card title `Reverse engineering`) | Card: "Turn the scan into a **solid body** CAD opens as one: sketch on a section, a fitted plane or a coordinate plane, extrude or revolve the sketch's regions, and export the bodies as STEP." / "The body is built in the frame the part stands in now — align the part onto its fitted planes first, in the 3D Measure workspace, so the model is square." / "Modelling runs on **Open CASCADE Technology** (LGPL-2.1 with the Open CASCADE exception), compiled to WebAssembly by occt-wasm and loaded as a separate file when this workspace is first opened. Everything stays in this browser." |
| 2 | Model slot "Scan" | dot (filled `#8b9099` when loaded, hollow ring otherwise) · bold name or `No scan loaded` · second line `{n} triangles` (en-US grouping) or the formats `STL, PLY, OBJ, 3MF or STEP · mm` · button `Open…` / `Replace…` | button disabled while the app is busy; opens a file picker (several files allowed: a textured OBJ with its MTL and image); the name has the full file name as tooltip; the button may pulse as a guided hint |
| 3 | Kernel status hint | `CAD kernel: waiting.` / spinner + `Loading the CAD kernel (Open CASCADE, about 5 MB)…` / `CAD kernel ready — {version}.` / `⚠ {message}` (warn) + button `Retry CAD kernel` | Retry sets the kernel back to idle, which reloads it |

### 3.2 Parameters group (folded section)

Heading row (`sec-head`), left to right:

| Control | Text | Tooltip | Action |
|---|---|---|---|
| Fold key | `▸ Parameters` (closed) / `▾ Parameters` (open); the triangle 10 px dim | `Show the parameters` / `Fold the parameters away` | open/close; **starts closed** |
| ⓘ | card `Parameters` | — | Card: "A parameter is a named number — a length, an angle, a whole number or a plain number — given as an expression: `25`, `25.4/12`, `width/2 + 1`. Every number field in the workspace takes an expression too, and a parameter by name: an extrude `height + 5` deep, a sketch's width `width`. Change the parameter and every number that reads it follows, and the part is built again." / "Numbers add and multiply as in a calculator, with `sin cos tan asin acos atan atan2 sqrt abs min max floor ceil round pow`, `pi` and `e`; angles are in degrees. A number may carry a unit — `1 in`, `2 cm`, `30 deg` — and one without takes the field's. Kinds are kept apart: a length and an angle cannot be added, and a length times a length is an area no field wants." / "A plain number typed over a field that read a parameter lets the parameter go; so does dragging the field's grip. Renaming a parameter renames it in every expression; one that something reads cannot be removed." |
| Count | bold `{n}` | — | — |
| Add key | `+ Add` (11 px, padding 1/7) | `Add a parameter` | adds a parameter, opens the section, focuses and selects the new row's name |

Body (only when open):

* Empty: hint `No parameters yet. Add one, name it, and use the name in any number field.`
* One **parameter row** per parameter — a 2-line grid (columns `1fr auto auto auto auto`,
  gap 3/6, 1 px bottom border):
  * Line 1: **name** text field (mono 11 px, full width of col 1; tooltip
    `The parameter's name, as expressions use it`) · **kind** select
    (`Length`, `Angle`, `Integer`, `Number`; tooltip
    `What the parameter is: a length in millimetres, an angle in degrees, a whole number, or a plain number`
    — "inches" when inch units) · **value** (bold, min-width 44, `—` when none; red when
    the row is refused; tooltip = the problem) · **note key** `✎` (tooltip
    `Note: {comment}` or `Add a note on this parameter`; pressed = accent colour) ·
    **remove** `×` (tooltip `Remove the parameter — refused while anything reads it`).
  * Line 2 (full width): **expression** text field (mono; tooltip
    `The expression the parameter is given — a number, or arithmetic over numbers and other parameters`).
  * Optional line 3 (when `✎` pressed): note field, placeholder
    `A note: what the parameter is for` (tooltip `A note on the parameter`).
  * Optional last line: refusal text (`hint warn rowerror`), e.g. why a name or expression
    was refused or why the parameter cannot be removed (names what reads it).
  * Text fields commit on blur / Enter; Escape restores the shown text.

### 3.3 Create feature group

Hidden entirely while any box is open.

1. Section head `Create feature` + ⓘ (card `Features`): "**Sketch** on a plane: the slice
   the plane cuts through the scan is fitted with lines and arcs, corners are inferred and
   the profile closed. The sketch then shows on the part. **Extrude** picks the sketch's
   regions in the viewport and pulls them by a distance, to a plane, between two planes or
   symmetrically; **Revolve** turns them about a line, a cylinder's axis or a sketch line.
   Both show the body as a ghost with grips to drag while they are set up." / "Every
   extrude or revolve makes a **new body**, or **joins** or **cuts** one made before.
   Features rebuild in order whenever one changes; one whose plane or sketch has gone says
   so and the rest still build."
2. Field `Sketch on` + **plane picker** (placeholder `Choose a plane…`). Options, grouped:
   `Sections` (every section by name), `Planes` (every fitted plane element), `Coordinate
   planes` (XY/XZ/YZ at zero once the part is aligned or with no scan; through the scan's
   centre while the part is unaligned). Hovering an option draws that plane on the part
   (amber sheet + name, §6.9). Choosing one opens a new sketch on it and sets status
   `Fit the slice chains, or draw the profile with the tools, then create the sketch.`;
   failure → error `That plane is not available any more.`
3. **Feature key grid** (4 columns), in this order. Every key: icon above word.

| # | Word | Icon | Enabled when | Tooltip | Opens |
|---|---|---|---|---|---|
| 1 | `Extrude` | extrude | kernel ready and ≥1 sketch | `Extrude a sketch's regions — pick them in the viewport` | Extrude box on the **last** sketch, its regions auto-picked if it has one |
| 2 | `Revolve` | revolve | kernel ready and ≥1 sketch | `Revolve a sketch's regions about an axis` | Revolve box on the last sketch |
| 3 | `Extract` | extract | scan loaded, kernel ready, and a plane/cylinder/cone/sphere/torus element or a shadow section exists | `A measured plane, cylinder, cone or sphere as a body at its extents, extended as in 3D Measure — a boss or a pin without a sketch — or a shadow section as its block, the outline through the part's thickness` | Extract box |
| 4 | `Import STEP` | importStep | kernel ready | `Bring a STEP file in as bodies of its own — a bought-in part, a mating part, the CAD model to check the scan against. A body for every solid in the file, named as the file names its parts; Move body puts them where they belong` | file picker (`.step,.stp`) first; a file not starting `ISO-10303-21;` → error `{name} is not a STEP file — it does not begin as one (ISO-10303-21).`; then Import box named after the file |
| 5 | `Shell` | shell | a body exists and kernel ready | `Hollow the body to a wall — click the faces to open, or none for a closed hollow; the wall comes from the Thickness map when one was measured` | Shell box (wall = Thickness-map mean to 0.05 mm if measured) |
| 6 | `Fillet` | fillet | body | `Round edges of the body — click them; the radius is read off the scan across the first` | Fillet box |
| 7 | `Chamfer` | chamfer | body | `Chamfer edges of the body — click them` | Chamfer box |
| 8 | `Draft` | draft | body | `Lean faces of the body about a neutral plane — a moulding's draft, put on after the fact: click the flat face that stays, then the faces to lean` | Draft box |
| 9 | `Move face` | moveFace | body | `Offset faces of the body by a distance — a flat face along its normal, a round face in radius; several picked together grow or shrink as a whole, a cylinder's wall and its ends. One face can be snapped onto the scan instead: pick it, take Snap to scan, and click the scan where it should lie` | Move face box |
| 10 | `Delete face` | deleteFace | body | `Delete faces of the body and heal the gap — a fillet, a chamfer, a hole or a boss taken out, the faces round it run on to close over where it was: click a face of it, and the rest of the round or hole comes with it` | Delete face box |
| 11 | `Hole` | hole | body | `A hole through the body along a cylinder or circle element, or a round face of the body` | Hole box |
| 12 | `Thread` | thread (own picture: bar + blue helix lines + axis) | body | `A modeled screw thread on a round face of the body — a bolt's on a boss, a nut's in a bore: ISO metric, UNC or UNF, a 45° form that prints without supports, or a diameter and pitch of your own, with an offset for clearance when printing` | Thread box |
| 13 | `Cut by plane` | cutPlane | body | `Cut the body at a plane and keep one side` | Cut by plane box |
| 14 | `Mirror` | mirror | body | `Mirror the body about a plane, joined with the source` | Mirror box (starts the symmetry-plane search) |
| 15 | `Combine` | combine | body and ≥2 bodies | `Join, cut or intersect two bodies made separately — or trim a sheet by a solid, or sew two sheets into one, before Thicken` | Combine bodies box |
| 16 | `Sheet` | sheet | kernel ready and ≥1 sketch | `A sketch's pieces — an open chain will do — extruded as a surface: a cutter to split a body with` | Sheet box |
| 17 | `Split` | split | body and ≥2 bodies | `Split a body by a sheet and keep the fragments that lie on the scan — or trim a sheet by sheets or solids, keeping its pieces on the scan` | Split box |
| 18 | `Loft` | loft | kernel ready and ≥2 sketches | `A solid through the profiles of two or more sketches on their own planes` | Loft box |
| 19 | `Sweep` | sweep (own picture: blue path, ink bent bar) | kernel ready and ≥2 sketches | `A sketch's region swept along a chain of lines and arcs in another sketch — a bent bar, a handle, a rim that follows a curve` | Sweep box |
| 20 | `Pattern` | pattern | body | `Copies of the body about an axis, along a direction, or in a grid` | Pattern box |
| 21 | `Move body` | moveBody | body | `Move and turn bodies — the ones selected on the part or picked in the body list (Ctrl+click or Shift+click rows for several), or click them on the part: drag the gizmo's arrows and rings, or type the distances and angles. Tick Copy to move a copy and leave the body where it is` | Move body box on: the bodies of list-picked rows + bodies of the 3D selection; else the only body if there is one; else none |
| 22 | `Offset body` | offsetBody | body | `Grow a body all round by a distance, or shrink it — kept beside it as a body of its own, a keep-out round a part or the pocket that holds it with clearance` | Offset body box |
| 23 | `Fit surface` | surface | scan loaded and kernel ready | `A smooth freeform surface fitted through a region of the scan — click faces or mark them — for the trim to close into a solid with the others` | Fit surface box |
| 24 | `Patch network` | surface | scan loaded and kernel ready | `Curves drawn on the scan, and the regions they close filled with patches fitted to the scan together — tangent across every curve not marked a crease, sharing their edges: a sheet, or a solid where the network closes. For a freeform skin that folds back or wants more than one patch` | Patch network box |
| 25 | `Trim` | trim | scan loaded, kernel ready, and a surface body or a solid element exists | `Close a set of fitted surfaces, sheets and measured planes, cylinders, cones and spheres into a solid: the faces they cut space into are kept where they lie on the scan and sewn` | Trim box (all surface bodies ticked) |
| 26 | `Thicken` | shell | kernel ready and a surface body exists | `Make a sheet or a fitted surface a solid of a thickness — a freeform thin shell's wall from its one skin, where a shell or an offset of a lofted body will not go` | Thicken box on the last sheet |

("body" = at least one body exists **and** the kernel is ready.)

4. Hints under the grid, when there is no sketch yet:
   * scan loaded: `Start with a sketch: choose a plane above, or cut a section in the 3D Measure workspace.`
   * no scan: `Start with a sketch: choose a coordinate plane above. No scan is needed to model.`

### 3.4 Context chips (the bar by a click on the part)

Object first, verb second (Fusion marking menu / Onshape context toolbar). Appears only
with **no box open**, after a click on something in the 3D view. Floating
(`position: fixed`, z 30), at the click + (12, 12) px, kept wholly on screen (§1.1).
Look: `--glass` at 96 %, 1 px line border, radius 3, shadow; padding 4, gap 4; buttons
12 px, padding 3/10, icon 16 px + word, no wrap. With readings it becomes a card
(`withinfo`): max width 300 px, readings on top (full width), verbs wrapping beneath.

It goes away on: a click on nothing / on the bare scan, `Esc`, or a chip being taken
(every chip first clears the selection, then runs). Orbit/pan/zoom does **not** clear it.

**Reading card** (`chip-info`) — shown when the selection is on bodies:

1. **Body line** (only for picks on bodies): colour dot (9 px, one body only) · name
   (bold; second line small dim `in {Component › Sub}` when inside components; for several
   bodies `{n} bodies`) · keys at right (11 px):
   * `Hide` — tooltip `Hide {name} — its eye in the body list brings it back` (several:
     `Hide these bodies — {names}`); clears the selection, hides the bodies.
   * `Hide others` / `Show all` — shown when other bodies exist; tooltip
     `Hide every body but {name|these}` / `Show every body again`.
   * `Make component` — shown for picks on ≥2 bodies; tooltip
     `Put {names} into a new component in the body list`; status
     `{Component n} holds them now — double-click its name in the body list to name it.`
   A 1 px line separates the body line from the readings.
2. **Title** (11.5 px, dim, 600): one item → `Point` / `Straight edge` / `Round edge` /
   `Edge` / `Flat face` / `Cylindrical face` / `Conical face` / `Spherical face` /
   `Toroidal face` / `Freeform face`; several → `{n} faces, {n} edges, {n} points`.
3. **Rows** (label dim left, value mono 11.5 right, selectable text):
   * corner: `X`, `Y`, `Z`
   * straight edge: `Length`, `Direction`, `From`, `To`; round edge: `Length`, `Radius`,
     `Diameter`, `Centre`, `Axis`; other edge: `Length`, `Middle`
   * face: `Area`; flat adds `Normal`, `Centre`; round adds `Radius`, `Diameter`; sphere
     `Centre`; cylinder/cone/torus `Axis`; cylinder/cone `On axis`
   * two items: point–point `Distance`, `ΔX`, `ΔY`, `ΔZ`; point–face `Distance to the face`;
     point–line `Distance to the axis` / `Distance to the edge`; face–face parallel
     `Distance` + `Angle: parallel`, else `Angle`; line–face along `Distance` + `Angle:
     parallel`, else `Angle to the face`; line–line parallel `Distance` (or
     `Distance between the axes`) + `Angle: parallel`, else `Angle` + `They cross: in one
     point` or `Closest: {d}`
   * sums: `Length, together` / `Edge length`; `Area, together` / `Face area`
4. Tip line (10.5 px, faint), single picks only: `Ctrl+click adds · Shift+drag a window`.

**Verb chips** by what was clicked (each with the feature key's icon):

| Selection | Chips, in order (label — tooltip) |
|---|---|
| Sketch region | `Extrude` — `Extrude this profile — the ghost and its arrows appear on the part`; `Revolve` — `Revolve this profile about an axis picked next` |
| Element line/cylinder/cone/circle | `Revolve about this` — `Revolve the last sketch's profile about {element}` |
| Element cylinder/circle (and a body exists) | `Hole` — `Drill a hole through the body along {element}` |
| Element plane | `Sketch` — `Start a sketch on {element}` |
| Element plane/cylinder/cone/sphere/torus | `Extract as body` — `{element} as a solid at its extents, extended as in 3D Measure` |
| Coordinate plane | `Sketch` — `Start a sketch on the {XY plane}`; `Hide` (eye-off icon) — `Put the {name} away — its eye in the body list brings it back` |
| Sketch line | `Revolve about this` — `Revolve the sketch about this line` |
| Body face | `Edit {feature}` — `Open {feature}, the feature that made this face, to change it` (when its maker is a feature); flat face: `Extrude` — `Pull this face: drag the arrow out to add to the body, in to cut into it`, `Sketch here` — `Start a sketch on this face of the body, its outline projected into the sketch`; movable and not a partial round: `Move face` — cylinder: `Offset this round face: type how much its radius grows or shrinks — Ctrl+click its ends too to make the whole cylinder longer and wider`, flat: `Offset this face along its normal by a typed distance — Ctrl+click more faces to move them together`; snappable and not partial: `Snap to scan` — `Move this face onto the scan: click the scan where the face should lie — a plane or a cylinder is fitted there and the face moved to it`; always `Shell` — `Hollow the body, this face taken away to open it`; `Delete face` — `Take this face off the body and heal the gap — with the rest of the round, the hole or the boss it is part of`; cylinder with centre: `Hole` — `A hole along this face’s axis`, and (full turn only) `Thread` — `A modeled screw thread on this round face — a bolt’s on a boss, a nut’s in a bore, the standard size for it to start`; flat: `Mirror` — `Mirror the body about this face`, `Draft` — `Lean faces of the body about this face as the neutral plane — click the faces to lean next`; always `Move body` — `Move or turn this body — drag the arrows and rings on the part, or type the distances and angles; tick Copy to move a copy instead` |
| Body edge | `Fillet` — `Round this edge — the radius is read off the scan`; `Chamfer` — `Chamfer this edge`; `Move body` |
| Body corner | none (readings only) |
| Several edges of one body | `Fillet` — `Round these {n} edges`; `Chamfer` — `Chamfer these {n} edges` |
| Picks on exactly 2 bodies | `Move bodies`; `Clearance` — `How far apart the two bodies stand at the closest — or how much of them runs into each other` (result → status line) |
| Picks on >2 bodies | `Move bodies` — `Move or turn these bodies — …` |
| Several faces of one body | (all movable, none partial) `Move faces` — `Offset these {n} faces together by one distance — the flat ones along their normals, the round ones in radius`; `Shell` — `Hollow the body, open at these {n} faces`; `Delete faces` — `Take these {n} faces off the body and heal the gap` |
| Any body pick, agent connection on | appended `Note` — `Pin a note for the agent on this {face|edge|corner}` / `…on these — say what to do with them` |

Status set by chip actions: sketch on a plane → `Fit the slice chains, or draw the profile
with the tools, then create the sketch.`; `Sketch here` → `Sketch on the face — its outline
is projected in[, all but {n} curved edge(s) the sketcher has no piece for]. Draw on it, or
Finish and extrude the outline itself.`; `Snap to scan` → `Now click the scan where this
face should lie.`

### 3.5 Features group (the feature list)

Shown when there is at least one feature (above or below the rollback bar).

**Heading** (`g-label`), left to right:
* `Features` + ⓘ (card `The feature list`): "The part is built from the top down. The
  **rollback bar** under the list is dragged up to go back in its history: the features
  below it are set aside, the part stands as it stood there, and whatever is made next
  goes in at the bar — a fillet before the shell, a cut before the pattern. Drag it back
  down, or press **To end**, and the rest is built again on top of what was added." / "A
  feature below the bar that loses what it stood on says so by name in its row, as any
  feature does." / "**Play** builds the part up again step by step for the camera: the bar
  goes to the top and moves down one feature at a time, each held a moment once it is
  built, and returns where it was at the end. **Space** holds it or carries it on, the
  **arrow keys** step it, **Escape** ends it. The caption in the viewport names each step."
* while building: spinner + `building`
* **Play key** (hidden during playback): play icon + `Play`; disabled while a box is open
  or the list is empty; tooltip `Play the feature list back step by step: the part built up from nothing, one feature at a time, each held a moment — for recording how it was made. Space holds it, the arrows step it, Escape ends it`
  (disabled: `Finish what is open first`).
* bold count: `{n}`, or `{built} of {total}` while rolled back.

**Playback bar** (only during playback; under the heading; well background, 1 px line,
radius 3, padding 6, wraps):
* keys (icon only, padding 3/6): **Step back** (tooltip `One step back — the playback held there (←)`, disabled at the start) ·
  **Pause** (`Hold the playback at this step (Space)`) *or* **Play** (`Carry on from this step (Space)` / at end `Play again from the top (Space)`) ·
  **Step forward** (`One step on — the playback held there (→)`, disabled at the end) ·
  **Stop** (`End the playback — the rollback bar back where it stood (Escape)`).
* `Hold` + select of `0.5 s`, `1 s`, `1.5 s` (default), `2 s`, `3 s`, `5 s` — tooltip
  `How long each step is held once it is built`.
* checkbox `Caption` (default on) — `Name each step in the top left of the viewport, for the recording`.
* checkbox `Skip sketches` (default off) — `Pass over the sketches: only the steps that change the geometry are shown, each profile going straight to the body made of it`.

**Rows** — one per feature, top to bottom, with the rollback bar in the gap after the last
built feature. A row (`ElementRow`), left to right:
1. kind icon (15 px) in the row colour — sketch rows `#3d8fd6`; other rows the colour of
   the body they made/changed (its style colour, else its auto colour).
   Icons: sketch→sketch, extrude→extrude, revolve/thread→revolve, shell/thicken/offset
   body→shell, fillet, chamfer, hole, extract, cut by plane→cutPlane, mirror, combine,
   sheet, split, loft/sweep→loft, pattern, draft, move face, delete face, import→importStep,
   move body→moveBody, fit surface/patch network→surface, trim.
2. name (`Sketch 1`, `Extrude 2`, …).
3. reading (only one of): failed → bold warn `⚠` (tooltip = the failure message); open trim
   → `⚠ open`; a number an agent fitted → bold `fit` (tooltip lists
   `{slot} fitted to the scan: {v} ± {σ} (nominal {n}, {k} σ off, {p} points)`).
4. **Edit** key ✎ (tooltip `Edit {name} — change what it is measured on or built from`,
   disabled with `Finish what is open first` while a box is open or playback runs).
5. **Eye** (sketch rows: show/hide the sketch on the part; tooltips
   `Hide {name} in the viewport` / `Show {name}`) — *or*, on every other row, the
   **suppress key** (built icon; tooltip `Suppress {name} — leave it out of the build, its picks and numbers kept`;
   disabled `Finish what is open first`).
6. **Bin** (tooltip `Delete {name}`); refused with error
   `{name} is built on by a later feature — delete that first.` if anything depends on it.

Row tooltip ("about"): sketch `{n pieces…} on {plane}`; extrude/revolve
`{Joins X | Cuts X | New body | Join · touches no body, a body of its own | Cut} · {10 mm | 360°}{ · slot = expr} · {n profile(s) of Sketch k | pulls a face of Body}`;
others `{reading} — {sentence}` (+ `· σ {rms}` for a fitted surface, `· solid, {k} of {n} faces`
or `· open, {k} open edge(s)` for a trim).

Row states: **selected** (accent soft fill) for the feature being edited or the one the
playback is on (kept scrolled into view); **muted/rolled** for features after the one being
edited; double-click = Edit (same conditions).

Under a row (indented 17 px, wraps anywhere): failure message (`hint warn`), a build note
(`hint`), a warning (`hint warn`), expression errors joined with ` · ` (`hint warn`).

**Suppressed row**: greyed; kind icon; name struck through; bold faint `suppressed`;
disabled ✎ (`Build {name} again to edit it`); struck suppress key (tooltip
`Suppressed — build {name} again`, click un-suppresses); bin `Delete {name}`. Row tooltip
`{name} is suppressed — left out of the build, its picks and numbers kept. Right-click the row for its menu.`

**Rolled-back rows** (below the bar): faded `kv ghost rolled`; dot; name; bold
`rolled back` (during playback `to come`); tooltip
`{name} is below the rollback bar — not built until the bar is moved below it` (playback:
`{name} is still to come in the playback`).

**Rollback bar** (in the gap after the last built row):
* a full-width rule: 3 px, radius 2, accent at 55 % opacity (100 % on hover/drag),
  min-height 14 px, cursor `ns-resize`. Locked (`.off`, faint colour, default cursor)
  while a box is open or playback runs.
* tooltip: `The rollback bar — drag it up to set the features below it aside and add one earlier in the history; drag it down, or press To end, to build them again`
  (locked: `Finish what is open first`; playback: `The playback moves the rollback bar — stop it to move the bar by hand`).
* drag (left button): the bar stays put; a **mark** (3 px accent rule with a 2 px
  accent-soft halo) shows the gap nearest the pointer (counted by row mid-points); on
  release the list is rolled there (one undo step).
* `To end` key (11 px) at the bar's right, only when something is rolled back and not
  dragging/playing; disabled while a box is open; tooltip
  `Move the rollback bar to the end of the list: every feature is built again`.

Last: build error `⚠ {message}` (`hint warn`).

**Feature-row context menu** (right click on any row, not while a box is open or the bar is
locked): head = feature name (condensed 10 px uppercase); items:
* `Suppress` (`Leave {name} out of the build — kept in the list, its picks and numbers as they are`)
  or `Build again` (`Put {name} back into the build, as it was`);
* for kinds fillet, chamfer, hole, thread, draft, delete face, pattern, shell, mirror,
  extrude, revolve with ≥2 of the kind: `Suppress every {kind} ({n})` /
  `Build every {kind} again ({n})`;
* on a fillet or chamfer when both kinds exist: `Suppress every round and chamfer ({n})` /
  `Build every round and chamfer again ({n})`.
Closes on an item, a click outside, Escape or a scroll.

### 3.6 Bodies group (the body list — an assembly browser)

**Heading** (`g-label`): `Bodies` · (when bodies exist) `Component` key (component icon,
condensed 10 px uppercase; tooltip with rows picked
`A new component holding {what} — a folder to sort bodies into`, else
`A new component — a folder to sort bodies into, an assembly's parts by what they make up: drag bodies onto it, or pick rows first to put them in`)
· **Show all / Hide all** key (eye icon 13 px + `Hide all` / `Show all`; tooltip
`Hide all bodies in the viewport` / `Show all bodies`) · bold count `{n}`.

**Coordinate-plane rows** (always first, order **XY, XZ, YZ**): skewed square dot in the
axis colour (XY = Z blue `#1877c0`, XZ = Y green `#2e7d46`, YZ = X red `#e5534b`) · name
`XY plane` etc. · bold `z = 0` / `y = 0` / `x = 0` · eye (`Hide the {name}` /
`Show the {name} at the origin`). Row tooltip
`The {name} through the origin — click it in the viewport to sketch on it`. Faded when
hidden. Default: shown on a blank stage, hidden over a scan.

**Component rows** (indented 14 px per nesting level): fold `▾`/`▸` (14 px; tooltip
`Close {name}` / `Open {name}`) · component icon (dim) · name (500 weight; double-click to
rename — tooltip `Double-click to name the component`; rename field: Enter keeps, Escape
cancels, blur keeps; placeholder `Name`) · bold dim count of bodies · eye of all its bodies
(`Hide every body in {name}` / `Show every body in {name}`; disabled when empty). Row
tooltip `{name} — {n} body/bodies. Drag bodies onto it to put them in; right-click for more`.
A closed component's row lights for what it holds.

**Body rows** (indented by depth): colour dot 11 px (a key: tooltip
`Colour the body — the STEP file carries it`; click toggles the swatch strip under the row)
· name (double-click renames; tooltip
`Double-click to name the body — the name it has in a STEP file; cleared, it is Solid n again`)
· bold reading: `surface` / volume (`{v} mm³`, 1 decimal) / `⚠ open`; tooltip
`{faces} faces · {triangles} triangles drawn` · eye (`Hide the body` / `Show the body`).
Swatch strip: 18 px squares radius 4 (tooltip = swatch name capitalised), the chosen one
ringed accent; then `Auto` (`The colour the body had — by the feature that made it`).

Row marks: hover = well fill; **selected** (picked in the list) = accent-soft fill; **lit**
(its body under the cursor in the viewport) = 1 px inset accent ring; **picked** (a face/edge
of it selected in the viewport) = 2 px accent bar on the left edge + bold name.

Row interaction:
* hover a row → its body (component: all bodies in it) lit in the viewport (§6.4).
* click = pick only that row; Ctrl/Cmd+click toggles; Shift+click spans from the anchor
  row (Ctrl+Shift adds the span). Clicking the list background clears the pick.
* **While a Move body box is open**, clicks put bodies into the Move or take them out
  (component row = all its bodies; Shift+click adds every row up to it), with hint
  `Click rows to put bodies into the Move or take them out — a component's row all of its bodies; Shift+click puts in every row up to it.`
* drag a row (or all picked rows, ghost label `{n} bodies` in an accent pill) onto a
  component row → into it (the target row shows a dashed/accent drop look); onto a body
  row → into that row's component; onto the strip `Drop here to take it out of its component`
  (shown under the list only while dragging) → to the top level.
* right click → menu (below).

**Selection bar** (under the rows, when rows are picked and no Move is open; accent-soft
background, radius 3, padding 5/6): bold words (`Shaft`, `3 bodies`, `2 bodies and a
component`) · `Move` (moveBody icon; tooltip
`Move and turn these bodies together — or copies of them — by the gizmo or by numbers`;
disabled with `Finish what is open first` when a box is open) · `Component` (`Put what is
picked into a new component`) · `Hide others` / `Show all` (only when 0 < picked < all) ·
`✕` (`Let the rows go (Escape)`).

**Footer hint** (when bodies exist): `Drawn solid, the scan a see-through veil over them. The view bar's Opaque key turns that round; its Scan key puts the scan away, its Bodies key the bodies — the map staying on the scan. Ctrl+click or Shift+click rows to pick several, and drag them onto a component to put them in; right-click a row for more.`

**Body-row context menu** (head = what is picked in words): `Hide others` / `Show all`;
`Move body` / `Move {n} bodies` (not while a box is open); for one component:
`New component inside`, `Rename`; otherwise `New component with this body` / `New component
with these`; `Move to {A › B}` for every component that can take it; `Move to the top of the
list` (when inside one); for one component `Remove component` (`Take {name} away — what it
holds moves up a level; no body is deleted`). Closes on item, outside click, Escape, scroll.

### 3.7 Notes for the agent (only while the agent connection is on)

**Heading**: `Notes for the agent` + ⓘ (card `Notes for the agent`: how to pin, select
several and take Note from the chips, how the agent answers, blue question pins, saved
with the project, not undo steps, shown while Settings → Agents is on) · bold count
`{n} open · {m} question(s)` or `all handled`.

**Keys** (tool row):
* `Note` (note icon 18 px; while placing: pressed + `Click the part…`). Disabled when no
  scan and no body (`Load a scan or build a body first — a note is pinned on the part`) or
  a box is open (`Finish what is open first`). Tooltips: placing
  `Click the part where the note goes — or here, or Escape, to stop`; idle
  `Pin a note for the agent on the part: click where it goes next`. Starting sets status
  `Click the part where the note goes — a face, an edge or a corner of a body, or the scan. Escape stops.`;
  the canvas cursor becomes a crosshair.
* `Delete all` (bin icon; only when notes exist) — tooltip
  `Take every note off the part — yours and the agent’s questions, open and handled. Not a step of the undo history`;
  asks `Delete all {n} note(s) for the agent? This cannot be undone.`

**Rows**: badge (pill 18 px, mono 10.5 bold, 1.5 px `#1d2026` border; amber `#f2a516`;
agent question blue `#1877c0` white text; handled grey `#9aa1ab`) · body key (opens the
note card at the row; tooltip `Edit the note` / `Answer the agent’s question`): text;
small `on {where}` or `The agent asks, on {where}`; reply line `Agent: …` / `You: …`;
`Click to answer` for an unanswered agent question · `✓` (`Mark handled` / `Handled — click
to open it again`) · `✕` (`Take the note off the part`). Where-text examples: `a round face
of Solid 1`, `3 edges of Solid 2`, `a corner of Solid 1`, `the scan`, `the part`.
Empty: `Pin what you want done on the part — the agent reads each note with the face, edge or place it is on.`

**Note card** (floating, chip-bar look, placed like the chip bar): head = badge + `on {where}`;
textarea (3 rows, autofocus, placeholder `What should the agent do here?`); agent reply
paragraph `Agent: {reply}`; keys `Delete` (`Take the note off the part`) and primary
`Pin note` / `Save` (`Keep the note (Enter, or a middle click)`; disabled when empty).
Enter keeps, Shift+Enter breaks the line, Escape closes unchanged, a **middle click
anywhere** (≤ 4 px movement) keeps — or closes when empty. Status on keep:
`Note {n} pinned — the agent reads it with what it is on.`
**Answer card** (agent question): head `The agent asks, on {where}`; question paragraph;
textarea (2 rows, placeholder `Your answer`); `Delete` (`Take the question off the part`),
primary `Answer` (`Give the agent this answer (Enter, or a middle click)`); status
`Note {n} answered — the agent reads the answer in its next look at the session.`

### 3.8 Export row (bottom of the panel)

Shown when a scan is loaded or any body exists; preceded by a divider. Tool row with two
buttons (icon 18 px + word):

| Button | Enabled when | Tooltip |
|---|---|---|
| `Export body STEP` | a visible body that is closed, or a visible Patch-network sheet, and not building | `Write every visible body into one STEP file as solids — what FreeCAD and Fusion open as a body` |
| `Compare in Deviation` | a scan is loaded, a visible non-surface body exists, not building, not busy | `Measure the scan against the visible bodies in the Deviation workspace — its scale, histogram, pins and report. The bodies go in as the reference part; no best fit is run, they stand in the scan's frame already` |

Export with nothing visible → error `No visible body to export — build one first.`
Compare → switches to Deviation; status
`The scan measured against the bodies from Reverse engineer, in the scan’s own frame — no best fit was run.`

---

## 4. Editor boxes

### 4.1 What every box shares

* Frame: `draftbox` (§2.2). Title (`sec-head`): `New {kind}` (lower-case kind:
  `New extrude`, `New fillet`, `New cut by plane`, `New combine bodies`, `New patch network` …),
  `Edit {name}` when reopened from a row; Import STEP and Move body use their bare title
  `Import STEP` / `Move body`.
* When editing an existing feature: `Name` field directly under the title.
* While a box is open: the Create-feature group is hidden; feature rows after the edited
  one are drawn muted; row keys stand down; the rollback bar is locked; the body list's
  Move key is disabled.
* **Preview line** (`hint`, warn when failed) — one sentence about the ghost:
  * idle: `Pick a profile to see the body.` (Fit surface: `Click the scan to gather the region; the surface follows.`;
    Patch network: `Click the corners of a patch on the scan; the first again closes it.`;
    Move body: `Drag an arrow or a ring of the gizmo on the part, or type the distances and angles.`;
    Import: `Choose the STEP file to bring in.`)
  * building: spinner + `Building the preview…` (Import: `Reading the file…`)
  * ok (solid): `Preview: {volume} — drag the grips in the viewport, or type below.`;
    not closed: `Preview: {volume} · ⚠ not a closed solid[. {reason}] — drag the grips …`
  * import ok: `The file holds {volume} of solid[, and faces without an inside — a surface body].`
  * trim: `Closed: {k} of the {n} faces the surfaces cut lie on the scan and sew into a solid of {volume}.` /
    `⚠ Not closed: {k} of {n} faces lie on the scan, and they leave {m} open edge(s) — drawn red on the part. Fit the surface that is missing there, or widen the tolerance; Create keeps the faces as a surface body meanwhile.`
  * fitted surface (freeform): `Fitted: σ {rms}, the worst point {max} off, a {u} × {v} net over {N} points.`
  * fitted plane/cylinder: `A plane` / `A cylinder of R {r}` + `: σ {rms} over {N} points` +
    either ` ({rmsFree} before it was held). {Parallel to XY plane} — the fit was {x} and {y}° off; ….`
    or `. Held to nothing.`
  * sheet: `A sheet of {n} face(s) — a cutter to split a body with, or a skin to thicken.`
  * network: `A solid of {volume}: the network closes.` /
    `A sheet: the network is open — Trim, Thicken or Split take it as they take a fitted surface.` / `⚠ {failure}`
  * failed: `⚠ {message}`
* **Analysis row** (live deviation map), in every box:
  * checkbox `Deviation map` + ⓘ (card `Deviation map`: "Colours the scan near the ghost by
    how far it lies from the ghost's surface — green on it, red outside the body, blue
    inside — on the Deviation ramp at ± this range, and darker past it. The rest of the scan
    keeps its colour. Drag the grips and watch the feature settle onto the scan.").
    Default **on**, remembered across sessions.
  * when on: number field `Range ±` (length, min 0.005 mm, default **0.1 mm**, max 100,
    remembered) and a hint: `The map appears with the ghost.` /
    `Reading the scan against the ghost…` / `No scan near the ghost.` /
    `{within} of {points} scan points near the ghost lie within ±{range} of it.`
* **Result** select (`OpFields`, where a feature makes a solid): label `Result`; options
  `Join` (tooltip `Made one with every body it touches; a body of its own where it touches none, one body of all where it bridges several`),
  `Cut` (disabled when no solid body to cut; `Taken out of every body it goes into; a body it parts is a body a piece`),
  `New body` (`A body of its own, whatever it overlaps — a tool to cut the scan with in Mesh Tools`).
* **Create** — primary full-width button, disabled until ready; tooltip
  `Enter or a middle click does the same` (feature boxes: `Enter does the same`). Labels:
  `Create extrude` / `Create revolve` / `Create coil` / `Create {kind}[ on {Body}]` /
  `Save changes` (editing) / `Import` / `Move the body|bodies` / `Copy the body|bodies` /
  `Delete the face|faces[ of {Body}]` / `Move the face` / `Move the faces` (Move face that
  edits history instead).
* **Cancel** — plain button in a tool row under Create. Escape does the same (except in a
  sketch: see §4.2).
* `Clear` key (`clearpicks`, beside a pick count) — tooltip `Let every picked one go and pick again`.

### 4.2 Sketch box (`SketchEditor`)

The sketch takes the stage: the 2D sheet lies over the viewport (§7) and the view bar
shows the three part-view keys (§5.2). **There is no Cancel**: Finish is the only exit;
Escape only backs out of the tool; an unwanted sketch is undone with Ctrl+Z.

Controls, top to bottom:

1. Title `New sketch` / `Edit {name}` + ⓘ (card `Sketching`): "The sketch lies on its plane —
   a section, a plane element, a coordinate plane or a face of a body — and is drawn on the
   part, which faces the plane and is cut open at it; the view bar's **Model** and **Slice**
   keys put the part away and show it whole." / "**Fit all chains** cuts every slice chain
   into lines and arcs, each keeping its slice points: the solver holds every piece to its
   points while the relations and the dimensions you type are met exactly. **Infer corners**
   joins ends within 2 mm and adds a tangency where a line runs into an arc; **Close**
   bridges what is left open." / "Every closed region is filled faintly; **Extrude** and
   **Revolve** appear as soon as one closes and take it straight into a feature. **Finish
   sketch** keeps the drawing as a feature for later; Escape only puts the tool down, and a
   sketch you did not want is undone with Ctrl+Z from the feature list." / "**Dimension**
   (D) and the relations work both ways round: on the pieces selected, or, with nothing
   selected, on what is clicked next. The Line tool (L) draws a chain, each line from the
   end of the last; press on that end and drag for a tangent arc. A new dimension's number
   follows the cursor until a click puts it down, and opens there to be typed; Escape takes
   it back. Later it is dragged to where it reads best, and clicked to be typed over."
2. **`Finish sketch`** — primary full-width (top copy). Tooltip
   `Finish the sketch — Enter or a middle click does the same. With nothing drawn, nothing is made.`
3. `Name` field (only when editing an existing sketch).
4. Plane hint: `On {plane description}` + ` · {n} slice chain(s)` once the slice is cut.
5. **Offset / plane position** (only for planes that take one): label
   `Plane at {X|Y|Z} =` (coordinate plane) or `Offset`; length expression field. Tooltips:
   `Where the coordinate plane lies along its axis — a number, or an expression over the parameters the sketch then follows`
   / `How far the sketch stands off its plane, along the plane’s normal — out of a face is positive. A number, or an expression over the parameters`.
6. **Level**: `Level` + length expression field (default 0). Tooltip
   `Cut the scan this far off the sketch’s plane, along its normal, and fit that level’s outline into this sketch — another height of a stacked part. Fit all chains then lands on the pieces already drawn: a wall they have is not fitted again, and the new walls end on theirs. 0 cuts at the plane (a face’s a hair inside it).`
7. **Picture row**
   * no picture: button `Picture…` (sketchImage icon 18) — tooltip
     `Lay a photo of the part or a dimensioned drawing under the sketch, to trace — then bring it to scale by two points a known length apart`
     (opens an image picker; error `{file} could not be read as a picture.`).
   * with a picture (`subbox`: grid gap 6, padding 8, 1 px border, well bg):
     * row: icon · file name (tooltip = name; `Picture` if none) · bold scale
       `{x.x} µm/px` (tooltip `Micrometres per pixel of the picture`) or `{n} px/in`
       (`Pixels per inch of the picture`) · `×` (`Take the picture away`).
     * keys (toggle, pressed while active): `Scale` (`Bring the picture to scale: click two points on it, then type how far apart they are on the part`),
       `Move` (`Move the picture: click a point on it, then where that point belongs — an end, the origin, an axis snaps`),
       `Level` (`Turn the picture level: click two points on it that should lie on a horizontal line`).
     * after two Scale points: `Apart on the part` + length field, and primary `Scale`
       (disabled until a length is typed; Enter in the field applies).
     * `Shown` slider 0.1 – 1, step 0.05 (tooltip
       `How much of the picture shows: faded, the scan and the pieces show through it`).
8. **Align nudge** (first sketch, part unaligned, scan loaded, new sketch): hint
   `Part not aligned yet.` + chip `Align in 3D Measure…` (tooltip
   `The part is still in the scanner's frame, and the body will be built in it — a STEP file that lies askew in CAD. Aligning the part onto its fitted planes first makes the model square. This opens the 3-2-1 alignment in the 3D Measure workspace; the sketch stays open and moves with the part.`).
9. **Fit row** (tool row, icon 18 + word):
   * `Fit all chains` — enabled when the slice is cut and some chain is unfitted; tooltip
     `Cut every slice chain into lines and arcs, each with its slice points behind it`.
   * `Infer corners` — enabled when something is fitted; tooltip
     `Join neighbouring ends within 2 mm, with a tangency where a line meets an arc`.
   * `Close` — enabled when something is fitted; tooltip
     `Bridge every gap left open along the fitted chains`.
10. Chip `Fit options…` — toggles the Fit options window (§4.3); tooltip
    `How fine the slice is cut, the fit tolerance, corners joined, lines laid along the axes — in a window of its own; a change there refits the chains`.
11. **Tool row** (`iconrow`, icon 22 px, each key's tooltip = `{label} — {text}`), in order:

| Key | Shortcut | Tooltip text after the label |
|---|---|---|
| `Select` | — | `Click a piece to select it, a second to pair it; click empty sheet to clear. Drag over empty sheet for a box: rightward it takes what lies wholly inside, leftward whatever it touches; hold Shift or Ctrl to add to the selection. Drag an end, a centre or a piece to move it — it snaps to other ends, the slice and the pieces, and a line moves square to itself; hold Ctrl to move it freely. The right button pans` |
| `Line (L)` | L | `Click the start, then every next corner — each line carries on from the last. Press on the last end and drag for an arc tangent to the piece before it. Escape, Enter or a second click on the last end ends the chain; so does landing on another piece’s end. Hold Ctrl to place a point without snapping` |
| `Rectangle (R)` ▸ | R | `Two opposite corners, the centre and a corner, or one side and the far side at any angle — rest the cursor on the key to choose; the key shows the way last chosen. The rectangle follows the cursor between the clicks` |
| `Polygon` ▸ | — | `A regular polygon: its centre, then a corner — or the middle of a side, for one sized across its flats like a hex nut. The sides are equal and drawn about a construction circle to dimension; how many there are is set under the keys while the tool is in hand` |
| `Circle (C)` ▸ | C | `The centre and a point on the rim, the two ends of a diameter, or three points on the rim — rest the cursor on the key to choose; the key shows the way last chosen. The circle follows the cursor between the clicks` |
| `Arc` ▸ | — | `The start, the end and a point it passes through, or the centre, the start and where it ends — rest the cursor on the key to choose; the key shows the way last chosen. The arc follows the cursor` |
| `Spline (S)` | S | `A smooth curve through the points clicked: click each point in turn — the curve follows the cursor. Click the last point again (or double-click it), press Enter or Escape to end it open; click the first point to close it round. Drag a fit point to bend it. Joined to a line or an arc at an end, it can be held tangent there. With a spline selected, Fit to slice pulls it onto the slice it was drawn over` |
| `Fillet (F)` | F | `Round a corner: click the corner point where two lines meet, or one line and then the other. With two lines selected, the key rounds those. The lines need not touch — a short piece bridging their ends is replaced by the round. The radius comes out as a dimension to type over; the next round starts with the last radius` |
| `Chamfer` | — | `Cut a corner off straight: click the corner point where two lines meet, or one line and then the other. With two lines selected, the key chamfers those. The two distances come out as dimensions to type over — alike for an even chamfer, apart for an uneven one; the next chamfer starts with the last distance` |
| `Project` | — | `Click an edge of a body, or a piece of another sketch, on the part under the sheet: it comes into this sketch as a piece of its own, fixed where it lies — to close a profile with, or to hang relations and dimensions on. It lights under the cursor when it can be taken` |
| `Snap to grid` (gridSnap icon; toggle) | — | full tooltip `Snap to grid — drawn and dragged points snap to the grid as well as to ends, the slice and pieces — off, the grid is only there to read against`; default **off**, remembered |
| `Show relations` (eye / eye-off; toggle) | — | full tooltip `Show relations — draw every relation beside its piece as a glyph. Off, only the one selected in the list below is drawn; the dimensions are always shown`; default **off**, remembered |

   * Pressed key = tool in hand (`on`). Pressing the tool key again puts it down (back to
     Select) for Fillet/Chamfer and for keyboard letters. Fillet/Chamfer pressed with exactly
     two lines selected act on them at once.
   * **Shape-tool flyouts** (▸ = keys with more ways; a small corner tick at the key's
     bottom-right, 45 % opacity): resting the mouse **250 ms** on the key (or tab focus, or
     pressing the key again while in hand on touch) opens a strip *under* the key (44 px
     keys, glass panel, shadow, z 25); it closes 180 ms after the pointer leaves. The chosen
     way becomes the key's picture and is remembered. Choosing a way also takes the tool.

| Tool | Way (picture) | Tooltip `{label} — {title}` | Hint per click |
|---|---|---|---|
| Rectangle | `Corner rectangle` (default) | `Two opposite corners` | `Click one corner.` → `Click the opposite corner.` |
| | `Centre rectangle` | `The centre, then a corner — the rectangle stays about its centre, a point where its construction diagonals cross` | `Click the centre.` → `Click a corner.` |
| | `3-point rectangle` | `The two ends of one side, then where the opposite side lies — a rectangle at any angle` | `Click one end of a side.` → `Click the other end of that side.` → `Click where the opposite side lies.` |
| Circle | `Centre circle` (default) | `The centre, then a point on the rim` | `Click the centre.` → `Click a point on the rim.` |
| | `2-point circle` | `The two ends of a diameter` | `Click one end of a diameter.` → `Click the other end.` |
| | `3-point circle` | `Three points on the rim` | `Click a point on the rim — 1 of 3.` → `— 2 of 3.` → `— 3 of 3.` |
| Arc | `3-point arc` (default) | `The start, the end, then a point the arc passes through` | `Click the start.` → `Click the end.` → `Click a point the arc passes through.` |
| | `Centre arc` | `The centre, the start, then where it ends — the shorter way round; a longer arc is drawn through three points` | `Click the centre.` → `Click the start — how far it lies is the radius.` → `Click where it ends — it runs the shorter way round.` |
| Polygon | `Inscribed polygon` (default) | `The centre, then a corner — the corners lie on the construction circle` | `Click the centre.` → `Click a corner.` |
| | `Circumscribed polygon` | `The centre, then the middle of a side — the sides touch the construction circle, whose diameter is the width across flats` | `Click the centre.` → `Click the middle of a side.` |

12. **Polygon sides** (only while Polygon is in hand): row `Sides` + number input 3 … max,
    step 1, default **6**, remembered; tooltip `How many sides the polygon has — three and up`.
13. **Fit spline to slice** chip (only when a spline is selected; enabled when the slice is
    cut) — tooltip `Fit the selected spline to the slice it was drawn over: the slice points along it — within a twentieth of its length — become its soft data, and the solver pulls the curve onto them, its fit points moving as far as its relations let them. Add fit points where the outline bends more than the curve can follow`.
14. **Hint line** (`hint`, only when there is something to say) — first that applies:
    the draft's error; the picture pick (`Click a point on the picture whose distance to another you know.`
    → `Click the other point.` → `Type how far apart the two points are on the part.`;
    Move: `Click a point on the picture.` → `Click where that point belongs.`; Level:
    `Click a point on a line of the picture that should lie level.` → `Click another point on that line.`);
    `Cutting the scan on the sketch plane…`; the shape tool's click hint (table above);
    Spline: `Click the first point of the curve.` → `Click the next point — the last one again, or Enter, ends the curve.`
    → (≥3) `Click the next point — the last one again, or Enter, ends it; the first point closes it.`;
    Dimension: `Click what to dimension: a line, a round piece, an end — or a coordinate plane’s line.`
    → `Click the second piece, end or coordinate plane.`; with a number in hand
    `Click where the number goes.` or `Click where the number goes — or a second piece, end or plane for the distance or angle to it.`;
    Fillet/Chamfer: `Click a corner point, or the first of its two lines.` → `Click the second line of the corner.`;
    Symmetric: `Symmetric: click the first of the two pieces that mirror each other.` →
    `Click the piece that mirrors it.` → `Click the line they mirror about — a construction line, or a coordinate plane’s line on the sheet.`;
    Coincident: `Coincident: click an end, a centre or a piece — then what it lies on: another end, a line, an arc, or a coordinate plane’s line (the amber lines).`
    → `Now the second: an end, a centre, a piece, or a coordinate plane’s line.`;
    other relations: `Click the piece to put it on.` → `Click the second piece.`
15. Group label `Dimension · Relations` with bold `{n} selected` at the right.
16. **Dimension + relations row** (`iconrow ten`, icons 20 px; tooltip = name only):

| # | Key | Tooltip | Click |
|---|---|---|---|
| 1 | Dimension (dimDistance icon) | `Dimension (D) — click a line for its length, an arc for its radius, a circle for its diameter (Ø — typed as R… it goes by the radius instead); click a second piece for what lies between the two: the distance between parallel lines, the angle between others, the distance from an end or a centre to a line or to another, or to one of the part’s coordinate planes (the amber lines named XY, YZ, ZX). The number then follows the cursor: click where it goes, and type the value; Escape takes it back. With pieces selected, the key dimensions those.` | with a selection → dimension it; else take/put down the Dimension tool |
| 2 | `Coincident` | `Coincident` | with a selection (and not armed) → apply; armed → disarm (Select); else arm (then click pieces) |
| 3 | `Horizontal` | `Horizontal` | same |
| 4 | `Vertical` | `Vertical` | same |
| 5 | `Parallel` | `Parallel` | same |
| 6 | `Perpendicular` | `Perpendicular` | same |
| 7 | `Tangent` | `Tangent` | same |
| 8 | `Concentric` | `Concentric` | same |
| 9 | `Equal` | `Equal` | same |
| 10 | `Symmetric` | `Symmetric` | same (3 picks: piece, mirror piece, mirror line) |

17. **Edit row** (`iconrow`, 22 px; all but Align need a selection):

| Key | Tooltip | Action |
|---|---|---|
| Delete (del icon) | `Delete the selected pieces` | delete |
| Sharp corner | `Sharp corner — Remove the selected fillet arc or chamfer line and carry its two lines to the corner` | sharpen |
| Fix | `Fix — Lock the selected pieces where they are` | toggle Fix |
| Construction | `Construction — Construction geometry takes no part in the regions — an axis to revolve about, a mirror line` | toggle construction |
| Pattern (toggle) | `Pattern — Repeat the selected pieces in a row, a grid or round a centre. Each copy is a piece of its own and keeps the relations and dimensions the pieces have among themselves` | open/close the Pattern box |
| Align | `Align — Turn the sheet so the selected line lies along its X axis — a crooked edge of the scan becomes the horizontal the rest is drawn against. Nothing moves on the part.` | enabled only with exactly **one line** selected |

18. **Pattern box** (`subbox`, while open):
    * mode radio: `Grid` (sketchPattern icon) / `Round` (sketchPatternRound icon).
      Grid starts `{count 3, X spacing 10 mm (0.5 in)}`; Round starts
      `{count 6, over 360°, centre 0,0}`.
    * Grid: `Along X` (× count, 1–500) · `X spacing` (length) · `Along Y` (× count;
      default 1) · `Y spacing` (only when Along Y > 1). Hint
      `Counts take the selection itself in; a negative spacing runs the other way.`
    * Round: `Count` (×) · `Over` (°) · `Centre X` · `Centre Y`. Hint
      `360° spreads the copies evenly round; less spans that angle, first to last.`
    * keys: primary `Pattern {n} piece(s)` (disabled with nothing selected) · `Cancel`.
    * Copies are drawn on the sheet where they would land until applied (capped at 50×50 /
      360 for the preview).
19. **Inferred chips** (`chiprow`, only when some): one `relchip` per inferred relation
    (dashed pill) reading like `H line 3`, `Coincident line 2 · arc 4`; tooltip
    `Inferred while drawing — the piece was drawn nearly so. Click to select it, × to take it off.`;
    inner `×` (`Take the inferred relation off`). Selected → solid accent; unmet → amber.
20. **Constraint list** (`relations`, max height 220 px, scrolls; only when any):
    * **Dimension rows**: label (click selects; e.g. `L line 3`, `R arc 4`, `Ø circle 5`,
      `Angle line 1 · line 2`, `d line 1 · XY plane`) · for radius/diameter a chip `→ Ø` /
      `→ R` (`Give it by its diameter instead` / `Give it by its radius instead`) · nominal
      chip `→ {round}` (not when an expression drives it) · expression field (mono; angle
      unit `°`, else the length unit; Enter takes it and ends the Dimension pick) · bold
      `⚠` when unmet · `×` (`Remove the dimension`). A dimension just made takes focus here
      with its number selected (except one placed by the Dimension tool, which opens on the
      sheet).
    * **Relation rows**: label (`Coincident line 1 · line 2`, `H line 3`, `V line 4`,
      `Parallel …`, `Symmetric … about XY plane`, `On point 2 · line 3`, `Collinear …`) ·
      `⚠` if unmet · `×` (`Remove the relation`).
    * **Fix rows**: `Fix {kind} {id}` · `×` (`Take the Fix off — the piece is free to move again`).
    * Selected row: `sel` styling; clicking a label selects the constraint (also lights it
      on the sheet).
21. **Status DRO**:
    * label row: `Sketch` · `{n} piece(s) · {k} region(s)` + ` · {m} loose` (tooltip
      `Pieces a relation or dimension does not pin down yet — drawn in the loose colour on the sheet; the rest in the constrained colour`).
    * window: empty sketch → dim 12 px `Needs input`; failed → alarm `⚠ {n} relation(s) not met`
      (tooltip lists each unmet one `… — off by {residual}`); solved →
      `Solved · {k} closed region(s)` or `Solved · no closed region yet` (tooltip: the
      soft RMS `The fitted pieces sit on the slice to {rms} RMS` or
      `Every relation and dimension is met`; `{n} relation(s) say nothing new (redundant)`;
      `{n} piece(s) not fully constrained — drawn in the loose colour` or
      `Every piece is fully constrained`; `{n} solver step(s)`).
22. **Feature keys** (only when ≥1 closed region): primary `Extrude` (tooltip
    `Finish the sketch and extrude every region of it — the body shows as a ghost with arrows to drag; a click on a region lets it go`)
    and primary `Revolve` (`Finish the sketch and revolve every region of it about an axis picked next`).
23. **`Finish sketch`** — primary full-width (bottom copy, same as #2).

### 4.3 Fit options window

Floating window (no backdrop; left 352, top 96, width 400; dragged by its head). `Esc`
closes it and goes no further.

| # | Control | Text | Details |
|---|---|---|---|
| 1 | Head | `Fit options` · `×` (aria `Close`) | drag the head to move the window |
| 2 | `Detail` slider | 0 – 1, step 0.05, default **0.5**; readout `Coarse`, `Coarser`, `Medium`, `Finer`, `Fine` | applied on release; note: "**Coarse** gives few, long pieces: a wall that bows or wanders is one line, a blob in a corner is cut out for the corner its sides make, and a round no bigger than the wander becomes a sharp corner. **Fine** gives every small feature a piece of its own." |
| 3 | `Fit tolerance` | number, placeholder `auto`, unit word | belongs to the sketch; blank = auto; refits on blur/Enter; note "How far an arc may miss its slice points. Left blank the chain's own scatter decides; a noisier scan wants it raised. Detail works on top of it." |
| 4 | checkbox | `Join corners — sides carried on to where they cross, fillets made tangent` | default **on** |
| 5 | checkbox | `Align lines to the sheet's axes when they run close to one` | default off |
| 6 | indented `Within` | number 0.5 – 10, step 0.5, unit `°`, default **3**; disabled unless #5 | — |
| 7 | indented checkbox | `Lines near 45° too` | disabled unless #5; default off |
| 8 | note | `Aligned lines get a horizontal or vertical relation, shown as an inferred chip that can be taken off. The axes are the sheet's: on a part that lies askew, select a side and use Align first.` | small dim |
| 9 | Foot | status `{n} piece(s) on {k} chain(s) — changes refit them` or `Takes effect on Fit all chains` · `Reset` (defaults) | every change refits fitted chains as one undo step; options remembered |

### 4.4 Extrude box

Opens with: last sketch (or the one chosen), its regions auto-picked when it has exactly
one, extent **Distance from 0 to 10 mm**, taper 0, Result **Join**. On open (and after a
profile pick) the `To` field takes focus with its number selected (unless another field is
focused) — so typing a number + Enter makes the extrude.

| # | Control | Text / options | Shown / enabled | Tooltip / ⓘ |
|---|---|---|---|---|
| 1 | Title | `New extrude` / `Edit {name}` | | |
| 2 | `Name` | text | editing | |
| 3a | `Profile` (face pull) | read-only: `{face description} of {Body}` | when pulling a body face | ⓘ `A face pulled`: "The face itself is the profile: it is found again on the body whenever this extrude is built, and pulled as it stands then — an earlier feature that reshapes the face reshapes what is pulled from it. Nothing is copied into a sketch." |
| 3b | `Sketch` | select: `Select…`, each sketch by name | otherwise | |
| 4 | `Profiles` | bold `{picked} of {found} picked` + key `All` / `None` | otherwise; key disabled with no region | key tooltip `Use every closed region of the sketch` / `Let every region go`; ⓘ `Profiles`: "The sketch's closed regions are drawn on the part while this box is open: click one to use it, click it again to let it go. An inner loop left unpicked becomes a hole of the region around it. A sketch with one region has it picked already." |
| 4h | hint | `The sketch has no closed region — edit it and close the profile.` / `Click the sketch's regions in the viewport.` | when nothing picked | |
| 5 | Preview line | §4.1 | | |
| 6 | Analysis row | §4.1 | | |
| 7 | `Extent` | `Distance`, `To a plane`, `To a face of a body`, `To a body`, `To next`, `To last / through all`, `Between two planes`, `Symmetric` | | switching to Distance starts `0 → 10 mm` (½ in); Symmetric starts Total 10 mm |
| 8 | `Reverse direction` (flip icon 16) | | enabled for Distance, To next, To last, To a body | `Extrude the other way along the sketch plane’s normal — From and To change sign, and the ghost turns over` / `Look for the body the other way along the sketch plane’s normal` / disabled: `A symmetric extrude has no direction to reverse` / `An extrude limited by a plane reaches the way the plane lies` |
| 9 | Distance: `To` | length + nominal chip; **Enter creates** (when untouched) | Distance | ⓘ "How far along the sketch plane's normal the body ends — in front of the plane. The arrow on that face drags it; a click on the part in front of the plane sets it." |
| 10 | Distance: `From` | length + nominal | Distance | ⓘ "How far along the normal the body starts — 0 is the plane itself, negative is behind it. The arrow on that face drags it; a click on the part behind the plane sets it." |
| 11 | Distance hint | `Or click the part: in front of the plane sets To, behind it From.` + ⓘ `Up to a point on the part` | Distance | card explains clicking scan/body/corner; part wholly behind: set To 0 by hand; height read once; a sketch region under the cursor still takes the click |
| 12 | Symmetric: `Total` | length + nominal | Symmetric | ⓘ "The whole thickness, half on each side of the sketch plane; either arrow drags it"; hint `Or click a point on the part: each half reaches it.` |
| 13 | To a plane: `From` + `Up to` (plane picker, `Choose the plane…`) | | To a plane | From ⓘ "How far along the normal the body starts" |
| 14 | To a face: `From` + `Up to` read-only (`{face}` or `Click a face of a body`) | | To a face | body faces become clickable (any body) |
| 15 | To a body: `From` + `Up to` select (`Choose the body…`, bodies) + hint `Runs until the profile meets the body’s surface — a solid’s skin or a sheet — the way the body lies; the body has to lie under the whole profile.` | | To a body | |
| 16 | To next / To last: `From` + hint | next: cut `Cuts through the first wall of the body the profile meets, and stops where it comes out.` / else `Runs until the profile meets the body — the whole profile has to land on it. Reverse direction looks the other way.`; last: cut `Cuts through all of the body that way.` / else `Runs to the far end of the body that way.`; + ` Measured against every body there is.` when Result is New body | | |
| 17 | Between planes: `From plane`, `To plane` (two plane pickers) | | Between two planes | |
| 18 | `Taper` | angle `°` + nominal; ±85° max; positive = sides lean **out** | always | ⓘ "Degrees the sides lean as the extrude goes away from the sketch, as Fusion's taper angle: positive leans them out, negative leans them in — a moulded boss's draft. 0 for straight sides. The sketch's outline stays the section at the sketch plane, or at the end of the extent nearest it, and an extrude reaching both ways leans both ways from there. The arrow on the far edge drags it. A taper that would close the profile before the end is refused, saying where it closes." |
| 19 | `Direction` | `The sketch plane's normal` (default), each measured line/cylinder/cone axis by name, `X axis`, `Y axis`, `Z axis`, `Typed…` (axes lying in the sketch plane disabled) | always | ⓘ `The way the profile is swept` (oblique prism; ends stay parallel; axis read once, turned to the plane's front) |
| 19a | typed direction | label `x, y, z` or `{a}° off the normal`; three number fields (4 places) | `Typed…` | |
| 19b | lean hint | `{a}° off the sketch plane's normal.` | a non-normal direction chosen | |
| 20 | `Result` | Join / Cut / New body (§4.1) | | |
| 21 | Create | `Create extrude` / `Save changes` | ready | `Enter or a middle click does the same` |
| 22 | `Cancel` | | | |

### 4.5 Revolve box

Opens with: last sketch (or chosen), regions auto-picked, axis none (or the one the chip
gave), angle **360°**, pitch 0 (not a coil), right-handed, Result **Join**.

| # | Control | Text / options | Shown | Tooltip / ⓘ |
|---|---|---|---|---|
| 1 | Title | `New revolve` / `New coil` (pitch ≠ 0) / `Edit {name}` | | |
| 2 | `Name` | | editing | |
| 3 | `Sketch` | select `Select…` + sketches (changing it clears the axis) | | |
| 4 | `Profiles` | as Extrude | | |
| 5 | `Axis` + ⓘ | plane-style picker, placeholder `Click in the viewport, or choose…`; groups `Sketch lines` (`{Sketch} line {id}[ (construction)]`), `Elements` (lines, cylinders, cones, circles), `Coordinate axes` (`X axis`, `Y axis`, `Z axis`) | | ⓘ `Axis`: "Click a line of the sketch on the part, or a line, cylinder, cone or circle element, or choose here. A construction line drawn down the middle of the profile is the usual axis." Hover/choose draws the axis across the part (amber line + name). |
| 6 | Preview line | idle until an axis exists | | |
| 7 | Analysis row | | | |
| 8 | checkbox `Coil — climb along the axis` + ⓘ `Coil` | | | card: turns past 360°, typed as turns or height; profile must reach less far along the axis than the pitch; thread recipe (groove section on a plane through the axis, Result Cut) |
| 9 | `Angle` | `°` + nominal | | ⓘ plain: "360° for a whole body of revolution — a full turn at most; tick Coil to turn on past it. The ring about the axis drags it."; coil: "Degrees turned — any number of turns: 1080° is three. The ring about the axis drags it, round and round." |
| 10 | `Pitch` | length (ignores 0) | coil | ⓘ "How far the profile climbs along the axis each turn — a thread's pitch, a spring's from one coil to the next. Negative climbs the other way along the axis, which Reverse direction does too." |
| 11 | `Turns` | unit `turns` | coil | ⓘ "How many times the profile goes round — the angle over 360°. A fraction is fine." |
| 12 | `Height` | length | coil | ⓘ "How far the profile climbs along the axis from start to end — the turns times the pitch. Typed, it sets the turns." |
| 13 | checkbox `Left-handed` | | coil | |
| 14 | `Reverse direction` (flip icon) | | coil | `Climb the other way along the axis — the hand stays as it is` |
| 15 | `Result` | | | |
| 16 | Create | `Create revolve` / `Create coil` / `Save changes` | | `Enter or a middle click does the same` |
| 17 | `Cancel` | | | |

A new coil starts with a pitch 1.5 × the profile's reach along the axis and three turns.

### 4.6 Feature boxes (`FeatureEditor`, one per kind)

Every feature box = title · (`Name` when editing) · the kind's fields (below) · Preview
line · Analysis row · Create · `Cancel` (§4.1). A Move face that edits history instead,
or a Move face + Draft, hides the Preview line and the Analysis row.

The **body select** used by many kinds: field `Body` + select `Select…` + every body by
name. Default body = the last solid body (else the last body). Changing it clears the
kind's picks.

What the 3D view arms while each box is open (§8.4): faces / edges of the box's body, the
other bodies dimmed to 15 %.

#### Shell — `New shell`
Defaults: wall **2 mm** (`thicknessFrom: default`) or the Thickness map's mean (`map`).
1. `Body`.
2. `Open` — read-only: `None — a closed hollow body. Click faces of the body to take them away`
   or `{n} face(s) picked` + `Clear`.
3. `Wall` (length + nominal) — ⓘ `The wall left after the inside is taken away, inward from every face that stays. The arrow on the opened face drags it.`
4. Hint: `The wall is the Thickness map’s mean over the part.` / `No Thickness map measured — 2 mm to start; measure the part in Wall thickness for its own.` / `Wall as typed.`
Grip: arrow on the first opened face, pointing into the body (`thickness`).

#### Fillet — `New fillet`
Defaults: radius **1 mm** until read off the scan, Tangent propagation **on**, mode Constant.
1. `Body`.
2. `Radius` select: `Constant` / `Variable — start to end` + ⓘ `Constant or variable radius`.
3. `Edges` — read-only: constant `Click the edges of the body — a round runs on along the edges tangent to it. Click a picked edge again to let it go`,
   variable `Click the edges of the body to round, each from a start radius to an end radius. Click a picked edge again to let it go`,
   else `{n} edge(s) picked` + `Clear`.
4. Variable only — one block per edge (`varyedge`): head `{i} · {edge description}` +
   swap key `⇄` (`Swap the two radii — the start radius at the other end`); `Start` (radius
   + nominal; ⓘ `The radius at the edge's start — the end named Start on the part`); `End`
   (ⓘ `The radius at the edge's other end; the round runs evenly from the one to the other`).
   Hint: none yet `Each edge clicked starts at the radius the last one has; type its two radii over.`;
   else `The part names each edge's Start and End, with the radius there[ — numbered as listed here]. Each edge is rounded on its own, after the others.`
   + ` And {an edge | n edges} at R {r}, kept — Constant shows it/them.` when constant edges exist too.
5. Constant only — `Radius` (radius + nominal; ⓘ `Read off the scan across the first picked edge; type your own to keep it. The arrow on the edge drags it.`)
   and the estimate hint: `Radius as typed.` / `Pick an edge and the radius is read off the scan there.` /
   `Reading the round off the scan…` / `Radius {r} read off the scan across the first edge ({n} section points).` /
   warn `No round found on the scan at the first edge — {r} to start.`; plus, when varying edges exist,
   `And {an edge | n edges} at a varying radius — {r0} → {r1}, … — kept; Variable shows it/them.`
6. Constant only — checkbox `Tangent propagation` + ⓘ (`The fillet runs on along every edge that leaves a picked one smoothly — all the way round a rounded rectangle from one click on its rim. The edges it reaches are lit in a lighter green on the part. Off, only the edges clicked are taken.`).
Grip: arrow at the first edge's middle, pointing into the body along the faces' bisector
(`radius`). Variable: Start/End chips on the part (§6.11).

#### Chamfer — `New chamfer`
Defaults: distance **1 mm**, distance 2 **1 mm**, angle **45°**, Equal distance, flip off, Tangent propagation on.
1. `Body`.
2. `Type` select: `Equal distance` / `Two distances` / `Distance and angle` + ⓘ `Chamfer type` ("**Equal distance** cuts both faces back alike. **Two distances** cuts the reference face back by the first and the face across by the second; **Distance and angle** cuts the reference face back by the distance, the chamfer at the angle to it. The reference face of each edge is the face most of the edges share — a rim's top — and Flip takes the other one.").
3. `Edges` (as fillet constant).
4. `Distance` (equal) / `Distance 1` (other) — length + nominal; ⓘ `How far back from the edge each face is cut. The arrow on the edge drags it.` / `How far back from the edge the reference face is cut. The arrow on the edge drags it.`
5. `Distance 2` (Two distances) — ⓘ `How far back from the edge the face across from the reference is cut.`
6. `Angle` `°` (Distance and angle; clamped 0.1 – 89.9) — ⓘ `Degrees between the chamfer and the reference face, under 90 — 45° on a square edge is the equal chamfer.`
7. checkbox `Flip — the other face is the reference` (not for Equal) — tooltip `Measure from the other face of each edge: the distances swap sides, the angle is read off the other face`.
8. checkbox `Tangent propagation` + ⓘ (as fillet, "The chamfer runs on…").
Grip: as fillet (`radius`).

#### Delete face — `New delete face`
Defaults: Tangent faces **on**.
1. `Body`.
2. `Faces` — `Click a face of the round, the hole or the boss to take off` / `{n} face(s) picked` + `Clear`.
3. checkbox `Tangent faces` + ⓘ `What a click takes` ("A click takes the face, the faces it is split from on its own surface — both halves of a bore — and every face the picked ones close in on all sides: a blind hole's floor, a boss's top. With Tangent faces on, a click on a round also takes the rounds that run on from it smoothly — a fillet's whole chain, up to the flats either side — and a bore's or a boss's wall the round at its mouth or foot. Click a picked face again to let go of what that click took.").
4. Hint `The faces are taken off and the faces round them run on to close the gap, each on its own surface. Pick every face of what is to go — a chamfer's faces one by one, or Shift+drag a window over them.`
Create label: `Delete the face|faces[ of {Body}]`.

#### Draft — `New draft`
Defaults: angle **2°**.
1. `Body` (changing it clears neutral and faces).
2. `Neutral` — plane picker; placeholder `Click a flat face of the body, or choose…` (or the
   picked face's description / `a fitted plane`) + ⓘ `The neutral plane` ("Where the leaned faces keep their place: they turn about the line in which they meet this plane. Click a flat face of the body — the face the part stands on in its mould, usually — or choose a plane. A positive angle leans the faces in as they go away from it.").
3. `Pick another neutral face` (only when a neutral is set) — `Pick the neutral plane again — the next flat face clicked is it`.
4. `Faces` — `First the neutral face` / `Click the faces of the body to lean` / `{n} face(s) picked` + `Clear`.
5. `Angle` `°` + nominal — ⓘ `Degrees the faces lean in as they go away from the neutral plane; negative leans them out`.
6. Hint (when command-set groups exist) `Then {n} faces at {a}°, …` — tooltip `More faces at angles of their own, set through the commands (groups); kept as they are when this box is applied`.
The first face click sets the neutral; later clicks pick faces.

#### Move face — `New move face`
Defaults: offset 0, Snap off.
1. `Face` / `Faces` — `Click the faces of the body to move — flat or round ones` /
   `{face description}` / `{n} faces picked` + `Clear`.
2. Snap key (only with exactly one snappable face): `Snap to scan` / pressed
   `Pick faces instead`; tooltip `Read the move off the scan: click the scan where this face should lie, and a plane or a cylinder fitted there says how far`
   / `Click faces of the body again — to add to the move, or to let one go`.
3. Number: label `Move by` (one flat face) / `Radius by` (all round) / `Offset` (several) —
   length + nominal; ⓘ several: `How far every picked face moves: a flat face along its outward normal — out adds to the body, in cuts into it — a round face in radius, a boss and a bore alike. A cylinder’s wall and its two ends together make it longer and wider at once.`;
   one round: `The change of the round face’s radius, the axis kept: a hole bored out or sleeved, a boss built up or turned down. Typed, dragged by the arrow, or read off a cylinder fitted where the scan was clicked.`;
   one flat: `How far along the face’s outward normal: out adds to the body, in cuts into it, the walls round the face running on as they stand. Typed, dragged by the arrow, or read off a plane fitted where the scan was clicked.`
4. Diameter hint (one round face): `Ø {before} → {after}`.
5. Reading hint (warn on error): `⚠ {error}` / spinner `Fitting the scan where you clicked…` /
   `Click the faces to move — one, or several to offset them together: a cylinder’s wall and both its ends, every face of a block. Shift+drag a window takes many at once.` /
   `Now click the scan where this face should lie — or type how far it moves.` /
   `Click a face to add it or to let it go.` (several) /
   `Type how far the face moves, or drag the arrow — or take Snap to scan and click the scan where it should lie. Click more faces to move them together.` /
   after a snap: `Read off the scan: a {plane|cylinder} fitted to σ {σ}, {d} {outside|inside} the face.`
   (cylinder: `, its radius {d} {larger|smaller}`) + lean/tilt/axis remarks.
6. checkbox `Draft it as well, by {a}°` (new feature, snap read a lean) — tooltip begins
   `A Draft feature after the move: the moved face turns {a}° about the line across its middle where the scan's plane crosses it, and lies on the scan. …`.
7. checkbox `Change what made it|them instead` (when the face traces back) — tooltip
   `Change what made the face — the extrude's distance, the sketch's line or circle — as you would by hand, instead of adding a Move face: the rounds, shells and cuts after it build again on the moved wall`.
8. Plan hint: `Create edits {what} — no Move face is added. Everything built after it — rounds, shells, cuts — builds again on the moved face(s); Ctrl+Z takes it back.`
   or `A Move face on the body: the walls round the face must stand square to it, and rounds on its rim do not follow — {why it cannot be traced | tick the box to change … instead, and they do}.`
   (+ ` Then a Draft leans the moved face {a}° onto the scan's plane, about the line across its middle — one Ctrl+Z takes both back.`)
Grip: arrow `move` on the face (flat: out along the normal; round: radially at the new radius).
While snapping, the bodies turn faint (tint + edges) and clicks go to the scan.

#### Hole — `New hole`
Defaults: Ø **6 mm** (radius 3), Through all, no insert.
1. `Body`.
2. `Along` — picker (group `Elements`: cylinders, circles, lines); placeholder
   `Click a round face — or a flat one to drill into at its centre — or choose…` (or the
   face, or `{face}, at its centre`). Hover draws the axis.
3. `Insert` select — tooltip `Size the hole for a heat-set insert: its hole diameter, and a millimetre deeper than the insert is long`;
   `None — a plain hole`, then `M2 — Ø 3.2 × 3`, `M2.5 — Ø 4 × 4`, `M3 — Ø 4 × 5.7`, `M3 short — Ø 4 × 3`,
   `M3 Voron — Ø 4.4 × 4`, `M4 — Ø 5.6 × 8.1`, `M4 short — Ø 5.6 × 4`, `M5 — Ø 6.4 × 9.5`,
   `M5 short — Ø 6.4 × 5.8`, `M6 — Ø 8 × 12.7`, `M8 — Ø 9.7 × 12.7`, `1/4"-20 — Ø 8 × 12.7` (+ unit word in inches).
4. `Diameter` (length; nominal unless an insert) — ⓘ `The hole's diameter — the element's, to start`; typing clears the insert.
5. checkbox `Through all` (unticking sets depth 10 mm).
6. `Depth` (when not through) — ⓘ `How far into the body from the face` (point) / `How far each way along the axis from its point`.
7. Insert hint: `Leave at least {wall} of wall round it — a boss of Ø {d} or more.`

#### Thread — `New thread`
Defaults: ISO metric, Ø 10, pitch 1.5, right hand, Full length, start 0, offset 0, chamfer off; sized from the face when opened from a face chip.
1. `Face` (bold read-only): `{face}{ · a bore: a nut’s thread | · a boss: a bolt’s thread}` or `Click a round face of the body`.
2. `Type` select: `ISO metric`, `Unified coarse (UNC)`, `Unified fine (UNF)`, `3D printing — 45° flanks`.
3. Print hint (print45): `45° flanks, a flat crest and root: printed with the axis upright, nothing overhangs past 45°. It mates only with itself.`
4. `Size` select: `M3` … (metric/print45) or `{label} — Ø{d}` (UNC/UNF); `Custom Ø{d}` when off-table.
5. `Designation` select (when a size is listed): each pitch's designation + ` — coarse` / ` — fine` (metric), ` — ISO coarse depth` / ` — ISO coarse pitch` (print45); `Custom pitch {p}` when off-table.
6. `Diameter` — ⓘ `The major diameter — a bolt's outside, a nut's at the bottom of its thread. Typed, the thread is cut to it, whatever the table has.`
7. `Pitch` — ⓘ `How far the thread climbs in a turn — {p} mm is {tpi} threads per inch.`
8. `Direction` select: `Right hand` / `Left hand`.
9. checkbox `Full length`; when off: `Length` (+ nominal; ⓘ `How far along the face the thread runs`), `Start` (ⓘ `How far in from the end of the face the thread starts — the end further along the face's axis, or the other one with the box below ticked`), checkbox `From the other end`.
10. checkbox `Chamfer the free ends (45°)` — tooltip `A 45° chamfer across the thread where it comes out of the body free — a bolt's tip, a bore's mouth at either side of a nut — from just past the root out through the crests, so the thread starts clean instead of on a knife-edge first turn. Not where the thread ends against a shoulder or a bore's floor, or short of the face's end.`
11. `Offset` — ⓘ `A clearance for 3D printing: every face of the thread — the flanks, the crest and the root — moved into the material by this much, so a printed bolt turns into a printed nut. Small values only: under {max} at this pitch, where the tooth would have nothing left at the pitch diameter. A tenth or two of a millimetre is a usual start — half each on a printed bolt and nut.`
12. Reading hint: `{M10×1.5}[ left-hand] · major Ø{d} · minor Ø{d} · {depth} mm deep[ · the Ø{f} bore filled in to the minor …][ · {o} mm off every face][ · countersunk|chamfered to Ø{c} at the free ends]`.
13. Problem hint (warn): `⚠ {problem}`.
Picks faces on **any** body (a click on another body moves the thread there).

#### Extract — `New extract`
Defaults: Result **New body**, slab 2 mm.
1. `From` select: `Click an element on the part, or select…`, optgroups `Elements` (planes,
   cylinders, cones, spheres, tori) and `Shadow sections`.
2. Shadow hint: `The outline stood up through the part, {t} thick — it follows the shadow as 3D Measure takes it again.` / `The shadow is still being taken.`
3. `Thickness` (plane only) — ⓘ `A plane becomes a slab this thick, centred on it`.
4. `Result`.

#### Cut by plane — `New cut by plane`
1. `Body`. 2. `At` — plane picker `Choose the plane…`. 3. block button (flip icon) `Keep the other side`.

#### Mirror — `New mirror`
Defaults: Join with the source **on**; the symmetry search starts at once.
1. `Body`.
2. `About` — plane picker; placeholder `Click a flat face of the body, or choose…` (or the face).
3. Symmetry hint: spinner `Looking for the scan's own symmetry plane…` /
   `{Set to the scan’s own symmetry plane | The scan has a symmetry plane}: its mirror image lies on the scan to σ {σ} over {m} of {n} samples.`
   + chip `Use it` (when another plane is chosen) + chip `Keep as a plane element`
   (tooltip `Make it a plane element in 3D Measure — Symmetry plane — for sketches, sections and dimensions to stand on; the mirror then refers to the element`) /
   `Set to {name}, the symmetry plane measured on this scan.` /
   `The scan has no symmetry plane of its own[ (the best mirror image stands off it by σ {σ})] — click a flat face of the body or choose a plane.`
4. checkbox `Join with the source`.

#### Combine bodies — `New combine bodies`
Defaults: Join; tool = first other solid; keep off.
1. `Body`. 2. `Operation`: `Join` / `Cut by` / `Intersect with`. 3. `Tool` (body select).
4. checkbox `Keep the tool body` — tooltip `Keep the tool as a body of its own after the combine, as it was — the body is joined, cut or intersected with a copy of it`.
5. Hint `The tool body stays as it is, beside the result.` / `The tool body goes into the result.`
6. Sheet hint (body is a sheet): join `A sheet joins a sheet: the two are sewn into one where they meet edge to edge — split each by the other first.`;
   else `The sheet is trimmed to the part of it {inside|outside} the solid tool, and stays a sheet for Thicken; the tool stays a body, to trim another sheet with.`

#### Sheet — `New sheet`
Defaults: last sketch, From **−10**, To **10** mm.
1. `Sketch` select. 2. `From` (ⓘ `How far along the sketch plane's normal the surface starts`). 3. `To` (ⓘ `How far along the normal it ends — past the body on both sides, so the split goes right through`).
4. Hint `Every drawn piece of the sketch, open or closed, becomes the surface. It is a body in the list, marked as a surface, and never exported on its own.`

#### Split — `New split`
Defaults: keep on the scan, tolerance **0.3 mm**, overlap **50 %**.
1. `Body`. 2. `Cut with` — a check row per other body (or `Make a sheet first`).
3. `Keep` select: `The {fragment|piece}s on the scan`, `The largest {…}[, by area]`, `The smallest {…}[, by area]`, `What lies between the sheets` (solids only).
4. Sheet hint: `The sheet is trimmed: the pieces kept are sewn into one sheet, for Thicken. A fitted skin laid past the part's edge loses its overshoot on the scan.`
5. When keeping on the scan: `Tolerance` (ⓘ `How far a sampled point of a fragment's face may lie from the scan and still count as on it — a few times the scan's scatter`), `Overlap` `%` (5–100; ⓘ `The share of a fragment's samples that must lie on the scan for it to be kept: a pocket's fragment has the pocket's roof where the scan has its floor, and falls out`).

#### Loft — `New loft`
Defaults: Straight **on**, Join.
1. `Sketches` — check row per sketch (order ticked = loft order). 2. checkbox `Straight between the profiles`. 3. `Result`. 4. Hint `In the order ticked, each sketch's one region — or its largest.`

#### Sweep — `New sweep`
Defaults: of the last two sketches, the one with a region = profile; Join.
1. `Profile` select. 2. `Path` select. 3. `Swap profile and path` (`The profile's sketch for the path's, and the path's for the profile's`; disabled when both empty).
4. Hint `The profile sketch's largest region is carried along the path sketch's one chain of lines and arcs, from the end nearest the profile.` + ⓘ `A sweep`.
   (Command-made route instead of a path: `Along a route in 3D: {n} points, {b} bend(s).` + ⓘ `A route`, and `Bend radius` field.)
5. `Result`.

#### Thicken — `New thicken`
Defaults: last sheet, **2 mm**, grows `in`, Result **New body**.
1. `Sheet` select (`Make a sheet or fit a surface first` / `Select…`). 2. `Thickness` (ⓘ `The wall, through the solid — every point of the sheet that far from the far skin`).
3. `Grows`: `Against the sheet's normal` / `Along the sheet's normal` / `Half each way`. 4. `Result`.
5. Hint `A freeform thin shell's wall from its one skin — the sheet stays a body of its own, put out of view. A face that bends tighter than the thickness towards the side it grows to is named, and the thickness it takes.`

#### Pattern — `New pattern`
Defaults: Circular, count **4**, over **360°**; linear spacing 10 mm; rows 2 × 10 mm.
1. `Body`. 2. `Kind`: `Circular, about an axis` / `Linear, along a direction`.
3. `About` / `Along` — picker (`Elements`, `Coordinate axes`), placeholder `Choose…`, hover shows the axis.
4. `Count` (ⓘ `Copies in all, the body itself counted`, ≥2).
5. `Over` `°` (ⓘ `The angle from the first copy to the last: 360° spreads them evenly round`) / `Spacing` (ⓘ `The distance from one copy to the next`).
6. Linear: `Then along` picker (placeholder `Nothing — one row`); when set: chip `× one row` (`One row only — no second direction`), `Rows` (ⓘ `Rows along the second direction, the first counted — a grid`), `Row spacing` (ⓘ `The distance from one row to the next`).

#### Import STEP — `Import STEP`
1. `File` read-only: `{name} · {size}` or `No file chosen`.
2. key `Choose a STEP file…` / `Another file…` — tooltip `Read another STEP file in this one's place — the features after it keep their picks where they still find them`.
3. Hint `A body for every solid in the file, in the frame the file was drawn in, named as the file names its parts. Put them in place with **Move body**.`
Create label `Import`.

#### Move body — `Move body`
Defaults: translate 0,0,0; rotate 0,0,0; Copy off; pivot = middle of the picked bodies.
1. `Body` / `Bodies` read-only: names joined, or `Click the bodies to move on the part, or their rows in the body list` + `Clear`.
2. `Add` select (when other bodies exist): `Or choose a body…` / `Another body…` + bodies.
3. checkbox `Copy — move a copy, the body stays where it is` — tooltip `Leave the bodies where they are and move copies of them — bodies of their own`.
4. `Move, mm` — three fields X, Y, Z in one row, each **underlined 2 px in its axis colour**
   (X red, Y green, Z blue); ⓘ `How far along the part's X, Y and Z, after the turn — the arrows on the part drag them`.
5. `Turn, °` — three fields likewise; ⓘ `Degrees about X, then about Y, then about Z, each through the middle of the bodies as they were picked — the rings on the part drag them; each ring stands about the axis its angle turns about`.
6. `Reset` (disabled when all zero) — `Put the bodies back where they stood: every distance and angle zero`.
7. Hint `Turned about {x, y, z} — the middle of the body|bodies as picked.`
Create label `Move the body|bodies` / `Copy the body|bodies`. Grips: the Move gizmo (§6.6).

#### Offset body — `New offset body`
Defaults: **+1 mm**, keep **on**.
1. `Body`. 2. `Distance` (ⓘ `How far all round — out, or in with a negative one. The convex edges come out rounded, as an offset rounds them`).
3. checkbox `Keep the body — the offset a new one` (tooltip `Leave the body as it is and make the offset a body of its own — a keep-out round a part, or what is cut from a case to hold it with clearance (Combine, Cut by, the offset as the tool)`).
4. Hint `Cut it from a case with Combine to leave a pocket with the clearance.` / `The body becomes the offset.`

#### Fit surface — `New fit surface`
Defaults: Freeform, flood **1°**, net auto, expand **35 %**, smoothness **0.01**, Hold to Auto.
1. `Shape` select `Freeform` / `Plane` / `Cylinder` + ⓘ `What is fitted` (auto-hold within 0.5° / 1 mm).
2. `Hold to` (plane/cylinder only): `Auto — what it lands near`, `Nothing — the free fit`, then `{Parallel to | Perpendicular to | Coaxial with | On …} {reference}` for every coordinate plane, measured element and earlier fitted plane/cylinder.
3. `Region` + ⓘ `The region`: bold `None — click the scan` / `{n} points` + `Clear` (`Start the region over`).
4. `Flood angle` `°` (0.1–89; ⓘ `How far the normals may turn over one edge for a click's flood to cross it`).
5. checkbox `Mark with the tools`; when ticked the app's **marking tools** appear:
   `Navigate` / `Window` / `Brush` / `Lasso` gesture keys, `Brush Ø ({unit})`, checkbox
   `Mark faces pointing away too`, `Erase` (pressed: `Erasing`), `Invert`, `Clear marking`,
   `Marked` count hidden; note `Esc puts the tools away`.
6. Freeform: checkbox `Net sized from the region`; when off, `Net U` (ⓘ `Control points across the region's long side, 3 to 20`) and `Net V` (ⓘ `Control points across the short side, 3 to 20`).
7. `Expand by` `%` (0–200; ⓘ `How far past the region the surface reaches, as a share of its extent — it has to meet its neighbours for the trim`).
8. Freeform: `Smoothness` (ⓘ `How hard the surface is held fair against the points: 0 follows them exactly, 0.01 fairs a scan's noise, more flattens detail`).
Clicks on the scan flood the region (Ctrl+click removes); status
`{n} points added to|taken out of the region — {N} in all. Click more of the scan, or Create.`

#### Trim — `New trim`
Defaults: every surface body ticked; tolerance 3 × median σ of the fitted surfaces (0.05–0.5 mm), 0.3 mm with none; overlap 50 %.
1. `Surfaces` + ⓘ `The surfaces` + bold `{ticked} of {total}`.
2. Check list (max height 180, scrolls): a row per body `{name}[ · σ {rms} | · solid]`, then a row per solid element `{name} · {kind}`; empty `No surface to close yet — fit surfaces here, or planes and cylinders in 3D Measure.`
3. `Tolerance` (ⓘ `How far a face's sampled point may lie from the scan and still count as on it — three times the scan's scatter, read off the surfaces' own fits`).
4. Hint `Tolerance as typed.` / `Three times the median σ of the fitted surfaces in the set.` / `No fitted surface in the set — 0.3 mm to start.`
5. `Overlap` `%` (ⓘ `The share of a face's samples that must lie on the scan for the face to be kept`).
Open edges of a non-closed trim are drawn red on the part.

#### Patch network — `New patch network` (`NetworkEditor`)
While open: the scan is coloured by curvature (restored on close); the network layer takes
every click on the scan (§8.6); the box has its own undo/redo.
1. `Click to` + ⓘ `Placing patches` (Draw / Changing a patch / Fill / Suggest creases — full rules in §8.6) · two toggle keys `Draw` / `Fill` (default Draw).
2. Tool row: `Fill all[ ({n})]` (disabled with no unfilled region; `Fill every region the curves close with a patch`) · `Suggest creases` / `Finding creases…` (`Find the scan's sharp edges and folds, to take as crease curves`) · `Undo` (`Undo the last edit of the network (Ctrl+Z)`) · `Redo` (`Redo (Ctrl+Y)`).
3. Suggestions hint: `No creases found on the scan.` / `{n} crease(s) suggested, dashed — click one to take it.` + chips `Accept all` (`Take every crease suggested as a crease curve`) and `Dismiss` (`Put the suggestions away`).
4. Placing hint: `{n} corner(s) — click the next corner;` / `one more for a patch, or Enter for a line;` / `click the first corner, press Enter, double-click or middle-click to close the patch;` + ` Backspace takes the last back, Esc lets it go.`
5. Status: `No patches yet.` / `{n} curves · building…` / `{n} patches · {k} open cells · {r} refused · worst edge {a}° · worst gap {g} µm · worst corner {c}° · closed — a solid`.
6. Editing hint (draw, not placing, patches exist): `Drag a corner or an edge to change a patch; click a patch to pick it.`
7. Patch list (`netpatch` rows, 11 px; picked = accent border + 14 % accent fill): `Patch {i} · σ {rms} · worst {max} · {u} × {v}`; tooltip `Off the scan beyond its noise: {misfit} (the tolerance {tol}). The patch's worst points: …`. Click picks/unpicks.
8. Picked patch: `Spans along` / `Spans across` rows — checkbox `Auto` (tooltip `The strip's span count chosen by the fit: 4, and more only where a patch on it misses the scan by more than the tolerance beyond its noise, while that helps — up to 20`) + number (4–20, disabled while Auto) + `spans`; key `Delete patch {i}` (`Take the patch out, with the edges no other patch has (Delete)`).
9. Refused regions (warn): `⚠ Region at ({x}, {y}, {z}): {why}` + chip `×` (`Empty the region: take its seed away`).
10. Body sides (curves along a body edge): `Along {Body}[ · {a}°]` + select `On the edge, no face`, `Contact — {face}`, `Tangent to {face}`; tooltip `How the patch beside this curve meets the body: on the body's edge (Contact), or tangent to one of the edge's two faces as well`.
11. `Smoothness` (default **0.001**; ⓘ `How hard the patches and their edges are held fair against the scan: 0 follows it exactly, 0.001 fairs its noise, 0.1 smooths well over it, from 1 detail goes`).
12. `Tolerance` (default **0.01 mm**; ⓘ `How far a patch may stray from the scan beyond its noise before Auto gives its strips more spans: they start at 4 and get more only where a patch misses by more than this, and only while that helps. Fewer spans are smoother`).
13. checkbox `Leave out stray points` — `Fit again without the points past three sigmas of the first fit — a rib or a stray edge caught in a cell`.

---

## 5. Stage overlays

### 5.1 View bar — normal (no sketch open)

Bottom-left stage corner, one wrapping row (`viewrow`) of key groups (`keys`), gap 6; the
Section view's own row stands **above** it while the section view is open. Read from the
left:

**Group 1 — part keys.** Each part key (`PartKey`) = a toggle button (icon 16 + word;
pressed/`on` while the part is shown) + a small caret `^` (8 px chevron) that opens its
menu. The menu also opens after the mouse rests **140 ms** on the key and closes **260 ms**
after it leaves both (caret click pins it open; a press anywhere else closes it). A small
**dot** on the key means "something in my menu differs from how the workspace opens".
A disabled key is greyed by `aria-disabled` (its menu still reachable) and shows the reason
as its tooltip. Caret tooltip: `How the {scan is | bodies are | elements are} drawn`.

| Key | Click | Tooltip | Disabled (reason as tooltip) | Menu (each row a toggle key: icon + word) |
|---|---|---|---|---|
| `Scan` (scan icon) | show/hide the scan | `Show the scan, or put it away. How it is drawn is in the menu over the key` | `No scan loaded`; or, with no body yet, `Build a body first — until then there is only the scan to look at` | `Transparent` (default = on whenever a body is shown — the scan is a see-through veil over the bodies; `Draw the scan transparent — what lies inside it, the fitted elements and the pinned readings show instead of hiding behind the surface in front of them`), `Mesh` (off; `Draw the triangle edges on the scan, to see how fine it is and where it is patchy. …`), `Backfaces` (on; `Colour the far side of every triangle — holes in the scan and inverted normals stop looking like solid part`), `Colours` (only for a textured scan; on), **`Curvature`** (curvature icon; off; disabled `Load a scan first — the map reads its surface`; `Colour the scan by how sharply it bends — flats green, convex rounds towards red, hollows towards blue, sharp edges dark. Where a flat ends and a round or a blend begins is where a surface region should end. While a feature is being set up its ghost takes the map over. Puts the Deviation key's map out`) |
| `Bodies` (combine icon) | show/hide all bodies as a whole (the map stays on the scan) | `Show the bodies, or put them away as a whole and leave the scan to stand alone — with the deviation map still on it, read against the bodies as they are. Each body's own eye in the panel takes it out of the model and the map instead` | `Build a body first` | `Transparent` (off; `Draw the bodies transparent — the scan, the elements and the bodies inside them show through. With the scan solid as well, each part shows where it stands proud of the other`), `Overlaps` (opIntersect icon; **on**; `Draw where two bodies run into each other in hatched red — what they share, over the faces of both that would otherwise fight for the same pixels. Off for bodies that overlap on purpose, a modifier volume for a slicer; saved with the project`), **`Tangent edges`** choice row (tangentEdges icon + label, then three radio keys `Drawn` (default; `Tangent edges drawn — where a round runs into a flat — as the sharp ones are, the way CAD draws them`), `Dimmed` (`Tangent edges dimmed`), `Off` (`Tangent edges left out, and not picked: a round shows by its outline against the background`); row tooltip `How the edges are drawn where a round runs into a flat. A round face's seam and the cut a join leaves across one surface are never drawn`) |
| `Elements` (elCylinder icon) | show/hide the 3D Measure elements and sections drawn as datums | `Show the elements and sections measured in the 3D Measure workspace on the part — the planes, cylinders and cuts the model is built against` | `Nothing measured yet — fit elements or cut sections in the 3D Measure workspace and they show here` | none; default **shown** |

**Group 2 — side keys (the workspace's own):**

| Key | Click | Tooltip | Disabled |
|---|---|---|---|
| `Deviation` (compare icon; toggle) | colour the scan against the committed bodies (turns Curvature off) | `Colour the scan by how far it lies from the bodies, on the deviation colour map at the range set in the feature boxes — outside the bodies up the ramp, inside down it — and violet where the scan lies deep inside them: a hole, a pocket or a slot the bodies lack. While a feature is being set up its ghost takes the map over. Puts the curvature colouring out` | no body (`Build a body first — the map reads the scan against it`) |

**Group 3 — `Section`** (sectionView icon; toggle): `Cut the parts open at a plane to look inside — the scan, the reference and everything measured on them. Closed parts are closed where they are cut; an open scan stays a shell. Slide the plane with the arrow, tilt it with the rings`.
Bodies are capped where the section cuts them, each in its own cut colour.

Curvature and Deviation exclude each other. A feature box's ghost map takes over either.

### 5.2 View bar — while a sketch is open

* Group 1: the **Scan** key only (no Curvature in its menu); the Bodies and Elements keys go.
  When **No part** is chosen, all model keys (Scan, Section) go too.
* Group 2 is replaced by a radio group (`What of the part lies under the sketch`):

| Key | Icon | Tooltip | Effect |
|---|---|---|---|
| `No part` | eyeOff | `The sketch on the bare stage, the 3D view under it put away` | hides the 3D viewport under the sheet |
| `Section` (default) | slice | `The part under the sketch, facing the sketch plane and following the sheet's pan and zoom, cut open at the plane where the plane runs through it — the half toward you taken away, so the slice being drawn on is the face in view` | part cut at the sketch plane, the cut closed with a flat slate cap (dark `#47535f`, light `#b9bfc6`), solid bodies capped in their cut colours |
| `Full part` | model | `The whole part under the sketch, facing the sketch plane and following the sheet's pan and zoom, nothing cut away` | uncut part |

* Number keys 0–9 (standard views) are held by the sketch: they do nothing while it is open.

### 5.3 Map legends (right edge)

Column at top 14 / right 14 / bottom 14, right-aligned, pointer-events none:
head (unit) · `+ {high end words}` · ramp (caps at both ends, 9 ticks with figures) ·
`− {low end words}` · stats grid.

| Legend | Shown when | Head | Ends | Stats | Editable |
|---|---|---|---|---|---|
| Live map (`re-map`) | a map report exists (ghost or bodies) | unit word (`mm`/`in`) | `+ scan outside the body` / `− scan inside the body` | `within` `{k} / {n}` (wide); `against` `the ghost` / `the bodies`; `deep inside (violet)` `{n}` (wide, only when > 0) | the top and bottom figures are keys (`Click to set the range of the scale`): click → number box with the magnitude; Enter/blur commits, Esc cancels; either end sets the symmetric range ±value |
| Curvature (`re-curvature`) | no live map and curvature on | `1/mm` | `+ convex — a round` / `− concave — a hollow` | `full colour at` `R {r}` | no |

Figures: signed deviation `+0.050` / `−0.050` (3 decimals mm); curvature 3 decimals. Caps
(beyond the scale) titled `Beyond the top of the scale` / `Beyond the bottom of the scale`.
Ramp = the colormap chosen in Settings (Deviation ramp: green at zero, red outside/positive,
blue inside/negative, darker caps).

### 5.4 Chips and cards on the stage

* **Ready chip** (top centre): `Part loaded — choose a plane in the panel to sketch on`.
* **Build chip** (top centre, ≥300 ms): spinner + `Loading the CAD kernel…` / `Building the part…`.
* **Playback caption** (top-left card, no pointer events): `STEP {k} / {n}` (condensed 11 px
  600 uppercase dim) · row: feature icon 20 px + name (18 px, 600; before the first step:
  scan icon + `The scan` or `An empty stage`) · reading (13 px tabular dim): sketch
  `on {plane}`; extrude/revolve `{New body|Join|Cut} · {10 mm|360°}`; others the row
  reading; failed `⚠ did not build` (amber); before the first step the file name.
* **Grip number field** (§6.6).

---

## 6. How things are drawn in the 3D view

Draw order (scene layers, front to back for picking): network layer (14) · grips (15) ·
sketches (20, covers what lies behind) · measured elements · bodies (elements + 10) ·
coordinate planes (last; take a click only where nothing else would).

### 6.1 Scan
* Normal, or a **see-through veil** over the bodies once any body is shown (Scan menu
  `Transparent`, default follows "a body is shown"). A veil hides nothing from a click:
  faces behind it are pickable.
* While Move face is snapping to the scan: scan solid, bodies **faint** (18 % tint + edges).
* Coloured by the live map / Deviation / Curvature when on (§6.10).

### 6.2 Bodies
* `MeshStandardMaterial`, double-sided, the scheme's finish, colour = style colour or the
  auto palette colour (§2.1); polygon offset +1 so edges win the depth test.
* Opacity states: **solid** (default, `Bodies › Transparent` off) · **transparent** 0.55,
  no depth write (Transparent on) · **dimmed** 0.15 (not in play for the open box's pick) ·
  **faint** 0.18 (snap to scan). A body a preview "replaces" is hidden (still pickable).
* **Edges at rest**: near-black `#0c0f13`, **1.1 px** (1.6 px while edges/everything are
  armed), opacity 0.85, depth-tested (hidden lines hidden). Tangent edges per the menu:
  drawn / dimmed (opacity 0.3) / off (and unpickable). Seams and same-surface splits never.
* **Silhouettes** of round faces from the current view (cylinder sides, sphere rims, where a
  fillet turns away), drawn like edges, recomputed when the camera turns > 0.2°.
* **Overlaps** (Overlaps on): shared volume of two solid bodies in matt hatched red
  `#e0564f`, drawn a hair nearer (polygon offset −1), unpickable; hidden when either body is
  hidden, replaced or dimmed.
* **Caps**: where the section view (or the sketch's Section view) cuts, each solid body is
  closed in its own cut colour; overlaps capped red hatched.
* Blank stage: the stage size follows the bodies; the camera flies out when they outgrow
  the framed sphere.

### 6.3 Picks and hover on bodies
| State | Face | Edge | Corner |
|---|---|---|---|
| Hover (what a click would take) | white `#ffffff` overlay 0.35, drawn through | white, **3 px**, drawn on top (no depth test) | white point **11 px** |
| Picked / selected | green `#3cb371` overlay 0.5 | green 3 px (depth-tested) | green point 11 px |
| Tangent chain (fillet/chamfer reach) | — | light green `#8fd6ae` 3 px | — |
| Open edges of an unclosed trim | — | red `#e0564f`, section width + 2, on top | — |

Reach: edge 8 px, corner 10 px on screen; a corner beats the edges ending in it, an edge
beats its faces. A face is taken only if at least as near as the solid scan under the same
pixel (the scan's own depth + max(0.5 mm, 1 % of the model radius)); a see-through scan
never blocks.

### 6.4 Body lit from the list
Hovering a body row (or a component row) draws the whole body in flat blue `#3a9be8` at
0.26 over **everything** (no depth test), plus its edges in that blue (1.6 px, depth-tested).
The reverse: the body under the cursor in 3D rings its row in the list.

### 6.5 Ghost (feature preview)
* A **tool body** (extrude/revolve/extract/loft/sweep/thicken/sheet …): see-through 0.45,
  no depth write, in the new body's colour; a **cut is red `#e0564f`** and drawn through the
  body it is inside of (no depth test).
* A **feature on a body** (fillet, chamfer, shell, draft, move face, delete face, thread,
  hole, mirror …) previews as **the whole body it will leave**: solid (opacity 1), in the
  body's place, with its own edges; the original is hidden.
* Move body: the moved bodies' own meshes carried by the gizmo's transform at once; a copy
  leaves the originals visible.
* While an extrude grip is dragged, the last built ghost is **stretched** frame by frame
  along the normal (affine map) until the kernel's new one lands.

### 6.6 Grips (manipulators) and the grip field
Colour = the ghost colour (body colour; red for a cut); the hovered grip turns white;
cursor `grab`. Arrows are slim cones held at **44 px** on screen (taper arrow 0.8 × 44);
opacity 0.95. Rings are tori (tube 4.5 % of radius), opacity 0.85.

| Box | Grips |
|---|---|
| Extrude (Distance) | arrow on the end face through the profiles' area-weighted centroid pointing **forward** (`To`), arrow on the start face pointing **back** (`From`); pulling outward always grows |
| Extrude (Symmetric) | both arrows move `Total` |
| Extrude (plane/face/next/last ends) | arrow only on the side with a number |
| Extrude taper | arrow on the far edge (outline's max-X side, half way along), along the sheet X, riding the ghost's lean — not when swept askew |
| Extrude of a pulled face | the arrows through the face centroid along its normal (or the chosen direction) |
| Revolve | ring about the axis through the profiles' centroid plane, radius = centroid's distance from the axis (so it runs through the body); none until an axis is chosen |
| Fillet / Chamfer | arrow at the first edge's middle, pointing into the body along the bisector of its two faces' normals (pull in = bigger) |
| Shell | arrow on the first opened face, into the body (pull in = thicker wall) |
| Move face | flat: arrow at the face centre carried by the offset, out along the normal; round: arrow radially at the new radius, half way along |
| Move body | gizmo at the bodies' (moved) middle: an arrow along X, Y, Z in red/green/blue, each **30 px** long standing off **97 px** (just outside the rings), and a ring per axis, **90 px** radius, in the same colours — gimbal style (the X ring rides on the Y and Z turns) |

**Grip field**: after a drag the grip shows a number field (CSS2D label, 6 px off the
arrow tip in the tip's screen direction; on a ring, centred). During the drag it is
read-only and follows the number; once released it takes typing — Enter commits (number or
expression; angle for taper/turn rings), Escape closes. Shows 2 decimals (mm) with the unit
word after it; tooltip `The number this grip drags — type a value and press Enter; Escape closes the field`
(with an expression: `{expr} = {value} {unit}`). Dragging a grip lets a parameter go.

### 6.7 Committed sketches on the part
* Every visible committed sketch drawn in its plane: pieces as fat lines in `#3d8fd6`
  (section line width), **drawn through the part** (no depth test), opacity 0.95;
  construction pieces **dashed** (dash 3, gap 2) at 0.7.
* A sketch is hidden after a feature uses it; it is drawn anyway while a feature made from
  it is open, and during playback at its step.
* Hovered sketch line (when lines are armed) is redrawn white, 2 px wider.

### 6.8 Sketch regions as click targets
Closed regions (holes cut out) as flat faces in the sketch colour: rest **0.16**, hover
**0.32**; picked → green `#3cb371` **0.5**, picked+hover **0.62**. Drawn through the part.
Armed while an Extrude/Revolve box is open (picked *through* the part) and when idle (then
only where the scan is not in front).

### 6.9 Plane and axis hints
* **Picker hover** (a plane list or axis list option under the pointer, or the chosen one
  while the box is open): plane = a square sheet 2.3 × the model radius wide, amber
  `#ffb020` at **0.18**, with a fat amber border drawn on top (0.9) and the name as a pinned
  label at its centre; axis = an amber fat line (section width + 1, on top) across the
  whole part with its name at the foot. Hinted plane/axis elements light too.
* **Sketch open**: the sketch plane as a faint sheet in the sketch blue `#3d8fd6` at
  **0.06** with the sketch's name.

### 6.10 Scan colouring
* Live map (box open, Deviation map ticked): scan near the ghost coloured by signed
  distance on the Deviation ramp ± Range (green on, red outside, blue inside, dark caps);
  scan further than 10 ranges away keeps its colour.
* Deviation key: same against committed bodies; scan **deeper inside** than the reach =
  violet `rgb(186,50,196)`, counted in the legend.
* Curvature: flats green, convex towards red, hollows towards blue, sharper than the scale
  in dark caps; scan rim left bare.

### 6.11 Coordinate planes
XY, XZ, YZ at the origin, each a square (side = model radius) in its axis colour (XY blue,
XZ green, YZ red): sheet **0.06**, border 0.45; hovered or selected: sheet **0.2**, border
1.0 and its name label at the corner on the positive side of both axes. Depth-tested (a
body stands in front). Clickable only when idle and only where nothing else takes the click.

### 6.12 Varying-radius fillet labels
For each variable edge: a dot on each end (picked green) and a pinned chip at each end —
title `Start` / `End` (`Start 2` / `End 2` when several, title in `--accent-d`) and value
`R {r} mm` — standing off the end in the direction the edge leaves it on screen (above the
end when seen end-on), re-placed as the view turns.

### 6.13 Note pins
Numbered pins drawn into the canvas (so pictures show them), held at fixed pixel size:
dot **8 px** (white rim, dark `#1d2026` core) at the pin; badge **24 px** up-right of it
(corner offset 25 % of its size): dark disc ring, fill amber `#f2a516` (open) / blue
`#1877c0` (agent question, white number) / grey `#9aa1ab` (handled); lit = white ring and
1.15× size. Faded to 0.45 when the face it marks turns away. The note's text beside it as a
label. Click on a badge (3 px slack) opens the note card. Pins show in every workspace but
take clicks only here.

### 6.14 Review pins (agent review pictures only)
Badge 28 px, dot 9 px; dark disc with a ring red `#e0413a` (missing material), blue
`#2f74d8` (extra), amber `#f2a516` (face stands off); hidden when facing away. Put up for a
picture and taken down after.

### 6.15 Selection windows
* 3D (Shift + left drag over bodies, or in a Fillet/Chamfer/Shell/Draft/Move face/Delete
  face/Move body box): rightward = **window** (solid 1 px `#3cb371`, fill 12 % green) takes
  what lies wholly inside; leftward = **crossing** (dashed `#4db8ff`, fill 12 % blue) takes
  what it touches — hidden edges included. A Shift+click that does not move (≤ 6 px) is an
  additive click.

### 6.16 Patch network layer
Curves amber `#f2a516` (fat line), creases dashed (dash 4 gap 3); hovered/dragged white;
picked patch outline/fill `#4fb3ff`; refused regions marked red `#e5484d` with the reason;
suggested creases dashed violet `#b794f6`; nodes as dots, patch corners as squares (dark rim,
white centre, 11 px), grips between them; the patch being placed: its sides traced along the
scan in white (the side back to the first corner dashed) and its area shaded in the network
body's colour at 0.22; a closed patch stays shaded until its fit arrives. Curves along a
body edge take the body's colour.

---

## 7. The 2D sketch sheet

### 7.1 Placement and camera
* The 2D Measure sheet (`FlatScene`, transparent canvas) is mounted over the 3D viewport
  while a sketch is open, with the stage veil at 58 %. Same navigation as the 2D workspace.
* The 3D camera faces the sketch plane, keeps the same point under the sheet's centre and
  the same mm/px, and **swings round in 500 ms** when the sheet is first laid (and when
  Align turns the frame); it swings back to the previous view when the sketch closes.
* The sheet's zero is the part's origin projected onto the plane; its X a coordinate axis;
  the grid (spacing follows the zoom) anchored on the origin. No sheet edge: a click past
  the framed card still lands on the plane.

### 7.2 Palette (dark / light)

| Role | Dark | Light |
|---|---|---|
| Cut face cap (3D, Section view) | `#47535f` | `#b9bfc6` |
| Slice chains (hairline) | `#aeb8c4` @ 0.5 | `#4a5560` @ 0.5 |
| Fitted piece (has slice points) | `#4db8ff` | `#0a63c9` |
| Free piece, still loose (under-constrained) | `#c48bff` | `#7a3cc9` |
| Free piece, fully constrained | `#f2f5f8` | `#14181d` |
| Projected piece | `#ff7ac8` | `#c2268a` |
| Construction | `#9aa5b1` | `#5d6875` |
| Selected | `#ffb020` | `#e07a00` |
| Hover / what a click takes / held by a pointed glyph | `#ffe27a` | `#ff9d14` |
| Datum traces (coordinate planes' lines) + origin dot | `#e8a33d` | `#a8660a` |
| Closed region fill (opacity 0.16) | `#3cb371` | `#1f8a4c` |
| End dots | `#ffffff` | `#14181d` |
| Dimension ink | `#e8edf2` | `#14181d` |

Priority of a piece's colour: hover > selected > construction > projected > fitted >
loose > constrained.

Line weights (px at default width): piece **2.5**, selected **3.5**, hover **5**,
construction **1.5**, datum trace **1.2**. Dots: end **4**, picked end **8**, hover **10**,
origin **6**.

### 7.3 What the sheet shows
* Slice chains as edge chains (hairline, half see-through) with a dot on every end.
* Pieces (lines, arcs, circles, splines, points) in their colours; a dot on every end of a
  line/arc and every spline fit point.
* The three **datum traces** (XY, YZ, ZX lines where the part's coordinate planes cross the
  sheet), amber, named, drawn about the sketch (reach ≥ 100 mm or 2 × the sketch diagonal);
  lit/selected like pieces; the **origin dot** (amber, 6 px) when the origin is on the sheet.
* Each closed region filled green at 0.16 (holes cut out).
* Pending clicks of a drawing tool as pins in the loose colour; the **rubber-band ghost**
  of the shape to the cursor (line; whole rectangle — centre mode also its two diagonals;
  polygon + its construction circle; circle; arc; spline through the points); a tangent arc
  while dragging off the chain's last end.
* **Snap marks** at the snapped point (size 0.6 × the hand's reach): cross ✕ = an end
  (anchor); upright cross ✚ = the origin; triangle = a line's midpoint; diamond = a slice
  point (edge); short dash = on a piece or on a datum line; small square = grid.
* Pattern copies where they would land while the Pattern box is open.
* The Project tool's candidate (edge/piece under the cursor) drawn as the piece it would
  become.
* **Relation glyphs** (only with *Show relations*, or the one selected in the list), lifted
  −7 px up / +6 px right of their spot, 11.5 px, pointer cursor, tooltip = the relation's
  name; hover → accent text on accent-soft with a 2 px accent outline and lights the pieces
  and ends it holds. Glyphs: Coincident `•`, Horizontal `H`, Vertical `V`, Parallel `∥`,
  Perpendicular `⊥`, Tangent `◠`, Concentric `◎`, Equal `=`, Symmetric `⋈`, Point on `•`,
  Collinear `≡`, Fix `⚓︎`. Spots: line middle; arc middle; circle top; coincident at the
  shared end; tangent at the touching point; symmetric between the two points; Fix a third
  along a line / at a round piece's centre.
* **Dimensions** as a drawing carries them (Onshape style): extension lines off the
  measured points, a dimension line with filled arrows (turned in from outside on a short
  stretch), the number in a gap along the line; radius = leader with arrow on the rim;
  diameter = line through the centre with arrows on the rim at both ends; angle = an arc
  between the lines. Number text: bare (no unit), `R {n}`, `Ø {n}`, `{a}°`; `ƒ ` prefix when
  an expression drives it (tooltip = the expression). Label: no plate, 12.5 px ink with a
  3/3/6 px halo of the stage colour; hover → accent text, accent-soft fill, 2 px outline,
  cursor `move`; **drag** moves it (kept relative to what it measures); **click** opens a
  72 px mono input in place (typing `Ø12` over an R or `R6` over a Ø flips the form).
* The picture (when laid) under the sheet in the 3D view at its opacity.

### 7.4 Sheet box selection
With **Select**: a drag starting on bare sheet draws a box — rightward **window** (solid
1 px `#3cb371`, fill 12 %), leftward **crossing** (dashed `#4db8ff`, fill 12 %); the pieces
it would take light (hover colour) while dragging; Shift or Ctrl adds. With any other tool a
left drag on bare sheet pans.

---

## 8. Interaction model

### 8.1 Mouse buttons (whole workspace)
* Left: pick / draw / drag grips and pieces. Shift+left drag over bodies: selection window.
* Middle: navigates (pan/turn per the navigation scheme); a middle **click** (no drag)
  = confirm (same as Enter).
* Right (on the sketch sheet): pans. Right click on panel rows: context menus.
* Wheel: zoom (on a network curve: span count ±1).
* Number keys 0 iso, 1 top, 2 bottom, 3 front, 4 rear, 5 left, 6 right, 7 iso front-left,
  8 iso rear-left, 9 iso rear-right (not while a sketch is open).
* Arrow keys (no modifiers) over the 3D view: step through what lies under the cursor —
  corners, then edges, then faces front to back (↓/→ next, ↑/← previous); the stepped-to
  item is lit and the next click takes it; moving the cursor > 4 px lets the step go.

### 8.2 Idle (no box open)
| Gesture | Result |
|---|---|
| Hover a body | face/edge/corner under the cursor lit white; cursor `pointer`; its list row ringed |
| Hover a committed sketch region (scan not in front) / sketch line | region 0.32 / line white |
| Hover a coordinate plane | plane lit + name |
| Hover an element | element light (hover lights on) |
| Click body face/edge/corner | select it; chip bar with readings + verbs (§3.4) |
| Ctrl+click (or still Shift+click) | add to the selection (multi-pick → combined readings/chips) |
| Shift+drag | selection window into the selection |
| Click sketch region | select; chips `Extrude` / `Revolve` (selected region lit green) |
| Click sketch line | chip `Revolve about this` |
| Click element | chips per kind |
| Click coordinate plane | select (lit); chips `Sketch` / `Hide` |
| Click scan / empty space | clear selection and list picks |
| Esc | Note key down (if placing) else clear selection and list picks |
| Enter / middle click | press the panel's confirm button of the moment |
| Double-click feature row | open its box |

### 8.3 Extrude / Revolve box
| Gesture | Result |
|---|---|
| Click a sketch region (through the part) | toggle it in the profiles (also switches sketch if from another) |
| Revolve: click a sketch line or a line/cylinder/cone/circle element | set the axis (other element kinds: status `{name} has no axis to revolve about — pick a line, a cylinder, a cone or a circle.`) |
| Extrude Distance/Symmetric: click the scan/a body/a body corner (not over a region) | in front of the plane sets **To**, behind sets **From** (Symmetric: either half reaches it); result in the status line |
| Extrude "To a face": click a body face | sets the face |
| Drag an arrow / ring / taper arrow | the number follows (box fields update live); ghost stretches |
| Grip field | type + Enter after release |
| Enter / middle click | create (when ready) |
| Esc | cancel the box |
| Tab | next number field |

### 8.4 Feature boxes on bodies
| Box | Armed in 3D | Click | Window (Shift+drag) |
|---|---|---|---|
| Shell, Draft, Delete face, Hole, Mirror | faces of its body | toggle face (Delete face: the whole round/hole/boss group, + tangent rounds when ticked; Draft: first click = neutral face; Hole: a round face = axis, a flat face = drill at its centre; Mirror: flat face = plane) | faces (Shell, Draft, Delete face) |
| Thread | faces of **any** body | the round face (another body moves it) | — |
| Move face | faces (any body for the first, then its body); none while snapping | toggle face; snapping: click the **scan** → fit plane/cylinder there | faces |
| Fillet, Chamfer | edges of its body | toggle edge (lit green; tangent chain light green) | edges |
| Move body | faces of the other bodies (copy: all) | add/remove a body | every body it touches |
| Hole, Extract | elements | Hole: cylinder/circle (sets Ø) or line; Extract: plane/cylinder/cone/sphere/torus (else a status reason) | — |
| Fit surface | the scan | flood the region (Ctrl+click removes); or the marking tools when ticked | — |
| Patch network | its own layer (§8.6) | | |
Bodies not in play are dimmed to 0.15. Enter = create when ready; Esc = cancel.

### 8.5 Sketch (on the sheet)
| Tool | Click | Drag | Notes |
|---|---|---|---|
| Select | piece → select (a second pairs it); empty → clear | on a piece's end/centre/body → move it (snaps; constraints hold; a line moves square to itself, Ctrl frees it; dropped on another end joins); on bare sheet → box selection | right button pans |
| Line (L) | each click a corner of a chain; second click on the last end, Enter, Esc, or landing on another piece's end ends it | press on the chain's last end and drag → tangent arc | lines within 3° of level/upright are made so (inferred chip) |
| Rectangle (R) / Polygon / Circle (C) / Arc | clicks per the chosen way (§4.2 table) | left drag pans | shape rubber-bands to the cursor |
| Spline (S) | each click a fit point; last point again / double-click / Enter / Esc ends open; first point closes | drag a pending pin to move it | — |
| Fillet (F) / Chamfer | a corner point, or line then line | — | size comes out as a dimension; with 2 lines selected the key applies at once |
| Project | an edge of a body or another sketch's piece under the sheet → projected, fixed (magenta) | — | status `Projected {source} into the sketch — fixed where it lies; hang relations and dimensions on it.` or `Nothing to project there — click an edge of a body, or a piece of another sketch, that lies flat to this plane or square to it.` |
| Dimension (D) | 1st: line (length) / arc (R) / circle (Ø) / end / datum line; 2nd (optional): another piece/end/datum → distance or angle; then the number **follows the cursor** — a click puts it down and opens it for typing | drag an existing number to move it | Enter puts it at the layout's place; Esc takes it back |
| Relation (armed) | the pieces (Symmetric: two pieces then the mirror line or a datum line) | — | with pieces pre-selected the key applies at once |
| All drawing tools | **Ctrl** held = place exactly at the cursor (no snap); snaps to ends (✕), origin (✚), midpoints (▽), slice points (◇), pieces/datums (–), grid (□, only with Snap to grid) | — | what a tool would take is lit under the cursor before the click |

Keys in a sketch:
* `L` Line, `R` Rectangle, `C` Circle, `S` Spline, `F` Fillet, `D` Dimension (with a
  selection under Select: dimension it). Pressing the active tool's letter again → Select.
  (No Ctrl/Alt/Meta.)
* `Esc` backs out one step: a spline's points become the curve; else pending points / the
  dimension in hand / the selection / the selected constraint are cleared; else the tool
  goes back to Select. **Never closes the sketch.**
* `Enter` / middle click: a dimension in hand → put it down; spline points → make the
  curve; a line chain in progress → end it; else **Finish sketch**.
* `Delete` / `Backspace`: remove the selected constraint, else the selected pieces.
* The sheet holds the view keys (0–9 do nothing).

### 8.6 Patch network (layer takes every click on the scan)
Draw: click patch corners on the scan (or on a node, an edge — split there — or a grip);
sides follow the scan; close with a click on the first corner, Enter, double-click (two
clicks within 400 ms / 6 px) or a middle click; Enter with two corners = a line only;
Backspace = last corner back; Esc = drop the patch. Drag an outer edge of a patch across
the scan → the next patch beside it (new corners snap to nodes within 10 px). Drag a
corner/point along the scan; drag a shared edge to bend it (a point inserted where
pressed); Shift+click an edge inserts a point. Click a patch → pick it (Esc unpicks).
`Delete`/`Backspace`: the grip/node/curve/split-centre under the cursor, else the picked
patch, else the patch under the cursor. `C`: toggle crease on the curve under the cursor.
Wheel over a curve: its strip's spans ±1 (4–20). Click a body edge: a curve along it.
Fill tool: click inside a closed region → fill; on a patch → pick; Ctrl+click a patch →
empty it. Click a suggested crease → take it. Ctrl+Z / Ctrl+Y = the network's own
undo/redo while the box is open. Reach: node 10 px, grip 8 px, curve 8 px; drag slack 6 px.

### 8.7 Note placing
With the Note key pressed every click on the part is the note's (a corner/edge/face a body
click would take, else the scan point); a note card opens at the click. Esc stops placing.
With faces/edges selected, the chip `Note` pins one note on all of them.

### 8.8 Playback
Space = pause/resume (not when a button has focus), ← / → = step, Esc = stop (bar returns).
The rollback bar moves itself; rows stand down; the reached row is lit and scrolled into
view. Leaving the workspace stops it. Agent commands wait.

### 8.9 Undo
Ctrl+Z undo; Ctrl+Shift+Z or Ctrl+Y redo (global). Undo steps: sketch commit, every feature
commit/edit/delete, rollback moves, suppress, body names/colours, components. Not undo
steps: notes, playback, view-bar settings.

### 8.10 Workspace keyboard summary (`keyboard.ts`)
Precedence: sketch → extrude → revolve → network layer keys → other feature box →
playback → idle. The workspace consumes every key while it is on screen.

| State | Esc | Enter (and middle click) | Other |
|---|---|---|---|
| Sketch | back out one step (never closes) | dimension down / spline made / chain ended / Finish | L R C S F D; Delete/Backspace |
| Extrude / Revolve | Cancel | Create if ready | — |
| Patch network | while placing: drop the patch; else unpick the picked patch, else Cancel | while placing: close the patch; else Create if ready | Backspace, Delete, C, Ctrl+Z/Y |
| Other feature box | Cancel | Create if ready | — |
| Playback | Stop | — | Space, ←, → |
| Idle | Note key down → else clear selection + list picks | confirm button | — |

A field's own Escape (number restored) stops there; an untouched `To` field passes Enter to
Create and Escape to the box.

---

## 9. Notes for the egui port

* Panel: `egui::SidePanel::left` exact width 336 (282 on narrow), `ScrollArea::vertical`.
  Section heads: small-caps condensed `RichText` + an `ⓘ` small button that toggles an
  `egui::Area` card placed to the right of the panel at the dot's y.
* Feature key grid: `egui::Grid` with 4 columns (or `ui.columns(4)`), each key a
  vertical icon+label button with `on_disabled_hover_text`.
* Editor boxes: `egui::Frame` with `--well` fill, 1 px stroke and a 3 px accent bar painted
  on the left; rows as `ui.horizontal` with the label left and the control right-aligned
  (`with_layout(right_to_left)`).
* Expression fields: a `TextEdit::singleline` holding the typed text; on lost focus / Enter
  evaluate; draw `= value` (italic, weak) to the right; error as a red small label below.
* RefPicker: `ComboBox` cannot report the hovered option — use a popup `Area` listing
  `selectable_label`s and read `response.hovered()` to drive the plane/axis hint.
* Flyouts (shape tools): popup `Area` under the key after 250 ms hover; close after 180 ms off.
* Context chip bar, note card, grip field, menus: `egui::Area` with `Order::Foreground`,
  fixed at the click + 12 px, clamped to the screen.
* View bar: an `Area` anchored bottom-left 12/12; part keys = toggle button + caret
  `menu_button`; the "changed" dot painted on the key.
* Legend: an `Area` anchored right-top 14/14 stretching to the bottom.
* Rollback bar: a 14 px tall `Sense::drag()` strip between rows; compute the target gap
  from row rect mid-points while dragging and paint a 3 px accent mark there.
* Feature/body rows: `Sense::click_and_drag()` rows; right click → `context_menu`; double
  click → edit/rename; drag-and-drop via `egui::DragAndDrop`.
* Keep all user-visible strings in one table so the exact labels above are reusable.
