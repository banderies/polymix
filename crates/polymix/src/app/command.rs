//! Mesh workspace commands: what can be done without the mouse, in one list. The menus,
//! hotkeys, test scripts and agents all run them through [`MeshWorkspace::execute`], so a
//! command behaves the same whoever asks for it.

use super::*;
use crate::commands::SelectCmd;
use crate::scene::ObjectId;
use mm_core::{topology, TriSelection};

/// A tool that can be opened by name: the name for [`MeshCommand::OpenTool`] (and scripts),
/// its title in the panels, and what it is for (shown to agents).
pub struct ToolInfo {
    pub name: &'static str,
    pub title: &'static str,
    pub about: &'static str,
}

const fn t(name: &'static str, title: &'static str, about: &'static str) -> ToolInfo {
    ToolInfo { name, title, about }
}

/// Every tool [`make_tool`] opens. Tools marked "(clicks)" are worked by clicking the model
/// (agents: `mesh_tool` click); the others by their settings or their panel.
pub const TOOLS: &[ToolInfo] = &[
    t("select", "Select", "Paint, lasso or click to select triangles (clicks); its panel edits the selection"),
    t("inspector", "Inspector", "Find holes, non-manifold vertices and small floating parts; Auto Repair All in its panel"),
    t("transform", "Transform", "Move, turn and scale the object (gizmo or numbers in its panel)"),
    t("facetransform", "Face Transform", "Transform the selected triangles"),
    t("softtransform", "Soft Transform", "Transform the selection with a smooth falloff into the rest"),
    t("warp", "Warp", "Deform the selection by dragging handles (clicks)"),
    t("align", "Align", "Put a picked face on the ground, the origin or another surface (clicks)"),
    t("attracttotarget", "Attract To Target", "Pull the surface (or the selection) onto the target object"),
    t("aligntotarget", "Align To Target", "Register the object to the target surface (iterative closest point)"),
    t("fitprimitive", "Fit Primitive", "Fit a plane, sphere, cylinder, … to the selection"),
    t("addtube", "Add Tube", "A tube between two points on the surface (clicks)"),
    t("units", "Units / Dimensions", "Read and set the object's size in millimetres"),
    t("measure", "Measure", "Distances, coordinates, thickness, cylinder fit and vessel diameter (clicks)"),
    t("meshquery", "Mesh Query", "Colour by normal, position, triangle quality or curvature (clicks for triangle info)"),
    t("clearance", "Clearance", "Minimum distance to another object, coloured by distance"),
    t("sculpt", "Surface brushes", "Sculpt without changing the triangles (clicks and drags)"),
    t("sculptvolume", "Volume brushes", "Sculpt, refining the mesh under the brush (clicks and drags)"),
    t("sculptsmooth", "Smoothing brush", "Robust smoothing brush (clicks and drags)"),
    t("stamp", "Stamp", "Cut a stencil shape into the surface (clicks and drags)"),
    t("remesh", "Remesh", "Even, well-shaped triangles at a target edge length (whole object or selection)"),
    t("reduce", "Reduce", "Fewer triangles (to a percentage, budget or deviation)"),
    t("smooth", "Smooth", "Smooth the object or the selection"),
    t("smoothboundary", "Smooth Boundary", "Smooth the border of the selection"),
    t("closecracks", "Close Cracks", "Zip open boundaries that lie on top of each other"),
    t("join", "Join", "Join two selected boundary rings with a band of triangles"),
    t("regionbridge", "Bridge", "Connect two selected regions with a tube"),
    t("tubehandle", "Tube Handle", "A handle tube between two selected regions"),
    t("facegroups", "Generate Face Groups", "Segment the mesh into face groups by crease or normal angle"),
    t("weldboundaries", "Weld Boundaries", "Weld open boundary vertices closer than a tolerance"),
    t("makesolid", "Make Solid", "Rebuild a closed, manifold, non-self-intersecting surface"),
    t("makepattern", "Make Pattern", "Turn the object into a frame of tubes, or perforate / fill it"),
    t("unwrap", "Unwrap", "Compute UVs (texture coordinates)"),
    t("setuv", "Set UVs", "Assign UVs by planar, cylindrical, spherical or box projection, without unwrapping"),
    t("convex", "Convex Decomposition", "Approximate the object by convex pieces"),
    t("filtercomplex", "Filter Complex", "Keep cells of a complex (Generate Complex) by which inputs they are inside, joined into one object"),
    t("hair", "Grow Hair", "Strands on the selected triangles"),
    t("pivot", "Create Pivot", "Drop a point with a coordinate frame on the object (clicks)"),
    t("planecut", "Plane Cut", "Cut the object with a plane (drag a line; Shift+click a vessel for a square cut)"),
    t("trim", "Trim branch", "Click a vessel to cut it square to its axis and cap it (clicks)"),
    t("flowext", "Flow extensions", "Straight extensions from open (cut) ends"),
    t("cap", "Cap open ends", "Flat-cap open boundaries"),
    t("patches", "Boundary patches / CFD export", "Name wall / inlet / outlet patches and export multi-solid STL"),
    t("erasefill", "Erase & Fill", "Delete the selection and fill the hole smoothly"),
    t("extrude", "Extrude / Offset", "Move the selected region out along its normals"),
    t("extract", "Extract", "Copy the selected region into a new, thickened object"),
    t("mirror", "Mirror", "Mirror the object across a plane"),
    t("offset", "Offset / Hollow", "Offset the whole surface, or hollow it out"),
    t("thicken", "Thicken", "A wall of a given thickness on one side of the surface, open ends kept open with flat rims (a hollow model to print)"),
    t("boolean", "Boolean", "Union, difference or intersection with another object"),
    t("bridge", "Bridge boundaries", "Join two open boundaries with a tube (pick their markers)"),
    t("deviation", "Deviation", "Colour map of the distance to a reference object, with statistics"),
    t("cfdcheck", "Check for CFD", "Watertight, manifold, single shell, orientation, self-intersections, quality, patches; fixes in its panel"),
    t("centerline", "Centerline", "Vessel centerline tree with branch lengths and diameters"),
    t("stent", "Virtual stent", "A bare or covered stent between two vessel cross-sections (clicks)"),
];

/// Primitive shapes for [`MeshCommand::AddShape`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Sphere,
    Plane,
    Cube,
    Cylinder,
}

impl Shape {
    pub const ALL: [Shape; 4] = [Shape::Sphere, Shape::Plane, Shape::Cube, Shape::Cylinder];

    pub fn label(self) -> &'static str {
        match self {
            Shape::Sphere => "Sphere",
            Shape::Plane => "Plane",
            Shape::Cube => "Cube",
            Shape::Cylinder => "Cylinder",
        }
    }

    fn mesh(self) -> Mesh {
        match self {
            Shape::Sphere => mm_core::primitives::icosphere(12.5, 4),
            Shape::Plane => mm_core::primitives::plane_grid(25.0, 25.0, 32, 32),
            Shape::Cube => mm_core::primitives::cube(25.0, 8),
            Shape::Cylinder => {
                let mut m = mm_core::primitives::cylinder(12.5, 25.0, 64, 16, true);
                m.transform(|p| glam::DVec3::new(p.x, p.z - 12.5, -p.y));
                m
            }
        }
    }
}

/// The view menu's standard views, as (name, yaw, pitch).
pub const STANDARD_VIEWS: [(&str, f64, f64); 6] = {
    use std::f64::consts::{FRAC_PI_2, PI};
    [("Front", 0.0, 0.0), ("Back", PI, 0.0), ("Left", -FRAC_PI_2, 0.0), ("Right", FRAC_PI_2, 0.0), ("Top", 0.0, FRAC_PI_2), ("Bottom", 0.0, -FRAC_PI_2)]
};

#[derive(Clone, Debug, PartialEq)]
pub enum MeshCommand {
    /// A selection command on the active object.
    Select(SelectCmd),
    /// Select the triangles of these face groups (added to the selection with `add`).
    SelectGroups { groups: Vec<i32>, add: bool },
    /// Select the triangles along open boundary number `boundary` (see [`boundary_loops`]):
    /// a hole's rim, as double-clicking it does.
    SelectBoundary { boundary: usize, add: bool },
    /// Select the triangles within `radius` of a point.
    SelectNear { at: DVec3, radius: f64, add: bool },
    /// Delete the selected triangles.
    Discard,
    /// The selection becomes its own object.
    Separate { keep_original: bool },
    FlipNormals,
    CreateFaceGroup,
    SeparateShells,
    CombineVisible,
    GenerateComplex,
    SplitComplex,
    Duplicate,
    Delete(ObjectId),
    /// Send a closed object to the Segmentation workspace, where its inside becomes a segment
    /// of the volume shown there.
    SendToSegmentation(ObjectId),
    /// Show an image on the object through its UVs (`None`: remove its texture).
    SetTexture(ObjectId, Option<std::path::PathBuf>),
    /// Make the object a reference (a see-through guide) or editable again.
    SetReference(ObjectId, bool),
    ClearScene,
    AddShape(Shape),
    Activate(ObjectId),
    SetVisible(ObjectId, bool),
    Rename(ObjectId, String),
    SetTarget(Option<ObjectId>),
    Undo,
    Redo,
    Focus,
    /// One of [`STANDARD_VIEWS`] (by index).
    StandardView(usize),
    /// Look from yaw / pitch (degrees; 0, 0 is the front view).
    Orbit { yaw_deg: f64, pitch_deg: f64 },
    Wireframe(bool),
    Normals(NormalMode),
    Colors(ColorMode),
    Boundaries(bool),
    Grid(bool),
    /// The Segmentation tab's image as slices among the meshes.
    ImageSlices(bool),
    /// Grid spacing in mm (0: automatic).
    GridSize(f64),
    /// Every object see-through (X-ray).
    XRay(bool),
    Orthographic(bool),
    Up(UpAxis),
    /// Open a tool by its name in [`TOOLS`] (the open tool is accepted first).
    OpenTool(String),
    /// Change the open tool's settings (see [`Tool::set_settings`]).
    ToolSettings(serde_json::Value),
    /// One of the open tool's actions (see [`Tool::actions`]) with its arguments.
    ToolAction(String, serde_json::Value),
    AcceptTool,
    CancelTool,
    /// Click the surface nearest to a point as a user would (for tools worked by clicking),
    /// turning the view to face it first if it is hidden.
    Click { at: DVec3, double: bool, shift: bool },
    Open(Vec<PathBuf>),
    Export { path: PathBuf, scope: ExportScope },
    SaveProject(PathBuf),
}

/// Open boundaries in a fixed order (by their smallest vertex id), so a boundary's number
/// stays the same between calls.
pub fn boundary_loops(m: &Mesh) -> Vec<topology::EdgeLoop> {
    let mut loops = topology::boundary_loops(m);
    loops.sort_by_key(|l| l.vertices.iter().min().copied());
    loops
}

impl MeshWorkspace {
    /// Run a command for a menu, hotkey or script: what went wrong goes to the status line.
    pub(super) fn run(&mut self, cmd: MeshCommand, ctx: &egui::Context, vp: Viewport) {
        if let Err(e) = self.execute(cmd, ctx, vp) {
            self.status = e;
        }
    }

    /// Run a command. `Ok` says what was done, `Err` why nothing was.
    pub(super) fn execute(&mut self, cmd: MeshCommand, ctx: &egui::Context, vp: Viewport) -> Result<String, String> {
        use MeshCommand as C;
        let before = self.history.generation();
        // Commands that change the scene: nothing happened if they left no undo step.
        let changed = |ws: &Self| -> Result<String, String> { if ws.history.generation() != before { Ok(ws.status.clone()) } else { Err(ws.status.clone()) } };
        match cmd {
            C::Select(c) => {
                self.active_id()?;
                commands::selection_command(&mut self.cmd_cx(), c);
                Ok(self.status.clone())
            }
            C::SelectGroups { groups, add } => {
                let id = self.active_id()?;
                self.history.push("Select: face groups", &self.scene);
                let o = self.scene.get_mut(id).unwrap();
                let m = o.mesh.clone();
                if !add {
                    o.selection = TriSelection::new(&m);
                }
                for t in m.triangle_ids().filter(|&t| groups.contains(&m.tri_group(t))) {
                    o.selection.insert(t);
                }
                o.touch_selection();
                self.status = format!("{} triangles selected", o.selection.len());
                Ok(self.status.clone())
            }
            C::SelectBoundary { boundary, add } => {
                let id = self.active_id()?;
                let m = self.scene.get(id).unwrap().mesh.clone();
                let loops = boundary_loops(&m);
                let lp = loops.get(boundary).ok_or_else(|| format!("No open boundary {boundary}: the object has {}", loops.len()))?;
                let rim: Vec<u32> = lp.edges.iter().map(|&e| m.edge_triangles(e)[0]).collect();
                self.history.push("Select: boundary", &self.scene);
                let o = self.scene.get_mut(id).unwrap();
                if !add {
                    o.selection = TriSelection::new(&m);
                }
                for t in rim {
                    o.selection.insert(t);
                }
                o.touch_selection();
                self.status = format!("{} triangles selected", o.selection.len());
                Ok(self.status.clone())
            }
            C::SelectNear { at, radius, add } => {
                let id = self.active_id()?;
                let o = self.scene.get_mut(id).unwrap();
                let bvh = o.bvh();
                let m = o.mesh.clone();
                let mut near = Vec::new();
                bvh.query_sphere(&m, at, radius, |t| near.push(t));
                if near.is_empty() {
                    return Err(format!("No triangles within {radius} mm of [{:.3}, {:.3}, {:.3}]", at.x, at.y, at.z));
                }
                self.history.push("Select: near a point", &self.scene);
                let o = self.scene.get_mut(id).unwrap();
                if !add {
                    o.selection = TriSelection::new(&m);
                }
                for t in near {
                    o.selection.insert(t);
                }
                o.touch_selection();
                self.status = format!("{} triangles selected", o.selection.len());
                Ok(self.status.clone())
            }
            C::Discard => {
                commands::discard_selection(&mut self.cmd_cx());
                changed(self)
            }
            C::Separate { keep_original } => {
                commands::separate_selection(&mut self.cmd_cx(), keep_original);
                self.scene_changed(ctx, vp);
                changed(self)
            }
            C::FlipNormals => {
                commands::flip_selection_normals(&mut self.cmd_cx());
                changed(self)
            }
            C::CreateFaceGroup => {
                commands::create_face_group(&mut self.cmd_cx(), false);
                changed(self)
            }
            C::SetReference(id, on) => {
                let o = self.scene.get_mut(id).ok_or("no such object")?;
                o.reference = on;
                o.color.w = if on { 0.35 } else { 1.0 };
                let name = o.name.clone();
                if on && self.scene.active == Some(id) {
                    self.scene.active = self.scene.objects.iter().find(|o| !o.reference).map(|o| o.id);
                }
                self.status = format!("{name} is {}", if on { "a reference (see-through, not picked or exported with the visible objects)" } else { "editable again" });
                Ok(self.status.clone())
            }
            C::SetTexture(id, path) => {
                let tex = match &path {
                    Some(p) => Some(std::sync::Arc::new(crate::scene::TextureImage::load(p).map_err(|e| format!("{e:#}"))?)),
                    None => None,
                };
                let o = self.scene.get_mut(id).ok_or("no such object")?;
                let has_uvs = o.uvs.as_ref().is_some_and(|u| u.topo == o.mesh.topology_version());
                let name = o.name.clone();
                o.texture = tex;
                self.status = match (&path, has_uvs) {
                    (None, _) => format!("Texture removed from {name}"),
                    (Some(p), true) => {
                        if !matches!(self.view.colors, ColorMode::Texture | ColorMode::TextureUnlit) {
                            self.view.colors = ColorMode::Texture;
                        }
                        format!("{} shown on {name}", p.file_name().map_or(String::new(), |n| n.to_string_lossy().to_string()))
                    }
                    (Some(_), false) => format!("Texture set on {name}; it shows once the object has UVs (Edit → Unwrap)"),
                };
                Ok(self.status.clone())
            }
            C::SendToSegmentation(id) => {
                let o = self.object(id)?;
                let (name, mesh) = (o.name.clone(), o.mesh.clone());
                self.to_segmentation.push((name.clone(), (*mesh).clone()));
                self.status = format!("Sent {name} to the Segmentation tab");
                Ok(self.status.clone())
            }
            C::GenerateComplex => {
                if !self.accept_tools(0, ctx, vp) {
                    return Err("Wait for the open tool to finish computing (or cancel it).".into());
                }
                // The booleans take seconds on large inputs: a tool computes them in the
                // background and adds the complex when they are done.
                match commands::complex_inputs(&self.scene) {
                    Ok(inputs) => {
                        let n = inputs.len();
                        self.tools.push(Box::new(crate::tools::complex::GenerateComplexTool::new(inputs, ctx)));
                        self.status = format!("Generate Complex: computing the cells of {n} objects…");
                        Ok("Generate Complex is computing the cells; app_wait, then the complex is the active object (named Complex)".into())
                    }
                    Err(e) => {
                        self.status = e;
                        Err(self.status.clone())
                    }
                }
            }
            C::SeparateShells | C::CombineVisible | C::SplitComplex | C::Duplicate | C::Delete(_) | C::ClearScene | C::AddShape(_) => {
                if !self.accept_tools(0, ctx, vp) {
                    return Err("Wait for the open tool to finish computing (or cancel it).".into());
                }
                match cmd {
                    C::SeparateShells => commands::separate_shells(&mut self.cmd_cx()),
                    C::CombineVisible => commands::combine_visible(&mut self.cmd_cx()),
                    C::SplitComplex => commands::split_complex(&mut self.cmd_cx()),
                    C::Duplicate => commands::duplicate_object(&mut self.cmd_cx()),
                    C::Delete(id) => {
                        self.object(id)?;
                        commands::delete_object(&mut self.cmd_cx(), id);
                    }
                    C::ClearScene => {
                        self.history.push("Clear scene", &self.scene);
                        self.scene.objects.clear();
                        self.scene.active = None;
                        self.status = "Scene cleared".into();
                    }
                    C::AddShape(shape) => {
                        self.history.push(format!("Add {}", shape.label()), &self.scene);
                        let first = self.scene.is_empty();
                        self.scene.add(shape.label().to_lowercase(), shape.mesh());
                        if first || self.prefs.reset_camera_on_import {
                            self.focus_scene();
                        }
                        self.status = format!("Added a {}", shape.label().to_lowercase());
                    }
                    _ => unreachable!(),
                }
                self.scene_changed(ctx, vp);
                changed(self)
            }
            C::Activate(id) => {
                let name = self.object(id)?.name.clone();
                self.scene.active = Some(id);
                Ok(format!("{name} is active"))
            }
            C::SetVisible(id, visible) => {
                self.object(id)?;
                self.scene.get_mut(id).unwrap().visible = visible;
                Ok(format!("{} {}", self.scene.get(id).unwrap().name, if visible { "shown" } else { "hidden" }))
            }
            C::Rename(id, name) => {
                let name = name.trim().to_string();
                if name.is_empty() {
                    return Err("A name cannot be empty.".into());
                }
                self.object(id)?;
                self.scene.get_mut(id).unwrap().name = name.clone();
                Ok(format!("Renamed to {name}"))
            }
            C::SetTarget(id) => {
                if let Some(id) = id {
                    self.object(id)?;
                    if self.scene.objects.len() < 2 {
                        return Err("A target needs a second object.".into());
                    }
                }
                commands::set_target(&mut self.cmd_cx(), id);
                Ok(self.status.clone())
            }
            C::Undo | C::Redo => {
                if self.tool_busy() {
                    return Err("Wait for the open tool to finish computing (or cancel it).".into());
                }
                let undo = cmd == C::Undo;
                if undo && !self.history.can_undo() || !undo && !self.history.can_redo() {
                    return Err(format!("Nothing to {}.", if undo { "undo" } else { "redo" }));
                }
                if undo {
                    self.undo(ctx);
                } else {
                    self.redo(ctx);
                }
                Ok(self.status.clone())
            }
            C::Focus => {
                self.focus_scene();
                Ok("View fitted to the visible objects".into())
            }
            C::StandardView(k) => {
                let (name, yaw, pitch) = *STANDARD_VIEWS.get(k).ok_or("no such view")?;
                self.standard_view(yaw, pitch);
                Ok(format!("{name} view"))
            }
            C::Orbit { yaw_deg, pitch_deg } => {
                self.standard_view(yaw_deg.to_radians(), pitch_deg.clamp(-90.0, 90.0).to_radians());
                Ok(format!("Looking from yaw {yaw_deg}°, pitch {pitch_deg}°"))
            }
            C::Wireframe(on) => {
                self.view.wireframe = on;
                Ok(format!("Wireframe {}", if on { "on" } else { "off" }))
            }
            C::Normals(m) => {
                self.view.normals = m;
                Ok(m.label().to_string())
            }
            C::Colors(m) => {
                self.view.colors = m;
                Ok(m.label().to_string())
            }
            C::Boundaries(on) => {
                self.view.boundaries = on;
                Ok(format!("Open boundaries {}", if on { "shown" } else { "hidden" }))
            }
            C::Grid(on) => {
                self.view.show_grid = on;
                Ok(format!("Grid {}", if on { "shown" } else { "hidden" }))
            }
            C::XRay(on) => {
                self.view.xray = on;
                Ok(format!("X-ray {}", if on { "on" } else { "off" }))
            }
            C::GridSize(mm) => {
                self.view.grid_mm = mm.max(0.0);
                Ok(if mm > 0.0 { format!("Grid every {mm} mm") } else { "Grid spacing automatic".into() })
            }
            C::ImageSlices(on) => {
                self.slices.show = on;
                Ok(match (on, &self.slices.source) {
                    (false, _) => "Image slices hidden".into(),
                    (true, Some(s)) => format!("Slices of {} shown", s.name),
                    (true, None) => "Image slices: open an image in the Segmentation tab to see its slices".into(),
                })
            }
            C::Orthographic(on) => {
                self.camera.orthographic = on;
                Ok(if on { "Orthographic view".into() } else { "Perspective view".into() })
            }
            C::Up(axis) => {
                self.camera.up_axis = axis;
                Ok(if axis == UpAxis::Z { "Z up".into() } else { "Y up".into() })
            }
            C::OpenTool(name) => {
                self.switch_to(SwitchTo::Tool(name.clone()), ctx, vp);
                if self.pending_switch.is_some() {
                    return Ok(format!("{name} opens when the open tool has finished computing"));
                }
                match self.tools.last() {
                    Some(t) => Ok(format!("{} is open", t.name())),
                    None => Err(format!("{name} did not open")),
                }
            }
            C::ToolSettings(v) => {
                let mut t = self.tools.pop().ok_or("No tool is open.")?;
                let r = t.set_settings(&v, &mut self.tool_cx(ctx, vp));
                let name = t.name();
                self.tools.push(t);
                r.map(|()| format!("{name} settings changed"))
            }
            C::ToolAction(name, args) => {
                let mut t = self.tools.pop().ok_or("No tool is open.")?;
                let actions = t.actions();
                let result = match actions.iter().find(|a| a.name == name) {
                    Some(a) => a.check(&args).and_then(|()| t.act(&name, &args, &mut self.tool_cx(ctx, vp))),
                    None if actions.is_empty() => Err(format!("{} has no actions", t.name())),
                    None => Err(format!("{} has no action {name}; it has {}", t.name(), actions.iter().map(|a| a.signature()).collect::<Vec<_>>().join(", "))),
                };
                // A tool may hand over to another one (Measure → Place stent).
                let sub = t.take_request();
                self.tools.push(t);
                self.tools.extend(sub);
                result
            }
            C::AcceptTool => {
                let mut t = self.tools.pop().ok_or("No tool is open.")?;
                let name = t.name();
                let mut cx = self.tool_cx(ctx, vp);
                if t.accept(&mut cx) {
                    t.on_close(&mut cx);
                    return Ok(format!("{name} applied. {}", self.status));
                }
                let busy = t.busy().is_some();
                self.tools.push(t);
                if self.history.generation() != before {
                    // Tools like Trim apply and stay open for the next one.
                    Ok(format!("{name} applied (it stays open for more). {}", self.status))
                } else if busy {
                    Ok(format!("{name} is computing; it is applied when done"))
                } else {
                    Err(format!("{name} has nothing to apply: it works through its panel or clicks (close it with cancel)"))
                }
            }
            C::CancelTool => {
                let mut t = self.tools.pop().ok_or("No tool is open.")?;
                let name = t.name();
                let mut cx = self.tool_cx(ctx, vp);
                if t.cancel(&mut cx) {
                    t.on_close(&mut cx);
                    Ok(format!("{name} closed"))
                } else {
                    self.tools.push(t);
                    Err(format!("{name} cannot be closed now"))
                }
            }
            C::Click { at, double, shift } => self.click_at(at, double, shift, vp),
            C::Open(paths) => {
                if self.import_job.is_some() || self.project_job.is_some() {
                    return Err("Files are being opened already.".into());
                }
                for p in &paths {
                    if !p.is_file() {
                        return Err(format!("No such file: {}", p.display()));
                    }
                    if !polymix_workspace::Workspace::accepts(self, p) {
                        return Err(format!("Not a mesh or project file: {}", p.display()));
                    }
                }
                let n = paths.len();
                self.start_import(paths, ctx);
                Ok(format!("Opening {n} file(s)"))
            }
            C::Export { path, scope } => self.export_to(path, scope, ctx),
            C::SaveProject(path) => self.save_project_to(path, ctx),
        }
    }

    /// Nothing left to happen from an agent's last command: no tool computing or waiting to
    /// open, no clicks queued, nothing loading or saving.
    pub(super) fn agent_quiet(&self) -> bool {
        !self.tool_busy() && !self.tools.last().is_some_and(|t| t.pending()) && self.synthetic.is_empty() && self.import_job.is_none() && self.pending_import.is_none() && self.project_job.is_none() && self.export_job.is_none() && self.pending_switch.is_none()
    }

    fn active_id(&self) -> Result<ObjectId, String> {
        self.scene.active.ok_or_else(|| if self.scene.is_empty() { "The scene is empty.".to_string() } else { "No object is active.".to_string() })
    }

    fn object(&self, id: ObjectId) -> Result<&crate::scene::SceneObject, String> {
        self.scene.get(id).ok_or_else(|| format!("No object with id {id}"))
    }

    /// Click the visible surface nearest to `at`, as a user would. If that point is hidden
    /// from the current view, the view turns to face it (the user then sees where the click
    /// went).
    fn click_at(&mut self, at: DVec3, double: bool, shift: bool, vp: Viewport) -> Result<String, String> {
        let diag = self.scene.bounds().diagonal().max(1e-9);
        let mut best: Option<(ObjectId, DVec3, DVec3)> = None;
        let mut best_d = f64::INFINITY;
        for o in self.scene.objects.iter_mut().filter(|o| o.visible) {
            let bvh = o.bvh();
            if let Some(h) = bvh.nearest(&o.mesh, at, diag) {
                if h.dist_sq < best_d {
                    best_d = h.dist_sq;
                    best = Some((o.id, h.point, o.mesh.tri_normal(h.tid)));
                }
            }
        }
        let (id, p, n) = best.ok_or("No visible surface to click.")?;
        let tol = diag * 2e-3;
        let screen = |ws: &mut Self| -> Option<egui::Pos2> {
            let s = vp.to_screen(ws.camera.project(p, vp.aspect())?);
            if !vp.rect.shrink(4.0).contains(s) {
                return None;
            }
            let ray = ws.camera.ray_from_ndc(vp.ndc(s), vp.aspect());
            let (hit_id, h) = tools::pick_ray(&mut ws.scene, &ray, None)?;
            (hit_id == id && (h.point - p).length() < tol).then_some(s)
        };
        let mut spot = screen(self);
        if spot.is_none() {
            // Face the point head-on, closer and closer until nothing is in front of it.
            self.remember_view();
            self.view_anim = None;
            self.camera.target = p;
            self.camera.look_along(-n);
            for _ in 0..8 {
                spot = screen(self);
                if spot.is_some() {
                    break;
                }
                self.camera.distance *= 0.5;
            }
        }
        let s = spot.ok_or("The point cannot be seen from any distance (it may be inside the model).")?;
        let base = PointerInput { shift, ..Default::default() };
        self.synthetic.push_back(PointerInput { pos: Some(s), ..base.clone() });
        self.synthetic.push_back(PointerInput { pos: Some(s), pressed: true, down: true, ..base.clone() });
        self.synthetic.push_back(PointerInput { pos: Some(s), released: true, clicked: true, double_clicked: double, ..base });
        Ok(format!("{} at [{:.3}, {:.3}, {:.3}]", if double { "Double-clicked" } else { "Clicked" }, p.x, p.y, p.z))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::scene::History;
    use serde_json::json;

    /// A Mesh workspace for tests: no GPU, no preferences, autosave in a scratch folder.
    pub fn workspace() -> (MeshWorkspace, egui::Context, Viewport) {
        static ENV: std::sync::Once = std::sync::Once::new();
        ENV.call_once(|| {
            std::env::set_var("POLYMIX_DATA_DIR", std::env::temp_dir().join(format!("polymix-command-test-{}", std::process::id())));
            std::env::set_var("POLYMIX_SCRIPT", "");
        });
        let ctx = egui::Context::default();
        let ws = MeshWorkspace::new(&ctx, None, None);
        (ws, ctx, Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) })
    }

    #[test]
    fn every_catalog_tool_opens_and_typed_settings_match() {
        let mut scene = Scene::default();
        scene.add("cube", mm_core::primitives::cube(10.0, 2));
        let (mut history, mut camera, mut status) = (History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: BrushScroll::Shift };
        for t in TOOLS {
            let tool = make_tool(t.name, &mut cx).unwrap_or_else(|| panic!("{} does not open", t.name));
            let schema = tool.settings_schema();
            assert_eq!(tool.settings().is_some(), schema.is_some(), "{}: settings and schema disagree", t.name);
            for a in tool.actions() {
                assert!(a.schema()["properties"].is_object(), "{}: {}", t.name, a.name);
                assert!(!a.about.is_empty(), "{}: {}", t.name, a.name);
            }
            if let (Some(s), Some(schema)) = (tool.settings(), schema) {
                let props = schema["properties"].as_object().map(|p| p.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
                let keys: Vec<String> = s.as_object().unwrap().keys().cloned().collect();
                assert_eq!(keys.len(), props.len(), "{}: {keys:?} vs {props:?}", t.name);
                assert!(keys.iter().all(|k| props.contains(k)), "{}: {keys:?} vs {props:?}", t.name);
            }
        }
    }

    #[test]
    fn commands_change_the_scene_and_undo() {
        let (mut ws, ctx, vp) = workspace();
        ws.execute(MeshCommand::AddShape(Shape::Cube), &ctx, vp).unwrap();
        let id = ws.scene.active.unwrap();
        let tris = ws.scene.get(id).unwrap().mesh.triangle_count();
        // Nothing selected: discarding is refused and leaves no undo step.
        assert!(ws.execute(MeshCommand::Discard, &ctx, vp).is_err());
        ws.history.by_agent = true;
        ws.execute(MeshCommand::SelectNear { at: DVec3::new(12.5, 0.0, 0.0), radius: 4.0, add: false }, &ctx, vp).unwrap();
        let picked = ws.scene.get(id).unwrap().selection.len();
        assert!(picked > 0 && picked < tris);
        ws.execute(MeshCommand::Discard, &ctx, vp).unwrap();
        assert_eq!(ws.scene.get(id).unwrap().mesh.triangle_count(), tris - picked);
        assert!(ws.history.undo_label().unwrap().starts_with("Agent: "));
        ws.history.by_agent = false;
        ws.execute(MeshCommand::Undo, &ctx, vp).unwrap();
        assert_eq!(ws.scene.get(id).unwrap().mesh.triangle_count(), tris);
        // The hole the discard made is gone again; open boundary 0 does not exist.
        assert!(ws.execute(MeshCommand::SelectBoundary { boundary: 0, add: false }, &ctx, vp).unwrap_err().contains("has 0"));
        ws.execute(MeshCommand::Redo, &ctx, vp).unwrap();
        ws.execute(MeshCommand::SelectBoundary { boundary: 0, add: false }, &ctx, vp).unwrap();
        assert!(!ws.scene.get(id).unwrap().selection.is_empty());
        assert!(ws.execute(MeshCommand::Delete(9999), &ctx, vp).is_err());
        ws.execute(MeshCommand::Rename(id, "box".into()), &ctx, vp).unwrap();
        assert_eq!(ws.scene.get(id).unwrap().name, "box");
    }

    #[test]
    fn tools_open_take_settings_and_close() {
        let (mut ws, ctx, vp) = workspace();
        ws.execute(MeshCommand::AddShape(Shape::Sphere), &ctx, vp).unwrap();
        assert_eq!(ws.execute(MeshCommand::OpenTool("remesh".into()), &ctx, vp).unwrap(), "Remesh is open");
        ws.execute(MeshCommand::ToolSettings(json!({"edge_length": 2.0, "preserve_sharp": true})), &ctx, vp).unwrap();
        let s = ws.tools.last().unwrap().settings().unwrap();
        assert_eq!((s["edge_length"].as_f64(), s["preserve_sharp"].as_bool()), (Some(2.0), Some(true)));
        let e = ws.execute(MeshCommand::ToolSettings(json!({"edge_lenght": 1.0})), &ctx, vp).unwrap_err();
        assert!(e.contains("no setting \"edge_lenght\"") && e.contains("edge_length"), "{e}");
        assert!(ws.execute(MeshCommand::ToolSettings(json!({"mode": "Sideways"})), &ctx, vp).is_err());
        ws.execute(MeshCommand::CancelTool, &ctx, vp).unwrap();
        assert!(ws.tools.is_empty());
        assert!(ws.execute(MeshCommand::AcceptTool, &ctx, vp).is_err());
    }

    #[test]
    fn faced_clicks_queue_pointer_events() {
        let (mut ws, ctx, vp) = workspace();
        ws.execute(MeshCommand::AddShape(Shape::Cube), &ctx, vp).unwrap();
        ws.camera.fit(&ws.scene.bounds(), vp.aspect());
        // The back face is hidden from the front: the view turns to face it.
        let before = ws.camera.forward();
        ws.execute(MeshCommand::Click { at: DVec3::new(0.0, 0.0, -20.0), double: false, shift: false }, &ctx, vp).unwrap();
        assert_eq!(ws.synthetic.len(), 3);
        assert!(ws.synthetic.back().unwrap().clicked);
        assert!(ws.camera.forward().dot(before) < 0.5, "the view should have turned");
    }
}
