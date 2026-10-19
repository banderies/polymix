# Agent access (MCP)

AI agents (Claude Code, Claude Desktop, Cursor and other [Model Context
Protocol](https://modelcontextprotocol.io) clients) can work with the session open in Polymix.
They can read the mesh scene, every tool's state (picked points, readouts, previews, warnings),
the image catalog, the segmentation, the log, a screenshot and the panels on screen. If you
allow it, they can also select, edit, run tools and their actions, segment and prepare images,
work the panels, and open and export files. Use it to debug Polymix with an agent, or to have an
agent help with the data on screen.

## Turning it on

Click **Agents** at the right end of the tab bar and tick *Let AI agents read this session*.
Tick *Let agents make changes* as well if agents may change things. The button's dot is green
while agents can connect, and blue for a moment after each call. Polymix remembers both
settings.

To connect Claude Code, press **Add to Claude Code**. It registers Polymix for all your
projects with Claude Code's `claude` command, and a new Claude Code session then has the
tools. Polymix looks for `claude` in the usual install places and in your login shell's
`PATH`, since apps opened from the Finder get a bare `PATH`. When the port or token changes,
the button becomes **Update Claude Code**.

The same by hand, or for another MCP client, uses *Copy command*, *Copy URL* and *Copy token*:

```sh
claude mcp add --transport http --scope user polymix http://127.0.0.1:47390/mcp --header "Authorization: Bearer <token>"
```

*New token* replaces the token, and clients set up with the old one stop working until they
are updated.

For development and scripted runs:
- `POLYMIX_MCP=1` turns access on for one session without saving it.
- `POLYMIX_MCP_WRITE=1` also allows changes.
- `POLYMIX_MCP_PORT` picks the port.

## What changes look like

- **Changes are visible.** A change shows the workspace it changes, so it happens in front of
  you, and the status line starts with "Agent:".
- **Changes are undoable.** Every change is an undo step labelled "Agent: …" (*Edit → Undo*),
  including changes made through a panel.
- **Agents cannot change agent access.** Acting on the Agent access window is refused, so an
  agent cannot allow itself changes or replace the token.
- **Without permission, nothing changes.** Tools that change things answer with a message
  asking you to tick *Let agents make changes*. Reading tools keep working.

## Tools

### App

| Tool | What it does |
|---|---|
| `app_status` | Version, the workspace tab shown, each workspace running / busy / stopped (with the error), window size |
| `app_screenshot` | PNG of the window (`max_size`, default 1568 px on the longest side) |
| `app_ui` | The widgets of the panels shown, one per line: role, label, value, state (checked, disabled, …) and an `#id` (`contains`, `limit`) |
| `app_ui_action` | Click, set a value, step, expand or collapse a widget by its `#id`, as a screen reader would |
| `app_wait` | Wait until tools, files and clicks have finished (`seconds`, up to 25) |
| `app_show` | Show a workspace tab |
| `app_log` | Recent log messages (`lines`, `level`): Polymix's from info up, libraries' warnings and errors |

### Mesh

| Tool | What it does |
|---|---|
| `mesh_scene` | Every object (id, name, counts, open edges, bounds, selection, face groups), active / target object, the tool stack and whether it is computing, status line, undo / redo steps, background jobs, camera, display options |
| `mesh_object` | One object (`object`: id or name): area, enclosed volume and orientation, edge lengths, validity, shells, numbered open boundaries with perimeters and centres, defects, face groups and patch names, pivots |
| `mesh_selection` | The selected triangles: count, area, bounds, separate regions, border length, face groups |
| `mesh_tool_state` | The open tool's full state: computing, settings (with their JSON Schema on request), what it holds and shows (picked points, readouts, previews, warnings, errors), and its actions with their arguments |
| `mesh_tools` | The tools `mesh_tool` opens with their actions, and one tool's settings schema and action arguments |
| `mesh_select` | all, clear, invert, grow, shrink, connected, …; by face `groups`; the rim of open `boundary` N; triangles `near` a point |
| `mesh_objects` | Activate, show, hide, rename; set or clear the target |
| `mesh_edit` | Discard / separate / flip the selection, new face group, duplicate, delete, separate shells, combine, add a shape, clear the scene |
| `mesh_history` | Undo or redo (`steps`) |
| `mesh_view` | Standard views, fit, orbit; wireframe, normals, colours, boundaries, grid, orthographic, up axis |
| `mesh_tool` | open (`tool`, `settings`), set (`settings`), run (`command`, `args`: one of the tool's actions), accept, cancel, click (`at`: the surface nearest a point) |
| `mesh_file` | Open meshes or a project, export (`scope`), save the project. Absolute paths only; existing files are kept unless `overwrite` |

### Volume and Segmentation

| Tool | What it does |
|---|---|
| `volume_catalog` | Folders, scan state, the DICOM tree (patients numbered), every volume (kind, format, dimensions, spacing), selected and previewed volume with its slice and window, image preparation (operation, running, last result) (`search`, `limit`) |
| `volume_info` | One volume (`volume`): grid and voxel → RAS affine, orientation, data type, warnings, files, header fields by group |
| `volume_do` | Volume's actions: select a volume (Segmentation then shows it), slice, window, add / remove a folder, rescan, cancel the scan, prepare an image (N4, denoise, resample, crop, reorient, register, subtract, normalise, vesselness, slab, time points, phase contrast, de-identified export), cancel it |
| `segmentation_state` | The volume shown (grid, value range, window), crosshair voxel / RAS / value, segments with voxel counts and volumes in mL, effect and every effect's settings (with Otsu's threshold), masking, undo, display, running operation, message, a waiting volume-switch prompt |
| `segmentation_do` | Segmentation's actions: add, select, rename / recolour / show / remove segments; threshold (or its preview), paint and erase strokes, flood fill, islands, margin, smoothing, logical operations; apply, cancel; masking; undo, redo; make a mesh for the Mesh tab; save (.seg.nrrd, .nii.gz); move the crosshair; window, opacity and views; answer the volume-switch prompt |

Lengths are in millimetres. Volume numbers stay valid until the folders are scanned again.
Mesh objects are addressed by their id, which never changes, and open boundaries by the number
`mesh_object` gives them. Segmentation positions are RAS millimetres, or voxel indices with
`in_voxels`, and segments are named by label number or name.

## How agents work the app

Agents read state and act through typed calls first. Screenshots and panel reads are a fallback
for what the typed calls do not cover.

1. **State.** Every Mesh tool reports what it holds and shows (`mesh_tool_state`): the points
   picked (a stent's ends with their centres, axes and diameters), measurements, previews and
   their checks, warnings and errors, and what to do next. `segmentation_state` and
   `volume_catalog` do the same for the image workspaces.
2. **Settings.** Every tool with settings takes them as typed JSON: open it with `settings`,
   or change them with `set`. Only the keys given change, and a misspelt key is an error that
   lists the valid ones. Each setting's meaning and range come from the source's doc comments,
   through the schema in `mesh_tools` and `mesh_tool_state`. A preview recomputes; `app_wait`,
   then `mesh_tool_state` shows the result and `accept` applies it.
3. **Actions.** What a tool's panel buttons and clicks do is offered as named actions with
   typed arguments, listed in `mesh_tool_state` and run with `mesh_tool` run. Examples:
   `set_end` (a stent's end at a point), Trim's `cut_at`, Measure's `measure_between`,
   Inspector's `repair_all`, the CFD check's `check`, Boundary patches' `rename`. The image
   workspaces' actions are the tools `volume_do` and `segmentation_do`, whose descriptions list
   every action with its arguments.
4. **Clicks.** `mesh_tool` click with a point in mm clicks the visible surface nearest that
   point, as a user would. If the point is hidden from the current view, the view first turns
   to face it, so you see where the click went.
5. **Panels.** Anything else: read the panel with `app_ui` and act with `app_ui_action`. These
   are the same widgets you use, read from egui's accessibility tree. Buttons that open a file
   dialog wait for you to answer it.

Work that takes time finishes after a call returns: computing previews, loading and saving,
and clicks, which land on the next frames. Replies say when something is still pending, and
`app_wait` waits for it.

## Privacy

Whatever an agent reads is sent to its provider, so agents get less than the window shows:

- Patient names, IDs, birth dates, sex, age and accession numbers are always masked (`•••`),
  whatever *Hide identifiers* is set to. Patients are numbered, never named.
- Dates are cut to the year (the HIPAA Safe Harbor rule counts exact dates of care as
  identifiers).
- The parts of patients' names and IDs found in the Volume catalog are masked in every text an
  agent receives. That includes mesh object names (which often come from file names), panel
  text and the log.
- File paths in the log and in error messages are replaced with `<path>`.
- Screenshots and panel reads show the window as it is. They are therefore refused while the
  Volume or Segmentation tab is shown, unless *Hide identifiers* is on. The Mesh tab is always
  allowed.
- The typed Volume and Segmentation tools work on every tab, whatever *Hide identifiers* is
  set to, since what they return is masked as above. Segmentation never names the volume.

Masking names in text relies on the DICOM patients the Volume workspace has found. A name in a
file that is not DICOM and was not matched to a DICOM patient is not known, so it is not
masked.

## Security

- Off by default. When on, Polymix listens on 127.0.0.1 only.
- Requests from web pages are always refused (anything with an `Origin` header), and so is any
  `Host` that is not a loopback name. A site open in a browser therefore cannot reach the
  server, even by DNS rebinding.
- With *Require a token* on (the default), every request needs the bearer token. The token is
  kept in the data folder (`agent-token`, readable by this user only). It keeps out other user
  accounts on the same computer, and sandboxed apps that may connect to the network but not
  read your files. Programs running as you can read the token file, as they can read your
  images, so on a single-user Mac turning it off costs little. Without it, the server works
  like most local MCP servers: any program on this computer can connect.
- Changes need a second, separate permission, and agents cannot grant it to themselves.

## How it works

`polymix-mcp` runs the server on its own thread with the official Rust SDK
([`rmcp`](https://github.com/modelcontextprotocol/rust-sdk)). It is stateless and speaks both
the 2026-07-28 protocol and the older ones that start with a handshake. The server holds no
app state. A tool call is queued for the UI thread, which takes it between frames (in the
shell's `logic`) and hands it to the workspace that owns the tool through the crash guard.
Tools are tagged read, change or destructive, and MCP clients use these hints to decide what
to ask you before running.

- **Reading.** Each workspace lists its tools (`Workspace::agent_tools`) and answers calls
  (`Workspace::agent_call`). A slow answer is `AgentReply::Later`: the workspace snapshots
  what it needs (meshes are `Arc`s, so the snapshot costs nothing), and the work runs on the
  server's worker threads, so the UI never waits for an agent.
- **Mesh commands.** `MeshCommand` (`app/command.rs`) is the one list of what the Mesh
  workspace does without the mouse. Menus, hotkeys, test scripts (`POLYMIX_SCRIPT`) and agents
  all run commands through `MeshWorkspace::execute`, so a command behaves the same whoever
  asks. The agent tools turn their arguments into commands. While an agent's commands (and any
  computing, loading or clicks they started) are under way, `History::by_agent` labels the
  undo steps.
- **Tool API.** Every Mesh tool implements `Tool::state` (everything it holds and shows, as
  JSON), `Tool::settings` / `settings_schema` / `set_settings` (its settings types derive
  `serde` and `schemars`, so doc comments become the schema's descriptions), and
  `Tool::actions` / `act`. An `Action` (`polymix_workspace::agent`) is a name, a purpose and
  the JSON Schema of its arguments; the same description lists it to agents and checks each
  call (`Action::check`). A test opens every tool in the catalog and checks that its settings,
  schema and actions agree.
- **Workspace actions.** A workspace lists its actions (`Workspace::agent_actions`) and does
  them (`agent_act`). The shell turns them into the tool `<workspace>_do`
  (`agent::do_tool`), whose description is generated from the actions' signatures, purposes
  and arguments, and checks each call against them before handing it over. A new action
  reaches agents without other code.
- **Panels.** `ui_access.rs` turns egui's AccessKit support on while agents use it. A plugin
  keeps the last frame's tree, and actions go back in as `Event::AccessKitActionRequest` with
  the next frame's input.

A call the UI thread does not take within 30 s (a dialog is open, or the window is minimized)
gets an error saying so.

## Limits

- Segmentation's undo steps are not labelled; its status message says "Agent: …" instead.
- The image preparation options are keys and values as the panel names them (listed in
  `volume_do`'s description), not a JSON Schema.
- *Hide identifiers* and the Volume browser's filters are left to the user.
- `app_ui` sees the shown tab only, plus open windows. Painted views (the 3D viewport, slice
  views) are not widgets, so `app_screenshot` and the read tools cover them.
- Nothing is pushed to agents: they ask again (or `app_wait`) to see changes.
- Long operations block nothing, but an agent's single wait is at most 25 s, so it waits again
  for longer work.
