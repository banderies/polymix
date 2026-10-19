# Agent access (MCP)

AI agents (Claude Code, Claude Desktop, Cursor and other [Model Context
Protocol](https://modelcontextprotocol.io) clients) can read the session open in Polymix: the
mesh scene, the image catalog, the segmentation, the log and a screenshot of the window. It is
meant for debugging Polymix with an agent and for asking an agent about the data on screen.

Agents only read. Changing the scene through agents is the next step (see the roadmap, M17).

## Turning it on

Click **Agents** at the right end of the tab bar and tick *Let AI agents read this session*.
The button's dot turns green while agents can connect and blue for a moment after each call.
Polymix remembers the setting.

To connect Claude Code, press *Copy command* and run it once in a terminal:

```sh
claude mcp add --transport http polymix http://127.0.0.1:47390/mcp --header "Authorization: Bearer <token>"
```

Other clients need the URL and the token (*Copy URL*, *Copy token*), sent as
`Authorization: Bearer <token>`. *New token* replaces the token, and clients set up with the
old one stop working.

For development and scripted runs, `POLYMIX_MCP=1` turns access on for one session without
saving it, and `POLYMIX_MCP_PORT` picks the port.

## Tools

| Tool | What it returns |
|---|---|
| `app_status` | Version, the workspace tab shown, each workspace running / busy / stopped (with the error), window size |
| `app_screenshot` | PNG of the window (`max_size`, default 1568 px on the longest side) |
| `app_log` | Recent log messages (`lines`, `level`): Polymix's from info up, libraries' warnings and errors |
| `mesh_scene` | Every object (id, name, counts, open edges, bounds, selection, face groups), active / target object, the tool stack and whether it is computing, status line, undo / redo steps, background jobs, camera, display options |
| `mesh_object` | One object (`object`: id or name, default the active one): area, enclosed volume and orientation, edge lengths, validity, shells, open boundaries with perimeters and centres, defects, face groups and patch names, pivots |
| `mesh_selection` | The selected triangles: count, area, bounds, separate regions, border length, face groups |
| `volume_catalog` | Folders, scan state, the DICOM tree (patients numbered), every volume (kind, format, dimensions, spacing), selected and previewed volume (`search`, `limit`) |
| `volume_info` | One volume (`volume`): grid and voxel → RAS affine, orientation, data type, warnings, files, header fields by group |
| `segmentation_state` | The volume shown (grid, value range, window), crosshair voxel / RAS / value, segments with voxel counts and volumes in mL, effect, masking, running operation, message |

Lengths are in millimetres. Volume numbers stay valid until the folders are scanned again.
Mesh objects are addressed by their id, which never changes.

## Privacy

Whatever an agent reads is sent to its provider, so agents get less than the window shows:

- Patient names, IDs, birth dates, sex, age and accession numbers are always masked (`•••`),
  whatever *Hide identifiers* is set to. Patients are numbered, never named.
- Dates are cut to the year (the HIPAA Safe Harbor rule counts exact dates of care as
  identifiers).
- The parts of patients' names and IDs found in the Volume catalog are masked in every text
  an agent receives, including mesh object names (which often come from file names) and the
  log.
- File paths in the log and in error messages are replaced with `<path>`.
- Screenshots show the window as it is, so they are refused while the Volume or Segmentation
  tab is shown unless *Hide identifiers* is on. Mesh tab screenshots are always allowed.

Masking names in text relies on the DICOM patients the Volume workspace has found. A name in a
file that is not DICOM and was not matched to a DICOM patient is not known, so it is not
masked.

## Security

- Off by default. When on, Polymix listens on 127.0.0.1 only.
- Every request needs the bearer token. The token is kept in the data folder (`agent-token`,
  readable by this user only).
- Requests from web pages are refused (anything with an `Origin` header), and so is any `Host`
  that is not a loopback name. A site open in a browser therefore cannot reach the server,
  even by DNS rebinding.

## How it works

`polymix-mcp` runs the server on its own thread with the official Rust SDK
([`rmcp`](https://github.com/modelcontextprotocol/rust-sdk)). It is stateless and speaks both
the 2026-07-28 protocol and the older ones that start with a handshake. The server holds no
app state. A tool call is queued for the UI thread, which takes it between frames (in the
shell's `logic`) and hands it to the workspace that owns the tool through the crash guard.

Each workspace lists its tools (`Workspace::agent_tools`) and answers calls
(`Workspace::agent_call`). An answer that would take more than a moment is `AgentReply::Later`.
The workspace snapshots what it needs (meshes are `Arc`s, so the snapshot costs nothing), and
the work runs on the server's worker threads, so the UI never waits for an agent. A call that
the UI thread does not take within 30 s (a dialog is open, or the window is minimized) gets an
error saying so. A screenshot is taken from the next frame and encoded off the UI thread.

## Limits

- Read only: agents cannot select, run tools, undo or save.
- Screenshots show the shown tab only, and agents cannot switch tabs.
- Tool panel settings (e.g. the Remesh panel's values) are not exposed, only the tool's name
  and progress.
- Nothing is pushed to agents: they ask again to see changes.
