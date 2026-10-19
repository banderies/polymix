//! The Segmentation workspace's agent tool: what it shows and the segments in it.
//!
//! The volume is not named here (its name can carry a patient's): `volume_catalog` says which
//! volume is selected in the Volume workspace, which is the one shown here.

use crate::{SegmentationWorkspace, PLANES};
use polymix_workspace::agent::{round, vec3, AgentReply, AgentTool};
use serde_json::{json, Value};

pub fn tools() -> Vec<AgentTool> {
    vec![AgentTool::new(
        "segmentation_state",
        "The Segmentation workspace: the volume shown (grid, spacing, orientation, value range, window), the crosshair (voxel, RAS mm, value), the segments (label, name, colour, visibility, voxel count and volume in mL), the selected segment and effect, masking, and any loading, running operation or message. The volume shown is the one selected in the Volume workspace (see volume_catalog).",
        json!({}),
    )]
}

pub fn call(ws: &mut SegmentationWorkspace, tool: &str, _args: &Value) -> AgentReply {
    match tool {
        "segmentation_state" => AgentReply::Now(Ok(state(ws))),
        _ => AgentReply::error(format!("the Segmentation workspace has no tool {tool}")),
    }
}

fn state(ws: &mut SegmentationWorkspace) -> Value {
    let volume = ws.volume.as_ref().map(|v| {
        let g = &v.geometry;
        json!({
            "dims": v.dims,
            "spacing_mm": g.spacing.map(|x| round(x, 4)),
            "orientation": g.orientation(),
            "value_range": [v.range.0, v.range.1],
            "auto_window": {"centre": round(v.auto_window.0, 3), "width": round(v.auto_window.1, 3)},
            "volumes_in_set": ws.source.as_ref().map_or(1, |s| s.volumes),
        })
    });
    let cursor = ws.volume.as_ref().zip(ws.state.as_ref()).map(|(v, st)| {
        let c = st.cursor;
        let (i, j, k) = (c.x.round().max(0.0) as usize, c.y.round().max(0.0) as usize, c.z.round().max(0.0) as usize);
        let inside = i < v.dims[0] && j < v.dims[1] && k < v.dims[2];
        let ras = v.geometry.world(c);
        json!({
            "voxel": [i, j, k],
            "ras_mm": vec3(ras.to_array(), 3),
            "value": inside.then(|| v.get(i, j, k)),
            "label": ws.seg.as_ref().filter(|_| inside).map(|s| s.labels[v.index(i, j, k)]),
        })
    });
    let voxel_ml = ws.volume.as_ref().map_or(0.0, |v| v.geometry.spacing.iter().product::<f64>() / 1000.0);
    let segments = ws.seg.as_mut().map(|s| {
        let counts = s.counts().to_vec();
        json!({
            "segments": s.segments.iter().map(|g| json!({
                "label": g.label,
                "name": g.name,
                "colour": format!("#{:02x}{:02x}{:02x}", g.colour.r(), g.colour.g(), g.colour.b()),
                "visible": g.visible,
                "voxels": counts[g.label as usize],
                "volume_ml": round(counts[g.label as usize] as f64 * voxel_ml, 3),
            })).collect::<Vec<_>>(),
            "selected_segment": s.selected_segment().map(|g| g.label),
            "modified": s.modified,
        })
    });
    let m = &ws.masking;
    json!({
        "volume": volume,
        "loading": ws.loading.as_ref().map(|l| round(l.progress() as f64, 2)),
        "error": ws.error.as_ref().map(|e| polymix_workspace::guard::scrub_paths(e)),
        "window": ws.state.as_ref().map(|st| json!({"centre": round(st.window.0, 3), "width": round(st.window.1, 3)})),
        "cursor": cursor,
        "segmentation": segments,
        "effect": ws.effect.map(|e| e.label()),
        "masking": {"editable_area": format!("{:?}", m.area), "intensity_range": m.intensity.map(|(a, b)| [a, b]), "overwrite": format!("{:?}", m.overwrite)},
        "running": ws.running.as_ref().map(|r| r.what.clone()),
        "message": ws.message.as_ref().map(|(text, error)| json!({"text": polymix_workspace::guard::scrub_paths(text), "error": error})),
        "maximized_view": ws.maximized.map(|k| match k {
            1 => "3d".to_string(),
            k => format!("{:?}", PLANES[[0, 0, 1, 2][k.min(3)]]).to_lowercase(),
        }),
        "meshes_waiting_for_mesh_tab": ws.outbox.len(),
        "note": ws.volume.is_none().then_some("No volume loaded: the Segmentation tab loads the volume selected in the Volume workspace when the tab is shown."),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmentation::Segmentation;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;
    use mm_volume::load::LoadedVolume;
    use polymix_imaging::ViewState;
    use std::sync::Arc;

    #[test]
    fn segments_with_their_volumes() {
        let mut ws = SegmentationWorkspace::new(&egui::Context::default(), None);
        assert_eq!(state(&mut ws)["volume"], Value::Null);
        // 4 × 4 × 4 voxels of 0.5 × 0.5 × 2 mm (0.5 mm³ each), values 0..64.
        let g = Geometry::from_axes([4, 4, 4, 1], [0.5, 0.5, 2.0], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("t", [4, 4, 4], (0..64).map(|v| v as f32).collect(), g);
        ws.state = Some(ViewState::new(&vol));
        let mut seg = Segmentation::new([4, 4, 4]);
        seg.add_segment(Some("aneurysm".into()));
        for v in &mut seg.labels[..10] {
            *v = 1;
        }
        seg.version += 1;
        ws.seg = Some(seg);
        ws.volume = Some(Arc::new(vol));
        let s = state(&mut ws);
        assert_eq!(s["volume"]["dims"], json!([4, 4, 4]));
        let a = &s["segmentation"]["segments"][0];
        assert_eq!(a["name"], "aneurysm");
        assert_eq!(a["voxels"], 10);
        assert_eq!(a["volume_ml"], 0.005);
        // The cursor starts in the middle voxel (1, 1, 1): value 1 + 4 + 16.
        assert_eq!(s["cursor"]["voxel"], json!([1, 1, 1]));
        assert_eq!(s["cursor"]["value"], 21.0);
        assert_eq!(s["cursor"]["ras_mm"], json!([0.5, 0.5, 2.0]));
    }
}
