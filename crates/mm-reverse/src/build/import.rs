// SPDX-License-Identifier: AGPL-3.0-only
//! The Import feature's build: a STEP file brought in as bodies of their own — a body per solid
//! in the file, named after its part where the file names it; a part of faces alone a surface
//! body. The file's text is the part's
//! (`Part::step_files`, by the key the feature holds). Ported from ScanRuler
//! (`plugins/reverse/core/build.ts` the import step, `kernelStep.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.

use solidcraft_kernel as kernel;

use super::{BodyState, Builder};
use crate::model::{Feature, ImportFeature, Part};

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &ImportFeature) -> Result<(), String> {
    // A box opened on no file yet.
    if x.file.key.is_empty() {
        return Err("Choose the STEP file to bring in.".into());
    }
    let file = part.step_files.get(&x.file.key).ok_or_else(|| format!("The STEP file {} is not in the part — import it again.", x.file.name))?;
    let read = kernel::step_import(&file.text).map_err(|e| format!("The STEP file {} could not be read: {e}", x.file.name))?;
    if read.bodies.is_empty() {
        return Err("The STEP file holds no solid and no surface.".into());
    }
    let mut ids = Vec::new();
    for ib in read.bodies {
        let id = b.fresh_id()?;
        let own = b.own_origins(&ib.body);
        let label = (!ib.name.trim().is_empty()).then(|| ib.name.clone());
        // A part of faces with no solid is a surface body (never joined or cut).
        let surface = ib.body.is_surface();
        let placed = if surface {
            let mut state = BodyState::new(ib.body, own);
            state.surface = true;
            b.set_body(id, state);
            vec![id]
        } else {
            b.settle(id, ib.body, own, false)?
        };
        for at in placed {
            if let Some(s) = b.body_state_mut(at) {
                s.label = label.clone();
            }
            ids.push(at);
        }
    }
    b.touched(ids);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::build::build;
    use crate::model::{FeatureKind, ImportFeature, Part, StepFileRef};

    // reImportMove.test.ts: fails by name when its file was never handed over; and a box
    // opened on no file yet asks for one.
    #[test]
    fn fails_by_name_with_no_file_or_one_never_handed_over() {
        let mut part = Part::default();
        let none = part.add(FeatureKind::Import(ImportFeature { file: StepFileRef { key: String::new(), name: String::new(), size: 0 } }));
        let gone = part.add(FeatureKind::Import(ImportFeature { file: StepFileRef { key: "9-abc".into(), name: "bracket.step".into(), size: 9 } }));
        let built = build(&part);
        assert_eq!(built.errors.get(&none).map(String::as_str), Some("Choose the STEP file to bring in."));
        assert_eq!(built.errors.get(&gone).map(String::as_str), Some("The STEP file bracket.step is not in the part — import it again."));
    }
}
