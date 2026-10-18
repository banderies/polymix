//! Voxel grids in patient space: size, spacing and the voxel → world (RAS, millimetres)
//! transform, with orientation codes ("RAS", "LPS", …) as nibabel's `aff2axcodes` gives them.

use glam::{DMat3, DMat4, DVec3, DVec4};
use serde::{Deserialize, Serialize};

/// A voxel grid placed in the world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    /// Voxels along i (columns), j (rows), k (slices), and the number of volumes or
    /// components (t), which is 1 for a plain 3D volume.
    pub dims: [usize; 4],
    /// Voxel size along i, j, k in millimetres.
    pub spacing: [f64; 3],
    /// Voxel index (i, j, k, 1) → RAS millimetres, as rows of a 4×4 matrix.
    pub affine: [[f64; 4]; 4],
}

impl Geometry {
    /// A grid from its spacing, axis directions (unit vectors in RAS) and the RAS position of
    /// voxel (0, 0, 0).
    pub fn from_axes(dims: [usize; 4], spacing: [f64; 3], axes: [DVec3; 3], origin: DVec3) -> Self {
        let m = DMat4::from_cols(
            (axes[0] * spacing[0]).extend(0.0),
            (axes[1] * spacing[1]).extend(0.0),
            (axes[2] * spacing[2]).extend(0.0),
            origin.extend(1.0),
        );
        Self { dims, spacing, affine: rows(&m) }
    }

    /// A grid from a voxel → RAS matrix; the spacing is the length of its columns.
    pub fn from_affine(dims: [usize; 4], m: DMat4) -> Self {
        let spacing = [m.x_axis.truncate().length(), m.y_axis.truncate().length(), m.z_axis.truncate().length()];
        Self { dims, spacing, affine: rows(&m) }
    }

    pub fn matrix(&self) -> DMat4 {
        let r = &self.affine;
        DMat4::from_cols(
            DVec4::new(r[0][0], r[1][0], r[2][0], r[3][0]),
            DVec4::new(r[0][1], r[1][1], r[2][1], r[3][1]),
            DVec4::new(r[0][2], r[1][2], r[2][2], r[3][2]),
            DVec4::new(r[0][3], r[1][3], r[2][3], r[3][3]),
        )
    }

    /// Unit direction (RAS) of each voxel axis.
    pub fn directions(&self) -> [DVec3; 3] {
        let m = self.matrix();
        [m.x_axis.truncate().normalize_or_zero(), m.y_axis.truncate().normalize_or_zero(), m.z_axis.truncate().normalize_or_zero()]
    }

    /// RAS position of voxel (i, j, k).
    pub fn world(&self, ijk: DVec3) -> DVec3 {
        self.matrix().transform_point3(ijk)
    }

    /// Orientation code: for each voxel axis, the patient direction it points to ("RAS"
    /// means i runs to the patient's right… no: i runs toward Right, j toward Anterior, k
    /// toward Superior).
    pub fn orientation(&self) -> String {
        axcodes(self.directions())
    }

    /// Largest angle (degrees) between a voxel axis and the nearest world axis: 0 for a grid
    /// aligned with the patient axes.
    pub fn obliquity_deg(&self) -> f64 {
        self.directions().iter().map(|d| d.abs().max_element().clamp(0.0, 1.0).acos().to_degrees()).fold(0.0, f64::max)
    }

    /// Whether the voxel axes form a left-handed frame (common for DICOM and NIfTI alike).
    pub fn left_handed(&self) -> bool {
        let [a, b, c] = self.directions();
        DMat3::from_cols(a, b, c).determinant() < 0.0
    }

    /// Size of the grid in millimetres along i, j, k.
    pub fn extent_mm(&self) -> [f64; 3] {
        [self.dims[0] as f64 * self.spacing[0], self.dims[1] as f64 * self.spacing[1], self.dims[2] as f64 * self.spacing[2]]
    }

    pub fn voxel_count(&self) -> usize {
        self.dims.iter().product()
    }
}

fn rows(m: &DMat4) -> [[f64; 4]; 4] {
    let t = m.transpose();
    [t.x_axis.to_array(), t.y_axis.to_array(), t.z_axis.to_array(), t.w_axis.to_array()]
}

/// Orientation letters for three axis directions in RAS: each axis gets the world axis it is
/// closest to (each used once), lettered by the direction it points.
pub fn axcodes(dirs: [DVec3; 3]) -> String {
    let mut used = [false; 3];
    let mut out = [' '; 3];
    // Assign the most decisive axes first, so near-ties do not steal a better match.
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&a, &b| dirs[b].abs().max_element().total_cmp(&dirs[a].abs().max_element()));
    for j in order {
        let d = dirs[j];
        let Some(i) = (0..3).filter(|&i| !used[i]).max_by(|&x, &y| d[x].abs().total_cmp(&d[y].abs())) else { continue };
        used[i] = true;
        out[j] = if d[i] >= 0.0 { ['R', 'A', 'S'][i] } else { ['L', 'P', 'I'][i] };
    }
    out.iter().collect()
}

/// DICOM patient coordinates (LPS) → RAS: x and y change sign.
pub fn lps_to_ras(v: DVec3) -> DVec3 {
    DVec3::new(-v.x, -v.y, v.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_codes() {
        let id = [DVec3::X, DVec3::Y, DVec3::Z];
        assert_eq!(axcodes(id), "RAS");
        assert_eq!(axcodes([-DVec3::X, -DVec3::Y, DVec3::Z]), "LPS");
        // A sagittal stack: i along A→P, j along S→I, k along L→R.
        assert_eq!(axcodes([-DVec3::Y, -DVec3::Z, DVec3::X]), "PIR");
        // Slightly oblique axes keep their nearest letter.
        let t = DVec3::new(1.0, 0.1, 0.0).normalize();
        assert_eq!(axcodes([t, DVec3::new(-0.1, 1.0, 0.0).normalize(), DVec3::Z]), "RAS");
    }

    #[test]
    fn affine_round_trip_and_measures() {
        let g = Geometry::from_axes([256, 240, 176, 1], [0.9, 0.9, 1.2], [-DVec3::X, -DVec3::Y, DVec3::Z], DVec3::new(10.0, 20.0, -30.0));
        assert_eq!(g.orientation(), "LPS");
        assert!(g.obliquity_deg() < 1e-9);
        assert!(!g.left_handed());
        let p = g.world(DVec3::new(1.0, 2.0, 3.0));
        assert!((p - DVec3::new(10.0 - 0.9, 20.0 - 1.8, -30.0 + 3.6)).length() < 1e-12);
        let h = Geometry::from_affine(g.dims, g.matrix());
        assert!((h.spacing[2] - 1.2).abs() < 1e-12);
        assert_eq!(h, g);
        let tilted = Geometry::from_axes([4, 4, 4, 1], [1.0; 3], [DVec3::X, DVec3::new(0.0, 0.94, 0.342).normalize(), DVec3::new(0.0, -0.342, 0.94).normalize()], DVec3::ZERO);
        assert!((tilted.obliquity_deg() - 20.0).abs() < 0.1);
    }
}
