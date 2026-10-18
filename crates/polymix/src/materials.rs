//! Material library (the Shaders palette): lighting presets for the mesh shader.

/// Lighting parameters understood by `mesh.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    pub name: &'static str,
    /// Colour the material gives the object (`None` keeps the object's own colour).
    pub color: Option<[f32; 3]>,
    /// Opacity the material gives the object (`None` keeps it).
    pub alpha: Option<f32>,
    /// Specular highlight strength and sharpness.
    pub specular: f32,
    pub shininess: f32,
    /// Rim (grazing-angle) brightening.
    pub rim: f32,
    /// 0..1: highlights take the surface colour and a soft environment reflection appears.
    pub metal: f32,
    /// 0..1: light wraps around the shadow side (soft, translucent look).
    pub wrap: f32,
}

impl Material {
    /// Shader uniform data: (specular, shininess, rim, metal), (wrap, 0, 0, 0).
    pub fn uniform(&self) -> ([f32; 4], [f32; 4]) {
        ([self.specular, self.shininess, self.rim, self.metal], [self.wrap, 0.0, 0.0, 0.0])
    }
}

#[allow(clippy::too_many_arguments)]
const fn mat(name: &'static str, color: Option<[f32; 3]>, alpha: Option<f32>, specular: f32, shininess: f32, rim: f32, metal: f32, wrap: f32) -> Material {
    Material { name, color, alpha, specular, shininess, rim, metal, wrap }
}

/// The library; the first entry is the default.
pub const MATERIALS: [Material; 10] = [
    mat("Clay", None, None, 0.18, 40.0, 0.25, 0.0, 0.0),
    mat("Soft clay", None, None, 0.08, 20.0, 0.15, 0.0, 0.35),
    mat("Satin", None, None, 0.35, 30.0, 0.4, 0.0, 0.1),
    mat("Glossy", None, None, 0.7, 120.0, 0.3, 0.0, 0.0),
    mat("Ceramic", Some([0.92, 0.91, 0.88]), None, 0.5, 70.0, 0.2, 0.0, 0.15),
    mat("Tissue", Some([0.84, 0.46, 0.42]), None, 0.3, 25.0, 0.35, 0.0, 0.5),
    mat("Resin", Some([0.55, 0.75, 0.95]), Some(0.7), 0.6, 90.0, 0.6, 0.0, 0.6),
    mat("Gold", Some([0.86, 0.66, 0.27]), None, 1.0, 80.0, 0.2, 1.0, 0.0),
    mat("Silver", Some([0.8, 0.82, 0.85]), None, 1.0, 100.0, 0.2, 1.0, 0.0),
    mat("Copper", Some([0.86, 0.46, 0.3]), None, 1.0, 70.0, 0.2, 1.0, 0.0),
];
