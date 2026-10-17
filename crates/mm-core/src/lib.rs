//! `mm-core`: the geometry kernel of Polymix.
//!
//! * [`mesh::Mesh`] — dynamic triangle mesh with explicit edge topology and local edits.
//! * [`bvh::Bvh`] — triangle BVH for ray casts, nearest-point and range queries.
//! * [`topology`] — components, boundary loops, defect detection, orientation repair.
//! * [`selection::TriSelection`] — triangle selections and region operations.
//! * [`primitives`] — procedural shapes.
//! * [`shading`] — display normals that keep sharp edges crisp.

pub mod bvh;
pub mod geom;
pub mod intersect;
pub mod measure;
pub mod mesh;
pub mod primitives;
pub mod selection;
pub mod shading;
pub mod topology;

pub use bvh::Bvh;
pub use geom::{Aabb, Plane, Ray};
pub use glam::{DMat3, DMat4, DQuat, DVec2, DVec3};
pub use mesh::{Mesh, MeshError, MeshResult, INVALID};
pub use selection::{BrushShape, TriSelection};

#[cfg(test)]
mod mesh_tests;
