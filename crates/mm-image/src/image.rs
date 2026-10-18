//! Conventions shared by the image operations: images are `LoadedVolume`s (values in voxel
//! order, i fastest, with their grid in RAS millimetres); operations report progress through
//! [`Progress`] and return new values on the same grid, or a new image when the grid changes.

use mm_volume::geometry::Geometry;
use mm_volume::load::LoadedVolume;

/// Progress callback: gets the fraction done (0–1), may be called from any thread; returning
/// `false` asks the operation to stop (it then returns an error).
pub type Progress<'a> = &'a (dyn Fn(f32) -> bool + Sync);

/// A progress callback that never cancels.
pub fn no_progress() -> Progress<'static> {
    &|_| true
}

/// An image on the grid of `src` with new values.
pub fn with_values(src: &LoadedVolume, name: &str, values: Vec<f32>) -> LoadedVolume {
    let mut out = LoadedVolume::from_values(name, src.dims, values, src.geometry.clone());
    out.invert = src.invert;
    out
}

/// An image on a new grid (`geometry.dims` gives its size).
pub fn on_grid(name: &str, geometry: Geometry, values: Vec<f32>) -> LoadedVolume {
    let dims = [geometry.dims[0], geometry.dims[1], geometry.dims[2]];
    let mut g = geometry;
    g.dims[3] = 1;
    LoadedVolume::from_values(name, dims, values, g)
}

/// Fail with "cancelled" when `progress` says so.
pub fn check(progress: Progress, f: f32) -> anyhow::Result<()> {
    if progress(f) { Ok(()) } else { anyhow::bail!("cancelled") }
}
