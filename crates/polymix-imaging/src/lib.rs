//! Image views shared by the Volume and Segmentation workspaces: the mouse controls every
//! image view uses ([`nav`]), one plane of a volume ([`slice_view`]), the 3D view
//! ([`render3d`]), the histogram shown while the window / level is dragged ([`histogram`]),
//! colour maps, slabs, a second image over the first and layouts ([`layers`]), and rulers,
//! angles and regions measured in the slice views ([`measure`]), and planes turned away from
//! the voxel grid ([`oblique`]).

pub mod histogram;
pub mod layers;
pub mod measure;
pub mod nav;
pub mod oblique;
pub mod slice_view;
pub mod render3d;

pub use layers::{ColourMap, Compare, Fusion, Layout, Slab, SlabMode};
pub use measure::{MeasureTool, Measurements};
pub use nav::{drag_window, Nav, NavEvents};
pub use render3d::{Mode3d, VolumeView3d};
pub use slice_view::{plane_colour, LabelOverlay, SliceOptions, SliceResponse, SliceView, ViewMap, ViewState, PLANES};
