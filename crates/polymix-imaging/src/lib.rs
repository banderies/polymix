//! Image views shared by the Volume and Segmentation workspaces: the mouse controls every
//! image view uses ([`nav`]), one plane of a volume ([`slice_view`]), the 3D view ([`render3d`])
//! and the histogram shown while the window / level is dragged ([`histogram`]).

pub mod histogram;
pub mod nav;
pub mod slice_view;
pub mod render3d;

pub use nav::{drag_window, Nav, NavEvents};
pub use render3d::{Mode3d, VolumeView3d};
pub use slice_view::{plane_colour, LabelOverlay, SliceOptions, SliceResponse, SliceView, ViewState};
