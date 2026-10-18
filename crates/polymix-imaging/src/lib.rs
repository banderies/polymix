//! Image views shared by the Volume and Segmentation workspaces: the mouse controls every
//! image view uses ([`nav`]), and more to come (slice views, 3D rendering).

pub mod nav;

pub use nav::{drag_window, Nav, NavEvents};
