// SPDX-License-Identifier: AGPL-3.0-only
//! Reverse engineering from scans for Polymix (no GUI): what turns a scanned mesh back into
//! a parametric model. Sections through the mesh come back as chains of points; the chains
//! are cut into lines and arcs; the pieces go into a sketch whose solver meets every
//! relation and dimension exactly and settles the rest onto the slice points; the sketch's
//! closed regions are what the solid features (in the app, on the SolidCraft kernel) are
//! built from.
//!
//! This crate is a Rust port of the reverse-engineering core of ScanRuler
//! (<https://github.com/CNCKitchen/scanruler>), © Stefan Hermann (CNC Kitchen), licensed
//! AGPL-3.0-only, and is itself AGPL-3.0-only. Each module names the ScanRuler file it was
//! ported from.

pub mod accept;
pub mod adjacency;
pub mod auto_constrain;
pub mod axis_align;
pub mod body_move;
pub mod box_select;
pub mod build;
pub mod chain;
pub mod chain_outline;
pub mod chamfer_sides;
pub mod compare;
pub mod corners;
pub mod curved_edge;
pub mod datums;
pub mod dim_layout;
pub mod dim_strokes;
pub mod display_units;
pub mod drag;
pub mod draw;
pub mod expression;
pub mod fillet_corner;
pub mod fillet_estimate;
pub mod fillet_num;
pub mod fillet_read;
pub mod fit2d;
pub mod fit3d;
pub mod flat_spline;
pub mod frame;
pub mod height_surface;
pub mod inserts;
pub mod intent;
pub mod js_fmt;
pub mod js_number;
pub mod length_fields;
pub mod linalg;
pub mod loft_fold;
pub mod loop_piece;
pub mod measure_elements;
pub mod mesh_graph;
pub mod mirror_relation;
pub mod model;
pub mod network;
pub mod network_body;
pub mod network_cells;
pub mod network_creases;
pub mod network_curve;
pub mod network_draft;
pub mod network_feature;
pub mod network_fit;
pub mod network_map;
pub mod network_seed;
pub mod network_spline;
pub mod network_split;
pub mod nominal;
pub mod outline;
pub mod param_slots;
pub mod parameters;
pub mod part_parameters;
pub mod pattern_copies;
pub mod pick_query;
pub mod principal_frame;
pub mod project;
pub mod regions;
pub mod round;
pub mod round_corner;
pub mod scan_path;
pub mod section;
pub mod section_skin;
pub mod segment;
pub mod shadow;
pub mod shadow_pieces;
pub mod share_fit;
pub mod silhouette;
pub mod simplify;
pub mod sketch;
pub mod sketch_draw;
pub mod sketch_fit;
pub mod sketch_image;
pub mod sketch_near_miss;
pub mod sketch_shapes;
pub mod sketch_spline;
pub mod slice;
pub mod snap;
// ---- DRAFT ----
pub mod draft_fit;
pub mod snap_edit;
pub mod snap_face;
// ---- DRAFT ----
pub mod solver;
pub mod sparse;
pub mod spline_solve;
pub mod stations;
pub mod surface_fit;
pub mod surface_region;
pub mod sweep_route;
pub mod symmetry_plane;
pub mod taper;
pub mod teeth;
pub mod thread;
pub mod thread_cut;
pub mod thread_read;
pub mod thread_scan;
pub mod topo;
pub mod tube_trace;
pub mod units;
// ---- EXT ----
pub mod extrude_draft;
pub mod extrude_grips;
pub mod extrude_scan;
pub mod profile_axis;

// ---- PICKLIST ----
pub mod body_clearance;
pub mod body_folders;
pub mod body_style;
pub mod selection_info;

#[cfg(test)]
pub(crate) mod fillet_testkit;
#[cfg(test)]
pub(crate) mod network_meshes;

// ---- VIEWCMP ----
pub mod audit;
pub mod body_check;
pub mod compare_detail;
pub mod edge_lines;
pub mod hotspots;
pub mod section_layer;
pub mod section_view;
pub mod volume_compare;

pub use fit2d::FitError;
pub use segment::{segment_chain, ChainSegment, Piece, PieceKind, SegmentOptions, SegmentedChain};

// ---- RND ----
pub mod face_group;
pub mod rounds;

// ---- BOOL ----
pub mod symmetry_search;
