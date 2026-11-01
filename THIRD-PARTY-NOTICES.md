# Third-party notices

Polymix is licensed under the GNU Affero General Public License v3.0 only (see `LICENSE`).
It contains or links the following third-party work. Every other dependency is listed in
`Cargo.lock`; each crate's licence travels with its source.

## ScanRuler

The Reverse Engineer workspace (`crates/mm-reverse`, `crates/polymix/src/reverse`) is a Rust
port of the Reverse engineer workspace and the scan-fitting code of ScanRuler,
<https://github.com/CNCKitchen/scanruler>, Copyright © Stefan Hermann (CNC Kitchen), licensed
under the GNU Affero General Public License v3.0 only. Each ported file names the ScanRuler
file it was ported from. ScanRuler and CNC Kitchen are trademarks of their owner and are not
licensed here; Polymix is not affiliated with or endorsed by CNC Kitchen.

## SolidCraft

The Reverse Engineer workspace builds its solids on the SolidCraft crates
(`solidcraft-geom`, `solidcraft-sketch`, `solidcraft-kernel`),
<https://github.com/tangency-labs/solidcraft>, Copyright © 2026 SolidCraft contributors,
licensed under the MIT License or the Apache License, Version 2.0, at your option.

## truck

SolidCraft's kernel builds on the truck crates (truck-modeling, truck-shapeops,
truck-meshalgo, truck-stepio, truck-topology, truck-geometry and their dependencies),
<https://github.com/ricosjp/truck>, Copyright © RICOS Co. Ltd., licensed under the Apache
License, Version 2.0. truck-geometry, truck-shapeops, truck-stepio, truck-meshalgo and
matext4cgmath are used in the versions SolidCraft vendors with its modifications (see
SolidCraft's `vendor/README.md`).

## Liberation Sans

`solidcraft-sketch` embeds the Liberation Sans font (Liberation Fonts 2.x, Copyright © Red Hat,
Inc.; digitized data © Google), licensed under the SIL Open Font License, Version 1.1
(<https://openfontlicense.org>). It is bundled unmodified.
