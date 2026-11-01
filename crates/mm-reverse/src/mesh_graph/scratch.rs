// SPDX-License-Identifier: AGPL-3.0-only
//! Visited stamps for a breadth-first walk over a mesh, so a walk never has to clear a
//! vertex-sized array between runs: a vertex is visited when its stamp equals the current
//! generation.
//!
//! Ported from ScanRuler (`src/core/geometry/scratch.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. ScanRuler keeps one buffer per mesh graph in a weak map and hands out the
//! next generation of it to every walk; here a walk owns its `Stamps`, and a walk that runs in
//! rounds (the region growing) takes the next generation of the same buffer each round, which
//! is what ScanRuler's shared buffer amounts to.

/// A stamp per vertex and the generation that currently marks a vertex as visited.
#[derive(Debug, Clone)]
pub struct Stamps {
    pub stamp: Vec<i32>,
    pub gen: i32,
}

impl Stamps {
    /// Stamps over `size` vertices, at generation 1: nothing visited yet.
    pub fn new(size: usize) -> Self {
        Self { stamp: vec![0; size], gen: 1 }
    }

    /// Move on to a fresh generation, so nothing reads as visited, and return it. The stamps
    /// are cleared only when the generation would run past `i32::MAX`.
    pub fn next_gen(&mut self) -> i32 {
        self.gen += 1;
        if self.gen == i32::MAX {
            self.stamp.fill(0);
            self.gen = 1;
        }
        self.gen
    }

    /// Whether vertex `v` is marked in the current generation.
    #[inline]
    pub fn seen(&self, v: u32) -> bool {
        self.stamp[v as usize] == self.gen
    }

    /// Mark vertex `v` in the current generation.
    #[inline]
    pub fn mark(&mut self, v: u32) {
        self.stamp[v as usize] = self.gen;
    }
}
