//! Texel rectangles and dirty-tile tracking for region-limited rebakes.
//!
//! Compositing is per texel except for spatial filters (blur, sharpen) and
//! the height-to-normal gradient, so recompositing only the texels an edit
//! touched reproduces a full bake exactly. [`DirtyTexels`] records those
//! texels on a fixed tile grid and hands them back as a few rectangles.

/// Edge length, in texels, of the tiles [`DirtyTexels`] tracks.
pub const DIRTY_TILE_SIZE: u32 = 64;

/// A block of texels: columns `x..x + width`, rows `y..y + height` (row 0
/// is the top of the texture).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TexelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl TexelRect {
    /// Every texel of a `width` × `height` texture.
    pub const fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Number of texels covered.
    pub const fn len(self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Exclusive right column.
    pub const fn right(self) -> u32 {
        self.x + self.width
    }

    /// Exclusive bottom row.
    pub const fn bottom(self) -> u32 {
        self.y + self.height
    }

    pub const fn contains(self, x: u32, y: u32) -> bool {
        x >= self.x && x < self.right() && y >= self.y && y < self.bottom()
    }

    /// Texels in both rectangles, or `None` when they do not overlap.
    pub fn intersect(self, other: Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        (right > x && bottom > y).then(|| Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        })
    }

    /// The rectangle grown by `by` texels on every side, clipped to a
    /// `width` × `height` texture.
    pub fn grow(self, by: u32, width: u32, height: u32) -> Self {
        let x = self.x.saturating_sub(by);
        let y = self.y.saturating_sub(by);
        let right = self.right().saturating_add(by).min(width);
        let bottom = self.bottom().saturating_add(by).min(height);
        Self {
            x,
            y,
            width: right.saturating_sub(x),
            height: bottom.saturating_sub(y),
        }
    }

    /// Row-major index of texture texel `(x, y)` within this rectangle.
    pub(crate) const fn local_index(self, x: u32, y: u32) -> usize {
        (y - self.y) as usize * self.width as usize + (x - self.x) as usize
    }
}

/// Texels of one texture that changed since a known state, tracked per
/// [`DIRTY_TILE_SIZE`] tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirtyTexels {
    width: u32,
    height: u32,
    columns: u32,
    rows: u32,
    tiles: Vec<bool>,
}

impl DirtyTexels {
    /// Nothing changed yet on a `width` × `height` texture.
    pub fn new(width: u32, height: u32) -> Self {
        let columns = width.div_ceil(DIRTY_TILE_SIZE);
        let rows = height.div_ceil(DIRTY_TILE_SIZE);
        Self {
            width,
            height,
            columns,
            rows,
            tiles: vec![false; columns as usize * rows as usize],
        }
    }

    /// Size of the texture tracked.
    pub const fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn is_clean(&self) -> bool {
        !self.tiles.contains(&true)
    }

    /// Marks every tile `rect` touches.
    pub fn mark(&mut self, rect: TexelRect) {
        let Some(rect) = rect.intersect(TexelRect::full(self.width, self.height)) else {
            return;
        };
        let (first_column, last_column) =
            (rect.x / DIRTY_TILE_SIZE, (rect.right() - 1) / DIRTY_TILE_SIZE);
        let (first_row, last_row) = (rect.y / DIRTY_TILE_SIZE, (rect.bottom() - 1) / DIRTY_TILE_SIZE);
        for row in first_row..=last_row {
            for column in first_column..=last_column {
                self.tiles[(row * self.columns + column) as usize] = true;
            }
        }
    }

    /// Share of the texture's tiles marked, `0..=1`.
    pub fn dirty_fraction(&self) -> f32 {
        if self.tiles.is_empty() {
            return 0.0;
        }
        self.tiles.iter().filter(|tile| **tile).count() as f32 / self.tiles.len() as f32
    }

    /// Marked tiles as disjoint rectangles clipped to the texture: runs of
    /// tiles along each row, merged with identical runs on the rows below.
    pub fn rects(&self) -> Vec<TexelRect> {
        // Open runs: (first column, last column, first row).
        let mut open: Vec<(u32, u32, u32)> = Vec::new();
        let mut done = Vec::new();
        for row in 0..=self.rows {
            let mut runs = Vec::new();
            if row < self.rows {
                let mut column = 0;
                while column < self.columns {
                    if self.tiles[(row * self.columns + column) as usize] {
                        let start = column;
                        while column < self.columns && self.tiles[(row * self.columns + column) as usize]
                        {
                            column += 1;
                        }
                        runs.push((start, column - 1));
                    } else {
                        column += 1;
                    }
                }
            }
            let mut still_open = Vec::new();
            for (first, last, top) in open {
                if let Some(position) = runs.iter().position(|run| *run == (first, last)) {
                    runs.remove(position);
                    still_open.push((first, last, top));
                } else {
                    done.push(self.tile_rect(first, last, top, row - 1));
                }
            }
            still_open.extend(runs.into_iter().map(|(first, last)| (first, last, row)));
            open = still_open;
        }
        done.sort_by_key(|rect: &TexelRect| (rect.y, rect.x));
        done
    }

    fn tile_rect(&self, first_column: u32, last_column: u32, top: u32, bottom: u32) -> TexelRect {
        let x = first_column * DIRTY_TILE_SIZE;
        let y = top * DIRTY_TILE_SIZE;
        let right = ((last_column + 1) * DIRTY_TILE_SIZE).min(self.width);
        let bottom = ((bottom + 1) * DIRTY_TILE_SIZE).min(self.height);
        TexelRect {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_intersect_grow_and_index() {
        let rect = TexelRect {
            x: 10,
            y: 20,
            width: 5,
            height: 4,
        };
        assert_eq!(rect.len(), 20);
        assert!(rect.contains(14, 23) && !rect.contains(15, 23));
        assert_eq!(rect.local_index(11, 21), 6);
        assert_eq!(
            rect.intersect(TexelRect {
                x: 12,
                y: 0,
                width: 100,
                height: 22
            }),
            Some(TexelRect {
                x: 12,
                y: 20,
                width: 3,
                height: 2
            })
        );
        assert_eq!(rect.intersect(TexelRect::full(5, 5)), None);
        assert_eq!(
            TexelRect {
                x: 0,
                y: 1,
                width: 3,
                height: 3
            }
            .grow(2, 4, 100),
            TexelRect {
                x: 0,
                y: 0,
                width: 4,
                height: 6
            }
        );
    }

    #[test]
    fn dirty_tiles_merge_into_disjoint_rects() {
        let mut dirty = DirtyTexels::new(200, 200);
        assert!(dirty.is_clean());
        // Tiles (0..=1, 0..=1) as one block, plus a lone clipped corner tile.
        dirty.mark(TexelRect {
            x: 10,
            y: 10,
            width: 100,
            height: 100,
        });
        dirty.mark(TexelRect {
            x: 199,
            y: 199,
            width: 1,
            height: 1,
        });
        assert_eq!(
            dirty.rects(),
            vec![
                TexelRect {
                    x: 0,
                    y: 0,
                    width: 128,
                    height: 128
                },
                TexelRect {
                    x: 192,
                    y: 192,
                    width: 8,
                    height: 8
                },
            ]
        );
        assert!((dirty.dirty_fraction() - 5.0 / 16.0).abs() < 1e-6);
    }

    #[test]
    fn rects_cover_exactly_the_marked_tiles() {
        let mut dirty = DirtyTexels::new(300, 260);
        for (x, y) in [(0, 0), (70, 0), (70, 70), (250, 130), (130, 250), (0, 200)] {
            dirty.mark(TexelRect {
                x,
                y,
                width: 3,
                height: 3,
            });
        }
        let rects = dirty.rects();
        let mut covered = vec![0_u8; 300 * 260];
        for rect in &rects {
            for y in rect.y..rect.bottom() {
                for x in rect.x..rect.right() {
                    covered[(y * 300 + x) as usize] += 1;
                }
            }
        }
        for y in 0..260 {
            for x in 0..300 {
                let tile = (y / DIRTY_TILE_SIZE) * dirty.columns + x / DIRTY_TILE_SIZE;
                let expected = u8::from(dirty.tiles[tile as usize]);
                assert_eq!(covered[(y * 300 + x) as usize], expected, "texel {x},{y}");
            }
        }
    }
}
