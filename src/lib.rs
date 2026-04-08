#![allow(clippy::needless_range_loop)]

const FULL_MASK: u16 = 0x1FF;
const ROW: [u8; 81] = generate_row();
const COL: [u8; 81] = generate_col();
const BOX: [u8; 81] = generate_box();

const fn generate_row() -> [u8; 81] {
    let mut arr = [0u8; 81];
    let mut i = 0;
    while i < 81 {
        arr[i] = (i / 9) as u8;
        i += 1;
    }
    arr
}

const fn generate_col() -> [u8; 81] {
    let mut arr = [0u8; 81];
    let mut i = 0;
    while i < 81 {
        arr[i] = (i % 9) as u8;
        i += 1;
    }
    arr
}

const fn generate_box() -> [u8; 81] {
    let mut arr = [0u8; 81];
    let mut i = 0;
    while i < 81 {
        arr[i] = ((i / 27) * 3 + (i % 9 / 3)) as u8;
        i += 1;
    }
    arr
}

/// Zero-allocation Sudoku solver backed by a 9-bit bitboard per constraint group.
///
/// The entire state fits in a single cache-line-aligned struct (~216 bytes),
/// keeping all working data in L1 cache throughout the solve.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Sudoku {
    pub cells: [u8; 81],
    rows: [u16; 9],
    cols: [u16; 9],
    boxes: [u16; 9],
    empty: [u8; 81],
}

impl Sudoku {
    /// Parses a Sudoku from a string of exactly 81 ASCII digits (`'0'` for empty cells).
    ///
    /// Returns `None` if the string is not 81 characters long, contains non-digit
    /// characters, or has duplicate digits in any row, column, or 3×3 box.
    pub fn from_string(s: &str) -> Option<Self> {
        if s.len() != 81 {
            return None;
        }
        let mut board = Sudoku {
            cells: [0; 81],
            rows: [0; 9],
            cols: [0; 9],
            boxes: [0; 9],
            empty: [0; 81],
        };
        for (i, ch) in s.bytes().enumerate() {
            // Reject anything outside ASCII '0'..='9'
            let val = match ch {
                b'0'..=b'9' => ch - b'0',
                _ => return None,
            };
            if val != 0 {
                let bit: u16 = 1 << (val - 1);
                let r = ROW[i] as usize;
                let c = COL[i] as usize;
                let b = BOX[i] as usize;
                if (board.rows[r] & bit) != 0
                    || (board.cols[c] & bit) != 0
                    || (board.boxes[b] & bit) != 0
                {
                    return None;
                }
                board.cells[i] = val;
                board.rows[r] |= bit;
                board.cols[c] |= bit;
                board.boxes[b] |= bit;
            }
        }
        Some(board)
    }

    /// Returns the bitmask of digits (bits 0–8 = digits 1–9) still available
    /// for the cell at `idx`, computed as the complement of the union of its
    /// row, column, and box constraints.
    ///
    /// # Safety
    /// `idx` must be in `0..81`.
    #[inline(always)]
    fn get_mask(&self, idx: usize) -> u16 {
        // SAFETY: idx must be in 0..81. ROW/COL/BOX values are in 0..9 by construction.
        debug_assert!(idx < 81, "get_mask: idx out of bounds: {}", idx);
        unsafe {
            let r = *ROW.get_unchecked(idx) as usize;
            let c = *COL.get_unchecked(idx) as usize;
            let b = *BOX.get_unchecked(idx) as usize;
            debug_assert!(r < 9 && c < 9 && b < 9);
            !(*self.rows.get_unchecked(r)
                | *self.cols.get_unchecked(c)
                | *self.boxes.get_unchecked(b))
                & FULL_MASK
        }
    }

    /// Solves the puzzle in-place using MRV (Minimum Remaining Values) heuristic
    /// backtracking with bit-level constraint propagation.
    ///
    /// Returns `true` if a solution was found, `false` if the puzzle is unsolvable.
    pub fn solve(&mut self) -> bool {
        let mut num_empty = 0;
        for i in 0..81 {
            if self.cells[i] == 0 {
                self.empty[num_empty] = i as u8;
                num_empty += 1;
            }
        }
        self.solve_recursive(num_empty)
    }

    fn solve_recursive(&mut self, num_empty: usize) -> bool {
        if num_empty == 0 {
            return true;
        }

        let mut best_i = 0;
        let mut min_c = 10;
        let mut best_mask = 0u16;

        for i in 0..num_empty {
            // SAFETY: i < num_empty <= 81, and empty[i] was written as (cell_index as u8)
            // where cell_index < 81, so the cast to usize is always in 0..81.
            debug_assert!(i < 81);
            let idx = unsafe { *self.empty.get_unchecked(i) as usize };
            debug_assert!(idx < 81, "solve_recursive: empty[{}] = {} out of bounds", i, idx);
            let mask = self.get_mask(idx);
            let count = mask.count_ones();
            if count == 0 {
                return false;
            }
            if count < min_c {
                min_c = count;
                best_i = i;
                best_mask = mask;
                if count == 1 {
                    break;
                }
            }
        }

        let idx = self.empty[best_i] as usize;
        let last_idx = num_empty - 1;
        let saved_val = self.empty[last_idx];
        self.empty[best_i] = saved_val;

        let mut m = best_mask;
        while m != 0 {
            let bit = m & m.wrapping_neg();
            m ^= bit;
            let digit = (bit.trailing_zeros() + 1) as u8;

            let r = ROW[idx] as usize;
            let c = COL[idx] as usize;
            let b = BOX[idx] as usize;

            self.cells[idx] = digit;
            self.rows[r] |= bit;
            self.cols[c] |= bit;
            self.boxes[b] |= bit;

            if self.solve_recursive(last_idx) {
                return true;
            }

            self.rows[r] &= !bit;
            self.cols[c] &= !bit;
            self.boxes[b] &= !bit;
        }

        self.cells[idx] = 0;
        self.empty[best_i] = idx as u8;
        self.empty[last_idx] = saved_val;
        false
    }

    /// Returns the current board state as an 81-character string of digits.
    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity(81);
        for i in 0..81 {
            s.push((self.cells[i] + b'0') as char);
        }
        s
    }

    /// Prints the board in a human-readable 9×9 grid with box dividers.
    pub fn print(&self) {
        for row in 0..9 {
            if row % 3 == 0 && row != 0 {
                println!("------+-------+------");
            }
            for col in 0..9 {
                if col % 3 == 0 && col != 0 {
                    print!("| ");
                }
                let val = self.cells[row * 9 + col];
                if val == 0 {
                    print!(". ");
                } else {
                    print!("{} ", val);
                }
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that a solved board satisfies all Sudoku constraints:
    /// every row, column, and 3×3 box contains each digit 1–9 exactly once.
    fn is_valid_solution(cells: &[u8; 81]) -> bool {
        let full: u16 = FULL_MASK;
        // Check rows
        for r in 0..9 {
            let mut mask = 0u16;
            for c in 0..9 {
                let v = cells[r * 9 + c];
                if v == 0 || v > 9 {
                    return false;
                }
                let bit = 1u16 << (v - 1);
                if mask & bit != 0 {
                    return false;
                }
                mask |= bit;
            }
            if mask != full {
                return false;
            }
        }
        // Check columns
        for c in 0..9 {
            let mut mask = 0u16;
            for r in 0..9 {
                let v = cells[r * 9 + c];
                let bit = 1u16 << (v - 1);
                if mask & bit != 0 {
                    return false;
                }
                mask |= bit;
            }
            if mask != full {
                return false;
            }
        }
        // Check 3×3 boxes
        for box_r in 0..3 {
            for box_c in 0..3 {
                let mut mask = 0u16;
                for dr in 0..3 {
                    for dc in 0..3 {
                        let v = cells[(box_r * 3 + dr) * 9 + (box_c * 3 + dc)];
                        let bit = 1u16 << (v - 1);
                        if mask & bit != 0 {
                            return false;
                        }
                        mask |= bit;
                    }
                }
                if mask != full {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_al_escargot() {
        let puzzle =
            "100007060900020008080500000000305070020010000800000400004000000000460010030900005";

        let mut solver = Sudoku::from_string(puzzle).unwrap();
        assert!(solver.solve());

        assert!(
            is_valid_solution(&solver.cells),
            "Solution failed full validity check"
        );

        // All original clues must be preserved
        let puzzle_bytes: Vec<u8> = puzzle.bytes().collect();
        for (i, &clue) in puzzle_bytes.iter().enumerate() {
            if clue != b'0' {
                assert_eq!(solver.cells[i], clue - b'0',
                    "Clue at position {} was overwritten", i);
            }
        }
    }

    #[test]
    fn test_platinum_blonde() {
        // "Platinum Blonde" — a well-known symmetrical hard puzzle.
        let puzzle =
            "000000012000000003002300400001800005060000070004000600000050090000200001000000000";

        let mut solver = Sudoku::from_string(puzzle).unwrap();
        assert!(solver.solve(), "Solver failed to find a solution");

        assert!(
            is_valid_solution(&solver.cells),
            "Solution failed full validity check (row/col/box constraints)"
        );

        let puzzle_bytes: Vec<u8> = puzzle.bytes().collect();
        for (i, &clue) in puzzle_bytes.iter().enumerate() {
            if clue != b'0' {
                assert_eq!(solver.cells[i], clue - b'0',
                    "Clue at position {} was overwritten", i);
            }
        }
    }

    #[test]
    fn test_hardest_2012() {
        let puzzle =
            "800000000003600000070090200050007000000045700000100030001000068008500010090000400";

        let mut solver = Sudoku::from_string(puzzle).unwrap();
        assert!(solver.solve());

        assert!(
            is_valid_solution(&solver.cells),
            "Solution failed full validity check (row/col/box constraints)"
        );

        let puzzle_bytes: Vec<u8> = puzzle.bytes().collect();
        for (i, &clue) in puzzle_bytes.iter().enumerate() {
            if clue != b'0' {
                assert_eq!(solver.cells[i], clue - b'0',
                    "Clue at position {} was overwritten", i);
            }
        }
    }

    #[test]
    fn test_invalid_puzzle_detection() {
        // Duplicate in row
        let invalid = "110000000000000000000000000000000000000000000000000000000000000000000000000000000";
        assert!(Sudoku::from_string(invalid).is_none());

        // Non-digit character
        let non_digit = "x00000000000000000000000000000000000000000000000000000000000000000000000000000000";
        assert!(Sudoku::from_string(non_digit).is_none());

        // Wrong length
        assert!(Sudoku::from_string("12345").is_none());
    }
}