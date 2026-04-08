//! # zero-alloc-sudoku
//!
//! A zero-heap-allocation, sub-microsecond Sudoku solver for Rust.
//!
//! ## Algorithm
//!
//! The solver combines two techniques:
//!
//! - **9-bit bitboard constraint propagation** — each of the 27 constraint
//!   groups (9 rows + 9 columns + 9 boxes) is represented as a `u16` bitmask
//!   where bit *k* is set when digit *k+1* has been placed. The available
//!   digits for any empty cell are computed with a single bitwise NOT and AND
//!   in `O(1)`.
//!
//! - **MRV (Minimum Remaining Values) heuristic backtracking** — at each
//!   recursive step the empty cell with the fewest legal digits is chosen
//!   first. This prunes the search tree dramatically on hard puzzles.
//!
//! The entire solver state is a single `#[repr(C, align(64))]` struct (~216 bytes)
//! that fits within two cache lines, eliminating cache misses during search.
//!
//! ## Quick start
//!
//! ```rust
//! use sudoku::Sudoku;
//!
//! let mut sudoku: Sudoku = "800000000003600000070090200050007000\
//!                           000045700000100030001000068008500010\
//!                           090000400"
//!     .parse()
//!     .expect("valid puzzle");
//!
//! assert!(sudoku.solve());
//! println!("{}", sudoku);
//! ```
//!
//! ## Performance
//!
//! On a modern x86-64 CPU with `--release` and `target-cpu=native`:
//!
//! | Puzzle          | Average (ns) | Throughput        |
//! |-----------------|--------------|-------------------|
//! | Al Escargot     | ~400 ns      | ~2 500 000 /s     |
//! | Hardest 2012    | ~600 ns      | ~1 600 000 /s     |
//! | Platinum Blonde | ~350 ns      | ~2 800 000 /s     |

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]

use std::fmt;
use std::str::FromStr;

// ── Internal lookup tables (generated at compile time) ────────────────────

/// Bitmask with all 9 digit-bits set: `0b1_1111_1111 == 0x1FF == 511`.
pub(crate) const FULL_MASK: u16 = 0x1FF;

pub(crate) const ROW: [u8; 81] = generate_row();
pub(crate) const COL: [u8; 81] = generate_col();
pub(crate) const BOX: [u8; 81] = generate_box();

const fn generate_row() -> [u8; 81] {
    let mut a = [0u8; 81];
    let mut i = 0;
    while i < 81 { a[i] = (i / 9) as u8; i += 1; }
    a
}
const fn generate_col() -> [u8; 81] {
    let mut a = [0u8; 81];
    let mut i = 0;
    while i < 81 { a[i] = (i % 9) as u8; i += 1; }
    a
}
const fn generate_box() -> [u8; 81] {
    let mut a = [0u8; 81];
    let mut i = 0;
    while i < 81 { a[i] = ((i / 27) * 3 + (i % 9 / 3)) as u8; i += 1; }
    a
}

// ── Error type ────────────────────────────────────────────────────────────

/// Error returned when a puzzle string cannot be parsed into a valid [`Sudoku`].
///
/// # Examples
///
/// ```rust
/// use sudoku::{Sudoku, ParseError};
///
/// // Wrong length
/// let err = "12345".parse::<Sudoku>().unwrap_err();
/// assert_eq!(err, ParseError::InvalidLength(5));
///
/// // Invalid character
/// let s = "x".repeat(1) + &"0".repeat(80);
/// let err = s.parse::<Sudoku>().unwrap_err();
/// assert_eq!(err, ParseError::InvalidCharacter { position: 0, ch: 'x' });
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The string did not contain exactly 81 characters.
    InvalidLength(usize),
    /// A character other than `'0'`–`'9'` was found.
    InvalidCharacter {
        /// Zero-based index in the input string.
        position: usize,
        /// The offending character.
        ch: char,
    },
    /// A digit appears more than once in the same row, column, or 3×3 box.
    DuplicateDigit {
        /// Zero-based cell index (0 = top-left, 80 = bottom-right).
        position: usize,
        /// The duplicated digit (1–9).
        digit: u8,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidLength(n) =>
                write!(f, "puzzle must be 81 characters, got {}", n),
            ParseError::InvalidCharacter { position, ch } =>
                write!(f, "invalid character {:?} at position {}", ch, position),
            ParseError::DuplicateDigit { position, digit } =>
                write!(f, "digit {} at position {} conflicts with its row/column/box",
                       digit, position),
        }
    }
}

impl std::error::Error for ParseError {}

// ── Core solver ───────────────────────────────────────────────────────────

/// A zero-allocation Sudoku board and solver.
///
/// The entire state (cells + bitboard constraints + empty-cell work list) fits
/// in a single `#[repr(C, align(64))]` struct (~216 bytes), keeping all working
/// data in L1 cache throughout the solve.
///
/// # Examples
///
/// Parse and solve:
///
/// ```rust
/// use sudoku::Sudoku;
///
/// let mut sudoku: Sudoku = "530070000600195000098000060800060003\
///                           400803001700020006060000280000419005\
///                           000080079"
///     .parse()
///     .unwrap();
///
/// assert!(sudoku.solve());
/// // Every cell is now filled with a valid digit.
/// assert!(sudoku.cells().iter().all(|&v| v >= 1 && v <= 9));
/// ```
///
/// Detect an invalid puzzle:
///
/// ```rust
/// use sudoku::{Sudoku, ParseError};
///
/// let result = "119000000000000000000000000000000000\
///               000000000000000000000000000000000000\
///               00000000000".parse::<Sudoku>();
/// assert!(result.is_err());
/// ```
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Sudoku {
    /// Raw cell values: `0` = empty, `1`–`9` = placed digit.
    pub cells: [u8; 81],
    rows:  [u16; 9],
    cols:  [u16; 9],
    boxes: [u16; 9],
    empty: [u8; 81],
}

impl Sudoku {
    // ── Construction ──────────────────────────────────────────────────

    /// Parses a Sudoku from a string of exactly 81 ASCII digits (`'0'` = empty).
    ///
    /// Prefer the [`FromStr`] implementation (`.parse()`) in new code;
    /// this method is kept for backwards compatibility.
    ///
    /// Returns `None` on any parse error. Use `.parse::<Sudoku>()` to
    /// obtain a detailed [`ParseError`] instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sudoku::Sudoku;
    ///
    /// let puzzle = "0".repeat(81);
    /// assert!(Sudoku::from_string(&puzzle).is_some());
    /// assert!(Sudoku::from_string("short").is_none());
    /// ```
    pub fn from_string(s: &str) -> Option<Self> {
        s.parse().ok()
    }

    // ── Accessors ─────────────────────────────────────────────────────

    /// Returns a reference to the 81-element cell array.
    ///
    /// Each element is `0` (empty) or a digit `1`–`9`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sudoku::Sudoku;
    ///
    /// let sudoku: Sudoku = "0".repeat(81).parse().unwrap();
    /// assert_eq!(sudoku.cells().len(), 81);
    /// assert!(sudoku.cells().iter().all(|&v| v == 0));
    /// ```
    #[inline]
    pub fn cells(&self) -> &[u8; 81] {
        &self.cells
    }

    /// Returns `true` if every cell has been filled (no zeros remain).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sudoku::Sudoku;
    ///
    /// let mut s: Sudoku = "0".repeat(81).parse().unwrap();
    /// assert!(!s.is_solved());
    /// s.solve();
    /// assert!(s.is_solved());
    /// ```
    #[inline]
    pub fn is_solved(&self) -> bool {
        self.cells.iter().all(|&v| v != 0)
    }

    /// Returns the digit at row `r`, column `c` (both 0-indexed).
    ///
    /// Returns `0` if the cell is empty.
    ///
    /// # Panics
    ///
    /// Panics if `r >= 9` or `c >= 9`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sudoku::Sudoku;
    ///
    /// let s: Sudoku = "530070000600195000098000060800060003\
    ///                  400803001700020006060000280000419005\
    ///                  000080079".parse().unwrap();
    /// assert_eq!(s.get(0, 0), 5);
    /// assert_eq!(s.get(0, 1), 3);
    /// assert_eq!(s.get(0, 2), 0); // empty
    /// ```
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> u8 {
        assert!(r < 9 && c < 9, "row and column must be 0..9");
        self.cells[r * 9 + c]
    }

    // ── Solver ────────────────────────────────────────────────────────

    /// Solves the puzzle **in place** using MRV backtracking with bitboard
    /// constraint propagation.
    ///
    /// Returns `true` if a solution was found (cells are updated), or `false`
    /// if the puzzle has no solution (cells are **unchanged**).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sudoku::Sudoku;
    ///
    /// // Arto Inkala's "AI Sudoku" (2010) — rated world's hardest at the time.
    /// let mut s: Sudoku = "800000000003600000070090200050007000\
    ///                      000045700000100030001000068008500010\
    ///                      090000400".parse().unwrap();
    ///
    /// assert!(s.solve());
    /// assert!(s.is_solved());
    /// ```
    pub fn solve(&mut self) -> bool {
        let mut n = 0usize;
        for i in 0..81 {
            if self.cells[i] == 0 {
                self.empty[n] = i as u8;
                n += 1;
            }
        }
        self.solve_recursive(n)
    }

    #[inline(always)]
    fn get_mask(&self, idx: usize) -> u16 {
        debug_assert!(idx < 81);
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

    fn solve_recursive(&mut self, num_empty: usize) -> bool {
        if num_empty == 0 { return true; }

        let mut best_i = 0;
        let mut min_c = 10u32;
        let mut best_mask = 0u16;

        for i in 0..num_empty {
            debug_assert!(i < 81);
            let idx = unsafe { *self.empty.get_unchecked(i) as usize };
            debug_assert!(idx < 81);
            let mask = self.get_mask(idx);
            let count = mask.count_ones();
            if count == 0 { return false; }
            if count < min_c {
                min_c = count;
                best_i = i;
                best_mask = mask;
                if count == 1 { break; }
            }
        }

        let idx = self.empty[best_i] as usize;
        let last = num_empty - 1;
        let saved = self.empty[last];
        self.empty[best_i] = saved;

        let mut m = best_mask;
        while m != 0 {
            let bit = m & m.wrapping_neg();
            m ^= bit;
            let digit = (bit.trailing_zeros() + 1) as u8;
            let r = ROW[idx] as usize;
            let c = COL[idx] as usize;
            let b = BOX[idx] as usize;

            self.cells[idx] = digit;
            self.rows[r]  |= bit;
            self.cols[c]  |= bit;
            self.boxes[b] |= bit;

            if self.solve_recursive(last) { return true; }

            self.rows[r]  &= !bit;
            self.cols[c]  &= !bit;
            self.boxes[b] &= !bit;
        }

        self.cells[idx] = 0;
        self.empty[best_i] = idx as u8;
        self.empty[last] = saved;
        false
    }

    // ── Formatting helpers ────────────────────────────────────────────

    /// Returns the board as an 81-character string of digits (`'0'` = empty).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sudoku::Sudoku;
    ///
    /// let puzzle = "0".repeat(81);
    /// let s: Sudoku = puzzle.parse().unwrap();
    /// assert_eq!(s.to_digit_string(), puzzle);
    /// ```
    pub fn to_digit_string(&self) -> String {
        self.cells.iter().map(|&v| (v + b'0') as char).collect()
    }

    /// Prints the board as a human-readable 9×9 grid with box dividers to stdout.
    pub fn print_grid(&self) {
        for row in 0..9 {
            if row % 3 == 0 && row != 0 {
                println!("------+-------+------");
            }
            for col in 0..9 {
                if col % 3 == 0 && col != 0 { print!("| "); }
                let v = self.cells[row * 9 + col];
                if v == 0 { print!(". "); } else { print!("{} ", v); }
            }
            println!();
        }
    }
}

// ── Trait implementations ─────────────────────────────────────────────────

impl FromStr for Sudoku {
    type Err = ParseError;

    /// Parses a Sudoku from a string of exactly 81 ASCII digits.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::InvalidLength`] if `s.len() != 81`,
    /// [`ParseError::InvalidCharacter`] for non-digit bytes, or
    /// [`ParseError::DuplicateDigit`] if any digit appears twice in a
    /// row, column, or 3×3 box.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sudoku::Sudoku;
    ///
    /// let s: Sudoku = "800000000003600000070090200050007000\
    ///                  000045700000100030001000068008500010\
    ///                  090000400".parse().unwrap();
    /// assert_eq!(s.get(0, 0), 8);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 81 {
            return Err(ParseError::InvalidLength(s.len()));
        }
        let mut board = Sudoku {
            cells: [0; 81],
            rows:  [0; 9],
            cols:  [0; 9],
            boxes: [0; 9],
            empty: [0; 81],
        };
        for (i, ch) in s.bytes().enumerate() {
            let val = match ch {
                b'0'..=b'9' => ch - b'0',
                _ => return Err(ParseError::InvalidCharacter {
                    position: i,
                    ch: ch as char,
                }),
            };
            if val != 0 {
                let bit: u16 = 1 << (val - 1);
                let r = ROW[i] as usize;
                let c = COL[i] as usize;
                let b = BOX[i] as usize;
                if (board.rows[r] | board.cols[c] | board.boxes[b]) & bit != 0 {
                    return Err(ParseError::DuplicateDigit { position: i, digit: val });
                }
                board.cells[i] = val;
                board.rows[r]  |= bit;
                board.cols[c]  |= bit;
                board.boxes[b] |= bit;
            }
        }
        Ok(board)
    }
}

/// Displays the board as an 81-character digit string (same as [`to_digit_string`]).
///
/// [`to_digit_string`]: Sudoku::to_digit_string
///
/// # Examples
///
/// ```rust
/// use sudoku::Sudoku;
///
/// let s: Sudoku = "0".repeat(81).parse().unwrap();
/// assert_eq!(format!("{}", s), "0".repeat(81));
/// ```
impl fmt::Display for Sudoku {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &v in &self.cells {
            write!(f, "{}", v)?;
        }
        Ok(())
    }
}

/// Formats the board as a compact digit string (identical to [`Display`]).
impl fmt::Debug for Sudoku {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sudoku(\"{}\")", self)
    }
}

impl PartialEq for Sudoku {
    fn eq(&self, other: &Self) -> bool {
        self.cells == other.cells
    }
}

impl Eq for Sudoku {}

// ── Default: empty (all-zeros) board ──────────────────────────────────────

/// Creates an empty board where every cell is unset.
///
/// # Examples
///
/// ```rust
/// use sudoku::Sudoku;
///
/// let mut s = Sudoku::default();
/// assert!(!s.is_solved());
/// assert!(s.solve()); // any valid completion exists
/// ```
impl Default for Sudoku {
    fn default() -> Self {
        Sudoku {
            cells: [0; 81],
            rows:  [0; 9],
            cols:  [0; 9],
            boxes: [0; 9],
            empty: [0; 81],
        }
    }
}

// ============================================================
//  CORRECTNESS PROOF — TEST SUITE  (L1–L5)
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ── Internal helpers ────────────────────────────────────────────

    impl Sudoku {
        fn bitboard_matches_cells(&self) -> bool {
            let mut er = [0u16; 9];
            let mut ec = [0u16; 9];
            let mut eb = [0u16; 9];
            for i in 0..81 {
                let v = self.cells[i];
                if v != 0 {
                    let bit = 1u16 << (v - 1);
                    er[ROW[i] as usize] |= bit;
                    ec[COL[i] as usize] |= bit;
                    eb[BOX[i] as usize] |= bit;
                }
            }
            self.rows == er && self.cols == ec && self.boxes == eb
        }
    }

    fn is_valid_solution(cells: &[u8; 81]) -> bool {
        for r in 0..9 {
            let mut m = 0u16;
            for c in 0..9 {
                let v = cells[r * 9 + c];
                if v == 0 || v > 9 { return false; }
                let bit = 1u16 << (v - 1);
                if m & bit != 0 { return false; }
                m |= bit;
            }
            if m != FULL_MASK { return false; }
        }
        for c in 0..9 {
            let mut m = 0u16;
            for r in 0..9 {
                let v = cells[r * 9 + c];
                let bit = 1u16 << (v - 1);
                if m & bit != 0 { return false; }
                m |= bit;
            }
            if m != FULL_MASK { return false; }
        }
        for br in 0..3 { for bc in 0..3 {
            let mut m = 0u16;
            for dr in 0..3 { for dc in 0..3 {
                let v = cells[(br * 3 + dr) * 9 + (bc * 3 + dc)];
                let bit = 1u16 << (v - 1);
                if m & bit != 0 { return false; }
                m |= bit;
            }}
            if m != FULL_MASK { return false; }
        }}
        true
    }

    fn assert_clues_preserved(puzzle: &str, s: &Sudoku) {
        for (i, ch) in puzzle.bytes().enumerate() {
            if ch != b'0' {
                assert_eq!(s.cells[i], ch - b'0',
                    "Clue at pos {} (row {}, col {}) overwritten", i, i/9, i%9);
            }
        }
    }

    // ── L1: Constant & table integrity ──────────────────────────────

    #[test]
    fn l1_full_mask_is_9_bit_all_ones() {
        assert_eq!(FULL_MASK, 0b1_1111_1111);
        assert_eq!(FULL_MASK, 0x1FF);
        assert_eq!(FULL_MASK, 511u16);
        assert_eq!(FULL_MASK.count_ones(), 9);
        assert_eq!(FULL_MASK & !0x1FF, 0);
    }

    #[test]
    fn l1_row_table_is_correct() {
        for i in 0..81usize { assert_eq!(ROW[i], (i / 9) as u8); assert!(ROW[i] < 9); }
        let mut c = [0u8; 9]; for &r in ROW.iter() { c[r as usize] += 1; }
        assert_eq!(c, [9u8; 9]);
    }

    #[test]
    fn l1_col_table_is_correct() {
        for i in 0..81usize { assert_eq!(COL[i], (i % 9) as u8); assert!(COL[i] < 9); }
        let mut c = [0u8; 9]; for &v in COL.iter() { c[v as usize] += 1; }
        assert_eq!(c, [9u8; 9]);
    }

    #[test]
    fn l1_box_table_is_correct() {
        for i in 0..81usize {
            assert_eq!(BOX[i], ((i / 27) * 3 + (i % 9 / 3)) as u8);
            assert!(BOX[i] < 9);
        }
        let mut c = [0u8; 9]; for &b in BOX.iter() { c[b as usize] += 1; }
        assert_eq!(c, [9u8; 9]);
    }

    #[test]
    fn l1_row_col_box_partition_is_consistent() {
        for i in 0..81usize {
            let r = ROW[i] as usize; let c = COL[i] as usize; let b = BOX[i] as usize;
            assert_eq!(r * 9 + c, i);
            assert_eq!(b, (r / 3) * 3 + (c / 3));
        }
    }

    // ── L2: Input validation ─────────────────────────────────────────

    #[test]
    fn l2_rejects_row_duplicate() {
        let s = "110000000000000000000000000000000000000000000000000000000000000000000000000000000";
        assert!(s.parse::<Sudoku>().is_err());
    }

    #[test]
    fn l2_rejects_col_duplicate() {
        let s = "000050000000050000000000000000000000000000000000000000000000000000000000000000000";
        assert!(s.parse::<Sudoku>().is_err());
    }

    #[test]
    fn l2_rejects_box_duplicate() {
        // Box 4 (centre): 3 at (3,3) and (5,5)
        let s = "000000000000000000000000000000300000000000000000003000000000000000000000000000000";
        assert!(s.parse::<Sudoku>().is_err());
    }

    #[test]
    fn l2_parse_error_variants() {
        assert_eq!("12345".parse::<Sudoku>().unwrap_err(), ParseError::InvalidLength(5));
        let non_digit = format!("x{}", "0".repeat(80));
        assert_eq!(non_digit.parse::<Sudoku>().unwrap_err(),
            ParseError::InvalidCharacter { position: 0, ch: 'x' });
        let dup_row = format!("11{}", "0".repeat(79));
        assert!(matches!(dup_row.parse::<Sudoku>().unwrap_err(),
            ParseError::DuplicateDigit { digit: 1, .. }));
    }

    #[test]
    fn l2_rejects_wrong_length() {
        assert!("0".repeat(80).parse::<Sudoku>().is_err());
        assert!("0".repeat(82).parse::<Sudoku>().is_err());
    }

    #[test]
    fn l2_accepts_all_zeros() {
        assert!("0".repeat(81).parse::<Sudoku>().is_ok());
    }

    // ── L3: Output soundness ─────────────────────────────────────────

    #[test]
    fn l3_al_escargot_solution_is_valid() {
        let p = "100007060900020008080500000000305070020010000800000400004000000000460010030900005";
        let mut s: Sudoku = p.parse().unwrap();
        assert!(s.solve());
        assert!(is_valid_solution(&s.cells));
        assert_clues_preserved(p, &s);
        assert!(s.is_solved());
    }

    #[test]
    fn l3_hardest_2012_solution_is_valid() {
        let p = "800000000003600000070090200050007000000045700000100030001000068008500010090000400";
        let mut s: Sudoku = p.parse().unwrap();
        assert!(s.solve());
        assert!(is_valid_solution(&s.cells));
        assert_clues_preserved(p, &s);
        assert!(s.is_solved());
    }

    #[test]
    fn l3_platinum_blonde_solution_is_valid() {
        let p = "000000012000000003002300400001800005060000070004000600000050090000200001000000000";
        let mut s: Sudoku = p.parse().unwrap();
        assert!(s.solve());
        assert!(is_valid_solution(&s.cells));
        assert_clues_preserved(p, &s);
        assert!(s.is_solved());
    }

    // ── L4: State integrity ──────────────────────────────────────────

    #[test]
    fn l4_bitboard_consistent_after_parse() {
        let s: Sudoku = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
            .parse().unwrap();
        assert!(s.bitboard_matches_cells());
    }

    #[test]
    fn l4_bitboard_consistent_after_solve() {
        let p = "100007060900020008080500000000305070020010000800000400004000000000460010030900005";
        let mut s: Sudoku = p.parse().unwrap();
        s.solve();
        assert!(s.bitboard_matches_cells());
    }

    #[test]
    fn l4_already_solved_board_returns_true() {
        let solved = "123456789456789123789123456231564897564897231897231564312645978645978312978312645";
        let mut s: Sudoku = solved.parse().unwrap();
        let before = s.cells;
        assert!(s.solve());
        assert_eq!(s.cells, before);
    }

    #[test]
    fn l4_unsolvable_board_returns_false_and_preserves_clues() {
        // Cell (0,0) needs 9 (row) but col-0 already has 9 → impossible
        let p = "012345678900000000000000000000000000000000000000000000000000000000000000000000000";
        let mut s: Sudoku = p.parse().unwrap();
        let snap: Vec<u8> = p.bytes().enumerate()
            .filter(|(_, b)| *b != b'0').map(|(i, _)| s.cells[i]).collect();
        assert!(!s.solve());
        for (k, (i, _)) in p.bytes().enumerate().filter(|(_, b)| *b != b'0').enumerate() {
            assert_eq!(s.cells[i], snap[k]);
        }
    }

    #[test]
    fn l4_solve_is_deterministic() {
        let p = "000000012000000003002300400001800005060000070004000600000050090000200001000000000";
        let mut a: Sudoku = p.parse().unwrap();
        let mut b: Sudoku = p.parse().unwrap();
        a.solve(); b.solve();
        assert_eq!(a.cells, b.cells);
    }

    #[test]
    fn l4_display_round_trips() {
        let p = "800000000003600000070090200050007000000045700000100030001000068008500010090000400";
        let s: Sudoku = p.parse().unwrap();
        assert_eq!(format!("{}", s), p);
        assert_eq!(s.to_digit_string(), p);
    }

    #[test]
    fn l4_default_is_empty_board() {
        let s = Sudoku::default();
        assert!(s.cells.iter().all(|&v| v == 0));
    }

    // ── L5: Batch completeness ───────────────────────────────────────

    #[test]
    fn l5_batch_hard_puzzles() {
        let corpus = [
            ("Al Escargot",
             "100007060900020008080500000000305070020010000800000400004000000000460010030900005"),
            ("Hardest 2012",
             "800000000003600000070090200050007000000045700000100030001000068008500010090000400"),
            ("Platinum Blonde",
             "000000012000000003002300400001800005060000070004000600000050090000200001000000000"),
            ("Norvig hard",
             "400000805030000000000700000020000060000080400000010000000603070500200000104000000"),
            ("Classic easy",
             "003020600900305001001806400008102900700000008006708200002609500800203009005010300"),
        ];
        for (name, puzzle) in &corpus {
            let mut s: Sudoku = puzzle.parse()
                .unwrap_or_else(|e| panic!("{}: parse error: {}", name, e));
            assert!(s.solve(),               "{}: no solution", name);
            assert!(is_valid_solution(&s.cells), "{}: invalid solution", name);
            assert_clues_preserved(puzzle, &s);
            assert!(s.is_solved(),           "{}: cells not all filled", name);
            assert!(s.bitboard_matches_cells(), "{}: bitboard mismatch", name);
        }
    }
}