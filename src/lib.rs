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

    #[inline(always)]
    fn get_mask(&self, idx: usize) -> u16 {
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

    /// Solves the puzzle in-place using MRV backtracking with bitboard constraint propagation.
    /// Returns `true` if a solution exists, `false` if unsolvable.
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
            debug_assert!(i < 81);
            let idx = unsafe { *self.empty.get_unchecked(i) as usize };
            debug_assert!(idx < 81);
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

    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity(81);
        for i in 0..81 {
            s.push((self.cells[i] + b'0') as char);
        }
        s
    }

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

// ============================================================
//  CRYPTOGRAPHIC CORRECTNESS PROOF — TEST SUITE
//
//  Organized as formal property layers:
//    L1 — Constant & lookup-table integrity
//    L2 — Input validation soundness  (reject iff invalid)
//    L3 — Output soundness            (solution ↔ valid Sudoku)
//    L4 — State integrity             (backtrack undo, determinism)
//    L5 — Batch completeness          (hard puzzle corpus)
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // Internal-only helpers
    // ----------------------------------------------------------------

    /// Checks that the solver's bitboard arrays (rows/cols/boxes) are
    /// exactly consistent with the cell array — i.e. no shadow state
    /// was left behind by incomplete backtracking.
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

    /// Full Sudoku validity check: every row, column, and 3×3 box must
    /// contain each of the digits 1–9 exactly once.
    fn is_valid_solution(cells: &[u8; 81]) -> bool {
        // rows
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
        // columns
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
        // 3×3 boxes
        for br in 0..3 {
            for bc in 0..3 {
                let mut m = 0u16;
                for dr in 0..3 {
                    for dc in 0..3 {
                        let v = cells[(br * 3 + dr) * 9 + (bc * 3 + dc)];
                        let bit = 1u16 << (v - 1);
                        if m & bit != 0 { return false; }
                        m |= bit;
                    }
                }
                if m != FULL_MASK { return false; }
            }
        }
        true
    }

    /// Asserts all original non-zero clues are unchanged in the solution.
    fn assert_clues_preserved(puzzle: &str, solver: &Sudoku) {
        for (i, ch) in puzzle.bytes().enumerate() {
            if ch != b'0' {
                assert_eq!(
                    solver.cells[i], ch - b'0',
                    "Clue at position {} (row {}, col {}) was overwritten: \
                     expected {} got {}",
                    i, i / 9, i % 9, ch - b'0', solver.cells[i]
                );
            }
        }
    }

    // ================================================================
    // L1 — Constant & lookup-table integrity
    // ================================================================

    /// FULL_MASK must be the 9-bit all-ones value (0b1_1111_1111 = 0x1FF = 511).
    /// Any deviation would silently allow duplicate digits to pass constraint checks.
    #[test]
    fn l1_full_mask_is_9_bit_all_ones() {
        assert_eq!(FULL_MASK, 0b1_1111_1111, "FULL_MASK wrong binary value");
        assert_eq!(FULL_MASK, 0x1FF,         "FULL_MASK wrong hex value");
        assert_eq!(FULL_MASK, 511u16,         "FULL_MASK wrong decimal value");
        assert_eq!(FULL_MASK.count_ones(), 9, "FULL_MASK must have exactly 9 set bits");
        assert_eq!(FULL_MASK & !0x1FF, 0,     "FULL_MASK must not set bits above bit-8");
    }

    /// ROW[i] must equal i/9 for every i in 0..81.
    #[test]
    fn l1_row_table_is_correct() {
        for i in 0..81usize {
            assert_eq!(ROW[i], (i / 9) as u8, "ROW[{}] wrong", i);
            assert!(ROW[i] < 9, "ROW[{}] out of range", i);
        }
        // Each row index must appear exactly 9 times (one per cell in that row).
        let mut counts = [0u8; 9];
        for &r in ROW.iter() { counts[r as usize] += 1; }
        assert_eq!(counts, [9u8; 9], "Each row index must appear 9 times in ROW table");
    }

    /// COL[i] must equal i%9 for every i in 0..81.
    #[test]
    fn l1_col_table_is_correct() {
        for i in 0..81usize {
            assert_eq!(COL[i], (i % 9) as u8, "COL[{}] wrong", i);
            assert!(COL[i] < 9, "COL[{}] out of range", i);
        }
        let mut counts = [0u8; 9];
        for &c in COL.iter() { counts[c as usize] += 1; }
        assert_eq!(counts, [9u8; 9], "Each col index must appear 9 times in COL table");
    }

    /// BOX[i] must equal (i/27)*3 + (i%9/3) and must be in 0..9.
    /// Each of the 9 box indices must appear exactly 9 times.
    #[test]
    fn l1_box_table_is_correct() {
        for i in 0..81usize {
            let expected = ((i / 27) * 3 + (i % 9 / 3)) as u8;
            assert_eq!(BOX[i], expected, "BOX[{}] wrong", i);
            assert!(BOX[i] < 9, "BOX[{}] out of range", i);
        }
        let mut counts = [0u8; 9];
        for &b in BOX.iter() { counts[b as usize] += 1; }
        assert_eq!(counts, [9u8; 9], "Each box index must appear 9 times in BOX table");
    }

    /// ROW, COL, BOX must partition the 81 cells consistently:
    /// for every cell, the triple (row, col, box) uniquely identifies it
    /// and the box membership must agree with the row/col grid position.
    #[test]
    fn l1_row_col_box_partition_is_consistent() {
        for i in 0..81usize {
            let r = ROW[i] as usize;
            let c = COL[i] as usize;
            let b = BOX[i] as usize;
            // Recompute canonical cell index and verify it round-trips.
            assert_eq!(r * 9 + c, i, "ROW/COL round-trip failed at index {}", i);
            // Verify box identity from (r, c).
            let expected_box = (r / 3) * 3 + (c / 3);
            assert_eq!(b, expected_box,
                "BOX[{}] = {} but (row={}, col={}) implies box {}", i, b, r, c, expected_box);
        }
    }

    // ================================================================
    // L2 — Input validation soundness
    // ================================================================

    #[test]
    fn l2_rejects_row_duplicate() {
        // Two 1s in the same row — must return None.
        let s = "110000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000";
        assert!(Sudoku::from_string(s).is_none(), "Row duplicate must be rejected");
    }

    #[test]
    fn l2_rejects_col_duplicate() {
        // Digit 5 in column 4 of rows 0 and 1.
        let s = "000050000\
                 000050000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000";
        assert!(Sudoku::from_string(s).is_none(), "Column duplicate must be rejected");
    }

    #[test]
    fn l2_rejects_box_duplicate() {
        // Digit 3 in two cells of the same 3×3 box but different rows and columns.
        // Box 4 (centre): rows 3-5, cols 3-5. Put 3 at (3,3) and (5,5).
        let s = "000000000\
                 000000000\
                 000000000\
                 000300000\
                 000000000\
                 000003000\
                 000000000\
                 000000000\
                 000000000";
        assert!(Sudoku::from_string(s).is_none(), "Box duplicate must be rejected");
    }

    #[test]
    fn l2_rejects_non_digit_character() {
        let s = "x00000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000\
                 000000000";
        assert!(Sudoku::from_string(s).is_none(), "Non-digit must be rejected");
    }

    #[test]
    fn l2_rejects_wrong_length() {
        assert!(Sudoku::from_string("").is_none(), "empty string must be rejected");
        assert!(Sudoku::from_string("123456789").is_none(), "9-char string must be rejected");

        // 80 chars — one too short
        let short: String = "0".repeat(80);
        assert!(Sudoku::from_string(&short).is_none(), "80-char string must be rejected");

        // 82 chars — one too long
        let long: String = "0".repeat(82);
        assert!(Sudoku::from_string(&long).is_none(), "82-char string must be rejected");
    }

    #[test]
    fn l2_accepts_all_zeros_empty_board() {
        let s: String = "0".repeat(81);
        assert!(Sudoku::from_string(&s).is_some(), "All-zeros board must be accepted");
    }

    // ================================================================
    // L3 — Output soundness
    // ================================================================

    /// Core property: the solver NEVER outputs an invalid Sudoku.
    /// Tested on Al Escargot (Arto Inkala, 2007).
    #[test]
    fn l3_al_escargot_solution_is_valid() {
        let puzzle = "100007060900020008080500000000305070020010000800000400004000000000460010030900005";
        let mut s = Sudoku::from_string(puzzle).unwrap();
        assert!(s.solve(), "solve() must return true");
        assert!(is_valid_solution(&s.cells), "Output is not a valid Sudoku");
        assert_clues_preserved(puzzle, &s);
        assert!(s.cells.iter().all(|&v| v != 0), "Solved board must have no empty cells");
    }

    /// Tested on Arto Inkala's 2010 "AI Sudoku" (a.k.a. "Hardest 2012").
    #[test]
    fn l3_hardest_2012_solution_is_valid() {
        let puzzle = "800000000003600000070090200050007000000045700000100030001000068008500010090000400";
        let mut s = Sudoku::from_string(puzzle).unwrap();
        assert!(s.solve());
        assert!(is_valid_solution(&s.cells));
        assert_clues_preserved(puzzle, &s);
        assert!(s.cells.iter().all(|&v| v != 0));
    }

    /// Tested on "Platinum Blonde" (well-known symmetrical hard puzzle).
    #[test]
    fn l3_platinum_blonde_solution_is_valid() {
        let puzzle = "000000012000000003002300400001800005060000070004000600000050090000200001000000000";
        let mut s = Sudoku::from_string(puzzle).unwrap();
        assert!(s.solve());
        assert!(is_valid_solution(&s.cells));
        assert_clues_preserved(puzzle, &s);
        assert!(s.cells.iter().all(|&v| v != 0));
    }

    // ================================================================
    // L4 — State integrity
    // ================================================================

    /// After `solve()`, the internal bitboards (rows/cols/boxes) must match
    /// the cells array exactly — no shadow state from backtracking.
    #[test]
    fn l4_bitboard_consistent_after_parse() {
        let puzzle = "800000000003600000070090200050007000000045700000100030001000068008500010090000400";
        let s = Sudoku::from_string(puzzle).unwrap();
        assert!(s.bitboard_matches_cells(),
            "Bitboard state must match cells immediately after from_string");
    }

    #[test]
    fn l4_bitboard_consistent_after_solve() {
        let puzzle = "100007060900020008080500000000305070020010000800000400004000000000460010030900005";
        let mut s = Sudoku::from_string(puzzle).unwrap();
        s.solve();
        assert!(s.bitboard_matches_cells(),
            "Bitboard state must match cells after solve — backtracking left shadow state");
    }

    /// solve() on an already-complete (zero empty cells) valid board must
    /// return true immediately without touching any cell.
    #[test]
    fn l4_already_solved_board_returns_true() {
        // Cyclic-shift valid solution; verified by inspection:
        // rows, cols, and boxes each contain 1–9 exactly once.
        let solved = "123456789456789123789123456231564897564897231897231564312645978645978312978312645";
        let mut s = Sudoku::from_string(solved).unwrap();
        let cells_before = s.cells;
        assert!(s.solve(), "solve() must return true for an already-solved board");
        assert_eq!(s.cells, cells_before, "solve() must not touch cells of a solved board");
    }

    /// solve() on an unsolvable board must return false AND leave the
    /// original clue cells untouched (clean backtracking invariant).
    ///
    /// Construction:
    ///   Row 0 cols 1-8 = digits 1-8  →  row 0 uses {1..8}, needs 9 at col 0.
    ///   Col 0 row 1    = digit 9     →  col 0 already has 9.
    ///   Cell (0,0) is empty: its row needs 9, its column already has 9 → no valid digit.
    #[test]
    fn l4_unsolvable_board_returns_false_and_preserves_clues() {
        //  Row 0:  _ 1 2 3 4 5 6 7 8   (cell 0,0 is empty; row needs 9)
        //  Row 1:  9 _ _ _ _ _ _ _ _   (col 0 already has 9)
        //  Rows 2-8: all empty
        let puzzle = "012345678\
                      900000000\
                      000000000\
                      000000000\
                      000000000\
                      000000000\
                      000000000\
                      000000000\
                      000000000";
        let mut s = Sudoku::from_string(puzzle)
            .expect("Puzzle must be accepted by from_string (no direct clue conflicts)");

        // Snapshot clue cells before the (failing) solve.
        let clue_snapshot: Vec<u8> = puzzle.bytes()
            .enumerate()
            .filter(|(_, b)| *b != b'0')
            .map(|(i, _)| s.cells[i])
            .collect();

        assert!(!s.solve(), "Unsolvable board must return false");

        // All original clues must be preserved even after backtracking.
        for (idx, (i, b)) in puzzle.bytes()
            .enumerate()
            .filter(|(_, b)| *b != b'0')
            .enumerate()
        {
            assert_eq!(s.cells[i], clue_snapshot[idx],
                "Clue at {} altered after failed solve (expected {}, got {})",
                i, clue_snapshot[idx], s.cells[i]);
            let _ = b; // satisfy the borrow checker
        }
    }

    /// Solving the same puzzle twice must yield bit-for-bit identical results
    /// (determinism required for reproducible behaviour).
    #[test]
    fn l4_solve_is_deterministic() {
        let puzzle = "000000012000000003002300400001800005060000070004000600000050090000200001000000000";
        let mut a = Sudoku::from_string(puzzle).unwrap();
        let mut b = Sudoku::from_string(puzzle).unwrap();
        a.solve();
        b.solve();
        assert_eq!(a.cells, b.cells, "Two independent solves of the same puzzle differ");
    }

    // ================================================================
    // L5 — Batch completeness on a corpus of hard puzzles
    // ================================================================

    /// Runs the solver against 5 independently-sourced hard puzzles and
    /// verifies full row/col/box correctness and clue preservation for each.
    #[test]
    fn l5_batch_hard_puzzles() {
        // (name, puzzle_string)
        let corpus: &[(&str, &str)] = &[
            // Arto Inkala "Al Escargot" (2007)
            ("Al Escargot",
             "100007060900020008080500000000305070020010000800000400004000000000460010030900005"),
            // Arto Inkala "AI Sudoku" / Hardest 2012
            ("Hardest 2012",
             "800000000003600000070090200050007000000045700000100030001000068008500010090000400"),
            // Platinum Blonde — symmetric 22-clue hard puzzle
            ("Platinum Blonde",
             "000000012000000003002300400001800005060000070004000600000050090000200001000000000"),
            // From norvig.com/sudoku.html — hard puzzle used in MRV benchmark
            ("Norvig hard",
             "400000805030000000000700000020000060000080400000010000000603070500200000104000000"),
            // Classic easy puzzle (quick sanity check at end of batch)
            ("Classic easy",
             "003020600900305001001806400008102900700000008006708200002609500800203009005010300"),
        ];

        for (name, puzzle) in corpus {
            let mut s = Sudoku::from_string(puzzle)
                .unwrap_or_else(|| panic!("{}: from_string returned None", name));

            assert!(s.solve(), "{}: solver returned false", name);

            assert!(
                is_valid_solution(&s.cells),
                "{}: solution is not a valid Sudoku", name
            );

            assert_clues_preserved(puzzle, &s);

            assert!(
                s.cells.iter().all(|&v| v != 0),
                "{}: solution contains empty cells", name
            );

            assert!(
                s.bitboard_matches_cells(),
                "{}: bitboard/cell mismatch after solve", name
            );
        }
    }
}