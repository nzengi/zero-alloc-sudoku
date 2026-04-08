# zero-alloc-sudoku

[![Crates.io](https://img.shields.io/crates/v/zero-alloc-sudoku)](https://crates.io/crates/zero-alloc-sudoku)
[![docs.rs](https://docs.rs/zero-alloc-sudoku/badge.svg)](https://docs.rs/zero-alloc-sudoku)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

A **zero-heap-allocation**, sub-microsecond Sudoku solver for Rust.

## Features

- **Zero allocations** — the entire solver state is a single `#[repr(C, align(64))]`
  struct (~216 bytes) that lives on the stack.
- **9-bit bitboard constraint propagation** — available digits for any cell
  are computed with a single bitwise NOT + AND in O(1).
- **MRV backtracking** — the empty cell with the fewest legal digits is always
  tried first, pruning the search tree dramatically on hard puzzles.
- **Idiomatic Rust API** — `FromStr`, `Display`, `Debug`, `Default`, `PartialEq`,
  `Eq`, and a rich `ParseError` type.
- **No `unsafe` in the hot path** beyond bounds-elision hints guarded by
  `debug_assert!`.

## Installation

```toml
[dependencies]
zero-alloc-sudoku = "1.0"
```

## Quick start

```rust
use sudoku::Sudoku;

// Parse
let mut s: Sudoku = "800000000003600000070090200050007000\
                     000045700000100030001000068008500010\
                     090000400"
    .parse()
    .expect("valid puzzle");

// Solve
assert!(s.solve());
assert!(s.is_solved());

// Display
println!("{}", s);   // 81-char digit string
s.print_grid();      // pretty 9×9 grid
```

## API overview

| Item | Description |
|------|-------------|
| `Sudoku::default()` | Empty board (all zeros) |
| `"…".parse::<Sudoku>()` | Parse from 81-char digit string |
| `Sudoku::from_string(&str)` | Same, returns `Option` |
| `s.solve() -> bool` | Solve in place; returns `false` if unsolvable |
| `s.is_solved() -> bool` | `true` when no empty cells remain |
| `s.get(row, col) -> u8` | Digit at (row, col) — `0` = empty |
| `s.cells() -> &[u8; 81]` | Raw cell array |
| `s.to_digit_string()` | 81-char `String` |
| `s.print_grid()` | Pretty-print to stdout |
| `format!("{}", s)` | Same as `to_digit_string` |
| `ParseError` | `InvalidLength` / `InvalidCharacter` / `DuplicateDigit` |

## Error handling

```rust
use sudoku::{Sudoku, ParseError};

match "12345".parse::<Sudoku>() {
    Err(ParseError::InvalidLength(n)) => eprintln!("need 81 chars, got {}", n),
    Err(ParseError::InvalidCharacter { position, ch }) =>
        eprintln!("bad char {:?} at {}", ch, position),
    Err(ParseError::DuplicateDigit { position, digit }) =>
        eprintln!("digit {} at {} conflicts", digit, position),
    Ok(mut s) => { s.solve(); }
}
```

## Performance

Benchmarked on Apple M2 (`--release`, `target-cpu=native`):

| Puzzle | Mean | Throughput |
|--------|------|------------|
| Al Escargot | ~400 ns | ~2 500 000 puzzles/s |
| Hardest 2012 (Inkala) | ~600 ns | ~1 600 000 puzzles/s |
| Platinum Blonde | ~350 ns | ~2 800 000 puzzles/s |

## Command-line usage

```bash
cargo install zero-alloc-sudoku

# Single puzzle as argument
sudoku-solver 800000000003600000070090200050007000000045700000100030001000068008500010090000400

# Stream puzzles from stdin (one per line, # for comments)
cat puzzles.txt | sudoku-solver
```

## Test suite

The library ships with a 5-layer correctness proof suite (35 tests total):

- **L1** — compile-time constant & lookup table integrity
- **L2** — input validation and `ParseError` variant coverage
- **L3** — solution soundness (row/column/box completeness + clue preservation)
- **L4** — internal state consistency (bitboard ↔ cells round-trip)
- **L5** — batch correctness across 5 world-record hard puzzles

```
cargo test        # runs all 35 tests including 13 doc tests
cargo test --doc  # doc tests only
```

## License

Licensed under either of

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.