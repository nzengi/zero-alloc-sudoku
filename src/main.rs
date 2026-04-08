//! Command-line interface for the `zero-alloc-sudoku` solver.
//!
//! Usage:
//! ```
//! echo "800000000003600000070090200050007000000045700000100030001000068008500010090000400" | sudoku-solver
//! ```
//! Or pass the puzzle as a command-line argument:
//! ```
//! sudoku-solver 800000000003600000070090200050007000000045700000100030001000068008500010090000400
//! ```

use sudoku::Sudoku;
use std::io::{self, Read};

fn solve_and_print(puzzle: &str) {
    let puzzle = puzzle.trim();
    match puzzle.parse::<Sudoku>() {
        Err(e) => eprintln!("Error: {}", e),
        Ok(mut s) => {
            if s.solve() {
                println!("{}", s);
            } else {
                eprintln!("No solution found.");
                std::process::exit(1);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        // Puzzle passed as CLI argument
        solve_and_print(&args[1]);
    } else {
        // Read from stdin
        let mut input = String::new();
        io::stdin().read_to_string(&mut input).expect("Failed to read stdin");
        for line in input.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') {
                solve_and_print(line);
            }
        }
    }
}