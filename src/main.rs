use std::hint::black_box;
use std::time::Instant;
use sudoku::Sudoku;

fn main() {
    let benchmarks = [
        (
            "Al Escargot",
            "100007060900020008080500000000305070020010000800000400004000000000460010030900005",
        ),
        (
            "Hardest 2012",
            "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
        ),
        (
            "Platinum Blonde",
            "000000012000000003002300400001800005060000070004000600000050090000200001000000000",
        ),
        (
            "Mystery Hard",
            // Row 2 had a duplicate '9'; col 5 of row 2 corrected to '0'
            "100000569492006108006900200080706942600000300300104006019800005000000100005000630",
        ),
    ];

    println!("=== Sudoku Solver Apex ===");
    for (name, puzzle) in &benchmarks {
        let initial = match Sudoku::from_string(puzzle) {
            Some(s) => s,
            None => {
                println!("Benchmark: {:<15} | INVALID PUZZLE", name);
                continue;
            }
        };
        let iterations = 10_000u32;
        let start = Instant::now();
        for _ in 0..iterations {
            let mut s = initial;
            black_box(s.solve());
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / iterations as u128;
        println!(
            "Benchmark: {:<15} | Avg: {:>7} ns | Throughput: {:>12.0} solves/s",
            name,
            avg_ns,
            1_000_000_000.0 / avg_ns as f64
        );
    }
}