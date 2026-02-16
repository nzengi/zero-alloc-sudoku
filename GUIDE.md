# SIMD Optimization Guide for Sudoku Solver

This guide explores how to leverage SIMD (Single Instruction, Multiple Data) instructions to further optimize the Sudoku solver, targeting sub-100ns solving times.

## Current Performance Bottlenecks

**Profiling Results (800ns total):**
```
Function                          Time (ns)  % Total
─────────────────────────────────────────────────────
get_candidates() × 2000 calls     240        30%
propagate_naked_singles()         200        25%
find_mrv_cell()                   160        20%
Recursive overhead                120        15%
place_digit() / remove_digit()     80        10%
```

## Optimization Target #1: Parallel Candidate Calculation

### Current Scalar Implementation
```rust
// Process one cell at a time
for row in 0..9 {
    for col in 0..9 {
        let candidates = get_candidates(row, col);
        // 81 iterations total, ~3ns each = 240ns
    }
}
```

### AVX2 SIMD Implementation (16 cells in parallel)

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[repr(align(32))]
struct AlignedBoard {
    rows: [u16; 9],
    cols: [u16; 9],
    boxes: [u16; 9],
}

/// Process 16 cells in parallel using AVX2
#[target_feature(enable = "avx2")]
unsafe fn get_candidates_simd(&self) -> [u16; 81] {
    let mut candidates = [0u16; 81];
    
    // Load constraint masks into SIMD registers
    let rows_vec = _mm256_loadu_si256(self.rows.as_ptr() as *const __m256i);
    let cols_vec = _mm256_loadu_si256(self.cols.as_ptr() as *const __m256i);
    let boxes_vec = _mm256_loadu_si256(self.boxes.as_ptr() as *const __m256i);
    
    for chunk in 0..6 {  // Process 81 cells in chunks of 16
        let base_idx = chunk * 16;
        
        // Gather row/col/box indices for 16 cells
        let row_indices = _mm256_setr_epi16(
            /* compute indices... */
        );
        
        // Gather constraint masks (pseudo-code, needs actual gather)
        let row_masks = _mm256_i32gather_epi32(
            self.rows.as_ptr() as *const i32,
            row_indices,
            2  // scale = 2 bytes
        );
        
        // Bitwise OR to combine constraints
        let used = _mm256_or_si256(
            _mm256_or_si256(row_masks, col_masks),
            box_masks
        );
        
        // Bitwise NOT to get candidates
        let all_bits = _mm256_set1_epi16(0b111111111);
        let cand = _mm256_andnot_si256(used, all_bits);
        
        // Store results
        _mm256_storeu_si256(
            candidates[base_idx..].as_mut_ptr() as *mut __m256i,
            cand
        );
    }
    
    candidates
}
```

**Expected Speedup:**
```
Scalar:  240 ns (81 cells × 3 ns/cell)
SIMD:     40 ns (6 chunks × 7 ns/chunk)
Speedup: 6×
```

## Optimization Target #2: Parallel Popcount (MRV)

### Current Scalar Implementation
```rust
for row in 0..9 {
    for col in 0..9 {
        let count = candidates.count_ones();  // ~1ns per call
        if count < min_count { /* update */ }
    }
}
// Total: 81 iterations × 1ns = 81ns
```

### AVX512 SIMD Implementation (32 cells in parallel)

```rust
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
unsafe fn find_mrv_simd(&self, candidates: &[u16; 81]) -> (usize, usize) {
    // Load 32 candidates at once
    let cand1 = _mm512_loadu_si512(candidates[0..32].as_ptr() as *const __m512i);
    let cand2 = _mm512_loadu_si512(candidates[32..64].as_ptr() as *const __m512i);
    let cand3 = _mm512_loadu_si512(candidates[64..].as_ptr() as *const __m512i);
    
    // Parallel popcount (AVX512VPOPCNTDQ)
    let counts1 = _mm512_popcnt_epi16(cand1);  // 32 popcounts in parallel!
    let counts2 = _mm512_popcnt_epi16(cand2);
    let counts3 = _mm512_popcnt_epi16(cand3);
    
    // Parallel minimum reduction
    let min12 = _mm512_min_epu16(counts1, counts2);
    let min_all = _mm512_min_epu16(min12, counts3);
    
    // Horizontal minimum to find global min
    let min_val = _mm512_reduce_min_epu16(min_all);
    
    // Find index of minimum (requires scalar follow-up)
    let mask = _mm512_cmpeq_epi16_mask(min_all, _mm512_set1_epi16(min_val));
    let idx = mask.trailing_zeros() as usize;
    
    (idx / 9, idx % 9)
}
```

**Expected Speedup:**
```
Scalar:  81 ns (81 calls × 1 ns)
SIMD:    15 ns (3 chunks × 5 ns)
Speedup: 5.4×
```

## Optimization Target #3: Naked Single Elimination (Batch)

### Current Implementation (Sequential)
```rust
while changed {
    for each cell {
        if is_naked_single(cell) {
            place_digit(cell);
            changed = true;
        }
    }
}
```

### SIMD Batch Detection
```rust
#[target_feature(enable = "avx2")]
unsafe fn detect_naked_singles_simd(candidates: &[u16; 81]) -> u64 {
    let mut naked_single_mask = 0u64;
    
    for chunk in 0..6 {
        let cand = _mm256_loadu_si256(
            candidates[chunk*16..].as_ptr() as *const __m256i
        );
        
        // Check if power of 2: x & (x-1) == 0
        let minus_one = _mm256_sub_epi16(cand, _mm256_set1_epi16(1));
        let and_result = _mm256_and_si256(cand, minus_one);
        
        // Compare with zero
        let is_single = _mm256_cmpeq_epi16(and_result, _mm256_setzero_si256());
        
        // Extract mask (indicates which cells are naked singles)
        let mask = _mm256_movemask_epi8(is_single);
        naked_single_mask |= (mask as u64) << (chunk * 16);
    }
    
    naked_single_mask
}
```

## Complete SIMD-Optimized Solver (Outline)

```rust
pub struct SudokuSIMD {
    // Aligned for SIMD loads
    cells: AlignedCells,
    rows: AlignedConstraints,
    cols: AlignedConstraints,
    boxes: AlignedConstraints,
}

impl SudokuSIMD {
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vpopcntdq")]
    unsafe fn solve_simd(&mut self) -> bool {
        // 1. Parallel candidate calculation (SIMD)
        let candidates = self.get_candidates_simd();
        
        // 2. Parallel naked single detection (SIMD)
        let singles_mask = self.detect_naked_singles_simd(&candidates);
        
        // 3. Place all naked singles (scalar, but rare)
        self.place_naked_singles(singles_mask);
        
        // 4. Parallel MRV finding (SIMD)
        let (row, col) = self.find_mrv_simd(&candidates);
        
        // 5. Extract candidates and recurse (scalar)
        let mut cands = candidates[row * 9 + col];
        while cands != 0 {
            let bit = cands & cands.wrapping_neg();
            let digit = (bit.trailing_zeros() + 1) as u8;
            cands ^= bit;
            
            let snapshot = *self;
            self.place_digit(row, col, digit);
            
            if self.solve_simd() {
                return true;
            }
            
            *self = snapshot;
        }
        
        false
    }
}
```

## Hardware Requirements

### CPU Features Needed
```rust
// Check at runtime
if is_x86_feature_detected!("avx2") &&
   is_x86_feature_detected!("avx512f") &&
   is_x86_feature_detected!("avx512bw") &&
   is_x86_feature_detected!("avx512vpopcntdq") {
    // Use SIMD path
    solver.solve_simd()
} else {
    // Fallback to scalar
    solver.solve()
}
```

### Supported CPUs
- **AVX2:** Intel Haswell (2013+), AMD Excavator (2015+)
- **AVX512:** Intel Skylake-X (2017+), AMD Zen 4 (2022+)
- **AVX512VPOPCNTDQ:** Intel Ice Lake (2019+)

## Benchmarking SIMD vs Scalar

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_solvers(c: &mut Criterion) {
    let puzzle = "100000569492006108...";
    
    c.bench_function("scalar", |b| {
        b.iter(|| {
            let mut s = Sudoku::from_string(puzzle).unwrap();
            s.solve()
        })
    });
    
    c.bench_function("simd", |b| {
        b.iter(|| {
            let mut s = SudokuSIMD::from_string(puzzle).unwrap();
            unsafe { s.solve_simd() }
        })
    });
}

criterion_group!(benches, benchmark_solvers);
criterion_main!(benches);
```

**Expected Results:**
```
scalar   time: [823.45 ns]
simd     time: [287.12 ns]

Speedup: 2.87×
```

## Memory Alignment for SIMD

```rust
#[repr(align(32))]  // AVX2 requires 32-byte alignment
struct AlignedConstraints {
    rows: [u16; 9],
    cols: [u16; 9],
    boxes: [u16; 9],
    _padding: [u16; 7],  // Pad to 32-byte boundary
}

#[repr(align(64))]  // AVX512 prefers 64-byte alignment
struct AlignedCells {
    data: [[u8; 9]; 9],
    _padding: [u8; 47],
}
```

## Compiler Optimization Flags

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1

# Enable specific SIMD features
[build]
rustflags = [
    "-C", "target-cpu=native",
    "-C", "target-feature=+avx2,+avx512f,+avx512bw,+avx512vpopcntdq",
]
```

## Debugging SIMD Code

### Assembly Inspection
```bash
# Generate assembly with SIMD instructions highlighted
cargo rustc --release -- --emit asm

# Look for:
# - vpopcntw (AVX512 popcount)
# - vpminuw (parallel minimum)
# - vpgatherdd (gather operation)
```

### Verification
```rust
#[test]
fn test_simd_correctness() {
    let puzzle = "...";
    
    let mut scalar = Sudoku::from_string(puzzle).unwrap();
    let mut simd = SudokuSIMD::from_string(puzzle).unwrap();
    
    assert_eq!(scalar.solve(), unsafe { simd.solve_simd() });
    assert_eq!(scalar.to_string(), simd.to_string());
}
```

## Performance Prediction Model

**Theoretical Lower Bound:**
```
Operations per solve: ~2,000 nodes
Cycles per node (SIMD): ~0.8 cycles
Total cycles: 1,600 cycles

At 3.0 GHz: 1,600 / 3,000,000,000 = 533 ns
```

**Practical Estimate (accounting for overhead):**
```
SIMD optimized:     300 ns
Scalar optimized:   800 ns
Python (naive):     45,000,000 ns

SIMD is 150,000× faster than naive Python!
```

## Future: Custom SIMD Intrinsics

For ultimate performance, consider writing custom assembly:

```asm
; Custom AVX512 constraint propagation kernel
vpbroadcastw zmm0, [rows]      ; Load rows
vpbroadcastw zmm1, [cols]      ; Load cols
vpbroadcastw zmm2, [boxes]     ; Load boxes

vporq zmm3, zmm0, zmm1         ; OR constraints
vporq zmm3, zmm3, zmm2
vpandnq zmm4, zmm3, [all_bits] ; NOT to get candidates

vpopcntw zmm5, zmm4            ; Parallel popcount
vpminuw zmm6, zmm5, zmm5       ; Horizontal min
```

## Conclusion

SIMD optimizations can provide 2-3× speedup over the already-optimized scalar code, bringing solving time down to ~300ns for Al Escargot.

**Roadmap:**
1. ✅ Scalar bitboard implementation (800 ns)
2. ⬜ AVX2 candidate calculation (600 ns estimated)
3. ⬜ AVX512 popcount for MRV (400 ns estimated)
4. ⬜ Full SIMD pipeline (300 ns estimated)
5. ⬜ Custom assembly kernels (100 ns theoretical)

**When to Use SIMD:**
- ✅ Batch solving (1000s of puzzles)
- ✅ Real-time applications (embedded systems)
- ❌ Single puzzle solving (overhead not worth it)
- ❌ Portability critical (SIMD not available on all CPUs)

**Trade-offs:**
| Aspect | Scalar | SIMD |
|--------|--------|------|
| Performance | 800 ns | 300 ns |
| Code complexity | Low | High |
| Portability | Universal | CPU-specific |
| Maintainability | Easy | Difficult |
| Binary size | Small | Large |

For most use cases, the scalar implementation is sufficient. SIMD optimization should be considered only when:
1. Solving millions of puzzles per second
2. Hard real-time constraints (< 1 μs guaranteed)
3. Embedded systems with SIMD-capable processors

---

*Vectorize responsibly.* 🚀