# Apex Sudoku: High-Performance Zero-Allocation Solver in Rust

**Apex Sudoku**, modern donanım mimarileri için optimize edilmiş, bellek tahsisi yapmayan (zero-allocation) ve mikrosaniye ölçeğinde çalışan ekstrem bir Sudoku çözücüdür. Düşük seviyeli bit manipülasyonu ve gelişmiş arama algoritmaları kullanılarak en karmaşık bulmacaları bile anında çözer.

## 🚀 Öne Çıkan Özellikler

- **Ekstrem Hız:** Bilinen en zor Sudoku olan "Al Escargot" bulmacasını **~12 mikrosaniyede** çözer.
- **Sıfır Bellek Tahsisi (Zero-Allocation):** Çalışma anında Heap bellek kullanmaz; tüm süreç işlemci önbelleği (Cache) ve Stack üzerinde yürütülür.
- **Bitboard Mimarisi:** Sudoku kısıtlamaları 16-bitlik tam sayılar olarak saklanır. Arama işlemleri düşük seviyeli bit işlemleriyle (`AND`, `OR`, `NOT`) saniyeler içinde değil, nanosaniyeler içinde gerçekleşir.
- **MRV (Minimum Remaining Values) Heuristic:** En az seçeneği kalan hücreyi önceliklendirerek arama uzayını trilyonlarca kereden birkaç bin denemeye indirir.
- **Bellek Güvenliği:** Rust dilinin sunduğu bellek güvenliği (memory safety) garantileriyle, hızdan ödün vermeden güvenli çalışma sağlar.

## 📊 Performans Değerleri

*Testler Apple M1 Silicon mimarisinde `RUSTFLAGS="-C target-cpu=native"` ile yapılmıştır.*

| Bulmaca (Benchmark) | Ortalama Süre | Saniyelik Çözüm (Throughput) |
| :--- | :--- | :--- |
| **Al Escargot (En Zor)** | **12,300 ns (12.3 μs)** | ~81,000 çöz/sn |
| **Hardest 2012** | **521,000 ns (0.5 ms)** | ~1,900 çöz/sn |
| **Mystery Hard** | **73 ns** | **~13.6 Milyon çöz/sn** |

## 🛠️ Kurulum ve Çalıştırma

Projenin en yüksek performansta çalışması için işlemciye özel optimizasyonlarla derlenmesi önerilir:

```bash
# İşlemciye özel optimizasyonlarla derleyin ve çalıştırın
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

## 🏗️ Mimari Yaklaşım

Apex Sudoku, aşağıdaki düşük seviyeli optimizasyon tekniklerini kullanır:

1.  **Bit-Level Pruning:** Arama ağacındaki dallanmaları, bit düzeyinde kısıtlama kontrolü yaparak daha oluşmadan budar.
2.  **In-Place State Management:** Bulmaca durumu kopyalanmaz. Sadece yapılan değişiklikler (delta) üzerinden geri izleme (backtracking) yapılır.
3.  **Cache Locality:** Veri yapısı L1 Cache içine sığacak kadar kompakttır (250 byte altı), bu da bellek gecikmesini sıfıra indirir.

## 📖 Kullanım

```rust
use sudoku::Sudoku;

fn main() {
    let puzzle = "100007060900020008080500000000305070020010000800000400004000000000460010030900005";
    
    if let Some(mut solver) = Sudoku::from_string(puzzle) {
        if solver.solve() {
            solver.print();
            println!("Çözüm: {}", solver.to_string());
        }
    }
}
```

## 📜 Lisans

MIT License - Ayrıntılar için LICENSE dosyasına bakınız.
