//! Pure Rust benchmark - no Python overhead
//! Run with: cargo run --release --bin benchmark

use std::time::Instant;

// Import from library
use poker_ehs::nlhe_engine::{run_cfr_batch_parallel, CFRConfig};

fn main() {
    println!("🦀 PURE RUST NLHE CFR BENCHMARK");
    println!("================================");
    println!("No Python. No FFI. Pure speed.\n");

    let config = CFRConfig {
        starting_stack: 100,
        max_raises: 2,
        epsilon: 0.1,
    };

    // Warm up
    println!("Warming up...");
    let _ = run_cfr_batch_parallel(1000, &config, 42, 1);

    // Benchmark different scales
    for &num_games in &[10_000, 100_000, 500_000, 1_000_000] {
        // Single-threaded
        let start = Instant::now();
        let stats = run_cfr_batch_parallel(num_games, &config, 42, 1);
        let elapsed_st = start.elapsed().as_secs_f64();

        // Multi-threaded (all cores)
        let num_threads = num_cpus::get();
        let start = Instant::now();
        let stats_mt = run_cfr_batch_parallel(num_games, &config, 42, num_threads);
        let elapsed_mt = start.elapsed().as_secs_f64();

        let speedup = elapsed_st / elapsed_mt;

        println!(
            "{:>7}k games | ST: {:>6.0} g/s | MT({:>2}): {:>7.0} g/s | {:.1}x speedup | {} infosets",
            num_games / 1000,
            num_games as f64 / elapsed_st,
            num_threads,
            num_games as f64 / elapsed_mt,
            speedup,
            stats_mt.unique_infosets,
        );
    }

    println!("\n✅ Benchmark complete.");
}

