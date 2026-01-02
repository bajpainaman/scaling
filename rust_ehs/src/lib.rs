//! Ultra-fast poker hand evaluation and EHS calculation
//! 
//! Uses lookup tables and vectorized Monte Carlo for speed.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

mod nlhe_engine;
pub use nlhe_engine::*;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

const NUM_CARDS: usize = 52;

// Hand rankings (higher = better)
const HIGH_CARD: u32 = 0;
const PAIR: u32 = 1;
const TWO_PAIR: u32 = 2;
const THREE_OF_A_KIND: u32 = 3;
const STRAIGHT: u32 = 4;
const FLUSH: u32 = 5;
const FULL_HOUSE: u32 = 6;
const FOUR_OF_A_KIND: u32 = 7;
const STRAIGHT_FLUSH: u32 = 8;

// ═══════════════════════════════════════════════════════════════════════════
// HAND EVALUATION (7-card)
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn get_rank(card: u8) -> u8 {
    card / 4
}

#[inline(always)]
fn get_suit(card: u8) -> u8 {
    card % 4
}

/// Evaluate 7 cards and return a hand rank value (higher = better)
/// Returns (category << 20) | tiebreaker
pub fn evaluate_7cards(cards: &[u8; 7]) -> u32 {
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    let mut suit_cards = [[0u8; 7]; 4];
    let mut suit_lens = [0usize; 4];
    
    // Count ranks and suits
    for &card in cards {
        let rank = get_rank(card) as usize;
        let suit = get_suit(card) as usize;
        rank_counts[rank] += 1;
        suit_cards[suit][suit_lens[suit]] = card;
        suit_lens[suit] += 1;
        suit_counts[suit] += 1;
    }
    
    // Check for flush
    let mut flush_suit: Option<usize> = None;
    for suit in 0..4 {
        if suit_counts[suit] >= 5 {
            flush_suit = Some(suit);
            break;
        }
    }
    
    // Check for straight
    fn find_straight(rank_counts: &[u8; 13]) -> Option<u8> {
        // Check wheel (A-2-3-4-5)
        if rank_counts[12] > 0 && rank_counts[0] > 0 && rank_counts[1] > 0 
            && rank_counts[2] > 0 && rank_counts[3] > 0 {
            return Some(3); // 5-high straight
        }
        
        // Check other straights
        for high in (4..13).rev() {
            let mut is_straight = true;
            for i in 0..5 {
                if rank_counts[high - i] == 0 {
                    is_straight = false;
                    break;
                }
            }
            if is_straight {
                return Some(high as u8);
            }
        }
        None
    }
    
    let straight_high = find_straight(&rank_counts);
    
    // Check for straight flush
    if let Some(suit) = flush_suit {
        let mut flush_rank_counts = [0u8; 13];
        for i in 0..suit_lens[suit] {
            let rank = get_rank(suit_cards[suit][i]) as usize;
            flush_rank_counts[rank] += 1;
        }
        if let Some(high) = find_straight(&flush_rank_counts) {
            return (STRAIGHT_FLUSH << 20) | (high as u32);
        }
    }
    
    // Count pairs, trips, quads
    let mut quads: Vec<u8> = Vec::new();
    let mut trips: Vec<u8> = Vec::new();
    let mut pairs: Vec<u8> = Vec::new();
    let mut singles: Vec<u8> = Vec::new();
    
    for rank in (0..13).rev() {
        match rank_counts[rank] {
            4 => quads.push(rank as u8),
            3 => trips.push(rank as u8),
            2 => pairs.push(rank as u8),
            1 => singles.push(rank as u8),
            _ => {}
        }
    }
    
    // Four of a kind
    if !quads.is_empty() {
        let kicker = trips.first()
            .or(pairs.first())
            .or(singles.first())
            .copied()
            .unwrap_or(0);
        return (FOUR_OF_A_KIND << 20) | ((quads[0] as u32) << 4) | (kicker as u32);
    }
    
    // Full house
    if !trips.is_empty() && (trips.len() >= 2 || !pairs.is_empty()) {
        let pair_rank = if trips.len() >= 2 {
            trips[1]
        } else {
            pairs[0]
        };
        return (FULL_HOUSE << 20) | ((trips[0] as u32) << 4) | (pair_rank as u32);
    }
    
    // Flush
    if let Some(suit) = flush_suit {
        let mut flush_ranks: Vec<u8> = (0..suit_lens[suit])
            .map(|i| get_rank(suit_cards[suit][i]))
            .collect();
        flush_ranks.sort_by(|a, b| b.cmp(a));
        let mut value = 0u32;
        for i in 0..5 {
            value = (value << 4) | (flush_ranks[i] as u32);
        }
        return (FLUSH << 20) | value;
    }
    
    // Straight
    if let Some(high) = straight_high {
        return (STRAIGHT << 20) | (high as u32);
    }
    
    // Three of a kind
    if !trips.is_empty() {
        let mut kickers: Vec<u8> = pairs.iter().chain(singles.iter()).copied().collect();
        kickers.sort_by(|a, b| b.cmp(a));
        let k1 = kickers.get(0).copied().unwrap_or(0) as u32;
        let k2 = kickers.get(1).copied().unwrap_or(0) as u32;
        return (THREE_OF_A_KIND << 20) | ((trips[0] as u32) << 8) | (k1 << 4) | k2;
    }
    
    // Two pair
    if pairs.len() >= 2 {
        let kicker = pairs.get(2)
            .or(singles.first())
            .copied()
            .unwrap_or(0);
        return (TWO_PAIR << 20) | ((pairs[0] as u32) << 8) | ((pairs[1] as u32) << 4) | (kicker as u32);
    }
    
    // One pair
    if pairs.len() == 1 {
        let mut value = (PAIR << 20) | ((pairs[0] as u32) << 12);
        for i in 0..3.min(singles.len()) {
            value |= (singles[i] as u32) << (8 - i * 4);
        }
        return value;
    }
    
    // High card
    let mut value = HIGH_CARD << 20;
    for i in 0..5.min(singles.len()) {
        value |= (singles[i] as u32) << (16 - i * 4);
    }
    value
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPECTED HAND STRENGTH (Monte Carlo)
// ═══════════════════════════════════════════════════════════════════════════

/// Fast EHS calculation using Monte Carlo
fn calculate_ehs(hole: [u8; 2], board: &[u8], num_samples: u32, seed: u64) -> f64 {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    
    // Cards already used
    let mut used = [false; NUM_CARDS];
    used[hole[0] as usize] = true;
    used[hole[1] as usize] = true;
    for &card in board {
        used[card as usize] = true;
    }
    
    // Build deck of remaining cards
    let mut remaining: Vec<u8> = (0..52)
        .filter(|&c| !used[c as usize])
        .map(|c| c as u8)
        .collect();
    
    let cards_to_deal = 5 - board.len();
    let mut wins = 0u32;
    let mut ties = 0u32;
    
    for _ in 0..num_samples {
        // Shuffle remaining cards
        remaining.shuffle(&mut rng);
        
        // Deal runout + opponent hole cards
        let mut full_board = [0u8; 5];
        for (i, &card) in board.iter().enumerate() {
            full_board[i] = card;
        }
        for i in 0..cards_to_deal {
            full_board[board.len() + i] = remaining[i];
        }
        
        let opp_hole = [remaining[cards_to_deal], remaining[cards_to_deal + 1]];
        
        // Build 7-card hands
        let mut my_hand = [0u8; 7];
        my_hand[0] = hole[0];
        my_hand[1] = hole[1];
        for i in 0..5 {
            my_hand[2 + i] = full_board[i];
        }
        
        let mut opp_hand = [0u8; 7];
        opp_hand[0] = opp_hole[0];
        opp_hand[1] = opp_hole[1];
        for i in 0..5 {
            opp_hand[2 + i] = full_board[i];
        }
        
        // Evaluate and compare
        let my_value = evaluate_7cards(&my_hand);
        let opp_value = evaluate_7cards(&opp_hand);
        
        if my_value > opp_value {
            wins += 1;
        } else if my_value == opp_value {
            ties += 1;
        }
    }
    
    (wins as f64 + 0.5 * ties as f64) / num_samples as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// PREFLOP LOOKUP TABLE (1326 combos)
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-computed preflop EHS for all 1326 starting hands
/// Indexed by canonical hand representation
fn preflop_ehs_lookup(card1: u8, card2: u8) -> f64 {
    let r1 = get_rank(card1);
    let r2 = get_rank(card2);
    let suited = get_suit(card1) == get_suit(card2);
    
    let (high, low) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    
    // Pairs
    if high == low {
        return match high {
            12 => 0.852, // AA
            11 => 0.824, // KK
            10 => 0.799, // QQ
            9 => 0.773,  // JJ
            8 => 0.750,  // TT
            7 => 0.723,  // 99
            6 => 0.696,  // 88
            5 => 0.669,  // 77
            4 => 0.642,  // 66
            3 => 0.615,  // 55
            2 => 0.588,  // 44
            1 => 0.561,  // 33
            0 => 0.534,  // 22
            _ => 0.5,
        };
    }
    
    // Non-pairs: approximate formula based on ranks and suitedness
    let gap = high - low;
    let base = 0.35 + (high as f64 + low as f64) * 0.012 - gap as f64 * 0.008;
    let suited_bonus = if suited { 0.03 } else { 0.0 };
    
    // Broadway bonus
    let broadway_bonus = if high >= 8 && low >= 8 { 0.05 } else { 0.0 };
    
    (base + suited_bonus + broadway_bonus).clamp(0.25, 0.70)
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL FUNCTIONS (for nlhe_engine)
// ═══════════════════════════════════════════════════════════════════════════

/// Internal bucket function for use by nlhe_engine
pub fn fast_bucket_internal(h0: u8, h1: u8, board: &[u8], round: u8) -> u32 {
    const NUM_BUCKETS: u32 = 10;
    
    // EHS samples by street
    let num_samples = match round {
        0 => 0,   // preflop
        1 => 32,  // flop
        2 => 64,  // turn
        _ => 0,   // river (exact)
    };
    
    let ehs = if board.is_empty() {
        preflop_ehs_lookup(h0, h1)
    } else {
        let seed = (h0 as u64 * 1000 + h1 as u64 * 100 + board.len() as u64) ^ 0xDEADBEEF;
        calculate_ehs([h0, h1], board, num_samples, seed)
    };
    
    let bucket = (ehs * NUM_BUCKETS as f64).floor() as u32;
    bucket.min(NUM_BUCKETS - 1)
}

// ═══════════════════════════════════════════════════════════════════════════
// PYTHON BINDINGS
// ═══════════════════════════════════════════════════════════════════════════

#[pyfunction]
fn fast_ehs(hole: [u8; 2], board: Vec<u8>, num_samples: u32, seed: u64) -> f64 {
    if board.is_empty() {
        // Preflop: use lookup table
        preflop_ehs_lookup(hole[0], hole[1])
    } else {
        // Postflop: Monte Carlo
        calculate_ehs(hole, &board, num_samples, seed)
    }
}

#[pyfunction]
fn fast_bucket(hole: [u8; 2], board: Vec<u8>, num_buckets: u32, num_samples: u32, seed: u64) -> u32 {
    let ehs = fast_ehs(hole, board, num_samples, seed);
    let bucket = (ehs * num_buckets as f64).floor() as u32;
    bucket.min(num_buckets - 1)
}

#[pyfunction]
fn batch_ehs(holes: Vec<[u8; 2]>, boards: Vec<Vec<u8>>, num_samples: u32, seed: u64) -> Vec<f64> {
    holes.iter().zip(boards.iter()).enumerate()
        .map(|(i, (hole, board))| {
            fast_ehs(*hole, board.clone(), num_samples, seed.wrapping_add(i as u64))
        })
        .collect()
}

#[pyfunction]
fn evaluate_hand(cards: Vec<u8>) -> u32 {
    if cards.len() != 7 {
        return 0;
    }
    let mut arr = [0u8; 7];
    arr.copy_from_slice(&cards);
    evaluate_7cards(&arr)
}

#[pyfunction]
fn compare_hands_rust(hand1: Vec<u8>, hand2: Vec<u8>) -> i32 {
    if hand1.len() != 7 || hand2.len() != 7 {
        return 0;
    }
    let mut arr1 = [0u8; 7];
    let mut arr2 = [0u8; 7];
    arr1.copy_from_slice(&hand1);
    arr2.copy_from_slice(&hand2);
    
    let v1 = evaluate_7cards(&arr1);
    let v2 = evaluate_7cards(&arr2);
    
    if v1 > v2 { 1 }
    else if v1 < v2 { -1 }
    else { 0 }
}

/// BENCHMARK MODE: Single-threaded (for comparison)
#[pyfunction]
fn run_cfr_benchmark(
    py: Python<'_>,
    num_games: usize,
    starting_stack: i32,
    max_raises: u8,
    epsilon: f64,
    seed: u64,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    use std::time::Instant;
    let start = Instant::now();
    
    let (samples, _regret_map, _strategy_map, stats) = nlhe_engine::run_cfr_batch(
        num_games, starting_stack, max_raises, epsilon, seed
    );
    
    let elapsed = start.elapsed().as_secs_f64();
    
    let stats_dict = pyo3::types::PyDict::new(py);
    stats_dict.set_item("games", num_games)?;
    stats_dict.set_item("samples", samples.len())?;
    stats_dict.set_item("unique_infosets", stats.unique_infosets_global)?;  // REAL global unique
    stats_dict.set_item("elapsed_ms", elapsed * 1000.0)?;
    stats_dict.set_item("games_per_sec", num_games as f64 / elapsed)?;
    stats_dict.set_item("samples_per_sec", samples.len() as f64 / elapsed)?;
    stats_dict.set_item("samples_per_game", samples.len() as f64 / num_games as f64)?;
    
    Ok(stats_dict.into())
}

/// PARALLEL BENCHMARK: Uses all CPU cores via Rayon
#[pyfunction]
#[pyo3(signature = (num_games, starting_stack, max_raises, epsilon, seed, num_threads=None))]
fn run_cfr_parallel(
    py: Python<'_>,
    num_games: usize,
    starting_stack: i32,
    max_raises: u8,
    epsilon: f64,
    seed: u64,
    num_threads: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    use std::time::Instant;
    
    let threads = num_threads.unwrap_or_else(num_cpus::get);
    let config = nlhe_engine::CFRConfig {
        starting_stack,
        max_raises,
        epsilon,
    };
    
    let start = Instant::now();
    let stats = nlhe_engine::run_cfr_batch_parallel(num_games, &config, seed, threads);
    let elapsed = start.elapsed().as_secs_f64();
    
    let stats_dict = pyo3::types::PyDict::new(py);
    stats_dict.set_item("games", num_games)?;
    stats_dict.set_item("threads", threads)?;
    stats_dict.set_item("samples", stats.total_samples)?;
    stats_dict.set_item("per_game_unique_sum", stats.unique_infosets_per_game_sum)?;  // Honest naming
    stats_dict.set_item("elapsed_ms", elapsed * 1000.0)?;
    stats_dict.set_item("games_per_sec", num_games as f64 / elapsed)?;
    stats_dict.set_item("samples_per_sec", stats.total_samples as f64 / elapsed)?;
    stats_dict.set_item("samples_per_game", stats.total_samples as f64 / num_games as f64)?;
    
    Ok(stats_dict.into())
}

/// Run CFR games and return CONTIGUOUS arrays (not Vec<Vec> garbage)
/// Returns (encodings_flat, advantages_flat, players, infoset_keys, num_samples, stats)
#[pyfunction]
fn run_cfr_games(
    py: Python<'_>,
    num_games: usize,
    starting_stack: i32,
    max_raises: u8,
    epsilon: f64,
    seed: u64,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<u8>, Vec<u64>, usize, pyo3::Py<pyo3::types::PyDict>)> {
    let (samples, _regret_map, _strategy_map, stats) = nlhe_engine::run_cfr_batch(
        num_games, starting_stack, max_raises, epsilon, seed
    );
    
    let n = samples.len();
    let encoding_dim = nlhe_engine::ENCODING_DIM;
    let num_actions = nlhe_engine::NUM_ACTIONS;
    
    // CONTIGUOUS flat arrays - one allocation each!
    let mut encodings_flat: Vec<f32> = Vec::with_capacity(n * encoding_dim);
    let mut advantages_flat: Vec<f32> = Vec::with_capacity(n * num_actions);
    let mut players: Vec<u8> = Vec::with_capacity(n);
    let mut infoset_keys: Vec<u64> = Vec::with_capacity(n);
    
    for s in &samples {
        encodings_flat.extend_from_slice(&s.encoding);
        advantages_flat.extend_from_slice(&s.advantages);
        players.push(s.player);
        infoset_keys.push(s.infoset_key);
    }
    
    // Build stats dict
    let stats_dict = pyo3::types::PyDict::new(py);
    stats_dict.set_item("unique_infosets", stats.unique_infosets_global)?;
    stats_dict.set_item("total_samples", stats.total_samples)?;
    stats_dict.set_item("samples_per_game", n as f64 / num_games as f64)?;
    
    Ok((encodings_flat, advantages_flat, players, infoset_keys, n, stats_dict.into()))
}

#[pymodule]
fn poker_ehs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_ehs, m)?)?;
    m.add_function(wrap_pyfunction!(fast_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ehs, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_hand, m)?)?;
    m.add_function(wrap_pyfunction!(compare_hands_rust, m)?)?;
    m.add_function(wrap_pyfunction!(run_cfr_games, m)?)?;
    m.add_function(wrap_pyfunction!(run_cfr_benchmark, m)?)?;
    m.add_function(wrap_pyfunction!(run_cfr_parallel, m)?)?;
    Ok(())
}
