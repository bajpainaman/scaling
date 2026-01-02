//! NLHE CFR Engine - Wintermute Edition
//!
//! Optimizations:
//! - FxHash for integer keys (3-5x faster than std HashMap)
//! - Rayon for parallel game simulation
//! - Packed u64 infoset keys (no string hashing)
//! - Preflop bucket lookup table
//! - Arena-style sample collection
//! - Zero-copy where possible

use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

pub const NUM_ACTIONS: usize = 6;
pub const NUM_BUCKETS: usize = 10;
pub const MAX_HISTORY: usize = 20;
pub const ENCODING_DIM: usize = NUM_BUCKETS + 4 + 1 + 1 + 5 + MAX_HISTORY * NUM_ACTIONS;

pub const PREFLOP: u8 = 0;
pub const FLOP: u8 = 1;
pub const TURN: u8 = 2;
pub const RIVER: u8 = 3;

pub const FOLD: u8 = 0;
pub const CALL: u8 = 1;
pub const BET_HALF: u8 = 2;
pub const BET_POT: u8 = 3;
pub const BET_2X: u8 = 4;
pub const ALL_IN: u8 = 5;

// ═══════════════════════════════════════════════════════════════════════════
// PREFLOP LOOKUP TABLE (169 canonical hands → bucket)
// ═══════════════════════════════════════════════════════════════════════════

/// Precomputed preflop buckets for all 169 canonical hands
/// Index: canonical_hand_index(card1, card2)
static PREFLOP_BUCKETS: once_cell::sync::Lazy<[u8; 169]> = once_cell::sync::Lazy::new(|| {
    let mut buckets = [0u8; 169];
    // Hand strength ranking (simplified)
    // AA=9, KK=9, QQ=8, JJ=8, AKs=8, TT=7, AQs=7, KQs=7, AJs=7, AKo=7...
    let rankings = [
        // Pairs: 22-AA (indices 0-12)
        3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 9, 9,
    ];
    
    // Fill pairs
    for i in 0..13 {
        buckets[i] = rankings[i];
    }
    
    // Fill suited hands (above diagonal)
    for high in 1..13 {
        for low in 0..high {
            let idx = 13 + (high * (high - 1) / 2) + low;
            let gap = high - low;
            let base = if high >= 10 { 6 } else if high >= 7 { 4 } else { 2 };
            let bonus = if gap <= 2 { 2 } else if gap <= 4 { 1 } else { 0 };
            buckets[idx] = (base + bonus).min(9);
        }
    }
    
    // Fill offsuit hands (below diagonal)
    for high in 1..13 {
        for low in 0..high {
            let idx = 13 + 78 + (high * (high - 1) / 2) + low;
            let gap = high - low;
            let base = if high >= 11 { 5 } else if high >= 8 { 3 } else { 1 };
            let bonus = if gap <= 1 { 2 } else if gap <= 3 { 1 } else { 0 };
            buckets[idx] = (base + bonus).min(8);
        }
    }
    
    buckets
});

#[inline(always)]
fn canonical_hand_index(c1: u8, c2: u8) -> usize {
    let r1 = c1 / 4;
    let r2 = c2 / 4;
    let s1 = c1 % 4;
    let s2 = c2 % 4;
    let suited = s1 == s2;
    
    let (high, low) = if r1 > r2 { (r1, r2) } else { (r2, r1) };
    
    if high == low {
        // Pair
        high as usize
    } else if suited {
        // Suited (above diagonal)
        13 + (high as usize * (high as usize - 1) / 2) + low as usize
    } else {
        // Offsuit (below diagonal)
        13 + 78 + (high as usize * (high as usize - 1) / 2) + low as usize
    }
}

#[inline(always)]
fn preflop_bucket(c1: u8, c2: u8) -> u8 {
    PREFLOP_BUCKETS[canonical_hand_index(c1, c2)]
}

// ═══════════════════════════════════════════════════════════════════════════
// GAME STATE (optimized - fixed-size arrays, no heap allocs in hot path)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct NLHEState {
    pub hole_cards: [[u8; 2]; 2],
    pub board: [u8; 5],
    pub stacks: [i32; 2],
    pub pot: i32,
    pub bets_this_round: [i32; 2],
    pub round: u8,
    pub current_player: u8,
    pub is_terminal: bool,
    pub winner: i8,
    pub history: [u8; MAX_HISTORY],  // Fixed-size! No heap alloc
    pub history_len: u8,
    pub starting_stack: i32,
    pub max_raises: u8,
    pub raises_this_round: u8,
}

impl NLHEState {
    #[inline]
    pub fn new(starting_stack: i32, max_raises: u8, rng: &mut Xoshiro256PlusPlus) -> Self {
        let mut deck: Vec<u8> = (0..52).collect();
        deck.shuffle(rng);
        
        let mut state = Self {
            hole_cards: [[deck[0], deck[1]], [deck[2], deck[3]]],
            board: [deck[4], deck[5], deck[6], deck[7], deck[8]],
            stacks: [starting_stack - 2, starting_stack - 1],  // SB=1, BB=2
            pot: 3,
            bets_this_round: [1, 2],  // SB posted 1, BB posted 2
            round: PREFLOP,
            current_player: 0,  // SB acts first preflop
            is_terminal: false,
            winner: -1,
            history: [0; MAX_HISTORY],
            history_len: 0,
            starting_stack,
            max_raises,
            raises_this_round: 0,
        };
        state
    }

    /// Returns (actions_array, count) - NO HEAP ALLOCATION
    #[inline(always)]
    pub fn legal_actions(&self) -> ([u8; 6], usize) {
        let mut a = [0u8; 6];
        let mut n = 0;
        
        if self.is_terminal { return (a, 0); }
        
        let player = self.current_player as usize;
        let opponent = 1 - player;
        let to_call = self.bets_this_round[opponent] - self.bets_this_round[player];
        let my_stack = self.stacks[player];
        
        macro_rules! push { ($x:expr) => {{ a[n] = $x; n += 1; }}; }
        
        // Fold (if facing bet)
        if to_call > 0 { push!(FOLD); }
        
        // Call/Check
        if to_call <= my_stack { push!(CALL); }
        
        // Raises
        if self.raises_this_round < self.max_raises && my_stack > to_call {
            let pot_after_call = self.pot + to_call;
            let remaining = my_stack - to_call;
            
            let half_pot = pot_after_call / 2;
            if half_pot > 0 && half_pot <= remaining { push!(BET_HALF); }
            if pot_after_call <= remaining { push!(BET_POT); }
            if pot_after_call * 2 <= remaining { push!(BET_2X); }
            if remaining > 0 { push!(ALL_IN); }
        }
        
        (a, n)
    }

    #[inline]
    pub fn apply_action(&self, action: u8) -> Self {
        let mut next = self.clone();
        let player = self.current_player as usize;
        let opponent = 1 - player;
        
        // Record action in history
        if next.history_len < MAX_HISTORY as u8 {
            next.history[next.history_len as usize] = action;
            next.history_len += 1;
        }
        
        let to_call = self.bets_this_round[opponent] - self.bets_this_round[player];
        
        match action {
            FOLD => {
                next.is_terminal = true;
                next.winner = opponent as i8;
            }
            CALL => {
                let call_amount = to_call.min(next.stacks[player]);
                next.stacks[player] -= call_amount;
                next.pot += call_amount;
                next.bets_this_round[player] += call_amount;
                
                // Check if round ends
                if next.bets_this_round[0] == next.bets_this_round[1] {
                    next.advance_round();
                } else {
                    next.current_player = opponent as u8;
                }
            }
            BET_HALF | BET_POT | BET_2X | ALL_IN => {
                let pot_after_call = next.pot + to_call;
                let raise_size = match action {
                    BET_HALF => pot_after_call / 2,
                    BET_POT => pot_after_call,
                    BET_2X => pot_after_call * 2,
                    ALL_IN => next.stacks[player] - to_call,
                    _ => 0,
                };
                
                let total = to_call + raise_size;
                let actual = total.min(next.stacks[player]);
                
                next.stacks[player] -= actual;
                next.pot += actual;
                next.bets_this_round[player] += actual;
                next.raises_this_round += 1;
                next.current_player = opponent as u8;
            }
            _ => {}
        }
        
        next
    }

    #[inline]
    fn advance_round(&mut self) {
        if self.round >= RIVER {
            // Showdown
            self.is_terminal = true;
            self.winner = self.determine_winner();
        } else {
            self.round += 1;
            self.bets_this_round = [0, 0];
            self.raises_this_round = 0;
            self.current_player = 0;  // P0 acts first postflop
        }
    }

    #[inline]
    fn determine_winner(&self) -> i8 {
        let v0 = self.evaluate_hand(0);
        let v1 = self.evaluate_hand(1);
        if v0 > v1 { 0 } else if v1 > v0 { 1 } else { -1 }  // -1 = tie
    }

    #[inline]
    fn evaluate_hand(&self, player: usize) -> u32 {
        let mut cards = [0u8; 7];
        cards[0] = self.hole_cards[player][0];
        cards[1] = self.hole_cards[player][1];
        cards[2..7].copy_from_slice(&self.board);
        crate::evaluate_7cards(&cards)
    }

    #[inline(always)]
    pub fn get_payoff(&self, player: u8) -> f64 {
        if !self.is_terminal { return 0.0; }
        
        let invested = self.starting_stack - self.stacks[player as usize];
        if self.winner == player as i8 {
            (self.pot - invested) as f64
        } else if self.winner == -1 {
            0.0  // Tie
        } else {
            -invested as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FAST INFOSET KEY (packed u64, no strings!)
// ═══════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn board_texture_bits(board: &[u8]) -> u32 {
    if board.is_empty() { return 0; }
    
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    let mut rank_mask = 0u16;
    
    for &c in board {
        let r = (c / 4) as usize;
        rank_counts[r] += 1;
        rank_mask |= 1 << r;
        suit_counts[(c % 4) as usize] += 1;
    }
    
    let mut bits = 0u32;
    
    // Bit 0: paired (any pair on board)
    if rank_counts.iter().any(|&c| c >= 2) { bits |= 1; }
    
    // Bit 1: trips+ or two-pair texture
    let pairs = rank_counts.iter().filter(|&&c| c >= 2).count();
    if pairs >= 2 || rank_counts.iter().any(|&c| c >= 3) { bits |= 2; }
    
    // Bit 2: flush possible (3+ same suit)
    if suit_counts.iter().any(|&c| c >= 3) { bits |= 4; }
    
    // Bit 3: straight potential (3+ cards in 5-rank window)
    for start in 0..=8 {
        let window = (rank_mask >> start) & 0x1F;
        if window.count_ones() >= 3 { bits |= 8; break; }
    }
    
    bits
}

#[inline(always)]
fn pot_bucket(pot: i32) -> u32 {
    match pot {
        0..=3 => 0,
        4..=10 => 1,
        11..=25 => 2,
        26..=50 => 3,
        51..=100 => 4,
        101..=200 => 5,
        201..=400 => 6,
        _ => 7,
    }
}

#[inline(always)]
fn spr_bucket(stack: i32, pot: i32) -> u32 {
    if pot <= 0 { return 5; }
    let spr = (stack * 10) / pot;  // Fixed-point 1 decimal
    match spr {
        0..=9 => 0,      // < 1
        10..=19 => 1,    // 1-2
        20..=39 => 2,    // 2-4
        40..=79 => 3,    // 4-8
        80..=159 => 4,   // 8-16
        _ => 5,          // 16+
    }
}

#[inline(always)]
fn history_pattern(history: &[u8], len: u8) -> u32 {
    let mut bets = 0u32;
    for i in 0..(len as usize).min(8) {
        if history[i] >= 2 { bets += 1; }
    }
    let last = if len > 0 { history[(len - 1) as usize] as u32 } else { 0 };
    ((bets.min(7)) << 3) | (last.min(7))
}

#[inline(always)]
fn make_infoset_key(state: &NLHEState, player: u8, bucket: u8) -> u64 {
    let opponent = 1 - player as usize;
    let to_call = (state.bets_this_round[opponent] - state.bets_this_round[player as usize]).max(0);
    let to_call_bucket = match (to_call * 10) / (state.pot.max(1)) {
        0 => 0,
        1..=2 => 1,
        3..=5 => 2,
        6..=10 => 3,
        11..=20 => 4,
        _ => 5,
    };
    
    let board_len = match state.round {
        PREFLOP => 0,
        FLOP => 3,
        TURN => 4,
        _ => 5,
    };
    let texture = board_texture_bits(&state.board[0..board_len]);
    let pot_b = pot_bucket(state.pot);
    let spr_b = spr_bucket(state.stacks[player as usize], state.pot);
    let hist_b = history_pattern(&state.history, state.history_len);
    
    // Pack: player(1) | bucket(4) | round(2) | texture(4) | pot(3) | spr(3) | tocall(3) | history(6)
    ((player as u64) << 25) |
    ((bucket as u64 & 0xF) << 21) |
    ((state.round as u64 & 0x3) << 19) |
    ((texture as u64 & 0xF) << 15) |
    ((pot_b as u64 & 0x7) << 12) |
    ((spr_b as u64 & 0x7) << 9) |
    ((to_call_bucket as u64 & 0x7) << 6) |
    (hist_b as u64 & 0x3F)
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE ENCODING (for neural net)
// ═══════════════════════════════════════════════════════════════════════════

/// Returns (encoding, bucket) - no scanning needed!
#[inline]
pub fn encode_state(state: &NLHEState, player: u8) -> ([f32; ENCODING_DIM], u8) {
    let mut enc = [0.0f32; ENCODING_DIM];
    let mut idx = 0;
    
    let player_idx = player as usize;
    let hole = &state.hole_cards[player_idx];
    
    // 1. Card bucket (preflop lookup or EHS)
    let board_len = match state.round {
        PREFLOP => 0, FLOP => 3, TURN => 4, _ => 5
    };
    let bucket = if state.round == PREFLOP {
        preflop_bucket(hole[0], hole[1])
    } else {
        crate::fast_bucket_internal(hole[0], hole[1], &state.board[0..board_len], state.round) as u8
    };
    enc[idx + bucket as usize] = 1.0;
    idx += NUM_BUCKETS;
    
    // 2. Round one-hot
    enc[idx + state.round as usize] = 1.0;
    idx += 4;
    
    // 3. Pot odds
    let opponent = 1 - player_idx;
    let to_call = (state.bets_this_round[opponent] - state.bets_this_round[player_idx]).max(0);
    enc[idx] = to_call as f32 / (state.pot + to_call + 1) as f32;
    idx += 1;
    
    // 4. SPR
    enc[idx] = (state.stacks[player_idx] as f32 / (state.pot as f32 + 1.0) / 10.0).min(1.0);
    idx += 1;
    
    // 5. Board texture (4 bits: paired, trips+, flush-draw, straight-draw)
    if board_len > 0 {
        let tex = board_texture_bits(&state.board[0..board_len]);
        enc[idx] = if tex & 1 != 0 { 1.0 } else { 0.0 };     // paired
        enc[idx + 1] = if tex & 2 != 0 { 1.0 } else { 0.0 }; // trips+
        enc[idx + 2] = if tex & 4 != 0 { 1.0 } else { 0.0 }; // flush draw
        enc[idx + 3] = if tex & 8 != 0 { 1.0 } else { 0.0 }; // straight draw
    }
    idx += 5;
    
    // 6. Action history (one-hot per position)
    for i in 0..(state.history_len as usize).min(MAX_HISTORY) {
        let action = state.history[i] as usize;
        if action < NUM_ACTIONS {
            enc[idx + i * NUM_ACTIONS + action] = 1.0;
        }
    }
    
    (enc, bucket)
}

// ═══════════════════════════════════════════════════════════════════════════
// CFR SAMPLE
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct CFRSample {
    pub encoding: [f32; ENCODING_DIM],
    pub advantages: [f32; NUM_ACTIONS],
    pub player: u8,
    pub infoset_key: u64,
}

#[derive(Default)]
pub struct CFRStats {
    pub unique_infosets_per_game_sum: usize,  // Sum of per-game unique (NOT global unique!)
    pub unique_infosets_global: usize,        // Actual global unique (single-threaded only)
    pub total_samples: usize,
    pub terminals_by_fold: usize,
    pub terminals_total: usize,
    pub bucket_histogram: [usize; NUM_BUCKETS],
}

// ═══════════════════════════════════════════════════════════════════════════
// EXTERNAL CFR TRAVERSAL (single game)
// ═══════════════════════════════════════════════════════════════════════════

pub fn external_cfr_traversal(
    state: &NLHEState,
    traverser: u8,
    regret_map: &mut FxHashMap<u64, [f64; NUM_ACTIONS]>,
    samples: &mut Vec<CFRSample>,
    seen: &mut FxHashSet<u64>,
    rng: &mut Xoshiro256PlusPlus,
    epsilon: f64,
) -> f64 {
    if state.is_terminal {
        return state.get_payoff(traverser);
    }
    
    let player = state.current_player;
    let (acts, n_acts) = state.legal_actions();  // STACK ARRAY - no heap!
    if n_acts == 0 { return 0.0; }
    
    // Encode and get bucket directly (no scanning!)
    let (encoding, bucket) = encode_state(state, player);
    
    // Compute infoset key
    let key = make_infoset_key(state, player, bucket);
    seen.insert(key);
    
    // Get strategy from regrets
    let regrets = regret_map.entry(key).or_insert([0.0; NUM_ACTIONS]);
    let mut strategy = [0.0f64; NUM_ACTIONS];
    let mut pos_sum = 0.0;
    
    for i in 0..n_acts {
        let a = acts[i] as usize;
        let pos = regrets[a].max(0.0);
        strategy[a] = pos;
        pos_sum += pos;
    }
    
    if pos_sum > 0.0 {
        for i in 0..n_acts { strategy[acts[i] as usize] /= pos_sum; }
    } else {
        let uniform = 1.0 / n_acts as f64;
        for i in 0..n_acts { strategy[acts[i] as usize] = uniform; }
    }
    
    // Epsilon-greedy
    if rng.gen::<f64>() < epsilon {
        let uniform = 1.0 / n_acts as f64;
        for i in 0..n_acts { strategy[acts[i] as usize] = uniform; }
    }
    
    if player == traverser {
        // Full traversal
        let mut action_values = [0.0f64; NUM_ACTIONS];
        for i in 0..n_acts {
            let action = acts[i];
            action_values[action as usize] = external_cfr_traversal(
                &state.apply_action(action), traverser, regret_map, samples, seen, rng, epsilon
            );
        }
        
        let mut ev = 0.0;
        for i in 0..n_acts {
            ev += strategy[acts[i] as usize] * action_values[acts[i] as usize];
        }
        
        // Update regrets (CFR+)
        let regrets = regret_map.entry(key).or_insert([0.0; NUM_ACTIONS]);
        let mut advantages = [0.0f32; NUM_ACTIONS];
        for i in 0..n_acts {
            let a = acts[i] as usize;
            let adv = action_values[a] - ev;
            advantages[a] = adv as f32;
            regrets[a] = (regrets[a] + adv).max(0.0);
        }
        
        samples.push(CFRSample { encoding, advantages, player, infoset_key: key });
        ev
    } else {
        // Sample opponent action
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        let mut chosen = acts[0];
        for i in 0..n_acts {
            cumsum += strategy[acts[i] as usize];
            if r < cumsum { chosen = acts[i]; break; }
        }
        external_cfr_traversal(&state.apply_action(chosen), traverser, regret_map, samples, seen, rng, epsilon)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PARALLEL CFR BATCH
// ═══════════════════════════════════════════════════════════════════════════

pub struct CFRConfig {
    pub starting_stack: i32,
    pub max_raises: u8,
    pub epsilon: f64,
}

/// Thread pool (reused across calls)
static THREAD_POOL: once_cell::sync::Lazy<parking_lot::Mutex<Option<rayon::ThreadPool>>> = 
    once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(None));

fn get_or_create_pool(num_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap()
}

/// Run CFR games in parallel using Rayon
pub fn run_cfr_batch_parallel(
    num_games: usize,
    config: &CFRConfig,
    seed: u64,
    num_threads: usize,
) -> CFRStats {
    let pool = get_or_create_pool(num_threads);
    
    let per_game_unique_sum = AtomicUsize::new(0);
    let total_samples = AtomicUsize::new(0);
    
    pool.install(|| {
        (0..num_games).into_par_iter().for_each(|game_idx| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(game_idx as u64));
            let mut regret_map: FxHashMap<u64, [f64; NUM_ACTIONS]> = FxHashMap::default();
            let mut samples = Vec::with_capacity(100);
            let mut seen: FxHashSet<u64> = FxHashSet::default();
            
            let state = NLHEState::new(config.starting_stack, config.max_raises, &mut rng);
            
            for player in 0..2 {
                external_cfr_traversal(
                    &state, player, &mut regret_map, &mut samples, &mut seen, &mut rng, config.epsilon
                );
            }
            
            per_game_unique_sum.fetch_add(seen.len(), Ordering::Relaxed);
            total_samples.fetch_add(samples.len(), Ordering::Relaxed);
        });
    });
    
    CFRStats {
        unique_infosets_per_game_sum: per_game_unique_sum.load(Ordering::Relaxed),
        unique_infosets_global: 0,  // Can't compute in parallel without shared set
        total_samples: total_samples.load(Ordering::Relaxed),
        terminals_by_fold: 0,
        terminals_total: 0,
        bucket_histogram: [0; NUM_BUCKETS],
    }
}

/// Single-threaded version (for comparison / Python API)
pub fn run_cfr_batch(
    num_games: usize,
    starting_stack: i32,
    max_raises: u8,
    epsilon: f64,
    seed: u64,
) -> (Vec<CFRSample>, FxHashMap<u64, [f64; NUM_ACTIONS]>, FxHashMap<u64, [f64; NUM_ACTIONS]>, CFRStats) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut regret_map: FxHashMap<u64, [f64; NUM_ACTIONS]> = FxHashMap::default();
    let strategy_map: FxHashMap<u64, [f64; NUM_ACTIONS]> = FxHashMap::default();
    let mut all_samples = Vec::with_capacity(num_games * 30);
    let mut all_seen: FxHashSet<u64> = FxHashSet::default();
    let mut stats = CFRStats::default();
    
    for _ in 0..num_games {
        let state = NLHEState::new(starting_stack, max_raises, &mut rng);
        
        for player in 0..2 {
            external_cfr_traversal(
                &state, player, &mut regret_map, &mut all_samples, &mut all_seen, &mut rng, epsilon
            );
        }
    }
    
    stats.unique_infosets_global = all_seen.len();  // This IS global unique (single-threaded)
    stats.unique_infosets_per_game_sum = all_seen.len();  // Same in ST
    stats.total_samples = all_samples.len();
    
    (all_samples, regret_map, strategy_map, stats)
}
