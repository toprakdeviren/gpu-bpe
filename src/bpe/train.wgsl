/**
 * @file train.wgsl
 * @brief GPU-Native BPE Training Pipeline (WebGPU/WGSL)
 *
 * Training kernels for BPE vocabulary learning. Each kernel is delimited by
 * "// --- KERNEL: <name> ---" markers. The JS host splits this file
 * at those markers and compiles each kernel as a separate GPUShaderModule,
 * prepending the shared utility section (everything before the first marker).
 *
 * Kernels (11):
 *   1. bpe_word_boundary         — GPU pre-tokenization (word boundary detection)
 *   2. bpe_clear_table           — Hash table reset
 *   3. bpe_find_max_pair         — Block-level max reduction
 *   4. bpe_find_max_pair_final   — Final max reduction (single workgroup)
 *   5. bpe_setup_merge           — GPU-driven merge orchestrator
 *   6. bpe_pair_count_b          — Batched two-level pair counting
 *   7. bpe_merge_b               — Fused fill_valid + pair merge (race-free self-validity)
 *   8. bpe_prefix_sum_reduce_b   — Batched prefix sum reduce
 *   9. bpe_prefix_sum_scan_blocks_b — Block scan (stages total to IterState._pad1)
 *  10. bpe_finalize_compact_b    — Fused Blelloch scan + scatter
 *  11. bpe_update_count          — Symbol count update + indirect dispatch (reads _pad1)
 */

// ════════════════════════════════════════════════════════════
// SHARED UTILITIES (prepended to every kernel module)
// ════════════════════════════════════════════════════════════

const WORKGROUP_SIZE: u32 = 256u;
const MAX_PROBE: u32 = 128u;
const MAX_WG_DIM: u32 = 65535u;  // WebGPU maxComputeWorkgroupsPerDimension
const INVALID_TOKEN: u32 = 0xFFFFFFFFu;
const WORD_START_BIT: u32 = 0x10000u;  // bit 16 = word-start flag
const TOKEN_MASK: u32 = 0xFFFFu;       // lower 16 bits = token ID

fn pack_pair(a: u32, b: u32) -> u32 { return (a << 16u) | b; }
fn unpack_first(pair: u32) -> u32  { return pair >> 16u; }
fn unpack_second(pair: u32) -> u32 { return pair & 0xFFFFu; }

// GPU-driven iteration state for batched training.
// Written by bpe_setup_merge, read by all batched kernels.
struct IterState {
    symbol_count: u32,      // [0]  Current valid symbol count
    table_size: u32,        // [1]  Hash table size (constant)
    early_stop: u32,        // [2]  1=stop, 0=continue
    next_token_id: u32,     // [3]  Next token ID to assign
    symbol_a: u32,          // [4]  First symbol of best pair
    symbol_b: u32,          // [5]  Second symbol of best pair
    new_symbol: u32,        // [6]  Merged symbol ID
    max_count: u32,         // [7]  Count of best pair
    merges_done: u32,       // [8]  Merges completed in this batch
    max_symbols: u32,       // [9]  Max dispatch size (initial count)
    _pad1: u32,
    _pad2: u32,
}

/// Murmur3 integer finalizer — 6 ALU ops instead of FNV-1a's 16.
/// Returns raw hash; caller applies `& table_mask` for power-of-2 tables.
fn pair_hash(pair: u32) -> u32 {
    var x = pair;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

/// Linearize a 2D dispatch grid into a flat 1D thread index.
/// JS host splits large dispatches into (X, Y, 1) where X*Y = total workgroups.
/// Each kernel should use flat_id() instead of gid.x to support >16M threads.
fn flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WORKGROUP_SIZE;
}

// Local hash table constants for bpe_pair_count_b
const LOCAL_TABLE_SIZE: u32 = 512u;
const LOCAL_TABLE_MASK: u32 = 511u;  // power-of-2 modulo
const LOCAL_MAX_PROBE: u32 = 64u;

// --- KERNEL: bpe_word_boundary ---
//
// GPU pre-tokenization: classify each symbol's byte value into a character
// class (letter/digit/space/punct), then compare adjacent classes to detect
// word boundaries. When a boundary is found, bit 16 (WORD_START_BIT) is set
// on the symbol at the start of the new word.
//
// This ensures bpe_pair_count never counts pairs across word boundaries,
// preventing multi-word token formation (e.g. "yakınlık▁ve▁" → 1 token).
//
// Character classes:
//   0 = letter (a-z, 0xC0-0x24F, Turkish ğışçöüİĞŞÇÖÜ, Arabic, etc.)
//   1 = digit  (0-9)
//   2 = space  (0x20 = ▁)
//   3 = punctuation / other

struct WordBoundaryParams { symbol_count: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read_write> symbols: array<u32>;
@group(0) @binding(1) var<uniform> params: WordBoundaryParams;

/// Character classification with reduced branch divergence.
/// Uses unsigned subtraction trick: `(tok - base) <= range` is branchless-friendly
/// and covers the entire range in a single comparison.
fn char_class(tok: u32) -> u32 {
    // Newline — own class (always a word boundary)
    if (tok == 0x0Au) { return 4u; }
    // Space
    if (tok == 0x20u) { return 2u; }
    // Digit 0-9 (0x30..0x39)
    if (tok - 0x30u <= 9u) { return 1u; }
    // UTF-8 continuation + leading bytes (0x80-0xFF) — all treated as letter
    // Covers multi-byte chars, Turkish İĞŞÇÖÜ, Arabic, etc.
    if (tok >= 0x80u) { return 0u; }
    // ASCII letter a-z (0x61..0x7A)
    if (tok - 0x61u <= 25u) { return 0u; }
    // ASCII letter A-Z (0x41..0x5A)
    if (tok - 0x41u <= 25u) { return 0u; }
    // Everything else = punctuation
    return 3u;
}

/**
 * @compute Kernel: bpe_word_boundary
 *
 * Scans the symbol sequence to identify word boundaries based on character classes.
 * When a boundary is detected, the WORD_START_BIT (bit 16) is set on that symbol.
 * This prevents the BPE pairing logic from merging tokens across distinct words,
 * following GPT-style pre-tokenization rules.
 *
 * Boundary Rules:
 * 1. The first symbol of the entire sequence is always a word start.
 * 2. Any transition between character classes (e.g., letter to punctuation) is a boundary.
 * 3. Special Case: A space (class 2) followed by a letter (0) or digit (1) is NOT a
 *    boundary; the space "attaches" to the start of the following word.
 * 4. Special Case: A newline (class 4) always forces a boundary both before and after.
 */
@compute @workgroup_size(256)
fn bpe_word_boundary(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let id = flat_id(gid, nwg);
    if (id >= params.symbol_count) { return; }

    let tok = symbols[id] & TOKEN_MASK;
    let cls = char_class(tok);

    // First symbol is always a word start
    if (id == 0u) {
        symbols[id] = tok | WORD_START_BIT;
        return;
    }

    let prev_tok = symbols[id - 1u] & TOKEN_MASK;
    let prev_cls = char_class(prev_tok);

    // Default: Word boundary exists if the character class changes
    var is_boundary = cls != prev_cls;

    // GPT-4 style: space(2) followed by letter(0) or digit(1) = same word (space attaches to next word).
    if (prev_cls == 2u && (cls == 0u || cls == 1u)) { 
        is_boundary = false; 
    }
    
    // But a space itself starting after a non-space is always a word start
    if (cls == 2u && prev_cls != 2u) { 
        is_boundary = true; 
    }

    // Newline(4) transitions always represent boundaries
    if (prev_cls == 4u || cls == 4u) { 
        is_boundary = true; 
    }

    if (is_boundary) {
        symbols[id] = tok | WORD_START_BIT;
    }
    // else: tok stays as-is (lower 16 bits only)
}

// --- KERNEL: bpe_clear_table ---

struct ClearParams { table_size: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read_write> pair_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> pair_ids: array<u32>;
@group(0) @binding(2) var<uniform> params: ClearParams;

@compute @workgroup_size(256)
fn bpe_clear_table(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= params.table_size) { return; }
    pair_counts[id] = 0u;
    pair_ids[id] = 0u;
}

// --- KERNEL: bpe_find_max_pair ---

struct FindMaxParams { table_size: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> pair_counts: array<u32>;
@group(0) @binding(1) var<storage, read> pair_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_max_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_max_pair_ids: array<u32>;
@group(0) @binding(4) var<uniform> params: FindMaxParams;

var<workgroup> sh_c: array<u32, 256>;
var<workgroup> sh_p: array<u32, 256>;

@compute @workgroup_size(256)
fn bpe_find_max_pair(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let fid = flat_id(gid, nwg);
    let block_idx = tgid.x + tgid.y * nwg.x;
    var c: u32 = 0u; var p: u32 = 0u;
    if (fid < params.table_size) { c = pair_counts[fid]; p = pair_ids[fid]; }
    sh_c[lid.x] = c; sh_p[lid.x] = p;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s && sh_c[lid.x + s] > sh_c[lid.x]) {
            sh_c[lid.x] = sh_c[lid.x + s]; sh_p[lid.x] = sh_p[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) { block_max_counts[block_idx] = sh_c[0]; block_max_pair_ids[block_idx] = sh_p[0]; }
}

// --- KERNEL: bpe_find_max_pair_final ---

struct FinalParams { block_count: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> block_max_counts: array<u32>;
@group(0) @binding(1) var<storage, read> block_max_pair_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> max_count: array<u32>;
@group(0) @binding(3) var<storage, read_write> max_pair_id: array<u32>;
@group(0) @binding(4) var<uniform> params: FinalParams;

var<workgroup> sh_c: array<u32, 256>;
var<workgroup> sh_p: array<u32, 256>;

@compute @workgroup_size(256)
fn bpe_find_max_pair_final(@builtin(local_invocation_id) lid: vec3<u32>) {
    var lm: u32 = 0u; var lp: u32 = 0u;
    var i: u32 = lid.x;
    while (i < params.block_count) {
        let c = block_max_counts[i];
        if (c > lm) { lm = c; lp = block_max_pair_ids[i]; }
        i += WORKGROUP_SIZE;
    }
    sh_c[lid.x] = lm; sh_p[lid.x] = lp;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s && sh_c[lid.x + s] > sh_c[lid.x]) {
            sh_c[lid.x] = sh_c[lid.x + s]; sh_p[lid.x] = sh_p[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) { max_count[0] = sh_c[0]; max_pair_id[0] = sh_p[0]; }
}

// ════════════════════════════════════════════════════════════
// BATCHED TRAINING KERNELS — GPU-driven merge loop
//
// These "_b" variants read iteration parameters from a shared
// IterState storage buffer instead of per-dispatch uniforms,
// enabling N merges to be encoded in a single command buffer
// with zero CPU readbacks between iterations.
// ════════════════════════════════════════════════════════════

// --- KERNEL: bpe_setup_merge ---
//
// Orchestrator (single thread). Runs after findMax each iteration.
// Reads best pair, writes merge params, increments token counter,
// checks early stop, logs the merge for CPU reconstruction.

@group(0) @binding(0) var<storage, read> sm_max_count: array<u32>;
@group(0) @binding(1) var<storage, read> sm_max_pair_id: array<u32>;
@group(0) @binding(2) var<storage, read_write> state: IterState;
@group(0) @binding(3) var<storage, read_write> merge_log: array<u32>;

@compute @workgroup_size(1)
fn bpe_setup_merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (state.early_stop != 0u) { return; }

    let mc = sm_max_count[0];
    if (mc < 2u || state.next_token_id > TOKEN_MASK) {
        state.early_stop = 1u;
        return;
    }

    let pair = sm_max_pair_id[0];
    state.symbol_a = pair >> 16u;
    state.symbol_b = pair & 0xFFFFu;
    state.max_count = mc;
    state.new_symbol = state.next_token_id;

    // Log for CPU vocab reconstruction: [pair, newTokenId, count]
    let log_idx = state.merges_done * 3u;
    merge_log[log_idx]      = pair;
    merge_log[log_idx + 1u] = state.next_token_id;
    merge_log[log_idx + 2u] = mc;

    state.next_token_id += 1u;
    state.merges_done += 1u;
}

// --- KERNEL: bpe_pair_count_b ---

@group(0) @binding(0) var<storage, read> symbols: array<u32>;
@group(0) @binding(1) var<storage, read_write> pair_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> pair_ids: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> state: IterState;

var<workgroup> local_ids: array<atomic<u32>, 512>;
var<workgroup> local_counts: array<atomic<u32>, 512>;

@compute @workgroup_size(256)
fn bpe_pair_count_b(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    atomicStore(&local_ids[lid.x], 0u);
    atomicStore(&local_counts[lid.x], 0u);
    atomicStore(&local_ids[lid.x + 256u], 0u);
    atomicStore(&local_counts[lid.x + 256u], 0u);
    workgroupBarrier();

    if (state.early_stop != 0u) { return; }

    // Phase 1: Aggregate into LOCAL shared-memory table
    let id = flat_id(gid, nwg);
    if (id + 1u < state.symbol_count) {
        let raw_b = symbols[id + 1u];
        if ((raw_b & WORD_START_BIT) == 0u) {
            let a = symbols[id] & TOKEN_MASK;
            let b = raw_b & TOKEN_MASK;
            if (a != 0u && b != 0u) {
                let pid = pack_pair(a, b);
                let h = pair_hash(pid);   // Murmur3 finalizer — better avalanche than Knuth
                for (var probe: u32 = 0u; probe < LOCAL_MAX_PROBE; probe++) {
                    let idx = (h + (probe * (probe + 1u)) / 2u) & LOCAL_TABLE_MASK;
                    let r = atomicCompareExchangeWeak(&local_ids[idx], 0u, pid);
                    if (r.exchanged || r.old_value == pid) {
                        atomicAdd(&local_counts[idx], 1u);
                        break;
                    }
                }
            }
        }
    }
    workgroupBarrier();

    // Phase 2: Flush local table → global table
    for (var slot: u32 = lid.x; slot < LOCAL_TABLE_SIZE; slot += WORKGROUP_SIZE) {
        let cnt = atomicLoad(&local_counts[slot]);
        if (cnt == 0u) { continue; }
        let pid = atomicLoad(&local_ids[slot]);
        if (pid == 0u) { continue; }
        let table_mask = state.table_size - 1u;
        let hash = pair_hash(pid) & table_mask;
        for (var probe: u32 = 0u; probe < MAX_PROBE; probe++) {
            let idx = (hash + (probe * (probe + 1u)) / 2u) & table_mask;
            let r = atomicCompareExchangeWeak(&pair_ids[idx], 0u, pid);
            if (r.exchanged || r.old_value == pid) {
                atomicAdd(&pair_counts[idx], cnt);
                break;
            }
        }
    }
}

// --- KERNEL: bpe_merge_b ---
//
// Fused fill_valid + merge: each thread writes its OWN valid_mask[id].
// A thread is invalid (valid=0) only if it is the B-side of a matching pair.
// The A-side thread writes the merged symbol. No cross-thread valid_mask
// writes — completely race-free without global barriers.

@group(0) @binding(0) var<storage, read_write> symbols: array<u32>;
@group(0) @binding(1) var<storage, read_write> valid_mask: array<u32>;
@group(0) @binding(2) var<storage, read> state: IterState;

@compute @workgroup_size(256)
fn bpe_merge_b(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if (state.early_stop != 0u) { return; }
    let id = flat_id(gid, nwg);
    if (id >= state.symbol_count) { return; }

    let raw = symbols[id];

    // ── A-side: check if (id, id+1) is the winning pair → write merged symbol ──
    if (id + 1u < state.symbol_count) {
        let raw_b = symbols[id + 1u];
        if ((raw_b & WORD_START_BIT) == 0u
            && (raw & TOKEN_MASK) == state.symbol_a
            && (raw_b & TOKEN_MASK) == state.symbol_b) {
            let flag = raw & WORD_START_BIT;
            symbols[id] = state.new_symbol | flag;
        }
    }

    // ── Self-validity: am I the B-side of a merge? ──
    // Check if (id-1, id) matches the winning pair → I'm consumed → valid=0
    var valid: u32 = 1u;
    if (id > 0u) {
        let raw_prev = symbols[id - 1u];
        if ((raw & WORD_START_BIT) == 0u
            && (raw_prev & TOKEN_MASK) == state.symbol_a
            && (raw & TOKEN_MASK) == state.symbol_b) {
            valid = 0u;
        }
    }
    valid_mask[id] = valid;
}

// --- KERNEL: bpe_prefix_sum_reduce_b ---

@group(0) @binding(0) var<storage, read> valid_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(2) var<storage, read> state: IterState;

var<workgroup> sh_data: array<u32, 256>;

@compute @workgroup_size(256)
fn bpe_prefix_sum_reduce_b(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let fid = flat_id(gid, nwg);
    let block_idx = tgid.x + tgid.y * nwg.x;
    var v: u32 = 0u;
    if (fid < state.symbol_count) { v = valid_mask[fid] & 1u; }
    sh_data[lid.x] = v;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) { sh_data[lid.x] += sh_data[lid.x + s]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { block_sums[block_idx] = sh_data[0]; }
}

// --- KERNEL: bpe_prefix_sum_scan_blocks_b ---
//
// Sequential exclusive prefix scan over block_sums (single thread).
// Writes total valid count to state._pad1 (staging field) instead of a
// separate buffer — eliminates the total_valid buffer entirely.
// Cannot update state.symbol_count here because finalize_compact_b
// runs next and needs the OLD symbol_count.

@group(0) @binding(0) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(1) var<storage, read_write> state: IterState;

@compute @workgroup_size(1)
fn bpe_prefix_sum_scan_blocks_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    if (state.early_stop != 0u) { return; }

    let block_count = (state.symbol_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < block_count; i++) {
        let v = block_sums[i];
        block_sums[i] = sum;
        sum += v;
    }
    // Stage new count in _pad1 — consumed by bpe_update_count after compact
    state._pad1 = sum;
}

// --- KERNEL: bpe_finalize_compact_b ---
//
// Fused prefix_sum_finalize + compact: Blelloch exclusive scan in shared memory,
// then directly scatter valid symbols to the output buffer.
// Eliminates the prefix_sum intermediate buffer entirely — the exclusive scan
// result stays in registers and is used immediately for the scatter index.
//
// Bindings:
//   0: valid_mask      (read)   — 1-bit per symbol: valid or merged-away
//   1: block_sums      (read)   — per-block totals from prefix_sum_reduce
//   2: input_symbols   (read)   — source symbol buffer (ping or pong)
//   3: output_symbols  (rw)     — destination symbol buffer
//   4: state           (read)   — IterState with symbol_count

@group(0) @binding(0) var<storage, read> valid_mask: array<u32>;
@group(0) @binding(1) var<storage, read> block_sums: array<u32>;
@group(0) @binding(2) var<storage, read> input_symbols: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_symbols: array<u32>;
@group(0) @binding(4) var<storage, read> state: IterState;

var<workgroup> sh_data: array<u32, 256>;

@compute @workgroup_size(256)
fn bpe_finalize_compact_b(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let fid = flat_id(gid, nwg);
    let block_idx = tgid.x + tgid.y * nwg.x;

    // Load valid bit (0 or 1)
    var v: u32 = 0u;
    if (fid < state.symbol_count) { v = valid_mask[fid] & 1u; }

    // ── Blelloch exclusive scan in shared memory ──
    sh_data[lid.x] = v;
    workgroupBarrier();

    // Up-sweep (reduce)
    for (var d: u32 = 1u; d < WORKGROUP_SIZE; d <<= 1u) {
        let idx = (lid.x + 1u) * (d << 1u) - 1u;
        if (idx < WORKGROUP_SIZE) { sh_data[idx] += sh_data[idx - d]; }
        workgroupBarrier();
    }

    // Clear root
    if (lid.x == 0u) { sh_data[WORKGROUP_SIZE - 1u] = 0u; }
    workgroupBarrier();

    // Down-sweep
    for (var d: u32 = WORKGROUP_SIZE >> 1u; d > 0u; d >>= 1u) {
        let idx = (lid.x + 1u) * (d << 1u) - 1u;
        if (idx < WORKGROUP_SIZE) {
            let t = sh_data[idx - d];
            sh_data[idx - d] = sh_data[idx];
            sh_data[idx] += t;
        }
        workgroupBarrier();
    }

    // ── Fused scatter: prefix_sum stays in register, write directly to output ──
    if (fid < state.symbol_count && v == 1u) {
        let dest = block_sums[block_idx] + sh_data[lid.x];
        output_symbols[dest] = input_symbols[fid];
    }
}

// --- KERNEL: bpe_update_count ---
//
// After compact: reads new symbol count from state._pad1 (staged by
// scan_blocks), updates symbol_count, and computes indirect dispatch
// params for the next iteration. Eliminates the total_valid buffer —
// the value is passed through IterState padding.
//
// Indirect buffer layout: [wgX, wgY, wgZ] (3 × u32)

@group(0) @binding(0) var<storage, read_write> state: IterState;
@group(0) @binding(1) var<storage, read_write> indirect: array<u32>;

@compute @workgroup_size(1)
fn bpe_update_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (state.early_stop != 0u) { return; }

    let new_count = state._pad1;
    state.symbol_count = new_count;

    let total_wg = (new_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    if (total_wg <= MAX_WG_DIM) {
        indirect[0] = max(total_wg, 1u);
        indirect[1] = 1u;
    } else {
        indirect[0] = MAX_WG_DIM;
        indirect[1] = (total_wg + MAX_WG_DIM - 1u) / MAX_WG_DIM;
    }
    indirect[2] = 1u;
}
