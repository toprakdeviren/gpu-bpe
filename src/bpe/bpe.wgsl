/**
 * @file bpe.wgsl
 * @brief GPU-Native BPE Pipeline — Training + Inference (WebGPU/WGSL)
 *
 * Ported from Metal (bpe.metal). Each kernel is delimited by
 * "// --- KERNEL: <name> ---" markers. The JS host splits this file
 * at those markers and compiles each kernel as a separate GPUShaderModule,
 * prepending the shared utility section (everything before the first marker).
 *
 * Training Kernels (1-13) + Inference Kernels (14-15).
 */

// ════════════════════════════════════════════════════════════
// SHARED UTILITIES (prepended to every kernel module)
// ════════════════════════════════════════════════════════════

const WORKGROUP_SIZE: u32 = 256u;
const MAX_PROBE: u32 = 128u;
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

fn pair_hash(pair: u32, table_size: u32) -> u32 {
    var h: u32 = 0x811C9DC5u;
    h = h ^ (pair & 0xFFu);          h = h * 0x01000193u;
    h = h ^ ((pair >> 8u) & 0xFFu);  h = h * 0x01000193u;
    h = h ^ ((pair >> 16u) & 0xFFu); h = h * 0x01000193u;
    h = h ^ ((pair >> 24u) & 0xFFu); h = h * 0x01000193u;
    return h % table_size;
}

/// Linearize a 2D dispatch grid into a flat 1D thread index.
/// JS host splits large dispatches into (X, Y, 1) where X*Y = total workgroups.
/// Each kernel should use flat_id() instead of gid.x to support >16M threads.
fn flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WORKGROUP_SIZE;
}

// Local hash table constants (shared by bpe_pair_count and bpe_pair_count_b)
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

fn char_class(tok: u32) -> u32 {
    // Newline — own class (always a word boundary)
    if (tok == 0x0Au) { return 4u; }
    // Space
    if (tok == 0x20u) { return 2u; }
    // Digit 0-9
    if (tok >= 0x30u && tok <= 0x39u) { return 1u; }
    // ASCII letter a-z (already lowercased in _byteLevelLoad)
    if (tok >= 0x61u && tok <= 0x7Au) { return 0u; }
    // ASCII uppercase A-Z (in case not lowercased)
    if (tok >= 0x41u && tok <= 0x5Au) { return 0u; }
    // Latin Extended (À-ɏ, covers Turkish İĞŞÇÖÜ mapped bytes and accented chars)
    if (tok >= 0xC0u && tok <= 0x24Fu) { return 0u; }
    // Common continuation byte range from multi-byte UTF-8 in byte-level mode.
    // In byte-level BPE, bytes 0x80-0xBF are UTF-8 continuations — treat as letter
    // so multi-byte characters don't get split at the wrong boundary.
    if (tok >= 0x80u && tok <= 0xBFu) { return 0u; }
    // UTF-8 leading bytes for multi-byte sequences (0xC0-0xFF) — treat as letter
    if (tok >= 0xC0u) { return 0u; }
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

// --- KERNEL: bpe_pair_count ---
//
// Optimization: Two-level hashing to reduce global atomic contention.
// Phase 1: Each workgroup aggregates pair counts into a LOCAL shared-memory
//          hash table (512 entries, power-of-2). Shared memory atomics are
//          10-100x faster than global memory atomics.
// Phase 2: Flush non-zero local entries to the global hash table.
//
// WORD BOUNDARY: Pairs where the second symbol has WORD_START_BIT set
// are SKIPPED — no cross-word merges allowed.

// (LOCAL_TABLE constants moved to preamble)

struct PairCountParams { symbol_count: u32, table_size: u32 }

@group(0) @binding(0) var<storage, read> symbols: array<u32>;
@group(0) @binding(1) var<storage, read_write> pair_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> pair_ids: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: PairCountParams;

var<workgroup> local_ids: array<atomic<u32>, 512>;
var<workgroup> local_counts: array<atomic<u32>, 512>;

/// Counts adjacent symbol pairs within words using a two-level hashing strategy.
///
/// This kernel reduces global memory contention by first aggregating pair counts
/// in a workgroup-local hash table before flushing them to the global hash table.
/// Pairs that cross word boundaries (indicated by WORD_START_BIT) are ignored.
@compute @workgroup_size(256)
fn bpe_pair_count(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    // Phase 0: Clear local table (256 threads clear 512 entries)
    atomicStore(&local_ids[lid.x], 0u);
    atomicStore(&local_counts[lid.x], 0u);
    atomicStore(&local_ids[lid.x + 256u], 0u);
    atomicStore(&local_counts[lid.x + 256u], 0u);
    workgroupBarrier();

    // Phase 1: Aggregate into LOCAL shared-memory table
    let id = flat_id(gid, nwg);
    if (id + 1u < params.symbol_count) {
        let raw_b = symbols[id + 1u];
        // Skip if next symbol starts a new word (boundary guard)
        if ((raw_b & WORD_START_BIT) == 0u) {
            let a = symbols[id] & TOKEN_MASK;
            let b = raw_b & TOKEN_MASK;
            if (a != 0u && b != 0u) {
                let pid = pack_pair(a, b);
                let h = pid ^ (pid >> 16u);  // fast hash for small table
                for (var probe: u32 = 0u; probe < LOCAL_MAX_PROBE; probe++) {
                    let idx = (h + probe) & LOCAL_TABLE_MASK;
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

    // Phase 2: Flush local table → global table (256 threads flush 512 entries)
    for (var slot: u32 = lid.x; slot < LOCAL_TABLE_SIZE; slot += WORKGROUP_SIZE) {
        let cnt = atomicLoad(&local_counts[slot]);
        if (cnt == 0u) { continue; }
        let pid = atomicLoad(&local_ids[slot]);
        if (pid == 0u) { continue; }
        let hash = pair_hash(pid, params.table_size);
        for (var probe: u32 = 0u; probe < MAX_PROBE; probe++) {
            let idx = (hash + probe) % params.table_size;
            let r = atomicCompareExchangeWeak(&pair_ids[idx], 0u, pid);
            if (r.exchanged || r.old_value == pid) {
                atomicAdd(&pair_counts[idx], cnt);  // flush aggregated count
                break;
            }
        }
    }
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

// --- KERNEL: bpe_merge ---
//
// Merges pairs while respecting word boundaries:
// - Only merge if both symbols match AND the second symbol is NOT a word-start.
// - The merged symbol inherits the word-start flag from the first symbol.

struct MergeParams { symbol_a: u32, symbol_b: u32, new_symbol: u32, symbol_count: u32 }

@group(0) @binding(0) var<storage, read_write> symbols: array<u32>;
@group(0) @binding(1) var<storage, read_write> valid_mask: array<u32>;
@group(0) @binding(2) var<uniform> params: MergeParams;

@compute @workgroup_size(256)
fn bpe_merge(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id + 1u >= params.symbol_count) { return; }

    let raw_a = symbols[id];
    let raw_b = symbols[id + 1u];

    // Don't merge across word boundaries
    if ((raw_b & WORD_START_BIT) != 0u) { return; }

    if ((raw_a & TOKEN_MASK) == params.symbol_a && (raw_b & TOKEN_MASK) == params.symbol_b) {
        // Preserve word-start flag from first symbol
        let flag = raw_a & WORD_START_BIT;
        symbols[id] = params.new_symbol | flag;
        valid_mask[id + 1u] = 0u;
    }
}

// --- KERNEL: bpe_prefix_sum_reduce ---

struct CountParams { count: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> valid_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(2) var<uniform> params: CountParams;

var<workgroup> sh_data: array<u32, 256>;

@compute @workgroup_size(256)
fn bpe_prefix_sum_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let fid = flat_id(gid, nwg);
    let block_idx = tgid.x + tgid.y * nwg.x;
    var v: u32 = 0u;
    if (fid < params.count) { v = valid_mask[fid] & 1u; }
    sh_data[lid.x] = v;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) { sh_data[lid.x] += sh_data[lid.x + s]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { block_sums[block_idx] = sh_data[0]; }
}

// --- KERNEL: bpe_prefix_sum_scan_blocks ---

struct ScanParams { block_count: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(1) var<storage, read_write> total_valid: array<u32>;
@group(0) @binding(2) var<uniform> params: ScanParams;

@compute @workgroup_size(1)
fn bpe_prefix_sum_scan_blocks(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < params.block_count; i++) {
        let v = block_sums[i];
        block_sums[i] = sum;
        sum += v;
    }
    total_valid[0] = sum;
}

// --- KERNEL: bpe_prefix_sum_finalize ---

struct CountParams2 { count: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> valid_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix_sum: array<u32>;
@group(0) @binding(2) var<storage, read> block_sums: array<u32>;
@group(0) @binding(3) var<uniform> params: CountParams2;

var<workgroup> sh_data: array<u32, 256>;

@compute @workgroup_size(256)
fn bpe_prefix_sum_finalize(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let fid = flat_id(gid, nwg);
    let block_idx = tgid.x + tgid.y * nwg.x;
    var v: u32 = 0u;
    if (fid < params.count) { v = valid_mask[fid] & 1u; }

    // Blelloch exclusive prefix sum in workgroup sh_data memory
    sh_data[lid.x] = v;
    workgroupBarrier();

    // Up-sweep
    for (var d: u32 = 1u; d < WORKGROUP_SIZE; d <<= 1u) {
        let idx = (lid.x + 1u) * (d << 1u) - 1u;
        if (idx < WORKGROUP_SIZE) { sh_data[idx] += sh_data[idx - d]; }
        workgroupBarrier();
    }
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

    if (fid < params.count) {
        prefix_sum[fid] = block_sums[block_idx] + sh_data[lid.x];
    }
}

// --- KERNEL: bpe_compact ---

struct CompactParams { input_count: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> input_symbols: array<u32>;
@group(0) @binding(1) var<storage, read> valid_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_symbols: array<u32>;
@group(0) @binding(3) var<storage, read> prefix_sum: array<u32>;
@group(0) @binding(4) var<uniform> params: CompactParams;

@compute @workgroup_size(256)
fn bpe_compact(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= params.input_count) { return; }
    if ((valid_mask[id] & 1u) == 1u) {
        output_symbols[prefix_sum[id]] = input_symbols[id];
    }
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

// --- KERNEL: bpe_fill_valid_mask ---

struct FillParams { count: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read_write> valid_mask: array<u32>;
@group(0) @binding(1) var<uniform> params: FillParams;

@compute @workgroup_size(256)
fn bpe_fill_valid_mask(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= params.count) { return; }
    valid_mask[id] = 1u;
}

// NOTE: Unicode preprocessing kernels (bpe_normalize_bytes, bpe_unicode_preprocess,
//       bpe_codepoint_to_symbol) have been removed. Text normalization is now
//       handled by the Decoder WASM module (Unicode 17.0 NFC) before training.

// --- KERNEL: trie_tokenizer_chunked ---
//
// Optimization: Cache root node's edge table in shared memory.
// Root (node 0) is accessed on EVERY token match restart — caching its
// edges eliminates repeated global memory pointer-chasing for the
// most frequently accessed trie level. With a byte-level trie,
// root has up to 256 children → 512 u32s (2KB) in shared memory.

const MAX_CACHED_EDGES: u32 = 256u;

struct TrieParams { input_length: u32, chunk_size: u32, max_tokens_per_chunk: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> nodes: array<u32>;  // 3 x u32 per node
@group(0) @binding(2) var<storage, read> edges: array<u32>;  // 2 x u32 per edge
@group(0) @binding(3) var<storage, read_write> token_output: array<u32>;
@group(0) @binding(4) var<storage, read_write> chunk_counts: array<u32>;
@group(0) @binding(5) var<uniform> params: TrieParams;

// Cached root edges: [symbol, targetNode] pairs in shared memory
var<workgroup> cached_edge_sym: array<u32, 256>;   // edge symbols
var<workgroup> cached_edge_tgt: array<u32, 256>;   // edge target nodes
var<workgroup> cached_root_fc: u32;                 // root firstChild
var<workgroup> cached_root_nc: u32;                 // root numChildren

/// Find child using CACHED root edges (shared memory — no global reads)
fn find_root_child(sym: u32) -> u32 {
    var lo: u32 = 0u; var hi = cached_root_nc;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        let es = cached_edge_sym[mid];
        if (es == sym) { return cached_edge_tgt[mid]; }
        else if (es < sym) { lo = mid + 1u; }
        else { hi = mid; }
    }
    return INVALID_TOKEN;
}

/// Find child using global memory (non-root nodes)
fn find_child_global(first: u32, num: u32, sym: u32) -> u32 {
    var lo = first; var hi = first + num;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        let es = edges[mid * 2u] & 0xFFu;
        if (es == sym) { return edges[mid * 2u + 1u]; }
        else if (es < sym) { lo = mid + 1u; }
        else { hi = mid; }
    }
    return INVALID_TOKEN;
}

@compute @workgroup_size(256)
fn trie_tokenizer_chunked(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // Cooperatively cache root's edge table into shared memory
    if (lid.x == 0u) {
        cached_root_fc = nodes[0];                    // root.firstChild
        cached_root_nc = nodes[1] & 0xFFFFu;          // root.numChildren
    }
    workgroupBarrier();
    let nc = min(cached_root_nc, MAX_CACHED_EDGES);
    let fc = cached_root_fc;
    // 256 threads load up to 256 edges cooperatively
    if (lid.x < nc) {
        cached_edge_sym[lid.x] = edges[(fc + lid.x) * 2u] & 0xFFu;
        cached_edge_tgt[lid.x] = edges[(fc + lid.x) * 2u + 1u];
    }
    workgroupBarrier();

    // Main tokenization loop
    let id = gid.x;
    let cs = id * params.chunk_size;
    if (cs >= params.input_length) { chunk_counts[id] = 0u; return; }
    let ce = min(cs + params.chunk_size, params.input_length);
    let ob = id * params.max_tokens_per_chunk;
    var tw: u32 = 0u; var pos = cs;

    while (pos < ce && tw < params.max_tokens_per_chunk) {
        var cn: u32 = 0u; var lmt: u32 = INVALID_TOKEN; var lmp = pos; var wp = pos;
        while (wp < ce) {
            let bv = input[wp] & 0xFFu;
            var nn: u32;
            if (cn == 0u) {
                // Fast path: use cached root edges (shared memory)
                nn = find_root_child(bv);
            } else {
                // Normal path: global memory lookup
                let nfc = nodes[cn * 3u]; let nnc = nodes[cn * 3u + 1u] & 0xFFFFu;
                nn = find_child_global(nfc, nnc, bv);
            }
            if (nn == INVALID_TOKEN) { break; }
            cn = nn; wp++;
            let ti = nodes[cn * 3u + 2u];
            if (ti != INVALID_TOKEN) { lmt = ti; lmp = wp; }
        }
        if (lmt != INVALID_TOKEN) {
            token_output[ob + tw] = lmt; tw++; pos = lmp;
        } else {
            token_output[ob + tw] = input[pos] & 0xFFu; tw++; pos++;
        }
    }
    chunk_counts[id] = tw;
}

// --- KERNEL: trie_tokenizer_compact ---

struct CompactTrieParams { max_tokens_per_chunk: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> chunked_tokens: array<u32>;
@group(0) @binding(1) var<storage, read> chunk_counts: array<u32>;
@group(0) @binding(2) var<storage, read> chunk_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> compact_output: array<u32>;
@group(0) @binding(4) var<uniform> params: CompactTrieParams;

@compute @workgroup_size(256)
fn trie_tokenizer_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let cnt = chunk_counts[id];
    if (cnt == 0u) { return; }
    let sb = id * params.max_tokens_per_chunk;
    let db = chunk_offsets[id];
    for (var i: u32 = 0u; i < cnt; i++) {
        compact_output[db + i] = chunked_tokens[sb + i];
    }
}

// ════════════════════════════════════════════════════════════
// GPU VOCAB MANAGEMENT KERNELS
// ════════════════════════════════════════════════════════════

// --- KERNEL: vocab_merge ---

const MAX_TOKEN_LENGTH: u32 = 256u;  // Max bytes per token

struct VocabMergeParams {
    token_a: u32,           // ID of first token to merge
    token_b: u32,           // ID of second token to merge
    new_token_id: u32,      // ID for merged token
    _pad: u32
}

@group(0) @binding(0) var<storage, read> vocab_lengths: array<u32>;      // Length of each token
@group(0) @binding(1) var<storage, read> vocab_data: array<u32>;         // Packed token bytes (4 bytes per u32)
@group(0) @binding(2) var<storage, read_write> new_vocab_lengths: array<u32>;
@group(0) @binding(3) var<storage, read_write> new_vocab_data: array<u32>;
@group(0) @binding(4) var<uniform> params: VocabMergeParams;

@compute @workgroup_size(1)
fn vocab_merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len_a = vocab_lengths[params.token_a];
    let len_b = vocab_lengths[params.token_b];
    let new_len = len_a + len_b;

    if (new_len > MAX_TOKEN_LENGTH) {
        return;  // Token too long
    }

    // Calculate byte offsets (each token gets MAX_TOKEN_LENGTH/4 u32s)
    let offset_a = params.token_a * (MAX_TOKEN_LENGTH / 4u);
    let offset_b = params.token_b * (MAX_TOKEN_LENGTH / 4u);
    let offset_new = params.new_token_id * (MAX_TOKEN_LENGTH / 4u);

    // Copy token A bytes
    let words_a = (len_a + 3u) / 4u;  // Round up to u32 count
    for (var i: u32 = 0u; i < words_a; i++) {
        new_vocab_data[offset_new + i] = vocab_data[offset_a + i];
    }

    // Copy token B bytes (starting after token A)
    let words_b = (len_b + 3u) / 4u;
    let byte_offset_in_word = len_a % 4u;

    if (byte_offset_in_word == 0u) {
        // Aligned: direct copy
        for (var i: u32 = 0u; i < words_b; i++) {
            new_vocab_data[offset_new + words_a + i] = vocab_data[offset_b + i];
        }
    } else {
        // Unaligned: need to shift and combine
        let shift_left = byte_offset_in_word * 8u;
        let shift_right = 32u - shift_left;

        var prev_word = vocab_data[offset_b];
        new_vocab_data[offset_new + words_a - 1u] |= (prev_word << shift_left);

        for (var i: u32 = 1u; i < words_b; i++) {
            let curr_word = vocab_data[offset_b + i];
            new_vocab_data[offset_new + words_a + i - 1u] |= (prev_word >> shift_right);
            new_vocab_data[offset_new + words_a + i] = (curr_word << shift_left);
            prev_word = curr_word;
        }

        if (words_b > 0u) {
            new_vocab_data[offset_new + words_a + words_b - 1u] |= (prev_word >> shift_right);
        }
    }

    // Set new token length
    new_vocab_lengths[params.new_token_id] = new_len;
}

// --- KERNEL: check_early_stop ---

struct EarlyStopParams {
    best_count: u32,        // Count of most frequent pair
    current_vocab_size: u32, // Current number of tokens
    max_vocab_size: u32,    // Target vocab size (e.g., 65535)
    _pad: u32
}

@group(0) @binding(0) var<storage, read> params: EarlyStopParams;
@group(0) @binding(1) var<storage, read_write> should_stop: array<atomic<u32>>;

@compute @workgroup_size(1)
fn check_early_stop(@builtin(global_invocation_id) gid: vec3<u32>) {
    var stop: u32 = 0u;

    // Stop if no more pairs to merge (count < 2)
    if (params.best_count < 2u) {
        stop = 1u;
    }

    // Stop if reached vocab size limit
    if (params.current_vocab_size >= params.max_vocab_size) {
        stop = 1u;
    }

    // Atomic write to ensure visibility
    if (stop == 1u) {
        atomicStore(&should_stop[0], 1u);
    }
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

    // Log for CPU vocab reconstruction: [pair, newTokenId]
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

    let id = flat_id(gid, nwg);
    if (id + 1u < state.symbol_count) {
        let raw_b = symbols[id + 1u];
        if ((raw_b & WORD_START_BIT) == 0u) {
            let a = symbols[id] & TOKEN_MASK;
            let b = raw_b & TOKEN_MASK;
            if (a != 0u && b != 0u) {
                let pid = pack_pair(a, b);
                let h = pid ^ (pid >> 16u);
                for (var probe: u32 = 0u; probe < LOCAL_MAX_PROBE; probe++) {
                    let idx = (h + probe) & LOCAL_TABLE_MASK;
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

    for (var slot: u32 = lid.x; slot < LOCAL_TABLE_SIZE; slot += WORKGROUP_SIZE) {
        let cnt = atomicLoad(&local_counts[slot]);
        if (cnt == 0u) { continue; }
        let pid = atomicLoad(&local_ids[slot]);
        if (pid == 0u) { continue; }
        let hash = pair_hash(pid, state.table_size);
        for (var probe: u32 = 0u; probe < MAX_PROBE; probe++) {
            let idx = (hash + probe) % state.table_size;
            let r = atomicCompareExchangeWeak(&pair_ids[idx], 0u, pid);
            if (r.exchanged || r.old_value == pid) {
                atomicAdd(&pair_counts[idx], cnt);
                break;
            }
        }
    }
}

// --- KERNEL: bpe_fill_valid_b ---

@group(0) @binding(0) var<storage, read_write> valid_mask: array<u32>;
@group(0) @binding(1) var<storage, read> state: IterState;

@compute @workgroup_size(256)
fn bpe_fill_valid_b(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if (state.early_stop != 0u) { return; }
    let id = flat_id(gid, nwg);
    if (id >= state.symbol_count) { return; }
    valid_mask[id] = 1u;
}

// --- KERNEL: bpe_merge_b ---

@group(0) @binding(0) var<storage, read_write> symbols: array<u32>;
@group(0) @binding(1) var<storage, read_write> valid_mask: array<u32>;
@group(0) @binding(2) var<storage, read> state: IterState;

@compute @workgroup_size(256)
fn bpe_merge_b(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if (state.early_stop != 0u) { return; }
    let id = flat_id(gid, nwg);
    if (id + 1u >= state.symbol_count) { return; }

    let raw_a = symbols[id];
    let raw_b = symbols[id + 1u];
    if ((raw_b & WORD_START_BIT) != 0u) { return; }

    if ((raw_a & TOKEN_MASK) == state.symbol_a && (raw_b & TOKEN_MASK) == state.symbol_b) {
        let flag = raw_a & WORD_START_BIT;
        symbols[id] = state.new_symbol | flag;
        valid_mask[id + 1u] = 0u;
    }
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

@group(0) @binding(0) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(1) var<storage, read_write> total_valid: array<u32>;
@group(0) @binding(2) var<storage, read> state: IterState;

@compute @workgroup_size(1)
fn bpe_prefix_sum_scan_blocks_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let block_count = (state.symbol_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < block_count; i++) {
        let v = block_sums[i];
        block_sums[i] = sum;
        sum += v;
    }
    total_valid[0] = sum;
}

// --- KERNEL: bpe_prefix_sum_finalize_b ---

@group(0) @binding(0) var<storage, read> valid_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix_sum: array<u32>;
@group(0) @binding(2) var<storage, read> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read> state: IterState;

var<workgroup> sh_data: array<u32, 256>;

@compute @workgroup_size(256)
fn bpe_prefix_sum_finalize_b(
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
    for (var d: u32 = 1u; d < WORKGROUP_SIZE; d <<= 1u) {
        let idx = (lid.x + 1u) * (d << 1u) - 1u;
        if (idx < WORKGROUP_SIZE) { sh_data[idx] += sh_data[idx - d]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { sh_data[WORKGROUP_SIZE - 1u] = 0u; }
    workgroupBarrier();
    for (var d: u32 = WORKGROUP_SIZE >> 1u; d > 0u; d >>= 1u) {
        let idx = (lid.x + 1u) * (d << 1u) - 1u;
        if (idx < WORKGROUP_SIZE) {
            let t = sh_data[idx - d];
            sh_data[idx - d] = sh_data[idx];
            sh_data[idx] += t;
        }
        workgroupBarrier();
    }

    if (fid < state.symbol_count) {
        prefix_sum[fid] = block_sums[block_idx] + sh_data[lid.x];
    }
}

// --- KERNEL: bpe_compact_b ---

@group(0) @binding(0) var<storage, read> input_symbols: array<u32>;
@group(0) @binding(1) var<storage, read> valid_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_symbols: array<u32>;
@group(0) @binding(3) var<storage, read> prefix_sum: array<u32>;
@group(0) @binding(4) var<storage, read> state: IterState;

@compute @workgroup_size(256)
fn bpe_compact_b(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = flat_id(gid, nwg);
    if (id >= state.symbol_count) { return; }
    if ((valid_mask[id] & 1u) == 1u) {
        output_symbols[prefix_sum[id]] = input_symbols[id];
    }
}

// --- KERNEL: bpe_update_count ---
//
// After compact, copies totalValid → state.symbol_count for next iteration.

@group(0) @binding(0) var<storage, read> total_valid: array<u32>;
@group(0) @binding(1) var<storage, read_write> state: IterState;

@compute @workgroup_size(1)
fn bpe_update_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (state.early_stop != 0u) { return; }
    state.symbol_count = total_valid[0];
}
