/**
 * @file tokenize.wgsl
 * @brief GPU Trie Tokenizer — Inference (WebGPU/WGSL)
 *
 * Inference kernels for BPE tokenization using a pre-compiled binary trie.
 * Each kernel is delimited by "// --- KERNEL: <name> ---" markers.
 * The JS host splits this file at those markers and compiles each kernel
 * as a separate GPUShaderModule, prepending the shared utility section.
 *
 * Kernels (2):
 *   1. trie_tokenizer_chunked  — Chunked greedy longest-match tokenization
 *   2. trie_tokenizer_compact  — Chunk output compaction
 */

// ════════════════════════════════════════════════════════════
// SHARED UTILITIES (prepended to every kernel module)
// ════════════════════════════════════════════════════════════

const WORKGROUP_SIZE: u32 = 256u;
const INVALID_TOKEN: u32 = 0xFFFFFFFFu;

// --- KERNEL: trie_tokenizer_chunked ---
//
// Optimization: Cache root node's edge table in shared memory.
// Root (node 0) is accessed on EVERY token match restart — caching its
// edges eliminates repeated global memory pointer-chasing for the
// most frequently accessed trie level. With a byte-level trie,
// root has up to 256 children → 512 u32s (2KB) in shared memory.

const MAX_CACHED_EDGES: u32 = 256u;

struct TrieParams { input_length: u32, chunk_size: u32, max_tokens_per_chunk: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> input: array<u32>;  // packed: 4 bytes per u32 (LE)
@group(0) @binding(1) var<storage, read> nodes: array<u32>;  // 3 x u32 per node
@group(0) @binding(2) var<storage, read> edges: array<u32>;  // 2 x u32 per edge
@group(0) @binding(3) var<storage, read_write> token_output: array<u32>;
@group(0) @binding(4) var<storage, read_write> chunk_counts: array<u32>;
@group(0) @binding(5) var<uniform> params: TrieParams;

/// Extract a single byte from the packed input buffer.
/// 4 bytes per u32, little-endian: byte 0 at bits [0:7], byte 3 at bits [24:31].
/// Uses hardware extractBits for bit extraction.
fn read_byte(pos: u32) -> u32 {
    return extractBits(input[pos >> 2u], (pos & 3u) * 8u, 8u);
}

// O(1) Root LUT: Direct byte→node lookup table in shared memory.
// Replaces binary search (up to 8 iterations, warp divergence)
// with a single branchless array read.
var<workgroup> root_lut: array<u32, 256>;
var<workgroup> cached_root_fc: u32;                 // root firstChild
var<workgroup> cached_root_nc: u32;                 // root numChildren

// Depth-1 metadata cache: saves 3 global reads per token.
// Every token starts at root (depth 0, cached via root_lut) then
// transitions to depth 1. Caching the depth-1 node's firstChild,
// numChildren, and tokenId eliminates 3 global reads per token.
// Indexed by the root byte that led to the depth-1 node.
// Cost: 3KB shared memory (256 × 3 × 4B).
var<workgroup> d1_fc:  array<u32, 256>;  // depth-1 firstChild
var<workgroup> d1_nc:  array<u32, 256>;  // depth-1 numChildren
var<workgroup> d1_tid: array<u32, 256>;  // depth-1 tokenId

/// Find child using global memory (non-root nodes).
/// Branchless lower_bound: `select()` compiles to predication — no SIMD
/// mask split, no warp divergence. All threads execute the same number
/// of iterations (⌈log₂(num)⌉), trading early-exit for uniform execution.
fn find_child_global(first: u32, num: u32, sym: u32) -> u32 {
    var lo: u32 = 0u;
    var n: u32 = num;
    while (n > 0u) {
        let half = n >> 1u;
        let mid = lo + half;
        let less = (edges[(first + mid) * 2u] & 0xFFu) < sym;
        lo = select(lo, mid + 1u, less);
        n = select(half, n - half - 1u, less);
    }
    if (lo < num) {
        let slot = (first + lo) * 2u;
        if ((edges[slot] & 0xFFu) == sym) {
            return edges[slot + 1u];
        }
    }
    return INVALID_TOKEN;
}

@compute @workgroup_size(256)
fn trie_tokenizer_chunked(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // ── Build shared-memory caches: 256 threads fill 256 slots ──

    // Step 1: Initialize all LUT + depth-1 cache slots
    root_lut[lid.x] = INVALID_TOKEN;
    d1_fc[lid.x]    = 0u;
    d1_nc[lid.x]    = 0u;
    d1_tid[lid.x]   = INVALID_TOKEN;
    if (lid.x == 0u) {
        cached_root_fc = nodes[0];                    // root.firstChild
        cached_root_nc = nodes[1] & 0xFFFFu;          // root.numChildren
    }

    workgroupBarrier();

    // Step 2: Scatter valid root edges into LUT + populate depth-1 cache
    let nc = min(cached_root_nc, MAX_CACHED_EDGES);
    let fc = cached_root_fc;
    if (lid.x < nc) {
        let sym     = edges[(fc + lid.x) * 2u] & 0xFFu;
        let d1_node = edges[(fc + lid.x) * 2u + 1u];       // depth-1 node index
        root_lut[sym] = d1_node;                            // byte → node (O(1))
        d1_fc[sym]    = nodes[d1_node * 3u];                // cache firstChild
        d1_nc[sym]    = nodes[d1_node * 3u + 1u] & 0xFFFFu; // cache numChildren
        d1_tid[sym]   = nodes[d1_node * 3u + 2u];           // cache tokenId
    }

    workgroupBarrier();

    // ── Main tokenization loop ──

    let id = gid.x;
    let cs = id * params.chunk_size;
    if (cs >= params.input_length) { chunk_counts[id] = 0u; return; }
    let ce = min(cs + params.chunk_size, params.input_length);
    let ob = id * params.max_tokens_per_chunk;
    var tw: u32 = 0u; var pos = cs;

    // Register cache for packed input — avoids re-reading the same u32
    // from global memory when sequential bytes fall within one word.
    var cached_word_idx: u32 = 0xFFFFFFFFu;
    var cached_word: u32 = 0u;

    while (pos < ce && tw < params.max_tokens_per_chunk) {
        var cn: u32 = 0u; var lmt: u32 = INVALID_TOKEN; var lmp = pos; var wp = pos;
        var depth: u32 = 0u;
        var rb: u32 = 0u;   // root byte — identifies which depth-1 cache entry to use
        while (wp < ce) {
            // Read byte with register cache
            let word_idx = wp >> 2u;
            if (word_idx != cached_word_idx) {
                cached_word = input[word_idx];
                cached_word_idx = word_idx;
            }
            let bv = extractBits(cached_word, (wp & 3u) * 8u, 8u);

            var nn: u32;
            if (cn == 0u) {
                // Depth 0 → 1: O(1) root LUT
                nn = root_lut[bv];
                rb = bv;
            } else if (depth == 1u) {
                // Depth 1 → 2: shared-memory cached metadata (3 global reads saved)
                nn = find_child_global(d1_fc[rb], d1_nc[rb], bv);
            } else {
                // Depth 2+: global memory lookup
                let nfc = nodes[cn * 3u]; let nnc = nodes[cn * 3u + 1u] & 0xFFFFu;
                nn = find_child_global(nfc, nnc, bv);
            }
            if (nn == INVALID_TOKEN) { break; }
            cn = nn; wp++; depth++;
            // tokenId: use cached value at depth 1, global otherwise
            let ti = select(nodes[cn * 3u + 2u], d1_tid[rb], depth == 1u);
            if (ti != INVALID_TOKEN) { lmt = ti; lmp = wp; }
        }
        if (lmt != INVALID_TOKEN) {
            token_output[ob + tw] = lmt; tw++; pos = lmp;
        } else {
            // Fallback byte — use read_byte (cache already warm in most cases)
            token_output[ob + tw] = read_byte(pos); tw++; pos++;
        }
    }
    chunk_counts[id] = tw;
}

// --- KERNEL: trie_prefix_sum ---
//
// GPU-side exclusive prefix sum over chunk_counts.
// Eliminates the CPU roundtrip that previously required:
//   1. mapAsync readback of chunk_counts (numChunks × 4 bytes)
//   2. CPU prefix sum loop
//   3. Upload of chunk_offsets (numChunks × 4 bytes)
//
// Single-thread sequential scan is sufficient because:
//   - ~100K chunks for 50MB input = ~0.1ms on GPU
//   - The real win is removing 2 PCIe transfers + 1 GPU fence
//
// Outputs total_tokens[0] so the host only reads back 4 bytes
// to allocate the correctly-sized compact buffer.

struct PrefixSumTrieParams { num_chunks: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read>       chunk_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write>  chunk_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write>  total_tokens: array<u32>;
@group(0) @binding(3) var<uniform>              params: PrefixSumTrieParams;

@compute @workgroup_size(1)
fn trie_prefix_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < params.num_chunks; i++) {
        chunk_offsets[i] = sum;
        sum += chunk_counts[i];
    }
    total_tokens[0] = sum;
}

// --- KERNEL: trie_tokenizer_compact ---
//
// Cooperative compaction: 1 workgroup (256 threads) = 1 chunk.
// All threads in a workgroup write to consecutive addresses → coalesced
// memory access. Previous design (1 thread = 1 chunk) caused stride-512
// writes across the workgroup, destroying memory bandwidth.

struct CompactTrieParams { max_tokens_per_chunk: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> chunked_tokens: array<u32>;
@group(0) @binding(1) var<storage, read> chunk_counts: array<u32>;
@group(0) @binding(2) var<storage, read> chunk_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> compact_output: array<u32>;
@group(0) @binding(4) var<uniform> params: CompactTrieParams;

@compute @workgroup_size(256)
fn trie_tokenizer_compact(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    // Linearize 2D workgroup grid (needed when numChunks > 65535)
    let chunk_id = wid.x + wid.y * nwg.x;
    let cnt = chunk_counts[chunk_id];
    if (cnt == 0u) { return; }

    let sb = chunk_id * params.max_tokens_per_chunk;
    let db = chunk_offsets[chunk_id];

    // 256 threads cooperatively copy — consecutive lid.x = consecutive addresses (coalesced)
    for (var i = lid.x; i < cnt; i += 256u) {
        compact_output[db + i] = chunked_tokens[sb + i];
    }
}
