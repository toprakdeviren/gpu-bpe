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

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> nodes: array<u32>;  // 3 x u32 per node
@group(0) @binding(2) var<storage, read> edges: array<u32>;  // 2 x u32 per edge
@group(0) @binding(3) var<storage, read_write> token_output: array<u32>;
@group(0) @binding(4) var<storage, read_write> chunk_counts: array<u32>;
@group(0) @binding(5) var<uniform> params: TrieParams;

// O(1) Root LUT: Direct byte→node lookup table in shared memory.
// Replaces binary search (up to 8 iterations, warp divergence)
// with a single branchless array read.
var<workgroup> root_lut: array<u32, 256>;
var<workgroup> cached_root_fc: u32;                 // root firstChild
var<workgroup> cached_root_nc: u32;                 // root numChildren

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
    // Build O(1) root LUT: 256 threads fill 256 slots
    // Step 1: Initialize all slots to INVALID (no child for this byte)
    root_lut[lid.x] = INVALID_TOKEN;
    if (lid.x == 0u) {
        cached_root_fc = nodes[0];                    // root.firstChild
        cached_root_nc = nodes[1] & 0xFFFFu;          // root.numChildren
    }
    workgroupBarrier();
    // Step 2: Scatter valid edges into LUT by byte value
    let nc = min(cached_root_nc, MAX_CACHED_EDGES);
    let fc = cached_root_fc;
    if (lid.x < nc) {
        let sym = edges[(fc + lid.x) * 2u] & 0xFFu;
        root_lut[sym] = edges[(fc + lid.x) * 2u + 1u];  // Direct: byte → target node
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
                // O(1) direct LUT lookup — branchless, no divergence
                nn = root_lut[bv];
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
