# gpu-bpe

GPU-accelerated Byte Pair Encoding that runs entirely in the browser. Training and inference happen on WebGPU compute shaders — no server, no Python, no CUDA.

**Demo:** [decoder.run/bpe](https://decoder.run/bpe)

---

## What it does

Drop a text file (or an entire folder) into the browser and train a BPE tokenizer on the GPU. The trained model can then encode arbitrary text in real time, also on the GPU.

The full pipeline:

1. **Pre-tokenize** — WASM-based Unicode 17.0 word boundary detection (GPT-4 style rules: contractions, space-prefix model, digit grouping)
2. **Train** — Batched merge loop on WebGPU compute shaders (128 merges per GPU roundtrip)
3. **Compile** — Flatten the merge table into a binary trie (8-byte nodes, 4-byte edges)
4. **Tokenize** — Chunked trie walk on the GPU with shared-memory root edge caching

## Architecture

```
index.html ─── app.js
                 ├── bpe/engine.js          WebGPU device + pipeline compiler
                 ├── bpe/trainer.js         Training loop (GPU)
                 ├── bpe/tokenizer.js       Trie tokenizer (GPU)
                 ├── bpe/bpe.wgsl           All 25 compute kernels
                 ├── wasm/decoder.mjs       Unicode 17.0 API (Decoder WASM)
                 ├── wasm/pre_tokenizer.mjs Word boundary detection
                 └── ui/                    File handling, training UI, encoder
```

### GPU Kernels (bpe.wgsl)

25 compute kernels in a single WGSL file, split at load time into per-kernel shader modules:

| Stage | Kernels | Notes |
|-------|---------|-------|
| Word boundaries | `bpe_word_boundary` | Byte-level heuristic fallback (normally bypassed by WASM pre-tokenizer) |
| Pair counting | `bpe_pair_count`, `bpe_pair_count_b` | Two-level open-addressing hash table (prime size 2,097,143) |
| Max-pair reduction | `bpe_find_max_pair`, `bpe_find_max_pair_final` | Two-pass parallel reduction |
| Merge | `bpe_merge`, `bpe_merge_b`, `bpe_setup_merge` | In-place symbol rewriting with word boundary preservation |
| Stream compaction | `bpe_prefix_sum_*`, `bpe_compact_*`, `bpe_fill_valid_*` | Blelloch prefix sum for gap removal |
| Batch control | `bpe_update_count`, `check_early_stop`, `vocab_merge` | GPU-driven iteration state |
| Tokenization | `trie_tokenizer_chunked`, `trie_tokenizer_compact` | Shared-memory trie walk |

### Pre-tokenization (WASM)

The WASM layer (Decoder) provides full Unicode 17.0 property tables. The `PreTokenizer` classifies codepoints into character classes (letter, digit, whitespace, punctuation, symbol, newline) and applies GPT-4 style boundary rules at the codepoint level — before the byte stream reaches the GPU. This solves the multi-byte punctuation merging problem where byte-level heuristics cannot distinguish continuation bytes of letters from continuation bytes of punctuation.

## Usage

### Development

Serve the project root with any static file server:

```sh
# Python
python3 -m http.server 8080

# Node
npx -y serve .
```

Open `http://localhost:8080` in a WebGPU-capable browser (Chrome 113+, Edge 113+, or Firefox Nightly with `dom.webgpu.enabled`).

### Training

1. Drop text files or select a folder
2. Pick a vocabulary size (1K to 64K)
3. Click Train

The log panel shows real-time progress: merge count, merges/sec, and ETA.

### Encoding

After training, switch to the Tokenizer tab. Type or paste text to see the token breakdown with IDs.

### Importing a vocabulary

You can also load a pre-trained vocabulary (JSON merge list) via the "Load Vocab" button without retraining.

No dependencies. No build step required.

## Browser Requirements

- WebGPU support (Chrome 113+, Edge 113+)
- `maxStorageBufferBindingSize >= 512 MB`
- `maxBufferSize >= 512 MB`

## License

MIT

## Author

[Uğur Toprakdeviren](https://github.com/toprakdeviren)
