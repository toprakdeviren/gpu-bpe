/**
 * Trie Tokenizer — GPU-accelerated BPE inference via trie walk
 *
 * Uploads compiled trie (nodes/edges) to GPU and performs chunked
 * tokenization with shared-memory root edge caching + depth-1 cache.
 *
 * Buffer pooling: persistent GPU buffers sized for a capacity threshold,
 * re-used across encode calls. Only re-allocated when input exceeds capacity.
 */

import { WORKGROUP_SIZE } from '../engine.js';
import { compileVocabToTrie, parseHeader, parseTrieBuffers } from './trie.js';
import { uploadBuffer, destroyBuffers } from '../gpu-utils.js';

// ─── Constants ───────────────────────────────────────────────

const DEFAULT_CHUNK_SIZE = 512;
const UTF8_REPLACEMENT = [0xEF, 0xBF, 0xBD]; // U+FFFD

// ─── Tokenizer ──────────────────────────────────────────────

export class TrieTokenizer {
    #engine;
    #device;
    #nodesBuf;
    #edgesBuf;
    #vocab;
    #chunkSize;

    // ── Buffer Pool ──────────────────────────────────────────
    // Persistent GPU buffers, re-used across encode calls.
    // Only re-created when input size exceeds current capacity.

    /** @type {number} Current pool capacity in bytes */
    #poolCapacity = 0;
    /** @type {GPUBuffer|null} */ #inputBuf = null;
    /** @type {GPUBuffer|null} */ #tokenBuf = null;
    /** @type {GPUBuffer|null} */ #countsBuf = null;
    /** @type {GPUBuffer|null} */ #offsetsBuf = null;
    /** @type {GPUBuffer|null} */ #totalBuf = null;
    /** @type {GPUBuffer|null} */ #compactBuf = null;

    // Persistent uniform buffers (updated via writeBuffer, never recreated)
    /** @type {GPUBuffer|null} */ #chunkedParamBuf = null;
    /** @type {GPUBuffer|null} */ #prefixSumParamBuf = null;
    /** @type {GPUBuffer|null} */ #compactParamBuf = null;

    /**
     * @param {import('../engine.js').BPEEngine} engine
     * @param {ArrayBuffer} trieData - Binary trie file contents
     * @param {number[][]} [vocab] - Vocab for decode (byte arrays)
     * @param {{ chunkSize?: number }} [options]
     */
    constructor(engine, trieData, vocab, options = {}) {
        this.#engine = engine;
        this.#device = engine.device;
        this.#vocab = vocab ?? Array.from({ length: 256 }, (_, i) => [i]);

        const header = parseHeader(trieData);
        const { nodes, edges } = parseTrieBuffers(trieData, header);

        this.nodeCount = header.nodeCount;
        this.edgeCount = header.edgeCount;
        this.maxTokenLen = header.maxTokenLen;

        // Adaptive chunk size
        const adaptiveChunk = Math.max(DEFAULT_CHUNK_SIZE, Math.min(2048, header.maxTokenLen * 8));
        this.#chunkSize = options.chunkSize ?? adaptiveChunk;

        this.#nodesBuf = uploadBuffer(this.#device, nodes, GPUBufferUsage.STORAGE);
        this.#edgesBuf = uploadBuffer(this.#device, edges, GPUBufferUsage.STORAGE);

        // Create persistent uniform buffers (16 bytes each, updated via writeBuffer)
        this.#chunkedParamBuf = this.#device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.#prefixSumParamBuf = this.#device.createBuffer({
            size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.#compactParamBuf = this.#device.createBuffer({
            size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        console.log(`[ok] TrieTokenizer: ${this.nodeCount} nodes, ${this.edgeCount} edges, chunk=${this.#chunkSize}`);
    }

    /**
     * Create a TrieTokenizer directly from a BPE vocabulary.
     *
     * @param {import('../engine.js').BPEEngine} engine
     * @param {number[][]} vocab
     * @param {{ chunkSize?: number }} [options]
     * @returns {TrieTokenizer}
     */
    static fromVocab(engine, vocab, options = {}) {
        const trieData = compileVocabToTrie(vocab);
        return new TrieTokenizer(engine, trieData, vocab, options);
    }

    // ─── Buffer Pool Management ─────────────────────────────

    /**
     * Ensure GPU buffer pool is large enough for the given input size.
     * Only re-allocates when capacity is exceeded (amortized O(1)).
     *
     * @param {number} inputLen - Input byte count
     */
    #ensurePoolCapacity(inputLen) {
        if (inputLen <= this.#poolCapacity) return;

        // Destroy existing pool if any
        if (this.#poolCapacity > 0) {
            destroyBuffers([
                this.#inputBuf, this.#tokenBuf, this.#countsBuf,
                this.#offsetsBuf, this.#totalBuf, this.#compactBuf,
            ]);
        }

        const device = this.#device;
        const maxBuf = this.#engine.limits.maxBufferSize;

        // Cap input size: token buffer and compact buffer are each ~4× inputSize,
        // so inputSize ≤ maxBuf/4 ensures all derived buffers ≤ maxBuf.
        const maxPoolInput = Math.floor(maxBuf / 4);
        const inputSize = Math.min(
            Math.ceil(inputLen * 1.5 / 4) * 4,   // 1.5× amortized growth
            maxPoolInput,                          // GPU constraint
        );
        // Derive chunk/token counts from actual buffer capacity
        const numChunks = Math.ceil(inputSize / this.#chunkSize);
        const maxTokensPerChunk = this.#chunkSize;

        // Derived buffer sizes (all guaranteed ≤ maxBuf since inputSize ≤ maxBuf/4)
        const tokenSize = numChunks * maxTokensPerChunk * 4;  // ≈ inputSize × 4
        const countsSize = numChunks * 4;
        const offsetsSize = numChunks * 4;
        const compactSize = inputSize * 4;  // worst case: 1 token per byte

        this.#inputBuf = device.createBuffer({
            size: inputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.#tokenBuf = device.createBuffer({
            size: tokenSize,
            usage: GPUBufferUsage.STORAGE,
        });
        this.#countsBuf = device.createBuffer({
            size: countsSize,
            usage: GPUBufferUsage.STORAGE,
        });
        this.#offsetsBuf = device.createBuffer({
            size: offsetsSize,
            usage: GPUBufferUsage.STORAGE,
        });
        this.#totalBuf = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.#compactBuf = device.createBuffer({
            size: compactSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        this.#poolCapacity = inputSize;
        console.log(`  [pool] resized: input=${(inputSize / 1048576).toFixed(0)}MB, tokens=${(tokenSize / 1048576).toFixed(0)}MB, compact=${(compactSize / 1048576).toFixed(0)}MB (max=${(maxBuf / 1048576).toFixed(0)}MB)`);
    }

    /**
     * Tokenize raw bytes into token IDs
     * @param {Uint8Array} bytes
     * @returns {Promise<Uint32Array>}
     */
    async encodeBytes(bytes) {
        const inputLen = bytes.length;
        if (inputLen === 0) return new Uint32Array(0);

        const maxBuf = this.#engine.limits.maxBufferSize;
        // Token buffer is the tightest constraint:
        //   numChunks * chunkSize * 4 ≤ maxBuf  →  maxInput = maxBuf / 4
        // Also limited by input buffer itself (maxBuf) and compact buffer (maxBuf/4)
        const maxInputPerPass = Math.floor(maxBuf / 4);
        // Align to chunk boundaries to avoid splitting mid-chunk
        const sliceSize = Math.floor(maxInputPerPass / this.#chunkSize) * this.#chunkSize;

        if (inputLen > sliceSize) {
            // Multi-pass: split input into slices that fit in GPU buffers
            const parts = [];
            for (let offset = 0; offset < inputLen; offset += sliceSize) {
                const end = Math.min(offset + sliceSize, inputLen);
                const slice = bytes.subarray(offset, end);
                const tokens = await this.#encodeSinglePass(slice);
                parts.push(tokens);
            }
            // Merge all token arrays
            const totalLen = parts.reduce((sum, p) => sum + p.length, 0);
            const merged = new Uint32Array(totalLen);
            let off = 0;
            for (const p of parts) {
                merged.set(p, off);
                off += p.length;
            }
            return merged;
        }

        return this.#encodeSinglePass(bytes);
    }

    /**
     * Tokenize a single pass (input must fit in GPU buffers)
     * @param {Uint8Array} bytes
     * @returns {Promise<Uint32Array>}
     */
    async #encodeSinglePass(bytes) {
        const inputLen = bytes.length;
        if (inputLen === 0) return new Uint32Array(0);

        const device = this.#device;
        const numChunks = Math.ceil(inputLen / this.#chunkSize);
        const maxTokensPerChunk = this.#chunkSize;

        // ── Ensure pool capacity & upload input ──────────────────

        this.#ensurePoolCapacity(inputLen);

        // writeBuffer requires 4-byte aligned data size
        const alignedLen = Math.ceil(inputLen / 4) * 4;
        if (inputLen === alignedLen) {
            device.queue.writeBuffer(this.#inputBuf, 0, bytes);
        } else {
            const padded = new Uint8Array(alignedLen);
            padded.set(bytes);
            device.queue.writeBuffer(this.#inputBuf, 0, padded);
        }

        // ── Update uniform params (no allocation — just writeBuffer) ─

        device.queue.writeBuffer(this.#chunkedParamBuf, 0,
            new Uint32Array([inputLen, this.#chunkSize, maxTokensPerChunk, 0]));
        device.queue.writeBuffer(this.#prefixSumParamBuf, 0,
            new Uint32Array([numChunks, 0]));
        device.queue.writeBuffer(this.#compactParamBuf, 0,
            new Uint32Array([maxTokensPerChunk, 0]));

        // ── Single submit: ALL 3 dispatches (GPU never idles) ────

        const enc = device.createCommandEncoder();

        // Pass 1: Chunked tokenization
        const p1 = enc.beginComputePass();
        p1.setPipeline(this.#engine.pipelines['trie_tokenizer_chunked']);
        p1.setBindGroup(0, device.createBindGroup({
            layout: this.#engine.pipelines['trie_tokenizer_chunked'].getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.#inputBuf } },
                { binding: 1, resource: { buffer: this.#nodesBuf } },
                { binding: 2, resource: { buffer: this.#edgesBuf } },
                { binding: 3, resource: { buffer: this.#tokenBuf } },
                { binding: 4, resource: { buffer: this.#countsBuf } },
                { binding: 5, resource: { buffer: this.#chunkedParamBuf } },
            ],
        }));
        p1.dispatchWorkgroups(Math.ceil(numChunks / WORKGROUP_SIZE));
        p1.end();

        // Pass 2: GPU prefix sum
        const p2 = enc.beginComputePass();
        p2.setPipeline(this.#engine.pipelines['trie_prefix_sum']);
        p2.setBindGroup(0, device.createBindGroup({
            layout: this.#engine.pipelines['trie_prefix_sum'].getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.#countsBuf } },
                { binding: 1, resource: { buffer: this.#offsetsBuf } },
                { binding: 2, resource: { buffer: this.#totalBuf } },
                { binding: 3, resource: { buffer: this.#prefixSumParamBuf } },
            ],
        }));
        p2.dispatchWorkgroups(1);
        p2.end();

        // Pass 3: Compact (1 workgroup = 1 chunk, coalesced writes)
        const p3 = enc.beginComputePass();
        p3.setPipeline(this.#engine.pipelines['trie_tokenizer_compact']);
        p3.setBindGroup(0, device.createBindGroup({
            layout: this.#engine.pipelines['trie_tokenizer_compact'].getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.#tokenBuf } },
                { binding: 1, resource: { buffer: this.#countsBuf } },
                { binding: 2, resource: { buffer: this.#offsetsBuf } },
                { binding: 3, resource: { buffer: this.#compactBuf } },
                { binding: 4, resource: { buffer: this.#compactParamBuf } },
            ],
        }));
        // dispatch numChunks workgroups (1 WG = 1 chunk); use 2D grid if > 65535
        if (numChunks <= 65535) {
            p3.dispatchWorkgroups(numChunks);
        } else {
            const x = Math.min(numChunks, 65535);
            const y = Math.ceil(numChunks / x);
            p3.dispatchWorkgroups(x, y);
        }
        p3.end();

        // Read total (4 bytes) — ALL compute already finished
        const totalReadBuf = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        enc.copyBufferToBuffer(this.#totalBuf, 0, totalReadBuf, 0, 4);
        device.queue.submit([enc.finish()]);

        // Near-instant: GPU finished all 3 passes before we get here
        await totalReadBuf.mapAsync(GPUMapMode.READ);
        const totalTokens = new Uint32Array(totalReadBuf.getMappedRange())[0];
        totalReadBuf.unmap();
        totalReadBuf.destroy();

        if (totalTokens === 0) return new Uint32Array(0);

        // ── Submit 2: ZERO compute — pure DMA copy of exact size ─

        const resultReadBuf = device.createBuffer({
            size: totalTokens * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const enc2 = device.createCommandEncoder();
        enc2.copyBufferToBuffer(this.#compactBuf, 0, resultReadBuf, 0, totalTokens * 4);
        device.queue.submit([enc2.finish()]);

        await resultReadBuf.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(resultReadBuf.getMappedRange().slice(0));
        resultReadBuf.unmap();
        resultReadBuf.destroy();

        return result;
    }

    // ─── Decode ─────────────────────────────────────────────

    /**
     * Decode token IDs back to raw bytes
     * @param {Uint32Array|number[]} tokens
     * @returns {Uint8Array}
     */
    decode(tokens) {
        const chunks = [];
        let totalLen = 0;

        for (const token of tokens) {
            const idx = Number(token);
            const bytes = idx < this.#vocab.length ? this.#vocab[idx] : UTF8_REPLACEMENT;
            chunks.push(bytes);
            totalLen += bytes.length;
        }

        const result = new Uint8Array(totalLen);
        let offset = 0;
        for (const chunk of chunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }

        return result;
    }

    // ─── Cleanup ────────────────────────────────────────────

    /** Release all GPU resources. */
    destroy() {
        destroyBuffers([this.#nodesBuf, this.#edgesBuf]);
        destroyBuffers([this.#chunkedParamBuf, this.#prefixSumParamBuf, this.#compactParamBuf]);

        if (this.#poolCapacity > 0) {
            destroyBuffers([
                this.#inputBuf, this.#tokenBuf, this.#countsBuf,
                this.#offsetsBuf, this.#totalBuf, this.#compactBuf,
            ]);
        }
    }
}
