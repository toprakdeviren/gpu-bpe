/**
 * Trie Tokenizer — GPU-accelerated BPE inference via trie walk
 *
 * Parses binary trie data, uploads nodes/edges to GPU, and performs
 * chunked tokenization with shared-memory root edge caching.
 */

import { WORKGROUP_SIZE, INVALID_TOKEN } from './engine.js';
import { compileVocabToTrie } from './trie-compiler.js';

// ─── Constants ───────────────────────────────────────────────
const TRIE_MAGIC = 0x54524945;
const SUPPORTED_VERSIONS = new Set([2, 3]);
const HEADER_SIZE = 28;
const DEFAULT_CHUNK_SIZE = 512;
const UTF8_REPLACEMENT = [0xEF, 0xBF, 0xBD]; // U+FFFD

// ─── Trie Binary Parsing ────────────────────────────────────

/** @typedef {{ nodeCount: number, edgeCount: number, maxTokenLen: number, version: number }} TrieHeader */

/**
 * @param {ArrayBuffer} data
 * @returns {TrieHeader}
 */
function parseHeader(data) {
    const view = new DataView(data, 0, HEADER_SIZE);
    const magic = view.getUint32(0, true);
    const version = view.getUint32(4, true);

    if (magic !== TRIE_MAGIC) {
        throw new Error(`Invalid trie magic: 0x${magic.toString(16)}`);
    }
    if (!SUPPORTED_VERSIONS.has(version)) {
        throw new Error(`Unsupported trie version: ${version}`);
    }

    return {
        version,
        nodeCount: view.getUint32(8, true),
        edgeCount: view.getUint32(12, true),
        maxTokenLen: view.getUint32(16, true),
    };
}

/**
 * @param {ArrayBuffer} data
 * @param {TrieHeader} header
 * @returns {{ nodes: Uint32Array, edges: Uint32Array }}
 */
function parseTrieBuffers(data, header) {
    const { version, nodeCount, edgeCount } = header;

    const bytesPerNode = version === 3 ? 12 : 8;
    const bytesPerEdge = version === 3 ? 8 : 4;
    const nodeBytes = nodeCount * bytesPerNode;
    const edgeBytes = edgeCount * bytesPerEdge;

    if (data.byteLength < HEADER_SIZE + nodeBytes + edgeBytes) {
        throw new Error('Truncated trie data');
    }

    const nodes = parseNodes(
        new DataView(data, HEADER_SIZE, nodeBytes),
        nodeCount, version, bytesPerNode,
    );

    const edges = parseEdges(
        new DataView(data, HEADER_SIZE + nodeBytes, edgeBytes),
        edgeCount, version, bytesPerEdge,
    );

    return { nodes, edges };
}

/**
 * Pack nodes into flat u32×3 array: [firstChild, numChildren, tokenId]
 */
function parseNodes(view, count, version, stride) {
    const packed = new Uint32Array(count * 3);

    for (let i = 0; i < count; i++) {
        const src = i * stride;
        const dst = i * 3;

        if (version === 3) {
            packed[dst] = view.getUint32(src, true);
            packed[dst + 1] = view.getUint32(src + 4, true);
            packed[dst + 2] = view.getUint32(src + 8, true);
        } else {
            packed[dst] = view.getUint16(src, true);
            packed[dst + 1] = view.getUint16(src + 2, true);
            const tokenId = view.getUint16(src + 4, true);
            packed[dst + 2] = tokenId === 0xFFFF ? INVALID_TOKEN : tokenId;
        }
    }

    return packed;
}

/**
 * Pack edges into flat u32×2 array: [symbol, targetNode]
 */
function parseEdges(view, count, version, stride) {
    const packed = new Uint32Array(count * 2);

    for (let i = 0; i < count; i++) {
        const src = i * stride;
        const dst = i * 2;

        if (version === 3) {
            packed[dst] = view.getUint8(src);
            packed[dst + 1] = view.getUint32(src + 4, true);
        } else {
            packed[dst] = view.getUint16(src, true) & 0xFF;
            packed[dst + 1] = view.getUint16(src + 2, true);
        }
    }

    return packed;
}

// ─── GPU Buffer Helpers ─────────────────────────────────────

function uploadBuffer(device, data, usage) {
    const buf = device.createBuffer({
        size: data.byteLength,
        usage: usage | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new data.constructor(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
}

function destroyBuffers(buffers) {
    for (const buf of buffers) buf.destroy();
}

// ─── Pure Utilities ─────────────────────────────────────────

/**
 * @param {Uint32Array} counts
 * @returns {{ offsets: Uint32Array, total: number }}
 */
function computePrefixSum(counts) {
    const offsets = new Uint32Array(counts.length);
    let total = 0;
    for (let i = 0; i < counts.length; i++) {
        offsets[i] = total;
        total += counts[i];
    }
    return { offsets, total };
}

/**
 * @param {number} size
 * @returns {number[][]}
 */
function buildByteVocab(size = 256) {
    return Array.from({ length: size }, (_, i) => [i]);
}

// ─── Tokenizer ──────────────────────────────────────────────

export class TrieTokenizer {
    #engine;
    #device;
    #nodesBuf;
    #edgesBuf;
    #vocab;
    #chunkSize;

    /**
     * @param {import('./engine.js').BPEEngine} engine
     * @param {ArrayBuffer} trieData - Binary trie file contents
     * @param {number[][]} [vocab] - Vocab for decode (byte arrays)
     * @param {{ chunkSize?: number }} [options]
     */
    constructor(engine, trieData, vocab, options = {}) {
        this.#engine = engine;
        this.#device = engine.device;
        this.#chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
        this.#vocab = vocab ?? buildByteVocab();

        const header = parseHeader(trieData);
        const { nodes, edges } = parseTrieBuffers(trieData, header);

        this.nodeCount = header.nodeCount;
        this.edgeCount = header.edgeCount;
        this.maxTokenLen = header.maxTokenLen;

        this.#nodesBuf = uploadBuffer(this.#device, nodes, GPUBufferUsage.STORAGE);
        this.#edgesBuf = uploadBuffer(this.#device, edges, GPUBufferUsage.STORAGE);

        console.log(`[ok] TrieTokenizer: ${this.nodeCount} nodes, ${this.edgeCount} edges`);
    }

    /**
     * Create a TrieTokenizer directly from a BPE vocabulary.
     * Compiles the vocab into a binary trie in-memory (no .trie file needed).
     *
     * @param {import('./engine.js').BPEEngine} engine
     * @param {number[][]} vocab - Vocab byte arrays from BPE training
     * @param {{ chunkSize?: number }} [options]
     * @returns {TrieTokenizer}
     */
    static fromVocab(engine, vocab, options = {}) {
        const trieData = compileVocabToTrie(vocab);
        return new TrieTokenizer(engine, trieData, vocab, options);
    }

    // ─── Encode ─────────────────────────────────────────────

    /**
     * Tokenize UTF-8 text into token IDs using GPU trie walk
     * @param {string} text
     * @returns {Promise<Uint32Array>}
     */
    async encode(text) {
        return this.encodeBytes(new TextEncoder().encode(text));
    }

    /**
     * Tokenize raw bytes into token IDs
     * @param {Uint8Array} bytes
     * @returns {Promise<Uint32Array>}
     */
    async encodeBytes(bytes) {
        const inputLen = bytes.length;
        if (inputLen === 0) return new Uint32Array(0);

        const numChunks = Math.ceil(inputLen / this.#chunkSize);
        const maxTokensPerChunk = this.#chunkSize;

        // Phase 1: Chunked tokenization → get per-chunk token counts
        const { counts, phase1Buffers } = await this.#runChunkedTokenization(bytes, numChunks, maxTokensPerChunk);

        // CPU prefix sum (numChunks is small)
        const { offsets, total } = computePrefixSum(counts);

        if (total === 0) {
            destroyBuffers(Object.values(phase1Buffers));
            return new Uint32Array(0);
        }

        // Phase 2: Compact scattered tokens into final array
        const result = await this.#runCompaction(phase1Buffers, numChunks, maxTokensPerChunk, offsets, total);

        return result;
    }

    /**
     * @returns {Promise<{ counts: Uint32Array, phase1Buffers: { tokenBuf: GPUBuffer, countsBuf: GPUBuffer } }>}
     */
    async #runChunkedTokenization(bytes, numChunks, maxTokensPerChunk) {
        const device = this.#device;
        const inputLen = bytes.length;

        // Upload input bytes as u32 (WGSL array<u32>)
        const inputU32 = new Uint32Array(inputLen);
        for (let i = 0; i < inputLen; i++) inputU32[i] = bytes[i];

        const inputBuf = uploadBuffer(device, inputU32, GPUBufferUsage.STORAGE);
        const tokenBuf = device.createBuffer({
            size: numChunks * maxTokensPerChunk * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const countsBuf = device.createBuffer({
            size: numChunks * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const paramBuf = uploadBuffer(
            device,
            new Uint32Array([inputLen, this.#chunkSize, maxTokensPerChunk, 0]),
            GPUBufferUsage.UNIFORM,
        );

        const pipeline = this.#engine.pipelines['trie_tokenizer_chunked'];
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inputBuf } },
                { binding: 1, resource: { buffer: this.#nodesBuf } },
                { binding: 2, resource: { buffer: this.#edgesBuf } },
                { binding: 3, resource: { buffer: tokenBuf } },
                { binding: 4, resource: { buffer: countsBuf } },
                { binding: 5, resource: { buffer: paramBuf } },
            ],
        });

        // Dispatch + readback counts
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(numChunks / WORKGROUP_SIZE));
        pass.end();

        const countsReadBuf = device.createBuffer({
            size: numChunks * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        encoder.copyBufferToBuffer(countsBuf, 0, countsReadBuf, 0, numChunks * 4);
        device.queue.submit([encoder.finish()]);

        await countsReadBuf.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(countsReadBuf.getMappedRange().slice(0));
        countsReadBuf.unmap();
        destroyBuffers([inputBuf, paramBuf, countsReadBuf]);

        return { counts, phase1Buffers: { tokenBuf, countsBuf } };
    }

    /**
     * @returns {Promise<Uint32Array>}
     */
    async #runCompaction(phase1Buffers, numChunks, maxTokensPerChunk, offsets, totalTokens) {
        const device = this.#device;
        const { tokenBuf, countsBuf } = phase1Buffers;

        const offsetsBuf = uploadBuffer(device, offsets, GPUBufferUsage.STORAGE);
        const compactBuf = device.createBuffer({
            size: totalTokens * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const compactParamBuf = uploadBuffer(
            device,
            new Uint32Array([maxTokensPerChunk, 0]),
            GPUBufferUsage.UNIFORM,
        );

        const pipeline = this.#engine.pipelines['trie_tokenizer_compact'];
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: tokenBuf } },
                { binding: 1, resource: { buffer: countsBuf } },
                { binding: 2, resource: { buffer: offsetsBuf } },
                { binding: 3, resource: { buffer: compactBuf } },
                { binding: 4, resource: { buffer: compactParamBuf } },
            ],
        });

        // Dispatch + readback result
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(numChunks / WORKGROUP_SIZE));
        pass.end();

        const resultReadBuf = device.createBuffer({
            size: totalTokens * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        encoder.copyBufferToBuffer(compactBuf, 0, resultReadBuf, 0, totalTokens * 4);
        device.queue.submit([encoder.finish()]);

        await resultReadBuf.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(resultReadBuf.getMappedRange().slice(0));
        resultReadBuf.unmap();
        destroyBuffers([tokenBuf, countsBuf, offsetsBuf, compactBuf, compactParamBuf, resultReadBuf]);

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

    /**
     * Decode token IDs back to string
     * @param {Uint32Array|number[]} tokens
     * @returns {string}
     */
    decodeToString(tokens) {
        return new TextDecoder().decode(this.decode(tokens));
    }
}
