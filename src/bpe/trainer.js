/**
 * BPE Trainer — GPU-accelerated BPE vocabulary training
 *
 * Manages the BPE training pipeline: word boundary detection, pair counting,
 * max pair finding, merging, and compaction.
 *
 * Text normalization (NFC, control chars, whitespace) is handled by the Decoder
 * WASM module before data is passed to the trainer.
 */

import { WORKGROUP_SIZE, TABLE_SIZE, dispatch2D } from './engine.js';

// ─── Constants ──────────────────────────────────────────────

const WORD_START_BIT = 0x10000;       // bit 16 — must match WGSL constant
const BATCH_SIZE = 128;
const MERGE_LOG_STRIDE = 3;           // [pair, newTokenId, count] per merge
const ITER_STATE_SIZE = 48;           // 12 × u32

const BUFFER_USAGE = {
    STORAGE_SRC: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    STORAGE_ALL: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    READBACK: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
};

// ─── Display Helpers ────────────────────────────────────────

/**
 * Convert a byte sequence to a human-readable display string.
 * Tries UTF-8 decode; falls back to hex for non-printable/invalid bytes.
 * Space (0x20) is shown as '▁' for visibility.
 *
 * @param {number[]} bytes
 * @returns {string}
 */
function bytesToDisplayString(bytes) {
    const parts = [];
    let i = 0;

    while (i < bytes.length) {
        const b = bytes[i];

        // ASCII range
        if (b < 0x80) {
            parts.push(formatAsciiByte(b));
            i++;
            continue;
        }

        // Orphan continuation byte
        if (b < 0xC0) {
            parts.push(formatHexByte(b));
            i++;
            continue;
        }

        // Multi-byte UTF-8 start
        const seqLen = b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
        const decoded = tryDecodeUtf8(bytes, i, seqLen);

        if (decoded !== null) {
            parts.push(decoded);
            i += seqLen;
        } else {
            parts.push(formatHexByte(b));
            i++;
        }
    }

    return parts.join('');
}

/** @param {number} b */
function formatAsciiByte(b) {
    if (b === 0x20) return '▁';
    if (b === 0x0A) return '\\n';
    if (b >= 0x21 && b <= 0x7E) return String.fromCharCode(b);
    return formatHexByte(b);
}

/** @param {number} b */
function formatHexByte(b) {
    return `<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`;
}

/**
 * @param {number[]} bytes
 * @param {number} offset
 * @param {number} len
 * @returns {string|null}
 */
function tryDecodeUtf8(bytes, offset, len) {
    if (offset + len > bytes.length) return null;

    // Validate continuation bytes
    for (let j = 1; j < len; j++) {
        if ((bytes[offset + j] & 0xC0) !== 0x80) return null;
    }

    try {
        const slice = new Uint8Array(bytes.slice(offset, offset + len));
        return new TextDecoder('utf-8', { fatal: true }).decode(slice);
    } catch {
        return null;
    }
}

// ─── Time Formatting ────────────────────────────────────────

/** @param {number} seconds */
function formatDuration(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

// ─── GPU Buffer Utilities ───────────────────────────────────

/**
 * @param {GPUDevice} device
 * @param {TypedArray} data
 * @param {GPUBufferUsageFlags} usage
 * @returns {GPUBuffer}
 */
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

/**
 * @param {GPUDevice} device
 * @param {number} size
 * @param {GPUBufferUsageFlags} usage
 * @returns {GPUBuffer}
 */
function allocBuffer(device, size, usage) {
    return device.createBuffer({ size, usage });
}

/** @param {GPUBuffer[]} buffers */
function destroyBuffers(buffers) {
    for (const buf of buffers) buf.destroy();
}

/** Yield to event loop so browser can repaint */
function yieldToEventLoop() {
    return new Promise(r => setTimeout(r, 0));
}

// ─── Bytes → Symbols ────────────────────────────────────────

/**
 * @param {Uint8Array} data
 * @returns {Uint32Array}
 */
function bytesToSymbols(data) {
    const symbols = new Uint32Array(data.length);
    for (let i = 0; i < data.length; i++) symbols[i] = data[i];
    return symbols;
}

// ─── Vocab ──────────────────────────────────────────────────

class Vocab {
    /** @type {number[][]} */
    entries = [];
    /** @type {string[]} */
    strings = [];
    /** @type {number} */
    nextTokenId = 256;

    constructor() {
        // Initialize 256 single-byte base tokens
        for (let i = 0; i < 256; i++) {
            this.entries.push([i]);
            this.strings.push(bytesToDisplayString([i]));
        }
    }

    get size() {
        return this.entries.length;
    }

    /**
     * Register a new merged token
     * @param {number} symbolA
     * @param {number} symbolB
     * @returns {number} newTokenId
     */
    addMerge(symbolA, symbolB) {
        const newTokenId = this.nextTokenId++;
        const merged = [...this.entries[symbolA], ...this.entries[symbolB]];
        this.entries.push(merged);
        this.strings.push(bytesToDisplayString(merged));
        return newTokenId;
    }

    /**
     * Export vocab as human-readable text
     * @returns {string}
     */
    export() {
        const lines = [
            `# GPU BPE Vocabulary (WebGPU Trainer)`,
            `# Total tokens: ${this.entries.length}`,
            '',
        ];

        for (let i = 0; i < this.entries.length; i++) {
            const bytes = this.entries[i].join(',');
            lines.push(`${i}\t${this.strings[i]}\t[${bytes}]`);
        }

        return lines.join('\n') + '\n';
    }
}

// ─── Training Buffers ───────────────────────────────────────

/**
 * Pre-allocate all persistent GPU buffers needed during training.
 *
 * @param {GPUDevice} device
 * @param {number} maxSymbols
 * @returns {TrainingBuffers}
 */
function allocTrainingBuffers(device, maxSymbols) {
    const maxBlocks = Math.ceil(maxSymbols / WORKGROUP_SIZE);
    const findMaxBlocks = Math.ceil(TABLE_SIZE / 256);

    return {
        // Hash table
        pairCounts: allocBuffer(device, TABLE_SIZE * 4, BUFFER_USAGE.STORAGE_SRC),
        pairIds: allocBuffer(device, TABLE_SIZE * 4, BUFFER_USAGE.STORAGE_SRC),

        // FindMax reduction
        blockMaxCounts: allocBuffer(device, findMaxBlocks * 4, GPUBufferUsage.STORAGE),
        blockMaxPairIds: allocBuffer(device, findMaxBlocks * 4, GPUBufferUsage.STORAGE),
        maxCount: allocBuffer(device, 4, BUFFER_USAGE.STORAGE_SRC),
        maxPairId: allocBuffer(device, 4, BUFFER_USAGE.STORAGE_SRC),

        // Compaction
        validMask: allocBuffer(device, maxSymbols * 4, GPUBufferUsage.STORAGE),
        blockSums: allocBuffer(device, maxBlocks * 4, GPUBufferUsage.STORAGE),
        totalValid: allocBuffer(device, 4, BUFFER_USAGE.STORAGE_SRC),
        compact: allocBuffer(device, maxSymbols * 4, BUFFER_USAGE.STORAGE_SRC),

        // Batched iteration
        iterState: allocBuffer(device, ITER_STATE_SIZE, BUFFER_USAGE.STORAGE_ALL),
        mergeLog: allocBuffer(device, BATCH_SIZE * MERGE_LOG_STRIDE * 4, BUFFER_USAGE.STORAGE_SRC),

        // Indirect dispatch buffer (3 × u32: wgX, wgY, wgZ)
        // Written by bpe_update_count kernel, read by dispatchWorkgroupsIndirect
        indirectDispatch: allocBuffer(device, 12,
            GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST),

        // Readback staging
        readbackState: allocBuffer(device, ITER_STATE_SIZE, BUFFER_USAGE.READBACK),
        readbackLog: allocBuffer(device, BATCH_SIZE * MERGE_LOG_STRIDE * 4, BUFFER_USAGE.READBACK),

        // Uniform param buffers (allocated once, reused across bind group recreations)
        clearTableParam: uploadBuffer(device, new Uint32Array([TABLE_SIZE, 0]), GPUBufferUsage.UNIFORM),
        findMaxParam1: uploadBuffer(device, new Uint32Array([TABLE_SIZE, 0]), GPUBufferUsage.UNIFORM),
        findMaxParam2: uploadBuffer(device, new Uint32Array([findMaxBlocks, 0]), GPUBufferUsage.UNIFORM),

        // Derived constants
        maxBlocks,
        findMaxBlocks,
        maxSymbols,
    };
}

/** @param {TrainingBuffers} b */
function destroyTrainingBuffers(b) {
    destroyBuffers([
        b.pairCounts, b.pairIds,
        b.blockMaxCounts, b.blockMaxPairIds,
        b.maxCount, b.maxPairId,
        b.validMask, b.blockSums, b.totalValid, b.compact,
        b.iterState, b.mergeLog, b.indirectDispatch,
        b.readbackState, b.readbackLog,
        b.clearTableParam, b.findMaxParam1, b.findMaxParam2,
    ]);
}

// ─── Bind Group Factory ─────────────────────────────────────

/**
 * Initialize all bind groups needed for batched training.
 *
 * @param {GPUDevice} device
 * @param {Record<string, GPUComputePipeline>} pipelines
 * @param {GPUBuffer} symbolBufA
 * @param {GPUBuffer} symbolBufB
 * @param {TrainingBuffers} tb
 * @returns {Record<string, GPUBindGroup>}
 */
function buildBindGroups(device, pipelines, symbolBufA, symbolBufB, tb) {
    const p = pipelines;

    /** Helper to init a bind group with less boilerplate */
    const bg = (pipeline, bindings) => device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    return {
        clearTable: bg(p.bpe_clear_table, [
            tb.pairCounts, tb.pairIds, tb.clearTableParam,
        ]),

        // Ping-pong pair counting
        pairCountA: bg(p.bpe_pair_count_b, [
            symbolBufA, tb.pairCounts, tb.pairIds, tb.iterState,
        ]),
        pairCountB: bg(p.bpe_pair_count_b, [
            symbolBufB, tb.pairCounts, tb.pairIds, tb.iterState,
        ]),

        // Two-pass findMax
        findMax1: bg(p.bpe_find_max_pair, [
            tb.pairCounts, tb.pairIds, tb.blockMaxCounts, tb.blockMaxPairIds, tb.findMaxParam1,
        ]),
        findMax2: bg(p.bpe_find_max_pair_final, [
            tb.blockMaxCounts, tb.blockMaxPairIds, tb.maxCount, tb.maxPairId, tb.findMaxParam2,
        ]),

        // Merge setup
        setupMerge: bg(p.bpe_setup_merge, [
            tb.maxCount, tb.maxPairId, tb.iterState, tb.mergeLog,
        ]),

        // Fill valid mask
        fillValid: bg(p.bpe_fill_valid_b, [
            tb.validMask, tb.iterState,
        ]),

        // Ping-pong merge
        mergeA: bg(p.bpe_merge_b, [
            symbolBufA, tb.validMask, tb.iterState,
        ]),
        mergeB: bg(p.bpe_merge_b, [
            symbolBufB, tb.validMask, tb.iterState,
        ]),

        // Two-phase prefix sum (reduce + scan)
        prefixReduce: bg(p.bpe_prefix_sum_reduce_b, [
            tb.validMask, tb.blockSums, tb.iterState,
        ]),
        prefixScan: bg(p.bpe_prefix_sum_scan_blocks_b, [
            tb.blockSums, tb.totalValid, tb.iterState,
        ]),

        // Fused finalize + compact (prefix_sum stays in registers)
        finalizeCompactAB: bg(p.bpe_finalize_compact_b, [
            tb.validMask, tb.blockSums, symbolBufA, symbolBufB, tb.iterState,
        ]),
        finalizeCompactBA: bg(p.bpe_finalize_compact_b, [
            tb.validMask, tb.blockSums, symbolBufB, symbolBufA, tb.iterState,
        ]),

        // Update count + indirect dispatch
        updateCount: bg(p.bpe_update_count, [
            tb.totalValid, tb.iterState, tb.indirectDispatch,
        ]),
    };
}

// ─── Batch Encoding ─────────────────────────────────────────

/**
 * Encode one batch of BPE merge iterations into a command encoder.
 */
function encodeBatch(cmd, pipelines, bg, batchMerges, maxDispatch, maxBlocks, findMaxBlocks, indirectBuf) {
    const p = pipelines;

    for (let i = 0; i < batchMerges; i++) {
        const even = (i % 2 === 0);

        // Phase A: clear → count → findMax → setupMerge
        encodePass(cmd, p.bpe_clear_table, bg.clearTable,
            Math.ceil(TABLE_SIZE / WORKGROUP_SIZE));

        // First iteration uses static maxDispatch; subsequent use indirect buffer
        if (i === 0) {
            encodePass(cmd, p.bpe_pair_count_b, even ? bg.pairCountA : bg.pairCountB,
                maxDispatch);
        } else {
            encodeIndirectPass(cmd, p.bpe_pair_count_b,
                even ? bg.pairCountA : bg.pairCountB, indirectBuf);
        }

        encodePass(cmd, p.bpe_find_max_pair, bg.findMax1, findMaxBlocks);
        encodePass(cmd, p.bpe_find_max_pair_final, bg.findMax2, 1);
        encodePass(cmd, p.bpe_setup_merge, bg.setupMerge, 1);

        // Phase B: fillValid → merge → prefixReduce → prefixScan → finalizeCompact → updateCount
        // All dynamic kernels use indirect dispatch (except first iteration)
        if (i === 0) {
            encodePass(cmd, p.bpe_fill_valid_b, bg.fillValid, maxDispatch);
            encodePass(cmd, p.bpe_merge_b, even ? bg.mergeA : bg.mergeB, maxDispatch);
            encodePass(cmd, p.bpe_prefix_sum_reduce_b, bg.prefixReduce, maxBlocks);
            encodePass(cmd, p.bpe_prefix_sum_scan_blocks_b, bg.prefixScan, 1);
            encodePass(cmd, p.bpe_finalize_compact_b,
                even ? bg.finalizeCompactAB : bg.finalizeCompactBA, maxDispatch);
        } else {
            encodeIndirectPass(cmd, p.bpe_fill_valid_b, bg.fillValid, indirectBuf);
            encodeIndirectPass(cmd, p.bpe_merge_b, even ? bg.mergeA : bg.mergeB, indirectBuf);
            encodeIndirectPass(cmd, p.bpe_prefix_sum_reduce_b, bg.prefixReduce, indirectBuf);
            encodePass(cmd, p.bpe_prefix_sum_scan_blocks_b, bg.prefixScan, 1);
            encodeIndirectPass(cmd, p.bpe_finalize_compact_b,
                even ? bg.finalizeCompactAB : bg.finalizeCompactBA, indirectBuf);
        }

        // Update count + write next indirect dispatch params
        encodePass(cmd, p.bpe_update_count, bg.updateCount, 1);
    }
}

/**
 * Encode a single compute pass with indirect dispatch.
 */
function encodeIndirectPass(cmd, pipeline, bindGroup, indirectBuffer) {
    const pass = cmd.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroupsIndirect(indirectBuffer, 0);
    pass.end();
}

/**
 * Encode a single compute pass.
 */
function encodePass(cmd, pipeline, bindGroup, workgroups) {
    const pass = cmd.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    dispatch2D(pass, workgroups);
    pass.end();
}

// ─── Input Preparation ──────────────────────────────────────

/**
 * Prepare input for training: pre-tokenize or fall back to byte-level.
 *
 * @param {Uint8Array|string} input
 * @param {import('../wasm/pre_tokenizer.mjs').PreTokenizer|null} preTokenizer
 * @returns {Promise<{ symbolData: Uint32Array, wordStarts: Uint8Array|null }>}
 */
async function prepareInput(input, preTokenizer) {
    // ── Byte-native path: Uint8Array → WASM normalizeBytes → classifyBytes → boundaries ──
    if (preTokenizer && input instanceof Uint8Array) {
        console.log('   Mode: WASM pre-tokenized bytes (zero-copy path)');
        console.log(`   Input: ${(input.length / 1048576).toFixed(1)} MB raw bytes`);
        await yieldToEventLoop();

        const result = preTokenizer.preTokenizeBytes(input);

        if (result.bytes.length === 0 && input.length > 0) {
            console.warn('   ⚠ preTokenizeBytes returned 0 bytes — falling back to byte-level');
            return { symbolData: bytesToSymbols(input), wordStarts: null };
        }

        const symbolData = new Uint32Array(result.bytes.length);
        for (let i = 0; i < result.bytes.length; i++) symbolData[i] = result.bytes[i];

        return { symbolData, wordStarts: result.wordStarts };
    }

    // ── String path: normalize via WASM string API → codepoint classify → boundaries ──
    if (preTokenizer && typeof input === 'string') {
        console.log('   Mode: WASM pre-tokenized (Unicode 17.0)');
        console.log(`   Input: ${input.length} chars (${(new TextEncoder().encode(input).length / 1048576).toFixed(1)} MB)`);
        await yieldToEventLoop();

        const result = preTokenizer.preTokenize(input);

        if (result.bytes.length === 0 && input.length > 0) {
            // PreTokenizer returned empty — WASM normalize may have failed on large input.
            // Fall back to byte-level encoding so training can proceed.
            console.warn('   ⚠ PreTokenizer returned 0 bytes — falling back to byte-level');
            const bytes = new TextEncoder().encode(input);
            return { symbolData: bytesToSymbols(bytes), wordStarts: null };
        }

        const symbolData = new Uint32Array(result.bytes.length);
        for (let i = 0; i < result.bytes.length; i++) symbolData[i] = result.bytes[i];

        return { symbolData, wordStarts: result.wordStarts };
    }

    // ── Fallback: raw byte-level (no pre-tokenization) ──
    console.log('   Mode: Byte-level (WASM-normalized)');
    const bytes = typeof input === 'string' ? new TextEncoder().encode(input) : input;

    return { symbolData: bytesToSymbols(bytes), wordStarts: null };
}

/**
 * Tag word boundaries into symbol data (mutates in place).
 *
 * @param {Uint32Array} symbolData
 * @param {Uint8Array} wordStarts
 */
function tagWordBoundaries(symbolData, wordStarts) {
    for (let i = 0; i < symbolData.length; i++) {
        if (wordStarts[i]) {
            symbolData[i] |= WORD_START_BIT;
        }
    }
}

// ─── BPE Trainer ────────────────────────────────────────────

export class BPETrainer {
    #engine;
    #device;
    #vocab;

    /**
     * @param {import('./engine.js').BPEEngine} engine
     */
    constructor(engine) {
        this.#engine = engine;
        this.#device = engine.device;
        this.#vocab = new Vocab();
    }

    /**
     * Train BPE on input text data.
     *
     * @param {Uint8Array|string} input - Raw text data (should be NFC-normalized)
     * @param {Object} options
     * @param {number} [options.targetVocabSize=4096]
     * @param {import('../wasm/pre_tokenizer.mjs').PreTokenizer} [options.preTokenizer]
     * @param {function} [options.onProgress]
     * @returns {Promise<TrainingResult>}
     */
    async train(input, { targetVocabSize = 4096, preTokenizer = null, onProgress = null } = {}) {
        const device = this.#device;
        const pipelines = this.#engine.pipelines;

        console.log('\n─── WebGPU BPE Training ───');
        console.log(`   Target: ${targetVocabSize} tokens`);

        // ── Prepare input ──
        const { symbolData, wordStarts } = await prepareInput(input, preTokenizer);
        const symbolCount = symbolData.length;
        console.log(`   Symbols: ${symbolCount}`);

        if (symbolCount === 0) {
            throw new Error('No symbols to train on — corpus is empty after pre-processing');
        }

        if (wordStarts) {
            tagWordBoundaries(symbolData, wordStarts);
            console.log('    → word boundaries tagged by WASM pre-tokenizer (Unicode-accurate)');
        }

        await yieldToEventLoop();

        // ── Upload symbols ──
        let symbolBuf = uploadBuffer(
            device, new Uint32Array(symbolData), BUFFER_USAGE.STORAGE_SRC,
        );

        if (!wordStarts) {
            await this.#runWordBoundary(symbolBuf, symbolCount);
            console.log('    → word boundaries tagged by GPU (byte-level heuristic)');
        }

        // ── Allocate training buffers ──
        const tb = allocTrainingBuffers(device, symbolCount);
        const maxDispatch = Math.ceil(symbolCount / WORKGROUP_SIZE);

        // ── Initialize iteration state ──
        device.queue.writeBuffer(tb.iterState, 0, new Uint32Array([
            symbolCount,              // symbol_count
            TABLE_SIZE,               // table_size
            0,                        // early_stop
            this.#vocab.nextTokenId,  // next_token_id
            0, 0, 0, 0,              // symbol_a, symbol_b, new_symbol, max_count
            0,                        // merges_done
            symbolCount,              // max_symbols
            0, 0,                     // padding
        ]));

        // ── Initialize indirect dispatch buffer ──
        const initWG = Math.ceil(symbolCount / WORKGROUP_SIZE);
        const wgX = Math.min(initWG, 65535);
        const wgY = initWG <= 65535 ? 1 : Math.ceil(initWG / 65535);
        device.queue.writeBuffer(tb.indirectDispatch, 0, new Uint32Array([wgX, wgY, 1]));

        // ── Build bind groups ──
        let bg = buildBindGroups(device, pipelines, symbolBuf, tb.compact, tb);

        // ── Training loop ──
        const mergesNeeded = targetVocabSize - this.#vocab.size;
        console.log(`\n    → training ${mergesNeeded} merges (batched, ${BATCH_SIZE}/batch)...`);

        const result = await this.#runTrainingLoop({
            tb, bg, symbolBuf, pipelines,
            mergesNeeded, maxDispatch,
        }, onProgress);

        // ── Cleanup ──
        destroyTrainingBuffers(tb);

        return result;
    }

    /**
     * @returns {Promise<TrainingResult>}
     */
    async #runTrainingLoop(ctx, onProgress) {
        const { tb, pipelines, mergesNeeded, maxDispatch } = ctx;
        let { bg, symbolBuf } = ctx;
        const device = this.#device;

        const startTime = performance.now();
        const merges = [];
        let totalMergesDone = 0;
        let earlyStop = false;

        while (totalMergesDone < mergesNeeded && !earlyStop) {
            const batchMerges = Math.min(BATCH_SIZE, mergesNeeded - totalMergesDone);

            // Reset batch merge counter
            device.queue.writeBuffer(tb.iterState, 32, new Uint32Array([0]));

            // ── Encode full batch ──
            const cmd = device.createCommandEncoder();

            encodeBatch(cmd, pipelines, bg, batchMerges,
                maxDispatch, tb.maxBlocks, tb.findMaxBlocks, tb.indirectDispatch);

            // Copy state + merge log for readback
            cmd.copyBufferToBuffer(tb.iterState, 0, tb.readbackState, 0, ITER_STATE_SIZE);
            cmd.copyBufferToBuffer(
                tb.mergeLog, 0, tb.readbackLog, 0,
                batchMerges * MERGE_LOG_STRIDE * 4,
            );

            device.queue.submit([cmd.finish()]);

            // ── Single readback per batch ──
            await tb.readbackState.mapAsync(GPUMapMode.READ);
            const stateData = new Uint32Array(tb.readbackState.getMappedRange().slice(0));
            tb.readbackState.unmap();

            await tb.readbackLog.mapAsync(GPUMapMode.READ);
            const logData = new Uint32Array(tb.readbackLog.getMappedRange().slice(0));
            tb.readbackLog.unmap();

            const mergesDone = stateData[8];
            const currentSymbolCount = stateData[0];
            earlyStop = stateData[2] !== 0;

            // ── Reconstruct vocab from merge log ──
            for (let i = 0; i < mergesDone; i++) {
                const pair = logData[i * MERGE_LOG_STRIDE];
                const symbolA = pair >>> 16;
                const symbolB = pair & 0xFFFF;
                const newTokenId = this.#vocab.addMerge(symbolA, symbolB);
                merges.push([symbolA, symbolB, newTokenId]);
            }

            totalMergesDone += mergesDone;

            // ── Handle ping-pong buffer swap ──
            if (mergesDone % 2 !== 0) {
                const temp = symbolBuf;
                symbolBuf = tb.compact;
                tb.compact = temp;
                bg = buildBindGroups(
                    device, pipelines, symbolBuf, tb.compact, tb,
                );
            }

            // ── Progress reporting ──
            const elapsed = (performance.now() - startTime) / 1000;
            const rate = totalMergesDone / elapsed;
            const lastMergeStr = mergesDone > 0
                ? this.#vocab.strings[this.#vocab.strings.length - 1]
                : '—';
            const lastCount = mergesDone > 0
                ? logData[(mergesDone - 1) * MERGE_LOG_STRIDE + 2]
                : 0;

            console.log(
                `   [${totalMergesDone}/${mergesNeeded}] '${lastMergeStr}' ` +
                `count:${lastCount} symbols:${currentSymbolCount} ` +
                `(${rate.toFixed(1)} merges/s)`,
            );

            if (onProgress) {
                onProgress({
                    mergeIndex: totalMergesDone,
                    totalMerges: mergesNeeded,
                    mergeString: lastMergeStr,
                    bestCount: lastCount,
                    symbolCount: currentSymbolCount,
                    mergesPerSecond: rate,
                });
            }

            await yieldToEventLoop();

            if (earlyStop) {
                console.log(`    ✗ early stop after ${totalMergesDone} merges`);
            }
        }

        const totalTime = (performance.now() - startTime) / 1000;
        console.log(`\n    ✓ training done: ${this.#vocab.size} tokens in ${formatDuration(totalTime)}`);
        console.log(`   Rate: ${(totalMergesDone / totalTime).toFixed(1)} merges/s`);

        return {
            vocab: this.#vocab.entries,
            vocabStrings: this.#vocab.strings,
            vocabSize: this.#vocab.size,
            merges,
            trainingTime: formatDuration(totalTime),
        };
    }

    // ─── GPU Word Boundary Detection ────────────────────────

    /**
     * Run GPU word boundary kernel (byte-level heuristic fallback).
     */
    async #runWordBoundary(symbolBuf, symbolCount) {
        const device = this.#device;
        const pipeline = this.#engine.pipelines.bpe_word_boundary;

        const paramBuf = uploadBuffer(
            device, new Uint32Array([symbolCount, 0]), GPUBufferUsage.UNIFORM,
        );

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: symbolBuf } },
                { binding: 1, resource: { buffer: paramBuf } },
            ],
        });

        const cmd = device.createCommandEncoder();
        encodePass(cmd, pipeline, bindGroup, Math.ceil(symbolCount / WORKGROUP_SIZE));
        device.queue.submit([cmd.finish()]);
        await device.queue.onSubmittedWorkDone();

        paramBuf.destroy();
    }

    // ─── Export ─────────────────────────────────────────────

    exportVocab() {
        return this.#vocab.export();
    }
}
