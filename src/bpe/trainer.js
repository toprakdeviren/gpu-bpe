/**
 * BPE Trainer — GPU-accelerated BPE vocabulary training
 *
 * Manages the BPE training pipeline: input preparation, training loop,
 * and vocab management. GPU orchestration delegated to training-pipeline.js.
 *
 * Text normalization (NFC, control chars, whitespace) is handled by the Decoder
 * WASM module before data is passed to the trainer.
 */

import { WORKGROUP_SIZE, TABLE_SIZE } from './engine.js';
import { uploadBuffer } from './gpu-utils.js';
import { Vocab } from './vocab.js';
import {
    BATCH_SIZE, MERGE_LOG_STRIDE, ITER_STATE_SIZE,
    allocTrainingBuffers, destroyTrainingBuffers,
    buildBindGroups, encodeBatch, encodePass,
} from './training-pipeline.js';

// ─── Constants ──────────────────────────────────────────────

const WORD_START_BIT = 0x10000;       // bit 16 — must match WGSL constant

const BUFFER_USAGE = {
    STORAGE_SRC: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
};

// ─── Utilities ──────────────────────────────────────────────

/** @param {number} seconds */
function formatDuration(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

/** Yield to event loop so browser can repaint */
function yieldToEventLoop() {
    return new Promise(r => setTimeout(r, 0));
}

// ─── Input Preparation ──────────────────────────────────────

/**
 * @param {Uint8Array} data
 * @returns {Uint32Array}
 */
function bytesToSymbols(data) {
    const symbols = new Uint32Array(data.length);
    for (let i = 0; i < data.length; i++) symbols[i] = data[i];
    return symbols;
}

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
                maxDispatch, tb.maxBlocks, tb.findMaxBlocks, tb.indirectDispatch, tb.useParallelScan);

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
