/**
 * Training Pipeline — GPU buffer allocation, bind groups, and batch encoding
 *
 * All the GPU orchestration primitives used by BPETrainer during the
 * training loop. Separated from trainer.js for clarity.
 */

import { WORKGROUP_SIZE, TABLE_SIZE, dispatch2D } from './engine.js';
import { uploadBuffer, allocBuffer, destroyBuffers } from './gpu-utils.js';

// ─── Constants ──────────────────────────────────────────────

export const BATCH_SIZE = 128;
export const MERGE_LOG_STRIDE = 3;           // [pair, newTokenId, count] per merge
export const ITER_STATE_SIZE = 48;           // 12 × u32

const BUFFER_USAGE = {
    STORAGE_SRC: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    STORAGE_ALL: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    READBACK: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
};

// ─── Training Buffers ───────────────────────────────────────

/**
 * Pre-allocate all persistent GPU buffers needed during training.
 *
 * @param {GPUDevice} device
 * @param {number} maxSymbols
 * @returns {TrainingBuffers}
 */
export function allocTrainingBuffers(device, maxSymbols) {
    const maxBlocks = Math.ceil(maxSymbols / WORKGROUP_SIZE);
    // find_max_pair4: 4 elements/thread × 256 threads = 1024 entries/workgroup
    const findMaxBlocks = Math.ceil(TABLE_SIZE / (256 * 4));
    // Parallel scan handles up to 256 blocks (65,536 symbols); above that → sequential fallback
    const useParallelScan = maxBlocks <= 256;

    return {
        // Hash table
        pairCounts: allocBuffer(device, TABLE_SIZE * 4, BUFFER_USAGE.STORAGE_SRC),
        pairIds: allocBuffer(device, TABLE_SIZE * 4, BUFFER_USAGE.STORAGE_SRC),

        // FindMax reduction (4× fewer blocks thanks to find_max_pair4)
        blockMaxCounts: allocBuffer(device, findMaxBlocks * 4, GPUBufferUsage.STORAGE),
        blockMaxPairIds: allocBuffer(device, findMaxBlocks * 4, GPUBufferUsage.STORAGE),
        maxCount: allocBuffer(device, 4, BUFFER_USAGE.STORAGE_SRC),
        maxPairId: allocBuffer(device, 4, BUFFER_USAGE.STORAGE_SRC),

        // Compaction (merge_reduce writes valid_mask + block_sums in one pass)
        validMask: allocBuffer(device, maxSymbols * 4, GPUBufferUsage.STORAGE),
        blockSums: allocBuffer(device, maxBlocks * 4, GPUBufferUsage.STORAGE),
        compact: allocBuffer(device, maxSymbols * 4, BUFFER_USAGE.STORAGE_SRC),

        // Batched iteration
        iterState: allocBuffer(device, ITER_STATE_SIZE, BUFFER_USAGE.STORAGE_ALL),
        mergeLog: allocBuffer(device, BATCH_SIZE * MERGE_LOG_STRIDE * 4, BUFFER_USAGE.STORAGE_SRC),

        // Indirect dispatch buffer (3 × u32: wgX, wgY, wgZ)
        // Written by scan_blocks kernel, read by dispatchWorkgroupsIndirect
        indirectDispatch: allocBuffer(device, 12,
            GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST),

        // Readback staging
        readbackState: allocBuffer(device, ITER_STATE_SIZE, BUFFER_USAGE.READBACK),
        readbackLog: allocBuffer(device, BATCH_SIZE * MERGE_LOG_STRIDE * 4, BUFFER_USAGE.READBACK),

        // Uniform param buffers
        clearTableParam: uploadBuffer(device, new Uint32Array([TABLE_SIZE, 0]), GPUBufferUsage.UNIFORM),
        findMaxParam1: uploadBuffer(device, new Uint32Array([TABLE_SIZE, 0]), GPUBufferUsage.UNIFORM),
        findMaxParam2: uploadBuffer(device, new Uint32Array([findMaxBlocks, 0]), GPUBufferUsage.UNIFORM),

        // Derived constants
        maxBlocks,
        findMaxBlocks,
        maxSymbols,
        useParallelScan,
    };
}

/** @param {TrainingBuffers} b */
export function destroyTrainingBuffers(b) {
    destroyBuffers([
        b.pairCounts, b.pairIds,
        b.blockMaxCounts, b.blockMaxPairIds,
        b.maxCount, b.maxPairId,
        b.validMask, b.blockSums, b.compact,
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
export function buildBindGroups(device, pipelines, symbolBufA, symbolBufB, tb) {
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

        // Two-pass findMax (4 elem/thread, deterministic tie-breaking)
        findMax1: bg(p.bpe_find_max_pair4, [
            tb.pairCounts, tb.pairIds, tb.blockMaxCounts, tb.blockMaxPairIds, tb.findMaxParam1,
        ]),
        findMax2: bg(p.bpe_find_max_pair_final_det, [
            tb.blockMaxCounts, tb.blockMaxPairIds, tb.maxCount, tb.maxPairId, tb.findMaxParam2,
        ]),

        // Merge setup
        setupMerge: bg(p.bpe_setup_merge, [
            tb.maxCount, tb.maxPairId, tb.iterState, tb.mergeLog,
        ]),

        // FUSED merge + reduce (ping-pong)
        mergeReduceA: bg(p.bpe_merge_reduce_b, [
            symbolBufA, tb.validMask, tb.blockSums, tb.iterState,
        ]),
        mergeReduceB: bg(p.bpe_merge_reduce_b, [
            symbolBufB, tb.validMask, tb.blockSums, tb.iterState,
        ]),

        // Parallel scan + update_count (primary path, ≤256 blocks)
        scanBlocksPar: bg(p.bpe_prefix_sum_scan_blocks_par, [
            tb.blockSums, tb.iterState, tb.indirectDispatch,
        ]),

        // Sequential scan fallback (>256 blocks)
        scanBlocksSeq: bg(p.bpe_prefix_sum_scan_blocks_b, [
            tb.blockSums, tb.iterState, tb.indirectDispatch,
        ]),

        // Fused finalize + compact
        finalizeCompactAB: bg(p.bpe_finalize_compact_b, [
            tb.validMask, tb.blockSums, symbolBufA, symbolBufB, tb.iterState,
        ]),
        finalizeCompactBA: bg(p.bpe_finalize_compact_b, [
            tb.validMask, tb.blockSums, symbolBufB, symbolBufA, tb.iterState,
        ]),
    };
}

// ─── Batch Encoding ─────────────────────────────────────────

/**
 * Encode one batch of BPE merge iterations into a command encoder.
 *
 * Optimized pipeline (8 dispatches/iteration, down from 10):
 *   clear → count → findMax4 → findMaxFinal → setupMerge
 *   → mergeReduce (FUSED) → scanBlocksPar → finalizeCompact
 */
export function encodeBatch(cmd, pipelines, bg, batchMerges, maxDispatch, maxBlocks, findMaxBlocks, indirectBuf, useParallelScan) {
    const p = pipelines;

    // Choose scan kernel + bind group based on block count
    const scanPipeline = useParallelScan
        ? p.bpe_prefix_sum_scan_blocks_par
        : p.bpe_prefix_sum_scan_blocks_b;
    const scanBG = useParallelScan ? bg.scanBlocksPar : bg.scanBlocksSeq;

    for (let i = 0; i < batchMerges; i++) {
        const even = (i % 2 === 0);

        // Phase A: clear → count → findMax4 → findMaxFinal → setupMerge
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

        encodePass(cmd, p.bpe_find_max_pair4, bg.findMax1, findMaxBlocks);
        encodePass(cmd, p.bpe_find_max_pair_final_det, bg.findMax2, 1);
        encodePass(cmd, p.bpe_setup_merge, bg.setupMerge, 1);

        // Phase B: mergeReduce (FUSED) → scanBlocks → finalizeCompact
        if (i === 0) {
            encodePass(cmd, p.bpe_merge_reduce_b,
                even ? bg.mergeReduceA : bg.mergeReduceB, maxDispatch);
            encodePass(cmd, scanPipeline, scanBG, useParallelScan ? 1 : 1);
            encodePass(cmd, p.bpe_finalize_compact_b,
                even ? bg.finalizeCompactAB : bg.finalizeCompactBA, maxDispatch);
        } else {
            encodeIndirectPass(cmd, p.bpe_merge_reduce_b,
                even ? bg.mergeReduceA : bg.mergeReduceB, indirectBuf);
            encodePass(cmd, scanPipeline, scanBG, 1);
            encodeIndirectPass(cmd, p.bpe_finalize_compact_b,
                even ? bg.finalizeCompactAB : bg.finalizeCompactBA, indirectBuf);
        }
    }
}

// ─── Pass Helpers ───────────────────────────────────────────

/** Encode a single compute pass with indirect dispatch. */
function encodeIndirectPass(cmd, pipeline, bindGroup, indirectBuffer) {
    const pass = cmd.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroupsIndirect(indirectBuffer, 0);
    pass.end();
}

/** Encode a single compute pass. */
export function encodePass(cmd, pipeline, bindGroup, workgroups) {
    const pass = cmd.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    dispatch2D(pass, workgroups);
    pass.end();
}
