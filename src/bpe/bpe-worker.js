/**
 * BPE Training Worker
 *
 * Runs the entire BPE training pipeline off the main thread:
 *   1. Creates its own GPUDevice (WebGPU is available in workers)
 *   2. Compiles pipelines from bpe.wgsl
 *   3. Runs the merge loop with zero main-thread blocking
 *   4. Posts progress + final result back via postMessage
 *
 * Messages IN:
 *   { cmd: 'train', corpus: ArrayBuffer, vocabSize: number, preTokenize: bool }
 *
 * Messages OUT:
 *   { type: 'progress', ... }
 *   { type: 'done', result: TrainingResult }
 *   { type: 'error', message: string }
 *   { type: 'log', text: string }
 */

// We import the pure-logic modules (no DOM dependencies)
import { BPEEngine } from './engine.js';
import { BPETrainer } from './trainer.js';

let engine = null;

/**
 * Initialize GPU device + compile pipelines (once per worker lifetime)
 */
async function ensureEngine() {
    if (engine) return engine;
    engine = new BPEEngine();
    await engine.init();
    postMessage({ type: 'log', text: `[worker] GPU engine ready (${Object.keys(engine.pipelines).length} kernels)` });
    return engine;
}

/**
 * Handle training request from main thread
 */
async function handleTrain({ corpus, vocabSize, preTokenized, wordStarts }) {
    const eng = await ensureEngine();
    const trainer = new BPETrainer(eng);

    // corpus arrives as ArrayBuffer — wrap as Uint8Array
    const corpusBytes = new Uint8Array(corpus);

    postMessage({ type: 'log', text: `[worker] Training: ${corpusBytes.length} bytes → ${vocabSize} vocab` });

    const result = await trainer.train(corpusBytes, {
        targetVocabSize: vocabSize,
        preTokenizer: null,  // Pre-tokenization done on main thread before transfer
        onProgress: (progress) => {
            postMessage({ type: 'progress', ...progress });
        },
    });

    // Transfer result back
    postMessage({
        type: 'done',
        result: {
            vocab: result.vocab,
            vocabStrings: result.vocabStrings,
            vocabSize: result.vocabSize,
            merges: result.merges,
            trainingTime: result.trainingTime,
        },
    });
}

// ─── Message Handler ────────────────────────────────────────

self.onmessage = async (e) => {
    const { cmd } = e.data;

    try {
        switch (cmd) {
            case 'train':
                await handleTrain(e.data);
                break;

            case 'ping':
                postMessage({ type: 'pong' });
                break;

            default:
                postMessage({ type: 'error', message: `Unknown command: ${cmd}` });
        }
    } catch (err) {
        postMessage({ type: 'error', message: err.message, stack: err.stack });
    }
};

postMessage({ type: 'log', text: '[worker] BPE worker loaded' });
