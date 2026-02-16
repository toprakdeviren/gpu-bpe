import { $, formatSize } from '../utils.js';
import { BPETrainer } from '../bpe/trainer.js';

// ─── Training Manager Class ───
export class TrainingManager {
    constructor(bpeEngine, fileManager, uiManager, logger, preTokenizer = null, onTrainingComplete = null) {
        this.bpeEngine = bpeEngine;
        this.fileManager = fileManager;
        this.uiManager = uiManager;
        this.logger = logger;
        this.lastTrainer = null;
        this.trainedModel = null;
        this.preTokenizer = preTokenizer;
        this.onTrainingComplete = onTrainingComplete;

        /** @type {Worker|null} */
        this._worker = null;
        this._useWorker = typeof Worker !== 'undefined' && !!navigator.gpu;
    }

    canTrain() {
        return this.bpeEngine && !this.fileManager.isEmpty();
    }

    async startTraining() {
        const trainBtn = $('trainBtn');
        if (trainBtn.disabled) return;

        trainBtn.disabled = true;
        this.uiManager.showTrainingProgress();

        const shouldShuffle = $('shuffleToggle').checked && this.fileManager.files.length > 1;

        // Always pass raw bytes — preTokenizeBytes handles NFC normalize + classify in WASM
        // (eliminates 5 unnecessary string<->bytes conversions; see WASM_NORMALIZE_BYTES_SPEC.md)
        const corpusData = this.fileManager.buildCorpus(shouldShuffle);

        const sizeStr = formatSize(corpusData.length);

        this.logger.log(`\n─ corpus: ${sizeStr} · vocab target: ${this.uiManager.selectedVocab.toLocaleString()}`);

        try {
            const result = this._useWorker
                ? await this._trainInWorker(corpusData)
                : await this._trainInline(corpusData);

            this.trainedModel = result;
            this.uiManager.displayTrainingComplete(result);
            this.uiManager.updateVocabStatus(result.vocabSize);

            // Reveal tokenizer encode section
            $('tokenizerSection').classList.remove('hidden');

            // Notify export controller
            if (this.onTrainingComplete) this.onTrainingComplete(result.vocabSize);
        } catch (error) {
            this.logger.log(`✗ Training failed: ${error.message}`);
            trainBtn.disabled = false;
        }
    }

    // ─── Worker-based Training ──────────────────────────────

    /**
     * Run training in a dedicated Web Worker (non-blocking).
     * Worker creates its own GPUDevice + pipelines.
     */
    async _trainInWorker(corpusData) {
        return new Promise((resolve, reject) => {
            // Pre-tokenize on main thread (needs WASM), then send raw bytes to worker
            let corpusBytes;
            if (this.preTokenizer && corpusData instanceof Uint8Array) {
                this.logger.log('  → pre-tokenizing bytes on main thread (zero-copy path)…');
                const result = this.preTokenizer.preTokenizeBytes(corpusData);
                if (result.bytes.length === 0 && corpusData.length > 0) {
                    this.logger.log('  ⚠ preTokenizeBytes returned 0 bytes — falling back to raw');
                    corpusBytes = corpusData;
                } else {
                    corpusBytes = result.bytes;
                }
            } else {
                corpusBytes = corpusData instanceof Uint8Array
                    ? corpusData
                    : new TextEncoder().encode(corpusData);
            }

            // Create worker (module type for ES imports)
            const worker = new Worker(
                new URL('../bpe/bpe-worker.js', import.meta.url),
                { type: 'module' }
            );
            this._worker = worker;

            worker.onmessage = (e) => {
                const msg = e.data;

                switch (msg.type) {
                    case 'progress':
                        this.uiManager.updateProgress(msg);
                        break;

                    case 'done':
                        this.logger.log('✓ Training complete (worker)');
                        worker.terminate();
                        this._worker = null;
                        resolve(msg.result);
                        break;

                    case 'error':
                        this.logger.log(`✗ Worker error: ${msg.message}`);
                        worker.terminate();
                        this._worker = null;
                        reject(new Error(msg.message));
                        break;

                    case 'log':
                        this.logger.log(msg.text);
                        break;
                }
            };

            worker.onerror = (err) => {
                this.logger.log(`✗ Worker crash: ${err.message}`);
                worker.terminate();
                this._worker = null;
                reject(new Error(err.message));
            };

            // Transfer corpus buffer to worker (zero-copy)
            const buffer = corpusBytes.buffer.slice(
                corpusBytes.byteOffset,
                corpusBytes.byteOffset + corpusBytes.byteLength
            );
            worker.postMessage({
                cmd: 'train',
                corpus: buffer,
                vocabSize: this.uiManager.selectedVocab,
            }, [buffer]);
        });
    }

    // ─── Inline Training (fallback) ─────────────────────────

    /**
     * Original inline training path (blocks main thread).
     * Used when Workers are not available.
     */
    async _trainInline(corpusData) {
        const trainer = new BPETrainer(this.bpeEngine);
        this.lastTrainer = trainer;

        return await trainer.train(corpusData, {
            targetVocabSize: this.uiManager.selectedVocab,
            preTokenizer: this.preTokenizer,
            onProgress: (progress) => this.uiManager.updateProgress(progress),
        });
    }

    getTrainedModel() {
        return this.trainedModel;
    }

    /**
     * Set a pre-loaded model (from JSON import)
     */
    setTrainedModel(model) {
        this.trainedModel = model;
    }

    /**
     * Download trained model as JSON
     */
    downloadModel() {
        const model = this.trainedModel;
        if (!model) return;

        const exportData = {
            version: 1,
            vocabSize: model.vocabSize,
            vocab: model.vocab,        // byte arrays
            merges: model.merges,      // [[a, b, newId], ...]
        };

        const blob = new Blob([JSON.stringify(exportData)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `bpe-vocab-${model.vocabSize}.json`;
        a.click();
        URL.revokeObjectURL(url);
        this.logger.log(`→ downloaded vocabulary (${model.vocabSize} tokens)`);
    }

    /**
     * Load model from JSON data
     */
    loadFromJSON(jsonData) {
        if (!jsonData.vocab || !jsonData.merges) {
            throw new Error('Invalid vocabulary file: missing vocab or merges');
        }

        // Reconstruct vocabStrings from byte arrays
        const vocab = jsonData.vocab;
        const vocabStrings = vocab.map(bytes => {
            if (!bytes || bytes.length === 0) return '';
            try {
                // Try UTF-8 decode for display
                return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
            } catch {
                return bytes.map(b => `<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`).join('');
            }
        });

        const model = {
            vocab,
            vocabStrings,
            vocabSize: vocab.length,
            merges: jsonData.merges,
        };

        this.trainedModel = model;
        this.logger.log(`→ loaded vocabulary: ${model.vocabSize} tokens, ${model.merges.length} merges`);
        return model;
    }
}
