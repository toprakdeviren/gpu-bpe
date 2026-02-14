import { $, formatSize } from '../utils.js';
import { BPETrainer } from '../bpe/trainer.js';

// ─── Training Manager Class ───
export class TrainingManager {
    constructor(bpeEngine, fileManager, uiManager, logger, preTokenizer = null) {
        this.bpeEngine = bpeEngine;
        this.fileManager = fileManager;
        this.uiManager = uiManager;
        this.logger = logger;
        this.lastTrainer = null;
        this.trainedModel = null;
        this.preTokenizer = preTokenizer;
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

        // When pre-tokenizer is available, pass corpus as string for Unicode-accurate processing
        const corpusData = this.preTokenizer
            ? this.fileManager.buildCorpusAsString(shouldShuffle)
            : this.fileManager.buildCorpus(shouldShuffle);

        const sizeStr = typeof corpusData === 'string'
            ? formatSize(new TextEncoder().encode(corpusData).length)
            : formatSize(corpusData.length);

        this.logger.log(`\n─ corpus: ${sizeStr} · vocab target: ${this.uiManager.selectedVocab.toLocaleString()}`);

        try {
            const result = await this.train(corpusData);
            this.trainedModel = result; // Store the result
            this.uiManager.displayTrainingComplete(result);
            this.uiManager.updateVocabStatus(result.vocabSize);

            // Reveal tokenizer encode section
            $('tokenizerSection').classList.remove('hidden');
        } catch (error) {
            this.logger.log(`✗ Training failed: ${error.message}`);
            trainBtn.disabled = false;
        }
    }

    async train(corpusData) {
        const trainer = new BPETrainer(this.bpeEngine);
        this.lastTrainer = trainer;

        return await trainer.train(corpusData, {
            targetVocabSize: this.uiManager.selectedVocab,
            preTokenizer: this.preTokenizer,
            onProgress: (progress) => this.uiManager.updateProgress(progress)
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
