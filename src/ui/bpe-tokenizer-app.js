import { $ } from '../utils.js';
import { Logger } from './logger.js';
import { StatusManager } from './status-manager.js';
import { FileManager } from './file-manager.js';
import { UIManager } from './ui-manager.js';
import { TrainingManager } from './training-manager.js';
import { TokenizerManager } from '../bpe/tokenizer-manager.js';
import { FileInputController } from './file-input-controller.js';
import { VocabLoaderController } from './vocab-loader-controller.js';
import { EncoderController } from './encoder-controller.js';
import { ExportController } from './export-controller.js';

// ─── Application Class ───
export class BPETokenizerApp {

    constructor() {
        this.logger = new Logger($('log'));
        this.statusManager = new StatusManager($('statusDot'), $('statusText'));
        this.fileManager = new FileManager(this.logger);
        this.uiManager = new UIManager(this.fileManager, this.logger);
        this.trainingManager = null;
        this.tokenizerManager = null;

        this.fileInput = new FileInputController(
            this.fileManager, this.uiManager, () => this._updateTrainButton()
        );
        this.vocabLoader = new VocabLoaderController(
            this.logger, this.statusManager, this.uiManager, () => this.trainingManager
        );
        this.encoder = new EncoderController(
            this.logger, () => this.tokenizerManager
        );
        this.bpeEngine = null;
        this.exportController = new ExportController(
            this.logger, () => this.trainingManager, () => this.bpeEngine
        );
    }

    initialize() {
        this._bindTabs();
        this.fileInput.bind();
        this.vocabLoader.bind();
        this.encoder.bind();
        this.exportController.bind();

        $('trainBtn').addEventListener('click', async () => {
            if (this.trainingManager) await this.trainingManager.startTraining();
        });
        $('downloadBtn').addEventListener('click', () => {
            this.trainingManager?.downloadModel();
        });

        this.uiManager.initializeVocabChips();
    }

    setBPEEngine(engine, preTokenizer = null) {
        this.bpeEngine = engine;
        this.trainingManager = new TrainingManager(
            engine, this.fileManager, this.uiManager, this.logger, preTokenizer,
            (vocabSize) => this.exportController.notifyTrainingComplete(vocabSize)
        );
        this.tokenizerManager = new TokenizerManager(
            engine, this.trainingManager, this.logger
        );
        this._updateTrainButton();
    }

    _updateTrainButton() {
        $('trainBtn').disabled = !(this.trainingManager?.canTrain());
    }

    _bindTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b =>
                    b.classList.toggle('active', b.dataset.tab === btn.dataset.tab));
                document.querySelectorAll('.tab-content').forEach(p =>
                    p.classList.toggle('active', p.dataset.tab === btn.dataset.tab));
            });
        });
    }
}
