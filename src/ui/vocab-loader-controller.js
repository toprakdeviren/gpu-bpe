import { $, ICONS } from '../utils.js';

// ─── Vocab Loader Controller ───
export class VocabLoaderController {
    /**
     * @param {import('./logger.js').Logger} logger
     * @param {import('./status-manager.js').StatusManager} statusManager
     * @param {import('./ui-manager.js').UIManager} uiManager
     * @param {() => import('./training-manager.js').TrainingManager | null} getTrainingManager
     */
    constructor(logger, statusManager, uiManager, getTrainingManager) {
        this.logger = logger;
        this.statusManager = statusManager;
        this.uiManager = uiManager;
        this.getTrainingManager = getTrainingManager;
    }

    bind() {
        const fileInput = $('vocabFileInput');

        $('loadVocabBtn').addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            await this._loadVocabFile(file);
            fileInput.value = '';
        });
    }

    async _loadVocabFile(file) {
        const trainingManager = this.getTrainingManager();

        try {
            const text = await file.text();
            const jsonData = JSON.parse(text);

            if (!trainingManager) {
                this.logger.log('✗ Engine not initialized yet');
                return;
            }

            const model = trainingManager.loadFromJSON(jsonData);
            this._showLoadedVocab(model.vocabSize);
        } catch (error) {
            this.logger.log(`✗ Failed to load vocabulary: ${error.message}`);
        }
    }

    _showLoadedVocab(vocabSize) {
        $('tokenizerSection').classList.remove('hidden');
        this.uiManager.updateVocabStatus(vocabSize);

        const downloadBtn = $('downloadBtn');
        downloadBtn.innerHTML = `${ICONS.download}<span>Download Vocabulary (${vocabSize.toLocaleString()} tokens)</span>`;
        downloadBtn.classList.remove('hidden');

        this.statusManager.setStatus('ok', `Loaded ${vocabSize} tokens`);
    }
}
