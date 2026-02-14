import { $, formatSize, DEFAULT_VOCAB_SIZE, CHIP_CLASSES } from '../utils.js';

// ─── UI Manager Class ───
export class UIManager {
    constructor(fileManager, logger) {
        this.fileManager = fileManager;
        this.logger = logger;
        this.selectedVocab = DEFAULT_VOCAB_SIZE;
    }

    renderFileList() {
        const fileListEl = $('fileList');
        fileListEl.innerHTML = this.fileManager.files.map((file, index) =>
            this.renderFileItem(file, index)
        ).join('');
        this.updateFileSummary();
    }

    renderFileItem(file, index) {
        return `
            <div class="file-item">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/></svg>
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatSize(file.size)}</span>
                <button data-idx="${index}" title="Remove" class="fi-remove file-remove">&times;</button>
            </div>
        `;
    }

    updateFileSummary() {
        if (this.fileManager.isEmpty()) {
            $('fileSummary').classList.add('hidden');
            return;
        }

        const totalSize = this.fileManager.calculateTotalSize();
        const fileLabel = this.fileManager.files.length === 1 ? 'file' : 'files';
        $('fileTotalInfo').textContent = `${this.fileManager.files.length} ${fileLabel} · ${formatSize(totalSize)}`;
        $('fileSummary').classList.remove('hidden');
    }

    initializeVocabChips() {
        const vocabChips = document.querySelectorAll('.vocab-chip');

        vocabChips.forEach(chip => {
            const isDefault = chip.dataset.value === String(DEFAULT_VOCAB_SIZE);
            const chipClass = isDefault ? CHIP_CLASSES.active : CHIP_CLASSES.inactive;
            chip.className = `vocab-chip ${CHIP_CLASSES.base} ${chipClass}`;

            chip.addEventListener('click', () => {
                this.selectVocabChip(chip, vocabChips);
            });
        });
    }

    selectVocabChip(selectedChip, allChips) {
        allChips.forEach(chip => {
            chip.className = `vocab-chip ${CHIP_CLASSES.base} ${CHIP_CLASSES.inactive}`;
        });
        selectedChip.className = `vocab-chip ${CHIP_CLASSES.base} ${CHIP_CLASSES.active}`;
        this.selectedVocab = parseInt(selectedChip.dataset.value);
    }

    updateProgress(progress) {
        const percentage = (progress.mergeIndex / progress.totalMerges * 100).toFixed(1);
        $('progressFill').style.width = `${percentage}%`;
        $('progressLabel').textContent = `Merge ${progress.mergeIndex.toLocaleString()} / ${progress.totalMerges.toLocaleString()}`;
        $('progressRate').textContent = `${progress.mergesPerSecond.toFixed(0)} merges/s · ${progress.symbolCount.toLocaleString()} syms`;

        // ETA — only show after 256 merges (let rate stabilize)
        const etaEl = $('progressEta');
        if (progress.mergeIndex >= 256 && progress.mergesPerSecond > 0) {
            const remaining = (progress.totalMerges - progress.mergeIndex) / progress.mergesPerSecond;
            etaEl.textContent = `~${this._formatDuration(remaining)} remaining`;
        } else {
            etaEl.textContent = 'estimating…';
        }
    }

    _formatDuration(seconds) {
        if (seconds < 60) return `${Math.round(seconds)}s`;
        const min = Math.floor(seconds / 60);
        const sec = Math.round(seconds % 60);
        if (min < 60) {
            return sec > 0 ? `${min}m ${sec}s` : `${min}m`;
        }
        const hr = Math.floor(min / 60);
        const rm = min % 60;
        return rm > 0 ? `${hr}h ${rm}m` : `${hr}h`;
    }

    displayTrainingComplete(result) {
        $('progressFill').style.width = '100%';
        $('progressLabel').textContent = `✓ done — ${result.vocabSize.toLocaleString()} tokens`;
        $('progressLabel').classList.add('text-amber-accent');
        $('progressEta').textContent = `${result.trainingTime}`;
        $('downloadBtn').innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"/></svg><span>Download Vocabulary (${result.vocabSize.toLocaleString()} tokens)</span>`;
        $('downloadBtn').classList.remove('hidden');
        this.logger.log(`→ training complete (${result.vocabSize.toLocaleString()} tokens)`);
    }

    showTrainingProgress() {
        $('downloadBtn').classList.add('hidden');
        $('progressWrap').classList.remove('hidden');
        $('progressFill').style.width = '0%';
        $('progressEta').textContent = '';
        $('progressLabel').classList.remove('text-amber-accent');
    }

    updateVocabStatus(vocabSize) {
        const statusEl = $('vocabStatus');
        const textEl = $('vocabStatusText');
        const iconEl = $('vocabStatusIcon');
        statusEl.classList.add('loaded');
        iconEl.innerHTML = `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>`;
        textEl.textContent = `${vocabSize.toLocaleString()} tokens loaded`;
    }
}
