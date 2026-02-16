import { $, formatSize } from '../utils.js';
import { TrieTokenizer } from '../bpe/tokenizer.js';

const MAGIC = 0x44584654; // 'DXFT' — Decoder eXport Format Tokens

/**
 * Export Controller — tokenize text files → .bin for Transformer training
 *
 * Uses GPU trie tokenizer for maximum speed. Pipeline:
 *   vocab (byte arrays) → compile trie → GPU trie walk → .bin export
 */
export class ExportController {
    /**
     * @param {import('./logger.js').Logger} logger
     * @param {() => import('./training-manager.js').TrainingManager | null} getTrainingManager
     * @param {() => import('../bpe/engine.js').BPEEngine | null} getEngine
     */
    constructor(logger, getTrainingManager, getEngine) {
        this.logger = logger;
        this.getTrainingManager = getTrainingManager;
        this.getEngine = getEngine;
        this._files = [];    // { name, size, data: Uint8Array }
        this._vocab = null;  // loaded from JSON (independent of Train tab)
        this._trieTokenizer = null; // GPU trie tokenizer instance
    }

    bind() {
        // Vocab loading
        $('exportLoadVocabBtn').addEventListener('click', () => $('exportVocabInput').click());
        $('exportVocabInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) await this._loadVocab(file);
            e.target.value = '';
        });

        // File input
        const dropZone = $('exportDropZone');
        const fileInput = $('exportFileInput');

        fileInput.addEventListener('change', async (e) => {
            await this._addFiles(e.target.files);
            fileInput.value = '';
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drop-active');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drop-active'));
        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            dropZone.classList.remove('drop-active');
            if (e.dataTransfer.files.length > 0) await this._addFiles(e.dataTransfer.files);
        });

        // Export
        $('exportBinBtn').addEventListener('click', () => this._export());
    }

    /**
     * Called externally when training completes — auto-update vocab status
     */
    notifyTrainingComplete(vocabSize) {
        if (!this._vocab) {
            this._updateVocabUI(vocabSize, 'from training');
            this._buildTrieFromTraining();
            this._updateExportBtn();
        }
    }

    // ── Trie Building ──

    _buildTrieFromTraining() {
        const model = this.getTrainingManager()?.getTrainedModel();
        const engine = this.getEngine();
        if (!model || !engine) return;

        try {
            const t0 = performance.now();
            this._trieTokenizer = TrieTokenizer.fromVocab(engine, model.vocab);
            const dt = ((performance.now() - t0)).toFixed(1);
            this.logger.log(`→ [export] trie compiled: ${this._trieTokenizer.nodeCount} nodes, ${this._trieTokenizer.edgeCount} edges (${dt}ms)`);
        } catch (e) {
            this.logger.log(`✗ [export] trie compile failed: ${e.message}`);
            this._trieTokenizer = null;
        }
    }

    _buildTrieFromVocab(vocab) {
        const engine = this.getEngine();
        if (!engine) {
            this.logger.log('✗ [export] GPU engine not ready');
            return;
        }

        try {
            const t0 = performance.now();
            this._trieTokenizer = TrieTokenizer.fromVocab(engine, vocab);
            const dt = ((performance.now() - t0)).toFixed(1);
            this.logger.log(`→ [export] trie compiled: ${this._trieTokenizer.nodeCount} nodes, ${this._trieTokenizer.edgeCount} edges (${dt}ms)`);
        } catch (e) {
            this.logger.log(`✗ [export] trie compile failed: ${e.message}`);
            this._trieTokenizer = null;
        }
    }

    // ── Vocab ──

    async _loadVocab(file) {
        try {
            const text = await file.text();
            const json = JSON.parse(text);

            if (!json.vocab || !json.merges) {
                throw new Error('Invalid vocabulary: missing vocab or merges');
            }

            // Also load into training manager for consistency
            const trainingManager = this.getTrainingManager();
            if (trainingManager) {
                trainingManager.loadFromJSON(json);
            }

            this._vocab = json;
            this._updateVocabUI(json.vocab.length, file.name);
            this.logger.log(`→ [export] loaded vocab: ${file.name} (${json.vocab.length} tokens)`);

            // Compile trie for GPU tokenization
            this._buildTrieFromVocab(json.vocab);

        } catch (e) {
            this.logger.log(`✗ [export] vocab error: ${e.message}`);
        }
        this._updateExportBtn();
    }

    _updateVocabUI(vocabSize, source) {
        const icon = $('exportVocabIcon');
        icon.innerHTML = `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>`;
        $('exportVocabText').textContent = `${vocabSize.toLocaleString()} tokens · ${source}`;
        $('exportVocabStatus').classList.add('loaded');
    }

    // ── Files ──

    async _addFiles(fileList) {
        for (const file of Array.from(fileList)) {
            const ab = await file.arrayBuffer();
            this._files.push({ name: file.name, size: ab.byteLength, data: new Uint8Array(ab) });
        }
        this._renderFileList();
        this._updateExportBtn();
    }

    _renderFileList() {
        const list = $('exportFileList');
        list.innerHTML = this._files.map(f =>
            `<div class="file-item"><span class="file-name">${f.name}</span><span class="file-size">${formatSize(f.size)}</span></div>`
        ).join('');

        const totalBytes = this._files.reduce((s, f) => s + f.size, 0);
        $('exportFileTotal').textContent = `${this._files.length} file(s) · ${formatSize(totalBytes)}`;
        $('exportFileSummary').classList.remove('hidden');
    }

    // ── Export ──

    _hasVocab() {
        return !!this._trieTokenizer;
    }

    _updateExportBtn() {
        $('exportBinBtn').disabled = !(this._trieTokenizer && this._files.length > 0);
    }

    async _export() {
        if (!this._trieTokenizer || this._files.length === 0) return;

        const btn = $('exportBinBtn');
        const btnText = $('exportBinBtnText');
        btn.disabled = true;
        btnText.textContent = 'Tokenizing…';
        $('exportProgress').classList.remove('hidden');
        $('exportProgressFill').style.width = '0%';

        try {
            // Merge all files into one byte array
            const sep = new TextEncoder().encode('\n\n');
            let totalLen = 0;
            for (const f of this._files) totalLen += f.data.length;
            totalLen += (this._files.length - 1) * sep.length;

            const merged = new Uint8Array(totalLen);
            let offset = 0;
            for (let i = 0; i < this._files.length; i++) {
                if (i > 0) { merged.set(sep, offset); offset += sep.length; }
                merged.set(this._files[i].data, offset); offset += this._files[i].data.length;
            }

            this.logger.log(`\n→ [export] GPU tokenizing ${formatSize(totalLen)}…`);
            $('exportProgressLabel').textContent = `GPU tokenizing ${formatSize(totalLen)}…`;
            $('exportProgressFill').style.width = '10%';

            await new Promise(r => setTimeout(r, 50)); // let UI update

            // GPU trie tokenization
            const t0 = performance.now();
            const tokens = await this._trieTokenizer.encodeBytes(merged);
            const dt = ((performance.now() - t0) / 1000).toFixed(2);
            const throughputMBs = (totalLen / 1048576 / parseFloat(dt)).toFixed(1);

            $('exportProgressFill').style.width = '90%';

            // Get vocab data
            const model = this.getTrainingManager()?.getTrainedModel();
            const vocabSize = this._vocab?.vocab?.length
                ?? model?.vocabSize
                ?? 256;

            // Serialize vocab as JSON bytes (for embedded decode in Transformer)
            const vocabExport = this._vocab ?? (model ? {
                version: 1,
                vocabSize: model.vocabSize,
                vocab: model.vocab,
                merges: model.merges,
            } : null);

            const vocabBytes = vocabExport
                ? new TextEncoder().encode(JSON.stringify(vocabExport))
                : new Uint8Array(0);

            // Build .bin v2: [MAGIC, vocabSize, tokenCount, vocabBytesLen, ...tokens, ...vocabJSON]
            const headerLen = 4; // 4 u32s
            const out = new Uint32Array(headerLen + tokens.length);
            out[0] = MAGIC;
            out[1] = vocabSize;
            out[2] = tokens.length;
            out[3] = vocabBytes.length;
            out.set(tokens, headerLen);

            // Combine: u32 array + vocab JSON bytes
            const tokenPart = new Uint8Array(out.buffer);
            const finalBuf = new Uint8Array(tokenPart.length + vocabBytes.length);
            finalBuf.set(tokenPart);
            finalBuf.set(vocabBytes, tokenPart.length);

            $('exportProgressFill').style.width = '100%';
            $('exportProgressLabel').textContent =
                `${tokens.length.toLocaleString()} tokens · ${dt}s · ${throughputMBs} MB/s · ${(totalLen / tokens.length).toFixed(2)}× compression`;

            // Download
            const blob = new Blob([finalBuf], { type: 'application/octet-stream' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `train-v${vocabSize}-${tokens.length}.bin`;
            a.click();
            URL.revokeObjectURL(url);

            const sizeMB = (finalBuf.byteLength / 1048576).toFixed(1);
            this.logger.log(`→ [export] ${tokens.length.toLocaleString()} tokens → ${a.download} (${sizeMB} MB)`);
            this.logger.log(`→ [export] GPU: ${dt}s · ${throughputMBs} MB/s · ${(totalLen / tokens.length).toFixed(2)}× compression`);
            this.logger.log(`→ [export] embedded vocab: ${formatSize(vocabBytes.length)}`);

            btnText.textContent = '✓ Exported — export again?';

        } catch (e) {
            this.logger.log(`✗ [export] failed: ${e.message}`);
            $('exportProgressLabel').textContent = `Error: ${e.message}`;
        } finally {
            btn.disabled = false;
        }
    }
}
