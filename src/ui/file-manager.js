import { NormalizationForm } from '../wasm/decoder.mjs';
import { PARAGRAPH_SEPARATOR, formatSize, shuffleArray } from '../utils.js';

// ─── File Manager Class ───
export class FileManager {
    constructor(logger, decoder = null) {
        this.files = [];
        this.logger = logger;
        this.decoder = decoder;
    }

    setDecoder(decoder) {
        this.decoder = decoder;
    }

    async addFiles(fileArray, onProgress = null) {
        const files = Array.from(fileArray);
        for (let i = 0; i < files.length; i++) {
            if (onProgress) onProgress(i + 1, files.length, files[i].name);
            await this.readFileAsArrayBuffer(files[i]);
        }

        const totalSize = this.calculateTotalSize();
        this.logger.log(`→ Added ${files.length} file(s) — ${formatSize(totalSize)} total`);
    }

    readFileAsArrayBuffer(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                let data = new Uint8Array(e.target.result);

                // Normalize text content using Decoder WASM (NFC + cleanup)
                if (this.decoder) {
                    try {
                        const text = new TextDecoder('utf-8', { fatal: false }).decode(data);
                        const normalized = this.decoder.normalize(text, NormalizationForm.NFC);
                        // Guard: normalize can silently return empty for large inputs (WASM OOM)
                        if (normalized && normalized.length > 0) {
                            data = new TextEncoder().encode(normalized);
                        } else if (text.length > 0) {
                            console.warn(`WASM normalize returned empty for ${file.name} (${data.length} bytes), keeping raw`);
                        }
                    } catch (err) {
                        // If normalization fails, use raw bytes
                        console.warn(`WASM normalize skipped for ${file.name}:`, err.message);
                    }
                }

                this.files.push({
                    name: file.name,
                    size: data.length,
                    data
                });
                resolve();
            };
            reader.onerror = () => {
                console.warn(`Failed to read ${file.name}:`, reader.error);
                resolve(); // resolve (not reject) so other files still get processed
            };
            reader.readAsArrayBuffer(file);
        });
    }

    removeFile(index) {
        this.files.splice(index, 1);
    }

    calculateTotalSize() {
        return this.files.reduce((total, file) => total + file.size, 0);
    }

    isEmpty() {
        return this.files.length === 0;
    }

    buildCorpus(shouldShuffle) {
        if (this.isEmpty()) return new Uint8Array(0);

        const encoder = new TextEncoder();
        const separator = encoder.encode(PARAGRAPH_SEPARATOR);

        if (!shouldShuffle) {
            return this.concatenateFiles(separator);
        }

        return this.buildShuffledCorpus(encoder);
    }

    concatenateFiles(separator) {
        const totalLen = this.files.reduce((sum, f) => sum + f.data.length, 0) +
            (this.files.length - 1) * separator.length;
        const result = new Uint8Array(totalLen);
        let offset = 0;

        this.files.forEach((file, index) => {
            if (index > 0) {
                result.set(separator, offset);
                offset += separator.length;
            }
            result.set(file.data, offset);
            offset += file.data.length;
        });

        return result;
    }

    buildShuffledCorpus(encoder) {
        const paragraphs = this.extractParagraphs();
        shuffleArray(paragraphs);
        this.logger.log(`→ shuffled ${paragraphs.length} paragraphs from ${this.files.length} files`);
        return encoder.encode(paragraphs.join(PARAGRAPH_SEPARATOR));
    }

    extractParagraphs() {
        const decoder = new TextDecoder();
        const paragraphs = [];

        for (const file of this.files) {
            const text = decoder.decode(file.data);
            const fileParagraphs = text.split(/\n\n+/).filter(p => p.trim().length > 0);
            paragraphs.push(...fileParagraphs);
        }

        return paragraphs;
    }

    /**
     * Build corpus as a JavaScript string (for PreTokenizer which needs codepoint access).
     * @param {boolean} shouldShuffle
     * @returns {string}
     */
    buildCorpusAsString(shouldShuffle) {
        if (this.isEmpty()) {
            console.warn('buildCorpusAsString: no files loaded');
            return '';
        }

        // Diagnostic: check for files with empty data
        const emptyFiles = this.files.filter(f => !f.data || f.data.length === 0);
        if (emptyFiles.length > 0) {
            console.warn(`buildCorpusAsString: ${emptyFiles.length}/${this.files.length} files have empty data:`,
                emptyFiles.map(f => f.name));
        }

        if (shouldShuffle) {
            const paragraphs = this.extractParagraphs();
            shuffleArray(paragraphs);
            this.logger.log(`→ shuffled ${paragraphs.length} paragraphs from ${this.files.length} files`);
            return paragraphs.join(PARAGRAPH_SEPARATOR);
        }

        const decoder = new TextDecoder();
        return this.files.map(f => decoder.decode(f.data)).join(PARAGRAPH_SEPARATOR);
    }
}
