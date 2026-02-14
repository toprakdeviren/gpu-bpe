import { $, TEXT_EXTENSIONS } from '../utils.js';

// ─── File Input Controller ───
export class FileInputController {
    /**
     * @param {import('./file-manager.js').FileManager} fileManager
     * @param {import('./ui-manager.js').UIManager} uiManager
     * @param {() => void} onFilesChanged
     */
    constructor(fileManager, uiManager, onFilesChanged) {
        this.fileManager = fileManager;
        this.uiManager = uiManager;
        this.onFilesChanged = onFilesChanged;
    }

    bind() {
        this._bindRemoveButtons();
        this._bindFileInput();
        this._bindFolderInput();
        this._bindDropZone();
    }

    // ─── Processing ─────────────────────────────────────────

    async _processFiles(files) {
        if (!files.length) return;

        const processingEl = $('fileProcessing');
        const processingText = $('fileProcessingText');

        processingEl.classList.remove('hidden');
        processingText.textContent = `Normalizing ${files.length} file${files.length > 1 ? 's' : ''}…`;

        await this.fileManager.addFiles(files, (done, total, name) => {
            processingText.textContent = total > 1
                ? `Processing ${done}/${total} — ${name}`
                : `Processing ${name}…`;
        });

        processingEl.classList.add('hidden');
        this.uiManager.renderFileList();
        this.onFilesChanged();
    }

    // ─── File Removal ───────────────────────────────────────

    _bindRemoveButtons() {
        $('fileList').addEventListener('click', (e) => {
            const removeBtn = e.target.closest('.fi-remove');
            if (!removeBtn) return;

            this.fileManager.removeFile(parseInt(removeBtn.dataset.idx, 10));
            this.uiManager.renderFileList();
            this.onFilesChanged();
        });
    }

    // ─── File Browse ────────────────────────────────────────

    _bindFileInput() {
        const input = $('fileInput');
        input.addEventListener('change', async (e) => {
            const files = e.target.files;
            if (!files.length) return;
            await this._processFiles(files);
            input.value = '';
        });
    }

    // ─── Folder Browse ──────────────────────────────────────

    _bindFolderInput() {
        const folderInput = $('folderInput');
        const browseFolder = $('browseFolder');

        if (!folderInput || !browseFolder) return;

        // "browse folder" link opens the native folder picker
        browseFolder.addEventListener('click', (e) => {
            e.stopPropagation(); // don't trigger dropZone's file input
            folderInput.click();
        });

        folderInput.addEventListener('change', async (e) => {
            const allFiles = Array.from(e.target.files);
            const textFiles = allFiles.filter(f => isTextFile(f.name));

            if (textFiles.length === 0) {
                this.fileManager.logger.log('→ No text files found in selected folder');
                folderInput.value = '';
                return;
            }

            this.fileManager.logger.log(
                `→ Found ${textFiles.length} text file${textFiles.length > 1 ? 's' : ''} ` +
                `(${allFiles.length} total in folder)`
            );

            await this._processFiles(textFiles);
            folderInput.value = '';
        });
    }

    // ─── Drop Zone ──────────────────────────────────────────

    _bindDropZone() {
        const dropZone = $('dropZone');

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drop-active');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drop-active');
        });

        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            dropZone.classList.remove('drop-active');

            // Try directory traversal via webkitGetAsEntry (supports folder drops)
            const items = e.dataTransfer.items;
            if (items && items.length > 0 && items[0].webkitGetAsEntry) {
                const files = await collectFilesFromEntries(items);
                if (files.length > 0) {
                    this.fileManager.logger.log(
                        `→ Scanned ${files.length} text file${files.length > 1 ? 's' : ''}`
                    );
                    await this._processFiles(files);
                    return;
                }
            }

            // Fallback: plain file list (no directory support)
            if (e.dataTransfer.files.length) {
                await this._processFiles(e.dataTransfer.files);
            }
        });
    }
}

// ─── Utilities ──────────────────────────────────────────────

/**
 * Check if a filename has a recognized text extension.
 * @param {string} name
 * @returns {boolean}
 */
function isTextFile(name) {
    const dot = name.lastIndexOf('.');
    if (dot === -1) return false;
    return TEXT_EXTENSIONS.has(name.slice(dot + 1).toLowerCase());
}

/**
 * Recursively collect File objects from dropped DataTransferItems.
 * Filters to text extensions only when traversing directories.
 *
 * @param {DataTransferItemList} items
 * @returns {Promise<File[]>}
 */
async function collectFilesFromEntries(items) {
    const files = [];
    const promises = [];

    for (const item of items) {
        const entry = item.webkitGetAsEntry();
        if (entry) {
            promises.push(traverseEntry(entry, files));
        }
    }

    await Promise.all(promises);
    return files;
}

/**
 * Recursively traverse a FileSystemEntry, collecting text files.
 *
 * @param {FileSystemEntry} entry
 * @param {File[]} accumulator
 */
async function traverseEntry(entry, accumulator) {
    if (entry.isFile) {
        if (isTextFile(entry.name)) {
            const file = await entryToFile(entry);
            if (file) accumulator.push(file);
        }
        return;
    }

    if (entry.isDirectory) {
        const entries = await readDirectoryEntries(entry);
        await Promise.all(entries.map(e => traverseEntry(e, accumulator)));
    }
}

/**
 * Convert a FileSystemFileEntry to a File object.
 * @param {FileSystemFileEntry} entry
 * @returns {Promise<File|null>}
 */
function entryToFile(entry) {
    return new Promise(resolve => {
        entry.file(
            f => resolve(f),
            () => resolve(null),
        );
    });
}

/**
 * Read all entries from a FileSystemDirectoryEntry.
 * Handles pagination (readEntries may return partial results).
 *
 * @param {FileSystemDirectoryEntry} dirEntry
 * @returns {Promise<FileSystemEntry[]>}
 */
function readDirectoryEntries(dirEntry) {
    return new Promise((resolve, reject) => {
        const reader = dirEntry.createReader();
        const allEntries = [];

        function readBatch() {
            reader.readEntries(
                (entries) => {
                    if (entries.length === 0) {
                        resolve(allEntries);
                    } else {
                        allEntries.push(...entries);
                        readBatch(); // readEntries may paginate
                    }
                },
                reject,
            );
        }

        readBatch();
    });
}
