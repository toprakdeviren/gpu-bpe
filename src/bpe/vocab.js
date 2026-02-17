/**
 * BPE Vocabulary — token registry with byte-sequence display
 *
 * Manages the growing vocabulary during BPE training:
 * 256 single-byte base tokens + merged tokens added during training.
 */

// ─── Display Helpers ────────────────────────────────────────

/**
 * Convert a byte sequence to a human-readable display string.
 * Tries UTF-8 decode; falls back to hex for non-printable/invalid bytes.
 * Space (0x20) is shown as '▁' for visibility.
 *
 * @param {number[]} bytes
 * @returns {string}
 */
function bytesToDisplayString(bytes) {
    const parts = [];
    let i = 0;

    while (i < bytes.length) {
        const b = bytes[i];

        // ASCII range
        if (b < 0x80) {
            parts.push(formatAsciiByte(b));
            i++;
            continue;
        }

        // Orphan continuation byte
        if (b < 0xC0) {
            parts.push(formatHexByte(b));
            i++;
            continue;
        }

        // Multi-byte UTF-8 start
        const seqLen = b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
        const decoded = tryDecodeUtf8(bytes, i, seqLen);

        if (decoded !== null) {
            parts.push(decoded);
            i += seqLen;
        } else {
            parts.push(formatHexByte(b));
            i++;
        }
    }

    return parts.join('');
}

/** @param {number} b */
function formatAsciiByte(b) {
    if (b === 0x20) return '▁';
    if (b === 0x0A) return '\\n';
    if (b >= 0x21 && b <= 0x7E) return String.fromCharCode(b);
    return formatHexByte(b);
}

/** @param {number} b */
function formatHexByte(b) {
    return `<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`;
}

/**
 * @param {number[]} bytes
 * @param {number} offset
 * @param {number} len
 * @returns {string|null}
 */
function tryDecodeUtf8(bytes, offset, len) {
    if (offset + len > bytes.length) return null;

    // Validate continuation bytes
    for (let j = 1; j < len; j++) {
        if ((bytes[offset + j] & 0xC0) !== 0x80) return null;
    }

    try {
        const slice = new Uint8Array(bytes.slice(offset, offset + len));
        return new TextDecoder('utf-8', { fatal: true }).decode(slice);
    } catch {
        return null;
    }
}

// ─── Vocab Class ────────────────────────────────────────────

export class Vocab {
    /** @type {number[][]} */
    entries = [];
    /** @type {string[]} */
    strings = [];
    /** @type {number} */
    nextTokenId = 256;

    constructor() {
        // Initialize 256 single-byte base tokens
        for (let i = 0; i < 256; i++) {
            this.entries.push([i]);
            this.strings.push(bytesToDisplayString([i]));
        }
    }

    get size() {
        return this.entries.length;
    }

    /**
     * Register a new merged token
     * @param {number} symbolA
     * @param {number} symbolB
     * @returns {number} newTokenId
     */
    addMerge(symbolA, symbolB) {
        const newTokenId = this.nextTokenId++;
        const merged = [...this.entries[symbolA], ...this.entries[symbolB]];
        this.entries.push(merged);
        this.strings.push(bytesToDisplayString(merged));
        return newTokenId;
    }

    /**
     * Export vocab as human-readable text
     * @returns {string}
     */
    export() {
        const lines = [
            `# GPU BPE Vocabulary (WebGPU Trainer)`,
            `# Total tokens: ${this.entries.length}`,
            '',
        ];

        for (let i = 0; i < this.entries.length; i++) {
            const bytes = this.entries[i].join(',');
            lines.push(`${i}\t${this.strings[i]}\t[${bytes}]`);
        }

        return lines.join('\n') + '\n';
    }
}
