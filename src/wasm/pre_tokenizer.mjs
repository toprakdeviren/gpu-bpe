// =============================================================================
// Pre-Tokenizer — Unicode-Accurate Word Boundary Detection for BPE Training
// =============================================================================
//
// This module provides GPT-4 style pre-tokenization using the Decoder library's
// full Unicode property tables. It solves the multi-byte Unicode punctuation
// merging problem where byte-level heuristics (as used in GPU kernels) cannot
// distinguish continuation bytes of letters (e.g. ğ: C4 9F) from continuation
// bytes of punctuation (e.g. ": E2 80 9C).
//
// Usage:
//   import { Decoder } from './decoder.mjs';
//   import { PreTokenizer } from './pre_tokenizer.mjs';
//
//   const decoder = await Decoder.init();
//   const pt = new PreTokenizer(decoder);
//   const { bytes, wordStarts } = pt.preTokenize(text);
//
//   // bytes:      Uint8Array — NFC-normalized UTF-8 byte stream
//   // wordStarts: Uint8Array — parallel mask, 1 = word-start position
//
// Integration with BPE trainer:
//   For each wordStarts[i] == 1, set symbols[i] |= WORD_START_BIT
//   Skip GPU word boundary kernel entirely.
//
// =============================================================================

// ─── Character Classes ──────────────────────────────────────

/**
 * Broad Unicode General Category groups for pre-tokenization.
 * @enum {number}
 */
export const CharClass = Object.freeze({
    LETTER: 0,  // Lu|Ll|Lt|Lm|Lo + Mn|Mc|Me (marks stay with letters)
    DIGIT: 1,  // Nd|Nl|No
    WHITESPACE: 2,  // Zs|Zl|Zp + Cc whitespace (tab, etc.)
    PUNCTUATION: 3,  // Pc|Pd|Ps|Pe|Pi|Pf|Po
    SYMBOL: 4,  // Sm|Sc|Sk|So
    NEWLINE: 5,  // \n, \r, U+0085, U+2028, U+2029
    OTHER: 6,  // Cc (non-whitespace control), Cf (format), etc.
});

const NEWLINE_CODEPOINTS = new Set([0x0A, 0x0D, 0x0085, 0x2028, 0x2029]);

// ─── Contraction Matching ───────────────────────────────────

// English contractions: 's 't 'm 'd 're 've 'll

/**
 * Single-char contraction suffixes (after apostrophe): s, t, m, d
 * @type {Set<number>}
 */
const SINGLE_CHAR_SUFFIXES = new Set([
    0x73, 0x53,  // s/S
    0x74, 0x54,  // t/T
    0x6D, 0x4D,  // m/M
    0x64, 0x44,  // d/D
]);

/**
 * Two-char contraction suffixes (after apostrophe): re, ve, ll
 * Each entry: [first_lower, first_upper, second_lower, second_upper]
 * @type {number[][]}
 */
const TWO_CHAR_SUFFIXES = [
    [0x72, 0x52, 0x65, 0x45],  // re/RE
    [0x76, 0x56, 0x65, 0x45],  // ve/VE
    [0x6C, 0x4C, 0x6C, 0x4C],  // ll/LL
];

/** Apostrophe codepoints: ASCII ' and Unicode right single quote ' */
const APOSTROPHES = new Set([0x27, 0x2019]);

/**
 * Try to match an English contraction starting at the apostrophe position.
 *
 * @param {Uint32Array} codepoints
 * @param {Uint8Array} classes
 * @param {number} i — position of the apostrophe
 * @returns {number} — codepoints consumed (including apostrophe), or 0
 */
function matchContraction(codepoints, classes, i) {
    const n = codepoints.length;
    if (i + 1 >= n) return 0;

    const next = codepoints[i + 1];
    const afterIsNonLetter = i + 2 >= n || classes[i + 2] !== CharClass.LETTER;

    // Single-char: 's 't 'm 'd
    if (SINGLE_CHAR_SUFFIXES.has(next) && afterIsNonLetter) {
        return 2;
    }

    // Two-char: 're 've 'll
    if (i + 2 < n) {
        const nextNext = codepoints[i + 2];
        const afterTwoIsNonLetter = i + 3 >= n || classes[i + 3] !== CharClass.LETTER;

        for (const [lo1, hi1, lo2, hi2] of TWO_CHAR_SUFFIXES) {
            if ((next === lo1 || next === hi1) &&
                (nextNext === lo2 || nextNext === hi2) &&
                afterTwoIsNonLetter) {
                return 3;
            }
        }
    }

    return 0;
}

// ─── Codepoint Classification ───────────────────────────────

/**
 * Build a classifier function that uses the Decoder's Unicode tables.
 * Results are cached per codepoint for amortized O(1) lookups.
 *
 * @param {import('./decoder.mjs').Decoder} decoder
 * @returns {(codepoints: Uint32Array) => Uint8Array}
 */
function buildClassifier(decoder) {
    /** @type {Map<number, number>} */
    const cache = new Map();

    /** @param {number} cp */
    function classify(cp) {
        if (NEWLINE_CODEPOINTS.has(cp)) return CharClass.NEWLINE;
        if (decoder.isLetter(cp) || decoder.isMark(cp)) return CharClass.LETTER;
        if (decoder.isDigit(cp) || decoder.isNumber(cp)) return CharClass.DIGIT;
        if (decoder.isWhitespace(cp)) return CharClass.WHITESPACE;
        if (decoder.isPunctuation(cp)) return CharClass.PUNCTUATION;
        if (decoder.isSymbol(cp)) return CharClass.SYMBOL;
        return CharClass.OTHER;
    }

    /**
     * @param {Uint32Array} codepoints
     * @returns {Uint8Array}
     */
    return function classifyAll(codepoints) {
        const n = codepoints.length;
        const classes = new Uint8Array(n);

        for (let i = 0; i < n; i++) {
            const cp = codepoints[i];
            let cls = cache.get(cp);
            if (cls === undefined) {
                cls = classify(cp);
                cache.set(cp, cls);
            }
            classes[i] = cls;
        }

        return classes;
    };
}

// ─── Word Boundary Detection ────────────────────────────────

/**
 * @param {number} cls
 * @returns {boolean}
 */
function isPunctOrSymbol(cls) {
    return cls === CharClass.PUNCTUATION || cls === CharClass.SYMBOL;
}

/**
 * Determine if a class transition constitutes a word boundary.
 *
 * @param {number} prev
 * @param {number} curr
 * @returns {boolean}
 */
function isClassTransitionBoundary(prev, curr) {
    // Letter ↔ Digit
    if (prev === CharClass.LETTER && curr === CharClass.DIGIT) return true;
    if (prev === CharClass.DIGIT && curr === CharClass.LETTER) return true;

    // Letter ↔ Punct/Symbol
    if (prev === CharClass.LETTER && isPunctOrSymbol(curr)) return true;
    if (isPunctOrSymbol(prev) && curr === CharClass.LETTER) return true;

    // Punct/Symbol ↔ Digit
    if (isPunctOrSymbol(prev) && curr === CharClass.DIGIT) return true;
    if (prev === CharClass.DIGIT && isPunctOrSymbol(curr)) return true;

    return false;
}

/**
 * Check if position `i` is a 3-digit boundary within a digit run.
 *
 * @param {Uint8Array} classes
 * @param {number} i
 * @returns {boolean}
 */
function isDigitRunSplitPoint(classes, i) {
    let runStart = i - 1;
    while (runStart > 0 && classes[runStart - 1] === CharClass.DIGIT) {
        runStart--;
    }
    return (i - runStart) % 3 === 0;
}

/**
 * Find word boundaries using GPT-4 style rules at codepoint level.
 *
 * Key design: whitespace is a PREFIX to the following word, not a standalone token.
 * "kabul edilmek" → [kabul][ edilmek], NOT [kabul][ ][edilmek]
 *
 * Rules (in priority order):
 * 1. Position 0 is always a word start
 * 2. Newlines always cause hard boundaries on both sides
 * 3. Whitespace after non-whitespace = new word (space becomes prefix)
 * 4. Non-whitespace after whitespace = CONTINUES the space-prefixed word
 * 5. English contractions merge with the preceding word
 * 6. Transition between non-whitespace classes starts a new word
 * 7. Digit runs split at 3-digit boundaries
 * 8. Consecutive punct/symbol stays as one word
 *
 * @param {Uint32Array} codepoints
 * @param {Uint8Array} classes
 * @returns {Uint8Array} — 1 = word start, 0 = continuation
 */
function findWordBoundaries(codepoints, classes) {
    const n = codepoints.length;
    const starts = new Uint8Array(n);

    if (n === 0) return starts;

    starts[0] = 1;

    let i = 1;
    while (i < n) {
        const prev = classes[i - 1];
        const curr = classes[i];

        // ── Newlines: always hard boundary ──
        if (curr === CharClass.NEWLINE || prev === CharClass.NEWLINE) {
            starts[i] = 1;
            i++;
            continue;
        }

        // ── Whitespace handling (space-prefix model) ──
        if (curr === CharClass.WHITESPACE) {
            if (prev !== CharClass.WHITESPACE) {
                starts[i] = 1; // Space after non-space → new word
            }
            // Consecutive whitespace stays in same chunk
            i++;
            continue;
        }

        if (prev === CharClass.WHITESPACE) {
            // Non-whitespace after whitespace → continues the ▁word unit
            i++;
            continue;
        }

        // ── English contractions ──
        if (prev === CharClass.LETTER && APOSTROPHES.has(codepoints[i])) {
            const consumed = matchContraction(codepoints, classes, i);
            if (consumed > 0) {
                i += consumed;
                continue;
            }
        }

        // ── Class transitions ──
        if (isClassTransitionBoundary(prev, curr)) {
            starts[i] = 1;
            i++;
            continue;
        }

        // ── Digit run splitting (every 3 digits) ──
        if (curr === CharClass.DIGIT && prev === CharClass.DIGIT) {
            if (isDigitRunSplitPoint(classes, i)) {
                starts[i] = 1;
            }
            i++;
            continue;
        }

        // Punct/symbol runs + same-class continuations fall through
        i++;
    }

    return starts;
}

// ─── UTF-8 Encoding ─────────────────────────────────────────

/** @param {number} cp */
function utf8ByteLength(cp) {
    if (cp <= 0x7F) return 1;
    if (cp <= 0x7FF) return 2;
    if (cp <= 0xFFFF) return 3;
    return 4;
}

/**
 * Encode a single codepoint to UTF-8 into a byte array at the given offset.
 *
 * @param {Uint8Array} out
 * @param {number} offset
 * @param {number} cp
 * @returns {number} — number of bytes written
 */
function encodeCodepoint(out, offset, cp) {
    if (cp <= 0x7F) {
        out[offset] = cp;
        return 1;
    }
    if (cp <= 0x7FF) {
        out[offset] = 0xC0 | (cp >> 6);
        out[offset + 1] = 0x80 | (cp & 0x3F);
        return 2;
    }
    if (cp <= 0xFFFF) {
        out[offset] = 0xE0 | (cp >> 12);
        out[offset + 1] = 0x80 | ((cp >> 6) & 0x3F);
        out[offset + 2] = 0x80 | (cp & 0x3F);
        return 3;
    }
    out[offset] = 0xF0 | (cp >> 18);
    out[offset + 1] = 0x80 | ((cp >> 12) & 0x3F);
    out[offset + 2] = 0x80 | ((cp >> 6) & 0x3F);
    out[offset + 3] = 0x80 | (cp & 0x3F);
    return 4;
}

/**
 * Encode codepoints to UTF-8 and map codepoint-level word starts to byte-level.
 *
 * @param {Uint32Array} codepoints
 * @param {Uint8Array} cpWordStarts — codepoint-level word start mask
 * @returns {{ bytes: Uint8Array, wordStarts: Uint8Array }}
 */
function encodeWithBoundaries(codepoints, cpWordStarts) {
    const n = codepoints.length;

    // First pass: total byte length
    let totalBytes = 0;
    for (let i = 0; i < n; i++) {
        totalBytes += utf8ByteLength(codepoints[i]);
    }

    const bytes = new Uint8Array(totalBytes);
    const wordStarts = new Uint8Array(totalBytes);

    // Second pass: encode + map boundaries
    let bytePos = 0;
    for (let i = 0; i < n; i++) {
        const startByte = bytePos;
        bytePos += encodeCodepoint(bytes, bytePos, codepoints[i]);

        if (cpWordStarts[i]) {
            wordStarts[startByte] = 1;
        }
    }

    return { bytes, wordStarts };
}

// ─── String → Codepoints ────────────────────────────────────

/**
 * Convert a JS string to a Uint32Array of Unicode codepoints.
 * Uses the string iterator for proper surrogate pair handling.
 *
 * @param {string} str
 * @returns {Uint32Array}
 */
function toCodepoints(str) {
    const result = new Uint32Array(str.length); // overallocate (surrogates shrink)
    let len = 0;
    for (const ch of str) {
        result[len++] = ch.codePointAt(0);
    }
    return result.subarray(0, len);
}

// ─── Empty Result ───────────────────────────────────────────

const EMPTY_RESULT = Object.freeze({
    bytes: new Uint8Array(0),
    wordStarts: new Uint8Array(0),
});

// ─── PreTokenizer ───────────────────────────────────────────

/**
 * GPT-4 style word boundary detection using full Unicode property tables.
 *
 * Produces a byte stream + parallel word-start mask suitable for GPU BPE training.
 * Replaces the GPU's byte-level char_class() heuristic with Unicode-accurate
 * codepoint-level classification.
 */
export class PreTokenizer {
    /** @type {import('./decoder.mjs').Decoder} */
    #decoder;

    /** @type {(codepoints: Uint32Array) => Uint8Array} */
    #classifyAll;

    /**
     * @param {import('./decoder.mjs').Decoder} decoder — initialized Decoder instance
     */
    constructor(decoder) {
        if (!decoder) {
            throw new Error('PreTokenizer requires an initialized Decoder instance');
        }
        this.#decoder = decoder;
        this.#classifyAll = buildClassifier(decoder);
    }

    /**
     * Pre-tokenize text into bytes + word-start mask.
     *
     * @param {string} text — input text (will be NFC-normalized)
     * @returns {{ bytes: Uint8Array, wordStarts: Uint8Array }}
     */
    preTokenize(text) {
        if (!text || text.length === 0) return EMPTY_RESULT;

        let normalized;
        try {
            normalized = this.#decoder.normalize(text, 0); // NFC
        } catch (e) {
            console.warn('PreTokenizer: WASM normalize failed, using raw text:', e.message);
            normalized = text;
        }

        // Guard: if normalize returned empty for non-empty input, use raw text
        if (!normalized || normalized.length === 0) {
            console.warn('PreTokenizer: normalize returned empty string, using raw text');
            normalized = text;
        }

        const codepoints = toCodepoints(normalized);
        const classes = this.#classifyAll(codepoints);
        const cpWordStarts = findWordBoundaries(codepoints, classes);

        return encodeWithBoundaries(codepoints, cpWordStarts);
    }
}
