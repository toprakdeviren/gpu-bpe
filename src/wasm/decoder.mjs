/**
 * Decoder — Unicode 17.0 Processing Library (WASM)
 * High-level JavaScript API (auto-generated from public C headers)
 *
 * @example
 *   import { Decoder } from './decoder.mjs';
 *   const d = await Decoder.init();
 *   d.isValidUtf8('Hello 🌍');       // true
 *   d.getScript(65);                  // Script.LATIN (3)
 *   d.countGraphemes('👨‍👩‍👧‍👦');         // 1
 */

import DecoderModule from './decoder.js';

// ─── Helpers ────────────────────────────────────────────────────

/** Build a frozen enum from sequential names (0, 1, 2…). */
const enumerate = (...names) =>
    Object.freeze(Object.fromEntries(names.map((n, i) => [n, i])));

// ─── Enums (from types.h) ──────────────────────────────────────

/** @enum {number} — Status uses explicit values (negative error codes), not sequential. */
export const Status = Object.freeze({
    SUCCESS: 0,
    ERROR_INVALID_INPUT: -1,
    ERROR_BUFFER_TOO_SMALL: -2,
    ERROR_INVALID_UTF8: -3,
    ERROR_INVALID_UTF16: -4,
    ERROR_INVALID_CODEPOINT: -5,
    ERROR_OUT_OF_MEMORY: -6,
    ERROR_NOT_IMPLEMENTED: -7,
    ERROR_IO: -8,
    ERROR_INVALID_ARGUMENT: -9,
    ERROR_OVERFLOW: -10,
});

/** @enum {number} */
export const Category = enumerate(
    'UNASSIGNED', 'UPPERCASE_LETTER', 'LOWERCASE_LETTER', 'TITLECASE_LETTER',
    'MODIFIER_LETTER', 'OTHER_LETTER', 'NONSPACING_MARK', 'SPACING_MARK',
    'ENCLOSING_MARK', 'DECIMAL_NUMBER', 'LETTER_NUMBER', 'OTHER_NUMBER',
    'CONNECTOR_PUNCTUATION', 'DASH_PUNCTUATION', 'OPEN_PUNCTUATION',
    'CLOSE_PUNCTUATION', 'INITIAL_PUNCTUATION', 'FINAL_PUNCTUATION',
    'OTHER_PUNCTUATION', 'MATH_SYMBOL', 'CURRENCY_SYMBOL', 'MODIFIER_SYMBOL',
    'OTHER_SYMBOL', 'SPACE_SEPARATOR', 'LINE_SEPARATOR', 'PARAGRAPH_SEPARATOR',
    'CONTROL', 'FORMAT', 'SURROGATE', 'PRIVATE_USE',
);

/** @enum {number} */
export const NormalizationForm = enumerate('NFC', 'NFD', 'NFKC', 'NFKD');

/** @enum {number} */
export const QuickCheck = enumerate('MAYBE', 'YES', 'NO');

/** @enum {number} */
export const Script = enumerate(
    'UNKNOWN', 'COMMON', 'INHERITED', 'LATIN', 'GREEK', 'CYRILLIC',
    'ARMENIAN', 'HEBREW', 'ARABIC', 'SYRIAC', 'THAANA', 'DEVANAGARI',
    'BENGALI', 'GURMUKHI', 'GUJARATI', 'ORIYA', 'TAMIL', 'TELUGU',
    'KANNADA', 'MALAYALAM', 'SINHALA', 'THAI', 'LAO', 'TIBETAN',
    'MYANMAR', 'GEORGIAN', 'HANGUL', 'ETHIOPIC', 'CHEROKEE',
    'CANADIAN_ABORIGINAL', 'OGHAM', 'RUNIC', 'KHMER', 'MONGOLIAN',
    'HIRAGANA', 'KATAKANA', 'BOPOMOFO', 'HAN', 'YI', 'OLD_ITALIC',
    'GOTHIC', 'DESERET', 'TAGALOG', 'HANUNOO', 'BUHID', 'TAGBANWA',
    'LIMBU', 'TAI_LE', 'LINEAR_B', 'UGARITIC', 'SHAVIAN', 'OSMANYA',
    'CYPRIOT', 'BRAILLE', 'BUGINESE', 'COPTIC', 'NEW_TAI_LUE',
    'GLAGOLITIC', 'TIFINAGH', 'SYLOTI_NAGRI', 'OLD_PERSIAN',
    'KHAROSHTHI', 'BALINESE', 'CUNEIFORM', 'PHOENICIAN', 'PHAGS_PA',
    'NKO', 'SUNDANESE', 'LEPCHA', 'OL_CHIKI', 'VAI', 'SAURASHTRA',
    'KAYAH_LI', 'REJANG', 'LYCIAN', 'CARIAN', 'LYDIAN', 'CHAM',
    'TAI_THAM', 'TAI_VIET', 'AVESTAN', 'EGYPTIAN_HIEROGLYPHS',
    'SAMARITAN', 'LISU', 'BAMUM', 'JAVANESE', 'MEETEI_MAYEK',
    'IMPERIAL_ARAMAIC', 'OLD_SOUTH_ARABIAN', 'INSCRIPTIONAL_PARTHIAN',
    'INSCRIPTIONAL_PAHLAVI', 'OLD_TURKIC', 'KAITHI', 'BATAK', 'BRAHMI',
    'MANDAIC', 'CHAKMA', 'MEROITIC_CURSIVE', 'MEROITIC_HIEROGLYPHS',
    'MIAO', 'SHARADA', 'SORA_SOMPENG', 'TAKRI', 'CAUCASIAN_ALBANIAN',
    'BASSA_VAH', 'DUPLOYAN', 'ELBASAN', 'GRANTHA', 'KHOJKI',
    'KHUDAWADI', 'LINEAR_A', 'MAHAJANI', 'MANICHAEAN', 'MENDE_KIKAKUI',
    'MODI', 'MRO', 'NABATAEAN', 'OLD_NORTH_ARABIAN', 'OLD_PERMIC',
    'PAHAWH_HMONG', 'PALMYRENE', 'PAU_CIN_HAU', 'PSALTER_PAHLAVI',
    'SIDDHAM', 'TIRHUTA', 'WARANG_CITI', 'AHOM', 'ANATOLIAN_HIEROGLYPHS',
    'HATRAN', 'MULTANI', 'OLD_HUNGARIAN', 'SIGNWRITING', 'ADLAM',
    'BHAIKSUKI', 'MARCHEN', 'NEWA', 'OSAGE', 'TANGUT', 'MASARAM_GONDI',
    'NUSHU', 'SOYOMBO', 'ZANABAZAR_SQUARE', 'DOGRA', 'GUNJALA_GONDI',
    'HANIFI_ROHINGYA', 'MAKASAR', 'MEDEFAIDRIN', 'OLD_SOGDIAN', 'SOGDIAN',
    'ELYMAIC', 'NANDINAGARI', 'NYIAKENG_PUACHUE_HMONG', 'WANCHO',
    'CHORASMIAN', 'DIVES_AKURU', 'KHITAN_SMALL_SCRIPT', 'YEZIDI',
    'CYPRO_MINOAN', 'OLD_UYGHUR', 'TANGSA', 'TOTO', 'VITHKUQI',
    'KATAKANA_OR_HIRAGANA', 'KAWI', 'NAG_MUNDARI', 'OL_ONAL', 'TITUS',
    'TOLONG_SIKI', 'SUNUWAR', 'TODHRI', 'ARA_NAUZ', 'GARAY',
    'GURUNG_KHEMA', 'KIRAT_RAI', 'ONEY', 'TULU_TIGALARI', 'SIDETIC',
    'BERIA_ERFE', 'TAI_YO', 'COUNT',
);

/** @enum {number} */
export const GraphemeBreak = enumerate(
    'OTHER', 'CR', 'LF', 'CONTROL', 'EXTEND', 'ZWJ',
    'REGIONAL_INDICATOR', 'PREPEND', 'SPACINGMARK',
    'L', 'V', 'T', 'LV', 'LVT',
);

/** @enum {number} */
export const WordBreak = enumerate(
    'OTHER', 'CR', 'LF', 'NEWLINE', 'EXTEND', 'ZWJ',
    'REGIONAL_INDICATOR', 'FORMAT', 'KATAKANA', 'HEBREWLETTER',
    'ALETTER', 'SINGLEQUOTE', 'DOUBLEQUOTE', 'MIDNUMLET',
    'MIDLETTER', 'MIDNUM', 'NUMERIC', 'EXTENDNUMLET', 'WSEGSPACE',
);

/** @enum {number} */
export const SentenceBreak = enumerate(
    'OTHER', 'CR', 'LF', 'EXTEND', 'SEP', 'FORMAT', 'SP',
    'LOWER', 'UPPER', 'OLETTER', 'NUMERIC', 'ATERM',
    'SCONTINUE', 'STERM', 'CLOSE',
);

/** @enum {number} */
export const ConfusableType = enumerate(
    'NONE', 'SINGLE_SCRIPT', 'MIXED_SCRIPT', 'WHOLE_SCRIPT', 'RESTRICTED',
);

// ─── Main Class ─────────────────────────────────────────────────

export class Decoder {
    /** @type {object} Emscripten module */ #wasm = null;

    // ═══════════════════════════════════════════════════════════
    //  LIFECYCLE
    // ═══════════════════════════════════════════════════════════

    /** @returns {Promise<Decoder>} */
    static async init() {
        const d = new Decoder();
        d.#wasm = await DecoderModule();
        d.#wasm.ccall('decoder_init', 'number', [], []);
        return d;
    }

    destroy() { this.#wasm.ccall('decoder_cleanup', null, [], []); }

    /** @returns {string} e.g. "17.0.0" */
    get version() {
        return this.#wasm.ccall('decoder_get_unicode_version', 'string', [], []);
    }

    // ═══════════════════════════════════════════════════════════
    //  INTERNAL HELPERS
    // ═══════════════════════════════════════════════════════════

    /** Boolean ccall — all args must be numbers. */
    #bool(fn, ...args) {
        return !!this.#wasm.ccall(
            fn, 'boolean', Array(args.length).fill('number'), args,
        );
    }

    /** Numeric ccall — all args must be numbers. */
    #int(fn, ...args) {
        return this.#wasm.ccall(
            fn, 'number', Array(args.length).fill('number'), args,
        );
    }

    /** Read `count` uint32 values starting at `ptr`. */
    #readU32(ptr, count) {
        const out = new Uint32Array(count);
        for (let i = 0; i < count; i++)
            out[i] = this.#wasm.getValue(ptr + i * 4, 'i32') >>> 0;
        return out;
    }

    /**
     * Allocate a uint32 output buffer, call `fn(...args, bufPtr, capacity)`,
     * and return the first N elements (N = return value) as Uint32Array.
     */
    #u32Result(fn, args, capacity) {
        const ptr = this.#wasm._malloc(capacity * 4);
        try {
            const n = this.#wasm.ccall(
                fn, 'number',
                [...Array(args.length).fill('number'), 'number', 'number'],
                [...args, ptr, capacity],
            );
            return this.#readU32(ptr, n);
        } finally {
            this.#wasm._free(ptr);
        }
    }

    /** Scope one or more `malloc`'d blocks — freed automatically. */
    #withAlloc(sizes, fn) {
        const ptrs = [];
        try {
            for (const s of sizes) ptrs.push(this.#wasm._malloc(s));
            return fn(...ptrs);
        } finally {
            for (const p of ptrs) this.#wasm._free(p);
        }
    }

    /** Scope a JS string as NUL-terminated UTF-8 in WASM heap. */
    #withUtf8(str, fn) {
        const len = this.#wasm.lengthBytesUTF8(str);
        const ptr = this.#wasm._malloc(len + 1);
        this.#wasm.stringToUTF8(str, ptr, len + 1);
        try { return fn(ptr, len); }
        finally { this.#wasm._free(ptr); }
    }

    /** Scope a JS string as a UTF-32 (uint32[]) block in WASM heap. */
    #withCodepoints(str, fn) {
        const cps = [...str].map(c => c.codePointAt(0));
        const ptr = this.#wasm._malloc(cps.length * 4);
        for (let i = 0; i < cps.length; i++)
            this.#wasm.setValue(ptr + i * 4, cps[i], 'i32');
        try { return fn(ptr, cps.length); }
        finally { this.#wasm._free(ptr); }
    }

    /** Segmentation count (graphemes / words / sentences). */
    #segCount(str, fn) {
        return this.#withCodepoints(str, (p, n) => this.#int(fn, p, n));
    }

    // ═══════════════════════════════════════════════════════════
    //  ENCODING (low-level pointer APIs)
    // ═══════════════════════════════════════════════════════════

    utf8ToUtf16(src, srcLen, dst, dstCap, written) { return this.#int('decoder_utf8_to_utf16', src, srcLen, dst, dstCap, written); }
    utf16ToUtf8(src, srcLen, dst, dstCap, written) { return this.#int('decoder_utf16_to_utf8', src, srcLen, dst, dstCap, written); }
    isValidUtf32(ptr, len) { return this.#bool('decoder_is_valid_utf32', ptr, len); }

    // ═══════════════════════════════════════════════════════════
    //  CODEPOINT PROPERTIES
    // ═══════════════════════════════════════════════════════════

    isValid(cp) { return this.#bool('decoder_is_valid', cp); }
    isAssigned(cp) { return this.#bool('decoder_is_assigned', cp); }
    isPrivateUse(cp) { return this.#bool('decoder_is_private_use', cp); }
    isSurrogate(cp) { return this.#bool('decoder_is_surrogate', cp); }
    isNoncharacter(cp) { return this.#bool('decoder_is_noncharacter', cp); }
    getCategory(cp) { return this.#int('decoder_get_category', cp); }
    isLetter(cp) { return this.#bool('decoder_is_letter', cp); }
    isUppercase(cp) { return this.#bool('decoder_is_uppercase', cp); }
    isLowercase(cp) { return this.#bool('decoder_is_lowercase', cp); }
    isTitlecase(cp) { return this.#bool('decoder_is_titlecase', cp); }
    isDigit(cp) { return this.#bool('decoder_is_digit', cp); }
    isNumber(cp) { return this.#bool('decoder_is_number', cp); }
    isPunctuation(cp) { return this.#bool('decoder_is_punctuation', cp); }
    isSymbol(cp) { return this.#bool('decoder_is_symbol', cp); }
    isMark(cp) { return this.#bool('decoder_is_mark', cp); }
    isSeparator(cp) { return this.#bool('decoder_is_separator', cp); }
    isControl(cp) { return this.#bool('decoder_is_control', cp); }
    isFormat(cp) { return this.#bool('decoder_is_format', cp); }
    isWhitespace(cp) { return this.#bool('decoder_is_whitespace', cp); }
    isSpace(cp) { return this.#bool('decoder_is_space', cp); }
    isAlphanumeric(cp) { return this.#bool('decoder_is_alphanumeric', cp); }
    isAlphabetic(cp) { return this.#bool('decoder_is_alphabetic', cp); }
    isNumeric(cp) { return this.#bool('decoder_is_numeric', cp); }
    getNumericValue(cp) { return this.#int('decoder_get_numeric_value', cp); }
    getDigitValue(cp) { return this.#int('decoder_get_digit_value', cp); }
    isInVersion(cp, major, minor) { return this.#bool('decoder_is_in_version', cp, major, minor); }

    /** Resolve a Unicode character name → codepoint. */
    fromName(name) {
        return this.#wasm.ccall('decoder_from_name', 'number', ['string'], [name]);
    }

    // ═══════════════════════════════════════════════════════════
    //  CASE MAPPING
    // ═══════════════════════════════════════════════════════════

    // Simple (1 → 1)
    toUpper(cp) { return this.#int('decoder_to_upper', cp); }
    toLower(cp) { return this.#int('decoder_to_lower', cp); }
    toTitle(cp) { return this.#int('decoder_to_title', cp); }
    caseFold(cp) { return this.#int('decoder_case_fold', cp); }

    // Full (1 → N) → Uint32Array
    toUpperFull(cp) { return this.#u32Result('decoder_to_upper_full', [cp], 4); }
    toLowerFull(cp) { return this.#u32Result('decoder_to_lower_full', [cp], 4); }
    toTitleFull(cp) { return this.#u32Result('decoder_to_title_full', [cp], 4); }
    caseFoldFull(cp) { return this.#u32Result('decoder_case_fold_full', [cp], 4); }

    // Case properties
    isCaseIgnorable(cp) { return this.#bool('decoder_is_case_ignorable', cp); }
    isCased(cp) { return this.#bool('decoder_is_cased', cp); }

    // String-level (low-level pointer APIs)
    stringToUpper(src, srcLen, dst, dstCap, dstLen) { return this.#int('decoder_string_to_upper', src, srcLen, dst, dstCap, dstLen); }
    stringToLower(src, srcLen, dst, dstCap, dstLen) { return this.#int('decoder_string_to_lower', src, srcLen, dst, dstCap, dstLen); }
    stringToTitle(src, srcLen, dst, dstCap, dstLen) { return this.#int('decoder_string_to_title', src, srcLen, dst, dstCap, dstLen); }
    stringCaseFold(src, srcLen, dst, dstCap, dstLen) { return this.#int('decoder_string_case_fold', src, srcLen, dst, dstCap, dstLen); }
    stringToUpperLocale(src, srcLen, loc, dst, dstCap, dstLen) { return this.#int('decoder_string_to_upper_locale', src, srcLen, loc, dst, dstCap, dstLen); }
    stringToLowerLocale(src, srcLen, loc, dst, dstCap, dstLen) { return this.#int('decoder_string_to_lower_locale', src, srcLen, loc, dst, dstCap, dstLen); }
    stringCaseFoldLocale(src, srcLen, loc, dst, dstCap, dstLen) { return this.#int('decoder_string_case_fold_locale', src, srcLen, loc, dst, dstCap, dstLen); }

    // ═══════════════════════════════════════════════════════════
    //  NORMALIZATION
    // ═══════════════════════════════════════════════════════════

    isCombining(cp) { return this.#bool('decoder_is_combining', cp); }
    canCompose(a, b) { return this.#bool('decoder_can_compose', a, b); }
    compose(a, b) { return this.#int('decoder_compose', a, b); }
    decompose(cp) { return this.#u32Result('decoder_decompose', [cp], 18); }

    /** @param {string} str @param {number} [form=NFC] @returns {string} */
    normalize(str, form = NormalizationForm.NFC) {
        return this.#withUtf8(str, (ptr, len) => {
            const cap = len * 4;
            return this.#withAlloc([cap, 4], (outPtr, lenPtr) => {
                this.#int('decoder_normalize_utf8', ptr, len, form, outPtr, cap, lenPtr);
                return this.#wasm.UTF8ToString(outPtr, this.#wasm.getValue(lenPtr, 'i32'));
            });
        });
    }

    /**
     * Normalize raw UTF-8 bytes without any JS string conversion.
     * @param {Uint8Array} bytes — UTF-8 encoded input
     * @param {number} [form=0] — NormalizationForm (0=NFC, 1=NFD, 2=NFKC, 3=NFKD)
     * @returns {Uint8Array} — normalized UTF-8 bytes
     */
    normalizeBytes(bytes, form = NormalizationForm.NFC) {
        const srcLen = bytes.length;
        const cap = srcLen * 4; // worst-case expansion for decomposition
        return this.#withAlloc([srcLen, cap, 4], (srcPtr, dstPtr, lenPtr) => {
            this.#wasm.HEAPU8.set(bytes, srcPtr);
            this.#int('decoder_normalize_utf8', srcPtr, srcLen, form, dstPtr, cap, lenPtr);
            const outLen = this.#wasm.getValue(lenPtr, 'i32');
            return new Uint8Array(this.#wasm.HEAPU8.buffer, dstPtr, outLen).slice();
        });
    }

    /**
     * Batch-classify UTF-8 bytes into per-codepoint CharClass values.
     * Returns { classes: Uint8Array, codepointCount: number }.
     * CharClass: 0=LETTER, 1=DIGIT, 2=WHITESPACE, 3=PUNCTUATION, 4=SYMBOL, 5=NEWLINE, 6=OTHER
     * @param {Uint8Array} bytes — UTF-8 encoded input
     * @returns {{ classes: Uint8Array, codepointCount: number }}
     */
    classifyBytes(bytes) {
        const srcLen = bytes.length;
        const cap = srcLen; // max codepoints ≤ bytes
        return this.#withAlloc([srcLen, cap, 4], (srcPtr, classPtr, countPtr) => {
            this.#wasm.HEAPU8.set(bytes, srcPtr);
            this.#int('decoder_classify_codepoints', srcPtr, srcLen, classPtr, cap, countPtr);
            const count = this.#wasm.getValue(countPtr, 'i32');
            return {
                classes: new Uint8Array(this.#wasm.HEAPU8.buffer, classPtr, count).slice(),
                codepointCount: count,
            };
        });
    }

    /** @param {string} str @param {number} [form=NFC] @returns {boolean} */
    isNormalized(str, form = NormalizationForm.NFC) {
        return this.#withUtf8(str, (ptr, len) =>
            this.#bool('decoder_is_normalized_utf8', ptr, len, form),
        );
    }

    // ═══════════════════════════════════════════════════════════
    //  SECURITY & IDENTIFIERS
    // ═══════════════════════════════════════════════════════════

    isIdentifierStart(cp) { return this.#bool('decoder_is_identifier_start', cp); }
    isIdentifierContinue(cp) { return this.#bool('decoder_is_identifier_continue', cp); }
    isPatternSyntax(cp) { return this.#bool('decoder_is_pattern_syntax', cp); }
    isPatternWhitespace(cp) { return this.#bool('decoder_is_pattern_whitespace', cp); }
    isRestrictedIdentifierStart(cp) { return this.#bool('decoder_is_restricted_identifier_start', cp); }
    isRestrictedIdentifierContinue(cp) { return this.#bool('decoder_is_restricted_identifier_continue', cp); }
    isConfusable(a, b) { return this.#bool('decoder_is_confusable', a, b); }
    getConfusableType(a, b) { return this.#int('decoder_get_confusable_type', a, b); }

    // ═══════════════════════════════════════════════════════════
    //  SCRIPT & BLOCK
    // ═══════════════════════════════════════════════════════════

    getScript(cp) { return this.#int('decoder_get_script', cp); }
    getBlock(cp) { return this.#int('decoder_get_block', cp); }
    isInBlock(cp, block) { return this.#bool('decoder_is_in_block', cp, block); }

    // ═══════════════════════════════════════════════════════════
    //  STRING OPERATIONS (JS string ↔ WASM memory)
    // ═══════════════════════════════════════════════════════════

    /** @param {string} str @returns {boolean} */
    isValidUtf8(str) {
        return this.#withUtf8(str, (p, n) => this.#bool('decoder_is_valid_utf8', p, n));
    }

    /** @param {string} str @returns {number} UTF-8 byte length */
    utf8Length(str) {
        return this.#withUtf8(str, (p, n) => this.#int('decoder_utf8_length', p, n));
    }

    /** @param {string} str @returns {number} codepoint count */
    charCount(str) {
        return this.#withUtf8(str, (p, n) => this.#int('decoder_utf8_char_count', p, n));
    }

    /** @param {string} str @returns {Uint32Array} codepoints */
    toCodepoints(str) {
        return this.#withUtf8(str, (ptr, len) =>
            this.#withAlloc([len * 4, 4], (outPtr, lenPtr) => {
                this.#int('decoder_utf8_to_utf32', ptr, len, outPtr, len, lenPtr);
                return this.#readU32(outPtr, this.#wasm.getValue(lenPtr, 'i32'));
            }),
        );
    }

    /** @param {number} cp @returns {string} Unicode character name */
    getName(cp) {
        return this.#withAlloc([256], buf => {
            this.#int('decoder_get_name', cp, buf, 256);
            return this.#wasm.UTF8ToString(buf);
        });
    }

    /** @param {string} str @returns {number} */
    countGraphemes(str) { return this.#segCount(str, 'decoder_count_graphemes'); }
    /** @param {string} str @returns {number} */
    countWords(str) { return this.#segCount(str, 'decoder_count_words'); }
    /** @param {string} str @returns {number} */
    countSentences(str) { return this.#segCount(str, 'decoder_count_sentences'); }

    /** @param {string} str @returns {boolean} suspicious mixed-script content */
    isSuspicious(str) {
        return this.#withCodepoints(str, (p, n) => this.#bool('decoder_is_suspicious', p, n));
    }

    /** @param {string} str @returns {boolean} */
    isWellFormed(str) {
        return this.#withCodepoints(str, (p, n) => this.#bool('decoder_is_well_formed', p, n));
    }
}
