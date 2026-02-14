// ─── Constants ───
export const BYTES_PER_KB = 1024;
export const BYTES_PER_MB = 1048576;
export const PARAGRAPH_SEPARATOR = '\n\n';
export const DEFAULT_VOCAB_SIZE = 65536;

/** Extensions recognized as text for folder-recursive file selection */
export const TEXT_EXTENSIONS = new Set([
    'txt', 'md', 'markdown', 'rst',
    'json', 'jsonl', 'csv', 'tsv', 'xml', 'yaml', 'yml', 'toml',
    'html', 'htm', 'css',
    'py', 'js', 'mjs', 'ts', 'tsx', 'jsx',
    'swift', 'rs', 'go', 'c', 'h', 'cpp', 'hpp', 'cc',
    'java', 'kt', 'kts', 'scala', 'rb', 'php', 'pl', 'lua',
    'sh', 'bash', 'zsh', 'fish',
    'sql', 'r', 'jl', 'zig', 'wgsl', 'glsl', 'hlsl',
    'tex', 'bib', 'srt', 'vtt', 'log',
]);

export const STATUS_CLASSES = {
    base: 'status-dot',
    loading: ['loading'],
    ok: ['ok'],
    err: ['err']
};

export const CHIP_CLASSES = {
    base: 'vocab-chip',
    inactive: '',
    active: 'active'
};

export const ICONS = {
    download: `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"/>
    </svg>`,
    encode: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M13 10V3L4 14h7v7l9-11h-7z"/>
    </svg>`,
};

// ─── Utilities ───
export const $ = id => document.getElementById(id);

export function formatSize(bytes) {
    if (bytes < BYTES_PER_KB) return `${bytes} B`;
    if (bytes < BYTES_PER_MB) return `${(bytes / BYTES_PER_KB).toFixed(1)} KB`;
    return `${(bytes / BYTES_PER_MB).toFixed(1)} MB`;
}

export function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// ─── Token Display Helpers ───
const WHITESPACE_REPLACEMENTS = [
    [/\n/g, '\\n'],
    [/\t/g, '\\t'],
    [/\r/g, '\\r'],
    [/ /g, '·'],
];

export function escapeWhitespace(text) {
    return WHITESPACE_REPLACEMENTS.reduce(
        (result, [pattern, replacement]) => result.replace(pattern, replacement),
        text
    );
}

export function resolveTokenText(tokenId, result) {
    if (result.vocabStrings?.[tokenId] != null) {
        return result.vocabStrings[tokenId];
    }
    if (result.vocab?.[tokenId]) {
        const entry = result.vocab[tokenId];
        const bytes = entry instanceof Uint8Array ? entry : new Uint8Array(entry);
        return new TextDecoder().decode(bytes);
    }
    return String.fromCharCode(tokenId);
}

export function renderTokenSpan(tokenId, result) {
    const display = escapeWhitespace(resolveTokenText(tokenId, result));
    return `<span class="token" title="Token ID: ${tokenId}">${display}<span class="token-id">#${tokenId}</span></span>`;
}
