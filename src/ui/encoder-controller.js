import { $, ICONS, renderTokenSpan } from '../utils.js';

// ─── Encoder Controller ───
export class EncoderController {
    /**
     * @param {import('./logger.js').Logger} logger
     * @param {() => import('./tokenizer-manager.js').TokenizerManager | null} getTokenizerManager
     */
    constructor(logger, getTokenizerManager) {
        this.logger = logger;
        this.getTokenizerManager = getTokenizerManager;
    }

    bind() {
        $('encodeBtn').addEventListener('click', () => this._encodeAndDisplay());
    }

    async _encodeAndDisplay() {
        const text = $('tokenizerInput').value.trim();
        if (!text) return;

        const tokenizerManager = this.getTokenizerManager();
        if (!tokenizerManager) return;

        const encodeBtn = $('encodeBtn');

        try {
            encodeBtn.disabled = true;
            encodeBtn.textContent = 'Encoding...';

            const result = await tokenizerManager.encode(text);
            this._renderResult(result, text);
        } catch (error) {
            this.logger.log(`✗ Encoding failed: ${error.message}`);
        } finally {
            encodeBtn.disabled = false;
            encodeBtn.innerHTML = `${ICONS.encode}<span>Tokenize</span>`;
        }
    }

    _renderResult(result, text) {
        const output = $('tokenizerOutput');

        output.innerHTML = result.tokens
            .map((/** @type {number} */ tokenId) => renderTokenSpan(tokenId, result))
            .join('');
        output.classList.remove('hidden');

        const byteCount = new TextEncoder().encode(text).length;
        const compression = (byteCount / result.tokens.length).toFixed(2);

        $('tokenCount').textContent = `${result.tokens.length} tokens`;
        $('compressionRatio').textContent = `${compression}x compression`;

        this.logger.log(`→ Encoded ${byteCount} bytes into ${result.tokens.length} tokens (${compression}x)`);
    }
}
