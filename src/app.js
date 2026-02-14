/**
 * BPE Tokenizer — Bootstrap
 *
 * Initializes the application, loads the WebGPU engine
 * and Decoder WASM module, then wires them together.
 */
import { BPETokenizerApp } from './ui/bpe-tokenizer-app.js';
import { BPEEngine } from './bpe/engine.js';
import { Decoder } from './wasm/decoder.mjs';
import { PreTokenizer } from './wasm/pre_tokenizer.mjs';

const app = new BPETokenizerApp();
app.initialize();

(async () => {
    try {
        app.statusManager.setStatus('loading', 'Initializing...');

        const [engine, decoder] = await Promise.all([
            new BPEEngine().init(),
            Decoder.init(),
        ]);

        console.log(`[ok] Decoder WASM loaded (Unicode ${decoder.version})`);
        const preTokenizer = new PreTokenizer(decoder);
        app.fileManager.setDecoder(decoder);
        app.setBPEEngine(engine, preTokenizer);
        app.statusManager.setStatus('ok', `Ready · Unicode ${decoder.version}`);
    } catch (error) {
        console.error('Failed to initialize:', error);
        app.statusManager.setStatus('err', 'Initialization failed');
    }
})();
