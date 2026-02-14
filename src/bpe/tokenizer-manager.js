// ─── Tokenizer Manager Class ───
export class TokenizerManager {
    constructor(bpeEngine, trainingManager, logger) {
        this.bpeEngine = bpeEngine;
        this.trainingManager = trainingManager;
        this.logger = logger;
    }

    /**
     * Greedy BPE encoder using trained merge rules.
     * Applies merge rules in the exact order they were learned during training.
     */
    async encode(text) {
        const model = this.trainingManager.getTrainedModel();
        if (!model) {
            throw new Error('No trained model available');
        }

        const { vocab, vocabStrings, merges } = model;

        if (!merges || merges.length === 0) {
            // Fallback: byte-level if no merges available
            const bytes = new TextEncoder().encode(text);
            return { tokens: Array.from(bytes), text, vocab, vocabStrings };
        }

        // Build merge priority map: pack(tokenA, tokenB) → { newTokenId, priority }
        const mergePriority = new Map();
        for (let i = 0; i < merges.length; i++) {
            const [a, b, newId] = merges[i];
            const key = (a << 16) | b;
            if (!mergePriority.has(key)) {
                mergePriority.set(key, { newId, priority: i });
            }
        }

        // Start with byte-level tokens
        const bytes = new TextEncoder().encode(text);
        let tokens = Array.from(bytes);

        // Apply merges in priority order (lowest priority = first learned = most frequent)
        // For each merge rule, scan the token list for matching adjacent pairs
        for (const [tokenA, tokenB, newTokenId] of merges) {
            if (tokens.length < 2) break;

            let i = 0;
            const merged = [];
            while (i < tokens.length) {
                if (i + 1 < tokens.length && tokens[i] === tokenA && tokens[i + 1] === tokenB) {
                    merged.push(newTokenId);
                    i += 2;
                } else {
                    merged.push(tokens[i]);
                    i++;
                }
            }
            tokens = merged;
        }

        return { tokens, text, vocab, vocabStrings };
    }
}
