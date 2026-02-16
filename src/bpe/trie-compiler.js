/**
 * Trie Compiler — vocab (byte arrays) → v3 binary trie for GPU tokenization
 *
 * Builds an in-memory trie from vocabulary entries, flattens via BFS,
 * and outputs the binary format consumed by TrieTokenizer.
 *
 * Binary Format (v3):
 *   Header (28 bytes): magic, version, nodeCount, edgeCount, maxTokenLen, vocabSize, flags
 *   Nodes (N × 12 bytes): firstChild(u32), numChildren(u32), tokenId(u32)
 *   Edges (E × 8 bytes):  symbol(u8+padding→u32), targetNode(u32)
 */

const TRIE_MAGIC = 0x54524945; // 'TRIE'
const VERSION = 3;
const INVALID_TOKEN = 0xFFFFFFFF;

/**
 * Compile a BPE vocabulary into a binary trie.
 *
 * @param {number[][]} vocab - Array of byte arrays, indexed by token ID
 * @returns {ArrayBuffer} Binary trie data (v3 format)
 */
export function compileVocabToTrie(vocab) {
    // ── 1. Build tree in memory ──
    const root = { children: new Map(), tokenId: INVALID_TOKEN };
    let maxTokenLen = 0;

    for (let tokenId = 0; tokenId < vocab.length; tokenId++) {
        const bytes = vocab[tokenId];
        if (!bytes || bytes.length === 0) continue;

        let node = root;
        for (const byte of bytes) {
            if (!node.children.has(byte)) {
                node.children.set(byte, { children: new Map(), tokenId: INVALID_TOKEN });
            }
            node = node.children.get(byte);
        }
        node.tokenId = tokenId;
        maxTokenLen = Math.max(maxTokenLen, bytes.length);
    }

    // ── 2. Flatten via BFS (cache-friendly ordering) ──
    const flatNodes = [];  // { firstChild, numChildren, tokenId }
    const flatEdges = [];  // { symbol, targetNode }

    const queue = [root];
    const nodeIndex = new Map(); // treeNode → flat index
    nodeIndex.set(root, 0);
    flatNodes.push(null); // placeholder for root

    let head = 0;
    while (head < queue.length) {
        const treeNode = queue[head++];
        const myIndex = nodeIndex.get(treeNode);

        // Sort children by symbol (required for GPU binary search)
        const sortedEntries = Array.from(treeNode.children.entries())
            .sort((a, b) => a[0] - b[0]);

        const firstChild = flatEdges.length;
        const numChildren = sortedEntries.length;

        for (const [symbol, childNode] of sortedEntries) {
            const childIndex = queue.length;
            nodeIndex.set(childNode, childIndex);
            queue.push(childNode);

            flatNodes.push(null); // placeholder
            flatEdges.push({ symbol, targetNode: childIndex });
        }

        flatNodes[myIndex] = {
            firstChild,
            numChildren,
            tokenId: treeNode.tokenId,
        };
    }

    const nodeCount = flatNodes.length;
    const edgeCount = flatEdges.length;

    // ── 3. Serialize to binary (v3) ──
    const headerSize = 28;
    const nodeBytes = nodeCount * 12; // 3 × u32
    const edgeBytes = edgeCount * 8;  // u32 + u32
    const totalSize = headerSize + nodeBytes + edgeBytes;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    // Header
    view.setUint32(0, TRIE_MAGIC, true);
    view.setUint32(4, VERSION, true);
    view.setUint32(8, nodeCount, true);
    view.setUint32(12, edgeCount, true);
    view.setUint32(16, maxTokenLen, true);
    view.setUint32(20, vocab.length, true);  // vocabSize
    view.setUint32(24, 0, true);             // flags

    // Nodes: [firstChild(u32), numChildren(u32), tokenId(u32)]
    let offset = headerSize;
    for (const node of flatNodes) {
        view.setUint32(offset, node.firstChild, true);
        view.setUint32(offset + 4, node.numChildren, true);
        view.setUint32(offset + 8, node.tokenId, true);
        offset += 12;
    }

    // Edges: [symbol(u8 + 3 pad → stored as u32 read), targetNode(u32)]
    for (const edge of flatEdges) {
        // v3: first byte is symbol, rest padding (read as getUint8 on JS side)
        view.setUint8(offset, edge.symbol);
        view.setUint8(offset + 1, 0);
        view.setUint8(offset + 2, 0);
        view.setUint8(offset + 3, 0);
        view.setUint32(offset + 4, edge.targetNode, true);
        offset += 8;
    }

    return buffer;
}
