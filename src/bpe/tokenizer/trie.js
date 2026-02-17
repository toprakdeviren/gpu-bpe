/**
 * Trie — compile & parse BPE vocabulary tries
 *
 * Compile: vocab (byte arrays) → v3 binary trie for GPU tokenization
 *   Builds an in-memory trie, flattens via BFS (cache-friendly), serializes.
 *
 * Parse: binary trie → GPU-ready typed arrays
 *   Reads header, unpacks nodes (u32×3) and edges (u32×2).
 *
 * Binary Format (v3):
 *   Header  (28 bytes): magic, version, nodeCount, edgeCount, maxTokenLen, vocabSize, flags
 *   Nodes   (N × 12B):  firstChild(u32), numChildren(u32), tokenId(u32)
 *   Edges   (E × 8B):   symbol(u8+pad→u32), targetNode(u32)
 */

import { INVALID_TOKEN } from '../engine.js';

// ─── Constants ───────────────────────────────────────────────

const TRIE_MAGIC = 0x54524945; // 'TRIE'
const VERSION = 3;
const SUPPORTED_VERSIONS = new Set([2, 3]);
const HEADER_SIZE = 28;

// ─── Types ───────────────────────────────────────────────────

/** @typedef {{ nodeCount: number, edgeCount: number, maxTokenLen: number, version: number }} TrieHeader */

// ═════════════════════════════════════════════════════════════
//  COMPILE: vocab → binary trie
// ═════════════════════════════════════════════════════════════

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

    // ── 3. Serialize to binary (v3) ──
    return serializeTrie(flatNodes, flatEdges, maxTokenLen, vocab.length);
}

// ═════════════════════════════════════════════════════════════
//  PARSE: binary trie → GPU-ready buffers
// ═════════════════════════════════════════════════════════════

/**
 * Parse the binary trie header.
 *
 * @param {ArrayBuffer} data
 * @returns {TrieHeader}
 */
export function parseHeader(data) {
    const view = new DataView(data, 0, HEADER_SIZE);
    const magic = view.getUint32(0, true);
    const version = view.getUint32(4, true);

    if (magic !== TRIE_MAGIC) {
        throw new Error(`Invalid trie magic: 0x${magic.toString(16)}`);
    }
    if (!SUPPORTED_VERSIONS.has(version)) {
        throw new Error(`Unsupported trie version: ${version}`);
    }

    return {
        version,
        nodeCount: view.getUint32(8, true),
        edgeCount: view.getUint32(12, true),
        maxTokenLen: view.getUint32(16, true),
    };
}

/**
 * Parse node + edge buffers from binary trie data.
 *
 * @param {ArrayBuffer} data
 * @param {TrieHeader} header
 * @returns {{ nodes: Uint32Array, edges: Uint32Array }}
 */
export function parseTrieBuffers(data, header) {
    const { version, nodeCount, edgeCount } = header;

    const bytesPerNode = version === 3 ? 12 : 8;
    const bytesPerEdge = version === 3 ? 8 : 4;
    const nodeBytes = nodeCount * bytesPerNode;
    const edgeBytes = edgeCount * bytesPerEdge;

    if (data.byteLength < HEADER_SIZE + nodeBytes + edgeBytes) {
        throw new Error('Truncated trie data');
    }

    const nodes = parseNodes(
        new DataView(data, HEADER_SIZE, nodeBytes),
        nodeCount, version, bytesPerNode,
    );

    const edges = parseEdges(
        new DataView(data, HEADER_SIZE + nodeBytes, edgeBytes),
        edgeCount, version, bytesPerEdge,
    );

    return { nodes, edges };
}

// ═════════════════════════════════════════════════════════════
//  Internal Helpers
// ═════════════════════════════════════════════════════════════

/** Serialize flat nodes + edges → ArrayBuffer (v3 binary format) */
function serializeTrie(flatNodes, flatEdges, maxTokenLen, vocabSize) {
    const nodeCount = flatNodes.length;
    const edgeCount = flatEdges.length;
    const nodeBytes = nodeCount * 12; // 3 × u32
    const edgeBytes = edgeCount * 8;  // u32 + u32
    const totalSize = HEADER_SIZE + nodeBytes + edgeBytes;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    // Header
    view.setUint32(0, TRIE_MAGIC, true);
    view.setUint32(4, VERSION, true);
    view.setUint32(8, nodeCount, true);
    view.setUint32(12, edgeCount, true);
    view.setUint32(16, maxTokenLen, true);
    view.setUint32(20, vocabSize, true);
    view.setUint32(24, 0, true); // flags

    // Nodes: [firstChild(u32), numChildren(u32), tokenId(u32)]
    let offset = HEADER_SIZE;
    for (const node of flatNodes) {
        view.setUint32(offset, node.firstChild, true);
        view.setUint32(offset + 4, node.numChildren, true);
        view.setUint32(offset + 8, node.tokenId, true);
        offset += 12;
    }

    // Edges: [symbol(u8 + 3 pad), targetNode(u32)]
    for (const edge of flatEdges) {
        view.setUint8(offset, edge.symbol);
        view.setUint8(offset + 1, 0);
        view.setUint8(offset + 2, 0);
        view.setUint8(offset + 3, 0);
        view.setUint32(offset + 4, edge.targetNode, true);
        offset += 8;
    }

    return buffer;
}

/** Pack nodes into flat u32×3 array: [firstChild, numChildren, tokenId] */
function parseNodes(view, count, version, stride) {
    const packed = new Uint32Array(count * 3);

    for (let i = 0; i < count; i++) {
        const src = i * stride;
        const dst = i * 3;

        if (version === 3) {
            packed[dst] = view.getUint32(src, true);
            packed[dst + 1] = view.getUint32(src + 4, true);
            packed[dst + 2] = view.getUint32(src + 8, true);
        } else {
            packed[dst] = view.getUint16(src, true);
            packed[dst + 1] = view.getUint16(src + 2, true);
            const tokenId = view.getUint16(src + 4, true);
            packed[dst + 2] = tokenId === 0xFFFF ? INVALID_TOKEN : tokenId;
        }
    }

    return packed;
}

/** Pack edges into flat u32×2 array: [symbol, targetNode] */
function parseEdges(view, count, version, stride) {
    const packed = new Uint32Array(count * 2);

    for (let i = 0; i < count; i++) {
        const src = i * stride;
        const dst = i * 2;

        if (version === 3) {
            packed[dst] = view.getUint8(src);
            packed[dst + 1] = view.getUint32(src + 4, true);
        } else {
            packed[dst] = view.getUint16(src, true) & 0xFF;
            packed[dst + 1] = view.getUint16(src + 2, true);
        }
    }

    return packed;
}
