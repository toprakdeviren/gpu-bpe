/**
 * WebGPU BPE Engine
 *
 * Initializes the WebGPU device, loads and splits the WGSL shader
 * source into per-kernel modules, and compiles compute pipelines.
 */

// ─── Constants ──────────────────────────────────────────────

export const WORKGROUP_SIZE = 256;
export const TABLE_SIZE = 2_097_152;      // 2^21 — power-of-2 for bitwise AND modulo
export const INVALID_TOKEN = 0xFFFF_FFFF;
export const MAX_WG_DIM = 65_535;         // WebGPU max workgroups per dimension

const MB = 1024 * 1024;

/** @readonly */
export const GPU_LIMITS = Object.freeze({
    MAX_STORAGE_BUFFER_SIZE: 512 * MB,
    MAX_BUFFER_SIZE: 512 * MB,
    MAX_COMPUTE_WORKGROUPS_PER_DIM: MAX_WG_DIM,
});

const SHADER_PATHS = ['./train.wgsl', './tokenize.wgsl'];

// ─── Dispatch Helper ────────────────────────────────────────

/**
 * Dispatch a compute pass, splitting into a 2D grid when
 * total workgroups exceeds the per-dimension limit (65,535).
 *
 * WGSL kernels use `flat_id(gid, num_workgroups)` to linearize.
 *
 * @param {GPUComputePassEncoder} pass
 * @param {number} totalWorkgroups
 */
export function dispatch2D(pass, totalWorkgroups) {
    if (totalWorkgroups <= 0) return;

    if (totalWorkgroups <= MAX_WG_DIM) {
        pass.dispatchWorkgroups(totalWorkgroups);
        return;
    }

    const x = Math.min(totalWorkgroups, MAX_WG_DIM);
    const y = Math.ceil(totalWorkgroups / x);
    pass.dispatchWorkgroups(x, y);
}

// ─── Shader Splitting ───────────────────────────────────────

/**
 * Split a single WGSL source into per-kernel sources.
 *
 * Uses `// --- KERNEL: <name> ---` markers. Everything before
 * the first marker becomes a shared preamble prepended to every
 * kernel module.
 *
 * @param {string} source
 * @returns {Record<string, string>}
 */
function splitKernels(source) {
    const marker = /^\/\/ --- KERNEL: (\S+) ---$/gm;
    const matches = [...source.matchAll(marker)];

    if (matches.length === 0) {
        throw new Error('No kernel markers found in shader source');
    }

    const preamble = source.slice(0, matches[0].index);
    const kernels = {};

    for (let i = 0; i < matches.length; i++) {
        const name = matches[i][1];
        const start = matches[i].index + matches[i][0].length;
        const end = matches[i + 1]?.index ?? source.length;
        kernels[name] = preamble + source.slice(start, end);
    }

    return kernels;
}

// ─── Shader Loading ─────────────────────────────────────────

/**
 * @param {string} path - Shader file path (relative to baseUrl)
 * @param {string} baseUrl - import.meta.url of the calling module
 * @returns {Promise<string>}
 */
async function loadShaderSource(path, baseUrl) {
    const url = new URL(path, baseUrl);
    url.searchParams.set('v', Date.now());   // cache-bust

    try {
        const response = await fetch(url);
        if (response.ok) {
            return await response.text();
        }
        throw new Error(`HTTP ${response.status}`);
    } catch (cause) {
        throw new Error(`Could not load shader: ${url}`, { cause });
    }
}

// ─── Pipeline Compilation ───────────────────────────────────

/**
 * Compile all kernel sources into compute pipelines.
 *
 * @param {GPUDevice} device
 * @param {Record<string, string>} kernelSources
 * @returns {Promise<Record<string, GPUComputePipeline>>}
 */
async function compilePipelines(device, kernelSources) {
    const pipelines = {};

    for (const [name, code] of Object.entries(kernelSources)) {
        const module = device.createShaderModule({ code, label: name });

        // Check for compilation errors before creating pipeline
        const info = await module.getCompilationInfo();
        for (const msg of info.messages) {
            if (msg.type === 'error') {
                console.error(`[WGSL] ${name}:${msg.lineNum}:${msg.linePos} ${msg.message}`);
            }
        }

        pipelines[name] = await device.createComputePipelineAsync({
            label: name,
            layout: 'auto',
            compute: { module, entryPoint: name },
        });
    }

    return pipelines;
}

// ─── Device Initialization ──────────────────────────────────

/**
 * @returns {Promise<GPUDevice>}
 */
async function requestGPUDevice() {
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported in this browser');
    }

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
    });

    if (!adapter) {
        throw new Error('No WebGPU adapter found');
    }

    return adapter.requestDevice({
        requiredLimits: {
            maxStorageBufferBindingSize: GPU_LIMITS.MAX_STORAGE_BUFFER_SIZE,
            maxBufferSize: GPU_LIMITS.MAX_BUFFER_SIZE,
            maxComputeWorkgroupsPerDimension: GPU_LIMITS.MAX_COMPUTE_WORKGROUPS_PER_DIM,
        },
    });
}

// ─── BPE Engine ─────────────────────────────────────────────

export class BPEEngine {
    /** @type {GPUDevice} */
    #device = null;

    /** @type {Record<string, GPUComputePipeline>} */
    #pipelines = {};

    /** @type {boolean} */
    #initialized = false;

    /** Read-only access to the GPU device */
    get device() {
        this.#assertInitialized();
        return this.#device;
    }

    /** Read-only access to compiled pipelines */
    get pipelines() {
        this.#assertInitialized();
        return this.#pipelines;
    }

    /**
     * Initialize the engine: request device, load shader, compile pipelines.
     * @returns {Promise<this>}
     */
    async init() {
        if (this.#initialized) return this;

        this.#device = await requestGPUDevice();

        // Load and compile all shader modules (train + tokenize)
        const allKernels = {};
        for (const path of SHADER_PATHS) {
            const source = await loadShaderSource(path, import.meta.url);
            const kernels = splitKernels(source);
            Object.assign(allKernels, kernels);
        }
        this.#pipelines = await compilePipelines(this.#device, allKernels);

        this.#initialized = true;

        const kernelCount = Object.keys(this.#pipelines).length;
        console.log(`[ok] BPE Engine initialized (${kernelCount} kernels)`);

        return this;
    }

    #assertInitialized() {
        if (!this.#initialized) {
            throw new Error('BPEEngine not initialized — call await engine.init() first');
        }
    }
}
