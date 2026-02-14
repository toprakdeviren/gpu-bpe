/**
 * WebGPU BPE Engine
 *
 * Initializes the WebGPU device, loads and splits the WGSL shader
 * source into per-kernel modules, and compiles compute pipelines.
 */

// ─── Constants ──────────────────────────────────────────────

export const WORKGROUP_SIZE = 256;
export const TABLE_SIZE = 2_097_143;      // prime near 2M (same as Metal)
export const INVALID_TOKEN = 0xFFFF_FFFF;
export const MAX_WG_DIM = 65_535;         // WebGPU max workgroups per dimension

const MB = 1024 * 1024;

/** @readonly */
export const GPU_LIMITS = Object.freeze({
    MAX_STORAGE_BUFFER_SIZE: 512 * MB,
    MAX_BUFFER_SIZE: 512 * MB,
    MAX_COMPUTE_WORKGROUPS_PER_DIM: MAX_WG_DIM,
});

const SHADER_PATH = './bpe.wgsl';

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
 * @param {string} baseUrl - import.meta.url of the calling module
 * @returns {Promise<string>}
 */
async function loadShaderSource(baseUrl) {
    const url = new URL(SHADER_PATH, baseUrl);

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
 * @returns {Record<string, GPUComputePipeline>}
 */
function compilePipelines(device, kernelSources) {
    const pipelines = {};

    for (const [name, code] of Object.entries(kernelSources)) {
        const module = device.createShaderModule({ code, label: name });

        pipelines[name] = device.createComputePipeline({
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

        const shaderSource = await loadShaderSource(import.meta.url);
        const kernelSources = splitKernels(shaderSource);
        this.#pipelines = compilePipelines(this.#device, kernelSources);

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
