/**
 * GPU Buffer Utilities — shared helpers for tokenizer + trainer
 */

/**
 * Upload typed array data to a new GPU buffer.
 *
 * @param {GPUDevice} device
 * @param {TypedArray} data
 * @param {GPUBufferUsageFlags} usage
 * @returns {GPUBuffer}
 */
export function uploadBuffer(device, data, usage) {
    const buf = device.createBuffer({
        size: data.byteLength,
        usage: usage | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new data.constructor(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
}

/**
 * Allocate an empty GPU buffer.
 *
 * @param {GPUDevice} device
 * @param {number} size
 * @param {GPUBufferUsageFlags} usage
 * @returns {GPUBuffer}
 */
export function allocBuffer(device, size, usage) {
    return device.createBuffer({ size, usage });
}

/**
 * Destroy an array of GPU buffers.
 *
 * @param {GPUBuffer[]} buffers
 */
export function destroyBuffers(buffers) {
    for (const buf of buffers) buf.destroy();
}
