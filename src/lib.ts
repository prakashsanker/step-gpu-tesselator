export async function getGPUDevice() {
    if (!('gpu' in navigator)) {
        throw new Error("WebGPU not supported in this browser");
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        throw new Error("Failed to get GPU adapter");
    }

    const device = await adapter.requestDevice();

    return device;
}