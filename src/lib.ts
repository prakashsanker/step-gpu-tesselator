// Cache GPU device to avoid creating multiple devices
let cachedDevice: GPUDevice | null = null;

export async function getGPUDevice(): Promise<GPUDevice> {
    // Return cached device if available
    if (cachedDevice) {
        return cachedDevice;
    }

    if (!('gpu' in navigator)) {
        throw new Error("WebGPU not supported in this browser");
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        throw new Error("Failed to get GPU adapter");
    }

    cachedDevice = await adapter.requestDevice();

    // Handle device loss by clearing cache
    cachedDevice.lost.then((info) => {
        console.error(`WebGPU device lost: ${info.reason}`, info.message);
        cachedDevice = null;
    });

    return cachedDevice;
}

export function normalizePoints(points: number[][]): number[][] {
    return points.map(p => {
        if (p.length === 2) return [p[0], p[1], 0, 0];
        if (p.length === 3) return [p[0], p[1], p[2], 0];
        throw new Error(`Invalid point dimension: ${p.length}`);
    });
}


