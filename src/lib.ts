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

export function normalizePoints(points: number[][]): number[][] {
    points.map(p => {
        if (p.length === 2) return [p[0], p[1], 0, 0];
        if (p.length === 3) return [p[0], p[1], p[2], 0];
        throw new Error(`Invalid point dimension: ${p.length}`);
    });
    return points;
}


