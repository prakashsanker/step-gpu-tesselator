import { getGPUDevice } from "./lib";

type Vec2 = [number, number];

interface TrimGridClassificationInput {
    boundary: Vec2[];
    holes: Vec2[][];
    gridDensityU: number;
    gridDensityV: number;
    uMin: number;
    vMin: number;
    du: number;
    dv: number;
    boundaryTolerance: number;
    useNearBoundary: boolean;
}

const TRIM_GRID_SHADER = /* wgsl */ `
struct Params {
    dims: vec4u,  // gridU, gridV, boundaryCount, holeCount
    uv: vec4f,    // uMin, vMin, du, dv
    misc: vec4f,  // boundaryTolerance, useNearBoundary, _, _
}

@group(0) @binding(0) var<storage, read> boundaryPts: array<vec2f>;
@group(0) @binding(1) var<storage, read> holePts: array<vec2f>;
@group(0) @binding(2) var<storage, read> holeOffsets: array<u32>; // size = holeCount + 1
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> maskOut: array<u32>;

fn pointInPolygonBoundary(x: f32, y: f32) -> bool {
    let count = params.dims.z;
    if (count < 3u) {
        return false;
    }

    var inside = false;
    var j = count - 1u;
    var i = 0u;
    loop {
        if (i >= count) {
            break;
        }
        let pi = boundaryPts[i];
        let pj = boundaryPts[j];
        let yi_gt = pi.y > y;
        let yj_gt = pj.y > y;
        if (yi_gt != yj_gt) {
            let xIntersect = (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y) + pi.x;
            if (x < xIntersect) {
                inside = !inside;
            }
        }
        j = i;
        i = i + 1u;
    }
    return inside;
}

fn pointInPolygonHoleRange(x: f32, y: f32, startIdx: u32, endIdx: u32) -> bool {
    let count = endIdx - startIdx;
    if (count < 3u) {
        return false;
    }

    var inside = false;
    var j = count - 1u;
    var i = 0u;
    loop {
        if (i >= count) {
            break;
        }
        let pi = holePts[startIdx + i];
        let pj = holePts[startIdx + j];
        let yi_gt = pi.y > y;
        let yj_gt = pj.y > y;
        if (yi_gt != yj_gt) {
            let xIntersect = (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y) + pi.x;
            if (x < xIntersect) {
                inside = !inside;
            }
        }
        j = i;
        i = i + 1u;
    }
    return inside;
}

fn isNearBoundary(x: f32, y: f32, tolerance: f32) -> bool {
    let count = params.dims.z;
    if (count < 2u) {
        return false;
    }
    var i = 0u;
    loop {
        if (i >= count) {
            break;
        }
        let p1 = boundaryPts[i];
        let p2 = boundaryPts[(i + 1u) % count];
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let lenSq = dx * dx + dy * dy;
        if (lenSq > 1e-12) {
            var t = ((x - p1.x) * dx + (y - p1.y) * dy) / lenSq;
            t = clamp(t, 0.0, 1.0);
            let cx = p1.x + t * dx;
            let cy = p1.y + t * dy;
            let ddx = x - cx;
            let ddy = y - cy;
            let dist = sqrt(ddx * ddx + ddy * ddy);
            if (dist < tolerance) {
                return true;
            }
        }
        i = i + 1u;
    }
    return false;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let gridU = params.dims.x;
    let gridV = params.dims.y;
    let cols = gridU + 1u;
    let rows = gridV + 1u;
    let total = cols * rows;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let i = idx % cols;
    let j = idx / cols;

    let u = params.uv.x + f32(i) * params.uv.z;
    let v = params.uv.y + f32(j) * params.uv.w;

    let insideBoundary = pointInPolygonBoundary(u, v);
    let useNear = params.misc.y > 0.5;
    let nearBoundary = useNear && isNearBoundary(u, v, params.misc.x);

    var insideHole = false;
    let holeCount = params.dims.w;
    if (holeCount > 0u) {
        var h = 0u;
        loop {
            if (h >= holeCount) {
                break;
            }
            let startIdx = holeOffsets[h];
            let endIdx = holeOffsets[h + 1u];
            if (pointInPolygonHoleRange(u, v, startIdx, endIdx)) {
                insideHole = true;
                break;
            }
            h = h + 1u;
        }
    }

    if ((insideBoundary || nearBoundary) && !insideHole) {
        maskOut[idx] = 1u;
    } else {
        maskOut[idx] = 0u;
    }
}
`;

const cache: {
    device: GPUDevice | null;
    pipeline: GPUComputePipeline | null;
    boundaryBuffer: GPUBuffer | null;
    holesBuffer: GPUBuffer | null;
    holeOffsetsBuffer: GPUBuffer | null;
    paramsBuffer: GPUBuffer | null;
    maskBuffer: GPUBuffer | null;
    maskStagingBuffer: GPUBuffer | null;
    boundaryCapacity: number;
    holesCapacity: number;
    holeOffsetsCapacity: number;
    maskCapacity: number;
    maskStagingCapacity: number;
} = {
    device: null,
    pipeline: null,
    boundaryBuffer: null,
    holesBuffer: null,
    holeOffsetsBuffer: null,
    paramsBuffer: null,
    maskBuffer: null,
    maskStagingBuffer: null,
    boundaryCapacity: 0,
    holesCapacity: 0,
    holeOffsetsCapacity: 0,
    maskCapacity: 0,
    maskStagingCapacity: 0,
};

function destroyCache() {
    cache.boundaryBuffer?.destroy();
    cache.holesBuffer?.destroy();
    cache.holeOffsetsBuffer?.destroy();
    cache.paramsBuffer?.destroy();
    cache.maskBuffer?.destroy();
    cache.maskStagingBuffer?.destroy();

    cache.boundaryBuffer = null;
    cache.holesBuffer = null;
    cache.holeOffsetsBuffer = null;
    cache.paramsBuffer = null;
    cache.maskBuffer = null;
    cache.maskStagingBuffer = null;

    cache.boundaryCapacity = 0;
    cache.holesCapacity = 0;
    cache.holeOffsetsCapacity = 0;
    cache.maskCapacity = 0;
    cache.maskStagingCapacity = 0;
}

function roundCapacity(size: number): number {
    const alignment = 256;
    return Math.max(alignment, Math.ceil(size / alignment) * alignment);
}

function ensureBuffer(
    device: GPUDevice,
    current: GPUBuffer | null,
    currentCapacity: number,
    requiredSize: number,
    usage: GPUBufferUsageFlags,
    label: string
): { buffer: GPUBuffer; capacity: number } {
    const needed = roundCapacity(requiredSize);
    if (current && currentCapacity >= needed) {
        return { buffer: current, capacity: currentCapacity };
    }

    current?.destroy();
    const buffer = device.createBuffer({
        label,
        size: needed,
        usage,
    });
    return { buffer, capacity: needed };
}

async function ensurePipeline(): Promise<{ device: GPUDevice; pipeline: GPUComputePipeline }> {
    const device = await getGPUDevice();
    if (cache.device !== device) {
        destroyCache();
        cache.pipeline = null;
        cache.device = device;
    }

    if (!cache.pipeline) {
        const module = device.createShaderModule({
            label: "trim-grid-classify-shader",
            code: TRIM_GRID_SHADER,
        });
        cache.pipeline = device.createComputePipeline({
            label: "trim-grid-classify-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "main",
            },
        });
    }

    if (!cache.paramsBuffer) {
        cache.paramsBuffer = device.createBuffer({
            label: "trim-grid-params",
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    return { device, pipeline: cache.pipeline };
}

let loggedGpuTrimGridError = false;

export async function classifyTrimGridGPU(input: TrimGridClassificationInput): Promise<Uint32Array | null> {
    try {
        const { device, pipeline } = await ensurePipeline();
        const boundaryFlat = new Float32Array(input.boundary.length * 2);
        for (let i = 0; i < input.boundary.length; i++) {
            boundaryFlat[i * 2 + 0] = input.boundary[i][0];
            boundaryFlat[i * 2 + 1] = input.boundary[i][1];
        }

        const holeCount = input.holes.length;
        let totalHolePoints = 0;
        for (const hole of input.holes) totalHolePoints += hole.length;

        const holesFlat = new Float32Array(totalHolePoints * 2);
        const holeOffsets = new Uint32Array(holeCount + 1);
        let holeCursor = 0;
        for (let h = 0; h < holeCount; h++) {
            holeOffsets[h] = holeCursor;
            const hole = input.holes[h];
            for (let i = 0; i < hole.length; i++) {
                holesFlat[(holeCursor + i) * 2 + 0] = hole[i][0];
                holesFlat[(holeCursor + i) * 2 + 1] = hole[i][1];
            }
            holeCursor += hole.length;
        }
        holeOffsets[holeCount] = holeCursor;

        const totalPoints = (input.gridDensityU + 1) * (input.gridDensityV + 1);
        const maskBytes = totalPoints * 4;

        const boundaryBufferInfo = ensureBuffer(
            device,
            cache.boundaryBuffer,
            cache.boundaryCapacity,
            boundaryFlat.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "trim-grid-boundary"
        );
        cache.boundaryBuffer = boundaryBufferInfo.buffer;
        cache.boundaryCapacity = boundaryBufferInfo.capacity;

        const holesBufferInfo = ensureBuffer(
            device,
            cache.holesBuffer,
            cache.holesCapacity,
            Math.max(4, holesFlat.byteLength),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "trim-grid-holes"
        );
        cache.holesBuffer = holesBufferInfo.buffer;
        cache.holesCapacity = holesBufferInfo.capacity;

        const holeOffsetsInfo = ensureBuffer(
            device,
            cache.holeOffsetsBuffer,
            cache.holeOffsetsCapacity,
            Math.max(4, holeOffsets.byteLength),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "trim-grid-hole-offsets"
        );
        cache.holeOffsetsBuffer = holeOffsetsInfo.buffer;
        cache.holeOffsetsCapacity = holeOffsetsInfo.capacity;

        const maskBufferInfo = ensureBuffer(
            device,
            cache.maskBuffer,
            cache.maskCapacity,
            maskBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "trim-grid-mask"
        );
        cache.maskBuffer = maskBufferInfo.buffer;
        cache.maskCapacity = maskBufferInfo.capacity;

        const maskStagingInfo = ensureBuffer(
            device,
            cache.maskStagingBuffer,
            cache.maskStagingCapacity,
            maskBytes,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "trim-grid-mask-staging"
        );
        cache.maskStagingBuffer = maskStagingInfo.buffer;
        cache.maskStagingCapacity = maskStagingInfo.capacity;

        const paramsU32 = new Uint32Array(12);
        const paramsF32 = new Float32Array(paramsU32.buffer);
        paramsU32[0] = input.gridDensityU;
        paramsU32[1] = input.gridDensityV;
        paramsU32[2] = input.boundary.length;
        paramsU32[3] = holeCount;

        paramsF32[4] = input.uMin;
        paramsF32[5] = input.vMin;
        paramsF32[6] = input.du;
        paramsF32[7] = input.dv;

        paramsF32[8] = input.boundaryTolerance;
        paramsF32[9] = input.useNearBoundary ? 1 : 0;
        paramsF32[10] = 0;
        paramsF32[11] = 0;

        device.queue.writeBuffer(cache.boundaryBuffer, 0, boundaryFlat);
        if (holesFlat.byteLength > 0) {
            device.queue.writeBuffer(cache.holesBuffer, 0, holesFlat);
        }
        device.queue.writeBuffer(cache.holeOffsetsBuffer, 0, holeOffsets);
        device.queue.writeBuffer(cache.paramsBuffer!, 0, paramsU32);

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cache.boundaryBuffer } },
                { binding: 1, resource: { buffer: cache.holesBuffer } },
                { binding: 2, resource: { buffer: cache.holeOffsetsBuffer } },
                { binding: 3, resource: { buffer: cache.paramsBuffer! } },
                { binding: 4, resource: { buffer: cache.maskBuffer } },
            ],
        });

        const encoder = device.createCommandEncoder({ label: "trim-grid-classify-encoder" });
        const pass = encoder.beginComputePass({ label: "trim-grid-classify-pass" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(totalPoints / 256));
        pass.end();

        encoder.copyBufferToBuffer(cache.maskBuffer, 0, cache.maskStagingBuffer, 0, maskBytes);
        device.queue.submit([encoder.finish()]);

        await cache.maskStagingBuffer.mapAsync(GPUMapMode.READ, 0, maskBytes);
        const mapped = cache.maskStagingBuffer.getMappedRange(0, maskBytes);
        const mask = new Uint32Array(mapped.slice(0));
        cache.maskStagingBuffer.unmap();

        return mask;
    } catch (err) {
        if (!loggedGpuTrimGridError) {
            console.warn("[trim-grid-gpu] GPU classification failed, falling back to CPU.", err);
            loggedGpuTrimGridError = true;
        }
        return null;
    }
}
