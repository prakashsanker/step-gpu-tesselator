/**
 * GPU-accelerated smooth vertex normal computation.
 *
 * Uses WebGPU compute shaders with atomic operations for parallel accumulation.
 * Computes angle-weighted vertex normals matching the CPU implementation.
 */

import { getGPUDevice } from "./lib";

type Vec3 = [number, number, number];

// Singleton for GPU resources
let device: GPUDevice | null = null;
let accumulatePipeline: GPUComputePipeline | null = null;
let normalizePipeline: GPUComputePipeline | null = null;

type BufferSlot =
    | "positionsBuffer"
    | "trianglesBuffer"
    | "accumBuffer"
    | "normalsBuffer"
    | "stagingBuffer";

type CapacitySlot =
    | "positionsCapacity"
    | "trianglesCapacity"
    | "accumCapacity"
    | "normalsCapacity"
    | "stagingCapacity";

const normalBufferCache: {
    positionsBuffer: GPUBuffer | null;
    trianglesBuffer: GPUBuffer | null;
    accumBuffer: GPUBuffer | null;
    normalsBuffer: GPUBuffer | null;
    stagingBuffer: GPUBuffer | null;
    accumParamsBuffer: GPUBuffer | null;
    normParamsBuffer: GPUBuffer | null;
    positionsCapacity: number;
    trianglesCapacity: number;
    accumCapacity: number;
    normalsCapacity: number;
    stagingCapacity: number;
} = {
    positionsBuffer: null,
    trianglesBuffer: null,
    accumBuffer: null,
    normalsBuffer: null,
    stagingBuffer: null,
    accumParamsBuffer: null,
    normParamsBuffer: null,
    positionsCapacity: 0,
    trianglesCapacity: 0,
    accumCapacity: 0,
    normalsCapacity: 0,
    stagingCapacity: 0,
};

function roundCapacity(size: number): number {
    const alignment = 256;
    const rounded = Math.ceil(size / alignment) * alignment;
    return Math.max(alignment, rounded);
}

function destroyCachedBuffers() {
    normalBufferCache.positionsBuffer?.destroy();
    normalBufferCache.trianglesBuffer?.destroy();
    normalBufferCache.accumBuffer?.destroy();
    normalBufferCache.normalsBuffer?.destroy();
    normalBufferCache.stagingBuffer?.destroy();
    normalBufferCache.accumParamsBuffer?.destroy();
    normalBufferCache.normParamsBuffer?.destroy();

    normalBufferCache.positionsBuffer = null;
    normalBufferCache.trianglesBuffer = null;
    normalBufferCache.accumBuffer = null;
    normalBufferCache.normalsBuffer = null;
    normalBufferCache.stagingBuffer = null;
    normalBufferCache.accumParamsBuffer = null;
    normalBufferCache.normParamsBuffer = null;
    normalBufferCache.positionsCapacity = 0;
    normalBufferCache.trianglesCapacity = 0;
    normalBufferCache.accumCapacity = 0;
    normalBufferCache.normalsCapacity = 0;
    normalBufferCache.stagingCapacity = 0;
}

function ensureCachedBuffer(
    dev: GPUDevice,
    slot: BufferSlot,
    capacitySlot: CapacitySlot,
    size: number,
    usage: GPUBufferUsageFlags,
    label: string
): GPUBuffer {
    const needed = roundCapacity(size);
    const current = normalBufferCache[slot];
    const currentCapacity = normalBufferCache[capacitySlot];
    if (current && currentCapacity >= needed) {
        return current;
    }

    current?.destroy();
    const next = dev.createBuffer({
        label,
        size: needed,
        usage,
    });
    normalBufferCache[slot] = next;
    normalBufferCache[capacitySlot] = needed;
    return next;
}

function ensureParamBuffers(dev: GPUDevice): { accum: GPUBuffer; norm: GPUBuffer } {
    if (!normalBufferCache.accumParamsBuffer) {
        normalBufferCache.accumParamsBuffer = dev.createBuffer({
            label: "accum-params",
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }
    if (!normalBufferCache.normParamsBuffer) {
        normalBufferCache.normParamsBuffer = dev.createBuffer({
            label: "norm-params",
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }
    return {
        accum: normalBufferCache.accumParamsBuffer,
        norm: normalBufferCache.normParamsBuffer,
    };
}

// Scale factor for integer atomics (floats scaled to i32)
const ATOMIC_SCALE = 1e6;

const accumulateShader = /* wgsl */ `
struct Params {
    triangleCount: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> triangles: array<u32>;
@group(0) @binding(2) var<storage, read_write> normalAccum: array<atomic<i32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn getPosition(idx: u32) -> vec3f {
    return vec3f(
        positions[idx * 3u + 0u],
        positions[idx * 3u + 1u],
        positions[idx * 3u + 2u]
    );
}

fn angleBetween(a: vec3f, b: vec3f) -> f32 {
    let lenA = length(a);
    let lenB = length(b);
    if (lenA < 1e-10 || lenB < 1e-10) {
        return 0.0;
    }
    let cosAngle = dot(a, b) / (lenA * lenB);
    return acos(clamp(cosAngle, -1.0, 1.0));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let triIdx = gid.x;
    if (triIdx >= params.triangleCount) {
        return;
    }

    let i0 = triangles[triIdx * 3u + 0u];
    let i1 = triangles[triIdx * 3u + 1u];
    let i2 = triangles[triIdx * 3u + 2u];

    let p0 = getPosition(i0);
    let p1 = getPosition(i1);
    let p2 = getPosition(i2);

    // Edge vectors
    let e01 = p1 - p0;
    let e02 = p2 - p0;
    let e12 = p2 - p1;

    // Face normal (cross product)
    let faceNormal = cross(e01, e02);

    // Compute angles at each vertex
    let angle0 = angleBetween(e01, e02);
    let angle1 = angleBetween(-e01, e12);
    let angle2 = angleBetween(-e02, -e12);

    let scale = params.scale;

    // Accumulate weighted normals using atomics
    // Vertex 0
    atomicAdd(&normalAccum[i0 * 3u + 0u], i32(faceNormal.x * angle0 * scale));
    atomicAdd(&normalAccum[i0 * 3u + 1u], i32(faceNormal.y * angle0 * scale));
    atomicAdd(&normalAccum[i0 * 3u + 2u], i32(faceNormal.z * angle0 * scale));

    // Vertex 1
    atomicAdd(&normalAccum[i1 * 3u + 0u], i32(faceNormal.x * angle1 * scale));
    atomicAdd(&normalAccum[i1 * 3u + 1u], i32(faceNormal.y * angle1 * scale));
    atomicAdd(&normalAccum[i1 * 3u + 2u], i32(faceNormal.z * angle1 * scale));

    // Vertex 2
    atomicAdd(&normalAccum[i2 * 3u + 0u], i32(faceNormal.x * angle2 * scale));
    atomicAdd(&normalAccum[i2 * 3u + 1u], i32(faceNormal.y * angle2 * scale));
    atomicAdd(&normalAccum[i2 * 3u + 2u], i32(faceNormal.z * angle2 * scale));
}
`;

const normalizeShader = /* wgsl */ `
struct Params {
    vertexCount: u32,
    invScale: f32,
}

@group(0) @binding(0) var<storage, read> normalAccum: array<i32>;
@group(0) @binding(1) var<storage, read_write> normals: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let vertIdx = gid.x;
    if (vertIdx >= params.vertexCount) {
        return;
    }

    let invScale = params.invScale;
    var nx = f32(normalAccum[vertIdx * 3u + 0u]) * invScale;
    var ny = f32(normalAccum[vertIdx * 3u + 1u]) * invScale;
    var nz = f32(normalAccum[vertIdx * 3u + 2u]) * invScale;

    let len = sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-10) {
        nx = nx / len;
        ny = ny / len;
        nz = nz / len;
    }

    normals[vertIdx * 3u + 0u] = nx;
    normals[vertIdx * 3u + 1u] = ny;
    normals[vertIdx * 3u + 2u] = nz;
}
`;

async function initPipelines(): Promise<GPUDevice> {
    const currentDevice = await getGPUDevice();

    // Invalidate cached pipelines if device changed
    if (device !== currentDevice) {
        destroyCachedBuffers();
        accumulatePipeline = null;
        normalizePipeline = null;
        device = currentDevice;
    }

    if (accumulatePipeline && normalizePipeline) {
        return device;
    }

    // Create accumulate pipeline
    const accumulateModule = device.createShaderModule({
        label: "smooth-normals-accumulate",
        code: accumulateShader,
    });

    accumulatePipeline = device.createComputePipeline({
        label: "smooth-normals-accumulate-pipeline",
        layout: "auto",
        compute: {
            module: accumulateModule,
            entryPoint: "main",
        },
    });

    // Create normalize pipeline
    const normalizeModule = device.createShaderModule({
        label: "smooth-normals-normalize",
        code: normalizeShader,
    });

    normalizePipeline = device.createComputePipeline({
        label: "smooth-normals-normalize-pipeline",
        layout: "auto",
        compute: {
            module: normalizeModule,
            entryPoint: "main",
        },
    });

    return device;
}

function flattenPositions(positions: Vec3[]): Float32Array {
    const positionsFlat = new Float32Array(positions.length * 3);
    for (let i = 0; i < positions.length; i++) {
        positionsFlat[i * 3 + 0] = positions[i][0];
        positionsFlat[i * 3 + 1] = positions[i][1];
        positionsFlat[i * 3 + 2] = positions[i][2];
    }
    return positionsFlat;
}

function flattenTriangles(triangles: [number, number, number][]): Uint32Array {
    const trianglesFlat = new Uint32Array(triangles.length * 3);
    for (let i = 0; i < triangles.length; i++) {
        trianglesFlat[i * 3 + 0] = triangles[i][0];
        trianglesFlat[i * 3 + 1] = triangles[i][1];
        trianglesFlat[i * 3 + 2] = triangles[i][2];
    }
    return trianglesFlat;
}

/**
 * Compute smooth normals on the GPU for flat buffers.
 * Returns a flat Float32Array [nx0, ny0, nz0, nx1, ...].
 */
export async function computeSmoothNormalsGPUFlat(
    positionsFlat: Float32Array,
    trianglesFlat: Uint32Array
): Promise<Float32Array> {
    const dev = await initPipelines();

    const vertexCount = Math.floor(positionsFlat.length / 3);
    const triangleCount = Math.floor(trianglesFlat.length / 3);

    if (triangleCount === 0 || vertexCount === 0) {
        const emptyNormals = new Float32Array(vertexCount * 3);
        for (let i = 0; i < vertexCount; i++) {
            emptyNormals[i * 3 + 2] = 1;
        }
        return emptyNormals;
    }

    // Reuse buffers between calls; this avoids repeated GPU allocation churn.
    const positionsBuffer = ensureCachedBuffer(
        dev,
        "positionsBuffer",
        "positionsCapacity",
        positionsFlat.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "positions-buffer"
    );
    dev.queue.writeBuffer(positionsBuffer, 0, positionsFlat);

    const trianglesBuffer = ensureCachedBuffer(
        dev,
        "trianglesBuffer",
        "trianglesCapacity",
        trianglesFlat.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "triangles-buffer"
    );
    dev.queue.writeBuffer(trianglesBuffer, 0, trianglesFlat);

    const accumBufferSize = vertexCount * 3 * 4;
    const accumBuffer = ensureCachedBuffer(
        dev,
        "accumBuffer",
        "accumCapacity",
        accumBufferSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "normal-accum-buffer"
    );

    const normalsBufferSize = vertexCount * 3 * 4;
    const normalsBuffer = ensureCachedBuffer(
        dev,
        "normalsBuffer",
        "normalsCapacity",
        normalsBufferSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        "normals-buffer"
    );

    const stagingBuffer = ensureCachedBuffer(
        dev,
        "stagingBuffer",
        "stagingCapacity",
        normalsBufferSize,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        "normals-staging"
    );

    // Accumulate params
    const accumParamsData = new ArrayBuffer(8);
    const accumParamsView = new DataView(accumParamsData);
    accumParamsView.setUint32(0, triangleCount, true);
    accumParamsView.setFloat32(4, ATOMIC_SCALE, true);

    const { accum: accumParamsBuffer, norm: normParamsBuffer } = ensureParamBuffers(dev);
    dev.queue.writeBuffer(accumParamsBuffer, 0, accumParamsData);

    // Normalize params
    const normParamsData = new ArrayBuffer(8);
    const normParamsView = new DataView(normParamsData);
    normParamsView.setUint32(0, vertexCount, true);
    normParamsView.setFloat32(4, 1.0 / ATOMIC_SCALE, true);

    dev.queue.writeBuffer(normParamsBuffer, 0, normParamsData);

    // Create bind groups
    const accumBindGroup = dev.createBindGroup({
        label: "accum-bind-group",
        layout: accumulatePipeline!.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: positionsBuffer } },
            { binding: 1, resource: { buffer: trianglesBuffer } },
            { binding: 2, resource: { buffer: accumBuffer } },
            { binding: 3, resource: { buffer: accumParamsBuffer } },
        ],
    });

    const normBindGroup = dev.createBindGroup({
        label: "norm-bind-group",
        layout: normalizePipeline!.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: accumBuffer } },
            { binding: 1, resource: { buffer: normalsBuffer } },
            { binding: 2, resource: { buffer: normParamsBuffer } },
        ],
    });

    // Dispatch compute passes
    const encoder = dev.createCommandEncoder({ label: "smooth-normals-encoder" });

    // Pass 1: Accumulate weighted normals
    encoder.clearBuffer(accumBuffer, 0, accumBufferSize);
    const accumPass = encoder.beginComputePass({ label: "accumulate-pass" });
    accumPass.setPipeline(accumulatePipeline!);
    accumPass.setBindGroup(0, accumBindGroup);
    accumPass.dispatchWorkgroups(Math.ceil(triangleCount / 256));
    accumPass.end();

    // Pass 2: Normalize
    const normPass = encoder.beginComputePass({ label: "normalize-pass" });
    normPass.setPipeline(normalizePipeline!);
    normPass.setBindGroup(0, normBindGroup);
    normPass.dispatchWorkgroups(Math.ceil(vertexCount / 256));
    normPass.end();

    // Copy to staging
    encoder.copyBufferToBuffer(normalsBuffer, 0, stagingBuffer, 0, normalsBufferSize);

    dev.queue.submit([encoder.finish()]);

    // Read back results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const mapped = stagingBuffer.getMappedRange(0, normalsBufferSize);
    const resultData = new Float32Array(mapped.slice(0));
    stagingBuffer.unmap();

    return resultData;
}

/**
 * Compute smooth vertex normals on the GPU.
 * Uses angle-weighted averaging matching the CPU implementation.
 */
export async function computeSmoothNormalsGPU(
    positions: Vec3[],
    triangles: [number, number, number][]
): Promise<Vec3[]> {
    const positionsFlat = flattenPositions(positions);
    const trianglesFlat = flattenTriangles(triangles);
    const normalsFlat = await computeSmoothNormalsGPUFlat(positionsFlat, trianglesFlat);

    const normals: Vec3[] = [];
    for (let i = 0; i < positions.length; i++) {
        normals.push([
            normalsFlat[i * 3 + 0],
            normalsFlat[i * 3 + 1],
            normalsFlat[i * 3 + 2],
        ]);
    }
    return normals;
}
