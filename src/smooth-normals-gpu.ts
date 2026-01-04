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
    if (device && accumulatePipeline && normalizePipeline) {
        return device;
    }

    device = await getGPUDevice();

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

/**
 * Compute smooth vertex normals on the GPU.
 * Uses angle-weighted averaging matching the CPU implementation.
 */
export async function computeSmoothNormalsGPU(
    positions: Vec3[],
    triangles: [number, number, number][]
): Promise<Vec3[]> {
    const dev = await initPipelines();

    const vertexCount = positions.length;
    const triangleCount = triangles.length;

    if (triangleCount === 0 || vertexCount === 0) {
        return positions.map(() => [0, 0, 1]);
    }

    // Flatten positions to Float32Array
    const positionsFlat = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i++) {
        positionsFlat[i * 3 + 0] = positions[i][0];
        positionsFlat[i * 3 + 1] = positions[i][1];
        positionsFlat[i * 3 + 2] = positions[i][2];
    }

    // Flatten triangles to Uint32Array
    const trianglesFlat = new Uint32Array(triangleCount * 3);
    for (let i = 0; i < triangleCount; i++) {
        trianglesFlat[i * 3 + 0] = triangles[i][0];
        trianglesFlat[i * 3 + 1] = triangles[i][1];
        trianglesFlat[i * 3 + 2] = triangles[i][2];
    }

    // Create buffers
    const positionsBuffer = dev.createBuffer({
        label: "positions-buffer",
        size: positionsFlat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(positionsBuffer, 0, positionsFlat);

    const trianglesBuffer = dev.createBuffer({
        label: "triangles-buffer",
        size: trianglesFlat.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(trianglesBuffer, 0, trianglesFlat);

    // Accumulator buffer (i32 atomics, initialized to 0)
    const accumBuffer = dev.createBuffer({
        label: "normal-accum-buffer",
        size: vertexCount * 3 * 4, // i32 per component
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Zero-initialize
    dev.queue.writeBuffer(accumBuffer, 0, new Int32Array(vertexCount * 3));

    // Output normals buffer
    const normalsBuffer = dev.createBuffer({
        label: "normals-buffer",
        size: vertexCount * 3 * 4, // f32 per component
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Staging buffer for readback
    const stagingBuffer = dev.createBuffer({
        label: "normals-staging",
        size: vertexCount * 3 * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Accumulate params
    const accumParamsData = new ArrayBuffer(8);
    const accumParamsView = new DataView(accumParamsData);
    accumParamsView.setUint32(0, triangleCount, true);
    accumParamsView.setFloat32(4, ATOMIC_SCALE, true);

    const accumParamsBuffer = dev.createBuffer({
        label: "accum-params",
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(accumParamsBuffer, 0, accumParamsData);

    // Normalize params
    const normParamsData = new ArrayBuffer(8);
    const normParamsView = new DataView(normParamsData);
    normParamsView.setUint32(0, vertexCount, true);
    normParamsView.setFloat32(4, 1.0 / ATOMIC_SCALE, true);

    const normParamsBuffer = dev.createBuffer({
        label: "norm-params",
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
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
    encoder.copyBufferToBuffer(normalsBuffer, 0, stagingBuffer, 0, vertexCount * 3 * 4);

    dev.queue.submit([encoder.finish()]);

    // Read back results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    // Convert to Vec3 array
    const normals: Vec3[] = [];
    for (let i = 0; i < vertexCount; i++) {
        normals.push([
            resultData[i * 3 + 0],
            resultData[i * 3 + 1],
            resultData[i * 3 + 2],
        ]);
    }

    // Cleanup buffers
    positionsBuffer.destroy();
    trianglesBuffer.destroy();
    accumBuffer.destroy();
    normalsBuffer.destroy();
    stagingBuffer.destroy();
    accumParamsBuffer.destroy();
    normParamsBuffer.destroy();

    return normals;
}
