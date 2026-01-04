/**
 * Optimized Ear Clipping Algorithm
 *
 * This version combines the best of both approaches:
 * 1. Single GPU dispatch (eliminates all CPU-GPU sync overhead)
 * 2. Parallel ear finding within the shader using workgroup parallelism
 * 3. Clips multiple independent ears per iteration using atomics
 *
 * Key insight: Use workgroupBarrier() instead of CPU sync for iteration control.
 * All threads cooperate within a single workgroup.
 */

import { getGPUDevice, normalizePoints } from "./lib";

const BYTE_SIZE = 4;

// Maximum vertices we can handle in one workgroup
// Limited by workgroup memory (16KB default) and thread count (256)
// 5 arrays * 256 * 4 bytes = 5120 bytes, well within 16KB
const MAX_VERTICES = 256;

const OPTIMIZED_SHADER = /* wgsl */`
// Optimized single-dispatch ear clipping with parallel processing

struct Point {
    x: f32,
    y: f32,
    z: f32,
    padding: f32
}

struct Uniforms {
    numVertices: u32,
    maxTriangles: u32,
}

// Input/Output buffers
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> outputIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> triangleCount: atomic<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// Workgroup shared memory for fast access
// Using 256 to stay within 16KB workgroup storage limit
var<workgroup> prevVertex: array<u32, 256>;
var<workgroup> nextVertex: array<u32, 256>;
var<workgroup> isActive: array<u32, 256>;
var<workgroup> isEar: array<u32, 256>;
var<workgroup> canClip: array<u32, 256>;
var<workgroup> activeCount: atomic<u32>;
var<workgroup> clippedThisRound: atomic<u32>;

fn cross2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    return ax * by - ay * bx;
}

fn isConvex(i: u32) -> bool {
    let p = prevVertex[i];
    let n = nextVertex[i];

    let A = points[p];
    let B = points[i];
    let C = points[n];

    let e1x = B.x - A.x;
    let e1y = B.y - A.y;
    let e2x = C.x - B.x;
    let e2y = C.y - B.y;

    return cross2d(e1x, e1y, e2x, e2y) > 1e-10;
}

fn pointInTriangle(A: Point, B: Point, C: Point, P: Point) -> bool {
    let c1 = cross2d(B.x - A.x, B.y - A.y, P.x - A.x, P.y - A.y);
    let c2 = cross2d(C.x - B.x, C.y - B.y, P.x - B.x, P.y - B.y);
    let c3 = cross2d(A.x - C.x, A.y - C.y, P.x - C.x, P.y - C.y);

    let eps = 1e-10;
    return (c1 >= -eps && c2 >= -eps && c3 >= -eps);
}

fn checkIsEar(i: u32, numVerts: u32) -> bool {
    if (isActive[i] == 0u) { return false; }
    if (!isConvex(i)) { return false; }

    let p = prevVertex[i];
    let n = nextVertex[i];
    let A = points[p];
    let B = points[i];
    let C = points[n];

    // Check if any active vertex is inside this triangle
    for (var j = 0u; j < numVerts; j++) {
        if (isActive[j] == 0u) { continue; }
        if (j == i || j == p || j == n) { continue; }

        let P = points[j];

        // Skip coincident points
        let sameAsA = abs(P.x - A.x) < 1e-9 && abs(P.y - A.y) < 1e-9;
        let sameAsB = abs(P.x - B.x) < 1e-9 && abs(P.y - B.y) < 1e-9;
        let sameAsC = abs(P.x - C.x) < 1e-9 && abs(P.y - C.y) < 1e-9;
        if (sameAsA || sameAsB || sameAsC) { continue; }

        if (pointInTriangle(A, B, C, P)) {
            return false;
        }
    }

    return true;
}

fn checkCanClip(i: u32) -> bool {
    if (isEar[i] == 0u) { return false; }

    let p = prevVertex[i];
    let n = nextVertex[i];

    // Can clip if this is the smallest ear index among neighbors
    if (isEar[p] == 1u && p < i) { return false; }
    if (isEar[n] == 1u && n < i) { return false; }

    let pp = prevVertex[p];
    if (isActive[pp] == 1u && isEar[pp] == 1u && pp < i) { return false; }

    let nn = nextVertex[n];
    if (isActive[nn] == 1u && isEar[nn] == 1u && nn < i) { return false; }

    return true;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let numVerts = uniforms.numVertices;

    // Initialize workgroup memory
    if (i < numVerts) {
        prevVertex[i] = select(numVerts - 1u, i - 1u, i > 0u);
        nextVertex[i] = select(0u, i + 1u, i < numVerts - 1u);
        isActive[i] = 1u;
        isEar[i] = 0u;
        canClip[i] = 0u;
    }
    if (i == 0u) {
        atomicStore(&activeCount, numVerts);
        atomicStore(&clippedThisRound, 0u);
    }

    workgroupBarrier();

    // Main loop: fixed iteration count (uniform control flow)
    // Most polygons complete in N-2 iterations, but we run all anyway
    // This ensures all threads hit all barriers uniformly
    for (var iter = 0u; iter < 254u; iter++) {
        // Reset per-iteration state
        if (i == 0u) {
            atomicStore(&clippedThisRound, 0u);
        }
        if (i < numVerts) {
            isEar[i] = 0u;
            canClip[i] = 0u;
        }

        workgroupBarrier();

        // Phase 1: Find all ears (parallel)
        // Only do work if we still have vertices to process
        if (i < numVerts && isActive[i] == 1u) {
            if (checkIsEar(i, numVerts)) {
                isEar[i] = 1u;
            }
        }

        workgroupBarrier();

        // Phase 2: Select independent ears (parallel)
        if (i < numVerts) {
            if (checkCanClip(i)) {
                canClip[i] = 1u;
            }
        }

        workgroupBarrier();

        // Phase 3: Clip selected ears (parallel with atomics)
        if (i < numVerts && canClip[i] == 1u) {
            let p = prevVertex[i];
            let n = nextVertex[i];

            // Write triangle
            let triIdx = atomicAdd(&triangleCount, 1u);
            outputIndices[triIdx * 3u + 0u] = p;
            outputIndices[triIdx * 3u + 1u] = i;
            outputIndices[triIdx * 3u + 2u] = n;

            // Update linked list
            nextVertex[p] = n;
            prevVertex[n] = p;

            // Mark inactive
            isActive[i] = 0u;

            atomicSub(&activeCount, 1u);
        }

        workgroupBarrier();
    }
}
`;

/**
 * Optimized ear clipping - single GPU dispatch with parallel processing.
 */
export async function earClippingOptimized(points: number[][]): Promise<number[][]> {
    if (points.length < 3) {
        throw new Error("Polygon must have at least 3 vertices");
    }

    if (points.length === 3) {
        return [[0, 1, 2]];
    }

    if (points.length > MAX_VERTICES) {
        throw new Error(`Polygon too large: ${points.length} vertices (max ${MAX_VERTICES})`);
    }

    const device = await getGPUDevice();
    const numVertices = points.length;
    const maxTriangles = numVertices - 2;

    // Prepare point data
    const normalizedPoints = normalizePoints(points);
    const pointsData = new Float32Array(normalizedPoints.flat());

    // Create buffers
    const pointsBuffer = device.createBuffer({
        size: pointsData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pointsBuffer, 0, pointsData);

    const outputIndicesBuffer = device.createBuffer({
        size: maxTriangles * 3 * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const triangleCountBuffer = device.createBuffer({
        size: BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(triangleCountBuffer, 0, new Uint32Array([0]));

    const uniformsData = new Uint32Array([numVertices, maxTriangles]);
    const uniformsBuffer = device.createBuffer({
        size: uniformsData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformsBuffer, 0, uniformsData);

    // Create pipeline
    const shaderModule = device.createShaderModule({
        label: "optimized ear clipping",
        code: OPTIMIZED_SHADER,
    });

    const pipeline = device.createComputePipeline({
        label: "optimized ear clipping pipeline",
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointsBuffer } },
            { binding: 1, resource: { buffer: outputIndicesBuffer } },
            { binding: 2, resource: { buffer: triangleCountBuffer } },
            { binding: 3, resource: { buffer: uniformsBuffer } },
        ],
    });

    // Single dispatch
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);  // Single workgroup handles all
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read back results
    const triangleCountStaging = device.createBuffer({
        size: BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const readEncoder = device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(triangleCountBuffer, 0, triangleCountStaging, 0, BYTE_SIZE);
    device.queue.submit([readEncoder.finish()]);

    await triangleCountStaging.mapAsync(GPUMapMode.READ);
    const numTriangles = new Uint32Array(triangleCountStaging.getMappedRange())[0];
    triangleCountStaging.unmap();
    triangleCountStaging.destroy();

    const indicesStaging = device.createBuffer({
        size: numTriangles * 3 * BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const readEncoder2 = device.createCommandEncoder();
    readEncoder2.copyBufferToBuffer(outputIndicesBuffer, 0, indicesStaging, 0, numTriangles * 3 * BYTE_SIZE);
    device.queue.submit([readEncoder2.finish()]);

    await indicesStaging.mapAsync(GPUMapMode.READ);
    const indicesData = new Uint32Array(indicesStaging.getMappedRange().slice(0));
    indicesStaging.unmap();
    indicesStaging.destroy();

    // Convert to triangle array
    const triangles: number[][] = [];
    for (let i = 0; i < numTriangles; i++) {
        triangles.push([
            indicesData[i * 3],
            indicesData[i * 3 + 1],
            indicesData[i * 3 + 2],
        ]);
    }

    // Cleanup
    pointsBuffer.destroy();
    outputIndicesBuffer.destroy();
    triangleCountBuffer.destroy();
    uniformsBuffer.destroy();

    return triangles;
}

export { earClippingOptimized as default };
