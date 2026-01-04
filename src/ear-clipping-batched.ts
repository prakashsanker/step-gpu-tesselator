/**
 * Batched Ear Clipping Algorithm
 *
 * Key optimization: Process ALL polygons in a single GPU dispatch.
 * This eliminates per-polygon overhead:
 * - Shader compilation (done once)
 * - Pipeline creation (done once)
 * - Buffer creation (reused)
 * - CPU-GPU round trips (one per batch instead of per polygon)
 */

import { getGPUDevice, normalizePoints } from "./lib";

const BYTE_SIZE = 4;
const MAX_VERTICES_PER_POLYGON = 256;

// Cached GPU resources
let cachedDevice: GPUDevice | null = null;
let cachedPipeline: GPUComputePipeline | null = null;
let cachedShaderModule: GPUShaderModule | null = null;

const BATCHED_SHADER = /* wgsl */`
// Batched ear clipping - process multiple polygons in parallel

struct Point {
    x: f32,
    y: f32,
    z: f32,
    padding: f32
}

struct PolygonInfo {
    startVertex: u32,
    numVertices: u32,
    outputOffset: u32,
    maxTriangles: u32,  // Expected triangles (numVerts - 2)
}

// Input buffers
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read> polygonInfos: array<PolygonInfo>;
@group(0) @binding(2) var<uniform> numPolygons: u32;

// Output buffers
@group(0) @binding(3) var<storage, read_write> outputIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> triangleCounts: array<atomic<u32>>;

// Workgroup shared memory (per polygon)
var<workgroup> prevVertex: array<u32, 256>;
var<workgroup> nextVertex: array<u32, 256>;
var<workgroup> isActive: array<u32, 256>;
var<workgroup> isEar: array<u32, 256>;
var<workgroup> canClip: array<u32, 256>;

fn cross2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    return ax * by - ay * bx;
}

fn isConvex(i: u32, startVertex: u32) -> bool {
    let p = prevVertex[i];
    let n = nextVertex[i];

    let A = points[startVertex + p];
    let B = points[startVertex + i];
    let C = points[startVertex + n];

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

fn checkIsEar(i: u32, numVerts: u32, startVertex: u32) -> bool {
    if (isActive[i] == 0u) { return false; }
    if (!isConvex(i, startVertex)) { return false; }

    let p = prevVertex[i];
    let n = nextVertex[i];
    let A = points[startVertex + p];
    let B = points[startVertex + i];
    let C = points[startVertex + n];

    for (var j = 0u; j < numVerts; j++) {
        if (isActive[j] == 0u) { continue; }
        if (j == i || j == p || j == n) { continue; }

        let P = points[startVertex + j];

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

    if (isEar[p] == 1u && p < i) { return false; }
    if (isEar[n] == 1u && n < i) { return false; }

    let pp = prevVertex[p];
    if (isActive[pp] == 1u && isEar[pp] == 1u && pp < i) { return false; }

    let nn = nextVertex[n];
    if (isActive[nn] == 1u && isEar[nn] == 1u && nn < i) { return false; }

    return true;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let polygonIdx = wid.x;
    if (polygonIdx >= numPolygons) { return; }

    let info = polygonInfos[polygonIdx];
    let numVerts = info.numVertices;
    let startVertex = info.startVertex;
    let outputOffset = info.outputOffset;
    let i = lid.x;

    // Initialize ALL workgroup memory (not just vertices)
    // Uninitialized entries could cause issues in edge cases
    prevVertex[i] = 0u;
    nextVertex[i] = 0u;
    isActive[i] = 0u;
    isEar[i] = 0u;
    canClip[i] = 0u;

    workgroupBarrier();

    // Now initialize actual vertices
    if (i < numVerts) {
        prevVertex[i] = select(numVerts - 1u, i - 1u, i > 0u);
        nextVertex[i] = select(0u, i + 1u, i < numVerts - 1u);
        isActive[i] = 1u;
    }

    workgroupBarrier();

    // Main loop
    for (var iter = 0u; iter < 254u; iter++) {
        if (i < numVerts) {
            isEar[i] = 0u;
            canClip[i] = 0u;
        }

        workgroupBarrier();

        // Phase 1: Find all ears
        if (i < numVerts && isActive[i] == 1u) {
            if (checkIsEar(i, numVerts, startVertex)) {
                isEar[i] = 1u;
            }
        }

        workgroupBarrier();

        // Phase 2: Select independent ears
        if (i < numVerts) {
            if (checkCanClip(i)) {
                canClip[i] = 1u;
            }
        }

        workgroupBarrier();

        // Phase 3: Clip selected ears
        // Add redundant isActive check to prevent any race conditions
        if (i < numVerts && canClip[i] == 1u && isActive[i] == 1u) {
            let p = prevVertex[i];
            let n = nextVertex[i];

            // Atomically get triangle slot, but check if we've exceeded max
            let triIdx = atomicAdd(&triangleCounts[polygonIdx], 1u);

            // CRITICAL: Only write if we haven't exceeded expected triangles
            // This prevents buffer overflow into next polygon's space
            if (triIdx < info.maxTriangles) {
                let outIdx = outputOffset + triIdx * 3u;
                outputIndices[outIdx + 0u] = p;
                outputIndices[outIdx + 1u] = i;
                outputIndices[outIdx + 2u] = n;
            }

            // Update linked list (even if we didn't write, to maintain consistency)
            nextVertex[p] = n;
            prevVertex[n] = p;

            // Mark inactive
            isActive[i] = 0u;
        }

        workgroupBarrier();
    }
}
`;

async function ensurePipeline(): Promise<{ device: GPUDevice; pipeline: GPUComputePipeline }> {
    const device = await getGPUDevice();

    if (cachedDevice !== device) {
        // Device changed, need to recreate everything
        cachedDevice = device;
        cachedShaderModule = null;
        cachedPipeline = null;
    }

    if (!cachedShaderModule) {
        cachedShaderModule = device.createShaderModule({
            label: "batched ear clipping shader",
            code: BATCHED_SHADER,
        });
    }

    if (!cachedPipeline) {
        cachedPipeline = device.createComputePipeline({
            label: "batched ear clipping pipeline",
            layout: "auto",
            compute: { module: cachedShaderModule, entryPoint: "main" },
        });
    }

    return { device, pipeline: cachedPipeline };
}

export interface BatchedPolygon {
    points: number[][];  // 2D or 3D points
}

export interface BatchedResult {
    triangles: number[][][];  // Array of triangle arrays per polygon
    totalTime: number;
}

/**
 * Process multiple polygons in a single GPU batch.
 * This is much more efficient than processing one at a time.
 */
export async function earClippingBatched(polygons: BatchedPolygon[]): Promise<BatchedResult> {
    const startTime = performance.now();

    if (polygons.length === 0) {
        return { triangles: [], totalTime: 0 };
    }

    // Filter out trivial cases and validate
    const validPolygons: { points: number[][]; originalIndex: number }[] = [];
    const trivialResults: Map<number, number[][]> = new Map();

    for (let i = 0; i < polygons.length; i++) {
        const pts = polygons[i].points;
        if (pts.length < 3) {
            trivialResults.set(i, []);
        } else if (pts.length === 3) {
            trivialResults.set(i, [[0, 1, 2]]);
        } else if (pts.length > MAX_VERTICES_PER_POLYGON) {
            throw new Error(`Polygon ${i} has ${pts.length} vertices (max ${MAX_VERTICES_PER_POLYGON})`);
        } else {
            validPolygons.push({ points: pts, originalIndex: i });
        }
    }

    // If all trivial, return early
    if (validPolygons.length === 0) {
        const results: number[][][] = [];
        for (let i = 0; i < polygons.length; i++) {
            results.push(trivialResults.get(i) || []);
        }
        return { triangles: results, totalTime: performance.now() - startTime };
    }

    const { device, pipeline } = await ensurePipeline();

    // Prepare batched data
    let totalVertices = 0;
    let totalMaxTriangles = 0;
    const polygonInfos: number[] = [];

    for (let polyIdx = 0; polyIdx < validPolygons.length; polyIdx++) {
        const poly = validPolygons[polyIdx];
        const numVerts = poly.points.length;
        const maxTris = numVerts - 2;
        const outputOffset = totalMaxTriangles * 3;
        // Pass maxTriangles so shader can prevent overflow
        polygonInfos.push(totalVertices, numVerts, outputOffset, maxTris);
        totalVertices += numVerts;
        totalMaxTriangles += maxTris;
    }

    // Flatten all points into one buffer
    const allPoints: number[] = [];
    for (const poly of validPolygons) {
        const normalized = normalizePoints(poly.points);
        for (const p of normalized) {
            allPoints.push(...p);
        }
    }

    // Create buffers
    const pointsBuffer = device.createBuffer({
        size: allPoints.length * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pointsBuffer, 0, new Float32Array(allPoints));

    const polygonInfosBuffer = device.createBuffer({
        size: polygonInfos.length * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(polygonInfosBuffer, 0, new Uint32Array(polygonInfos));

    const numPolygonsBuffer = device.createBuffer({
        size: BYTE_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(numPolygonsBuffer, 0, new Uint32Array([validPolygons.length]));

    const outputIndicesBuffer = device.createBuffer({
        size: totalMaxTriangles * 3 * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // Initialize output buffer to zeros (prevents garbage from uninitialized memory)
    device.queue.writeBuffer(outputIndicesBuffer, 0, new Uint32Array(totalMaxTriangles * 3));

    const triangleCountsBuffer = device.createBuffer({
        size: validPolygons.length * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(triangleCountsBuffer, 0, new Uint32Array(validPolygons.length));

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointsBuffer } },
            { binding: 1, resource: { buffer: polygonInfosBuffer } },
            { binding: 2, resource: { buffer: numPolygonsBuffer } },
            { binding: 3, resource: { buffer: outputIndicesBuffer } },
            { binding: 4, resource: { buffer: triangleCountsBuffer } },
        ],
    });

    // Single dispatch for ALL polygons
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(validPolygons.length);  // One workgroup per polygon
    pass.end();

    // Copy results to staging buffers
    const triangleCountsStaging = device.createBuffer({
        size: validPolygons.length * BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(
        triangleCountsBuffer, 0,
        triangleCountsStaging, 0,
        validPolygons.length * BYTE_SIZE
    );

    const indicesStaging = device.createBuffer({
        size: totalMaxTriangles * 3 * BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(
        outputIndicesBuffer, 0,
        indicesStaging, 0,
        totalMaxTriangles * 3 * BYTE_SIZE
    );

    device.queue.submit([encoder.finish()]);

    // Read results
    await triangleCountsStaging.mapAsync(GPUMapMode.READ);
    const triangleCounts = new Uint32Array(triangleCountsStaging.getMappedRange().slice(0));
    triangleCountsStaging.unmap();

    await indicesStaging.mapAsync(GPUMapMode.READ);
    const allIndices = new Uint32Array(indicesStaging.getMappedRange().slice(0));
    indicesStaging.unmap();


    // Parse results - no more overflow workaround needed, shader prevents it
    const gpuResults: Map<number, number[][]> = new Map();
    for (let p = 0; p < validPolygons.length; p++) {
        const numTris = triangleCounts[p];
        const outputOffset = polygonInfos[p * 4 + 2];
        const numVerts = validPolygons[p].points.length;
        const expectedTris = numVerts - 2;
        const triangles: number[][] = [];

        // Use actual triangle count (shader limits to maxTriangles internally)
        const actualTris = Math.min(numTris, expectedTris);

        for (let t = 0; t < actualTris; t++) {
            const idx = outputOffset + t * 3;
            const i0 = allIndices[idx];
            const i1 = allIndices[idx + 1];
            const i2 = allIndices[idx + 2];

            // Validate indices (should always pass now, but keep as safety)
            if (i0 < numVerts && i1 < numVerts && i2 < numVerts) {
                triangles.push([i0, i1, i2]);
            }
        }

        gpuResults.set(validPolygons[p].originalIndex, triangles);
    }

    // Combine results in original order
    const results: number[][][] = [];
    for (let i = 0; i < polygons.length; i++) {
        if (trivialResults.has(i)) {
            results.push(trivialResults.get(i)!);
        } else {
            results.push(gpuResults.get(i) || []);
        }
    }

    // Cleanup
    pointsBuffer.destroy();
    polygonInfosBuffer.destroy();
    numPolygonsBuffer.destroy();
    outputIndicesBuffer.destroy();
    triangleCountsBuffer.destroy();
    triangleCountsStaging.destroy();
    indicesStaging.destroy();

    return {
        triangles: results,
        totalTime: performance.now() - startTime,
    };
}
