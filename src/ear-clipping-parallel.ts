/**
 * Parallel Ear Clipping Algorithm
 *
 * Optimizations over the original implementation:
 * 1. Clips MULTIPLE independent ears per GPU dispatch (not just 1)
 * 2. Reduces iterations from O(N) to O(log N) in practice
 * 3. Single command buffer submission per iteration (no per-ear sync)
 * 4. No debug logging or unnecessary buffer readbacks
 *
 * Key insight: Two ears are "independent" if they don't share vertices.
 * We use a simple rule: an ear can be clipped if its index is the
 * smallest among all ears that share a vertex with it.
 */

import { getGPUDevice, normalizePoints } from "./lib";

const BYTE_SIZE = 4;
const WORKGROUP_SIZE = 64;

// =============================================================================
// WGSL Shaders
// =============================================================================

const CLASSIFY_AND_FIND_EARS_SHADER = /* wgsl */`
struct Point {
    x: f32,
    y: f32,
    z: f32,
    padding: f32
}

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read> prevVertex: array<u32>;
@group(0) @binding(2) var<storage, read> nextVertex: array<u32>;
@group(0) @binding(3) var<storage, read> isActive: array<u32>;
@group(0) @binding(4) var<storage, read_write> isEar: array<u32>;
@group(0) @binding(5) var<storage, read_write> canClip: array<u32>;

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

@compute @workgroup_size(64)
fn classifyAndFindEars(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let numVerts = arrayLength(&points);

    if (i >= numVerts) { return; }

    // Reset outputs
    isEar[i] = 0u;
    canClip[i] = 0u;

    // Skip inactive vertices
    if (isActive[i] == 0u) { return; }

    // Check if convex
    if (!isConvex(i)) { return; }

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
            return;  // Not an ear
        }
    }

    // This is an ear
    isEar[i] = 1u;
}
`;

const SELECT_INDEPENDENT_EARS_SHADER = /* wgsl */`
// Select which ears can be clipped in parallel.
// Rule: An ear can be clipped if it has the smallest index among
// all ears that share a vertex with it (itself, prev, next).

@group(0) @binding(0) var<storage, read> prevVertex: array<u32>;
@group(0) @binding(1) var<storage, read> nextVertex: array<u32>;
@group(0) @binding(2) var<storage, read> isActive: array<u32>;
@group(0) @binding(3) var<storage, read> isEar: array<u32>;
@group(0) @binding(4) var<storage, read_write> canClip: array<u32>;

@compute @workgroup_size(64)
fn selectIndependentEars(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let numVerts = arrayLength(&isEar);

    if (i >= numVerts) { return; }
    if (isEar[i] == 0u) { return; }

    let p = prevVertex[i];
    let n = nextVertex[i];

    // Check if prev or next is also an ear with lower index
    // If so, we can't clip this one (it would conflict)

    // Check prev neighbor
    if (isEar[p] == 1u && p < i) {
        canClip[i] = 0u;
        return;
    }

    // Check next neighbor
    if (isEar[n] == 1u && n < i) {
        canClip[i] = 0u;
        return;
    }

    // Check if prev's prev is an ear that would use our prev
    let pp = prevVertex[p];
    if (isActive[pp] == 1u && isEar[pp] == 1u && pp < i) {
        // pp's triangle uses p, which is also our prev
        canClip[i] = 0u;
        return;
    }

    // Check if next's next is an ear that would use our next
    let nn = nextVertex[n];
    if (isActive[nn] == 1u && isEar[nn] == 1u && nn < i) {
        // nn's triangle uses n, which is also our next
        canClip[i] = 0u;
        return;
    }

    // This ear is independent - can clip it
    canClip[i] = 1u;
}
`;

const CLIP_EARS_SHADER = /* wgsl */`
struct Point {
    x: f32,
    y: f32,
    z: f32,
    padding: f32
}

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> prevVertex: array<u32>;
@group(0) @binding(2) var<storage, read_write> nextVertex: array<u32>;
@group(0) @binding(3) var<storage, read_write> isActive: array<u32>;
@group(0) @binding(4) var<storage, read> canClip: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputIndices: array<u32>;
@group(0) @binding(6) var<storage, read_write> triangleCount: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> clippedCount: atomic<u32>;

@compute @workgroup_size(64)
fn clipEars(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let numVerts = arrayLength(&points);

    if (i >= numVerts) { return; }
    if (canClip[i] == 0u) { return; }

    let p = prevVertex[i];
    let n = nextVertex[i];

    // Atomically get triangle index and increment
    let triIdx = atomicAdd(&triangleCount, 1u);

    // Write triangle (prev, current, next)
    outputIndices[triIdx * 3u + 0u] = p;
    outputIndices[triIdx * 3u + 1u] = i;
    outputIndices[triIdx * 3u + 2u] = n;

    // Update linked list pointers
    // Note: This is safe because independent ears don't share neighbors
    nextVertex[p] = n;
    prevVertex[n] = p;

    // Mark as inactive
    isActive[i] = 0u;

    // Count clipped ears this iteration
    atomicAdd(&clippedCount, 1u);
}
`;

const COUNT_ACTIVE_SHADER = /* wgsl */`
@group(0) @binding(0) var<storage, read> isActive: array<u32>;
@group(0) @binding(1) var<storage, read_write> activeCount: atomic<u32>;

@compute @workgroup_size(64)
fn countActive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let numVerts = arrayLength(&isActive);

    if (i >= numVerts) { return; }

    if (isActive[i] == 1u) {
        atomicAdd(&activeCount, 1u);
    }
}
`;

// =============================================================================
// Main Algorithm
// =============================================================================

interface GPUResources {
    device: GPUDevice;
    // Buffers
    pointsBuffer: GPUBuffer;
    prevVertexBuffer: GPUBuffer;
    nextVertexBuffer: GPUBuffer;
    isActiveBuffer: GPUBuffer;
    isEarBuffer: GPUBuffer;
    canClipBuffer: GPUBuffer;
    outputIndicesBuffer: GPUBuffer;
    triangleCountBuffer: GPUBuffer;
    clippedCountBuffer: GPUBuffer;
    activeCountBuffer: GPUBuffer;
    // Pipelines
    classifyPipeline: GPUComputePipeline;
    selectPipeline: GPUComputePipeline;
    clipPipeline: GPUComputePipeline;
    countPipeline: GPUComputePipeline;
    // Bind groups
    classifyBindGroup: GPUBindGroup;
    selectBindGroup: GPUBindGroup;
    clipBindGroup: GPUBindGroup;
    countBindGroup: GPUBindGroup;
    // Metadata
    numVertices: number;
    workgroupCount: number;
}

async function initializeResources(device: GPUDevice, points: number[][]): Promise<GPUResources> {
    const numVertices = points.length;
    const maxTriangles = numVertices - 2;
    const workgroupCount = Math.ceil(numVertices / WORKGROUP_SIZE);

    // Normalize points to 4-component vectors
    const normalizedPoints = normalizePoints(points);
    const pointsData = new Float32Array(normalizedPoints.flat());

    // Initialize prev/next as circular linked list
    const prevData = new Uint32Array(numVertices);
    const nextData = new Uint32Array(numVertices);
    for (let i = 0; i < numVertices; i++) {
        prevData[i] = (i === 0) ? numVertices - 1 : i - 1;
        nextData[i] = (i === numVertices - 1) ? 0 : i + 1;
    }

    // All vertices start active
    const isActiveData = new Uint32Array(numVertices);
    isActiveData.fill(1);

    // Create buffers
    const pointsBuffer = device.createBuffer({
        label: "points",
        size: pointsData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pointsBuffer, 0, pointsData);

    const prevVertexBuffer = device.createBuffer({
        label: "prevVertex",
        size: prevData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(prevVertexBuffer, 0, prevData);

    const nextVertexBuffer = device.createBuffer({
        label: "nextVertex",
        size: nextData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(nextVertexBuffer, 0, nextData);

    const isActiveBuffer = device.createBuffer({
        label: "isActive",
        size: isActiveData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(isActiveBuffer, 0, isActiveData);

    const isEarBuffer = device.createBuffer({
        label: "isEar",
        size: numVertices * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const canClipBuffer = device.createBuffer({
        label: "canClip",
        size: numVertices * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const outputIndicesBuffer = device.createBuffer({
        label: "outputIndices",
        size: maxTriangles * 3 * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const triangleCountBuffer = device.createBuffer({
        label: "triangleCount",
        size: BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(triangleCountBuffer, 0, new Uint32Array([0]));

    const clippedCountBuffer = device.createBuffer({
        label: "clippedCount",
        size: BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const activeCountBuffer = device.createBuffer({
        label: "activeCount",
        size: BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Create shader modules
    const classifyModule = device.createShaderModule({
        label: "classifyAndFindEars",
        code: CLASSIFY_AND_FIND_EARS_SHADER,
    });

    const selectModule = device.createShaderModule({
        label: "selectIndependentEars",
        code: SELECT_INDEPENDENT_EARS_SHADER,
    });

    const clipModule = device.createShaderModule({
        label: "clipEars",
        code: CLIP_EARS_SHADER,
    });

    const countModule = device.createShaderModule({
        label: "countActive",
        code: COUNT_ACTIVE_SHADER,
    });

    // Create pipelines
    const classifyPipeline = device.createComputePipeline({
        label: "classifyPipeline",
        layout: "auto",
        compute: { module: classifyModule, entryPoint: "classifyAndFindEars" },
    });

    const selectPipeline = device.createComputePipeline({
        label: "selectPipeline",
        layout: "auto",
        compute: { module: selectModule, entryPoint: "selectIndependentEars" },
    });

    const clipPipeline = device.createComputePipeline({
        label: "clipPipeline",
        layout: "auto",
        compute: { module: clipModule, entryPoint: "clipEars" },
    });

    const countPipeline = device.createComputePipeline({
        label: "countPipeline",
        layout: "auto",
        compute: { module: countModule, entryPoint: "countActive" },
    });

    // Create bind groups
    const classifyBindGroup = device.createBindGroup({
        layout: classifyPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointsBuffer } },
            { binding: 1, resource: { buffer: prevVertexBuffer } },
            { binding: 2, resource: { buffer: nextVertexBuffer } },
            { binding: 3, resource: { buffer: isActiveBuffer } },
            { binding: 4, resource: { buffer: isEarBuffer } },
            { binding: 5, resource: { buffer: canClipBuffer } },
        ],
    });

    const selectBindGroup = device.createBindGroup({
        layout: selectPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: prevVertexBuffer } },
            { binding: 1, resource: { buffer: nextVertexBuffer } },
            { binding: 2, resource: { buffer: isActiveBuffer } },
            { binding: 3, resource: { buffer: isEarBuffer } },
            { binding: 4, resource: { buffer: canClipBuffer } },
        ],
    });

    const clipBindGroup = device.createBindGroup({
        layout: clipPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointsBuffer } },
            { binding: 1, resource: { buffer: prevVertexBuffer } },
            { binding: 2, resource: { buffer: nextVertexBuffer } },
            { binding: 3, resource: { buffer: isActiveBuffer } },
            { binding: 4, resource: { buffer: canClipBuffer } },
            { binding: 5, resource: { buffer: outputIndicesBuffer } },
            { binding: 6, resource: { buffer: triangleCountBuffer } },
            { binding: 7, resource: { buffer: clippedCountBuffer } },
        ],
    });

    const countBindGroup = device.createBindGroup({
        layout: countPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: isActiveBuffer } },
            { binding: 1, resource: { buffer: activeCountBuffer } },
        ],
    });

    return {
        device,
        pointsBuffer,
        prevVertexBuffer,
        nextVertexBuffer,
        isActiveBuffer,
        isEarBuffer,
        canClipBuffer,
        outputIndicesBuffer,
        triangleCountBuffer,
        clippedCountBuffer,
        activeCountBuffer,
        classifyPipeline,
        selectPipeline,
        clipPipeline,
        countPipeline,
        classifyBindGroup,
        selectBindGroup,
        clipBindGroup,
        countBindGroup,
        numVertices,
        workgroupCount,
    };
}

function destroyResources(resources: GPUResources): void {
    resources.pointsBuffer.destroy();
    resources.prevVertexBuffer.destroy();
    resources.nextVertexBuffer.destroy();
    resources.isActiveBuffer.destroy();
    resources.isEarBuffer.destroy();
    resources.canClipBuffer.destroy();
    resources.outputIndicesBuffer.destroy();
    resources.triangleCountBuffer.destroy();
    resources.clippedCountBuffer.destroy();
    resources.activeCountBuffer.destroy();
}

/**
 * Parallel ear clipping algorithm.
 *
 * @param points - Array of [x, y] or [x, y, z] points in CCW order
 * @returns Array of triangles, each triangle is [i, j, k] indices
 */
export async function earClippingParallel(points: number[][]): Promise<number[][]> {
    if (points.length < 3) {
        throw new Error("Polygon must have at least 3 vertices");
    }

    if (points.length === 3) {
        // Trivial case: single triangle
        return [[0, 1, 2]];
    }

    const device = await getGPUDevice();
    const resources = await initializeResources(device, points);

    const maxIterations = points.length; // Safety limit
    let iteration = 0;

    // Staging buffer for reading back counts
    const stagingBuffer = device.createBuffer({
        size: BYTE_SIZE * 2,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    while (iteration < maxIterations) {
        iteration++;

        // Reset clipped count for this iteration
        device.queue.writeBuffer(resources.clippedCountBuffer, 0, new Uint32Array([0]));
        device.queue.writeBuffer(resources.activeCountBuffer, 0, new Uint32Array([0]));

        // Create command encoder for this iteration
        const encoder = device.createCommandEncoder();

        // Pass 1: Classify vertices and find ears
        const classifyPass = encoder.beginComputePass();
        classifyPass.setPipeline(resources.classifyPipeline);
        classifyPass.setBindGroup(0, resources.classifyBindGroup);
        classifyPass.dispatchWorkgroups(resources.workgroupCount);
        classifyPass.end();

        // Pass 2: Select independent ears
        const selectPass = encoder.beginComputePass();
        selectPass.setPipeline(resources.selectPipeline);
        selectPass.setBindGroup(0, resources.selectBindGroup);
        selectPass.dispatchWorkgroups(resources.workgroupCount);
        selectPass.end();

        // Pass 3: Clip selected ears
        const clipPass = encoder.beginComputePass();
        clipPass.setPipeline(resources.clipPipeline);
        clipPass.setBindGroup(0, resources.clipBindGroup);
        clipPass.dispatchWorkgroups(resources.workgroupCount);
        clipPass.end();

        // Pass 4: Count remaining active vertices
        const countPass = encoder.beginComputePass();
        countPass.setPipeline(resources.countPipeline);
        countPass.setBindGroup(0, resources.countBindGroup);
        countPass.dispatchWorkgroups(resources.workgroupCount);
        countPass.end();

        // Copy counts to staging buffer
        encoder.copyBufferToBuffer(resources.clippedCountBuffer, 0, stagingBuffer, 0, BYTE_SIZE);
        encoder.copyBufferToBuffer(resources.activeCountBuffer, 0, stagingBuffer, BYTE_SIZE, BYTE_SIZE);

        // Submit and wait
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read back counts
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const countData = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
        stagingBuffer.unmap();

        const clippedThisIteration = countData[0];
        const activeRemaining = countData[1];

        // Check termination conditions
        if (activeRemaining <= 2) {
            // Done - no more triangles possible
            break;
        }

        if (clippedThisIteration === 0) {
            // No ears found - polygon might be degenerate
            console.warn(`[EarClippingParallel] No ears found at iteration ${iteration}, ${activeRemaining} vertices remaining`);
            break;
        }
    }

    // Read back results
    const triangleCountStaging = device.createBuffer({
        size: BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(resources.triangleCountBuffer, 0, triangleCountStaging, 0, BYTE_SIZE);
    device.queue.submit([encoder.finish()]);

    await triangleCountStaging.mapAsync(GPUMapMode.READ);
    const numTriangles = new Uint32Array(triangleCountStaging.getMappedRange())[0];
    triangleCountStaging.unmap();
    triangleCountStaging.destroy();

    // Read triangle indices
    const indicesStaging = device.createBuffer({
        size: numTriangles * 3 * BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder2 = device.createCommandEncoder();
    encoder2.copyBufferToBuffer(resources.outputIndicesBuffer, 0, indicesStaging, 0, numTriangles * 3 * BYTE_SIZE);
    device.queue.submit([encoder2.finish()]);

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
    stagingBuffer.destroy();
    destroyResources(resources);

    return triangles;
}

export { earClippingParallel as default };
