import { getGPUDevice, normalizePoints } from "./lib";

const BYTE_SIZE = 4;

/**
 * Single-dispatch ear clipping algorithm.
 *
 * Runs the ENTIRE ear clipping loop on the GPU in one dispatch,
 * eliminating all CPU-GPU synchronization overhead.
 *
 * Trade-off: Single-threaded on GPU (only thread 0 does work),
 * but no sync points means much lower latency for small polygons.
 */

function createSingleDispatchEarClippingShader(device: GPUDevice) {
    return device.createShaderModule({
        label: "Single-dispatch ear clipping",
        code: `
/* Single-dispatch ear clipping shader
 *
 * This shader runs the entire ear clipping algorithm in a single dispatch.
 * Only thread 0 does actual work - this trades parallelism for eliminating
 * CPU-GPU sync overhead which dominates for small/medium polygons.
 */

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

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> prevVertex: array<u32>;
@group(0) @binding(2) var<storage, read_write> nextVertex: array<u32>;
@group(0) @binding(3) var<storage, read_write> isActive: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> triangleCount: atomic<u32>;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

const CONVEX: u32 = 1u;
const REFLEX: u32 = 0u;

// Cross product for 2D vectors (returns z component)
fn cross2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    return ax * by - ay * bx;
}

// Check if vertex at index i is convex (left turn)
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

// Check if point P is inside triangle ABC (using barycentric/cross product method)
fn pointInTriangle(A: Point, B: Point, C: Point, P: Point) -> bool {
    let c1 = cross2d(B.x - A.x, B.y - A.y, P.x - A.x, P.y - A.y);
    let c2 = cross2d(C.x - B.x, C.y - B.y, P.x - B.x, P.y - B.y);
    let c3 = cross2d(A.x - C.x, A.y - C.y, P.x - C.x, P.y - C.y);

    let eps = 1e-10;
    // All same sign (or zero) means inside
    return (c1 >= -eps && c2 >= -eps && c3 >= -eps);
}

// Check if vertex i is an ear
fn isEar(i: u32, numVerts: u32) -> bool {
    if (!isConvex(i)) {
        return false;
    }

    let p = prevVertex[i];
    let n = nextVertex[i];

    let A = points[p];
    let B = points[i];
    let C = points[n];

    // Check if any other active vertex is inside this triangle
    for (var j = 0u; j < numVerts; j++) {
        if (isActive[j] == 0u) { continue; }
        if (j == i || j == p || j == n) { continue; }

        let P = points[j];

        // Skip if P is at same position as a triangle vertex
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

// Clip ear at vertex i
fn clipEar(i: u32) {
    let p = prevVertex[i];
    let n = nextVertex[i];

    // Write triangle
    let triIdx = atomicAdd(&triangleCount, 1u);
    outputIndices[triIdx * 3u + 0u] = p;
    outputIndices[triIdx * 3u + 1u] = i;
    outputIndices[triIdx * 3u + 2u] = n;

    // Remove vertex from linked list
    nextVertex[p] = n;
    prevVertex[n] = p;

    // Mark as inactive
    isActive[i] = 0u;
}

// Count active vertices
fn countActive(numVerts: u32) -> u32 {
    var count = 0u;
    for (var i = 0u; i < numVerts; i++) {
        if (isActive[i] == 1u) {
            count++;
        }
    }
    return count;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Only thread 0 does work
    if (id.x != 0u) {
        return;
    }

    let numVerts = uniforms.numVertices;
    let maxTris = uniforms.maxTriangles;

    // Main ear clipping loop
    for (var iteration = 0u; iteration < maxTris; iteration++) {
        let activeCount = countActive(numVerts);

        // Done when only 3 vertices left
        if (activeCount <= 3u) {
            if (activeCount == 3u) {
                // Find the 3 remaining vertices and output final triangle
                var idx0 = 0xFFFFFFFFu;
                var idx1 = 0xFFFFFFFFu;
                var idx2 = 0xFFFFFFFFu;
                var found = 0u;

                for (var i = 0u; i < numVerts; i++) {
                    if (isActive[i] == 1u) {
                        if (found == 0u) { idx0 = i; }
                        else if (found == 1u) { idx1 = i; }
                        else if (found == 2u) { idx2 = i; }
                        found++;
                    }
                }

                let triIdx = atomicAdd(&triangleCount, 1u);
                outputIndices[triIdx * 3u + 0u] = idx0;
                outputIndices[triIdx * 3u + 1u] = idx1;
                outputIndices[triIdx * 3u + 2u] = idx2;
            }
            return;
        }

        // Find first ear and clip it
        var foundEar = false;
        for (var i = 0u; i < numVerts; i++) {
            if (isActive[i] == 0u) { continue; }

            if (isEar(i, numVerts)) {
                clipEar(i);
                foundEar = true;
                break;
            }
        }

        // Safety: if no ear found, something is wrong - bail out
        if (!foundEar) {
            return;
        }
    }
}
`
    });
}

export async function earClippingSingleDispatch(points: number[][]): Promise<number[][]> {
    const device = await getGPUDevice();

    if (points.length < 3) {
        throw new Error("Polygon must have at least 3 vertices");
    }

    const numVertices = points.length;
    const maxTriangles = numVertices - 2;

    // Normalize points to [x, y, z, padding] format
    const normalizedPoints = normalizePoints(points);
    const pointsData = new Float32Array(normalizedPoints.flat());

    // Initialize prev/next linked list
    const prevData = new Uint32Array(numVertices);
    const nextData = new Uint32Array(numVertices);
    for (let i = 0; i < numVertices; i++) {
        prevData[i] = (i === 0) ? (numVertices - 1) : (i - 1);
        nextData[i] = (i === numVertices - 1) ? 0 : (i + 1);
    }

    // Initialize all vertices as active
    const activeData = new Uint32Array(numVertices);
    activeData.fill(1);

    // Create buffers
    const pointsBuffer = device.createBuffer({
        label: "points",
        size: pointsData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pointsBuffer, 0, pointsData);

    const prevBuffer = device.createBuffer({
        label: "prevVertex",
        size: prevData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(prevBuffer, 0, prevData);

    const nextBuffer = device.createBuffer({
        label: "nextVertex",
        size: nextData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(nextBuffer, 0, nextData);

    const activeBuffer = device.createBuffer({
        label: "active",
        size: activeData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(activeBuffer, 0, activeData);

    const outputBuffer = device.createBuffer({
        label: "outputIndices",
        size: maxTriangles * 3 * BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const triangleCountBuffer = device.createBuffer({
        label: "triangleCount",
        size: BYTE_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(triangleCountBuffer, 0, new Uint32Array([0]));

    const uniformData = new Uint32Array([numVertices, maxTriangles]);
    const uniformBuffer = device.createBuffer({
        label: "uniforms",
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create shader and pipeline
    const shader = createSingleDispatchEarClippingShader(device);

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ]
    });

    const pipeline = device.createComputePipeline({
        label: "ear-clipping-single-dispatch",
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shader, entryPoint: "main" }
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: pointsBuffer } },
            { binding: 1, resource: { buffer: prevBuffer } },
            { binding: 2, resource: { buffer: nextBuffer } },
            { binding: 3, resource: { buffer: activeBuffer } },
            { binding: 4, resource: { buffer: outputBuffer } },
            { binding: 5, resource: { buffer: triangleCountBuffer } },
            { binding: 6, resource: { buffer: uniformBuffer } },
        ]
    });

    // Single dispatch - entire algorithm runs on GPU
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1); // Single workgroup, single thread
    pass.end();

    // Copy results to readback buffers
    const triangleCountReadback = device.createBuffer({
        size: BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(triangleCountBuffer, 0, triangleCountReadback, 0, BYTE_SIZE);

    const outputReadback = device.createBuffer({
        size: maxTriangles * 3 * BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(outputBuffer, 0, outputReadback, 0, maxTriangles * 3 * BYTE_SIZE);

    // Submit and wait ONCE
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read results
    await triangleCountReadback.mapAsync(GPUMapMode.READ);
    const numTriangles = new Uint32Array(triangleCountReadback.getMappedRange())[0];
    triangleCountReadback.unmap();

    await outputReadback.mapAsync(GPUMapMode.READ);
    const indicesData = new Uint32Array(outputReadback.getMappedRange());

    // Convert to array of triangles
    const triangles: number[][] = [];
    for (let i = 0; i < numTriangles; i++) {
        triangles.push([
            indicesData[i * 3],
            indicesData[i * 3 + 1],
            indicesData[i * 3 + 2]
        ]);
    }
    outputReadback.unmap();

    // Cleanup
    pointsBuffer.destroy();
    prevBuffer.destroy();
    nextBuffer.destroy();
    activeBuffer.destroy();
    outputBuffer.destroy();
    triangleCountBuffer.destroy();
    uniformBuffer.destroy();
    triangleCountReadback.destroy();
    outputReadback.destroy();

    return triangles;
}
