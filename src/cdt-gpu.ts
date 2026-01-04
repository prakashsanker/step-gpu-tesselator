/**
 * GPU-accelerated Constrained Delaunay Triangulation (CDT)
 *
 * Algorithm: Two-phase approach
 * 1. Initial triangulation via ear clipping (already have this)
 * 2. Edge flipping to achieve Delaunay property
 *
 * The constraint edges (polygon boundary) are marked and never flipped.
 * Interior edges are flipped if they violate the Delaunay condition
 * (opposite vertex is inside the circumcircle of the triangle).
 */

import { getGPUDevice } from "./lib";
import { earClippingSingleDispatch } from "./ear-clipping-single-dispatch";

// Edge representation for the half-edge data structure
interface HalfEdge {
    origin: number;      // Vertex index
    twin: number;        // Index of twin half-edge (-1 if boundary)
    next: number;        // Next half-edge in face
    prev: number;        // Previous half-edge in face
    face: number;        // Face index
    isConstraint: boolean; // True if this edge is a constraint (cannot be flipped)
}

/**
 * Build half-edge data structure from triangles
 */
function buildHalfEdges(
    _vertices: [number, number][],
    triangles: [number, number, number][],
    constraintEdges: [number, number][]
): HalfEdge[] {
    const halfEdges: HalfEdge[] = [];
    const edgeMap = new Map<string, number>(); // "v1,v2" -> halfEdge index

    // Create constraint edge set for O(1) lookup
    const constraintSet = new Set<string>();
    for (const [a, b] of constraintEdges) {
        constraintSet.add(`${Math.min(a, b)},${Math.max(a, b)}`);
    }

    // Create half-edges for each triangle
    for (let f = 0; f < triangles.length; f++) {
        const [a, b, c] = triangles[f];
        const baseIdx = halfEdges.length;

        // Create 3 half-edges for this triangle
        for (let i = 0; i < 3; i++) {
            const verts = [a, b, c];
            const origin = verts[i];
            const dest = verts[(i + 1) % 3];
            const edgeKey = `${origin},${dest}`;
            const constraintKey = `${Math.min(origin, dest)},${Math.max(origin, dest)}`;

            halfEdges.push({
                origin,
                twin: -1,  // Will be filled in later
                next: baseIdx + (i + 1) % 3,
                prev: baseIdx + (i + 2) % 3,
                face: f,
                isConstraint: constraintSet.has(constraintKey)
            });

            edgeMap.set(edgeKey, baseIdx + i);
        }
    }

    // Link twin half-edges
    for (let i = 0; i < halfEdges.length; i++) {
        const he = halfEdges[i];
        const dest = halfEdges[he.next].origin;
        const twinKey = `${dest},${he.origin}`;
        const twinIdx = edgeMap.get(twinKey);
        if (twinIdx !== undefined) {
            he.twin = twinIdx;
        }
    }

    return halfEdges;
}

/**
 * Create the GPU shader for edge flipping
 */
function createEdgeFlipShader(device: GPUDevice) {
    return device.createShaderModule({
        label: "CDT Edge Flip",
        code: `
/*
 * GPU Edge Flipping for Constrained Delaunay Triangulation
 *
 * Each thread checks one edge. If the edge is not a constraint and
 * violates the Delaunay condition, it marks the edge for flipping.
 *
 * The actual flip is done in a second pass to avoid race conditions.
 */

struct Vertex {
    x: f32,
    y: f32
}

struct HalfEdge {
    origin: u32,
    twin: i32,
    next: u32,
    prev: u32,
    face: u32,
    isConstraint: u32
}

struct Uniforms {
    numEdges: u32,
    numVertices: u32
}

@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, read_write> halfEdges: array<HalfEdge>;
@group(0) @binding(2) var<storage, read_write> needsFlip: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// In-circle test: returns true if D is inside circumcircle of ABC
fn inCircle(a: Vertex, b: Vertex, c: Vertex, d: Vertex) -> bool {
    let adx = a.x - d.x;
    let ady = a.y - d.y;
    let bdx = b.x - d.x;
    let bdy = b.y - d.y;
    let cdx = c.x - d.x;
    let cdy = c.y - d.y;

    let abdet = adx * bdy - bdx * ady;
    let bcdet = bdx * cdy - cdx * bdy;
    let cadet = cdx * ady - adx * cdy;

    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    return (alift * bcdet + blift * cadet + clift * abdet) > 0.0;
}

@compute @workgroup_size(64)
fn detectFlips(@builtin(global_invocation_id) id: vec3<u32>) {
    let edgeIdx = id.x;
    if (edgeIdx >= uniforms.numEdges) {
        return;
    }

    let he = halfEdges[edgeIdx];

    // Skip if this is a constraint edge or boundary edge
    if (he.isConstraint == 1u || he.twin < 0) {
        needsFlip[edgeIdx] = 0u;
        return;
    }

    // Only process edges where edgeIdx < twin to avoid double-processing
    if (edgeIdx > u32(he.twin)) {
        needsFlip[edgeIdx] = 0u;
        return;
    }

    // Get the quad formed by the two triangles sharing this edge
    //     C
    //    /|\\
    //   / | \\
    //  A--+--B  (edge being checked is A-B)
    //   \\ | /
    //    \\|/
    //     D

    let heNext = halfEdges[he.next];
    let twinHe = halfEdges[u32(he.twin)];
    let twinNext = halfEdges[twinHe.next];

    let A = vertices[he.origin];
    let B = vertices[heNext.origin];
    let C = vertices[halfEdges[heNext.next].origin];  // Opposite vertex in our triangle
    let D = vertices[halfEdges[twinNext.next].origin];  // Opposite vertex in twin triangle

    // Check if D is inside circumcircle of ABC
    if (inCircle(A, B, C, D)) {
        needsFlip[edgeIdx] = 1u;
    } else {
        needsFlip[edgeIdx] = 0u;
    }
}

@compute @workgroup_size(64)
fn applyFlips(@builtin(global_invocation_id) id: vec3<u32>) {
    let edgeIdx = id.x;
    if (edgeIdx >= uniforms.numEdges) {
        return;
    }

    if (needsFlip[edgeIdx] == 0u) {
        return;
    }

    // Perform edge flip
    // Before:          After:
    //     C               C
    //    /|\\            / \\
    //   / | \\          /   \\
    //  A--+--B   =>    A     B
    //   \\ | /          \\   /
    //    \\|/            \\ /
    //     D               D
    //
    // Edge A-B becomes edge C-D

    let he = halfEdges[edgeIdx];
    let twin = halfEdges[u32(he.twin)];

    let heNext = halfEdges[he.next];
    let hePrev = halfEdges[he.prev];
    let twinNext = halfEdges[twin.next];
    let twinPrev = halfEdges[twin.prev];

    // Get vertices
    let A = he.origin;
    let B = heNext.origin;
    let C = halfEdges[heNext.next].origin;
    let D = halfEdges[twinNext.next].origin;

    // Update edge origins to flip from A-B to C-D
    halfEdges[edgeIdx].origin = C;
    halfEdges[u32(he.twin)].origin = D;

    // Update next/prev pointers for the flip
    // This is the complex part - we need to rewire the connectivity

    // For edge AB->CD:
    // he: was A->B, becomes C->D
    // twin: was B->A, becomes D->C

    // Triangle 1 (was ABC, becomes ACD):
    halfEdges[edgeIdx].next = hePrev.next;  // D's incoming edge
    halfEdges[edgeIdx].prev = he.next;       // A's outgoing edge

    // Triangle 2 (was ABD, becomes BCD):
    halfEdges[u32(he.twin)].next = twinPrev.next;
    halfEdges[u32(he.twin)].prev = twin.next;

    // Clear flip flag
    needsFlip[edgeIdx] = 0u;
}
`
    });
}

/**
 * GPU-accelerated edge flipping to achieve Delaunay property
 */
export async function gpuEdgeFlip(
    vertices: [number, number][],
    triangles: [number, number, number][],
    constraintEdges: [number, number][],
    maxIterations: number = 100
): Promise<[number, number, number][]> {
    const device = await getGPUDevice();

    // Build half-edge data structure
    const halfEdges = buildHalfEdges(vertices, triangles, constraintEdges);
    const numEdges = halfEdges.length;

    // Create GPU buffers
    const vertexData = new Float32Array(vertices.length * 2);
    vertices.forEach((v, i) => {
        vertexData[i * 2] = v[0];
        vertexData[i * 2 + 1] = v[1];
    });

    const vertexBuffer = device.createBuffer({
        label: "vertices",
        size: vertexData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertexData);

    // Half-edge buffer (6 u32s per half-edge)
    const heData = new Int32Array(numEdges * 6);
    halfEdges.forEach((he, i) => {
        heData[i * 6 + 0] = he.origin;
        heData[i * 6 + 1] = he.twin;
        heData[i * 6 + 2] = he.next;
        heData[i * 6 + 3] = he.prev;
        heData[i * 6 + 4] = he.face;
        heData[i * 6 + 5] = he.isConstraint ? 1 : 0;
    });

    const halfEdgeBuffer = device.createBuffer({
        label: "halfEdges",
        size: heData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(halfEdgeBuffer, 0, heData);

    const needsFlipBuffer = device.createBuffer({
        label: "needsFlip",
        size: numEdges * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const uniformData = new Uint32Array([numEdges, vertices.length]);
    const uniformBuffer = device.createBuffer({
        label: "uniforms",
        size: uniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create shader and pipelines
    const shader = createEdgeFlipShader(device);

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ]
    });

    const detectPipeline = device.createComputePipeline({
        label: "detect-flips",
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shader, entryPoint: "detectFlips" }
    });

    const applyPipeline = device.createComputePipeline({
        label: "apply-flips",
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shader, entryPoint: "applyFlips" }
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: vertexBuffer } },
            { binding: 1, resource: { buffer: halfEdgeBuffer } },
            { binding: 2, resource: { buffer: needsFlipBuffer } },
            { binding: 3, resource: { buffer: uniformBuffer } },
        ]
    });

    const workgroupCount = Math.ceil(numEdges / 64);

    // Iterate until no more flips needed
    for (let iter = 0; iter < maxIterations; iter++) {
        const encoder = device.createCommandEncoder();

        // Detect edges that need flipping
        const detectPass = encoder.beginComputePass();
        detectPass.setPipeline(detectPipeline);
        detectPass.setBindGroup(0, bindGroup);
        detectPass.dispatchWorkgroups(workgroupCount);
        detectPass.end();

        // Apply flips
        const applyPass = encoder.beginComputePass();
        applyPass.setPipeline(applyPipeline);
        applyPass.setBindGroup(0, bindGroup);
        applyPass.dispatchWorkgroups(workgroupCount);
        applyPass.end();

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // TODO: Check if any flips were made and break early if not
    }

    // Read back results
    const readbackBuffer = device.createBuffer({
        size: heData.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(halfEdgeBuffer, 0, readbackBuffer, 0, heData.byteLength);
    device.queue.submit([copyEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Int32Array(readbackBuffer.getMappedRange());

    // Reconstruct triangles from half-edges
    const resultTriangles: [number, number, number][] = [];
    const processedFaces = new Set<number>();

    for (let i = 0; i < numEdges; i++) {
        const face = resultData[i * 6 + 4];
        if (processedFaces.has(face)) continue;
        processedFaces.add(face);

        const v0 = resultData[i * 6 + 0];
        const next1 = resultData[i * 6 + 2];
        const v1 = resultData[next1 * 6 + 0];
        const next2 = resultData[next1 * 6 + 2];
        const v2 = resultData[next2 * 6 + 0];

        resultTriangles.push([v0, v1, v2]);
    }

    readbackBuffer.unmap();

    // Cleanup
    vertexBuffer.destroy();
    halfEdgeBuffer.destroy();
    needsFlipBuffer.destroy();
    uniformBuffer.destroy();
    readbackBuffer.destroy();

    return resultTriangles;
}

/**
 * Build constraint edges from boundary polygon
 * Each consecutive pair of vertices forms a constraint edge
 */
function buildConstraintEdges(numVertices: number): [number, number][] {
    const edges: [number, number][] = [];
    for (let i = 0; i < numVertices; i++) {
        edges.push([i, (i + 1) % numVertices]);
    }
    return edges;
}

/**
 * Full CDT pipeline:
 * 1. Initial triangulation via ear clipping
 * 2. Edge flipping for Delaunay property
 *
 * @param boundary - 2D boundary polygon vertices (CCW order)
 * @param holes - Optional array of hole polygons (CW order)
 * @param applyDelaunay - Whether to apply edge flipping (default: true)
 */
export async function constrainedDelaunayTriangulation(
    boundary: [number, number][],
    holes: [number, number][][] = [],
    applyDelaunay: boolean = true
): Promise<[number, number, number][]> {
    // Step 1: Merge holes into boundary (if any)
    // For now, assume no holes - just triangulate the boundary
    // TODO: Integrate hole bridging from step-parser

    if (holes.length > 0) {
        console.warn("[CDT] Holes not yet supported in CDT, ignoring them");
    }

    // Convert 2D boundary to 3D for ear clipping (z=0)
    const boundary3d: [number, number, number][] = boundary.map(([x, y]) => [x, y, 0]);

    // Step 2: Initial triangulation via ear clipping
    const initialTriangles = await earClippingSingleDispatch(boundary3d);

    if (initialTriangles.length === 0) {
        console.warn("[CDT] Ear clipping returned no triangles");
        return [];
    }

    // Debug: check ear clipping output for degenerate triangles
    const degenerateTris = initialTriangles.filter(t =>
        t[0] === t[1] || t[1] === t[2] || t[0] === t[2]
    );
    if (degenerateTris.length > 0) {
        console.warn(`[CDT] Ear clipping produced ${degenerateTris.length} degenerate triangles:`);
        degenerateTris.slice(0, 5).forEach(t => console.warn(`  [${t.join(',')}]`));
    }

    // If Delaunay optimization is disabled, return ear clipping result directly
    if (!applyDelaunay) {
        return initialTriangles as [number, number, number][];
    }

    // Step 3: Build constraint edges (boundary edges cannot be flipped)
    const constraintEdges = buildConstraintEdges(boundary.length);

    // Step 4: Apply edge flipping for Delaunay property
    const delaunayTriangles = await gpuEdgeFlip(
        boundary,
        initialTriangles as [number, number, number][],
        constraintEdges,
        50  // Max iterations
    );

    // Debug: check edge flip output for degenerate triangles
    const degenerateAfterFlip = delaunayTriangles.filter(t =>
        t[0] === t[1] || t[1] === t[2] || t[0] === t[2]
    );
    if (degenerateAfterFlip.length > 0) {
        console.warn(`[CDT] Edge flipping produced ${degenerateAfterFlip.length} degenerate triangles:`);
        degenerateAfterFlip.slice(0, 5).forEach(t => console.warn(`  [${t.join(',')}]`));
    }

    return delaunayTriangles;
}

/**
 * Triangulate a 2D polygon with optional Steiner points for mesh quality.
 * This is a higher-level API that adds interior points for better triangles.
 *
 * @param boundary - 2D boundary polygon
 * @param minAngle - Minimum angle constraint (degrees, default 20)
 * @param maxArea - Maximum triangle area constraint
 */
export async function qualityTriangulation(
    boundary: [number, number][],
    _minAngle: number = 20,
    _maxArea?: number
): Promise<{
    vertices: [number, number][];
    triangles: [number, number, number][];
}> {
    // For now, just use CDT without Steiner points
    // TODO: Implement Steiner point insertion based on quality metrics

    const triangles = await constrainedDelaunayTriangulation(boundary, [], true);

    return {
        vertices: boundary,
        triangles
    };
}
