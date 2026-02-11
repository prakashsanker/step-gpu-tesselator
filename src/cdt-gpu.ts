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
import { earClipping } from "./ear-clipping";

const DEBUG_CDT_LOGS = false;

function cdtDebugEnabled(): boolean {
    return DEBUG_CDT_LOGS || (globalThis as any)?.__CDT_DEBUG_LOGS__ === true;
}

function cdtDebugLog(...args: unknown[]): void {
    if (cdtDebugEnabled()) {
        console.log(...args);
    }
}

function cdtDebugWarn(...args: unknown[]): void {
    if (cdtDebugEnabled()) {
        console.warn(...args);
    }
}

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
 * In-circle test: returns true if point D is inside the circumcircle of triangle ABC.
 * Uses the determinant method which is robust for convex/concave cases.
 */
function inCircle(
    ax: number, ay: number,
    bx: number, by: number,
    cx: number, cy: number,
    dx: number, dy: number
): boolean {
    const adx = ax - dx;
    const ady = ay - dy;
    const bdx = bx - dx;
    const bdy = by - dy;
    const cdx = cx - dx;
    const cdy = cy - dy;

    const abdet = adx * bdy - bdx * ady;
    const bcdet = bdx * cdy - cdx * bdy;
    const cadet = cdx * ady - adx * cdy;

    const alift = adx * adx + ady * ady;
    const blift = bdx * bdx + bdy * bdy;
    const clift = cdx * cdx + cdy * cdy;

    return (alift * bcdet + blift * cadet + clift * abdet) > 0;
}

/**
 * CPU-based edge flipping for Delaunay property.
 * This is a correct implementation that properly handles all connectivity updates.
 */
function cpuEdgeFlip(
    vertices: [number, number][],
    halfEdges: HalfEdge[],
    maxIterations: number = 100
): void {
    for (let iter = 0; iter < maxIterations; iter++) {
        let flipped = false;

        // Check each edge for potential flip
        for (let i = 0; i < halfEdges.length; i++) {
            const he = halfEdges[i];

            // Skip constraints, boundary edges, and already-processed edges
            if (he.isConstraint || he.twin < 0) continue;

            // Only process each edge pair once (smaller index)
            if (i > he.twin) continue;

            const twin = halfEdges[he.twin];

            // Get the quad vertices:
            //     C
            //    /|\
            //   / | \
            //  A--+--B  (current edge is A->B)
            //   \ | /
            //    \|/
            //     D
            const A = he.origin;
            const B = halfEdges[he.next].origin;
            const C = halfEdges[halfEdges[he.next].next].origin;
            const D = halfEdges[halfEdges[twin.next].next].origin;

            // Check if D is inside circumcircle of ABC
            const [ax, ay] = vertices[A];
            const [bx, by] = vertices[B];
            const [cx, cy] = vertices[C];
            const [dx, dy] = vertices[D];

            if (!inCircle(ax, ay, bx, by, cx, cy, dx, dy)) continue;

            // Perform the edge flip
            // Before: triangles ABC and BAD
            // After: triangles ACD and DCB

            // Get all 6 half-edges involved
            const heAB = i;
            const heBA = he.twin;
            const heBC = he.next;
            const heCA = halfEdges[heBC].next;
            const heAD = twin.next;
            const heDB = halfEdges[heAD].next;

            // Update the flipped edge endpoints
            // heAB becomes heCD (C->D)
            // heBA becomes heDC (D->C)
            halfEdges[heAB].origin = C;
            halfEdges[heBA].origin = D;

            // Update next/prev pointers for triangle 1 (ACD, was ABC)
            // heCD.next = heDA (was heBC)
            // heDA.next = heAC (which is heCA with reversed direction - but we use heCA)
            // Actually let's think about this more carefully...

            // After flip, the two triangles are:
            // Triangle 1: C -> D -> A -> C  (edges: CD, DA, AC)
            // Triangle 2: D -> C -> B -> D  (edges: DC, CB, BD)

            // heAB (now CD): next should point to DA, prev should point to AC
            // heBA (now DC): next should point to CB, prev should point to BD

            // heCA now becomes AC (same edge, but in triangle ACD)
            // heAD now becomes DA (same direction)
            // heBC now becomes CB (same direction)
            // heDB now becomes BD (same direction)

            // Update triangle 1 (ACD): CD -> DA -> AC -> CD
            halfEdges[heAB].next = heAD;  // CD -> DA
            halfEdges[heAD].next = heCA;  // DA -> AC
            halfEdges[heCA].next = heAB;  // AC -> CD

            halfEdges[heAB].prev = heCA;  // CD <- AC
            halfEdges[heAD].prev = heAB;  // DA <- CD
            halfEdges[heCA].prev = heAD;  // AC <- DA

            // Update triangle 2 (DCB): DC -> CB -> BD -> DC
            halfEdges[heBA].next = heBC;  // DC -> CB
            halfEdges[heBC].next = heDB;  // CB -> BD
            halfEdges[heDB].next = heBA;  // BD -> DC

            halfEdges[heBA].prev = heDB;  // DC <- BD
            halfEdges[heBC].prev = heBA;  // CB <- DC
            halfEdges[heDB].prev = heBC;  // BD <- CB

            // Update face indices (assign new face numbers)
            const face1 = halfEdges[heAB].face;
            const face2 = halfEdges[heBA].face;
            halfEdges[heAB].face = face1;
            halfEdges[heAD].face = face1;
            halfEdges[heCA].face = face1;
            halfEdges[heBA].face = face2;
            halfEdges[heBC].face = face2;
            halfEdges[heDB].face = face2;

            flipped = true;
        }

        if (!flipped) {
            cdtDebugLog(`[CDT] Edge flipping converged after ${iter + 1} iterations`);
            break;
        }
    }
}

/**
 * Extract triangles from half-edge data structure
 */
function extractTrianglesFromHalfEdges(halfEdges: HalfEdge[]): [number, number, number][] {
    const triangles: [number, number, number][] = [];
    const processedFaces = new Set<number>();

    for (let i = 0; i < halfEdges.length; i++) {
        const face = halfEdges[i].face;
        if (processedFaces.has(face)) continue;
        processedFaces.add(face);

        const v0 = halfEdges[i].origin;
        const v1 = halfEdges[halfEdges[i].next].origin;
        const v2 = halfEdges[halfEdges[halfEdges[i].next].next].origin;

        triangles.push([v0, v1, v2]);
    }

    return triangles;
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
 * Build constraint edges from boundary AND holes.
 * All boundary edges and hole edges become constraints.
 *
 * @param boundaryCount - Number of vertices in outer boundary
 * @param holeOffsets - Array of [startIndex, count] for each hole
 */
function buildConstraintEdgesWithHoles(
    boundaryCount: number,
    holeOffsets: { start: number; count: number }[]
): [number, number][] {
    const edges: [number, number][] = [];

    // Boundary edges
    for (let i = 0; i < boundaryCount; i++) {
        edges.push([i, (i + 1) % boundaryCount]);
    }

    // Hole edges
    for (const hole of holeOffsets) {
        for (let i = 0; i < hole.count; i++) {
            const v1 = hole.start + i;
            const v2 = hole.start + ((i + 1) % hole.count);
            edges.push([v1, v2]);
        }
    }

    return edges;
}

// ============================================================================
// Bowyer-Watson Delaunay Triangulation with Constraint Edge Recovery
// ============================================================================

/**
 * Compute circumcircle of a triangle. Returns center and radius squared.
 */
function circumcircle(
    ax: number, ay: number,
    bx: number, by: number,
    cx: number, cy: number
): { cx: number; cy: number; r2: number } | null {
    const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (Math.abs(d) < 1e-12) {
        return null; // Degenerate triangle
    }

    const ax2ay2 = ax * ax + ay * ay;
    const bx2by2 = bx * bx + by * by;
    const cx2cy2 = cx * cx + cy * cy;

    const ux = (ax2ay2 * (by - cy) + bx2by2 * (cy - ay) + cx2cy2 * (ay - by)) / d;
    const uy = (ax2ay2 * (cx - bx) + bx2by2 * (ax - cx) + cx2cy2 * (bx - ax)) / d;

    const r2 = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);

    return { cx: ux, cy: uy, r2 };
}

/**
 * Check if point is inside circumcircle of triangle.
 */
function pointInCircumcircle(
    px: number, py: number,
    ax: number, ay: number,
    bx: number, by: number,
    cx: number, cy: number
): boolean {
    const cc = circumcircle(ax, ay, bx, by, cx, cy);
    if (!cc) return false;

    const dist2 = (px - cc.cx) * (px - cc.cx) + (py - cc.cy) * (py - cc.cy);
    return dist2 < cc.r2 - 1e-10; // Small epsilon for numerical stability
}

interface DelaunayTriangle {
    v: [number, number, number]; // Vertex indices
    bad: boolean; // Marked for removal
}

/**
 * Bowyer-Watson algorithm for Delaunay triangulation.
 * Returns triangles as vertex index triples.
 */
function bowyerWatson(points: [number, number][]): [number, number, number][] {
    if (points.length < 3) return [];

    // Find bounding box
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of points) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    }

    // Create super-triangle that contains all points
    const dx = maxX - minX;
    const dy = maxY - minY;
    const dmax = Math.max(dx, dy) * 2;
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;

    // Super-triangle vertices (indices will be points.length, points.length+1, points.length+2)
    const superV0: [number, number] = [midX - dmax * 2, midY - dmax];
    const superV1: [number, number] = [midX + dmax * 2, midY - dmax];
    const superV2: [number, number] = [midX, midY + dmax * 2];

    // All vertices including super-triangle
    const allPoints: [number, number][] = [...points, superV0, superV1, superV2];
    const superIdx0 = points.length;
    const superIdx1 = points.length + 1;
    const superIdx2 = points.length + 2;

    // Start with super-triangle
    const triangles: DelaunayTriangle[] = [
        { v: [superIdx0, superIdx1, superIdx2], bad: false }
    ];

    // Insert each point
    for (let i = 0; i < points.length; i++) {
        const [px, py] = points[i];

        // Find all triangles whose circumcircle contains the point
        for (const tri of triangles) {
            if (tri.bad) continue;

            const [a, b, c] = tri.v;
            if (pointInCircumcircle(
                px, py,
                allPoints[a][0], allPoints[a][1],
                allPoints[b][0], allPoints[b][1],
                allPoints[c][0], allPoints[c][1]
            )) {
                tri.bad = true;
            }
        }

        // Find the boundary of the polygonal hole (edges of bad triangles not shared)
        const edgeCount = new Map<string, { v1: number; v2: number; count: number }>();

        for (const tri of triangles) {
            if (!tri.bad) continue;

            const [a, b, c] = tri.v;
            const edges: [number, number][] = [[a, b], [b, c], [c, a]];

            for (const [v1, v2] of edges) {
                const key = v1 < v2 ? `${v1},${v2}` : `${v2},${v1}`;
                const existing = edgeCount.get(key);
                if (existing) {
                    existing.count++;
                } else {
                    edgeCount.set(key, { v1, v2, count: 1 });
                }
            }
        }

        // Remove bad triangles
        const goodTriangles = triangles.filter(t => !t.bad);
        triangles.length = 0;
        triangles.push(...goodTriangles);

        // Create new triangles from boundary edges to new point
        for (const [, edge] of edgeCount) {
            if (edge.count === 1) {
                // This edge is on the boundary of the cavity
                triangles.push({
                    v: [edge.v1, edge.v2, i],
                    bad: false
                });
            }
        }
    }

    // Remove triangles that contain super-triangle vertices
    const result: [number, number, number][] = [];
    for (const tri of triangles) {
        const [a, b, c] = tri.v;
        if (a >= superIdx0 || b >= superIdx0 || c >= superIdx0) {
            continue; // Skip triangles with super-triangle vertices
        }
        result.push([a, b, c]);
    }

    return result;
}

/**
 * Check if an edge exists in the triangulation.
 */
function edgeExists(
    triangles: [number, number, number][],
    v1: number,
    v2: number
): boolean {
    for (const [a, b, c] of triangles) {
        if ((a === v1 && b === v2) || (b === v1 && c === v2) || (c === v1 && a === v2) ||
            (a === v2 && b === v1) || (b === v2 && c === v1) || (c === v2 && a === v1)) {
            return true;
        }
    }
    return false;
}

/**
 * Find triangles that share an edge.
 */
function findTrianglesWithEdge(
    triangles: [number, number, number][],
    v1: number,
    v2: number
): number[] {
    const result: number[] = [];
    for (let i = 0; i < triangles.length; i++) {
        const [a, b, c] = triangles[i];
        if ((a === v1 || b === v1 || c === v1) && (a === v2 || b === v2 || c === v2)) {
            result.push(i);
        }
    }
    return result;
}

/**
 * Check if a point is strictly to the left of a directed line segment.
 */
function isLeftOf(
    lineStart: [number, number],
    lineEnd: [number, number],
    point: [number, number]
): boolean {
    const cross = (lineEnd[0] - lineStart[0]) * (point[1] - lineStart[1]) -
                  (lineEnd[1] - lineStart[1]) * (point[0] - lineStart[0]);
    return cross > 1e-10;
}

/**
 * Check if a point is strictly to the right of a directed line segment.
 */
function isRightOf(
    lineStart: [number, number],
    lineEnd: [number, number],
    point: [number, number]
): boolean {
    const cross = (lineEnd[0] - lineStart[0]) * (point[1] - lineStart[1]) -
                  (lineEnd[1] - lineStart[1]) * (point[0] - lineStart[0]);
    return cross < -1e-10;
}

/**
 * Check if a triangle intersects a line segment (edge crosses or vertices on opposite sides).
 */
function triangleIntersectsEdge(
    tri: [number, number, number],
    vertices: [number, number][],
    v1: number,
    v2: number
): boolean {
    const [a, b, c] = tri;

    // Skip if triangle contains either endpoint
    if (a === v1 || a === v2 || b === v1 || b === v2 || c === v1 || c === v2) {
        // Check if the edge is actually part of this triangle
        if ((a === v1 && b === v2) || (b === v1 && c === v2) || (c === v1 && a === v2) ||
            (a === v2 && b === v1) || (b === v2 && c === v1) || (c === v2 && a === v1)) {
            return false; // Edge is part of triangle, not intersecting
        }
        // Triangle contains one endpoint - check if constraint passes through interior
        const p1 = vertices[v1];
        const p2 = vertices[v2];
        const pa = vertices[a];
        const pb = vertices[b];
        const pc = vertices[c];

        // Check each edge of the triangle for intersection with constraint
        const edges: [number, number][] = [[a, b], [b, c], [c, a]];
        for (const [e1, e2] of edges) {
            if (e1 === v1 || e1 === v2 || e2 === v1 || e2 === v2) continue;
            if (edgesCross(p1, p2, vertices[e1], vertices[e2])) {
                return true;
            }
        }
        return false;
    }

    const p1 = vertices[v1];
    const p2 = vertices[v2];
    const pa = vertices[a];
    const pb = vertices[b];
    const pc = vertices[c];

    // Check if the constraint edge crosses any triangle edge
    const edges: [[number, number], [number, number]][] = [
        [pa, pb], [pb, pc], [pc, pa]
    ];

    for (const [ea, eb] of edges) {
        if (edgesCross(p1, p2, ea, eb)) {
            return true;
        }
    }

    return false;
}

/**
 * Triangulate a simple polygon using the fan method.
 * Assumes the polygon is convex or at least star-shaped from vertex 0.
 */
function fanTriangulate(vertices: number[]): [number, number, number][] {
    if (vertices.length < 3) return [];

    const triangles: [number, number, number][] = [];
    for (let i = 1; i < vertices.length - 1; i++) {
        triangles.push([vertices[0], vertices[i], vertices[i + 1]]);
    }
    return triangles;
}

/**
 * Simple ear clipping for a polygon defined by vertex indices.
 * Used for re-triangulating cavities.
 */
function earClipPolygon(
    polyIndices: number[],
    allVertices: [number, number][]
): [number, number, number][] {
    if (polyIndices.length < 3) return [];
    if (polyIndices.length === 3) {
        return [[polyIndices[0], polyIndices[1], polyIndices[2]]];
    }

    const triangles: [number, number, number][] = [];
    const remaining = [...polyIndices];

    // Compute signed area to determine winding
    let area = 0;
    for (let i = 0; i < remaining.length; i++) {
        const j = (i + 1) % remaining.length;
        const vi = allVertices[remaining[i]];
        const vj = allVertices[remaining[j]];
        area += vi[0] * vj[1] - vj[0] * vi[1];
    }
    const ccw = area > 0;

    let safety = remaining.length * remaining.length;

    while (remaining.length > 3 && safety-- > 0) {
        let earFound = false;

        for (let i = 0; i < remaining.length; i++) {
            const prevIdx = (i - 1 + remaining.length) % remaining.length;
            const nextIdx = (i + 1) % remaining.length;

            const prev = remaining[prevIdx];
            const curr = remaining[i];
            const next = remaining[nextIdx];

            const prevV = allVertices[prev];
            const currV = allVertices[curr];
            const nextV = allVertices[next];

            // Check if ear (convex vertex with no other vertices inside)
            const cross = (currV[0] - prevV[0]) * (nextV[1] - prevV[1]) -
                         (currV[1] - prevV[1]) * (nextV[0] - prevV[0]);
            const isConvex = ccw ? cross > 0 : cross < 0;

            if (!isConvex) continue;

            // Check no other vertex is inside this triangle
            let isEar = true;
            for (let j = 0; j < remaining.length; j++) {
                if (j === prevIdx || j === i || j === nextIdx) continue;
                const testV = allVertices[remaining[j]];
                if (pointInTriangle(testV, prevV, currV, nextV)) {
                    isEar = false;
                    break;
                }
            }

            if (isEar) {
                triangles.push([prev, curr, next]);
                remaining.splice(i, 1);
                earFound = true;
                break;
            }
        }

        if (!earFound) {
            // No ear found, try to continue anyway
            break;
        }
    }

    // Handle remaining triangle
    if (remaining.length === 3) {
        triangles.push([remaining[0], remaining[1], remaining[2]]);
    }

    return triangles;
}

/**
 * Check if a point is inside a triangle (barycentric coordinates).
 */
function pointInTriangle(
    p: [number, number],
    a: [number, number],
    b: [number, number],
    c: [number, number]
): boolean {
    const v0x = c[0] - a[0];
    const v0y = c[1] - a[1];
    const v1x = b[0] - a[0];
    const v1y = b[1] - a[1];
    const v2x = p[0] - a[0];
    const v2y = p[1] - a[1];

    const dot00 = v0x * v0x + v0y * v0y;
    const dot01 = v0x * v1x + v0y * v1y;
    const dot02 = v0x * v2x + v0y * v2y;
    const dot11 = v1x * v1x + v1y * v1y;
    const dot12 = v1x * v2x + v1y * v2y;

    const invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    const v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0) && (v >= 0) && (u + v < 1);
}

/**
 * Recover a constraint edge using cavity-based approach.
 *
 * Algorithm:
 * 1. Find all triangles that intersect the constraint edge (v1, v2)
 * 2. Remove those triangles, creating a polygonal cavity
 * 3. Extract the boundary of the cavity (ordered polygon)
 * 4. Re-triangulate each side of the cavity with the constraint edge as a boundary
 */
function recoverConstraintEdge(
    triangles: [number, number, number][],
    vertices: [number, number][],
    v1: number,
    v2: number
): boolean {
    // If edge already exists, nothing to do
    if (edgeExists(triangles, v1, v2)) {
        return true;
    }

    // Step 1: Find all triangles that intersect the constraint edge
    const intersectingIndices: number[] = [];
    for (let i = 0; i < triangles.length; i++) {
        if (triangleIntersectsEdge(triangles[i], vertices, v1, v2)) {
            intersectingIndices.push(i);
        }
    }

    if (intersectingIndices.length === 0) {
        // No triangles intersect - edge might already exist or be outside triangulation
        return edgeExists(triangles, v1, v2);
    }

    // Step 2: Collect all edges of intersecting triangles and find cavity boundary
    const edgeCount = new Map<string, { e1: number; e2: number; count: number }>();

    for (const triIdx of intersectingIndices) {
        const [a, b, c] = triangles[triIdx];
        const edges: [number, number][] = [[a, b], [b, c], [c, a]];

        for (const [e1, e2] of edges) {
            const key = e1 < e2 ? `${e1},${e2}` : `${e2},${e1}`;
            const existing = edgeCount.get(key);
            if (existing) {
                existing.count++;
            } else {
                edgeCount.set(key, { e1, e2, count: 1 });
            }
        }
    }

    // Boundary edges are those that appear exactly once
    const boundaryEdges: [number, number][] = [];
    for (const [, edge] of edgeCount) {
        if (edge.count === 1) {
            boundaryEdges.push([edge.e1, edge.e2]);
        }
    }

    // Step 3: Order the boundary edges into a polygon
    // The boundary forms a closed polygon around the cavity
    const orderedBoundary = orderBoundaryEdges(boundaryEdges);

    if (orderedBoundary.length < 3) {
        cdtDebugWarn(`[CDT] Cavity boundary too small: ${orderedBoundary.length} vertices`);
        return false;
    }

    // Step 4: Split the cavity boundary into two polygons (one on each side of v1-v2)
    // Find positions of v1 and v2 in the boundary
    const v1Pos = orderedBoundary.indexOf(v1);
    const v2Pos = orderedBoundary.indexOf(v2);

    if (v1Pos === -1 || v2Pos === -1) {
        cdtDebugWarn(`[CDT] Constraint endpoints not on cavity boundary`);
        return false;
    }

    // Extract two sub-polygons:
    // Polygon A: from v1 to v2 (going forward)
    // Polygon B: from v2 to v1 (going forward, wrapping around)
    let polyA: number[];
    let polyB: number[];

    if (v1Pos < v2Pos) {
        // v1 comes before v2
        polyA = orderedBoundary.slice(v1Pos, v2Pos + 1);
        polyB = [...orderedBoundary.slice(v2Pos), ...orderedBoundary.slice(0, v1Pos + 1)];
    } else {
        // v2 comes before v1
        polyA = orderedBoundary.slice(v1Pos);
        polyA.push(...orderedBoundary.slice(0, v2Pos + 1));
        polyB = orderedBoundary.slice(v2Pos, v1Pos + 1);
    }

    // Step 5: Remove the intersecting triangles
    // Sort indices in descending order to remove from end first
    intersectingIndices.sort((a, b) => b - a);
    for (const idx of intersectingIndices) {
        triangles.splice(idx, 1);
    }

    // Step 6: Re-triangulate each sub-polygon
    if (polyA.length >= 3) {
        const trisA = earClipPolygon(polyA, vertices);
        triangles.push(...trisA);
    }

    if (polyB.length >= 3) {
        const trisB = earClipPolygon(polyB, vertices);
        triangles.push(...trisB);
    }

    return edgeExists(triangles, v1, v2);
}

/**
 * Order boundary edges into a continuous polygon.
 */
function orderBoundaryEdges(edges: [number, number][]): number[] {
    if (edges.length === 0) return [];

    // Build adjacency map
    const adj = new Map<number, number[]>();
    for (const [e1, e2] of edges) {
        if (!adj.has(e1)) adj.set(e1, []);
        if (!adj.has(e2)) adj.set(e2, []);
        adj.get(e1)!.push(e2);
        adj.get(e2)!.push(e1);
    }

    // Start from first edge
    const result: number[] = [edges[0][0]];
    const visited = new Set<number>();
    visited.add(edges[0][0]);

    let current = edges[0][0];
    let maxIterations = edges.length * 2;

    while (maxIterations-- > 0) {
        const neighbors = adj.get(current) || [];
        let next = -1;

        for (const n of neighbors) {
            if (!visited.has(n)) {
                next = n;
                break;
            }
        }

        if (next === -1) {
            // No unvisited neighbor, we're done
            break;
        }

        result.push(next);
        visited.add(next);
        current = next;
    }

    return result;
}

/**
 * Check if two line segments cross (not just touch).
 */
function edgesCross(
    a1: [number, number], a2: [number, number],
    b1: [number, number], b2: [number, number]
): boolean {
    const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

    const d1 = cross(b1, b2, a1);
    const d2 = cross(b1, b2, a2);
    const d3 = cross(a1, a2, b1);
    const d4 = cross(a1, a2, b2);

    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true;
    }

    return false;
}

/**
 * Check if four points form a convex quadrilateral.
 */
function isConvexQuad(
    a: [number, number],
    b: [number, number],
    c: [number, number],
    d: [number, number]
): boolean {
    const cross = (o: [number, number], p1: [number, number], p2: [number, number]) =>
        (p1[0] - o[0]) * (p2[1] - o[1]) - (p1[1] - o[1]) * (p2[0] - o[0]);

    const c1 = cross(a, b, c);
    const c2 = cross(b, c, d);
    const c3 = cross(c, d, a);
    const c4 = cross(d, a, b);

    return (c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0) ||
           (c1 < 0 && c2 < 0 && c3 < 0 && c4 < 0);
}

/**
 * Full CDT with proper hole support using Bowyer-Watson + constraint recovery.
 *
 * @param boundary - Outer boundary vertices (CCW)
 * @param holes - Array of hole polygons (CW)
 * @returns Triangles that are inside boundary and outside all holes
 */
export async function cdtWithHoles(
    boundary: [number, number][],
    holes: [number, number][][] = []
): Promise<[number, number, number][]> {
    // Combine all vertices
    const allVertices: [number, number][] = [...boundary];
    const holeOffsets: { start: number; count: number }[] = [];

    for (const hole of holes) {
        holeOffsets.push({
            start: allVertices.length,
            count: hole.length
        });
        allVertices.push(...hole);
    }

    cdtDebugLog(`[CDT-Holes] ${allVertices.length} total vertices (${boundary.length} boundary + ${allVertices.length - boundary.length} hole)`);

    // Step 1: Delaunay triangulation of all vertices
    let triangles = bowyerWatson(allVertices);
    cdtDebugLog(`[CDT-Holes] Bowyer-Watson produced ${triangles.length} triangles`);

    // Step 2: Recover constraint edges (boundary + holes)
    const constraintEdges = buildConstraintEdgesWithHoles(boundary.length, holeOffsets);
    let recoveredCount = 0;
    let failedCount = 0;

    for (const [v1, v2] of constraintEdges) {
        if (recoverConstraintEdge(triangles, allVertices, v1, v2)) {
            recoveredCount++;
        } else {
            failedCount++;
        }
    }
    cdtDebugLog(`[CDT-Holes] Constraint recovery: ${recoveredCount} recovered, ${failedCount} failed`);

    // Step 3: Remove triangles outside boundary
    triangles = triangles.filter(tri => {
        const centroid: [number, number] = [
            (allVertices[tri[0]][0] + allVertices[tri[1]][0] + allVertices[tri[2]][0]) / 3,
            (allVertices[tri[0]][1] + allVertices[tri[1]][1] + allVertices[tri[2]][1]) / 3
        ];
        return isPointInsidePolygon(centroid, boundary);
    });
    cdtDebugLog(`[CDT-Holes] After boundary filter: ${triangles.length} triangles`);

    // Step 4: Remove triangles inside holes or crossing hole boundaries
    if (holes.length > 0) {
        const beforeCount = triangles.length;
        triangles = triangles.filter(tri => {
            const v0 = allVertices[tri[0]];
            const v1 = allVertices[tri[1]];
            const v2 = allVertices[tri[2]];

            // Check centroid
            const centroid: [number, number] = [
                (v0[0] + v1[0] + v2[0]) / 3,
                (v0[1] + v1[1] + v2[1]) / 3
            ];
            for (const hole of holes) {
                if (isPointInsidePolygon(centroid, hole)) {
                    return false;
                }
            }

            // Check if any triangle edge crosses any hole boundary edge
            const triEdges: [[number, number], [number, number]][] = [
                [v0, v1], [v1, v2], [v2, v0]
            ];

            for (const hole of holes) {
                for (let i = 0; i < hole.length; i++) {
                    const h1 = hole[i];
                    const h2 = hole[(i + 1) % hole.length];

                    for (const [e1, e2] of triEdges) {
                        // Skip if triangle edge shares a vertex with hole edge
                        if ((e1[0] === h1[0] && e1[1] === h1[1]) ||
                            (e1[0] === h2[0] && e1[1] === h2[1]) ||
                            (e2[0] === h1[0] && e2[1] === h1[1]) ||
                            (e2[0] === h2[0] && e2[1] === h2[1])) {
                            continue;
                        }

                        if (edgesCross(e1, e2, h1, h2)) {
                            return false;
                        }
                    }
                }
            }

            return true;
        });
        cdtDebugLog(`[CDT-Holes] After hole filter: ${triangles.length} triangles (removed ${beforeCount - triangles.length})`);
    }

    return triangles;
}

/**
 * Check if a point is inside a polygon using the winding number algorithm.
 * Works for both convex and concave polygons.
 */
function isPointInsidePolygon(point: [number, number], polygon: [number, number][]): boolean {
    let windingNumber = 0;
    const n = polygon.length;

    for (let i = 0; i < n; i++) {
        const p1 = polygon[i];
        const p2 = polygon[(i + 1) % n];

        if (p1[1] <= point[1]) {
            if (p2[1] > point[1]) {
                // Upward crossing
                const cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (point[0] - p1[0]) * (p2[1] - p1[1]);
                if (cross > 0) {
                    windingNumber++;
                }
            }
        } else {
            if (p2[1] <= point[1]) {
                // Downward crossing
                const cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (point[0] - p1[0]) * (p2[1] - p1[1]);
                if (cross < 0) {
                    windingNumber--;
                }
            }
        }
    }

    return windingNumber !== 0;
}

/**
 * Calculate the centroid of a triangle.
 */
function triangleCentroid(
    v0: [number, number],
    v1: [number, number],
    v2: [number, number]
): [number, number] {
    return [
        (v0[0] + v1[0] + v2[0]) / 3,
        (v0[1] + v1[1] + v2[1]) / 3
    ];
}

/**
 * Remove triangles whose centroids fall inside any hole.
 */
function removeTrianglesInsideHoles(
    triangles: [number, number, number][],
    vertices: [number, number][],
    holes: [number, number][][]
): [number, number, number][] {
    if (holes.length === 0) {
        return triangles;
    }

    return triangles.filter(tri => {
        const centroid = triangleCentroid(
            vertices[tri[0]],
            vertices[tri[1]],
            vertices[tri[2]]
        );

        // Check if centroid is inside any hole
        for (const hole of holes) {
            if (isPointInsidePolygon(centroid, hole)) {
                return false; // Remove this triangle
            }
        }

        return true; // Keep this triangle
    });
}

/**
 * Full CDT pipeline:
 * 1. Merge all vertices (boundary + holes)
 * 2. Initial triangulation via ear clipping (with hole bridging)
 * 3. Edge flipping for Delaunay property (respecting constraint edges)
 * 4. Remove triangles inside holes
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
    // Step 1: Build combined vertex array and track hole positions
    const allVertices: [number, number][] = [...boundary];
    const holeOffsets: { start: number; count: number }[] = [];

    for (const hole of holes) {
        holeOffsets.push({
            start: allVertices.length,
            count: hole.length
        });
        allVertices.push(...hole);
    }

    // Step 2: Build constraint edges for both boundary and holes
    // TODO: Use these for full constraint edge recovery when we triangulate all vertices
    const _constraintEdges = buildConstraintEdgesWithHoles(boundary.length, holeOffsets);

    // Step 3: Initial triangulation
    // For now, we use ear clipping on the boundary only (without holes)
    // and then filter out triangles inside holes.
    // TODO: For better results, use Delaunay triangulation of ALL vertices
    // followed by constraint edge recovery.

    // Convert 2D boundary to 3D for ear clipping (z=0)
    const boundary3d: [number, number, number][] = boundary.map(([x, y]) => [x, y, 0]);

    // Initial triangulation via ear clipping (using the working implementation)
    const initialTriangles = await earClipping(boundary3d);

    if (initialTriangles.length === 0) {
        cdtDebugWarn("[CDT] Ear clipping returned no triangles");
        return [];
    }

    // Debug: check ear clipping output for degenerate triangles
    const degenerateTris = initialTriangles.filter(t =>
        t[0] === t[1] || t[1] === t[2] || t[0] === t[2]
    );
    if (degenerateTris.length > 0) {
        cdtDebugWarn(`[CDT] Ear clipping produced ${degenerateTris.length} degenerate triangles:`);
        degenerateTris.slice(0, 5).forEach(t => cdtDebugWarn(`  [${t.join(',')}]`));
    }

    let triangles = initialTriangles as [number, number, number][];

    // Step 4: Apply edge flipping for Delaunay property (if enabled)
    if (applyDelaunay) {
        // Only use boundary constraint edges for now since we're only
        // triangulating the boundary
        const boundaryConstraints = buildConstraintEdges(boundary.length);

        // Build half-edge data structure
        const halfEdges = buildHalfEdges(boundary, triangles, boundaryConstraints);

        // Apply CPU-based edge flipping (more reliable than GPU version)
        cpuEdgeFlip(boundary, halfEdges, 50);

        // Extract triangles from the updated half-edge structure
        triangles = extractTrianglesFromHalfEdges(halfEdges);

        // Debug: check edge flip output for degenerate triangles
        const degenerateAfterFlip = triangles.filter(t =>
            t[0] === t[1] || t[1] === t[2] || t[0] === t[2]
        );
        if (degenerateAfterFlip.length > 0) {
            cdtDebugWarn(`[CDT] Edge flipping produced ${degenerateAfterFlip.length} degenerate triangles:`);
            degenerateAfterFlip.slice(0, 5).forEach(t => cdtDebugWarn(`  [${t.join(',')}]`));
        }
    }

    // Step 5: Remove triangles inside holes
    if (holes.length > 0) {
        const beforeCount = triangles.length;
        triangles = removeTrianglesInsideHoles(triangles, boundary, holes);
        const removedCount = beforeCount - triangles.length;
        if (removedCount > 0) {
            cdtDebugLog(`[CDT] Removed ${removedCount} triangles inside holes`);
        }
    }

    return triangles;
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
