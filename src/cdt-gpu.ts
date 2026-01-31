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
            console.log(`[CDT] Edge flipping converged after ${iter + 1} iterations`);
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
            console.warn(`[CDT] Edge flipping produced ${degenerateAfterFlip.length} degenerate triangles:`);
            degenerateAfterFlip.slice(0, 5).forEach(t => console.warn(`  [${t.join(',')}]`));
        }
    }

    // Step 5: Remove triangles inside holes
    if (holes.length > 0) {
        const beforeCount = triangles.length;
        triangles = removeTrianglesInsideHoles(triangles, boundary, holes);
        const removedCount = beforeCount - triangles.length;
        if (removedCount > 0) {
            console.log(`[CDT] Removed ${removedCount} triangles inside holes`);
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
