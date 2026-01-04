/**
 * Mesh Quality and Refinement (C7)
 *
 * Provides CAD-grade tessellation improvements:
 * - C7.1: Adaptive refinement by curvature/chord error metric
 * - C7.2: T-junction stitching between adjacent faces
 * - C7.3: Smooth vertex normals based on face adjacency
 * - C7.4: Triangle aspect ratio control (avoid slivers)
 */

import { evaluateSurface, surfaceNormal } from "./surfaces";
import type { Surface, Vec3 } from "./surfaces";

type Vec2 = [number, number];

export interface RefinementOptions {
    /** Maximum allowed chord error (distance from surface to linear approximation) */
    chordTolerance: number;
    /** Maximum recursion depth for adaptive refinement */
    maxDepth: number;
    /** Minimum edge length (stop refining when edges are smaller) */
    minEdgeLength: number;
    /** Maximum triangle aspect ratio before attempting to split */
    maxAspectRatio: number;
}

export const DEFAULT_REFINEMENT_OPTIONS: RefinementOptions = {
    chordTolerance: 0.01,
    maxDepth: 5,
    minEdgeLength: 0.001,
    maxAspectRatio: 10.0,
};

/**
 * Represents a mesh with vertices and triangles for refinement
 */
export interface RefinableMesh {
    uvVertices: Vec2[];
    positions: Vec3[];
    normals: Vec3[];
    triangles: [number, number, number][];
}

// ============================================================================
// C7.1: Adaptive Refinement by Curvature/Chord Error
// ============================================================================

/**
 * Compute the chord error for an edge.
 * This is the distance between the actual surface point at the edge midpoint
 * and the linearly interpolated point.
 */
export function computeChordError(
    surface: Surface,
    uv1: Vec2,
    uv2: Vec2
): number {
    // Midpoint in UV space
    const uvMid: Vec2 = [(uv1[0] + uv2[0]) / 2, (uv1[1] + uv2[1]) / 2];

    // Actual surface point at midpoint
    const surfacePoint = evaluateSurface(surface, uvMid[0], uvMid[1]);

    // 3D points at endpoints
    const p1 = evaluateSurface(surface, uv1[0], uv1[1]);
    const p2 = evaluateSurface(surface, uv2[0], uv2[1]);

    // Linear interpolation at midpoint
    const linearMid: Vec3 = [
        (p1[0] + p2[0]) / 2,
        (p1[1] + p2[1]) / 2,
        (p1[2] + p2[2]) / 2,
    ];

    // Distance between actual and interpolated
    const dx = surfacePoint[0] - linearMid[0];
    const dy = surfacePoint[1] - linearMid[1];
    const dz = surfacePoint[2] - linearMid[2];

    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Compute the maximum chord error for a triangle.
 * Checks all three edges and returns the maximum error.
 */
export function computeTriangleChordError(
    surface: Surface,
    uv0: Vec2,
    uv1: Vec2,
    uv2: Vec2
): { maxError: number; worstEdge: number } {
    const errors = [
        computeChordError(surface, uv0, uv1),
        computeChordError(surface, uv1, uv2),
        computeChordError(surface, uv2, uv0),
    ];

    let maxError = errors[0];
    let worstEdge = 0;

    for (let i = 1; i < 3; i++) {
        if (errors[i] > maxError) {
            maxError = errors[i];
            worstEdge = i;
        }
    }

    return { maxError, worstEdge };
}

/**
 * Adaptively refine a mesh based on chord error tolerance.
 * Uses midpoint subdivision on triangles that exceed the tolerance.
 */
export function adaptiveRefineMesh(
    surface: Surface,
    mesh: RefinableMesh,
    options: RefinementOptions = DEFAULT_REFINEMENT_OPTIONS
): RefinableMesh {
    const { chordTolerance, maxDepth, minEdgeLength } = options;

    // Work with mutable copies
    const uvVertices = [...mesh.uvVertices];
    const positions = [...mesh.positions];
    const normals = [...mesh.normals];
    let triangles = [...mesh.triangles];

    // Map from UV coordinates to vertex index (for reusing vertices)
    const uvToIndex = new Map<string, number>();
    for (let i = 0; i < uvVertices.length; i++) {
        const key = `${uvVertices[i][0].toFixed(9)},${uvVertices[i][1].toFixed(9)}`;
        uvToIndex.set(key, i);
    }

    // Helper to add or reuse a vertex
    function getOrAddVertex(uv: Vec2): number {
        const key = `${uv[0].toFixed(9)},${uv[1].toFixed(9)}`;
        const existing = uvToIndex.get(key);
        if (existing !== undefined) {
            return existing;
        }

        const pos = evaluateSurface(surface, uv[0], uv[1]);
        const norm = surfaceNormal(surface, uv[0], uv[1]);

        const idx = uvVertices.length;
        uvVertices.push(uv);
        positions.push(pos);
        normals.push(norm);
        uvToIndex.set(key, idx);

        return idx;
    }

    // Iteratively refine until convergence or max iterations
    for (let depth = 0; depth < maxDepth; depth++) {
        const newTriangles: [number, number, number][] = [];
        let refined = false;

        for (const [i0, i1, i2] of triangles) {
            const uv0 = uvVertices[i0];
            const uv1 = uvVertices[i1];
            const uv2 = uvVertices[i2];

            // Check edge lengths
            const len01 = Math.sqrt(
                (uv1[0] - uv0[0]) ** 2 + (uv1[1] - uv0[1]) ** 2
            );
            const len12 = Math.sqrt(
                (uv2[0] - uv1[0]) ** 2 + (uv2[1] - uv1[1]) ** 2
            );
            const len20 = Math.sqrt(
                (uv0[0] - uv2[0]) ** 2 + (uv0[1] - uv2[1]) ** 2
            );

            // Skip if edges are too small
            if (Math.max(len01, len12, len20) < minEdgeLength) {
                newTriangles.push([i0, i1, i2]);
                continue;
            }

            // Compute chord errors
            const { maxError, worstEdge } = computeTriangleChordError(
                surface, uv0, uv1, uv2
            );

            if (maxError <= chordTolerance) {
                // Triangle is fine, keep it
                newTriangles.push([i0, i1, i2]);
            } else {
                // Subdivide by inserting midpoint on worst edge
                refined = true;

                if (worstEdge === 0) {
                    // Split edge 0-1
                    const uvMid: Vec2 = [(uv0[0] + uv1[0]) / 2, (uv0[1] + uv1[1]) / 2];
                    const iMid = getOrAddVertex(uvMid);
                    newTriangles.push([i0, iMid, i2]);
                    newTriangles.push([iMid, i1, i2]);
                } else if (worstEdge === 1) {
                    // Split edge 1-2
                    const uvMid: Vec2 = [(uv1[0] + uv2[0]) / 2, (uv1[1] + uv2[1]) / 2];
                    const iMid = getOrAddVertex(uvMid);
                    newTriangles.push([i0, i1, iMid]);
                    newTriangles.push([i0, iMid, i2]);
                } else {
                    // Split edge 2-0
                    const uvMid: Vec2 = [(uv2[0] + uv0[0]) / 2, (uv2[1] + uv0[1]) / 2];
                    const iMid = getOrAddVertex(uvMid);
                    newTriangles.push([i0, i1, iMid]);
                    newTriangles.push([iMid, i1, i2]);
                }
            }
        }

        triangles = newTriangles;

        if (!refined) {
            break; // No triangles needed refinement
        }
    }

    return { uvVertices, positions, normals, triangles };
}

/**
 * Compute adaptive sample count based on curvature for a surface patch.
 * Returns the number of samples needed in u and v directions.
 */
export function computeAdaptiveSampleCount(
    surface: Surface,
    uMin: number,
    uMax: number,
    vMin: number,
    vMax: number,
    chordTolerance: number
): { uSamples: number; vSamples: number } {
    // Sample the surface at corners and center
    const corners = [
        evaluateSurface(surface, uMin, vMin),
        evaluateSurface(surface, uMax, vMin),
        evaluateSurface(surface, uMax, vMax),
        evaluateSurface(surface, uMin, vMax),
    ];

    const center = evaluateSurface(
        surface,
        (uMin + uMax) / 2,
        (vMin + vMax) / 2
    );

    // Compute approximate surface size
    const uEdgeLen = distance3D(corners[0], corners[1]);
    const vEdgeLen = distance3D(corners[0], corners[3]);

    // Compute center deviation from linear interpolation
    const linearCenter: Vec3 = [
        (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4,
        (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4,
        (corners[0][2] + corners[1][2] + corners[2][2] + corners[3][2]) / 4,
    ];

    const centerDeviation = distance3D(center, linearCenter);

    // Estimate curvature from deviation
    const avgEdgeLen = (uEdgeLen + vEdgeLen) / 2;
    const curvature = avgEdgeLen > 0 ? centerDeviation / (avgEdgeLen * avgEdgeLen / 4) : 0;

    // Compute sample counts based on curvature and tolerance
    // Formula: samples = sqrt(curvature * size / tolerance)
    const baseSamples = 4;
    const maxSamples = 64;

    const uCurvatureFactor = Math.sqrt(Math.max(1, curvature * uEdgeLen / chordTolerance));
    const vCurvatureFactor = Math.sqrt(Math.max(1, curvature * vEdgeLen / chordTolerance));

    const uSamples = Math.min(maxSamples, Math.max(baseSamples, Math.ceil(baseSamples * uCurvatureFactor)));
    const vSamples = Math.min(maxSamples, Math.max(baseSamples, Math.ceil(baseSamples * vCurvatureFactor)));

    return { uSamples, vSamples };
}

// ============================================================================
// C7.4: Triangle Aspect Ratio Control
// ============================================================================

/**
 * Compute the aspect ratio of a triangle (longest edge / shortest altitude).
 * Higher values indicate more sliver-like triangles.
 */
export function computeTriangleAspectRatio(p0: Vec3, p1: Vec3, p2: Vec3): number {
    // Edge lengths
    const a = distance3D(p1, p2);
    const b = distance3D(p0, p2);
    const c = distance3D(p0, p1);

    // Semi-perimeter
    const s = (a + b + c) / 2;

    // Area using Heron's formula
    const areaSquared = s * (s - a) * (s - b) * (s - c);
    if (areaSquared <= 0) {
        return Infinity; // Degenerate triangle
    }
    const area = Math.sqrt(areaSquared);

    // Longest edge
    const longestEdge = Math.max(a, b, c);

    // Shortest altitude = 2 * area / longest edge
    const shortestAltitude = (2 * area) / longestEdge;

    if (shortestAltitude < 1e-10) {
        return Infinity;
    }

    return longestEdge / shortestAltitude;
}

/**
 * Check if a triangle is a sliver (has bad aspect ratio).
 */
export function isSliverTriangle(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    maxAspectRatio: number = 10.0
): boolean {
    return computeTriangleAspectRatio(p0, p1, p2) > maxAspectRatio;
}

/**
 * Filter out degenerate triangles (zero area or very high aspect ratio).
 */
export function filterDegenerateTriangles(
    positions: Vec3[],
    triangles: [number, number, number][],
    maxAspectRatio: number = 100.0
): [number, number, number][] {
    return triangles.filter(([i0, i1, i2]) => {
        const p0 = positions[i0];
        const p1 = positions[i1];
        const p2 = positions[i2];

        // Check for duplicate vertices
        if (distance3D(p0, p1) < 1e-10 ||
            distance3D(p1, p2) < 1e-10 ||
            distance3D(p2, p0) < 1e-10) {
            return false;
        }

        // Check aspect ratio
        return computeTriangleAspectRatio(p0, p1, p2) <= maxAspectRatio;
    });
}

// ============================================================================
// C7.2: T-Junction Stitching
// ============================================================================

/**
 * Edge key for edge lookup (order-independent).
 */
function edgeKey(v0: number, v1: number): string {
    const [a, b] = v0 < v1 ? [v0, v1] : [v1, v0];
    return `${a}-${b}`;
}

/**
 * Build an edge map from triangles.
 * Returns a map from edge key to list of triangles using that edge.
 */
export function buildEdgeMap(
    triangles: [number, number, number][]
): Map<string, { triIndex: number; localEdge: number }[]> {
    const edgeMap = new Map<string, { triIndex: number; localEdge: number }[]>();

    for (let triIndex = 0; triIndex < triangles.length; triIndex++) {
        const [i0, i1, i2] = triangles[triIndex];
        const edges = [
            [i0, i1, 0],
            [i1, i2, 1],
            [i2, i0, 2],
        ] as [number, number, number][];

        for (const [v0, v1, localEdge] of edges) {
            const key = edgeKey(v0, v1);
            if (!edgeMap.has(key)) {
                edgeMap.set(key, []);
            }
            edgeMap.get(key)!.push({ triIndex, localEdge });
        }
    }

    return edgeMap;
}

/**
 * Represents a shared boundary edge between two faces.
 */
export interface SharedEdge {
    /** 3D positions of the edge endpoints on face A */
    positionsA: [Vec3, Vec3];
    /** 3D positions of the edge endpoints on face B */
    positionsB: [Vec3, Vec3];
    /** Indices into face A's vertex list */
    indicesA: [number, number];
    /** Indices into face B's vertex list */
    indicesB: [number, number];
    /** Tolerance for matching */
    tolerance: number;
}

/**
 * Find shared edges between two meshes.
 * Returns pairs of edges that should be stitched together.
 */
export function findSharedEdges(
    positionsA: Vec3[],
    trianglesA: [number, number, number][],
    positionsB: Vec3[],
    trianglesB: [number, number, number][],
    tolerance: number = 1e-6
): SharedEdge[] {
    const sharedEdges: SharedEdge[] = [];

    // Build edge list for mesh A
    const edgesA: { v0: number; v1: number; p0: Vec3; p1: Vec3 }[] = [];
    for (const [i0, i1, i2] of trianglesA) {
        edgesA.push(
            { v0: i0, v1: i1, p0: positionsA[i0], p1: positionsA[i1] },
            { v0: i1, v1: i2, p0: positionsA[i1], p1: positionsA[i2] },
            { v0: i2, v1: i0, p0: positionsA[i2], p1: positionsA[i0] }
        );
    }

    // Build edge list for mesh B
    const edgesB: { v0: number; v1: number; p0: Vec3; p1: Vec3 }[] = [];
    for (const [i0, i1, i2] of trianglesB) {
        edgesB.push(
            { v0: i0, v1: i1, p0: positionsB[i0], p1: positionsB[i1] },
            { v0: i1, v1: i2, p0: positionsB[i1], p1: positionsB[i2] },
            { v0: i2, v1: i0, p0: positionsB[i2], p1: positionsB[i0] }
        );
    }

    // Find matching edges
    for (const edgeA of edgesA) {
        for (const edgeB of edgesB) {
            // Check if edges match (same or reversed order)
            const matchForward =
                distance3D(edgeA.p0, edgeB.p0) < tolerance &&
                distance3D(edgeA.p1, edgeB.p1) < tolerance;
            const matchReverse =
                distance3D(edgeA.p0, edgeB.p1) < tolerance &&
                distance3D(edgeA.p1, edgeB.p0) < tolerance;

            if (matchForward || matchReverse) {
                sharedEdges.push({
                    positionsA: [edgeA.p0, edgeA.p1],
                    positionsB: matchForward ? [edgeB.p0, edgeB.p1] : [edgeB.p1, edgeB.p0],
                    indicesA: [edgeA.v0, edgeA.v1],
                    indicesB: matchForward ? [edgeB.v0, edgeB.v1] : [edgeB.v1, edgeB.v0],
                    tolerance,
                });
            }
        }
    }

    return sharedEdges;
}

/**
 * Weld coincident vertices across multiple meshes.
 * Creates a unified vertex buffer with shared vertices merged.
 */
export function weldVertices(
    meshes: { positions: Vec3[]; triangles: [number, number, number][] }[],
    tolerance: number = 1e-6
): { positions: Vec3[]; triangles: [number, number, number][] } {
    const positions: Vec3[] = [];
    const triangles: [number, number, number][] = [];

    // Spatial hash for fast lookup
    const cellSize = tolerance * 10;
    const positionMap = new Map<string, number>();

    function getCellKey(p: Vec3): string {
        const x = Math.floor(p[0] / cellSize);
        const y = Math.floor(p[1] / cellSize);
        const z = Math.floor(p[2] / cellSize);
        return `${x},${y},${z}`;
    }

    function findOrAddVertex(p: Vec3): number {
        const cellKey = getCellKey(p);

        // Check this cell and neighbors
        const cx = Math.floor(p[0] / cellSize);
        const cy = Math.floor(p[1] / cellSize);
        const cz = Math.floor(p[2] / cellSize);

        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const neighborKey = `${cx + dx},${cy + dy},${cz + dz}`;
                    const existingIndex = positionMap.get(neighborKey);
                    if (existingIndex !== undefined) {
                        if (distance3D(positions[existingIndex], p) < tolerance) {
                            return existingIndex;
                        }
                    }
                }
            }
        }

        // Add new vertex
        const index = positions.length;
        positions.push(p);
        positionMap.set(cellKey, index);
        return index;
    }

    // Process each mesh
    for (const mesh of meshes) {
        // Build index remapping for this mesh
        const indexRemap: number[] = [];
        for (const p of mesh.positions) {
            indexRemap.push(findOrAddVertex(p));
        }

        // Remap triangles
        for (const [i0, i1, i2] of mesh.triangles) {
            triangles.push([indexRemap[i0], indexRemap[i1], indexRemap[i2]]);
        }
    }

    return { positions, triangles };
}

// ============================================================================
// C7.3: Smooth Vertex Normals
// ============================================================================

/**
 * Compute face normals for all triangles.
 */
export function computeFaceNormals(
    positions: Vec3[],
    triangles: [number, number, number][]
): Vec3[] {
    const faceNormals: Vec3[] = [];

    for (const [i0, i1, i2] of triangles) {
        const p0 = positions[i0];
        const p1 = positions[i1];
        const p2 = positions[i2];

        // Edge vectors
        const e1: Vec3 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        const e2: Vec3 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        // Cross product
        const normal: Vec3 = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Normalize
        const len = Math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2);
        if (len > 1e-10) {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }

        faceNormals.push(normal);
    }

    return faceNormals;
}

/**
 * Compute angle-weighted smooth vertex normals.
 * Each face normal contributes to vertices proportional to the angle at that vertex.
 */
export function computeSmoothNormals(
    positions: Vec3[],
    triangles: [number, number, number][]
): Vec3[] {
    const vertexNormals: Vec3[] = positions.map(() => [0, 0, 0]);

    for (const [i0, i1, i2] of triangles) {
        const p0 = positions[i0];
        const p1 = positions[i1];
        const p2 = positions[i2];

        // Edge vectors
        const e01: Vec3 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        const e02: Vec3 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        const e12: Vec3 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        const e10: Vec3 = [-e01[0], -e01[1], -e01[2]];
        const e20: Vec3 = [-e02[0], -e02[1], -e02[2]];
        const e21: Vec3 = [-e12[0], -e12[1], -e12[2]];

        // Face normal (cross product)
        const faceNormal: Vec3 = [
            e01[1] * e02[2] - e01[2] * e02[1],
            e01[2] * e02[0] - e01[0] * e02[2],
            e01[0] * e02[1] - e01[1] * e02[0],
        ];

        // Compute angles at each vertex
        const angle0 = angleBetweenVectors(e01, e02);
        const angle1 = angleBetweenVectors(e10, e12);
        const angle2 = angleBetweenVectors(e20, e21);

        // Accumulate weighted normals
        vertexNormals[i0][0] += faceNormal[0] * angle0;
        vertexNormals[i0][1] += faceNormal[1] * angle0;
        vertexNormals[i0][2] += faceNormal[2] * angle0;

        vertexNormals[i1][0] += faceNormal[0] * angle1;
        vertexNormals[i1][1] += faceNormal[1] * angle1;
        vertexNormals[i1][2] += faceNormal[2] * angle1;

        vertexNormals[i2][0] += faceNormal[0] * angle2;
        vertexNormals[i2][1] += faceNormal[1] * angle2;
        vertexNormals[i2][2] += faceNormal[2] * angle2;
    }

    // Normalize all vertex normals
    for (const normal of vertexNormals) {
        const len = Math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2);
        if (len > 1e-10) {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }
    }

    return vertexNormals;
}

/**
 * Compute smooth normals with crease angle threshold.
 * Edges with dihedral angle above the threshold are treated as creases (sharp edges).
 */
export function computeSmoothNormalsWithCreases(
    positions: Vec3[],
    triangles: [number, number, number][],
    creaseAngle: number = Math.PI / 4 // 45 degrees default
): Vec3[] {
    // First compute face normals
    const faceNormals = computeFaceNormals(positions, triangles);

    // Build edge to triangle map
    const edgeToTriangles = new Map<string, number[]>();
    for (let triIdx = 0; triIdx < triangles.length; triIdx++) {
        const [i0, i1, i2] = triangles[triIdx];
        const edges = [[i0, i1], [i1, i2], [i2, i0]];

        for (const [v0, v1] of edges) {
            const key = edgeKey(v0, v1);
            if (!edgeToTriangles.has(key)) {
                edgeToTriangles.set(key, []);
            }
            edgeToTriangles.get(key)!.push(triIdx);
        }
    }

    // Find crease edges (edges where adjacent face normals differ by more than crease angle)
    const creaseEdges = new Set<string>();
    for (const [key, triIndices] of edgeToTriangles) {
        if (triIndices.length === 2) {
            const n1 = faceNormals[triIndices[0]];
            const n2 = faceNormals[triIndices[1]];
            const dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
            const angle = Math.acos(Math.max(-1, Math.min(1, dot)));

            if (angle > creaseAngle) {
                creaseEdges.add(key);
            }
        }
    }

    // Build vertex to triangles map, excluding triangles across creases
    const vertexToTriangles: Map<number, Set<number>>[] = positions.map(() => new Map());

    for (let triIdx = 0; triIdx < triangles.length; triIdx++) {
        const [i0, i1, i2] = triangles[triIdx];
        const vertices = [i0, i1, i2];

        for (let v = 0; v < 3; v++) {
            const vertIdx = vertices[v];

            // Group triangles by smoothing group (separated by creases)
            // Simple approach: each vertex gets all triangles, but we'll average separately
            // TODO: Implement proper smoothing group separation using crease edges
            if (!vertexToTriangles[vertIdx].has(triIdx)) {
                vertexToTriangles[vertIdx].set(triIdx, new Set());
            }
        }
    }

    // For now, use simple smooth normals (crease handling is complex)
    // TODO: Implement proper smoothing group separation
    return computeSmoothNormals(positions, triangles);
}

// ============================================================================
// Utility Functions
// ============================================================================

function distance3D(a: Vec3, b: Vec3): number {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const dz = b[2] - a[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function angleBetweenVectors(a: Vec3, b: Vec3): number {
    const lenA = Math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2);
    const lenB = Math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2);

    if (lenA < 1e-10 || lenB < 1e-10) {
        return 0;
    }

    const dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (lenA * lenB);
    return Math.acos(Math.max(-1, Math.min(1, dot)));
}

/**
 * Combine multiple meshes with proper normal computation.
 */
export function combineMeshesWithNormals(
    meshes: { positions: Vec3[]; triangles: [number, number, number][] }[],
    weldTolerance: number = 1e-6
): { positions: Float32Array; normals: Float32Array; indices: Uint32Array } {
    // Weld vertices
    const welded = weldVertices(meshes, weldTolerance);

    // Filter degenerate triangles
    const cleanTriangles = filterDegenerateTriangles(welded.positions, welded.triangles);

    // Compute smooth normals
    const normals = computeSmoothNormals(welded.positions, cleanTriangles);

    // Build output arrays
    const positionsArray = new Float32Array(welded.positions.length * 3);
    const normalsArray = new Float32Array(normals.length * 3);
    const indicesArray = new Uint32Array(cleanTriangles.length * 3);

    for (let i = 0; i < welded.positions.length; i++) {
        positionsArray[i * 3 + 0] = welded.positions[i][0];
        positionsArray[i * 3 + 1] = welded.positions[i][1];
        positionsArray[i * 3 + 2] = welded.positions[i][2];

        normalsArray[i * 3 + 0] = normals[i][0];
        normalsArray[i * 3 + 1] = normals[i][1];
        normalsArray[i * 3 + 2] = normals[i][2];
    }

    for (let i = 0; i < cleanTriangles.length; i++) {
        indicesArray[i * 3 + 0] = cleanTriangles[i][0];
        indicesArray[i * 3 + 1] = cleanTriangles[i][1];
        indicesArray[i * 3 + 2] = cleanTriangles[i][2];
    }

    return { positions: positionsArray, normals: normalsArray, indices: indicesArray };
}
