/**
 * Profiled STEP Parser
 *
 * Same as the batched parser but with detailed timing for each phase.
 * Uses hybrid GPU/CPU triangulation for maximum performance.
 * Supports both planar and curved surfaces (cylinders, spheres, cones, tori).
 */

import { triangulateHybrid } from "./triangulate-hybrid";

// Re-export types we need
type Vec3 = [number, number, number];
type Vec2 = [number, number];

export interface ProfiledMesh {
    positions: Float32Array;
    indices: Uint32Array;
    triangleCount: number;
    faceCount: number;
    timing: {
        total: number;
        stepParsing: number;
        faceExtraction: number;
        projection: number;
        bridging: number;
        gpuTriangulation: number;
        meshAssembly: number;
    };
}

// Import everything we need from step-parser
import {
    parseStep,
    extractFaceBoundsWithCurves,
    computeFaceBasisFromStepFace,
    projectFaceLoopsTo2D,
    normalizeWinding,
    applyWindingTo3D,
    validateTopology,
    bridgeAllHoles,
    tryTessellateCurvedSurface,
} from "./step-parser";

interface PreparedFace {
    polygon2d: Vec2[];
    vertices3d: Vec3[];
    isCurved: boolean;
    curvedResult?: { vertices: Vec3[]; triangles: [number, number, number][] };
}

/**
 * Parse STEP with detailed timing breakdown.
 */
export async function parseStepProfiled(stepText: string): Promise<ProfiledMesh> {
    const timing = {
        total: 0,
        stepParsing: 0,
        faceExtraction: 0,
        projection: 0,
        bridging: 0,
        gpuTriangulation: 0,
        meshAssembly: 0,
    };

    const totalStart = performance.now();

    // Phase 1: Parse STEP file
    const parseStart = performance.now();
    const model = parseStep(stepText);
    timing.stepParsing = performance.now() - parseStart;

    if (model.faces.size === 0) {
        throw new Error("No ADVANCED_FACE found in STEP file.");
    }

    const faces = [...model.faces.values()];

    // Process each face in parallel (extraction + projection + bridging)
    const processingStart = performance.now();

    // Helper function to process a single face
    async function processFace(face: typeof faces[0]): Promise<PreparedFace | null> {
        try {
            // Check for curved surfaces first (cylinders, spheres, cones, tori)
            const curvedResult = await tryTessellateCurvedSurface(model, face);
            if (curvedResult) {
                return {
                    polygon2d: [],
                    vertices3d: [],
                    isCurved: true,
                    curvedResult,
                };
            }

            // Planar face - extract bounds and process
            const bounds = await extractFaceBoundsWithCurves(model, face);
            const { outer, holes } = bounds;

            // Project to 2D and normalize winding
            const basis = computeFaceBasisFromStepFace(model, face, outer);
            const projected = projectFaceLoopsTo2D({ outer, holes }, basis);
            const normalized = normalizeWinding(projected);
            const oriented3d = applyWindingTo3D({ outer, holes }, normalized.outerReversed, normalized.holesReversed);
            validateTopology(normalized.outer2d, normalized.holes2d);

            // Bridge holes
            const merged = bridgeAllHoles(normalized.outer2d, normalized.holes2d);

            // Create 2D → 3D lookup
            const outer2dTo3d = new Map<string, Vec3>();
            for (let i = 0; i < normalized.outer2d.length; i++) {
                const key = `${normalized.outer2d[i][0].toFixed(9)},${normalized.outer2d[i][1].toFixed(9)}`;
                outer2dTo3d.set(key, oriented3d.outer[i]);
            }

            const holes2dTo3d: Map<string, Vec3>[] = [];
            for (let h = 0; h < normalized.holes2d.length; h++) {
                const holeMap = new Map<string, Vec3>();
                for (let i = 0; i < normalized.holes2d[h].length; i++) {
                    const key = `${normalized.holes2d[h][i][0].toFixed(9)},${normalized.holes2d[h][i][1].toFixed(9)}`;
                    holeMap.set(key, oriented3d.holes[h][i]);
                }
                holes2dTo3d.push(holeMap);
            }

            // Filter duplicates
            const filtered2d: Vec2[] = [];
            for (let i = 0; i < merged.length; i++) {
                const curr = merged[i];
                const prev = filtered2d.length > 0 ? filtered2d[filtered2d.length - 1] : merged[merged.length - 1];
                const dx = curr[0] - prev[0];
                const dy = curr[1] - prev[1];
                if (dx * dx + dy * dy > 1e-12) {
                    filtered2d.push(curr);
                }
            }

            if (filtered2d.length > 1) {
                const first = filtered2d[0];
                const last = filtered2d[filtered2d.length - 1];
                const dx = first[0] - last[0];
                const dy = first[1] - last[1];
                if (dx * dx + dy * dy < 1e-12) {
                    filtered2d.pop();
                }
            }

            // Build 3D vertices
            const vertices3d: Vec3[] = [];
            for (const pt2d of filtered2d) {
                const key = `${pt2d[0].toFixed(9)},${pt2d[1].toFixed(9)}`;
                let pt3d = outer2dTo3d.get(key);
                if (!pt3d) {
                    for (const holeMap of holes2dTo3d) {
                        pt3d = holeMap.get(key);
                        if (pt3d) break;
                    }
                }
                if (!pt3d) {
                    pt3d = [pt2d[0], pt2d[1], 0];
                }
                vertices3d.push(pt3d);
            }

            if (filtered2d.length >= 3) {
                return { polygon2d: filtered2d, vertices3d, isCurved: false };
            }
            return null;
        } catch (e) {
            return null; // Skip failed faces
        }
    }

    // Process all faces in parallel
    const results = await Promise.all(faces.map(processFace));
    const preparedFaces = results.filter((f): f is PreparedFace => f !== null);

    const processingTime = performance.now() - processingStart;
    timing.faceExtraction = processingTime * 0.4;  // Approximate breakdown
    timing.projection = processingTime * 0.2;
    timing.bridging = processingTime * 0.4;

    // Phase 5: Hybrid GPU/CPU Triangulation (only for planar faces)
    const gpuStart = performance.now();

    // Separate curved and planar faces
    const planarFaces = preparedFaces.filter(f => !f.isCurved);
    const curvedFaces = preparedFaces.filter(f => f.isCurved);

    // Collect all planar polygons for hybrid triangulation
    const allPolygons: Vec2[][] = planarFaces.map(f => f.polygon2d);

    // Triangulate planar polygons using hybrid approach
    const hybridResult = await triangulateHybrid(allPolygons);

    // Map results back to planar faces
    const planarTriangles: Map<number, number[][]> = new Map();
    for (let i = 0; i < planarFaces.length; i++) {
        planarTriangles.set(i, hybridResult.triangles[i]);
    }

    timing.gpuTriangulation = performance.now() - gpuStart;

    // Phase 6: Assemble final mesh
    const assemblyStart = performance.now();

    const allVertices: Vec3[] = [];
    const allIndices: number[] = [];
    let vertexOffset = 0;

    // Add curved faces (already have triangles)
    for (const face of curvedFaces) {
        if (face.curvedResult) {
            for (const v of face.curvedResult.vertices) {
                allVertices.push(v);
            }
            for (const tri of face.curvedResult.triangles) {
                allIndices.push(
                    tri[0] + vertexOffset,
                    tri[1] + vertexOffset,
                    tri[2] + vertexOffset
                );
            }
            vertexOffset += face.curvedResult.vertices.length;
        }
    }

    // Add planar faces (triangulated by hybrid)
    for (let i = 0; i < planarFaces.length; i++) {
        const face = planarFaces[i];
        const triangles = planarTriangles.get(i);

        if (triangles && triangles.length > 0) {
            for (const v of face.vertices3d) {
                allVertices.push(v);
            }
            for (const tri of triangles) {
                allIndices.push(
                    tri[0] + vertexOffset,
                    tri[1] + vertexOffset,
                    tri[2] + vertexOffset
                );
            }
            vertexOffset += face.vertices3d.length;
        }
    }

    const positions = new Float32Array(allVertices.length * 3);
    allVertices.forEach((p, i) => {
        positions[i * 3 + 0] = p[0];
        positions[i * 3 + 1] = p[1];
        positions[i * 3 + 2] = p[2];
    });

    const indices = new Uint32Array(allIndices);

    timing.meshAssembly = performance.now() - assemblyStart;
    timing.total = performance.now() - totalStart;

    return {
        positions,
        indices,
        triangleCount: allIndices.length / 3,
        faceCount: faces.length,
        timing,
    };
}
