/**
 * Surface Tessellation Pipeline
 *
 * Tessellates parametric surfaces (cylinders, spheres, etc.) by:
 * 1. Creating UV boundary from surface patch parameters
 * 2. Triangulating in UV space using CDT
 * 3. Evaluating UV vertices back to 3D positions
 * 4. Computing surface normals
 *
 * C7 additions:
 * - Adaptive refinement by curvature/chord error (C7.1)
 * - Smooth vertex normals (C7.3)
 * - Triangle aspect ratio control (C7.4)
 */

import { constrainedDelaunayTriangulation, cdtWithHoles } from "./cdt-gpu";
import { triangulateWithHoles } from "./triangulate-fast";
import {
    classifyTrimGridGPU,
    buildTrimGridTrianglesGPU,
    classifyAndBuildTrimGridTrianglesGPU,
} from "./trim-grid-gpu";
import { evaluateSurface, surfaceNormal } from "./surfaces";
import { evaluateSurfaceMeshGPU, evaluateSurfaceDenseGridGPU } from "./surface-eval-gpu";
import { createRectangularUVBoundary } from "./uv-extraction";
import {
    adaptiveRefineMesh,
    computeAdaptiveSampleCount,
    filterDegenerateTriangles,
} from "./mesh-quality";
import type { RefinementOptions } from "./mesh-quality";

// Define surface types locally to avoid Vite import issues
type Vec3 = [number, number, number];

interface Axis2Placement3D {
    location: Vec3;
    axis: Vec3;
    refDirection: Vec3;
}

interface PlaneSurface {
    type: "PLANE";
    placement: Axis2Placement3D;
}

interface CylindricalSurface {
    type: "CYLINDRICAL_SURFACE";
    placement: Axis2Placement3D;
    radius: number;
}

interface SphericalSurface {
    type: "SPHERICAL_SURFACE";
    placement: Axis2Placement3D;
    radius: number;
}

interface ConicalSurface {
    type: "CONICAL_SURFACE";
    placement: Axis2Placement3D;
    radius: number;
    semiAngle: number;
}

interface ToroidalSurface {
    type: "TOROIDAL_SURFACE";
    placement: Axis2Placement3D;
    majorRadius: number;
    minorRadius: number;
}

interface BSplineSurfaceLocal {
    type: "B_SPLINE_SURFACE";
    controlPoints: Vec3[][];
    uDegree: number;
    vDegree: number;
    uKnots: number[];
    vKnots: number[];
    weights?: number[][];
}

type Surface =
    | PlaneSurface
    | CylindricalSurface
    | SphericalSurface
    | ConicalSurface
    | ToroidalSurface
    | BSplineSurfaceLocal;

type Vec2 = [number, number];

interface TessellatedMesh {
    positions: Float32Array;   // 3D vertex positions
    normals: Float32Array;     // Vertex normals
    indices: Uint32Array;      // Triangle indices
    uvs: Float32Array;         // UV coordinates (for texturing)
}

export type LocalUvClassifierMode = "shadow" | "candidate";

export interface LocalUvClassifierEdgeInput {
    points: Vec2[];
    // Optional max tolerance observed on edge end vertices.
    vertexToleranceMax?: number;
    // Hint that this edge should be treated as low-confidence in Stage-A.
    hasHighVertexTolerance?: boolean;
    // Hint that edge sampling produced a degenerate chain.
    isDegenerate?: boolean;
}

export interface LocalUvClassifierWireInput {
    loop: Vec2[];
    orientationBit: 0 | 1;
    edges?: LocalUvClassifierEdgeInput[];
    hasBadEdges?: boolean;
}

export interface LocalUvClassifierSummary {
    mode: LocalUvClassifierMode;
    faceIndex?: number;
    surfaceType?: string;
    buildLabel?: string;
    gridPoints: number;
    occDecisions: number;
    localInside: number;
    localOutside: number;
    localUncertain: number;
    localFallbackCalls: number;
    boundaryBandSamples: number;
    seamProximateSamples: number;
    mismatchCount: number;
    effectiveMismatchCount: number;
    mismatchBoundaryBand: number;
    mismatchInterior: number;
    mismatchSeamProximate: number;
    mismatchNonSeam: number;
    falseInsideCount: number;
    falseOutsideCount: number;
    stageAEvaluations: number;
    stageAInside: number;
    stageAOutside: number;
    stageAUncertain: number;
    stageABadWire: number;
    stageBEvaluations: number;
    stageBForcedEvaluations: number;
    stageBTriggeredByUncertain: number;
    stageBTriggeredByBadWire: number;
    stageBInside: number;
    stageBOutside: number;
    stageBUncertain: number;
    stageBResolvedByUncertain: number;
    stageBResolvedByBadWire: number;
    stageBProbeFallbacks: number;
    stageBBundleSkips: number;
    stageBSkippedNearVertexHits: number;
    stageBTransitionTies: number;
    decisionFromStageA: number;
    decisionFromStageB: number;
    decisionFromStageBForced: number;
    decisionFromDomainUnsafeStageA: number;
    decisionFromDomainUnsafeStageB: number;
    mismatchFromStageA: number;
    mismatchFromStageB: number;
    mismatchFromStageBForced: number;
    mismatchFromDomainUnsafe: number;
    mismatchFromDomainUnsafeStageA: number;
    mismatchFromDomainUnsafeStageB: number;
    mismatchDomainStageBOutside: number;
    mismatchDomainStageBWithTransitionTie: number;
    mismatchDomainStageBWithProbeFallback: number;
    mismatchDomainStageBNearBoundary: number;
    mismatchDomainStageBWithStageAInsideHint: number;
    mismatchDomainStageBWithStageAOutsideHint: number;
    mismatchDomainStageBWithStageAUncertainHint: number;
}

export interface TrimmedSurfaceBuildOptions {
    // Optional classifier used as the source of truth for UV inclusion.
    // Return true for points that should be considered inside/on the face.
    uvInsideTest?: (u: number, v: number) => boolean;
    // Optional triangle gate using UV samples (typically 7-point sampling).
    // Return true to keep triangle, false to drop it.
    keepTriangle?: (samples: ReadonlyArray<[number, number]>) => boolean;
    // If false, only emit full quad triangles (skip partial boundary quads).
    allowPartialCellTriangles?: boolean;
    // Optional anisotropic grid density overrides for trimmed-grid tessellation.
    gridDensityU?: number;
    gridDensityV?: number;
    // If true, skip CDT-with-holes and use the trimmed grid path even when holes exist.
    preferGridForHoles?: boolean;
    // Optional label used in logs.
    logLabel?: string;
    // Optional profiling hook for trimmed-surface sub-phases.
    recordProfileSample?: (phase: TrimmedSurfaceProfilePhase, elapsedMs: number) => void;
    // Optional callback to report local-vs-OCC classifier summary (shadow/candidate modes).
    recordLocalUvClassifierSummary?: (summary: LocalUvClassifierSummary) => void;
    // Optional wire-level UV loops (typically from OCC pcurves) used by the local classifier.
    // When provided, these loops are preferred over the trimmed patch loops for shadow/candidate checks.
    localUvClassifierWires?: LocalUvClassifierWireInput[];
    // Optional metadata used in classifier summary records.
    classifierFaceIndex?: number;
    classifierSurfaceType?: string;
}

export type TrimmedSurfaceProfilePhase =
    | "hole_triangulation"
    | "hole_triangle_gate"
    | "hole_evaluate_mesh"
    | "gpu_classify_build"
    | "gpu_dense_eval"
    | "gpu_mask_classify"
    | "gpu_mask_triangles"
    | "cpu_grid_classify"
    | "cpu_triangle_build"
    | "final_evaluate_mesh"
    | "uvmesh_gpu_eval"
    | "uvmesh_cpu_eval";

function trimDebugLog(...args: unknown[]): void {
    if ((globalThis as any)?.__TRIM_VERBOSE_LOGS__ === true) {
        console.log(...args);
    }
}

function trimDebugEnabled(): boolean {
    return (globalThis as any)?.__TRIM_VERBOSE_LOGS__ === true;
}

function readTrimNumber(key: string): number | null {
    const raw = (globalThis as any)?.[key];
    if (typeof raw !== "number" || !Number.isFinite(raw)) {
        return null;
    }
    return raw;
}

function readTrimBoolean(key: string, fallback: boolean): boolean {
    const raw = (globalThis as any)?.[key];
    return typeof raw === "boolean" ? raw : fallback;
}

function recordTrimProfileSample(
    buildOptions: TrimmedSurfaceBuildOptions | undefined,
    phase: TrimmedSurfaceProfilePhase,
    elapsedMs: number
): void {
    if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
        return;
    }
    try {
        buildOptions?.recordProfileSample?.(phase, elapsedMs);
    } catch {
        // Profiling hooks are best-effort and must never affect tessellation.
    }
}

/**
 * Tessellate a cylindrical surface patch
 */
export async function tessellateCylinder(
    surface: CylindricalSurface,
    angleStart: number = 0,
    angleEnd: number = Math.PI * 2,
    heightStart: number = 0,
    heightEnd: number = 1,
    numAngleSamples: number = 16,
    numHeightSamples: number = 2
): Promise<TessellatedMesh> {
    // Create a full UV grid (not just boundary)
    const uvVertices: [number, number][] = [];
    const triangles: [number, number, number][] = [];

    const dAngle = (angleEnd - angleStart) / numAngleSamples;
    const dHeight = (heightEnd - heightStart) / numHeightSamples;

    // Generate grid vertices
    for (let j = 0; j <= numHeightSamples; j++) {
        const h = heightStart + j * dHeight;
        for (let i = 0; i <= numAngleSamples; i++) {
            const angle = angleStart + i * dAngle;
            uvVertices.push([angle, h]);
        }
    }

    // Generate triangles from grid
    const cols = numAngleSamples + 1;
    for (let j = 0; j < numHeightSamples; j++) {
        for (let i = 0; i < numAngleSamples; i++) {
            const topLeft = j * cols + i;
            const topRight = j * cols + i + 1;
            const bottomLeft = (j + 1) * cols + i;
            const bottomRight = (j + 1) * cols + i + 1;

            // Two triangles per quad
            triangles.push([topLeft, bottomLeft, bottomRight]);
            triangles.push([topLeft, bottomRight, topRight]);
        }
    }

    // Evaluate to 3D
    return evaluateUVMesh(surface, uvVertices, triangles);
}

/**
 * Tessellate a spherical surface patch using a proper UV grid
 */
export async function tessellateSphere(
    surface: SphericalSurface,
    lonStart: number = 0,
    lonEnd: number = Math.PI * 2,
    latStart: number = -Math.PI / 2,
    latEnd: number = Math.PI / 2,
    numLonSamples: number = 16,
    numLatSamples: number = 8
): Promise<TessellatedMesh> {
    // Create a full UV grid (not just boundary)
    const uvVertices: [number, number][] = [];
    const triangles: [number, number, number][] = [];

    const dLon = (lonEnd - lonStart) / numLonSamples;
    const dLat = (latEnd - latStart) / numLatSamples;

    // Generate grid vertices
    for (let j = 0; j <= numLatSamples; j++) {
        const lat = latStart + j * dLat;
        for (let i = 0; i <= numLonSamples; i++) {
            const lon = lonStart + i * dLon;
            uvVertices.push([lon, lat]);
        }
    }

    // Generate triangles from grid
    const cols = numLonSamples + 1;
    for (let j = 0; j < numLatSamples; j++) {
        for (let i = 0; i < numLonSamples; i++) {
            const topLeft = j * cols + i;
            const topRight = j * cols + i + 1;
            const bottomLeft = (j + 1) * cols + i;
            const bottomRight = (j + 1) * cols + i + 1;

            // Two triangles per quad
            triangles.push([topLeft, bottomLeft, bottomRight]);
            triangles.push([topLeft, bottomRight, topRight]);
        }
    }

    // Evaluate to 3D
    return evaluateUVMesh(surface, uvVertices, triangles);
}

/**
 * Tessellate a conical surface patch using a proper UV grid
 */
export async function tessellateCone(
    surface: ConicalSurface,
    angleStart: number = 0,
    angleEnd: number = Math.PI * 2,
    heightStart: number = 0,
    heightEnd: number = 1,
    numAngleSamples: number = 16,
    numHeightSamples: number = 2
): Promise<TessellatedMesh> {
    // Create a full UV grid (not just boundary)
    const uvVertices: [number, number][] = [];
    const triangles: [number, number, number][] = [];

    const dAngle = (angleEnd - angleStart) / numAngleSamples;
    const dHeight = (heightEnd - heightStart) / numHeightSamples;

    // Generate grid vertices
    for (let j = 0; j <= numHeightSamples; j++) {
        const h = heightStart + j * dHeight;
        for (let i = 0; i <= numAngleSamples; i++) {
            const angle = angleStart + i * dAngle;
            uvVertices.push([angle, h]);
        }
    }

    // Generate triangles from grid
    const cols = numAngleSamples + 1;
    for (let j = 0; j < numHeightSamples; j++) {
        for (let i = 0; i < numAngleSamples; i++) {
            const topLeft = j * cols + i;
            const topRight = j * cols + i + 1;
            const bottomLeft = (j + 1) * cols + i;
            const bottomRight = (j + 1) * cols + i + 1;

            // Two triangles per quad
            triangles.push([topLeft, bottomLeft, bottomRight]);
            triangles.push([topLeft, bottomRight, topRight]);
        }
    }

    // Evaluate to 3D
    return evaluateUVMesh(surface, uvVertices, triangles);
}

/**
 * Tessellate a toroidal surface patch using a proper UV grid
 */
export async function tessellateTorus(
    surface: ToroidalSurface,
    majorAngleStart: number = 0,
    majorAngleEnd: number = Math.PI * 2,
    minorAngleStart: number = 0,
    minorAngleEnd: number = Math.PI * 2,
    numMajorSamples: number = 24,
    numMinorSamples: number = 12
): Promise<TessellatedMesh> {
    // Create a full UV grid (not just boundary)
    const uvVertices: [number, number][] = [];
    const triangles: [number, number, number][] = [];

    const dMajor = (majorAngleEnd - majorAngleStart) / numMajorSamples;
    const dMinor = (minorAngleEnd - minorAngleStart) / numMinorSamples;

    // Generate grid vertices
    for (let j = 0; j <= numMinorSamples; j++) {
        const minor = minorAngleStart + j * dMinor;
        for (let i = 0; i <= numMajorSamples; i++) {
            const major = majorAngleStart + i * dMajor;
            uvVertices.push([major, minor]);
        }
    }

    // Generate triangles from grid
    const cols = numMajorSamples + 1;
    for (let j = 0; j < numMinorSamples; j++) {
        for (let i = 0; i < numMajorSamples; i++) {
            const topLeft = j * cols + i;
            const topRight = j * cols + i + 1;
            const bottomLeft = (j + 1) * cols + i;
            const bottomRight = (j + 1) * cols + i + 1;

            // Two triangles per quad
            triangles.push([topLeft, bottomLeft, bottomRight]);
            triangles.push([topLeft, bottomRight, topRight]);
        }
    }

    // Evaluate to 3D
    return evaluateUVMesh(surface, uvVertices, triangles);
}

/**
 * Tessellate a B-spline surface patch using a UV grid
 */
export async function tessellateBSplineSurface(
    surface: BSplineSurfaceLocal,
    numUSamples: number = 16,
    numVSamples: number = 16
): Promise<TessellatedMesh> {
    // Get UV parameter range from knot vectors
    const { uKnots, vKnots, uDegree, vDegree } = surface;

    // Valid parameter range is [knots[degree], knots[n+1]] where n = numControlPoints - 1
    const uMin = uKnots[uDegree];
    const uMax = uKnots[uKnots.length - uDegree - 1];
    const vMin = vKnots[vDegree];
    const vMax = vKnots[vKnots.length - vDegree - 1];

    // Create a full UV grid
    const uvVertices: [number, number][] = [];
    const triangles: [number, number, number][] = [];

    const dU = (uMax - uMin) / numUSamples;
    const dV = (vMax - vMin) / numVSamples;

    // Generate grid vertices
    for (let j = 0; j <= numVSamples; j++) {
        const v = vMin + j * dV;
        for (let i = 0; i <= numUSamples; i++) {
            const u = uMin + i * dU;
            uvVertices.push([u, v]);
        }
    }

    // Generate triangles from grid
    const cols = numUSamples + 1;
    for (let j = 0; j < numVSamples; j++) {
        for (let i = 0; i < numUSamples; i++) {
            const topLeft = j * cols + i;
            const topRight = j * cols + i + 1;
            const bottomLeft = (j + 1) * cols + i;
            const bottomRight = (j + 1) * cols + i + 1;

            // Two triangles per quad
            triangles.push([topLeft, bottomLeft, bottomRight]);
            triangles.push([topLeft, bottomRight, topRight]);
        }
    }

    // Evaluate to 3D
    return evaluateUVMesh(surface, uvVertices, triangles);
}

/**
 * Tessellate a planar surface patch
 */
export async function tessellatePlane(
    surface: PlaneSurface,
    uMin: number = 0,
    uMax: number = 1,
    vMin: number = 0,
    vMax: number = 1
): Promise<TessellatedMesh> {
    const uvBoundary = createRectangularUVBoundary(uMin, uMax, vMin, vMax);
    const triangles = await constrainedDelaunayTriangulation(uvBoundary, [], false);
    return evaluateUVMesh(surface, uvBoundary, triangles);
}

/**
 * Generic surface tessellation using UV boundary
 */
export async function tessellateSurface(
    surface: Surface,
    uvBoundary: Vec2[],
    useDelaunay: boolean = true
): Promise<TessellatedMesh> {
    const triangles = await constrainedDelaunayTriangulation(uvBoundary, [], useDelaunay);
    return evaluateUVMesh(surface, uvBoundary, triangles);
}

/**
 * C6: Tessellate a surface using an arbitrary UV boundary polygon
 * This handles trimmed surfaces where the boundary isn't rectangular
 *
 * C6.4: Now supports holes via the uvHoles parameter
 *
 * Uses grid-based tessellation clipped to the boundary for better quality
 * on curved surfaces (avoids long-spanning triangles from ear clipping).
 *
 * @param bbox3d Optional 3D bounding box to filter grid points. For horizontal
 *               cylinders, the UV polygon may span full U range but we only want
 *               points whose 3D positions fall within the boundary's 3D extent.
 */
export async function tessellateTrimmedSurface(
    surface: Surface,
    uvBoundary: Vec2[],
    gridDensity: number = 16,
    uvHoles: Vec2[][] = [],
    bbox3d?: { xMin: number; xMax: number; yMin: number; yMax: number; zMin: number; zMax: number },
    buildOptions?: TrimmedSurfaceBuildOptions
): Promise<TessellatedMesh> {
    if (uvBoundary.length < 3) {
        throw new Error("UV boundary must have at least 3 points");
    }

    // Check if UV boundary crosses the ±π discontinuity (for cylindrical surfaces)
    // This happens when we have U values near both +π and -π
    const PI = Math.PI;
    const nearPosPI = uvBoundary.filter(([u]) => u > PI - 0.5).length;
    const nearNegPI = uvBoundary.filter(([u]) => u < -PI + 0.5).length;

    // Also check if this is a full circle (U spans nearly 2π)
    let uMinBoundary = Infinity, uMaxBoundary = -Infinity;
    for (const [u] of uvBoundary) {
        uMinBoundary = Math.min(uMinBoundary, u);
        uMaxBoundary = Math.max(uMaxBoundary, u);
    }
    const uSpan = uMaxBoundary - uMinBoundary;
    const isFullCircle = uSpan > 5.5; // More than ~315 degrees

    // Only treat as discontinuity if it's NOT a full circle
    const hasDiscontinuity = nearPosPI > 0 && nearNegPI > 0 && !isFullCircle;

    const debugTrim = trimDebugEnabled();
    if (debugTrim) {
        trimDebugLog(`[tessellateTrimmedSurface] nearPosPI=${nearPosPI}, nearNegPI=${nearNegPI}, uSpan=${uSpan.toFixed(3)}, isFullCircle=${isFullCircle}, hasDiscontinuity=${hasDiscontinuity}`);
    }

    // Create a continuous version of the boundary for polygon testing
    let continuousBoundary: Vec2[];
    let continuousHoles: Vec2[][];

    if (hasDiscontinuity) {
        // Determine which half of the circle the boundary is actually in
        // by counting points in each half
        const positiveHalf = uvBoundary.filter(([u]) => u >= 0).length;
        const negativeHalf = uvBoundary.filter(([u]) => u < 0).length;

        if (positiveHalf > negativeHalf) {
            // Boundary is mostly in [0, π], move -π values to +π
            continuousBoundary = uvBoundary.map(([u, v]) => {
                if (u < -PI + 0.5) {
                    return [u + 2 * PI, v] as Vec2;
                }
                return [u, v] as Vec2;
            });
            continuousHoles = uvHoles.map(hole =>
                hole.map(([u, v]) => {
                    if (u < -PI + 0.5) {
                        return [u + 2 * PI, v] as Vec2;
                    }
                    return [u, v] as Vec2;
                })
            );
        } else {
            // Boundary is mostly in [-π, 0], move +π values to -π
            continuousBoundary = uvBoundary.map(([u, v]) => {
                if (u > PI - 0.5) {
                    return [u - 2 * PI, v] as Vec2;
                }
                return [u, v] as Vec2;
            });
            continuousHoles = uvHoles.map(hole =>
                hole.map(([u, v]) => {
                    if (u > PI - 0.5) {
                        return [u - 2 * PI, v] as Vec2;
                    }
                    return [u, v] as Vec2;
                })
            );
        }
    } else {
        continuousBoundary = uvBoundary;
        continuousHoles = uvHoles;
    }
    const useOccUvInside = typeof buildOptions?.uvInsideTest === 'function';
    const keepTriangle = buildOptions?.keepTriangle;
    const allowPartialCellTriangles = buildOptions?.allowPartialCellTriangles ?? true;
    const gridDensityU = Math.max(4, Math.floor(buildOptions?.gridDensityU ?? gridDensity));
    const gridDensityV = Math.max(2, Math.floor(buildOptions?.gridDensityV ?? gridDensity));
    const preferGridForHoles = buildOptions?.preferGridForHoles ?? false;
    const buildLabel = buildOptions?.logLabel ?? 'default';

    // Debug: log continuous boundary bounds
    if (debugTrim) {
        let cbUMin = Infinity, cbUMax = -Infinity, cbVMin = Infinity, cbVMax = -Infinity;
        for (const [u, v] of continuousBoundary) {
            cbUMin = Math.min(cbUMin, u);
            cbUMax = Math.max(cbUMax, u);
            cbVMin = Math.min(cbVMin, v);
            cbVMax = Math.max(cbVMax, v);
        }
        trimDebugLog(`[tessellateTrimmedSurface] Continuous boundary bounds: U=[${cbUMin.toFixed(3)}, ${cbUMax.toFixed(3)}], V=[${cbVMin.toFixed(3)}, ${cbVMax.toFixed(3)}]`);
        trimDebugLog(`[tessellateTrimmedSurface] Continuous boundary first 5: ${continuousBoundary.slice(0, 5).map(([u, v]) => `(${u.toFixed(2)},${v.toFixed(2)})`).join(' ')}`);
        trimDebugLog(`[tessellateTrimmedSurface] Continuous boundary has ${continuousBoundary.length} points`);
        if (continuousHoles.length > 0) {
            for (let i = 0; i < continuousHoles.length; i++) {
                const hole = continuousHoles[i];
                let hUMin = Infinity, hUMax = -Infinity, hVMin = Infinity, hVMax = -Infinity;
                for (const [u, v] of hole) {
                    hUMin = Math.min(hUMin, u);
                    hUMax = Math.max(hUMax, u);
                    hVMin = Math.min(hVMin, v);
                    hVMax = Math.max(hVMax, v);
                }
                trimDebugLog(`[tessellateTrimmedSurface] Continuous hole ${i} bounds: U=[${hUMin.toFixed(3)}, ${hUMax.toFixed(3)}], V=[${hVMin.toFixed(3)}, ${hVMax.toFixed(3)}]`);
            }
        }
    }

    // ===== Use CDT with holes for proper hole support =====
    // CDT (Constrained Delaunay Triangulation) with cavity-based constraint recovery
    // ensures hole boundary edges are preserved in the triangulation
    if (continuousHoles.length > 0 && !preferGridForHoles) {
        trimDebugLog(`[tessellateTrimmedSurface] Triangulating ${continuousHoles.length} hole loops`);
        const holeTriangulationStart = performance.now();

        // If boundary is sparse (e.g., just 4 rectangle corners), densify it
        // to get better triangulation quality on curved surfaces
        let denseBoundary: Vec2[];
        if (continuousBoundary.length <= 8) {
            denseBoundary = [];
            const maxDenseBoundaryPoints = Math.max(
                24,
                Math.floor(readTrimNumber("__TRIM_MAX_DENSE_BOUNDARY_POINTS__") ?? 96)
            );
            const edgePointCap = Math.max(4, Math.floor(maxDenseBoundaryPoints / Math.max(1, continuousBoundary.length)));
            for (let i = 0; i < continuousBoundary.length; i++) {
                const p1 = continuousBoundary[i];
                const p2 = continuousBoundary[(i + 1) % continuousBoundary.length];
                // Add points along this edge
                const edgePoints = Math.max(4, Math.min(edgePointCap, Math.max(6, Math.floor(gridDensity * 0.5))));
                for (let t = 0; t < edgePoints; t++) {
                    const u = p1[0] + (p2[0] - p1[0]) * (t / edgePoints);
                    const v = p1[1] + (p2[1] - p1[1]) * (t / edgePoints);
                    denseBoundary.push([u, v]);
                }
            }
            trimDebugLog(`[tessellateTrimmedSurface] Densified boundary from ${continuousBoundary.length} to ${denseBoundary.length} points`);
        } else {
            denseBoundary = continuousBoundary;
        }

        // Complexity dispatch:
        // - earcut for small/medium simple hole domains (lower overhead)
        // - CDT for heavier domains or when earcut fails
        const totalHolePoints = continuousHoles.reduce((sum, hole) => sum + hole.length, 0);
        const totalLoopPoints = denseBoundary.length + totalHolePoints;
        const holeCount = continuousHoles.length;
        const forceMode = (globalThis as any)?.__TRIM_HOLE_TRIANGULATION_MODE__;
        const earcutMaxPoints = Math.max(64, Math.floor(readTrimNumber("__TRIM_HOLE_EARCUT_MAX_POINTS__") ?? 512));
        const earcutMaxHoles = Math.max(1, Math.floor(readTrimNumber("__TRIM_HOLE_EARCUT_MAX_HOLES__") ?? 24));
        const canUseEarcut =
            forceMode === "earcut" ||
            (forceMode !== "cdt" && totalLoopPoints <= earcutMaxPoints && holeCount <= earcutMaxHoles);

        let triangles: [number, number, number][] = [];
        if (canUseEarcut) {
            const earcutTriangles = triangulateWithHoles(denseBoundary, continuousHoles);
            triangles = earcutTriangles.map((tri) => [tri[0], tri[1], tri[2]] as [number, number, number]);
            if (triangles.length === 0) {
                triangles = await cdtWithHoles(denseBoundary, continuousHoles);
            } else {
                trimDebugLog(
                    `[tessellateTrimmedSurface] Hole mode=earcut points=${totalLoopPoints} holes=${holeCount} tris=${triangles.length}`
                );
            }
        } else {
            triangles = await cdtWithHoles(denseBoundary, continuousHoles);
            trimDebugLog(
                `[tessellateTrimmedSurface] Hole mode=cdt points=${totalLoopPoints} holes=${holeCount} tris=${triangles.length}`
            );
        }
        trimDebugLog(`[tessellateTrimmedSurface] Hole triangulation generated ${triangles.length} triangles`);
        recordTrimProfileSample(buildOptions, "hole_triangulation", performance.now() - holeTriangulationStart);

        // Build combined vertex array (boundary + all holes)
        const allVertices: Vec2[] = [...denseBoundary];
        for (const hole of continuousHoles) {
            allVertices.push(...hole);
        }
        if (!keepTriangle) {
            return evaluateUVMesh(surface, allVertices, triangles);
        }

        const triSamples = (a: Vec2, b: Vec2, c: Vec2): [number, number][] => {
            const uCentroid = (a[0] + b[0] + c[0]) / 3;
            const vCentroid = (a[1] + b[1] + c[1]) / 3;
            return [
                [a[0], a[1]],
                [b[0], b[1]],
                [c[0], c[1]],
                [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
                [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2],
                [(c[0] + a[0]) / 2, (c[1] + a[1]) / 2],
                [uCentroid, vCentroid],
            ];
        };

        const filteredTriangles: [number, number, number][] = [];
        let droppedByGate = 0;
        const holeTriangleGateStart = performance.now();
        for (const [ia, ib, ic] of triangles) {
            const a = allVertices[ia];
            const b = allVertices[ib];
            const c = allVertices[ic];
            if (!a || !b || !c) continue;
            let keep = true;
            try {
                keep = !!keepTriangle(triSamples(a, b, c));
            } catch {
                keep = true;
            }
            if (keep) {
                filteredTriangles.push([ia, ib, ic]);
            } else {
                droppedByGate++;
            }
        }
        recordTrimProfileSample(buildOptions, "hole_triangle_gate", performance.now() - holeTriangleGateStart);

        if (filteredTriangles.length === 0) {
            console.warn(`[tessellateTrimmedSurface] Triangle gate rejected all CDT triangles; keeping original CDT output (mode=${buildLabel})`);
            const holeEvalStart = performance.now();
            const mesh = await evaluateUVMesh(surface, allVertices, triangles, buildOptions?.recordProfileSample);
            recordTrimProfileSample(buildOptions, "hole_evaluate_mesh", performance.now() - holeEvalStart);
            return mesh;
        }

        if (droppedByGate > 0) {
            trimDebugLog(`[tessellateTrimmedSurface] Triangle gate dropped ${droppedByGate}/${triangles.length} CDT triangles (mode=${buildLabel})`);
        }
        const holeEvalStart = performance.now();
        const mesh = await evaluateUVMesh(surface, allVertices, filteredTriangles, buildOptions?.recordProfileSample);
        recordTrimProfileSample(buildOptions, "hole_evaluate_mesh", performance.now() - holeEvalStart);
        return mesh;
    }
    // ===== End CDT with holes =====

    // Find UV bounding box from the continuous boundary
    let uMin = Infinity, uMax = -Infinity;
    let vMin = Infinity, vMax = -Infinity;

    for (const [u, v] of continuousBoundary) {
        uMin = Math.min(uMin, u);
        uMax = Math.max(uMax, u);
        vMin = Math.min(vMin, v);
        vMax = Math.max(vMax, v);
    }

    // Create a grid of UV vertices
    const du = (uMax - uMin) / gridDensityU;
    const dv = (vMax - vMin) / gridDensityV;

    // OCC-backed UV classifiers can be stateful on first query for a face.
    // Do a single warmup probe so classification is stable without relying on
    // debug-only probe code paths.
    if (useOccUvInside) {
        const probeU = (uMin + uMax) * 0.5;
        const probeV = (vMin + vMax) * 0.5;
        try {
            buildOptions!.uvInsideTest!(probeU, probeV);
        } catch {
            // Best-effort warmup only; main classification keeps fail-open behavior.
        }
    }

    const uvVertices: Vec2[] = [];
    const vertexGrid: (number | null)[][] = []; // Maps grid position to vertex index

    let insideCount = 0;
    let outsideCount = 0;

    // Helper: check if point is close to polygon boundary (within tolerance)
    function isNearBoundary(point: Vec2, polygon: Vec2[], tolerance: number): boolean {
        const [px, py] = point;
        for (let i = 0; i < polygon.length; i++) {
            const [x1, y1] = polygon[i];
            const [x2, y2] = polygon[(i + 1) % polygon.length];

            // Distance from point to line segment
            const dx = x2 - x1;
            const dy = y2 - y1;
            const lenSq = dx * dx + dy * dy;

            if (lenSq < 1e-12) continue; // Skip degenerate edges

            // Parameter t for closest point on line
            let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
            t = Math.max(0, Math.min(1, t)); // Clamp to segment

            // Closest point on segment
            const closestX = x1 + t * dx;
            const closestY = y1 + t * dy;

            // Distance to closest point
            const dist = Math.sqrt((px - closestX) ** 2 + (py - closestY) ** 2);
            if (dist < tolerance) {
                return true;
            }
        }
        return false;
    }

    // Tolerance for including points near the boundary
    const boundaryTolerance = Math.max(du, dv) * 0.5;
    const preferGeometryOnlyLoad = (globalThis as any)?.__PERF_GEOMETRY_ONLY_LOAD__ === true;
    const perfDisableNearBoundaryDefault = true;
    const perfDisableNearBoundaryRaw = (globalThis as any)?.__PERF_TRIM_DISABLE_NEAR_BOUNDARY__;
    const perfDisableNearBoundary = perfDisableNearBoundaryRaw == null
        ? perfDisableNearBoundaryDefault
        : perfDisableNearBoundaryRaw === true;
    const skipNearBoundaryChecks = preferGeometryOnlyLoad && uvHoles.length === 0 && perfDisableNearBoundary;
    const enableLocalUvClassifierCandidate = readTrimBoolean("__ENABLE_LOCAL_UV_CLASSIFIER_CANDIDATE__", false);
    const enableLocalUvClassifierShadow = readTrimBoolean("__ENABLE_LOCAL_UV_CLASSIFIER_SHADOW__", false);
    const localUvClassifierFallbackToOcc = readTrimBoolean("__LOCAL_UV_CLASSIFIER_FALLBACK_TO_OCC__", true);
    const localUvClassifierCandidateStrict = readTrimBoolean("__LOCAL_UV_CLASSIFIER_CANDIDATE_STRICT__", false);
    const localUvClassifierBandScale = Math.max(
        0,
        readTrimNumber("__LOCAL_UV_CLASSIFIER_BOUNDARY_BAND_SCALE__") ?? 1
    );
    const localUvBoundaryTolerance = Math.max(0, boundaryTolerance * localUvClassifierBandScale);
    const localUvStageBWeightedMixedSign = readTrimBoolean("__LOCAL_UV_STAGEB_WEIGHTED_MIXED_SIGN__", true);
    const localUvStageBAccumulateCrossings = readTrimBoolean("__LOCAL_UV_STAGEB_ACCUMULATE_CROSSINGS__", false);
    const localUvStageBMultiProbeConsensus = readTrimBoolean("__LOCAL_UV_STAGEB_MULTI_PROBE_CONSENSUS__", false);
    const localUvStageBTopTransTieAccumulator = readTrimBoolean(
        "__LOCAL_UV_STAGEB_TOPTRANS_TIE_ACCUMULATOR__",
        false
    );
    const localUvStageBMultiProbePreferOutside = readTrimBoolean(
        "__LOCAL_UV_STAGEB_MULTI_PROBE_PREFER_OUTSIDE__",
        true
    );
    const localUvTransitionEdgeToleranceScale = Math.max(
        0.01,
        readTrimNumber("__LOCAL_UV_TRANSITION_EDGE_TOL_SCALE__") ?? 0.25
    );
    const localUvTransitionMaxProbes = Math.max(
        1,
        Math.floor(readTrimNumber("__LOCAL_UV_TRANSITION_MAX_PROBES__") ?? 3)
    );
    const localUvStageBFaceWideProbeTraversal = readTrimBoolean(
        "__LOCAL_UV_STAGEB_FACE_WIDE_PROBE_TRAVERSAL__",
        true
    );
    const localUvSeamTolerance = Math.max(
        localUvBoundaryTolerance,
        Math.max(du, dv) * 1.5,
        1e-4
    );
    const enableLocalUvFaceClassifierFallback = readTrimBoolean("__ENABLE_LOCAL_UV_FACE_CLASSIFIER_FALLBACK__", true);
    const useLocalUvClassifierShadow = useOccUvInside && enableLocalUvClassifierShadow;
    const useLocalUvClassifierCandidate = useOccUvInside && enableLocalUvClassifierCandidate;
    const localUvStageAUseEdgeLoop = readTrimBoolean("__LOCAL_UV_STAGEA_USE_EDGE_LOOP__", true);
    // Domain-unsafe wires (typically cone seam/self-intersection cases) should
    // follow OCCT's bad-wire behavior: rely on Stage-B/topological classification.
    const localUvDomainUnsafeForceStageB = readTrimBoolean("__LOCAL_UV_DOMAIN_UNSAFE_FORCE_STAGEB__", true);
    // Hard gate for parity work: on domain-unsafe faces, skip Stage-A arbitration
    // and use the forced Stage-B decision directly.
    const localUvDomainUnsafeStageBOnly = readTrimBoolean("__LOCAL_UV_DOMAIN_UNSAFE_STAGEB_ONLY__", false);
    // Reframe sample points into each wire's periodic band before wire tests.
    // This better matches OCCT wire-local domain behavior on seam-crossing cones.
    const localUvWireLocalRecadre = readTrimBoolean("__LOCAL_UV_WIRE_LOCAL_RECADRE__", true);
    // In domain-unsafe mode, allow confident Stage-A decisions to override ambiguous
    // forced Stage-B outcomes (transition ties, probe fallbacks, boundary ambiguity).
    const localUvDomainUnsafeArbitrateStageA = readTrimBoolean("__LOCAL_UV_DOMAIN_UNSAFE_ARBITRATE_STAGEA__", true);
    const localUvDomainUnsafeArbitrateTiesOnly = readTrimBoolean("__LOCAL_UV_DOMAIN_UNSAFE_ARBITRATE_TIES_ONLY__", false);
    // Stage-A may return an uncertain boundary-band result with a provisional
    // in/out hint. For cone-heavy domain-unsafe faces, this hint is often
    // directionally correct even when Stage-B picks a deterministic outside.
    const localUvDomainUnsafeUseStageAProvisional = readTrimBoolean(
        "__LOCAL_UV_DOMAIN_UNSAFE_USE_STAGEA_PROVISIONAL__",
        true
    );
    const localUvDomainUnsafeStrictStageAConfidence = readTrimBoolean(
        "__LOCAL_UV_DOMAIN_UNSAFE_STRICT_STAGEA_CONFIDENCE__",
        false
    );
    // Domain-unsafe Stage-A confidence should behave closer to OCCT fallback order:
    // do not treat near-boundary Stage-A votes as confident.
    const localUvDomainUnsafeStageAConfidenceBandMultiplier = Math.max(
        1,
        readTrimNumber("__LOCAL_UV_DOMAIN_UNSAFE_STAGEA_CONFIDENCE_BAND_MULTIPLIER__") ?? 1.5
    );
    const localUvDomainUnsafeStageAConfidenceMinDistance = Math.max(
        0,
        readTrimNumber("__LOCAL_UV_DOMAIN_UNSAFE_STAGEA_CONFIDENCE_MIN_DISTANCE__") ?? 1e-3
    );
    // Targeted cone parity guard:
    // when Stage-A says INSIDE but forced Stage-B says OUTSIDE, only allow
    // Stage-A override if the point is clearly interior (not near boundary).
    const localUvDomainUnsafeGuardInsideVsOutside = readTrimBoolean(
        "__LOCAL_UV_DOMAIN_UNSAFE_GUARD_INSIDE_VS_OUTSIDE__",
        false
    );
    const localUvDomainStageAProvisionalInsideMinSamples = Math.max(
        4,
        Math.floor(readTrimNumber("__LOCAL_UV_DOMAIN_STAGEA_PROVISIONAL_INSIDE_MIN_SAMPLES__") ?? 24)
    );
    const localUvDomainStageAProvisionalInsideOutsideRatio = Math.max(
        0,
        Math.min(1, readTrimNumber("__LOCAL_UV_DOMAIN_STAGEA_PROVISIONAL_INSIDE_OUTSIDE_RATIO__") ?? 0.6)
    );
    const localUvDomainUnsafeInsideOverrideBandMultiplier = Math.max(
        1,
        readTrimNumber("__LOCAL_UV_DOMAIN_UNSAFE_INSIDE_OVERRIDE_BAND_MULTIPLIER__") ?? 2
    );
    const localUvDomainUnsafeInsideOverrideMinDistance = Math.max(
        0,
        readTrimNumber("__LOCAL_UV_DOMAIN_UNSAFE_INSIDE_OVERRIDE_MIN_DISTANCE__") ?? 2e-3
    );
    const localUvClassifierStats = {
        inside: 0,
        outside: 0,
        uncertain: 0,
        fallbackCalls: 0,
        occDecisions: 0,
        boundaryBandSamples: 0,
        seamProximateSamples: 0,
        mismatchCount: 0,
        effectiveMismatchCount: 0,
        mismatchBoundaryBand: 0,
        mismatchInterior: 0,
        mismatchSeamProximate: 0,
        mismatchNonSeam: 0,
        falseInsideCount: 0,
        falseOutsideCount: 0,
        stageAEvaluations: 0,
        stageAInside: 0,
        stageAOutside: 0,
        stageAUncertain: 0,
        stageABadWire: 0,
        stageBEvaluations: 0,
        stageBForcedEvaluations: 0,
        stageBTriggeredByUncertain: 0,
        stageBTriggeredByBadWire: 0,
        stageBInside: 0,
        stageBOutside: 0,
        stageBUncertain: 0,
        stageBResolvedByUncertain: 0,
        stageBResolvedByBadWire: 0,
        stageBProbeFallbacks: 0,
        stageBBundleSkips: 0,
        stageBSkippedNearVertexHits: 0,
        stageBTransitionTies: 0,
        decisionFromStageA: 0,
        decisionFromStageB: 0,
        decisionFromStageBForced: 0,
        decisionFromDomainUnsafeStageA: 0,
        decisionFromDomainUnsafeStageB: 0,
        mismatchFromStageA: 0,
        mismatchFromStageB: 0,
        mismatchFromStageBForced: 0,
        mismatchFromDomainUnsafe: 0,
        mismatchFromDomainUnsafeStageA: 0,
        mismatchFromDomainUnsafeStageB: 0,
        mismatchDomainStageBOutside: 0,
        mismatchDomainStageBWithTransitionTie: 0,
        mismatchDomainStageBWithProbeFallback: 0,
        mismatchDomainStageBNearBoundary: 0,
        mismatchDomainStageBWithStageAInsideHint: 0,
        mismatchDomainStageBWithStageAOutsideHint: 0,
        mismatchDomainStageBWithStageAUncertainHint: 0,
    };

    type LocalUvDecision = "inside" | "outside" | "uncertain";
    type LocalUvDecisionSource =
        | "stageA"
        | "stageB"
        | "stageB_forced"
        | "domain_stageA"
        | "domain_stageB"
        | "domain_uncertain";
    interface LocalUvClassification {
        decision: LocalUvDecision;
        nearBoundaryBand: boolean;
        seamProximate: boolean;
        source: LocalUvDecisionSource;
        domainStageAHint?: LocalUvDecision;
        domainStageAProvisionalHint?: Exclude<LocalUvDecision, "uncertain">;
        stageBMetrics?: LocalUvTransitionMetrics;
    }

    type LoopPointRelation = "inside" | "outside" | "on";
    interface PreparedLoopClassifier {
        pointsX: number[];
        pointsY: number[];
        pointCount: number;
        uMin: number;
        uMax: number;
        vMin: number;
        vMax: number;
        uRange: number;
        vRange: number;
        tolUOriginal: number;
        tolVOriginal: number;
        tolUNormalized: number;
        tolVNormalized: number;
    }
    const classifierPointEpsilon = Math.max(
        1e-6,
        Math.min(5e-3, Math.max(du, dv) * 0.02)
    );

    const normalizeLoopForClassifier = (loop: Vec2[]): Vec2[] => {
        const normalized: Vec2[] = [];
        for (const [u, v] of loop) {
            if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
            if (normalized.length === 0) {
                normalized.push([u, v]);
                continue;
            }
            const [prevU, prevV] = normalized[normalized.length - 1];
            if (Math.abs(u - prevU) <= classifierPointEpsilon && Math.abs(v - prevV) <= classifierPointEpsilon) {
                continue;
            }
            normalized.push([u, v]);
        }
        if (normalized.length > 1) {
            const [firstU, firstV] = normalized[0];
            const [lastU, lastV] = normalized[normalized.length - 1];
            if (
                Math.abs(firstU - lastU) <= classifierPointEpsilon &&
                Math.abs(firstV - lastV) <= classifierPointEpsilon
            ) {
                normalized.pop();
            }
        }
        return normalized;
    };

    const normalizeEdgePointsForClassifier = (points: Vec2[]): Vec2[] => {
        const normalized: Vec2[] = [];
        for (const [u, v] of points) {
            if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
            if (normalized.length === 0) {
                normalized.push([u, v]);
                continue;
            }
            const [prevU, prevV] = normalized[normalized.length - 1];
            if (Math.abs(u - prevU) <= classifierPointEpsilon && Math.abs(v - prevV) <= classifierPointEpsilon) {
                continue;
            }
            normalized.push([u, v]);
        }
        return normalized;
    };

    const unwrapLoopPeriodicUForClassifier = (loop: Vec2[], period: number): Vec2[] => {
        if (loop.length === 0 || !Number.isFinite(period) || period <= 1e-9) {
            return loop;
        }
        const unwrapped: Vec2[] = [[loop[0][0], loop[0][1]]];
        let prevU = loop[0][0];
        for (let i = 1; i < loop.length; i++) {
            const [uRaw, v] = loop[i];
            let bestU = uRaw;
            let bestDelta = Math.abs(bestU - prevU);
            const baseShift = Math.round((prevU - uRaw) / period);
            for (const offset of [-1, 0, 1]) {
                const candidateU = uRaw + (baseShift + offset) * period;
                const candidateDelta = Math.abs(candidateU - prevU);
                if (candidateDelta < bestDelta) {
                    bestDelta = candidateDelta;
                    bestU = candidateU;
                }
            }
            unwrapped.push([bestU, v]);
            prevU = bestU;
        }
        return unwrapped;
    };

    const transformToNormalized = (value: number, min: number, range: number): number => {
        return range > 1e-10 ? (value - min) / range : value;
    };

    const buildPreparedLoopClassifier = (
        loop: Vec2[],
        domain: { uMin: number; uMax: number; vMin: number; vMax: number },
        tolUOriginal: number,
        tolVOriginal: number
    ): PreparedLoopClassifier | null => {
        if (loop.length < 3) return null;
        const uRange = domain.uMax - domain.uMin;
        const vRange = domain.vMax - domain.vMin;
        const pointCount = loop.length;
        const pointsX = new Array(pointCount + 1);
        const pointsY = new Array(pointCount + 1);
        for (let i = 0; i < pointCount; i++) {
            pointsX[i] = transformToNormalized(loop[i][0], domain.uMin, uRange);
            pointsY[i] = transformToNormalized(loop[i][1], domain.vMin, vRange);
        }
        pointsX[pointCount] = pointsX[0];
        pointsY[pointCount] = pointsY[0];
        const tolUNormalized = uRange > 1e-10 ? tolUOriginal / uRange : tolUOriginal;
        const tolVNormalized = vRange > 1e-10 ? tolVOriginal / vRange : tolVOriginal;
        return {
            pointsX,
            pointsY,
            pointCount,
            uMin: domain.uMin,
            uMax: domain.uMax,
            vMin: domain.vMin,
            vMax: domain.vMax,
            uRange,
            vRange,
            tolUOriginal,
            tolVOriginal,
            tolUNormalized,
            tolVNormalized,
        };
    };

    const internalSiDans = (
        classifier: PreparedLoopClassifier,
        px: number,
        py: number
    ): boolean => {
        let nbCrossings = 0;
        let prevDx = classifier.pointsX[0] - px;
        let prevDy = classifier.pointsY[0] - py;
        let prevYIsNegative = prevDy < 0;
        for (let nextIdx = 1; nextIdx <= classifier.pointCount; nextIdx++) {
            const currDx = classifier.pointsX[nextIdx] - px;
            const currDy = classifier.pointsY[nextIdx] - py;
            const currYIsNegative = currDy < 0;
            if (currYIsNegative !== prevYIsNegative) {
                if (prevDx > 0 && currDx > 0) {
                    nbCrossings++;
                } else if (prevDx > 0 || currDx > 0) {
                    const xIntersect = prevDx - (prevDy * (currDx - prevDx)) / (currDy - prevDy);
                    if (xIntersect > 0) {
                        nbCrossings++;
                    }
                }
                prevYIsNegative = currYIsNegative;
            }
            prevDx = currDx;
            prevDy = currDy;
        }
        return (nbCrossings & 1) !== 0;
    };

    const internalSiDansOuOn = (
        classifier: PreparedLoopClassifier,
        px: number,
        py: number
    ): LoopPointRelation => {
        let nbCrossings = 0;
        const tolU = classifier.tolUNormalized;
        const tolV = classifier.tolVNormalized;
        let prevDx = classifier.pointsX[0] - px;
        let prevDy = classifier.pointsY[0] - py;
        let prevYIsNegative = prevDy < 0;

        for (let nextIdx = 1; nextIdx <= classifier.pointCount; nextIdx++) {
            const prevIdx = nextIdx - 1;
            const currDx = classifier.pointsX[nextIdx] - px;
            const currDy = classifier.pointsY[nextIdx] - py;

            if (currDx < tolU && currDx > -tolU && currDy < tolV && currDy > -tolV) {
                return "on";
            }

            const edgeDx = classifier.pointsX[nextIdx] - classifier.pointsX[prevIdx];
            if (
                (classifier.pointsX[prevIdx] - px) * currDx < 0 &&
                Math.abs(edgeDx) > 1e-12
            ) {
                const interpY =
                    classifier.pointsY[nextIdx] -
                    ((classifier.pointsY[nextIdx] - classifier.pointsY[prevIdx]) / edgeDx) * currDx;
                const deltaY = interpY - py;
                if (deltaY >= -tolV && deltaY <= tolV) {
                    return "on";
                }
            }

            const currYIsNegative = currDy < 0;
            if (currYIsNegative !== prevYIsNegative) {
                if (prevDx > 0 && currDx > 0) {
                    nbCrossings++;
                } else if (prevDx > 0 || currDx > 0) {
                    const xIntersect = prevDx - (prevDy * (currDx - prevDx)) / (currDy - prevDy);
                    if (xIntersect > 0) {
                        nbCrossings++;
                    }
                }
                prevYIsNegative = currYIsNegative;
            }
            prevDx = currDx;
            prevDy = currDy;
        }

        return (nbCrossings & 1) !== 0 ? "inside" : "outside";
    };

    const classifyPointAgainstLoop = (
        point: Vec2,
        classifier: PreparedLoopClassifier | null
    ): LoopPointRelation => {
        if (!classifier || classifier.pointCount < 3) return "outside";
        const [u, v] = point;
        if (
            u < classifier.uMin - classifier.tolUOriginal ||
            u > classifier.uMax + classifier.tolUOriginal ||
            v < classifier.vMin - classifier.tolVOriginal ||
            v > classifier.vMax + classifier.tolVOriginal
        ) {
            return "outside";
        }

        const px = transformToNormalized(u, classifier.uMin, classifier.uRange);
        const py = transformToNormalized(v, classifier.vMin, classifier.vRange);

        const result = internalSiDansOuOn(classifier, px, py);
        if (result === "on") {
            return "on";
        }

        const tolU = classifier.tolUNormalized;
        const tolV = classifier.tolVNormalized;
        if (tolU > 0 || tolV > 0) {
            const isInside = result === "inside";
            const cornerInsideA = internalSiDans(classifier, px - tolU, py - tolV);
            const cornerInsideB = internalSiDans(classifier, px + tolU, py - tolV);
            const cornerInsideC = internalSiDans(classifier, px - tolU, py + tolV);
            const cornerInsideD = internalSiDans(classifier, px + tolU, py + tolV);
            if (
                isInside !== cornerInsideA ||
                isInside !== cornerInsideB ||
                isInside !== cornerInsideC ||
                isInside !== cornerInsideD
            ) {
                return "on";
            }
        }

        return result;
    };

    const orientation2D = (a: Vec2, b: Vec2, c: Vec2): number => {
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    };

    const loopsEdgeIntersects = (a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2, tol: number): boolean => {
        const onSegment = (p: Vec2, q: Vec2, r: Vec2): boolean => {
            return (
                r[0] >= Math.min(p[0], q[0]) - tol &&
                r[0] <= Math.max(p[0], q[0]) + tol &&
                r[1] >= Math.min(p[1], q[1]) - tol &&
                r[1] <= Math.max(p[1], q[1]) + tol
            );
        };
        const o1 = orientation2D(a1, a2, b1);
        const o2 = orientation2D(a1, a2, b2);
        const o3 = orientation2D(b1, b2, a1);
        const o4 = orientation2D(b1, b2, a2);
        const s1 = Math.abs(o1) <= tol ? 0 : Math.sign(o1);
        const s2 = Math.abs(o2) <= tol ? 0 : Math.sign(o2);
        const s3 = Math.abs(o3) <= tol ? 0 : Math.sign(o3);
        const s4 = Math.abs(o4) <= tol ? 0 : Math.sign(o4);
        if (s1 * s2 < 0 && s3 * s4 < 0) {
            return true;
        }
        if (s1 === 0 && onSegment(a1, a2, b1)) return true;
        if (s2 === 0 && onSegment(a1, a2, b2)) return true;
        if (s3 === 0 && onSegment(b1, b2, a1)) return true;
        if (s4 === 0 && onSegment(b1, b2, a2)) return true;
        return false;
    };

    const hasSelfIntersections = (loop: Vec2[], tol: number): boolean => {
        const n = loop.length;
        if (n < 4) return false;
        for (let i = 0; i < n; i++) {
            const a1 = loop[i];
            const a2 = loop[(i + 1) % n];
            for (let j = i + 1; j < n; j++) {
                if (j === i) continue;
                if ((j + 1) % n === i || (i + 1) % n === j) continue;
                const b1 = loop[j];
                const b2 = loop[(j + 1) % n];
                if (loopsEdgeIntersects(a1, a2, b1, b2, tol)) {
                    return true;
                }
            }
        }
        return false;
    };

    const computeLoopBounds = (loop: Vec2[]): { uMin: number; uMax: number; vMin: number; vMax: number } => {
        let uMinLocal = Infinity;
        let uMaxLocal = -Infinity;
        let vMinLocal = Infinity;
        let vMaxLocal = -Infinity;
        for (const [u, v] of loop) {
            if (u < uMinLocal) uMinLocal = u;
            if (u > uMaxLocal) uMaxLocal = u;
            if (v < vMinLocal) vMinLocal = v;
            if (v > vMaxLocal) vMaxLocal = v;
        }
        return {
            uMin: uMinLocal,
            uMax: uMaxLocal,
            vMin: vMinLocal,
            vMax: vMaxLocal,
        };
    };

    // For local-vs-OCC parity, prefer raw UV loops (before continuity seam rewrites).
    // OCCT classifiers evaluate wires in their own periodic domains instead of a
    // globally unwrapped loop.
    const rawOuterClassifierLoopBase = normalizeLoopForClassifier(uvBoundary);
    const rawHoleClassifierLoopsBase = uvHoles
        .map((hole) => normalizeLoopForClassifier(hole))
        .filter((hole) => hole.length >= 3);
    const rawOuterClassifierLoop = hasDiscontinuity
        ? unwrapLoopPeriodicUForClassifier(rawOuterClassifierLoopBase, 2 * PI)
        : rawOuterClassifierLoopBase;
    const rawHoleClassifierLoops = hasDiscontinuity
        ? rawHoleClassifierLoopsBase.map((hole) => unwrapLoopPeriodicUForClassifier(hole, 2 * PI))
        : rawHoleClassifierLoopsBase;
    const normalizedOuterClassifierLoop = normalizeLoopForClassifier(continuousBoundary);
    const fallbackOuterClassifierLoop =
        normalizedOuterClassifierLoop.length >= 3 ? normalizedOuterClassifierLoop : continuousBoundary;
    const fallbackHoleClassifierLoops = continuousHoles
        .map((hole) => normalizeLoopForClassifier(hole))
        .filter((hole) => hole.length >= 3);

    const defaultOuterClassifierLoop =
        rawOuterClassifierLoop.length >= 3 ? rawOuterClassifierLoop : fallbackOuterClassifierLoop;
    const defaultHoleClassifierLoops = rawHoleClassifierLoops.length > 0
        ? rawHoleClassifierLoops
        : fallbackHoleClassifierLoops;

    interface ClassifierWireSpec {
        loop: Vec2[];
        orientationBit: 0 | 1;
        edges?: LocalUvClassifierEdgeInput[];
        hasBadEdges?: boolean;
    }

    const defaultClassifierWireSpecs: ClassifierWireSpec[] = [];
    if (defaultOuterClassifierLoop.length >= 3) {
        defaultClassifierWireSpecs.push({
            loop: defaultOuterClassifierLoop,
            orientationBit: 1,
        });
    }
    for (const holeLoop of defaultHoleClassifierLoops) {
        if (holeLoop.length < 3) continue;
        defaultClassifierWireSpecs.push({
            loop: holeLoop,
            orientationBit: 0,
        });
    }

    const customClassifierWireSpecs: ClassifierWireSpec[] = [];
    const customClassifierWires = buildOptions?.localUvClassifierWires ?? [];
    for (const wire of customClassifierWires) {
        if (!wire || !Array.isArray(wire.loop)) continue;
        const normalizedBase = normalizeLoopForClassifier(wire.loop);
        const normalizedLoop = hasDiscontinuity
            ? unwrapLoopPeriodicUForClassifier(normalizedBase, 2 * PI)
            : normalizedBase;
        if (normalizedLoop.length < 3) continue;
        customClassifierWireSpecs.push({
            loop: normalizedLoop,
            orientationBit: wire.orientationBit === 0 ? 0 : 1,
            edges: Array.isArray(wire.edges)
                ? wire.edges.reduce<LocalUvClassifierEdgeInput[]>((acc, edge) => {
                    const pointsBase = normalizeEdgePointsForClassifier(edge?.points ?? []);
                    const points = hasDiscontinuity
                        ? unwrapLoopPeriodicUForClassifier(pointsBase, 2 * PI)
                        : pointsBase;
                    if (points.length < 2) return acc;
                    const edgeInput: LocalUvClassifierEdgeInput = {
                        points,
                        hasHighVertexTolerance: edge.hasHighVertexTolerance === true,
                        isDegenerate: edge.isDegenerate === true,
                    };
                    if (typeof edge.vertexToleranceMax === "number" && Number.isFinite(edge.vertexToleranceMax)) {
                        edgeInput.vertexToleranceMax = edge.vertexToleranceMax;
                    }
                    acc.push(edgeInput);
                    return acc;
                }, [])
                : undefined,
            hasBadEdges: wire.hasBadEdges === true,
        });
    }

    const classifierWireSpecs =
        customClassifierWireSpecs.length > 0 ? customClassifierWireSpecs : defaultClassifierWireSpecs;

    const loopPerimeter = (loop: Vec2[]): number => {
        if (loop.length < 2) return 0;
        let total = 0;
        for (let i = 0; i < loop.length; i++) {
            const a = loop[i];
            const b = loop[(i + 1) % loop.length];
            total += Math.hypot(b[0] - a[0], b[1] - a[1]);
        }
        return total;
    };

    let outerClassifierLoop = defaultOuterClassifierLoop;
    const explicitOuterWires = classifierWireSpecs.filter((wire) => wire.orientationBit === 1);
    const outerCandidates = explicitOuterWires.length > 0 ? explicitOuterWires : classifierWireSpecs;
    if (outerCandidates.length > 0) {
        outerClassifierLoop = outerCandidates[0].loop;
        let bestPerimeter = loopPerimeter(outerClassifierLoop);
        for (let i = 1; i < outerCandidates.length; i++) {
            const candidateLoop = outerCandidates[i].loop;
            const candidatePerimeter = loopPerimeter(candidateLoop);
            if (candidatePerimeter > bestPerimeter) {
                bestPerimeter = candidatePerimeter;
                outerClassifierLoop = candidateLoop;
            }
        }
    }
    const holeClassifierLoops = classifierWireSpecs
        .filter((wire) => wire.orientationBit === 0 && wire.loop !== outerClassifierLoop)
        .map((wire) => wire.loop);

    const outerClassifierBounds = computeLoopBounds(outerClassifierLoop);
    const outerClassifierUMin = outerClassifierBounds.uMin;
    const outerClassifierUMax = outerClassifierBounds.uMax;

    let classifierDomainUMin = outerClassifierBounds.uMin;
    let classifierDomainUMax = outerClassifierBounds.uMax;
    let classifierDomainVMin = outerClassifierBounds.vMin;
    let classifierDomainVMax = outerClassifierBounds.vMax;
    const updateClassifierDomain = (bounds: { uMin: number; uMax: number; vMin: number; vMax: number }): void => {
        if (bounds.uMin < classifierDomainUMin) classifierDomainUMin = bounds.uMin;
        if (bounds.uMax > classifierDomainUMax) classifierDomainUMax = bounds.uMax;
        if (bounds.vMin < classifierDomainVMin) classifierDomainVMin = bounds.vMin;
        if (bounds.vMax > classifierDomainVMax) classifierDomainVMax = bounds.vMax;
    };
    for (const wire of classifierWireSpecs) {
        updateClassifierDomain(computeLoopBounds(wire.loop));
    }
    if (!Number.isFinite(classifierDomainUMin) || !Number.isFinite(classifierDomainVMin)) {
        classifierDomainUMin = outerClassifierUMin;
        classifierDomainUMax = outerClassifierUMax;
        classifierDomainVMin = vMin;
        classifierDomainVMax = vMax;
    }
    const classifierLoopToleranceU = Math.max(1e-7, classifierPointEpsilon);
    const classifierLoopToleranceV = Math.max(1e-7, classifierPointEpsilon);
    const classifierPreparationFailed = outerClassifierLoop.length < 3;
    const outerClassifierUSpan = Math.max(1e-9, outerClassifierUMax - outerClassifierUMin);
    const periodicEdgeBand = Math.max(
        localUvBoundaryTolerance * 2,
        Math.min(
            outerClassifierUSpan * 0.2,
            Math.max(
                0.05,
                readTrimNumber("__LOCAL_UV_CLASSIFIER_PERIODIC_EDGE_BAND__") ?? 0.2
            )
        )
    );
    const isNearClassifierUSeam = (uRaw: number): boolean => {
        if (!hasDiscontinuity) return false;
        return (
            uRaw <= outerClassifierUMin + periodicEdgeBand ||
            uRaw >= outerClassifierUMax - periodicEdgeBand
        );
    };
    const surfaceTypeForClassifier = buildOptions?.classifierSurfaceType ?? "";
    const useLocalUvTopologicalFallback =
        enableLocalUvFaceClassifierFallback &&
        surfaceTypeForClassifier === "Cone" &&
        useLocalUvClassifierShadow;
    const classifierLoopsForSafety = classifierWireSpecs.length > 0
        ? classifierWireSpecs.map((wire) => wire.loop)
        : [outerClassifierLoop, ...holeClassifierLoops];
    const coneClassifierDomainUnsafe =
        surfaceTypeForClassifier === "Cone" &&
        classifierLoopsForSafety.some((loop) => hasSelfIntersections(loop, classifierPointEpsilon));
    // OCCT keeps cone classification in periodic-domain traversal semantics even
    // when loop topology appears valid. Restricting this path to self-intersections
    // leaves a large Stage-B mismatch bucket on otherwise-valid cone trims.
    const forceConeDomainUnsafe = readTrimBoolean("__LOCAL_UV_CONE_FORCE_DOMAIN_UNSAFE__", true);
    const localClassifierDomainUnsafe =
        classifierPreparationFailed ||
        coneClassifierDomainUnsafe ||
        (forceConeDomainUnsafe && surfaceTypeForClassifier === "Cone");

    const distancePointToLoop = (point: Vec2, loop: Vec2[]): number => {
        if (loop.length < 2) return Infinity;
        const [px, py] = point;
        let minDistSq = Infinity;
        for (let i = 0; i < loop.length; i++) {
            const [x1, y1] = loop[i];
            const [x2, y2] = loop[(i + 1) % loop.length];
            const dx = x2 - x1;
            const dy = y2 - y1;
            const lenSq = dx * dx + dy * dy;
            let cx = x1;
            let cy = y1;
            if (lenSq > 1e-14) {
                let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
                t = Math.max(0, Math.min(1, t));
                cx = x1 + t * dx;
                cy = y1 + t * dy;
            }
            const distSq = (px - cx) * (px - cx) + (py - cy) * (py - cy);
            if (distSq < minDistSq) minDistSq = distSq;
        }
        return Math.sqrt(minDistSq);
    };

    interface LocalClassifierWire {
        loop: Vec2[];
        edges: LocalUvClassifierEdgeInput[];
        prepared: PreparedLoopClassifier;
        orientationBit: 0 | 1;
        bounds: { uMin: number; uMax: number; vMin: number; vMax: number };
        hasBadEdges: boolean;
    }

    interface LocalUvCandidateEval {
        decision: LocalUvDecision;
        nearBoundaryBand: boolean;
        minBoundaryDistance: number;
        badWire: boolean;
        source: "stageA" | "stageB" | "stageB_forced";
        provisionalDecision?: Exclude<LocalUvDecision, "uncertain">;
        stageBMetrics?: LocalUvTransitionMetrics;
    }

    interface LocalUvTransitionMetrics {
        probeFallbacks: number;
        bundleSkips: number;
        skippedNearVertexHits: number;
        transitionTies: number;
    }

    const localClassifierWires: LocalClassifierWire[] = [];
    const buildClassifierLoopFromEdges = (edges: LocalUvClassifierEdgeInput[] | undefined): Vec2[] | null => {
        if (!Array.isArray(edges) || edges.length === 0) return null;
        const stitched: Vec2[] = [];
        for (const edge of edges) {
            const pts = normalizeEdgePointsForClassifier(edge.points);
            if (pts.length < 2) continue;
            if (stitched.length === 0) {
                stitched.push(...pts);
            } else {
                const [lastU, lastV] = stitched[stitched.length - 1];
                const [firstU, firstV] = pts[0];
                const isSharedEndpoint =
                    Math.abs(lastU - firstU) <= classifierPointEpsilon &&
                    Math.abs(lastV - firstV) <= classifierPointEpsilon;
                if (isSharedEndpoint) stitched.push(...pts.slice(1));
                else stitched.push(...pts);
            }
        }
        if (stitched.length < 3) return null;
        return normalizeLoopForClassifier(stitched);
    };
    const addClassifierWire = (
        loop: Vec2[],
        orientationBit: 0 | 1,
        edges: LocalUvClassifierEdgeInput[] | undefined,
        hasBadEdgesHint: boolean | undefined
    ): void => {
        const sanitizedEdges = Array.isArray(edges)
            ? edges.filter((edge) => Array.isArray(edge.points) && edge.points.length >= 2)
            : [];
        const edgeLoop =
            localUvStageAUseEdgeLoop && surfaceTypeForClassifier === "Cone"
                ? buildClassifierLoopFromEdges(sanitizedEdges)
                : null;
        const loopForClassifier = edgeLoop && edgeLoop.length >= 3 ? edgeLoop : loop;
        if (loopForClassifier.length < 3) return;
        const bounds = computeLoopBounds(loopForClassifier);
        if (!Number.isFinite(bounds.uMin) || !Number.isFinite(bounds.vMin)) return;
        const prepared = buildPreparedLoopClassifier(
            loopForClassifier,
            bounds,
            classifierLoopToleranceU,
            classifierLoopToleranceV
        );
        if (!prepared) return;
        localClassifierWires.push({
            loop: loopForClassifier,
            edges: sanitizedEdges,
            prepared,
            orientationBit,
            bounds,
            hasBadEdges:
                hasBadEdgesHint === true ||
                (Array.isArray(edges) &&
                    edges.some((edge) => edge.hasHighVertexTolerance === true || edge.isDegenerate === true)),
        });
    };
    for (const wire of classifierWireSpecs) {
        addClassifierWire(wire.loop, wire.orientationBit, wire.edges, wire.hasBadEdges);
    }

    const evaluateLocalUvCandidate = (
        u: number,
        v: number
    ): LocalUvCandidateEval => {
        let minBoundaryDistance = Infinity;
        let forceOutside = false;
        let badWire = false;
        let sawOnBoundary = false;

        const recadreValueNearBounds = (
            value: number,
            boundsMin: number,
            boundsMax: number,
            period: number
        ): number => {
            if (!localUvWireLocalRecadre || period <= 1e-12 || !Number.isFinite(value)) {
                return value;
            }
            const center = (boundsMin + boundsMax) * 0.5;
            let shifted = value + Math.round((center - value) / period) * period;
            const halfPeriod = period * 0.5;
            if (shifted < boundsMin - halfPeriod) shifted += period;
            if (shifted > boundsMax + halfPeriod) shifted -= period;
            return shifted;
        };
        const recadrePointForWire = (wire: LocalClassifierWire): Vec2 => {
            const uu = hasDiscontinuity
                ? recadreValueNearBounds(u, wire.bounds.uMin, wire.bounds.uMax, 2 * PI)
                : u;
            return [uu, v];
        };

        for (const wire of localClassifierWires) {
            if (wire.hasBadEdges) {
                badWire = true;
            }
            const wirePoint = recadrePointForWire(wire);
            const relation = classifyPointAgainstLoop(wirePoint, wire.prepared);
            const boundaryDistance = distancePointToLoop(wirePoint, wire.loop);
            if (boundaryDistance < minBoundaryDistance) {
                minBoundaryDistance = boundaryDistance;
            }
            if (relation === "on") {
                sawOnBoundary = true;
                continue;
            }
            if (
                (relation === "inside" && wire.orientationBit === 0) ||
                (relation === "outside" && wire.orientationBit === 1)
            ) {
                forceOutside = true;
                break;
            }
        }

        if (sawOnBoundary) {
            return {
                decision: "uncertain",
                nearBoundaryBand: true,
                minBoundaryDistance,
                badWire,
                source: "stageA",
            };
        }

        return {
            decision: forceOutside ? "outside" : "inside",
            nearBoundaryBand: false,
            minBoundaryDistance,
            badWire,
            source: "stageA",
        };
    };

    const forEachWireSegment = (
        wire: LocalClassifierWire,
        visitor: (edge: LocalUvClassifierEdgeInput, a: Vec2, b: Vec2) => void
    ): void => {
        if (wire.edges.length > 0) {
            for (const edge of wire.edges) {
                const edgePoints = edge.points;
                if (!Array.isArray(edgePoints) || edgePoints.length < 2) continue;
                for (let i = 0; i < edgePoints.length - 1; i++) {
                    visitor(edge, edgePoints[i], edgePoints[i + 1]);
                }
            }
            return;
        }
        const syntheticEdge: LocalUvClassifierEdgeInput = {
            points: wire.loop,
            hasHighVertexTolerance: false,
            isDegenerate: false,
        };
        for (let i = 0; i < wire.loop.length; i++) {
            const a = wire.loop[i];
            const b = wire.loop[(i + 1) % wire.loop.length];
            visitor(syntheticEdge, a, b);
        }
    };

    const distancePointToWire = (point: Vec2, wire: LocalClassifierWire): number => {
        const [px, py] = point;
        let minDistSq = Infinity;
        forEachWireSegment(wire, (_edge, a, b) => {
            const dx = b[0] - a[0];
            const dy = b[1] - a[1];
            const lenSq = dx * dx + dy * dy;
            let cx = a[0];
            let cy = a[1];
            if (lenSq > 1e-14) {
                let t = ((px - a[0]) * dx + (py - a[1]) * dy) / lenSq;
                t = Math.max(0, Math.min(1, t));
                cx = a[0] + t * dx;
                cy = a[1] + t * dy;
            }
            const distSq = (px - cx) * (px - cx) + (py - cy) * (py - cy);
            if (distSq < minDistSq) minDistSq = distSq;
        });
        return minDistSq === Infinity ? Infinity : Math.sqrt(minDistSq);
    };

    interface LocalUvDirectionClassification {
        relation: LoopPointRelation;
        metrics: LocalUvTransitionMetrics;
        closestHitX: number;
        eventKind: "cross" | "touch" | "ambiguous" | "none";
        eventSign: -1 | 0 | 1;
    }

    const createTransitionMetrics = (): LocalUvTransitionMetrics => ({
        probeFallbacks: 0,
        bundleSkips: 0,
        skippedNearVertexHits: 0,
        transitionTies: 0,
    });

    const accumulateTransitionMetrics = (
        target: LocalUvTransitionMetrics,
        delta: LocalUvTransitionMetrics
    ): void => {
        target.probeFallbacks += delta.probeFallbacks;
        target.bundleSkips += delta.bundleSkips;
        target.skippedNearVertexHits += delta.skippedNearVertexHits;
        target.transitionTies += delta.transitionTies;
    };

    const classifyPointAgainstWireByDirection = (
        point: Vec2,
        wire: LocalClassifierWire,
        dir: Vec2,
        options?: { edgeToleranceScale?: number }
    ): LocalUvDirectionClassification => {
        const metrics = createTransitionMetrics();
        const [px, py] = point;
        const edgeToleranceScale = Math.max(0.01, options?.edgeToleranceScale ?? 1);
        const tolU = Math.max(
            1e-8,
            Math.max(classifierPointEpsilon, wire.prepared.tolUOriginal) * edgeToleranceScale
        );
        const tolV = Math.max(
            1e-8,
            Math.max(classifierPointEpsilon, wire.prepared.tolVOriginal) * edgeToleranceScale
        );

        if (
            px < wire.bounds.uMin - tolU ||
            px > wire.bounds.uMax + tolU ||
            py < wire.bounds.vMin - tolV ||
            py > wire.bounds.vMax + tolV
        ) {
            return {
                relation: "outside",
                metrics,
                closestHitX: Number.POSITIVE_INFINITY,
                eventKind: "none",
                eventSign: 0,
            };
        }

        const [dx, dy] = dir;
        const rayLen = Math.hypot(dx, dy);
        if (rayLen <= 1e-12) {
            return {
                relation: "outside",
                metrics,
                closestHitX: Number.POSITIVE_INFINITY,
                eventKind: "none",
                eventSign: 0,
            };
        }
        const ux = dx / rayLen;
        const uy = dy / rayLen;
        const nx = -uy;
        const ny = ux;
        type IntersectionHit = {
            x: number;
            transition: 1 | -1;
            nearVertex: boolean;
            highTolVertex: boolean;
            vertexRole: "start" | "end" | "none";
            vertexOtherY: number;
        };
        const hits: IntersectionHit[] = [];
        forEachWireSegment(wire, (edge, a, b) => {
            const edgeTolRaw =
                typeof edge.vertexToleranceMax === "number" && Number.isFinite(edge.vertexToleranceMax)
                    ? edge.vertexToleranceMax
                    : 0;
            const edgeTol = Math.max(0, edgeTolRaw);
            const edgeTolU = Math.max(tolU, edgeTol);
            const edgeTolV = Math.max(tolV, edgeTol);
            const aRelX = a[0] - px;
            const aRelY = a[1] - py;
            const bRelX = b[0] - px;
            const bRelY = b[1] - py;
            const ax = aRelX * ux + aRelY * uy;
            const ay = aRelX * nx + aRelY * ny;
            const bx = bRelX * ux + bRelY * uy;
            const by = bRelX * nx + bRelY * ny;

            const segDx = bx - ax;
            const segDy = by - ay;
            const segLen2 = segDx * segDx + segDy * segDy;
            if (segLen2 > 1e-16) {
                const t = Math.max(0, Math.min(1, ((-ax) * segDx + (-ay) * segDy) / segLen2));
                const qx = ax + t * segDx;
                const qy = ay + t * segDy;
                if (Math.abs(qx) <= edgeTolU && Math.abs(qy) <= edgeTolV) {
                    hits.length = 0;
                    hits.push({
                        x: 0,
                        transition: 1,
                        nearVertex: false,
                        highTolVertex: false,
                        vertexRole: "none",
                        vertexOtherY: 0,
                    });
                    return;
                }
            } else if (Math.abs(ax) <= edgeTolU && Math.abs(ay) <= edgeTolV) {
                hits.length = 0;
                hits.push({
                    x: 0,
                    transition: 1,
                    nearVertex: false,
                    highTolVertex: false,
                    vertexRole: "none",
                    vertexOtherY: 0,
                });
                return;
            }

            // Ignore nearly horizontal segments for crossing count.
            if (Math.abs(by - ay) <= edgeTolV) {
                return;
            }
            // Half-open vertical test avoids double counting shared vertices.
            const y1 = ay;
            const y2 = by;
            const crosses =
                (y1 <= 0 && y2 > 0) ||
                (y2 <= 0 && y1 > 0);
            if (!crosses) return;

            const xIntersect = ax + ((-ay) * (bx - ax)) / (by - ay);
            if (Math.abs(xIntersect) <= edgeTolU) {
                hits.length = 0;
                hits.push({
                    x: 0,
                    transition: 1,
                    nearVertex: false,
                    highTolVertex: false,
                    vertexRole: "none",
                    vertexOtherY: 0,
                });
                return;
            }
            if (xIntersect > edgeTolU) {
                const t = (-ay) / (by - ay);
                const nearVertexTol = Math.max(1e-6, edgeTolV / (Math.abs(by - ay) + 1e-9));
                const nearStart = t <= nearVertexTol;
                const nearEnd = t >= 1 - nearVertexTol;
                const nearVertex = nearStart || nearEnd;
                const vertexRole: "start" | "end" | "none" = nearStart ? "start" : nearEnd ? "end" : "none";
                const vertexOtherY = nearStart ? by : nearEnd ? ay : 0;
                hits.push({
                    x: xIntersect,
                    transition: by > ay ? 1 : -1,
                    nearVertex,
                    highTolVertex: edge.hasHighVertexTolerance === true,
                    vertexRole,
                    vertexOtherY,
                });
            }
        });

        if (hits.length === 1 && hits[0].x === 0) {
            return {
                relation: "on",
                metrics,
                closestHitX: 0,
                eventKind: "touch",
                eventSign: 0,
            };
        }

        if (hits.length === 0) {
            return {
                relation: "outside",
                metrics,
                closestHitX: Number.POSITIVE_INFINITY,
                eventKind: "none",
                eventSign: 0,
            };
        }
        hits.sort((a, b) => a.x - b.x);

        const closestBandTol = Math.max(tolU * 2, 1e-7);
        const hitBundles: IntersectionHit[][] = [];
        for (const hit of hits) {
            const last = hitBundles[hitBundles.length - 1];
            if (!last || Math.abs(hit.x - last[0].x) > closestBandTol) {
                hitBundles.push([hit]);
            } else {
                last.push(hit);
            }
        }
        const isUnstableBundle = (bundle: IntersectionHit[]): boolean => {
            if (bundle.length === 0) return false;
            const allNearVertex = bundle.every((hit) => hit.nearVertex);
            if (!allNearVertex) return false;
            const hasHighTol = bundle.some((hit) => hit.highTolVertex);
            if (hasHighTol) return true;
            // OCCT CheckSkip analogue extension: even without explicit high vertex
            // tolerance metadata, a closest near-vertex bundle with mixed in/out
            // transitions is unstable and should be bridged to the next bundle.
            const hasPositive = bundle.some((hit) => hit.transition > 0);
            const hasNegative = bundle.some((hit) => hit.transition < 0);
            return hasPositive && hasNegative;
        };
        const resolveComplexVertexBundle = (
            bundle: IntersectionHit[]
        ): { resolved: boolean; transition: -1 | 0 | 1 } => {
            const nearHits = bundle.filter((hit) => hit.nearVertex);
            if (nearHits.length < 2) {
                return { resolved: false, transition: 0 };
            }
            const nonNearHits = bundle.filter((hit) => !hit.nearVertex);
            if (nonNearHits.length > 0) {
                return { resolved: false, transition: 0 };
            }

            let otherAbove = 0;
            let otherBelow = 0;
            let transitionSum = 0;
            for (const hit of nearHits) {
                transitionSum += hit.transition;
                if (hit.vertexOtherY > tolV) otherAbove++;
                else if (hit.vertexOtherY < -tolV) otherBelow++;
            }

            // Same-side neighboring endpoints => vertex-touch (local extremum).
            if (otherAbove === 0 || otherBelow === 0) {
                return { resolved: true, transition: 0 };
            }

            // Endpoints on both sides of the probe line => true crossing through vertex.
            if (transitionSum > 0) return { resolved: true, transition: 1 };
            if (transitionSum < 0) return { resolved: true, transition: -1 };

            // If transition sum still cancels, prefer stable non-high-tolerance hint.
            const stableHit = nearHits.find((hit) => !hit.highTolVertex);
            if (stableHit) {
                return { resolved: true, transition: stableHit.transition };
            }
            return { resolved: true, transition: nearHits[0]?.transition ?? 0 };
        };

        type BundleEvent =
            | { kind: "cross"; sign: -1 | 1; count: number; hasPositive: boolean; hasNegative: boolean }
            | { kind: "touch"; count: number; hasPositive: boolean; hasNegative: boolean }
            | { kind: "ambiguous" };

        const chooseWeightedMixedSign = (
            hitsSubset: IntersectionHit[]
        ): -1 | 1 | null => {
            if (hitsSubset.length === 0) return null;
            let posWeight = 0;
            let negWeight = 0;
            for (const hit of hitsSubset) {
                let w = 1;
                if (!hit.nearVertex) w += 3;
                if (!hit.highTolVertex) w += 2;
                if (Math.abs(hit.vertexOtherY) > tolV) w += 1;
                if (hit.transition > 0) posWeight += w;
                else negWeight += w;
            }
            if (posWeight === negWeight) return null;
            return posWeight > negWeight ? 1 : -1;
        };

        const evaluateBundleEvent = (
            bundle: IntersectionHit[],
            skipHighTolNearVertex: boolean
        ): BundleEvent => {
            const filtered: IntersectionHit[] = [];
            for (const hit of bundle) {
                const skip =
                    skipHighTolNearVertex &&
                    hit.nearVertex &&
                    hit.highTolVertex &&
                    bundle.length > 1;
                if (skip) {
                    metrics.skippedNearVertexHits++;
                    continue;
                }
                filtered.push(hit);
            }
            if (filtered.length === 0) {
                return { kind: "ambiguous" };
            }

            const complexVertex = resolveComplexVertexBundle(filtered);
            if (complexVertex.resolved) {
                if (complexVertex.transition === 0) {
                    return {
                        kind: "touch",
                        count: filtered.length,
                        hasPositive: true,
                        hasNegative: true,
                    };
                }
                return {
                    kind: "cross",
                    sign: complexVertex.transition,
                    count: filtered.length,
                    hasPositive: complexVertex.transition > 0,
                    hasNegative: complexVertex.transition < 0,
                };
            }
            let sum = 0;
            let count = 0;
            let hasPositive = false;
            let hasNegative = false;
            for (const hit of filtered) {
                sum += hit.transition;
                count++;
                if (hit.transition > 0) hasPositive = true;
                if (hit.transition < 0) hasNegative = true;
            }
            if (count === 0) {
                return { kind: "ambiguous" };
            }
            if (hasPositive && hasNegative) {
                if (sum === 0) {
                    return { kind: "touch", count, hasPositive, hasNegative };
                }
                if (localUvStageBWeightedMixedSign) {
                    const stable = filtered.filter((hit) => !hit.nearVertex && !hit.highTolVertex);
                    const semiStable = filtered.filter((hit) => !hit.highTolVertex);
                    const weightedSign =
                        chooseWeightedMixedSign(stable) ??
                        chooseWeightedMixedSign(semiStable);
                    if (weightedSign != null) {
                        return {
                            kind: "cross",
                            sign: weightedSign,
                            count,
                            hasPositive,
                            hasNegative,
                        };
                    }
                }
                return {
                    kind: "cross",
                    sign: sum > 0 ? 1 : -1,
                    count,
                    hasPositive,
                    hasNegative,
                };
            }
            return {
                kind: "cross",
                sign: hasPositive ? 1 : -1,
                count,
                hasPositive,
                hasNegative,
            };
        };

        // OCCT CheckSkip analogue:
        // if the leading intersection bundle is entirely unstable (high-tolerance
        // near-vertex), bridge to the next bundle before classification.
        let bundleIndex = 0;
        while (
            bundleIndex < hitBundles.length - 1 &&
            isUnstableBundle(hitBundles[bundleIndex])
        ) {
            bundleIndex++;
            metrics.bundleSkips++;
        }
        if (!localUvStageBAccumulateCrossings) {
            // OCCT-like closest-hit ownership: classify from the nearest stable
            // intersection bundle only. If that bundle is touch/ambiguous, let the
            // caller try another probe direction.
            if (bundleIndex >= hitBundles.length) {
                return {
                    relation: "outside",
                    metrics,
                    closestHitX: Number.POSITIVE_INFINITY,
                    eventKind: "none",
                    eventSign: 0,
                };
            }
            const firstBundle = hitBundles[bundleIndex];
            const closestHitX = firstBundle[0]?.x ?? Number.POSITIVE_INFINITY;
            let event = evaluateBundleEvent(firstBundle, true);
            if (event.kind === "ambiguous") {
                event = evaluateBundleEvent(firstBundle, false);
            }
            if (event.kind === "cross") {
                return {
                    relation: event.sign > 0 ? "inside" : "outside",
                    metrics,
                    closestHitX,
                    eventKind: "cross",
                    eventSign: event.sign,
                };
            }
            if (event.kind === "touch") {
                if (event.count > 1 && event.hasPositive && event.hasNegative) {
                    metrics.transitionTies++;
                }
                return {
                    relation: "on",
                    metrics,
                    closestHitX,
                    eventKind: "touch",
                    eventSign: 0,
                };
            }
            metrics.transitionTies++;
            return {
                relation: "on",
                metrics,
                closestHitX,
                eventKind: "ambiguous",
                eventSign: 0,
            };
        }

        let sawTouchEvent = false;
        let sawAmbiguousEvent = false;
        let sawCrossEvent = false;
        let crossingSum = 0;
        let firstCrossSign: -1 | 1 | 0 = 0;
        for (let nextBundleIndex = bundleIndex; nextBundleIndex < hitBundles.length; nextBundleIndex++) {
            const nextBundle = hitBundles[nextBundleIndex];
            if (isUnstableBundle(nextBundle)) {
                metrics.bundleSkips++;
                continue;
            }
            let event = evaluateBundleEvent(nextBundle, true);
            if (event.kind === "ambiguous") {
                event = evaluateBundleEvent(nextBundle, false);
            }
            if (event.kind === "cross") {
                sawCrossEvent = true;
                if (firstCrossSign === 0) firstCrossSign = event.sign;
                crossingSum += event.sign;
                continue;
            }
            if (event.kind === "touch") {
                sawTouchEvent = true;
                if (event.count > 1 && event.hasPositive && event.hasNegative) {
                    metrics.transitionTies++;
                }
                continue;
            }
            sawAmbiguousEvent = true;
            metrics.transitionTies++;
        }

        if (sawTouchEvent || sawAmbiguousEvent) {
            return {
                relation: "on",
                metrics,
                closestHitX: hitBundles[bundleIndex]?.[0]?.x ?? Number.POSITIVE_INFINITY,
                eventKind: sawAmbiguousEvent ? "ambiguous" : "touch",
                eventSign: 0,
            };
        }
        if (sawCrossEvent) {
            if (crossingSum > 0) {
                return {
                    relation: "inside",
                    metrics,
                    closestHitX: hitBundles[bundleIndex]?.[0]?.x ?? Number.POSITIVE_INFINITY,
                    eventKind: "cross",
                    eventSign: 1,
                };
            }
            if (crossingSum < 0) {
                return {
                    relation: "outside",
                    metrics,
                    closestHitX: hitBundles[bundleIndex]?.[0]?.x ?? Number.POSITIVE_INFINITY,
                    eventKind: "cross",
                    eventSign: -1,
                };
            }
            if (firstCrossSign > 0) {
                return {
                    relation: "inside",
                    metrics,
                    closestHitX: hitBundles[bundleIndex]?.[0]?.x ?? Number.POSITIVE_INFINITY,
                    eventKind: "cross",
                    eventSign: 1,
                };
            }
            if (firstCrossSign < 0) {
                return {
                    relation: "outside",
                    metrics,
                    closestHitX: hitBundles[bundleIndex]?.[0]?.x ?? Number.POSITIVE_INFINITY,
                    eventKind: "cross",
                    eventSign: -1,
                };
            }
            return {
                relation: "outside",
                metrics,
                closestHitX: hitBundles[bundleIndex]?.[0]?.x ?? Number.POSITIVE_INFINITY,
                eventKind: "none",
                eventSign: 0,
            };
        }
        return {
            relation: "outside",
            metrics,
            closestHitX: Number.POSITIVE_INFINITY,
            eventKind: "none",
            eventSign: 0,
        };
    };

    interface LocalUvTransitionClassification {
        relation: LoopPointRelation;
        metrics: LocalUvTransitionMetrics;
    }

    const appendProbeCandidatesForWire = (
        point: Vec2,
        wire: LocalClassifierWire,
        outCandidates: Array<{ dir: Vec2; quality: number }>,
        maxCandidates = 16
    ): void => {
        if (outCandidates.length >= maxCandidates) return;
        // OCCT-inspired probing constants from FaceExplorer.cxx
        const probeStart = 0.123;
        const probeEnd = 0.7;
        const probeStep = 0.2111;
        const smallAngleSin = 0.001;
        const [px, py] = point;
        forEachWireSegment(wire, (_edge, a, b) => {
            const ex = b[0] - a[0];
            const ey = b[1] - a[1];
            const eLen = Math.hypot(ex, ey);
            if (eLen <= 1e-12 || outCandidates.length >= maxCandidates) return;
            const etx = ex / eLen;
            const ety = ey / eLen;

            for (let t = probeStart; t < probeEnd && outCandidates.length < maxCandidates; t += probeStep) {
                const sx = a[0] * t + b[0] * (1 - t);
                const sy = a[1] * t + b[1] * (1 - t);
                const vx = sx - px;
                const vy = sy - py;
                const vLen = Math.hypot(vx, vy);
                if (vLen <= 1e-12) continue;
                const vtx = vx / vLen;
                const vty = vy / vLen;
                const sinA = Math.abs(etx * vty - ety * vtx);
                if (sinA < smallAngleSin) continue;
                outCandidates.push({ dir: [vtx, vty], quality: sinA });
            }
        });
    };

    const finalizeProbeDirections = (
        probeCandidates: Array<{ dir: Vec2; quality: number }>,
        maxProbeDirections: number
    ): Vec2[] => {
        if (probeCandidates.length === 0) {
            return [[1, 0]];
        }
        probeCandidates.sort((a, b) => b.quality - a.quality);
        const probeDirections: Vec2[] = [];
        const dedupeDot = 0.9995;
        for (const candidate of probeCandidates) {
            const [cx, cy] = candidate.dir;
            let duplicate = false;
            for (const existing of probeDirections) {
                const dot = cx * existing[0] + cy * existing[1];
                if (Math.abs(dot) >= dedupeDot) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            probeDirections.push(candidate.dir);
            if (probeDirections.length >= maxProbeDirections) break;
        }
        if (probeDirections.length === 0) {
            probeDirections.push([1, 0]);
        }
        return probeDirections;
    };

    const classifyPointAgainstWireByTransitions = (
        point: Vec2,
        wire: LocalClassifierWire,
        options?: { probeDirections?: Vec2[]; edgeToleranceScale?: number; maxProbeDirections?: number }
    ): LocalUvTransitionClassification => {
        const metrics = createTransitionMetrics();
        const edgeToleranceScale = Math.max(0.01, options?.edgeToleranceScale ?? localUvTransitionEdgeToleranceScale);
        const maxProbeDirections = Math.max(1, Math.floor(options?.maxProbeDirections ?? localUvTransitionMaxProbes));
        let probeDirections: Vec2[];
        if (Array.isArray(options?.probeDirections) && options.probeDirections.length > 0) {
            probeDirections = options.probeDirections;
        } else {
            const probeCandidates: Array<{ dir: Vec2; quality: number }> = [];
            appendProbeCandidatesForWire(point, wire, probeCandidates, 16);
            if (probeCandidates.length === 0) {
                metrics.probeFallbacks++;
            }
            probeDirections = finalizeProbeDirections(probeCandidates, maxProbeDirections);
        }

        let firstResolved: LoopPointRelation | null = null;
        let sawInside = false;
        let sawOutside = false;
        for (const direction of probeDirections) {
            const directional = classifyPointAgainstWireByDirection(point, wire, direction, {
                edgeToleranceScale,
            });
            accumulateTransitionMetrics(metrics, directional.metrics);
            if (directional.relation === "on") {
                continue;
            }
            if (firstResolved == null) {
                firstResolved = directional.relation;
                if (!localUvStageBMultiProbeConsensus) {
                    return { relation: directional.relation, metrics };
                }
            }
            if (directional.relation === "inside") sawInside = true;
            else sawOutside = true;
        }

        if (firstResolved == null) {
            // Every tested probe ended up ON/ambiguous.
            metrics.transitionTies++;
            return { relation: "on", metrics };
        }

        if (!localUvStageBMultiProbeConsensus) {
            return { relation: firstResolved, metrics };
        }

        if (sawInside && sawOutside) {
            // OCCT Segment/OtherSegment inspiration: if alternate robust probes disagree,
            // avoid trusting a single segment. Prefer conservative OUT unless configured otherwise.
            metrics.transitionTies++;
            return {
                relation: localUvStageBMultiProbePreferOutside ? "outside" : "on",
                metrics,
            };
        }

        return { relation: firstResolved, metrics };
    };

    const evaluateLocalUvCandidateByWireTransitions = (
        u: number,
        v: number,
        options?: { allowBoundaryBandUncertain?: boolean }
    ): LocalUvCandidateEval => {
        let minBoundaryDistance = Infinity;
        let forceOutside = false;
        let badWire = false;
        const transitionMetrics = createTransitionMetrics();
        const recadreValueNearBounds = (
            value: number,
            boundsMin: number,
            boundsMax: number,
            period: number
        ): number => {
            if (!localUvWireLocalRecadre || period <= 1e-12 || !Number.isFinite(value)) {
                return value;
            }
            const center = (boundsMin + boundsMax) * 0.5;
            let shifted = value + Math.round((center - value) / period) * period;
            const halfPeriod = period * 0.5;
            if (shifted < boundsMin - halfPeriod) shifted += period;
            if (shifted > boundsMax + halfPeriod) shifted -= period;
            return shifted;
        };
        const recadrePointForWire = (wire: LocalClassifierWire): Vec2 => {
            const uu = hasDiscontinuity
                ? recadreValueNearBounds(u, wire.bounds.uMin, wire.bounds.uMax, 2 * PI)
                : u;
            return [uu, v];
        };

        if (localUvStageBFaceWideProbeTraversal) {
            const wirePoints = localClassifierWires.map((wire) => recadrePointForWire(wire));
            const probeCandidates: Array<{ dir: Vec2; quality: number }> = [];
            for (let wireIndex = 0; wireIndex < localClassifierWires.length; wireIndex++) {
                appendProbeCandidatesForWire(wirePoints[wireIndex], localClassifierWires[wireIndex], probeCandidates, 64);
            }
            if (probeCandidates.length === 0) {
                transitionMetrics.probeFallbacks++;
            }
            const probeDirections = finalizeProbeDirections(probeCandidates, localUvTransitionMaxProbes);

            let firstResolvedDecision: LocalUvDecision | null = null;
            let firstResolvedDistance = Infinity;
            let firstResolvedBadWire = false;
            let sawInside = false;
            let sawOutside = false;
            const closestCompareTol = Math.max(
                classifierPointEpsilon * 4,
                localUvBoundaryTolerance * 0.25,
                1e-7
            );

            for (const direction of probeDirections) {
                let directionBadWire = false;
                let directionMinBoundaryDistance = Infinity;
                let directionDecision: LocalUvDecision | null = null;
                let directionClosestHitX = Number.POSITIVE_INFINITY;
                let directionHadTie = false;
                const directionMetrics = createTransitionMetrics();
                let closestInsideVotes = 0;
                let closestOutsideVotes = 0;
                let closestCrossSignSum = 0;
                let closestCrossCount = 0;
                let closestTouchCount = 0;
                let closestAmbiguousCount = 0;

                for (let wireIndex = 0; wireIndex < localClassifierWires.length; wireIndex++) {
                    const wire = localClassifierWires[wireIndex];
                    if (wire.hasBadEdges) {
                        directionBadWire = true;
                    }
                    const wirePoint = wirePoints[wireIndex];
                    const boundaryDistance = distancePointToWire(wirePoint, wire);
                    if (boundaryDistance < directionMinBoundaryDistance) {
                        directionMinBoundaryDistance = boundaryDistance;
                    }
                    const directional = classifyPointAgainstWireByDirection(
                        wirePoint,
                        wire,
                        direction,
                        { edgeToleranceScale: localUvTransitionEdgeToleranceScale }
                    );
                    accumulateTransitionMetrics(directionMetrics, directional.metrics);

                    if (!Number.isFinite(directional.closestHitX)) {
                        continue;
                    }

                    const wireDecision: LocalUvDecision =
                        directional.relation === "on"
                            ? "uncertain"
                            : (
                                (directional.relation === "inside" && wire.orientationBit === 0) ||
                                (directional.relation === "outside" && wire.orientationBit === 1)
                            )
                                ? "outside"
                                : "inside";
                    const voteDecision = (decision: LocalUvDecision): void => {
                        if (decision === "inside") closestInsideVotes++;
                        else if (decision === "outside") closestOutsideVotes++;
                    };
                    const voteTransition = (): void => {
                        if (!localUvStageBTopTransTieAccumulator) {
                            return;
                        }
                        if (directional.eventKind === "cross" && directional.eventSign !== 0) {
                            const transitionDecisionSign =
                                wire.orientationBit === 0 ? -directional.eventSign : directional.eventSign;
                            closestCrossSignSum += transitionDecisionSign;
                            closestCrossCount++;
                        } else if (directional.eventKind === "touch") {
                            closestTouchCount++;
                        } else if (directional.eventKind === "ambiguous") {
                            closestAmbiguousCount++;
                        }
                    };
                    const hitDelta = directional.closestHitX - directionClosestHitX;
                    if (hitDelta < -closestCompareTol) {
                        directionClosestHitX = directional.closestHitX;
                        directionDecision = wireDecision;
                        directionHadTie = false;
                        closestInsideVotes = 0;
                        closestOutsideVotes = 0;
                        closestCrossSignSum = 0;
                        closestCrossCount = 0;
                        closestTouchCount = 0;
                        closestAmbiguousCount = 0;
                        voteDecision(wireDecision);
                        voteTransition();
                    } else if (Math.abs(hitDelta) <= closestCompareTol) {
                        voteDecision(wireDecision);
                        voteTransition();
                        if (directionDecision == null) {
                            directionDecision = wireDecision;
                        } else if (directionDecision !== wireDecision) {
                            directionHadTie = true;
                            if (localUvStageBTopTransTieAccumulator && closestCrossCount > 0) {
                                if (closestCrossSignSum > 0) {
                                    directionDecision = "inside";
                                } else if (closestCrossSignSum < 0) {
                                    directionDecision = "outside";
                                } else {
                                    directionDecision = "uncertain";
                                }
                            } else if (closestInsideVotes !== closestOutsideVotes) {
                                directionDecision =
                                    closestInsideVotes > closestOutsideVotes ? "inside" : "outside";
                            } else {
                                directionDecision = "uncertain";
                            }
                        }
                    }
                }

                if (
                    localUvStageBTopTransTieAccumulator &&
                    directionHadTie &&
                    directionDecision === "uncertain"
                ) {
                    if (closestCrossCount > 0 && closestCrossSignSum !== 0) {
                        directionDecision = closestCrossSignSum > 0 ? "inside" : "outside";
                    } else if (closestInsideVotes !== closestOutsideVotes) {
                        directionDecision =
                            closestInsideVotes > closestOutsideVotes ? "inside" : "outside";
                    } else if (closestAmbiguousCount === 0 && closestTouchCount > 0) {
                        directionDecision = localUvStageBMultiProbePreferOutside ? "outside" : "uncertain";
                    }
                }

                accumulateTransitionMetrics(transitionMetrics, directionMetrics);
                if (directionHadTie) {
                    transitionMetrics.transitionTies++;
                }
                if (directionDecision == null) {
                    // No intersections on this probe direction; OCCT classifier treats as OUT.
                    directionDecision = "outside";
                }
                if (directionDecision === "uncertain") {
                    continue;
                }

                if (firstResolvedDecision == null) {
                    firstResolvedDecision = directionDecision;
                    firstResolvedDistance = directionMinBoundaryDistance;
                    firstResolvedBadWire = directionBadWire;
                    if (!localUvStageBMultiProbeConsensus) {
                        break;
                    }
                }
                if (directionDecision === "inside") {
                    sawInside = true;
                } else {
                    sawOutside = true;
                }
            }

            if (firstResolvedDecision == null) {
                transitionMetrics.transitionTies++;
                return {
                    decision: "uncertain",
                    nearBoundaryBand: true,
                    minBoundaryDistance,
                    badWire,
                    source: "stageB",
                    stageBMetrics: transitionMetrics,
                };
            }

            minBoundaryDistance = firstResolvedDistance;
            badWire = firstResolvedBadWire;
            let stageBDecision: LocalUvDecision = firstResolvedDecision;
            if (localUvStageBMultiProbeConsensus && sawInside && sawOutside) {
                transitionMetrics.transitionTies++;
                stageBDecision = localUvStageBMultiProbePreferOutside ? "outside" : "uncertain";
            }
            if (stageBDecision === "uncertain") {
                return {
                    decision: "uncertain",
                    nearBoundaryBand: true,
                    minBoundaryDistance,
                    badWire,
                    source: "stageB",
                    stageBMetrics: transitionMetrics,
                };
            }
            forceOutside = stageBDecision === "outside";
        } else {
            for (const wire of localClassifierWires) {
                if (wire.hasBadEdges) {
                    badWire = true;
                }
                const wirePoint = recadrePointForWire(wire);
                const transitionClassification = classifyPointAgainstWireByTransitions(wirePoint, wire, {
                    edgeToleranceScale: localUvTransitionEdgeToleranceScale,
                    maxProbeDirections: localUvTransitionMaxProbes,
                });
                accumulateTransitionMetrics(transitionMetrics, transitionClassification.metrics);
                const relation = transitionClassification.relation;
                const boundaryDistance = distancePointToWire(wirePoint, wire);
                if (boundaryDistance < minBoundaryDistance) {
                    minBoundaryDistance = boundaryDistance;
                }
                if (relation === "on") {
                    return {
                        decision: "uncertain",
                        nearBoundaryBand: true,
                        minBoundaryDistance,
                        badWire,
                        source: "stageB",
                        stageBMetrics: transitionMetrics,
                    };
                }
                if (
                    (relation === "inside" && wire.orientationBit === 0) ||
                    (relation === "outside" && wire.orientationBit === 1)
                ) {
                    forceOutside = true;
                }
            }
        }

        const nearBoundaryBand =
            !skipNearBoundaryChecks &&
            localUvBoundaryTolerance > classifierPointEpsilon &&
            minBoundaryDistance <= localUvBoundaryTolerance;
        const allowBoundaryBandUncertain = options?.allowBoundaryBandUncertain ?? true;
        if (nearBoundaryBand && allowBoundaryBandUncertain) {
            return {
                decision: "uncertain",
                nearBoundaryBand: true,
                minBoundaryDistance,
                badWire,
                source: "stageB",
                stageBMetrics: transitionMetrics,
            };
        }

        return {
            decision: forceOutside ? "outside" : "inside",
            nearBoundaryBand,
            minBoundaryDistance,
            badWire,
            source: "stageB",
            stageBMetrics: transitionMetrics,
        };
    };

    const recadrePeriodic = (value: number, minValue: number, period: number): number => {
        if (period <= 1e-12 || !Number.isFinite(value) || !Number.isFinite(minValue)) {
            return value;
        }
        let recadred = value;
        if (recadred < minValue) {
            while (recadred < minValue) {
                recadred += period;
            }
            return recadred;
        }
        while (recadred >= minValue) {
            recadred -= period;
        }
        return recadred + period;
    };

    // OCCT-inspired periodic traversal:
    // evaluate one shifted point against all wires, then iterate periodic images
    // until IN/ON is found or the periodic domain is exhausted.
    const evaluateLocalUvWithPeriodicTraversal = (
        uRaw: number,
        vRaw: number,
        options?: {
            forceStageB?: boolean;
            disableStageB?: boolean;
            allowStageABoundaryBandUncertain?: boolean;
        }
    ): LocalUvCandidateEval => {
        if (localClassifierWires.length === 0) {
            return {
                decision: "uncertain",
                nearBoundaryBand: true,
                minBoundaryDistance: Infinity,
                badWire: true,
                source: "stageA",
            };
        }

        const isUPeriodic = hasDiscontinuity;
        const isVPeriodic = false;
        const uPeriod = isUPeriodic ? 2 * PI : 0;
        const vPeriod = isVPeriodic ? 2 * PI : 0;

        let u = uRaw;
        let v = vRaw;
        let uu = uRaw;
        let vv = vRaw;
        if (isUPeriodic) {
            uu = recadrePeriodic(uRaw, classifierDomainUMin, uPeriod);
        }
        if (isVPeriodic) {
            vv = recadrePeriodic(vRaw, classifierDomainVMin, vPeriod);
        }

        let uRecadred = false;
        let vRecadred = false;
        let guard = 0;
        let lastEval: LocalUvCandidateEval = {
            decision: "outside",
            nearBoundaryBand: false,
            minBoundaryDistance: Infinity,
            badWire: false,
            source: "stageA",
        };
        const forceStageB = options?.forceStageB === true;
        const disableStageB = options?.disableStageB === true;

        while (guard < 64) {
            guard++;
            const stageAEval = forceStageB
                ? {
                    decision: "uncertain" as const,
                    nearBoundaryBand: false,
                    minBoundaryDistance: Infinity,
                    badWire: true,
                    source: "stageA" as const,
                }
                : evaluateLocalUvCandidate(u, v);
            if (!forceStageB) {
                localUvClassifierStats.stageAEvaluations++;
                if (stageAEval.decision === "inside") {
                    localUvClassifierStats.stageAInside++;
                } else if (stageAEval.decision === "outside") {
                    localUvClassifierStats.stageAOutside++;
                } else {
                    localUvClassifierStats.stageAUncertain++;
                }
                if (stageAEval.badWire) {
                    localUvClassifierStats.stageABadWire++;
                }
            }
            let evalAtPoint = stageAEval;
            const shouldRunStageB =
                !disableStageB &&
                useLocalUvTopologicalFallback &&
                (forceStageB || stageAEval.decision === "uncertain" || stageAEval.badWire);
            if (shouldRunStageB) {
                localUvClassifierStats.stageBEvaluations++;
                if (forceStageB) {
                    localUvClassifierStats.stageBForcedEvaluations++;
                }
                if (!forceStageB && stageAEval.decision === "uncertain") {
                    localUvClassifierStats.stageBTriggeredByUncertain++;
                }
                if (!forceStageB && stageAEval.badWire) {
                    localUvClassifierStats.stageBTriggeredByBadWire++;
                }
                const stageBEval = evaluateLocalUvCandidateByWireTransitions(u, v, {
                    allowBoundaryBandUncertain: false,
                });
                if (stageBEval.decision === "inside") {
                    localUvClassifierStats.stageBInside++;
                } else if (stageBEval.decision === "outside") {
                    localUvClassifierStats.stageBOutside++;
                } else {
                    localUvClassifierStats.stageBUncertain++;
                }
                if (stageBEval.stageBMetrics) {
                    localUvClassifierStats.stageBProbeFallbacks += stageBEval.stageBMetrics.probeFallbacks;
                    localUvClassifierStats.stageBBundleSkips += stageBEval.stageBMetrics.bundleSkips;
                    localUvClassifierStats.stageBSkippedNearVertexHits += stageBEval.stageBMetrics.skippedNearVertexHits;
                    localUvClassifierStats.stageBTransitionTies += stageBEval.stageBMetrics.transitionTies;
                }
                // OCCT-style flow: Stage-B owns ambiguous/bad-wire paths.
                if (forceStageB) {
                    evalAtPoint = {
                        ...stageBEval,
                        source: "stageB_forced",
                    };
                } else if (stageAEval.badWire) {
                    localUvClassifierStats.stageBResolvedByBadWire++;
                    if (stageBEval.decision === "uncertain") {
                        evalAtPoint = {
                            decision: "uncertain",
                            nearBoundaryBand: stageAEval.nearBoundaryBand || stageBEval.nearBoundaryBand,
                            minBoundaryDistance: Math.min(stageAEval.minBoundaryDistance, stageBEval.minBoundaryDistance),
                            badWire: true,
                            source: "stageB",
                            stageBMetrics: stageBEval.stageBMetrics,
                        };
                    } else {
                        evalAtPoint = {
                            ...stageBEval,
                            source: "stageB",
                        };
                    }
                } else if (stageAEval.decision === "uncertain" && stageBEval.decision !== "uncertain") {
                    localUvClassifierStats.stageBResolvedByUncertain++;
                    evalAtPoint = {
                        decision: stageBEval.decision,
                        nearBoundaryBand: stageAEval.nearBoundaryBand || stageBEval.nearBoundaryBand,
                        minBoundaryDistance: Math.min(stageAEval.minBoundaryDistance, stageBEval.minBoundaryDistance),
                        badWire: stageAEval.badWire || stageBEval.badWire,
                        source: "stageB",
                        stageBMetrics: stageBEval.stageBMetrics,
                    };
                }
            }
            lastEval = evalAtPoint;

            if (!isUPeriodic && !isVPeriodic) {
                return evalAtPoint;
            }

            // OCCT returns early for IN/ON.
            if (evalAtPoint.decision === "inside" || evalAtPoint.decision === "uncertain") {
                return evalAtPoint;
            }

            if (!uRecadred) {
                u = uu;
                uRecadred = true;
            } else if (isUPeriodic) {
                u += uPeriod;
            }

            if (u > classifierDomainUMax + classifierPointEpsilon || !isUPeriodic) {
                if (!vRecadred) {
                    v = vv;
                    vRecadred = true;
                } else if (isVPeriodic) {
                    v += vPeriod;
                }
                u = uu;

                if (v > classifierDomainVMax + classifierPointEpsilon || !isVPeriodic) {
                    return lastEval;
                }
            }
        }

        return lastEval;
    };

    const isSeamProximate = (uRaw: number): boolean => {
        if (!hasDiscontinuity) return false;
        return isNearClassifierUSeam(uRaw) || Math.abs(Math.abs(uRaw) - PI) <= localUvSeamTolerance;
    };

    let localUvPolarityInverted = false;
    let localUvPolarityCalibrationSamples = 0;
    let localUvPolarityCalibrationDirectMismatch = 0;
    let localUvPolarityCalibrationInvertedMismatch = 0;
    type DomainFaceRoutingPolicy = "default" | "prefer_stageB_disagreement";
    type DomainDisagreementPreference = 0 | 1 | 2; // 0 unknown, 1 prefer Stage-A, 2 prefer Stage-B
    const enableDomainDisagreementMap = readTrimBoolean("__LOCAL_UV_DOMAIN_DISAGREEMENT_MAP__", false);

    let domainDisableStageAProvisionalInside = false;
    const domainFaceRoutingPolicy: DomainFaceRoutingPolicy = (() => {
        if (
            !localClassifierDomainUnsafe ||
            localUvDomainUnsafeStageBOnly ||
            !useLocalUvTopologicalFallback ||
            !localUvDomainUnsafeForceStageB ||
            localClassifierWires.length === 0
        ) {
            return "default";
        }

        const sampleGrid = Math.max(
            4,
            Math.floor(readTrimNumber("__LOCAL_UV_DOMAIN_POLICY_SAMPLE_GRID__") ?? 9)
        );
        const minStageAUncertain = Math.max(
            16,
            Math.floor(readTrimNumber("__LOCAL_UV_DOMAIN_POLICY_MIN_STAGEA_UNCERTAIN__") ?? 48)
        );

        let stageAInside = 0;
        let stageAOutside = 0;
        let stageAUncertain = 0;
        let stageBInside = 0;
        let stageBOutside = 0;
        let stageBUncertain = 0;
        let disagreeAInsideBOutside = 0;
        let disagreeAOutsideBInside = 0;
        let stageAProvisionalInside = 0;
        let stageAProvisionalInsideVsStageBOutside = 0;

        const uSpan = Math.max(1e-9, uMax - uMin);
        const vSpan = Math.max(1e-9, vMax - vMin);
        for (let j = 1; j < sampleGrid; j++) {
            for (let i = 1; i < sampleGrid; i++) {
                const u = uMin + (uSpan * i) / sampleGrid;
                const v = vMin + (vSpan * j) / sampleGrid;
                const stageAOnly = evaluateLocalUvWithPeriodicTraversal(u, v, {
                    disableStageB: true,
                    allowStageABoundaryBandUncertain: true,
                });
                const forcedStageB = evaluateLocalUvWithPeriodicTraversal(u, v, { forceStageB: true });

                if (stageAOnly.decision === "inside") stageAInside++;
                else if (stageAOnly.decision === "outside") stageAOutside++;
                else stageAUncertain++;
                if (stageAOnly.decision === "uncertain" && stageAOnly.provisionalDecision === "inside") {
                    stageAProvisionalInside++;
                    if (forcedStageB.decision === "outside") {
                        stageAProvisionalInsideVsStageBOutside++;
                    }
                }

                if (forcedStageB.decision === "inside") stageBInside++;
                else if (forcedStageB.decision === "outside") stageBOutside++;
                else stageBUncertain++;

                if (
                    stageAOnly.decision !== "uncertain" &&
                    forcedStageB.decision !== "uncertain" &&
                    stageAOnly.decision !== forcedStageB.decision
                ) {
                    if (stageAOnly.decision === "inside" && forcedStageB.decision === "outside") {
                        disagreeAInsideBOutside++;
                    } else if (stageAOnly.decision === "outside" && forcedStageB.decision === "inside") {
                        disagreeAOutsideBInside++;
                    }
                }
            }
        }

        if (
            stageAProvisionalInside >= localUvDomainStageAProvisionalInsideMinSamples &&
            stageAProvisionalInsideVsStageBOutside >=
                stageAProvisionalInside * localUvDomainStageAProvisionalInsideOutsideRatio
        ) {
            domainDisableStageAProvisionalInside = true;
        }

        const pathologicalStageACluster =
            stageAInside === 0 &&
            stageAOutside > 0 &&
            stageAUncertain >= minStageAUncertain &&
            stageBOutside > stageBInside &&
            disagreeAInsideBOutside > 0;

        if (pathologicalStageACluster) {
            return "prefer_stageB_disagreement";
        }
        return "default";
    })();
    const domainDisagreementGridSize = Math.max(
        4,
        Math.floor(readTrimNumber("__LOCAL_UV_DOMAIN_DISAGREEMENT_MAP_GRID__") ?? 9)
    );
    const domainDisagreementMap: DomainDisagreementPreference[] | null = (() => {
        if (
            !enableDomainDisagreementMap ||
            !localClassifierDomainUnsafe ||
            localUvDomainUnsafeStageBOnly ||
            !useLocalUvTopologicalFallback ||
            !localUvDomainUnsafeForceStageB ||
            localClassifierWires.length === 0
        ) {
            return null;
        }
        const cellsPerAxis = domainDisagreementGridSize - 1;
        const map = new Array<DomainDisagreementPreference>(cellsPerAxis * cellsPerAxis).fill(0);
        const uSpan = Math.max(1e-9, uMax - uMin);
        const vSpan = Math.max(1e-9, vMax - vMin);
        for (let j = 1; j < domainDisagreementGridSize; j++) {
            for (let i = 1; i < domainDisagreementGridSize; i++) {
                const u = uMin + (uSpan * i) / domainDisagreementGridSize;
                const v = vMin + (vSpan * j) / domainDisagreementGridSize;
                const stageAOnly = evaluateLocalUvWithPeriodicTraversal(u, v, {
                    disableStageB: true,
                    allowStageABoundaryBandUncertain: true,
                });
                const forcedStageB = evaluateLocalUvWithPeriodicTraversal(u, v, { forceStageB: true });

                let pref: DomainDisagreementPreference = 0;
                if (
                    stageAOnly.decision !== "uncertain" &&
                    forcedStageB.decision !== "uncertain" &&
                    stageAOnly.decision !== forcedStageB.decision
                ) {
                    // Conservative routing for current failure mode: prefer the
                    // branch that votes outside when A/B disagree.
                    pref = stageAOnly.decision === "outside" ? 1 : 2;
                } else if (stageAOnly.decision !== "uncertain" && forcedStageB.decision === "uncertain") {
                    pref = 1;
                } else if (stageAOnly.decision === "uncertain" && forcedStageB.decision !== "uncertain") {
                    pref = 2;
                }

                const idx = (j - 1) * cellsPerAxis + (i - 1);
                map[idx] = pref;
            }
        }
        return map;
    })();
    const getDomainDisagreementPreference = (uRaw: number, vRaw: number): DomainDisagreementPreference => {
        if (!domainDisagreementMap) return 0;
        const cellsPerAxis = domainDisagreementGridSize - 1;
        const uSpan = Math.max(1e-9, uMax - uMin);
        const vSpan = Math.max(1e-9, vMax - vMin);
        const fu = Math.max(0, Math.min(1, (uRaw - uMin) / uSpan));
        const fv = Math.max(0, Math.min(1, (vRaw - vMin) / vSpan));
        const iu = Math.max(0, Math.min(cellsPerAxis - 1, Math.floor(fu * cellsPerAxis)));
        const iv = Math.max(0, Math.min(cellsPerAxis - 1, Math.floor(fv * cellsPerAxis)));
        return domainDisagreementMap[iv * cellsPerAxis + iu] ?? 0;
    };
    const shouldPreferStageAUnderDisagreement = (
        disagreementPref: DomainDisagreementPreference,
        preferStageBForDisagreement: boolean,
        stageBDecision: LocalUvDecision,
        stageADecision: LocalUvDecision
    ): boolean => {
        if (disagreementPref === 1) return true;
        if (disagreementPref === 2) return false;
        if (
            preferStageBForDisagreement &&
            stageBDecision !== "uncertain" &&
            stageBDecision !== stageADecision
        ) {
            return false;
        }
        return true;
    };

    const classifyWithLocalUvClassifierRaw = (uRaw: number, v: number): LocalUvClassification => {
        if (localClassifierDomainUnsafe) {
            // Candidate mode is quality-gated; keep domain-unsafe faces on OCC until
            // local parity for these faces is proven safe.
            if (useLocalUvClassifierCandidate && !localUvClassifierCandidateStrict) {
                return {
                    decision: "uncertain",
                    nearBoundaryBand: true,
                    seamProximate: isSeamProximate(uRaw),
                    source: "domain_uncertain",
                };
            }
            if (localClassifierWires.length > 0) {
                if (useLocalUvTopologicalFallback && localUvDomainUnsafeForceStageB) {
                    const forcedStageB = evaluateLocalUvWithPeriodicTraversal(uRaw, v, { forceStageB: true });
                    if (localUvDomainUnsafeStageBOnly) {
                        return {
                            decision: forcedStageB.decision,
                            nearBoundaryBand: forcedStageB.nearBoundaryBand,
                            seamProximate: isSeamProximate(uRaw),
                            source: "domain_stageB",
                            stageBMetrics: forcedStageB.stageBMetrics,
                        };
                    }
                    let stageAOnly: LocalUvCandidateEval | null = null;
                    if (localUvDomainUnsafeArbitrateStageA) {
                        stageAOnly = evaluateLocalUvWithPeriodicTraversal(uRaw, v, {
                            disableStageB: true,
                            allowStageABoundaryBandUncertain: true,
                        });
                        const stageAConfidenceDistance = Math.max(
                            classifierPointEpsilon * 4,
                            localUvBoundaryTolerance * localUvDomainUnsafeStageAConfidenceBandMultiplier,
                            localUvDomainUnsafeStageAConfidenceMinDistance
                        );
                        const stageAConfident =
                            localUvDomainUnsafeStrictStageAConfidence
                                ? (
                                    stageAOnly.decision !== "uncertain" &&
                                    !stageAOnly.badWire &&
                                    !stageAOnly.nearBoundaryBand &&
                                    stageAOnly.minBoundaryDistance > stageAConfidenceDistance
                                )
                                : (
                                    stageAOnly.decision !== "uncertain" &&
                                    !stageAOnly.badWire &&
                                    stageAOnly.minBoundaryDistance > classifierPointEpsilon * 0.5
                                );
                        const stageBMetrics = forcedStageB.stageBMetrics;
                        const forcedAmbiguous =
                            forcedStageB.decision === "uncertain" ||
                            forcedStageB.nearBoundaryBand ||
                            !!(
                                stageBMetrics &&
                                (
                                    stageBMetrics.transitionTies > 0 ||
                                    stageBMetrics.probeFallbacks > 0
                                )
                            );
                        const shouldArbitrate = localUvDomainUnsafeArbitrateTiesOnly ? forcedAmbiguous : true;
                        const preferStageBForDisagreement = domainFaceRoutingPolicy === "prefer_stageB_disagreement";
                        const disagreementPref = getDomainDisagreementPreference(uRaw, v);
                        const minInsideOverrideDistance = Math.max(
                            localUvBoundaryTolerance * localUvDomainUnsafeInsideOverrideBandMultiplier,
                            localUvDomainUnsafeInsideOverrideMinDistance
                        );
                        const canUseStageAInsideVsStageBOutside =
                            !localUvDomainUnsafeGuardInsideVsOutside ||
                            stageAOnly.decision !== "inside" ||
                            forcedStageB.decision !== "outside" ||
                            (
                                !stageAOnly.nearBoundaryBand &&
                                stageAOnly.minBoundaryDistance >= minInsideOverrideDistance
                            );
                        if (shouldArbitrate && stageAConfident) {
                            const blockConfidentInsideOverride =
                                domainDisableStageAProvisionalInside &&
                                stageAOnly.decision === "inside" &&
                                forcedStageB.decision === "outside";
                            if (forcedStageB.decision === "uncertain" || forcedStageB.decision !== stageAOnly.decision) {
                                if (blockConfidentInsideOverride || !canUseStageAInsideVsStageBOutside) {
                                    // Keep Stage-B on boundary-adjacent inside-vs-outside disagreements.
                                    // This targets repeated cone false-inside cohorts.
                                    // Continue to provisional handling below.
                                } else
                                if (shouldPreferStageAUnderDisagreement(
                                    disagreementPref,
                                    preferStageBForDisagreement,
                                    forcedStageB.decision,
                                    stageAOnly.decision
                                )) {
                                    return {
                                        decision: stageAOnly.decision,
                                        nearBoundaryBand: stageAOnly.nearBoundaryBand,
                                        seamProximate: isSeamProximate(uRaw),
                                        source: "domain_stageA",
                                    };
                                }
                            }
                        }
                        const stageAProvisional =
                            stageAOnly.decision === "uncertain" && !stageAOnly.badWire
                                ? stageAOnly.provisionalDecision
                                : undefined;
                        if (
                            shouldArbitrate &&
                            (forcedAmbiguous || localUvDomainUnsafeUseStageAProvisional) &&
                            stageAProvisional &&
                            forcedStageB.decision !== stageAProvisional
                        ) {
                            const blockProvisionalInsideOverride =
                                domainDisableStageAProvisionalInside &&
                                stageAProvisional === "inside" &&
                                forcedStageB.decision === "outside";
                            const canUseProvisionalInsideVsStageBOutside =
                                !localUvDomainUnsafeGuardInsideVsOutside ||
                                stageAProvisional !== "inside" ||
                                forcedStageB.decision !== "outside" ||
                                (
                                    !stageAOnly.nearBoundaryBand &&
                                    stageAOnly.minBoundaryDistance >= minInsideOverrideDistance
                                );
                            if (blockProvisionalInsideOverride || !canUseProvisionalInsideVsStageBOutside) {
                                // Do not force inside from provisional hint near boundary.
                            } else
                            if (shouldPreferStageAUnderDisagreement(
                                disagreementPref,
                                preferStageBForDisagreement,
                                forcedStageB.decision,
                                stageAProvisional
                            )) {
                                return {
                                    decision: stageAProvisional,
                                    nearBoundaryBand: true,
                                    seamProximate: isSeamProximate(uRaw),
                                    source: "domain_stageA",
                                };
                            }
                        }
                    }
                    return {
                        decision: forcedStageB.decision,
                        nearBoundaryBand: forcedStageB.nearBoundaryBand,
                        seamProximate: isSeamProximate(uRaw),
                        source: "domain_stageB",
                        domainStageAHint: stageAOnly?.decision,
                        domainStageAProvisionalHint: stageAOnly?.provisionalDecision,
                        stageBMetrics: forcedStageB.stageBMetrics,
                    };
                }
                const candidate = evaluateLocalUvWithPeriodicTraversal(uRaw, v);
                if (candidate.source === "stageA") {
                    if (candidate.decision !== "uncertain" && !candidate.badWire) {
                        return {
                            decision: candidate.decision,
                            nearBoundaryBand: candidate.nearBoundaryBand,
                            seamProximate: isSeamProximate(uRaw),
                            source: "domain_stageA",
                        };
                    }
                } else {
                    return {
                        decision: candidate.decision,
                        nearBoundaryBand: candidate.nearBoundaryBand,
                        seamProximate: isSeamProximate(uRaw),
                        source: "domain_stageB",
                        stageBMetrics: candidate.stageBMetrics,
                    };
                }
            }
            return {
                decision: "uncertain",
                nearBoundaryBand: true,
                seamProximate: isSeamProximate(uRaw),
                source: "domain_uncertain",
            };
        }
        const candidate = evaluateLocalUvWithPeriodicTraversal(uRaw, v);
        const seamProximate = isSeamProximate(uRaw);
        return {
            decision: candidate.decision,
            nearBoundaryBand: candidate.nearBoundaryBand,
            seamProximate,
            source: candidate.source,
        };
    };

    const applyLocalUvPolarity = (decision: LocalUvDecision): LocalUvDecision => {
        if (!localUvPolarityInverted) return decision;
        if (decision === "inside") return "outside";
        if (decision === "outside") return "inside";
        return decision;
    };

    const classifyWithLocalUvClassifier = (uRaw: number, v: number): LocalUvClassification => {
        const raw = classifyWithLocalUvClassifierRaw(uRaw, v);
        if (raw.decision === "uncertain") return raw;
        return {
            ...raw,
            decision: applyLocalUvPolarity(raw.decision),
        };
    };

    const recordLocalDecisionSource = (
        decision: LocalUvClassification,
        mismatched: boolean
    ): void => {
        const source = decision.source;
        if (source === "stageA") {
            localUvClassifierStats.decisionFromStageA++;
            if (mismatched) localUvClassifierStats.mismatchFromStageA++;
            return;
        }
        if (source === "stageB") {
            localUvClassifierStats.decisionFromStageB++;
            if (mismatched) localUvClassifierStats.mismatchFromStageB++;
            return;
        }
        if (source === "stageB_forced") {
            localUvClassifierStats.decisionFromStageBForced++;
            if (mismatched) localUvClassifierStats.mismatchFromStageBForced++;
            return;
        }
        if (source === "domain_stageA") {
            localUvClassifierStats.decisionFromDomainUnsafeStageA++;
            if (mismatched) {
                localUvClassifierStats.mismatchFromDomainUnsafe++;
                localUvClassifierStats.mismatchFromDomainUnsafeStageA++;
            }
            return;
        }
        if (source === "domain_stageB") {
            localUvClassifierStats.decisionFromDomainUnsafeStageB++;
            if (mismatched) {
                localUvClassifierStats.mismatchFromDomainUnsafe++;
                localUvClassifierStats.mismatchFromDomainUnsafeStageB++;
                if (decision.decision === "outside") {
                    localUvClassifierStats.mismatchDomainStageBOutside++;
                }
                if (decision.nearBoundaryBand) {
                    localUvClassifierStats.mismatchDomainStageBNearBoundary++;
                }
                if (decision.domainStageAHint === "inside") {
                    localUvClassifierStats.mismatchDomainStageBWithStageAInsideHint++;
                } else if (decision.domainStageAHint === "outside") {
                    localUvClassifierStats.mismatchDomainStageBWithStageAOutsideHint++;
                } else if (decision.domainStageAHint === "uncertain") {
                    localUvClassifierStats.mismatchDomainStageBWithStageAUncertainHint++;
                }
                if ((decision.stageBMetrics?.transitionTies ?? 0) > 0) {
                    localUvClassifierStats.mismatchDomainStageBWithTransitionTie++;
                }
                if ((decision.stageBMetrics?.probeFallbacks ?? 0) > 0) {
                    localUvClassifierStats.mismatchDomainStageBWithProbeFallback++;
                }
            }
        }
    };

    const maybeCalibrateConeLocalUvPolarity = (): void => {
        const strictCandidateNoOcc = localUvClassifierCandidateStrict && useLocalUvClassifierCandidate && !useLocalUvClassifierShadow;
        if (
            strictCandidateNoOcc ||
            surfaceTypeForClassifier !== "Cone" ||
            !useOccUvInside ||
            !(useLocalUvClassifierShadow || useLocalUvClassifierCandidate) ||
            typeof buildOptions?.uvInsideTest !== "function"
        ) {
            return;
        }

        const sampleGrid = Math.max(
            4,
            Math.floor(readTrimNumber("__LOCAL_UV_CLASSIFIER_POLARITY_GRID__") ?? 6)
        );
        const minSamples = Math.max(
            8,
            Math.floor(readTrimNumber("__LOCAL_UV_CLASSIFIER_POLARITY_MIN_SAMPLES__") ?? 12)
        );
        const minImprovement = Math.max(
            1,
            Math.floor(readTrimNumber("__LOCAL_UV_CLASSIFIER_POLARITY_MIN_IMPROVEMENT__") ?? 3)
        );

        const uSpanCalib = Math.max(1e-9, uMax - uMin);
        const vSpanCalib = Math.max(1e-9, vMax - vMin);
        let directMismatch = 0;
        let invertedMismatch = 0;
        let considered = 0;

        for (let j = 1; j < sampleGrid; j++) {
            for (let i = 1; i < sampleGrid; i++) {
                const u = uMin + (uSpanCalib * i) / sampleGrid;
                const v = vMin + (vSpanCalib * j) / sampleGrid;
                const localRaw = classifyWithLocalUvClassifierRaw(u, v);
                if (localRaw.decision === "uncertain") continue;
                const occInside = !!buildOptions.uvInsideTest(u, v);
                const localInside = localRaw.decision === "inside";
                considered++;
                if (localInside !== occInside) directMismatch++;
                if ((!localInside) !== occInside) invertedMismatch++;
            }
        }

        localUvPolarityCalibrationSamples = considered;
        localUvPolarityCalibrationDirectMismatch = directMismatch;
        localUvPolarityCalibrationInvertedMismatch = invertedMismatch;
        if (
            considered >= minSamples &&
            invertedMismatch + minImprovement < directMismatch
        ) {
            localUvPolarityInverted = true;
        }
    };
    maybeCalibrateConeLocalUvPolarity();

    const enableGpuTrimGridClassification = readTrimBoolean("__ENABLE_GPU_TRIM_GRID_CLASSIFICATION__", true);
    const totalGridPoints = (gridDensityU + 1) * (gridDensityV + 1);
    const gpuGridMinPoints = Math.max(
        512,
        Math.floor(readTrimNumber("__GPU_TRIM_GRID_MIN_POINTS__") ?? 2048)
    );
    const canUseGpuTrimGridClassification =
        enableGpuTrimGridClassification &&
        !useOccUvInside &&
        totalGridPoints >= gpuGridMinPoints;
    // GPU-first path: build triangle connectivity directly from the trim mask.
    // Keep this conservative so we can fall back to the existing CPU path for
    // classifier- or bbox-sensitive faces.
    const enableGpuTrimTriangleBuild = readTrimBoolean("__ENABLE_GPU_TRIM_TRIANGLE_BUILD__", true);
    const gpuTrimTriangleMinCells = Math.max(
        256,
        Math.floor(readTrimNumber("__GPU_TRIM_TRIANGLE_BUILD_MIN_CELLS__") ?? 1024)
    );
    const totalGridCells = gridDensityU * gridDensityV;
    const canUseGpuTrimTriangleBuild =
        enableGpuTrimTriangleBuild &&
        !bbox3d &&
        !keepTriangle &&
        !useOccUvInside &&
        totalGridCells >= gpuTrimTriangleMinCells;

    if (canUseGpuTrimTriangleBuild) {
        const gpuClassifyBuildStart = performance.now();
        const gpuTriangles = await classifyAndBuildTrimGridTrianglesGPU({
            boundary: continuousBoundary,
            holes: continuousHoles,
            gridDensityU,
            gridDensityV,
            uMin,
            vMin,
            du,
            dv,
            boundaryTolerance,
            useNearBoundary: !skipNearBoundaryChecks,
            allowPartialCellTriangles,
        });
        recordTrimProfileSample(buildOptions, "gpu_classify_build", performance.now() - gpuClassifyBuildStart);
        if (gpuTriangles && gpuTriangles.triangleCount > 0) {
            const gpuDenseEvalStart = performance.now();
            const denseGpuEval = await evaluateSurfaceDenseGridGPU(
                surface as any,
                gridDensityU,
                gridDensityV,
                uMin,
                vMin,
                du,
                dv
            );
            recordTrimProfileSample(buildOptions, "gpu_dense_eval", performance.now() - gpuDenseEvalStart);
            if (denseGpuEval) {
                const uvs = new Float32Array(totalGridPoints * 2);
                for (let j = 0; j <= gridDensityV; j++) {
                    for (let i = 0; i <= gridDensityU; i++) {
                        const idx = j * (gridDensityU + 1) + i;
                        uvs[idx * 2 + 0] = uMin + i * du;
                        uvs[idx * 2 + 1] = vMin + j * dv;
                    }
                }
                if (debugTrim) {
                    trimDebugLog(
                        `[tessellateTrimmedSurface] GPU dense-grid eval accepted: ` +
                        `verts=${totalGridPoints}, tris=${gpuTriangles.triangleCount}`
                    );
                }
                return {
                    positions: denseGpuEval.positions,
                    normals: denseGpuEval.normals,
                    indices: gpuTriangles.indices,
                    uvs,
                };
            }

            const uvVerticesDense: Vec2[] = new Array(totalGridPoints);
            for (let j = 0; j <= gridDensityV; j++) {
                for (let i = 0; i <= gridDensityU; i++) {
                    const idx = j * (gridDensityU + 1) + i;
                    uvVerticesDense[idx] = [uMin + i * du, vMin + j * dv];
                }
            }
            if (debugTrim) {
                trimDebugLog(
                    `[tessellateTrimmedSurface] GPU classify+triangle build accepted: ` +
                    `cells=${totalGridCells}, triangles=${gpuTriangles.triangleCount}`
                );
            }
            const finalEvalStart = performance.now();
            const mesh = await evaluateUVMesh(surface, uvVerticesDense, gpuTriangles.indices, buildOptions?.recordProfileSample);
            recordTrimProfileSample(buildOptions, "final_evaluate_mesh", performance.now() - finalEvalStart);
            return mesh;
        }
    }

    let gpuTrimMask: Uint32Array | null = null;
    if (canUseGpuTrimGridClassification) {
        const gpuMaskClassifyStart = performance.now();
        gpuTrimMask = await classifyTrimGridGPU({
            boundary: continuousBoundary,
            holes: continuousHoles,
            gridDensityU,
            gridDensityV,
            uMin,
            vMin,
            du,
            dv,
            boundaryTolerance,
            useNearBoundary: !skipNearBoundaryChecks,
        });
        recordTrimProfileSample(buildOptions, "gpu_mask_classify", performance.now() - gpuMaskClassifyStart);
        if (gpuTrimMask && debugTrim) {
            trimDebugLog(
                `[tessellateTrimmedSurface] GPU trim classification enabled for ${totalGridPoints} grid points`
            );
        }
    }

    if (canUseGpuTrimTriangleBuild && gpuTrimMask) {
        const gpuMaskTrianglesStart = performance.now();
        const gpuTriangles = await buildTrimGridTrianglesGPU({
            mask: gpuTrimMask,
            gridDensityU,
            gridDensityV,
            allowPartialCellTriangles,
        });
        recordTrimProfileSample(buildOptions, "gpu_mask_triangles", performance.now() - gpuMaskTrianglesStart);
        if (gpuTriangles && gpuTriangles.triangleCount > 0) {
            const uvVerticesDense: Vec2[] = new Array(totalGridPoints);
            let insideByMask = 0;
            for (let j = 0; j <= gridDensityV; j++) {
                for (let i = 0; i <= gridDensityU; i++) {
                    const idx = j * (gridDensityU + 1) + i;
                    uvVerticesDense[idx] = [uMin + i * du, vMin + j * dv];
                    if (gpuTrimMask[idx] !== 0) {
                        insideByMask++;
                    }
                }
            }
            if (debugTrim) {
                trimDebugLog(
                    `[tessellateTrimmedSurface] GPU trim triangle build accepted: ` +
                    `cells=${totalGridCells}, includedVerts=${insideByMask}/${totalGridPoints}, ` +
                    `triangles=${gpuTriangles.triangleCount}`
                );
            }
            const finalEvalStart = performance.now();
            const mesh = await evaluateUVMesh(surface, uvVerticesDense, gpuTriangles.indices, buildOptions?.recordProfileSample);
            recordTrimProfileSample(buildOptions, "final_evaluate_mesh", performance.now() - finalEvalStart);
            return mesh;
        }
    }

    // 3D bbox tolerance - keep tight to avoid protrusions
    // For horizontal cylinders, even 0.5 tolerance can create visible artifacts
    const bbox3dTol = 0.05;

    // Helper to check if a 3D point is within the bbox
    const isIn3DBbox = (pos: Vec3): boolean => {
        if (!bbox3d) return true;
        return pos[0] >= bbox3d.xMin - bbox3dTol && pos[0] <= bbox3d.xMax + bbox3dTol &&
               pos[1] >= bbox3d.yMin - bbox3dTol && pos[1] <= bbox3d.yMax + bbox3dTol &&
               pos[2] >= bbox3d.zMin - bbox3dTol && pos[2] <= bbox3d.zMax + bbox3dTol;
    };

    let bbox3dFilteredCount = 0;
    const cpuGridClassifyStart = performance.now();

    for (let j = 0; j <= gridDensityV; j++) {
        vertexGrid[j] = [];
        for (let i = 0; i <= gridDensityU; i++) {
            const u = uMin + i * du;
            const v = vMin + j * dv;
            const maskIdx = j * (gridDensityU + 1) + i;
            let includePoint = false;
            if (gpuTrimMask) {
                includePoint = gpuTrimMask[maskIdx] !== 0;
            } else {
                // Check if this point is inside the continuous boundary and outside holes.
                // For selected OCC paths, use the classifier as the source of truth.
                if (useLocalUvClassifierShadow) {
                    const localDecision = classifyWithLocalUvClassifier(u, v);
                    localUvClassifierStats.occDecisions++;
                    if (localDecision.decision === "inside") {
                        localUvClassifierStats.inside++;
                    } else if (localDecision.decision === "outside") {
                        localUvClassifierStats.outside++;
                    } else if (localUvClassifierFallbackToOcc) {
                        localUvClassifierStats.uncertain++;
                    } else {
                        localUvClassifierStats.uncertain++;
                    }
                    if (localDecision.nearBoundaryBand) {
                        localUvClassifierStats.boundaryBandSamples++;
                    }
                    if (localDecision.seamProximate) {
                        localUvClassifierStats.seamProximateSamples++;
                    }
                    const occInside = !!buildOptions!.uvInsideTest!(u, v);
                    includePoint = occInside;
                    if (localDecision.decision !== "uncertain") {
                        const localInside = localDecision.decision === "inside";
                        recordLocalDecisionSource(localDecision, localInside !== occInside);
                        if (localInside !== occInside) {
                            localUvClassifierStats.mismatchCount++;
                            if (localInside) {
                                localUvClassifierStats.falseInsideCount++;
                            } else {
                                localUvClassifierStats.falseOutsideCount++;
                            }
                            if (localDecision.nearBoundaryBand) {
                                localUvClassifierStats.mismatchBoundaryBand++;
                            } else {
                                localUvClassifierStats.mismatchInterior++;
                            }
                            if (localDecision.seamProximate) {
                                localUvClassifierStats.mismatchSeamProximate++;
                            } else {
                                localUvClassifierStats.mismatchNonSeam++;
                            }
                        }
                    }
                } else if (useLocalUvClassifierCandidate) {
                    const localDecision = classifyWithLocalUvClassifier(u, v);
                    if (localDecision.decision === "inside") {
                        recordLocalDecisionSource(localDecision, false);
                        localUvClassifierStats.inside++;
                        includePoint = true;
                    } else if (localDecision.decision === "outside") {
                        recordLocalDecisionSource(localDecision, false);
                        localUvClassifierStats.outside++;
                        includePoint = false;
                    } else if (localUvClassifierFallbackToOcc) {
                        localUvClassifierStats.uncertain++;
                        localUvClassifierStats.fallbackCalls++;
                        localUvClassifierStats.occDecisions++;
                        includePoint = !!buildOptions!.uvInsideTest!(u, v);
                    } else {
                        localUvClassifierStats.uncertain++;
                        // Fail-open without OCC fallback to preserve old classifier bias.
                        includePoint = true;
                    }
                    if (localDecision.nearBoundaryBand) {
                        localUvClassifierStats.boundaryBandSamples++;
                    }
                    if (localDecision.seamProximate) {
                        localUvClassifierStats.seamProximateSamples++;
                    }
                } else {
                    const insideBoundary = useOccUvInside
                        ? !!buildOptions!.uvInsideTest!(u, v)
                        : isPointInPolygon([u, v], continuousBoundary);
                    const nearBoundary = useOccUvInside
                        ? false
                        : (skipNearBoundaryChecks
                            ? false
                            : isNearBoundary([u, v], continuousBoundary, boundaryTolerance));
                    const insideHole = useOccUvInside
                        ? false
                        : continuousHoles.some(hole => isPointInPolygon([u, v], hole));
                    includePoint = (insideBoundary || nearBoundary) && !insideHole;
                }
            }

            if (includePoint) {
                // If 3D bbox is provided, also check if the 3D position is within bounds
                // This is crucial for horizontal cylinders where UV polygon spans full U
                // but we only want the portion of the surface within the 3D boundary
                if (bbox3d) {
                    const pos3d = evaluateSurface(surface, u, v);
                    if (!isIn3DBbox(pos3d)) {
                        vertexGrid[j][i] = null;
                        bbox3dFilteredCount++;
                        outsideCount++;
                        continue;
                    }
                }

                vertexGrid[j][i] = uvVertices.length;
                // Store the unwrapped UV for surface evaluation
                // (cos/sin are 2π periodic, so values in [π, 2π] work correctly)
                uvVertices.push([u, v]);
                insideCount++;
            } else {
                vertexGrid[j][i] = null;
                outsideCount++;
            }
        }
    }
    recordTrimProfileSample(buildOptions, "cpu_grid_classify", performance.now() - cpuGridClassifyStart);

    if (bbox3d && debugTrim) {
        trimDebugLog(`[tessellateTrimmedSurface] 3D bbox provided: X=[${bbox3d.xMin.toFixed(2)}, ${bbox3d.xMax.toFixed(2)}], Y=[${bbox3d.yMin.toFixed(2)}, ${bbox3d.yMax.toFixed(2)}], Z=[${bbox3d.zMin.toFixed(2)}, ${bbox3d.zMax.toFixed(2)}]`);
        trimDebugLog(`[tessellateTrimmedSurface] 3D bbox filtered ${bbox3dFilteredCount} points outside bounds`);

        // Sample a few evaluated positions to debug
        if (uvVertices.length > 0) {
            trimDebugLog(`[tessellateTrimmedSurface] Sample 3D positions from evaluated surface:`);
            for (let k = 0; k < Math.min(5, uvVertices.length); k++) {
                const [u, v] = uvVertices[k];
                const pos = evaluateSurface(surface, u, v);
                trimDebugLog(`  UV(${u.toFixed(2)}, ${v.toFixed(2)}) -> 3D(${pos[0].toFixed(2)}, ${pos[1].toFixed(2)}, ${pos[2].toFixed(2)})`);
            }
        }
    }

    if (debugTrim) {
        trimDebugLog(`[tessellateTrimmedSurface] Grid bounds: U=[${uMin.toFixed(3)}, ${uMax.toFixed(3)}], V=[${vMin.toFixed(3)}, ${vMax.toFixed(3)}]`);
        trimDebugLog(`[tessellateTrimmedSurface] Grid points: ${insideCount} inside, ${outsideCount} outside${useOccUvInside ? ` (mode=${buildLabel})` : ''}`);
        if (useLocalUvClassifierShadow || useLocalUvClassifierCandidate) {
            localUvClassifierStats.effectiveMismatchCount =
                localUvClassifierStats.mismatchCount + localUvClassifierStats.uncertain;
            const mode = useLocalUvClassifierShadow ? 'shadow' : 'candidate';
            trimDebugLog(
                `[tessellateTrimmedSurface] Local UV classifier stats (${mode}): ` +
                `inside=${localUvClassifierStats.inside}, outside=${localUvClassifierStats.outside}, ` +
                `uncertain=${localUvClassifierStats.uncertain}, fallbackCalls=${localUvClassifierStats.fallbackCalls}, ` +
                `occDecisions=${localUvClassifierStats.occDecisions}, ` +
                `mismatch=${localUvClassifierStats.mismatchCount}, ` +
                `effectiveMismatch=${localUvClassifierStats.effectiveMismatchCount}, ` +
                `falseInside=${localUvClassifierStats.falseInsideCount}, falseOutside=${localUvClassifierStats.falseOutsideCount}, ` +
                `polarityInverted=${localUvPolarityInverted}, ` +
                `polarityCalib=${localUvPolarityCalibrationSamples}/${localUvPolarityCalibrationDirectMismatch}->${localUvPolarityCalibrationInvertedMismatch}, ` +
                `boundaryMismatch=${localUvClassifierStats.mismatchBoundaryBand}, ` +
                `seamMismatch=${localUvClassifierStats.mismatchSeamProximate}, ` +
                `stageA=${localUvClassifierStats.stageAEvaluations}, stageB=${localUvClassifierStats.stageBEvaluations}, ` +
                `stageBForced=${localUvClassifierStats.stageBForcedEvaluations}, badWireTriggers=${localUvClassifierStats.stageBTriggeredByBadWire}`
            );
        }

        // Diagnostic: test a specific point that should be inside (center of grid)
        const testU = (uMin + uMax) / 2;
        const testV = (vMin + vMax) / 2;
        const testInside = useOccUvInside
            ? !!buildOptions!.uvInsideTest!(testU, testV)
            : isPointInPolygon([testU, testV], continuousBoundary);
        const testInHole = useOccUvInside
            ? false
            : continuousHoles.some(hole => isPointInPolygon([testU, testV], hole));
        trimDebugLog(`[tessellateTrimmedSurface] Test point (${testU.toFixed(3)}, ${testV.toFixed(3)}): insideBoundary=${testInside}, insideHole=${testInHole}`);

        // Debug: trace through point-in-polygon for the center point
        if (!testInside && continuousHoles.length === 0) {
            trimDebugLog(`[DEBUG] Tracing point-in-polygon for center point:`);
            let crossings = 0;
            for (let i = 0, j = continuousBoundary.length - 1; i < continuousBoundary.length; j = i++) {
                const [xi, yi] = continuousBoundary[i];
                const [xj, yj] = continuousBoundary[j];
                const yCrossesRay = ((yi > testV) !== (yj > testV));
                if (yCrossesRay) {
                    const xIntersect = (xj - xi) * (testV - yi) / (yj - yi) + xi;
                    if (testU < xIntersect) {
                        crossings++;
                        if (crossings <= 10) {
                            trimDebugLog(`  Crossing ${crossings}: edge [${j}]->[${i}] from (${xj.toFixed(2)},${yj.toFixed(2)}) to (${xi.toFixed(2)},${yi.toFixed(2)}), intersect at x=${xIntersect.toFixed(2)}`);
                        }
                    }
                }
            }
            trimDebugLog(`[DEBUG] Total crossings: ${crossings} -> inside=${crossings % 2 === 1}`);

            // Show boundary polygon structure
            trimDebugLog(`[DEBUG] Boundary polygon (${continuousBoundary.length} points):`);
            trimDebugLog(`  First 5: ${continuousBoundary.slice(0, 5).map((p, i) => `[${i}](${p[0].toFixed(2)},${p[1].toFixed(2)})`).join(' ')}`);
            const quarter = Math.floor(continuousBoundary.length / 4);
            trimDebugLog(`  At ${quarter}: (${continuousBoundary[quarter][0].toFixed(2)},${continuousBoundary[quarter][1].toFixed(2)})`);
            const half = Math.floor(continuousBoundary.length / 2);
            trimDebugLog(`  At ${half}: (${continuousBoundary[half][0].toFixed(2)},${continuousBoundary[half][1].toFixed(2)})`);
            const threeQuarter = Math.floor(continuousBoundary.length * 3 / 4);
            trimDebugLog(`  At ${threeQuarter}: (${continuousBoundary[threeQuarter][0].toFixed(2)},${continuousBoundary[threeQuarter][1].toFixed(2)})`);
            trimDebugLog(`  Last 5: ${continuousBoundary.slice(-5).map((p, i) => `[${continuousBoundary.length-5+i}](${p[0].toFixed(2)},${p[1].toFixed(2)})`).join(' ')}`);
        }

        // Additional diagnostic: test a point that SHOULD be inside the hole
        if (continuousHoles.length > 0) {
            const hole = continuousHoles[0];
            const holeUs = hole.map(p => p[0]);
            const holeVs = hole.map(p => p[1]);
            const holeCenterU = (Math.min(...holeUs) + Math.max(...holeUs)) / 2;
            const holeCenterV = (Math.min(...holeVs) + Math.max(...holeVs)) / 2;
            const holeCenterInHole = isPointInPolygon([holeCenterU, holeCenterV], hole);
            trimDebugLog(`[tessellateTrimmedSurface] Hole center (${holeCenterU.toFixed(3)}, ${holeCenterV.toFixed(3)}): insideHole=${holeCenterInHole}`);

            // Debug: manually trace through point-in-polygon for the hole center
            let debugCrossings = 0;
            const testX = holeCenterU;
            const testY = holeCenterV;
            for (let i = 0, j = hole.length - 1; i < hole.length; j = i++) {
                const [xi, yi] = hole[i];
                const [xj, yj] = hole[j];
                const yCrossesRay = ((yi > testY) !== (yj > testY));
                if (yCrossesRay) {
                    const xIntersect = (xj - xi) * (testY - yi) / (yj - yi) + xi;
                    if (testX < xIntersect) {
                        debugCrossings++;
                        if (debugCrossings <= 5) {
                            trimDebugLog(`[DEBUG] Crossing ${debugCrossings}: edge [${j}]->[${i}] from (${xj.toFixed(2)},${yj.toFixed(2)}) to (${xi.toFixed(2)},${yi.toFixed(2)})`);
                        }
                    }
                }
            }
            trimDebugLog(`[DEBUG] Total crossings: ${debugCrossings} -> inside=${debugCrossings % 2 === 1}`);

            // Check if all hole points are in expected range
            const outOfRangeU = hole.filter(p => p[0] < -4 || p[0] > 7);
            const outOfRangeV = hole.filter(p => p[1] < -1 || p[1] > 3);
            if (outOfRangeU.length > 0 || outOfRangeV.length > 0) {
                trimDebugLog(`[DEBUG] WARNING: ${outOfRangeU.length} points have U outside [-4,7], ${outOfRangeV.length} have V outside [-1,3]`);
            }

            // Show the polygon vertices at key positions
            trimDebugLog(`[DEBUG] Hole polygon shape (${hole.length} points):`);
            trimDebugLog(`  [0]: (${hole[0][0].toFixed(3)}, ${hole[0][1].toFixed(3)})`);
            trimDebugLog(`  [1]: (${hole[1][0].toFixed(3)}, ${hole[1][1].toFixed(3)})`);
            const mid = Math.floor(hole.length / 2);
            trimDebugLog(`  [${mid}]: (${hole[mid][0].toFixed(3)}, ${hole[mid][1].toFixed(3)})`);
            trimDebugLog(`  [${hole.length-2}]: (${hole[hole.length-2][0].toFixed(3)}, ${hole[hole.length-2][1].toFixed(3)})`);
            trimDebugLog(`  [${hole.length-1}]: (${hole[hole.length-1][0].toFixed(3)}, ${hole[hole.length-1][1].toFixed(3)})`);

            // Check if first point at V=0.6, last points approaching first
            const firstV = hole[0][1];
            const lastV = hole[hole.length-1][1];
            trimDebugLog(`[DEBUG] First point V=${firstV.toFixed(3)}, last point V=${lastV.toFixed(3)}`);
            trimDebugLog(`[DEBUG] Test point V=${testY.toFixed(3)} - should be between ${Math.min(firstV, lastV).toFixed(3)} and ${Math.max(firstV, lastV).toFixed(3)} for the vertical edges`);
        }
    }

    // Create triangles from the grid
    const triangles: [number, number, number][] = [];
    const triSamples = (a: Vec2, b: Vec2, c: Vec2): [number, number][] => {
        const uCentroid = (a[0] + b[0] + c[0]) / 3;
        const vCentroid = (a[1] + b[1] + c[1]) / 3;
        return [
            [a[0], a[1]],
            [b[0], b[1]],
            [c[0], c[1]],
            [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
            [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2],
            [(c[0] + a[0]) / 2, (c[1] + a[1]) / 2],
            [uCentroid, vCentroid],
        ];
    };
    const shouldKeepTriangle = (ia: number, ib: number, ic: number): boolean => {
        if (!keepTriangle) return true;
        const a = uvVertices[ia];
        const b = uvVertices[ib];
        const c = uvVertices[ic];
        if (!a || !b || !c) return false;
        try {
            return !!keepTriangle(triSamples(a, b, c));
        } catch {
            // Fail open to avoid catastrophic face loss on classifier runtime issues.
            return true;
        }
    };
    let triangleDropsByClassifier = 0;
    const pushIfKept = (ia: number, ib: number, ic: number) => {
        if (shouldKeepTriangle(ia, ib, ic)) {
            triangles.push([ia, ib, ic]);
        } else {
            triangleDropsByClassifier++;
        }
    };

    const cpuTriangleBuildStart = performance.now();
    for (let j = 0; j < gridDensityV; j++) {
        for (let i = 0; i < gridDensityU; i++) {
            const v00 = vertexGrid[j][i];
            const v10 = vertexGrid[j][i + 1];
            const v01 = vertexGrid[j + 1][i];
            const v11 = vertexGrid[j + 1][i + 1];

            // Create triangles only if all vertices are valid
            if (v00 !== null && v10 !== null && v01 !== null && v11 !== null) {
                // Two triangles per quad
                pushIfKept(v00, v01, v11);
                pushIfKept(v00, v11, v10);
            } else {
                // Handle partial quads - create triangles where possible
                if (!allowPartialCellTriangles) {
                    continue;
                }
                if (v00 !== null && v01 !== null && v11 !== null) {
                    pushIfKept(v00, v01, v11);
                }
                if (v00 !== null && v11 !== null && v10 !== null) {
                    pushIfKept(v00, v11, v10);
                }
                if (v00 !== null && v01 !== null && v10 !== null) {
                    pushIfKept(v00, v01, v10);
                }
                if (v01 !== null && v11 !== null && v10 !== null) {
                    pushIfKept(v01, v11, v10);
                }
            }
        }
    }
    recordTrimProfileSample(buildOptions, "cpu_triangle_build", performance.now() - cpuTriangleBuildStart);


    trimDebugLog(`[tessellateTrimmedSurface] Generated ${triangles.length} triangles from ${uvVertices.length} vertices`);
    if (triangleDropsByClassifier > 0) {
        trimDebugLog(`[tessellateTrimmedSurface] Triangle gate dropped ${triangleDropsByClassifier} candidate triangles (mode=${buildLabel})`);
    }
    if (useLocalUvClassifierShadow || useLocalUvClassifierCandidate) {
        try {
            localUvClassifierStats.effectiveMismatchCount =
                localUvClassifierStats.mismatchCount + localUvClassifierStats.uncertain;
            buildOptions?.recordLocalUvClassifierSummary?.({
                mode: useLocalUvClassifierShadow ? "shadow" : "candidate",
                faceIndex: buildOptions?.classifierFaceIndex,
                surfaceType: buildOptions?.classifierSurfaceType,
                buildLabel,
                gridPoints: totalGridPoints,
                occDecisions: localUvClassifierStats.occDecisions,
                localInside: localUvClassifierStats.inside,
                localOutside: localUvClassifierStats.outside,
                localUncertain: localUvClassifierStats.uncertain,
                localFallbackCalls: localUvClassifierStats.fallbackCalls,
                boundaryBandSamples: localUvClassifierStats.boundaryBandSamples,
                seamProximateSamples: localUvClassifierStats.seamProximateSamples,
                mismatchCount: localUvClassifierStats.mismatchCount,
                effectiveMismatchCount: localUvClassifierStats.effectiveMismatchCount,
                mismatchBoundaryBand: localUvClassifierStats.mismatchBoundaryBand,
                mismatchInterior: localUvClassifierStats.mismatchInterior,
                mismatchSeamProximate: localUvClassifierStats.mismatchSeamProximate,
                mismatchNonSeam: localUvClassifierStats.mismatchNonSeam,
                falseInsideCount: localUvClassifierStats.falseInsideCount,
                falseOutsideCount: localUvClassifierStats.falseOutsideCount,
                stageAEvaluations: localUvClassifierStats.stageAEvaluations,
                stageAInside: localUvClassifierStats.stageAInside,
                stageAOutside: localUvClassifierStats.stageAOutside,
                stageAUncertain: localUvClassifierStats.stageAUncertain,
                stageABadWire: localUvClassifierStats.stageABadWire,
                stageBEvaluations: localUvClassifierStats.stageBEvaluations,
                stageBForcedEvaluations: localUvClassifierStats.stageBForcedEvaluations,
                stageBTriggeredByUncertain: localUvClassifierStats.stageBTriggeredByUncertain,
                stageBTriggeredByBadWire: localUvClassifierStats.stageBTriggeredByBadWire,
                stageBInside: localUvClassifierStats.stageBInside,
                stageBOutside: localUvClassifierStats.stageBOutside,
                stageBUncertain: localUvClassifierStats.stageBUncertain,
                stageBResolvedByUncertain: localUvClassifierStats.stageBResolvedByUncertain,
                stageBResolvedByBadWire: localUvClassifierStats.stageBResolvedByBadWire,
                stageBProbeFallbacks: localUvClassifierStats.stageBProbeFallbacks,
                stageBBundleSkips: localUvClassifierStats.stageBBundleSkips,
                stageBSkippedNearVertexHits: localUvClassifierStats.stageBSkippedNearVertexHits,
                stageBTransitionTies: localUvClassifierStats.stageBTransitionTies,
                decisionFromStageA: localUvClassifierStats.decisionFromStageA,
                decisionFromStageB: localUvClassifierStats.decisionFromStageB,
                decisionFromStageBForced: localUvClassifierStats.decisionFromStageBForced,
                decisionFromDomainUnsafeStageA: localUvClassifierStats.decisionFromDomainUnsafeStageA,
                decisionFromDomainUnsafeStageB: localUvClassifierStats.decisionFromDomainUnsafeStageB,
                mismatchFromStageA: localUvClassifierStats.mismatchFromStageA,
                mismatchFromStageB: localUvClassifierStats.mismatchFromStageB,
                mismatchFromStageBForced: localUvClassifierStats.mismatchFromStageBForced,
                mismatchFromDomainUnsafe: localUvClassifierStats.mismatchFromDomainUnsafe,
                mismatchFromDomainUnsafeStageA: localUvClassifierStats.mismatchFromDomainUnsafeStageA,
                mismatchFromDomainUnsafeStageB: localUvClassifierStats.mismatchFromDomainUnsafeStageB,
                mismatchDomainStageBOutside: localUvClassifierStats.mismatchDomainStageBOutside,
                mismatchDomainStageBWithTransitionTie: localUvClassifierStats.mismatchDomainStageBWithTransitionTie,
                mismatchDomainStageBWithProbeFallback: localUvClassifierStats.mismatchDomainStageBWithProbeFallback,
                mismatchDomainStageBNearBoundary: localUvClassifierStats.mismatchDomainStageBNearBoundary,
                mismatchDomainStageBWithStageAInsideHint: localUvClassifierStats.mismatchDomainStageBWithStageAInsideHint,
                mismatchDomainStageBWithStageAOutsideHint: localUvClassifierStats.mismatchDomainStageBWithStageAOutsideHint,
                mismatchDomainStageBWithStageAUncertainHint: localUvClassifierStats.mismatchDomainStageBWithStageAUncertainHint,
            });
        } catch {
            // Classifier summary hooks are best-effort only.
        }
    }

    // Note: We do NOT add boundary stitching triangles here.
    // The boundary stitching was causing visible spike artifacts on curved surfaces.
    // The grid-based tessellation alone provides clean results, even if there are small gaps
    // at the edges. This matches the approach used in the benchmark-research branch.

    const finalEvalStart = performance.now();
    const mesh = await evaluateUVMesh(surface, uvVertices, triangles, buildOptions?.recordProfileSample);
    recordTrimProfileSample(buildOptions, "final_evaluate_mesh", performance.now() - finalEvalStart);
    return mesh;
}

/**
 * C6.4: Bridge holes into outer boundary in UV space.
 * Uses the same algorithm as for planar faces.
 */
function bridgeAllHolesUV(outer: Vec2[], holes: Vec2[][]): Vec2[] {
    if (holes.length === 0) {
        return outer;
    }

    // Ensure outer is CCW
    let outerArea = 0;
    for (let i = 0; i < outer.length; i++) {
        const j = (i + 1) % outer.length;
        outerArea += outer[i][0] * outer[j][1] - outer[j][0] * outer[i][1];
    }
    outerArea /= 2;
    let currentOuter = outerArea > 0 ? outer : [...outer].reverse();

    // Ensure holes are CW
    const normalizedHoles = holes.map(hole => {
        let holeArea = 0;
        for (let i = 0; i < hole.length; i++) {
            const j = (i + 1) % hole.length;
            holeArea += hole[i][0] * hole[j][1] - hole[j][0] * hole[i][1];
        }
        return holeArea < 0 ? hole : [...hole].reverse();
    });

    // Sort holes by rightmost X (descending)
    const holesWithRightmost = normalizedHoles.map((hole, idx) => {
        let maxU = -Infinity;
        let maxUIndex = 0;
        for (let i = 0; i < hole.length; i++) {
            if (hole[i][0] > maxU) {
                maxU = hole[i][0];
                maxUIndex = i;
            }
        }
        return { hole, idx, rightmostU: maxU, rightmostIndex: maxUIndex };
    });

    holesWithRightmost.sort((a, b) => b.rightmostU - a.rightmostU);

    // Merge each hole
    let merged = currentOuter;
    for (const { hole, rightmostIndex } of holesWithRightmost) {
        merged = mergeHoleIntoOuterUV(merged, hole, rightmostIndex);
    }

    return merged;
}

/**
 * C6.4: Merge a single hole into the outer boundary.
 */
function mergeHoleIntoOuterUV(outer: Vec2[], hole: Vec2[], holeRightmostIndex: number): Vec2[] {
    const holeVertex = hole[holeRightmostIndex];

    // Cast a ray from holeVertex in the +U direction to find the closest edge on outer
    let bestDist = Infinity;
    let bestOuterIndex = 0;

    for (let i = 0; i < outer.length; i++) {
        const j = (i + 1) % outer.length;
        const p1 = outer[i];
        const p2 = outer[j];

        // Check if the edge crosses the horizontal ray from holeVertex
        const minV = Math.min(p1[1], p2[1]);
        const maxV = Math.max(p1[1], p2[1]);

        if (holeVertex[1] >= minV && holeVertex[1] <= maxV && p1[1] !== p2[1]) {
            // Find intersection point
            const t = (holeVertex[1] - p1[1]) / (p2[1] - p1[1]);
            const intersectU = p1[0] + t * (p2[0] - p1[0]);

            // Must be to the right of holeVertex
            if (intersectU >= holeVertex[0]) {
                const dist = intersectU - holeVertex[0];
                if (dist < bestDist) {
                    bestDist = dist;
                    // Pick the endpoint closest to the intersection point in V
                    // The intersection is at (intersectU, holeVertex[1])
                    const distToP1 = Math.abs(p1[1] - holeVertex[1]);
                    const distToP2 = Math.abs(p2[1] - holeVertex[1]);
                    bestOuterIndex = distToP1 < distToP2 ? i : j;
                }
            }
        }
    }

    // Check if any reflex vertex on outer is visible and closer
    for (let i = 0; i < outer.length; i++) {
        const v = outer[i];
        // Must be to the right of holeVertex and within V range
        if (v[0] >= holeVertex[0] && Math.abs(v[1] - holeVertex[1]) < 0.001) {
            const dist = v[0] - holeVertex[0];
            if (dist < bestDist) {
                bestDist = dist;
                bestOuterIndex = i;
            }
        }
    }

    // Build merged polygon
    const merged: Vec2[] = [];

    // Part A: outer[0..bestOuterIndex]
    for (let i = 0; i <= bestOuterIndex; i++) {
        merged.push(outer[i]);
    }

    // Part B: all hole vertices starting from rightmost
    for (let i = 0; i < hole.length; i++) {
        const idx = (holeRightmostIndex + i) % hole.length;
        merged.push(hole[idx]);
    }

    // Part C: bridge back
    merged.push([...hole[holeRightmostIndex]]);
    merged.push([...outer[bestOuterIndex]]);

    // Part D: remaining outer
    for (let i = bestOuterIndex + 1; i < outer.length; i++) {
        merged.push(outer[i]);
    }

    return merged;
}

/**
 * Point-in-polygon test using ray casting
 */
function isPointInPolygon(point: Vec2, polygon: Vec2[]): boolean {
    const [x, y] = point;
    let inside = false;

    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const [xi, yi] = polygon[i];
        const [xj, yj] = polygon[j];

        if (((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }

    return inside;
}

/**
 * Evaluate UV mesh vertices to 3D positions and normals
 */
async function evaluateUVMesh(
    surface: Surface,
    uvVertices: Vec2[],
    triangleIndices: [number, number, number][] | Uint32Array,
    profileSample?: (phase: TrimmedSurfaceProfilePhase, elapsedMs: number) => void
): Promise<TessellatedMesh> {
    const numVertices = uvVertices.length;

    // Allocate UVs + indices once regardless of eval path.
    const uvs = new Float32Array(numVertices * 2);
    const indices = triangleIndices instanceof Uint32Array
        ? triangleIndices
        : new Uint32Array(triangleIndices.length * 3);

    for (let i = 0; i < numVertices; i++) {
        const [u, v] = uvVertices[i];
        uvs[i * 2 + 0] = u;
        uvs[i * 2 + 1] = v;
    }

    if (!(triangleIndices instanceof Uint32Array)) {
        for (let i = 0; i < triangleIndices.length; i++) {
            const [a, b, c] = triangleIndices[i];
            indices[i * 3 + 0] = a;
            indices[i * 3 + 1] = b;
            indices[i * 3 + 2] = c;
        }
    }

    // Try GPU evaluation for large primitive faces (CPU fallback below).
    const gpuEvalStart = performance.now();
    const gpuEvaluated = await evaluateSurfaceMeshGPU(surface as any, uvs);
    if (profileSample) {
        profileSample("uvmesh_gpu_eval", performance.now() - gpuEvalStart);
    }
    if (gpuEvaluated) {
        return {
            positions: gpuEvaluated.positions,
            normals: gpuEvaluated.normals,
            indices,
            uvs,
        };
    }

    const positions = new Float32Array(numVertices * 3);
    const normals = new Float32Array(numVertices * 3);

    let nanCount = 0;
    const cpuEvalStart = performance.now();

    // CPU fallback path.
    for (let i = 0; i < numVertices; i++) {
        const [u, v] = uvVertices[i];

        // Get 3D position
        const pos = evaluateSurface(surface, u, v);

        // Check for NaN/invalid positions
        if (!pos || isNaN(pos[0]) || isNaN(pos[1]) || isNaN(pos[2])) {
            nanCount++;
            // Use a fallback position at origin
            positions[i * 3 + 0] = 0;
            positions[i * 3 + 1] = 0;
            positions[i * 3 + 2] = 0;
        } else {
            positions[i * 3 + 0] = pos[0];
            positions[i * 3 + 1] = pos[1];
            positions[i * 3 + 2] = pos[2];
        }

        // Get normal
        const norm = surfaceNormal(surface, u, v);
        normals[i * 3 + 0] = norm[0];
        normals[i * 3 + 1] = norm[1];
        normals[i * 3 + 2] = norm[2];

    }

    if (nanCount > 0) {
        console.warn(`[evaluateUVMesh] ${nanCount}/${numVertices} vertices had NaN positions`);
    }
    if (profileSample) {
        profileSample("uvmesh_cpu_eval", performance.now() - cpuEvalStart);
    }

    return { positions, normals, indices, uvs };
}

/**
 * Add interior Steiner points for better mesh quality
 * Uses a simple grid-based approach
 */
export async function tessellateWithRefinement(
    surface: Surface,
    uvBoundary: Vec2[],
    gridDensity: number = 5
): Promise<TessellatedMesh> {
    // Find UV bounding box
    let uMin = Infinity, uMax = -Infinity;
    let vMin = Infinity, vMax = -Infinity;

    for (const [u, v] of uvBoundary) {
        uMin = Math.min(uMin, u);
        uMax = Math.max(uMax, u);
        vMin = Math.min(vMin, v);
        vMax = Math.max(vMax, v);
    }

    // Add interior grid points
    const allVertices: Vec2[] = [...uvBoundary];
    const du = (uMax - uMin) / (gridDensity + 1);
    const dv = (vMax - vMin) / (gridDensity + 1);

    for (let i = 1; i <= gridDensity; i++) {
        for (let j = 1; j <= gridDensity; j++) {
            const u = uMin + i * du;
            const v = vMin + j * dv;

            // Simple point-in-polygon test (for convex boundaries)
            // TODO: Use proper point-in-polygon for concave boundaries
            if (isPointInConvexPolygon([u, v], uvBoundary)) {
                allVertices.push([u, v]);
            }
        }
    }

    // Triangulate with Steiner points
    // Note: This requires a different approach since CDT expects just boundary
    // For now, return simple triangulation
    const triangles = await constrainedDelaunayTriangulation(uvBoundary, [], true);
    return evaluateUVMesh(surface, uvBoundary, triangles);
}

/**
 * Simple point-in-convex-polygon test
 */
function isPointInConvexPolygon(point: Vec2, polygon: Vec2[]): boolean {
    const n = polygon.length;
    let sign: number | null = null;

    for (let i = 0; i < n; i++) {
        const a = polygon[i];
        const b = polygon[(i + 1) % n];

        // Cross product of edge vector and vector to point
        const cross = (b[0] - a[0]) * (point[1] - a[1]) -
            (b[1] - a[1]) * (point[0] - a[0]);

        if (sign === null) {
            sign = Math.sign(cross);
        } else if (Math.sign(cross) !== 0 && Math.sign(cross) !== sign) {
            return false;
        }
    }

    return true;
}

/**
 * Create a full cylinder mesh (both end caps and lateral surface)
 */
export async function createFullCylinderMesh(
    placement: Axis2Placement3D,
    radius: number,
    height: number,
    numSegments: number = 24
): Promise<TessellatedMesh> {
    const surface: CylindricalSurface = {
        type: "CYLINDRICAL_SURFACE",
        placement,
        radius
    };

    // Tessellate the lateral surface
    return tessellateCylinder(
        surface,
        0,
        Math.PI * 2,
        0,
        height,
        numSegments
    );
}

/**
 * Create a full sphere mesh
 */
export async function createFullSphereMesh(
    placement: Axis2Placement3D,
    radius: number,
    numLonSegments: number = 24,
    numLatSegments: number = 12
): Promise<TessellatedMesh> {
    const surface: SphericalSurface = {
        type: "SPHERICAL_SURFACE",
        placement,
        radius
    };

    return tessellateSphere(
        surface,
        0,
        Math.PI * 2,
        -Math.PI / 2,
        Math.PI / 2,
        numLonSegments,
        numLatSegments
    );
}

// ============================================================================
// C7.1: Adaptive Tessellation Functions
// ============================================================================

/**
 * Tessellation options with quality controls
 */
export interface TessellationQualityOptions {
    /** Enable adaptive refinement based on chord error */
    adaptiveRefinement: boolean;
    /** Maximum chord error tolerance */
    chordTolerance: number;
    /** Maximum refinement depth */
    maxRefinementDepth: number;
    /** Minimum edge length (stop refining smaller edges) */
    minEdgeLength: number;
    /** Maximum triangle aspect ratio before filtering */
    maxAspectRatio: number;
}

export const DEFAULT_TESSELLATION_QUALITY: TessellationQualityOptions = {
    adaptiveRefinement: true,
    chordTolerance: 0.01,
    maxRefinementDepth: 5,
    minEdgeLength: 0.001,
    maxAspectRatio: 50.0,
};

/**
 * Tessellate a cylindrical surface with adaptive refinement (C7.1)
 */
export async function tessellateCylinderAdaptive(
    surface: CylindricalSurface,
    angleStart: number = 0,
    angleEnd: number = Math.PI * 2,
    heightStart: number = 0,
    heightEnd: number = 1,
    options: TessellationQualityOptions = DEFAULT_TESSELLATION_QUALITY
): Promise<TessellatedMesh> {
    // Compute adaptive sample count based on curvature
    const { uSamples, vSamples } = computeAdaptiveSampleCount(
        surface,
        angleStart,
        angleEnd,
        heightStart,
        heightEnd,
        options.chordTolerance
    );

    // Initial tessellation with adaptive sample counts
    const mesh = await tessellateCylinder(
        surface,
        angleStart,
        angleEnd,
        heightStart,
        heightEnd,
        uSamples,
        vSamples
    );

    if (!options.adaptiveRefinement) {
        return mesh;
    }

    // Convert to refinable format
    const uvVertices: Vec2[] = [];
    const positions: Vec3[] = [];
    const normals: Vec3[] = [];

    for (let i = 0; i < mesh.positions.length / 3; i++) {
        positions.push([
            mesh.positions[i * 3],
            mesh.positions[i * 3 + 1],
            mesh.positions[i * 3 + 2],
        ]);
        normals.push([
            mesh.normals[i * 3],
            mesh.normals[i * 3 + 1],
            mesh.normals[i * 3 + 2],
        ]);
        uvVertices.push([mesh.uvs[i * 2], mesh.uvs[i * 2 + 1]]);
    }

    const triangles: [number, number, number][] = [];
    for (let i = 0; i < mesh.indices.length / 3; i++) {
        triangles.push([
            mesh.indices[i * 3],
            mesh.indices[i * 3 + 1],
            mesh.indices[i * 3 + 2],
        ]);
    }

    // Apply adaptive refinement
    const refinementOptions: RefinementOptions = {
        chordTolerance: options.chordTolerance,
        maxDepth: options.maxRefinementDepth,
        minEdgeLength: options.minEdgeLength,
        maxAspectRatio: options.maxAspectRatio,
    };

    const refined = adaptiveRefineMesh(
        surface,
        { uvVertices, positions, normals, triangles },
        refinementOptions
    );

    // Filter degenerate triangles (C7.4)
    const cleanTriangles = filterDegenerateTriangles(
        refined.positions,
        refined.triangles,
        options.maxAspectRatio
    );

    // Convert back to TessellatedMesh format
    return buildTessellatedMesh(refined.uvVertices, refined.positions, refined.normals, cleanTriangles);
}

/**
 * Tessellate a spherical surface with adaptive refinement (C7.1)
 */
export async function tessellateSphereAdaptive(
    surface: SphericalSurface,
    lonStart: number = 0,
    lonEnd: number = Math.PI * 2,
    latStart: number = -Math.PI / 2,
    latEnd: number = Math.PI / 2,
    options: TessellationQualityOptions = DEFAULT_TESSELLATION_QUALITY
): Promise<TessellatedMesh> {
    // Compute adaptive sample count based on curvature
    const { uSamples, vSamples } = computeAdaptiveSampleCount(
        surface,
        lonStart,
        lonEnd,
        latStart,
        latEnd,
        options.chordTolerance
    );

    // Initial tessellation with adaptive sample counts
    const mesh = await tessellateSphere(
        surface,
        lonStart,
        lonEnd,
        latStart,
        latEnd,
        uSamples,
        vSamples
    );

    if (!options.adaptiveRefinement) {
        return mesh;
    }

    // Convert and refine (same as cylinder)
    const uvVertices: Vec2[] = [];
    const positions: Vec3[] = [];
    const normals: Vec3[] = [];

    for (let i = 0; i < mesh.positions.length / 3; i++) {
        positions.push([
            mesh.positions[i * 3],
            mesh.positions[i * 3 + 1],
            mesh.positions[i * 3 + 2],
        ]);
        normals.push([
            mesh.normals[i * 3],
            mesh.normals[i * 3 + 1],
            mesh.normals[i * 3 + 2],
        ]);
        uvVertices.push([mesh.uvs[i * 2], mesh.uvs[i * 2 + 1]]);
    }

    const triangles: [number, number, number][] = [];
    for (let i = 0; i < mesh.indices.length / 3; i++) {
        triangles.push([
            mesh.indices[i * 3],
            mesh.indices[i * 3 + 1],
            mesh.indices[i * 3 + 2],
        ]);
    }

    const refinementOptions: RefinementOptions = {
        chordTolerance: options.chordTolerance,
        maxDepth: options.maxRefinementDepth,
        minEdgeLength: options.minEdgeLength,
        maxAspectRatio: options.maxAspectRatio,
    };

    const refined = adaptiveRefineMesh(
        surface,
        { uvVertices, positions, normals, triangles },
        refinementOptions
    );

    const cleanTriangles = filterDegenerateTriangles(
        refined.positions,
        refined.triangles,
        options.maxAspectRatio
    );

    return buildTessellatedMesh(refined.uvVertices, refined.positions, refined.normals, cleanTriangles);
}

/**
 * Helper to build TessellatedMesh from arrays
 */
function buildTessellatedMesh(
    uvVertices: Vec2[],
    positions: Vec3[],
    normals: Vec3[],
    triangles: [number, number, number][]
): TessellatedMesh {
    const numVertices = positions.length;

    const positionsArray = new Float32Array(numVertices * 3);
    const normalsArray = new Float32Array(numVertices * 3);
    const uvsArray = new Float32Array(numVertices * 2);

    for (let i = 0; i < numVertices; i++) {
        positionsArray[i * 3 + 0] = positions[i][0];
        positionsArray[i * 3 + 1] = positions[i][1];
        positionsArray[i * 3 + 2] = positions[i][2];

        normalsArray[i * 3 + 0] = normals[i][0];
        normalsArray[i * 3 + 1] = normals[i][1];
        normalsArray[i * 3 + 2] = normals[i][2];

        uvsArray[i * 2 + 0] = uvVertices[i][0];
        uvsArray[i * 2 + 1] = uvVertices[i][1];
    }

    const indicesArray = new Uint32Array(triangles.length * 3);
    for (let i = 0; i < triangles.length; i++) {
        indicesArray[i * 3 + 0] = triangles[i][0];
        indicesArray[i * 3 + 1] = triangles[i][1];
        indicesArray[i * 3 + 2] = triangles[i][2];
    }

    return {
        positions: positionsArray,
        normals: normalsArray,
        indices: indicesArray,
        uvs: uvsArray,
    };
}
