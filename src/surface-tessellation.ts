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
import { evaluateSurface, surfaceNormal } from "./surfaces";
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
 */
export async function tessellateTrimmedSurface(
    surface: Surface,
    uvBoundary: Vec2[],
    gridDensity: number = 16,
    uvHoles: Vec2[][] = []
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

    console.log(`[tessellateTrimmedSurface] nearPosPI=${nearPosPI}, nearNegPI=${nearNegPI}, uSpan=${uSpan.toFixed(3)}, isFullCircle=${isFullCircle}, hasDiscontinuity=${hasDiscontinuity}`);

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

    // Debug: log continuous boundary bounds
    {
        let cbUMin = Infinity, cbUMax = -Infinity, cbVMin = Infinity, cbVMax = -Infinity;
        for (const [u, v] of continuousBoundary) {
            cbUMin = Math.min(cbUMin, u);
            cbUMax = Math.max(cbUMax, u);
            cbVMin = Math.min(cbVMin, v);
            cbVMax = Math.max(cbVMax, v);
        }
        console.log(`[tessellateTrimmedSurface] Continuous boundary bounds: U=[${cbUMin.toFixed(3)}, ${cbUMax.toFixed(3)}], V=[${cbVMin.toFixed(3)}, ${cbVMax.toFixed(3)}]`);
        console.log(`[tessellateTrimmedSurface] Continuous boundary first 5: ${continuousBoundary.slice(0, 5).map(([u, v]) => `(${u.toFixed(2)},${v.toFixed(2)})`).join(' ')}`);
        console.log(`[tessellateTrimmedSurface] Continuous boundary has ${continuousBoundary.length} points`);
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
                console.log(`[tessellateTrimmedSurface] Continuous hole ${i} bounds: U=[${hUMin.toFixed(3)}, ${hUMax.toFixed(3)}], V=[${hVMin.toFixed(3)}, ${hVMax.toFixed(3)}]`);
            }
        }
    }

    // ===== Use CDT with holes for proper hole support =====
    // CDT (Constrained Delaunay Triangulation) with cavity-based constraint recovery
    // ensures hole boundary edges are preserved in the triangulation
    if (continuousHoles.length > 0) {
        console.log(`[tessellateTrimmedSurface] Using CDT with ${continuousHoles.length} holes`);

        // If boundary is sparse (e.g., just 4 rectangle corners), densify it
        // to get better triangulation quality on curved surfaces
        let denseBoundary: Vec2[];
        if (continuousBoundary.length <= 8) {
            denseBoundary = [];
            for (let i = 0; i < continuousBoundary.length; i++) {
                const p1 = continuousBoundary[i];
                const p2 = continuousBoundary[(i + 1) % continuousBoundary.length];
                // Add points along this edge
                const edgePoints = Math.max(8, gridDensity);
                for (let t = 0; t < edgePoints; t++) {
                    const u = p1[0] + (p2[0] - p1[0]) * (t / edgePoints);
                    const v = p1[1] + (p2[1] - p1[1]) * (t / edgePoints);
                    denseBoundary.push([u, v]);
                }
            }
            console.log(`[tessellateTrimmedSurface] Densified boundary from ${continuousBoundary.length} to ${denseBoundary.length} points`);
        } else {
            denseBoundary = continuousBoundary;
        }

        // Use CDT with holes - proper Constrained Delaunay Triangulation
        // that recovers constraint edges using cavity-based approach
        const triangles = await cdtWithHoles(denseBoundary, continuousHoles);
        console.log(`[tessellateTrimmedSurface] CDT with holes generated ${triangles.length} triangles`);

        // Build combined vertex array (boundary + all holes)
        const allVertices: Vec2[] = [...denseBoundary];
        for (const hole of continuousHoles) {
            allVertices.push(...hole);
        }

        return evaluateUVMesh(surface, allVertices, triangles);
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
    const du = (uMax - uMin) / gridDensity;
    const dv = (vMax - vMin) / gridDensity;

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

    for (let j = 0; j <= gridDensity; j++) {
        vertexGrid[j] = [];
        for (let i = 0; i <= gridDensity; i++) {
            const u = uMin + i * du;
            const v = vMin + j * dv;

            // Check if this point is inside the continuous boundary and outside holes
            const insideBoundary = isPointInPolygon([u, v], continuousBoundary);
            const nearBoundary = isNearBoundary([u, v], continuousBoundary, boundaryTolerance);
            const insideHole = continuousHoles.some(hole => isPointInPolygon([u, v], hole));

            // Include points that are inside OR very close to the boundary
            if ((insideBoundary || nearBoundary) && !insideHole) {
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

    console.log(`[tessellateTrimmedSurface] Grid bounds: U=[${uMin.toFixed(3)}, ${uMax.toFixed(3)}], V=[${vMin.toFixed(3)}, ${vMax.toFixed(3)}]`);
    console.log(`[tessellateTrimmedSurface] Grid points: ${insideCount} inside, ${outsideCount} outside`);

    // Diagnostic: test a specific point that should be inside (center of grid)
    const testU = (uMin + uMax) / 2;
    const testV = (vMin + vMax) / 2;
    const testInside = isPointInPolygon([testU, testV], continuousBoundary);
    const testInHole = continuousHoles.some(hole => isPointInPolygon([testU, testV], hole));
    console.log(`[tessellateTrimmedSurface] Test point (${testU.toFixed(3)}, ${testV.toFixed(3)}): insideBoundary=${testInside}, insideHole=${testInHole}`);

    // Debug: trace through point-in-polygon for the center point
    if (!testInside && continuousHoles.length === 0) {
        console.log(`[DEBUG] Tracing point-in-polygon for center point:`);
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
                        console.log(`  Crossing ${crossings}: edge [${j}]->[${i}] from (${xj.toFixed(2)},${yj.toFixed(2)}) to (${xi.toFixed(2)},${yi.toFixed(2)}), intersect at x=${xIntersect.toFixed(2)}`);
                    }
                }
            }
        }
        console.log(`[DEBUG] Total crossings: ${crossings} -> inside=${crossings % 2 === 1}`);

        // Show boundary polygon structure
        console.log(`[DEBUG] Boundary polygon (${continuousBoundary.length} points):`);
        console.log(`  First 5: ${continuousBoundary.slice(0, 5).map((p, i) => `[${i}](${p[0].toFixed(2)},${p[1].toFixed(2)})`).join(' ')}`);
        const quarter = Math.floor(continuousBoundary.length / 4);
        console.log(`  At ${quarter}: (${continuousBoundary[quarter][0].toFixed(2)},${continuousBoundary[quarter][1].toFixed(2)})`);
        const half = Math.floor(continuousBoundary.length / 2);
        console.log(`  At ${half}: (${continuousBoundary[half][0].toFixed(2)},${continuousBoundary[half][1].toFixed(2)})`);
        const threeQuarter = Math.floor(continuousBoundary.length * 3 / 4);
        console.log(`  At ${threeQuarter}: (${continuousBoundary[threeQuarter][0].toFixed(2)},${continuousBoundary[threeQuarter][1].toFixed(2)})`);
        console.log(`  Last 5: ${continuousBoundary.slice(-5).map((p, i) => `[${continuousBoundary.length-5+i}](${p[0].toFixed(2)},${p[1].toFixed(2)})`).join(' ')}`);
    }

    // Additional diagnostic: test a point that SHOULD be inside the hole
    if (continuousHoles.length > 0) {
        const hole = continuousHoles[0];
        const holeUs = hole.map(p => p[0]);
        const holeVs = hole.map(p => p[1]);
        const holeCenterU = (Math.min(...holeUs) + Math.max(...holeUs)) / 2;
        const holeCenterV = (Math.min(...holeVs) + Math.max(...holeVs)) / 2;
        const holeCenterInHole = isPointInPolygon([holeCenterU, holeCenterV], hole);
        console.log(`[tessellateTrimmedSurface] Hole center (${holeCenterU.toFixed(3)}, ${holeCenterV.toFixed(3)}): insideHole=${holeCenterInHole}`);

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
                        console.log(`[DEBUG] Crossing ${debugCrossings}: edge [${j}]->[${i}] from (${xj.toFixed(2)},${yj.toFixed(2)}) to (${xi.toFixed(2)},${yi.toFixed(2)}), intersect at x=${xIntersect.toFixed(2)}`);
                    }
                }
            }
        }
        console.log(`[DEBUG] Total crossings: ${debugCrossings} -> inside=${debugCrossings % 2 === 1}`);

        // Check if all hole points are in expected range
        const outOfRangeU = hole.filter(p => p[0] < -4 || p[0] > 7);
        const outOfRangeV = hole.filter(p => p[1] < -1 || p[1] > 3);
        if (outOfRangeU.length > 0 || outOfRangeV.length > 0) {
            console.log(`[DEBUG] WARNING: ${outOfRangeU.length} points have U outside [-4,7], ${outOfRangeV.length} have V outside [-1,3]`);
        }

        // Show the polygon vertices at key positions
        console.log(`[DEBUG] Hole polygon shape (${hole.length} points):`);
        console.log(`  [0]: (${hole[0][0].toFixed(3)}, ${hole[0][1].toFixed(3)})`);
        console.log(`  [1]: (${hole[1][0].toFixed(3)}, ${hole[1][1].toFixed(3)})`);
        const mid = Math.floor(hole.length / 2);
        console.log(`  [${mid}]: (${hole[mid][0].toFixed(3)}, ${hole[mid][1].toFixed(3)})`);
        console.log(`  [${hole.length-2}]: (${hole[hole.length-2][0].toFixed(3)}, ${hole[hole.length-2][1].toFixed(3)})`);
        console.log(`  [${hole.length-1}]: (${hole[hole.length-1][0].toFixed(3)}, ${hole[hole.length-1][1].toFixed(3)})`);

        // Check if first point at V=0.6, last points approaching first
        const firstV = hole[0][1];
        const lastV = hole[hole.length-1][1];
        console.log(`[DEBUG] First point V=${firstV.toFixed(3)}, last point V=${lastV.toFixed(3)}`);
        console.log(`[DEBUG] Test point V=${testY.toFixed(3)} - should be between ${Math.min(firstV, lastV).toFixed(3)} and ${Math.max(firstV, lastV).toFixed(3)} for the vertical edges`);
    }

    // Create triangles from the grid
    const triangles: [number, number, number][] = [];

    for (let j = 0; j < gridDensity; j++) {
        for (let i = 0; i < gridDensity; i++) {
            const v00 = vertexGrid[j][i];
            const v10 = vertexGrid[j][i + 1];
            const v01 = vertexGrid[j + 1][i];
            const v11 = vertexGrid[j + 1][i + 1];

            // Create triangles only if all vertices are valid
            if (v00 !== null && v10 !== null && v01 !== null && v11 !== null) {
                // Two triangles per quad
                triangles.push([v00, v01, v11]);
                triangles.push([v00, v11, v10]);
            } else {
                // Handle partial quads - create triangles where possible
                if (v00 !== null && v01 !== null && v11 !== null) {
                    triangles.push([v00, v01, v11]);
                }
                if (v00 !== null && v11 !== null && v10 !== null) {
                    triangles.push([v00, v11, v10]);
                }
                if (v00 !== null && v01 !== null && v10 !== null) {
                    triangles.push([v00, v01, v10]);
                }
                if (v01 !== null && v11 !== null && v10 !== null) {
                    triangles.push([v01, v11, v10]);
                }
            }
        }
    }


    console.log(`[tessellateTrimmedSurface] Generated ${triangles.length} triangles from ${uvVertices.length} vertices`);

    // Note: We do NOT add boundary stitching triangles here.
    // The boundary stitching was causing visible spike artifacts on curved surfaces.
    // The grid-based tessellation alone provides clean results, even if there are small gaps
    // at the edges. This matches the approach used in the benchmark-research branch.

    return evaluateUVMesh(surface, uvVertices, triangles);
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
function evaluateUVMesh(
    surface: Surface,
    uvVertices: Vec2[],
    triangleIndices: [number, number, number][]
): TessellatedMesh {
    const numVertices = uvVertices.length;

    // Allocate arrays
    const positions = new Float32Array(numVertices * 3);
    const normals = new Float32Array(numVertices * 3);
    const uvs = new Float32Array(numVertices * 2);

    let nanCount = 0;

    // Evaluate each vertex
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

        // Store UVs
        uvs[i * 2 + 0] = u;
        uvs[i * 2 + 1] = v;
    }

    if (nanCount > 0) {
        console.warn(`[evaluateUVMesh] ${nanCount}/${numVertices} vertices had NaN positions`);
    }

    // Build index buffer
    const indices = new Uint32Array(triangleIndices.length * 3);
    for (let i = 0; i < triangleIndices.length; i++) {
        const [a, b, c] = triangleIndices[i];
        indices[i * 3 + 0] = a;
        indices[i * 3 + 1] = b;
        indices[i * 3 + 2] = c;
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
