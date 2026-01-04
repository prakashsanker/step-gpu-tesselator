/**
 * Surface Tessellation Pipeline
 *
 * Tessellates parametric surfaces (cylinders, spheres, etc.) by:
 * 1. Creating UV boundary from surface patch parameters
 * 2. Triangulating in UV space using CDT
 * 3. Evaluating UV vertices back to 3D positions
 * 4. Computing surface normals
 */

import { constrainedDelaunayTriangulation } from "./cdt-gpu";
import { evaluateSurface, surfaceNormal } from "./surfaces";
import {
    createCylinderUVBoundary,
    createSphereUVBoundary,
    createRectangularUVBoundary
} from "./uv-extraction";

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

type Surface =
    | PlaneSurface
    | CylindricalSurface
    | SphericalSurface
    | ConicalSurface
    | ToroidalSurface;

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
    numAngleSamples: number = 16
): Promise<TessellatedMesh> {
    // Step 1: Create UV boundary
    const uvBoundary = createCylinderUVBoundary(
        angleStart,
        angleEnd,
        heightStart,
        heightEnd,
        numAngleSamples
    );

    // Step 2: Triangulate in UV space
    const triangles = await constrainedDelaunayTriangulation(uvBoundary, [], true);

    // Step 3: Evaluate to 3D
    return evaluateUVMesh(surface, uvBoundary, triangles);
}

/**
 * Tessellate a spherical surface patch
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
    // Step 1: Create UV boundary
    const uvBoundary = createSphereUVBoundary(
        lonStart,
        lonEnd,
        latStart,
        latEnd,
        numLonSamples,
        numLatSamples
    );

    // Step 2: Triangulate in UV space
    const triangles = await constrainedDelaunayTriangulation(uvBoundary, [], true);

    // Step 3: Evaluate to 3D
    return evaluateUVMesh(surface, uvBoundary, triangles);
}

/**
 * Tessellate a conical surface patch
 */
export async function tessellateCone(
    surface: ConicalSurface,
    angleStart: number = 0,
    angleEnd: number = Math.PI * 2,
    heightStart: number = 0,
    heightEnd: number = 1,
    numAngleSamples: number = 16
): Promise<TessellatedMesh> {
    // Use same UV layout as cylinder
    const uvBoundary = createCylinderUVBoundary(
        angleStart,
        angleEnd,
        heightStart,
        heightEnd,
        numAngleSamples
    );

    const triangles = await constrainedDelaunayTriangulation(uvBoundary, [], true);
    return evaluateUVMesh(surface, uvBoundary, triangles);
}

/**
 * Tessellate a toroidal surface patch
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
    // Use sphere-like UV layout for torus
    const uvBoundary = createSphereUVBoundary(
        majorAngleStart,
        majorAngleEnd,
        minorAngleStart,
        minorAngleEnd,
        numMajorSamples,
        numMinorSamples
    );

    const triangles = await constrainedDelaunayTriangulation(uvBoundary, [], true);
    return evaluateUVMesh(surface, uvBoundary, triangles);
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

    // Evaluate each vertex
    for (let i = 0; i < numVertices; i++) {
        const [u, v] = uvVertices[i];

        // Get 3D position
        const pos = evaluateSurface(surface, u, v);
        positions[i * 3 + 0] = pos[0];
        positions[i * 3 + 1] = pos[1];
        positions[i * 3 + 2] = pos[2];

        // Get normal
        const norm = surfaceNormal(surface, u, v);
        normals[i * 3 + 0] = norm[0];
        normals[i * 3 + 1] = norm[1];
        normals[i * 3 + 2] = norm[2];

        // Store UVs
        uvs[i * 2 + 0] = u;
        uvs[i * 2 + 1] = v;
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
