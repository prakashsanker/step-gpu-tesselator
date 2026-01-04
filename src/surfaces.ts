/**
 * STEP Surface Definitions and UV-to-3D Evaluation
 *
 * Supports:
 * - PLANE
 * - CYLINDRICAL_SURFACE
 * - SPHERICAL_SURFACE
 * - CONICAL_SURFACE
 * - TOROIDAL_SURFACE
 *
 * Each surface type provides:
 * - evaluate(u, v) -> [x, y, z]  : UV to 3D point
 * - normal(u, v) -> [nx, ny, nz] : Surface normal at UV
 */

export type Vec3 = [number, number, number];

export interface Axis2Placement3D {
    location: Vec3;      // Origin point
    axis: Vec3;          // Z direction (normal)
    refDirection: Vec3;  // X direction
}

/**
 * Compute the Y direction from axis and refDirection
 */
function computeYDirection(placement: Axis2Placement3D): Vec3 {
    const [ax, ay, az] = placement.axis;
    const [rx, ry, rz] = placement.refDirection;
    // Y = Z × X (cross product)
    return [
        ay * rz - az * ry,
        az * rx - ax * rz,
        ax * ry - ay * rx
    ];
}

/**
 * Normalize a vector
 */
function normalize(v: Vec3): Vec3 {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < 1e-10) return [0, 0, 1];
    return [v[0] / len, v[1] / len, v[2] / len];
}

// ============================================================================
// PLANE
// ============================================================================

export interface PlaneSurface {
    type: "PLANE";
    placement: Axis2Placement3D;
}

export function evaluatePlane(surface: PlaneSurface, u: number, v: number): Vec3 {
    const { location, refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);

    return [
        location[0] + u * refDirection[0] + v * yDir[0],
        location[1] + u * refDirection[1] + v * yDir[1],
        location[2] + u * refDirection[2] + v * yDir[2]
    ];
}

export function normalPlane(surface: PlaneSurface, _u: number, _v: number): Vec3 {
    return normalize(surface.placement.axis);
}

// ============================================================================
// CYLINDRICAL_SURFACE
// ============================================================================

export interface CylindricalSurface {
    type: "CYLINDRICAL_SURFACE";
    placement: Axis2Placement3D;
    radius: number;
}

/**
 * Evaluate cylindrical surface at (u, v)
 * u = angle in radians (around axis)
 * v = distance along axis
 *
 * P = location + radius * cos(u) * X + radius * sin(u) * Y + v * Z
 */
export function evaluateCylinder(surface: CylindricalSurface, u: number, v: number): Vec3 {
    const { location, axis, refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);
    const r = surface.radius;

    const cosU = Math.cos(u);
    const sinU = Math.sin(u);

    return [
        location[0] + r * cosU * refDirection[0] + r * sinU * yDir[0] + v * axis[0],
        location[1] + r * cosU * refDirection[1] + r * sinU * yDir[1] + v * axis[1],
        location[2] + r * cosU * refDirection[2] + r * sinU * yDir[2] + v * axis[2]
    ];
}

/**
 * Normal points radially outward
 * N = cos(u) * X + sin(u) * Y
 */
export function normalCylinder(surface: CylindricalSurface, u: number, _v: number): Vec3 {
    const { refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);

    const cosU = Math.cos(u);
    const sinU = Math.sin(u);

    return normalize([
        cosU * refDirection[0] + sinU * yDir[0],
        cosU * refDirection[1] + sinU * yDir[1],
        cosU * refDirection[2] + sinU * yDir[2]
    ]);
}

// ============================================================================
// SPHERICAL_SURFACE
// ============================================================================

export interface SphericalSurface {
    type: "SPHERICAL_SURFACE";
    placement: Axis2Placement3D;
    radius: number;
}

/**
 * Evaluate spherical surface at (u, v)
 * u = longitude (angle around axis)
 * v = latitude (angle from equator, -π/2 to π/2)
 *
 * P = location + R * cos(v) * cos(u) * X + R * cos(v) * sin(u) * Y + R * sin(v) * Z
 */
export function evaluateSphere(surface: SphericalSurface, u: number, v: number): Vec3 {
    const { location, axis, refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);
    const r = surface.radius;

    const cosU = Math.cos(u);
    const sinU = Math.sin(u);
    const cosV = Math.cos(v);
    const sinV = Math.sin(v);

    return [
        location[0] + r * cosV * cosU * refDirection[0] + r * cosV * sinU * yDir[0] + r * sinV * axis[0],
        location[1] + r * cosV * cosU * refDirection[1] + r * cosV * sinU * yDir[1] + r * sinV * axis[1],
        location[2] + r * cosV * cosU * refDirection[2] + r * cosV * sinU * yDir[2] + r * sinV * axis[2]
    ];
}

/**
 * Normal points radially outward
 */
export function normalSphere(surface: SphericalSurface, u: number, v: number): Vec3 {
    const { axis, refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);

    const cosU = Math.cos(u);
    const sinU = Math.sin(u);
    const cosV = Math.cos(v);
    const sinV = Math.sin(v);

    return normalize([
        cosV * cosU * refDirection[0] + cosV * sinU * yDir[0] + sinV * axis[0],
        cosV * cosU * refDirection[1] + cosV * sinU * yDir[1] + sinV * axis[1],
        cosV * cosU * refDirection[2] + cosV * sinU * yDir[2] + sinV * axis[2]
    ]);
}

// ============================================================================
// CONICAL_SURFACE
// ============================================================================

export interface ConicalSurface {
    type: "CONICAL_SURFACE";
    placement: Axis2Placement3D;
    radius: number;      // Base radius at v=0
    semiAngle: number;   // Cone half-angle in radians
}

/**
 * Evaluate conical surface at (u, v)
 * u = angle around axis
 * v = distance along axis from apex
 *
 * Local radius at height v: r(v) = radius + v * tan(semiAngle)
 * P = location + r(v) * cos(u) * X + r(v) * sin(u) * Y + v * Z
 */
export function evaluateCone(surface: ConicalSurface, u: number, v: number): Vec3 {
    const { location, axis, refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);

    const localRadius = surface.radius + v * Math.tan(surface.semiAngle);
    const cosU = Math.cos(u);
    const sinU = Math.sin(u);

    return [
        location[0] + localRadius * cosU * refDirection[0] + localRadius * sinU * yDir[0] + v * axis[0],
        location[1] + localRadius * cosU * refDirection[1] + localRadius * sinU * yDir[1] + v * axis[1],
        location[2] + localRadius * cosU * refDirection[2] + localRadius * sinU * yDir[2] + v * axis[2]
    ];
}

/**
 * Cone normal
 */
export function normalCone(surface: ConicalSurface, u: number, _v: number): Vec3 {
    const { axis, refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);
    const cosAngle = Math.cos(surface.semiAngle);
    const sinAngle = Math.sin(surface.semiAngle);

    const cosU = Math.cos(u);
    const sinU = Math.sin(u);

    // Normal is perpendicular to the cone surface
    // N = cosAngle * (cosU * X + sinU * Y) - sinAngle * Z
    return normalize([
        cosAngle * (cosU * refDirection[0] + sinU * yDir[0]) - sinAngle * axis[0],
        cosAngle * (cosU * refDirection[1] + sinU * yDir[1]) - sinAngle * axis[1],
        cosAngle * (cosU * refDirection[2] + sinU * yDir[2]) - sinAngle * axis[2]
    ]);
}

// ============================================================================
// TOROIDAL_SURFACE
// ============================================================================

export interface ToroidalSurface {
    type: "TOROIDAL_SURFACE";
    placement: Axis2Placement3D;
    majorRadius: number;  // Distance from center to tube center
    minorRadius: number;  // Tube radius
}

/**
 * Evaluate toroidal surface at (u, v)
 * u = major angle (around the torus axis)
 * v = minor angle (around the tube)
 *
 * P = location + (R + r*cos(v)) * cos(u) * X + (R + r*cos(v)) * sin(u) * Y + r*sin(v) * Z
 */
export function evaluateTorus(surface: ToroidalSurface, u: number, v: number): Vec3 {
    const { location, axis, refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);
    const R = surface.majorRadius;
    const r = surface.minorRadius;

    const cosU = Math.cos(u);
    const sinU = Math.sin(u);
    const cosV = Math.cos(v);
    const sinV = Math.sin(v);

    const tubeCenter = R + r * cosV;

    return [
        location[0] + tubeCenter * cosU * refDirection[0] + tubeCenter * sinU * yDir[0] + r * sinV * axis[0],
        location[1] + tubeCenter * cosU * refDirection[1] + tubeCenter * sinU * yDir[1] + r * sinV * axis[1],
        location[2] + tubeCenter * cosU * refDirection[2] + tubeCenter * sinU * yDir[2] + r * sinV * axis[2]
    ];
}

/**
 * Torus normal
 */
export function normalTorus(surface: ToroidalSurface, u: number, v: number): Vec3 {
    const { axis, refDirection } = surface.placement;
    const yDir = computeYDirection(surface.placement);

    const cosU = Math.cos(u);
    const sinU = Math.sin(u);
    const cosV = Math.cos(v);
    const sinV = Math.sin(v);

    // Normal direction at the tube surface
    return normalize([
        cosV * (cosU * refDirection[0] + sinU * yDir[0]) + sinV * axis[0],
        cosV * (cosU * refDirection[1] + sinU * yDir[1]) + sinV * axis[1],
        cosV * (cosU * refDirection[2] + sinU * yDir[2]) + sinV * axis[2]
    ]);
}

// ============================================================================
// UNIFIED INTERFACE
// ============================================================================

export type Surface =
    | PlaneSurface
    | CylindricalSurface
    | SphericalSurface
    | ConicalSurface
    | ToroidalSurface;

/**
 * Evaluate any surface at (u, v) -> [x, y, z]
 */
export function evaluateSurface(surface: Surface, u: number, v: number): Vec3 {
    switch (surface.type) {
        case "PLANE":
            return evaluatePlane(surface, u, v);
        case "CYLINDRICAL_SURFACE":
            return evaluateCylinder(surface, u, v);
        case "SPHERICAL_SURFACE":
            return evaluateSphere(surface, u, v);
        case "CONICAL_SURFACE":
            return evaluateCone(surface, u, v);
        case "TOROIDAL_SURFACE":
            return evaluateTorus(surface, u, v);
    }
}

/**
 * Get surface normal at (u, v) -> [nx, ny, nz]
 */
export function surfaceNormal(surface: Surface, u: number, v: number): Vec3 {
    switch (surface.type) {
        case "PLANE":
            return normalPlane(surface, u, v);
        case "CYLINDRICAL_SURFACE":
            return normalCylinder(surface, u, v);
        case "SPHERICAL_SURFACE":
            return normalSphere(surface, u, v);
        case "CONICAL_SURFACE":
            return normalCone(surface, u, v);
        case "TOROIDAL_SURFACE":
            return normalTorus(surface, u, v);
    }
}

/**
 * Evaluate a list of UV points to 3D
 */
export function evaluateUVPoints(
    surface: Surface,
    uvPoints: [number, number][]
): Vec3[] {
    return uvPoints.map(([u, v]) => evaluateSurface(surface, u, v));
}

/**
 * Evaluate triangles from UV space to 3D
 * Returns positions and normals for rendering
 */
export function evaluateTriangles(
    surface: Surface,
    uvVertices: [number, number][],
    triangleIndices: [number, number, number][]
): { positions: Float32Array; normals: Float32Array; indices: Uint32Array } {
    const positions = new Float32Array(uvVertices.length * 3);
    const normals = new Float32Array(uvVertices.length * 3);

    for (let i = 0; i < uvVertices.length; i++) {
        const [u, v] = uvVertices[i];
        const pos = evaluateSurface(surface, u, v);
        const norm = surfaceNormal(surface, u, v);

        positions[i * 3 + 0] = pos[0];
        positions[i * 3 + 1] = pos[1];
        positions[i * 3 + 2] = pos[2];

        normals[i * 3 + 0] = norm[0];
        normals[i * 3 + 1] = norm[1];
        normals[i * 3 + 2] = norm[2];
    }

    const indices = new Uint32Array(triangleIndices.length * 3);
    for (let i = 0; i < triangleIndices.length; i++) {
        const [a, b, c] = triangleIndices[i];
        indices[i * 3 + 0] = a;
        indices[i * 3 + 1] = b;
        indices[i * 3 + 2] = c;
    }

    return { positions, normals, indices };
}
