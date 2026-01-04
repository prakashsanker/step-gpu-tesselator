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
    if (!surface.placement) {
        console.warn("[evaluateCylinder] Missing placement");
        return [0, 0, 0];
    }
    const { location, axis, refDirection } = surface.placement;
    if (!location || !axis || !refDirection) {
        console.warn("[evaluateCylinder] Missing placement properties");
        return [0, 0, 0];
    }
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
// B_SPLINE_SURFACE (C5)
// ============================================================================

export interface BSplineSurface {
    type: "B_SPLINE_SURFACE";
    controlPoints: Vec3[][];  // 2D array [v][u] of 3D points
    uDegree: number;
    vDegree: number;
    uKnots: number[];
    vKnots: number[];
    weights?: number[][];     // For rational B-splines (NURBS)
}

/**
 * De Boor's algorithm for B-spline curve evaluation at parameter t
 * @param points - Control points
 * @param knots - Knot vector
 * @param degree - Spline degree
 * @param t - Parameter value
 */
function deBoor1D(points: Vec3[], knots: number[], degree: number, t: number): Vec3 {
    const n = points.length - 1;  // Number of control points - 1

    // Clamp t to valid range
    const tMin = knots[degree];
    const tMax = knots[n + 1];
    t = Math.max(tMin, Math.min(tMax, t));

    // Find the knot span index k where knots[k] <= t < knots[k+1]
    let k = degree;
    for (let i = degree; i <= n; i++) {
        if (t >= knots[i] && t < knots[i + 1]) {
            k = i;
            break;
        }
    }
    // Handle t == tMax case
    if (t >= tMax) k = n;

    // Copy relevant control points
    const d: Vec3[] = [];
    for (let j = 0; j <= degree; j++) {
        const idx = Math.max(0, Math.min(n, k - degree + j));
        d.push([...points[idx]]);
    }

    // De Boor recursion
    for (let r = 1; r <= degree; r++) {
        for (let j = degree; j >= r; j--) {
            const i = k - degree + j;
            const denominator = knots[i + degree - r + 1] - knots[i];
            const alpha = denominator > 1e-10 ? (t - knots[i]) / denominator : 0;

            d[j][0] = (1 - alpha) * d[j - 1][0] + alpha * d[j][0];
            d[j][1] = (1 - alpha) * d[j - 1][1] + alpha * d[j][1];
            d[j][2] = (1 - alpha) * d[j - 1][2] + alpha * d[j][2];
        }
    }

    return d[degree];
}

/**
 * Evaluate B-spline surface at (u, v)
 */
export function evaluateBSplineSurface(surface: BSplineSurface, u: number, v: number): Vec3 {
    const { controlPoints, uDegree, vDegree, uKnots, vKnots } = surface;

    // First evaluate in v direction to get a curve of control points
    const vCurvePoints: Vec3[] = [];
    for (let i = 0; i < controlPoints[0].length; i++) {
        // Extract column i (all v-rows at u-index i)
        const columnPoints: Vec3[] = controlPoints.map(row => row[i]);
        vCurvePoints.push(deBoor1D(columnPoints, vKnots, vDegree, v));
    }

    // Then evaluate in u direction
    return deBoor1D(vCurvePoints, uKnots, uDegree, u);
}

/**
 * Approximate B-spline surface normal using finite differences
 */
export function normalBSplineSurface(surface: BSplineSurface, u: number, v: number): Vec3 {
    const eps = 0.001;

    const p = evaluateBSplineSurface(surface, u, v);
    const pu = evaluateBSplineSurface(surface, u + eps, v);
    const pv = evaluateBSplineSurface(surface, u, v + eps);

    // Tangent vectors
    const du: Vec3 = [(pu[0] - p[0]) / eps, (pu[1] - p[1]) / eps, (pu[2] - p[2]) / eps];
    const dv: Vec3 = [(pv[0] - p[0]) / eps, (pv[1] - p[1]) / eps, (pv[2] - p[2]) / eps];

    // Normal = du × dv
    const normal: Vec3 = [
        du[1] * dv[2] - du[2] * dv[1],
        du[2] * dv[0] - du[0] * dv[2],
        du[0] * dv[1] - du[1] * dv[0]
    ];

    return normalize(normal);
}

// ============================================================================
// UNIFIED INTERFACE
// ============================================================================

export type Surface =
    | PlaneSurface
    | CylindricalSurface
    | SphericalSurface
    | ConicalSurface
    | ToroidalSurface
    | BSplineSurface;

/**
 * Evaluate any surface at (u, v) -> [x, y, z]
 */
export function evaluateSurface(surface: Surface, u: number, v: number): Vec3 {
    if (!surface || !surface.type) {
        console.warn("[evaluateSurface] Invalid surface:", surface);
        return [0, 0, 0];
    }
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
        case "B_SPLINE_SURFACE":
            return evaluateBSplineSurface(surface, u, v);
        default:
            console.warn("[evaluateSurface] Unknown surface type:", (surface as { type: string }).type);
            return [0, 0, 0];
    }
}

/**
 * Get surface normal at (u, v) -> [nx, ny, nz]
 */
export function surfaceNormal(surface: Surface, u: number, v: number): Vec3 {
    if (!surface || !surface.type) {
        return [0, 0, 1];
    }
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
        case "B_SPLINE_SURFACE":
            return normalBSplineSurface(surface, u, v);
        default:
            return [0, 0, 1];
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
