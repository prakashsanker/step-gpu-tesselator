import { earClipping } from "./ear-clipping";
import { earClippingSingleDispatch } from "./ear-clipping-single-dispatch";
import { earClippingOptimized } from "./ear-clipping-optimized";
import { earClippingBatched, type BatchedPolygon } from "./ear-clipping-batched";
import { triangulateHybrid } from "./triangulate-hybrid";
import {
  tessellateCylinder,
  tessellateSphere,
  tessellateCone,
  tessellateTorus,
  tessellateBSplineSurface,
  tessellateTrimmedSurface,
} from "./surface-tessellation";
import { evaluateBSplineSurface, type BSplineSurface as BSplineSurfaceType } from "./surfaces";
import { computeSmoothNormals, filterDegenerateTriangles } from "./mesh-quality";
import { computeSmoothNormalsGPU } from "./smooth-normals-gpu";

// Minimal STEP → mesh parser for the square face example
type Vec3 = [number, number, number];
type Vec2 = [number, number];

// =============================================================================
// Ear Clipping Algorithm Selection
// =============================================================================

// Use VITE_USE_OPTIMIZED_EAR_CLIPPING=true to enable optimized version
const USE_OPTIMIZED = import.meta.env.VITE_USE_OPTIMIZED_EAR_CLIPPING === 'true';

// Maximum vertices for optimized version (workgroup memory limit)
const OPTIMIZED_MAX_VERTICES = 256;

/**
 * Select and run the appropriate ear clipping algorithm.
 * - Optimized: Single GPU dispatch with parallel processing (≤256 vertices)
 * - SingleDispatch: Single GPU dispatch, single-threaded (>256 vertices or fallback)
 * - Original: Multiple GPU dispatches (when env var not set)
 */
async function runEarClipping(points: Vec3[]): Promise<number[][]> {
  if (USE_OPTIMIZED) {
    if (points.length <= OPTIMIZED_MAX_VERTICES) {
      return await earClippingOptimized(points);
    } else {
      // Fall back to single-dispatch for large polygons
      return await earClippingSingleDispatch(points);
    }
  }
  // Default: original ear clipping
  return await earClipping(points);
}

// =============================================================================
// Vector Helpers
// =============================================================================

const EPSILON = 1e-10;

// Vec3 operations
function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function vec3Len(v: Vec3): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function vec3Normalize(v: Vec3): Vec3 {
  const len = vec3Len(v);
  if (len < EPSILON) {
    return [0, 0, 0];
  }
  return [v[0] / len, v[1] / len, v[2] / len];
}

function vec3Scale(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

// Vec2 operations
function vec2Dot(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}

function vec2Len(v: Vec2): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1]);
}

// =============================================================================
// C6b: B-Spline Curve Evaluation (De Boor's Algorithm)
// =============================================================================

/**
 * Evaluate a B-spline curve at parameter t using De Boor's algorithm.
 * @param controlPoints - Array of 3D control points
 * @param knots - Full knot vector (with multiplicities expanded)
 * @param degree - Curve degree
 * @param t - Parameter value
 */
function evaluateBSplineCurve(
  controlPoints: Vec3[],
  knots: number[],
  degree: number,
  t: number
): Vec3 {
  const n = controlPoints.length - 1;

  // Clamp t to valid range
  const tMin = knots[degree];
  const tMax = knots[n + 1];
  t = Math.max(tMin, Math.min(tMax - 1e-10, t));

  // Find the knot span index k where knots[k] <= t < knots[k+1]
  let k = degree;
  for (let i = degree; i <= n; i++) {
    if (t >= knots[i] && t < knots[i + 1]) {
      k = i;
      break;
    }
  }
  // Handle edge case at tMax
  if (t >= tMax - 1e-10) {
    k = n;
  }

  // De Boor's algorithm: work with control points P[k-degree] ... P[k]
  // Copy the relevant control points
  const d: Vec3[] = [];
  for (let i = 0; i <= degree; i++) {
    const idx = k - degree + i;
    if (idx >= 0 && idx <= n) {
      d.push([...controlPoints[idx]]);
    } else {
      d.push([0, 0, 0]);
    }
  }

  // Apply the triangular scheme
  for (let r = 1; r <= degree; r++) {
    for (let j = degree; j >= r; j--) {
      const i = k - degree + j;
      const denom = knots[i + degree - r + 1] - knots[i];
      if (Math.abs(denom) < 1e-10) {
        continue; // Skip degenerate case
      }
      const alpha = (t - knots[i]) / denom;
      d[j][0] = (1 - alpha) * d[j - 1][0] + alpha * d[j][0];
      d[j][1] = (1 - alpha) * d[j - 1][1] + alpha * d[j][1];
      d[j][2] = (1 - alpha) * d[j - 1][2] + alpha * d[j][2];
    }
  }

  return d[degree];
}

/**
 * Sample a B-spline curve at uniform parameter intervals.
 * @param controlPoints - Array of 3D control points
 * @param knotMultiplicities - Knot multiplicities
 * @param knotValues - Knot values (before expansion)
 * @param degree - Curve degree
 * @param numSamples - Number of samples to generate
 */
function sampleBSplineCurve(
  controlPoints: Vec3[],
  knotMultiplicities: number[],
  knotValues: number[],
  degree: number,
  numSamples: number
): Vec3[] {
  // Build full knot vector from multiplicities
  const knots: number[] = [];
  for (let i = 0; i < knotValues.length; i++) {
    const multiplicity = knotMultiplicities[i] || 1;
    for (let j = 0; j < multiplicity; j++) {
      knots.push(knotValues[i]);
    }
  }

  // Valid parameter range
  const tMin = knots[degree];
  const tMax = knots[controlPoints.length];

  const samples: Vec3[] = [];
  for (let i = 0; i < numSamples; i++) {
    const t = tMin + (i / (numSamples - 1)) * (tMax - tMin);
    samples.push(evaluateBSplineCurve(controlPoints, knots, degree, t));
  }

  return samples;
}

/** Compute signed area of a 2D polygon (positive = CCW) */
function computeSignedArea2D(points: Vec2[]): number {
  let sum = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const curr = points[i];
    const next = points[(i + 1) % n];
    sum += curr[0] * next[1] - next[0] * curr[1];
  }
  return sum / 2;
}

// =============================================================================
// Face Basis & Projection (3D → 2D)
// =============================================================================

/** A local coordinate system on a planar face */
export interface FaceBasis {
  origin: Vec3;  // A point on the plane
  u: Vec3;       // Unit vector for local X axis
  v: Vec3;       // Unit vector for local Y axis
  n: Vec3;       // Unit normal to the plane
}

/** Result of projecting face loops to 2D */
export interface ProjectedLoops {
  outer2d: Vec2[];
  holes2d: Vec2[][];
}

/** Result of winding normalization */
export interface NormalizedLoops {
  outer2d: Vec2[];
  holes2d: Vec2[][];
  outerReversed: boolean;
  holesReversed: boolean[];  // Track which holes were reversed
}

/**
 * Compute a face basis from STEP plane data if available,
 * otherwise fall back to computing from the outer loop geometry.
 */
function computeFaceBasisFromStepFace(
  model: StepModel,
  face: AdvancedFace,
  outerLoop: Vec3[]
): FaceBasis {
  // Try to get basis from STEP plane entity
  const plane = model.planes.get(face.surfaceId);
  if (plane) {
    const placement = model.axis2Placements.get(plane.placementId);
    if (placement) {
      const originPoint = model.points.get(placement.locationId);
      if (originPoint) {
        const origin = originPoint.coords;

        // Get normal (axis) - defaults to Z if not specified
        let n: Vec3 = [0, 0, 1];
        if (placement.axisId !== null) {
          const axisDir = model.directions.get(placement.axisId);
          if (axisDir) {
            n = vec3Normalize(axisDir.dir);
          }
        }

        // Flip normal if face.sameSense is false
        // sameSense indicates whether the face normal agrees with the surface normal
        if (!face.sameSense) {
          n = [-n[0], -n[1], -n[2]];
        }

        // Get reference direction (u) - defaults to X if not specified
        let u: Vec3 = [1, 0, 0];
        if (placement.refDirectionId !== null) {
          const refDir = model.directions.get(placement.refDirectionId);
          if (refDir) {
            u = vec3Normalize(refDir.dir);
          }
        }

        // Compute v = normalize(cross(n, u))
        let v = vec3Normalize(vec3Cross(n, u));

        // Re-orthonormalize u to ensure u ⟂ n and u ⟂ v
        // u = normalize(cross(v, n))
        u = vec3Normalize(vec3Cross(v, n));

        // Check for degenerate basis
        if (vec3Len(u) > EPSILON && vec3Len(v) > EPSILON && vec3Len(n) > EPSILON) {
          return { origin, u, v, n };
        }
      }
    }
  }

  // Fallback: compute basis from outer loop geometry
  return computeFaceBasisFromLoop(outerLoop);
}

/**
 * Compute a face basis geometrically from the outer loop vertices.
 * Used as fallback when STEP plane data is not available.
 */
function computeFaceBasisFromLoop(loop: Vec3[]): FaceBasis {
  if (loop.length < 3) {
    throw new Error("Cannot compute basis from loop with fewer than 3 vertices");
  }

  const origin = loop[0];

  // Find first non-degenerate triangle to compute normal
  let n: Vec3 = [0, 0, 0];
  for (let i = 1; i < loop.length - 1; i++) {
    const edge1 = vec3Sub(loop[i], origin);
    const edge2 = vec3Sub(loop[i + 1], origin);
    const cross = vec3Cross(edge1, edge2);
    const len = vec3Len(cross);
    if (len > EPSILON) {
      n = vec3Scale(cross, 1 / len);  // normalize
      break;
    }
  }

  // Check if we found a valid normal
  if (vec3Len(n) < EPSILON) {
    throw new Error("Degenerate polygon: could not compute normal");
  }

  // Compute u from first edge, projected onto the plane
  const edge1 = vec3Sub(loop[1], origin);
  // Remove component along normal: u' = edge1 - dot(edge1, n) * n
  const dotEN = vec3Dot(edge1, n);
  let u = vec3Sub(edge1, vec3Scale(n, dotEN));
  u = vec3Normalize(u);

  // If u is degenerate, try other edges
  if (vec3Len(u) < EPSILON) {
    for (let i = 2; i < loop.length; i++) {
      const edge = vec3Sub(loop[i], origin);
      const dot = vec3Dot(edge, n);
      u = vec3Sub(edge, vec3Scale(n, dot));
      u = vec3Normalize(u);
      if (vec3Len(u) > EPSILON) break;
    }
  }

  if (vec3Len(u) < EPSILON) {
    throw new Error("Degenerate polygon: could not compute u basis vector");
  }

  // Compute v = normalize(cross(n, u))
  const v = vec3Normalize(vec3Cross(n, u));

  return { origin, u, v, n };
}

/**
 * Project a single 3D point to 2D using the given basis.
 *
 * Formula:
 *   d = p - origin
 *   x = dot(d, u)
 *   y = dot(d, v)
 */
function projectPointTo2D(p: Vec3, basis: FaceBasis): Vec2 {
  const d = vec3Sub(p, basis.origin);
  const x = vec3Dot(d, basis.u);
  const y = vec3Dot(d, basis.v);
  return [x, y];
}

/**
 * Project all face loops (outer + holes) from 3D to 2D.
 */
function projectFaceLoopsTo2D(
  faceLoops3d: { outer: Vec3[]; holes: Vec3[][] },
  basis: FaceBasis
): ProjectedLoops {
  const outer2d = faceLoops3d.outer.map(p => projectPointTo2D(p, basis));
  const holes2d = faceLoops3d.holes.map(hole =>
    hole.map(p => projectPointTo2D(p, basis))
  );
  return { outer2d, holes2d };
}

// =============================================================================
// C2.3: Winding Normalization
// =============================================================================

/**
 * Normalize winding order for triangulation:
 * - Outer loop: CCW (counter-clockwise, positive signed area)
 * - Holes: CW (clockwise, negative signed area)
 *
 * This is required because:
 * 1. Ear clipping expects CCW winding for the outer boundary
 * 2. When bridging holes (C2.5a), holes must wind opposite to outer
 * 3. Consistent winding ensures correct interior/exterior classification
 *
 * @param projected - The 2D projected loops (outer + holes)
 * @returns Normalized loops with tracking of which were reversed
 */
function normalizeWinding(projected: ProjectedLoops): NormalizedLoops {
  const { outer2d, holes2d } = projected;

  // Step 1: Check outer loop winding
  const outerArea = computeSignedArea2D(outer2d);
  const outerReversed = outerArea < 0;  // CW needs to be reversed

  // If outer is CW (negative area), reverse to make CCW
  const normalizedOuter = outerReversed
    ? outer2d.slice().reverse()
    : outer2d;

  if (outerReversed) {
  } else {
  }

  // Step 2: Check each hole's winding
  const holesReversed: boolean[] = [];
  const normalizedHoles: Vec2[][] = [];

  for (let i = 0; i < holes2d.length; i++) {
    const hole = holes2d[i];
    const holeArea = computeSignedArea2D(hole);
    const needsReverse = holeArea > 0;  // CCW needs to be reversed to CW

    holesReversed.push(needsReverse);

    if (needsReverse) {
      normalizedHoles.push(hole.slice().reverse());
    } else {
      normalizedHoles.push(hole);
    }
  }

  return {
    outer2d: normalizedOuter,
    holes2d: normalizedHoles,
    outerReversed,
    holesReversed,
  };
}

/**
 * Apply the same reversal to 3D loops that was applied to 2D loops.
 * This keeps 3D and 2D arrays in sync for correct vertex indexing.
 */
function applyWindingTo3D(
  loops3d: { outer: Vec3[]; holes: Vec3[][] },
  outerReversed: boolean,
  holesReversed: boolean[]
): { outer: Vec3[]; holes: Vec3[][] } {
  const outer = outerReversed
    ? loops3d.outer.slice().reverse()
    : loops3d.outer;

  const holes = loops3d.holes.map((hole, i) =>
    holesReversed[i] ? hole.slice().reverse() : hole
  );

  return { outer, holes };
}

// =============================================================================
// C2.4: Topology Validation
// =============================================================================

/**
 * Result of topology validation
 */
export interface TopologyValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * Check if a point is inside a polygon using ray casting algorithm.
 * Casts a ray from the point to the right (+X) and counts edge crossings.
 * Odd crossings = inside, even = outside.
 */
function pointInPolygon(point: Vec2, polygon: Vec2[]): boolean {
  const [px, py] = point;
  let inside = false;
  const n = polygon.length;

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];

    // Check if horizontal ray from point crosses this edge
    //
    // Step 1: Does the edge straddle our y-level?
    //         One vertex must be above py, one below
    const edgeStraddlesRay = (yi > py) !== (yj > py);

    // Step 2: Where does the edge cross the horizontal line y = py?
    //         Using linear interpolation along the edge:
    //         t = (py - yi) / (yj - yi)     [how far along the edge]
    //         xCrossing = xi + t * (xj - xi) [x-coord at that point]
    const t = (py - yi) / (yj - yi);
    const xCrossing = xi + t * (xj - xi);

    // Step 3: Is the crossing point to the RIGHT of our point?
    //         (We're casting ray in +X direction)
    const crossingIsToRight = px < xCrossing;

    // If edge straddles ray AND crossing is to the right, we hit this edge
    if (edgeStraddlesRay && crossingIsToRight) {
      inside = !inside;
    }
  }

  return inside;
}

/**
 * Check if two line segments intersect (proper intersection, not touching endpoints).
 * Uses the orientation test method.
 *
 * @param a1, a2 - First segment endpoints
 * @param b1, b2 - Second segment endpoints
 * @returns true if segments properly intersect (cross each other)
 */
function segmentsIntersect(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2): boolean {
  // Compute orientation of three points (sign of cross product)
  function orientation(p: Vec2, q: Vec2, r: Vec2): number {
    const val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
    if (Math.abs(val) < EPSILON) return 0;  // Collinear
    return val > 0 ? 1 : -1;  // Clockwise or counter-clockwise
  }

  // Check if point q lies on segment pr (when collinear)
  function onSegment(p: Vec2, q: Vec2, r: Vec2): boolean {
    return q[0] <= Math.max(p[0], r[0]) && q[0] >= Math.min(p[0], r[0]) &&
           q[1] <= Math.max(p[1], r[1]) && q[1] >= Math.min(p[1], r[1]);
  }

  const o1 = orientation(a1, a2, b1);
  const o2 = orientation(a1, a2, b2);
  const o3 = orientation(b1, b2, a1);
  const o4 = orientation(b1, b2, a2);

  // General case: segments intersect if orientations differ
  if (o1 !== o2 && o3 !== o4) {
    return true;
  }

  // Special collinear cases
  if (o1 === 0 && onSegment(a1, b1, a2)) return true;
  if (o2 === 0 && onSegment(a1, b2, a2)) return true;
  if (o3 === 0 && onSegment(b1, a1, b2)) return true;
  if (o4 === 0 && onSegment(b1, a2, b2)) return true;

  return false;
}

/**
 * Check if two line segments share a vertex (are adjacent in a polygon).
 */
function segmentsShareVertex(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2): boolean {
  const eq = (p: Vec2, q: Vec2) =>
    Math.abs(p[0] - q[0]) < EPSILON && Math.abs(p[1] - q[1]) < EPSILON;
  return eq(a1, b1) || eq(a1, b2) || eq(a2, b1) || eq(a2, b2);
}

/**
 * Check if a loop is simple (doesn't self-intersect).
 * Checks all non-adjacent edge pairs for intersection.
 */
function isSimpleLoop(loop: Vec2[]): { valid: boolean; error?: string } {
  const n = loop.length;
  if (n < 3) {
    return { valid: false, error: "Loop has fewer than 3 vertices" };
  }

  // Check all pairs of non-adjacent edges
  for (let i = 0; i < n; i++) {
    const a1 = loop[i];
    const a2 = loop[(i + 1) % n];

    // Start from i+2 to skip adjacent edges
    for (let j = i + 2; j < n; j++) {
      // Skip if j+1 wraps around to i (adjacent)
      if (j === n - 1 && i === 0) continue;

      const b1 = loop[j];
      const b2 = loop[(j + 1) % n];

      // Skip if segments share a vertex (adjacent edges)
      if (segmentsShareVertex(a1, a2, b1, b2)) continue;

      if (segmentsIntersect(a1, a2, b1, b2)) {
        return {
          valid: false,
          error: `Self-intersection: edges ${i}-${(i+1)%n} and ${j}-${(j+1)%n}`
        };
      }
    }
  }

  return { valid: true };
}

/**
 * Check if two loops intersect each other.
 * Tests all edge pairs between the two loops.
 */
function loopsIntersect(loop1: Vec2[], loop2: Vec2[], loop1Name: string, loop2Name: string): { intersects: boolean; error?: string } {
  for (let i = 0; i < loop1.length; i++) {
    const a1 = loop1[i];
    const a2 = loop1[(i + 1) % loop1.length];

    for (let j = 0; j < loop2.length; j++) {
      const b1 = loop2[j];
      const b2 = loop2[(j + 1) % loop2.length];

      if (segmentsIntersect(a1, a2, b1, b2)) {
        return {
          intersects: true,
          error: `${loop1Name} edge ${i} intersects ${loop2Name} edge ${j}`
        };
      }
    }
  }

  return { intersects: false };
}

/**
 * Validate the topology of a face with holes.
 *
 * Checks:
 * 1. All loops are simple (no self-intersection)
 * 2. All hole vertices are inside the outer boundary
 * 3. No loop intersects any other loop (outer↔hole, hole↔hole)
 *
 * @param outer2d - Outer boundary in 2D (should be CCW)
 * @param holes2d - Holes in 2D (should each be CW)
 * @returns Validation result with any errors found
 */
function validateTopology(outer2d: Vec2[], holes2d: Vec2[][]): TopologyValidationResult {
  const errors: string[] = [];

  // 1. Check outer loop is simple
  const outerSimple = isSimpleLoop(outer2d);
  if (!outerSimple.valid) {
    errors.push(`Outer loop: ${outerSimple.error}`);
  }

  // 2. Check each hole
  for (let h = 0; h < holes2d.length; h++) {
    const hole = holes2d[h];

    // 2a. Check hole is simple
    const holeSimple = isSimpleLoop(hole);
    if (!holeSimple.valid) {
      errors.push(`Hole ${h}: ${holeSimple.error}`);
    }

    // 2b. Check all hole vertices are inside outer
    for (let v = 0; v < hole.length; v++) {
      if (!pointInPolygon(hole[v], outer2d)) {
        errors.push(`Hole ${h} vertex ${v} is outside outer boundary`);
        break;  // One error per hole is enough
      }
    }

    // 2c. Check hole doesn't intersect outer
    const outerHoleIntersect = loopsIntersect(outer2d, hole, "outer", `hole ${h}`);
    if (outerHoleIntersect.intersects) {
      errors.push(outerHoleIntersect.error!);
    }
  }

  // 3. Check holes don't intersect each other
  for (let i = 0; i < holes2d.length; i++) {
    for (let j = i + 1; j < holes2d.length; j++) {
      const holeHoleIntersect = loopsIntersect(holes2d[i], holes2d[j], `hole ${i}`, `hole ${j}`);
      if (holeHoleIntersect.intersects) {
        errors.push(holeHoleIntersect.error!);
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

// =============================================================================
// C2.5: Hole Bridging Algorithm
// =============================================================================
//
// GOAL: Merge holes into the outer boundary to create a single polygon
//       that can be triangulated with ear clipping.
//
// VISUAL OVERVIEW:
//
//     BEFORE (outer + hole)              AFTER (single merged polygon)
//
//     0 ────────────→ 1                  0 ────────────→ 1
//     ↑               │                  ↑               │
//     │    a → b      │                  │    5 → 6      │
//     │    ↑   ↓      ↓                  │    ↑   ↓      ↓
//     │    d ← c      │        →→→       │    8 ← 7      │
//     │               │                  │  ↗         ↘  │
//     │               │                  │ 4=9         3 │
//     3 ←──────────── 2                  ↑←──────────────┘
//                                        (4 and 9 are same point - bridge)
//                                        (3 appears twice - bridge back)
//
// The merged polygon visits: 0→1→2→3→[bridge to hole]→5→6→7→8→[bridge back]→3→0
//
// =============================================================================

/**
 * Find the index of the vertex with the maximum X coordinate.
 * This is the "rightmost" vertex in the polygon.
 *
 * WHY: We start bridging from the rightmost hole vertex because:
 *      1. A ray cast to +X from this point is guaranteed to hit the outer boundary
 *      2. Processing holes right-to-left prevents bridge edges from crossing
 */
function findRightmostVertexIndex(polygon: Vec2[]): number {
  let maxIndex = 0;
  let maxX = polygon[0][0];

  for (let i = 1; i < polygon.length; i++) {
    // If this vertex has a larger X coordinate, it becomes the new rightmost
    if (polygon[i][0] > maxX) {
      maxX = polygon[i][0];
      maxIndex = i;
    }
  }

  return maxIndex;
}

/**
 * Cast a horizontal ray from point P in the +X direction and find where
 * it first intersects an edge of the polygon.
 *
 * VISUAL:
 *
 *         P ─────────────────●─────────────→ +X
 *                            ↑
 *                       intersection
 *                            │
 *                      ┌─────┴─────┐
 *                      │           │
 *                      │   outer   │
 *                      │  polygon  │
 *                      └───────────┘
 *
 * RETURNS:
 *   - edgeStartIndex: which edge was hit (edge goes from [i] to [i+1])
 *   - intersectionX: the X coordinate where ray hits the edge
 *   - edgeParameter: how far along the edge (0.0 = at start, 1.0 = at end)
 */
function castRayToPolygon(
  rayOrigin: Vec2,
  polygon: Vec2[]
): { edgeStartIndex: number; intersectionX: number; edgeParameter: number } | null {

  const rayY = rayOrigin[1];  // Ray is horizontal at this Y level
  const rayX = rayOrigin[0];  // Starting X position

  let closestIntersectionX = Infinity;
  let closestEdgeIndex = -1;
  let closestEdgeParameter = 0;

  const numVertices = polygon.length;

  // Check each edge of the polygon
  for (let i = 0; i < numVertices; i++) {
    const edgeStart = polygon[i];
    const edgeEnd = polygon[(i + 1) % numVertices];

    // STEP 1: Check if this edge crosses our ray's Y level
    //
    //    edgeStart.y ●
    //                 \
    //    rayY ─────────\────────→  (ray at this Y level)
    //                   \
    //    edgeEnd.y       ●
    //
    // The edge crosses rayY if one endpoint is above and one is below

    const startAboveRay = edgeStart[1] > rayY;
    const endAboveRay = edgeEnd[1] > rayY;

    // If both endpoints are on the same side of the ray, no intersection
    if (startAboveRay === endAboveRay) {
      continue;
    }

    // STEP 2: Find WHERE on the edge the intersection occurs
    //
    // Using linear interpolation:
    //   t = (rayY - startY) / (endY - startY)
    //   intersectX = startX + t * (endX - startX)
    //
    // t is the "edge parameter" - how far along the edge (0 to 1)

    const edgeParameter = (rayY - edgeStart[1]) / (edgeEnd[1] - edgeStart[1]);
    const intersectionX = edgeStart[0] + edgeParameter * (edgeEnd[0] - edgeStart[0]);

    // STEP 3: Only consider intersections to the RIGHT of ray origin
    // (we're casting in +X direction)

    if (intersectionX <= rayX) {
      continue;  // Intersection is behind the ray origin
    }

    // STEP 4: Keep track of the CLOSEST intersection
    if (intersectionX < closestIntersectionX) {
      closestIntersectionX = intersectionX;
      closestEdgeIndex = i;
      closestEdgeParameter = edgeParameter;
    }
  }

  // No intersection found
  if (closestEdgeIndex === -1) {
    return null;
  }

  return {
    edgeStartIndex: closestEdgeIndex,
    intersectionX: closestIntersectionX,
    edgeParameter: closestEdgeParameter
  };
}

/**
 * Check if a line segment from A to B intersects with segment from C to D.
 * Uses the cross product orientation test.
 *
 * VISUAL - Intersection case:
 *
 *       A
 *        \
 *     C───\────D
 *          \
 *           B
 *
 * VISUAL - No intersection case:
 *
 *       A────B
 *
 *     C────D
 */
function doSegmentsIntersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2): boolean {
  // Helper: compute orientation of triplet (p, q, r)
  // Returns: positive if CCW, negative if CW, zero if collinear
  function orientation(p: Vec2, q: Vec2, r: Vec2): number {
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
  }

  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);

  // General case: segments intersect if orientations are different
  // (c and d are on opposite sides of line AB) AND
  // (a and b are on opposite sides of line CD)
  if (o1 * o2 < 0 && o3 * o4 < 0) {
    return true;
  }

  return false;
}

/**
 * Check if we can draw a straight line from point A to point B
 * without crossing any edge of the polygon.
 *
 * VISUAL - Visible:
 *
 *     ┌───────────┐
 *     │           │
 *     │  A ────── B  (no edges crossed)
 *     │           │
 *     └───────────┘
 *
 * VISUAL - Not visible:
 *
 *     ┌─────┬─────┐
 *     │     │     │
 *     │  A ─┼── B    (edge crossed!)
 *     │     │     │
 *     └─────┴─────┘
 *
 * @param skipVertexIndices - Don't check edges that touch these vertices
 *                            (because A or B might BE one of these vertices)
 */
function isVisible(
  a: Vec2,
  b: Vec2,
  polygon: Vec2[],
  skipVertexIndices: Set<number>
): boolean {
  const n = polygon.length;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;

    // Skip edges that include vertices we're connecting to
    if (skipVertexIndices.has(i) || skipVertexIndices.has(j)) {
      continue;
    }

    const edgeStart = polygon[i];
    const edgeEnd = polygon[j];

    // Check if our line A→B crosses this edge
    if (doSegmentsIntersect(a, b, edgeStart, edgeEnd)) {
      return false;  // Blocked!
    }
  }

  return true;  // No edges block the view
}

/**
 * Check if point P is inside triangle ABC.
 *
 * Uses the "same side" test: P is inside if it's on the same side
 * of each edge as the interior of the triangle.
 *
 * VISUAL:
 *
 *           A
 *          /\
 *         /  \
 *        / P  \     ← P is inside
 *       /      \
 *      B────────C
 */
function isPointInTriangle(a: Vec2, b: Vec2, c: Vec2, p: Vec2): boolean {
  // Compute vectors from A to other points
  const v0x = c[0] - a[0];
  const v0y = c[1] - a[1];
  const v1x = b[0] - a[0];
  const v1y = b[1] - a[1];
  const v2x = p[0] - a[0];
  const v2y = p[1] - a[1];

  // Compute dot products for barycentric coordinates
  const dot00 = v0x * v0x + v0y * v0y;
  const dot01 = v0x * v1x + v0y * v1y;
  const dot02 = v0x * v2x + v0y * v2y;
  const dot11 = v1x * v1x + v1y * v1y;
  const dot12 = v1x * v2x + v1y * v2y;

  // Compute barycentric coordinates
  const invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
  const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  // Check if point is in triangle
  return (u >= -EPSILON) && (v >= -EPSILON) && (u + v <= 1 + EPSILON);
}

/**
 * Find the vertex on the outer polygon that we should connect to
 * from the hole's rightmost vertex.
 *
 * ALGORITHM:
 *
 * 1. Cast ray from hole vertex P in +X direction
 * 2. Find where ray hits outer polygon (intersection point I)
 * 3. The edge we hit goes from vertex V1 to V2
 * 4. Pick the endpoint with larger X coordinate as candidate M
 * 5. If P can "see" M directly (no edges block), return M
 * 6. Otherwise, find a better vertex inside triangle P-I-M
 *
 * VISUAL:
 *
 *                     V1
 *                    /
 *     P ───────────●I        M = V2 (rightmost endpoint)
 *        (ray)    /  \
 *                /    \
 *               V2 ════ rest of outer
 *                ↑
 *                M (candidate - rightmost endpoint of hit edge)
 */
function findBridgeTargetVertex(
  holeVertex: Vec2,
  outerPolygon: Vec2[]
): number {
  // STEP 1: Cast ray from hole vertex
  const rayHit = castRayToPolygon(holeVertex, outerPolygon);

  if (rayHit === null) {
    // Ray cast failed - hole may be outside outer polygon (handled gracefully)
    return 0;
  }

  const { edgeStartIndex, intersectionX } = rayHit;

  // STEP 2: Get the edge endpoints
  const edgeStart = outerPolygon[edgeStartIndex];
  const edgeEnd = outerPolygon[(edgeStartIndex + 1) % outerPolygon.length];


  // STEP 3: Pick candidate M = rightmost endpoint of the hit edge
  let candidateIndex: number;
  let candidateVertex: Vec2;

  if (edgeStart[0] >= edgeEnd[0]) {
    candidateIndex = edgeStartIndex;
    candidateVertex = edgeStart;
  } else {
    candidateIndex = (edgeStartIndex + 1) % outerPolygon.length;
    candidateVertex = edgeEnd;
  }


  // STEP 4: Check if we can directly see M from P
  const skipIndices = new Set([candidateIndex]);
  // Also skip the adjacent edges
  const prevIndex = (candidateIndex - 1 + outerPolygon.length) % outerPolygon.length;
  skipIndices.add(prevIndex);

  if (isVisible(holeVertex, candidateVertex, outerPolygon, skipIndices)) {
    return candidateIndex;
  }

  // STEP 5: M is not directly visible
  // We need to find a "reflex" vertex inside triangle P-I-M
  // that IS visible from P


  const intersectionPoint: Vec2 = [intersectionX, holeVertex[1]];

  // Search all vertices to find one inside the triangle P-I-M
  // that is visible from P and has minimum angle to the ray direction

  let bestIndex = candidateIndex;  // Fallback to M
  let bestAngle = Math.PI;  // Start with worst angle

  for (let i = 0; i < outerPolygon.length; i++) {
    if (i === candidateIndex) continue;

    const vertex = outerPolygon[i];

    // Is this vertex inside triangle (P, I, M)?
    if (!isPointInTriangle(holeVertex, intersectionPoint, candidateVertex, vertex)) {
      continue;
    }

    // Is it visible from P?
    const checkSkip = new Set([i, (i - 1 + outerPolygon.length) % outerPolygon.length]);
    if (!isVisible(holeVertex, vertex, outerPolygon, checkSkip)) {
      continue;
    }

    // Compute angle from ray direction (+X) to P→vertex direction
    const dx = vertex[0] - holeVertex[0];
    const dy = vertex[1] - holeVertex[1];
    const angle = Math.abs(Math.atan2(dy, dx));

    // Keep the vertex with smallest angle (closest to ray direction)
    if (angle < bestAngle) {
      bestAngle = angle;
      bestIndex = i;
    }
  }

  return bestIndex;
}

/**
 * Merge a single hole into the outer polygon by creating a bridge.
 *
 * The bridge creates two new edges:
 *   1. outer[M] → hole[P]   (bridge INTO the hole)
 *   2. hole[P] → outer[M]   (bridge OUT of the hole)
 *
 * BEFORE:
 *
 *   Outer (CCW):  0 → 1 → 2 → 3 → 0
 *   Hole (CW):    a → d → c → b → a  (CW = indices go backwards visually)
 *
 *     0 ────→ 1
 *     ↑       ↓
 *     │ a → b │
 *     │ ↑   ↓ │     hole[P] = b (rightmost)
 *     │ d ← c │     outer[M] = 2 (found by ray cast)
 *     ↑       ↓
 *     3 ←──── 2
 *
 * AFTER (single polygon, CCW):
 *
 *     0 ────→ 1
 *     ↑       ↓
 *     │ 6 → 7 │
 *     │ ↑   ↓ │     Merged: 0→1→2→[b→c→d→a→b]→2→3→0
 *     │ 9 ← 8 │              └─────────────┘
 *     ↑   ↓   ↓                 hole traversed
 *     3 ←─5←─ 2
 *         ↑
 *     (2 appears twice: index 2 and index 10)
 *     (b appears twice: index 3 and index 7)
 *
 * The merged polygon is:
 *   [0, 1, 2, b, c, d, a, b, 2, 3]
 *            ↑           ↑
 *         bridge in   bridge out
 */
function mergeHoleIntoOuter(outer: Vec2[], hole: Vec2[]): Vec2[] {
  // STEP 1: Find rightmost vertex on the hole (this is P)
  const holeRightmostIndex = findRightmostVertexIndex(hole);
  const holeVertex = hole[holeRightmostIndex];


  // STEP 2: Find the vertex on outer to connect to (this is M)
  const outerTargetIndex = findBridgeTargetVertex(holeVertex, outer);
  const outerVertex = outer[outerTargetIndex];


  // STEP 3: Build the merged polygon
  //
  // We construct the new polygon by:
  //   Part A: outer[0] through outer[M] (inclusive)
  //   Part B: hole[P] through hole[P-1] (wrapping around, all vertices)
  //   Part C: hole[P] again (to complete the bridge)
  //   Part D: outer[M] again (to complete the bridge)
  //   Part E: outer[M+1] through outer[end]

  const merged: Vec2[] = [];
  const outerLen = outer.length;
  const holeLen = hole.length;

  // Part A: outer vertices from 0 to M (inclusive)
  for (let i = 0; i <= outerTargetIndex; i++) {
    merged.push(outer[i]);
  }

  // Part B: ALL hole vertices starting from P, going FORWARD (in array order)
  // The hole is CW. When we enter the hole from an outer CCW polygon, we must
  // traverse the hole in its CW direction to EXCLUDE the hole interior.
  // Going backward (CCW) would INCLUDE the hole interior - wrong!
  //
  // VISUAL - Why forward traversal excludes the hole:
  //
  //   Outer CCW:    ─────────→ (left side is interior)
  //                 ↑         │
  //                 │ [hole]  ↓
  //                 ←─────────
  //
  //   Hole CW:      ←─────────  (right side is interior)
  //                 │         ↑
  //                 ↓ [hole]  │
  //                 ─────────→
  //
  //   By traversing the hole CW (forward), the hole's interior stays on
  //   our RIGHT while the outer's interior stays on our LEFT.
  //   This correctly excludes the hole from the triangulation.
  //
  for (let i = 0; i < holeLen; i++) {
    const holeIndex = (holeRightmostIndex + i) % holeLen;
    merged.push(hole[holeIndex]);
  }

  // Part C: hole[P] again - back to bridge entry point
  // Note: we use the same coordinates (no epsilon offset) because:
  // 1. The 2D→3D mapping needs exact coordinates
  // 2. The ear clipping should handle collinear/coincident points
  merged.push([...hole[holeRightmostIndex]]);

  // Part D: outer[M] again - return to outer at bridge point
  merged.push([...outer[outerTargetIndex]]);

  // Part E: remaining outer vertices from M+1 to end
  for (let i = outerTargetIndex + 1; i < outerLen; i++) {
    merged.push(outer[i]);
  }


  // Debug: verify merged polygon is CCW
  const mergedArea = computeSignedArea2D(merged);
  if (mergedArea < 0) {
    merged.reverse();
  }

  return merged;
}

/**
 * Bridge ALL holes into the outer polygon.
 *
 * IMPORTANT: We process holes from RIGHT to LEFT (sorted by rightmost X).
 * This prevents bridge edges from crossing each other.
 *
 * VISUAL - Why right-to-left matters:
 *
 *   Processing left-to-right (WRONG):
 *
 *     ┌─────────────────────────┐
 *     │   ┌──┐         ┌──┐     │
 *     │   │H1│─────────│H2│     │  ← Bridge to H1 crosses bridge to H2!
 *     │   └──┘    ╳    └──┘     │
 *     └──────────────────────────┘
 *
 *   Processing right-to-left (CORRECT):
 *
 *     ┌──────────────────────────┐
 *     │   ┌──┐         ┌──┐     │
 *     │   │H1│         │H2│─────│  ← Bridge H2 first (rightmost)
 *     │   └──┘         └──┘     │
 *     └──────────────────────────┘
 *     Then bridge H1 - no crossing!
 */
function bridgeAllHoles(outer: Vec2[], holes: Vec2[][]): Vec2[] {
  if (holes.length === 0) {
    return outer;
  }


  // STEP 1: Sort holes by rightmost X coordinate (descending = right to left)
  const holesWithRightmostX = holes.map((hole, index) => {
    const rightmostIndex = findRightmostVertexIndex(hole);
    const rightmostX = hole[rightmostIndex][0];
    return { hole, index, rightmostX };
  });

  holesWithRightmostX.sort((a, b) => b.rightmostX - a.rightmostX);


  // STEP 2: Merge each hole one by one
  let currentPolygon = outer;

  for (let i = 0; i < holesWithRightmostX.length; i++) {
    const { hole } = holesWithRightmostX[i];
    currentPolygon = mergeHoleIntoOuter(currentPolygon, hole);
  }


  return currentPolygon;
}

/**
 * Debug function to verify projection sanity.
 * Checks:
 * 1. Signed area of outer loop is non-zero (valid polygon)
 * 2. All points are approximately planar (z-deviation in local coords ≈ 0)
 */
function debugVerifyProjection(
  points3d: Vec3[],
  points2d: Vec2[],
  basis: FaceBasis,
  loopName: string = "loop"
): { valid: boolean; signedArea: number; maxZDeviation: number } {
  // Check signed area
  const signedArea = computeSignedArea2D(points2d);
  const areaValid = Math.abs(signedArea) > EPSILON;

  // Check planarity: z in local coords should be ~0 for all points
  let maxZDeviation = 0;
  for (const p of points3d) {
    const d = vec3Sub(p, basis.origin);
    const z = Math.abs(vec3Dot(d, basis.n));
    if (z > maxZDeviation) {
      maxZDeviation = z;
    }
  }

  const planarityValid = maxZDeviation < 1e-6;

  if (!areaValid) {
  }
  if (!planarityValid) {
  }
  if (areaValid && planarityValid) {
  }

  return { valid: areaValid && planarityValid, signedArea, maxZDeviation };
}

export interface Mesh {
  positions: Float32Array;
  indices: Uint32Array;
  normals?: Float32Array;       // Vertex normals for smooth shading (C7.3)
  color?: ResolvedColor;        // Material color from STYLED_ITEM (C8.3) - single color fallback
  vertexColors?: Float32Array;  // Per-vertex RGB colors for multi-colored models
  // Timing information for benchmarking
  parseTime?: number;           // Time spent parsing STEP file (ms)
  triangulationTime?: number;   // Time spent in GPU triangulation (ms)
  totalTime?: number;           // Total time from start to finish (ms)
}

/** Resolved color for rendering (C8.3) */
export interface ResolvedColor {
  r: number;
  g: number;
  b: number;
}

/** Solid with color information (C8) */
export interface SolidWithColor {
  solidId: number;
  name: string;
  faceIds: number[];
  color?: ResolvedColor;
}

// --- Internal STEP structures we care about ---

interface CartesianPoint {
  id: number;
  coords: Vec3;
}

interface VertexPoint {
  id: number;
  pointId: number;
}

interface EdgeCurve {
  id: number;
  startVertexId: number;
  endVertexId: number;
  curveId: number;   // e.g. LINE id (we ignore its geometry for now)
  sameSense: boolean;
}

interface OrientedEdge {
  id: number;
  edgeElementId: number; // refers to EdgeCurve.id
  orientation: boolean;  // .T. or .F.
}

interface EdgeLoop {
  id: number;
  orientedEdgeIds: number[];
}

interface FaceBound {
  id: number;
  loopId: number;
  orientation: boolean;
  isOuter: boolean;
}

interface AdvancedFace {
  id: number;
  boundIds: number[]; // here only one: outer bound
  surfaceId: number;
  sameSense: boolean;
}

interface Direction {
  id: number;
  dir: Vec3;
}

interface Axis2Placement3D {
  id: number;
  locationId: number;  // CARTESIAN_POINT for origin
  axisId: number | null;  // DIRECTION for Z axis (normal), may be omitted
  refDirectionId: number | null;  // DIRECTION for X axis, may be omitted
}

interface Plane {
  id: number;
  placementId: number;  // AXIS2_PLACEMENT_3D
}

// =============================================================================
// C4: Curved Surface Types
// =============================================================================

interface CylindricalSurface {
  id: number;
  placementId: number;  // AXIS2_PLACEMENT_3D (axis of cylinder)
  radius: number;
}

interface SphericalSurface {
  id: number;
  placementId: number;  // AXIS2_PLACEMENT_3D (center of sphere)
  radius: number;
}

interface ConicalSurface {
  id: number;
  placementId: number;  // AXIS2_PLACEMENT_3D (apex of cone)
  radius: number;       // Base radius
  semiAngle: number;    // Half-angle in radians
}

interface ToroidalSurface {
  id: number;
  placementId: number;  // AXIS2_PLACEMENT_3D (center of torus)
  majorRadius: number;  // Distance from center to tube center
  minorRadius: number;  // Tube radius
}

/** B-spline surface with knots (C5) */
interface BSplineSurface {
  id: number;
  uDegree: number;
  vDegree: number;
  controlPointIds: number[][];  // 2D array [v][u] of CARTESIAN_POINT refs
  uKnotMultiplicities: number[];
  vKnotMultiplicities: number[];
  uKnots: number[];
  vKnots: number[];
  weights?: number[][];         // For rational B-splines (NURBS)
  uClosed: boolean;
  vClosed: boolean;
}

// =============================================================================
// C3: Curve Geometry Types
// =============================================================================

/** Vector entity (direction + magnitude) */
interface Vector {
  id: number;
  directionId: number;
  magnitude: number;
}

/** Line curve: origin + direction */
interface Line {
  id: number;
  pointId: number;      // Origin point
  vectorId: number;     // Direction vector
}

/** Circle curve: center + radius */
interface Circle {
  id: number;
  placementId: number;  // AXIS2_PLACEMENT_3D (center, normal, refDir)
  radius: number;
}

/** Ellipse curve: center + major/minor radii */
interface Ellipse {
  id: number;
  placementId: number;  // AXIS2_PLACEMENT_3D (center, normal, refDir=major axis)
  majorRadius: number;
  minorRadius: number;
}

/** B-Spline curve with knots */
interface BSplineCurve {
  id: number;
  degree: number;
  controlPointIds: number[];  // References to CARTESIAN_POINT
  knotMultiplicities: number[];
  knots: number[];
  weights?: number[];         // For rational B-splines (NURBS)
  closed: boolean;
}

/** Surface curve wrapper (references the 3D curve and PCURVEs) */
interface SurfaceCurve {
  id: number;
  curve3dId: number;      // Reference to LINE, CIRCLE, ELLIPSE, or B_SPLINE
  pcurveIds: number[];    // References to PCURVE entities
  preference: string;     // .PCURVE_S1., .PCURVE_S2., or .CURVE_3D.
}

/** PCURVE: 2D curve on a parametric surface */
interface PCurve {
  id: number;
  surfaceId: number;               // Reference to surface (PLANE, CYLINDRICAL_SURFACE, etc.)
  representationId: number;        // Reference to DEFINITIONAL_REPRESENTATION
}

/** DEFINITIONAL_REPRESENTATION: Container for 2D curve geometry */
interface DefinitionalRepresentation {
  id: number;
  curveIds: number[];     // References to 2D LINE, CIRCLE, etc.
}

// =============================================================================
// 2D Geometry Types (for PCURVE UV boundaries)
// =============================================================================

/** 2D Cartesian point (in UV parameter space) */
interface Point2D {
  id: number;
  coords: Vec2;
}

/** 2D Direction vector */
interface Direction2D {
  id: number;
  dir: Vec2;
}

/** 2D Vector (direction + magnitude) */
interface Vector2D {
  id: number;
  directionId: number;
  magnitude: number;
}

/** 2D Line (in UV parameter space) */
interface Line2D {
  id: number;
  pointId: number;      // Start point
  vectorId: number;     // Direction vector
}

/** 2D Circle (in UV parameter space) */
interface Circle2D {
  id: number;
  center: Vec2;         // Center point (resolved)
  radius: number;
}

/** 2D Axis placement */
interface Axis2Placement2D {
  id: number;
  locationId: number;
  refDirectionId: number | null;
}

// =============================================================================
// C8: Full Solids / Assemblies
// =============================================================================

/** CLOSED_SHELL: Collection of faces forming a closed solid boundary */
interface ClosedShell {
  id: number;
  name: string;
  faceIds: number[];
}

/** MANIFOLD_SOLID_BREP: Solid body containing a closed shell */
interface ManifoldSolidBrep {
  id: number;
  name: string;
  shellId: number;
}

/** BREP_WITH_VOIDS: Solid body with outer shell and void shells */
interface BrepWithVoids {
  id: number;
  name: string;
  outerShellId: number;
  voidShellIds: number[];
}

/** COLOUR_RGB: RGB color values (0-1 range) */
interface ColourRgb {
  id: number;
  name: string;
  r: number;
  g: number;
  b: number;
}

/** FILL_AREA_STYLE_COLOUR: Links a color to a fill area */
interface FillAreaStyleColour {
  id: number;
  name: string;
  colourId: number;
}

/** FILL_AREA_STYLE: Collection of fill area style colours */
interface FillAreaStyle {
  id: number;
  name: string;
  fillStyleIds: number[];
}

/** SURFACE_STYLE_FILL_AREA: Links fill area style to surface style */
interface SurfaceStyleFillArea {
  id: number;
  fillAreaStyleId: number;
}

/** SURFACE_SIDE_STYLE: Collection of surface styles */
interface SurfaceSideStyle {
  id: number;
  name: string;
  styleIds: number[];
}

/** SURFACE_STYLE_USAGE: Links surface style to a side */
interface SurfaceStyleUsage {
  id: number;
  side: string;
  styleId: number;
}

/** PRESENTATION_STYLE_ASSIGNMENT: Collection of presentation styles */
interface PresentationStyleAssignment {
  id: number;
  styleIds: number[];
}

/** STYLED_ITEM: Assigns styles to a geometric item */
interface StyledItem {
  id: number;
  name: string;
  styleIds: number[];
  itemId: number;
}

/** SHAPE_REPRESENTATION: Top-level geometry container */
interface ShapeRepresentation {
  id: number;
  name: string;
  itemIds: number[];
  contextId: number;
}

/** REPRESENTATION_RELATIONSHIP: Links two representations */
interface RepresentationRelationship {
  id: number;
  name: string;
  description: string;
  rep1Id: number;
  rep2Id: number;
  transformationId?: number; // Reference to ITEM_DEFINED_TRANSFORMATION
}

/** ITEM_DEFINED_TRANSFORMATION: Transform between representations */
interface ItemDefinedTransformation {
  id: number;
  name: string;
  description: string;
  transformItem1Id: number;
  transformItem2Id: number;
}

/** Resolved color for rendering */
export interface ResolvedColor {
  r: number;
  g: number;
  b: number;
}

/** Transform matrix (4x4) for assembly positioning */
export interface Transform {
  // 4x4 transformation matrix in column-major order
  // [m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03, m13, m23, m33]
  matrix: number[];
}

/** Apply a 4x4 transform matrix to a 3D point */
function applyTransformToPoint(point: Vec3, transform: Transform): Vec3 {
  const m = transform.matrix;
  const x = point[0];
  const y = point[1];
  const z = point[2];

  // Matrix multiplication for column-major 4x4 matrix
  // result = M * [x, y, z, 1]^T
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14]
  ];
}

/** Apply a 4x4 transform matrix to a 3D normal (rotation only, no translation) */
function applyTransformToNormal(normal: Vec3, transform: Transform): Vec3 {
  const m = transform.matrix;
  const x = normal[0];
  const y = normal[1];
  const z = normal[2];

  // For normals, only apply rotation (ignore translation)
  const result: Vec3 = [
    m[0] * x + m[4] * y + m[8] * z,
    m[1] * x + m[5] * y + m[9] * z,
    m[2] * x + m[6] * y + m[10] * z
  ];

  // Normalize the result
  const len = Math.sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
  if (len > 0) {
    result[0] /= len;
    result[1] /= len;
    result[2] /= len;
  }
  return result;
}

/** Solid with color and transform information */
export interface SolidWithColor {
  solidId: number;
  name: string;
  faceIds: number[];
  color?: ResolvedColor;
  transform?: Transform; // Assembly transform to apply to all vertices
}

/** Resolved curve geometry ready for sampling */
type CurveType = 'LINE' | 'CIRCLE' | 'ELLIPSE' | 'B_SPLINE';

interface ResolvedCircle {
  type: 'CIRCLE';
  center: Vec3;
  normal: Vec3;
  refDirection: Vec3;
  radius: number;
}

interface ResolvedEllipse {
  type: 'ELLIPSE';
  center: Vec3;
  normal: Vec3;
  refDirection: Vec3;  // Major axis direction
  majorRadius: number;
  minorRadius: number;
}

interface ResolvedBSpline {
  type: 'B_SPLINE';
  degree: number;
  controlPoints: Vec3[];
  knots: number[];       // Expanded knot vector (with multiplicities)
  weights?: number[];
  closed: boolean;
}

interface ResolvedLine {
  type: 'LINE';
  origin: Vec3;
  direction: Vec3;
}

type ResolvedCurve = ResolvedLine | ResolvedCircle | ResolvedEllipse | ResolvedBSpline;

// Simple container to hold all parsed entities
interface StepModel {
  points: Map<number, CartesianPoint>;
  vertices: Map<number, VertexPoint>;
  edgeCurves: Map<number, EdgeCurve>;
  orientedEdges: Map<number, OrientedEdge>;
  edgeLoops: Map<number, EdgeLoop>;
  faceBounds: Map<number, FaceBound>;
  faces: Map<number, AdvancedFace>;
  directions: Map<number, Direction>;
  axis2Placements: Map<number, Axis2Placement3D>;
  planes: Map<number, Plane>;
  // C4: Curved surfaces
  cylindricalSurfaces: Map<number, CylindricalSurface>;
  sphericalSurfaces: Map<number, SphericalSurface>;
  conicalSurfaces: Map<number, ConicalSurface>;
  toroidalSurfaces: Map<number, ToroidalSurface>;
  // C5: B-spline surfaces
  bSplineSurfaces: Map<number, BSplineSurface>;
  // C3: Curve geometry
  vectors: Map<number, Vector>;
  lines: Map<number, Line>;
  circles: Map<number, Circle>;
  ellipses: Map<number, Ellipse>;
  bsplines: Map<number, BSplineCurve>;
  surfaceCurves: Map<number, SurfaceCurve>;
  // C4: PCURVE support
  pcurves: Map<number, PCurve>;
  definitionalRepresentations: Map<number, DefinitionalRepresentation>;
  // 2D geometry (for PCURVE UV boundaries)
  points2d: Map<number, Point2D>;
  directions2d: Map<number, Direction2D>;
  vectors2d: Map<number, Vector2D>;
  lines2d: Map<number, Line2D>;
  circles2d: Map<number, Circle2D>;
  axis2Placements2d: Map<number, Axis2Placement2D>;
  // C8: Full solids / assemblies
  closedShells: Map<number, ClosedShell>;
  manifoldSolidBreps: Map<number, ManifoldSolidBrep>;
  styledItems: Map<number, StyledItem>;
  colourRgbs: Map<number, ColourRgb>;
  fillAreaStyleColours: Map<number, FillAreaStyleColour>;
  fillAreaStyles: Map<number, FillAreaStyle>;
  surfaceStyleFillAreas: Map<number, SurfaceStyleFillArea>;
  surfaceSideStyles: Map<number, SurfaceSideStyle>;
  surfaceStyleUsages: Map<number, SurfaceStyleUsage>;
  presentationStyleAssignments: Map<number, PresentationStyleAssignment>;
  shapeRepresentations: Map<number, ShapeRepresentation>;
  representationRelationships: Map<number, RepresentationRelationship>;
  itemDefinedTransformations: Map<number, ItemDefinedTransformation>;
}

// --- Public API: parse STEP text into a Mesh (one face) ---

/**
 * Try to tessellate a curved surface (cylinder, sphere, cone, torus).
 * Returns null if the face uses a planar surface.
 */
async function tryTessellateCurvedSurface(
  model: StepModel,
  face: AdvancedFace
): Promise<{ vertices: Vec3[]; triangles: [number, number, number][] } | null> {
  const surfaceId = face.surfaceId;

  // Extract edge loop vertices from the face to determine UV bounds
  const boundaryVertices = extractFaceBoundaryVertices(model, face);
  if (boundaryVertices.length === 0) {
    return null;
  }

  // Check for each curved surface type
  const cylinder = model.cylindricalSurfaces.get(surfaceId);
  if (cylinder) {
    const placement = getPlacementData(model, cylinder.placementId);

    // Compute face Y range for arc direction validation
    const yVals = boundaryVertices.map(v => v[1]);
    const faceYRange: [number, number] = [Math.min(...yVals), Math.max(...yVals)];

    // C6: Try UV boundary polygon approach for proper trimmed surfaces
    // Use 36 samples per edge to match planar face curve sampling (angular tolerance ~5°)
    const samplesPerEdge = 36;
    const uvBoundary = extractUVBoundaryLoop(
      model, face,
      (pt) => pointToCylinderUV(pt, placement),
      samplesPerEdge,
      faceYRange
    );

    if (uvBoundary.length >= 3) {
      // C6.4: Extract outer and hole loops separately FIRST
      // This must happen before area calculation because the combined boundary is wrong for faces with holes
      const { outer: outerLoop, holes: holeLoops } = extractUVBoundaryLoopsSeparate(
        model, face,
        (pt) => pointToCylinderUV(pt, placement),
        samplesPerEdge,
        faceYRange
      );

      // Use the OUTER loop for area calculation (not the combined boundary which includes holes)
      const loopForArea = outerLoop.length > 0 ? outerLoop : uvBoundary;

      // Fix angle wrapping for continuous UV polygon
      const fixedUV = fixUVAngleWrapping(loopForArea);

      // Compute signed area of UV polygon (using shoelace formula)
      let uvArea = 0;
      for (let i = 0; i < fixedUV.length; i++) {
        const j = (i + 1) % fixedUV.length;
        uvArea += fixedUV[i][0] * fixedUV[j][1] - fixedUV[j][0] * fixedUV[i][1];
      }
      uvArea = Math.abs(uvArea) / 2;

      // Expected area for full cylinder = 2π * height
      // For half cylinder = π * height
      let vMin = Infinity, vMax = -Infinity;
      for (const [_u, v] of fixedUV) {
        vMin = Math.min(vMin, v);
        vMax = Math.max(vMax, v);
      }
      const height = vMax - vMin;
      const expectedFullArea = Math.PI * 2 * height;
      const areaRatio = uvArea / expectedFullArea;

      // Use trimmed tessellation if area ratio suggests partial surface OR if there are holes
      const isPartialSurface = areaRatio < 0.8;

      console.log(`[Cylinder] Face ${face.id}: outerLoop.length=${outerLoop.length}, holeLoops.length=${holeLoops.length}, isPartialSurface=${isPartialSurface}, areaRatio=${areaRatio.toFixed(3)}`);

      // Debug: log UV bounds
      if (outerLoop.length > 0) {
        let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
        for (const [u, v] of outerLoop) {
          uMin = Math.min(uMin, u);
          uMax = Math.max(uMax, u);
          vMin = Math.min(vMin, v);
          vMax = Math.max(vMax, v);
        }
        console.log(`[Cylinder] Outer UV bounds: U=[${uMin.toFixed(3)}, ${uMax.toFixed(3)}], V=[${vMin.toFixed(3)}, ${vMax.toFixed(3)}]`);
        // Show first and last 5 points
        console.log(`[Cylinder] Outer loop first 5: ${outerLoop.slice(0, 5).map(([u, v]) => `(${u.toFixed(2)},${v.toFixed(2)})`).join(' ')}`);
        console.log(`[Cylinder] Outer loop last 5: ${outerLoop.slice(-5).map(([u, v]) => `(${u.toFixed(2)},${v.toFixed(2)})`).join(' ')}`);
      }
      if (holeLoops.length > 0) {
        for (let i = 0; i < holeLoops.length; i++) {
          const hole = holeLoops[i];
          let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
          for (const [u, v] of hole) {
            uMin = Math.min(uMin, u);
            uMax = Math.max(uMax, u);
            vMin = Math.min(vMin, v);
            vMax = Math.max(vMax, v);
          }
          console.log(`[Cylinder] Hole ${i} UV bounds: U=[${uMin.toFixed(3)}, ${uMax.toFixed(3)}], V=[${vMin.toFixed(3)}, ${vMax.toFixed(3)}]`);
          console.log(`[Cylinder] Hole ${i} first 5: ${hole.slice(0, 5).map(([u, v]) => `(${u.toFixed(2)},${v.toFixed(2)})`).join(' ')}`);
        }
      }

      if (isPartialSurface || holeLoops.length > 0) {
        // Use trimmed surface tessellation with actual UV boundary
        console.log(`[Cylinder] Using tessellateTrimmedSurface for Face ${face.id}`);

        // For full cylinders with holes, use a simple rectangular boundary
        // The extracted loop forms a path around the rectangle which confuses point-in-polygon
        let effectiveOuterLoop: Vec2[];
        if (!isPartialSurface && holeLoops.length > 0) {
          // Full cylinder with holes - use rectangular boundary
          // Create a simple rectangle from (0, vMin) to (2π, vMax)
          const rectPoints: Vec2[] = [];
          const numSamples = 32;
          // Bottom edge: U from 0 to 2π
          for (let i = 0; i <= numSamples; i++) {
            rectPoints.push([i * Math.PI * 2 / numSamples, vMin]);
          }
          // Right edge: V from vMin to vMax
          for (let i = 1; i <= numSamples; i++) {
            rectPoints.push([Math.PI * 2, vMin + i * (vMax - vMin) / numSamples]);
          }
          // Top edge: U from 2π to 0
          for (let i = numSamples - 1; i >= 0; i--) {
            rectPoints.push([i * Math.PI * 2 / numSamples, vMax]);
          }
          // Left edge: V from vMax to vMin (but not including the start point again)
          for (let i = numSamples - 1; i > 0; i--) {
            rectPoints.push([0, vMin + i * (vMax - vMin) / numSamples]);
          }
          effectiveOuterLoop = rectPoints;
          console.log(`[Cylinder] Using rectangular boundary for full cylinder with holes: ${rectPoints.length} points`);
        } else {
          effectiveOuterLoop = outerLoop.length > 0 ? outerLoop : uvBoundary;
        }

        const mesh = await tessellateTrimmedSurface(
          {
            type: "CYLINDRICAL_SURFACE",
            placement,
            radius: cylinder.radius,
          },
          effectiveOuterLoop,
          64,  // Grid density for tessellation (higher = less gaps at boundary)
          holeLoops  // C6.4: Pass hole loops
        );

        console.log(`[Cylinder] tessellateTrimmedSurface returned: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} triangles`);

        // Check for NaN in positions
        let nanCount = 0;
        for (let i = 0; i < mesh.positions.length; i++) {
          if (isNaN(mesh.positions[i])) nanCount++;
        }
        if (nanCount > 0) {
          console.warn(`[Cylinder] WARNING: ${nanCount} NaN values in positions!`);
        }

        // Log position bounds
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;
        let zMin = Infinity, zMax = -Infinity;
        for (let i = 0; i < mesh.positions.length; i += 3) {
          xMin = Math.min(xMin, mesh.positions[i]);
          xMax = Math.max(xMax, mesh.positions[i]);
          yMin = Math.min(yMin, mesh.positions[i + 1]);
          yMax = Math.max(yMax, mesh.positions[i + 1]);
          zMin = Math.min(zMin, mesh.positions[i + 2]);
          zMax = Math.max(zMax, mesh.positions[i + 2]);
        }
        console.log(`[Cylinder] Position bounds: X=[${xMin.toFixed(3)}, ${xMax.toFixed(3)}], Y=[${yMin.toFixed(3)}, ${yMax.toFixed(3)}], Z=[${zMin.toFixed(3)}, ${zMax.toFixed(3)}]`);

        return meshToVerticesAndTriangles(mesh);
      }
    }

    // Fall back to grid tessellation for full cylinders
    const { uMin, uMax, vMin, vMax } = computeCylinderUVBounds(
      boundaryVertices, placement
    );
    const mesh = await tessellateCylinder(
      {
        type: "CYLINDRICAL_SURFACE",
        placement,
        radius: cylinder.radius,
      },
      uMin, uMax,
      vMin, vMax,
      16
    );
    return meshToVerticesAndTriangles(mesh);
  }

  const sphere = model.sphericalSurfaces.get(surfaceId);
  if (sphere) {
    const placement = getPlacementData(model, sphere.placementId);
    const { uMin, uMax, vMin, vMax } = computeSphereUVBounds(
      boundaryVertices, placement, sphere.radius
    );
    const mesh = await tessellateSphere(
      {
        type: "SPHERICAL_SURFACE",
        placement,
        radius: sphere.radius,
      },
      uMin, uMax,
      vMin, vMax,
      16, 8
    );
    return meshToVerticesAndTriangles(mesh);
  }

  const cone = model.conicalSurfaces.get(surfaceId);
  if (cone) {
    const placement = getPlacementData(model, cone.placementId);
    const { uMin, uMax, vMin, vMax } = computeConeUVBounds(
      boundaryVertices, placement, cone.semiAngle
    );
    const mesh = await tessellateCone(
      {
        type: "CONICAL_SURFACE",
        placement,
        radius: cone.radius,
        semiAngle: cone.semiAngle,
      },
      uMin, uMax,
      vMin, vMax,
      16
    );
    return meshToVerticesAndTriangles(mesh);
  }

  const torus = model.toroidalSurfaces.get(surfaceId);
  if (torus) {
    const placement = getPlacementData(model, torus.placementId);
    const { uMin, uMax, vMin, vMax } = computeTorusUVBounds(
      boundaryVertices, placement, torus.majorRadius, torus.minorRadius
    );
    const mesh = await tessellateTorus(
      {
        type: "TOROIDAL_SURFACE",
        placement,
        majorRadius: torus.majorRadius,
        minorRadius: torus.minorRadius,
      },
      uMin, uMax,
      vMin, vMax,
      24, 12
    );
    return meshToVerticesAndTriangles(mesh);
  }

  // C5/C6b: Check for B-spline surface (with trimmed boundary support)
  const bspline = model.bSplineSurfaces.get(surfaceId);
  if (bspline) {
    // Resolve control point IDs to actual 3D coordinates
    // STEP B-spline control points are stored as [u_index][v_index]
    // But our evaluateBSplineSurface expects [v_index][u_index]
    // So we need to transpose the control point array
    const rawControlPoints: Vec3[][] = [];
    for (const row of bspline.controlPointIds) {
      const cpRow: Vec3[] = [];
      for (const cpId of row) {
        const point = model.points.get(cpId);
        if (!point) {
          return null;
        }
        cpRow.push(point.coords);
      }
      rawControlPoints.push(cpRow);
    }

    // Transpose: rawControlPoints[u][v] -> controlPoints[v][u]
    const numU = rawControlPoints.length;
    const numV = rawControlPoints[0]?.length || 0;
    const controlPoints: Vec3[][] = [];
    for (let v = 0; v < numV; v++) {
      const row: Vec3[] = [];
      for (let u = 0; u < numU; u++) {
        row.push(rawControlPoints[u][v]);
      }
      controlPoints.push(row);
    }

    // Build full knot vectors from multiplicities

    const uKnots: number[] = [];
    for (let i = 0; i < bspline.uKnots.length; i++) {
      const multiplicity = bspline.uKnotMultiplicities[i] || 1;
      for (let j = 0; j < multiplicity; j++) {
        uKnots.push(bspline.uKnots[i]);
      }
    }

    const vKnots: number[] = [];
    for (let i = 0; i < bspline.vKnots.length; i++) {
      const multiplicity = bspline.vKnotMultiplicities[i] || 1;
      for (let j = 0; j < multiplicity; j++) {
        vKnots.push(bspline.vKnots[i]);
      }
    }


    // Build the surface object for evaluation
    const surfaceObj: BSplineSurfaceType = {
      type: "B_SPLINE_SURFACE",
      controlPoints,
      uDegree: bspline.uDegree,
      vDegree: bspline.vDegree,
      uKnots,
      vKnots,
      weights: bspline.weights,
    };

    // Try to extract UV boundary from PCURVE data (avoids surface inversion)
    const uvBoundary = extractUVBoundaryFromPCurves(model, face, surfaceId, 16);

    if (uvBoundary && uvBoundary.length >= 3) {
      // Use trimmed surface tessellation with actual UV boundary
      const mesh = await tessellateTrimmedSurface(
        surfaceObj,
        uvBoundary,
        32,  // Higher grid density for B-spline surfaces
        []   // No holes for now
      );
      return meshToVerticesAndTriangles(mesh);
    }

    // Fallback: Use full rectangular tessellation (when no PCURVE data available)
    const mesh = await tessellateBSplineSurface(surfaceObj, 32, 32);
    return meshToVerticesAndTriangles(mesh);
  }

  // Not a curved surface (probably a PLANE)
  return null;
}

/**
 * Extract UV boundary from PCURVE data for a face on a specific surface.
 * This avoids the need to invert surface parameterization (Newton-Raphson).
 *
 * @param model - The STEP model
 * @param face - The face to extract UV boundary for
 * @param surfaceId - The surface ID to find PCURVEs for
 * @param samplesPerEdge - Number of samples for curved edges
 * @returns Array of UV points forming the boundary, or null if no PCURVEs found
 */
function extractUVBoundaryFromPCurves(
  model: StepModel,
  face: AdvancedFace,
  surfaceId: number,
  samplesPerEdge: number = 16
): Vec2[] | null {
  const uvPoints: Vec2[] = [];
  let foundAnyPcurve = false;

  for (const boundId of face.boundIds) {
    const bound = model.faceBounds.get(boundId);
    if (!bound) continue;

    const loop = model.edgeLoops.get(bound.loopId);
    if (!loop) continue;

    for (const orientedEdgeId of loop.orientedEdgeIds) {
      const orientedEdge = model.orientedEdges.get(orientedEdgeId);
      if (!orientedEdge) continue;

      const edgeCurve = model.edgeCurves.get(orientedEdge.edgeElementId);
      if (!edgeCurve) continue;

      // Check if this edge's curve is a SURFACE_CURVE with PCURVEs
      const surfaceCurve = model.surfaceCurves.get(edgeCurve.curveId);
      if (!surfaceCurve) {
        // Not a surface curve, can't extract PCURVE data
        continue;
      }

      // Find the PCURVE for our target surface
      let targetPcurve: PCurve | null = null;
      for (const pcurveId of surfaceCurve.pcurveIds) {
        const pcurve = model.pcurves.get(pcurveId);
        if (pcurve && pcurve.surfaceId === surfaceId) {
          targetPcurve = pcurve;
          break;
        }
      }

      if (!targetPcurve) {
        // No PCURVE for this surface on this edge
        continue;
      }

      foundAnyPcurve = true;

      // Get the 2D curve from DEFINITIONAL_REPRESENTATION
      const defRep = model.definitionalRepresentations.get(targetPcurve.representationId);
      if (!defRep || defRep.curveIds.length === 0) {
        continue;
      }

      // Sample the 2D curve
      const curveId = defRep.curveIds[0]; // Usually just one curve per def rep
      const edgeUvPoints = sample2DCurve(model, curveId, samplesPerEdge);

      // Handle edge orientation
      const effectiveOrientation = orientedEdge.orientation === edgeCurve.sameSense;
      if (!effectiveOrientation) {
        edgeUvPoints.reverse();
      }

      // Add points (skip last to avoid duplicates at edge boundaries)
      for (let i = 0; i < edgeUvPoints.length - 1; i++) {
        uvPoints.push(edgeUvPoints[i]);
      }
    }
  }

  if (!foundAnyPcurve || uvPoints.length < 3) {
    return null;
  }

  return uvPoints;
}

/**
 * Sample a 2D curve from model data.
 * Supports LINE entities with 2D points.
 */
function sample2DCurve(model: StepModel, curveId: number, numSamples: number): Vec2[] {
  const points: Vec2[] = [];

  // Check if it's a 2D line
  const line = model.lines2d.get(curveId);
  if (line) {
    const startPoint = model.points2d.get(line.pointId);
    const vector = model.vectors2d.get(line.vectorId);

    if (startPoint && vector) {
      const direction = model.directions2d.get(vector.directionId);
      if (direction) {
        // For lines, we just need start and end points
        // The parameter range is typically [0, 1] or based on vector magnitude
        const start = startPoint.coords;
        const dir = direction.dir;
        const mag = vector.magnitude;

        // Sample along the line
        for (let i = 0; i <= numSamples; i++) {
          const t = i / numSamples;
          const u = start[0] + t * dir[0] * mag;
          const v = start[1] + t * dir[1] * mag;
          points.push([u, v]);
        }
        return points;
      }
    }
  }

  // If we couldn't sample the curve, return empty
  // Could not sample curve - this may be expected for unsupported curve types
  return points;
}

/**
 * Extract all boundary vertices from a face's edge loops.
 * For circular edges (full circles), samples points along the circle.
 */
function extractFaceBoundaryVertices(model: StepModel, face: AdvancedFace): Vec3[] {
  const vertices: Vec3[] = [];

  for (const boundId of face.boundIds) {
    const bound = model.faceBounds.get(boundId);
    if (!bound) continue;

    const loop = model.edgeLoops.get(bound.loopId);
    if (!loop) continue;

    for (const orientedEdgeId of loop.orientedEdgeIds) {
      const orientedEdge = model.orientedEdges.get(orientedEdgeId);
      if (!orientedEdge) continue;

      const edgeCurve = model.edgeCurves.get(orientedEdge.edgeElementId);
      if (!edgeCurve) continue;

      // Check if this is a circular edge (full circle when start=end)
      const isFullCircle = edgeCurve.startVertexId === edgeCurve.endVertexId;

      // Try to get curve geometry for circles
      const circle = model.circles.get(edgeCurve.curveId);
      if (circle) {
        // Sample points around the circle (full or partial arc)
        const circlePlacement = model.axis2Placements.get(circle.placementId);
        if (circlePlacement) {
          const center = model.points.get(circlePlacement.locationId)?.coords || [0,0,0];
          let axis: Vec3 = [0, 0, 1];
          let refDir: Vec3 = [1, 0, 0];

          if (circlePlacement.axisId !== null) {
            const dir = model.directions.get(circlePlacement.axisId);
            if (dir) axis = dir.dir;
          }
          if (circlePlacement.refDirectionId !== null) {
            const dir = model.directions.get(circlePlacement.refDirectionId);
            if (dir) refDir = dir.dir;
          }

          const yDir = vec3Cross(axis, refDir);

          if (isFullCircle) {
            // Sample 8 points around the full circle
            for (let i = 0; i < 8; i++) {
              const angle = (i / 8) * Math.PI * 2;
              const x = center[0] + circle.radius * (Math.cos(angle) * refDir[0] + Math.sin(angle) * yDir[0]);
              const y = center[1] + circle.radius * (Math.cos(angle) * refDir[1] + Math.sin(angle) * yDir[1]);
              const z = center[2] + circle.radius * (Math.cos(angle) * refDir[2] + Math.sin(angle) * yDir[2]);
              vertices.push([x, y, z]);
            }
          } else {
            // Partial arc - get start and end points, sample between them
            const startVertex = model.vertices.get(edgeCurve.startVertexId);
            const endVertex = model.vertices.get(edgeCurve.endVertexId);

            if (startVertex && endVertex) {
              const startPt = model.points.get(startVertex.pointId)?.coords;
              const endPt = model.points.get(endVertex.pointId)?.coords;

              if (startPt && endPt) {
                // Compute angles for start and end points
                const d1: Vec3 = [startPt[0] - center[0], startPt[1] - center[1], startPt[2] - center[2]];
                const d2: Vec3 = [endPt[0] - center[0], endPt[1] - center[1], endPt[2] - center[2]];

                const angle1 = Math.atan2(vec3Dot(d1, yDir), vec3Dot(d1, refDir));
                let angle2 = Math.atan2(vec3Dot(d2, yDir), vec3Dot(d2, refDir));

                // Ensure we go the right way around
                if (angle2 < angle1) angle2 += Math.PI * 2;

                // Sample 4 points along the arc
                for (let i = 0; i <= 4; i++) {
                  const t = i / 4;
                  const angle = angle1 + t * (angle2 - angle1);
                  const x = center[0] + circle.radius * (Math.cos(angle) * refDir[0] + Math.sin(angle) * yDir[0]);
                  const y = center[1] + circle.radius * (Math.cos(angle) * refDir[1] + Math.sin(angle) * yDir[1]);
                  const z = center[2] + circle.radius * (Math.cos(angle) * refDir[2] + Math.sin(angle) * yDir[2]);
                  vertices.push([x, y, z]);
                }
              }
            }
          }
          continue;
        }
      }

      // Get start vertex
      const startVertex = model.vertices.get(edgeCurve.startVertexId);
      if (startVertex) {
        const point = model.points.get(startVertex.pointId);
        if (point) {
          vertices.push(point.coords);
        }
      }

      // Get end vertex (skip if same as start for non-circle edges)
      if (!isFullCircle) {
        const endVertex = model.vertices.get(edgeCurve.endVertexId);
        if (endVertex) {
          const point = model.points.get(endVertex.pointId);
          if (point) {
            vertices.push(point.coords);
          }
        }
      }
    }
  }

  return vertices;
}

/**
 * C6: Convert a 3D point to UV coordinates on a cylindrical surface
 */
function pointToCylinderUV(
  point: Vec3,
  placement: { location: Vec3; axis: Vec3; refDirection: Vec3 }
): Vec2 {
  const { location, axis, refDirection } = placement;
  const yDir = vec3Cross(axis, refDirection);

  const d: Vec3 = [point[0] - location[0], point[1] - location[1], point[2] - location[2]];
  const v = vec3Dot(d, axis);
  const x = vec3Dot(d, refDirection);
  const y = vec3Dot(d, yDir);
  const u = Math.atan2(y, x);

  return [u, v];
}

/**
 * C6: Convert a 3D point to UV coordinates on a spherical surface
 */
function pointToSphereUV(
  point: Vec3,
  placement: { location: Vec3; axis: Vec3; refDirection: Vec3 }
): Vec2 {
  const { location, axis, refDirection } = placement;
  const yDir = vec3Cross(axis, refDirection);

  const d: Vec3 = [point[0] - location[0], point[1] - location[1], point[2] - location[2]];
  const len = Math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
  if (len < 1e-10) return [0, 0];

  const dn: Vec3 = [d[0]/len, d[1]/len, d[2]/len];
  const sinLat = vec3Dot(dn, axis);
  const v = Math.asin(Math.max(-1, Math.min(1, sinLat)));
  const x = vec3Dot(dn, refDirection);
  const y = vec3Dot(dn, yDir);
  const u = Math.atan2(y, x);

  return [u, v];
}

/**
 * C6: Convert a 3D point to UV coordinates on a conical surface
 */
function pointToConeUV(
  point: Vec3,
  placement: { location: Vec3; axis: Vec3; refDirection: Vec3 },
  _semiAngle: number
): Vec2 {
  const { location, axis, refDirection } = placement;
  const yDir = vec3Cross(axis, refDirection);

  const d: Vec3 = [point[0] - location[0], point[1] - location[1], point[2] - location[2]];
  const v = vec3Dot(d, axis);
  const x = vec3Dot(d, refDirection);
  const y = vec3Dot(d, yDir);
  const u = Math.atan2(y, x);

  return [u, v];
}

/**
 * C6: Convert a 3D point to UV coordinates on a toroidal surface
 */
function pointToTorusUV(
  point: Vec3,
  placement: { location: Vec3; axis: Vec3; refDirection: Vec3 },
  majorRadius: number,
  minorRadius: number
): Vec2 {
  const { location, axis, refDirection } = placement;
  const yDir = vec3Cross(axis, refDirection);

  const d: Vec3 = [point[0] - location[0], point[1] - location[1], point[2] - location[2]];

  // Project onto the equatorial plane to find major angle
  const dFlat: Vec3 = [
    d[0] - vec3Dot(d, axis) * axis[0],
    d[1] - vec3Dot(d, axis) * axis[1],
    d[2] - vec3Dot(d, axis) * axis[2]
  ];
  const flatLen = Math.sqrt(dFlat[0]*dFlat[0] + dFlat[1]*dFlat[1] + dFlat[2]*dFlat[2]);

  let u = 0;
  if (flatLen > 1e-10) {
    const xProj = vec3Dot(dFlat, refDirection) / flatLen;
    const yProj = vec3Dot(dFlat, yDir) / flatLen;
    u = Math.atan2(yProj, xProj);
  }

  // Find minor angle (angle within the tube cross-section)
  const tubeCenterDist = flatLen - majorRadius;
  const tubeCenterHeight = vec3Dot(d, axis);
  const v = Math.atan2(tubeCenterHeight, tubeCenterDist);

  return [u, v];
}

/**
 * C6b: Convert a 3D point to UV coordinates on a B-spline surface
 * Uses Newton-Raphson iteration to invert the surface parameterization
 */
function pointToBSplineSurfaceUV(
  point: Vec3,
  surface: BSplineSurfaceType,
  initialGuess?: Vec2
): Vec2 {
  const maxIterations = 20;
  const tolerance = 1e-8;
  const eps = 1e-6;

  const { uKnots, vKnots, uDegree, vDegree, controlPoints } = surface;

  // Check if surface data is valid
  if (!uKnots || uKnots.length === 0 || !vKnots || vKnots.length === 0) {
    return [0.5, 0.5];
  }
  if (!controlPoints || controlPoints.length === 0 || !controlPoints[0] || controlPoints[0].length === 0) {
    return [0.5, 0.5];
  }

  // For B-spline, valid parameter range is [knots[degree], knots[numControlPoints]]
  // controlPoints is organized as [v_row][u_col], so:
  // - numU = controlPoints[0].length (columns)
  // - numV = controlPoints.length (rows)
  const numU = controlPoints[0].length;
  const numV = controlPoints.length;

  // The valid parameter range is [knots[degree], knots[n+1]] where n = numControlPoints - 1
  // which equals [knots[degree], knots[numControlPoints]]
  // But we need enough knots: knots.length must be >= numControlPoints + degree + 1
  const uMin = uKnots[uDegree];
  const uMax = uKnots.length > numU ? uKnots[numU] : uKnots[uKnots.length - 1];
  const vMin = vKnots[vDegree];
  const vMax = vKnots.length > numV ? vKnots[numV] : vKnots[vKnots.length - 1];

  // Check if UV range is valid
  if (isNaN(uMin) || isNaN(uMax) || isNaN(vMin) || isNaN(vMax) ||
      uMin === undefined || uMax === undefined || vMin === undefined || vMax === undefined) {
    return [0.5, 0.5];
  }

  let u = initialGuess ? initialGuess[0] : (uMin + uMax) / 2;
  let v = initialGuess ? initialGuess[1] : (vMin + vMax) / 2;

  for (let iter = 0; iter < maxIterations; iter++) {
    u = Math.max(uMin, Math.min(uMax - eps, u));
    v = Math.max(vMin, Math.min(vMax - eps, v));

    const p = evaluateBSplineSurface(surface, u, v);
    const dx = point[0] - p[0];
    const dy = point[1] - p[1];
    const dz = point[2] - p[2];
    const residual = dx * dx + dy * dy + dz * dz;

    if (residual < tolerance) break;

    const pu = evaluateBSplineSurface(surface, u + eps, v);
    const pv = evaluateBSplineSurface(surface, u, v + eps);

    const Su: Vec3 = [(pu[0] - p[0]) / eps, (pu[1] - p[1]) / eps, (pu[2] - p[2]) / eps];
    const Sv: Vec3 = [(pv[0] - p[0]) / eps, (pv[1] - p[1]) / eps, (pv[2] - p[2]) / eps];

    const r: Vec3 = [dx, dy, dz];
    const a11 = vec3Dot(Su, Su);
    const a12 = vec3Dot(Su, Sv);
    const a22 = vec3Dot(Sv, Sv);
    const b1 = vec3Dot(Su, r);
    const b2 = vec3Dot(Sv, r);

    const det = a11 * a22 - a12 * a12;
    if (Math.abs(det) < 1e-12) {
      u += (Math.random() - 0.5) * 0.1 * (uMax - uMin);
      v += (Math.random() - 0.5) * 0.1 * (vMax - vMin);
      continue;
    }

    const du = (b1 * a22 - b2 * a12) / det;
    const dv = (a11 * b2 - a12 * b1) / det;

    u += du * 0.8;
    v += dv * 0.8;
  }

  // Final check: verify convergence
  const finalP = evaluateBSplineSurface(surface, u, v);
  const finalDist = Math.sqrt(
    (point[0] - finalP[0]) ** 2 +
    (point[1] - finalP[1]) ** 2 +
    (point[2] - finalP[2]) ** 2
  );
  // Poor convergence is handled silently - UV will be clamped

  return [Math.max(uMin, Math.min(uMax, u)), Math.max(vMin, Math.min(vMax, v))];
}

/**
 * C6: Extract ordered UV boundary polygon from a face's edge loops
 * This properly maintains edge order for trimmed surface triangulation
 * @param faceYRange - Optional [minY, maxY] range of the face's 3D boundary for arc validation
 */
function extractUVBoundaryLoop(
  model: StepModel,
  face: AdvancedFace,
  pointToUV: (point: Vec3) => Vec2,
  samplesPerEdge: number = 8,
  faceYRange?: [number, number]
): Vec2[] {
  const uvPoints: Vec2[] = [];

  for (const boundId of face.boundIds) {
    const bound = model.faceBounds.get(boundId);
    if (!bound) continue;

    const loop = model.edgeLoops.get(bound.loopId);
    if (!loop) continue;

    for (const orientedEdgeId of loop.orientedEdgeIds) {
      const orientedEdge = model.orientedEdges.get(orientedEdgeId);
      if (!orientedEdge) continue;

      const edgeCurve = model.edgeCurves.get(orientedEdge.edgeElementId);
      if (!edgeCurve) continue;

      const edgeOrientation = orientedEdge.orientation;

      // Get edge endpoints
      const startVertexId = edgeOrientation ? edgeCurve.startVertexId : edgeCurve.endVertexId;
      const endVertexId = edgeOrientation ? edgeCurve.endVertexId : edgeCurve.startVertexId;

      const startVertex = model.vertices.get(startVertexId);
      const endVertex = model.vertices.get(endVertexId);
      if (!startVertex || !endVertex) continue;

      const startPt = model.points.get(startVertex.pointId)?.coords;
      const endPt = model.points.get(endVertex.pointId)?.coords;
      if (!startPt || !endPt) continue;

      // Check if edge is a circle
      const circle = model.circles.get(edgeCurve.curveId);
      if (circle) {
        // Check if this is a full circle (start and end are the same vertex)
        const isFullCircle = edgeCurve.startVertexId === edgeCurve.endVertexId;

        const circlePlacement = model.axis2Placements.get(circle.placementId);
        if (circlePlacement) {
          const center = model.points.get(circlePlacement.locationId)?.coords || [0, 0, 0];
          let axis: Vec3 = [0, 0, 1];
          let refDir: Vec3 = [1, 0, 0];

          if (circlePlacement.axisId !== null) {
            const dir = model.directions.get(circlePlacement.axisId);
            if (dir) axis = dir.dir;
          }
          if (circlePlacement.refDirectionId !== null) {
            const dir = model.directions.get(circlePlacement.refDirectionId);
            if (dir) refDir = dir.dir;
          }

          const yDir = vec3Cross(axis, refDir);

          // For full circles, sample around the entire circle
          if (isFullCircle) {
            for (let i = 0; i < samplesPerEdge; i++) {
              const angle = (i / samplesPerEdge) * Math.PI * 2;
              const x = center[0] + circle.radius * (Math.cos(angle) * refDir[0] + Math.sin(angle) * yDir[0]);
              const y = center[1] + circle.radius * (Math.cos(angle) * refDir[1] + Math.sin(angle) * yDir[1]);
              const z = center[2] + circle.radius * (Math.cos(angle) * refDir[2] + Math.sin(angle) * yDir[2]);
              uvPoints.push(pointToUV([x, y, z]));
            }
            continue;
          }

          // Compute start and end angles for partial arcs
          const d1: Vec3 = [startPt[0] - center[0], startPt[1] - center[1], startPt[2] - center[2]];
          const d2: Vec3 = [endPt[0] - center[0], endPt[1] - center[1], endPt[2] - center[2]];

          let angle1 = Math.atan2(vec3Dot(d1, yDir), vec3Dot(d1, refDir));
          let angle2 = Math.atan2(vec3Dot(d2, yDir), vec3Dot(d2, refDir));

          // Choose the shorter path around the circle
          let ccwDist = angle2 - angle1;
          if (ccwDist < 0) ccwDist += Math.PI * 2;
          const cwDist = Math.PI * 2 - ccwDist;

          // Initial choice: pick shorter path
          let angleSpan: number;
          if (ccwDist < cwDist) {
            angleSpan = ccwDist;
          } else if (cwDist < ccwDist) {
            angleSpan = -cwDist;
          } else {
            // Equal distance - default to CCW
            angleSpan = ccwDist;
          }

          // VALIDATION: Check if the arc midpoint is within the face's Y range
          // If we have a face Y range, validate that the arc doesn't go outside it
          const midAngle = angle1 + angleSpan / 2;
          const midPt: Vec3 = [
            center[0] + circle.radius * (Math.cos(midAngle) * refDir[0] + Math.sin(midAngle) * yDir[0]),
            center[1] + circle.radius * (Math.cos(midAngle) * refDir[1] + Math.sin(midAngle) * yDir[1]),
            center[2] + circle.radius * (Math.cos(midAngle) * refDir[2] + Math.sin(midAngle) * yDir[2])
          ];

          if (faceYRange) {
            const [faceYMin, faceYMax] = faceYRange;
            const tolerance = 1.0; // Small tolerance for numerical precision
            const midpointOutsideRange = midPt[1] < faceYMin - tolerance || midPt[1] > faceYMax + tolerance;

            if (midpointOutsideRange) {
              // Midpoint is outside face's Y range - flip the arc direction
              angleSpan = angleSpan > 0 ? -cwDist : ccwDist;
            }
          }

          // Sample along arc
          for (let i = 0; i < samplesPerEdge; i++) {
            const t = i / samplesPerEdge;
            const angle = angle1 + t * angleSpan;
            const x = center[0] + circle.radius * (Math.cos(angle) * refDir[0] + Math.sin(angle) * yDir[0]);
            const y = center[1] + circle.radius * (Math.cos(angle) * refDir[1] + Math.sin(angle) * yDir[1]);
            const z = center[2] + circle.radius * (Math.cos(angle) * refDir[2] + Math.sin(angle) * yDir[2]);
            uvPoints.push(pointToUV([x, y, z]));
          }
          continue;
        }
      }

      // C6b: Check for B-spline curve
      const bspline = model.bsplines.get(edgeCurve.curveId);
      if (bspline) {
        // Resolve control points to 3D coordinates
        const controlPoints: Vec3[] = [];
        for (const cpId of bspline.controlPointIds) {
          const pt = model.points.get(cpId);
          if (pt) {
            controlPoints.push(pt.coords);
          }
        }

        if (controlPoints.length >= 2) {
          // Sample the B-spline curve
          const samples = sampleBSplineCurve(
            controlPoints,
            bspline.knotMultiplicities,
            bspline.knots,
            bspline.degree,
            samplesPerEdge
          );

          // Reverse if edge orientation is flipped
          if (!edgeOrientation) {
            samples.reverse();
          }

          for (const pt of samples) {
            uvPoints.push(pointToUV(pt));
          }
          continue;
        }
      }

      // Check if edge is an ellipse
      const ellipse = model.ellipses.get(edgeCurve.curveId);
      if (ellipse) {
        const isFullEllipse = edgeCurve.startVertexId === edgeCurve.endVertexId;

        const ellipsePlacement = model.axis2Placements.get(ellipse.placementId);
        if (ellipsePlacement) {
          const center = model.points.get(ellipsePlacement.locationId)?.coords || [0, 0, 0];
          let axis: Vec3 = [0, 0, 1];
          let refDir: Vec3 = [1, 0, 0];

          if (ellipsePlacement.axisId !== null) {
            const dir = model.directions.get(ellipsePlacement.axisId);
            if (dir) axis = dir.dir;
          }
          if (ellipsePlacement.refDirectionId !== null) {
            const dir = model.directions.get(ellipsePlacement.refDirectionId);
            if (dir) refDir = dir.dir;
          }

          const yDir = vec3Cross(axis, refDir);
          const majorR = ellipse.majorRadius;
          const minorR = ellipse.minorRadius;

          if (isFullEllipse) {
            for (let i = 0; i < samplesPerEdge; i++) {
              const angle = (i / samplesPerEdge) * Math.PI * 2;
              const x = center[0] + majorR * Math.cos(angle) * refDir[0] + minorR * Math.sin(angle) * yDir[0];
              const y = center[1] + majorR * Math.cos(angle) * refDir[1] + minorR * Math.sin(angle) * yDir[1];
              const z = center[2] + majorR * Math.cos(angle) * refDir[2] + minorR * Math.sin(angle) * yDir[2];
              uvPoints.push(pointToUV([x, y, z]));
            }
            continue;
          }

          // Partial ellipse arc - compute angles using ellipse parametric form
          const d1: Vec3 = [startPt[0] - center[0], startPt[1] - center[1], startPt[2] - center[2]];
          const d2: Vec3 = [endPt[0] - center[0], endPt[1] - center[1], endPt[2] - center[2]];

          const x1 = vec3Dot(d1, refDir);
          const y1 = vec3Dot(d1, yDir);
          const x2 = vec3Dot(d2, refDir);
          const y2 = vec3Dot(d2, yDir);

          const angle1 = Math.atan2(y1 / minorR, x1 / majorR);
          let angle2 = Math.atan2(y2 / minorR, x2 / majorR);

          // Choose the shorter path around the ellipse
          let ccwDist = angle2 - angle1;
          if (ccwDist < 0) ccwDist += Math.PI * 2;
          const cwDist = Math.PI * 2 - ccwDist;

          let angleSpan: number;
          if (ccwDist < cwDist) {
            angleSpan = ccwDist;
          } else if (cwDist < ccwDist) {
            angleSpan = -cwDist;
          } else {
            angleSpan = ccwDist;
          }

          for (let i = 0; i < samplesPerEdge; i++) {
            const t = i / samplesPerEdge;
            const angle = angle1 + t * angleSpan;
            const x = center[0] + majorR * Math.cos(angle) * refDir[0] + minorR * Math.sin(angle) * yDir[0];
            const y = center[1] + majorR * Math.cos(angle) * refDir[1] + minorR * Math.sin(angle) * yDir[1];
            const z = center[2] + majorR * Math.cos(angle) * refDir[2] + minorR * Math.sin(angle) * yDir[2];
            uvPoints.push(pointToUV([x, y, z]));
          }
          continue;
        }
      }

      // For lines and other curves, sample linearly
      for (let i = 0; i < samplesPerEdge; i++) {
        const t = i / samplesPerEdge;
        const pt: Vec3 = [
          startPt[0] + t * (endPt[0] - startPt[0]),
          startPt[1] + t * (endPt[1] - startPt[1]),
          startPt[2] + t * (endPt[2] - startPt[2])
        ];
        uvPoints.push(pointToUV(pt));
      }
    }
  }

  return uvPoints;
}

/**
 * C6.4: Extract UV boundary loops separately for outer and holes.
 * Returns the outer boundary and an array of hole boundaries.
 * @param faceYRange - Optional [minY, maxY] range of the face's 3D boundary for arc validation
 */
function extractUVBoundaryLoopsSeparate(
  model: StepModel,
  face: AdvancedFace,
  pointToUV: (point: Vec3) => Vec2,
  samplesPerEdge: number = 8,
  faceYRange?: [number, number]
): { outer: Vec2[]; holes: Vec2[][] } {
  let outer: Vec2[] = [];
  const holes: Vec2[][] = [];

  for (const boundId of face.boundIds) {
    const bound = model.faceBounds.get(boundId);
    if (!bound) continue;

    const loop = model.edgeLoops.get(bound.loopId);
    if (!loop) continue;

    const loopPoints: Vec2[] = [];

    for (const orientedEdgeId of loop.orientedEdgeIds) {
      const orientedEdge = model.orientedEdges.get(orientedEdgeId);
      if (!orientedEdge) continue;

      const edgeCurve = model.edgeCurves.get(orientedEdge.edgeElementId);
      if (!edgeCurve) continue;

      const edgeOrientation = orientedEdge.orientation;

      // Get edge endpoints
      const startVertexId = edgeOrientation ? edgeCurve.startVertexId : edgeCurve.endVertexId;
      const endVertexId = edgeOrientation ? edgeCurve.endVertexId : edgeCurve.startVertexId;

      const startVertex = model.vertices.get(startVertexId);
      const endVertex = model.vertices.get(endVertexId);
      if (!startVertex || !endVertex) continue;

      const startPt = model.points.get(startVertex.pointId)?.coords;
      const endPt = model.points.get(endVertex.pointId)?.coords;
      if (!startPt || !endPt) continue;

      // Check if edge is a circle
      const circle = model.circles.get(edgeCurve.curveId);
      if (circle) {
        const isFullCircle = edgeCurve.startVertexId === edgeCurve.endVertexId;

        const circlePlacement = model.axis2Placements.get(circle.placementId);
        if (circlePlacement) {
          const center = model.points.get(circlePlacement.locationId)?.coords || [0, 0, 0];
          let axis: Vec3 = [0, 0, 1];
          let refDir: Vec3 = [1, 0, 0];

          if (circlePlacement.axisId !== null) {
            const dir = model.directions.get(circlePlacement.axisId);
            if (dir) axis = dir.dir;
          }
          if (circlePlacement.refDirectionId !== null) {
            const dir = model.directions.get(circlePlacement.refDirectionId);
            if (dir) refDir = dir.dir;
          }

          const yDir = vec3Cross(axis, refDir);

          if (isFullCircle) {
            // For full circles, we need to maintain angle continuity
            // The direction is determined by sameSense and edgeOrientation
            const effectiveDirection = edgeOrientation === edgeCurve.sameSense;

            for (let i = 0; i < samplesPerEdge; i++) {
              // Sample from 0 to 2π (or 2π to 0 if reversed)
              const t = i / samplesPerEdge;
              const angle = effectiveDirection ? t * Math.PI * 2 : (1 - t) * Math.PI * 2;
              const x = center[0] + circle.radius * (Math.cos(angle) * refDir[0] + Math.sin(angle) * yDir[0]);
              const y = center[1] + circle.radius * (Math.cos(angle) * refDir[1] + Math.sin(angle) * yDir[1]);
              const z = center[2] + circle.radius * (Math.cos(angle) * refDir[2] + Math.sin(angle) * yDir[2]);
              // Compute V (height along cylinder axis) the normal way
              const uv = pointToUV([x, y, z]);
              // But use the unwrapped angle for U to maintain continuity
              loopPoints.push([angle, uv[1]]);
            }
            continue;
          }

          // Partial arc
          const d1: Vec3 = [startPt[0] - center[0], startPt[1] - center[1], startPt[2] - center[2]];
          const d2: Vec3 = [endPt[0] - center[0], endPt[1] - center[1], endPt[2] - center[2]];

          let angle1 = Math.atan2(vec3Dot(d1, yDir), vec3Dot(d1, refDir));
          let angle2 = Math.atan2(vec3Dot(d2, yDir), vec3Dot(d2, refDir));

          let ccwDist = angle2 - angle1;
          if (ccwDist < 0) ccwDist += Math.PI * 2;
          const cwDist = Math.PI * 2 - ccwDist;

          // Use EDGE_CURVE sameSense to determine arc direction:
          // - sameSense=.T. -> curve parameterization agrees with edge direction (CCW for standard circles)
          // - sameSense=.F. -> curve parameterization is reversed (CW direction)
          // Combined with orientedEdge.orientation to get effective direction
          const effectiveDirection = edgeOrientation === edgeCurve.sameSense;

          let angleSpan: number;
          if (effectiveDirection) {
            // Go CCW (positive direction)
            angleSpan = ccwDist;
          } else {
            // Go CW (negative direction)
            angleSpan = -cwDist;
          }

          // Fallback validation using face Y range
          // Only apply if the chosen direction results in midpoint outside face bounds
          const midAngle = angle1 + angleSpan / 2;
          const midPt: Vec3 = [
            center[0] + circle.radius * (Math.cos(midAngle) * refDir[0] + Math.sin(midAngle) * yDir[0]),
            center[1] + circle.radius * (Math.cos(midAngle) * refDir[1] + Math.sin(midAngle) * yDir[1]),
            center[2] + circle.radius * (Math.cos(midAngle) * refDir[2] + Math.sin(midAngle) * yDir[2])
          ];

          if (faceYRange) {
            const [faceYMin, faceYMax] = faceYRange;
            const tolerance = 1.0; // Small tolerance for numerical precision
            const midpointOutsideRange = midPt[1] < faceYMin - tolerance || midPt[1] > faceYMax + tolerance;

            if (midpointOutsideRange) {
              // Midpoint is outside face's Y range - flip the arc direction
              angleSpan = angleSpan > 0 ? -cwDist : ccwDist;
            }
          }

          for (let i = 0; i < samplesPerEdge; i++) {
            const t = i / samplesPerEdge;
            const angle = angle1 + t * angleSpan;
            const x = center[0] + circle.radius * (Math.cos(angle) * refDir[0] + Math.sin(angle) * yDir[0]);
            const y = center[1] + circle.radius * (Math.cos(angle) * refDir[1] + Math.sin(angle) * yDir[1]);
            const z = center[2] + circle.radius * (Math.cos(angle) * refDir[2] + Math.sin(angle) * yDir[2]);
            loopPoints.push(pointToUV([x, y, z]));
          }
          continue;
        }
      }

      // C6b: Check for B-spline curve
      const bspline = model.bsplines.get(edgeCurve.curveId);
      if (bspline) {
        // Resolve control points to 3D coordinates
        const controlPoints: Vec3[] = [];
        for (const cpId of bspline.controlPointIds) {
          const pt = model.points.get(cpId);
          if (pt) {
            controlPoints.push(pt.coords);
          }
        }

        if (controlPoints.length >= 2) {
          // Sample the B-spline curve
          const samples = sampleBSplineCurve(
            controlPoints,
            bspline.knotMultiplicities,
            bspline.knots,
            bspline.degree,
            samplesPerEdge
          );

          // Reverse if edge orientation is flipped
          if (!edgeOrientation) {
            samples.reverse();
          }

          for (const pt of samples) {
            loopPoints.push(pointToUV(pt));
          }
          continue;
        }
      }

      // Check if edge is an ellipse
      const ellipse = model.ellipses.get(edgeCurve.curveId);
      if (ellipse) {
        const isFullEllipse = edgeCurve.startVertexId === edgeCurve.endVertexId;

        const ellipsePlacement = model.axis2Placements.get(ellipse.placementId);
        if (ellipsePlacement) {
          const center = model.points.get(ellipsePlacement.locationId)?.coords || [0, 0, 0];
          let axis: Vec3 = [0, 0, 1];
          let refDir: Vec3 = [1, 0, 0];

          if (ellipsePlacement.axisId !== null) {
            const dir = model.directions.get(ellipsePlacement.axisId);
            if (dir) axis = dir.dir;
          }
          if (ellipsePlacement.refDirectionId !== null) {
            const dir = model.directions.get(ellipsePlacement.refDirectionId);
            if (dir) refDir = dir.dir;
          }

          const yDir = vec3Cross(axis, refDir);
          const majorR = ellipse.majorRadius;
          const minorR = ellipse.minorRadius;

          if (isFullEllipse) {
            for (let i = 0; i < samplesPerEdge; i++) {
              const angle = (i / samplesPerEdge) * Math.PI * 2;
              const x = center[0] + majorR * Math.cos(angle) * refDir[0] + minorR * Math.sin(angle) * yDir[0];
              const y = center[1] + majorR * Math.cos(angle) * refDir[1] + minorR * Math.sin(angle) * yDir[1];
              const z = center[2] + majorR * Math.cos(angle) * refDir[2] + minorR * Math.sin(angle) * yDir[2];
              loopPoints.push(pointToUV([x, y, z]));
            }
            continue;
          }

          // Partial ellipse arc - compute angles using ellipse parametric form
          const d1: Vec3 = [startPt[0] - center[0], startPt[1] - center[1], startPt[2] - center[2]];
          const d2: Vec3 = [endPt[0] - center[0], endPt[1] - center[1], endPt[2] - center[2]];

          const x1 = vec3Dot(d1, refDir);
          const y1 = vec3Dot(d1, yDir);
          const x2 = vec3Dot(d2, refDir);
          const y2 = vec3Dot(d2, yDir);

          const angle1 = Math.atan2(y1 / minorR, x1 / majorR);
          let angle2 = Math.atan2(y2 / minorR, x2 / majorR);

          // Choose the shorter path around the ellipse
          let ccwDist = angle2 - angle1;
          if (ccwDist < 0) ccwDist += Math.PI * 2;
          const cwDist = Math.PI * 2 - ccwDist;

          let angleSpan: number;
          if (ccwDist < cwDist) {
            angleSpan = ccwDist;
          } else if (cwDist < ccwDist) {
            angleSpan = -cwDist;
          } else {
            angleSpan = ccwDist;
          }

          for (let i = 0; i < samplesPerEdge; i++) {
            const t = i / samplesPerEdge;
            const angle = angle1 + t * angleSpan;
            const x = center[0] + majorR * Math.cos(angle) * refDir[0] + minorR * Math.sin(angle) * yDir[0];
            const y = center[1] + majorR * Math.cos(angle) * refDir[1] + minorR * Math.sin(angle) * yDir[1];
            const z = center[2] + majorR * Math.cos(angle) * refDir[2] + minorR * Math.sin(angle) * yDir[2];
            loopPoints.push(pointToUV([x, y, z]));
          }
          continue;
        }
      }

      // For lines and other curves, sample linearly
      for (let i = 0; i < samplesPerEdge; i++) {
        const t = i / samplesPerEdge;
        const pt: Vec3 = [
          startPt[0] + t * (endPt[0] - startPt[0]),
          startPt[1] + t * (endPt[1] - startPt[1]),
          startPt[2] + t * (endPt[2] - startPt[2])
        ];
        loopPoints.push(pointToUV(pt));
      }
    }

    if (loopPoints.length >= 3) {
      if (bound.isOuter) {
        outer = loopPoints;
      } else {
        holes.push(loopPoints);
      }
    }
  }

  return { outer, holes };
}

/**
 * C6: Fix UV angles to minimize the angular span
 * This handles cases like half-cylinders where we want angles in [0, π] not [0, 2π]
 */
function fixUVAngleWrapping(uvPoints: Vec2[]): Vec2[] {
  if (uvPoints.length < 2) return uvPoints;

  // First, make angles continuous (no jumps > π)
  const continuous: Vec2[] = [[...uvPoints[0]]];
  let prevU = uvPoints[0][0];

  for (let i = 1; i < uvPoints.length; i++) {
    let u = uvPoints[i][0];
    const v = uvPoints[i][1];

    // If there's a large jump in u, adjust by ±2π
    while (u - prevU > Math.PI) u -= Math.PI * 2;
    while (prevU - u > Math.PI) u += Math.PI * 2;

    continuous.push([u, v]);
    prevU = u;
  }

  // Find the min/max of the continuous version
  let uMin = Infinity, uMax = -Infinity;
  for (const [u, _v] of continuous) {
    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
  }

  // If the span is > π, try shifting everything by π to see if we get a smaller span
  const currentSpan = uMax - uMin;
  if (currentSpan > Math.PI) {
    // Try shifting all angles by adding π (shift the range)
    const shifted: Vec2[] = continuous.map(([u, v]) => {
      let newU = u - Math.PI;
      // Normalize to [-π, π]
      while (newU < -Math.PI) newU += Math.PI * 2;
      while (newU > Math.PI) newU -= Math.PI * 2;
      return [newU, v] as Vec2;
    });

    // Make shifted version continuous too
    const shiftedContinuous: Vec2[] = [[...shifted[0]]];
    prevU = shifted[0][0];
    for (let i = 1; i < shifted.length; i++) {
      let u = shifted[i][0];
      const v = shifted[i][1];
      while (u - prevU > Math.PI) u -= Math.PI * 2;
      while (prevU - u > Math.PI) u += Math.PI * 2;
      shiftedContinuous.push([u, v]);
      prevU = u;
    }

    // Check if shifted version has smaller span
    let shiftedMin = Infinity, shiftedMax = -Infinity;
    for (const [u, _v] of shiftedContinuous) {
      shiftedMin = Math.min(shiftedMin, u);
      shiftedMax = Math.max(shiftedMax, u);
    }
    const shiftedSpan = shiftedMax - shiftedMin;

    if (shiftedSpan < currentSpan) {
      return shiftedContinuous;
    }
  }

  return continuous;
}

/**
 * Compute UV bounds for a cylinder from 3D boundary vertices
 */
function computeCylinderUVBounds(
  vertices: Vec3[],
  placement: { location: Vec3; axis: Vec3; refDirection: Vec3 }
): { uMin: number; uMax: number; vMin: number; vMax: number } {
  const { location, axis, refDirection } = placement;

  // Compute Y direction (perpendicular to axis and refDirection)
  const yDir = vec3Cross(axis, refDirection);

  let uMin = Infinity, uMax = -Infinity;
  let vMin = Infinity, vMax = -Infinity;

  for (const p of vertices) {
    // Vector from cylinder origin to point
    const d: Vec3 = [p[0] - location[0], p[1] - location[1], p[2] - location[2]];

    // v = projection onto axis (height)
    const v = vec3Dot(d, axis);

    // u = angle in the plane perpendicular to axis
    const x = vec3Dot(d, refDirection);
    const y = vec3Dot(d, yDir);
    const u = Math.atan2(y, x);

    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
    vMin = Math.min(vMin, v);
    vMax = Math.max(vMax, v);
  }

  // Handle full circle case (if angle span covers most of 2π)
  if (uMax - uMin > Math.PI * 1.6) {
    uMin = 0;
    uMax = Math.PI * 2;
  }

  return { uMin, uMax, vMin, vMax };
}

/**
 * Compute UV bounds for a sphere from 3D boundary vertices
 */
function computeSphereUVBounds(
  vertices: Vec3[],
  placement: { location: Vec3; axis: Vec3; refDirection: Vec3 },
  radius: number
): { uMin: number; uMax: number; vMin: number; vMax: number } {
  const { location, axis, refDirection } = placement;
  const yDir = vec3Cross(axis, refDirection);

  let uMin = Infinity, uMax = -Infinity;
  let vMin = Infinity, vMax = -Infinity;

  for (const p of vertices) {
    const d: Vec3 = [p[0] - location[0], p[1] - location[1], p[2] - location[2]];
    const len = Math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    if (len < 1e-10) continue;

    // Normalize
    const dn: Vec3 = [d[0]/len, d[1]/len, d[2]/len];

    // v = latitude (angle from equator)
    const sinLat = vec3Dot(dn, axis);
    const v = Math.asin(Math.max(-1, Math.min(1, sinLat)));

    // u = longitude
    const x = vec3Dot(dn, refDirection);
    const y = vec3Dot(dn, yDir);
    const u = Math.atan2(y, x);

    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
    vMin = Math.min(vMin, v);
    vMax = Math.max(vMax, v);
  }

  // Handle wraparound case
  if (uMax - uMin > Math.PI * 1.9) {
    uMin = 0;
    uMax = Math.PI * 2;
  }

  const uRange = uMax - uMin;
  const vRange = vMax - vMin;

  // Check if latitude spans pole to pole (full sphere indicator)
  const isFullLatitude = vRange > Math.PI * 0.9;

  // If latitude is full (pole-to-pole) and longitude is ~π (half sphere from boundary),
  // this is a complete sphere defined by two meridian arcs - expand to full 2π
  if (isFullLatitude && uRange > Math.PI * 0.8 && uRange < Math.PI * 1.2) {
    uMin = 0;
    uMax = Math.PI * 2;
  }

  // Also handle degenerate case where all vertices are on same meridian
  if (uRange < 0.1 && isFullLatitude) {
    uMin = 0;
    uMax = Math.PI * 2;
  }

  return { uMin, uMax, vMin, vMax };
}

/**
 * Compute UV bounds for a cone from 3D boundary vertices
 */
function computeConeUVBounds(
  vertices: Vec3[],
  placement: { location: Vec3; axis: Vec3; refDirection: Vec3 },
  semiAngle: number
): { uMin: number; uMax: number; vMin: number; vMax: number } {
  const { location, axis, refDirection } = placement;
  const yDir = vec3Cross(axis, refDirection);

  let uMin = Infinity, uMax = -Infinity;
  let vMin = Infinity, vMax = -Infinity;

  for (const p of vertices) {
    const d: Vec3 = [p[0] - location[0], p[1] - location[1], p[2] - location[2]];

    // v = projection onto axis (height from apex)
    const v = vec3Dot(d, axis);

    // u = angle around axis
    const x = vec3Dot(d, refDirection);
    const y = vec3Dot(d, yDir);
    const u = Math.atan2(y, x);

    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
    vMin = Math.min(vMin, v);
    vMax = Math.max(vMax, v);
  }

  if (uMax - uMin > Math.PI * 1.6) {
    uMin = 0;
    uMax = Math.PI * 2;
  }

  return { uMin, uMax, vMin, vMax };
}

/**
 * Compute UV bounds for a torus from 3D boundary vertices
 */
function computeTorusUVBounds(
  vertices: Vec3[],
  placement: { location: Vec3; axis: Vec3; refDirection: Vec3 },
  majorRadius: number,
  minorRadius: number
): { uMin: number; uMax: number; vMin: number; vMax: number } {
  const { location, axis, refDirection } = placement;
  const yDir = vec3Cross(axis, refDirection);

  let uMin = Infinity, uMax = -Infinity;
  let vMin = Infinity, vMax = -Infinity;

  for (const p of vertices) {
    const d: Vec3 = [p[0] - location[0], p[1] - location[1], p[2] - location[2]];

    // Project onto the XY plane (perpendicular to axis)
    const projX = vec3Dot(d, refDirection);
    const projY = vec3Dot(d, yDir);
    const projZ = vec3Dot(d, axis);

    // u = major angle (around the torus center)
    const u = Math.atan2(projY, projX);

    // Distance from axis in the XY plane
    const distFromAxis = Math.sqrt(projX*projX + projY*projY);

    // v = minor angle (around the tube)
    const dx = distFromAxis - majorRadius;
    const v = Math.atan2(projZ, dx);

    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
    vMin = Math.min(vMin, v);
    vMax = Math.max(vMax, v);
  }

  const uRange = uMax - uMin;
  const vRange = vMax - vMin;

  // Detect full torus: if range covers most of 2π (> 80%), expand to full range
  if (uRange > Math.PI * 1.6) {
    uMin = 0;
    uMax = Math.PI * 2;
  }
  if (vRange > Math.PI * 1.6) {
    vMin = 0;
    vMax = Math.PI * 2;
  }

  return { uMin, uMax, vMin, vMax };
}

/**
 * Helper to get placement data from model
 */
function getPlacementData(model: StepModel, placementId: number): {
  location: Vec3;
  axis: Vec3;
  refDirection: Vec3;
} {
  const placement = model.axis2Placements.get(placementId);
  if (!placement) {
    return {
      location: [0, 0, 0],
      axis: [0, 0, 1],
      refDirection: [1, 0, 0],
    };
  }

  const location = model.points.get(placement.locationId)?.coords || [0, 0, 0];

  let axis: Vec3 = [0, 0, 1];
  if (placement.axisId !== null) {
    const dir = model.directions.get(placement.axisId);
    if (dir) axis = dir.dir;
  }

  let refDirection: Vec3 = [1, 0, 0];
  if (placement.refDirectionId !== null) {
    const dir = model.directions.get(placement.refDirectionId);
    if (dir) refDirection = dir.dir;
  }

  return { location, axis, refDirection };
}

/**
 * Convert TessellatedMesh to vertices and triangles format.
 * Filters out degenerate triangles (zero area or high aspect ratio).
 */
function meshToVerticesAndTriangles(mesh: {
  positions: Float32Array;
  indices: Uint32Array;
}): { vertices: Vec3[]; triangles: [number, number, number][] } {
  const vertices: Vec3[] = [];
  for (let i = 0; i < mesh.positions.length; i += 3) {
    vertices.push([mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]]);
  }

  const rawTriangles: [number, number, number][] = [];
  for (let i = 0; i < mesh.indices.length; i += 3) {
    rawTriangles.push([mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]]);
  }

  // Filter out degenerate triangles (zero area, duplicate vertices, or very high aspect ratio)
  const triangles = filterDegenerateTriangles(vertices, rawTriangles, 100.0);

  return { vertices, triangles };
}

/**
 * Process a single face and return its 3D vertices and triangle indices.
 * This is a helper for parseStepToMesh that handles one ADVANCED_FACE.
 */
async function processSingleFace(
  model: StepModel,
  face: AdvancedFace,
  faceIndex: number
): Promise<{ vertices: Vec3[]; triangles: [number, number, number][] }> {
  // C4: Check if this is a curved surface and tessellate accordingly
  const curvedResult = await tryTessellateCurvedSurface(model, face);
  if (curvedResult) {
    return curvedResult;
  }

  // C2.1: Extract outer boundary and holes using helper functions
  // C3: Use async version that supports curved edges (GPU curve sampling)
  const { outer, holes } = await extractFaceBoundsWithCurves(model, face);

  // C2.2: Compute face basis and project to 2D
  const basis = computeFaceBasisFromStepFace(model, face, outer);
  const { outer2d, holes2d } = projectFaceLoopsTo2D({ outer, holes }, basis);

  // C2.3: Normalize winding order (outer=CCW, holes=CW)
  const normalized = normalizeWinding({ outer2d, holes2d });

  // Apply same winding changes to 3D points (keeps indices in sync)
  const oriented3d = applyWindingTo3D(
    { outer, holes },
    normalized.outerReversed,
    normalized.holesReversed
  );

  // C2.4: Validate topology (holes inside outer, no intersections, simple loops)
  const topology = validateTopology(normalized.outer2d, normalized.holes2d);
  if (!topology.valid) {
  }

  // C2.5: Bridge holes and triangulate
  const mergedPolygon2d = bridgeAllHoles(normalized.outer2d, normalized.holes2d);

  // Create lookup maps for fast 2D → 3D mapping
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

  // Filter out consecutive duplicate vertices before ear clipping
  const filteredMerged2d: Vec2[] = [];
  for (let i = 0; i < mergedPolygon2d.length; i++) {
    const curr = mergedPolygon2d[i];
    const prev = filteredMerged2d.length > 0
      ? filteredMerged2d[filteredMerged2d.length - 1]
      : mergedPolygon2d[mergedPolygon2d.length - 1];

    const dx = curr[0] - prev[0];
    const dy = curr[1] - prev[1];
    const distSq = dx * dx + dy * dy;

    if (distSq > 1e-12) {
      filteredMerged2d.push(curr);
    }
  }

  // Also check if last vertex equals first (wrap-around duplicate)
  if (filteredMerged2d.length > 1) {
    const first = filteredMerged2d[0];
    const last = filteredMerged2d[filteredMerged2d.length - 1];
    const dx = first[0] - last[0];
    const dy = first[1] - last[1];
    if (dx * dx + dy * dy < 1e-12) {
      filteredMerged2d.pop();
    }
  }

  // Build 3D positions array from filtered vertices
  const vertices: Vec3[] = [];
  for (const pt2d of filteredMerged2d) {
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
    vertices.push(pt3d);
  }

  // Convert 2D points to Vec3 with z=0 for ear clipping
  const filtered2dAsVec3: Vec3[] = filteredMerged2d.map(p => [p[0], p[1], 0]);

  // Run ear clipping on the filtered (bridged) polygon
  const triangles = await runEarClipping(filtered2dAsVec3) as [number, number, number][];

  return { vertices, triangles };
}

export async function parseStepToMesh(stepText: string): Promise<Mesh> {
  // Optimized version with parallel face processing and hybrid GPU/CPU triangulation.
  // Supports both planar and curved surfaces (cylinders, spheres, cones, tori).


  const totalStart = performance.now();
  const parseStart = performance.now();

  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  // Minimal stats logging
  console.log(`[parseStepToMesh] ${model.faces.size} faces, ${model.styledItems.size} styled items, ${model.colourRgbs.size} colors`);

  // C8: Use CLOSED_SHELL face ordering when available for consistent rendering
  let faces: AdvancedFace[];
  let solidColor: ResolvedColor | undefined;

  // Map from faceId -> color for per-face coloring
  const faceColorMap = new Map<number, ResolvedColor>();

  // Map from faceId -> transform for assembly positioning
  const faceTransformMap = new Map<number, Transform>();

  // Check if we have solid structure (MANIFOLD_SOLID_BREP -> CLOSED_SHELL)
  if (model.manifoldSolidBreps.size > 0) {
    const solids = extractSolidsWithColors(model);
    if (solids.length > 0) {
      // Collect faces from ALL solids and map each face to its solid's color and transform
      const allFaceIds = new Set<number>();
      let transformCount = 0;
      for (const solid of solids) {
        for (const faceId of solid.faceIds) {
          allFaceIds.add(faceId);
          // Map this face to the solid's color (can be overridden by per-face color below)
          if (solid.color) {
            faceColorMap.set(faceId, solid.color);
          }
          // Map this face to the solid's transform
          if (solid.transform) {
            faceTransformMap.set(faceId, solid.transform);
            transformCount++;
          }
        }
      }
      faces = [...allFaceIds]
        .map(id => model.faces.get(id))
        .filter((f): f is AdvancedFace => f !== undefined);

      // Use first solid's color as fallback for single-color mode
      solidColor = solids[0].color;
    } else {
      faces = [...model.faces.values()];
    }
  } else if (model.closedShells.size > 0) {
    // Fallback: collect faces from ALL closed shells
    const allFaceIds = new Set<number>();
    for (const shell of model.closedShells.values()) {
      const shellColor = resolveColorForItem(model, shell.id);
      for (const faceId of shell.faceIds) {
        allFaceIds.add(faceId);
        if (shellColor) {
          faceColorMap.set(faceId, shellColor);
        }
      }
    }
    faces = [...allFaceIds]
      .map(id => model.faces.get(id))
      .filter((f): f is AdvancedFace => f !== undefined);
    // Use first shell's color as fallback
    const firstShell = [...model.closedShells.values()][0];
    solidColor = resolveColorForItem(model, firstShell.id);
  } else {
    // Legacy: use all faces from model
    faces = [...model.faces.values()];
  }

  // Check for per-face colors (STYLED_ITEM referencing ADVANCED_FACE directly)
  let perFaceColorCount = 0;
  const facesToCheck = faces.length > 0 ? faces : [...model.faces.values()];
  for (const face of facesToCheck) {
    const faceColor = resolveColorForItem(model, face.id);
    if (faceColor) {
      faceColorMap.set(face.id, faceColor);
      perFaceColorCount++;
    }
  }
  console.log(`[parseStepToMesh] ${perFaceColorCount}/${facesToCheck.length} faces have colors`);
  const parseEnd = performance.now();
  const triangulationStart = performance.now();

  // Helper function to process a single face (runs in parallel)
  async function processFaceOptimized(face: AdvancedFace): Promise<{
    faceId: number;  // Track face ID for color lookup
    polygon2d: Vec2[];
    vertices3d: Vec3[];
    isCurved: boolean;
    curvedResult?: { vertices: Vec3[]; triangles: [number, number, number][] };
    basis?: FaceBasis;  // Store basis for normal extraction
  } | null> {
    try {
      // Check for curved surfaces first (cylinders, spheres, cones, tori)
      const curvedResult = await tryTessellateCurvedSurface(model, face);
      if (curvedResult) {
        return {
          faceId: face.id,
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
        // Validate that 2D and 3D vertex counts match
        if (filtered2d.length !== vertices3d.length) {
          return null;
        }
        return { faceId: face.id, polygon2d: filtered2d, vertices3d, isCurved: false, basis };
      }
      return null;
    } catch {
      // Face processing failed - skip this face (counted in final stats)
      return null;
    }
  }

  // Process all faces in parallel
  const results = await Promise.all(faces.map(processFaceOptimized));
  const preparedFaces = results.filter((f): f is NonNullable<typeof f> => f !== null);

  const failedCount = results.filter(f => f === null).length;
  console.log(`[parseStepToMesh] Face processing: ${preparedFaces.length} succeeded, ${failedCount} failed out of ${faces.length} total`);

  // Separate curved and planar faces
  const planarFaces = preparedFaces.filter(f => !f.isCurved);
  const curvedFaces = preparedFaces.filter(f => f.isCurved);

  // Triangulate planar polygons using hybrid GPU/CPU approach
  const allPolygons: Vec2[][] = planarFaces.map(f => f.polygon2d);
  const hybridResult = await triangulateHybrid(allPolygons);

  // Map triangulation results back to planar faces
  const planarTriangles: Map<number, number[][]> = new Map();
  for (let i = 0; i < planarFaces.length; i++) {
    planarTriangles.set(i, hybridResult.triangles[i]);
  }

  // Assemble final mesh
  const allVertices: Vec3[] = [];
  const allNormals: Vec3[] = [];  // Store normals from STEP geometry
  const allColors: Vec3[] = [];   // Per-vertex RGB colors
  const allIndices: number[] = [];
  let vertexOffset = 0;

  // Default color (light gray) for faces without color
  const defaultColor: Vec3 = [0.7, 0.7, 0.7];

  // Add curved faces (already have triangles from surface tessellation)
  console.log(`[parseStepToMesh] Processing ${curvedFaces.length} curved faces, ${planarFaces.length} planar faces`);
  for (let fi = 0; fi < curvedFaces.length; fi++) {
    const face = curvedFaces[fi];
    if (face.curvedResult) {
      const numVerts = face.curvedResult.vertices.length;
      const numTris = face.curvedResult.triangles.length;
      console.log(`[parseStepToMesh] Curved face ${face.faceId}: ${numVerts} vertices, ${numTris} triangles`);

      // Validate face mesh before adding
      let faceMaxIndex = 0;
      for (const tri of face.curvedResult.triangles) {
        faceMaxIndex = Math.max(faceMaxIndex, tri[0], tri[1], tri[2]);
      }
      if (faceMaxIndex >= numVerts) {
        console.error(`[CURVED FACE ${fi}] Invalid indices! maxIndex=${faceMaxIndex}, numVerts=${numVerts}`);
        continue; // Skip this face
      }

      // Get color for this face
      const faceColor = faceColorMap.get(face.faceId);
      const colorVec: Vec3 = faceColor
        ? [faceColor.r, faceColor.g, faceColor.b]
        : defaultColor;

      // Get transform for this face (if any)
      const faceTransform = faceTransformMap.get(face.faceId);

      for (const v of face.curvedResult.vertices) {
        // Apply assembly transform if present
        const transformedV = faceTransform ? applyTransformToPoint(v, faceTransform) : v;
        allVertices.push(transformedV);
        // For curved faces, use a placeholder normal - will be computed from geometry
        allNormals.push([0, 0, 1]);
        allColors.push(colorVec);
      }
      for (const tri of face.curvedResult.triangles) {
        allIndices.push(
          tri[0] + vertexOffset,
          tri[1] + vertexOffset,
          tri[2] + vertexOffset
        );
      }
      vertexOffset += numVerts;
    }
  }

  // Add planar faces (triangulated by hybrid approach)
  for (let i = 0; i < planarFaces.length; i++) {
    const face = planarFaces[i];
    const triangles = planarTriangles.get(i);

    if (triangles && triangles.length > 0) {
      const numVerts = face.vertices3d.length;
      console.log(`[parseStepToMesh] Planar face ${face.faceId}: ${numVerts} vertices, ${triangles.length} triangles`);

      // Validate face mesh before adding
      let faceMaxIndex = 0;
      for (const tri of triangles) {
        faceMaxIndex = Math.max(faceMaxIndex, tri[0], tri[1], tri[2]);
      }
      if (faceMaxIndex >= numVerts) {
        console.error(`[PLANAR FACE ${i}] Invalid indices! maxIndex=${faceMaxIndex}, numVerts=${numVerts}`);
        continue; // Skip this face
      }

      // Get the face normal from STEP geometry (stored in basis)
      let faceNormal: Vec3 = face.basis?.n || [0, 0, 1];

      // Get color for this face
      const faceColor = faceColorMap.get(face.faceId);
      const colorVec: Vec3 = faceColor
        ? [faceColor.r, faceColor.g, faceColor.b]
        : defaultColor;

      // Get transform for this face (if any)
      const faceTransform = faceTransformMap.get(face.faceId);

      // Transform the normal if we have a transform
      if (faceTransform) {
        faceNormal = applyTransformToNormal(faceNormal, faceTransform);
      }

      for (const v of face.vertices3d) {
        // Apply assembly transform if present
        const transformedV = faceTransform ? applyTransformToPoint(v, faceTransform) : v;
        allVertices.push(transformedV);
        allNormals.push(faceNormal);  // All vertices of this face share the same normal
        allColors.push(colorVec);
      }
      for (const tri of triangles) {
        allIndices.push(
          tri[0] + vertexOffset,
          tri[1] + vertexOffset,
          tri[2] + vertexOffset
        );
      }
      vertexOffset += numVerts;
    }
  }

  const triangulationEnd = performance.now();

  // Build final positions array
  const positions = new Float32Array(allVertices.length * 3);
  allVertices.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  const indices = new Uint32Array(allIndices);

  // Use normals from STEP geometry for planar faces
  // For curved faces (placeholder normals), compute from geometry
  const trianglesForNormals: [number, number, number][] = [];
  for (let i = 0; i < allIndices.length; i += 3) {
    trianglesForNormals.push([allIndices[i], allIndices[i + 1], allIndices[i + 2]]);
  }

  // Compute smooth normals for curved surfaces (they have placeholder [0,0,1])
  const smoothNormals = await computeSmoothNormalsGPU(allVertices, trianglesForNormals);

  // Build final normals: use STEP normals for planar faces, computed for curved
  const normals = new Float32Array(allNormals.length * 3);
  for (let i = 0; i < allNormals.length; i++) {
    const stepNormal = allNormals[i];
    // Check if this is a placeholder normal (curved face) - use computed normal instead
    if (stepNormal[0] === 0 && stepNormal[1] === 0 && stepNormal[2] === 1) {
      // This might be a curved face placeholder OR an actual Z-up face
      // Use computed normal for better results on curved surfaces
      normals[i * 3 + 0] = smoothNormals[i][0];
      normals[i * 3 + 1] = smoothNormals[i][1];
      normals[i * 3 + 2] = smoothNormals[i][2];
    } else {
      // Use the STEP-provided normal for planar faces
      normals[i * 3 + 0] = stepNormal[0];
      normals[i * 3 + 1] = stepNormal[1];
      normals[i * 3 + 2] = stepNormal[2];
    }
  }

  const totalEnd = performance.now();

  // Validate mesh indices before returning
  const numVertices = allVertices.length;
  let maxIndex = 0;
  let invalidCount = 0;
  for (let i = 0; i < allIndices.length; i++) {
    const idx = allIndices[i];
    if (idx >= numVertices) {
      invalidCount++;
      if (invalidCount <= 5) {
        console.error(`[MESH VALIDATION] Invalid index ${idx} at position ${i}, numVertices=${numVertices}`);
      }
    }
    maxIndex = Math.max(maxIndex, idx);
  }
  if (invalidCount > 0) {
    console.error(`[MESH VALIDATION] ${invalidCount} invalid indices found! maxIndex=${maxIndex}, numVertices=${numVertices}`);
  }

  // Calculate timing
  const parseTime = parseEnd - parseStart;
  const triangulationTime = triangulationEnd - triangulationStart;
  const totalTime = totalEnd - totalStart;

  // Build vertex colors array (RGB per vertex)
  const vertexColors = new Float32Array(allColors.length * 3);
  for (let i = 0; i < allColors.length; i++) {
    vertexColors[i * 3 + 0] = allColors[i][0];
    vertexColors[i * 3 + 1] = allColors[i][1];
    vertexColors[i * 3 + 2] = allColors[i][2];
  }

  console.log(`[parseStepToMesh] Built mesh with ${allVertices.length} vertices, ${allColors.length} vertex colors`);

  return { positions, indices, normals, color: solidColor, vertexColors, parseTime, triangulationTime, totalTime };
}

/**
 * Parse STEP file using single-dispatch GPU ear clipping.
 * This version runs the entire ear clipping algorithm in one GPU dispatch,
 * eliminating CPU-GPU synchronization overhead.
 */
export async function parseStepToMeshSingleDispatch(stepText: string): Promise<Mesh> {
  const totalStart = performance.now();
  const parseStart = performance.now();

  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  const face = [...model.faces.values()][0];
  const { outer, holes } = extractFaceBounds(model, face);
  const basis = computeFaceBasisFromStepFace(model, face, outer);
  const { outer2d, holes2d } = projectFaceLoopsTo2D({ outer, holes }, basis);
  const normalized = normalizeWinding({ outer2d, holes2d });
  const oriented3d = applyWindingTo3D(
    { outer, holes },
    normalized.outerReversed,
    normalized.holesReversed
  );

  const topology = validateTopology(normalized.outer2d, normalized.holes2d);
  if (!topology.valid) {
  }

  const mergedPolygon2d = bridgeAllHoles(normalized.outer2d, normalized.holes2d);

  // Create lookup maps
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
  const filteredMerged2d: Vec2[] = [];
  for (let i = 0; i < mergedPolygon2d.length; i++) {
    const curr = mergedPolygon2d[i];
    const prev = filteredMerged2d.length > 0
      ? filteredMerged2d[filteredMerged2d.length - 1]
      : mergedPolygon2d[mergedPolygon2d.length - 1];
    const dx = curr[0] - prev[0];
    const dy = curr[1] - prev[1];
    if (dx * dx + dy * dy > 1e-12) {
      filteredMerged2d.push(curr);
    }
  }

  if (filteredMerged2d.length > 1) {
    const first = filteredMerged2d[0];
    const last = filteredMerged2d[filteredMerged2d.length - 1];
    const dx = first[0] - last[0];
    const dy = first[1] - last[1];
    if (dx * dx + dy * dy < 1e-12) {
      filteredMerged2d.pop();
    }
  }

  // Build 3D positions
  const filtered3d: Vec3[] = [];
  for (const pt2d of filteredMerged2d) {
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
    filtered3d.push(pt3d);
  }

  const positions = new Float32Array(filtered3d.length * 3);
  filtered3d.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  const filtered2dAsVec3: Vec3[] = filteredMerged2d.map(p => [p[0], p[1], 0]);

  const parseEnd = performance.now();
  const triangulationStart = performance.now();

  // Use single-dispatch ear clipping
  const triangles = await earClippingSingleDispatch(filtered2dAsVec3);

  const triangulationEnd = performance.now();
  const totalEnd = performance.now();

  const indicesArray: number[] = [];
  for (const triangle of triangles) {
    indicesArray.push(triangle[0], triangle[1], triangle[2]);
  }
  const indices = new Uint32Array(indicesArray);

  const parseTime = parseEnd - parseStart;
  const triangulationTime = triangulationEnd - triangulationStart;
  const totalTime = totalEnd - totalStart;

  return { positions, indices, parseTime, triangulationTime, totalTime };
}

/**
 * Parse STEP file using optimized GPU ear clipping.
 * This version uses parallel processing within a single GPU workgroup.
 * Limited to polygons with ≤256 vertices; falls back to single-dispatch for larger ones.
 */
export async function parseStepToMeshOptimized(stepText: string): Promise<Mesh> {
  const totalStart = performance.now();
  const parseStart = performance.now();

  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  const faces = [...model.faces.values()];

  const allVertices: Vec3[] = [];
  const allIndices: number[] = [];
  let vertexOffset = 0;

  const parseEnd = performance.now();
  const triangulationStart = performance.now();

  for (let faceIndex = 0; faceIndex < faces.length; faceIndex++) {
    const face = faces[faceIndex];

    try {
      // Use optimized ear clipping algorithm
      const { vertices, triangles } = await processSingleFaceOptimized(model, face, faceIndex);

      for (const v of vertices) {
        allVertices.push(v);
      }

      for (const tri of triangles) {
        allIndices.push(
          tri[0] + vertexOffset,
          tri[1] + vertexOffset,
          tri[2] + vertexOffset
        );
      }

      vertexOffset += vertices.length;
    } catch (e) {
    }
  }

  const triangulationEnd = performance.now();

  const positions = new Float32Array(allVertices.length * 3);
  allVertices.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  const indices = new Uint32Array(allIndices);
  const totalEnd = performance.now();

  const parseTime = parseEnd - parseStart;
  const triangulationTime = triangulationEnd - triangulationStart;
  const totalTime = totalEnd - totalStart;

  const triangleCount = allIndices.length / 3;

  return { positions, indices, parseTime, triangulationTime, totalTime };
}

/**
 * Process a single face using optimized ear clipping.
 */
async function processSingleFaceOptimized(
  model: StepModel,
  face: AdvancedFace,
  faceIndex: number
): Promise<{ vertices: Vec3[]; triangles: number[][] }> {
  const { outer, holes } = extractFaceBounds(model, face);
  const basis = computeFaceBasisFromStepFace(model, face, outer);
  const { outer2d, holes2d } = projectFaceLoopsTo2D({ outer, holes }, basis);
  const normalized = normalizeWinding({ outer2d, holes2d });
  const oriented3d = applyWindingTo3D(
    { outer, holes },
    normalized.outerReversed,
    normalized.holesReversed
  );

  const topology = validateTopology(normalized.outer2d, normalized.holes2d);
  if (!topology.valid) {
  }

  const mergedPolygon2d = bridgeAllHoles(normalized.outer2d, normalized.holes2d);

  // Create lookup maps
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
  const filteredMerged2d: Vec2[] = [];
  for (let i = 0; i < mergedPolygon2d.length; i++) {
    const curr = mergedPolygon2d[i];
    const prev = filteredMerged2d.length > 0
      ? filteredMerged2d[filteredMerged2d.length - 1]
      : mergedPolygon2d[mergedPolygon2d.length - 1];
    const dx = curr[0] - prev[0];
    const dy = curr[1] - prev[1];
    if (dx * dx + dy * dy > 1e-12) {
      filteredMerged2d.push(curr);
    }
  }

  if (filteredMerged2d.length > 1) {
    const first = filteredMerged2d[0];
    const last = filteredMerged2d[filteredMerged2d.length - 1];
    const dx = first[0] - last[0];
    const dy = first[1] - last[1];
    if (dx * dx + dy * dy < 1e-12) {
      filteredMerged2d.pop();
    }
  }

  // Build 3D positions
  const vertices: Vec3[] = [];
  for (const pt2d of filteredMerged2d) {
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
    vertices.push(pt3d);
  }

  const filtered2dAsVec3: Vec3[] = filteredMerged2d.map(p => [p[0], p[1], 0]);

  // Use optimized ear clipping (falls back to single-dispatch for large polygons)
  let triangles: number[][];
  if (filtered2dAsVec3.length <= OPTIMIZED_MAX_VERTICES) {
    triangles = await earClippingOptimized(filtered2dAsVec3);
  } else {
    triangles = await earClippingSingleDispatch(filtered2dAsVec3);
  }

  return { vertices, triangles };
}

/**
 * Browser helper: take a `File` (e.g. from an `<input type="file">`) and
 * parse it into a `Mesh`. Uses the standard `File.text()` API, so it works
 * anywhere the DOM `File` type is available.
 */
export async function parseBrowserFileToMesh(file: File): Promise<Mesh> {
  const stepText = await file.text();
  return parseStepToMesh(stepText);
}

/**
 * Parse a STEP file and return the face bounds (outer boundary and holes).
 * Useful for testing the parsing logic before triangulation.
 */
export function parseStepFaceBounds(stepText: string): { outer: Vec3[]; holes: Vec3[][] } {
  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  const face = [...model.faces.values()][0];
  return extractFaceBounds(model, face);
}

/** Result of projection for testing */
export interface ProjectionTestResult {
  basis: FaceBasis;
  outer3d: Vec3[];
  outer2d: Vec2[];
  holes3d: Vec3[][];
  holes2d: Vec2[][];
  signedArea2d: number;
  maxZDeviation: number;
  usedStepPlane: boolean;
}

/** Result of winding normalization for testing */
export interface WindingTestResult {
  // Before normalization
  rawOuter2d: Vec2[];
  rawHoles2d: Vec2[][];
  rawOuterArea: number;
  rawHoleAreas: number[];
  // After normalization
  normalizedOuter2d: Vec2[];
  normalizedHoles2d: Vec2[][];
  normalizedOuterArea: number;
  normalizedHoleAreas: number[];
  // What changed
  outerReversed: boolean;
  holesReversed: boolean[];
}

/**
 * Parse a STEP file and return projection results for testing.
 * Includes basis, 2D points, and sanity check results.
 */
export function parseStepProjection(stepText: string): ProjectionTestResult {
  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  const face = [...model.faces.values()][0];
  const { outer, holes } = extractFaceBounds(model, face);

  // Check if STEP plane data is available
  const plane = model.planes.get(face.surfaceId);
  const usedStepPlane = plane !== undefined &&
    model.axis2Placements.has(plane.placementId);

  // Compute basis and project
  const basis = computeFaceBasisFromStepFace(model, face, outer);
  const { outer2d, holes2d } = projectFaceLoopsTo2D({ outer, holes }, basis);

  // Compute sanity checks
  const signedArea2d = computeSignedArea2D(outer2d);

  let maxZDeviation = 0;
  for (const p of outer) {
    const d = vec3Sub(p, basis.origin);
    const z = Math.abs(vec3Dot(d, basis.n));
    if (z > maxZDeviation) maxZDeviation = z;
  }
  for (const hole of holes) {
    for (const p of hole) {
      const d = vec3Sub(p, basis.origin);
      const z = Math.abs(vec3Dot(d, basis.n));
      if (z > maxZDeviation) maxZDeviation = z;
    }
  }

  return {
    basis,
    outer3d: outer,
    outer2d,
    holes3d: holes,
    holes2d,
    signedArea2d,
    maxZDeviation,
    usedStepPlane,
  };
}

/**
 * Parse a STEP file and return winding normalization results for testing.
 * Shows before/after state of winding normalization.
 */
export function parseStepWinding(stepText: string): WindingTestResult {
  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  const face = [...model.faces.values()][0];
  const { outer, holes } = extractFaceBounds(model, face);

  // Compute basis and project to 2D
  const basis = computeFaceBasisFromStepFace(model, face, outer);
  const { outer2d, holes2d } = projectFaceLoopsTo2D({ outer, holes }, basis);

  // Compute raw areas before normalization
  const rawOuterArea = computeSignedArea2D(outer2d);
  const rawHoleAreas = holes2d.map(h => computeSignedArea2D(h));

  // Normalize winding
  const normalized = normalizeWinding({ outer2d, holes2d });

  // Compute normalized areas
  const normalizedOuterArea = computeSignedArea2D(normalized.outer2d);
  const normalizedHoleAreas = normalized.holes2d.map(h => computeSignedArea2D(h));

  return {
    rawOuter2d: outer2d,
    rawHoles2d: holes2d,
    rawOuterArea,
    rawHoleAreas,
    normalizedOuter2d: normalized.outer2d,
    normalizedHoles2d: normalized.holes2d,
    normalizedOuterArea,
    normalizedHoleAreas,
    outerReversed: normalized.outerReversed,
    holesReversed: normalized.holesReversed,
  };
}

/**
 * Parse a STEP file and return topology validation results for testing.
 */
export function parseStepTopology(stepText: string): TopologyValidationResult & {
  outerVertexCount: number;
  holeCount: number;
} {
  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  const face = [...model.faces.values()][0];
  const { outer, holes } = extractFaceBounds(model, face);

  // Compute basis and project to 2D
  const basis = computeFaceBasisFromStepFace(model, face, outer);
  const { outer2d, holes2d } = projectFaceLoopsTo2D({ outer, holes }, basis);

  // Normalize winding
  const normalized = normalizeWinding({ outer2d, holes2d });

  // Validate topology
  const result = validateTopology(normalized.outer2d, normalized.holes2d);

  return {
    ...result,
    outerVertexCount: normalized.outer2d.length,
    holeCount: normalized.holes2d.length,
  };
}

// --- Helpers ---

function vec3Equal(a: Vec3, b: Vec3): boolean {
  return (
    Math.abs(a[0] - b[0]) < 1e-9 &&
    Math.abs(a[1] - b[1]) < 1e-9 &&
    Math.abs(a[2] - b[2]) < 1e-9
  );
}

/**
 * Extract boundary points from an edge loop.
 * Walks the oriented edges in order and collects vertex coordinates.
 */
/**
 * Extract loop points with curve sampling (async, uses GPU for curves).
 * This is the C3-enhanced version that handles CIRCLE, ELLIPSE, and B_SPLINE edges.
 */
async function extractLoopPointsWithCurves(
  model: StepModel,
  loopId: number,
  options: { angularTolerance?: number; minSamples?: number; maxSamples?: number } = {}
): Promise<Vec3[]> {
  const loop = model.edgeLoops.get(loopId);
  if (!loop) throw new Error(`EDGE_LOOP #${loopId} not found`);

  // First pass: collect edge info and identify curves
  const edgeInfos: Array<{
    startPoint: Vec3;
    endPoint: Vec3;
    curve: ResolvedCurve | null;
    reversed: boolean;
  }> = [];

  for (const orientedEdgeId of loop.orientedEdgeIds) {
    const oedge = model.orientedEdges.get(orientedEdgeId);
    if (!oedge) throw new Error(`ORIENTED_EDGE #${orientedEdgeId} not found`);

    const edgeCurve = model.edgeCurves.get(oedge.edgeElementId);
    if (!edgeCurve) throw new Error(`EDGE_CURVE #${oedge.edgeElementId} not found`);

    // Figure out start/end vertex IDs depending on orientation
    let startVertexId = edgeCurve.startVertexId;
    let endVertexId = edgeCurve.endVertexId;
    let reversed = false;

    // If orientation is false (.F.), we reverse the direction
    if (!oedge.orientation) {
      [startVertexId, endVertexId] = [endVertexId, startVertexId];
      reversed = true;
    }

    // Also consider edge curve's sameSense
    if (!edgeCurve.sameSense) {
      reversed = !reversed;
    }

    const startVertex = model.vertices.get(startVertexId);
    const endVertex = model.vertices.get(endVertexId);
    if (!startVertex || !endVertex) {
      throw new Error(`VERTEX_POINT (#${startVertexId} or #${endVertexId}) not found`);
    }

    const startPoint = model.points.get(startVertex.pointId);
    const endPoint = model.points.get(endVertex.pointId);
    if (!startPoint || !endPoint) {
      throw new Error(
        `CARTESIAN_POINT (#${startVertex.pointId} or #${endVertex.pointId}) not found`
      );
    }

    // Resolve the curve geometry
    const curve = resolveCurve(model, edgeCurve.curveId);

    edgeInfos.push({
      startPoint: startPoint.coords,
      endPoint: endPoint.coords,
      curve,
      reversed,
    });
  }

  // Collect non-line curves for GPU sampling
  const curvesToSample: ResolvedCurve[] = [];
  const curveStartPoints: Vec3[] = [];
  const curveEndPoints: Vec3[] = [];
  const curveReversed: boolean[] = [];
  const curveEdgeIndices: number[] = [];

  for (let i = 0; i < edgeInfos.length; i++) {
    const { curve, startPoint, endPoint, reversed } = edgeInfos[i];
    if (curve && curve.type !== 'LINE') {
      curvesToSample.push(curve);
      curveStartPoints.push(startPoint);
      curveEndPoints.push(endPoint);
      curveReversed.push(reversed);
      curveEdgeIndices.push(i);
    }
  }

  // Sample curves on GPU (if any)
  let sampledCurves: Vec3[][] = [];
  if (curvesToSample.length > 0) {
    // Dynamic import to avoid circular dependency
    const { sampleCurvesGPU } = await import('./curve-sampling');
    sampledCurves = await sampleCurvesGPU(
      curvesToSample,
      curveStartPoints,
      curveEndPoints,
      curveReversed,
      options
    );
    // Debug: show first few sampled points with actual values
    for (let i = 0; i < sampledCurves.length; i++) {
      const samples = sampledCurves[i];
    }
  }

  // Build boundary points with sampled curve points
  const boundaryPoints: Vec3[] = [];
  let sampledCurveIdx = 0;

  for (let i = 0; i < edgeInfos.length; i++) {
    const { startPoint, curve } = edgeInfos[i];

    // Always add start point
    boundaryPoints.push(startPoint);

    // If this edge has a curve, add intermediate sampled points
    if (curve && curve.type !== 'LINE' && sampledCurveIdx < curveEdgeIndices.length && curveEdgeIndices[sampledCurveIdx] === i) {
      const samples = sampledCurves[sampledCurveIdx];
      // Add intermediate points (skip first and last - they are start/end vertices)
      for (let j = 1; j < samples.length - 1; j++) {
        boundaryPoints.push(samples[j]);
      }
      sampledCurveIdx++;
    }
  }


  // Close the loop explicitly if needed
  const first = boundaryPoints[0];
  const last = boundaryPoints[boundaryPoints.length - 1];
  if (!vec3Equal(first, last)) {
    boundaryPoints.push(first);
  }

  // Return unique points (without the closing duplicate)
  const result = boundaryPoints.slice(0, boundaryPoints.length - 1);
  return result;
}

/**
 * Extract loop points (synchronous, straight edges only).
 * This is the legacy version for files without curves.
 */
function extractLoopPoints(model: StepModel, loopId: number): Vec3[] {
  const loop = model.edgeLoops.get(loopId);
  if (!loop) throw new Error(`EDGE_LOOP #${loopId} not found`);

  const boundaryPoints: Vec3[] = [];

  for (const orientedEdgeId of loop.orientedEdgeIds) {
    const oedge = model.orientedEdges.get(orientedEdgeId);
    if (!oedge) throw new Error(`ORIENTED_EDGE #${orientedEdgeId} not found`);

    const edgeCurve = model.edgeCurves.get(oedge.edgeElementId);
    if (!edgeCurve) throw new Error(`EDGE_CURVE #${oedge.edgeElementId} not found`);

    // Figure out start/end vertex IDs depending on orientation
    let startVertexId = edgeCurve.startVertexId;
    let endVertexId = edgeCurve.endVertexId;

    // If orientation is false (.F.), we reverse the direction
    if (!oedge.orientation) {
      [startVertexId, endVertexId] = [endVertexId, startVertexId];
    }

    const startVertex = model.vertices.get(startVertexId);
    const endVertex = model.vertices.get(endVertexId);
    if (!startVertex || !endVertex) {
      throw new Error(`VERTEX_POINT (#${startVertexId} or #${endVertexId}) not found`);
    }

    const startPoint = model.points.get(startVertex.pointId);
    const endPoint = model.points.get(endVertex.pointId);
    if (!startPoint || !endPoint) {
      throw new Error(
        `CARTESIAN_POINT (#${startVertex.pointId} or #${endVertex.pointId}) not found`
      );
    }

    // Push the start point of each edge
    boundaryPoints.push(startPoint.coords);
  }

  // Close the loop explicitly if needed
  const first = boundaryPoints[0];
  const last = boundaryPoints[boundaryPoints.length - 1];
  if (!vec3Equal(first, last)) {
    boundaryPoints.push(first);
  }

  // Return unique points (without the closing duplicate)
  return boundaryPoints.slice(0, boundaryPoints.length - 1);
}

interface FaceBoundsResult {
  outer: Vec3[];
  holes: Vec3[][];
}

/**
 * Extract the outer boundary and hole boundaries from a face.
 * Returns { outer: Vec3[], holes: Vec3[][] }
 */
function extractFaceBounds(model: StepModel, face: AdvancedFace): FaceBoundsResult {
  let outer: Vec3[] | null = null;
  const holes: Vec3[][] = [];

  for (const boundId of face.boundIds) {
    const bound = model.faceBounds.get(boundId);
    if (!bound) throw new Error(`Face bound #${boundId} not found`);

    const points = extractLoopPoints(model, bound.loopId);

    if (bound.isOuter) {
      if (outer !== null) {
        throw new Error(`Multiple outer bounds found for face #${face.id}`);
      }
      outer = points;
    } else {
      holes.push(points);
    }
  }

  if (outer === null) {
    throw new Error(`No outer bound found for face #${face.id}`);
  }

  return { outer, holes };
}

/**
 * Extract face bounds with curve sampling (async, uses GPU).
 * This is the C3-enhanced version for files with curved edges.
 */
async function extractFaceBoundsWithCurves(
  model: StepModel,
  face: AdvancedFace,
  options: { angularTolerance?: number; minSamples?: number; maxSamples?: number } = {}
): Promise<FaceBoundsResult> {
  let outer: Vec3[] | null = null;
  const holes: Vec3[][] = [];
  const allBounds: { points: Vec3[]; isOuter: boolean }[] = [];

  for (const boundId of face.boundIds) {
    const bound = model.faceBounds.get(boundId);
    if (!bound) throw new Error(`Face bound #${boundId} not found`);

    const points = await extractLoopPointsWithCurves(model, bound.loopId, options);
    allBounds.push({ points, isOuter: bound.isOuter });

    if (bound.isOuter) {
      if (outer !== null) {
        throw new Error(`Multiple outer bounds found for face #${face.id}`);
      }
      outer = points;
    } else {
      holes.push(points);
    }
  }

  // If no explicit outer bound, determine it heuristically
  if (outer === null) {
    if (allBounds.length === 1) {
      // Single bound - it's the outer
      outer = allBounds[0].points;
      holes.length = 0; // Clear holes array
    } else if (allBounds.length > 1) {
      // Multiple bounds - use largest area as outer
      // Project to 2D and compute signed areas
      let maxArea = -Infinity;
      let outerIdx = 0;
      for (let i = 0; i < allBounds.length; i++) {
        const pts = allBounds[i].points;
        // Simple 2D area calculation (use x,y coordinates)
        let area = 0;
        for (let j = 0; j < pts.length; j++) {
          const curr = pts[j];
          const next = pts[(j + 1) % pts.length];
          area += curr[0] * next[1] - next[0] * curr[1];
        }
        area = Math.abs(area / 2);
        if (area > maxArea) {
          maxArea = area;
          outerIdx = i;
        }
      }
      outer = allBounds[outerIdx].points;
      holes.length = 0;
      for (let i = 0; i < allBounds.length; i++) {
        if (i !== outerIdx) {
          holes.push(allBounds[i].points);
        }
      }
    } else {
      throw new Error(`No bounds found for face #${face.id}`);
    }
  }

  return { outer, holes };
}

// --- STEP parsing (very constrained to our example) ---

function parseStep(stepText: string): StepModel {
  const model: StepModel = {
    points: new Map(),
    vertices: new Map(),
    edgeCurves: new Map(),
    orientedEdges: new Map(),
    edgeLoops: new Map(),
    faceBounds: new Map(),
    faces: new Map(),
    directions: new Map(),
    axis2Placements: new Map(),
    planes: new Map(),
    // C4: Curved surfaces
    cylindricalSurfaces: new Map(),
    sphericalSurfaces: new Map(),
    conicalSurfaces: new Map(),
    toroidalSurfaces: new Map(),
    // C5: B-spline surfaces
    bSplineSurfaces: new Map(),
    // C3: Curve geometry
    vectors: new Map(),
    lines: new Map(),
    circles: new Map(),
    ellipses: new Map(),
    bsplines: new Map(),
    surfaceCurves: new Map(),
    // C4: PCURVE support
    pcurves: new Map(),
    definitionalRepresentations: new Map(),
    // 2D geometry (for PCURVE UV boundaries)
    points2d: new Map(),
    directions2d: new Map(),
    vectors2d: new Map(),
    lines2d: new Map(),
    circles2d: new Map(),
    axis2Placements2d: new Map(),
    // C8: Full solids / assemblies
    closedShells: new Map(),
    manifoldSolidBreps: new Map(),
    brepWithVoids: new Map<number, BrepWithVoids>(),
    styledItems: new Map(),
    colourRgbs: new Map(),
    fillAreaStyleColours: new Map(),
    fillAreaStyles: new Map(),
    surfaceStyleFillAreas: new Map(),
    surfaceSideStyles: new Map(),
    surfaceStyleUsages: new Map(),
    presentationStyleAssignments: new Map(),
    shapeRepresentations: new Map(),
    representationRelationships: new Map(),
    itemDefinedTransformations: new Map(),
  };

  // Remove comments (/* ... */, / ... */, and -- ... end-of-line)
  let text = stepText.replace(/\/\*[\s\S]*?\*\//g, "");     // block comments /* ... */
  text = text.replace(/\/[^*][\s\S]*?\*\//g, "");          // block comments / ... */ (single slash)
  text = text.replace(/--.*$/gm, "");                       // line comments

  // Join multi-line entities: STEP entities can span multiple lines, ending with semicolon
  // Normalize by removing newlines and collapsing multiple spaces
  const rawLines = text.split(/\r?\n/);
  const entities: string[] = [];
  let currentEntity = "";

  // Regex to detect start of a new entity: #number= or #number =
  // This distinguishes entity starts from continuation lines containing references like #123,#456
  const entityStartRegex = /^#\d+\s*=/;

  for (const line of rawLines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // Check if this line starts a new entity (e.g., "#123=TYPE(...)" or "#123 = TYPE(...)")
    // Not just "#..." which could be continuation lines like "#456,#789,..."
    if (entityStartRegex.test(trimmed)) {
      // Start of a new entity - save previous if exists
      if (currentEntity) {
        entities.push(currentEntity);
      }
      currentEntity = trimmed;
    } else if (currentEntity) {
      // Continuation of current entity
      currentEntity += " " + trimmed;
    }

    // Check if current entity is complete (ends with semicolon)
    if (currentEntity && currentEntity.endsWith(";")) {
      entities.push(currentEntity);
      currentEntity = "";
    }
  }
  // Add any remaining entity
  if (currentEntity) {
    entities.push(currentEntity);
  }

  // Regex updated to handle:
  // - Spaces around '=' sign
  // - Spaces between entity type and opening paren: CARTESIAN_POINT ( 'NONE', ...)
  // - Spaces before semicolon: ...) ) ;
  const entityRegex = /^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(\s*(.*)\s*\)\s*;?\s*$/;
  // Complex entity regex: #id=( TYPE1(...) TYPE2(...) ... );
  const complexEntityRegex = /^#(\d+)\s*=\s*\(\s*([\s\S]*)\s*\)\s*;?\s*$/;

  for (const entity of entities) {
    const trimmed = entity.trim();

    if (!trimmed.startsWith("#")) continue;

    // Try simple entity format first
    const match = trimmed.match(entityRegex);
    if (match) {
      const id = parseInt(match[1], 10);
      const type = match[2];
      const args = match[3]; // raw argument string inside (...)
      parseSimpleEntity(id, type, args, model);
      continue;
    }

    // Try complex entity format: #id=( TYPE1(...) TYPE2(...) ... );
    const complexMatch = trimmed.match(complexEntityRegex);
    if (complexMatch) {
      const id = parseInt(complexMatch[1], 10);
      const content = complexMatch[2];
      parseComplexEntity(id, content, model);
      continue;
    }
  }

  // Post-processing: Resolve ORIENTED_CLOSED_SHELL references
  // These shells reference other closed shells and need to copy their face IDs
  for (const shell of model.closedShells.values()) {
    const shellWithRef = shell as ClosedShell & { _referencedShellId?: number; _isReversed?: boolean };
    if (shellWithRef._referencedShellId !== undefined) {
      const referencedShell = model.closedShells.get(shellWithRef._referencedShellId);
      if (referencedShell) {
        // Copy face IDs from the referenced shell
        shellWithRef.faceIds = [...referencedShell.faceIds];
      }
    }
  }

  return model;
}

/**
 * Parse a complex STEP entity that combines multiple types.
 * Format: #id=( TYPE1(...) TYPE2(...) ... );
 *
 * Example for rational B-spline surface:
 * #37682=(
 *   BOUNDED_SURFACE()
 *   B_SPLINE_SURFACE(3,2,((control_points...)),...)
 *   B_SPLINE_SURFACE_WITH_KNOTS((u_mults),(v_mults),(u_knots),(v_knots),...)
 *   RATIONAL_B_SPLINE_SURFACE((weights))
 *   ...
 * );
 */
function parseComplexEntity(id: number, content: string, model: StepModel): void {
  // Extract all sub-entity types and their arguments
  // Pattern: TYPE_NAME(...) or TYPE_NAME()
  const subEntityRegex = /([A-Z][A-Z0-9_]*)\s*\(([^()]*(?:\([^()]*(?:\([^()]*\)[^()]*)*\)[^()]*)*)\)/g;

  const subEntities: { type: string; args: string }[] = [];
  let subMatch;
  while ((subMatch = subEntityRegex.exec(content)) !== null) {
    subEntities.push({
      type: subMatch[1],
      args: subMatch[2].trim()
    });
  }

  if (subEntities.length === 0) {
    return;
  }

  // Check what types are present
  const types = new Set(subEntities.map(e => e.type));

  // Handle complex B-spline surface (combines B_SPLINE_SURFACE + B_SPLINE_SURFACE_WITH_KNOTS + optional RATIONAL)
  if (types.has('B_SPLINE_SURFACE') && types.has('B_SPLINE_SURFACE_WITH_KNOTS')) {
    parseComplexBSplineSurface(id, subEntities, model);
    return;
  }

  // Handle REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION (assembly transforms)
  // Format: REPRESENTATION_RELATIONSHIP(' ',' ',#rep1,#rep2) REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#transform) SHAPE_REPRESENTATION_RELATIONSHIP()
  if (types.has('REPRESENTATION_RELATIONSHIP') && types.has('REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION')) {
    parseComplexRepresentationRelationship(id, subEntities, model);
    return;
  }

  // Handle other complex entity types as needed
  // For now, just try to parse any recognizable sub-entities
  for (const sub of subEntities) {
    // Some entities might be parseable directly
    switch (sub.type) {
      case 'REPRESENTATION_ITEM':
      case 'GEOMETRIC_REPRESENTATION_ITEM':
      case 'SURFACE':
      case 'BOUNDED_SURFACE':
        // These are abstract supertypes, skip them
        break;
      default:
        // Could add more handlers here
        break;
    }
  }
}

/**
 * Parse a complex B-spline surface entity that combines:
 * - B_SPLINE_SURFACE: degrees and control points
 * - B_SPLINE_SURFACE_WITH_KNOTS: knot vectors and multiplicities
 * - RATIONAL_B_SPLINE_SURFACE: weights (optional)
 */
function parseComplexBSplineSurface(
  id: number,
  subEntities: { type: string; args: string }[],
  model: StepModel
): void {
  let uDegree = 0;
  let vDegree = 0;
  let controlPointIds: number[][] = [];
  let uKnotMultiplicities: number[] = [];
  let vKnotMultiplicities: number[] = [];
  let uKnots: number[] = [];
  let vKnots: number[] = [];
  let weights: number[][] | undefined;
  let uClosed = false;
  let vClosed = false;

  for (const sub of subEntities) {
    if (sub.type === 'B_SPLINE_SURFACE') {
      // B_SPLINE_SURFACE(u_degree, v_degree, control_points, surface_form, u_closed, v_closed, self_intersect)
      // Example: B_SPLINE_SURFACE(3,2,((#cp1,#cp2),(#cp3,#cp4)),.UNSPECIFIED.,.F.,.F.,.F.)

      const degreeMatch = sub.args.match(/^(\d+)\s*,\s*(\d+)\s*,/);
      if (degreeMatch) {
        uDegree = parseInt(degreeMatch[1], 10);
        vDegree = parseInt(degreeMatch[2], 10);
      }

      // Extract control points array
      const cpStart = sub.args.indexOf('((');
      if (cpStart >= 0) {
        // Find matching closing ))
        let depth = 0;
        let cpEnd = cpStart;
        for (let i = cpStart; i < sub.args.length; i++) {
          if (sub.args[i] === '(') depth++;
          else if (sub.args[i] === ')') {
            depth--;
            if (depth === 0) {
              cpEnd = i + 1;
              break;
            }
          }
        }

        const cpArrayStr = sub.args.substring(cpStart, cpEnd);
        // Parse rows: (#id1,#id2,...),(#id3,#id4,...)
        const rowRegex = /\(\s*(#[\d\s,#]+)\s*\)/g;
        let rowMatch;
        while ((rowMatch = rowRegex.exec(cpArrayStr)) !== null) {
          const rowStr = rowMatch[1];
          const pointIds: number[] = [];
          const pointRegex = /#(\d+)/g;
          let pointMatch;
          while ((pointMatch = pointRegex.exec(rowStr)) !== null) {
            pointIds.push(parseInt(pointMatch[1], 10));
          }
          if (pointIds.length > 0) {
            controlPointIds.push(pointIds);
          }
        }
      }

      // Parse closed flags
      const flagsMatch = sub.args.match(/\.\w+\.\s*,\s*\.([TF])\.\s*,\s*\.([TF])\.\s*,\s*\.([TF])\./);
      if (flagsMatch) {
        uClosed = flagsMatch[1] === 'T';
        vClosed = flagsMatch[2] === 'T';
      }
    }
    else if (sub.type === 'B_SPLINE_SURFACE_WITH_KNOTS') {
      // B_SPLINE_SURFACE_WITH_KNOTS(u_multiplicities, v_multiplicities, u_knots, v_knots, knot_spec)
      // Example: B_SPLINE_SURFACE_WITH_KNOTS((4,1,1,4),(3,3),(0.,0.5,0.75,1.),(0.,1.),.UNSPECIFIED.)

      // Find all parenthesized lists
      const listRegex = /\(\s*([-0-9.Ee+\s,]+)\s*\)/g;
      const allLists: { nums: number[], isFloat: boolean }[] = [];
      let listMatch;
      while ((listMatch = listRegex.exec(sub.args)) !== null) {
        const content = listMatch[1];
        const nums = content.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
        if (nums.length > 0) {
          const isFloat = content.includes('.');
          allLists.push({ nums, isFloat });
        }
      }

      // Separate into integer lists (multiplicities) and float lists (knots)
      const intLists = allLists.filter(l => !l.isFloat).map(l => l.nums.map(n => Math.round(n)));
      const realLists = allLists.filter(l => l.isFloat).map(l => l.nums);

      uKnotMultiplicities = intLists[0] || [];
      vKnotMultiplicities = intLists[1] || [];
      uKnots = realLists[0] || [];
      vKnots = realLists[1] || [];
    }
    else if (sub.type === 'RATIONAL_B_SPLINE_SURFACE') {
      // RATIONAL_B_SPLINE_SURFACE(weights_array)
      // Example: RATIONAL_B_SPLINE_SURFACE(((1.,0.707,1.),(1.,0.707,1.)))

      const weightsStart = sub.args.indexOf('((');
      if (weightsStart >= 0) {
        // Find matching ))
        let depth = 0;
        let weightsEnd = weightsStart;
        for (let i = weightsStart; i < sub.args.length; i++) {
          if (sub.args[i] === '(') depth++;
          else if (sub.args[i] === ')') {
            depth--;
            if (depth === 0) {
              weightsEnd = i + 1;
              break;
            }
          }
        }

        const weightsStr = sub.args.substring(weightsStart, weightsEnd);
        weights = [];

        // Parse rows of weights
        const rowRegex = /\(\s*([-0-9.Ee+\s,]+)\s*\)/g;
        let rowMatch;
        while ((rowMatch = rowRegex.exec(weightsStr)) !== null) {
          const rowStr = rowMatch[1];
          const rowWeights = rowStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
          if (rowWeights.length > 0) {
            weights.push(rowWeights);
          }
        }
      }
    }
  }

  // Only store if we got valid data
  if (controlPointIds.length > 0 && uKnots.length > 0 && vKnots.length > 0) {
    model.bSplineSurfaces.set(id, {
      id,
      uDegree,
      vDegree,
      controlPointIds,
      uKnotMultiplicities,
      vKnotMultiplicities,
      uKnots,
      vKnots,
      uClosed,
      vClosed,
      weights,
    });
  }
}

/**
 * Parse a complex REPRESENTATION_RELATIONSHIP with TRANSFORMATION.
 * This handles assembly transforms that position components.
 */
function parseComplexRepresentationRelationship(
  id: number,
  subEntities: { type: string; args: string }[],
  model: StepModel
): void {
  let name = '';
  let description = '';
  let rep1Id = 0;
  let rep2Id = 0;
  let transformationId: number | undefined;

  for (const sub of subEntities) {
    if (sub.type === 'REPRESENTATION_RELATIONSHIP') {
      // REPRESENTATION_RELATIONSHIP(' ',' ',#rep1,#rep2)
      const match = sub.args.match(/'([^']*)'\s*,\s*'([^']*)'\s*,\s*#(\d+)\s*,\s*#(\d+)/);
      if (match) {
        name = match[1];
        description = match[2];
        rep1Id = parseInt(match[3], 10);
        rep2Id = parseInt(match[4], 10);
      }
    } else if (sub.type === 'REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION') {
      // REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#transform_id)
      const match = sub.args.match(/#(\d+)/);
      if (match) {
        transformationId = parseInt(match[1], 10);
      }
    }
    // SHAPE_REPRESENTATION_RELATIONSHIP() has no args, just skip it
  }

  if (rep1Id > 0 && rep2Id > 0) {
    model.representationRelationships.set(id, {
      id,
      name,
      description,
      rep1Id,
      rep2Id,
      transformationId
    });
  }
}

/**
 * Parse a simple STEP entity (single type).
 */
function parseSimpleEntity(id: number, type: string, args: string, model: StepModel): void {

    switch (type) {
      case "CARTESIAN_POINT":
        parseCartesianPoint(id, args, model);
        break;
      case "VERTEX_POINT":
        parseVertexPoint(id, args, model);
        break;
      case "EDGE_CURVE":
        parseEdgeCurve(id, args, model);
        break;
      case "ORIENTED_EDGE":
        parseOrientedEdge(id, args, model);
        break;
      case "EDGE_LOOP":
        parseEdgeLoop(id, args, model);
        break;
      case "FACE_OUTER_BOUND":
        parseFaceBound(id, args, model, true);
        break;
      case "FACE_BOUND":
        parseFaceBound(id, args, model, false);
        break;
      case "ADVANCED_FACE":
        parseAdvancedFace(id, args, model);
        break;
      case "DIRECTION":
        parseDirection(id, args, model);
        break;
      case "AXIS2_PLACEMENT_3D":
        parseAxis2Placement3D(id, args, model);
        break;
      case "PLANE":
        parsePlane(id, args, model);
        break;
      // C4: Curved surface entities
      case "CYLINDRICAL_SURFACE":
        parseCylindricalSurface(id, args, model);
        break;
      case "SPHERICAL_SURFACE":
        parseSphericalSurface(id, args, model);
        break;
      case "CONICAL_SURFACE":
        parseConicalSurface(id, args, model);
        break;
      case "TOROIDAL_SURFACE":
        parseToroidalSurface(id, args, model);
        break;
      case "DEGENERATE_TOROIDAL_SURFACE":
        // Degenerate torus - treat similarly to regular torus for rendering
        parseDegenerateToroidalSurface(id, args, model);
        break;
      // C5: B-spline surfaces
      case "B_SPLINE_SURFACE_WITH_KNOTS":
        parseBSplineSurface(id, args, model);
        break;
      // C3: Curve geometry entities
      case "VECTOR":
        parseVector(id, args, model);
        break;
      case "LINE":
        parseLine(id, args, model);
        break;
      case "CIRCLE":
        parseCircle(id, args, model);
        break;
      case "ELLIPSE":
        parseEllipse(id, args, model);
        break;
      // C6b: B-spline curves (for trimmed B-spline surfaces)
      case "B_SPLINE_CURVE_WITH_KNOTS":
        parseBSplineCurve(id, args, model);
        break;
      case "SURFACE_CURVE":
        parseSurfaceCurve(id, args, model);
        break;
      // C4: PCURVE entities
      case "PCURVE":
        parsePCurve(id, args, model);
        break;
      case "DEFINITIONAL_REPRESENTATION":
        parseDefinitionalRepresentation(id, args, model);
        break;
      // C8: Full solids / assemblies
      case "CLOSED_SHELL":
        parseClosedShell(id, args, model);
        break;
      case "ORIENTED_CLOSED_SHELL":
        parseOrientedClosedShell(id, args, model);
        break;
      case "MANIFOLD_SOLID_BREP":
        parseManifoldSolidBrep(id, args, model);
        break;
      case "BREP_WITH_VOIDS":
        parseBrepWithVoids(id, args, model);
        break;
      case "ITEM_DEFINED_TRANSFORMATION":
        parseItemDefinedTransformation(id, args, model);
        break;
      case "COLOUR_RGB":
        parseColourRgb(id, args, model);
        break;
      case "FILL_AREA_STYLE_COLOUR":
        parseFillAreaStyleColour(id, args, model);
        break;
      case "FILL_AREA_STYLE":
        parseFillAreaStyle(id, args, model);
        break;
      case "SURFACE_STYLE_FILL_AREA":
        parseSurfaceStyleFillArea(id, args, model);
        break;
      case "SURFACE_SIDE_STYLE":
        parseSurfaceSideStyle(id, args, model);
        break;
      case "SURFACE_STYLE_USAGE":
        parseSurfaceStyleUsage(id, args, model);
        break;
      case "PRESENTATION_STYLE_ASSIGNMENT":
        parsePresentationStyleAssignment(id, args, model);
        break;
      case "STYLED_ITEM":
        parseStyledItem(id, args, model);
        break;
      case "ADVANCED_BREP_SHAPE_REPRESENTATION":
      case "SHAPE_REPRESENTATION":
        parseShapeRepresentation(id, args, model);
        break;
      case "SHAPE_REPRESENTATION_RELATIONSHIP":
        parseShapeRepresentationRelationship(id, args, model);
        break;
      // We ignore other entity types for now
    }
}

// --- Individual entity parsers (all tailored to our example syntax) ---

function parseCartesianPoint(id: number, args: string, model: StepModel) {
  // CARTESIAN_POINT('', (x, y, z)) - 3D point
  // CARTESIAN_POINT('', (x, y)) - 2D point (used in PCURVE, we skip these)
  const coord3dMatch = args.match(/\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)\s*$/);
  if (coord3dMatch) {
    const x = parseFloat(coord3dMatch[1]);
    const y = parseFloat(coord3dMatch[2]);
    const z = parseFloat(coord3dMatch[3]);
    model.points.set(id, { id, coords: [x, y, z] });
    return;
  }

  // Check for 2D point - store in points2d for PCURVE UV boundary extraction
  const coord2dMatch = args.match(/\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)\s*$/);
  if (coord2dMatch) {
    const u = parseFloat(coord2dMatch[1]);
    const v = parseFloat(coord2dMatch[2]);
    model.points2d.set(id, { id, coords: [u, v] });
    return;
  }

  throw new Error(`Failed to parse CARTESIAN_POINT args: ${args}`);
}

function parseVertexPoint(id: number, args: string, model: StepModel) {
  // VERTEX_POINT('', #10)
  const m = args.match(/'.*'\s*,\s*#(\d+)/);
  if (!m) throw new Error(`Failed to parse VERTEX_POINT args: ${args}`);
  const pointId = parseInt(m[1], 10);
  model.vertices.set(id, { id, pointId });
}

function parseEdgeCurve(id: number, args: string, model: StepModel) {
  // EDGE_CURVE('', #20, #21, #40, .T.)
  const m = args.match(/'.*'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*(\.[TF]\.)/);
  if (!m) throw new Error(`Failed to parse EDGE_CURVE args: ${args}`);
  const startVertexId = parseInt(m[1], 10);
  const endVertexId = parseInt(m[2], 10);
  const curveId = parseInt(m[3], 10);
  const sameSense = m[4] === ".T.";

  model.edgeCurves.set(id, {
    id,
    startVertexId,
    endVertexId,
    curveId,
    sameSense,
  });
}

function parseOrientedEdge(id: number, args: string, model: StepModel) {
  // ORIENTED_EDGE('', *, *, #50, .T.)
  const m = args.match(/'.*'\s*,\s*\*\s*,\s*\*\s*,\s*#(\d+)\s*,\s*(\.[TF]\.)/);
  if (!m) throw new Error(`Failed to parse ORIENTED_EDGE args: ${args}`);
  const edgeElementId = parseInt(m[1], 10);
  const orientation = m[2] === ".T.";


  model.orientedEdges.set(id, {
    id,
    edgeElementId,
    orientation,
  });
}

function parseEdgeLoop(id: number, args: string, model: StepModel) {
  // EDGE_LOOP('', (#60, #61, #62, #63))
  const m = args.match(/'.*'\s*,\s*\(([^)]*)\)/);
  if (!m) throw new Error(`Failed to parse EDGE_LOOP args: ${args}`);
  const idsStr = m[1].trim();
  const orientedEdgeIds = idsStr
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const mm = s.match(/^#(\d+)$/);
      if (!mm) throw new Error(`Invalid oriented edge ref in EDGE_LOOP: ${s}`);
      return parseInt(mm[1], 10);
    });

  model.edgeLoops.set(id, { id, orientedEdgeIds });
}

function parseFaceBound(id: number, args: string, model: StepModel, isOuter: boolean) {
  // FACE_OUTER_BOUND('', #70, .T.) or FACE_BOUND('', #70, .T.)
  const m = args.match(/'.*'\s*,\s*#(\d+)\s*,\s*(\.[TF]\.)/);
  if (!m) throw new Error(`Failed to parse FACE_BOUND args: ${args}`);
  const loopId = parseInt(m[1], 10);
  const orientation = m[2] === ".T.";

  model.faceBounds.set(id, { id, loopId, orientation, isOuter });
}

function parseAdvancedFace(id: number, args: string, model: StepModel) {
  // ADVANCED_FACE('', (#80), #5, .T.)
  const m = args.match(/'.*'\s*,\s*\(([^)]*)\)\s*,\s*#(\d+)\s*,\s*(\.[TF]\.)/);
  if (!m) throw new Error(`Failed to parse ADVANCED_FACE args: ${args}`);
  const boundList = m[1]
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const mm = s.match(/^#(\d+)$/);
      if (!mm) throw new Error(`Invalid bound ref in ADVANCED_FACE: ${s}`);
      return parseInt(mm[1], 10);
    });

  const surfaceId = parseInt(m[2], 10);
  const sameSense = m[3] === ".T.";

  model.faces.set(id, { id, boundIds: boundList, surfaceId, sameSense });
}

function parseDirection(id: number, args: string, model: StepModel) {
  // DIRECTION('', (x, y, z)) - 3D direction
  // DIRECTION('', (x, y)) - 2D direction (used in PCURVE, we skip these)
  const coord3dMatch = args.match(/\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)\s*$/);
  if (coord3dMatch) {
    const x = parseFloat(coord3dMatch[1]);
    const y = parseFloat(coord3dMatch[2]);
    const z = parseFloat(coord3dMatch[3]);
    model.directions.set(id, { id, dir: [x, y, z] });
    return;
  }

  // Check for 2D direction - store in directions2d for PCURVE UV boundary extraction
  const coord2dMatch = args.match(/\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)\s*$/);
  if (coord2dMatch) {
    const u = parseFloat(coord2dMatch[1]);
    const v = parseFloat(coord2dMatch[2]);
    model.directions2d.set(id, { id, dir: [u, v] });
    return;
  }

  throw new Error(`Failed to parse DIRECTION args: ${args}`);
}

function parseAxis2Placement3D(id: number, args: string, model: StepModel) {
  // AXIS2_PLACEMENT_3D('', #1, #2, #3) - with axis and ref_direction
  // AXIS2_PLACEMENT_3D('', #1, #2, $) - with axis, no ref_direction
  // AXIS2_PLACEMENT_3D('', #1, $, $) - no axis, no ref_direction
  // AXIS2_PLACEMENT_3D('', #1) - minimal form

  // Match the location (required)
  const locationMatch = args.match(/'.*'\s*,\s*#(\d+)/);
  if (!locationMatch) {
    throw new Error(`Failed to parse AXIS2_PLACEMENT_3D location: ${args}`);
  }
  const locationId = parseInt(locationMatch[1], 10);

  // Try to match axis (optional, can be $ or #id)
  let axisId: number | null = null;
  const axisMatch = args.match(/'.*'\s*,\s*#\d+\s*,\s*(#(\d+)|\$)/);
  if (axisMatch && axisMatch[2]) {
    axisId = parseInt(axisMatch[2], 10);
  }

  // Try to match ref_direction (optional, can be $ or #id)
  let refDirectionId: number | null = null;
  const refMatch = args.match(/'.*'\s*,\s*#\d+\s*,\s*(?:#\d+|\$)\s*,\s*(#(\d+)|\$)/);
  if (refMatch && refMatch[2]) {
    refDirectionId = parseInt(refMatch[2], 10);
  }

  model.axis2Placements.set(id, { id, locationId, axisId, refDirectionId });
}

function parsePlane(id: number, args: string, model: StepModel) {
  // PLANE('', #4)
  const m = args.match(/'.*'\s*,\s*#(\d+)/);
  if (!m) throw new Error(`Failed to parse PLANE args: ${args}`);
  const placementId = parseInt(m[1], 10);

  model.planes.set(id, { id, placementId });
}

// =============================================================================
// C4: Curved Surface Parsers
// =============================================================================

function parseCylindricalSurface(id: number, args: string, model: StepModel) {
  // CYLINDRICAL_SURFACE('', #placement, radius)
  // Example: CYLINDRICAL_SURFACE('',#127,5.)
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)/);
  if (!m) throw new Error(`Failed to parse CYLINDRICAL_SURFACE args: ${args}`);
  const placementId = parseInt(m[1], 10);
  const radius = parseFloat(m[2]);

  model.cylindricalSurfaces.set(id, { id, placementId, radius });
}

function parseSphericalSurface(id: number, args: string, model: StepModel) {
  // SPHERICAL_SURFACE('', #placement, radius)
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)/);
  if (!m) throw new Error(`Failed to parse SPHERICAL_SURFACE args: ${args}`);
  const placementId = parseInt(m[1], 10);
  const radius = parseFloat(m[2]);

  model.sphericalSurfaces.set(id, { id, placementId, radius });
}

function parseConicalSurface(id: number, args: string, model: StepModel) {
  // CONICAL_SURFACE('', #placement, radius, semi_angle)
  // semi_angle is in radians
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)/);
  if (!m) throw new Error(`Failed to parse CONICAL_SURFACE args: ${args}`);
  const placementId = parseInt(m[1], 10);
  const radius = parseFloat(m[2]);
  const semiAngle = parseFloat(m[3]);

  model.conicalSurfaces.set(id, { id, placementId, radius, semiAngle });
}

function parseToroidalSurface(id: number, args: string, model: StepModel) {
  // TOROIDAL_SURFACE('', #placement, major_radius, minor_radius)
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)/);
  if (!m) throw new Error(`Failed to parse TOROIDAL_SURFACE args: ${args}`);
  const placementId = parseInt(m[1], 10);
  const majorRadius = parseFloat(m[2]);
  const minorRadius = parseFloat(m[3]);

  model.toroidalSurfaces.set(id, { id, placementId, majorRadius, minorRadius });
}

function parseDegenerateToroidalSurface(id: number, args: string, model: StepModel) {
  // DEGENERATE_TOROIDAL_SURFACE('', #placement, major_radius, minor_radius, .T./.F.)
  // The last parameter indicates if it's a self-intersecting apple torus
  // We treat it the same as a regular torus for rendering purposes
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)/);
  if (!m) throw new Error(`Failed to parse DEGENERATE_TOROIDAL_SURFACE args: ${args}`);
  const placementId = parseInt(m[1], 10);
  const majorRadius = parseFloat(m[2]);
  const minorRadius = parseFloat(m[3]);

  // Store in the same toroidalSurfaces map - the tessellation will handle it the same way
  model.toroidalSurfaces.set(id, { id, placementId, majorRadius, minorRadius });
}

// =============================================================================
// C5: B-Spline Surface Parser
// =============================================================================

function parseBSplineSurface(id: number, args: string, model: StepModel) {
  // B_SPLINE_SURFACE_WITH_KNOTS('name', u_degree, v_degree,
  //   control_points_list,  // 2D array: ((#p1,#p2,...),(#p3,#p4,...),...)
  //   surface_form, u_closed, v_closed, self_intersect,
  //   u_multiplicities, v_multiplicities,
  //   u_knots, v_knots,
  //   knot_spec)

  // Parse degrees
  const degreeMatch = args.match(/'[^']*'\s*,\s*(\d+)\s*,\s*(\d+)\s*,/);
  if (!degreeMatch) {
    return;
  }
  const uDegree = parseInt(degreeMatch[1], 10);
  const vDegree = parseInt(degreeMatch[2], 10);

  // Parse 2D control points array: ((#1,#2,#3),(#4,#5,#6),...)
  // Use a regex that matches nested parentheses
  const cpArrayMatch = args.match(/\(\s*\([^)]+\)(?:\s*,\s*\([^)]+\))*\s*\)/);
  if (!cpArrayMatch) {
    return;
  }

  // Extract the 2D array content
  const cpArrayStr = cpArrayMatch[0];
  const controlPointIds: number[][] = [];

  // Match each row: (#1,#2,#3,#4)
  const rowRegex = /\(\s*(#[\d\s,#]+)\s*\)/g;
  let rowMatch;
  while ((rowMatch = rowRegex.exec(cpArrayStr)) !== null) {
    const rowStr = rowMatch[1];
    const pointIds: number[] = [];
    const pointRegex = /#(\d+)/g;
    let pointMatch;
    while ((pointMatch = pointRegex.exec(rowStr)) !== null) {
      pointIds.push(parseInt(pointMatch[1], 10));
    }
    if (pointIds.length > 0) {
      controlPointIds.push(pointIds);
    }
  }

  if (controlPointIds.length === 0) {
    return;
  }

  // Parse logical flags: .F. or .T.
  const flagsMatch = args.match(/\.\w+\.\s*,\s*\.([TF])\.\s*,\s*\.([TF])\.\s*,\s*\.([TF])\./);
  const uClosed = flagsMatch ? flagsMatch[1] === 'T' : false;
  const vClosed = flagsMatch ? flagsMatch[2] === 'T' : false;

  // Parse multiplicities and knots from after the control points
  // Structure: ...control_points, surface_form, u_closed, v_closed, self_intersect,
  //            u_mults, v_mults, u_knots, v_knots, knot_spec)
  const afterCp = args.substring(args.indexOf(cpArrayStr) + cpArrayStr.length);

  // Find all parenthesized number lists
  const allLists: { nums: number[], isFloat: boolean }[] = [];
  const listRegex = /\(\s*([-0-9.Ee+\s,]+)\s*\)/g;
  let listMatch;
  while ((listMatch = listRegex.exec(afterCp)) !== null) {
    const content = listMatch[1];
    const nums = content.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    if (nums.length > 0) {
      // Check if it contains decimal points (float list) or all integers
      const isFloat = content.includes('.');
      allLists.push({ nums, isFloat });
    }
  }

  // Separate into integer lists (multiplicities) and float lists (knots)
  const intLists = allLists.filter(l => !l.isFloat).map(l => l.nums.map(n => Math.round(n)));
  const realLists = allLists.filter(l => l.isFloat).map(l => l.nums);

  // Assign parsed values (best effort)
  const uKnotMultiplicities = intLists[0] || [];
  const vKnotMultiplicities = intLists[1] || [];
  const uKnots = realLists[0] || [];
  const vKnots = realLists[1] || [];

  model.bSplineSurfaces.set(id, {
    id,
    uDegree,
    vDegree,
    controlPointIds,
    uKnotMultiplicities,
    vKnotMultiplicities,
    uKnots,
    vKnots,
    uClosed,
    vClosed,
  });
}

// =============================================================================
// C3: Curve Entity Parsers
// =============================================================================

function parseVector(id: number, args: string, model: StepModel) {
  // VECTOR('', #direction, magnitude)
  // Example: VECTOR('', #30, 1.)
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)/);
  if (!m) throw new Error(`Failed to parse VECTOR args: ${args}`);
  const directionId = parseInt(m[1], 10);
  const magnitude = parseFloat(m[2]);

  // Store in 3D vectors - we'll determine 2D vs 3D based on referenced direction later
  model.vectors.set(id, { id, directionId, magnitude });
  // Also store as 2D vector if the direction is 2D
  model.vectors2d.set(id, { id, directionId, magnitude });
}

function parseLine(id: number, args: string, model: StepModel) {
  // LINE('', #point, #vector)
  // Example: LINE('', #28, #29)
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*#(\d+)/);
  if (!m) throw new Error(`Failed to parse LINE args: ${args}`);
  const pointId = parseInt(m[1], 10);
  const vectorId = parseInt(m[2], 10);

  // Store in 3D lines - we'll determine 2D vs 3D based on referenced point later
  model.lines.set(id, { id, pointId, vectorId });
  // Also store as 2D line (will be used if the point is 2D)
  model.lines2d.set(id, { id, pointId, vectorId });
}

function parseCircle(id: number, args: string, model: StepModel) {
  // CIRCLE('', #axis2_placement, radius)
  // Example: CIRCLE('', #224, 5.)
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)/);
  if (!m) throw new Error(`Failed to parse CIRCLE args: ${args}`);
  const placementId = parseInt(m[1], 10);
  const radius = parseFloat(m[2]);

  model.circles.set(id, { id, placementId, radius });
}

function parseEllipse(id: number, args: string, model: StepModel) {
  // ELLIPSE('', #axis2_placement, major_radius, minor_radius)
  // Example: ELLIPSE('', #189412, 545.26, 196.83)
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)/);
  if (!m) throw new Error(`Failed to parse ELLIPSE args: ${args}`);
  const placementId = parseInt(m[1], 10);
  const majorRadius = parseFloat(m[2]);
  const minorRadius = parseFloat(m[3]);

  model.ellipses.set(id, { id, placementId, majorRadius, minorRadius });
}

// =============================================================================
// C6b: B-Spline Curve Parsing (for trimmed B-spline surfaces)
// =============================================================================

function parseBSplineCurve(id: number, args: string, model: StepModel) {
  // B_SPLINE_CURVE_WITH_KNOTS('name', degree,
  //   (#cp1, #cp2, ...),      -- control points
  //   .form.,                  -- curve form
  //   .closed.,               -- closed curve flag
  //   .self_intersect.,       -- self-intersect flag
  //   (mult1, mult2, ...),    -- knot multiplicities
  //   (knot1, knot2, ...),    -- knot values
  //   .knot_type.)            -- knot specification

  // Extract degree
  const degreeMatch = args.match(/'[^']*'\s*,\s*(\d+)/);
  if (!degreeMatch) {
    return; // Skip if can't parse
  }
  const degree = parseInt(degreeMatch[1], 10);

  // Extract control point IDs
  const cpListMatch = args.match(/,\s*\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)/);
  if (!cpListMatch) {
    return;
  }
  const cpMatches = cpListMatch[1].match(/#(\d+)/g);
  if (!cpMatches) {
    return;
  }
  const controlPointIds = cpMatches.map(m => parseInt(m.substring(1), 10));

  // Extract knot multiplicities - find second parenthesized list of numbers
  const afterCPs = args.substring(args.indexOf(cpListMatch[0]) + cpListMatch[0].length);
  const multMatch = afterCPs.match(/\(\s*([\d\s,]+)\s*\)/);
  if (!multMatch) {
    return;
  }
  const knotMultiplicities = multMatch[1].split(',').map(s => parseInt(s.trim(), 10));

  // Extract knot values - find the list of floats after multiplicities
  const afterMults = afterCPs.substring(afterCPs.indexOf(multMatch[0]) + multMatch[0].length);
  const knotsMatch = afterMults.match(/\(\s*([-0-9.Ee+\s,]+)\s*\)/);
  if (!knotsMatch) {
    return;
  }
  const knots = knotsMatch[1].split(',').map(s => parseFloat(s.trim()));

  model.bsplines.set(id, {
    id,
    degree,
    controlPointIds,
    knotMultiplicities,
    knots,
    closed: false,  // TODO: Detect from STEP data
  });
}

function parseSurfaceCurve(id: number, args: string, model: StepModel) {
  // SURFACE_CURVE('', #3d_curve, (#pcurve1, #pcurve2), .PCURVE_S1.)
  // Example: SURFACE_CURVE('', #223, (#228, #239), .PCURVE_S1.)

  // Extract 3D curve reference
  const curve3dMatch = args.match(/'[^']*'\s*,\s*#(\d+)/);
  if (!curve3dMatch) throw new Error(`Failed to parse SURFACE_CURVE 3D curve: ${args}`);
  const curve3dId = parseInt(curve3dMatch[1], 10);

  // Extract PCURVE references from the list
  const pcurveListMatch = args.match(/\((\s*#\d+(?:\s*,\s*#\d+)*\s*)\)/);
  const pcurveIds: number[] = [];
  if (pcurveListMatch) {
    const pcurveMatches = pcurveListMatch[1].match(/#(\d+)/g);
    if (pcurveMatches) {
      for (const match of pcurveMatches) {
        pcurveIds.push(parseInt(match.substring(1), 10));
      }
    }
  }

  // Extract preference (.PCURVE_S1., .PCURVE_S2., .CURVE_3D.)
  const prefMatch = args.match(/\.(PCURVE_S1|PCURVE_S2|CURVE_3D)\./);
  const preference = prefMatch ? prefMatch[1] : 'CURVE_3D';

  model.surfaceCurves.set(id, { id, curve3dId, pcurveIds, preference });
}

// =============================================================================
// C4: PCURVE Parsing
// =============================================================================

function parsePCurve(id: number, args: string, model: StepModel) {
  // PCURVE('', #surface, #definitional_representation)
  // Example: PCURVE('',#126,#260)
  const m = args.match(/'[^']*'\s*,\s*#(\d+)\s*,\s*#(\d+)/);
  if (!m) throw new Error(`Failed to parse PCURVE args: ${args}`);
  const surfaceId = parseInt(m[1], 10);
  const representationId = parseInt(m[2], 10);

  model.pcurves.set(id, { id, surfaceId, representationId });
}

function parseDefinitionalRepresentation(id: number, args: string, model: StepModel) {
  // DEFINITIONAL_REPRESENTATION('', (#curve1, #curve2, ...), #context)
  // Example: DEFINITIONAL_REPRESENTATION('',(#261),#265)
  // We only care about the curve references

  const curveListMatch = args.match(/\((\s*#\d+(?:\s*,\s*#\d+)*\s*)\)/);
  const curveIds: number[] = [];
  if (curveListMatch) {
    const curveMatches = curveListMatch[1].match(/#(\d+)/g);
    if (curveMatches) {
      for (const match of curveMatches) {
        curveIds.push(parseInt(match.substring(1), 10));
      }
    }
  }

  model.definitionalRepresentations.set(id, { id, curveIds });
}

// =============================================================================
// C8: Full Solids / Assemblies Parsing
// =============================================================================

function parseClosedShell(id: number, args: string, model: StepModel) {
  // CLOSED_SHELL ( 'NONE', ( #1417, #3481, #2185, ... ) )
  const nameMatch = args.match(/^'([^']*)'/);
  const name = nameMatch ? nameMatch[1] : '';

  const faceListMatch = args.match(/\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)\s*\)?\s*$/);
  const faceIds: number[] = [];
  if (faceListMatch) {
    const faceMatches = faceListMatch[1].match(/#(\d+)/g);
    if (faceMatches) {
      for (const match of faceMatches) {
        faceIds.push(parseInt(match.substring(1), 10));
      }
    }
  }

  model.closedShells.set(id, { id, name, faceIds });
}

function parseOrientedClosedShell(id: number, args: string, model: StepModel) {
  // ORIENTED_CLOSED_SHELL('',*,#139308,.F.);
  // Format: name, *, shell_ref, orientation
  // We need to get the faces from the referenced closed shell
  const match = args.match(/#(\d+)/);
  if (!match) return;

  const referencedShellId = parseInt(match[1], 10);

  // Store a reference - we'll resolve it after all shells are parsed
  // The orientation (.T./.F.) indicates whether faces should be flipped
  const orientationMatch = args.match(/\.(T|F)\.\s*$/);
  const isReversed = orientationMatch ? orientationMatch[1] === 'F' : false;

  // Store with a special marker that it's an oriented reference
  model.closedShells.set(id, {
    id,
    name: `OrientedRef_${referencedShellId}`,
    faceIds: [], // Will be populated in post-processing
    _referencedShellId: referencedShellId,
    _isReversed: isReversed
  } as ClosedShell & { _referencedShellId?: number; _isReversed?: boolean });
}

function parseManifoldSolidBrep(id: number, args: string, model: StepModel) {
  // MANIFOLD_SOLID_BREP ( 'Fillet3', #903 )
  const match = args.match(/^'([^']*)'\s*,\s*#(\d+)/);
  if (!match) return;

  const name = match[1];
  const shellId = parseInt(match[2], 10);

  model.manifoldSolidBreps.set(id, { id, name, shellId });
}

function parseBrepWithVoids(id: number, args: string, model: StepModel) {
  // BREP_WITH_VOIDS('',#139307,(#533,#534));
  // Format: name, outer_shell, (void_shells...)
  const match = args.match(/^'([^']*)'\s*,\s*#(\d+)\s*,\s*\(([^)]*)\)/);
  if (!match) return;

  const name = match[1];
  const outerShellId = parseInt(match[2], 10);
  const voidShellIds: number[] = [];

  // Parse void shell IDs
  const voidRefs = match[3].match(/#(\d+)/g);
  if (voidRefs) {
    for (const ref of voidRefs) {
      const voidId = parseInt(ref.slice(1), 10);
      voidShellIds.push(voidId);
    }
  }

  model.brepWithVoids.set(id, { id, name, outerShellId, voidShellIds });
}

function parseItemDefinedTransformation(id: number, args: string, model: StepModel) {
  // ITEM_DEFINED_TRANSFORMATION(' ',' ',#189384,#191213);
  // Format: name, description, source_axis2_placement, target_axis2_placement
  const match = args.match(/'([^']*)'\s*,\s*'([^']*)'\s*,\s*#(\d+)\s*,\s*#(\d+)/);
  if (!match) return;

  const name = match[1];
  const description = match[2];
  const transformItem1Id = parseInt(match[3], 10);
  const transformItem2Id = parseInt(match[4], 10);

  model.itemDefinedTransformations.set(id, {
    id,
    name,
    description,
    transformItem1Id,
    transformItem2Id
  });
}

function parseColourRgb(id: number, args: string, model: StepModel) {
  // COLOUR_RGB ( '',0.792, 0.820, 0.933 )
  const match = args.match(/^'([^']*)'\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)/);
  if (!match) return;

  const name = match[1];
  const r = parseFloat(match[2]);
  const g = parseFloat(match[3]);
  const b = parseFloat(match[4]);

  model.colourRgbs.set(id, { id, name, r, g, b });
}

function parseFillAreaStyleColour(id: number, args: string, model: StepModel) {
  // FILL_AREA_STYLE_COLOUR ( '', #1833 )
  const match = args.match(/^'([^']*)'\s*,\s*#(\d+)/);
  if (!match) return;

  const name = match[1];
  const colourId = parseInt(match[2], 10);

  model.fillAreaStyleColours.set(id, { id, name, colourId });
}

function parseFillAreaStyle(id: number, args: string, model: StepModel) {
  // FILL_AREA_STYLE ('',( #3957 ) )
  const nameMatch = args.match(/^'([^']*)'/);
  const name = nameMatch ? nameMatch[1] : '';

  const styleListMatch = args.match(/\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)/);
  const fillStyleIds: number[] = [];
  if (styleListMatch) {
    const styleMatches = styleListMatch[1].match(/#(\d+)/g);
    if (styleMatches) {
      for (const match of styleMatches) {
        fillStyleIds.push(parseInt(match.substring(1), 10));
      }
    }
  }

  model.fillAreaStyles.set(id, { id, name, fillStyleIds });
}

function parseSurfaceStyleFillArea(id: number, args: string, model: StepModel) {
  // SURFACE_STYLE_FILL_AREA ( #2251 )
  const match = args.match(/#(\d+)/);
  if (!match) return;

  const fillAreaStyleId = parseInt(match[1], 10);

  model.surfaceStyleFillAreas.set(id, { id, fillAreaStyleId });
}

function parseSurfaceSideStyle(id: number, args: string, model: StepModel) {
  // SURFACE_SIDE_STYLE ('',( #3519 ) )
  const nameMatch = args.match(/^'([^']*)'/);
  const name = nameMatch ? nameMatch[1] : '';

  const styleListMatch = args.match(/\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)/);
  const styleIds: number[] = [];
  if (styleListMatch) {
    const styleMatches = styleListMatch[1].match(/#(\d+)/g);
    if (styleMatches) {
      for (const match of styleMatches) {
        styleIds.push(parseInt(match.substring(1), 10));
      }
    }
  }

  model.surfaceSideStyles.set(id, { id, name, styleIds });
}

function parseSurfaceStyleUsage(id: number, args: string, model: StepModel) {
  // SURFACE_STYLE_USAGE ( .BOTH. , #667 )
  const match = args.match(/(\.[A-Z_]+\.)\s*,\s*#(\d+)/);
  if (!match) return;

  const side = match[1];
  const styleId = parseInt(match[2], 10);

  model.surfaceStyleUsages.set(id, { id, side, styleId });
}

function parsePresentationStyleAssignment(id: number, args: string, model: StepModel) {
  // PRESENTATION_STYLE_ASSIGNMENT (( #1623 ) )
  const styleListMatch = args.match(/\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)/);
  const styleIds: number[] = [];
  if (styleListMatch) {
    const styleMatches = styleListMatch[1].match(/#(\d+)/g);
    if (styleMatches) {
      for (const match of styleMatches) {
        styleIds.push(parseInt(match.substring(1), 10));
      }
    }
  }

  model.presentationStyleAssignments.set(id, { id, styleIds });
}

function parseStyledItem(id: number, args: string, model: StepModel) {
  // STYLED_ITEM ( 'NONE', ( #2892 ), #3700 )
  const nameMatch = args.match(/^'([^']*)'/);
  const name = nameMatch ? nameMatch[1] : '';

  // Match style list (may have multiple styles)
  const styleListMatch = args.match(/\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)/);
  const styleIds: number[] = [];
  if (styleListMatch) {
    const styleMatches = styleListMatch[1].match(/#(\d+)/g);
    if (styleMatches) {
      for (const match of styleMatches) {
        styleIds.push(parseInt(match.substring(1), 10));
      }
    }
  }

  // Match the item reference (last #id in the args)
  const itemMatch = args.match(/#(\d+)\s*\)?\s*$/);
  if (!itemMatch) return;
  const itemId = parseInt(itemMatch[1], 10);

  model.styledItems.set(id, { id, name, styleIds, itemId });
}

function parseShapeRepresentation(id: number, args: string, model: StepModel) {
  // ADVANCED_BREP_SHAPE_REPRESENTATION ( 'VM-001', ( #3700, #448 ), #2830 )
  const nameMatch = args.match(/^'([^']*)'/);
  const name = nameMatch ? nameMatch[1] : '';

  // Match item list
  const itemListMatch = args.match(/\(\s*(#\d+(?:\s*,\s*#\d+)*)\s*\)/);
  const itemIds: number[] = [];
  if (itemListMatch) {
    const itemMatches = itemListMatch[1].match(/#(\d+)/g);
    if (itemMatches) {
      for (const match of itemMatches) {
        itemIds.push(parseInt(match.substring(1), 10));
      }
    }
  }

  // Match the context reference (last #id in the args)
  const contextMatch = args.match(/#(\d+)\s*\)?\s*$/);
  const contextId = contextMatch ? parseInt(contextMatch[1], 10) : 0;

  model.shapeRepresentations.set(id, { id, name, itemIds, contextId });
}

/**
 * Parse SHAPE_REPRESENTATION_RELATIONSHIP - relates two shape representations without transform.
 * Format: SHAPE_REPRESENTATION_RELATIONSHIP('name', 'desc', #rep1, #rep2)
 */
function parseShapeRepresentationRelationship(id: number, args: string, model: StepModel) {
  // Extract name, description, and two rep IDs
  const match = args.match(/'([^']*)'\s*,\s*'([^']*)'\s*,\s*#(\d+)\s*,\s*#(\d+)/);
  if (!match) return;

  model.representationRelationships.set(id, {
    id,
    name: match[1],
    description: match[2],
    rep1Id: parseInt(match[3], 10),
    rep2Id: parseInt(match[4], 10),
    // No transform for simple relationships
  });
}

// =============================================================================
// C8: Color Resolution (follow STYLED_ITEM -> color chain)
// =============================================================================

/**
 * Resolve the color for a styled item by following the STEP style chain.
 */
function resolveColorForItem(model: StepModel, itemId: number): ResolvedColor | undefined {
  for (const styledItem of model.styledItems.values()) {
    if (styledItem.itemId === itemId) {
      return resolveColorFromStyledItem(model, styledItem);
    }
  }
  return undefined;
}

function resolveColorFromStyledItem(model: StepModel, styledItem: StyledItem): ResolvedColor | undefined {
  for (const styleId of styledItem.styleIds) {
    const psa = model.presentationStyleAssignments.get(styleId);
    if (psa) {
      const color = resolveColorFromPSA(model, psa);
      if (color) {
        return color;
      }
    }
  }
  return undefined;
}

function resolveColorFromPSA(model: StepModel, psa: PresentationStyleAssignment): ResolvedColor | undefined {
  for (const styleId of psa.styleIds) {
    const ssu = model.surfaceStyleUsages.get(styleId);
    if (ssu) {
      const sss = model.surfaceSideStyles.get(ssu.styleId);
      if (sss) {
        for (const fillStyleId of sss.styleIds) {
          const ssfa = model.surfaceStyleFillAreas.get(fillStyleId);
          if (ssfa) {
            const fas = model.fillAreaStyles.get(ssfa.fillAreaStyleId);
            if (fas) {
              for (const fascId of fas.fillStyleIds) {
                const fasc = model.fillAreaStyleColours.get(fascId);
                if (fasc) {
                  const colour = model.colourRgbs.get(fasc.colourId);
                  if (colour) {
                    return { r: colour.r, g: colour.g, b: colour.b };
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return undefined;
}

// Enable verbose transform debugging with VITE_VERBOSE_TRANSFORM=true
const VERBOSE_TRANSFORM_DEBUG = typeof import.meta !== 'undefined' &&
  (import.meta.env?.VITE_VERBOSE_TRANSFORM === 'true');

/**
 * Build a 4x4 transformation matrix from source and target AXIS2_PLACEMENT_3D.
 * The transform maps coordinates from source local space to target (assembly) space.
 */
function buildTransformMatrix(
  model: StepModel,
  sourceId: number,
  targetId: number
): Transform | undefined {
  const source = model.axis2Placements.get(sourceId);
  const target = model.axis2Placements.get(targetId);
  if (!source || !target) return undefined;

  // Get source coordinate system
  const sourceOrigin = model.points.get(source.locationId)?.coords || [0, 0, 0];
  let sourceZ: Vec3 = [0, 0, 1];
  let sourceX: Vec3 = [1, 0, 0];
  if (source.axisId !== null) {
    const dir = model.directions.get(source.axisId);
    if (dir) sourceZ = dir.dir;
  }
  if (source.refDirectionId !== null) {
    const dir = model.directions.get(source.refDirectionId);
    if (dir) sourceX = dir.dir;
  }
  const sourceY = vec3Cross(sourceZ, sourceX);

  // Get target coordinate system
  const targetOrigin = model.points.get(target.locationId)?.coords || [0, 0, 0];
  let targetZ: Vec3 = [0, 0, 1];
  let targetX: Vec3 = [1, 0, 0];
  if (target.axisId !== null) {
    const dir = model.directions.get(target.axisId);
    if (dir) targetZ = dir.dir;
  }
  if (target.refDirectionId !== null) {
    const dir = model.directions.get(target.refDirectionId);
    if (dir) targetX = dir.dir;
  }
  const targetY = vec3Cross(targetZ, targetX);

  // For ITEM_DEFINED_TRANSFORMATION, the convention is:
  // transformItem1 is the source placement (local coordinate system)
  // transformItem2 is the target placement (assembly coordinate system)
  // We need to transform points from source local space to target assembly space
  //
  // The transformation is: P_target = R * (P_source - O_source) + O_target
  // Where R maps source axes to target axes
  //
  // In matrix form: M = T_target * R * T_source^-1
  // Where T_source^-1 translates by -sourceOrigin, R rotates, T_target translates by targetOrigin

  // Build rotation matrix R that maps source axes to target axes
  // If source has axes (sX, sY, sZ) and target has (tX, tY, tZ),
  // R maps sX->tX, sY->tY, sZ->tZ
  // R = [tX|tY|tZ] * [sX|sY|sZ]^T (target axes expressed in terms of source axes)

  // For the simple case where source axes are identity (or we want to ignore source rotation):
  // Just use target axes as the rotation
  // But we need to account for sourceOrigin

  // Build 4x4 matrix in column-major order
  // The full transform is: P' = R * (P - O_source) + O_target
  // Expanding: P' = R*P - R*O_source + O_target
  // So translation = O_target - R*O_source

  const matrix = new Array(16).fill(0);

  // Rotation part: target axes (assuming source is at standard orientation)
  matrix[0] = targetX[0]; matrix[1] = targetX[1]; matrix[2] = targetX[2]; matrix[3] = 0;
  matrix[4] = targetY[0]; matrix[5] = targetY[1]; matrix[6] = targetY[2]; matrix[7] = 0;
  matrix[8] = targetZ[0]; matrix[9] = targetZ[1]; matrix[10] = targetZ[2]; matrix[11] = 0;

  // Translation part: O_target - R*O_source
  // R*O_source = [targetX·O_s, targetY·O_s, targetZ·O_s]
  const rotatedSourceOrigin: Vec3 = [
    targetX[0] * sourceOrigin[0] + targetY[0] * sourceOrigin[1] + targetZ[0] * sourceOrigin[2],
    targetX[1] * sourceOrigin[0] + targetY[1] * sourceOrigin[1] + targetZ[1] * sourceOrigin[2],
    targetX[2] * sourceOrigin[0] + targetY[2] * sourceOrigin[1] + targetZ[2] * sourceOrigin[2]
  ];

  matrix[12] = targetOrigin[0] - rotatedSourceOrigin[0];
  matrix[13] = targetOrigin[1] - rotatedSourceOrigin[1];
  matrix[14] = targetOrigin[2] - rotatedSourceOrigin[2];
  matrix[15] = 1;

  if (VERBOSE_TRANSFORM_DEBUG) {
    console.log(`[buildTransformMatrix] Source: origin=[${sourceOrigin.map(v => v.toFixed(2)).join(', ')}]`);
    console.log(`[buildTransformMatrix] Target: origin=[${targetOrigin.map(v => v.toFixed(2)).join(', ')}]`);
    console.log(`[buildTransformMatrix] Final translation=[${matrix[12].toFixed(2)}, ${matrix[13].toFixed(2)}, ${matrix[14].toFixed(2)}]`);
  }

  return { matrix };
}

/**
 * Find the transform for a solid by tracing through shape representations.
 * This walks up the assembly hierarchy to find and compose transforms.
 */
function findTransformForSolid(model: StepModel, solidId: number): Transform | undefined {
  // The solidId might be:
  // 1. A MANIFOLD_SOLID_BREP directly in a SHAPE_REPRESENTATION
  // 2. A BREP_WITH_VOIDS in a SHAPE_REPRESENTATION
  // 3. A CLOSED_SHELL that's part of a BREP_WITH_VOIDS
  // 4. A standalone CLOSED_SHELL

  // First, check if this is a shell that's part of a BREP_WITH_VOIDS
  let effectiveId = solidId;
  for (const brep of model.brepWithVoids.values()) {
    if (brep.outerShellId === solidId || brep.voidShellIds.includes(solidId)) {
      // Use the BREP_WITH_VOIDS ID instead, as that's what's in the shape representation
      effectiveId = brep.id;
      if (VERBOSE_TRANSFORM_DEBUG) {
        console.log(`[findTransformForSolid] Shell #${solidId} is part of BREP_WITH_VOIDS #${brep.id}`);
      }
      break;
    }
  }

  // Find which SHAPE_REPRESENTATION contains this solid (or its parent BREP)
  let currentRepId: number | undefined;
  for (const rep of model.shapeRepresentations.values()) {
    if (rep.itemIds.includes(effectiveId)) {
      currentRepId = rep.id;
      if (VERBOSE_TRANSFORM_DEBUG) {
        console.log(`[findTransformForSolid] Solid #${solidId} (effective #${effectiveId}) found in Shape Rep #${currentRepId}`);
      }
      break;
    }
  }

  if (currentRepId === undefined) {
    if (VERBOSE_TRANSFORM_DEBUG) {
      console.log(`[findTransformForSolid] Solid #${solidId} not found in any shape representation`);
    }
    return undefined;
  }

  // Walk up the representation hierarchy looking for transforms
  // The hierarchy is: ADVANCED_BREP_SHAPE_REPRESENTATION -> SHAPE_REPRESENTATION -> ... with transforms
  // Relationships: SHAPE_REPRESENTATION_RELATIONSHIP(rep1, rep2) where rep2 is the "child" (detailed rep)
  // Complex entities with transforms: REPRESENTATION_RELATIONSHIP(rep1, rep2) where rep1 is being positioned in rep2
  const visited = new Set<number>();
  let composedTransform: Transform | undefined;

  while (currentRepId !== undefined && !visited.has(currentRepId)) {
    visited.add(currentRepId);

    let foundNext = false;
    for (const rel of model.representationRelationships.values()) {
      // For SHAPE_REPRESENTATION_RELATIONSHIP: rep2 is the detailed (child) representation
      // For REPRESENTATION_RELATIONSHIP with transform: rep1 is positioned relative to rep2

      // Case 1: currentRepId is rep2 (child), we go to rep1 (parent) - no transform on this type
      if (rel.rep2Id === currentRepId && rel.transformationId === undefined) {
        if (VERBOSE_TRANSFORM_DEBUG) {
          console.log(`[findTransformForSolid] Following relationship: Rep #${currentRepId} (child) -> Rep #${rel.rep1Id} (parent)`);
        }
        currentRepId = rel.rep1Id;
        foundNext = true;
        break;
      }

      // Case 2: currentRepId is rep1, and there's a transform to rep2 (assembly)
      if (rel.rep1Id === currentRepId && rel.transformationId !== undefined) {
        const transform = model.itemDefinedTransformations.get(rel.transformationId);
        if (transform) {
          if (VERBOSE_TRANSFORM_DEBUG) {
            console.log(`[findTransformForSolid] Found transform: Rep #${currentRepId} -> Rep #${rel.rep2Id} via Transform #${rel.transformationId}`);
          }
          const matrix = buildTransformMatrix(
            model,
            transform.transformItem1Id,
            transform.transformItem2Id
          );
          if (matrix) {
            if (VERBOSE_TRANSFORM_DEBUG) {
              console.log(`[findTransformForSolid] Transform translation=[${matrix.matrix[12].toFixed(2)}, ${matrix.matrix[13].toFixed(2)}, ${matrix.matrix[14].toFixed(2)}]`);
            }
            // DISABLED: return matrix to see baseline without transforms
            // return matrix;
            // Compose transforms - new transform should be applied AFTER existing ones
            if (composedTransform) {
              composedTransform = multiplyTransforms(matrix, composedTransform);
            } else {
              composedTransform = matrix;
            }
          }
        }
        // Continue up the hierarchy
        currentRepId = rel.rep2Id;
        foundNext = true;
        break;
      }
    }

    if (!foundNext) {
      break;
    }
  }

  if (VERBOSE_TRANSFORM_DEBUG && composedTransform) {
    console.log(`[findTransformForSolid] Final transform for solid #${solidId}: translation=[${composedTransform.matrix[12].toFixed(2)}, ${composedTransform.matrix[13].toFixed(2)}, ${composedTransform.matrix[14].toFixed(2)}]`);
  }

  return composedTransform;
}

/**
 * Multiply two 4x4 transform matrices (column-major order).
 */
function multiplyTransforms(a: Transform, b: Transform): Transform {
  const ma = a.matrix;
  const mb = b.matrix;
  const result = new Array(16);

  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      result[col * 4 + row] =
        ma[0 * 4 + row] * mb[col * 4 + 0] +
        ma[1 * 4 + row] * mb[col * 4 + 1] +
        ma[2 * 4 + row] * mb[col * 4 + 2] +
        ma[3 * 4 + row] * mb[col * 4 + 3];
    }
  }

  return { matrix: result };
}

/**
 * Extract all solids with their face IDs and colors from the model.
 */
export function extractSolidsWithColors(model: StepModel): SolidWithColor[] {
  const solids: SolidWithColor[] = [];

  for (const brep of model.manifoldSolidBreps.values()) {
    const shell = model.closedShells.get(brep.shellId);
    if (!shell) continue;

    const color = resolveColorForItem(model, brep.id);
    const transform = findTransformForSolid(model, brep.id);

    solids.push({
      solidId: brep.id,
      name: brep.name || shell.name || `Solid_${brep.id}`,
      faceIds: shell.faceIds,
      color,
      transform,
    });
  }

  // Also add any standalone CLOSED_SHELLs that aren't referenced by MANIFOLD_SOLID_BREP
  for (const shell of model.closedShells.values()) {
    const isReferenced = [...model.manifoldSolidBreps.values()].some(b => b.shellId === shell.id);
    if (isReferenced) continue;

    // Check if this shell is part of a BREP_WITH_VOIDS - if so, use that ID for color lookup
    let colorLookupId = shell.id;
    for (const brep of model.brepWithVoids.values()) {
      if (brep.outerShellId === shell.id || brep.voidShellIds.includes(shell.id)) {
        colorLookupId = brep.id;
        break;
      }
    }

    const color = resolveColorForItem(model, colorLookupId);
    const transform = findTransformForSolid(model, shell.id);

    solids.push({
      solidId: shell.id,
      name: shell.name || `Shell_${shell.id}`,
      faceIds: shell.faceIds,
      color,
      transform,
    });
  }

  return solids;
}

// =============================================================================
// C3: Curve Resolution (follow reference chain to get geometry)
// =============================================================================

/**
 * Resolve an AXIS2_PLACEMENT_3D to get origin, Z-axis (normal), and X-axis (refDir).
 * Used by circles and ellipses to define their coordinate system.
 */
function resolveAxis2Placement(
  model: StepModel,
  placementId: number
): { origin: Vec3; normal: Vec3; refDirection: Vec3 } {
  const placement = model.axis2Placements.get(placementId);
  if (!placement) {
    throw new Error(`AXIS2_PLACEMENT_3D #${placementId} not found`);
  }

  // Get origin
  const originPoint = model.points.get(placement.locationId);
  if (!originPoint) {
    throw new Error(`Origin point #${placement.locationId} not found`);
  }
  const origin = originPoint.coords;

  // Get normal (Z axis) - defaults to (0, 0, 1) if not specified
  let normal: Vec3 = [0, 0, 1];
  if (placement.axisId !== null) {
    const axisDir = model.directions.get(placement.axisId);
    if (axisDir) {
      normal = axisDir.dir;
    }
  }

  // Get ref direction (X axis) - defaults to (1, 0, 0) if not specified
  let refDirection: Vec3 = [1, 0, 0];
  if (placement.refDirectionId !== null) {
    const refDir = model.directions.get(placement.refDirectionId);
    if (refDir) {
      refDirection = refDir.dir;
    }
  }

  return { origin, normal, refDirection };
}

/**
 * Resolve a curve ID to its full geometry.
 * Follows SURFACE_CURVE → actual curve type → resolved geometry.
 */
function resolveCurve(model: StepModel, curveId: number): ResolvedCurve | null {
  // Check if it's a SURFACE_CURVE wrapper
  const surfaceCurve = model.surfaceCurves.get(curveId);
  if (surfaceCurve) {
    curveId = surfaceCurve.curve3dId;
  }

  // Check for LINE
  const line = model.lines.get(curveId);
  if (line) {
    const originPoint = model.points.get(line.pointId);
    if (!originPoint) {
      return null;
    }
    const vector = model.vectors.get(line.vectorId);
    if (!vector) {
      return null;
    }
    const dir = model.directions.get(vector.directionId);
    if (!dir) {
      return null;
    }
    return {
      type: 'LINE',
      origin: originPoint.coords,
      direction: dir.dir,
    };
  }

  // Check for CIRCLE
  const circle = model.circles.get(curveId);
  if (circle) {
    const { origin, normal, refDirection } = resolveAxis2Placement(model, circle.placementId);
    return {
      type: 'CIRCLE',
      center: origin,
      normal,
      refDirection,
      radius: circle.radius,
    };
  }

  // Check for ELLIPSE
  const ellipse = model.ellipses.get(curveId);
  if (ellipse) {
    const { origin, normal, refDirection } = resolveAxis2Placement(model, ellipse.placementId);
    return {
      type: 'ELLIPSE',
      center: origin,
      normal,
      refDirection,
      majorRadius: ellipse.majorRadius,
      minorRadius: ellipse.minorRadius,
    };
  }

  // Check for B_SPLINE (not yet parsed - TODO in next phase)
  const bspline = model.bsplines.get(curveId);
  if (bspline) {
    // Resolve control points
    const controlPoints: Vec3[] = [];
    for (const cpId of bspline.controlPointIds) {
      const cp = model.points.get(cpId);
      if (!cp) {
        return null;
      }
      controlPoints.push(cp.coords);
    }

    // Expand knots with multiplicities
    const expandedKnots: number[] = [];
    for (let i = 0; i < bspline.knots.length; i++) {
      const mult = bspline.knotMultiplicities[i] || 1;
      for (let j = 0; j < mult; j++) {
        expandedKnots.push(bspline.knots[i]);
      }
    }

    return {
      type: 'B_SPLINE',
      degree: bspline.degree,
      controlPoints,
      knots: expandedKnots,
      weights: bspline.weights,
      closed: bspline.closed,
    };
  }

  // Unknown curve type
  return null;
}

/** Export curve resolution functions and types for use in curve-sampling.ts */
export { resolveCurve, resolveAxis2Placement };
export type { Vec3, Vec2, ResolvedCurve, ResolvedCircle, ResolvedEllipse, ResolvedBSpline, ResolvedLine };

/** Export internal functions for profiling and testing */
export {
    parseStep,
    extractFaceBounds,
    extractFaceBoundsWithCurves,
    computeFaceBasisFromStepFace,
    projectFaceLoopsTo2D,
    normalizeWinding,
    applyWindingTo3D,
    validateTopology,
    bridgeAllHoles,
    tryTessellateCurvedSurface,
};

// =============================================================================
// Batched Processing (Most Efficient)
// =============================================================================

interface PreparedFace {
  polygon2d: Vec2[];
  vertices3d: Vec3[];
  isCurved: boolean;
  curvedResult?: { vertices: Vec3[]; triangles: [number, number, number][] };
}

/**
 * Prepare a face for batched triangulation (extract polygon without triangulating).
 */
async function prepareFaceForBatch(
  model: StepModel,
  face: AdvancedFace,
  faceIndex: number
): Promise<PreparedFace> {
  // Check for curved surfaces first
  const curvedResult = await tryTessellateCurvedSurface(model, face);
  if (curvedResult) {
    return {
      polygon2d: [],
      vertices3d: [],
      isCurved: true,
      curvedResult,
    };
  }

  // Extract and process face bounds
  const { outer, holes } = await extractFaceBoundsWithCurves(model, face);
  const basis = computeFaceBasisFromStepFace(model, face, outer);
  const { outer2d, holes2d } = projectFaceLoopsTo2D({ outer, holes }, basis);
  const normalized = normalizeWinding({ outer2d, holes2d });
  const oriented3d = applyWindingTo3D(
    { outer, holes },
    normalized.outerReversed,
    normalized.holesReversed
  );

  const topology = validateTopology(normalized.outer2d, normalized.holes2d);
  if (!topology.valid) {
  }

  const mergedPolygon2d = bridgeAllHoles(normalized.outer2d, normalized.holes2d);

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
  const filteredMerged2d: Vec2[] = [];
  for (let i = 0; i < mergedPolygon2d.length; i++) {
    const curr = mergedPolygon2d[i];
    const prev = filteredMerged2d.length > 0
      ? filteredMerged2d[filteredMerged2d.length - 1]
      : mergedPolygon2d[mergedPolygon2d.length - 1];
    const dx = curr[0] - prev[0];
    const dy = curr[1] - prev[1];
    if (dx * dx + dy * dy > 1e-12) {
      filteredMerged2d.push(curr);
    }
  }

  if (filteredMerged2d.length > 1) {
    const first = filteredMerged2d[0];
    const last = filteredMerged2d[filteredMerged2d.length - 1];
    const dx = first[0] - last[0];
    const dy = first[1] - last[1];
    if (dx * dx + dy * dy < 1e-12) {
      filteredMerged2d.pop();
    }
  }

  // Build 3D positions
  const vertices3d: Vec3[] = [];
  for (const pt2d of filteredMerged2d) {
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

  return {
    polygon2d: filteredMerged2d,
    vertices3d,
    isCurved: false,
  };
}

/**
 * Parse STEP file using batched GPU processing.
 * This is the most efficient version - processes ALL faces in a single GPU dispatch.
 */
export async function parseStepToMeshBatched(stepText: string): Promise<Mesh> {
  const totalStart = performance.now();
  const parseStart = performance.now();

  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  const faces = [...model.faces.values()];

  // Phase 1: Prepare all faces (extract polygons, but don't triangulate yet)
  const preparedFaces: PreparedFace[] = [];
  for (let i = 0; i < faces.length; i++) {
    try {
      const prepared = await prepareFaceForBatch(model, faces[i], i);
      preparedFaces.push(prepared);
    } catch (e) {
      preparedFaces.push({ polygon2d: [], vertices3d: [], isCurved: false });
    }
  }

  const parseEnd = performance.now();
  const triangulationStart = performance.now();

  // Phase 2: Batch triangulate all non-curved faces
  const polygonsForBatch: BatchedPolygon[] = [];
  const batchIndexToFaceIndex: number[] = [];

  for (let i = 0; i < preparedFaces.length; i++) {
    const face = preparedFaces[i];
    if (!face.isCurved && face.polygon2d.length >= 3) {
      polygonsForBatch.push({ points: face.polygon2d });
      batchIndexToFaceIndex.push(i);
    }
  }

  // Run batched ear clipping (single GPU dispatch for ALL polygons!)
  const batchResult = await earClippingBatched(polygonsForBatch);

  // Phase 3: Combine all results
  const allVertices: Vec3[] = [];
  const allIndices: number[] = [];
  let vertexOffset = 0;

  for (let i = 0; i < preparedFaces.length; i++) {
    const face = preparedFaces[i];

    if (face.isCurved && face.curvedResult) {
      // Use pre-computed curved surface result
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
    } else if (face.vertices3d.length >= 3) {
      // Find the batch result for this face
      const batchIdx = batchIndexToFaceIndex.indexOf(i);
      if (batchIdx !== -1) {
        const triangles = batchResult.triangles[batchIdx];

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
  }

  const triangulationEnd = performance.now();

  // Build final mesh
  const positions = new Float32Array(allVertices.length * 3);
  allVertices.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  const indices = new Uint32Array(allIndices);
  const totalEnd = performance.now();

  const parseTime = parseEnd - parseStart;
  const triangulationTime = triangulationEnd - triangulationStart;
  const totalTime = totalEnd - totalStart;

  const triangleCount = allIndices.length / 3;

  return { positions, indices, parseTime, triangulationTime, totalTime };
}

