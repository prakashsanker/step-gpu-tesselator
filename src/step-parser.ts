import { earClipping } from "./ear-clipping";
// Minimal STEP → mesh parser for the square face example
type Vec3 = [number, number, number];
type Vec2 = [number, number];

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
          console.log("[Projection] Using STEP plane basis");
          return { origin, u, v, n };
        }
      }
    }
  }

  // Fallback: compute basis from outer loop geometry
  console.log("[Projection] Falling back to geometric basis from outer loop");
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
    console.log(`[Winding] Outer loop reversed: area was ${outerArea.toFixed(4)} (CW → CCW)`);
  } else {
    console.log(`[Winding] Outer loop OK: area is ${outerArea.toFixed(4)} (CCW)`);
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
      console.log(`[Winding] Hole ${i} reversed: area was ${holeArea.toFixed(4)} (CCW → CW)`);
    } else {
      normalizedHoles.push(hole);
      console.log(`[Winding] Hole ${i} OK: area is ${holeArea.toFixed(4)} (CW)`);
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
    // This shouldn't happen if topology validation passed (C2.4)
    console.error("[Bridging] Ray cast failed - hole may be outside outer polygon");
    return 0;
  }

  const { edgeStartIndex, intersectionX, edgeParameter } = rayHit;

  // STEP 2: Get the edge endpoints
  const edgeStart = outerPolygon[edgeStartIndex];
  const edgeEnd = outerPolygon[(edgeStartIndex + 1) % outerPolygon.length];

  console.log(`[Bridging] Ray hit edge ${edgeStartIndex}→${(edgeStartIndex + 1) % outerPolygon.length} at x=${intersectionX.toFixed(3)}`);

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

  console.log(`[Bridging] Candidate vertex M: index=${candidateIndex}, coords=(${candidateVertex[0].toFixed(3)}, ${candidateVertex[1].toFixed(3)})`);

  // STEP 4: Check if we can directly see M from P
  const skipIndices = new Set([candidateIndex]);
  // Also skip the adjacent edges
  const prevIndex = (candidateIndex - 1 + outerPolygon.length) % outerPolygon.length;
  const nextIndex = (candidateIndex + 1) % outerPolygon.length;
  skipIndices.add(prevIndex);

  if (isVisible(holeVertex, candidateVertex, outerPolygon, skipIndices)) {
    console.log(`[Bridging] Direct visibility to M confirmed`);
    return candidateIndex;
  }

  // STEP 5: M is not directly visible
  // We need to find a "reflex" vertex inside triangle P-I-M
  // that IS visible from P

  console.log(`[Bridging] M not directly visible, searching for alternate vertex`);

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

  console.log(`[Bridging] Selected vertex: index=${bestIndex}`);
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

  console.log(`[Bridging] Hole rightmost vertex P: index=${holeRightmostIndex}, coords=(${holeVertex[0].toFixed(3)}, ${holeVertex[1].toFixed(3)})`);

  // STEP 2: Find the vertex on outer to connect to (this is M)
  const outerTargetIndex = findBridgeTargetVertex(holeVertex, outer);
  const outerVertex = outer[outerTargetIndex];

  console.log(`[Bridging] Bridge target M: index=${outerTargetIndex}, coords=(${outerVertex[0].toFixed(3)}, ${outerVertex[1].toFixed(3)})`);

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

  console.log(`[Bridging] Merged: ${outerLen} outer + ${holeLen} hole → ${merged.length} total vertices`);

  // Debug: verify merged polygon is CCW
  const mergedArea = computeSignedArea2D(merged);
  console.log(`[Bridging] Merged polygon signed area: ${mergedArea.toFixed(4)} (should be positive for CCW)`);
  if (mergedArea < 0) {
    console.warn(`[Bridging] WARNING: Merged polygon is CW (negative area), reversing!`);
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

  console.log(`[Bridging] Starting to bridge ${holes.length} hole(s)`);

  // STEP 1: Sort holes by rightmost X coordinate (descending = right to left)
  const holesWithRightmostX = holes.map((hole, index) => {
    const rightmostIndex = findRightmostVertexIndex(hole);
    const rightmostX = hole[rightmostIndex][0];
    return { hole, index, rightmostX };
  });

  holesWithRightmostX.sort((a, b) => b.rightmostX - a.rightmostX);

  console.log(`[Bridging] Processing order (right to left): ${holesWithRightmostX.map(h => `hole${h.index}(x=${h.rightmostX.toFixed(2)})`).join(', ')}`);

  // STEP 2: Merge each hole one by one
  let currentPolygon = outer;

  for (let i = 0; i < holesWithRightmostX.length; i++) {
    const { hole, index } = holesWithRightmostX[i];
    console.log(`[Bridging] Processing hole ${i + 1}/${holes.length} (original index ${index})`);

    currentPolygon = mergeHoleIntoOuter(currentPolygon, hole);
  }

  console.log(`[Bridging] Complete! Final polygon has ${currentPolygon.length} vertices`);

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
    console.warn(`[Projection] ${loopName}: signed area is ~0 (${signedArea.toExponential(2)}), polygon may be degenerate`);
  }
  if (!planarityValid) {
    console.warn(`[Projection] ${loopName}: max Z deviation = ${maxZDeviation.toExponential(2)}, points may not be coplanar`);
  }
  if (areaValid && planarityValid) {
    console.log(`[Projection] ${loopName}: OK (area=${signedArea.toFixed(4)}, maxZ=${maxZDeviation.toExponential(2)})`);
  }

  return { valid: areaValid && planarityValid, signedArea, maxZDeviation };
}

export interface Mesh {
  positions: Float32Array;
  indices: Uint32Array;
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
}

// --- Public API: parse STEP text into a Mesh (one face) ---

export async function parseStepToMesh(stepText: string): Promise<Mesh> {
  // This function is browser-safe: it expects the STEP file contents as a string,
  // leaving file I/O (File API, fetch, Node fs, etc.) to the caller.

  const model = parseStep(stepText);
  if (model.faces.size === 0) {
    throw new Error("No ADVANCED_FACE found in STEP file.");
  }

  // For this minimal example, just take the first face
  const face = [...model.faces.values()][0];

  // C2.1: Extract outer boundary and holes using helper functions
  const { outer, holes } = extractFaceBounds(model, face);

  // C2.2: Compute face basis and project to 2D
  const basis = computeFaceBasisFromStepFace(model, face, outer);
  const { outer2d, holes2d } = projectFaceLoopsTo2D({ outer, holes }, basis);

  // Debug: verify projection sanity
  debugVerifyProjection(outer, outer2d, basis, "outer");
  for (let i = 0; i < holes.length; i++) {
    debugVerifyProjection(holes[i], holes2d[i], basis, `hole[${i}]`);
  }

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
    console.warn("[StepParser] Topology validation failed:");
    for (const error of topology.errors) {
      console.warn(`  - ${error}`);
    }
    // For now, continue anyway but warn. Could throw in strict mode.
  } else {
    console.log("[StepParser] Topology validation passed");
  }

  // ==========================================================================
  // C2.5: Bridge holes and triangulate
  // ==========================================================================

  // Log hole information
  if (normalized.holes2d.length > 0) {
    console.log(`[StepParser] Face has ${normalized.holes2d.length} hole(s)`);
    for (let i = 0; i < normalized.holes2d.length; i++) {
      const holeArea = computeSignedArea2D(normalized.holes2d[i]);
      console.log(`[StepParser] Hole ${i}: ${normalized.holes2d[i].length} vertices, area=${holeArea.toFixed(4)} (CW)`);
    }
  }

  // Bridge holes into outer polygon (if there are any holes)
  const mergedPolygon2d = bridgeAllHoles(normalized.outer2d, normalized.holes2d);

  console.log(`[StepParser] Merged polygon has ${mergedPolygon2d.length} vertices`);

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
  // (bridge points create duplicates that can confuse the algorithm)
  const filteredMerged2d: Vec2[] = [];
  for (let i = 0; i < mergedPolygon2d.length; i++) {
    const curr = mergedPolygon2d[i];
    const prev = filteredMerged2d.length > 0
      ? filteredMerged2d[filteredMerged2d.length - 1]
      : mergedPolygon2d[mergedPolygon2d.length - 1];

    // Skip if this vertex is at same position as previous
    const dx = curr[0] - prev[0];
    const dy = curr[1] - prev[1];
    const distSq = dx * dx + dy * dy;

    if (distSq > 1e-12) {
      filteredMerged2d.push(curr);
    } else {
      console.log(`[StepParser] Filtered duplicate vertex at index ${i}`);
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
      console.log(`[StepParser] Filtered wrap-around duplicate vertex`);
    }
  }

  console.log(`[StepParser] Filtered from ${mergedPolygon2d.length} to ${filteredMerged2d.length} vertices`);

  // Build 3D positions array from filtered vertices
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
      console.warn(`[StepParser] Could not find 3D coordinate for 2D point (${pt2d[0]}, ${pt2d[1]})`);
      pt3d = [pt2d[0], pt2d[1], 0];
    }
    filtered3d.push(pt3d);
  }

  // Build positions array from filtered 3D vertices
  const positions = new Float32Array(filtered3d.length * 3);
  filtered3d.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  // Convert 2D points to Vec3 with z=0 for ear clipping
  const filtered2dAsVec3: Vec3[] = filteredMerged2d.map(p => [p[0], p[1], 0]);

  // Run ear clipping on the filtered (bridged) polygon
  const triangles = await earClipping(filtered2dAsVec3);

  console.log("[StepParser] Triangulation complete:", triangles.length, "triangles");

  // Convert triangles array to flat indices array
  const indicesArray: number[] = [];
  for (const triangle of triangles) {
    indicesArray.push(triangle[0], triangle[1], triangle[2]);
  }
  const indices = new Uint32Array(indicesArray);

  console.log("[StepParser] Final mesh:", filtered3d.length, "vertices,", triangles.length, "triangles");

  return { positions, indices };
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
  };

  // Remove comments (/* ... */, / ... */, and -- ... end-of-line)
  let text = stepText.replace(/\/\*[\s\S]*?\*\//g, "");     // block comments /* ... */
  text = text.replace(/\/[^*][\s\S]*?\*\//g, "");          // block comments / ... */ (single slash)
  text = text.replace(/--.*$/gm, "");                       // line comments

  // Split into lines and process entity lines starting with '#'
  const lines = text.split(/\r?\n/);

  const entityRegex = /^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\);?$/;

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed.startsWith("#")) continue;
    const match = trimmed.match(entityRegex);
    if (!match) continue;

    const id = parseInt(match[1], 10);
    const type = match[2];
    const args = match[3]; // raw argument string inside (...)

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
      // We ignore other entity types (LINE, VECTOR, etc.) for now
    }
  }

  return model;
}

// --- Individual entity parsers (all tailored to our example syntax) ---

function parseCartesianPoint(id: number, args: string, model: StepModel) {
  // CARTESIAN_POINT('', (x, y, z))
  const coordMatch = args.match(/\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)\s*$/);
  if (!coordMatch) {
    throw new Error(`Failed to parse CARTESIAN_POINT args: ${args}`);
  }
  const x = parseFloat(coordMatch[1]);
  const y = parseFloat(coordMatch[2]);
  const z = parseFloat(coordMatch[3]);

  model.points.set(id, { id, coords: [x, y, z] });
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
  // DIRECTION('', (0.0, 0.0, 1.0))
  const coordMatch = args.match(/\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)\s*$/);
  if (!coordMatch) {
    throw new Error(`Failed to parse DIRECTION args: ${args}`);
  }
  const x = parseFloat(coordMatch[1]);
  const y = parseFloat(coordMatch[2]);
  const z = parseFloat(coordMatch[3]);

  model.directions.set(id, { id, dir: [x, y, z] });
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

