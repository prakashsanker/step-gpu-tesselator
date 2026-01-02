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

  // Build positions array from oriented 3D outer boundary (for rendering)
  const positions = new Float32Array(oriented3d.outer.length * 3);
  oriented3d.outer.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  // Log hole information for now (triangulation with holes not yet implemented)
  if (normalized.holes2d.length > 0) {
    console.log(`[StepParser] Face has ${normalized.holes2d.length} hole(s) - hole triangulation not yet implemented`);
    for (let i = 0; i < normalized.holes2d.length; i++) {
      const holeArea = computeSignedArea2D(normalized.holes2d[i]);
      console.log(`[StepParser] Hole ${i}: ${normalized.holes2d[i].length} vertices, area=${holeArea.toFixed(4)} (should be negative/CW)`);
    }
  }

  // Convert 2D points to Vec3 with z=0 for ear clipping
  // (ear clipping algorithm expects Vec3 but operates in 2D)
  const outer2dAsVec3: Vec3[] = normalized.outer2d.map(p => [p[0], p[1], 0]);

  // Use ear clipping algorithm for triangulation (outer boundary only for now)
  const triangles = await earClipping(outer2dAsVec3);

  console.log("[StepParser] Received triangles from earClipping:", triangles);

  // Convert triangles array to flat indices array
  const indicesArray: number[] = [];
  for (const triangle of triangles) {
    indicesArray.push(triangle[0], triangle[1], triangle[2]);
  }
  const indices = new Uint32Array(indicesArray);

  console.log("[StepParser] Final indices array:", Array.from(indices));
  console.log("[StepParser] Indices length:", indices.length);
  console.log("[StepParser] Expected triangle count:", triangles.length);

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

