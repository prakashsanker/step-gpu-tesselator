/**
 * Fast Hybrid Triangulation
 *
 * Uses the optimal algorithm based on polygon size:
 * - earcut (CPU): Small polygons (<50 vertices) - fast JS, no GPU overhead
 * - ear-clipping-optimized (GPU): Medium polygons (50-256 vertices) - single dispatch
 * - ear-clipping-parallel (GPU): Large polygons (>256 vertices) - parallel multi-dispatch
 */

import earcut from 'earcut';
import { earClippingOptimized } from './ear-clipping-optimized';
import { earClippingParallel } from './ear-clipping-parallel';

// Thresholds for algorithm selection
const EARCUT_THRESHOLD = 50;
const OPTIMIZED_THRESHOLD = 256;

type Vec2 = [number, number];
type Vec3 = [number, number, number];

/**
 * Triangulate a simple polygon using the optimal algorithm.
 */
export async function triangulateFast(points: Vec2[] | Vec3[]): Promise<number[][]> {
  const n = points.length;

  if (n < 3) return [];
  if (n === 3) return [[0, 1, 2]];

  if (n < EARCUT_THRESHOLD) {
    return triangulateWithEarcut(points);
  }

  if (n <= OPTIMIZED_THRESHOLD) {
    return triangulateWithGPUOptimized(points);
  }

  return triangulateWithGPUParallel(points);
}

/**
 * Triangulate with earcut (fast CPU implementation).
 */
export function triangulateWithEarcut(points: Vec2[] | Vec3[]): number[][] {
  const flatCoords: number[] = [];
  for (const p of points) {
    flatCoords.push(p[0], p[1]);
  }

  const indices = earcut(flatCoords);

  const triangles: number[][] = [];
  for (let i = 0; i < indices.length; i += 3) {
    triangles.push([indices[i], indices[i + 1], indices[i + 2]]);
  }

  return triangles;
}

/**
 * Triangulate with GPU-optimized ear clipping (single dispatch).
 */
async function triangulateWithGPUOptimized(points: Vec2[] | Vec3[]): Promise<number[][]> {
  const points3d: Vec3[] = points.map(p =>
    p.length === 3 ? p as Vec3 : [p[0], p[1], 0]
  );
  return earClippingOptimized(points3d);
}

/**
 * Triangulate with GPU-parallel ear clipping.
 */
async function triangulateWithGPUParallel(points: Vec2[] | Vec3[]): Promise<number[][]> {
  const points3d: Vec3[] = points.map(p =>
    p.length === 3 ? p as Vec3 : [p[0], p[1], 0]
  );
  return earClippingParallel(points3d);
}

// Helper to compute signed area (positive = CCW, negative = CW)
function computeSignedArea(vertices: Vec2[]): number {
  let area = 0;
  for (let i = 0; i < vertices.length; i++) {
    const j = (i + 1) % vertices.length;
    area += vertices[i][0] * vertices[j][1];
    area -= vertices[j][0] * vertices[i][1];
  }
  return area / 2;
}

/**
 * Triangulate a polygon with holes using earcut.
 */
export function triangulateWithHoles(outer: Vec2[], holes: Vec2[][]): number[][] {
  console.log(`[triangulateWithHoles] outer: ${outer.length} vertices, holes: ${holes.length} (sizes: ${holes.map(h => h.length).join(', ')})`);

  // Check winding orders
  const outerArea = computeSignedArea(outer);
  const outerWinding = outerArea > 0 ? 'CCW' : 'CW';
  console.log(`[triangulateWithHoles] Outer winding: ${outerWinding} (signed area: ${outerArea.toFixed(3)})`);

  // Debug: dump vertices when area is near zero
  if (Math.abs(outerArea) < 1) {
    console.error(`[triangulateWithHoles] DEGENERATE OUTER (area near zero)! First 5 vertices:`);
    for (let i = 0; i < Math.min(5, outer.length); i++) {
      console.error(`  [${i}]: (${outer[i][0].toFixed(6)}, ${outer[i][1].toFixed(6)})`);
    }
    console.error(`  ... last vertex: (${outer[outer.length-1][0].toFixed(6)}, ${outer[outer.length-1][1].toFixed(6)})`);
  }

  for (let h = 0; h < holes.length; h++) {
    const holeArea = computeSignedArea(holes[h]);
    const holeWinding = holeArea > 0 ? 'CCW' : 'CW';
    console.log(`[triangulateWithHoles] Hole ${h} winding: ${holeWinding} (signed area: ${holeArea.toFixed(3)})`);

    // Debug: dump vertices when area is near zero
    if (Math.abs(holeArea) < 1) {
      console.error(`[triangulateWithHoles] DEGENERATE HOLE ${h} (area near zero)! First 5 vertices:`);
      const hole = holes[h];
      for (let i = 0; i < Math.min(5, hole.length); i++) {
        console.error(`  [${i}]: (${hole[i][0].toFixed(6)}, ${hole[i][1].toFixed(6)})`);
      }
      console.error(`  ... last vertex: (${hole[hole.length-1][0].toFixed(6)}, ${hole[hole.length-1][1].toFixed(6)})`);
    }

    if (outerWinding === holeWinding) {
      console.error(`[triangulateWithHoles] ERROR: Outer and hole ${h} have SAME winding (both ${outerWinding})! This will cause earcut to fail.`);
    }
  }

  // Check for duplicate vertices in outer
  const EPSILON = 1e-9;
  let duplicateCount = 0;
  for (let i = 0; i < outer.length; i++) {
    const next = (i + 1) % outer.length;
    const dx = outer[i][0] - outer[next][0];
    const dy = outer[i][1] - outer[next][1];
    if (Math.sqrt(dx*dx + dy*dy) < EPSILON) {
      duplicateCount++;
      if (duplicateCount <= 5) {
        console.warn(`[triangulateWithHoles] DUPLICATE in outer at index ${i}: (${outer[i][0].toFixed(6)}, ${outer[i][1].toFixed(6)})`);
      }
    }
  }
  if (duplicateCount > 0) {
    console.warn(`[triangulateWithHoles] Found ${duplicateCount} duplicate consecutive vertices in outer!`);
  }

  // Check for duplicate vertices in holes
  for (let h = 0; h < holes.length; h++) {
    const hole = holes[h];
    let holeDups = 0;
    for (let i = 0; i < hole.length; i++) {
      const next = (i + 1) % hole.length;
      const dx = hole[i][0] - hole[next][0];
      const dy = hole[i][1] - hole[next][1];
      if (Math.sqrt(dx*dx + dy*dy) < EPSILON) {
        holeDups++;
      }
    }
    if (holeDups > 0) {
      console.warn(`[triangulateWithHoles] Found ${holeDups} duplicate consecutive vertices in hole ${h}!`);
    }
  }

  // Log bounding boxes to check spatial relationship
  let outerMinX = Infinity, outerMinY = Infinity, outerMaxX = -Infinity, outerMaxY = -Infinity;
  for (const p of outer) {
    outerMinX = Math.min(outerMinX, p[0]);
    outerMinY = Math.min(outerMinY, p[1]);
    outerMaxX = Math.max(outerMaxX, p[0]);
    outerMaxY = Math.max(outerMaxY, p[1]);
  }
  console.log(`[triangulateWithHoles] Outer bbox: (${outerMinX.toFixed(3)}, ${outerMinY.toFixed(3)}) to (${outerMaxX.toFixed(3)}, ${outerMaxY.toFixed(3)})`);

  for (let h = 0; h < holes.length; h++) {
    let holeMinX = Infinity, holeMinY = Infinity, holeMaxX = -Infinity, holeMaxY = -Infinity;
    for (const p of holes[h]) {
      holeMinX = Math.min(holeMinX, p[0]);
      holeMinY = Math.min(holeMinY, p[1]);
      holeMaxX = Math.max(holeMaxX, p[0]);
      holeMaxY = Math.max(holeMaxY, p[1]);
    }
    console.log(`[triangulateWithHoles] Hole ${h} bbox: (${holeMinX.toFixed(3)}, ${holeMinY.toFixed(3)}) to (${holeMaxX.toFixed(3)}, ${holeMaxY.toFixed(3)})`);

    // Check if hole is inside outer
    const holeInside = holeMinX >= outerMinX && holeMaxX <= outerMaxX &&
                       holeMinY >= outerMinY && holeMaxY <= outerMaxY;
    if (!holeInside) {
      console.error(`[triangulateWithHoles] ERROR: Hole ${h} bbox extends outside outer bbox!`);
    }
  }

  const flatCoords: number[] = [];
  const holeIndices: number[] = [];

  for (const p of outer) {
    flatCoords.push(p[0], p[1]);
  }

  let currentIndex = outer.length;
  for (const hole of holes) {
    holeIndices.push(currentIndex);
    for (const p of hole) {
      flatCoords.push(p[0], p[1]);
    }
    currentIndex += hole.length;
  }

  console.log(`[triangulateWithHoles] flatCoords: ${flatCoords.length / 2} points, holeIndices: [${holeIndices.join(', ')}]`);

  const indices = earcut(flatCoords, holeIndices);
  const triangleCount = indices.length / 3;
  const expectedTriangles = outer.length + holes.reduce((sum, h) => sum + h.length, 0) + holes.length - 2;
  console.log(`[triangulateWithHoles] earcut returned ${triangleCount} triangles (expected ~${expectedTriangles})`);

  if (triangleCount < expectedTriangles * 0.7) {
    console.error(`[triangulateWithHoles] WARNING: earcut returned significantly fewer triangles than expected!`);
  }

  const triangles: number[][] = [];
  for (let i = 0; i < indices.length; i += 3) {
    triangles.push([indices[i], indices[i + 1], indices[i + 2]]);
  }

  return triangles;
}
