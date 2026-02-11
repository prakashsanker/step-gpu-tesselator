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

// Thresholds for algorithm selection.
// Empirically, earcut outperforms our GPU paths for many medium polygons once
// dispatch/setup overhead is included, so keep CPU coverage broad and reserve
// GPU paths for truly large loops.
const EARCUT_THRESHOLD = 1024;
const OPTIMIZED_THRESHOLD = 4096;

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
  const debug = (globalThis as any)?.__TRIANGULATE_HOLES_DEBUG__ === true;

  if (debug) {
    console.log(`[triangulateWithHoles] outer: ${outer.length} vertices, holes: ${holes.length} (sizes: ${holes.map(h => h.length).join(', ')})`);

    const outerArea = computeSignedArea(outer);
    const outerWinding = outerArea > 0 ? 'CCW' : 'CW';
    console.log(`[triangulateWithHoles] Outer winding: ${outerWinding} (signed area: ${outerArea.toFixed(3)})`);

    if (Math.abs(outerArea) < 1) {
      console.error(`[triangulateWithHoles] DEGENERATE OUTER (area near zero)`);
    }

    for (let h = 0; h < holes.length; h++) {
      const holeArea = computeSignedArea(holes[h]);
      const holeWinding = holeArea > 0 ? 'CCW' : 'CW';
      console.log(`[triangulateWithHoles] Hole ${h} winding: ${holeWinding} (signed area: ${holeArea.toFixed(3)})`);
      if (outerWinding === holeWinding) {
        console.error(`[triangulateWithHoles] ERROR: Outer and hole ${h} have same winding`);
      }
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

  const indices = earcut(flatCoords, holeIndices);
  if (debug) {
    const triangleCount = indices.length / 3;
    const expectedTriangles = outer.length + holes.reduce((sum, h) => sum + h.length, 0) + holes.length - 2;
    console.log(`[triangulateWithHoles] flatCoords: ${flatCoords.length / 2} points, holeIndices: [${holeIndices.join(', ')}]`);
    console.log(`[triangulateWithHoles] earcut returned ${triangleCount} triangles (expected ~${expectedTriangles})`);
    if (triangleCount < expectedTriangles * 0.7) {
      console.error(`[triangulateWithHoles] WARNING: earcut returned significantly fewer triangles than expected`);
    }
  }

  const triangles: number[][] = [];
  for (let i = 0; i < indices.length; i += 3) {
    triangles.push([indices[i], indices[i + 1], indices[i + 2]]);
  }

  return triangles;
}
