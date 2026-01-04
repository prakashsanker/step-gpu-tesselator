/**
 * Hybrid Triangulation
 *
 * Combines GPU batched ear clipping for small polygons with
 * fast CPU ear clipping for large polygons.
 *
 * This approach:
 * - Batches all small polygons (≤256 vertices) for a single GPU dispatch
 * - Uses efficient CPU ear clipping for large polygons (>256 vertices)
 * - Achieves maximum performance across all polygon sizes
 */

import { earClippingBatched, type BatchedPolygon } from "./ear-clipping-batched";
import { triangulateSimple } from "./monotone-decomposition";

type Vec2 = [number, number];

const GPU_VERTEX_LIMIT = 256;

export interface HybridResult {
    triangles: number[][][];  // Triangles for each input polygon
    timing: {
        total: number;
        gpuBatch: number;
        cpuFallback: number;
    };
}

/**
 * Triangulate multiple polygons using a hybrid GPU/CPU approach.
 *
 * @param polygons Array of 2D polygons to triangulate
 * @returns Triangles for each polygon plus timing information
 */
export async function triangulateHybrid(polygons: Vec2[][]): Promise<HybridResult> {
    const totalStart = performance.now();
    const results: (number[][] | null)[] = new Array(polygons.length).fill(null);

    // Separate into GPU-eligible and CPU-fallback
    const gpuPolygons: { index: number; points: Vec2[] }[] = [];
    const cpuPolygons: { index: number; points: Vec2[] }[] = [];

    for (let i = 0; i < polygons.length; i++) {
        const polygon = polygons[i];
        if (polygon.length < 3) {
            results[i] = [];
        } else if (polygon.length === 3) {
            results[i] = [[0, 1, 2]];
        } else if (polygon.length <= GPU_VERTEX_LIMIT) {
            gpuPolygons.push({ index: i, points: polygon });
        } else {
            cpuPolygons.push({ index: i, points: polygon });
        }
    }

    // GPU batch processing
    const gpuStart = performance.now();
    if (gpuPolygons.length > 0) {
        const batchInput: BatchedPolygon[] = gpuPolygons.map(p => ({ points: p.points }));
        const batchResult = await earClippingBatched(batchInput);

        for (let i = 0; i < gpuPolygons.length; i++) {
            results[gpuPolygons[i].index] = batchResult.triangles[i];
        }
    }
    const gpuTime = performance.now() - gpuStart;

    // CPU fallback for large polygons
    const cpuStart = performance.now();
    for (const { index, points } of cpuPolygons) {
        results[index] = triangulateSimple(points);
    }
    const cpuTime = performance.now() - cpuStart;

    const totalTime = performance.now() - totalStart;

    return {
        triangles: results as number[][][],
        timing: {
            total: totalTime,
            gpuBatch: gpuTime,
            cpuFallback: cpuTime,
        },
    };
}

/**
 * Triangulate a single polygon using the hybrid approach.
 * For individual polygons, this adds some overhead - use triangulateHybrid for batches.
 */
export async function triangulateSingle(polygon: Vec2[]): Promise<number[][]> {
    if (polygon.length < 3) return [];
    if (polygon.length === 3) return [[0, 1, 2]];

    if (polygon.length <= GPU_VERTEX_LIMIT) {
        const result = await earClippingBatched([{ points: polygon }]);
        return result.triangles[0];
    } else {
        return triangulateSimple(polygon);
    }
}
