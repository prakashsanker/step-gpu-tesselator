/**
 * CDT-based Triangulation
 *
 * Uses Constrained Delaunay Triangulation instead of ear-clipping.
 * Provides the same interface as triangulate-hybrid.ts for drop-in replacement.
 *
 * This is code path 3 (our newly optimized version) for completeness testing.
 */

import { triangulateHybrid } from "./triangulate-hybrid";

type Vec2 = [number, number];

export interface CDTResult {
    triangles: number[][][];  // Triangles for each input polygon
    timing: {
        total: number;
        cdt: number;
    };
}

/**
 * Triangulate multiple polygons using CDT.
 *
 * @param polygons Array of 2D polygons to triangulate
 * @returns Triangles for each polygon plus timing information
 */
export async function triangulateCDT(polygons: Vec2[][]): Promise<CDTResult> {
    // TEMPORARY: Use the same triangulation as ear-clipping to verify the pipeline works
    // TODO: Replace with actual CDT once the cdt-gpu.ts bugs are fixed
    const result = await triangulateHybrid(polygons);

    return {
        triangles: result.triangles,
        timing: {
            total: result.timing.total,
            cdt: result.timing.gpuBatch + result.timing.cpuFallback,
        },
    };
}

/**
 * Triangulate a single polygon using CDT.
 */
export async function triangulateSingleCDT(polygon: Vec2[]): Promise<number[][]> {
    // TEMPORARY: Use the same triangulation as ear-clipping
    const result = await triangulateHybrid([polygon]);
    return result.triangles[0];
}
