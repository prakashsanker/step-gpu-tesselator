/**
 * Tessellator - Main API for STEP file parsing and tessellation
 *
 * This module provides the canonical API for parsing STEP CAD files and
 * converting them to triangulated meshes suitable for rendering.
 *
 * Pipeline: OpenCascade.js (STEP parsing + XCAF colors) -> GPU tessellation -> Mesh
 */

// Re-export the main parsing function from the OCC tessellator
export { parseStepWithOCC as parseStepFile } from './occ-test';

// Re-export the Mesh type and geometry utilities
export type { Mesh, Vec3, Vec2 } from './step-parser';
export {
  computeFaceBasisFromLoop,
  projectFaceLoopsTo2D,
  normalizeWinding,
  applyWindingTo3D,
  bridgeAllHoles,
} from './step-parser';

// Re-export profiling utilities
export {
  tessellationProfile,
  resetTessellationProfile,
  getTessellationProfileReport,
} from './occ-test';

/**
 * Parse a STEP file from a browser File object.
 * Convenience wrapper for handling File input from <input type="file">.
 *
 * @param file - Browser File object from file input
 * @returns Triangulated mesh with positions, indices, normals, and colors
 */
export async function parseBrowserFile(file: File): Promise<import('./step-parser').Mesh> {
  const { parseStepFile } = await import('./tessellator');

  // Read file as ArrayBuffer for large file support
  const arrayBuffer = await file.arrayBuffer();
  const uint8Array = new Uint8Array(arrayBuffer);

  return parseStepFile(uint8Array);
}
