/**
 * OpenCascade.js integration checkpoints
 * Goal: Parse STEP files with OCC and connect to existing GPU tessellator
 */

import type { Vec3, Mesh } from './step-parser';
import {
  computeFaceBasisFromLoop,
  projectFaceLoopsTo2D,
  normalizeWinding,
  applyWindingTo3D,
} from './step-parser';
import { earClipping } from './ear-clipping';
import { constrainedDelaunayTriangulation } from './cdt-gpu';
import { bridgeAllHoles } from './step-parser';
import { triangulateFast, triangulateWithHoles } from './triangulate-fast';

// Triangulation method type
export type TriangulationMethod = 'ear-clipping' | 'cdt';
import { createThreeMeshFromTesselation } from './threejs-render';
import * as occtimportjsModule from 'occt-import-js';
const occtimportjs = (occtimportjsModule as any).default || occtimportjsModule;
import { buildComprehensiveFaceColorMap, getAdvancedFaceIds, type RGBColor as ParsedRGBColor } from './step-color-parser';

// Profiling accumulator for tessellation functions
export const tessellationProfile = {
  // Top-level phases (STEP loading + face extraction)
  loadStepFile: { total: 0, calls: 0 },
  // loadStepFile sub-phases
  loadStepFile_initOC: { total: 0, calls: 0 },
  loadStepFile_createDoc: { total: 0, calls: 0 },
  loadStepFile_readFile: { total: 0, calls: 0 },
  loadStepFile_transfer: { total: 0, calls: 0 },
  loadStepFile_getTools: { total: 0, calls: 0 },
  loadStepFile_colorParsing: { total: 0, calls: 0 },
  extractFacesWithEdges: { total: 0, calls: 0 },
  // Tessellation phases
  occEdgesToPolygon: { total: 0, calls: 0 },
  computeFaceBasisFromLoop: { total: 0, calls: 0 },
  projectFaceLoopsTo2D: { total: 0, calls: 0 },
  normalizeWinding: { total: 0, calls: 0 },
  applyWindingTo3D: { total: 0, calls: 0 },
  bridgeAllHoles: { total: 0, calls: 0 },
  earClipping: { total: 0, calls: 0 },
  tessellatePlanarFace: { total: 0, calls: 0 },
  tessellateCurvedFace: { total: 0, calls: 0 },
  computeNormals: { total: 0, calls: 0 },
  meshAssembly: { total: 0, calls: 0 },
  tessellateOCCShape: { total: 0, calls: 0 },
};

export function resetTessellationProfile() {
  for (const key of Object.keys(tessellationProfile)) {
    tessellationProfile[key as keyof typeof tessellationProfile] = { total: 0, calls: 0 };
  }
}

export function getTessellationProfileReport(): string {
  const lines: string[] = [];
  // Total pipeline time = load + extract + tessellate
  const total = (tessellationProfile.loadStepFile.total || 0) +
                (tessellationProfile.extractFacesWithEdges.total || 0) +
                (tessellationProfile.tessellateOCCShape.total || 0) || 1;

  lines.push('=== FULL PIPELINE PROFILE ===');
  lines.push(`Total pipeline time: ${total.toFixed(2)}ms`);
  lines.push('');
  for (const [name, data] of Object.entries(tessellationProfile)) {
    if (data.calls > 0) {
      const pct = ((data.total / total) * 100).toFixed(1);
      const avg = (data.total / data.calls).toFixed(3);
      lines.push(`${name.padEnd(25)} ${data.total.toFixed(2).padStart(10)}ms (${pct.padStart(5)}%) | ${data.calls.toString().padStart(5)} calls | ${avg.padStart(8)}ms/call`);
    }
  }
  return lines.join('\n');
}
import {
  tessellateCylinder,
  tessellateSphere,
  tessellateCone,
  tessellateTorus,
  tessellateBSplineSurface,
  tessellateTrimmedSurface,
  type TrimmedSurfaceBuildOptions,
} from './surface-tessellation';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Set to true to enable debug logging (significantly impacts performance)
const DEBUG_OCC = false;
const DEBUG_TESSELLATION_VERBOSE = false;

function tessellationVerboseEnabled(): boolean {
  return DEBUG_TESSELLATION_VERBOSE || (globalThis as any)?.__TESSELLATION_VERBOSE_LOGS__ === true;
}

function tessellationVerboseLog(...args: unknown[]): void {
  if (tessellationVerboseEnabled()) {
    console.log(...args);
  }
}

function logOCC(...args: unknown[]): void {
  if (DEBUG_OCC) {
    console.log('[OCC]', ...args);
  }
}

function curveDebugLog(...args: unknown[]): void {
  if ((globalThis as any)?.__CURVE_VERBOSE_LOGS__ === true) {
    console.log(...args);
  }
}

function faceExtractionLog(...args: unknown[]): void {
  if ((globalThis as any)?.__FACE_EXTRACTION_LOGS__ === true) {
    console.log(...args);
  }
}

// OpenCascade instance type (using any for now since types aren't well-defined)
type OpenCascadeInstance = any;

/**
 * RGB color (0-1 range)
 */
interface RGBColor {
  r: number;
  g: number;
  b: number;
}

let oc: OpenCascadeInstance | null = null;

async function initOC(): Promise<OpenCascadeInstance> {
  if (oc) return oc;
  console.log('[OCC] Initializing OpenCascade.js v2 (with XCAF support)...');
  const startTime = performance.now();

  // Import the main OpenCascade module directly (not using initOpenCascade which has issues with Vite)
  // @ts-ignore
  const { default: opencascade } = await import('opencascade.js/dist/opencascade.js');

  // Initialize with locateFile to find WASM files
  oc = await opencascade({
    locateFile(path: string) {
      return '/node_modules/opencascade.js/dist/' + path;
    }
  });

  console.log(`[OCC] Core initialized in ${(performance.now() - startTime).toFixed(0)}ms`);

  // Load additional dynamic libraries needed for XCAF and Document framework
  // These must be loaded in dependency order
  const xcafLibs = [
    // Core geometry/topology
    'module.TKMath.wasm',
    'module.TKG2d.wasm',
    'module.TKG3d.wasm',
    'module.TKGeomBase.wasm',
    'module.TKGeomAlgo.wasm',
    'module.TKBRep.wasm',
    'module.TKTopAlgo.wasm',
    'module.TKShHealing.wasm',
    // Document framework
    'module.TKCDF.wasm',
    'module.TKLCAF.wasm',
    'module.TKCAF.wasm',
    // XCAF
    'module.TKVCAF.wasm',
    'module.TKXCAF.wasm',
    // STEP I/O
    'module.TKXSBase.wasm',
    'module.TKSTEPBase.wasm',
    'module.TKSTEP209.wasm',
    'module.TKSTEPAttr.wasm',
    'module.TKSTEP.wasm',
    'module.TKXDESTEP.wasm',
    // Meshing
    'module.TKMesh.wasm',
  ];

  console.log('[OCC] Loading dynamic libraries for XCAF support...');
  for (const lib of xcafLibs) {
    try {
      const libPath = '/node_modules/opencascade.js/dist/' + lib;
      await oc.loadDynamicLibrary(libPath, {loadAsync: true, global: true, nodelete: true, allowUndefined: true});
    } catch (err: any) {
      console.warn(`[OCC] Warning: Failed to load ${lib}: ${err.message}`);
    }
  }

  console.log(`[OCC] Full initialization completed in ${(performance.now() - startTime).toFixed(0)}ms`);

  // Check for XCAF APIs
  const hasXCAF = typeof oc.XCAFDoc_DocumentTool !== 'undefined';
  const hasTDFLabelSeq = typeof oc.TDF_LabelSequence_1 !== 'undefined';
  console.log(`[OCC] XCAF support: ${hasXCAF ? 'YES' : 'NO'}`);
  console.log(`[OCC] TDF_LabelSequence support: ${hasTDFLabelSeq ? 'YES' : 'NO'}`);

  // Debug APIs logged only when DEBUG_OCC is true
  if (DEBUG_OCC) {
    console.log('[OCC] B-spline surface APIs:', Object.keys(oc).filter(k => k.includes('BSpline') && k.includes('Surface')).slice(0, 20));
    const xcafApis = Object.keys(oc).filter(k => k.includes('XCAF') || k.includes('XDE'));
    const colorApis = Object.keys(oc).filter(k => k.includes('Color') && !k.includes('ColorScale'));
    const quantityApis = Object.keys(oc).filter(k => k.startsWith('Quantity_'));
    const tdfApis = Object.keys(oc).filter(k => k.startsWith('TDF_'));
    console.log('[OCC] XCAF/XDE APIs found:', xcafApis.length, xcafApis.slice(0, 10));
    console.log('[OCC] Color APIs found:', colorApis.length, colorApis.slice(0, 10));
    console.log('[OCC] Quantity APIs found:', quantityApis.length, quantityApis.slice(0, 10));
    console.log('[OCC] TDF APIs found:', tdfApis.length, tdfApis.slice(0, 10));
  }

  return oc;
}

/**
 * Result from loading a STEP file with color information
 */
interface StepLoadResult {
  shape: any;
  colorTool: any | null;
  shapeTool: any | null;
  doc: any | null;
  stepColors: Map<number, RGBColor>; // Fallback: colors parsed from STEP text, keyed by entity ID
  shapeColorMap: Map<number, RGBColor>; // Map from shape label tag to color (for propagation to child faces)
  faceIdOrder: number[]; // Ordered list of ADVANCED_FACE entity IDs from STEP text
  geometryColorMap: Map<string, RGBColor>; // Map from geometry key (vertex position) to color
  solidMatchedColors: Map<number, RGBColor>; // Map from face hash code to color (from solid matching)
  faceToSolid: Map<number, number>; // Map from face hash code to OCC solid index
  solidToColor: Map<number, RGBColor>; // Map from OCC solid index to color
}

/**
 * Parse colors directly from STEP file text as a fallback when XCAF is not available.
 * This parses COLOUR_RGB, DRAUGHTING_PRE_DEFINED_COLOUR, and STYLED_ITEM entities.
 */
function parseStepColors(stepContent: string): Map<number, RGBColor> {
  const colors = new Map<number, RGBColor>();

  // Parse COLOUR_RGB entities: #123=COLOUR_RGB('',0.8,0.2,0.1);
  const colorRgbRegex = /#(\d+)\s*=\s*COLOUR_RGB\s*\(\s*'[^']*'\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)/gi;
  let match;
  while ((match = colorRgbRegex.exec(stepContent)) !== null) {
    const id = parseInt(match[1]);
    const r = parseFloat(match[2]);
    const g = parseFloat(match[3]);
    const b = parseFloat(match[4]);
    colors.set(id, { r, g, b });
  }

  // Parse predefined colors: DRAUGHTING_PRE_DEFINED_COLOUR('red')
  const predefColorRegex = /#(\d+)\s*=\s*DRAUGHTING_PRE_DEFINED_COLOUR\s*\(\s*'([^']+)'\s*\)/gi;
  const predefinedColors: Record<string, RGBColor> = {
    'red': { r: 1, g: 0, b: 0 },
    'green': { r: 0, g: 1, b: 0 },
    'blue': { r: 0, g: 0, b: 1 },
    'yellow': { r: 1, g: 1, b: 0 },
    'cyan': { r: 0, g: 1, b: 1 },
    'magenta': { r: 1, g: 0, b: 1 },
    'white': { r: 1, g: 1, b: 1 },
    'black': { r: 0, g: 0, b: 0 },
  };
  while ((match = predefColorRegex.exec(stepContent)) !== null) {
    const id = parseInt(match[1]);
    const colorName = match[2].toLowerCase();
    if (predefinedColors[colorName]) {
      colors.set(id, predefinedColors[colorName]);
    }
  }

  if (colors.size > 0) {
    console.log(`[StepColors] Parsed ${colors.size} color definitions from STEP text`);
  }

  return colors;
}

/**
 * Extract the ordered list of ADVANCED_FACE entity IDs from STEP content.
 * This order should match OCC's face iteration order.
 */
function extractFaceIdOrder(stepContent: string): number[] {
  const faceIds: number[] = [];

  // Match ADVANCED_FACE entities: #123=ADVANCED_FACE(...)
  const advancedFaceRegex = /#(\d+)\s*=\s*ADVANCED_FACE\s*\(/gi;
  let match;
  while ((match = advancedFaceRegex.exec(stepContent)) !== null) {
    faceIds.push(parseInt(match[1]));
  }

  console.log(`[StepColors] Extracted ${faceIds.length} ADVANCED_FACE entity IDs in order`);
  return faceIds;
}

/**
 * Build a map from face geometry key (based on vertex positions) to color.
 * This allows matching OCC faces to STEP colors by geometry rather than index order.
 */
function buildGeometryColorMap(
  stepContent: string,
  faceColors: Map<number, RGBColor>
): Map<string, RGBColor> {
  const geometryColorMap = new Map<string, RGBColor>();

  // Build entity reference map
  const entityRefs = new Map<number, string>();
  const entityRegex = /#(\d+)\s*=\s*([^;]+);/g;
  let match;
  while ((match = entityRegex.exec(stepContent)) !== null) {
    entityRefs.set(parseInt(match[1]), match[2].trim());
  }

  // Helper to extract references from an entity
  const extractRefs = (content: string): number[] => {
    const refs: number[] = [];
    const refRegex = /#(\d+)/g;
    let refMatch;
    while ((refMatch = refRegex.exec(content)) !== null) {
      refs.push(parseInt(refMatch[1]));
    }
    return refs;
  };

  // Helper to get CARTESIAN_POINT coordinates
  const getCartesianPoint = (pointId: number): [number, number, number] | null => {
    const content = entityRefs.get(pointId);
    if (!content || !content.startsWith('CARTESIAN_POINT')) return null;

    // CARTESIAN_POINT('',(-123.456,78.9,0.))
    const coordMatch = content.match(/\(\s*'[^']*'\s*,\s*\(\s*([^)]+)\s*\)\s*\)/);
    if (!coordMatch) return null;

    const coords = coordMatch[1].split(',').map(s => parseFloat(s.trim()));
    if (coords.length >= 3 && coords.every(n => !isNaN(n))) {
      return [coords[0], coords[1], coords[2]];
    }
    return null;
  };

  // Helper to get vertex point coordinate
  const getVertexPoint = (vertexId: number): [number, number, number] | null => {
    const content = entityRefs.get(vertexId);
    if (!content || !content.startsWith('VERTEX_POINT')) return null;

    const refs = extractRefs(content);
    for (const ref of refs) {
      const point = getCartesianPoint(ref);
      if (point) return point;
    }
    return null;
  };

  // Helper to get first vertex from an edge
  const getEdgeFirstVertex = (edgeId: number): [number, number, number] | null => {
    const content = entityRefs.get(edgeId);
    if (!content) return null;

    if (content.startsWith('ORIENTED_EDGE')) {
      // ORIENTED_EDGE('',*,*,#edgeCurve,.T.)
      const refs = extractRefs(content);
      for (const ref of refs) {
        const vertex = getEdgeFirstVertex(ref);
        if (vertex) return vertex;
      }
    } else if (content.startsWith('EDGE_CURVE')) {
      // EDGE_CURVE('',#vertex1,#vertex2,#curve,.T.)
      const refs = extractRefs(content);
      if (refs.length >= 1) {
        return getVertexPoint(refs[0]);
      }
    }
    return null;
  };

  // Helper to get all vertices from a face's outer loop (for computing centroid)
  const getFaceOuterVertices = (faceId: number): [number, number, number][] => {
    const vertices: [number, number, number][] = [];
    const content = entityRefs.get(faceId);
    if (!content || !content.startsWith('ADVANCED_FACE')) return vertices;

    // ADVANCED_FACE('',(#bound1,#bound2,...),#surface,.T.)
    const refs = extractRefs(content);

    for (const boundRef of refs) {
      const boundContent = entityRefs.get(boundRef);
      if (!boundContent) continue;

      // Check for both FACE_OUTER_BOUND and FACE_BOUND (some files use one or the other)
      // Take the first bound we find (typically the outer bound comes first)
      if (boundContent.startsWith('FACE_OUTER_BOUND') || boundContent.startsWith('FACE_BOUND')) {
        // FACE_OUTER_BOUND('',#edgeLoop,.T.) or FACE_BOUND('',#edgeLoop,.T.)
        const boundRefs = extractRefs(boundContent);
        for (const loopRef of boundRefs) {
          const loopContent = entityRefs.get(loopRef);
          if (!loopContent || !loopContent.startsWith('EDGE_LOOP')) continue;

          // EDGE_LOOP('',(#edge1,#edge2,...))
          const edgeRefs = extractRefs(loopContent);
          for (const edgeRef of edgeRefs) {
            const vertex = getEdgeFirstVertex(edgeRef);
            if (vertex) {
              vertices.push(vertex);
            }
          }
        }
        // Only use the first bound (outer), not inner holes
        break;
      }
    }
    return vertices;
  };

  // Create geometry key from centroid (more unique than first vertex)
  const makeCentroidKey = (vertices: [number, number, number][]): string | null => {
    if (vertices.length < 3) return null;

    // Compute centroid
    let cx = 0, cy = 0, cz = 0;
    for (const [x, y, z] of vertices) {
      cx += x;
      cy += y;
      cz += z;
    }
    cx /= vertices.length;
    cy /= vertices.length;
    cz /= vertices.length;

    // Round to 1 decimal place (more forgiving for slight variations)
    const x = Math.round(cx * 10) / 10;
    const y = Math.round(cy * 10) / 10;
    const z = Math.round(cz * 10) / 10;
    return `${x},${y},${z}`;
  };

  // Process each face that has a color
  let mapped = 0;
  let unmapped = 0;
  for (const [faceId, color] of faceColors) {
    if (faceId < 0) continue; // Skip sentinel values

    const vertices = getFaceOuterVertices(faceId);
    const key = makeCentroidKey(vertices);
    if (key) {
      geometryColorMap.set(key, color);
      mapped++;
    } else {
      unmapped++;
    }
  }

  console.log(`[GeometryColorMap] Built map with ${geometryColorMap.size} entries (${mapped} mapped, ${unmapped} unmapped)`);

  // Log some sample entries for debugging
  let sampleCount = 0;
  for (const [key, color] of geometryColorMap) {
    if (sampleCount < 5) {
      console.log(`[GeometryColorMap] Sample: ${key} -> RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)})`);
      sampleCount++;
    } else {
      break;
    }
  }

  return geometryColorMap;
}

/**
 * Build a map from ADVANCED_FACE entity IDs to their colors by following the styling chain.
 * The styling chain in STEP files goes:
 * STYLED_ITEM -> PRESENTATION_STYLE_ASSIGNMENT -> SURFACE_STYLE_USAGE ->
 * SURFACE_SIDE_STYLE -> SURFACE_STYLE_FILL_AREA -> FILL_AREA_STYLE ->
 * FILL_AREA_STYLE_COLOUR -> COLOUR_RGB
 *
 * STYLED_ITEM can target:
 * - An ADVANCED_FACE directly
 * - A MANIFOLD_SOLID_BREP (applies to all faces in the solid)
 * - A CLOSED_SHELL (applies to all faces in the shell)
 */
function buildFaceColorMap(stepContent: string, colorEntities: Map<number, RGBColor>): Map<number, RGBColor> {
  const faceColors = new Map<number, RGBColor>();

  // We need to follow reference chains. Let's build an entity reference map first.
  const entityRefs = new Map<number, string>(); // entity ID -> full entity line
  const entityRegex = /#(\d+)\s*=\s*([^;]+);/g;
  let match;
  while ((match = entityRegex.exec(stepContent)) !== null) {
    const id = parseInt(match[1]);
    const content = match[2].trim();
    entityRefs.set(id, content);
  }

  // Helper to extract references from an entity
  const extractRefs = (content: string): number[] => {
    const refs: number[] = [];
    const refRegex = /#(\d+)/g;
    let refMatch;
    while ((refMatch = refRegex.exec(content)) !== null) {
      refs.push(parseInt(refMatch[1]));
    }
    return refs;
  };

  // Helper to get all face IDs from a shape (solid, shell, compound)
  const getFaceIdsFromShape = (shapeId: number, visited: Set<number> = new Set()): number[] => {
    if (visited.has(shapeId)) return [];
    visited.add(shapeId);

    const content = entityRefs.get(shapeId);
    if (!content) return [];

    const faces: number[] = [];

    // Check entity type
    if (content.startsWith('ADVANCED_FACE')) {
      faces.push(shapeId);
    } else if (content.startsWith('CLOSED_SHELL') || content.startsWith('OPEN_SHELL')) {
      // Shell contains faces: CLOSED_SHELL('NONE', (#face1, #face2, ...))
      const refs = extractRefs(content);
      for (const ref of refs) {
        faces.push(...getFaceIdsFromShape(ref, visited));
      }
    } else if (content.startsWith('MANIFOLD_SOLID_BREP')) {
      // Solid references a shell: MANIFOLD_SOLID_BREP('name', #shell)
      const refs = extractRefs(content);
      for (const ref of refs) {
        faces.push(...getFaceIdsFromShape(ref, visited));
      }
    }

    return faces;
  };

  // Find STYLED_ITEM entities and trace to colors
  // STYLED_ITEM('',(...styles...),#target)
  const styledItemRegex = /#(\d+)\s*=\s*STYLED_ITEM\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)\s*,\s*#(\d+)\s*\)/gi;
  while ((match = styledItemRegex.exec(stepContent)) !== null) {
    const styledItemId = parseInt(match[1]);
    const stylesStr = match[2];
    const targetId = parseInt(match[3]);

    // Extract style references
    const styleRefs = extractRefs(stylesStr);

    // Try to find a color by following the chain
    for (const styleRef of styleRefs) {
      const color = traceToColor(styleRef, entityRefs, colorEntities, new Set());
      if (color) {
        // Get all faces that this style applies to
        const faceIds = getFaceIdsFromShape(targetId);

        if (faceIds.length > 0) {
          for (const faceId of faceIds) {
            faceColors.set(faceId, color);
          }
        } else {
          // Target might be the solid/shell itself, store the color for later use
          faceColors.set(targetId, color);
        }
        break;
      }
    }
  }

  // Also look for a default color if we found colors but no face associations
  // This happens when STYLED_ITEM targets a solid but we couldn't resolve faces
  if (faceColors.size === 0 && colorEntities.size > 0) {
    // No STYLED_ITEM found the faces, but we have colors - use the first one as default
    const firstColor = colorEntities.values().next().value;
    if (firstColor) {
      faceColors.set(-1, firstColor); // -1 as a sentinel for "default color"
    }
  }

  if (faceColors.size > 0) {
    console.log(`[StepColors] Built face->color map with ${faceColors.size} entries`);
  }

  return faceColors;
}

/**
 * Build a map from MANIFOLD_SOLID_BREP entity ID to its color and face count.
 * This allows matching OCC solids to STEP solids by face count.
 */
function buildSolidColorMap(stepContent: string, colorEntities: Map<number, RGBColor>): Map<number, { color: RGBColor; faceCount: number; faceIds: number[] }> {
  const solidMap = new Map<number, { color: RGBColor; faceCount: number; faceIds: number[] }>();

  // Build entity reference map
  const entityRefs = new Map<number, string>();
  const entityRegex = /#(\d+)\s*=\s*([^;]+);/g;
  let match;
  while ((match = entityRegex.exec(stepContent)) !== null) {
    entityRefs.set(parseInt(match[1]), match[2].trim());
  }

  console.log(`[SolidColorMap] Parsed ${entityRefs.size} entities from STEP file, have ${colorEntities.size} colors`);

  // Verify some key entity types exist
  let manifoldCount = 0;
  let presentationStyleCount = 0;
  let surfaceStyleCount = 0;
  for (const content of entityRefs.values()) {
    if (content.startsWith('MANIFOLD_SOLID_BREP')) manifoldCount++;
    if (content.startsWith('PRESENTATION_STYLE_ASSIGNMENT')) presentationStyleCount++;
    if (content.startsWith('SURFACE_STYLE')) surfaceStyleCount++;
  }
  console.log(`[SolidColorMap] Entity types: ${manifoldCount} MANIFOLD_SOLID_BREP, ${presentationStyleCount} PRESENTATION_STYLE_ASSIGNMENT, ${surfaceStyleCount} SURFACE_STYLE*`);

  // Helper to extract references
  const extractRefs = (content: string): number[] => {
    const refs: number[] = [];
    const refRegex = /#(\d+)/g;
    let refMatch;
    while ((refMatch = refRegex.exec(content)) !== null) {
      refs.push(parseInt(refMatch[1]));
    }
    return refs;
  };

  // Helper to get all face IDs from a solid
  const getFaceIdsFromSolid = (solidId: number): number[] => {
    const content = entityRefs.get(solidId);
    if (!content || !content.startsWith('MANIFOLD_SOLID_BREP')) return [];

    const faces: number[] = [];
    const refs = extractRefs(content);

    for (const shellRef of refs) {
      const shellContent = entityRefs.get(shellRef);
      if (!shellContent) continue;
      if (shellContent.startsWith('CLOSED_SHELL') || shellContent.startsWith('OPEN_SHELL')) {
        const faceRefs = extractRefs(shellContent);
        for (const faceRef of faceRefs) {
          const faceContent = entityRefs.get(faceRef);
          if (faceContent && faceContent.startsWith('ADVANCED_FACE')) {
            faces.push(faceRef);
          }
        }
      }
    }

    return faces;
  };

  // Find STYLED_ITEM → MANIFOLD_SOLID_BREP mappings
  const styledItemRegex = /#(\d+)\s*=\s*STYLED_ITEM\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)\s*,\s*#(\d+)\s*\)/gi;
  let styledItemCount = 0;
  let solidTargetCount = 0;
  let colorFoundCount = 0;
  let colorNotFoundCount = 0;

  while ((match = styledItemRegex.exec(stepContent)) !== null) {
    styledItemCount++;
    const styledItemId = parseInt(match[1]);
    const stylesStr = match[2];
    const targetId = parseInt(match[3]);

    const targetContent = entityRefs.get(targetId);
    if (!targetContent) {
      if (styledItemCount <= 3) {
        console.log(`[SolidColorMap] STYLED_ITEM #${styledItemId}: target #${targetId} not found in entityRefs`);
      }
      continue;
    }

    // Only process MANIFOLD_SOLID_BREP targets
    if (!targetContent.startsWith('MANIFOLD_SOLID_BREP')) continue;
    solidTargetCount++;

    // Extract color
    const styleRefs = extractRefs(stylesStr);
    let color: RGBColor | null = null;

    for (const styleRef of styleRefs) {
      color = traceToColor(styleRef, entityRefs, colorEntities, new Set());
      if (color) break;
    }

    if (color) {
      colorFoundCount++;
      const faceIds = getFaceIdsFromSolid(targetId);
      solidMap.set(targetId, { color, faceCount: faceIds.length, faceIds });
      if (solidMap.size <= 3) {
        console.log(`[SolidColorMap] Found solid #${targetId}: ${faceIds.length} faces, RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)})`);
      }
    } else {
      colorNotFoundCount++;
      if (colorNotFoundCount <= 3) {
        console.log(`[SolidColorMap] NO COLOR for solid #${targetId}, styleRefs: [${styleRefs.join(', ')}]`);
        // Debug: show the style chain
        for (const ref of styleRefs) {
          const refContent = entityRefs.get(ref);
          console.log(`[SolidColorMap]   #${ref} = ${refContent ? refContent.substring(0, 100) : 'NOT FOUND'}`);
        }
      }
    }
  }

  console.log(`[SolidColorMap] Stats: ${styledItemCount} STYLED_ITEMs, ${solidTargetCount} target solids, ${colorFoundCount} with colors, ${colorNotFoundCount} without colors, ${solidMap.size} in map`)

  return solidMap;
}

/**
 * Recursively trace through STEP entity references to find a color.
 */
function traceToColor(
  entityId: number,
  entityRefs: Map<number, string>,
  colorEntities: Map<number, RGBColor>,
  visited: Set<number>
): RGBColor | null {
  if (visited.has(entityId)) return null;
  visited.add(entityId);

  // Check if this is a color entity directly
  if (colorEntities.has(entityId)) {
    return colorEntities.get(entityId)!;
  }

  // Get the entity content
  const content = entityRefs.get(entityId);
  if (!content) return null;

  // Extract all references and recursively search
  const refs: number[] = [];
  const refRegex = /#(\d+)/g;
  let match;
  while ((match = refRegex.exec(content)) !== null) {
    refs.push(parseInt(match[1]));
  }

  for (const ref of refs) {
    const color = traceToColor(ref, entityRefs, colorEntities, visited);
    if (color) return color;
  }

  return null;
}

/**
 * DIAGNOSTIC: Comprehensively test all XCAF color extraction methods
 * This helps us understand exactly what's available and working in opencascade.js
 */
function diagnoseXCAFColorExtraction(
  oc: any,
  shape: any,
  colorTool: any,
  shapeTool: any
): void {
  console.log('\n========== XCAF COLOR DIAGNOSTIC ==========');

  if (!colorTool || !shapeTool) {
    console.log('[XCAF_DIAG] Missing colorTool or shapeTool');
    return;
  }

  // Get a sample face to test with
  let sampleFace: any = null;
  let sampleFaceHash = 0;
  try {
    const explorer = new oc.TopExp_Explorer_2(
      shape,
      oc.TopAbs_ShapeEnum.TopAbs_FACE,
      oc.TopAbs_ShapeEnum.TopAbs_SHAPE
    );
    if (explorer.More()) {
      sampleFace = explorer.Current();
      sampleFaceHash = sampleFace.HashCode(2147483647);
      console.log(`[XCAF_DIAG] Testing with sample face (hash: ${sampleFaceHash})`);
    }
  } catch (e) {
    console.log('[XCAF_DIAG] Failed to get sample face:', e);
    return;
  }

  if (!sampleFace) {
    console.log('[XCAF_DIAG] No faces found in shape');
    return;
  }

  // List all colorTool methods
  const colorMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(colorTool) || {})
    .filter(k => typeof colorTool[k] === 'function');
  console.log(`[XCAF_DIAG] ColorTool has ${colorMethods.length} methods`);

  // List GetColor variants
  const getColorMethods = colorMethods.filter(m => m.startsWith('GetColor'));
  console.log(`[XCAF_DIAG] GetColor variants: ${getColorMethods.join(', ')}`);

  // List IsSet variants
  const isSetMethods = colorMethods.filter(m => m.startsWith('IsSet'));
  console.log(`[XCAF_DIAG] IsSet variants: ${isSetMethods.join(', ')}`);

  // Test 1: Check if any colors are defined in document
  console.log('\n--- Test 1: Colors in document ---');
  try {
    // Check if TDF_LabelSequence exists
    const tdfSeqAPIs = Object.keys(oc).filter(k => k.includes('TDF_LabelSequence'));
    console.log(`[XCAF_DIAG] TDF_LabelSequence APIs: ${tdfSeqAPIs.length > 0 ? tdfSeqAPIs.join(', ') : 'NONE'}`);

    if (tdfSeqAPIs.length > 0) {
      for (const api of tdfSeqAPIs) {
        try {
          const seq = new oc[api]();
          colorTool.GetColors(seq);
          console.log(`[XCAF_DIAG] ${api} -> GetColors returned ${seq.Length()} colors`);
        } catch (e) {
          console.log(`[XCAF_DIAG] ${api} -> GetColors failed: ${e}`);
        }
      }
    }
  } catch (e) {
    console.log(`[XCAF_DIAG] Test 1 error: ${e}`);
  }

  // Test 2: Try IsSet on sample face with all color types
  console.log('\n--- Test 2: IsSet on sample face ---');
  const colorTypes = [
    { val: 0, name: 'XCAFDoc_ColorGen' },
    { val: 1, name: 'XCAFDoc_ColorSurf' },
    { val: 2, name: 'XCAFDoc_ColorCurv' }
  ];

  for (const method of isSetMethods) {
    for (const ct of colorTypes) {
      try {
        const result = colorTool[method](sampleFace, ct.val);
        if (result) {
          console.log(`[XCAF_DIAG] ✓ ${method}(face, ${ct.name}) = TRUE`);
        }
      } catch (e) {
        // Silent - method doesn't match signature
      }
    }
  }

  // Test 3: Try ALL GetColor variants with sample face
  console.log('\n--- Test 3: GetColor variants with face ---');
  const qColor = new oc.Quantity_Color_1();

  for (const method of getColorMethods) {
    for (const ct of colorTypes) {
      try {
        // Try (face, colorType, color) signature
        const result = colorTool[method](sampleFace, ct.val, qColor);
        if (result) {
          console.log(`[XCAF_DIAG] ✓ ${method}(face, ${ct.name}, color) = TRUE -> RGB(${qColor.Red().toFixed(3)}, ${qColor.Green().toFixed(3)}, ${qColor.Blue().toFixed(3)})`);
        }
      } catch (e) {
        // Try without colorType - (face, color) signature
        try {
          const result2 = colorTool[method](sampleFace, qColor);
          if (result2) {
            console.log(`[XCAF_DIAG] ✓ ${method}(face, color) = TRUE -> RGB(${qColor.Red().toFixed(3)}, ${qColor.Green().toFixed(3)}, ${qColor.Blue().toFixed(3)})`);
          }
        } catch (e2) {
          // This method doesn't match any face signature
        }
      }
    }
  }

  // Test 4: Try to get label for face, then get color from label
  console.log('\n--- Test 4: Get label for face, then color from label ---');
  try {
    // Try FindShape to get label for the face
    const findShapeMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(shapeTool) || {})
      .filter(k => k.startsWith('FindShape'));
    console.log(`[XCAF_DIAG] ShapeTool FindShape variants: ${findShapeMethods.join(', ')}`);

    for (const method of findShapeMethods) {
      try {
        const label = shapeTool[method](sampleFace);
        if (label && !label.IsNull()) {
          console.log(`[XCAF_DIAG] ✓ shapeTool.${method}(face) returned valid label`);

          // Now try to get color from this label
          for (const getMethod of getColorMethods) {
            for (const ct of colorTypes) {
              try {
                const result = colorTool[getMethod](label, ct.val, qColor);
                if (result) {
                  console.log(`[XCAF_DIAG] ✓ colorTool.${getMethod}(label, ${ct.name}, color) = TRUE -> RGB(${qColor.Red().toFixed(3)}, ${qColor.Green().toFixed(3)}, ${qColor.Blue().toFixed(3)})`);
                }
              } catch (e) {
                // Signature doesn't match
              }
            }
          }
        }
      } catch (e) {
        // Method doesn't match signature
      }
    }
  } catch (e) {
    console.log(`[XCAF_DIAG] Test 4 error: ${e}`);
  }

  // Test 5: Try iterating first 5 faces and check each
  console.log('\n--- Test 5: Check first 5 faces for colors ---');
  try {
    const explorer = new oc.TopExp_Explorer_2(
      shape,
      oc.TopAbs_ShapeEnum.TopAbs_FACE,
      oc.TopAbs_ShapeEnum.TopAbs_SHAPE
    );

    let faceIdx = 0;
    while (explorer.More() && faceIdx < 5) {
      const face = explorer.Current();
      let foundColor = false;

      // Try GetColor_7 specifically (should be shape, type, color)
      for (const ct of colorTypes) {
        try {
          if (colorTool.GetColor_7) {
            const result = colorTool.GetColor_7(face, ct.val, qColor);
            if (result) {
              console.log(`[XCAF_DIAG] Face ${faceIdx}: GetColor_7(face, ${ct.name}) -> RGB(${qColor.Red().toFixed(3)}, ${qColor.Green().toFixed(3)}, ${qColor.Blue().toFixed(3)})`);
              foundColor = true;
            }
          }
        } catch (e) {
          // Signature doesn't match
        }
      }

      if (!foundColor) {
        console.log(`[XCAF_DIAG] Face ${faceIdx}: No color found via GetColor_7`);
      }

      explorer.Next();
      faceIdx++;
    }
  } catch (e) {
    console.log(`[XCAF_DIAG] Test 5 error: ${e}`);
  }

  // Test 6: Check XCAFDoc_ColorType enum availability
  console.log('\n--- Test 6: XCAFDoc_ColorType enum ---');
  const enumKeys = Object.keys(oc).filter(k => k.includes('XCAFDoc_Color'));
  console.log(`[XCAF_DIAG] XCAFDoc_Color* APIs: ${enumKeys.join(', ')}`);

  if (oc.XCAFDoc_ColorType) {
    console.log(`[XCAF_DIAG] XCAFDoc_ColorType values:`, Object.keys(oc.XCAFDoc_ColorType));
  }

  // Test 7: Check for other XCAF tools that might have color info
  console.log('\n--- Test 7: Other XCAF tools ---');
  const xcafTools = Object.keys(oc).filter(k => k.includes('XCAFDoc_') && k.includes('Tool'));
  console.log(`[XCAF_DIAG] Available XCAF tools: ${xcafTools.join(', ')}`);

  // Test 8: Check VisMaterialTool if available
  console.log('\n--- Test 8: VisMaterialTool check ---');
  try {
    if (oc.XCAFDoc_VisMaterialTool) {
      console.log('[XCAF_DIAG] XCAFDoc_VisMaterialTool is available');
    }
    if (oc.XCAFDoc_MaterialTool) {
      console.log('[XCAF_DIAG] XCAFDoc_MaterialTool is available');
    }
  } catch (e) {
    console.log(`[XCAF_DIAG] Material tool check error: ${e}`);
  }

  // Test 9: Try to directly query a face's visual properties
  console.log('\n--- Test 9: Direct face visual property query ---');
  try {
    // Some implementations use Quantity_Color directly on shapes
    if (sampleFace && oc.BRepTools && oc.BRepTools.Read) {
      console.log('[XCAF_DIAG] BRepTools is available');
    }

    // Check if there's a way to get presentation (visual) attributes
    const presentationAPIs = Object.keys(oc).filter(k =>
      k.includes('Presentation') || k.includes('AIS_') || k.includes('Prs3d'));
    console.log(`[XCAF_DIAG] Presentation APIs available: ${presentationAPIs.length > 0 ? presentationAPIs.slice(0, 10).join(', ') + '...' : 'NONE'}`);
  } catch (e) {
    console.log(`[XCAF_DIAG] Test 9 error: ${e}`);
  }

  // Test 10: Try getting color from the whole shape (not just faces)
  console.log('\n--- Test 10: Color on whole shape ---');
  try {
    for (const ct of colorTypes) {
      try {
        if (colorTool.GetColor_7) {
          const result = colorTool.GetColor_7(shape, ct.val, qColor);
          if (result) {
            console.log(`[XCAF_DIAG] ✓ Whole shape has color via GetColor_7(shape, ${ct.name}) -> RGB(${qColor.Red().toFixed(3)}, ${qColor.Green().toFixed(3)}, ${qColor.Blue().toFixed(3)})`);
          }
        }
      } catch (e) {
        // Doesn't match
      }
    }
  } catch (e) {
    console.log(`[XCAF_DIAG] Test 10 error: ${e}`);
  }

  console.log('\n========== END XCAF DIAGNOSTIC ==========\n');
}

/**
 * Build a map from face HashCodes to colors by traversing the XCAF document.
 * For each shape that has a color, we iterate its faces and map each face's
 * HashCode to that color. This allows direct lookup when processing faces.
 */
function buildShapeColorMap(
  oc: any,
  shape: any,
  shapeToolInput: any,
  colorToolInput: any,
  docHandle?: any  // Optional: Handle_TDocStd_Document for XCAFPrs_DocumentExplorer
): Map<number, RGBColor> {
  const faceColorMap = new Map<number, RGBColor>();

  if (!shapeToolInput || !colorToolInput) {
    logOCC('[ShapeColorMap] Missing shapeTool or colorTool');
    return faceColorMap;
  }

  // === PRIMARY APPROACH: Use XCAFPrs_DocumentExplorer ===
  // This is the canonical way to iterate XCAF documents with colors
  if (docHandle && oc.XCAFPrs_DocumentExplorer) {
    logOCC('[ShapeColorMap] Trying XCAFPrs_DocumentExplorer approach...');
    try {
      let explorer = null;
      let explorerInitialized = false;

      // Try different constructors
      // XCAFPrs_DocumentExplorer_1: default constructor, then Init
      // XCAFPrs_DocumentExplorer_2: takes document handle
      // XCAFPrs_DocumentExplorer_3: takes document handle + flags
      if (oc.XCAFPrs_DocumentExplorer_2) {
        try {
          explorer = new oc.XCAFPrs_DocumentExplorer_2(docHandle);
          explorerInitialized = true;
          logOCC('[ShapeColorMap] Created explorer via XCAFPrs_DocumentExplorer_2');
        } catch (e) {
          logOCC('[ShapeColorMap] XCAFPrs_DocumentExplorer_2 failed:', e);
        }
      }

      if (!explorerInitialized && oc.XCAFPrs_DocumentExplorer_1) {
        try {
          explorer = new oc.XCAFPrs_DocumentExplorer_1();
          // Init_1 or Init_2 to initialize with document
          if (explorer.Init_1) {
            explorer.Init_1(docHandle);
            explorerInitialized = true;
            logOCC('[ShapeColorMap] Created explorer via XCAFPrs_DocumentExplorer_1 + Init_1');
          } else if (explorer.Init_2) {
            explorer.Init_2(docHandle);
            explorerInitialized = true;
            logOCC('[ShapeColorMap] Created explorer via XCAFPrs_DocumentExplorer_1 + Init_2');
          }
        } catch (e) {
          logOCC('[ShapeColorMap] XCAFPrs_DocumentExplorer_1 failed:', e);
        }
      }

      if (explorerInitialized && explorer) {
        let nodeCount = 0;
        let nodesWithColor = 0;
        let facesColored = 0;

        // Iterate through the document
        while (explorer.More()) {
          nodeCount++;
          try {
            // Get the current node - returns XCAFPrs_DocumentNode
            const current = explorer.Current_1 ? explorer.Current_1() : explorer.Current_2();

            if (current) {
              // XCAFPrs_DocumentNode has: Id (label), LocalTrsf, Location, Style, etc.
              // Try to get the style which contains color info
              let style = null;
              let nodeShape = null;

              // Get style - XCAFPrs_Style contains color information
              if (current.Style) {
                style = current.Style();
              }

              // Get the shape from the node
              if (current.RefLabel) {
                const refLabel = current.RefLabel();
                if (shapeToolInput.GetShape) {
                  nodeShape = shapeToolInput.GetShape(refLabel);
                }
              } else if (current.Label) {
                const label = current.Label();
                if (shapeToolInput.GetShape) {
                  nodeShape = shapeToolInput.GetShape(label);
                }
              } else if (current.Id) {
                const label = current.Id();
                if (shapeToolInput.GetShape) {
                  nodeShape = shapeToolInput.GetShape(label);
                }
              }

              // Extract color from style
              let nodeColor: RGBColor | null = null;
              if (style) {
                // XCAFPrs_Style has GetColorSurf, GetColorCurv, GetColorSurfRGBA, etc.
                try {
                  if (style.IsSetColorSurf && style.IsSetColorSurf()) {
                    const surfColor = style.GetColorSurfRGBA ? style.GetColorSurfRGBA() : style.GetColorSurf();
                    if (surfColor) {
                      if (surfColor.GetRGB) {
                        const rgb = surfColor.GetRGB();
                        nodeColor = { r: rgb.Red(), g: rgb.Green(), b: rgb.Blue() };
                      } else {
                        nodeColor = { r: surfColor.Red(), g: surfColor.Green(), b: surfColor.Blue() };
                      }
                    }
                  }
                } catch (styleErr) {
                  // Try alternative methods
                }

                // Fallback to curve color if surface color not set
                if (!nodeColor) {
                  try {
                    if (style.IsSetColorCurv && style.IsSetColorCurv()) {
                      const curvColor = style.GetColorCurv();
                      if (curvColor) {
                        nodeColor = { r: curvColor.Red(), g: curvColor.Green(), b: curvColor.Blue() };
                      }
                    }
                  } catch (styleErr) {
                    // No curve color
                  }
                }
              }

              // If we have a color and shape, map all faces
              if (nodeColor && nodeShape && !nodeShape.IsNull()) {
                nodesWithColor++;

                // Iterate faces in this shape
                const faceExplorer = new oc.TopExp_Explorer_2(
                  nodeShape,
                  oc.TopAbs_ShapeEnum.TopAbs_FACE,
                  oc.TopAbs_ShapeEnum.TopAbs_SHAPE
                );

                while (faceExplorer.More()) {
                  const face = faceExplorer.Current();
                  const hashCode = face.HashCode(2147483647);
                  if (!faceColorMap.has(hashCode)) {
                    faceColorMap.set(hashCode, nodeColor);
                    facesColored++;
                  }
                  faceExplorer.Next();
                }
              }

              if (nodeCount <= 5) {
                const depth = explorer.CurrentDepth ? explorer.CurrentDepth() : -1;
                logOCC(`[ShapeColorMap] Node ${nodeCount}: depth=${depth}, hasColor=${!!nodeColor}, hasShape=${!!(nodeShape && !nodeShape.IsNull())}`);
              }
            }
          } catch (nodeErr) {
            if (nodeCount <= 3) {
              logOCC(`[ShapeColorMap] Error processing node ${nodeCount}:`, nodeErr);
            }
          }

          explorer.Next();
        }

        logOCC(`[ShapeColorMap] XCAFPrs_DocumentExplorer: ${nodeCount} nodes, ${nodesWithColor} with colors, ${facesColored} faces colored`);

        if (faceColorMap.size > 0) {
          logOCC(`[ShapeColorMap] SUCCESS: XCAFPrs_DocumentExplorer extracted ${faceColorMap.size} face colors`);
          return faceColorMap; // Return early if we got colors
        }
      }
    } catch (explorerErr) {
      logOCC('[ShapeColorMap] XCAFPrs_DocumentExplorer failed:', explorerErr);
    }
  } else {
    if (!docHandle) {
      logOCC('[ShapeColorMap] No docHandle provided for XCAFPrs_DocumentExplorer');
    }
    if (!oc.XCAFPrs_DocumentExplorer) {
      logOCC('[ShapeColorMap] XCAFPrs_DocumentExplorer not available');
    }
  }

  // === FALLBACK APPROACHES below ===

  // DIAGNOSTIC: Check for static GetShape method
  logOCC('\n=== STATIC METHOD CHECK ===');
  const shapeToolStaticMethods = Object.keys(oc).filter(k => k.includes('XCAFDoc_ShapeTool') && k.includes('GetShape'));
  logOCC('[Static] XCAFDoc_ShapeTool GetShape methods:', shapeToolStaticMethods.length > 0 ? shapeToolStaticMethods.join(', ') : 'NONE');

  // Check if XCAFDoc_ShapeTool has static methods
  if (oc.XCAFDoc_ShapeTool) {
    const staticMethods = Object.getOwnPropertyNames(oc.XCAFDoc_ShapeTool).filter(k => typeof oc.XCAFDoc_ShapeTool[k] === 'function');
    logOCC('[Static] XCAFDoc_ShapeTool class methods:', staticMethods.slice(0, 15).join(', '));
    logOCC('[Static] Has GetShape:', typeof oc.XCAFDoc_ShapeTool.GetShape === 'function');
    logOCC('[Static] Has GetShape_1:', typeof oc.XCAFDoc_ShapeTool.GetShape_1 === 'function');
  }

  // colorTool and shapeTool might be Handles - try to unwrap them
  let colorTool = colorToolInput;
  if (typeof colorToolInput.get === 'function') {
    try {
      colorTool = colorToolInput.get();
      logOCC('[ShapeColorMap] Unwrapped colorTool handle');
    } catch (e) {
      logOCC('[ShapeColorMap] Failed to unwrap colorTool:', e);
    }
  }

  let shapeTool = shapeToolInput;
  if (typeof shapeToolInput.get === 'function') {
    try {
      shapeTool = shapeToolInput.get();
      logOCC('[ShapeColorMap] Unwrapped shapeTool handle');
    } catch (e) {
      logOCC('[ShapeColorMap] Failed to unwrap shapeTool:', e);
    }
  }

  // Log available colorTool methods for debugging (use prototype, not Object.keys)
  const colorMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(colorTool) || {})
    .filter(k => typeof colorTool[k] === 'function');
  logOCC('[ShapeColorMap] colorTool methods:', colorMethods.slice(0, 20).join(', '));
  logOCC('[ShapeColorMap] colorTool method count:', colorMethods.length);

  // Check how many colors are defined in the document
  let docColorCount = 0;
  try {
    if (typeof colorTool.GetColors === 'function') {
      // Try different TDF_LabelSequence constructors
      let colorLabels = null;
      if (oc.TDF_LabelSequence_1) {
        colorLabels = new oc.TDF_LabelSequence_1();
      } else if (oc.TDF_LabelSequence) {
        colorLabels = new oc.TDF_LabelSequence();
      }

      if (colorLabels) {
        colorTool.GetColors(colorLabels);
        docColorCount = colorLabels.Length();
        logOCC(`[ShapeColorMap] Colors defined in XCAF document: ${docColorCount}`);
      } else {
        // List available TDF_LabelSequence APIs
        const tdfApis = Object.keys(oc).filter(k => k.includes('TDF_LabelSequence'));
        logOCC('[ShapeColorMap] Available TDF_LabelSequence APIs:', tdfApis.join(', '));
      }
    }
  } catch (e) {
    logOCC('[ShapeColorMap] GetColors failed:', e);
    // List available TDF_LabelSequence APIs on error
    const tdfApis = Object.keys(oc).filter(k => k.includes('TDF_LabelSequence'));
    logOCC('[ShapeColorMap] Available TDF_LabelSequence APIs:', tdfApis.join(', '));
  }

  let colorsFoundViaShape = 0;
  let colorsFoundViaLabel = 0;

  // Helper to get color from a shape directly
  // Shape-based GetColor methods have DIFFERENT output types:
  // - GetColor_6: (shape, colorType, out_TDF_Label) → returns color LABEL
  // - GetColor_7: (shape, colorType, out_Quantity_Color) → returns RGB color (what we need!)
  // - GetColor_8: (shape, colorType, out_Quantity_ColorRGBA) → returns RGBA color
  let shapeColorAttempts = 0;
  let shapeColorErrors: string[] = [];

  const getShapeColor = (shape: any): RGBColor | null => {
    if (!shape || shape.IsNull()) return null;

    // XCAFDoc_ColorType: 0=Gen, 1=Surf, 2=Curv
    for (const colorType of [1, 0, 2]) {
      // Try GetColor_7 first - it takes Quantity_Color as output
      try {
        if (typeof colorTool.GetColor_7 === 'function') {
          const color = new oc.Quantity_Color_1();
          shapeColorAttempts++;
          const hasColor = colorTool.GetColor_7(shape, colorType, color);
          if (hasColor) {
            colorsFoundViaShape++;
            return { r: color.Red(), g: color.Green(), b: color.Blue() };
          }
        }
      } catch (e: any) {
        if (shapeColorErrors.length < 3) {
          shapeColorErrors.push(`GetColor_7(colorType=${colorType}): ${e.message || e}`);
        }
      }

      // Fallback to GetColor_8 with Quantity_ColorRGBA if GetColor_7 didn't work
      try {
        if (typeof colorTool.GetColor_8 === 'function' && oc.Quantity_ColorRGBA_1) {
          const colorRGBA = new oc.Quantity_ColorRGBA_1();
          shapeColorAttempts++;
          const hasColor = colorTool.GetColor_8(shape, colorType, colorRGBA);
          if (hasColor) {
            colorsFoundViaShape++;
            const rgb = colorRGBA.GetRGB();
            return { r: rgb.Red(), g: rgb.Green(), b: rgb.Blue() };
          }
        }
      } catch (e: any) {
        if (shapeColorErrors.length < 6) {
          shapeColorErrors.push(`GetColor_8(colorType=${colorType}): ${e.message || e}`);
        }
      }
    }
    return null;
  };

  // Helper to get color from a label
  // Label-based GetColor methods have different signatures:
  // - GetColor_1: (label, out_color) → 2 args, returns bool
  // - GetColor_2: (label, out_colorRGBA) → 2 args, returns bool
  // - GetColor_4: (label, colorType, out_color) → 3 args, returns bool
  // - GetColor_5: (label, colorType, out_colorRGBA) → 3 args, returns bool (if exists)
  const getLabelColor = (label: any): RGBColor | null => {
    if (!label || label.IsNull()) return null;

    const color = new oc.Quantity_Color_1();

    // First try GetColor_4 with colorType (most specific)
    // XCAFDoc_ColorType: 0=Gen, 1=Surf, 2=Curv
    for (const colorType of [1, 0, 2]) {
      try {
        if (typeof colorTool.GetColor_4 === 'function') {
          const hasColor = colorTool.GetColor_4(label, colorType, color);
          if (hasColor) {
            colorsFoundViaLabel++;
            return { r: color.Red(), g: color.Green(), b: color.Blue() };
          }
        }
      } catch (e) {
        // Method signature doesn't match
      }
    }

    // Try GetColor_1: (label, color) - 2 args, no colorType
    try {
      if (typeof colorTool.GetColor_1 === 'function') {
        const hasColor = colorTool.GetColor_1(label, color);
        if (hasColor) {
          colorsFoundViaLabel++;
          return { r: color.Red(), g: color.Green(), b: color.Blue() };
        }
      }
    } catch (e) {
      // Method signature doesn't match
    }

    // Try GetColor_2 with RGBA (if available)
    try {
      if (typeof colorTool.GetColor_2 === 'function' && oc.Quantity_ColorRGBA_1) {
        const colorRGBA = new oc.Quantity_ColorRGBA_1();
        const hasColor = colorTool.GetColor_2(label, colorRGBA);
        if (hasColor) {
          colorsFoundViaLabel++;
          // Extract RGB from RGBA
          const rgb = colorRGBA.GetRGB();
          return { r: rgb.Red(), g: rgb.Green(), b: rgb.Blue() };
        }
      }
    } catch (e) {
      // Method signature doesn't match
    }

    return null;
  };

  // Helper to map all faces of a shape to a color
  const mapFacesToColor = (shape: any, color: RGBColor) => {
    if (!shape || shape.IsNull()) return;

    try {
      const explorer = new oc.TopExp_Explorer_2(
        shape,
        oc.TopAbs_ShapeEnum.TopAbs_FACE,
        oc.TopAbs_ShapeEnum.TopAbs_SHAPE
      );

      while (explorer.More()) {
        const face = explorer.Current();
        const hashCode = face.HashCode(2147483647); // Max int for unique hash
        if (!faceColorMap.has(hashCode)) {
          faceColorMap.set(hashCode, color);
        }
        explorer.Next();
      }
    } catch (e) {
      // Silent fail
    }
  };

  // Recursive function to traverse labels and collect colors
  let labelsProcessed = 0;
  let labelsWithColor = 0;
  let labelsWithShape = 0;

  const processLabel = (label: any, inheritedColor: RGBColor | null, depth: number = 0) => {
    if (depth > 30 || !label || label.IsNull()) return;
    labelsProcessed++;

    try {
      // Get shape for this label using STATIC method
      let shape: any = null;

      // Use static GetShape_1(label, outShape) - 2 args
      if (oc.XCAFDoc_ShapeTool && typeof oc.XCAFDoc_ShapeTool.GetShape_1 === 'function') {
        try {
          const outShape = new oc.TopoDS_Shape();
          const success = oc.XCAFDoc_ShapeTool.GetShape_1(label, outShape);
          if (success && outShape && !outShape.IsNull()) {
            shape = outShape;
          }
        } catch (e) {
          // GetShape_1 failed
        }
      }

      if (shape && !shape.IsNull()) {
        labelsWithShape++;
      }

      // Check for color on this label/shape
      let color = getLabelColor(label);
      let colorSource = 'label';
      if (!color && shape && !shape.IsNull()) {
        color = getShapeColor(shape);
        colorSource = 'shape';
      }

      // DIAGNOSTIC: If no color found, try IsSet_2 on shape (limit logging)
      if (!color && shape && !shape.IsNull() && labelsProcessed <= 20) {
        for (const colorType of [0, 1, 2]) {
          if (typeof colorTool.IsSet_2 === 'function') {
            try {
              const isSet = colorTool.IsSet_2(shape, colorType);
              if (isSet) {
                logOCC(`[processLabel] Depth ${depth}: IsSet_2(shape, ${colorType}) = TRUE but getLabelColor/getShapeColor returned null!`);

                // Try GetColor_7 directly
                if (typeof colorTool.GetColor_7 === 'function') {
                  const outColor = new oc.Quantity_Color_1();
                  const hasColor = colorTool.GetColor_7(shape, colorType, outColor);
                  if (hasColor) {
                    logOCC(`[processLabel] GetColor_7 SUCCESS: RGB(${outColor.Red().toFixed(3)}, ${outColor.Green().toFixed(3)}, ${outColor.Blue().toFixed(3)})`);
                    color = { r: outColor.Red(), g: outColor.Green(), b: outColor.Blue() };
                    colorSource = 'IsSet_2+GetColor_7';
                  }
                }
              }
            } catch (e) {
              // Silent
            }
          }
        }
      }

      if (color) {
        labelsWithColor++;
        if (labelsWithColor <= 5) {
          logOCC(`[processLabel] Found color at depth ${depth} via ${colorSource}: RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)})`);
        }
      }

      // Use inherited color if no direct color
      const effectiveColor = color || inheritedColor;

      // If we have a color and a shape, map all faces to this color
      if (effectiveColor && shape && !shape.IsNull()) {
        mapFacesToColor(shape, effectiveColor);
      }

      // Traverse children using TDF_ChildIterator_2(label, allLevels)
      if (oc.TDF_ChildIterator_2) {
        const childIter = new oc.TDF_ChildIterator_2(label, false);
        while (childIter.More()) {
          processLabel(childIter.Value(), effectiveColor, depth + 1);
          childIter.Next();
        }
      }
    } catch (e) {
      // Silent fail
    }
  };

  // DEBUG: List available methods on shapeTool and colorTool (use prototype)
  const shapeToolProto = Object.getOwnPropertyNames(Object.getPrototypeOf(shapeTool) || {}).filter(k => typeof shapeTool[k] === 'function');
  logOCC('[ShapeColorMap] shapeTool prototype methods:', shapeToolProto.join(', '));
  logOCC('[ShapeColorMap] shapeTool has Search:', typeof shapeTool.Search === 'function');
  logOCC('[ShapeColorMap] shapeTool has Search_1:', typeof shapeTool.Search_1 === 'function');
  logOCC('[ShapeColorMap] colorTool methods:', Object.keys(colorTool).filter(k => typeof colorTool[k] === 'function').slice(0, 30));

  // NEW: Try to get the main label and traverse from there
  try {
    // Get the main label from the document
    const mainLabel = shapeToolInput.BaseLabel ? shapeToolInput.BaseLabel() : null;
    if (mainLabel) {
      logOCC('[ShapeColorMap] BaseLabel found, checking for colors...');

      // Recursive function to traverse label tree
      const traverseForColors = (label: any, depth: number) => {
        if (depth > 10) return;

        // Check if this label has a shape using STATIC GetShape_1(label, outShape)
        let labelShape = null;
        try {
          if (oc.XCAFDoc_ShapeTool?.GetShape_1) {
            const outShape = new oc.TopoDS_Shape();
            const success = oc.XCAFDoc_ShapeTool.GetShape_1(label, outShape);
            if (success && outShape && !outShape.IsNull()) {
              labelShape = outShape;
            }
          }
        } catch (e) {
          // GetShape_1 failed
        }

        // Check if this label has a color
        const labelColor = getLabelColor(label);
        if (labelColor && depth < 5) {
          logOCC(`[ShapeColorMap] Label at depth ${depth} has color: RGB(${labelColor.r.toFixed(2)}, ${labelColor.g.toFixed(2)}, ${labelColor.b.toFixed(2)})`);

          // If it has a shape, map the shape's faces to this color
          if (labelShape && !labelShape.IsNull()) {
            mapFacesToColor(labelShape, labelColor);
          }
        }

        // Traverse children using TDF_ChildIterator_2(label, allLevels)
        try {
          if (oc.TDF_ChildIterator_2) {
            const childIter = new oc.TDF_ChildIterator_2(label, false);
            while (childIter.More()) {
              traverseForColors(childIter.Value(), depth + 1);
              childIter.Next();
            }
          }
        } catch (e) {}
      };

      traverseForColors(mainLabel, 0);
    }
  } catch (e) {
    logOCC('[ShapeColorMap] BaseLabel traversal failed:', e);
  }

  // NEW: Iterate over all SOLIDS and get colors directly from XCAF
  // This doesn't require TDF_LabelSequence which is unavailable in OpenCascade.js
  let solidsWithColors = 0;
  let facesColoredFromSolids = 0;
  try {
    const solidExplorer = new oc.TopExp_Explorer_2(
      shape,
      oc.TopAbs_ShapeEnum.TopAbs_SOLID,
      oc.TopAbs_ShapeEnum.TopAbs_SHAPE
    );

    let solidIndex = 0;
    while (solidExplorer.More()) {
      const solid = solidExplorer.Current();

      // Try to get color for this solid directly via XCAF
      let solidColor = getShapeColor(solid);

      // DEBUG: Log what we're checking for solid 0
      if (solidIndex === 0) {
        logOCC(`[ShapeColorMap] Solid 0 getShapeColor result: ${solidColor ? 'found' : 'null'}`);
        logOCC(`[ShapeColorMap] Solid 0 HashCode: ${solid.HashCode(2147483647)}`);
        logOCC(`[ShapeColorMap] shapeTool.FindShape available: ${typeof shapeTool.FindShape}`);
      }

      // If direct shape lookup fails, try finding the label for this shape
      // occt-import-js uses shapeTool->Search() which is the correct approach
      if (!solidColor && shapeTool) {
        // Try Search method (like occt-import-js does)
        // Search signature in emscripten needs ALL 5 args (no default params):
        // Search(shape, out_label, findInstance, findComponent, findSubShape) → bool
        if (typeof shapeTool.Search === 'function') {
          try {
            const solidLabel = new oc.TDF_Label();
            // Pass all 5 arguments - findInstance=true, findComponent=true, findSubShape=true
            const found = shapeTool.Search(solid, solidLabel, true, true, true);
            if (found && solidLabel && !solidLabel.IsNull()) {
              solidColor = getLabelColor(solidLabel);
              if (solidColor && solidIndex < 3) {
                logOCC(`[ShapeColorMap] Solid ${solidIndex} found color via Search`);
              }
            } else if (solidIndex === 0) {
              logOCC(`[ShapeColorMap] Solid 0: Search returned found=${found}, label.IsNull=${solidLabel?.IsNull?.()}`);
            }
          } catch (e: any) {
            if (solidIndex === 0) {
              logOCC(`[ShapeColorMap] Solid 0: Search(5 args) error: ${e.message || e}`);
            }
          }
        }

        // Fallback to FindShape variants if Search didn't work
        if (!solidColor) {
          for (const findMethodName of ['FindShape', 'FindShape_1', 'FindShape_2']) {
            if (typeof shapeTool[findMethodName] === 'function') {
              try {
                const solidLabel = new oc.TDF_Label();
                // Try 2-arg signature first (shape, out_label)
                let found = false;
                try {
                  found = shapeTool[findMethodName](solid, solidLabel);
                } catch (e2) {
                  // Try 3-arg signature (shape, out_label, findInstance)
                  try {
                    found = shapeTool[findMethodName](solid, solidLabel, false);
                  } catch (e3) {
                    // Neither worked
                  }
                }
                if (found && solidLabel && !solidLabel.IsNull()) {
                  solidColor = getLabelColor(solidLabel);
                  if (solidColor && solidIndex < 3) {
                    logOCC(`[ShapeColorMap] Solid ${solidIndex} found color via ${findMethodName}`);
                  }
                  break;
                }
              } catch (e) {
                // Silent
              }
            }
          }
        }
      }

      if (solidColor) {
        solidsWithColors++;
        // Map all faces in this solid to the solid's color
        const faceExplorer = new oc.TopExp_Explorer_2(
          solid,
          oc.TopAbs_ShapeEnum.TopAbs_FACE,
          oc.TopAbs_ShapeEnum.TopAbs_SHAPE
        );

        while (faceExplorer.More()) {
          const face = faceExplorer.Current();
          const hashCode = face.HashCode(2147483647);
          if (!faceColorMap.has(hashCode)) {
            faceColorMap.set(hashCode, solidColor);
            facesColoredFromSolids++;
          }
          faceExplorer.Next();
        }

        if (solidIndex < 5) {
          logOCC(`[ShapeColorMap] Solid ${solidIndex} has XCAF color: RGB(${solidColor.r.toFixed(2)}, ${solidColor.g.toFixed(2)}, ${solidColor.b.toFixed(2)})`);
        }
      } else if (solidIndex < 3) {
        // Debug: check if IsSet returns true for this solid
        for (const ct of [0, 1, 2]) {
          try {
            if (colorTool.IsSet_1) {
              const isSet = colorTool.IsSet_1(solid, ct);
              if (isSet) logOCC(`[ShapeColorMap] Solid ${solidIndex}: IsSet_1(${ct}) = true`);
            }
          } catch (e) {}
        }
      }

      solidIndex++;
      solidExplorer.Next();
    }

    logOCC(`[ShapeColorMap] XCAF solid colors: ${solidsWithColors}/${solidIndex} solids have colors, ${facesColoredFromSolids} faces colored`);
  } catch (e) {
    logOCC('[ShapeColorMap] Error iterating solids for XCAF colors:', e);
  }

  // Fallback: Try TDF_LabelSequence approach (may fail if not available)
  try {
    const labels = new oc.TDF_LabelSequence_1();
    shapeTool.GetFreeShapes(labels);

    logOCC(`[ShapeColorMap] Processing ${labels.Length()} free shapes for face colors...`);

    // DIAGNOSTIC: Examine each free shape label in detail
    for (let i = 1; i <= labels.Length(); i++) {
      const label = labels.Value(i);
      logOCC(`\n=== FREE SHAPE LABEL ${i} ===`);

      // Check label type using STATIC methods on XCAFDoc_ShapeTool
      try {
        const ST = oc.XCAFDoc_ShapeTool;
        const isShape = ST?.IsShape ? ST.IsShape(label) : 'N/A';
        const isAssembly = ST?.IsAssembly ? ST.IsAssembly(label) : 'N/A';
        const isComponent = ST?.IsComponent ? ST.IsComponent(label) : 'N/A';
        const isReference = ST?.IsReference ? ST.IsReference(label) : 'N/A';
        const isSimpleShape = ST?.IsSimpleShape ? ST.IsSimpleShape(label) : 'N/A';
        const isCompound = ST?.IsCompound ? ST.IsCompound(label) : 'N/A';
        const isSubShape = ST?.IsSubShape_1 ? ST.IsSubShape_1(label) : 'N/A';

        logOCC(`[Label ${i}] IsShape=${isShape}, IsAssembly=${isAssembly}, IsComponent=${isComponent}`);
        logOCC(`[Label ${i}] IsReference=${isReference}, IsSimpleShape=${isSimpleShape}, IsCompound=${isCompound}, IsSubShape=${isSubShape}`);
      } catch (e: any) {
        logOCC(`[Label ${i}] Error checking label type: ${e.message || e}`);
      }

      // Check if label has shape using STATIC GetShape_1(label, outShape)
      try {
        let shape = null;
        if (oc.XCAFDoc_ShapeTool?.GetShape_1) {
          const outShape = new oc.TopoDS_Shape();
          const success = oc.XCAFDoc_ShapeTool.GetShape_1(label, outShape);
          if (success && outShape && !outShape.IsNull()) {
            shape = outShape;
          }
        }
        const hasShape = shape && !shape.IsNull();
        if (hasShape) {
          const shapeType = shape.ShapeType ? shape.ShapeType() : 'unknown';
          logOCC(`[Label ${i}] Has shape, type=${shapeType}`);
        } else {
          logOCC(`[Label ${i}] No shape (IsNull or undefined)`);
        }
      } catch (e: any) {
        logOCC(`[Label ${i}] GetShape_1 error: ${e.message || e}`);
      }

      // Check if label has direct color
      const labelColor = getLabelColor(label);
      logOCC(`[Label ${i}] Direct color: ${labelColor ? `RGB(${labelColor.r.toFixed(2)}, ${labelColor.g.toFixed(2)}, ${labelColor.b.toFixed(2)})` : 'none'}`);

      // OPTION 2: Try IsSet_1 and IsSet_2 to check if color IS assigned
      // IsSet checks if a color attribute exists without retrieving it
      try {
        const colorTypes = [
          { val: 0, name: 'XCAFDoc_ColorGen' },
          { val: 1, name: 'XCAFDoc_ColorSurf' },
          { val: 2, name: 'XCAFDoc_ColorCurv' }
        ];

        for (const ct of colorTypes) {
          // Try IsSet_1(label, colorType) - label-based
          if (typeof colorTool.IsSet_1 === 'function') {
            try {
              const isSet = colorTool.IsSet_1(label, ct.val);
              if (isSet) {
                logOCC(`[Label ${i}] IsSet_1(${ct.name}) = TRUE - color IS set!`);
              }
            } catch (e: any) {
              logOCC(`[Label ${i}] IsSet_1(${ct.name}) error: ${e.message || e}`);
            }
          }

          // Try IsSet_2(shape, colorType) - shape-based
          if (typeof colorTool.IsSet_2 === 'function') {
            try {
              // Get the shape from this label first
              let labelShape = null;
              if (oc.XCAFDoc_ShapeTool?.GetShape_1) {
                const outShape = new oc.TopoDS_Shape();
                const success = oc.XCAFDoc_ShapeTool.GetShape_1(label, outShape);
                if (success && outShape && !outShape.IsNull()) {
                  labelShape = outShape;
                }
              }

              if (labelShape && !labelShape.IsNull()) {
                const isSet = colorTool.IsSet_2(labelShape, ct.val);
                if (isSet) {
                  logOCC(`[Label ${i}] IsSet_2(shape, ${ct.name}) = TRUE - shape has color!`);

                  // OPTION 3: Try GetColor_7 directly on this XCAF shape
                  if (typeof colorTool.GetColor_7 === 'function') {
                    try {
                      const color = new oc.Quantity_Color_1();
                      const hasColor = colorTool.GetColor_7(labelShape, ct.val, color);
                      if (hasColor) {
                        logOCC(`[Label ${i}] GetColor_7(xcafShape, ${ct.name}) = RGB(${color.Red().toFixed(3)}, ${color.Green().toFixed(3)}, ${color.Blue().toFixed(3)})`);
                      } else {
                        logOCC(`[Label ${i}] GetColor_7(xcafShape, ${ct.name}) returned false despite IsSet=true`);
                      }
                    } catch (e: any) {
                      logOCC(`[Label ${i}] GetColor_7 error: ${e.message || e}`);
                    }
                  }
                }
              }
            } catch (e: any) {
              // IsSet_2 failed silently
            }
          }
        }
      } catch (e: any) {
        logOCC(`[Label ${i}] IsSet diagnostic error: ${e.message || e}`);
      }

      // Count children using TDF_ChildIterator_2(label, allLevels)
      // Note: TDF_ChildIterator_1 = default ctor (0 args), TDF_ChildIterator_2 = (label, allLevels)
      try {
        if (oc.TDF_ChildIterator_2) {
          let childCount = 0;
          const childIter = new oc.TDF_ChildIterator_2(label, false);
          while (childIter.More()) {
            childCount++;
            childIter.Next();
          }
          logOCC(`[Label ${i}] Child count: ${childCount}`);
        } else {
          logOCC(`[Label ${i}] TDF_ChildIterator_2 not available`);
        }
      } catch (e: any) {
        logOCC(`[Label ${i}] Error counting children: ${e.message || e}`);
      }

      processLabel(label, null, 0);
    }

    logOCC(`\n[ShapeColorMap] Label traversal stats: ${labelsProcessed} processed, ${labelsWithShape} with shapes, ${labelsWithColor} with colors`);

    // DIAGNOSTIC: Check where the 24 colors are stored
    logOCC(`\n=== COLOR LABELS DIAGNOSTIC ===`);
    if (colorTool.GetColors) {
      const colorLabels = new oc.TDF_LabelSequence_1();
      colorTool.GetColors(colorLabels);
      logOCC(`[ColorLabels] GetColors returned ${colorLabels.Length()} color labels`);

      // Examine first few color labels
      for (let i = 1; i <= Math.min(colorLabels.Length(), 5); i++) {
        const colorLabel = colorLabels.Value(i);
        logOCC(`\n--- Color Label ${i} ---`);

        // Try to get the color from this label
        const color = getLabelColor(colorLabel);
        logOCC(`[ColorLabel ${i}] Color: ${color ? `RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)})` : 'none'}`);

        // Check if this color label has a shape using STATIC GetShape_1(label, outShape)
        try {
          let shape = null;
          if (oc.XCAFDoc_ShapeTool?.GetShape_1) {
            const outShape = new oc.TopoDS_Shape();
            const success = oc.XCAFDoc_ShapeTool.GetShape_1(colorLabel, outShape);
            if (success && outShape && !outShape.IsNull()) {
              shape = outShape;
            }
          }
          const hasShape = shape && !shape.IsNull();
          logOCC(`[ColorLabel ${i}] Has shape: ${hasShape}`);
        } catch (e) {
          logOCC(`[ColorLabel ${i}] GetShape_1 failed`);
        }

        // Try GetShapesOfColor - get shapes that have this color
        try {
          if (typeof colorTool.GetShapesOfColor === 'function') {
            const shapesWithColor = new oc.TDF_LabelSequence_1();
            colorTool.GetShapesOfColor(colorLabel, shapesWithColor);
            logOCC(`[ColorLabel ${i}] Shapes with this color: ${shapesWithColor.Length()}`);
          }
        } catch (e: any) {
          logOCC(`[ColorLabel ${i}] GetShapesOfColor failed: ${e.message || e}`);
        }
      }
    }
  } catch (e: any) {
    // TDF_LabelSequence not available - expected in OpenCascade.js
    logOCC('[ShapeColorMap] TDF_LabelSequence fallback error:', e.message || e);
  }

  // Log summary of what was found
  logOCC(`[ShapeColorMap] Color extraction stats: ${colorsFoundViaLabel} via label, ${colorsFoundViaShape} via shape`);
  logOCC(`[ShapeColorMap] getShapeColor attempts: ${shapeColorAttempts}`);
  if (shapeColorErrors.length > 0) {
    logOCC(`[ShapeColorMap] getShapeColor errors (first 3): ${shapeColorErrors.join('; ')}`);
  }
  if (faceColorMap.size > 0) {
    const uniqueColors = new Set<string>();
    for (const color of faceColorMap.values()) {
      uniqueColors.add(`${color.r.toFixed(2)},${color.g.toFixed(2)},${color.b.toFixed(2)}`);
    }
    logOCC(`[ShapeColorMap] Found ${faceColorMap.size} face->color mappings with ${uniqueColors.size} unique colors`);
  } else {
    logOCC('[ShapeColorMap] No colors extracted from XCAF labels');
  }

  return faceColorMap;
}

/**
 * Build a map of OCC solid index to face count and face hash codes.
 * Also builds a reverse map: face hash code → solid index.
 * This allows matching OCC solids to STEP solids by face count,
 * and looking up which solid a face belongs to.
 */
function buildOCCSolidFaceCounts(oc: any, shape: any): {
  solidMap: Map<number, { faceCount: number; faceHashCodes: number[] }>;
  faceToSolid: Map<number, number>;
} {
  const enableLoadDiagnostics = readGlobalBoolean('__ENABLE_LOAD_DIAGNOSTICS__', false);
  const solidMap = new Map<number, { faceCount: number; faceHashCodes: number[] }>();
  const faceToSolid = new Map<number, number>(); // face hash code → solid index

  try {
    // Iterate over all solids in the shape
    const solidExplorer = new oc.TopExp_Explorer_2(
      shape,
      oc.TopAbs_ShapeEnum.TopAbs_SOLID,
      oc.TopAbs_ShapeEnum.TopAbs_SHAPE
    );

    let solidIndex = 0;
    while (solidExplorer.More()) {
      const solid = solidExplorer.Current();
      const faceHashCodes: number[] = [];

      // Count faces in this solid
      const faceExplorer = new oc.TopExp_Explorer_2(
        solid,
        oc.TopAbs_ShapeEnum.TopAbs_FACE,
        oc.TopAbs_ShapeEnum.TopAbs_SHAPE
      );

      while (faceExplorer.More()) {
        const face = faceExplorer.Current();
        const hashCode = face.HashCode(2147483647);
        faceHashCodes.push(hashCode);
        faceToSolid.set(hashCode, solidIndex);
        faceExplorer.Next();
      }

      solidMap.set(solidIndex, { faceCount: faceHashCodes.length, faceHashCodes });
      solidIndex++;
      solidExplorer.Next();
    }

    if (enableLoadDiagnostics) {
      console.log(`[OCC_Solids] Found ${solidMap.size} solids, ${faceToSolid.size} face→solid mappings`);
      for (const [idx, data] of solidMap) {
        console.log(`[OCC_Solids] Solid ${idx}: ${data.faceCount} faces`);
      }
    }
  } catch (e) {
    if (enableLoadDiagnostics) {
      console.log('[OCC_Solids] Error iterating solids:', e);
    }
  }

  return { solidMap, faceToSolid };
}

/**
 * Match OCC solids to STEP solids by face count and build a face hash -> color map.
 * Also returns a solidToColor map (OCC solid index -> color) for propagation.
 */
function matchSolidsAndBuildColorMap(
  oc: any,
  shape: any,
  stepSolidMap: Map<number, { color: RGBColor; faceCount: number; faceIds: number[] }>,
  occSolidMap: Map<number, { faceCount: number; faceHashCodes: number[] }>
): { faceColorMap: Map<number, RGBColor>; solidToColor: Map<number, RGBColor> } {
  const enableLoadDiagnostics = readGlobalBoolean('__ENABLE_LOAD_DIAGNOSTICS__', false);
  const faceColorMap = new Map<number, RGBColor>();
  const solidToColor = new Map<number, RGBColor>(); // OCC solid index -> color

  // Group STEP solids by face count
  const stepByFaceCount = new Map<number, Array<{ solidId: number; color: RGBColor; faceCount: number }>>();
  for (const [solidId, data] of stepSolidMap) {
    const count = data.faceCount;
    if (!stepByFaceCount.has(count)) {
      stepByFaceCount.set(count, []);
    }
    stepByFaceCount.get(count)!.push({ solidId, color: data.color, faceCount: count });
  }

  // Group OCC solids by face count
  const occByFaceCount = new Map<number, Array<{ solidIdx: number; faceHashCodes: number[] }>>();
  for (const [solidIdx, data] of occSolidMap) {
    const count = data.faceCount;
    if (!occByFaceCount.has(count)) {
      occByFaceCount.set(count, []);
    }
    occByFaceCount.get(count)!.push({ solidIdx, faceHashCodes: data.faceHashCodes });
  }

  // Match solids by face count
  let matchedSolids = 0;
  let matchedFaces = 0;

  for (const [faceCount, stepSolids] of stepByFaceCount) {
    const occSolids = occByFaceCount.get(faceCount);
    if (!occSolids) continue;

    // If there's exactly one STEP solid and one OCC solid with this face count, match them
    if (stepSolids.length === 1 && occSolids.length === 1) {
      const color = stepSolids[0].color;
      const faceHashCodes = occSolids[0].faceHashCodes;
      const occSolidIdx = occSolids[0].solidIdx;

      solidToColor.set(occSolidIdx, color);
      for (const hashCode of faceHashCodes) {
        faceColorMap.set(hashCode, color);
        matchedFaces++;
      }
      matchedSolids++;

      if (enableLoadDiagnostics) {
        console.log(`[SolidMatch] Matched STEP #${stepSolids[0].solidId} to OCC solid ${occSolidIdx} (${faceCount} faces) -> RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)})`);
      }
    } else if (stepSolids.length > 0 && occSolids.length > 0) {
      // Multiple solids with same face count - try to match by order (imperfect but better than nothing)
      const minCount = Math.min(stepSolids.length, occSolids.length);
      for (let i = 0; i < minCount; i++) {
        const color = stepSolids[i].color;
        const faceHashCodes = occSolids[i].faceHashCodes;
        const occSolidIdx = occSolids[i].solidIdx;

        solidToColor.set(occSolidIdx, color);
        for (const hashCode of faceHashCodes) {
          faceColorMap.set(hashCode, color);
          matchedFaces++;
        }
        matchedSolids++;
      }
      if (enableLoadDiagnostics) {
        console.log(`[SolidMatch] Matched ${minCount} solids with ${faceCount} faces each (ambiguous)`);
      }
    }
  }

  if (enableLoadDiagnostics) {
    console.log(`[SolidMatch] Total: ${matchedSolids} solids matched, ${matchedFaces} faces colored`);
  }
  return { faceColorMap, solidToColor };
}

/**
 * Load a STEP file and return the TopoDS_Shape with color information
 * Accepts either Uint8Array (for large files) or string
 */
async function loadStepFile(fileContent: Uint8Array | string, fileName: string): Promise<StepLoadResult> {
  // Profile: initOC
  const initOCStart = performance.now();
  const oc = await initOC();
  tessellationProfile.loadStepFile_initOC.total += performance.now() - initOCStart;
  tessellationProfile.loadStepFile_initOC.calls++;

  // Perf-mode switches (used by benchmark harness to isolate geometry throughput).
  const preferGeometryOnlyLoad = readGlobalBoolean('__PERF_GEOMETRY_ONLY_LOAD__', false);
  const useXCAFReader = readGlobalBoolean('__ENABLE_XCAF_READER__', !preferGeometryOnlyLoad);
  const enableStepColorParsing = readGlobalBoolean('__ENABLE_STEP_COLOR_PARSING__', !preferGeometryOnlyLoad);
  // Expensive load-time diagnostics are opt-in. Keep disabled by default so
  // perf runs and normal interactive loads avoid heavy introspection/log spam.
  const enableLoadDiagnostics = readGlobalBoolean('__ENABLE_LOAD_DIAGNOSTICS__', false);

  // Debug APIs logged only when DEBUG_OCC is true
  if (DEBUG_OCC) {
    console.log('[OCC] Available STEPCAFControl_Reader constructors:',
      Object.keys(oc).filter(k => k.startsWith('STEPCAFControl_Reader')));
    console.log('[OCC] Available XCAFDoc APIs:',
      Object.keys(oc).filter(k => k.startsWith('XCAFDoc')).slice(0, 20));
  }

  // Convert to Uint8Array if string was passed
  let fileData: Uint8Array;
  let fileContentString: string | null = null;

  if (typeof fileContent === 'string') {
    // Legacy string input - convert to Uint8Array
    const encoder = new TextEncoder();
    fileData = encoder.encode(fileContent);
    fileContentString = fileContent;
  } else {
    // Uint8Array input (preferred for large files)
    fileData = fileContent;
  }

  // Write file to virtual filesystem using writeFile (handles large files better)
  oc.FS.writeFile('/' + fileName, fileData);

  let shape: any = null;
  let colorTool: any = null;
  let shapeTool: any = null;
  let doc: any = null;
  let xcafDocHandle: any = null;  // Handle for XCAFPrs_DocumentExplorer

  // Try XCAF reader first (supports colors)
  const hasXCAF = useXCAFReader && (oc.STEPCAFControl_Reader_1 || oc.STEPCAFControl_Reader);

  if (hasXCAF) {
    try {
      logOCC('Using STEPCAFControl_Reader for color support...');

      // Profile: create document
      const createDocStart = performance.now();

      // Create XDE document using Handle (correct OpenCascade.js API)
      const app = new oc.TDocStd_Application();
      const docHandle = new oc.Handle_TDocStd_Document_1();
      // TCollection_ExtendedString_2 requires (string, isMultiByte) parameters
      app.NewDocument(new oc.TCollection_ExtendedString_2("XmlXCAF", true), docHandle);

      if (docHandle.IsNull()) {
        throw new Error('Failed to create XCAF document');
      }
      doc = docHandle.get();
      xcafDocHandle = docHandle;  // Store for XCAFPrs_DocumentExplorer

      // Create XCAF STEP reader
      let cafReader;
      if (oc.STEPCAFControl_Reader_1) {
        cafReader = new oc.STEPCAFControl_Reader_1();
      } else {
        cafReader = new oc.STEPCAFControl_Reader();
      }

      tessellationProfile.loadStepFile_createDoc.total += performance.now() - createDocStart;
      tessellationProfile.loadStepFile_createDoc.calls++;

      // Enable all relevant reading modes (no profiling - fast)
      if (cafReader.SetColorMode) {
        cafReader.SetColorMode(true);
      }
      if (cafReader.SetLayerMode) {
        cafReader.SetLayerMode(true);
      }
      if (cafReader.SetNameMode) {
        cafReader.SetNameMode(true);
      }
      if (cafReader.SetMatMode) {
        cafReader.SetMatMode(true);
        if (enableLoadDiagnostics) {
          console.log('[XCAF] SetMatMode(true) - materials');
        }
      }
      if (cafReader.SetGDTMode) {
        cafReader.SetGDTMode(true);
      }

      // Profile: ReadFile
      const readFileStart = performance.now();
      const readResult = cafReader.ReadFile('/' + fileName);
      tessellationProfile.loadStepFile_readFile.total += performance.now() - readFileStart;
      tessellationProfile.loadStepFile_readFile.calls++;
      logOCC('[XCAF] ReadFile result:', readResult?.value);

      // IFSelect_RetDone is typically 1 in OpenCascade (0=RetVoid, 1=RetDone, 2=RetError...)
      // Check for value 1 (RetDone) or value 0 if enum mapping differs
      const isDone = (typeof readResult === 'object' && (readResult.value === 1 || readResult.value === 0)) ||
                     readResult === 1 || readResult === 0;

      if (isDone) {
        // Profile: Transfer
        const transferStart = performance.now();
        const progressRange = new oc.Message_ProgressRange_1();
        let transferResult: any;
        try {
          if (cafReader.Transfer_1) {
            transferResult = cafReader.Transfer_1(docHandle, progressRange);
          } else if (cafReader.Transfer) {
            transferResult = cafReader.Transfer(docHandle, progressRange);
          }
        } catch (transferErr: any) {
          console.error('[XCAF] Transfer failed:', transferErr.message || transferErr);
        }
        tessellationProfile.loadStepFile_transfer.total += performance.now() - transferStart;
        tessellationProfile.loadStepFile_transfer.calls++;
        logOCC('[XCAF] Transfer result:', transferResult);

        // === CHECK TRANSFER APIs for STEP entity to shape mapping ===
        // NOTE: Expensive debug logging - only run in DEBUG mode
        if (DEBUG_OCC) {
          console.log('[TransferAPI] Checking Transfer APIs for entity-to-shape mapping...');
          if (cafReader.Reader) {
            try {
              const innerReader = cafReader.Reader();
              console.log('[TransferAPI] cafReader.Reader() succeeded');
              const innerMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(innerReader) || {})
                .filter(k => typeof innerReader[k] === 'function');
              console.log('[TransferAPI] STEPControl_Reader methods:', innerMethods.join(', '));

              if (innerReader.WS) {
                const ws = innerReader.WS();
                console.log('[TransferAPI] innerReader.WS() succeeded (XSControl_WorkSession)');
                const wsMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(ws) || {})
                  .filter(k => typeof ws[k] === 'function');
                console.log('[TransferAPI] WorkSession methods:', wsMethods.join(', '));

                if (ws.TransferReader) {
                  const tr = ws.TransferReader();
                  console.log('[TransferAPI] ws.TransferReader() succeeded');
                  const trMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(tr) || {})
                    .filter(k => typeof tr[k] === 'function');
                  console.log('[TransferAPI] TransferReader methods:', trMethods.join(', '));

                  if (tr.TransientProcess) {
                    const tp = tr.TransientProcess();
                    console.log('[TransferAPI] tr.TransientProcess() succeeded');
                    const tpMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(tp) || {})
                      .filter(k => typeof tp[k] === 'function');
                    console.log('[TransferAPI] TransientProcess methods:', tpMethods.join(', '));

                    // Check key methods we need for entity mapping
                    console.log('[TransferAPI] KEY METHODS:');
                    console.log('  NbMapped:', typeof tp.NbMapped);
                    console.log('  Mapped:', typeof tp.Mapped);
                    console.log('  MapIndex:', typeof tp.MapIndex);
                    console.log('  MapItem:', typeof tp.MapItem);
                    console.log('  Find:', typeof tp.Find);
                    console.log('  FindTransient:', typeof tp.FindTransient);

                    if (typeof tp.NbMapped === 'function') {
                      console.log('[TransferAPI] NbMapped() =', tp.NbMapped());
                    }
                  } else {
                    console.log('[TransferAPI] TransferReader does NOT have TransientProcess');
                  }
                } else {
                  console.log('[TransferAPI] WorkSession does NOT have TransferReader');
                }
              } else {
                console.log('[TransferAPI] STEPControl_Reader does NOT have WS method');
              }
            } catch (e) {
              console.log('[TransferAPI] Error exploring Transfer APIs:', e);
            }
          } else {
            console.log('[TransferAPI] cafReader does NOT have Reader method');
          }
        }
        // === END CHECK TRANSFER APIs ===

        // Profile: getTools
        const getToolsStart = performance.now();

        // Get tools from document
        shapeTool = oc.XCAFDoc_DocumentTool.ShapeTool(doc.Main());
        colorTool = oc.XCAFDoc_DocumentTool.ColorTool(doc.Main());

        tessellationProfile.loadStepFile_getTools.total += performance.now() - getToolsStart;
        tessellationProfile.loadStepFile_getTools.calls++;
        logOCC('Got shapeTool:', !!shapeTool, 'colorTool:', !!colorTool);

        // DIAGNOSTIC: Investigate colorTool APIs - only in debug mode
        if (DEBUG_OCC) {
          const colorToolAPIs = Object.keys(oc).filter(k => k.includes('XCAFDoc_ColorTool'));
          console.log('[ColorDiag] XCAFDoc_ColorTool APIs in OC:', colorToolAPIs.join(', '));

          // Try to instantiate XCAFDoc_ColorTool directly
          for (const apiName of colorToolAPIs) {
            try {
              const api = oc[apiName];
              if (typeof api === 'function') {
                console.log(`[ColorDiag] ${apiName} is a function/constructor`);
                // Check prototype methods
                if (api.prototype) {
                  const protoMethods = Object.getOwnPropertyNames(api.prototype);
                  console.log(`[ColorDiag] ${apiName}.prototype methods:`, protoMethods.slice(0, 20).join(', '));
                }
              }
            } catch (e) {
              console.log(`[ColorDiag] ${apiName} inspection error:`, e);
            }
          }

          // Also check what the DocumentTool.ColorTool returns
          console.log('[ColorDiag] XCAFDoc_DocumentTool methods:',
            Object.getOwnPropertyNames(oc.XCAFDoc_DocumentTool || {}).join(', '));
          console.log('[ColorDiag] colorTool type:', typeof colorTool);
          console.log('[ColorDiag] colorTool constructor:', colorTool?.constructor?.name);

          // Check if it's a Handle that needs unwrapping
          if (colorTool) {
            const hasGet = typeof colorTool.get === 'function';
            const hasIsNull = typeof colorTool.IsNull === 'function';
            console.log('[ColorDiag] colorTool.get exists:', hasGet);
            console.log('[ColorDiag] colorTool.IsNull exists:', hasIsNull);

            if (hasIsNull) {
              console.log('[ColorDiag] colorTool.IsNull():', colorTool.IsNull());
            }

            // List all methods on colorTool
            const allProps = Object.getOwnPropertyNames(Object.getPrototypeOf(colorTool) || {});
            console.log('[ColorDiag] colorTool prototype methods:', allProps.slice(0, 30).join(', '));

            // Try to get the actual tool if it's a handle
            let actualColorTool = colorTool;
            if (hasGet && !colorTool.IsNull?.()) {
              try {
                actualColorTool = colorTool.get();
                console.log('[ColorDiag] Unwrapped colorTool type:', actualColorTool?.constructor?.name);
                const actualProps = Object.getOwnPropertyNames(Object.getPrototypeOf(actualColorTool) || {});
                console.log('[ColorDiag] Unwrapped colorTool methods:', actualProps.slice(0, 30).join(', '));
              } catch (e) {
                console.log('[ColorDiag] Failed to unwrap colorTool:', e);
              }
            }
          }

          // === CHECK FOR XCAFPrs_DocumentExplorer ===
          console.log('\n=== XCAFPrs_DocumentExplorer Check ===');
          const xcafPrsAPIs = Object.keys(oc).filter(k => k.includes('XCAFPrs'));
          console.log('[XCAFPrs] Available XCAFPrs APIs:', xcafPrsAPIs.length > 0 ? xcafPrsAPIs.join(', ') : 'NONE');

          if (oc.XCAFPrs_DocumentExplorer) {
            console.log('[XCAFPrs] XCAFPrs_DocumentExplorer IS available!');
            try {
              const explorerProto = Object.getOwnPropertyNames(oc.XCAFPrs_DocumentExplorer.prototype || {});
              console.log('[XCAFPrs] XCAFPrs_DocumentExplorer.prototype methods:', explorerProto.join(', '));
              const explorerConstructors = Object.keys(oc).filter(k => k.startsWith('XCAFPrs_DocumentExplorer'));
              console.log('[XCAFPrs] DocumentExplorer constructors:', explorerConstructors.join(', '));
            } catch (e) {
              console.log('[XCAFPrs] Error inspecting XCAFPrs_DocumentExplorer:', e);
            }
          } else {
            console.log('[XCAFPrs] XCAFPrs_DocumentExplorer is NOT directly available');
            const xcafPrsStyles = Object.keys(oc).filter(k => k.includes('XCAFPrs_Style'));
            console.log('[XCAFPrs] XCAFPrs_Style APIs:', xcafPrsStyles.length > 0 ? xcafPrsStyles.join(', ') : 'NONE');
          }
          console.log('=== End XCAFPrs Check ===\n');
        }

        // Get all shapes from the document
        // TDF_LabelSequence may not be available in this build, try alternatives
        let labelsLength = 0;
        let labels: any = null;

        // Try to create TDF_LabelSequence
        try {
          if (oc.TDF_LabelSequence_1) {
            labels = new oc.TDF_LabelSequence_1();
          } else if (oc.TDF_LabelSequence) {
            labels = new oc.TDF_LabelSequence();
          }

          if (labels && shapeTool.GetFreeShapes) {
            shapeTool.GetFreeShapes(labels);
            labelsLength = labels.Length();
          }
        } catch (labelErr) {
          console.log('[ColorDiag] TDF_LabelSequence not available:', labelErr);
          // List what TDF APIs ARE available
          const tdfApis = Object.keys(oc).filter(k => k.startsWith('TDF_')).slice(0, 20);
          console.log('[ColorDiag] Available TDF APIs:', tdfApis.join(', '));
        }

        // Alternative: try to get shape directly from reader
        if (labelsLength === 0 && cafReader) {
          if (enableLoadDiagnostics) {
            console.log('[ColorDiag] Trying alternative shape extraction...');
          }

          // Check what methods the reader has
          if (enableLoadDiagnostics) {
            const readerMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(cafReader) || {})
              .filter(k => typeof cafReader[k] === 'function');
            console.log('[ColorDiag] cafReader methods:', readerMethods.slice(0, 20).join(', '));
          }

          // Try Reader().OneShape() pattern
          if (cafReader.Reader && typeof cafReader.Reader === 'function') {
            try {
              const innerReader = cafReader.Reader();
              if (innerReader && innerReader.OneShape) {
                shape = innerReader.OneShape();
                if (enableLoadDiagnostics) {
                  const isNull = shape?.IsNull?.() ?? true;
                  const shapeType = shape?.ShapeType?.() ?? 'unknown';
                  console.log('[ColorDiag] Got shape from cafReader.Reader().OneShape()',
                    'IsNull:', isNull, 'ShapeType:', shapeType);
                }
              }
            } catch (e) {
              if (enableLoadDiagnostics) {
                console.log('[ColorDiag] cafReader.Reader().OneShape() failed:', e);
              }
            }
          }

          // Try NbRootsForTransfer pattern
          if ((!shape || shape.IsNull?.()) && cafReader.NbRootsForTransfer) {
            try {
              const numRoots = cafReader.NbRootsForTransfer();
              if (enableLoadDiagnostics) {
                console.log('[ColorDiag] NbRootsForTransfer:', numRoots);
              }
            } catch (e) {
              // Not available
            }
          }
        }

        logOCC('Free shapes count:', labelsLength);

        if (labelsLength > 0 && labels) {
          // DIAGNOSTIC: Try to extract colors from labels
          console.log('[ColorDiag] Trying to extract colors from', labels.Length(), 'free shapes...');

          // Get the actual colorTool (unwrap if needed)
          let actualColorTool = colorTool;
          if (typeof colorTool?.get === 'function' && !colorTool.IsNull?.()) {
            try {
              actualColorTool = colorTool.get();
            } catch (e) {
              console.log('[ColorDiag] Could not unwrap colorTool');
            }
          }

          // Try different color extraction approaches on the first label
          const firstLabel = labels.Value(1);
          const firstShape = shapeTool.GetShape(firstLabel);

          // Systematically test all GetColor variants on both label and shape
          if (actualColorTool) {
            const color = new oc.Quantity_Color_1();
            const methodNames = ['GetColor_1', 'GetColor_2', 'GetColor_4', 'GetColor_5', 'GetColor_6', 'GetColor_7', 'GetColor_8'];

            console.log('[ColorDiag] Testing GetColor methods on first shape/label...');

            // Test each method with shape (colorType 1 = Surface)
            for (const methodName of methodNames) {
              if (typeof actualColorTool[methodName] === 'function') {
                try {
                  const result = actualColorTool[methodName](firstShape, 1, color);
                  if (result) {
                    console.log(`[ColorDiag] ${methodName}(shape, 1, color) = TRUE! RGB(${color.Red().toFixed(2)}, ${color.Green().toFixed(2)}, ${color.Blue().toFixed(2)})`);
                  }
                } catch (e) {
                  // Method signature didn't match - that's OK
                }
              }
            }

            // Test each method with label (colorType 1 = Surface)
            for (const methodName of methodNames) {
              if (typeof actualColorTool[methodName] === 'function') {
                try {
                  const result = actualColorTool[methodName](firstLabel, 1, color);
                  if (result) {
                    console.log(`[ColorDiag] ${methodName}(label, 1, color) = TRUE! RGB(${color.Red().toFixed(2)}, ${color.Green().toFixed(2)}, ${color.Blue().toFixed(2)})`);
                  }
                } catch (e) {
                  // Method signature didn't match - that's OK
                }
              }
            }

            // Try IsSet variants
            const isSetMethods = ['IsSet_1', 'IsSet_2'];
            for (const methodName of isSetMethods) {
              if (typeof actualColorTool[methodName] === 'function') {
                try {
                  const result = actualColorTool[methodName](firstLabel, 1);
                  console.log(`[ColorDiag] ${methodName}(label, 1) = ${result}`);
                } catch (e) {
                  // Signature didn't match
                }
                try {
                  const result = actualColorTool[methodName](firstShape, 1);
                  console.log(`[ColorDiag] ${methodName}(shape, 1) = ${result}`);
                } catch (e) {
                  // Signature didn't match
                }
              }
            }
          }

          // Get compound of all shapes
          const builder = new oc.BRep_Builder();
          const compound = new oc.TopoDS_Compound();
          builder.MakeCompound(compound);

          for (let i = 1; i <= labels.Length(); i++) {
            const label = labels.Value(i);
            const shapeFromLabel = shapeTool.GetShape(label);
            if (shapeFromLabel && !shapeFromLabel.IsNull()) {
              builder.Add(compound, shapeFromLabel);
            }
          }
          shape = compound;
        }
      }
    } catch (xcafErr) {
      logOCC('XCAF reader failed, falling back to basic reader:', xcafErr);
    }
  }

  // Fallback to basic reader if XCAF failed
  if (!shape || (shape.IsNull && shape.IsNull())) {
    logOCC('Using basic STEPControl_Reader (no color support)...');

    let reader;
    if (oc.STEPControl_Reader_1) {
      reader = new oc.STEPControl_Reader_1();
    } else if (oc.STEPControl_Reader) {
      reader = new oc.STEPControl_Reader();
    } else {
      throw new Error('STEPControl_Reader not found');
    }

    const readResult = reader.ReadFile('/' + fileName);
    logOCC('ReadFile result:', readResult);

    // IFSelect_RetDone is typically 0 in OpenCascade
    // ReadFile returns an object with .value in emscripten bindings
    const isDone = readResult === oc.IFSelect_ReturnStatus?.IFSelect_RetDone ||
                   readResult === 0 ||
                   (typeof readResult === 'object' && (readResult.value === 0 || readResult.value === 1));
    if (!isDone) {
      oc.FS.unlink('/' + fileName);
      const resultValue = typeof readResult === 'object' ? JSON.stringify(readResult) : readResult;
      throw new Error(`Failed to read STEP file: ${resultValue}`);
    }

    logOCC('Transferring roots...');
    if (reader.TransferRoots) {
      try {
        reader.TransferRoots();
      } catch (e) {
        if (oc.Message_ProgressRange_1) {
          reader.TransferRoots(new oc.Message_ProgressRange_1());
        } else if (oc.Message_ProgressRange) {
          reader.TransferRoots(new oc.Message_ProgressRange());
        }
      }
    } else if (reader.TransferRoot) {
      reader.TransferRoot();
    }

    shape = reader.OneShape();
  }

  // Debug shape info only when DEBUG_OCC is true
  if (DEBUG_OCC) {
    console.log('[OCC] Getting shape...');
    if (!shape || shape.IsNull()) {
      console.error('[OCC] Shape is null or empty!');
    } else {
      console.log('[OCC] Shape type:', shape.ShapeType ? shape.ShapeType() : 'unknown');
      const countShapes = (_shapeType: string, enumValue: any) => {
        const exp = new oc.TopExp_Explorer_2(shape, enumValue, oc.TopAbs_ShapeEnum.TopAbs_SHAPE);
        let count = 0;
        while (exp.More()) {
          count++;
          exp.Next();
        }
        return count;
      };
      console.log('[OCC] Shape contains:');
      console.log('  - Compounds:', countShapes('Compound', oc.TopAbs_ShapeEnum.TopAbs_COMPOUND));
      console.log('  - Solids:', countShapes('Solid', oc.TopAbs_ShapeEnum.TopAbs_SOLID));
      console.log('  - Shells:', countShapes('Shell', oc.TopAbs_ShapeEnum.TopAbs_SHELL));
      console.log('  - Faces:', countShapes('Face', oc.TopAbs_ShapeEnum.TopAbs_FACE));
    }
  }

  // Clean up
  oc.FS.unlink('/' + fileName);

  const buildEmptyColorResult = (): StepLoadResult => {
    // Still need to unwrap tools before returning.
    let unwrappedColorTool = colorTool;
    let unwrappedShapeTool = shapeTool;
    if (colorTool && typeof colorTool.get === 'function' && !colorTool.IsNull?.()) {
      try { unwrappedColorTool = colorTool.get(); } catch (e) {}
    }
    if (shapeTool && typeof shapeTool.get === 'function' && !shapeTool.IsNull?.()) {
      try { unwrappedShapeTool = shapeTool.get(); } catch (e) {}
    }

    return {
      shape,
      colorTool: unwrappedColorTool,
      shapeTool: unwrappedShapeTool,
      doc,
      stepColors: new Map<number, RGBColor>(),
      shapeColorMap: new Map<number, RGBColor>(),
      faceIdOrder: [],
      geometryColorMap: new Map<string, RGBColor>(),
      solidMatchedColors: new Map<number, RGBColor>(),
      faceToSolid: new Map<number, number>(),
      solidToColor: new Map<number, RGBColor>(),
    };
  };

  // Profile: Color parsing
  const colorParsingStart = performance.now();
  if (!enableStepColorParsing) {
    const colorParsingTotal = performance.now() - colorParsingStart;
    tessellationProfile.loadStepFile_colorParsing.total += colorParsingTotal;
    tessellationProfile.loadStepFile_colorParsing.calls++;
    if (enableLoadDiagnostics) {
      console.log(`[ColorParsing] Disabled by __ENABLE_STEP_COLOR_PARSING__/__PERF_GEOMETRY_ONLY_LOAD__, skipped in ${colorParsingTotal.toFixed(1)}ms`);
    }
    return buildEmptyColorResult();
  }

  if (fileContentString === null) {
    const decoder = new TextDecoder('utf-8');
    fileContentString = decoder.decode(fileData);
  }
  logOCC('Parsing colors from STEP text...');

  // Sub-profile: parseStepColors
  let t0 = performance.now();
  const colorEntities = parseStepColors(fileContentString);
  const parseStepColorsTime = performance.now() - t0;

  // FAST PATH: If no color entities found, skip all expensive color extraction
  if (colorEntities.size === 0) {
    const colorParsingTotal = performance.now() - colorParsingStart;
    tessellationProfile.loadStepFile_colorParsing.total += colorParsingTotal;
    tessellationProfile.loadStepFile_colorParsing.calls++;
    if (enableLoadDiagnostics) {
      console.log(`[ColorParsing] No colors in file, skipped in ${colorParsingTotal.toFixed(1)}ms`);
    }
    return buildEmptyColorResult();
  }

  // Sub-profile: buildFaceColorMap
  t0 = performance.now();
  const stepColors = buildFaceColorMap(fileContentString, colorEntities);
  const buildFaceColorMapTime = performance.now() - t0;

  // Sub-profile: extractFaceIdOrder
  t0 = performance.now();
  const faceIdOrder = extractFaceIdOrder(fileContentString);
  const extractFaceIdOrderTime = performance.now() - t0;

  // Sub-profile: buildGeometryColorMap
  t0 = performance.now();
  const geometryColorMap = buildGeometryColorMap(fileContentString, stepColors);
  const buildGeometryColorMapTime = performance.now() - t0;

  // Sub-profile: buildSolidColorMap
  t0 = performance.now();
  const solidColorMap = buildSolidColorMap(fileContentString, colorEntities);
  const buildSolidColorMapTime = performance.now() - t0;

  // Sub-profile: buildOCCSolidFaceCounts
  t0 = performance.now();
  const { solidMap: occSolidFaceCounts, faceToSolid } = buildOCCSolidFaceCounts(oc, shape);
  const buildOCCSolidFaceCountsTime = performance.now() - t0;

  // Sub-profile: matchSolidsAndBuildColorMap
  t0 = performance.now();
  const { faceColorMap: solidMatchedColors, solidToColor } = matchSolidsAndBuildColorMap(oc, shape, solidColorMap, occSolidFaceCounts);
  const matchSolidsTime = performance.now() - t0;

  // Run comprehensive XCAF diagnostic to understand what's available
  // NOTE: This is expensive - only run in debug mode
  if (DEBUG_OCC) {
    diagnoseXCAFColorExtraction(oc, shape, colorTool, shapeTool);
  }

  // Sub-profile: buildShapeColorMap
  t0 = performance.now();
  const shapeColorMap = buildShapeColorMap(oc, shape, shapeTool, colorTool, xcafDocHandle);
  const buildShapeColorMapTime = performance.now() - t0;

  const colorParsingTotal = performance.now() - colorParsingStart;
  tessellationProfile.loadStepFile_colorParsing.total += colorParsingTotal;
  tessellationProfile.loadStepFile_colorParsing.calls++;

  // Log color parsing breakdown (always, for now)
  if (enableLoadDiagnostics) {
    console.log(`[ColorParsing] Breakdown: parseStepColors=${parseStepColorsTime.toFixed(1)}ms, buildFaceColorMap=${buildFaceColorMapTime.toFixed(1)}ms, extractFaceIdOrder=${extractFaceIdOrderTime.toFixed(1)}ms, buildGeometryColorMap=${buildGeometryColorMapTime.toFixed(1)}ms, buildSolidColorMap=${buildSolidColorMapTime.toFixed(1)}ms, buildOCCSolidFaceCounts=${buildOCCSolidFaceCountsTime.toFixed(1)}ms, matchSolids=${matchSolidsTime.toFixed(1)}ms, buildShapeColorMap=${buildShapeColorMapTime.toFixed(1)}ms, total=${colorParsingTotal.toFixed(1)}ms`);
  }

  // IMPORTANT: Unwrap colorTool and shapeTool handles before returning
  // The tools from XCAFDoc_DocumentTool are Handles - getFaceColor needs the actual tool
  let unwrappedColorTool = colorTool;
  let unwrappedShapeTool = shapeTool;

  if (colorTool && typeof colorTool.get === 'function' && !colorTool.IsNull?.()) {
    try {
      unwrappedColorTool = colorTool.get();
      logOCC('[loadStepFile] Unwrapped colorTool for return');
    } catch (e) {
      logOCC('[loadStepFile] Failed to unwrap colorTool:', e);
    }
  }

  if (shapeTool && typeof shapeTool.get === 'function' && !shapeTool.IsNull?.()) {
    try {
      unwrappedShapeTool = shapeTool.get();
      logOCC('[loadStepFile] Unwrapped shapeTool for return');
    } catch (e) {
      logOCC('[loadStepFile] Failed to unwrap shapeTool:', e);
    }
  }

  return { shape, colorTool: unwrappedColorTool, shapeTool: unwrappedShapeTool, doc, stepColors, shapeColorMap, faceIdOrder, geometryColorMap, solidMatchedColors, faceToSolid, solidToColor };
}

/**
 * Count the number of faces in a shape
 */
async function countFaces(shape: any): Promise<number> {
  const oc = await initOC();

  let faceCount = 0;
  const explorer = new oc.TopExp_Explorer_2(
    shape,
    oc.TopAbs_ShapeEnum.TopAbs_FACE,
    oc.TopAbs_ShapeEnum.TopAbs_SHAPE
  );

  while (explorer.More()) {
    faceCount++;
    explorer.Next();
  }

  return faceCount;
}

/**
 * Run Checkpoint 1 test
 */
async function runCheckpoint1(stepFileContent: string): Promise<{ success: boolean; faceCount: number; error?: string }> {
  try {
    console.log('[Checkpoint 1] Loading STEP file...');
    const { shape } = await loadStepFile(stepFileContent, 'test.step');

    console.log('[Checkpoint 1] Counting faces...');
    const faceCount = await countFaces(shape);

    console.log(`[Checkpoint 1] Found ${faceCount} faces`);

    const success = faceCount === 6;
    if (success) {
      console.log('[Checkpoint 1] ✓ PASSED: simple-cube.step has 6 faces');
    } else {
      console.log(`[Checkpoint 1] ✗ FAILED: Expected 6 faces, got ${faceCount}`);
    }

    return { success, faceCount };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('[Checkpoint 1] ✗ ERROR:', errorMsg);
    return { success: false, faceCount: 0, error: errorMsg };
  }
}

// ============================================================================
// CHECKPOINT 2: Extract parametric surfaces from each face
// ============================================================================

interface SurfaceInfo {
  faceIndex: number;
  surfaceType: string;
  uvBounds: {
    uMin: number;
    uMax: number;
    vMin: number;
    vMax: number;
  };
}

/**
 * Get the surface type name from a Geom_Surface Handle
 * @param oc - OpenCascade instance
 * @param surfaceHandle - Handle<Geom_Surface> (do NOT call .get() on this)
 */
function getSurfaceTypeName(oc: any, surfaceHandle: any): string {
  if (!surfaceHandle) {
    return 'Unknown(null)';
  }

  // Try using GeomAdaptor_Surface to identify the type
  // This expects the Handle directly, NOT the unwrapped surface
  try {
    if (oc.GeomAdaptor_Surface_2) {
      const adaptor = new oc.GeomAdaptor_Surface_2(surfaceHandle);
      const surfType = adaptor.GetType();

      // Map enum to string - GeomAbs_SurfaceType values
      const typeMap: Record<number, string> = {
        0: 'Plane',
        1: 'Cylinder',
        2: 'Cone',
        3: 'Sphere',
        4: 'Torus',
        5: 'BezierSurface',
        6: 'BSplineSurface',
        7: 'SurfaceOfRevolution',
        8: 'SurfaceOfExtrusion',
        9: 'OffsetSurface',
        10: 'OtherSurface'
      };

      // surfType might be an object with a .value property (opencascade.js enum)
      const typeValue = typeof surfType === 'object' && surfType !== null ? surfType.value : surfType;
      return typeMap[typeValue] || `Unknown(${typeValue})`;
    }
  } catch (e) {
    logOCC('GeomAdaptor_Surface failed:', e);
  }

  // Fallback: Try to get the dynamic type name from the unwrapped surface
  try {
    const actualSurface = typeof surfaceHandle.get === 'function' ? surfaceHandle.get() : surfaceHandle;
    if (actualSurface && typeof actualSurface.DynamicType === 'function') {
      const typeHandle = actualSurface.DynamicType();
      if (typeHandle) {
        if (typeof typeHandle.Name === 'function') {
          return typeHandle.Name();
        }
        if (typeHandle.$$ && typeHandle.$$.ptrType && typeHandle.$$.ptrType.registeredClass) {
          return typeHandle.$$.ptrType.registeredClass.name;
        }
      }
    }
  } catch (e) {
    logOCC('DynamicType fallback failed:', e);
  }

  return `Unknown(${String(surfaceHandle)})`;
}

/**
 * Extract surface information from all faces in a shape
 */
async function extractSurfaces(shape: any): Promise<SurfaceInfo[]> {
  const oc = await initOC();
  const surfaces: SurfaceInfo[] = [];

  // Log available TopoDS APIs for debugging (only when DEBUG_OCC is true)
  if (DEBUG_OCC) {
    console.log('[OCC] Available TopoDS APIs:',
      Object.keys(oc).filter(k => k.startsWith('TopoDS')).slice(0, 30));
    console.log('[OCC] Available BRep_Tool APIs:',
      Object.keys(oc).filter(k => k.startsWith('BRep_Tool')));
    console.log('[OCC] Available BRepTools APIs:',
      Object.keys(oc).filter(k => k.startsWith('BRepTools')));
  }

  const explorer = new oc.TopExp_Explorer_2(
    shape,
    oc.TopAbs_ShapeEnum.TopAbs_FACE,
    oc.TopAbs_ShapeEnum.TopAbs_SHAPE
  );

  let faceIndex = 0;
  while (explorer.More()) {
    const currentShape = explorer.Current();
    logOCC(`Face ${faceIndex} - currentShape type:`, typeof currentShape);

    try {
      // Try different approaches to cast to face
      let face = currentShape;

      // In opencascade.js, TopoDS static methods might be accessed differently
      if (oc.TopoDS && typeof oc.TopoDS.Face_1 === 'function') {
        face = oc.TopoDS.Face_1(currentShape);
      } else if (oc.TopoDS && typeof oc.TopoDS.Face === 'function') {
        face = oc.TopoDS.Face(currentShape);
      }

      // Get the surface from the face using BRep_Tool::Surface
      let surface = null;

      // Try different API patterns
      if (oc.BRep_Tool && typeof oc.BRep_Tool.Surface_2 === 'function') {
        surface = oc.BRep_Tool.Surface_2(face);
      } else if (oc.BRep_Tool && typeof oc.BRep_Tool.Surface === 'function') {
        surface = oc.BRep_Tool.Surface(face);
      } else if (typeof oc.BRep_Tool_Surface_2 === 'function') {
        surface = oc.BRep_Tool_Surface_2(face);
      } else if (typeof oc.BRep_Tool_Surface === 'function') {
        surface = oc.BRep_Tool_Surface(face);
      }

      if (!surface) {
        logOCC(`No surface found for face ${faceIndex}`);
        explorer.Next();
        faceIndex++;
        continue;
      }

      // BRep_Tool.Surface_2 returns a Handle<Geom_Surface>
      const surfaceType = getSurfaceTypeName(oc, surface);

      // Get UV bounds using BRepAdaptor_Surface
      let uMin = 0, uMax = 0, vMin = 0, vMax = 0;

      try {
        // BRepAdaptor_Surface_2 constructor takes a TopoDS_Face
        if (oc.BRepAdaptor_Surface_2) {
          const faceAdaptor = new oc.BRepAdaptor_Surface_2(face, true);
          uMin = faceAdaptor.FirstUParameter();
          uMax = faceAdaptor.LastUParameter();
          vMin = faceAdaptor.FirstVParameter();
          vMax = faceAdaptor.LastVParameter();
        } else if (oc.BRepAdaptor_Surface_1) {
          const faceAdaptor = new oc.BRepAdaptor_Surface_1();
          faceAdaptor.Initialize_1(face, true);
          uMin = faceAdaptor.FirstUParameter();
          uMax = faceAdaptor.LastUParameter();
          vMin = faceAdaptor.FirstVParameter();
          vMax = faceAdaptor.LastVParameter();
        }
      } catch (boundsErr) {
        logOCC('BRepAdaptor_Surface failed:', boundsErr);
      }

      surfaces.push({
        faceIndex,
        surfaceType,
        uvBounds: { uMin, uMax, vMin, vMax }
      });

      logOCC(`Face ${faceIndex}: ${surfaceType}, UV=[${uMin.toFixed(2)},${uMax.toFixed(2)}]x[${vMin.toFixed(2)},${vMax.toFixed(2)}]`);

    } catch (e) {
      console.error(`[OCC] Error extracting surface for face ${faceIndex}:`, e);
    }

    explorer.Next();
    faceIndex++;
  }

  return surfaces;
}

/**
 * Run Checkpoint 2 test
 */
async function runCheckpoint2(stepFileContent: string): Promise<{
  success: boolean;
  surfaces: SurfaceInfo[];
  error?: string;
}> {
  try {
    console.log('[Checkpoint 2] Loading STEP file...');
    const { shape } = await loadStepFile(stepFileContent, 'test.step');

    console.log('[Checkpoint 2] Extracting surfaces...');
    const surfaces = await extractSurfaces(shape);

    console.log(`[Checkpoint 2] Extracted ${surfaces.length} surfaces`);

    // For simple-cube.step, all 6 faces should be planes
    const allPlanes = surfaces.every(s => s.surfaceType === 'Plane');
    const hasValidUV = surfaces.every(s =>
      isFinite(s.uvBounds.uMin) && isFinite(s.uvBounds.uMax) &&
      isFinite(s.uvBounds.vMin) && isFinite(s.uvBounds.vMax)
    );

    const success = surfaces.length === 6 && allPlanes && hasValidUV;

    if (success) {
      console.log('[Checkpoint 2] ✓ PASSED: All 6 faces are planes with valid UV bounds');
    } else {
      if (surfaces.length !== 6) {
        console.log(`[Checkpoint 2] ✗ FAILED: Expected 6 surfaces, got ${surfaces.length}`);
      }
      if (!allPlanes) {
        const nonPlanes = surfaces.filter(s => s.surfaceType !== 'Plane');
        console.log(`[Checkpoint 2] ✗ FAILED: Not all surfaces are planes:`, nonPlanes);
      }
      if (!hasValidUV) {
        console.log('[Checkpoint 2] ✗ FAILED: Some UV bounds are invalid');
      }
    }

    return { success, surfaces };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('[Checkpoint 2] ✗ ERROR:', errorMsg);
    return { success: false, surfaces: [], error: errorMsg };
  }
}

/**
 * Information about an edge/curve
 */
interface EdgeInfo {
  edgeIndex: number;
  curveType: string;
  startPoint: { x: number; y: number; z: number };
  endPoint: { x: number; y: number; z: number };
  // For lines: direction vector
  // For circles/arcs: center and radius
  parameters?: Record<string, number>;
  // Sampled points along the curve (for curved edges)
  sampledPoints?: Array<{ x: number; y: number; z: number }>;
}

/**
 * Axis2Placement3D - local coordinate system
 */
interface Axis2Placement3D {
  location: Vec3;
  axis: Vec3;        // Z direction
  refDirection: Vec3; // X direction
}

/**
 * B-spline surface parameters
 */
interface BSplineParams {
  controlPoints: Vec3[][];  // 2D grid [v][u] of control points
  uDegree: number;
  vDegree: number;
  uKnots: number[];         // Expanded knot vector (with multiplicities)
  vKnots: number[];
  weights?: number[][];     // For NURBS (rational B-splines)
}

/**
 * Surface parameters extracted from OCC
 */
interface SurfaceParams {
  placement?: Axis2Placement3D;  // For analytic surfaces (cylinder, sphere, etc.)
  radius?: number;       // For cylinder, sphere, cone
  semiAngle?: number;    // For cone
  majorRadius?: number;  // For torus
  minorRadius?: number;  // For torus
  bspline?: BSplineParams; // For B-spline surfaces
}

/**
 * Information about a face with its boundary edges
 */
interface FaceWithEdgesInfo {
  faceIndex: number;
  surfaceType: string;
  uvBounds: { uMin: number; uMax: number; vMin: number; vMax: number };
  outerLoop: EdgeInfo[];
  innerLoops: EdgeInfo[][]; // Holes
  occFace?: any; // TopoDS_Face (used for robust point-in-face UV classification)
  occSurface?: any; // Handle<Geom_Surface> (used for UV projection on curved faces)
  surfaceParams?: SurfaceParams; // For curved surfaces
  color?: RGBColor; // Face color from STEP styling
  isReversed?: boolean; // True if face orientation is REVERSED (normals should flip)
}

type FaceTessellationStatus = 'ok' | 'skipped' | 'error';

export interface OccFaceTessellationDiagnostic {
  faceIndex: number;
  stepFaceId?: number;
  surfaceType: string;
  status: FaceTessellationStatus;
  isReversed: boolean;
  outerEdgeCount: number;
  innerLoopCount: number;
  outputVertexCount: number;
  outputTriangleCount: number;
  durationMs: number;
  error?: string;
}

export interface OcctImportFaceDiagnostic {
  globalFaceIndex: number;
  meshIndex: number;
  meshName?: string;
  firstTriangle: number;
  lastTriangle: number;
  triangleCount: number;
  hasNativeColor: boolean;
}

export interface FaceDiffRow {
  faceIndex: number;
  stepFaceId?: number;
  ours?: OccFaceTessellationDiagnostic;
  reference?: OcctImportFaceDiagnostic;
  triangleDelta?: number;
  triangleDeltaPct?: number;
}

export interface FaceDiffReport {
  generatedAtIso: string;
  targetFaceIndices: number[];
  assumptions: string[];
  totals: {
    oursFaceCount: number;
    referenceFaceCount: number;
    oursTriangles: number;
    referenceTriangles: number;
  };
  rows: FaceDiffRow[];
}

/**
 * Get the curve type name from a Geom_Curve Handle
 */
function getCurveTypeName(oc: any, curveHandle: any): string {
  if (!curveHandle) {
    return 'Unknown(null)';
  }

  try {
    // Use GeomAdaptor_Curve to identify the curve type
    if (oc.GeomAdaptor_Curve_2) {
      const adaptor = new oc.GeomAdaptor_Curve_2(curveHandle);
      const curveType = adaptor.GetType();

      // Map enum to string - GeomAbs_CurveType values
      const typeMap: Record<number, string> = {
        0: 'Line',
        1: 'Circle',
        2: 'Ellipse',
        3: 'Hyperbola',
        4: 'Parabola',
        5: 'BezierCurve',
        6: 'BSplineCurve',
        7: 'OffsetCurve',
        8: 'OtherCurve'
      };

      const typeValue = typeof curveType === 'object' && curveType !== null ? curveType.value : curveType;
      return typeMap[typeValue] || `Unknown(${typeValue})`;
    }
  } catch (e) {
    logOCC('GeomAdaptor_Curve failed:', e);
  }

  return 'Unknown';
}

/**
 * Extract boundary edges from a face
 */
async function extractFaceEdges(oc: any, face: any, faceIndex: number): Promise<{ outerLoop: EdgeInfo[]; innerLoops: EdgeInfo[][] }> {
  const wires: EdgeInfo[][] = [];
  const wireHashes: number[] = [];

  // Get the outer wire hash for comparison
  let outerWireHash: number | null = null;
  if (oc.BRepTools && oc.BRepTools.OuterWire) {
    try {
      const outerWire = oc.BRepTools.OuterWire(face);
      if (outerWire && !outerWire.IsNull()) {
        outerWireHash = outerWire.HashCode ? outerWire.HashCode(2147483647) : null;
      }
    } catch (e) {
      // OuterWire not available for this face
    }
  }

  try {
    // Use TopExp_Explorer to iterate over wires in the face
    const wireExplorer = new oc.TopExp_Explorer_2(
      face,
      oc.TopAbs_ShapeEnum.TopAbs_WIRE,
      oc.TopAbs_ShapeEnum.TopAbs_SHAPE
    );

    let wireIndex = 0;
    while (wireExplorer.More()) {
      const wireShape = wireExplorer.Current();
      const wire = oc.TopoDS.Wire_1(wireShape);

      // Track wire hash for outer wire identification
      const wireHash = wire.HashCode ? wire.HashCode(2147483647) : wireIndex;
      wireHashes.push(wireHash);

      const edges: EdgeInfo[] = [];

      // Use BRepTools_WireExplorer to iterate edges in order
      let edgeExplorer;
      if (oc.BRepTools_WireExplorer_2) {
        edgeExplorer = new oc.BRepTools_WireExplorer_2(wire);
      } else if (oc.BRepTools_WireExplorer_3) {
        edgeExplorer = new oc.BRepTools_WireExplorer_3(wire, face);
      }

      if (!edgeExplorer) {
        logOCC('No WireExplorer available');
        wireExplorer.Next();
        wireIndex++;
        continue;
      }

      // Count total edges for debugging and collect their hash codes
      let totalEdgesInWire = 0;
      const edgeHashes: number[] = [];
      const tempExplorer = new oc.TopExp_Explorer_2(wire, oc.TopAbs_ShapeEnum.TopAbs_EDGE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE);
      while (tempExplorer.More()) {
        const tempEdge = tempExplorer.Current();
        const hash = tempEdge.HashCode ? tempEdge.HashCode(2147483647) : totalEdgesInWire;
        edgeHashes.push(hash);
        totalEdgesInWire++;
        tempExplorer.Next();
      }
      faceExtractionLog(`[extractFaceEdges] Face ${faceIndex} Wire ${wireIndex}: TopExp_Explorer found ${totalEdgesInWire} edges, hashes: [${edgeHashes.join(', ')}]`);

      // Also count edges from WireExplorer for comparison
      let wireExplorerEdgeCount = 0;
      const wireExplorerHashes: number[] = [];

      let edgeIndex = 0;
      while (edgeExplorer.More()) {
        const edgeShape = edgeExplorer.Current();
        wireExplorerEdgeCount++;

        try {
          // Use BRepAdaptor_Curve to get curve information from the edge
          let curveType = 'Unknown';
          let startPoint = { x: 0, y: 0, z: 0 };
          let endPoint = { x: 0, y: 0, z: 0 };

          // Cast to TopoDS_Edge
          const edge = oc.TopoDS.Edge_1(edgeShape);
          const edgeHash = edge.HashCode ? edge.HashCode(2147483647) : edgeIndex;
          wireExplorerHashes.push(edgeHash);

          // Check if curve adaptor is available (needed for curve type detection and sampling)
          if (!oc.BRepAdaptor_Curve_2) {
            console.warn(`[extractFaceEdges] WARNING: BRepAdaptor_Curve_2 not available - curve sampling disabled!`);
          }

          // Try to use TopExp::FirstVertex and LastVertex which respect edge orientation
          // These are the proper OCC way to get vertices with correct orientation
          let gotVerticesFromTopExp = false;

          if (oc.TopExp && oc.TopExp.FirstVertex && oc.TopExp.LastVertex) {
            try {
              // FirstVertex(edge, true) - true means CumOri (cumulative orientation)
              const firstVertex = oc.TopExp.FirstVertex(edge, true);
              const lastVertex = oc.TopExp.LastVertex(edge, true);

              if (firstVertex && lastVertex) {
                const firstPnt = oc.BRep_Tool.Pnt(firstVertex);
                const lastPnt = oc.BRep_Tool.Pnt(lastVertex);

                startPoint = { x: firstPnt.X(), y: firstPnt.Y(), z: firstPnt.Z() };
                endPoint = { x: lastPnt.X(), y: lastPnt.Y(), z: lastPnt.Z() };
                gotVerticesFromTopExp = true;
              }
            } catch (topExpErr) {
              logOCC('TopExp vertex extraction failed:', topExpErr);
            }
          }

          // Fallback to BRepAdaptor_Curve if TopExp didn't work
          if (!gotVerticesFromTopExp && oc.BRepAdaptor_Curve_2) {
            const curveAdaptor = new oc.BRepAdaptor_Curve_2(edge);

            const first = curveAdaptor.FirstParameter();
            const last = curveAdaptor.LastParameter();

            const startPnt = curveAdaptor.Value(first);
            const endPnt = curveAdaptor.Value(last);

            startPoint = { x: startPnt.X(), y: startPnt.Y(), z: startPnt.Z() };
            endPoint = { x: endPnt.X(), y: endPnt.Y(), z: endPnt.Z() };
          }

          // Get curve type and sample curved edges
          let sampledPoints: Array<{ x: number; y: number; z: number }> | undefined;

          if (oc.BRepAdaptor_Curve_2) {
            const curveAdaptor = new oc.BRepAdaptor_Curve_2(edge);

            // Get curve type using GetType()
            const curveTypeEnum = curveAdaptor.GetType();
            const typeValue = typeof curveTypeEnum === 'object' && curveTypeEnum !== null
              ? curveTypeEnum.value
              : curveTypeEnum;

            // Map GeomAbs_CurveType enum to string
            const curveTypeMap: Record<number, string> = {
              0: 'Line',
              1: 'Circle',
              2: 'Ellipse',
              3: 'Hyperbola',
              4: 'Parabola',
              5: 'BezierCurve',
              6: 'BSplineCurve',
              7: 'OffsetCurve',
              8: 'OtherCurve'
            };
            curveType = curveTypeMap[typeValue] || `Unknown(${typeValue})`;

            // Sample curved edges (non-lines) to create polyline approximation
            if (curveType !== 'Line') {
              const first = curveAdaptor.FirstParameter();
              const last = curveAdaptor.LastParameter();

              // Number of samples depends on curve type and arc length
              // For circles, use more samples for full circles
              const paramRange = last - first;

              // Log arc parameters for debugging (only for first few faces)
              if (faceIndex < 3 && curveType === 'Circle') {
                const angleDegrees = (paramRange * 180 / Math.PI).toFixed(1);
                faceExtractionLog(`[extractFaceEdges] Face ${faceIndex} Wire ${wireIndex} Edge ${edgeIndex}: Circle arc ${angleDegrees}° (params: ${first.toFixed(3)} to ${last.toFixed(3)})`);
              }
              // Keep extraction sampling moderate by default so planar/trim loops
              // do not explode in vertex count. Values remain runtime-tunable.
              const MIN_SAMPLES = Math.max(4, Math.floor(readGlobalNumber('__EDGE_MIN_SAMPLES__') ?? 6));
              const MAX_SAMPLES = Math.max(MIN_SAMPLES, Math.floor(readGlobalNumber('__EDGE_MAX_SAMPLES__') ?? 64));
              let numSamples = Math.max(MIN_SAMPLES, Math.floor(readGlobalNumber('__EDGE_DEFAULT_SAMPLES__') ?? 16));

              if (curveType === 'Circle') {
                // Keep planar circular boundaries denser so cylinder caps don't become
                // visibly polygonal after planar tessellation.
                const defaultAngleStep = face.surfaceType === 'Plane' ? (Math.PI / 36) : (Math.PI / 12);
                const angleStepKey = face.surfaceType === 'Plane'
                  ? '__EDGE_CIRCLE_ANGLE_STEP_PLANE__'
                  : '__EDGE_CIRCLE_ANGLE_STEP__';
                const angleStep = readGlobalNumber(angleStepKey) ?? defaultAngleStep;
                numSamples = Math.ceil(Math.abs(paramRange) / angleStep);
              } else if (curveType === 'Ellipse') {
                numSamples = Math.max(MIN_SAMPLES, Math.floor(readGlobalNumber('__EDGE_ELLIPSE_SAMPLES__') ?? 20));
              } else if (curveType === 'BSplineCurve') {
                numSamples = Math.max(MIN_SAMPLES, Math.floor(readGlobalNumber('__EDGE_BSPLINE_SAMPLES__') ?? 16));
              }

              if (!isFinite(numSamples)) numSamples = Math.max(MIN_SAMPLES, Math.floor(readGlobalNumber('__EDGE_DEFAULT_SAMPLES__') ?? 16));
              numSamples = Math.max(MIN_SAMPLES, Math.min(MAX_SAMPLES, numSamples));

              sampledPoints = [];
              for (let i = 0; i <= numSamples; i++) {
                const t = first + (i / numSamples) * (last - first);
                const pnt = curveAdaptor.Value(t);
                sampledPoints.push({
                  x: pnt.X(),
                  y: pnt.Y(),
                  z: pnt.Z()
                });
              }

              // Check if edge orientation requires reversing the sampled points
              // Compare first sampled point with startPoint (which respects edge orientation)
              if (sampledPoints.length > 0 && gotVerticesFromTopExp) {
                const sampledFirst = sampledPoints[0];
                const distToStart = Math.sqrt(
                  Math.pow(sampledFirst.x - startPoint.x, 2) +
                  Math.pow(sampledFirst.y - startPoint.y, 2) +
                  Math.pow(sampledFirst.z - startPoint.z, 2)
                );
                const sampledLast = sampledPoints[sampledPoints.length - 1];
                const distToEnd = Math.sqrt(
                  Math.pow(sampledLast.x - startPoint.x, 2) +
                  Math.pow(sampledLast.y - startPoint.y, 2) +
                  Math.pow(sampledLast.z - startPoint.z, 2)
                );

                // If the last sampled point is closer to startPoint, edge is reversed
                if (distToEnd < distToStart) {
                  sampledPoints.reverse();
                  faceExtractionLog(`[extractFaceEdges] Face ${faceIndex} Wire ${wireIndex} Edge ${edgeIndex}: Reversed sampled points (edge orientation)`);
                }
              }

            }
          }

          // Warn if a non-Line edge doesn't have sampled points (this would cause tessellation issues)
          if (curveType !== 'Line' && (!sampledPoints || sampledPoints.length === 0)) {
            console.warn(`[extractFaceEdges] Face ${faceIndex} Wire ${wireIndex} Edge ${edgeIndex}: ${curveType} edge has NO sampled points!`);
          }

          edges.push({
            edgeIndex,
            curveType,
            startPoint,
            endPoint,
            sampledPoints
          });

        } catch (edgeErr) {
          logOCC(`Error processing edge ${edgeIndex}:`, edgeErr);
          edges.push({
            edgeIndex,
            curveType: 'Error',
            startPoint: { x: 0, y: 0, z: 0 },
            endPoint: { x: 0, y: 0, z: 0 }
          });
        }

        edgeExplorer.Next();
        edgeIndex++;
      }

      // Log comparison of TopExp vs WireExplorer edge counts
      if (totalEdgesInWire !== wireExplorerEdgeCount) {
        console.warn(`[extractFaceEdges] Face ${faceIndex} Wire ${wireIndex}: MISMATCH! TopExp found ${totalEdgesInWire}, WireExplorer found ${wireExplorerEdgeCount}`);
        console.warn(`  TopExp hashes: [${edgeHashes.join(', ')}]`);
        console.warn(`  WireExplorer hashes: [${wireExplorerHashes.join(', ')}]`);
      }

      // Log edge vertex positions for inner wires (holes) for debugging
      if (wireIndex > 0 && edges.length < 3) {
        console.warn(`[extractFaceEdges] Face ${faceIndex} Wire ${wireIndex}: Only ${edges.length} edges - hole may be degenerate`);
        edges.forEach((e, i) => {
          console.warn(`  Edge ${i}: (${e.startPoint.x.toFixed(2)}, ${e.startPoint.y.toFixed(2)}) -> (${e.endPoint.x.toFixed(2)}, ${e.endPoint.y.toFixed(2)})`);
        });
      }

      wires.push(edges);

      wireExplorer.Next();
      wireIndex++;
    }

  } catch (e) {
    console.error(`[OCC] Error extracting edges for face ${faceIndex}:`, e);
  }

  faceExtractionLog(`[extractFaceEdges] Face ${faceIndex}: Found ${wires.length} wire(s)`);
  for (let i = 0; i < wires.length; i++) {
    const w = wires[i];
    faceExtractionLog(`  Wire ${i}: ${w.length} edges, types: [${w.map(e => e.curveType).join(', ')}]`);
  }

  if (wires.length === 0) {
    return { outerLoop: [], innerLoops: [] };
  }

  const wireApproxLength = (edges: EdgeInfo[]): number => {
    let length = 0;
    for (const edge of edges) {
      const pts = edge.sampledPoints && edge.sampledPoints.length >= 2
        ? edge.sampledPoints
        : [edge.startPoint, edge.endPoint];
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i];
        const b = pts[i + 1];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dz = b.z - a.z;
        length += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
    }
    return length;
  };

  // Try to use BRepTools.OuterWire hash for reliable outer wire identification
  let outerIdx = 0;
  let usedOuterWire = false;

  if (outerWireHash !== null) {
    // Find which wire index matches the outer wire by comparing hash codes
    for (let i = 0; i < wireHashes.length; i++) {
      if (wireHashes[i] === outerWireHash) {
        outerIdx = i;
        usedOuterWire = true;
        faceExtractionLog(`[extractFaceEdges] Face ${faceIndex}: Using OuterWire (hash ${outerWireHash}) -> Wire ${outerIdx}`);
        break;
      }
    }
  }

  // Fallback: pick the outer loop as the longest wire
  if (!usedOuterWire) {
    let bestLen = -Infinity;
    const wireLengths: number[] = [];
    for (let i = 0; i < wires.length; i++) {
      const len = wireApproxLength(wires[i]);
      wireLengths.push(len);
      if (len > bestLen) {
        bestLen = len;
        outerIdx = i;
      }
    }

    // Log wire lengths to debug outer/inner selection
    if (wires.length > 1) {
      faceExtractionLog(`[extractFaceEdges] Face ${faceIndex}: Wire lengths: [${wireLengths.map(l => l.toFixed(2)).join(', ')}], selected outer: Wire ${outerIdx} (by length heuristic)`);
    }
  }

  const outerLoop = wires[outerIdx];
  const innerLoops = wires.filter((_, i) => i !== outerIdx);

  return { outerLoop, innerLoops };
}

/**
 * Get color for a face from the colorTool, including inherited colors from parent shapes.
 * In STEP files, colors are often assigned to solids/shells, not individual faces.
 * This function checks the face directly, then traverses up the label hierarchy.
 */
function getFaceColor(
  oc: any,
  face: any,
  colorTool: any,
  shapeTool: any,
  shapeColorCache?: Map<string, RGBColor>
): RGBColor | undefined {
  if (!colorTool || !shapeTool) {
    return undefined;
  }

  // Helper to try getting color from a shape
  const tryGetShapeColor = (shape: any): RGBColor | undefined => {
    try {
      const color = new oc.Quantity_Color_1();
      let hasColor = false;

      if (colorTool.GetColor_1) {
        // Try XCAFDoc_ColorSurf first (surface color)
        hasColor = colorTool.GetColor_1(shape, 1, color);
        if (!hasColor) {
          // Try XCAFDoc_ColorGen (general color)
          hasColor = colorTool.GetColor_1(shape, 0, color);
        }
      } else if (colorTool.GetColor) {
        hasColor = colorTool.GetColor(shape, 1, color);
        if (!hasColor) {
          hasColor = colorTool.GetColor(shape, 0, color);
        }
      }

      if (hasColor) {
        return { r: color.Red(), g: color.Green(), b: color.Blue() };
      }
    } catch (e) {
      // Silent fail, try next approach
    }
    return undefined;
  };

  // Helper to try getting color from a label
  const tryGetLabelColor = (label: any): RGBColor | undefined => {
    try {
      const color = new oc.Quantity_Color_1();
      let hasColor = false;

      // Try GetColor_3 (label overload) if available
      if (colorTool.GetColor_3) {
        hasColor = colorTool.GetColor_3(label, 1, color);
        if (!hasColor) {
          hasColor = colorTool.GetColor_3(label, 0, color);
        }
      }

      if (hasColor) {
        return { r: color.Red(), g: color.Green(), b: color.Blue() };
      }
    } catch (e) {
      // Silent fail
    }
    return undefined;
  };

  try {
    // Step 1: Try to get color directly from the face shape
    let result = tryGetShapeColor(face);
    if (result) {
      return result;
    }

    // Step 2: Try to find the face's label and check for color
    if (shapeTool.FindShape) {
      const faceLabel = new oc.TDF_Label();
      // FindShape signature: (shape, out_label, findInstance) - 3 args
      if (shapeTool.FindShape(face, faceLabel, false) && !faceLabel.IsNull()) {
        result = tryGetLabelColor(faceLabel);
        if (result) {
          return result;
        }

        // Step 3: Walk up the label hierarchy to find parent with color
        let parentLabel = faceLabel.Father();
        let depth = 0;
        while (parentLabel && !parentLabel.IsNull() && depth < 10) {
          result = tryGetLabelColor(parentLabel);
          if (result) {
            return result;
          }

          // Also try to get shape from parent label and check its color
          if (shapeTool.GetShape) {
            try {
              const parentShape = shapeTool.GetShape(parentLabel);
              if (parentShape && !parentShape.IsNull()) {
                result = tryGetShapeColor(parentShape);
                if (result) {
                  return result;
                }
              }
            } catch (e) {
              // Parent doesn't have a shape, continue up
            }
          }

          parentLabel = parentLabel.Father();
          depth++;
        }
      }
    }

    // Step 4: If we have a color cache from the initial shape traversal, use it
    if (shapeColorCache && shapeColorCache.size > 0) {
      // Return first cached color as fallback
      const firstColor = shapeColorCache.values().next().value;
      if (firstColor) {
        return firstColor;
      }
    }

  } catch (e) {
    // Color extraction failed silently
  }

  return undefined;
}

/**
 * Extract surfaces and boundary edges from all faces
 */
async function extractFacesWithEdges(
  shape: any,
  colorTool?: any,
  shapeTool?: any,
  stepColors?: Map<number, RGBColor>,
  shapeColorMap?: Map<number, RGBColor>,
  faceIdOrder?: number[],
  geometryColorMap?: Map<string, RGBColor>,
  solidMatchedColors?: Map<number, RGBColor>,
  faceToSolid?: Map<number, number>,
  solidToColor?: Map<number, RGBColor>
): Promise<FaceWithEdgesInfo[]> {
  const oc = await initOC();
  const faces: FaceWithEdgesInfo[] = [];

  logOCC('extractFacesWithEdges: colorTool available:', !!colorTool);
  logOCC('extractFacesWithEdges: stepColors available:', !!stepColors, stepColors?.size || 0);
  logOCC('extractFacesWithEdges: shapeColorMap available:', !!shapeColorMap, shapeColorMap?.size || 0);
  logOCC('extractFacesWithEdges: faceIdOrder available:', !!faceIdOrder, faceIdOrder?.length || 0);
  logOCC('extractFacesWithEdges: geometryColorMap available:', !!geometryColorMap, geometryColorMap?.size || 0);

  // Track color source statistics
  let colorsFromShapeHashMap = 0;
  let colorsFromXCAF = 0;
  let colorsFromFaceIdOrder = 0;
  let colorsFromGeometry = 0;
  let colorsFromFallback = 0;
  let facesWithNoColor = 0;

  // Helper to create geometry key from OCC face centroid
  // Using centroid of outer wire vertices is more unique than first vertex
  const makeGeometryKey = (face: any): string | null => {
    try {
      // Get outer wire of the face
      const outerWire = oc.BRepTools.OuterWire(face);
      if (!outerWire || outerWire.IsNull()) {
        return null;
      }

      // Collect all edge start vertices to compute centroid
      const vertices: Array<{ x: number; y: number; z: number }> = [];
      const wireExplorer = new oc.BRepTools_WireExplorer_2(outerWire, face);

      while (wireExplorer.More()) {
        const edge = wireExplorer.Current();
        const firstVertex = oc.TopExp.FirstVertex(edge, true);
        const pnt = oc.BRep_Tool.Pnt(firstVertex);
        vertices.push({ x: pnt.X(), y: pnt.Y(), z: pnt.Z() });
        wireExplorer.Next();
      }

      if (vertices.length < 3) {
        return null;
      }

      // Compute centroid
      let cx = 0, cy = 0, cz = 0;
      for (const v of vertices) {
        cx += v.x;
        cy += v.y;
        cz += v.z;
      }
      cx /= vertices.length;
      cy /= vertices.length;
      cz /= vertices.length;

      // Round to 1 decimal place (same as STEP parsing side)
      const x = Math.round(cx * 10) / 10;
      const y = Math.round(cy * 10) / 10;
      const z = Math.round(cz * 10) / 10;
      return `${x},${y},${z}`;
    } catch (e) {
      // Fallback to vertex-based approach
      try {
        const vertices: Array<{ x: number; y: number; z: number }> = [];
        const edgeExplorer = new oc.TopExp_Explorer_2(
          face,
          oc.TopAbs_ShapeEnum.TopAbs_VERTEX,
          oc.TopAbs_ShapeEnum.TopAbs_SHAPE
        );

        while (edgeExplorer.More()) {
          const vertex = oc.TopoDS.Vertex_1(edgeExplorer.Current());
          const pnt = oc.BRep_Tool.Pnt(vertex);
          vertices.push({ x: pnt.X(), y: pnt.Y(), z: pnt.Z() });
          edgeExplorer.Next();
        }

        if (vertices.length >= 3) {
          let cx = 0, cy = 0, cz = 0;
          for (const v of vertices) {
            cx += v.x;
            cy += v.y;
            cz += v.z;
          }
          cx /= vertices.length;
          cy /= vertices.length;
          cz /= vertices.length;

          const x = Math.round(cx * 10) / 10;
          const y = Math.round(cy * 10) / 10;
          const z = Math.round(cz * 10) / 10;
          return `${x},${y},${z}`;
        }
      } catch (e2) {
        // Failed to get vertices
      }
    }
    return null;
  };

  const explorer = new oc.TopExp_Explorer_2(
    shape,
    oc.TopAbs_ShapeEnum.TopAbs_FACE,
    oc.TopAbs_ShapeEnum.TopAbs_SHAPE
  );

  let faceIndex = 0;
  while (explorer.More()) {
    const currentShape = explorer.Current();

    try {
      const face = oc.TopoDS.Face_1(currentShape);

      // Get surface info (reuse logic from extractSurfaces)
      let surface = null;
      if (oc.BRep_Tool && typeof oc.BRep_Tool.Surface_2 === 'function') {
        surface = oc.BRep_Tool.Surface_2(face);
      }

      const surfaceType = surface ? getSurfaceTypeName(oc, surface) : 'Unknown';

      // Get UV bounds and surface parameters
      let uMin = 0, uMax = 0, vMin = 0, vMax = 0;
      let surfaceParams: SurfaceParams | undefined;

      if (oc.BRepAdaptor_Surface_2) {
        const faceAdaptor = new oc.BRepAdaptor_Surface_2(face, true);
        uMin = faceAdaptor.FirstUParameter();
        uMax = faceAdaptor.LastUParameter();
        vMin = faceAdaptor.FirstVParameter();
        vMax = faceAdaptor.LastVParameter();

        // Extract surface-specific parameters
        if (surfaceType === 'Cylinder') {
          try {
            const cylinder = faceAdaptor.Cylinder();
            const axis = cylinder.Axis();
            const location = axis.Location();
            const direction = axis.Direction();

            // Get X direction from cylinder's coordinate system
            const xDir = cylinder.XAxis().Direction();

            surfaceParams = {
              placement: {
                location: [location.X(), location.Y(), location.Z()],
                axis: [direction.X(), direction.Y(), direction.Z()],
                refDirection: [xDir.X(), xDir.Y(), xDir.Z()]
              },
              radius: cylinder.Radius()
            };

            logOCC(`Cylinder params: radius=${surfaceParams.radius}`);
          } catch (e) {
            logOCC('Failed to extract cylinder params:', e);
          }
        } else if (surfaceType === 'Sphere') {
          try {
            const sphere = faceAdaptor.Sphere();
            const location = sphere.Location();
            const axis = sphere.Position().Axis();
            const direction = axis.Direction();
            const xDir = sphere.XAxis().Direction();

            surfaceParams = {
              placement: {
                location: [location.X(), location.Y(), location.Z()],
                axis: [direction.X(), direction.Y(), direction.Z()],
                refDirection: [xDir.X(), xDir.Y(), xDir.Z()]
              },
              radius: sphere.Radius()
            };

            logOCC(`Sphere params: radius=${surfaceParams.radius}`);
          } catch (e) {
            logOCC('Failed to extract sphere params:', e);
          }
        } else if (surfaceType === 'Cone') {
          try {
            const cone = faceAdaptor.Cone();
            const axis = cone.Axis();
            const location = axis.Location();
            const direction = axis.Direction();
            const xDir = cone.XAxis().Direction();

            surfaceParams = {
              placement: {
                location: [location.X(), location.Y(), location.Z()],
                axis: [direction.X(), direction.Y(), direction.Z()],
                refDirection: [xDir.X(), xDir.Y(), xDir.Z()]
              },
              radius: cone.RefRadius(),
              semiAngle: cone.SemiAngle()
            };

            logOCC(`Cone params: radius=${surfaceParams.radius}, semiAngle=${surfaceParams.semiAngle}`);
          } catch (e) {
            logOCC('Failed to extract cone params:', e);
          }
        } else if (surfaceType === 'Torus') {
          try {
            const torus = faceAdaptor.Torus();
            const axis = torus.Axis();
            const location = axis.Location();
            const direction = axis.Direction();
            const xDir = torus.XAxis().Direction();

            surfaceParams = {
              placement: {
                location: [location.X(), location.Y(), location.Z()],
                axis: [direction.X(), direction.Y(), direction.Z()],
                refDirection: [xDir.X(), xDir.Y(), xDir.Z()]
              },
              majorRadius: torus.MajorRadius(),
              minorRadius: torus.MinorRadius()
            };

            logOCC(`Torus params: majorRadius=${surfaceParams.majorRadius}, minorRadius=${surfaceParams.minorRadius}`);
          } catch (e) {
            logOCC('Failed to extract torus params:', e);
          }
        } else if (surfaceType === 'BSplineSurface') {
          try {
            logOCC('Attempting to extract B-spline surface...');

            // Try multiple approaches to get the B-spline surface
            let bspline = null;

            // Approach 1: BSpline() method on faceAdaptor
            if (faceAdaptor.BSpline) {
              const bsplineSurf = faceAdaptor.BSpline();
              if (bsplineSurf) {
                // Check if we need to call .get() or can use directly
                if (typeof bsplineSurf.get === 'function') {
                  bspline = bsplineSurf.get();
                } else if (bsplineSurf.UDegree) {
                  // It might already be unwrapped
                  bspline = bsplineSurf;
                }
              }
            }

            // Approach 2: Get surface from BRep_Tool and cast
            if (!bspline && surface) {
              const actualSurface = typeof surface.get === 'function' ? surface.get() : surface;
              if (actualSurface && typeof actualSurface.UDegree === 'function') {
                bspline = actualSurface;
              }
            }

            // Approach 3: Try DownCast if available
            if (!bspline && surface && oc.Geom_BSplineSurface) {
              try {
                if (oc.Geom_BSplineSurface.DownCast) {
                  const downcast = oc.Geom_BSplineSurface.DownCast(surface);
                  if (downcast && !downcast.IsNull()) {
                    bspline = downcast.get ? downcast.get() : downcast;
                  }
                }
              } catch (downcastErr) {
                logOCC('DownCast failed:', downcastErr);
              }
            }

            if (bspline) {
              // Get degrees
              const uDegree = bspline.UDegree();
              const vDegree = bspline.VDegree();

              // Get number of control points
              const numUPoles = bspline.NbUPoles();
              const numVPoles = bspline.NbVPoles();

              // Extract control points (1-indexed in OCC)
              const controlPoints: Vec3[][] = [];
              for (let v = 1; v <= numVPoles; v++) {
                const row: Vec3[] = [];
                for (let u = 1; u <= numUPoles; u++) {
                  const pole = bspline.Pole(u, v);
                  row.push([pole.X(), pole.Y(), pole.Z()]);
                }
                controlPoints.push(row);
              }
              // Extract knots with multiplicities (OCC provides them separately)
              const numUKnots = bspline.NbUKnots();
              const numVKnots = bspline.NbVKnots();

              const uKnots: number[] = [];
              for (let i = 1; i <= numUKnots; i++) {
                const knot = bspline.UKnot(i);
                const mult = bspline.UMultiplicity(i);
                for (let m = 0; m < mult; m++) {
                  uKnots.push(knot);
                }
              }

              const vKnots: number[] = [];
              for (let i = 1; i <= numVKnots; i++) {
                const knot = bspline.VKnot(i);
                const mult = bspline.VMultiplicity(i);
                for (let m = 0; m < mult; m++) {
                  vKnots.push(knot);
                }
              }

              // Check if it's a rational B-spline (NURBS)
              let weights: number[][] | undefined;
              if (bspline.IsURational && bspline.IsVRational) {
                if (bspline.IsURational() || bspline.IsVRational()) {
                  weights = [];
                  for (let v = 1; v <= numVPoles; v++) {
                    const row: number[] = [];
                    for (let u = 1; u <= numUPoles; u++) {
                      row.push(bspline.Weight(u, v));
                    }
                    weights.push(row);
                  }
                }
              }

              surfaceParams = {
                bspline: {
                  controlPoints,
                  uDegree,
                  vDegree,
                  uKnots,
                  vKnots,
                  weights
                }
              };

              logOCC(`BSpline params: degree=(${uDegree},${vDegree}), poles=(${numUPoles}x${numVPoles}), knots=(${uKnots.length},${vKnots.length})`);
            } else {
              logOCC('Could not get B-spline object from either approach');
            }
          } catch (e) {
            logOCC('Failed to extract B-spline params:', e);
          }
        }
      }

      // Extract boundary edges
      const { outerLoop, innerLoops } = await extractFaceEdges(oc, face, faceIndex);

      // Extract face orientation (FORWARD or REVERSED)
      // REVERSED means the face normal should be flipped relative to the surface normal
      let isReversed = false;
      try {
        const orientation = face.Orientation_1();
        // Compare by value property (opencascade.js enum pattern)
        if (oc.TopAbs_Orientation && oc.TopAbs_Orientation.TopAbs_REVERSED) {
          isReversed = orientation.value === oc.TopAbs_Orientation.TopAbs_REVERSED.value;
        }
        if (faceIndex < 5) {
          faceExtractionLog(`[extractFacesWithEdges] Face ${faceIndex}: orientation=${isReversed ? 'REVERSED' : 'FORWARD'} (raw: ${orientation.value})`);
        }
      } catch (e) {
        // Orientation not available, assume FORWARD
        logOCC(`Face ${faceIndex}: Could not get orientation:`, e);
      }

      // Debug logging for edge extraction
      faceExtractionLog(`[extractFacesWithEdges] Face ${faceIndex}: surfaceType=${surfaceType}, outerLoop=${outerLoop.length} edges, innerLoops=${innerLoops.length}, reversed=${isReversed}`);
      if (outerLoop.length === 0) {
        console.warn(`[extractFacesWithEdges] Face ${faceIndex} has 0 edges in outerLoop!`);
      } else if (outerLoop.length > 0 && outerLoop.length < 5) {
        faceExtractionLog(`[extractFacesWithEdges] Face ${faceIndex} outerLoop edges:`, outerLoop.map(e => ({
          curveType: e.curveType,
          start: [e.startPoint.x.toFixed(2), e.startPoint.y.toFixed(2), e.startPoint.z.toFixed(2)],
          end: [e.endPoint.x.toFixed(2), e.endPoint.y.toFixed(2), e.endPoint.z.toFixed(2)]
        })));
      }

      // Extract face color using multiple strategies:
      // 1. Try shapeColorMap first (face HashCode lookup - most reliable)
      // 2. Try XCAF direct lookup
      // 3. Use geometry-based matching
      // 4. Use faceIdOrder to correlate OCC face index to STEP entity ID
      // 5. Fall back to first available color
      let color: RGBColor | undefined = undefined;

      // Get face hash code once for reuse
      let faceHashCode = 0;
      try {
        faceHashCode = face.HashCode(2147483647);
      } catch (e) {
        // HashCode not available
      }

      // Strategy 0 (NEW): Look up face by HashCode in solidMatchedColors
      // This uses solid-level matching by face count
      if (!color && solidMatchedColors && solidMatchedColors.size > 0 && faceHashCode) {
        if (solidMatchedColors.has(faceHashCode)) {
          color = solidMatchedColors.get(faceHashCode);
          colorsFromShapeHashMap++; // Count this as hash-based
          if (faceIndex < 5) {
            console.log(`[FaceColor] Face ${faceIndex} -> SolidMatch (hash ${faceHashCode}) -> RGB(${color!.r.toFixed(2)}, ${color!.g.toFixed(2)}, ${color!.b.toFixed(2)})`);
          }
        }
      }

      // Strategy 1: Look up face by HashCode in shapeColorMap (XCAF-based)
      if (!color && shapeColorMap && shapeColorMap.size > 0 && faceHashCode) {
        if (shapeColorMap.has(faceHashCode)) {
          color = shapeColorMap.get(faceHashCode);
          colorsFromShapeHashMap++;
        }
      }

      // Strategy 2: Try XCAF direct lookup and parent traversal
      if (!color) {
        // DIAGNOSTIC: Check if IsSet returns true for this face
        if (faceIndex < 5 && colorTool) {
          try {
            // Try all IsSet variants and color types
            for (const isSetMethod of ['IsSet_1', 'IsSet_2']) {
              if (typeof colorTool[isSetMethod] === 'function') {
                for (const colorType of [0, 1, 2]) { // Gen, Surf, Curv
                  try {
                    const isSet = colorTool[isSetMethod](face, colorType);
                    if (isSet) {
                      console.log(`[XCAF_DIAG] Face ${faceIndex}: ${isSetMethod}(face, ${colorType}) = TRUE!`);
                    }
                  } catch (e) {
                    // This variant doesn't match signature
                  }
                }
              }
            }
          } catch (e) {
            console.log(`[XCAF_DIAG] Face ${faceIndex}: IsSet check failed:`, e);
          }
        }

        color = getFaceColor(oc, face, colorTool, shapeTool);
        if (color) {
          colorsFromXCAF++;
        }
      }

      // Strategy 3: Use geometry-based matching with fuzzy nearest-neighbor search
      // First try exact match, then fall back to nearest neighbor within tolerance
      if (!color && geometryColorMap && geometryColorMap.size > 0) {
        const geoKey = makeGeometryKey(face);
        if (geoKey) {
          // Try exact match first
          if (geometryColorMap.has(geoKey)) {
            color = geometryColorMap.get(geoKey);
            colorsFromGeometry++;
            if (faceIndex < 10) {
              console.log(`[FaceColor] Face ${faceIndex} -> Geometry EXACT ${geoKey} -> RGB(${color!.r.toFixed(2)}, ${color!.g.toFixed(2)}, ${color!.b.toFixed(2)})`);
            }
          } else {
            // Fuzzy nearest-neighbor search with tolerance
            // Model coordinates range up to 600+ units, so use larger tolerance
            const GEOMETRY_TOLERANCE = 50.0; // Max distance to consider a match
            const [gx, gy, gz] = geoKey.split(',').map(parseFloat);
            let closestKey: string | null = null;
            let closestDist = Infinity;
            let closestColor: RGBColor | undefined;

            for (const [mapKey, mapColor] of geometryColorMap.entries()) {
              const [mx, my, mz] = mapKey.split(',').map(parseFloat);
              const dist = Math.sqrt((gx-mx)**2 + (gy-my)**2 + (gz-mz)**2);
              if (dist < closestDist) {
                closestDist = dist;
                closestKey = mapKey;
                closestColor = mapColor;
              }
            }

            if (closestDist <= GEOMETRY_TOLERANCE && closestColor) {
              color = closestColor;
              colorsFromGeometry++;
              if (faceIndex < 10) {
                console.log(`[FaceColor] Face ${faceIndex} -> Geometry FUZZY ${geoKey} ~ ${closestKey} (dist=${closestDist.toFixed(1)}) -> RGB(${color!.r.toFixed(2)}, ${color!.g.toFixed(2)}, ${color!.b.toFixed(2)})`);
              }
            } else if (faceIndex < 20) {
              console.log(`[FaceColor] Face ${faceIndex} -> Geometry ${geoKey} -> NO MATCH (closest: ${closestKey}, dist=${closestDist.toFixed(1)} > ${GEOMETRY_TOLERANCE})`);
            }
          }
        }
      }

      // Strategy 3.5: Solid membership propagation
      // If a face's solid was matched (but this specific face wasn't in solidMatchedColors),
      // inherit the solid's color. This handles faces in matched solids that weren't directly matched.
      if (!color && faceToSolid && solidToColor && faceHashCode) {
        const solidIdx = faceToSolid.get(faceHashCode);
        if (solidIdx !== undefined && solidToColor.has(solidIdx)) {
          color = solidToColor.get(solidIdx);
          colorsFromShapeHashMap++; // Count as solid-based
          if (faceIndex < 20 && faceIndex >= 10) {
            console.log(`[FaceColor] Face ${faceIndex} -> SolidProp (solid ${solidIdx}) -> RGB(${color!.r.toFixed(2)}, ${color!.g.toFixed(2)}, ${color!.b.toFixed(2)})`);
          }
        }
      }

      // Strategy 4: Use faceIdOrder to look up color in stepColors (fallback if geometry fails)
      // This correlates OCC face index to STEP entity ID
      // NOTE: This is unreliable because OCC face order may differ from STEP document order
      if (!color && stepColors && stepColors.size > 0 && faceIdOrder && faceIdOrder.length > faceIndex) {
        const stepEntityId = faceIdOrder[faceIndex];
        if (stepColors.has(stepEntityId)) {
          color = stepColors.get(stepEntityId);
          colorsFromFaceIdOrder++;
          // Log first 10 faces to verify correlation
          if (faceIndex < 10) {
            console.log(`[FaceColor] Face ${faceIndex} -> STEP #${stepEntityId} -> RGB(${color!.r.toFixed(2)}, ${color!.g.toFixed(2)}, ${color!.b.toFixed(2)})`);
          }
        }
      }

      // Strategy 5: If shapeColorMap exists but HashCode lookup missed, use first color
      if (!color && shapeColorMap && shapeColorMap.size > 0) {
        const firstColor = shapeColorMap.values().next().value;
        if (firstColor) {
          color = firstColor;
          colorsFromFallback++;
        }
      }

      // Final fallback: use first color from stepColors
      if (!color && stepColors && stepColors.size > 0) {
        const firstColor = stepColors.values().next().value;
        if (firstColor) {
          color = firstColor;
          colorsFromFallback++;
        }
      }

      if (!color) {
        facesWithNoColor++;
      }

      faces.push({
        faceIndex,
        surfaceType,
        uvBounds: { uMin, uMax, vMin, vMax },
        outerLoop,
        innerLoops,
        occFace: face,
        occSurface: surface,
        surfaceParams,
        color,
        isReversed
      });

      // Warn if we have a B-spline surface but couldn't extract params
      if (surfaceType === 'BSplineSurface' && (!surfaceParams || !surfaceParams.bspline)) {
        logOCC(`Face ${faceIndex}: B-spline surface detected but params extraction FAILED`);
      }

    } catch (e) {
      logOCC(`Error processing face ${faceIndex}:`, e);
    }

    explorer.Next();
    faceIndex++;
  }

  // Log color source statistics
  console.log(`[FaceColors] Color sources: shapeHashMap=${colorsFromShapeHashMap}, xcaf=${colorsFromXCAF}, geometry=${colorsFromGeometry}, faceIdOrder=${colorsFromFaceIdOrder}, fallback=${colorsFromFallback}, noColor=${facesWithNoColor}`);

  return faces;
}

/**
 * Run Checkpoint 3 test - Extract boundary curves/edges
 */
async function runCheckpoint3(stepFileContent: string): Promise<{
  success: boolean;
  faces: FaceWithEdgesInfo[];
  error?: string;
}> {
  try {
    console.log('[Checkpoint 3] Loading STEP file...');
    const { shape } = await loadStepFile(stepFileContent, 'test.step');

    console.log('[Checkpoint 3] Extracting faces with edges...');
    const faces = await extractFacesWithEdges(shape);

    console.log(`[Checkpoint 3] Extracted ${faces.length} faces`);

    // For simple-cube.step:
    // - 6 faces
    // - Each face should have 4 edges (rectangle)
    // - All edges should be lines
    // - No inner loops (holes)
    const hasSixFaces = faces.length === 6;
    const allHaveFourEdges = faces.every(f => f.outerLoop.length === 4);
    const allEdgesAreLines = faces.every(f =>
      f.outerLoop.every(e => e.curveType === 'Line')
    );
    const noHoles = faces.every(f => f.innerLoops.length === 0);

    const success = hasSixFaces && allHaveFourEdges && allEdgesAreLines && noHoles;

    if (success) {
      console.log('[Checkpoint 3] ✓ PASSED: All 6 faces have 4 line edges each, no holes');
    } else {
      if (!hasSixFaces) {
        console.log(`[Checkpoint 3] ✗ FAILED: Expected 6 faces, got ${faces.length}`);
      }
      if (!allHaveFourEdges) {
        faces.forEach((f, i) => {
          if (f.outerLoop.length !== 4) {
            console.log(`[Checkpoint 3] ✗ FAILED: Face ${i} has ${f.outerLoop.length} edges, expected 4`);
          }
        });
      }
      if (!allEdgesAreLines) {
        faces.forEach((f, i) => {
          const nonLines = f.outerLoop.filter(e => e.curveType !== 'Line');
          if (nonLines.length > 0) {
            console.log(`[Checkpoint 3] ✗ FAILED: Face ${i} has non-line edges:`, nonLines.map(e => e.curveType));
          }
        });
      }
      if (!noHoles) {
        faces.forEach((f, i) => {
          if (f.innerLoops.length > 0) {
            console.log(`[Checkpoint 3] ✗ FAILED: Face ${i} has ${f.innerLoops.length} holes`);
          }
        });
      }
    }

    return { success, faces };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('[Checkpoint 3] ✗ ERROR:', errorMsg);
    return { success: false, faces: [], error: errorMsg };
  }
}

// =============================================================================
// Checkpoint 4: Connect to GPU tessellator and render
// =============================================================================

/**
 * Convert OCC face edge data to Vec3 array (outer boundary polygon)
 *
 * BRepTools_WireExplorer iterates edges in wire order, and if we're using
 * TopExp::FirstVertex/LastVertex with CumOri=true, the edges should already
 * have the correct start/end points for the wire traversal direction.
 *
 * For line edges: just use the start point
 * For curved edges: use the sampled points (excluding the last one to avoid duplicates)
 */
function occEdgesToPolygon(
  edges: EdgeInfo[],
  debugLabel?: string,
  options?: { lineSubdivideStep?: number }
): Vec3[] {
  if (edges.length === 0) return [];
  const lineSubdivideStep = options?.lineSubdivideStep ?? 0;

  const polygon: Vec3[] = [];

  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    const hasSamples = edge.sampledPoints && edge.sampledPoints.length > 0;
    if (debugLabel) {
      tessellationVerboseLog(`[occEdgesToPolygon ${debugLabel}] edge ${i}: type=${edge.curveType}, sampledPoints=${edge.sampledPoints?.length || 0}`);
    }

    if (hasSamples) {
      // For curved edges, use sampled points (skip last point to avoid duplicate with next edge's start)
      for (let j = 0; j < edge.sampledPoints!.length - 1; j++) {
        const pt = edge.sampledPoints![j];
        polygon.push([pt.x, pt.y, pt.z]);
      }
    } else {
      // For line edges, add start point
      // We don't skip end point for LINE edges because they need proper polygon closure
      polygon.push([edge.startPoint.x, edge.startPoint.y, edge.startPoint.z]);

      // Optional line densification for UV projection on curved periodic faces.
      // Keep it off by default for planar tessellation to avoid triangle blowups.
      if (lineSubdivideStep > 1e-9) {
        const dx = edge.endPoint.x - edge.startPoint.x;
        const dy = edge.endPoint.y - edge.startPoint.y;
        const dz = edge.endPoint.z - edge.startPoint.z;
        const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (length > lineSubdivideStep) {
          const numSegments = Math.max(2, Math.ceil(length / lineSubdivideStep));
          for (let j = 1; j < numSegments; j++) {
            const t = j / numSegments;
            polygon.push([
              edge.startPoint.x + t * dx,
              edge.startPoint.y + t * dy,
              edge.startPoint.z + t * dz
            ]);
          }
        }
      }
    }
  }

  return polygon;
}

/**
 * Tessellate a single planar face from OCC data using GPU triangulation
 * Each sub-function is profiled for performance analysis
 * @param face - Face geometry info extracted from OCC
 * @param triangulationMethod - 'ear-clipping' (default) or 'cdt'
 */
async function tessellatePlanarFaceFromOCC(
  face: FaceWithEdgesInfo,
  triangulationMethod: TriangulationMethod = 'ear-clipping'
): Promise<{
  vertices: Vec3[];
  triangles: number[][];
}> {
  const faceStart = performance.now();

  // 1. Convert edge data to polygon vertices
  let t0 = performance.now();
  const hasHoles = face.innerLoops.length > 0;
  const outer: Vec3[] = occEdgesToPolygon(face.outerLoop, hasHoles ? 'outer' : undefined);
  const holes: Vec3[][] = face.innerLoops.map((loop, i) => occEdgesToPolygon(loop, `hole${i}`));
  tessellationProfile.occEdgesToPolygon.total += performance.now() - t0;
  tessellationProfile.occEdgesToPolygon.calls++;

  // Diagnostic logging for planar faces with holes
  tessellationVerboseLog(`[tessellatePlanarFace] outerLoop: ${face.outerLoop.length} edges -> ${outer.length} vertices`);
  tessellationVerboseLog(`[tessellatePlanarFace] innerLoops: ${face.innerLoops.length} loops`);
  for (let i = 0; i < face.innerLoops.length; i++) {
    const loop = face.innerLoops[i];
    const holeVerts = holes[i];
    tessellationVerboseLog(`  - hole ${i}: ${loop.length} edges -> ${holeVerts.length} vertices, types: [${loop.map(e => e.curveType).join(', ')}]`);
  }

  if (outer.length < 3) {
    console.warn(`[tessellatePlanarFace] Not enough vertices: ${outer.length} (need >= 3). outerLoop has ${face.outerLoop.length} edges`);
    if (face.outerLoop.length > 0) {
      console.warn(`[tessellatePlanarFace] First edge:`, JSON.stringify(face.outerLoop[0]));
    }
    return { vertices: [], triangles: [] };
  }

  // 2. Compute face basis from outer loop
  t0 = performance.now();
  const basis = computeFaceBasisFromLoop(outer);
  tessellationProfile.computeFaceBasisFromLoop.total += performance.now() - t0;
  tessellationProfile.computeFaceBasisFromLoop.calls++;

  // 3. Project to 2D
  t0 = performance.now();
  const projected = projectFaceLoopsTo2D({ outer, holes }, basis);
  tessellationProfile.projectFaceLoopsTo2D.total += performance.now() - t0;
  tessellationProfile.projectFaceLoopsTo2D.calls++;

  // 4. Normalize winding (CCW outer, CW holes)
  t0 = performance.now();
  const normalized = normalizeWinding(projected);
  tessellationProfile.normalizeWinding.total += performance.now() - t0;
  tessellationProfile.normalizeWinding.calls++;

  // 5. Apply same winding changes to 3D
  t0 = performance.now();
  const oriented3d = applyWindingTo3D(
    { outer, holes },
    normalized.outerReversed,
    normalized.holesReversed
  );
  tessellationProfile.applyWindingTo3D.total += performance.now() - t0;
  tessellationProfile.applyWindingTo3D.calls++;

  // 6. Triangulate based on whether we have holes
  t0 = performance.now();
  let triangles: number[][];
  let vertices3d: Vec3[];

  tessellationVerboseLog(`[tessellatePlanarFace] After projection: outer2d=${normalized.outer2d.length} verts, holes2d=${normalized.holes2d.length} holes`);
  if (normalized.holes2d.length > 0) {
    for (let i = 0; i < normalized.holes2d.length; i++) {
      tessellationVerboseLog(`  - hole2d ${i}: ${normalized.holes2d[i].length} vertices`);
    }
  }

  if (normalized.holes2d.length > 0) {
    // HAS HOLES: Use earcut with native hole support (no bridging needed)
    // triangulateWithHoles expects vertices in order: outer, then holes concatenated
    // and returns indices into that combined array
    tessellationVerboseLog(`[tessellatePlanarFace] Using earcut with holes: outer=${normalized.outer2d.length}, holes=${normalized.holes2d.map(h => h.length).join(',')}`);
    triangles = triangulateWithHoles(normalized.outer2d, normalized.holes2d);
    tessellationVerboseLog(`[tessellatePlanarFace] earcut produced ${triangles.length} triangles`);

    // Build combined 3D vertices: outer + all holes
    vertices3d = [...oriented3d.outer];
    for (const hole of oriented3d.holes) {
      vertices3d.push(...hole);
    }
  } else {
    // NO HOLES: Use hybrid triangulation (earcut for medium loops, GPU for large loops)
    tessellationVerboseLog(`[tessellatePlanarFace] Using hybrid triangulation (no holes)`);
    const points2dAsVec3: Vec3[] = normalized.outer2d.map(p => [p[0], p[1], 0]);
    try {
      triangles = await triangulateFast(points2dAsVec3);
      if (triangles.length === 0 && points2dAsVec3.length >= 3) {
        // Safety fallback for rare degenerate/predictor misses.
        triangles = await earClipping(points2dAsVec3);
      }
    } catch {
      triangles = await earClipping(points2dAsVec3);
    }
    vertices3d = oriented3d.outer;
  }

  tessellationProfile.earClipping.total += performance.now() - t0;
  tessellationProfile.earClipping.calls++;

  tessellationProfile.tessellatePlanarFace.total += performance.now() - faceStart;
  tessellationProfile.tessellatePlanarFace.calls++;

  return { vertices: vertices3d, triangles };
}

/**
 * Helper to convert TessellatedMesh to vertices/triangles format
 */
function tessellatedMeshToVerticesAndTriangles(mesh: { positions: Float32Array; indices: Uint32Array }): {
  vertices: Vec3[];
  triangles: number[][];
} {
  const vertices: Vec3[] = [];
  for (let i = 0; i < mesh.positions.length; i += 3) {
    vertices.push([mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]]);
  }

  const triangles: number[][] = [];
  for (let i = 0; i < mesh.indices.length; i += 3) {
    triangles.push([mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]]);
  }

  return { vertices, triangles };
}

function topAbsStateToValue(state: any): number | undefined {
  if (typeof state === 'number') return state;
  if (state && typeof state === 'object' && typeof state.value === 'number') {
    return state.value;
  }
  return undefined;
}

function topAbsStateEquals(lhs: any, rhs: any): boolean {
  const lhsValue = topAbsStateToValue(lhs);
  const rhsValue = topAbsStateToValue(rhs);
  if (lhsValue !== undefined && rhsValue !== undefined) {
    return lhsValue === rhsValue;
  }
  return lhs === rhs;
}

type DebugMode = 'off' | 'skip' | 'only';

function readGlobalString(key: string): string | undefined {
  const raw = (globalThis as any)?.[key];
  return typeof raw === 'string' ? raw : undefined;
}

function readGlobalNumber(key: string): number | undefined {
  const raw = (globalThis as any)?.[key];
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined;
}

function readGlobalBoolean(key: string, fallback: boolean): boolean {
  const raw = (globalThis as any)?.[key];
  return typeof raw === 'boolean' ? raw : fallback;
}

function readDebugModeFromGlobal(key: string, fallback: DebugMode): DebugMode {
  const raw = readGlobalString(key)?.trim().toLowerCase();
  if (raw === 'off' || raw === 'skip' || raw === 'only') {
    return raw;
  }
  return fallback;
}

function readFaceIdsFromGlobal(key: string, fallback: number[]): Set<number> {
  const raw = (globalThis as any)?.[key];
  if (!raw) return new Set(fallback);
  if (Array.isArray(raw)) {
    const ids = raw
      .map((v) => Number(v))
      .filter((v) => Number.isInteger(v) && v >= 0);
    return new Set(ids.length ? ids : fallback);
  }
  if (typeof raw === 'string') {
    const ids = raw
      .split(',')
      .map((s) => Number(s.trim()))
      .filter((v) => Number.isInteger(v) && v >= 0);
    return new Set(ids.length ? ids : fallback);
  }
  return new Set(fallback);
}

function filterMeshTrianglesByFaceUVClassification<T extends { positions: Float32Array; indices: Uint32Array; uvs?: Float32Array }>(
  oc: any,
  occFace: any,
  mesh: T,
  options?: {
    tol?: number;
    recadreOnPeriodic?: boolean;
    label?: string;
    logAlways?: boolean;
    sampleMode?: 'centroid' | 'multi7';
    maxOutSamples?: number;
    maxDropRatio?: number;
    faceIndex?: number;
    surfaceType?: string;
    periodicProof?: boolean;
  }
): T {
  if (!mesh.uvs || mesh.uvs.length < (mesh.positions.length / 3) * 2) {
    return mesh;
  }

  if (!occFace || !oc?.BRepTopAdaptor_FClass2d || !oc?.gp_Pnt2d_3 || !oc?.TopAbs_State) {
    return mesh;
  }

  const tol = options?.tol ?? 1e-7;
  const recadreOnPeriodic = options?.recadreOnPeriodic ?? true;
  const label = options?.label ?? 'mesh';
  const logAlways = options?.logAlways ?? false;
  const sampleMode = options?.sampleMode ?? 'multi7';
  const maxOutSamples = options?.maxOutSamples ?? (sampleMode === 'multi7' ? 1 : 0);
  const defaultMaxDropRatio = readGlobalNumber('__SEAM_MAX_DROP_RATIO__') ?? 0.35;
  const maxDropRatio = options?.maxDropRatio ?? defaultMaxDropRatio;
  const proofFailOpen = readGlobalBoolean('__SEAM_PROOF_FAIL_OPEN__', true);
  const triCount = mesh.indices.length / 3;
  const kept: number[] = [];
  let dropped = 0;
  let classifier: any | undefined;
  const faceIndex = options?.faceIndex ?? -1;
  const surfaceType = options?.surfaceType ?? 'Unknown';
  const periodicProof = options?.periodicProof ?? false;
  let proofRecoverableByU = 0;
  let proofRecoverableByUV = 0;
  let proofCheckedDropped = 0;

  function classifyOut(classifierInst: any, u: number, v: number): boolean {
    const uvPoint = new oc.gp_Pnt2d_3(u, v);
    try {
      const state = classifierInst.Perform(uvPoint, recadreOnPeriodic);
      return topAbsStateEquals(state, oc.TopAbs_State.TopAbs_OUT);
    } catch {
      // Treat failures as OUT for proof conservatism.
      return true;
    } finally {
      uvPoint.delete?.();
    }
  }

  function outSamplesWithShiftModes(classifierInst: any, samples: Array<[number, number]>): { outU: number; outUV: number } {
    const twoPi = Math.PI * 2;
    let outU = 0;
    let outUV = 0;
    const uShifts = [-twoPi, 0, twoPi];
    const uvShifts = surfaceType === 'Torus'
      ? [
          [-twoPi, -twoPi], [-twoPi, 0], [-twoPi, twoPi],
          [0, -twoPi], [0, 0], [0, twoPi],
          [twoPi, -twoPi], [twoPi, 0], [twoPi, twoPi],
        ]
      : [[0, 0]];

    for (const [u, v] of samples) {
      let uEquivalentInside = false;
      for (const du of uShifts) {
        if (!classifyOut(classifierInst, u + du, v)) {
          uEquivalentInside = true;
          break;
        }
      }
      if (!uEquivalentInside) {
        outU++;
        if (outU > maxOutSamples) {
          // Keep counting outUV for signal quality, but u-only already failed.
        }
      }

      let uvEquivalentInside = false;
      for (const [du, dv] of uvShifts) {
        if (!classifyOut(classifierInst, u + du, v + dv)) {
          uvEquivalentInside = true;
          break;
        }
      }
      if (!uvEquivalentInside) {
        outUV++;
      }
    }

    return { outU, outUV };
  }

  try {
    classifier = new oc.BRepTopAdaptor_FClass2d(occFace, tol);

    for (let i = 0; i < mesh.indices.length; i += 3) {
      const ia = mesh.indices[i + 0];
      const ib = mesh.indices[i + 1];
      const ic = mesh.indices[i + 2];

      const uvs = mesh.uvs!;
      const uA = uvs[ia * 2 + 0];
      const vA = uvs[ia * 2 + 1];
      const uB = uvs[ib * 2 + 0];
      const vB = uvs[ib * 2 + 1];
      const uC = uvs[ic * 2 + 0];
      const vC = uvs[ic * 2 + 1];
      const uCentroid = (uA + uB + uC) / 3;
      const vCentroid = (vA + vB + vC) / 3;

      const samplePoints: Array<[number, number]> = sampleMode === 'centroid'
        ? [[uCentroid, vCentroid]]
        : [
            [uA, vA],
            [uB, vB],
            [uC, vC],
            [(uA + uB) / 2, (vA + vB) / 2],
            [(uB + uC) / 2, (vB + vC) / 2],
            [(uC + uA) / 2, (vC + vA) / 2],
            [uCentroid, vCentroid],
          ];

      let outSamples = 0;
      for (const [u, v] of samplePoints) {
        try {
          if (classifyOut(classifier, u, v)) {
            outSamples++;
            if (outSamples > maxOutSamples) {
              break;
            }
          }
        } catch {
          // Ignore per-point classification failures and keep evaluating other samples.
        }
      }

      if (outSamples <= maxOutSamples) {
        kept.push(ia, ib, ic);
      } else {
        if (periodicProof) {
          proofCheckedDropped++;
          const { outU, outUV } = outSamplesWithShiftModes(classifier, samplePoints);
          if (outU <= maxOutSamples) {
            proofRecoverableByU++;
          }
          if (outUV <= maxOutSamples) {
            proofRecoverableByUV++;
          }
        }
        dropped++;
      }
    }
  } catch (e) {
    console.warn(`[seam-filter] ${label}: classifier setup failed, keeping original mesh`, e);
    return mesh;
  } finally {
    classifier?.delete?.();
  }

  if (dropped === 0) {
    if (logAlways) {
      console.log(
        `[seam-filter] ${label}: dropped 0/${triCount} triangles (kept ${triCount}) mode=${sampleMode} maxOut=${maxOutSamples}`
      );
    }
    return mesh;
  }

  if (kept.length === 0) {
    console.warn(`[seam-filter] ${label}: classifier dropped all ${triCount} triangles, keeping original mesh`);
    return mesh;
  }

  if (periodicProof && proofFailOpen && proofCheckedDropped > 0 && proofRecoverableByUV > 0) {
    const pctUV = (proofRecoverableByUV / proofCheckedDropped * 100).toFixed(2);
    console.warn(
      `[seam-filter] ${label}: proof found ${proofRecoverableByUV}/${proofCheckedDropped} dropped triangles recoverable by UV shifts (${pctUV}%), keeping original mesh`
    );
    return mesh;
  }

  if (triCount > 0) {
    const dropRatio = dropped / triCount;
    if (dropRatio > maxDropRatio) {
      console.warn(
        `[seam-filter] ${label}: dropped ${(dropRatio * 100).toFixed(1)}% (>${(maxDropRatio * 100).toFixed(1)}% cap), keeping original mesh`
      );
      return mesh;
    }
  }

  console.log(
    `[seam-filter] ${label}: dropped ${dropped}/${triCount} triangles (kept ${kept.length / 3}) mode=${sampleMode} maxOut=${maxOutSamples}`
  );
  if (periodicProof && proofCheckedDropped > 0) {
    const pctU = (proofRecoverableByU / proofCheckedDropped * 100).toFixed(2);
    const pctUV = (proofRecoverableByUV / proofCheckedDropped * 100).toFixed(2);
    console.log(
      `[seam-proof] face=${faceIndex} type=${surfaceType} dropped=${proofCheckedDropped} recoverableByU=${proofRecoverableByU} (${pctU}%) recoverableByUV=${proofRecoverableByUV} (${pctUV}%) mode=${sampleMode} maxOut=${maxOutSamples}`
    );
  }
  return { ...mesh, indices: new Uint32Array(kept) } as T;
}

type Vec2 = [number, number];
interface TrimLoopsUV {
  uvOuter: Vec2[];
  uvHoles: Vec2[][];
  uvOuterRawWrapped?: Vec2[];
}

interface TrimLoopValidationResult {
  ok: boolean;
  reason?: string;
  uvOuter: Vec2[];
  uvHoles: Vec2[][];
  outerAreaAbs: number;
  totalHoleAreaAbs: number;
}

interface ConeSeamSplitResult {
  ok: boolean;
  reason?: string;
  uSplit?: number;
  splitMode?: 'seam-crossing' | 'u-span-fallback';
  leftPatch?: TrimLoopsUV;
  rightPatch?: TrimLoopsUV;
  leftAreaAbs?: number;
  rightAreaAbs?: number;
  areaBalance?: number;
}

interface TessellatedMeshLike {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  uvs?: Float32Array;
}

function wrapToPi(angleRad: number): number {
  const twoPi = Math.PI * 2;
  // Normalize into (-2π, 2π) first for numerical stability
  angleRad = angleRad % twoPi;
  if (angleRad > Math.PI) angleRad -= twoPi;
  if (angleRad < -Math.PI) angleRad += twoPi;
  return angleRad;
}

function unwrapPeriodicLoopU(points: Vec2[], period: number = Math.PI * 2): Vec2[] {
  return unwrapPeriodicLoopComponent(points, 0, period);
}

function unwrapClosedPeriodicLoopUOnce(points: Vec2[], period: number = Math.PI * 2): Vec2[] {
  if (points.length < 2) return points.map(([u, v]): Vec2 => [u, v]);
  // Include the closing edge so seam jumps on the loop closure are unwrapped once as well.
  const closed = [...points, points[0]];
  const unwrapped = unwrapPeriodicLoopU(closed, period);
  unwrapped.pop();
  return simplifyLoop2D(unwrapped);
}

function shiftLoopU(points: Vec2[], deltaU: number): Vec2[] {
  if (deltaU === 0) return points;
  return points.map(([u, v]): Vec2 => [u + deltaU, v]);
}

function getLoopUBounds(points: Vec2[]): { uMin: number; uMax: number } {
  let uMin = Infinity;
  let uMax = -Infinity;
  for (const [u] of points) {
    uMin = Math.min(uMin, u);
    uMax = Math.max(uMax, u);
  }
  return { uMin, uMax };
}

function meanLoopU(points: Vec2[]): number {
  if (points.length === 0) return 0;
  let sum = 0;
  for (const [u] of points) {
    sum += u;
  }
  return sum / points.length;
}

function chooseShiftToRange(
  uMin: number,
  uMax: number,
  period: number,
  targetMin: number,
  targetMax: number
): number {
  let bestK = 0;
  let bestScore = Infinity;
  const targetCenter = (targetMin + targetMax) * 0.5;

  for (let k = -3; k <= 3; k++) {
    const shiftedMin = uMin + k * period;
    const shiftedMax = uMax + k * period;
    const under = Math.max(0, targetMin - shiftedMin);
    const over = Math.max(0, shiftedMax - targetMax);
    const center = (shiftedMin + shiftedMax) * 0.5;
    const centerPenalty = Math.abs(center - targetCenter) * 1e-3;
    const score = under + over + centerPenalty;

    if (score < bestScore) {
      bestScore = score;
      bestK = k;
    }
  }

  return bestK;
}

function unwrapPeriodicLoopComponent(
  points: Vec2[],
  component: 0 | 1,
  period: number = Math.PI * 2
): Vec2[] {
  if (points.length === 0) return [];
  // Only unwrap near full-period jumps (seam crossings), not ~half-period jumps.
  const unwrapThreshold = period * 0.75;
  const eps = 1e-6;
  const out: Vec2[] = [];

  let offset = 0;
  let prevUnwrapped = points[0][component];
  out.push([points[0][0], points[0][1]]);

  for (let i = 1; i < points.length; i++) {
    const raw = points[i][component];
    let unwrapped = raw + offset;
    const delta = unwrapped - prevUnwrapped;

    if (delta > unwrapThreshold + eps) {
      offset -= period;
      unwrapped = raw + offset;
    } else if (delta < -unwrapThreshold - eps) {
      offset += period;
      unwrapped = raw + offset;
    }

    const nextPoint: Vec2 = [points[i][0], points[i][1]];
    nextPoint[component] = unwrapped;
    out.push(nextPoint);
    prevUnwrapped = unwrapped;
  }

  return out;
}

function shiftLoopComponent(points: Vec2[], component: 0 | 1, delta: number): Vec2[] {
  if (delta === 0) return points;
  return points.map(([u, v]): Vec2 => {
    if (component === 0) {
      return [u + delta, v];
    }
    return [u, v + delta];
  });
}

function getLoopComponentBounds(points: Vec2[], component: 0 | 1): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const p of points) {
    const value = p[component];
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return { min, max };
}

function meanLoopComponent(points: Vec2[], component: 0 | 1): number {
  if (points.length === 0) return 0;
  let sum = 0;
  for (const p of points) {
    sum += p[component];
  }
  return sum / points.length;
}

function normalizePeriodicTrimLoops(
  uvOuter: Vec2[],
  uvHoles: Vec2[][],
  options: {
    periodicU: boolean;
    periodicV: boolean;
    periodU?: number;
    periodV?: number;
  }
): { uvOuter: Vec2[]; uvHoles: Vec2[][] } {
  let outer = uvOuter;
  let holes = uvHoles.map((h) => h);
  const periodU = options.periodU ?? (Math.PI * 2);
  const periodV = options.periodV ?? (Math.PI * 2);

  const normalizeComponent = (
    component: 0 | 1,
    period: number,
    targetMin: number,
    targetMax: number
  ) => {
    outer = unwrapPeriodicLoopComponent(outer, component, period);
    const { min, max } = getLoopComponentBounds(outer, component);
    const outerShiftK = chooseShiftToRange(min, max, period, targetMin, targetMax);
    if (outerShiftK !== 0) {
      outer = shiftLoopComponent(outer, component, outerShiftK * period);
    }

    const outerMean = meanLoopComponent(outer, component);
    holes = holes.map((hole) => {
      let alignedHole = unwrapPeriodicLoopComponent(hole, component, period);
      if (outerShiftK !== 0) {
        alignedHole = shiftLoopComponent(alignedHole, component, outerShiftK * period);
      }
      const holeMean = meanLoopComponent(alignedHole, component);
      const alignK = Math.round((outerMean - holeMean) / period);
      if (alignK !== 0) {
        alignedHole = shiftLoopComponent(alignedHole, component, alignK * period);
      }
      return alignedHole;
    });
  };

  if (options.periodicU) {
    normalizeComponent(0, periodU, 0, periodU);
  }
  if (options.periodicV) {
    normalizeComponent(1, periodV, 0, periodV);
  }

  return {
    uvOuter: simplifyLoop2D(outer),
    uvHoles: holes.map((h) => simplifyLoop2D(h)),
  };
}

function simplifyLoop2D(points: Vec2[], eps: number = 1e-10): Vec2[] {
  if (points.length === 0) return points;

  const out: Vec2[] = [];
  const almostEqual = (a: Vec2, b: Vec2) =>
    Math.abs(a[0] - b[0]) <= eps && Math.abs(a[1] - b[1]) <= eps;

  for (const p of points) {
    if (out.length === 0 || !almostEqual(out[out.length - 1], p)) {
      out.push(p);
    }
  }

  // If loop ended up with same start/end, drop the duplicate end.
  if (out.length >= 2 && almostEqual(out[0], out[out.length - 1])) {
    out.pop();
  }

  return out;
}

function pointSegmentDistance2D(point: Vec2, a: Vec2, b: Vec2): number {
  const abU = b[0] - a[0];
  const abV = b[1] - a[1];
  const apU = point[0] - a[0];
  const apV = point[1] - a[1];
  const abLen2 = abU * abU + abV * abV;
  if (abLen2 <= 1e-24) {
    const du = point[0] - a[0];
    const dv = point[1] - a[1];
    return Math.sqrt(du * du + dv * dv);
  }
  const t = Math.max(0, Math.min(1, (apU * abU + apV * abV) / abLen2));
  const projU = a[0] + t * abU;
  const projV = a[1] + t * abV;
  const du = point[0] - projU;
  const dv = point[1] - projV;
  return Math.sqrt(du * du + dv * dv);
}

function simplifyPolylineRDP(points: Vec2[], tolerance: number): Vec2[] {
  if (points.length <= 2 || tolerance <= 0) {
    return points.slice();
  }

  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: Array<[number, number]> = [[0, points.length - 1]];

  while (stack.length > 0) {
    const [start, end] = stack.pop()!;
    let maxDist = -1;
    let maxIdx = -1;
    for (let i = start + 1; i < end; i++) {
      const dist = pointSegmentDistance2D(points[i], points[start], points[end]);
      if (dist > maxDist) {
        maxDist = dist;
        maxIdx = i;
      }
    }
    if (maxIdx > start && maxIdx < end && maxDist > tolerance) {
      keep[maxIdx] = 1;
      stack.push([start, maxIdx], [maxIdx, end]);
    }
  }

  const simplified: Vec2[] = [];
  for (let i = 0; i < points.length; i++) {
    if (keep[i]) simplified.push(points[i]);
  }
  return simplified;
}

function loopBounds2D(points: Vec2[]): { minU: number; maxU: number; minV: number; maxV: number } {
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const [u, v] of points) {
    minU = Math.min(minU, u);
    maxU = Math.max(maxU, u);
    minV = Math.min(minV, v);
    maxV = Math.max(maxV, v);
  }
  return { minU, maxU, minV, maxV };
}

function simplifyClosedLoopRDP(loopInput: Vec2[], tolerance: number, minPoints: number = 3): Vec2[] {
  const loop = simplifyLoop2D(loopInput);
  if (loop.length <= minPoints || tolerance <= 0) {
    return loop;
  }

  // Split the closed loop at a stable pivot (farthest point from centroid)
  // before running open-polyline RDP.
  const centroid = loopCentroid2D(loop);
  let pivot = 0;
  let maxDist2 = -1;
  for (let i = 0; i < loop.length; i++) {
    const du = loop[i][0] - centroid[0];
    const dv = loop[i][1] - centroid[1];
    const dist2 = du * du + dv * dv;
    if (dist2 > maxDist2) {
      maxDist2 = dist2;
      pivot = i;
    }
  }

  const rotated: Vec2[] = [];
  for (let i = 0; i < loop.length; i++) {
    rotated.push(loop[(pivot + i) % loop.length]);
  }
  rotated.push(rotated[0]);

  const simplifiedOpen = simplifyPolylineRDP(rotated, tolerance);
  if (simplifiedOpen.length < minPoints + 1) {
    return loop;
  }

  const simplifiedClosed = simplifyLoop2D(simplifiedOpen);
  return simplifiedClosed.length >= minPoints ? simplifiedClosed : loop;
}

function simplifyLoopForMeshing(loopInput: Vec2[], targetPoints: number, maxAreaErrorRatio: number): Vec2[] {
  const loop = simplifyLoop2D(loopInput);
  if (loop.length <= 3 || loop.length <= targetPoints) {
    return loop;
  }

  const sourceArea = loopAreaAbs2D(loop);
  const bounds = loopBounds2D(loop);
  const diag = Math.hypot(bounds.maxU - bounds.minU, bounds.maxV - bounds.minV);
  if (!Number.isFinite(diag) || diag <= 1e-12) {
    return loop;
  }

  let lo = 0;
  let hi = diag * 0.5;
  let best = loop;

  for (let iter = 0; iter < 22; iter++) {
    const mid = (lo + hi) * 0.5;
    const candidate = simplifyClosedLoopRDP(loop, mid, 3);
    if (candidate.length < 3) {
      hi = mid;
      continue;
    }
    const candidateArea = loopAreaAbs2D(candidate);
    const areaErrorRatio = sourceArea > 1e-12
      ? Math.abs(candidateArea - sourceArea) / sourceArea
      : 0;
    const areaOk = areaErrorRatio <= maxAreaErrorRatio;
    if (areaOk) {
      best = candidate;
      lo = mid;
    } else {
      hi = mid;
    }
  }

  if (best.length <= targetPoints) {
    return best;
  }

  // Final cap for extreme loops: uniform decimation with area guard.
  const step = Math.ceil(best.length / targetPoints);
  const decimated: Vec2[] = [];
  for (let i = 0; i < best.length; i += step) {
    decimated.push(best[i]);
  }
  const decimatedLoop = simplifyLoop2D(decimated);
  if (decimatedLoop.length < 3) {
    return best;
  }
  const decimatedArea = loopAreaAbs2D(decimatedLoop);
  const decimatedAreaErrorRatio = sourceArea > 1e-12
    ? Math.abs(decimatedArea - sourceArea) / sourceArea
    : 0;
  const relaxedAreaErrorRatio = Math.max(maxAreaErrorRatio * 4, 0.12);
  if (
    decimatedAreaErrorRatio <= maxAreaErrorRatio ||
    (loop.length > targetPoints * 4 && decimatedAreaErrorRatio <= relaxedAreaErrorRatio)
  ) {
    return decimatedLoop;
  }
  if (best.length > targetPoints * 2) {
    const hardStep = Math.ceil(best.length / targetPoints);
    const hardDecimated: Vec2[] = [];
    for (let i = 0; i < best.length; i += hardStep) {
      hardDecimated.push(best[i]);
    }
    const hardLoop = simplifyLoop2D(hardDecimated);
    if (hardLoop.length >= 3) {
      return hardLoop;
    }
  }
  return best;
}

function signedLoopArea2D(points: Vec2[]): number {
  if (points.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }
  return area * 0.5;
}

function loopAreaAbs2D(points: Vec2[]): number {
  return Math.abs(signedLoopArea2D(points));
}

function loopCentroid2D(points: Vec2[]): Vec2 {
  if (points.length === 0) return [0, 0];
  let su = 0;
  let sv = 0;
  for (const [u, v] of points) {
    su += u;
    sv += v;
  }
  return [su / points.length, sv / points.length];
}

function pointOnSegment2DInclusive(point: Vec2, a: Vec2, b: Vec2, eps: number = 1e-8): boolean {
  const cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]);
  if (Math.abs(cross) > eps) return false;
  const minU = Math.min(a[0], b[0]) - eps;
  const maxU = Math.max(a[0], b[0]) + eps;
  const minV = Math.min(a[1], b[1]) - eps;
  const maxV = Math.max(a[1], b[1]) + eps;
  return point[0] >= minU && point[0] <= maxU && point[1] >= minV && point[1] <= maxV;
}

function isPointInPolygonInclusive2D(point: Vec2, polygon: Vec2[], eps: number = 1e-8): boolean {
  if (polygon.length < 3) return false;
  for (let i = 0; i < polygon.length; i++) {
    if (pointOnSegment2DInclusive(point, polygon[i], polygon[(i + 1) % polygon.length], eps)) {
      return true;
    }
  }
  let inside = false;
  const [x, y] = point;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > y) === (yj > y)) continue;
    const xAtY = ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (x < xAtY) inside = !inside;
  }
  return inside;
}

function isLoopInsidePolygonInclusive2D(loop: Vec2[], polygon: Vec2[], eps: number = 1e-8): boolean {
  if (loop.length < 3 || polygon.length < 3) return false;
  for (const p of loop) {
    if (!isPointInPolygonInclusive2D(p, polygon, eps)) {
      return false;
    }
  }
  return true;
}

function intersectSegmentAtU(a: Vec2, b: Vec2, uCut: number): Vec2 {
  const du = b[0] - a[0];
  if (Math.abs(du) < 1e-12) {
    return [uCut, (a[1] + b[1]) * 0.5];
  }
  const t = Math.max(0, Math.min(1, (uCut - a[0]) / du));
  return [uCut, a[1] + t * (b[1] - a[1])];
}

function clipLoopByUHalfPlane(points: Vec2[], uCut: number, keepLowerU: boolean, eps: number = 1e-9): Vec2[] {
  if (points.length < 3) return [];
  const isInside = (p: Vec2) => keepLowerU ? p[0] <= uCut + eps : p[0] >= uCut - eps;
  const out: Vec2[] = [];
  let prev = points[points.length - 1];
  let prevInside = isInside(prev);

  for (let i = 0; i < points.length; i++) {
    const curr = points[i];
    const currInside = isInside(curr);

    if (currInside !== prevInside) {
      out.push(intersectSegmentAtU(prev, curr, uCut));
    }
    if (currInside) {
      out.push(curr);
    }

    prev = curr;
    prevInside = currInside;
  }

  return simplifyLoop2D(out);
}

function validateAndSanitizeTrimLoops(
  uvOuterInput: Vec2[],
  uvHolesInput: Vec2[][],
  options?: {
    minAreaAbs?: number;
    maxHoleToOuterRatio?: number;
    failOnHoleOutside?: boolean;
    failOnHugeHole?: boolean;
  }
): TrimLoopValidationResult {
  const minAreaAbs = options?.minAreaAbs ?? 1e-7;
  const maxHoleToOuterRatio = options?.maxHoleToOuterRatio ?? 0.98;
  const failOnHoleOutside = options?.failOnHoleOutside ?? false;
  const failOnHugeHole = options?.failOnHugeHole ?? false;

  const uvOuter = simplifyLoop2D(uvOuterInput);
  const outerAreaAbs = loopAreaAbs2D(uvOuter);
  if (uvOuter.length < 3 || outerAreaAbs <= minAreaAbs) {
    return {
      ok: false,
      reason: `outer-invalid(len=${uvOuter.length}, area=${outerAreaAbs.toExponential(3)})`,
      uvOuter,
      uvHoles: [],
      outerAreaAbs,
      totalHoleAreaAbs: 0,
    };
  }

  const uvHoles: Vec2[][] = [];
  let totalHoleAreaAbs = 0;
  for (const rawHole of uvHolesInput) {
    const hole = simplifyLoop2D(rawHole);
    const holeAreaAbs = loopAreaAbs2D(hole);
    if (hole.length < 3 || holeAreaAbs <= minAreaAbs) {
      continue;
    }
    if (holeAreaAbs >= outerAreaAbs * maxHoleToOuterRatio) {
      if (failOnHugeHole) {
        return {
          ok: false,
          reason: `hole-too-large(area=${holeAreaAbs.toExponential(3)}, outer=${outerAreaAbs.toExponential(3)})`,
          uvOuter,
          uvHoles,
          outerAreaAbs,
          totalHoleAreaAbs,
        };
      }
      continue;
    }
    const c = loopCentroid2D(hole);
    if (!isPointInPolygonInclusive2D(c, uvOuter)) {
      if (failOnHoleOutside) {
        return {
          ok: false,
          reason: 'hole-outside-outer',
          uvOuter,
          uvHoles,
          outerAreaAbs,
          totalHoleAreaAbs,
        };
      }
      continue;
    }
    uvHoles.push(hole);
    totalHoleAreaAbs += holeAreaAbs;
  }

  if (totalHoleAreaAbs >= outerAreaAbs * maxHoleToOuterRatio) {
    return {
      ok: false,
      reason: `hole-area-ratio-invalid(sum=${totalHoleAreaAbs.toExponential(3)}, outer=${outerAreaAbs.toExponential(3)})`,
      uvOuter,
      uvHoles,
      outerAreaAbs,
      totalHoleAreaAbs,
    };
  }

  return { ok: true, uvOuter, uvHoles, outerAreaAbs, totalHoleAreaAbs };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[mid];
  }
  return 0.5 * (sorted[mid - 1] + sorted[mid]);
}

function findConeSeamSplitUFromCrossings(
  wrappedOuter: Vec2[],
  unwrappedOuter: Vec2[],
  period: number = Math.PI * 2
): number | null {
  if (wrappedOuter.length < 3 || unwrappedOuter.length < 3) {
    return null;
  }
  const bounds = getLoopUBounds(unwrappedOuter);
  const margin = Math.max(1e-6, (bounds.uMax - bounds.uMin) * 1e-4);
  if (!(bounds.uMax > bounds.uMin + margin * 2)) {
    return null;
  }

  const seamCrossingU: number[] = [];
  const seamJumpThreshold = Math.PI + 1e-6; // true periodic seam crossing
  for (let i = 0; i < wrappedOuter.length; i++) {
    const j = (i + 1) % wrappedOuter.length;
    const rawJump = Math.abs(wrappedOuter[j][0] - wrappedOuter[i][0]);
    if (rawJump <= seamJumpThreshold) continue;
    let cutU = 0.5 * (wrappedOuter[i][0] + wrappedOuter[j][0]);
    while (cutU < bounds.uMin) cutU += period;
    while (cutU > bounds.uMax) cutU -= period;
    if (cutU > bounds.uMin + margin && cutU < bounds.uMax - margin) {
      seamCrossingU.push(cutU);
    }
  }
  if (seamCrossingU.length < 2) {
    return null;
  }

  let uSplit = median(seamCrossingU);
  while (uSplit < bounds.uMin) uSplit += period;
  while (uSplit > bounds.uMax) uSplit -= period;
  if (uSplit <= bounds.uMin + margin || uSplit >= bounds.uMax - margin) {
    return null;
  }
  return uSplit;
}

function splitConeTrimLoopsIntoTwoPatches(
  wrappedOuterForSeam: Vec2[],
  uvOuterUnwrapped: Vec2[],
  uvHoles: Vec2[][],
): ConeSeamSplitResult {
  const sourceValidation = validateAndSanitizeTrimLoops(uvOuterUnwrapped, uvHoles, {
    minAreaAbs: 1e-7,
    maxHoleToOuterRatio: 0.98,
    failOnHoleOutside: true,
    failOnHugeHole: true,
  });
  if (!sourceValidation.ok) {
    return { ok: false, reason: `source-invalid:${sourceValidation.reason ?? 'unknown'}` };
  }

  let uSplit = findConeSeamSplitUFromCrossings(wrappedOuterForSeam, sourceValidation.uvOuter);
  let splitMode: ConeSeamSplitResult['splitMode'] = 'seam-crossing';
  if (uSplit == null) {
    const bounds = getLoopUBounds(sourceValidation.uvOuter);
    const span = bounds.uMax - bounds.uMin;
    const margin = Math.max(1e-6, span * 1e-4);
    if (!(span > margin * 2)) {
      return { ok: false, reason: 'no-seam-crossing-cut' };
    }

    // Fallback for faces that do not present a detectable seam jump in wrapped UV,
    // but still benefit from split tessellation due to heavy trim-hole topology.
    const candidateFractions = [0.50, 0.45, 0.55, 0.40, 0.60];
    let bestCandidate: number | null = null;
    let bestBalance = -Infinity;
    let bestMinArea = -Infinity;
    for (const t of candidateFractions) {
      const cutU = bounds.uMin + span * t;
      if (cutU <= bounds.uMin + margin || cutU >= bounds.uMax - margin) continue;
      const left = clipLoopByUHalfPlane(sourceValidation.uvOuter, cutU, true);
      const right = clipLoopByUHalfPlane(sourceValidation.uvOuter, cutU, false);
      if (left.length < 3 || right.length < 3) continue;
      const leftAreaAbs = loopAreaAbs2D(left);
      const rightAreaAbs = loopAreaAbs2D(right);
      if (leftAreaAbs <= 1e-7 || rightAreaAbs <= 1e-7) continue;
      const minArea = Math.min(leftAreaAbs, rightAreaAbs);
      const maxArea = Math.max(leftAreaAbs, rightAreaAbs);
      const balance = maxArea > 0 ? minArea / maxArea : 0;
      if (
        balance > bestBalance + 1e-9 ||
        (Math.abs(balance - bestBalance) <= 1e-9 && minArea > bestMinArea)
      ) {
        bestCandidate = cutU;
        bestBalance = balance;
        bestMinArea = minArea;
      }
    }

    if (bestCandidate == null) {
      return { ok: false, reason: 'no-usable-fallback-cut' };
    }
    uSplit = bestCandidate;
    splitMode = 'u-span-fallback';
  }

  const leftOuter = clipLoopByUHalfPlane(sourceValidation.uvOuter, uSplit, true);
  const rightOuter = clipLoopByUHalfPlane(sourceValidation.uvOuter, uSplit, false);
  const leftHoles: Vec2[][] = [];
  const rightHoles: Vec2[][] = [];
  for (const hole of sourceValidation.uvHoles) {
    const hb = getLoopUBounds(hole);
    // Preserve non-crossing holes exactly; clipping those can distort hole shape.
    if (hb.maxU <= uSplit + 1e-9) {
      leftHoles.push(hole);
      continue;
    }
    if (hb.minU >= uSplit - 1e-9) {
      rightHoles.push(hole);
      continue;
    }

    // Hole crosses split line: clip into both patches.
    const leftClip = clipLoopByUHalfPlane(hole, uSplit, true);
    const rightClip = clipLoopByUHalfPlane(hole, uSplit, false);
    if (leftClip.length >= 3) leftHoles.push(leftClip);
    if (rightClip.length >= 3) rightHoles.push(rightClip);
  }

  const leftValidation = validateAndSanitizeTrimLoops(leftOuter, leftHoles, {
    minAreaAbs: 1e-7,
    maxHoleToOuterRatio: 0.98,
    failOnHoleOutside: true,
    failOnHugeHole: true,
  });
  if (!leftValidation.ok) {
    return { ok: false, reason: `left-invalid:${leftValidation.reason ?? 'unknown'}` };
  }

  const rightValidation = validateAndSanitizeTrimLoops(rightOuter, rightHoles, {
    minAreaAbs: 1e-7,
    maxHoleToOuterRatio: 0.98,
    failOnHoleOutside: true,
    failOnHugeHole: true,
  });
  if (!rightValidation.ok) {
    return { ok: false, reason: `right-invalid:${rightValidation.reason ?? 'unknown'}` };
  }

  const leftAreaAbs = leftValidation.outerAreaAbs;
  const rightAreaAbs = rightValidation.outerAreaAbs;
  const minArea = Math.min(leftAreaAbs, rightAreaAbs);
  const maxArea = Math.max(leftAreaAbs, rightAreaAbs);
  const areaBalance = maxArea > 0 ? minArea / maxArea : 0;
  if (areaBalance < 0.05) {
    return {
      ok: false,
      reason: `split-unbalanced(balance=${areaBalance.toFixed(4)})`,
      uSplit,
      splitMode,
      leftAreaAbs,
      rightAreaAbs,
      areaBalance,
    };
  }

  return {
    ok: true,
    uSplit,
    splitMode,
    leftPatch: { uvOuter: leftValidation.uvOuter, uvHoles: leftValidation.uvHoles },
    rightPatch: { uvOuter: rightValidation.uvOuter, uvHoles: rightValidation.uvHoles },
    leftAreaAbs,
    rightAreaAbs,
    areaBalance,
  };
}

function mergeTessellatedMeshes(meshes: TessellatedMeshLike[]): TessellatedMeshLike {
  if (meshes.length === 0) {
    return {
      positions: new Float32Array(),
      normals: new Float32Array(),
      indices: new Uint32Array(),
      uvs: new Float32Array(),
    };
  }
  if (meshes.length === 1) {
    return meshes[0];
  }

  let totalVerts = 0;
  let totalIndices = 0;
  let hasUvs = false;
  for (const mesh of meshes) {
    totalVerts += mesh.positions.length / 3;
    totalIndices += mesh.indices.length;
    hasUvs = hasUvs || !!mesh.uvs;
  }

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const indices = new Uint32Array(totalIndices);
  const uvs = hasUvs ? new Float32Array(totalVerts * 2) : undefined;

  let vertBase = 0;
  let posOffset = 0;
  let idxOffset = 0;
  let uvOffset = 0;
  for (const mesh of meshes) {
    positions.set(mesh.positions, posOffset);
    normals.set(mesh.normals, posOffset);
    posOffset += mesh.positions.length;

    if (uvs) {
      if (mesh.uvs) {
        uvs.set(mesh.uvs, uvOffset);
      } else {
        for (let i = 0; i < (mesh.positions.length / 3) * 2; i++) {
          uvs[uvOffset + i] = 0;
        }
      }
      uvOffset += (mesh.positions.length / 3) * 2;
    }

    for (let i = 0; i < mesh.indices.length; i++) {
      indices[idxOffset + i] = mesh.indices[i] + vertBase;
    }
    idxOffset += mesh.indices.length;
    vertBase += mesh.positions.length / 3;
  }

  return { positions, normals, indices, uvs };
}

function projectPointsToUV(
  oc: any,
  shapeAnalysisSurface: any,
  points: Vec3[],
  opts: { wrapU: boolean; wrapV: boolean }
): Vec2[] {
  const uv: Vec2[] = [];
  const tol = 1e-7;

  for (const p of points) {
    const gp = new oc.gp_Pnt_3(p[0], p[1], p[2]);
    const p2d = shapeAnalysisSurface.ValueOfUV(gp, tol);

    let u = p2d.X();
    let v = p2d.Y();

    if (opts.wrapU) u = wrapToPi(u);
    if (opts.wrapV) v = wrapToPi(v);

    if (isFinite(u) && isFinite(v)) {
      uv.push([u, v]);
    }

    // Avoid leaking WASM heap allocations
    gp.delete?.();
    p2d.delete?.();
  }

  return simplifyLoop2D(uv);
}

function curveTypeNameFromGeomAbs(curveTypeEnum: any): string {
  const typeValue = typeof curveTypeEnum === 'object' && curveTypeEnum !== null
    ? curveTypeEnum.value
    : curveTypeEnum;
  const typeMap: Record<number, string> = {
    0: 'Line',
    1: 'Circle',
    2: 'Ellipse',
    3: 'Hyperbola',
    4: 'Parabola',
    5: 'BezierCurve',
    6: 'BSplineCurve',
    7: 'OffsetCurve',
    8: 'OtherCurve'
  };
  return typeMap[typeValue] || `Unknown(${typeValue})`;
}

function curveSampleCount(curveType: string, paramRange: number): number {
  const MIN_SAMPLES = Math.max(2, Math.floor(readGlobalNumber('__PCURVE_MIN_SAMPLES__') ?? 6));
  const MAX_SAMPLES = Math.max(MIN_SAMPLES, Math.floor(readGlobalNumber('__PCURVE_MAX_SAMPLES__') ?? 96));
  let count = Math.max(MIN_SAMPLES, Math.floor(readGlobalNumber('__PCURVE_DEFAULT_SAMPLES__') ?? 20));
  if (curveType === 'Line') {
    count = 2;
  } else if (curveType === 'Circle') {
    // ~15 degree step for trimmed p-curves (full circle ~= 24 samples).
    const angleStep = Math.PI / 12;
    count = Math.ceil(Math.abs(paramRange) / angleStep);
  } else if (curveType === 'Ellipse' || curveType === 'BSplineCurve') {
    count = Math.max(MIN_SAMPLES, Math.floor(readGlobalNumber('__PCURVE_COMPLEX_SAMPLES__') ?? 28));
  }
  if (!isFinite(count)) count = 32;
  return Math.max(MIN_SAMPLES, Math.min(MAX_SAMPLES, count));
}

function dist2D(a: Vec2, b: Vec2): number {
  const du = a[0] - b[0];
  const dv = a[1] - b[1];
  return Math.sqrt(du * du + dv * dv);
}

function appendLoopSegment(loop: Vec2[], segment: Vec2[], eps: number = 1e-8): void {
  if (segment.length === 0) return;
  if (loop.length === 0) {
    loop.push(...segment);
    return;
  }
  const last = loop[loop.length - 1];
  const first = segment[0];
  if (dist2D(last, first) <= eps) {
    loop.push(...segment.slice(1));
  } else {
    loop.push(...segment);
  }
}

function sampleEdgePcurveUV(
  oc: any,
  edge: any,
  occFace: any
): Vec2[] | null {
  if (!oc.BRepAdaptor_Curve2d_2) return null;

  let adaptor2d: any | null = null;
  try {
    adaptor2d = new oc.BRepAdaptor_Curve2d_2(edge, occFace);
    const first = adaptor2d.FirstParameter();
    const last = adaptor2d.LastParameter();
    if (!isFinite(first) || !isFinite(last) || Math.abs(last - first) < 1e-12) {
      adaptor2d.delete?.();
      return null;
    }

    const curveType = curveTypeNameFromGeomAbs(adaptor2d.GetType());
    const sampleCount = curveSampleCount(curveType, last - first);
    const sampled: Vec2[] = [];
    for (let i = 0; i <= sampleCount; i++) {
      const t = first + (i / sampleCount) * (last - first);
      const p2 = adaptor2d.Value(t);
      sampled.push([p2.X(), p2.Y()]);
      p2.delete?.();
    }

    // Re-align to topological orientation so edge chaining is stable.
    let startUV: Vec2 | null = null;
    if (oc.TopExp?.FirstVertex && oc.BRep_Tool?.Parameters) {
      try {
        const firstVertex = oc.TopExp.FirstVertex(edge, true);
        if (firstVertex && !firstVertex.IsNull?.()) {
          const uvFirst = oc.BRep_Tool.Parameters(firstVertex, occFace);
          if (uvFirst) {
            startUV = [uvFirst.X(), uvFirst.Y()];
            uvFirst.delete?.();
          }
        }
        firstVertex?.delete?.();
      } catch {
        // Keep sampled order when vertex UV lookup fails.
      }
    }
    if (startUV && sampled.length >= 2) {
      const dFirst = dist2D(sampled[0], startUV);
      const dLast = dist2D(sampled[sampled.length - 1], startUV);
      if (dLast + 1e-10 < dFirst) {
        sampled.reverse();
      }
    }

    adaptor2d.delete?.();
    return simplifyLoop2D(sampled);
  } catch {
    adaptor2d?.delete?.();
    return null;
  }
}

function loopPerimeterUV(points: Vec2[]): number {
  if (points.length < 2) return 0;
  let perimeter = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    perimeter += dist2D(a, b);
  }
  return perimeter;
}

interface PcurveWireLoopsResult {
  wires: Vec2[][];
  wireHashes: number[];
  outerWireHash: number | null;
}

interface OcctInspiredConeTrimDomain {
  uvOuter: Vec2[];
  gridDensity: number;
  sourceLoopCount: number;
  sourcePointCount: number;
  uMin: number;
  uMax: number;
  vMin: number;
  vMax: number;
}

function getFaceTrimWireLoopsUVFromPCurves(
  oc: any,
  face: FaceWithEdgesInfo
): PcurveWireLoopsResult | null {
  if (!face.occFace || !oc.TopExp_Explorer_2 || !oc.BRepTools_WireExplorer_2 || !oc.TopoDS?.Wire_1 || !oc.TopoDS?.Edge_1) {
    return null;
  }

  const wires: Vec2[][] = [];
  const wireHashes: number[] = [];
  let outerWireHash: number | null = null;
  if (oc.BRepTools?.OuterWire) {
    try {
      const outerWire = oc.BRepTools.OuterWire(face.occFace);
      if (outerWire && !outerWire.IsNull?.()) {
        outerWireHash = outerWire.HashCode ? outerWire.HashCode(2147483647) : null;
      }
      outerWire?.delete?.();
    } catch {
      // Keep heuristic fallback.
    }
  }

  try {
    const wireExplorer = new oc.TopExp_Explorer_2(
      face.occFace,
      oc.TopAbs_ShapeEnum.TopAbs_WIRE,
      oc.TopAbs_ShapeEnum.TopAbs_SHAPE
    );

    while (wireExplorer.More()) {
      const wireShape = wireExplorer.Current();
      const wire = oc.TopoDS.Wire_1(wireShape);
      const wireHash = wire.HashCode ? wire.HashCode(2147483647) : wireHashes.length;

      const edgeExplorer = oc.BRepTools_WireExplorer_3
        ? new oc.BRepTools_WireExplorer_3(wire, face.occFace)
        : new oc.BRepTools_WireExplorer_2(wire);
      const loopUV: Vec2[] = [];

      while (edgeExplorer.More()) {
        const edgeShape = edgeExplorer.Current();
        const edge = oc.TopoDS.Edge_1(edgeShape);
        const sampled = sampleEdgePcurveUV(oc, edge, face.occFace);
        if (sampled && sampled.length >= 2) {
          appendLoopSegment(loopUV, sampled);
        }
        edgeExplorer.Next();
      }
      edgeExplorer.delete?.();

      const simplified = simplifyLoop2D(loopUV);
      if (simplified.length >= 3) {
        wires.push(simplified);
        wireHashes.push(wireHash);
      }
      wireExplorer.Next();
    }
    wireExplorer.delete?.();
  } catch {
    return null;
  }

  if (wires.length === 0) {
    return null;
  }

  return { wires, wireHashes, outerWireHash };
}

function buildOcctInspiredConeTrimDomainFromPCurves(
  oc: any,
  face: FaceWithEdgesInfo
): OcctInspiredConeTrimDomain | null {
  const pcurveLoops = getFaceTrimWireLoopsUVFromPCurves(oc, face);
  if (!pcurveLoops || pcurveLoops.wires.length === 0) {
    return null;
  }

  const period = Math.PI * 2;
  const unwrappedWires = pcurveLoops.wires
    .map((wire) => unwrapClosedPeriodicLoopUOnce(wire, period))
    .filter((wire) => wire.length >= 3);

  if (unwrappedWires.length === 0) {
    return null;
  }

  // Align all wires to a shared U band so the domain rectangle is compact/stable.
  const referenceMeanU = meanLoopU(unwrappedWires[0]);
  let alignedWires = unwrappedWires.map((wire) => {
    let aligned = wire;
    const alignK = Math.round((referenceMeanU - meanLoopU(aligned)) / period);
    if (alignK !== 0) {
      aligned = shiftLoopU(aligned, alignK * period);
    }
    return aligned;
  });

  let uMin = Infinity;
  let uMax = -Infinity;
  let vMin = Infinity;
  let vMax = -Infinity;
  for (const wire of alignedWires) {
    for (const [u, v] of wire) {
      uMin = Math.min(uMin, u);
      uMax = Math.max(uMax, u);
      vMin = Math.min(vMin, v);
      vMax = Math.max(vMax, v);
    }
  }

  if (!Number.isFinite(uMin) || !Number.isFinite(uMax) || !Number.isFinite(vMin) || !Number.isFinite(vMax)) {
    return null;
  }
  if (uMax - uMin <= 1e-6 || vMax - vMin <= 1e-6) {
    return null;
  }

  const shiftK = chooseShiftToRange(uMin, uMax, period, 0, period);
  if (shiftK !== 0) {
    const shiftU = shiftK * period;
    alignedWires = alignedWires.map((wire) => shiftLoopU(wire, shiftU));
    uMin += shiftU;
    uMax += shiftU;
  }

  const padFactor = Math.max(0, readGlobalNumber('__OCCT_INSPIRED_TRIM_DOMAIN_PAD_FACTOR__') ?? 0.03);
  const minPad = Math.max(1e-6, readGlobalNumber('__OCCT_INSPIRED_TRIM_DOMAIN_PAD_MIN__') ?? 1e-4);
  const padU = Math.max(minPad, (uMax - uMin) * padFactor);
  const padV = Math.max(minPad, (vMax - vMin) * padFactor);

  const sourcePointCount = alignedWires.reduce((sum, wire) => sum + wire.length, 0);
  const hasTrimHoles = face.innerLoops.length > 0;
  const gridScale = Math.max(0.1, readGlobalNumber('__OCCT_INSPIRED_TRIM_GRID_SCALE__') ?? 0.9);
  let minGrid = Math.max(8, Math.floor(readGlobalNumber('__OCCT_INSPIRED_TRIM_MIN_GRID__') ?? 14));
  let maxGrid = Math.max(minGrid, Math.floor(readGlobalNumber('__OCCT_INSPIRED_TRIM_MAX_GRID__') ?? 32));
  if (!hasTrimHoles) {
    const minGridNoHoles = Math.max(
      minGrid,
      Math.floor(readGlobalNumber('__OCCT_INSPIRED_TRIM_MIN_GRID_NO_HOLES__') ?? 26)
    );
    const maxGridNoHoles = Math.max(
      minGridNoHoles,
      Math.floor(readGlobalNumber('__OCCT_INSPIRED_TRIM_MAX_GRID_NO_HOLES__') ?? 48)
    );
    minGrid = minGridNoHoles;
    maxGrid = Math.max(maxGrid, maxGridNoHoles);
  }
  const gridDensity = Math.max(
    minGrid,
    Math.min(maxGrid, Math.ceil(Math.sqrt(sourcePointCount) * gridScale))
  );

  return {
    uvOuter: [
      [uMin - padU, vMin - padV],
      [uMax + padU, vMin - padV],
      [uMax + padU, vMax + padV],
      [uMin - padU, vMax + padV],
    ],
    gridDensity,
    sourceLoopCount: alignedWires.length,
    sourcePointCount,
    uMin: uMin - padU,
    uMax: uMax + padU,
    vMin: vMin - padV,
    vMax: vMax + padV,
  };
}

function getFaceTrimLoopsUVFromPCurves(
  oc: any,
  face: FaceWithEdgesInfo
): { uvOuter: Vec2[]; uvHoles: Vec2[][] } | null {
  const pcurveLoops = getFaceTrimWireLoopsUVFromPCurves(oc, face);
  if (!pcurveLoops || pcurveLoops.wires.length === 0) {
    return null;
  }
  const { wires, wireHashes, outerWireHash } = pcurveLoops;

  // Cone seam faces are sensitive to wire ordering/hash mismatches.
  // Build loop labels from p-curve geometry directly:
  // - unwrap seam jumps once
  // - choose the largest loop as outer
  // - only accept holes that are truly contained by that outer loop
  // Any inconsistency falls back to the projection path for this face.
  if (face.surfaceType === 'Cone') {
    const minAreaAbs = 1e-7;
    const period = Math.PI * 2;
    const classifiedWires = wires.map((wire) => unwrapClosedPeriodicLoopUOnce(wire, period));

    const areasAbs = classifiedWires.map((loop) => loopAreaAbs2D(loop));

    let outerIdx = -1;
    let bestAreaAbs = -Infinity;
    for (let i = 0; i < classifiedWires.length; i++) {
      if (classifiedWires[i].length < 3 || areasAbs[i] <= minAreaAbs) continue;
      if (areasAbs[i] > bestAreaAbs) {
        bestAreaAbs = areasAbs[i];
        outerIdx = i;
      }
    }

    if (outerIdx < 0 || bestAreaAbs <= minAreaAbs) {
      console.warn(`[pcurve-loop-label] face ${face.faceIndex} Cone: invalid outer by area, fallback to projection`);
      return null;
    }

    let outerClassified = classifiedWires[outerIdx];
    const outerBounds = getLoopUBounds(outerClassified);
    const outerShiftK = chooseShiftToRange(outerBounds.uMin, outerBounds.uMax, period, 0, period);
    if (outerShiftK !== 0) {
      outerClassified = shiftLoopU(outerClassified, outerShiftK * period);
    }
    const outerMidU = meanLoopU(outerClassified);

    const holeIdx: number[] = [];
    const alignedHolesForValidation: Vec2[][] = [];
    let totalHoleAreaAbs = 0;
    for (let i = 0; i < classifiedWires.length; i++) {
      if (i === outerIdx || classifiedWires[i].length < 3 || areasAbs[i] <= minAreaAbs) continue;
      let aligned = classifiedWires[i];
      if (outerShiftK !== 0) {
        aligned = shiftLoopU(aligned, outerShiftK * period);
      }
      const alignK = Math.round((outerMidU - meanLoopU(aligned)) / period);
      if (alignK !== 0) {
        aligned = shiftLoopU(aligned, alignK * period);
      }
      if (!isLoopInsidePolygonInclusive2D(aligned, outerClassified)) {
        console.warn(
          `[pcurve-loop-label] face ${face.faceIndex} Cone: loop ${i} not inside selected outer ${outerIdx}, fallback to projection`
        );
        return null;
      }
      if (areasAbs[i] >= bestAreaAbs * 0.98) {
        console.warn(
          `[pcurve-loop-label] face ${face.faceIndex} Cone: hole too large ` +
          `(hole=${areasAbs[i].toExponential(3)}, outer=${bestAreaAbs.toExponential(3)}), fallback to projection`
        );
        return null;
      }
      holeIdx.push(i);
      alignedHolesForValidation.push(aligned);
      totalHoleAreaAbs += areasAbs[i];
    }

    if (totalHoleAreaAbs >= bestAreaAbs * 0.98) {
      console.warn(
        `[pcurve-loop-label] face ${face.faceIndex} Cone: hole area sum too large ` +
        `(holes=${totalHoleAreaAbs.toExponential(3)}, outer=${bestAreaAbs.toExponential(3)}), fallback to projection`
      );
      return null;
    }

    const sanity = validateAndSanitizeTrimLoops(outerClassified, alignedHolesForValidation, {
      minAreaAbs,
      maxHoleToOuterRatio: 0.98,
      failOnHoleOutside: true,
      failOnHugeHole: true,
    });
    if (!sanity.ok) {
      console.warn(
        `[pcurve-loop-label] face ${face.faceIndex} Cone: invalid classified loops (${sanity.reason ?? 'unknown'}), fallback to projection`
      );
      return null;
    }

    const uvOuter = wires[outerIdx];
    const uvHoles = holeIdx.map((idx) => wires[idx]);
    curveDebugLog(
      `[pcurve-loop-label] face ${face.faceIndex} Cone: outerIdx=${outerIdx}, ` +
      `outerPts=${uvOuter.length}, holes=${uvHoles.length}, ` +
      `outerArea=${bestAreaAbs.toExponential(3)}, holeArea=${totalHoleAreaAbs.toExponential(3)}`
    );
    return { uvOuter, uvHoles };
  }

  let outerIdx = 0;
  if (outerWireHash !== null) {
    const matchIdx = wireHashes.findIndex((h) => h === outerWireHash);
    if (matchIdx >= 0) {
      outerIdx = matchIdx;
    }
  } else {
    let bestPerimeter = -Infinity;
    for (let i = 0; i < wires.length; i++) {
      const perimeter = loopPerimeterUV(wires[i]);
      if (perimeter > bestPerimeter) {
        bestPerimeter = perimeter;
        outerIdx = i;
      }
    }
  }

  const uvOuter = wires[outerIdx];
  const uvHoles = wires.filter((_, i) => i !== outerIdx && wires[i].length >= 3);
  if (!uvOuter || uvOuter.length < 3) {
    return null;
  }
  return { uvOuter, uvHoles };
}

function getFaceTrimLoopsUV(
  oc: any,
  face: FaceWithEdgesInfo
): TrimLoopsUV | null {
  if (!face.occSurface) return null;

  const isUPeriodic = ['Cylinder', 'Sphere', 'Cone', 'Torus'].includes(face.surfaceType);
  const isVPeriodic = face.surfaceType === 'Torus';

  const normalizeAndFinalize = (
    source: 'pcurve' | 'projection',
    uvOuter: Vec2[],
    uvHoles: Vec2[][]
  ): TrimLoopsUV | null => {
    const rawOuterWrapped = simplifyLoop2D(uvOuter);
    let normalizedOuter = uvOuter;
    let normalizedHoles = uvHoles;
    const shouldNormalizePeriodicTrimLoops = face.surfaceType === 'Cone' || face.surfaceType === 'Torus';
    if (shouldNormalizePeriodicTrimLoops && normalizedOuter.length >= 3) {
      const beforeUBounds = getLoopComponentBounds(normalizedOuter, 0);
      const beforeVBounds = getLoopComponentBounds(normalizedOuter, 1);
      const normalized = normalizePeriodicTrimLoops(normalizedOuter, normalizedHoles, {
        periodicU: isUPeriodic,
        periodicV: isVPeriodic,
      });
      normalizedOuter = normalized.uvOuter;
      normalizedHoles = normalized.uvHoles;
      const afterUBounds = getLoopComponentBounds(normalizedOuter, 0);
      const afterVBounds = getLoopComponentBounds(normalizedOuter, 1);
      curveDebugLog(
        `[trim-loop-normalize] face ${face.faceIndex} ${face.surfaceType} source=${source}: ` +
        `U [${beforeUBounds.min.toFixed(3)}, ${beforeUBounds.max.toFixed(3)}] -> ` +
        `[${afterUBounds.min.toFixed(3)}, ${afterUBounds.max.toFixed(3)}], ` +
        `V [${beforeVBounds.min.toFixed(3)}, ${beforeVBounds.max.toFixed(3)}] -> ` +
        `[${afterVBounds.min.toFixed(3)}, ${afterVBounds.max.toFixed(3)}]`
      );
    }

    // Ensure closed loops for robust polygon tests; duplicate endpoint is removed by simplifyLoop2D.
    normalizedOuter = simplifyLoop2D(normalizedOuter);
    normalizedHoles = normalizedHoles.map((h) => simplifyLoop2D(h));

    // Perf-mode: apply conservative trim-loop simplification across all curved
    // surfaces before grid selection/CDT. This is benchmark-oriented and off
    // unless geometry-only perf mode is enabled.
    const preferGeometryOnlyLoad = readGlobalBoolean('__PERF_GEOMETRY_ONLY_LOAD__', false);
    const enablePerfTrimSimplify = readGlobalBoolean('__PERF_TRIM_SIMPLIFY_LOOPS__', true);
    if (preferGeometryOnlyLoad && enablePerfTrimSimplify) {
      const maxAreaErrorRatio = Math.max(1e-4, readGlobalNumber('__PERF_TRIM_MAX_AREA_ERR_RATIO__') ?? 0.02);
      const maxOuterAreaErrorNoHoles = Math.max(
        maxAreaErrorRatio,
        readGlobalNumber('__PERF_TRIM_MAX_OUTER_AREA_ERR_RATIO_NO_HOLES__') ?? 0.05
      );
      const defaultOuterCap = normalizedHoles.length === 0 ? 96 : 128;
      const maxOuterPts = Math.max(24, Math.floor(readGlobalNumber('__PERF_TRIM_MAX_OUTER_PTS__') ?? defaultOuterCap));
      const maxHolePts = Math.max(8, Math.floor(readGlobalNumber('__PERF_TRIM_MAX_HOLE_PTS__') ?? 20));
      const beforeOuterPts = normalizedOuter.length;
      const beforeHolePts = normalizedHoles.reduce((sum, h) => sum + h.length, 0);

      const outerAreaErrRatio = normalizedHoles.length === 0 ? maxOuterAreaErrorNoHoles : maxAreaErrorRatio;
      const simplifiedOuter = simplifyLoopForMeshing(normalizedOuter, maxOuterPts, outerAreaErrRatio);
      const simplifiedHoles = normalizedHoles
        .map((hole) => simplifyLoopForMeshing(hole, maxHolePts, maxAreaErrorRatio))
        .filter((hole) => hole.length >= 3);

      const simplifiedValidation = validateAndSanitizeTrimLoops(simplifiedOuter, simplifiedHoles, {
        minAreaAbs: 1e-7,
        maxHoleToOuterRatio: 0.98,
        failOnHoleOutside: true,
        failOnHugeHole: true,
      });

      if (simplifiedValidation.ok) {
        normalizedOuter = simplifiedValidation.uvOuter;
        normalizedHoles = simplifiedValidation.uvHoles;
      } else {
        curveDebugLog(
          `[perf-trim-simplify] face ${face.faceIndex} ${face.surfaceType} source=${source}: skipped ` +
          `(${simplifiedValidation.reason ?? 'unknown'})`
        );
      }

      const afterHolePts = normalizedHoles.reduce((sum, h) => sum + h.length, 0);
      if (beforeOuterPts !== normalizedOuter.length || beforeHolePts !== afterHolePts) {
        curveDebugLog(
          `[perf-trim-simplify] face ${face.faceIndex} ${face.surfaceType} source=${source}: ` +
          `outer ${beforeOuterPts} -> ${normalizedOuter.length}, holes ${beforeHolePts} -> ${afterHolePts} ` +
          `(count=${normalizedHoles.length})`
        );
      }
    }

    // OCCT-inspired stability: keep trim loops geometrically equivalent but bounded in
    // complexity before triangulation. This avoids pathological CDT inputs on cone seams.
    if (face.surfaceType === 'Cone') {
      const maxOuterPts = Math.max(64, Math.floor(readGlobalNumber('__CONE_TRIM_MAX_OUTER_PTS__') ?? 192));
      const maxHolePts = Math.max(12, Math.floor(readGlobalNumber('__CONE_TRIM_MAX_HOLE_PTS__') ?? 32));
      const maxAreaErrorRatio = Math.max(1e-4, readGlobalNumber('__CONE_TRIM_MAX_AREA_ERR_RATIO__') ?? 0.03);
      const maxOuterAreaErrorNoHoles = Math.max(
        maxAreaErrorRatio,
        readGlobalNumber('__CONE_TRIM_MAX_AREA_ERR_RATIO_NO_HOLES__') ?? 0.08
      );
      const beforeOuterPts = normalizedOuter.length;
      const beforeHolePts = normalizedHoles.reduce((sum, h) => sum + h.length, 0);

      const outerAreaErrRatio = normalizedHoles.length === 0 ? maxOuterAreaErrorNoHoles : maxAreaErrorRatio;
      const simplifiedOuter = simplifyLoopForMeshing(normalizedOuter, maxOuterPts, outerAreaErrRatio);
      const simplifiedHoles = normalizedHoles
        .map((hole) => simplifyLoopForMeshing(hole, maxHolePts, maxAreaErrorRatio))
        .filter((hole) => hole.length >= 3);

      const simplifiedValidation = validateAndSanitizeTrimLoops(simplifiedOuter, simplifiedHoles, {
        minAreaAbs: 1e-7,
        maxHoleToOuterRatio: 0.98,
        failOnHoleOutside: true,
        failOnHugeHole: true,
      });

      if (simplifiedValidation.ok) {
        normalizedOuter = simplifiedValidation.uvOuter;
        normalizedHoles = simplifiedValidation.uvHoles;
      } else if (source === 'pcurve') {
        console.warn(
          `[trim-loop-simplify] face ${face.faceIndex} Cone source=${source}: invalid simplified loops ` +
          `(${simplifiedValidation.reason ?? 'unknown'}), fallback to projection`
        );
        return null;
      } else {
        console.warn(
          `[trim-loop-simplify] face ${face.faceIndex} Cone source=${source}: invalid simplified loops ` +
          `(${simplifiedValidation.reason ?? 'unknown'}), keeping unsimplified projection loops`
        );
      }

      const afterHolePts = normalizedHoles.reduce((sum, h) => sum + h.length, 0);
      if (beforeOuterPts !== normalizedOuter.length || beforeHolePts !== afterHolePts) {
        curveDebugLog(
          `[trim-loop-simplify] face ${face.faceIndex} Cone source=${source}: ` +
          `outer ${beforeOuterPts} -> ${normalizedOuter.length}, holes ${beforeHolePts} -> ${afterHolePts} ` +
          `(count=${normalizedHoles.length})`
        );
      }
    }

    if (normalizedOuter.length < 3) return null;
    return { uvOuter: normalizedOuter, uvHoles: normalizedHoles, uvOuterRawWrapped: rawOuterWrapped };
  };

  const pcurveLoops = getFaceTrimLoopsUVFromPCurves(oc, face);
  if (pcurveLoops && pcurveLoops.uvOuter.length >= 3) {
    const finalized = normalizeAndFinalize('pcurve', pcurveLoops.uvOuter, pcurveLoops.uvHoles);
    if (finalized) {
      curveDebugLog(`[trim-loop-source] face ${face.faceIndex} ${face.surfaceType}: source=pcurve`);
      return finalized;
    }
  }

  const sa = new oc.ShapeAnalysis_Surface(face.occSurface);
  const lineSubdivideStep = Math.max(0.25, readGlobalNumber('__CURVED_UV_LINE_SUBDIVIDE_STEP__') ?? 2.0);
  const outer3d = occEdgesToPolygon(face.outerLoop, undefined, { lineSubdivideStep });
  if (outer3d.length < 3) {
    sa.delete?.();
    return null;
  }

  // Debug: show outer3d for cone and torus
  if (face.surfaceType === 'Torus' || face.surfaceType === 'Cone') {
    curveDebugLog(`[getFaceTrimLoopsUV] ${face.surfaceType} outer3d has ${outer3d.length} points`);
    if (outer3d.length > 0) {
      curveDebugLog(`[getFaceTrimLoopsUV] First 5 points: ${outer3d.slice(0, 5).map(p => `(${p[0].toFixed(3)}, ${p[1].toFixed(3)}, ${p[2].toFixed(3)})`).join(', ')}`);
    }
    curveDebugLog(`[getFaceTrimLoopsUV] outerLoop has ${face.outerLoop.length} edges`);
    face.outerLoop.forEach((e, i) => {
      curveDebugLog(`[getFaceTrimLoopsUV] Edge ${i}: curveType=${e.curveType}, sampledPoints=${e.sampledPoints?.length || 0}`);
    });
  }

  const uvOuter = projectPointsToUV(oc, sa, outer3d, { wrapU: isUPeriodic, wrapV: isVPeriodic });

  // Debug: show uvOuter for cone and torus
  if (face.surfaceType === 'Torus' || face.surfaceType === 'Cone') {
    curveDebugLog(`[getFaceTrimLoopsUV] ${face.surfaceType} uvOuter has ${uvOuter.length} points after projection`);
    if (uvOuter.length > 0) {
      curveDebugLog(`[getFaceTrimLoopsUV] UV First 10: ${uvOuter.slice(0, 10).map(p => `(${p[0].toFixed(3)}, ${p[1].toFixed(3)})`).join(', ')}`);
      // Also show middle and end
      const mid = Math.floor(uvOuter.length / 2);
      curveDebugLog(`[getFaceTrimLoopsUV] UV Middle 10 (${mid}): ${uvOuter.slice(mid, mid+10).map(p => `(${p[0].toFixed(3)}, ${p[1].toFixed(3)})`).join(', ')}`);
      curveDebugLog(`[getFaceTrimLoopsUV] UV Last 10: ${uvOuter.slice(-10).map(p => `(${p[0].toFixed(3)}, ${p[1].toFixed(3)})`).join(', ')}`);
    }
  }
  // Debug: log inner loop edge info for cylinders
  if (face.surfaceType === 'Cylinder' && face.innerLoops.length > 0) {
    curveDebugLog(`[getFaceTrimLoopsUV] Cylinder has ${face.innerLoops.length} inner loops`);
    face.innerLoops.forEach((loop, loopIdx) => {
      curveDebugLog(`[getFaceTrimLoopsUV] Inner loop ${loopIdx}: ${loop.length} edges`);
      loop.forEach((e, edgeIdx) => {
        curveDebugLog(`  Edge ${edgeIdx}: curveType=${e.curveType}, sampledPoints=${e.sampledPoints?.length || 0}, start=(${e.startPoint.x.toFixed(3)},${e.startPoint.y.toFixed(3)},${e.startPoint.z.toFixed(3)}), end=(${e.endPoint.x.toFixed(3)},${e.endPoint.y.toFixed(3)},${e.endPoint.z.toFixed(3)})`);
      });
    });
  }

  const uvHoles = face.innerLoops
    .map((loop, loopIdx) => {
      const poly3d = occEdgesToPolygon(loop, undefined, { lineSubdivideStep });
      if (face.surfaceType === 'Cylinder') {
        curveDebugLog(`[getFaceTrimLoopsUV] Inner loop ${loopIdx} -> 3D polygon: ${poly3d.length} points`);
        if (poly3d.length > 0) {
          curveDebugLog(`  First 5 3D: ${poly3d.slice(0, 5).map((p, i) => `[${i}](${p[0].toFixed(2)},${p[1].toFixed(2)},${p[2].toFixed(2)})`).join(' ')}`);
        }
      }
      return poly3d;
    })
    .filter((loop3d) => loop3d.length >= 3)
    .map((loop3d) => projectPointsToUV(oc, sa, loop3d, { wrapU: isUPeriodic, wrapV: isVPeriodic }))
    .filter((loop2d) => loop2d.length >= 3);

  const finalized = normalizeAndFinalize('projection', uvOuter, uvHoles);
  sa.delete?.();
  return finalized;
}

function chooseTrimGridDensity(face: FaceWithEdgesInfo, uvOuter: Vec2[], uvHoles: Vec2[][]): number {
  const isPeriodicSeamSensitiveTrim = (): boolean => {
    if (uvOuter.length < 3) return false;
    const period = Math.PI * 2;
    const seamJumpThreshold = period * 0.75;
    let maxJump = 0;
    for (let i = 0; i < uvOuter.length; i++) {
      const u1 = uvOuter[i][0];
      const u2 = uvOuter[(i + 1) % uvOuter.length][0];
      const jump = Math.abs(u2 - u1);
      if (jump > maxJump) maxJump = jump;
    }
    const nearPosPI = uvOuter.filter((p) => p[0] > Math.PI - 0.3).length;
    const nearNegPI = uvOuter.filter((p) => p[0] < -Math.PI + 0.3).length;
    const uBounds = getLoopUBounds(uvOuter);
    const uSpan = uBounds.uMax - uBounds.uMin;
    const crossesByCounts = nearPosPI >= 2 && nearNegPI >= 2;
    const crossesByJump = maxJump > seamJumpThreshold && uSpan > Math.PI;
    return crossesByCounts || crossesByJump;
  };

  const totalPts = uvHoles.reduce((acc, h) => acc + h.length, uvOuter.length);
  // Keep trim-grid growth sublinear and capped by default. This is a key speed
  // control for complex trims where boundary sampling can be very dense.
  const trimGridScale = Math.max(0.25, readGlobalNumber('__TRIM_GRID_SCALE__') ?? 0.85);
  const trimDensityBias = Math.max(0.5, readGlobalNumber('__TRIM_DENSITY_BIAS__') ?? 1.0);
  let base = Math.ceil(Math.sqrt(totalPts) * trimGridScale);
  base = Math.ceil(base * trimDensityBias);

  const minGrid = Math.max(8, Math.floor(readGlobalNumber('__TRIM_MIN_GRID_DENSITY__') ?? 10));
  let effectiveMinGrid = minGrid;
  const maxGridGlobal = Math.max(minGrid, Math.floor(readGlobalNumber('__TRIM_MAX_GRID_DENSITY__') ?? 32));
  const maxGridNoHoles = Math.max(minGrid, Math.floor(readGlobalNumber('__TRIM_MAX_GRID_DENSITY_NO_HOLES__') ?? 20));
  const maxGridWithHoles = Math.max(minGrid, Math.floor(readGlobalNumber('__TRIM_MAX_GRID_DENSITY_WITH_HOLES__') ?? 24));
  let effectiveMaxGrid = uvHoles.length === 0 ? maxGridNoHoles : maxGridWithHoles;

  // OCCT-inspired guardrail: trim mesh density should not explode just because
  // trim loops are densely sampled. Cone seam faces are especially sensitive.
  if (face.surfaceType === 'Cone') {
    const coneMaxGrid = Math.max(14, Math.floor(readGlobalNumber('__CONE_TRIM_MAX_GRID_DENSITY__') ?? 24));
    const coneMaxGridNoHoles = Math.max(10, Math.floor(readGlobalNumber('__CONE_TRIM_MAX_GRID_DENSITY_NO_HOLES__') ?? 14));
    const outerPointSoftCap = Math.max(128, Math.floor(readGlobalNumber('__CONE_TRIM_OUTER_POINT_SOFT_CAP__') ?? 512));

    if (uvOuter.length > outerPointSoftCap || uvHoles.length > 0) {
      effectiveMaxGrid = Math.min(effectiveMaxGrid, coneMaxGrid);
    }
    if (uvHoles.length === 0) {
      effectiveMaxGrid = Math.min(effectiveMaxGrid, coneMaxGridNoHoles);
    }
  }

  // Additional throttle for highly sampled trims to curb triangle inflation.
  const highComplexPointThreshold = Math.max(256, Math.floor(readGlobalNumber('__TRIM_HIGH_COMPLEXITY_POINT_THRESHOLD__') ?? 600));
  if (totalPts > highComplexPointThreshold) {
    const complexityGridCap = Math.max(minGrid, Math.floor(effectiveMaxGrid * 0.75));
    effectiveMaxGrid = Math.min(effectiveMaxGrid, complexityGridCap);
  }

  // Perf benchmark mode: tighten trim density budgets so complex no-hole curved
  // faces do not dominate runtime via over-sampled grids.
  const preferGeometryOnlyLoad = readGlobalBoolean('__PERF_GEOMETRY_ONLY_LOAD__', false);
  if (preferGeometryOnlyLoad) {
    const perfMinGrid = Math.max(4, Math.floor(readGlobalNumber('__PERF_TRIM_MIN_GRID_DENSITY__') ?? 4));
    const perfGridScaleMult = Math.max(0.4, readGlobalNumber('__PERF_TRIM_GRID_SCALE_MULT__') ?? 0.65);
    const perfNoHolesMax = Math.max(perfMinGrid, Math.floor(readGlobalNumber('__PERF_TRIM_MAX_GRID_DENSITY_NO_HOLES__') ?? 8));
    const perfWithHolesMax = Math.max(perfMinGrid, Math.floor(readGlobalNumber('__PERF_TRIM_MAX_GRID_DENSITY_WITH_HOLES__') ?? 12));

    effectiveMinGrid = Math.min(effectiveMinGrid, perfMinGrid);
    base = Math.ceil(base * perfGridScaleMult);
    effectiveMaxGrid = Math.min(effectiveMaxGrid, uvHoles.length === 0 ? perfNoHolesMax : perfWithHolesMax);

    // Dense trim loops can still over-sample UV grids after regular caps.
    // Apply an additional complexity-aware downscale only in perf mode.
    const perfHighComplexPts = Math.max(
      256,
      Math.floor(readGlobalNumber('__PERF_TRIM_HIGH_COMPLEXITY_POINT_THRESHOLD__') ?? 700)
    );
    if (totalPts > perfHighComplexPts) {
      const perfScale = Math.sqrt(perfHighComplexPts / totalPts);
      const perfHighNoHolesMax = Math.max(
        perfMinGrid,
        Math.floor(readGlobalNumber('__PERF_TRIM_HIGH_COMPLEXITY_NO_HOLES_MAX_GRID__') ?? 6)
      );
      const perfHighWithHolesMax = Math.max(
        perfMinGrid,
        Math.floor(readGlobalNumber('__PERF_TRIM_HIGH_COMPLEXITY_WITH_HOLES_MAX_GRID__') ?? 10)
      );
      base = Math.max(effectiveMinGrid, Math.ceil(base * perfScale));
      effectiveMaxGrid = Math.min(
        effectiveMaxGrid,
        uvHoles.length === 0 ? perfHighNoHolesMax : perfHighWithHolesMax
      );
    }
  }

  return Math.max(effectiveMinGrid, Math.min(effectiveMaxGrid, base));
  // Keep seam-sensitive periodic trims away from under-sampled fold artifacts.
  // This applies to cone/cylinder trims that cross the U seam and is intentionally
  // scoped so we do not globally increase cost.
  if ((face.surfaceType === 'Cone' || face.surfaceType === 'Cylinder') && isPeriodicSeamSensitiveTrim()) {
    const seamMinGrid = Math.max(
      effectiveMinGrid,
      Math.floor(
        readGlobalNumber(
          face.surfaceType === 'Cone'
            ? '__CONE_PERIODIC_SEAM_MIN_GRID_DENSITY__'
            : '__CYLINDER_PERIODIC_SEAM_MIN_GRID_DENSITY__'
        ) ?? (face.surfaceType === 'Cone' ? 18 : 22)
      )
    );
    base = Math.max(base, seamMinGrid);
    effectiveMaxGrid = Math.max(effectiveMaxGrid, seamMinGrid);
  }

  return Math.max(effectiveMinGrid, Math.min(effectiveMaxGrid, base));
}

function estimateFaceSizeFromLoops(face: FaceWithEdgesInfo): number {
  const points: Vec3[] = [];
  for (const edge of face.outerLoop) {
    points.push([edge.startPoint.x, edge.startPoint.y, edge.startPoint.z]);
    points.push([edge.endPoint.x, edge.endPoint.y, edge.endPoint.z]);
  }
  for (const loop of face.innerLoops) {
    for (const edge of loop) {
      points.push([edge.startPoint.x, edge.startPoint.y, edge.startPoint.z]);
      points.push([edge.endPoint.x, edge.endPoint.y, edge.endPoint.z]);
    }
  }
  if (points.length === 0) return 1.0;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const [x, y, z] of points) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  const avgSize = ((maxX - minX) + (maxY - minY) + (maxZ - minZ)) / 3;
  return Number.isFinite(avgSize) && avgSize > 0 ? avgSize : 1.0;
}

function unwrapPolyTriangulation(rawTriangulation: any): any | null {
  if (!rawTriangulation) return null;
  try {
    if (typeof rawTriangulation.IsNull === 'function' && rawTriangulation.IsNull()) {
      return null;
    }
  } catch {
    // Ignore IsNull failures and continue with best-effort unwrapping.
  }
  if (typeof rawTriangulation.get === 'function') {
    try {
      const unwrapped = rawTriangulation.get();
      if (unwrapped && unwrapped !== rawTriangulation) {
        return unwrapPolyTriangulation(unwrapped);
      }
    } catch {
      // Ignore get() failures and continue with raw object.
    }
  }
  if (
    typeof rawTriangulation.NbNodes === 'function' &&
    typeof rawTriangulation.NbTriangles === 'function' &&
    typeof rawTriangulation.Node === 'function' &&
    typeof rawTriangulation.Triangle === 'function'
  ) {
    return rawTriangulation;
  }
  return null;
}

async function tessellateFaceFromOcctTriangulation(face: FaceWithEdgesInfo): Promise<{
  vertices: Vec3[];
  triangles: number[][];
} | null> {
  if (!face.occFace) return null;
  const oc = await initOC();

  const linDeflectionRatio = readGlobalNumber('__OCCT_NATIVE_LIN_DEFLECTION_RATIO__') ?? 0.001;
  const explicitLinDeflection = readGlobalNumber('__OCCT_NATIVE_LIN_DEFLECTION__');
  const angDeflection = readGlobalNumber('__OCCT_NATIVE_ANG_DEFLECTION__') ?? 0.5;
  const isRelative = readGlobalBoolean('__OCCT_NATIVE_RELATIVE__', false);
  const isInParallel = readGlobalBoolean('__OCCT_NATIVE_PARALLEL__', false);

  const fallbackFaceSize = estimateFaceSizeFromLoops(face);
  const linDeflection = explicitLinDeflection ?? Math.max(1e-6, fallbackFaceSize * linDeflectionRatio);

  let location: any = null;
  const newLocation = () => {
    if (oc.TopLoc_Location_1) return new oc.TopLoc_Location_1();
    if (oc.TopLoc_Location) return new oc.TopLoc_Location();
    return null;
  };
  const readTriangulation = () => {
    location?.delete?.();
    location = newLocation();
    if (!oc.BRep_Tool?.Triangulation || !location) {
      return null;
    }
    const rawTri = oc.BRep_Tool.Triangulation(face.occFace, location);
    return unwrapPolyTriangulation(rawTri);
  };

  let triangulation = readTriangulation();
  if (!triangulation || triangulation.NbNodes() === 0 || triangulation.NbTriangles() === 0) {
    if (oc.BRepMesh_IncrementalMesh_2) {
      try {
        const mesher = new oc.BRepMesh_IncrementalMesh_2(
          face.occFace,
          linDeflection,
          isRelative,
          angDeflection,
          isInParallel
        );
        mesher.delete?.();
      } catch (meshErr) {
        console.warn(
          `[occt-native] face ${face.faceIndex}: BRepMesh_IncrementalMesh_2 failed`,
          meshErr
        );
      }
    } else if (oc.BRepTools?.Triangulation) {
      try {
        oc.BRepTools.Triangulation(face.occFace, linDeflection, false);
      } catch (meshErr) {
        console.warn(
          `[occt-native] face ${face.faceIndex}: BRepTools.Triangulation failed`,
          meshErr
        );
      }
    }
    triangulation = readTriangulation();
  }

  if (!triangulation) {
    location?.delete?.();
    return null;
  }

  const nodeCount = triangulation.NbNodes();
  const triCount = triangulation.NbTriangles();
  if (nodeCount <= 0 || triCount <= 0) {
    location?.delete?.();
    return null;
  }

  let locationTransform: any = null;
  try {
    if (location && typeof location.IsIdentity === 'function' && !location.IsIdentity()) {
      locationTransform = location.Transformation?.();
    }
  } catch {
    locationTransform = null;
  }

  const vertices: Vec3[] = [];
  for (let i = 1; i <= nodeCount; i++) {
    const node = triangulation.Node(i);
    const worldNode =
      locationTransform && typeof node.Transformed === 'function'
        ? node.Transformed(locationTransform)
        : node;
    vertices.push([worldNode.X(), worldNode.Y(), worldNode.Z()]);
    if (worldNode !== node) {
      worldNode.delete?.();
    }
  }

  const triangles: number[][] = [];
  for (let i = 1; i <= triCount; i++) {
    const tri = triangulation.Triangle(i);
    const v0 = tri.Value(1);
    const v1 = tri.Value(2);
    const v2 = tri.Value(3);
    if (
      Number.isInteger(v0) && Number.isInteger(v1) && Number.isInteger(v2) &&
      v0 > 0 && v1 > 0 && v2 > 0 &&
      v0 <= nodeCount && v1 <= nodeCount && v2 <= nodeCount
    ) {
      triangles.push([v0 - 1, v1 - 1, v2 - 1]);
    }
  }

  locationTransform?.delete?.();
  location?.delete?.();
  return triangles.length > 0 ? { vertices, triangles } : null;
}

/**
 * Tessellate a curved surface face using existing surface-tessellation functions
 */
async function tessellateCurvedFaceFromOCC(face: FaceWithEdgesInfo): Promise<{
  vertices: Vec3[];
  triangles: number[][];
}> {
  const faceStart = performance.now();
  const STRICT_CLASSIFIER_FACE_IDS = readFaceIdsFromGlobal('__STRICT_CLASSIFIER_FACE_IDS__', [14, 63, 64, 65, 66, 994]);
  const PERIODIC_PROOF_FACE_IDS = readFaceIdsFromGlobal('__PERIODIC_PROOF_FACE_IDS__', []);
  // Default-on for the OCCT-inspired cone path. Face filters remain opt-in:
  // when no FACE_IDS global is provided, all cone faces are eligible.
  const enableConeSeamSplit = readGlobalBoolean('__ENABLE_CONE_SEAM_SPLIT__', true);
  const coneSeamSplitFaceIdsRaw = (globalThis as any)?.__CONE_SEAM_SPLIT_FACE_IDS__;
  const coneSeamSplitFaceIds = readFaceIdsFromGlobal('__CONE_SEAM_SPLIT_FACE_IDS__', []);
  // Default-on for cylinder seam split (trimmed seam-crossing cylinders only).
  const enableCylinderSeamSplit = readGlobalBoolean('__ENABLE_CYLINDER_SEAM_SPLIT__', true);
  const cylinderSeamSplitFaceIdsRaw = (globalThis as any)?.__CYLINDER_SEAM_SPLIT_FACE_IDS__;
  const cylinderSeamSplitFaceIds = readFaceIdsFromGlobal('__CYLINDER_SEAM_SPLIT_FACE_IDS__', []);
  const enableOcctInspiredTrimGraph = readGlobalBoolean('__ENABLE_OCCT_INSPIRED_TRIM_GRAPH__', true);
  const occtInspiredTrimGraphFaceIdsRaw = (globalThis as any)?.__OCCT_INSPIRED_TRIM_GRAPH_FACE_IDS__;
  const occtInspiredTrimGraphFaceIds = readFaceIdsFromGlobal('__OCCT_INSPIRED_TRIM_GRAPH_FACE_IDS__', []);
  const enableOcctNativeFaceTessellation = readGlobalBoolean('__ENABLE_OCCT_NATIVE_FACE_TESSELLATION__', false);
  const allowOcctOraclePath = readGlobalBoolean('__ALLOW_OCCT_ORACLE_PATH__', false);
  const occtNativeFaceIds = readFaceIdsFromGlobal('__OCCT_NATIVE_FACE_IDS__', [63, 64, 65, 66]);
  const shouldRunPeriodicProof = PERIODIC_PROOF_FACE_IDS.has(face.faceIndex);
  const isKnownLidFace = STRICT_CLASSIFIER_FACE_IDS.has(face.faceIndex);
  // Runtime debug controls (set in browser console):
  //   globalThis.__FACE_DEBUG_MODE__ = 'off' | 'skip' | 'only'
  //   globalThis.__FACE_DEBUG_IDS__ = [14,63,64,65,66,994] or "14,63,64,65,66,994"
  const FACE_DEBUG_IDS = readFaceIdsFromGlobal('__FACE_DEBUG_IDS__', [14, 63, 64, 65, 66, 994]);
  const FACE_DEBUG_MODE = readDebugModeFromGlobal('__FACE_DEBUG_MODE__', 'off');
  const isTargetFace = FACE_DEBUG_IDS.has(face.faceIndex);
  if (FACE_DEBUG_MODE === 'skip' && isTargetFace) {
    console.warn(`[FACE DEBUG] skip face=${face.faceIndex} type=${face.surfaceType}`);
    return { vertices: [], triangles: [] };
  }
  if (FACE_DEBUG_MODE === 'only' && !isTargetFace) {
    return { vertices: [], triangles: [] };
  }
  if (FACE_DEBUG_MODE === 'only' && isTargetFace) {
    console.warn(`[FACE DEBUG] keep-only face=${face.faceIndex} type=${face.surfaceType}`);
  }
  // Runtime sliver debug controls:
  //   globalThis.__SLIVER_DEBUG_MODE__ = 'off' | 'skip' | 'only'
  //   globalThis.__SLIVER_EPS__ = 0.005
  const SLIVER_EPS = readGlobalNumber('__SLIVER_EPS__') ?? 0.005;
  const SLIVER_DEBUG_MODE = readDebugModeFromGlobal('__SLIVER_DEBUG_MODE__', 'off');

  if (allowOcctOraclePath && enableOcctNativeFaceTessellation && occtNativeFaceIds.has(face.faceIndex)) {
    const occtNativeMesh = await tessellateFaceFromOcctTriangulation(face);
    if (occtNativeMesh && occtNativeMesh.triangles.length > 0) {
      console.log(
        `[occt-native] face ${face.faceIndex} ${face.surfaceType}: using OCC triangulation ` +
        `(${occtNativeMesh.vertices.length} verts, ${occtNativeMesh.triangles.length} tris)`
      );
      return occtNativeMesh;
    }
    console.warn(
      `[occt-native] face ${face.faceIndex} ${face.surfaceType}: no native triangulation, falling back`
    );
  }

  if (!face.surfaceParams) {
    console.warn(`[Tessellate] No surface params for ${face.surfaceType} face ${face.faceIndex}`);
    return { vertices: [], triangles: [] };
  }

  const params = face.surfaceParams;
  const isSeamSensitivePeriodicFace = face.surfaceType === 'Cylinder' || face.surfaceType === 'Cone' || face.surfaceType === 'Torus';
  let avoidFullSurfaceFallback = false;

  // Prefer UV-trimmed tessellation using the actual face boundary wires.
  // The previous approach tessellated the whole (u,v) bounds rectangle, which drops trim details.
  try {
    const oc = await initOC();

    const shouldTryOcctInspiredTrimGraph =
      face.surfaceType === 'Cone' &&
      enableOcctInspiredTrimGraph &&
      (occtInspiredTrimGraphFaceIdsRaw == null || occtInspiredTrimGraphFaceIds.has(face.faceIndex));
    if (
      shouldTryOcctInspiredTrimGraph &&
      face.occFace &&
      params.radius !== undefined &&
      params.semiAngle !== undefined &&
      params.placement
    ) {
      const domain = buildOcctInspiredConeTrimDomainFromPCurves(oc, face);
      if (!domain) {
        console.warn(
          `[occt-inspired-trim] face ${face.faceIndex} ${face.surfaceType}: ` +
          `domain build failed, falling back`
        );
      } else if (oc?.BRepTopAdaptor_FClass2d && oc?.gp_Pnt2d_3 && oc?.TopAbs_State) {
        let occBuildClassifier: any | undefined;
        try {
          const hasTrimHoles = face.innerLoops.length > 0;
          occBuildClassifier = new oc.BRepTopAdaptor_FClass2d(face.occFace, 1e-7);
          const useTriangleGate = hasTrimHoles || readGlobalBoolean('__OCCT_INSPIRED_TRIM_GATE_NO_HOLES__', false);
          const maxOutSamples = Math.max(
            0,
            Math.floor(
              readGlobalNumber('__OCCT_INSPIRED_TRIM_MAX_OUT_SAMPLES__') ??
              (hasTrimHoles ? 1 : 3)
            )
          );
          const allowPartialCellTriangles = readGlobalBoolean(
            '__OCCT_INSPIRED_TRIM_ALLOW_PARTIAL_CELL_TRIANGLES__',
            !hasTrimHoles
          );
          const classifyInside = (u: number, v: number): boolean => {
            const uvPoint = new oc.gp_Pnt2d_3(u, v);
            try {
              const state = occBuildClassifier.Perform(uvPoint, true);
              return !topAbsStateEquals(state, oc.TopAbs_State.TopAbs_OUT);
            } catch {
              return true;
            } finally {
              uvPoint.delete?.();
            }
          };
          const buildOptions: TrimmedSurfaceBuildOptions = {
            uvInsideTest: classifyInside,
            keepTriangle: useTriangleGate
              ? (samples) => {
                  let outCount = 0;
                  for (const [u, v] of samples) {
                    if (!classifyInside(u, v)) {
                      outCount++;
                      if (outCount > maxOutSamples) {
                        return false;
                      }
                    }
                  }
                  return true;
                }
              : undefined,
            allowPartialCellTriangles,
            logLabel: `occt-inspired-face-${face.faceIndex}`,
          };
          const coneSurface = {
            type: 'CONICAL_SURFACE' as const,
            placement: params.placement,
            radius: params.radius,
            semiAngle: params.semiAngle,
          };
          const mesh = await tessellateTrimmedSurface(
            coneSurface,
            domain.uvOuter,
            domain.gridDensity,
            [],
            undefined,
            buildOptions
          );
          if (mesh.indices.length > 0) {
            console.log(
              `[occt-inspired-trim] face ${face.faceIndex} ${face.surfaceType}: ` +
              `domain U=[${domain.uMin.toFixed(3)}, ${domain.uMax.toFixed(3)}] ` +
              `V=[${domain.vMin.toFixed(3)}, ${domain.vMax.toFixed(3)}], ` +
              `loops=${domain.sourceLoopCount}, loopPts=${domain.sourcePointCount}, ` +
              `grid=${domain.gridDensity}, verts=${mesh.positions.length / 3}, tris=${mesh.indices.length / 3}`
            );
            tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
            tessellationProfile.tessellateCurvedFace.calls++;
            return tessellatedMeshToVerticesAndTriangles(mesh);
          }
          console.warn(
            `[occt-inspired-trim] face ${face.faceIndex} ${face.surfaceType}: ` +
            `produced empty mesh, falling back`
          );
        } catch (e) {
          console.warn(
            `[occt-inspired-trim] face ${face.faceIndex} ${face.surfaceType}: ` +
            `build failed, falling back`,
            e
          );
        } finally {
          occBuildClassifier?.delete?.();
        }
      } else {
        console.warn(
          `[occt-inspired-trim] face ${face.faceIndex} ${face.surfaceType}: ` +
          `OCC classifier unavailable, falling back`
        );
      }
    }

    const loops = getFaceTrimLoopsUV(oc, face);
    if (loops) {
      let coneCrossesSeam = false;
      let coneWrappedOuterForSplit: Vec2[] | null = null;
      let cylinderCrossesSeam = false;
      let cylinderWrappedOuterForSplit: Vec2[] | null = null;
      let torusCrossesSeam = false;
      let degeneratePeriodicTrim = false;

      // Check if the UV boundary is degenerate (all points have same U or same V)
      // This happens for complete periodic surfaces like full torus where the boundary
      // in 3D maps to a line in UV space
      const uvOuter = loops.uvOuter;
      if (uvOuter.length >= 3) {
        const uValues = uvOuter.map(p => p[0]);
        const vValues = uvOuter.map(p => p[1]);
        const uMin = Math.min(...uValues);
        const uMax = Math.max(...uValues);
        const vMin = Math.min(...vValues);
        const vMax = Math.max(...vValues);
        const uSpan = uMax - uMin;
        const vSpan = vMax - vMin;

        // Check if UV boundary represents the full periodic domain
        // For a complete torus, the 3D boundary edges form a "seam" that maps to a
        // rectangle in UV space that doesn't actually enclose the domain center.
        const fullPeriodTolerance = 0.1; // ~6 degrees, only for full-period seam heuristics
        const degenerateSpanTolerance = 1e-6; // true UV collapse only
        const fullPeriodSpanThreshold = (2 * Math.PI) - fullPeriodTolerance;
        const centerU = (uMin + uMax) / 2;
        const centerV = (vMin + vMax) / 2;
        curveDebugLog(`[tessellateCurvedFace] UV bounds: U=[${uMin.toFixed(3)}, ${uMax.toFixed(3)}], V=[${vMin.toFixed(3)}, ${vMax.toFixed(3)}]`);
        curveDebugLog(`[tessellateCurvedFace] UV center: (${centerU.toFixed(3)}, ${centerV.toFixed(3)}), spans: uSpan=${uSpan.toFixed(3)}, vSpan=${vSpan.toFixed(3)}`);
        const isSliver = Math.abs(uSpan) < SLIVER_EPS || Math.abs(vSpan) < SLIVER_EPS;
        if (SLIVER_DEBUG_MODE === 'skip' && isSliver) {
          console.warn(
            `[SLIVER] skip face=${face.faceIndex} type=${face.surfaceType} uSpan=${uSpan.toFixed(6)} vSpan=${vSpan.toFixed(6)}`
          );
          return { vertices: [], triangles: [] };
        }
        if (SLIVER_DEBUG_MODE === 'only' && !isSliver) {
          return { vertices: [], triangles: [] };
        }
        if (SLIVER_DEBUG_MODE === 'only' && isSliver) {
          console.warn(
            `[SLIVER] keep-only face=${face.faceIndex} type=${face.surfaceType} uSpan=${uSpan.toFixed(6)} vSpan=${vSpan.toFixed(6)}`
          );
        }

        // For periodic surfaces (torus), if the UV boundary spans close to 2π in both dimensions,
        // check if the bounding box center is actually inside the polygon.
        // If not, the boundary is a "seam rectangle" that doesn't enclose the intended area.
        const PI = Math.PI;
        const isNearFullPeriod = uSpan > fullPeriodSpanThreshold && vSpan > fullPeriodSpanThreshold;

        if (face.surfaceType === 'Torus' && isNearFullPeriod) {
          // Quick point-in-polygon test for the center
          const centerInside = isPointInPolygonSimple([centerU, centerV], uvOuter);
          curveDebugLog(`[tessellateCurvedFace] Torus center (${centerU.toFixed(3)}, ${centerV.toFixed(3)}) inside polygon: ${centerInside}`);
          if (!centerInside) {
            curveDebugLog(`[tessellateCurvedFace] Torus UV boundary doesn't enclose center - falling back to full surface`);
            throw new Error('Torus seam boundary - use full surface tessellation');
          }
        }

        // For torus with V spanning ~2π (full circle in minor radius direction), check V-seam
        // This handles cases like half-torus where U is partial but V is full
        // Detection: look for consecutive points with a V-jump > π (seam crossing)
        if (face.surfaceType === 'Torus' && vSpan > 5.5) {
          let crossesVSeam = false;
          for (let i = 0; i < uvOuter.length; i++) {
            const v1 = uvOuter[i][1];
            const v2 = uvOuter[(i + 1) % uvOuter.length][1];
            const vJump = Math.abs(v2 - v1);
            if (vJump > PI) {
              crossesVSeam = true;
              curveDebugLog(`[tessellateCurvedFace] Torus V-seam detected: jump of ${vJump.toFixed(3)} at index ${i} (${v1.toFixed(3)} -> ${v2.toFixed(3)})`);
              break;
            }
          }

          curveDebugLog(`[tessellateCurvedFace] Torus V-seam check: vSpan=${vSpan.toFixed(3)}, crossesVSeam=${crossesVSeam}`);

          if (crossesVSeam) {
            torusCrossesSeam = true;
            avoidFullSurfaceFallback = true;
            curveDebugLog(`[tessellateCurvedFace] Torus V boundary crosses seam - shifting V to [0, 2π]`);

            // Shift V coordinates from [-π, π] to [0, 2π] for continuous boundary
            loops.uvOuter = uvOuter.map(([u, v]): Vec2 => {
              if (v < 0) {
                return [u, v + 2 * PI];
              }
              return [u, v];
            });

            // Update vMin/vMax after shift
            const shiftedVs = loops.uvOuter.map(p => p[1]);
            const shiftedVMin = Math.min(...shiftedVs);
            const shiftedVMax = Math.max(...shiftedVs);
            curveDebugLog(`[tessellateCurvedFace] Torus V shifted to [0, 2π]: V=[${shiftedVMin.toFixed(3)}, ${shiftedVMax.toFixed(3)}]`);

            // Shift holes as well
            if (loops.uvHoles.length > 0) {
              loops.uvHoles = loops.uvHoles.map((hole, h) => {
                const shiftedHole = hole.map(([u, v]): Vec2 => {
                  if (v < 0) {
                    return [u, v + 2 * PI];
                  }
                  return [u, v];
                });
                const holeVs = shiftedHole.map(p => p[1]);
                curveDebugLog(`[tessellateCurvedFace] Torus hole ${h} V shifted to [0, 2π]: V=[${Math.min(...holeVs).toFixed(3)}, ${Math.max(...holeVs).toFixed(3)}]`);
                return shiftedHole;
              });
            }
          }
        }

        // For torus with U spanning ~2π (full circle around the axis), check U-seam
        // This handles cases like partial-height torus rings where V is partial but U is full
        // Detection: look for consecutive points with a U-jump > π (seam crossing)
        if (face.surfaceType === 'Torus' && uSpan > 5.5) {
          let crossesUSeam = false;
          for (let i = 0; i < loops.uvOuter.length; i++) {
            const u1 = loops.uvOuter[i][0];
            const u2 = loops.uvOuter[(i + 1) % loops.uvOuter.length][0];
            const uJump = Math.abs(u2 - u1);
            if (uJump > PI) {
              crossesUSeam = true;
              curveDebugLog(`[tessellateCurvedFace] Torus U-seam detected: jump of ${uJump.toFixed(3)} at index ${i} (${u1.toFixed(3)} -> ${u2.toFixed(3)})`);
              break;
            }
          }

          curveDebugLog(`[tessellateCurvedFace] Torus U-seam check: uSpan=${uSpan.toFixed(3)}, crossesUSeam=${crossesUSeam}`);

          if (crossesUSeam) {
            torusCrossesSeam = true;
            avoidFullSurfaceFallback = true;
            curveDebugLog(`[tessellateCurvedFace] Torus U boundary crosses seam - shifting U to [0, 2π]`);

            // Shift U coordinates from [-π, π] to [0, 2π] for continuous boundary
            loops.uvOuter = loops.uvOuter.map(([u, v]): Vec2 => {
              if (u < 0) {
                return [u + 2 * PI, v];
              }
              return [u, v];
            });

            // Update uMin/uMax after shift
            const shiftedUs = loops.uvOuter.map(p => p[0]);
            const shiftedUMin = Math.min(...shiftedUs);
            const shiftedUMax = Math.max(...shiftedUs);
            curveDebugLog(`[tessellateCurvedFace] Torus U shifted to [0, 2π]: U=[${shiftedUMin.toFixed(3)}, ${shiftedUMax.toFixed(3)}]`);

            // Shift holes as well
            if (loops.uvHoles.length > 0) {
              loops.uvHoles = loops.uvHoles.map((hole, h) => {
                const shiftedHole = hole.map(([u, v]): Vec2 => {
                  if (u < 0) {
                    return [u + 2 * PI, v];
                  }
                  return [u, v];
                });
                const holeUs = shiftedHole.map(p => p[0]);
                curveDebugLog(`[tessellateCurvedFace] Torus hole ${h} U shifted to [0, 2π]: U=[${Math.min(...holeUs).toFixed(3)}, ${Math.max(...holeUs).toFixed(3)}]`);
                return shiftedHole;
              });
            }
          }
        }

        // For cones with full-circle base (U spans ~2π), the UV boundary may cross the ±π seam.
        // We must unwrap by edge continuity (not simple sign shift), otherwise seam-adjacent
        // points can collapse to the same U and clip away valid surface regions.
        if (face.surfaceType === 'Cone' && uSpan > 5.5) {
          // Use wrapped p-curve UVs for seam detection; normalized UVs can hide seam jumps.
          const seamProbeOuter = (loops.uvOuterRawWrapped && loops.uvOuterRawWrapped.length >= 3)
            ? loops.uvOuterRawWrapped
            : uvOuter;

          const nearPosPI = seamProbeOuter.filter(p => p[0] > PI - 0.3).length;
          const nearNegPI = seamProbeOuter.filter(p => p[0] < -PI + 0.3).length;
          const crossesByCounts = nearPosPI > 2 && nearNegPI > 2;

          const period = 2 * PI;
          const seamJumpThreshold = period * 0.75; // near full-wrap jump (avoid legitimate ~π meridian jumps)
          let maxJump = 0;
          let crossesByJump = false;
          for (let i = 0; i < seamProbeOuter.length; i++) {
            const u1 = seamProbeOuter[i][0];
            const u2 = seamProbeOuter[(i + 1) % seamProbeOuter.length][0];
            const jump = Math.abs(u2 - u1);
            maxJump = Math.max(maxJump, jump);
            if (jump > seamJumpThreshold) {
              crossesByJump = true;
            }
          }

          // Some seam-crossing trims only have a few samples on one side of ±π.
          // Use either robust count-based detection or a near-2π continuity jump.
          const crossesSeam = crossesByCounts || crossesByJump;
          coneCrossesSeam = crossesSeam;
          if (crossesSeam) {
            avoidFullSurfaceFallback = true;
            // Keep wrapped p-curve loop for seam-edge detection during split planning.
            coneWrappedOuterForSplit = seamProbeOuter.map(([u, v]): Vec2 => [u, v]);
          }

          curveDebugLog(`[tessellateCurvedFace] Cone seam check: nearPosPI=${nearPosPI}, nearNegPI=${nearNegPI}, maxJump=${maxJump.toFixed(3)}, byCounts=${crossesByCounts}, byJump=${crossesByJump}, crossesSeam=${crossesSeam}`);

          if (crossesSeam) {
            curveDebugLog(`[tessellateCurvedFace] Cone crosses seam - unwrapping U by continuity`);

            let unwrappedOuter = unwrapPeriodicLoopU(uvOuter, period);
            const unwrappedBounds = getLoopUBounds(unwrappedOuter);
            const outerShiftK = chooseShiftToRange(unwrappedBounds.uMin, unwrappedBounds.uMax, period, 0, period);
            if (outerShiftK !== 0) {
              unwrappedOuter = shiftLoopU(unwrappedOuter, outerShiftK * period);
            }
            loops.uvOuter = unwrappedOuter;

            const shiftedBounds = getLoopUBounds(loops.uvOuter);
            const outerMidU = meanLoopU(loops.uvOuter);
            console.log(
              `[tessellateCurvedFace] Cone unwrapped boundary: U=[${shiftedBounds.uMin.toFixed(3)}, ${shiftedBounds.uMax.toFixed(3)}], shiftK=${outerShiftK}, meanU=${outerMidU.toFixed(3)}, V=[${vMin.toFixed(2)}, ${vMax.toFixed(2)}]`
            );

            // Unwrap holes by continuity and align each hole near the outer loop band.
            if (loops.uvHoles.length > 0) {
              loops.uvHoles = loops.uvHoles.map((hole, h) => {
                let alignedHole = unwrapPeriodicLoopU(hole, period);
                if (outerShiftK !== 0) {
                  alignedHole = shiftLoopU(alignedHole, outerShiftK * period);
                }
                const holeMidU = meanLoopU(alignedHole);
                const alignK = Math.round((outerMidU - holeMidU) / period);
                if (alignK !== 0) {
                  alignedHole = shiftLoopU(alignedHole, alignK * period);
                }
                const holeBounds = getLoopUBounds(alignedHole);
                curveDebugLog(`[tessellateCurvedFace] Cone hole ${h} unwrapped: U=[${holeBounds.uMin.toFixed(3)}, ${holeBounds.uMax.toFixed(3)}], alignK=${alignK}`);
                return alignedHole;
              });
            }
          }
        }

        // For spheres with nearly full coverage, check if the center is inside
        if (face.surfaceType === 'Sphere' && uSpan > 5.5) {
          const centerInside = isPointInPolygonSimple([centerU, centerV], uvOuter);
          curveDebugLog(`[tessellateCurvedFace] Sphere center (${centerU.toFixed(3)}, ${centerV.toFixed(3)}) inside polygon: ${centerInside}`);
          if (!centerInside) {
            curveDebugLog(`[tessellateCurvedFace] Sphere UV boundary doesn't enclose center - falling back to full surface`);
            throw new Error('Sphere seam boundary - use full surface tessellation');
          }
        }

        // For cylinders with full-circle U span, seam-crossing trims need periodic unwrap.
        // For trimmed faces (holes/slots), prefer continuity unwrap and optional split later.
        // For non-trimmed full wraps, keep the rectangular fallback for stability.
        if (face.surfaceType === 'Cylinder' && uSpan > 5.5) {
          const seamProbeOuter = (loops.uvOuterRawWrapped && loops.uvOuterRawWrapped.length >= 3)
            ? loops.uvOuterRawWrapped
            : uvOuter;

          // Check if boundary crosses the seam (has points near both +π and -π)
          const nearPosPI = seamProbeOuter.filter(p => p[0] > PI - 0.3).length;
          const nearNegPI = seamProbeOuter.filter(p => p[0] < -PI + 0.3).length;
          const crossesByCounts = nearPosPI > 2 && nearNegPI > 2;
          const period = 2 * PI;
          const seamJumpThreshold = period * 0.75;
          let maxJump = 0;
          let crossesByJump = false;
          for (let i = 0; i < seamProbeOuter.length; i++) {
            const u1 = seamProbeOuter[i][0];
            const u2 = seamProbeOuter[(i + 1) % seamProbeOuter.length][0];
            const jump = Math.abs(u2 - u1);
            maxJump = Math.max(maxJump, jump);
            if (jump > seamJumpThreshold) {
              crossesByJump = true;
            }
          }
          const crossesSeam = crossesByCounts || crossesByJump;
          cylinderCrossesSeam = crossesSeam;
          if (crossesSeam) {
            avoidFullSurfaceFallback = true;
            cylinderWrappedOuterForSplit = seamProbeOuter.map(([u, v]): Vec2 => [u, v]);
          }

          curveDebugLog(`[tessellateCurvedFace] Cylinder seam check: nearPosPI=${nearPosPI}, nearNegPI=${nearNegPI}, maxJump=${maxJump.toFixed(3)}, byCounts=${crossesByCounts}, byJump=${crossesByJump}, crossesSeam=${crossesSeam}`);
          curveDebugLog(`[tessellateCurvedFace] Cylinder has ${loops.uvHoles.length} holes`);

          if (crossesSeam) {
            // Preserve true seam-crossing trim topology by continuity unwrapping
            // both outer loop and holes; sign-shift can distort hole footprints
            // when a loop straddles U=0.
            let unwrappedOuter = unwrapPeriodicLoopU(loops.uvOuter, period);
            const unwrappedBounds = getLoopUBounds(unwrappedOuter);
            const outerShiftK = chooseShiftToRange(unwrappedBounds.uMin, unwrappedBounds.uMax, period, 0, period);
            if (outerShiftK !== 0) {
              unwrappedOuter = shiftLoopU(unwrappedOuter, outerShiftK * period);
            }
            loops.uvOuter = unwrappedOuter;

            const shiftedBounds = getLoopUBounds(loops.uvOuter);
            const outerMidU = meanLoopU(loops.uvOuter);
            curveDebugLog(
              `[tessellateCurvedFace] Cylinder unwrapped boundary: U=[${shiftedBounds.uMin.toFixed(3)}, ${shiftedBounds.uMax.toFixed(3)}], ` +
              `shiftK=${outerShiftK}, meanU=${outerMidU.toFixed(3)}, V=[${vMin.toFixed(2)}, ${vMax.toFixed(2)}]`
            );

            if (loops.uvHoles.length > 0) {
              loops.uvHoles = loops.uvHoles.map((hole, h) => {
                let alignedHole = unwrapPeriodicLoopU(hole, period);
                if (outerShiftK !== 0) {
                  alignedHole = shiftLoopU(alignedHole, outerShiftK * period);
                }
                const holeMidU = meanLoopU(alignedHole);
                const alignK = Math.round((outerMidU - holeMidU) / period);
                if (alignK !== 0) {
                  alignedHole = shiftLoopU(alignedHole, alignK * period);
                }
                const holeBounds = getLoopUBounds(alignedHole);
                curveDebugLog(
                  `[tessellateCurvedFace] Cylinder hole ${h} unwrapped: ` +
                  `U=[${holeBounds.uMin.toFixed(3)}, ${holeBounds.uMax.toFixed(3)}], alignK=${alignK}`
                );
                return alignedHole;
              });
            }
          }
        }

        if (uSpan < degenerateSpanTolerance || vSpan < degenerateSpanTolerance) {
          if (isSeamSensitivePeriodicFace) {
            degeneratePeriodicTrim = true;
            avoidFullSurfaceFallback = true;
            console.warn(`[tessellateCurvedFace] Face ${face.faceIndex} (${face.surfaceType}) has degenerate UV span (uSpan=${uSpan.toFixed(6)}, vSpan=${vSpan.toFixed(6)}) - continuing trimmed path and disabling full-surface fallback`);
          } else {
            curveDebugLog(`[tessellateCurvedFace] UV boundary degenerate - falling back to full surface`);
            throw new Error('Degenerate UV boundary - use full surface tessellation');
          }
        }
      }

      // Simple point-in-polygon test (ray casting algorithm)
      function isPointInPolygonSimple(point: [number, number], polygon: [number, number][]): boolean {
        const [x, y] = point;
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
          const [xi, yi] = polygon[i];
          const [xj, yj] = polygon[j];
          if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
            inside = !inside;
          }
        }
        return inside;
      }

      let surface: any | null = null;

      if (face.surfaceType === 'Cylinder' && params.radius !== undefined && params.placement) {
        surface = { type: 'CYLINDRICAL_SURFACE', placement: params.placement, radius: params.radius };
      } else if (face.surfaceType === 'Sphere' && params.radius !== undefined && params.placement) {
        surface = { type: 'SPHERICAL_SURFACE', placement: params.placement, radius: params.radius };
      } else if (face.surfaceType === 'Cone' && params.radius !== undefined && params.semiAngle !== undefined && params.placement) {
        surface = { type: 'CONICAL_SURFACE', placement: params.placement, radius: params.radius, semiAngle: params.semiAngle };
      } else if (face.surfaceType === 'Torus' && params.majorRadius !== undefined && params.minorRadius !== undefined && params.placement) {
        surface = { type: 'TOROIDAL_SURFACE', placement: params.placement, majorRadius: params.majorRadius, minorRadius: params.minorRadius };
      } else if (face.surfaceType === 'BSplineSurface' && params.bspline) {
        const { controlPoints, uDegree, vDegree, uKnots, vKnots, weights } = params.bspline;
        surface = { type: 'B_SPLINE_SURFACE', controlPoints, uDegree, vDegree, uKnots, vKnots, weights };
      }

      if (surface) {
        const isNoHoleCylinderSurface =
          face.surfaceType === 'Cylinder' &&
          surface.type === 'CYLINDRICAL_SURFACE' &&
          loops.uvHoles.length === 0 &&
          loops.uvOuter.length >= 3;
        if (isNoHoleCylinderSurface) {
          // For no-hole cylinders, use the dedicated cylindrical patch tessellator.
          // This avoids coarse square-grid artifacts from generic trimmed-surface meshing.
          const uBounds = getLoopUBounds(loops.uvOuter);
          const vBounds = getLoopComponentBounds(loops.uvOuter, 1);
          const uSpanLocal = uBounds.uMax - uBounds.uMin;
          const vSpanLocal = vBounds.max - vBounds.min;
          const defaultUSamples = uSpanLocal > 5.5 ? 64 : 48;
          const defaultVSamples = Math.max(3, Math.min(12, Math.ceil(vSpanLocal / 2)));
          const numUSamples = Math.max(16, Math.floor(readGlobalNumber('__CYLINDER_PATCH_U_SAMPLES__') ?? defaultUSamples));
          const numVSamples = Math.max(2, Math.floor(readGlobalNumber('__CYLINDER_PATCH_V_SAMPLES__') ?? defaultVSamples));
          const mesh = await tessellateCylinder(
            surface,
            uBounds.uMin,
            uBounds.uMax,
            vBounds.min,
            vBounds.max,
            numUSamples,
            numVSamples
          );
          tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
          tessellationProfile.tessellateCurvedFace.calls++;
          return tessellatedMeshToVerticesAndTriangles(mesh);
        }

        const gridDensity = chooseTrimGridDensity(face, loops.uvOuter, loops.uvHoles);

        // For cylinders, compute 3D bounding box from boundary edges to filter
        // grid points whose 3D positions fall outside the intended region.
        // This is essential for horizontal cylinders where UV spans full circle
        // but we only want a portion of the surface (e.g., upper half of a hole).
        let bbox3d: { xMin: number; xMax: number; yMin: number; yMax: number; zMin: number; zMax: number } | undefined;
        if (face.surfaceType === 'Cylinder') {
          const outer3d = occEdgesToPolygon(face.outerLoop);
          if (outer3d.length > 0) {
            const xs = outer3d.map(p => p[0]);
            const ys = outer3d.map(p => p[1]);
            const zs = outer3d.map(p => p[2]);
            bbox3d = {
              xMin: Math.min(...xs), xMax: Math.max(...xs),
              yMin: Math.min(...ys), yMax: Math.max(...ys),
              zMin: Math.min(...zs), zMax: Math.max(...zs)
            };
            curveDebugLog(`[tessellateCurvedFace] Face ${face.faceIndex}: Cylinder 3D bbox:`);
            curveDebugLog(`  X: [${bbox3d.xMin.toFixed(2)}, ${bbox3d.xMax.toFixed(2)}]`);
            curveDebugLog(`  Y: [${bbox3d.yMin.toFixed(2)}, ${bbox3d.yMax.toFixed(2)}]`);
            curveDebugLog(`  Z: [${bbox3d.zMin.toFixed(2)}, ${bbox3d.zMax.toFixed(2)}]`);
          }
        }

        let occBuildClassifier: any | undefined;
        let usedOccBuildClassifier = false;
        let trimmedBuildOptions: TrimmedSurfaceBuildOptions | undefined;

        const isCylinderSurfaceForTrim = face.surfaceType === 'Cylinder' && surface.type === 'CYLINDRICAL_SURFACE';
        if (isCylinderSurfaceForTrim) {
          const period = Math.PI * 2;
          const uBounds = getLoopUBounds(loops.uvOuter);
          const vBounds = getLoopComponentBounds(loops.uvOuter, 1);
          const uSpan = Math.max(1e-6, uBounds.uMax - uBounds.uMin);
          const vSpan = Math.max(1e-6, vBounds.max - vBounds.min);

          const fullUSamples = Math.max(32, Math.floor(readGlobalNumber('__CYLINDER_TRIM_U_SAMPLES_FULL__') ?? 64));
          const minUSamples = Math.max(16, Math.floor(readGlobalNumber('__CYLINDER_TRIM_U_SAMPLES_MIN__') ?? 32));
          const maxUSamples = Math.max(minUSamples, Math.floor(readGlobalNumber('__CYLINDER_TRIM_U_SAMPLES_MAX__') ?? 128));
          const minUSamplesWithHoles = Math.max(minUSamples, Math.floor(readGlobalNumber('__CYLINDER_TRIM_U_SAMPLES_HOLES_MIN__') ?? 48));
          let gridDensityU = Math.round(fullUSamples * (uSpan / period));
          if (loops.uvHoles.length > 0) {
            gridDensityU = Math.max(gridDensityU, minUSamplesWithHoles);
          }
          gridDensityU = Math.max(minUSamples, Math.min(maxUSamples, gridDensityU));

          const vStep = Math.max(0.2, readGlobalNumber('__CYLINDER_TRIM_V_STEP__') ?? 2.0);
          const minVDensity = Math.max(4, Math.floor(readGlobalNumber('__CYLINDER_TRIM_V_SAMPLES_MIN__') ?? 8));
          const maxVDensity = Math.max(minVDensity, Math.floor(readGlobalNumber('__CYLINDER_TRIM_V_SAMPLES_MAX__') ?? 24));
          let gridDensityV = Math.ceil(vSpan / vStep);
          if (loops.uvHoles.length > 0) {
            gridDensityV = Math.max(gridDensityV, Math.floor(readGlobalNumber('__CYLINDER_TRIM_V_SAMPLES_HOLES_MIN__') ?? 10));
          }
          gridDensityV = Math.max(minVDensity, Math.min(maxVDensity, gridDensityV));

          trimmedBuildOptions = {
            ...trimmedBuildOptions,
            // Cylinder trims are best represented by anisotropic UV grids.
            gridDensityU,
            gridDensityV,
            // Keep the OCCT-inspired cylinder hole path enabled by default.
            // This is the known-stable visual mode (with slight corner curvature).
            preferGridForHoles: true,
          };
        }

        const useOccPrimaryBuildForKnownFace = isKnownLidFace && !!face.occFace;
        if (useOccPrimaryBuildForKnownFace && oc?.BRepTopAdaptor_FClass2d && oc?.gp_Pnt2d_3 && oc?.TopAbs_State) {
          try {
            // For known problematic lid faces, use OCC classification during triangle build.
            // This avoids destructive post-filtering that can drop most of a valid face.
            occBuildClassifier = new oc.BRepTopAdaptor_FClass2d(face.occFace, 1e-7);
            usedOccBuildClassifier = true;
            const classifyInside = (u: number, v: number): boolean => {
              const uvPoint = new oc.gp_Pnt2d_3(u, v);
              try {
                const state = occBuildClassifier.Perform(uvPoint, true);
                return !topAbsStateEquals(state, oc.TopAbs_State.TopAbs_OUT);
              } catch {
                // Fail open to avoid catastrophic face loss when classifier is unstable.
                return true;
              } finally {
                uvPoint.delete?.();
              }
            };
            const maxOutSamples = 1;
            trimmedBuildOptions = {
              ...trimmedBuildOptions,
              uvInsideTest: classifyInside,
              keepTriangle: (samples) => {
                let outCount = 0;
                for (const [u, v] of samples) {
                  if (!classifyInside(u, v)) {
                    outCount++;
                    if (outCount > maxOutSamples) {
                      return false;
                    }
                  }
                }
                return true;
              },
              // Partial quad triangles are the most frequent source of slivers on these faces.
              allowPartialCellTriangles: false,
              logLabel: `occ-face-${face.faceIndex}`,
            };
            console.log(`[seam-build] face ${face.faceIndex} ${face.surfaceType}: OCC-guided triangle build enabled`);
          } catch (e) {
            console.warn(`[seam-build] face ${face.faceIndex} ${face.surfaceType}: OCC-guided build unavailable, falling back`, e);
          }
        }

        const simplifySeamPatch = (
          patch: { uvOuter: Vec2[]; uvHoles: Vec2[][] },
          mode: 'cone' | 'cylinder'
        ) => {
          const defaultOuterPts = mode === 'cone' ? 192 : 256;
          const defaultHolePts = mode === 'cone' ? 32 : 64;
          const defaultAreaErr = mode === 'cone' ? 0.03 : 0.02;
          const defaultAreaErrNoHoles = mode === 'cone' ? 0.08 : 0.06;
          const keyPrefix = mode === 'cone' ? '__CONE_TRIM' : '__CYLINDER_TRIM';

          const maxOuterPts = Math.max(64, Math.floor(readGlobalNumber(`${keyPrefix}_MAX_OUTER_PTS__`) ?? defaultOuterPts));
          const maxHolePts = Math.max(12, Math.floor(readGlobalNumber(`${keyPrefix}_MAX_HOLE_PTS__`) ?? defaultHolePts));
          const maxAreaErrorRatio = Math.max(1e-4, readGlobalNumber(`${keyPrefix}_MAX_AREA_ERR_RATIO__`) ?? defaultAreaErr);
          const maxOuterAreaErrorNoHoles = Math.max(
            maxAreaErrorRatio,
            readGlobalNumber(`${keyPrefix}_MAX_AREA_ERR_RATIO_NO_HOLES__`) ?? defaultAreaErrNoHoles
          );

          const outerAreaErrRatio = patch.uvHoles.length === 0 ? maxOuterAreaErrorNoHoles : maxAreaErrorRatio;
          const simplifiedOuter = simplifyLoopForMeshing(patch.uvOuter, maxOuterPts, outerAreaErrRatio);
          const simplifiedHoles = patch.uvHoles
            .map((hole) => simplifyLoopForMeshing(hole, maxHolePts, maxAreaErrorRatio))
            .filter((hole) => hole.length >= 3);
          const validation = validateAndSanitizeTrimLoops(simplifiedOuter, simplifiedHoles, {
            minAreaAbs: 1e-7,
            maxHoleToOuterRatio: 0.98,
            failOnHoleOutside: true,
            failOnHugeHole: true,
          });
          if (validation.ok) {
            return { uvOuter: validation.uvOuter, uvHoles: validation.uvHoles };
          }
          return patch;
        };

        const isConeSeamSplitFaceEnabled =
          coneSeamSplitFaceIdsRaw == null || coneSeamSplitFaceIds.has(face.faceIndex);
        const isCylinderSeamSplitFaceEnabled =
          cylinderSeamSplitFaceIdsRaw == null || cylinderSeamSplitFaceIds.has(face.faceIndex);
        const cylinderOuterBounds = getLoopUBounds(loops.uvOuter);
        const cylinderOuterUSpan = cylinderOuterBounds.uMax - cylinderOuterBounds.uMin;
        const cylinderLikelyPeriodic = face.surfaceType === 'Cylinder' && cylinderOuterUSpan > 5.5;
        const shouldTryConeSeamSplit =
          face.surfaceType === 'Cone' &&
          enableConeSeamSplit &&
          isConeSeamSplitFaceEnabled &&
          loops.uvOuter.length >= 3 &&
          loops.uvHoles.length > 0;
        const shouldTryCylinderSeamSplit =
          face.surfaceType === 'Cylinder' &&
          enableCylinderSeamSplit &&
          isCylinderSeamSplitFaceEnabled &&
          cylinderCrossesSeam &&
          loops.uvOuter.length >= 3 &&
          loops.uvHoles.length > 0;
        const coneSplitSourceOuter =
          (coneWrappedOuterForSplit && coneWrappedOuterForSplit.length >= 3)
            ? coneWrappedOuterForSplit
            : ((loops.uvOuterRawWrapped && loops.uvOuterRawWrapped.length >= 3)
              ? loops.uvOuterRawWrapped
              : loops.uvOuter);
        const cylinderSplitSourceOuter =
          (cylinderWrappedOuterForSplit && cylinderWrappedOuterForSplit.length >= 3)
            ? cylinderWrappedOuterForSplit
            : ((loops.uvOuterRawWrapped && loops.uvOuterRawWrapped.length >= 3)
              ? loops.uvOuterRawWrapped
              : loops.uvOuter);
        if (
          face.surfaceType === 'Cone' &&
          enableConeSeamSplit &&
          isConeSeamSplitFaceEnabled &&
          !shouldTryConeSeamSplit
        ) {
          console.log(
            `[cone-seam-split] face ${face.faceIndex} ${face.surfaceType}: skipped ` +
            `(crossesSeam=${coneCrossesSeam}, wrappedOuter=${!!coneWrappedOuterForSplit}, ` +
            `rawOuter=${!!loops.uvOuterRawWrapped}, ` +
            `outerPts=${loops.uvOuter.length}, holes=${loops.uvHoles.length})`
          );
        }
        if (
          face.surfaceType === 'Cylinder' &&
          enableCylinderSeamSplit &&
          isCylinderSeamSplitFaceEnabled &&
          !shouldTryCylinderSeamSplit
        ) {
          console.log(
            `[cylinder-seam-split] face ${face.faceIndex} ${face.surfaceType}: skipped ` +
            `(crossesSeam=${cylinderCrossesSeam}, likelyPeriodic=${cylinderLikelyPeriodic}, wrappedOuter=${!!cylinderWrappedOuterForSplit}, ` +
            `rawOuter=${!!loops.uvOuterRawWrapped}, ` +
            `outerPts=${loops.uvOuter.length}, holes=${loops.uvHoles.length})`
          );
        }

        let mesh;
        try {
          if (shouldTryConeSeamSplit || shouldTryCylinderSeamSplit) {
            const splitLabel = shouldTryConeSeamSplit ? 'cone-seam-split' : 'cylinder-seam-split';
            const splitModeName: 'cone' | 'cylinder' = shouldTryConeSeamSplit ? 'cone' : 'cylinder';
            const splitSourceOuter = shouldTryConeSeamSplit ? coneSplitSourceOuter : cylinderSplitSourceOuter;
            const splitResult = splitConeTrimLoopsIntoTwoPatches(
              splitSourceOuter,
              loops.uvOuter,
              loops.uvHoles
            );
            if (splitResult.ok && splitResult.leftPatch && splitResult.rightPatch && splitResult.uSplit != null) {
              const leftPatch = simplifySeamPatch(splitResult.leftPatch, splitModeName);
              const rightPatch = simplifySeamPatch(splitResult.rightPatch, splitModeName);
              const patchMeshes: TessellatedMeshLike[] = [];
              patchMeshes.push(
                await tessellateTrimmedSurface(
                  surface,
                  leftPatch.uvOuter,
                  gridDensity,
                  leftPatch.uvHoles,
                  bbox3d,
                  trimmedBuildOptions
                ) as TessellatedMeshLike
              );
              patchMeshes.push(
                await tessellateTrimmedSurface(
                  surface,
                  rightPatch.uvOuter,
                  gridDensity,
                  rightPatch.uvHoles,
                  bbox3d,
                  trimmedBuildOptions
                ) as TessellatedMeshLike
              );
              mesh = mergeTessellatedMeshes(patchMeshes);
              console.log(
                `[${splitLabel}] face ${face.faceIndex} ${face.surfaceType}: split at U=${splitResult.uSplit.toFixed(3)} ` +
                `mode=${splitResult.splitMode ?? 'unknown'} ` +
                `left(outer=${leftPatch.uvOuter.length}, holes=${leftPatch.uvHoles.length}) ` +
                `right(outer=${rightPatch.uvOuter.length}, holes=${rightPatch.uvHoles.length}) ` +
                `areas=(${(splitResult.leftAreaAbs ?? 0).toExponential(3)}, ${(splitResult.rightAreaAbs ?? 0).toExponential(3)}) ` +
                `balance=${(splitResult.areaBalance ?? 0).toFixed(4)}`
              );
            } else {
              console.warn(
                `[${splitLabel}] face ${face.faceIndex} ${face.surfaceType}: split unavailable/failed (${splitResult.reason ?? 'unknown'}), using single trimmed tessellation`
              );
              mesh = await tessellateTrimmedSurface(surface, loops.uvOuter, gridDensity, loops.uvHoles, bbox3d, trimmedBuildOptions);
            }
          } else {
            mesh = await tessellateTrimmedSurface(surface, loops.uvOuter, gridDensity, loops.uvHoles, bbox3d, trimmedBuildOptions);
          }
        } finally {
          occBuildClassifier?.delete?.();
        }

        // Robust safety pass: keep only triangles whose UV samples are classified
        // as not OUT by OCC. This is critical for seam-sensitive periodic trims and
        // for known problematic faces that produce lid artifacts.
        const aggressivePeriodicFiltering = readGlobalBoolean('__ENABLE_AGGRESSIVE_PERIODIC_FILTER__', false);
        const shouldClassifyPeriodicTrim = !usedOccBuildClassifier && isSeamSensitivePeriodicFace &&
          (coneCrossesSeam || cylinderCrossesSeam || torusCrossesSeam || degeneratePeriodicTrim);
        const shouldClassifyComplexConeOrTorus = aggressivePeriodicFiltering && !usedOccBuildClassifier && (
          (face.surfaceType === 'Torus') ||
          (face.surfaceType === 'Cone' && (loops.uvHoles.length > 0 || face.outerLoop.length > 8))
        );
        const shouldClassifyKnownLidFace = isKnownLidFace && !usedOccBuildClassifier;
        const shouldClassifyByOcc =
          (shouldClassifyPeriodicTrim || shouldClassifyComplexConeOrTorus || shouldClassifyKnownLidFace) && !!face.occFace;
        if (shouldClassifyByOcc && face.occFace) {
          const reason: string[] = [];
          if (coneCrossesSeam) reason.push('cone-seam');
          if (cylinderCrossesSeam) reason.push('cylinder-seam');
          if (torusCrossesSeam) reason.push('torus-seam');
          if (degeneratePeriodicTrim) reason.push('degenerate');
          if (face.surfaceType === 'Torus' && !torusCrossesSeam) reason.push('torus-complex');
          if (face.surfaceType === 'Cone' && !coneCrossesSeam) reason.push('cone-complex');
          if (shouldClassifyKnownLidFace) reason.push('known-lid-face');
          // Keep torus filtering strict, but keep cones tolerant and capped.
          // Cone strictness was removing valid side-wall triangles.
          const isComplexTorus = shouldClassifyComplexConeOrTorus && face.surfaceType === 'Torus';
          const isComplexCone = shouldClassifyComplexConeOrTorus && face.surfaceType === 'Cone';
          const maxOutSamples = isComplexTorus ? 0 : 1;
          const maxDropRatio = (shouldClassifyKnownLidFace || isComplexCone) ? 0.30 : undefined;
          mesh = filterMeshTrianglesByFaceUVClassification(oc, face.occFace, mesh, {
            tol: 1e-7,
            recadreOnPeriodic: true,
            label: `face ${face.faceIndex} ${face.surfaceType} ${reason.join('+')}`,
            logAlways: shouldClassifyComplexConeOrTorus || shouldClassifyKnownLidFace,
            sampleMode: 'multi7',
            maxOutSamples,
            maxDropRatio,
            faceIndex: face.faceIndex,
            surfaceType: face.surfaceType,
            periodicProof: shouldRunPeriodicProof,
          });
        }

        tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
        tessellationProfile.tessellateCurvedFace.calls++;
        return tessellatedMeshToVerticesAndTriangles(mesh);
      }
    }
  } catch (e) {
    if (avoidFullSurfaceFallback && isSeamSensitivePeriodicFace) {
      const message = e instanceof Error ? e.message : String(e);
      console.warn(`[tessellateCurvedFace] Face ${face.faceIndex} (${face.surfaceType}) trimmed path failed (${message}) - skipping unsafe full-surface fallback`);
      tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
      tessellationProfile.tessellateCurvedFace.calls++;
      return { vertices: [], triangles: [] };
    }
    // Fall through to non-trimmed fallback below
  }

  // Fallback: rectangular parameter patch (kept as a backstop if UV projection fails).
  const { uMin, uMax, vMin, vMax } = face.uvBounds;

  if (face.surfaceType === 'Cylinder' && params.radius !== undefined && params.placement) {
    const mesh = await tessellateCylinder(
      { type: 'CYLINDRICAL_SURFACE', placement: params.placement, radius: params.radius },
      uMin, uMax,
      vMin, vMax,
      64, 4
    );
    tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
    tessellationProfile.tessellateCurvedFace.calls++;
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  if (face.surfaceType === 'Sphere' && params.radius !== undefined && params.placement) {
    const mesh = await tessellateSphere(
      { type: 'SPHERICAL_SURFACE', placement: params.placement, radius: params.radius },
      uMin, uMax,
      vMin, vMax,
      64, 32
    );
    tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
    tessellationProfile.tessellateCurvedFace.calls++;
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  if (face.surfaceType === 'Cone' && params.radius !== undefined && params.semiAngle !== undefined && params.placement) {
    // Compute height samples proportional to the height range for smooth tessellation
    const heightRange = Math.abs(vMax - vMin);
    const numHeightSamples = Math.max(16, Math.ceil(heightRange * 8));
    const mesh = await tessellateCone(
      { type: 'CONICAL_SURFACE', placement: params.placement, radius: params.radius, semiAngle: params.semiAngle },
      uMin, uMax,
      vMin, vMax,
      64,
      numHeightSamples
    );
    tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
    tessellationProfile.tessellateCurvedFace.calls++;
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  if (face.surfaceType === 'Torus' && params.majorRadius !== undefined && params.minorRadius !== undefined && params.placement) {
    const mesh = await tessellateTorus(
      { type: 'TOROIDAL_SURFACE', placement: params.placement, majorRadius: params.majorRadius, minorRadius: params.minorRadius },
      uMin, uMax,
      vMin, vMax,
      64, 32
    );
    tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
    tessellationProfile.tessellateCurvedFace.calls++;
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  if (face.surfaceType === 'BSplineSurface' && params.bspline) {
    const { controlPoints, uDegree, vDegree, uKnots, vKnots, weights } = params.bspline;
    const mesh = await tessellateBSplineSurface(
      { type: 'B_SPLINE_SURFACE', controlPoints, uDegree, vDegree, uKnots, vKnots, weights },
      32, 32
    );
    tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
    tessellationProfile.tessellateCurvedFace.calls++;
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  console.warn(`[Tessellate] Unsupported curved surface type: ${face.surfaceType}`);
  tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
  tessellationProfile.tessellateCurvedFace.calls++;
  return { vertices: [], triangles: [] };
}

/**
 * Compute the normal of a triangle given three vertices
 */
function computeTriangleNormal(v0: Vec3, v1: Vec3, v2: Vec3): Vec3 {
  // Edge vectors
  const e1: Vec3 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
  const e2: Vec3 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

  // Cross product
  const n: Vec3 = [
    e1[1] * e2[2] - e1[2] * e2[1],
    e1[2] * e2[0] - e1[0] * e2[2],
    e1[0] * e2[1] - e1[1] * e2[0]
  ];

  // Normalize
  const len = Math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (len > 1e-10) {
    n[0] /= len;
    n[1] /= len;
    n[2] /= len;
  }

  return n;
}

/**
 * Tessellate all faces from OCC and create a Mesh
 */
async function tessellateOCCShape(
  faces: FaceWithEdgesInfo[],
  triangulationMethod: TriangulationMethod = 'ear-clipping',
  onProgress?: (percent: number) => void,
  faceDiagnosticsOut?: OccFaceTessellationDiagnostic[],
  stepFaceIdOrder?: number[],
  targetFaceSet?: Set<number>
): Promise<Mesh> {
  const shapeStart = performance.now();
  console.log(`[Tessellate] Starting tessellation of ${faces.length} faces...`);
  const FACE_DEBUG_IDS = readFaceIdsFromGlobal('__FACE_DEBUG_IDS__', [14, 63, 64, 65, 66, 994]);
  const FACE_DEBUG_MODE = readDebugModeFromGlobal('__FACE_DEBUG_MODE__', 'off');

  const allVertices: Vec3[] = [];
  const allNormals: Vec3[] = [];
  const allColors: RGBColor[] = []; // Per-vertex colors
  const allIndices: number[] = [];
  let vertexOffset = 0;
  let hasAnyColor = false;
  let processedCount = 0;
  let skippedCount = 0;
  let errorCount = 0;
  const progressInterval = Math.max(1, Math.floor(faces.length / 20)); // Log every 5%
  const shouldCollectDiagnostics = Array.isArray(faceDiagnosticsOut);

  const pushFaceDiagnostic = (
    face: FaceWithEdgesInfo,
    status: FaceTessellationStatus,
    durationMs: number,
    outputVertexCount: number,
    outputTriangleCount: number,
    error?: string
  ) => {
    if (!shouldCollectDiagnostics || !faceDiagnosticsOut) return;
    faceDiagnosticsOut.push({
      faceIndex: face.faceIndex,
      stepFaceId: stepFaceIdOrder?.[face.faceIndex],
      surfaceType: face.surfaceType,
      status,
      isReversed: !!face.isReversed,
      outerEdgeCount: face.outerLoop.length,
      innerLoopCount: face.innerLoops.length,
      outputVertexCount,
      outputTriangleCount,
      durationMs,
      error,
    });
  };

  for (const face of faces) {
    let faceStart = performance.now();
    try {
      if (targetFaceSet && !targetFaceSet.has(face.faceIndex)) {
        // In targeted diagnostic mode, skip non-target faces quietly so outputs stay focused.
        skippedCount++;
        processedCount++;
        continue;
      }
      const isFaceDebugTarget = FACE_DEBUG_IDS.has(face.faceIndex);
      if (FACE_DEBUG_MODE === 'skip' && isFaceDebugTarget) {
        pushFaceDiagnostic(face, 'skipped', performance.now() - faceStart, 0, 0, 'face-debug-skip');
        skippedCount++;
        processedCount++;
        continue;
      }
      if (FACE_DEBUG_MODE === 'only' && !isFaceDebugTarget) {
        pushFaceDiagnostic(face, 'skipped', performance.now() - faceStart, 0, 0, 'face-debug-only');
        skippedCount++;
        processedCount++;
        continue;
      }
      if (FACE_DEBUG_MODE === 'only' && isFaceDebugTarget) {
        console.warn(`[FACE DEBUG] keep-only face=${face.faceIndex} type=${face.surfaceType}`);
      }

      // Progress logging
      if (processedCount % progressInterval === 0 || processedCount < 10) {
        const elapsed = ((performance.now() - shapeStart) / 1000).toFixed(1);
        const pct = ((processedCount / faces.length) * 100).toFixed(1);
        console.log(`[Tessellate] Face ${processedCount}/${faces.length} (${pct}%) - ${elapsed}s elapsed - type: ${face.surfaceType}`);
        if (onProgress) {
          // Map tessellation progress to 20-100% of overall progress
          const overallPct = Math.round(20 + (processedCount / faces.length) * 80);
          onProgress(overallPct);
          // Yield to let the UI update
          await new Promise(r => setTimeout(r, 0));
        }
      }
      let result: { vertices: Vec3[]; triangles: number[][] };

      // DEBUG FLAG: Set to true to skip torus faces for isolation testing
      const SKIP_TORUS_FACES = false;

      // DEBUG FLAGS: Set to true to skip specific faces for isolation testing
      // Cylinder faces summary:
      // Face 1:  V=[4.75, 85.00], reversed=false (half cylinder)
      // Face 2:  V=[4.75, 57.50], reversed=false (full cylinder, crosses seam)
      // Face 4:  V=[2.24, 4.99],  reversed=false (half cylinder, short)
      // Face 7:  V=[2.24, 4.99],  reversed=false (full cylinder, short)
      // Face 10: V=[0.00, 27.25], reversed=true  ⚠️ V STARTS AT 0!
      // Face 13: V=[4.15, 54.40], reversed=true  (half cylinder)
      // Face 15: V=[4.15, 62.37], reversed=true  (full cylinder)
      // Face 16: V=?,            reversed=true  (hole wall)
      // Face 17: V=?,            reversed=true  (hole wall)
      const SKIP_FACE_10 = false;  // Fixed: 3D bbox filtering now in tessellateTrimmedSurface
      const SKIP_FACE_16 = false;  // Hole wall cylinder
      const SKIP_FACE_17 = false;  // Hole wall cylinder

      if (SKIP_TORUS_FACES && face.surfaceType === 'Torus') {
        console.log(`[Tessellate] SKIPPING Torus face ${face.faceIndex} (debug flag)`);
        pushFaceDiagnostic(face, 'skipped', performance.now() - faceStart, 0, 0, 'debug-skip-torus');
        skippedCount++;
        processedCount++;
        continue;
      }

      if (SKIP_FACE_10 && face.faceIndex === 10) {
        console.log(`[Tessellate] SKIPPING Face 10 (debug flag) - Cylinder with V starting at 0`);
        pushFaceDiagnostic(face, 'skipped', performance.now() - faceStart, 0, 0, 'debug-skip-face-10');
        skippedCount++;
        processedCount++;
        continue;
      }

      if (SKIP_FACE_16 && face.faceIndex === 16) {
        console.log(`[Tessellate] SKIPPING Face 16 (debug flag) - Cylinder`);
        pushFaceDiagnostic(face, 'skipped', performance.now() - faceStart, 0, 0, 'debug-skip-face-16');
        skippedCount++;
        processedCount++;
        continue;
      }

      if (SKIP_FACE_17 && face.faceIndex === 17) {
        console.log(`[Tessellate] SKIPPING Face 17 (debug flag) - Cylinder`);
        pushFaceDiagnostic(face, 'skipped', performance.now() - faceStart, 0, 0, 'debug-skip-face-17');
        skippedCount++;
        processedCount++;
        continue;
      }

      if (face.surfaceType === 'Plane') {
        result = await tessellatePlanarFaceFromOCC(face, triangulationMethod);
      } else if (['Cylinder', 'Sphere', 'Cone', 'Torus', 'BSplineSurface'].includes(face.surfaceType)) {
        // 3D bbox filtering for cylinders is now done inside tessellateTrimmedSurface
        // during grid generation, which produces better results than post-filtering
        result = await tessellateCurvedFaceFromOCC(face);
      } else {
        // Skip unsupported surface types
        pushFaceDiagnostic(face, 'skipped', performance.now() - faceStart, 0, 0, `unsupported-surface:${face.surfaceType}`);
        skippedCount++;
        processedCount++;
        continue;
      }

      // Compute per-vertex normals by averaging face normals
      const normalStart = performance.now();

      // First, initialize normals to zero
      const vertexNormals: Vec3[] = result.vertices.map(() => [0, 0, 0] as Vec3);

      // Accumulate face normals at each vertex
      for (const tri of result.triangles) {
        const v0 = result.vertices[tri[0]];
        const v1 = result.vertices[tri[1]];
        const v2 = result.vertices[tri[2]];
        const faceNormal = computeTriangleNormal(v0, v1, v2);

        // Add face normal to each vertex of the triangle
        for (const idx of tri) {
          vertexNormals[idx][0] += faceNormal[0];
          vertexNormals[idx][1] += faceNormal[1];
          vertexNormals[idx][2] += faceNormal[2];
        }
      }

      // Normalize the accumulated normals
      for (const n of vertexNormals) {
        const len = Math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 1e-10) {
          n[0] /= len;
          n[1] /= len;
          n[2] /= len;
        } else {
          // Default to up vector if degenerate
          n[0] = 0;
          n[1] = 0;
          n[2] = 1;
        }
      }

      // Handle REVERSED face orientation
      // When a face is REVERSED, the surface normal should point in the opposite direction.
      // This is common for hole inner walls - the cylinder surface naturally faces outward,
      // but for a hole it should face inward.
      if (face.isReversed) {
        // Flip all normals
        for (const n of vertexNormals) {
          n[0] = -n[0];
          n[1] = -n[1];
          n[2] = -n[2];
        }
        // Reverse triangle winding (swap indices 1 and 2) so front face is correct
        for (const tri of result.triangles) {
          const temp = tri[1];
          tri[1] = tri[2];
          tri[2] = temp;
        }
        tessellationVerboseLog(`[Tessellate] Face ${face.faceIndex}: Applied REVERSED orientation (flipped normals + winding)`);
      }

      tessellationProfile.computeNormals.total += performance.now() - normalStart;
      tessellationProfile.computeNormals.calls++;

      // Add vertices, normals and assign face color to each vertex
      const faceColor = face.color || { r: 0.4, g: 0.6, b: 1.0 }; // Default blue-ish
      if (face.color) {
        hasAnyColor = true;
      }

      for (let i = 0; i < result.vertices.length; i++) {
        allVertices.push(result.vertices[i]);
        allNormals.push(vertexNormals[i]);
        allColors.push(faceColor);
      }

      for (const tri of result.triangles) {
        allIndices.push(
          tri[0] + vertexOffset,
          tri[1] + vertexOffset,
          tri[2] + vertexOffset
        );
      }

      vertexOffset += result.vertices.length;
      processedCount++;

      // Log slow faces
      const faceTime = performance.now() - faceStart;
      if (faceTime > 500) {
        console.warn(`[Tessellate] SLOW face ${face.faceIndex} (${face.surfaceType}): ${(faceTime/1000).toFixed(2)}s, ${result.vertices.length} verts, ${result.triangles.length} tris`);
      }
      pushFaceDiagnostic(face, 'ok', faceTime, result.vertices.length, result.triangles.length);
    } catch (e) {
      console.error(`[Tessellate] Error tessellating face ${face.faceIndex}:`, e);
      const errorMsg = e instanceof Error ? e.message : String(e);
      pushFaceDiagnostic(face, 'error', performance.now() - faceStart, 0, 0, errorMsg);
      errorCount++;
      processedCount++;
    }
  }

  const tessTime = ((performance.now() - shapeStart) / 1000).toFixed(2);
  console.log(`[Tessellate] Completed: ${processedCount} faces in ${tessTime}s (${skippedCount} skipped, ${errorCount} errors)`);
  console.log(`[Tessellate] Generated: ${allVertices.length} vertices, ${allIndices.length / 3} triangles`);

  // Build final mesh
  const assemblyStart = performance.now();

  const positions = new Float32Array(allVertices.length * 3);
  allVertices.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  const normals = new Float32Array(allNormals.length * 3);
  allNormals.forEach((n, i) => {
    normals[i * 3 + 0] = n[0];
    normals[i * 3 + 1] = n[1];
    normals[i * 3 + 2] = n[2];
  });

  const indices = new Uint32Array(allIndices);

  // Build vertex colors array if we have any colors
  // NOTE: STEP file colors are typically already in linear color space (CAD convention)
  // Three.js with outputColorSpace=SRGBColorSpace expects linear vertex colors
  // So we pass them through directly without conversion
  let vertexColors: Float32Array | undefined;
  if (hasAnyColor) {
    vertexColors = new Float32Array(allColors.length * 3);
    allColors.forEach((c, i) => {
      // Pass through colors directly - STEP colors are already linear
      vertexColors![i * 3 + 0] = c.r;
      vertexColors![i * 3 + 1] = c.g;
      vertexColors![i * 3 + 2] = c.b;
    });
  }

  tessellationProfile.meshAssembly.total += performance.now() - assemblyStart;
  tessellationProfile.meshAssembly.calls++;

  tessellationProfile.tessellateOCCShape.total += performance.now() - shapeStart;
  tessellationProfile.tessellateOCCShape.calls++;

  // Log color summary
  if (vertexColors) {
    // Count unique colors
    const uniqueColors = new Set<string>();
    for (let i = 0; i < allColors.length; i++) {
      const c = allColors[i];
      uniqueColors.add(`${c.r.toFixed(2)},${c.g.toFixed(2)},${c.b.toFixed(2)}`);
    }
    console.log(`[Tessellate] Vertex colors created: ${vertexColors.length / 3} vertices, ${uniqueColors.size} unique colors`);
  } else {
    console.log('[Tessellate] No vertex colors (using default material color)');
  }

  return { positions, indices, normals, vertexColors };
}

/**
 * Render a Three.js mesh in the render-container element
 */
function renderInContainer(threeMesh: THREE.Mesh) {
  const container = document.getElementById('render-container');
  if (!container) {
    console.error('[Checkpoint 4] render-container element not found');
    return;
  }

  // Clear any existing content
  container.innerHTML = '';

  const width = container.clientWidth;
  const height = container.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x404040);

  // Compute bounding box
  threeMesh.geometry.computeBoundingBox();
  const boundingBox = threeMesh.geometry.boundingBox!;
  const center = new THREE.Vector3();
  boundingBox.getCenter(center);
  const size = new THREE.Vector3();
  boundingBox.getSize(size);

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = 45;
  const cameraDistance = maxDim / (2 * Math.tan((fov * Math.PI) / 360)) * 1.5;

  const camera = new THREE.PerspectiveCamera(fov, width / height, 0.1, cameraDistance * 5);
  camera.position.set(
    center.x + cameraDistance * 0.7,
    center.y + cameraDistance * 0.7,
    center.z + cameraDistance * 0.7
  );
  camera.lookAt(center);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.copy(center);

  // Lights - use multiple lights to ensure all surfaces are visible
  // Ambient light provides base illumination for all surfaces
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  // Hemisphere light for sky/ground gradient
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x888888, 1.0);
  scene.add(hemiLight);

  // Main directional light
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(center.x + cameraDistance, center.y + cameraDistance * 1.5, center.z + cameraDistance);
  scene.add(dirLight);

  // Fill light from opposite side to reduce dark areas
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
  fillLight.position.set(center.x - cameraDistance, center.y - cameraDistance * 0.5, center.z - cameraDistance);
  scene.add(fillLight);

  scene.add(threeMesh);

  // Grid
  const gridSize = Math.ceil(maxDim * 2);
  const grid = new THREE.GridHelper(gridSize, Math.min(gridSize, 20));
  grid.position.y = boundingBox.min.y;
  scene.add(grid);

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

/**
 * Run Checkpoint 4 test - Tessellate and render with Three.js
 */
async function runCheckpoint4(stepFileContent: string): Promise<{
  success: boolean;
  mesh?: Mesh;
  vertexCount?: number;
  triangleCount?: number;
  error?: string;
}> {
  try {
    console.log('[Checkpoint 4] Loading STEP file with OCC...');
    const { shape } = await loadStepFile(stepFileContent, 'test.step');

    console.log('[Checkpoint 4] Extracting faces with edges...');
    const faces = await extractFacesWithEdges(shape);
    console.log(`[Checkpoint 4] Found ${faces.length} faces`);

    console.log('[Checkpoint 4] Tessellating faces...');
    const mesh = await tessellateOCCShape(faces);

    const vertexCount = mesh.positions.length / 3;
    const triangleCount = mesh.indices.length / 3;

    console.log(`[Checkpoint 4] Mesh: ${vertexCount} vertices, ${triangleCount} triangles`);

    // Render with Three.js
    console.log('[Checkpoint 4] Rendering with Three.js...');
    const threeMesh = createThreeMeshFromTesselation(mesh);

    // Override material to basic (no lighting) for debugging winding issues
    threeMesh.material = new THREE.MeshBasicMaterial({
      color: 0x6699ff,
      side: THREE.DoubleSide,
      wireframe: false,
    });

    renderInContainer(threeMesh);

    // For simple-cube.step: expect 6 faces * ~2 triangles each = ~12 triangles
    const success = vertexCount > 0 && triangleCount >= 12;

    if (success) {
      console.log('[Checkpoint 4] ✓ PASSED: Rendered cube with OCC-parsed geometry');
    } else {
      console.log(`[Checkpoint 4] ✗ FAILED: Unexpected mesh size (${vertexCount} verts, ${triangleCount} tris)`);
    }

    return { success, mesh, vertexCount, triangleCount };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('[Checkpoint 4] ✗ ERROR:', errorMsg);
    return { success: false, error: errorMsg };
  }
}

/**
 * Run Checkpoint 5 test - Tessellate cylinder (curved surfaces)
 */
async function runCheckpoint5(stepFileContent: string): Promise<{
  success: boolean;
  faceCount?: number;
  faceTypes?: string[];
  mesh?: Mesh;
  vertexCount?: number;
  triangleCount?: number;
  error?: string;
}> {
  try {
    console.log('[Checkpoint 5] Loading STEP file with OCC...');
    const { shape } = await loadStepFile(stepFileContent, 'test.step');

    console.log('[Checkpoint 5] Extracting faces with edges and surface params...');
    const faces = await extractFacesWithEdges(shape);
    const faceTypes = faces.map(f => f.surfaceType);
    console.log(`[Checkpoint 5] Found ${faces.length} faces: ${faceTypes.join(', ')}`);

    // Check for expected cylinder structure (1 cylinder + 2 planar caps)
    const hasCylinder = faceTypes.includes('Cylinder');

    if (!hasCylinder) {
      console.warn('[Checkpoint 5] No cylindrical surface found in file');
    }

    console.log('[Checkpoint 5] Tessellating faces...');
    const mesh = await tessellateOCCShape(faces);

    const vertexCount = mesh.positions.length / 3;
    const triangleCount = mesh.indices.length / 3;

    console.log(`[Checkpoint 5] Mesh: ${vertexCount} vertices, ${triangleCount} triangles`);

    // Render with Three.js
    console.log('[Checkpoint 5] Rendering with Three.js...');
    const threeMesh = createThreeMeshFromTesselation(mesh);

    // Use basic material (ignores lighting) to debug geometry without normal issues
    threeMesh.material = new THREE.MeshBasicMaterial({
      color: 0x6699ff,
      side: THREE.DoubleSide,
      wireframe: false,
    });

    renderInContainer(threeMesh);

    // Success criteria: has cylinder face, produced triangles
    const success = hasCylinder && vertexCount > 0 && triangleCount > 0;

    if (success) {
      console.log('[Checkpoint 5] ✓ PASSED: Rendered cylinder with OCC-parsed geometry');
    } else {
      console.log(`[Checkpoint 5] ✗ FAILED: hasCylinder=${hasCylinder}, verts=${vertexCount}, tris=${triangleCount}`);
    }

    return { success, faceCount: faces.length, faceTypes, mesh, vertexCount, triangleCount };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('[Checkpoint 5] ✗ ERROR:', errorMsg);
    return { success: false, error: errorMsg };
  }
}

// =============================================================================
// Main Adapter: parseStepWithOCC
// =============================================================================
// This function provides the same interface as parseStepToMesh but uses
// OpenCascade.js for STEP parsing instead of the custom parser.
// It reuses the existing tessellation functions from step-parser.ts and
// surface-tessellation.ts.
// =============================================================================

/**
 * Parse a STEP file using OpenCascade.js and return a Mesh.
 * This is a drop-in replacement for parseStepToMesh() that uses OCC for parsing.
 *
 * Benefits of using OCC:
 * - Handles complex assemblies and transforms
 * - Properly resolves edge orientations from STEP topology
 * - Handles edge cases in STEP files that the custom parser may miss
 *
 * The tessellation still uses our existing GPU-accelerated tessellator.
 */
export async function parseStepWithOCC(
  stepFileContent: Uint8Array | string,
  triangulationMethod: TriangulationMethod = 'ear-clipping',
  onProgress?: (percent: number) => void
): Promise<Mesh> {
  console.log(`[parseStepWithOCC] Starting tessellation...`);
  const startTime = performance.now();

  // Step 1: Load STEP file with OCC (with color support)
  if (onProgress) onProgress(0);
  const loadStart = performance.now();
  const { shape, colorTool, shapeTool, stepColors, shapeColorMap, faceIdOrder, geometryColorMap, solidMatchedColors, faceToSolid, solidToColor } = await loadStepFile(stepFileContent, 'input.step');
  tessellationProfile.loadStepFile.total += performance.now() - loadStart;
  tessellationProfile.loadStepFile.calls++;

  // Log color extraction summary
  console.log(`[parseStepWithOCC] Color sources: shapeColorMap=${shapeColorMap.size}, stepColors=${stepColors.size}, faceIdOrder=${faceIdOrder.length}, geometryColorMap=${geometryColorMap.size}, solidMatchedColors=${solidMatchedColors.size}, faceToSolid=${faceToSolid.size}, solidToColor=${solidToColor.size}, colorTool=${!!colorTool}`);

  // Step 2: Extract faces with edges, surface parameters, and colors
  if (onProgress) onProgress(10);
  const extractStart = performance.now();
  const faces = await extractFacesWithEdges(shape, colorTool, shapeTool, stepColors, shapeColorMap, faceIdOrder, geometryColorMap, solidMatchedColors, faceToSolid, solidToColor);
  tessellationProfile.extractFacesWithEdges.total += performance.now() - extractStart;
  tessellationProfile.extractFacesWithEdges.calls++;

  // Count faces with colors
  const facesWithColor = faces.filter(f => f.color !== undefined).length;
  console.log(`[parseStepWithOCC] Extracted ${faces.length} faces (${facesWithColor} with colors)`);

  // Step 3: Tessellate all faces
  if (onProgress) onProgress(20);
  const mesh = await tessellateOCCShape(faces, triangulationMethod, onProgress);

  const endTime = performance.now();
  console.log(`[parseStepWithOCC] Complete in ${(endTime - startTime).toFixed(0)}ms: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} triangles`);

  return mesh;
}

// =============================================================================
// OCCT-IMPORT-JS Parser (simpler, with working color extraction)
// =============================================================================

interface OcctBrepFace {
  first: number;  // First triangle index
  last: number;   // Last triangle index
  color?: number[]; // RGB color array [r, g, b] in 0-1 range
}

interface OcctMesh {
  attributes: {
    position: { array: number[] };
    normal?: { array: number[] };
  };
  index: { array: number[] };
  color?: number[];  // Mesh-level color
  name?: string;
  brep_faces?: OcctBrepFace[];  // Per-face colors
}

interface OcctResult {
  success: boolean;
  meshes: OcctMesh[];
}

/**
 * Parse a STEP file using occt-import-js with custom color extraction.
 * occt-import-js doesn't extract per-face colors, so we parse them ourselves.
 */
export async function parseStepWithOcctImport(stepFileContent: string): Promise<Mesh> {
  console.log('[parseStepWithOcctImport] Starting...');
  const startTime = performance.now();

  // Parse colors from STEP file using our custom parser
  console.log('[parseStepWithOcctImport] Extracting colors from STEP...');
  const colorStart = performance.now();
  const faceColorMap = buildComprehensiveFaceColorMap(stepFileContent);
  const faceIds = getAdvancedFaceIds(stepFileContent);
  console.log(`[parseStepWithOcctImport] Color extraction: ${faceColorMap.size} colored targets, ${faceIds.length} faces in ${(performance.now() - colorStart).toFixed(0)}ms`);

  // Build ordered face color array (indexed by face position in STEP file)
  const orderedFaceColors: (ParsedRGBColor | null)[] = faceIds.map(id => faceColorMap.get(id) || null);
  const coloredFaceCount = orderedFaceColors.filter(c => c !== null).length;
  console.log(`[parseStepWithOcctImport] ${coloredFaceCount}/${faceIds.length} faces have colors`);

  // Initialize occt-import-js with custom WASM path
  const initStart = performance.now();
  const occt = await occtimportjs({
    locateFile: (file: string) => {
      if (file.endsWith('.wasm')) {
        return '/occt-import-js.wasm';
      }
      return file;
    }
  });
  console.log(`[parseStepWithOcctImport] WASM initialized in ${(performance.now() - initStart).toFixed(0)}ms`);

  // Convert string to Uint8Array
  const encoder = new TextEncoder();
  const fileBuffer = encoder.encode(stepFileContent);

  // Parse the STEP file
  const parseStart = performance.now();
  const result: OcctResult = occt.ReadStepFile(fileBuffer, null);
  console.log(`[parseStepWithOcctImport] Parsed in ${(performance.now() - parseStart).toFixed(0)}ms`);

  if (!result || !result.success) {
    throw new Error('occt-import-js failed to parse STEP file');
  }

  if (!result.meshes || result.meshes.length === 0) {
    throw new Error('occt-import-js returned no meshes');
  }

  console.log(`[parseStepWithOcctImport] Got ${result.meshes.length} meshes`);

  // Count total brep_faces and check for native colors from occt-import-js
  let totalBrepFaces = 0;
  let nativeColorCount = 0;
  let meshesWithColor = 0;
  for (const mesh of result.meshes) {
    if (mesh.color) {
      meshesWithColor++;
    }
    if (mesh.brep_faces) {
      totalBrepFaces += mesh.brep_faces.length;
      for (const face of mesh.brep_faces) {
        if (face.color && face.color.length >= 3) {
          nativeColorCount++;
        }
      }
    }
  }
  console.log(`[parseStepWithOcctImport] Total brep_faces: ${totalBrepFaces} (STEP has ${faceIds.length} ADVANCED_FACE)`);
  console.log(`[parseStepWithOcctImport] Native colors: ${meshesWithColor} meshes with color, ${nativeColorCount}/${totalBrepFaces} faces with native color`);

  // Log first few sample colors from our STEP parser
  const sampleColoredFaces = orderedFaceColors
    .map((c, i) => c ? { index: i, faceId: faceIds[i], color: c } : null)
    .filter(x => x !== null)
    .slice(0, 5);
  if (sampleColoredFaces.length > 0) {
    console.log(`[parseStepWithOcctImport] Sample STEP parser colors:`);
    for (const f of sampleColoredFaces) {
      console.log(`  Face ${f.index} (#${f.faceId}): RGB(${f.color.r.toFixed(2)}, ${f.color.g.toFixed(2)}, ${f.color.b.toFixed(2)})`);
    }
  }

  // Combine all meshes into a single Mesh
  const allPositions: number[] = [];
  const allIndices: number[] = [];
  const allNormals: number[] = [];
  const allColors: RGBColor[] = [];

  let vertexOffset = 0;
  let globalFaceIndex = 0;  // Track face index across all meshes
  let colorsApplied = 0;

  const defaultGray: RGBColor = { r: 0.7, g: 0.7, b: 0.7 };

  for (const mesh of result.meshes) {
    const positions = mesh.attributes.position.array;
    const indices = mesh.index.array;
    const normals = mesh.attributes.normal?.array;

    // Get mesh-level color as default (from occt-import-js, if available)
    const meshDefaultColor: RGBColor = mesh.color
      ? { r: mesh.color[0], g: mesh.color[1], b: mesh.color[2] }
      : defaultGray;

    // Build vertex-to-color mapping from brep_faces using our parsed colors
    const numVertices = positions.length / 3;
    const vertexColors: RGBColor[] = new Array(numVertices);
    for (let i = 0; i < numVertices; i++) {
      vertexColors[i] = meshDefaultColor;
    }

    if (mesh.brep_faces) {
      for (let faceIdx = 0; faceIdx < mesh.brep_faces.length; faceIdx++) {
        const face = mesh.brep_faces[faceIdx];

        // Priority 1: Use native color from occt-import-js if available
        let faceColor: RGBColor = meshDefaultColor;
        let colorSource = 'default';

        if (face.color && face.color.length >= 3) {
          faceColor = { r: face.color[0], g: face.color[1], b: face.color[2] };
          colorsApplied++;
          colorSource = 'native';
        } else if (globalFaceIndex < orderedFaceColors.length) {
          // Priority 2: Fall back to our STEP text parser
          const parsedColor = orderedFaceColors[globalFaceIndex];
          if (parsedColor) {
            faceColor = { r: parsedColor.r, g: parsedColor.g, b: parsedColor.b };
            colorsApplied++;
            colorSource = 'parsed';
          }
        }

        // Apply color to all vertices in this face's triangles
        for (let triIdx = face.first; triIdx <= face.last; triIdx++) {
          const baseIdx = triIdx * 3;
          for (let j = 0; j < 3; j++) {
            const vertexIdx = indices[baseIdx + j];
            if (vertexIdx !== undefined && vertexIdx < numVertices) {
              vertexColors[vertexIdx] = faceColor;
            }
          }
        }

        globalFaceIndex++;
      }
    }

    // Add positions (Z-up to Y-up conversion done by createThreeMeshFromTesselation)
    for (let i = 0; i < positions.length; i++) {
      allPositions.push(positions[i]);
    }

    // Add normals
    if (normals) {
      for (let i = 0; i < normals.length; i++) {
        allNormals.push(normals[i]);
      }
    }

    // Add indices with offset
    for (const idx of indices) {
      allIndices.push(idx + vertexOffset);
    }

    // Add vertex colors
    for (const color of vertexColors) {
      allColors.push(color);
    }

    vertexOffset += numVertices;
  }

  // Track color sources
  const colorSources = { native: 0, parsed: 0 };
  for (const mesh of result.meshes) {
    if (mesh.brep_faces) {
      for (const face of mesh.brep_faces) {
        if (face.color && face.color.length >= 3) {
          colorSources.native++;
        }
      }
    }
  }
  colorSources.parsed = colorsApplied - colorSources.native;
  console.log(`[parseStepWithOcctImport] Color sources: ${colorSources.native} native, ${colorSources.parsed} parsed, ${globalFaceIndex - colorsApplied} default`);
  console.log(`[parseStepWithOcctImport] Applied ${colorsApplied} colors from STEP parser (${globalFaceIndex} total brep_faces)`);
  if (globalFaceIndex !== totalBrepFaces) {
    console.warn(`[parseStepWithOcctImport] Face count mismatch: processed ${globalFaceIndex}, expected ${totalBrepFaces}`);
  }

  // Build the final Mesh object
  const mesh: Mesh = {
    positions: new Float32Array(allPositions),
    indices: new Uint32Array(allIndices),
    normals: allNormals.length > 0 ? new Float32Array(allNormals) : undefined,
    vertexColors: new Float32Array(allColors.length * 3),
  };

  // Fill vertex colors array
  for (let i = 0; i < allColors.length; i++) {
    mesh.vertexColors![i * 3 + 0] = allColors[i].r;
    mesh.vertexColors![i * 3 + 1] = allColors[i].g;
    mesh.vertexColors![i * 3 + 2] = allColors[i].b;
  }

  const endTime = performance.now();
  console.log(`[parseStepWithOcctImport] Complete in ${(endTime - startTime).toFixed(0)}ms: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} triangles`);

  return mesh;
}

/**
 * Parse STEP file using occt-import-js for comparison tests.
 * Returns a simplified result suitable for comparison with our OCC tessellator.
 */
export async function parseStepWithOcctImportSimple(stepFileContent: string | Uint8Array): Promise<{
  success: boolean;
  meshCount: number;
  vertexCount: number;
  triangleCount: number;
  boundingBox: { min: [number, number, number]; max: [number, number, number] };
  totalTime: number;
}> {
  const startTime = performance.now();

  // Initialize occt-import-js with custom WASM path
  // The WASM file is in the public directory to ensure it's served correctly
  const occt = await occtimportjs({
    locateFile: (file: string) => {
      if (file.endsWith('.wasm')) {
        return '/occt-import-js.wasm';
      }
      return file;
    }
  });

  // Convert to Uint8Array if needed
  let fileBuffer: Uint8Array;
  if (typeof stepFileContent === 'string') {
    fileBuffer = new TextEncoder().encode(stepFileContent);
  } else {
    fileBuffer = stepFileContent;
  }

  // Parse the STEP file
  const result = occt.ReadStepFile(fileBuffer, null);
  const parseTime = performance.now() - startTime;

  if (!result || !result.success) {
    return {
      success: false,
      meshCount: 0,
      vertexCount: 0,
      triangleCount: 0,
      boundingBox: { min: [0, 0, 0], max: [0, 0, 0] },
      totalTime: parseTime,
    };
  }

  if (!result.meshes || result.meshes.length === 0) {
    return {
      success: false,
      meshCount: 0,
      vertexCount: 0,
      triangleCount: 0,
      boundingBox: { min: [0, 0, 0], max: [0, 0, 0] },
      totalTime: parseTime,
    };
  }

  // Count totals
  let totalVertices = 0;
  let totalIndices = 0;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const mesh of result.meshes) {
    totalVertices += mesh.attributes.position.array.length / 3;
    totalIndices += mesh.index.array.length;

    // Compute bounding box
    const pos = mesh.attributes.position.array;
    for (let i = 0; i < pos.length; i += 3) {
      minX = Math.min(minX, pos[i]);
      maxX = Math.max(maxX, pos[i]);
      minY = Math.min(minY, pos[i + 1]);
      maxY = Math.max(maxY, pos[i + 1]);
      minZ = Math.min(minZ, pos[i + 2]);
      maxZ = Math.max(maxZ, pos[i + 2]);
    }
  }

  return {
    success: true,
    meshCount: result.meshes.length,
    vertexCount: totalVertices,
    triangleCount: totalIndices / 3,
    boundingBox: {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    },
    totalTime: parseTime,
  };
}

export async function parseStepWithOcctImportFaceDiagnostics(
  stepFileContent: string | Uint8Array
): Promise<OcctImportFaceDiagnostic[]> {
  const occt = await occtimportjs({
    locateFile: (file: string) => {
      if (file.endsWith('.wasm')) {
        return '/occt-import-js.wasm';
      }
      return file;
    }
  });

  const fileBuffer = typeof stepFileContent === 'string'
    ? new TextEncoder().encode(stepFileContent)
    : stepFileContent;
  const result: OcctResult = occt.ReadStepFile(fileBuffer, null);
  if (!result || !result.success || !result.meshes || result.meshes.length === 0) {
    throw new Error('occt-import-js returned no meshes while building face diagnostics');
  }

  const rows: OcctImportFaceDiagnostic[] = [];
  let globalFaceIndex = 0;
  for (let meshIndex = 0; meshIndex < result.meshes.length; meshIndex++) {
    const mesh = result.meshes[meshIndex];
    const brepFaces = mesh.brep_faces || [];
    for (const face of brepFaces) {
      const firstTriangle = Number.isFinite(face.first) ? face.first : 0;
      const lastTriangle = Number.isFinite(face.last) ? face.last : -1;
      const triangleCount = Math.max(0, lastTriangle - firstTriangle + 1);
      rows.push({
        globalFaceIndex,
        meshIndex,
        meshName: mesh.name,
        firstTriangle,
        lastTriangle,
        triangleCount,
        hasNativeColor: !!(face.color && face.color.length >= 3),
      });
      globalFaceIndex++;
    }
  }

  return rows;
}

export async function buildFaceDiffReport(
  stepFileContent: Uint8Array | string,
  options?: {
    targetFaceIndices?: number[];
    triangulationMethod?: TriangulationMethod;
  }
): Promise<FaceDiffReport> {
  const triangulationMethod = options?.triangulationMethod ?? 'ear-clipping';

  const { shape, colorTool, shapeTool, stepColors, shapeColorMap, faceIdOrder, geometryColorMap, solidMatchedColors, faceToSolid, solidToColor } =
    await loadStepFile(stepFileContent, 'input.step');
  const faces = await extractFacesWithEdges(
    shape,
    colorTool,
    shapeTool,
    stepColors,
    shapeColorMap,
    faceIdOrder,
    geometryColorMap,
    solidMatchedColors,
    faceToSolid,
    solidToColor
  );

  const normalizedTargetFaceIndices = options?.targetFaceIndices
    ? Array.from(new Set(
      options.targetFaceIndices
        .filter((value) => Number.isFinite(value))
        .map((value) => Math.trunc(value))
        .filter((value) => value >= 0)
    )).sort((a, b) => a - b)
    : undefined;
  const targetFaceSet = normalizedTargetFaceIndices && normalizedTargetFaceIndices.length > 0
    ? new Set<number>(normalizedTargetFaceIndices)
    : undefined;

  const oursDiagnostics: OccFaceTessellationDiagnostic[] = [];
  await tessellateOCCShape(
    faces,
    triangulationMethod,
    undefined,
    oursDiagnostics,
    faceIdOrder,
    targetFaceSet
  );

  const referenceDiagnostics = await parseStepWithOcctImportFaceDiagnostics(stepFileContent);
  const oursByFaceIndex = new Map<number, OccFaceTessellationDiagnostic>();
  for (const row of oursDiagnostics) {
    oursByFaceIndex.set(row.faceIndex, row);
  }
  const referenceByFaceIndex = new Map<number, OcctImportFaceDiagnostic>();
  for (const row of referenceDiagnostics) {
    referenceByFaceIndex.set(row.globalFaceIndex, row);
  }

  const targetFaceIndices = normalizedTargetFaceIndices && normalizedTargetFaceIndices.length > 0
    ? normalizedTargetFaceIndices
    : Array.from(new Set([
      ...Array.from(oursByFaceIndex.keys()),
      ...Array.from(referenceByFaceIndex.keys()),
    ])).sort((a, b) => a - b);

  const rows: FaceDiffRow[] = targetFaceIndices.map((faceIndex) => {
    const ours = oursByFaceIndex.get(faceIndex);
    const reference = referenceByFaceIndex.get(faceIndex);
    const triangleDelta =
      ours && reference
        ? ours.outputTriangleCount - reference.triangleCount
        : undefined;
    const triangleDeltaPct =
      triangleDelta !== undefined && reference && reference.triangleCount > 0
        ? (triangleDelta / reference.triangleCount) * 100
        : undefined;

    return {
      faceIndex,
      stepFaceId: ours?.stepFaceId ?? faceIdOrder[faceIndex],
      ours,
      reference,
      triangleDelta,
      triangleDeltaPct,
    };
  });

  const oursTriangles = rows.reduce((acc, row) => {
    if (!row.ours || row.ours.status !== 'ok') return acc;
    return acc + row.ours.outputTriangleCount;
  }, 0);
  const referenceTriangles = rows.reduce((acc, row) => acc + (row.reference?.triangleCount ?? 0), 0);
  const oursFaceCount = rows.reduce((acc, row) => acc + (row.ours ? 1 : 0), 0);
  const referenceFaceCount = rows.reduce((acc, row) => acc + (row.reference ? 1 : 0), 0);

  return {
    generatedAtIso: new Date().toISOString(),
    targetFaceIndices,
    assumptions: [
      'Face index alignment assumes OCC faceIndex order matches occt-import-js brep_faces global order.',
      'Rows with missing reference data usually indicate meshes without brep_faces entries.',
    ],
    totals: {
      oursFaceCount,
      referenceFaceCount,
      oursTriangles,
      referenceTriangles,
    },
    rows,
  };
}

/**
 * Parse STEP file with occt-import-js and return full mesh data for rendering.
 * This is used by the visual validation harness to render occt-import-js output.
 */
export async function parseStepWithOcctImportFull(stepFileContent: string | Uint8Array): Promise<{
  success: boolean;
  meshCount: number;
  vertexCount: number;
  triangleCount: number;
  totalTime: number;
  mesh?: {
    positions: Float32Array;
    normals: Float32Array;
    indices: Uint32Array;
  };
  error?: string;
}> {
  const startTime = performance.now();

  try {
    // Initialize occt-import-js with custom WASM path
    const occt = await occtimportjs({
      locateFile: (file: string) => {
        if (file.endsWith('.wasm')) {
          return '/occt-import-js.wasm';
        }
        return file;
      }
    });

    // Convert to Uint8Array if needed
    let fileBuffer: Uint8Array;
    if (typeof stepFileContent === 'string') {
      fileBuffer = new TextEncoder().encode(stepFileContent);
    } else {
      fileBuffer = stepFileContent;
    }

    // Parse the STEP file
    const result = occt.ReadStepFile(fileBuffer, null);
    const parseTime = performance.now() - startTime;

    if (!result || !result.success || !result.meshes || result.meshes.length === 0) {
      return {
        success: false,
        meshCount: 0,
        vertexCount: 0,
        triangleCount: 0,
        totalTime: parseTime,
        error: 'No meshes found',
      };
    }

    // Combine all meshes into a single mesh for rendering
    let totalVertices = 0;
    let totalIndices = 0;

    // First pass: count totals
    for (const mesh of result.meshes) {
      totalVertices += mesh.attributes.position.array.length / 3;
      totalIndices += mesh.index.array.length;
    }

    // Allocate combined arrays
    const positions = new Float32Array(totalVertices * 3);
    const normals = new Float32Array(totalVertices * 3);
    const indices = new Uint32Array(totalIndices);

    // Second pass: copy data
    let vertexOffset = 0;
    let indexOffset = 0;
    let vertexIndexOffset = 0;

    for (const mesh of result.meshes) {
      const pos = mesh.attributes.position.array;
      const norm = mesh.attributes.normal?.array;
      const idx = mesh.index.array;

      // Copy positions
      positions.set(pos, vertexOffset * 3);

      // Copy normals (or compute later if not available)
      if (norm) {
        normals.set(norm, vertexOffset * 3);
      }

      // Copy indices with offset
      for (let i = 0; i < idx.length; i++) {
        indices[indexOffset + i] = idx[i] + vertexIndexOffset;
      }

      vertexOffset += pos.length / 3;
      vertexIndexOffset += pos.length / 3;
      indexOffset += idx.length;
    }

    return {
      success: true,
      meshCount: result.meshes.length,
      vertexCount: totalVertices,
      triangleCount: totalIndices / 3,
      totalTime: parseTime,
      mesh: {
        positions,
        normals,
        indices,
      },
    };
  } catch (e) {
    return {
      success: false,
      meshCount: 0,
      vertexCount: 0,
      triangleCount: 0,
      totalTime: performance.now() - startTime,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

// Export for use in browser
export { initOC, loadStepFile, countFaces, runCheckpoint1, extractSurfaces, runCheckpoint2, extractFacesWithEdges, runCheckpoint3, runCheckpoint4, runCheckpoint5 };
