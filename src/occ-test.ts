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
import { bridgeAllHoles } from './step-parser';
import { createThreeMeshFromTesselation } from './threejs-render';
import * as occtimportjsModule from 'occt-import-js';
const occtimportjs = (occtimportjsModule as any).default || occtimportjsModule;
import { buildComprehensiveFaceColorMap, getAdvancedFaceIds, type RGBColor as ParsedRGBColor } from './step-color-parser';

// Profiling accumulator for tessellation functions
export const tessellationProfile = {
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
  const total = tessellationProfile.tessellateOCCShape.total || 1;

  lines.push('=== TESSELLATION PROFILE ===');
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
} from './surface-tessellation';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Set to true to enable debug logging (significantly impacts performance)
const DEBUG_OCC = false;

function logOCC(...args: unknown[]): void {
  if (DEBUG_OCC) {
    console.log('[OCC]', ...args);
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
  colorToolInput: any
): Map<number, RGBColor> {
  const faceColorMap = new Map<number, RGBColor>();

  if (!shapeToolInput || !colorToolInput) {
    console.log('[ShapeColorMap] Missing shapeTool or colorTool');
    return faceColorMap;
  }

  // colorTool and shapeTool might be Handles - try to unwrap them
  let colorTool = colorToolInput;
  if (typeof colorToolInput.get === 'function') {
    try {
      colorTool = colorToolInput.get();
      console.log('[ShapeColorMap] Unwrapped colorTool handle');
    } catch (e) {
      console.log('[ShapeColorMap] Failed to unwrap colorTool:', e);
    }
  }

  let shapeTool = shapeToolInput;
  if (typeof shapeToolInput.get === 'function') {
    try {
      shapeTool = shapeToolInput.get();
      console.log('[ShapeColorMap] Unwrapped shapeTool handle');
    } catch (e) {
      console.log('[ShapeColorMap] Failed to unwrap shapeTool:', e);
    }
  }

  // Log available colorTool methods for debugging (use prototype, not Object.keys)
  const colorMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(colorTool) || {})
    .filter(k => typeof colorTool[k] === 'function');
  console.log('[ShapeColorMap] colorTool methods:', colorMethods.slice(0, 20).join(', '));
  console.log('[ShapeColorMap] colorTool method count:', colorMethods.length);

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
        console.log(`[ShapeColorMap] Colors defined in XCAF document: ${docColorCount}`);
      } else {
        // List available TDF_LabelSequence APIs
        const tdfApis = Object.keys(oc).filter(k => k.includes('TDF_LabelSequence'));
        console.log('[ShapeColorMap] Available TDF_LabelSequence APIs:', tdfApis.join(', '));
      }
    }
  } catch (e) {
    console.log('[ShapeColorMap] GetColors failed:', e);
    // List available TDF_LabelSequence APIs on error
    const tdfApis = Object.keys(oc).filter(k => k.includes('TDF_LabelSequence'));
    console.log('[ShapeColorMap] Available TDF_LabelSequence APIs:', tdfApis.join(', '));
  }

  let colorsFoundViaShape = 0;
  let colorsFoundViaLabel = 0;

  // Helper to get color from a shape directly
  // Shape-based GetColor methods: GetColor_6, GetColor_7, GetColor_8
  // Signature: (shape, colorType, out_color) → 3 args, returns bool
  let shapeColorAttempts = 0;
  let shapeColorErrors: string[] = [];

  const getShapeColor = (shape: any): RGBColor | null => {
    if (!shape || shape.IsNull()) return null;

    const color = new oc.Quantity_Color_1();
    // XCAFDoc_ColorType: 0=Gen, 1=Surf, 2=Curv
    for (const colorType of [1, 0, 2]) {
      // Try shape-based GetColor variants (GetColor_6, GetColor_7, GetColor_8 are shape-based)
      for (const methodName of ['GetColor_6', 'GetColor_7', 'GetColor_8']) {
        try {
          if (typeof colorTool[methodName] === 'function') {
            shapeColorAttempts++;
            const hasColor = colorTool[methodName](shape, colorType, color);
            if (hasColor) {
              colorsFoundViaShape++;
              return { r: color.Red(), g: color.Green(), b: color.Blue() };
            }
          }
        } catch (e: any) {
          if (shapeColorErrors.length < 3) {
            shapeColorErrors.push(`${methodName}(colorType=${colorType}): ${e.message || e}`);
          }
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
      // Get shape for this label
      const shape = shapeTool.GetShape(label);
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

      if (color) {
        labelsWithColor++;
        if (labelsWithColor <= 5) {
          console.log(`[processLabel] Found color at depth ${depth} via ${colorSource}: RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)})`);
        }
      }

      // Use inherited color if no direct color
      const effectiveColor = color || inheritedColor;

      // If we have a color and a shape, map all faces to this color
      if (effectiveColor && shape && !shape.IsNull()) {
        mapFacesToColor(shape, effectiveColor);
      }

      // Traverse children
      if (oc.TDF_ChildIterator_1) {
        const childIter = new oc.TDF_ChildIterator_1(label, false);
        while (childIter.More()) {
          processLabel(childIter.Value(), effectiveColor, depth + 1);
          childIter.Next();
        }
      }
    } catch (e) {
      // Silent fail
    }
  };

  // DEBUG: List available methods on shapeTool and colorTool
  console.log('[ShapeColorMap] shapeTool methods:', Object.keys(shapeTool).filter(k => typeof shapeTool[k] === 'function').slice(0, 30));
  console.log('[ShapeColorMap] colorTool methods:', Object.keys(colorTool).filter(k => typeof colorTool[k] === 'function').slice(0, 30));

  // NEW: Try to get the main label and traverse from there
  try {
    // Get the main label from the document
    const mainLabel = shapeToolInput.BaseLabel ? shapeToolInput.BaseLabel() : null;
    if (mainLabel) {
      console.log('[ShapeColorMap] BaseLabel found, checking for colors...');

      // Recursive function to traverse label tree
      const traverseForColors = (label: any, depth: number) => {
        if (depth > 10) return;

        // Check if this label has a shape
        let labelShape = null;
        try {
          if (shapeTool.GetShape) {
            labelShape = shapeTool.GetShape(label);
          }
        } catch (e) {}

        // Check if this label has a color
        const labelColor = getLabelColor(label);
        if (labelColor && depth < 5) {
          console.log(`[ShapeColorMap] Label at depth ${depth} has color: RGB(${labelColor.r.toFixed(2)}, ${labelColor.g.toFixed(2)}, ${labelColor.b.toFixed(2)})`);

          // If it has a shape, map the shape's faces to this color
          if (labelShape && !labelShape.IsNull()) {
            mapFacesToColor(labelShape, labelColor);
          }
        }

        // Traverse children using TDF_ChildIterator
        try {
          if (oc.TDF_ChildIterator_1) {
            const childIter = new oc.TDF_ChildIterator_1(label, false);
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
    console.log('[ShapeColorMap] BaseLabel traversal failed:', e);
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
        console.log(`[ShapeColorMap] Solid 0 getShapeColor result: ${solidColor ? 'found' : 'null'}`);
        console.log(`[ShapeColorMap] Solid 0 HashCode: ${solid.HashCode(2147483647)}`);
        console.log(`[ShapeColorMap] shapeTool.FindShape available: ${typeof shapeTool.FindShape}`);
      }

      // If direct shape lookup fails, try finding the label for this shape
      if (!solidColor && shapeTool) {
        // FindShape signature: (shape, out_label, findInstance) - 3 args with output param
        // Try different FindShape variants
        for (const findMethodName of ['FindShape', 'FindShape_1', 'FindShape_2']) {
          if (typeof shapeTool[findMethodName] === 'function') {
            try {
              // Create output label - FindShape fills this with the result
              const solidLabel = new oc.TDF_Label();
              // Call with 3 args: shape, output label, findInstance flag
              const found = shapeTool[findMethodName](solid, solidLabel, false);
              if (found && solidLabel && !solidLabel.IsNull()) {
                solidColor = getLabelColor(solidLabel);
                if (solidColor && solidIndex < 3) {
                  console.log(`[ShapeColorMap] Solid ${solidIndex} found color via ${findMethodName} label lookup`);
                }
                break;
              } else if (solidIndex === 0) {
                console.log(`[ShapeColorMap] Solid 0: ${findMethodName} returned found=${found}, label null/empty`);
              }
            } catch (e) {
              if (solidIndex === 0) {
                console.log(`[ShapeColorMap] Solid 0: ${findMethodName} threw error:`, e);
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
          console.log(`[ShapeColorMap] Solid ${solidIndex} has XCAF color: RGB(${solidColor.r.toFixed(2)}, ${solidColor.g.toFixed(2)}, ${solidColor.b.toFixed(2)})`);
        }
      } else if (solidIndex < 3) {
        // Debug: check if IsSet returns true for this solid
        for (const ct of [0, 1, 2]) {
          try {
            if (colorTool.IsSet_1) {
              const isSet = colorTool.IsSet_1(solid, ct);
              if (isSet) console.log(`[ShapeColorMap] Solid ${solidIndex}: IsSet_1(${ct}) = true`);
            }
          } catch (e) {}
        }
      }

      solidIndex++;
      solidExplorer.Next();
    }

    console.log(`[ShapeColorMap] XCAF solid colors: ${solidsWithColors}/${solidIndex} solids have colors, ${facesColoredFromSolids} faces colored`);
  } catch (e) {
    console.log('[ShapeColorMap] Error iterating solids for XCAF colors:', e);
  }

  // Fallback: Try TDF_LabelSequence approach (may fail if not available)
  try {
    const labels = new oc.TDF_LabelSequence_1();
    shapeTool.GetFreeShapes(labels);

    console.log(`[ShapeColorMap] Processing ${labels.Length()} free shapes for face colors...`);

    for (let i = 1; i <= labels.Length(); i++) {
      const label = labels.Value(i);
      processLabel(label, null, 0);
    }

    console.log(`[ShapeColorMap] Label traversal stats: ${labelsProcessed} processed, ${labelsWithShape} with shapes, ${labelsWithColor} with colors`);

    // Also check colorTool's GetColors for any additional colored shapes
    if (colorTool.GetColors) {
      const colorLabels = new oc.TDF_LabelSequence_1();
      colorTool.GetColors(colorLabels);
      // Color labels found in document - could be used for additional shape lookups
    }
  } catch (e) {
    // TDF_LabelSequence not available - expected in OpenCascade.js
    console.log('[ShapeColorMap] TDF_LabelSequence fallback not available (expected)');
  }

  // Log summary of what was found
  console.log(`[ShapeColorMap] Color extraction stats: ${colorsFoundViaLabel} via label, ${colorsFoundViaShape} via shape`);
  console.log(`[ShapeColorMap] getShapeColor attempts: ${shapeColorAttempts}`);
  if (shapeColorErrors.length > 0) {
    console.log(`[ShapeColorMap] getShapeColor errors (first 3): ${shapeColorErrors.join('; ')}`);
  }
  if (faceColorMap.size > 0) {
    const uniqueColors = new Set<string>();
    for (const color of faceColorMap.values()) {
      uniqueColors.add(`${color.r.toFixed(2)},${color.g.toFixed(2)},${color.b.toFixed(2)}`);
    }
    console.log(`[ShapeColorMap] Found ${faceColorMap.size} face->color mappings with ${uniqueColors.size} unique colors`);
  } else {
    console.log('[ShapeColorMap] No colors extracted from XCAF labels');
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

    console.log(`[OCC_Solids] Found ${solidMap.size} solids, ${faceToSolid.size} face→solid mappings`);
    for (const [idx, data] of solidMap) {
      console.log(`[OCC_Solids] Solid ${idx}: ${data.faceCount} faces`);
    }
  } catch (e) {
    console.log('[OCC_Solids] Error iterating solids:', e);
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

      console.log(`[SolidMatch] Matched STEP #${stepSolids[0].solidId} to OCC solid ${occSolidIdx} (${faceCount} faces) -> RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)})`);
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
      console.log(`[SolidMatch] Matched ${minCount} solids with ${faceCount} faces each (ambiguous)`);
    }
  }

  console.log(`[SolidMatch] Total: ${matchedSolids} solids matched, ${matchedFaces} faces colored`);
  return { faceColorMap, solidToColor };
}

/**
 * Load a STEP file and return the TopoDS_Shape with color information
 */
async function loadStepFile(fileContent: string, fileName: string): Promise<StepLoadResult> {
  const oc = await initOC();

  // Debug APIs logged only when DEBUG_OCC is true
  if (DEBUG_OCC) {
    console.log('[OCC] Available STEPCAFControl_Reader constructors:',
      Object.keys(oc).filter(k => k.startsWith('STEPCAFControl_Reader')));
    console.log('[OCC] Available XCAFDoc APIs:',
      Object.keys(oc).filter(k => k.startsWith('XCAFDoc')).slice(0, 20));
  }

  // Write file to virtual filesystem
  oc.FS.createDataFile('/', fileName, fileContent, true, true, true);

  let shape: any = null;
  let colorTool: any = null;
  let shapeTool: any = null;
  let doc: any = null;

  // Try XCAF reader first (supports colors)
  const hasXCAF = oc.STEPCAFControl_Reader_1 || oc.STEPCAFControl_Reader;

  if (hasXCAF) {
    try {
      logOCC('Using STEPCAFControl_Reader for color support...');

      // Create XDE document using Handle (correct OpenCascade.js API)
      const app = new oc.TDocStd_Application();
      const docHandle = new oc.Handle_TDocStd_Document_1();
      // TCollection_ExtendedString_2 requires (string, isMultiByte) parameters
      app.NewDocument(new oc.TCollection_ExtendedString_2("XmlXCAF", true), docHandle);

      if (docHandle.IsNull()) {
        throw new Error('Failed to create XCAF document');
      }
      doc = docHandle.get();

      // Create XCAF STEP reader
      let cafReader;
      if (oc.STEPCAFControl_Reader_1) {
        cafReader = new oc.STEPCAFControl_Reader_1();
      } else {
        cafReader = new oc.STEPCAFControl_Reader();
      }

      // Enable all relevant reading modes
      if (cafReader.SetColorMode) {
        cafReader.SetColorMode(true);
        console.log('[XCAF] SetColorMode(true)');
      }
      if (cafReader.SetLayerMode) {
        cafReader.SetLayerMode(true);
        console.log('[XCAF] SetLayerMode(true)');
      }
      if (cafReader.SetNameMode) {
        cafReader.SetNameMode(true);
        console.log('[XCAF] SetNameMode(true)');
      }
      if (cafReader.SetMatMode) {
        cafReader.SetMatMode(true);
        console.log('[XCAF] SetMatMode(true) - materials');
      }
      if (cafReader.SetGDTMode) {
        cafReader.SetGDTMode(true);
        console.log('[XCAF] SetGDTMode(true)');
      }

      // Log available reader methods for debugging
      const readerMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(cafReader) || {})
        .filter(k => typeof cafReader[k] === 'function');
      console.log('[XCAF] Reader methods:', readerMethods.filter(m => m.startsWith('Set') || m.startsWith('Get')).join(', '));

      // Read file
      const readResult = cafReader.ReadFile(fileName);
      console.log('[XCAF] ReadFile result:', readResult, 'type:', typeof readResult, 'value:', readResult?.value);

      // IFSelect_RetDone is typically 1 in OpenCascade (0=RetVoid, 1=RetDone, 2=RetError...)
      // Check for value 1 (RetDone) or value 0 if enum mapping differs
      const isDone = (typeof readResult === 'object' && (readResult.value === 1 || readResult.value === 0)) ||
                     readResult === 1 || readResult === 0;

      if (isDone) {
        // Transfer to document (use handle for Transfer)
        // Beta v2 requires progress range as second argument
        const progressRange = new oc.Message_ProgressRange_1();
        console.log('[XCAF] Calling Transfer to build shapes...');
        let transferResult: any;
        try {
          if (cafReader.Transfer_1) {
            transferResult = cafReader.Transfer_1(docHandle, progressRange);
            console.log('[XCAF] Transfer_1 result:', transferResult);
          } else if (cafReader.Transfer) {
            transferResult = cafReader.Transfer(docHandle, progressRange);
            console.log('[XCAF] Transfer result:', transferResult);
          } else {
            console.error('[XCAF] No Transfer method available!');
          }
        } catch (transferErr: any) {
          console.error('[XCAF] Transfer failed:', transferErr.message || transferErr);
        }

        // Check how many roots were transferred
        if (cafReader.NbRootsForTransfer) {
          try {
            const nbRoots = cafReader.NbRootsForTransfer();
            console.log('[XCAF] NbRootsForTransfer:', nbRoots);
          } catch (e) { /* ignore */ }
        }

        // === CHECK TRANSFER APIs for STEP entity to shape mapping ===
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
        // === END CHECK TRANSFER APIs ===

        // Get tools from document
        shapeTool = oc.XCAFDoc_DocumentTool.ShapeTool(doc.Main());
        colorTool = oc.XCAFDoc_DocumentTool.ColorTool(doc.Main());

        logOCC('Got shapeTool:', !!shapeTool, 'colorTool:', !!colorTool);

        // DIAGNOSTIC: Investigate colorTool APIs
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
          console.log('[ColorDiag] Trying alternative shape extraction...');

          // Check what methods the reader has
          const readerMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(cafReader) || {})
            .filter(k => typeof cafReader[k] === 'function');
          console.log('[ColorDiag] cafReader methods:', readerMethods.slice(0, 20).join(', '));

          // Try Reader().OneShape() pattern
          if (cafReader.Reader && typeof cafReader.Reader === 'function') {
            try {
              const innerReader = cafReader.Reader();
              if (innerReader && innerReader.OneShape) {
                shape = innerReader.OneShape();
                const isNull = shape?.IsNull?.() ?? true;
                const shapeType = shape?.ShapeType?.() ?? 'unknown';
                console.log('[ColorDiag] Got shape from cafReader.Reader().OneShape()',
                  'IsNull:', isNull, 'ShapeType:', shapeType);
              }
            } catch (e) {
              console.log('[ColorDiag] cafReader.Reader().OneShape() failed:', e);
            }
          }

          // Try NbRootsForTransfer pattern
          if ((!shape || shape.IsNull?.()) && cafReader.NbRootsForTransfer) {
            try {
              const numRoots = cafReader.NbRootsForTransfer();
              console.log('[ColorDiag] NbRootsForTransfer:', numRoots);
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

    const readResult = reader.ReadFile(fileName);
    logOCC('ReadFile result:', readResult);

    // IFSelect_RetDone is typically 0 in OpenCascade
    // ReadFile returns an object with .value in emscripten bindings
    const isDone = readResult === oc.IFSelect_ReturnStatus?.IFSelect_RetDone ||
                   readResult === 0 ||
                   (typeof readResult === 'object' && (readResult.value === 0 || readResult.value === 1));
    if (!isDone) {
      oc.FS.unlink(fileName);
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
  oc.FS.unlink(fileName);

  // Parse colors directly from STEP text as fallback
  logOCC('Parsing colors from STEP text...');
  const colorEntities = parseStepColors(fileContent);
  const stepColors = buildFaceColorMap(fileContent, colorEntities);

  // Extract face ID order from STEP text (correlates OCC face index to STEP entity ID)
  const faceIdOrder = extractFaceIdOrder(fileContent);

  // Build geometry-based color map for more reliable face matching
  const geometryColorMap = buildGeometryColorMap(fileContent, stepColors);

  // Build solid-level color map for solid-based matching
  const solidColorMap = buildSolidColorMap(fileContent, colorEntities);

  // Build OCC solid face counts for matching (also returns faceToSolid map)
  const { solidMap: occSolidFaceCounts, faceToSolid } = buildOCCSolidFaceCounts(oc, shape);

  // Match OCC solids to STEP solids by face count and build face->color map
  const { faceColorMap: solidMatchedColors, solidToColor } = matchSolidsAndBuildColorMap(oc, shape, solidColorMap, occSolidFaceCounts);

  // Run comprehensive XCAF diagnostic to understand what's available
  diagnoseXCAFColorExtraction(oc, shape, colorTool, shapeTool);

  // Build shape color map from XCAF labels for color propagation
  const shapeColorMap = buildShapeColorMap(oc, shape, shapeTool, colorTool);

  // IMPORTANT: Unwrap colorTool and shapeTool handles before returning
  // The tools from XCAFDoc_DocumentTool are Handles - getFaceColor needs the actual tool
  let unwrappedColorTool = colorTool;
  let unwrappedShapeTool = shapeTool;

  if (colorTool && typeof colorTool.get === 'function' && !colorTool.IsNull?.()) {
    try {
      unwrappedColorTool = colorTool.get();
      console.log('[loadStepFile] Unwrapped colorTool for return');
    } catch (e) {
      console.log('[loadStepFile] Failed to unwrap colorTool:', e);
    }
  }

  if (shapeTool && typeof shapeTool.get === 'function' && !shapeTool.IsNull?.()) {
    try {
      unwrappedShapeTool = shapeTool.get();
      console.log('[loadStepFile] Unwrapped shapeTool for return');
    } catch (e) {
      console.log('[loadStepFile] Failed to unwrap shapeTool:', e);
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
  occSurface?: any; // Handle<Geom_Surface> (used for UV projection on curved faces)
  surfaceParams?: SurfaceParams; // For curved surfaces
  color?: RGBColor; // Face color from STEP styling
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

      let edgeIndex = 0;
      while (edgeExplorer.More()) {
        const edgeShape = edgeExplorer.Current();

        try {
          // Use BRepAdaptor_Curve to get curve information from the edge
          let curveType = 'Unknown';
          let startPoint = { x: 0, y: 0, z: 0 };
          let endPoint = { x: 0, y: 0, z: 0 };

          // Cast to TopoDS_Edge
          const edge = oc.TopoDS.Edge_1(edgeShape);

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
              const MIN_SAMPLES = 32;
              const MAX_SAMPLES = 256;
              let numSamples = 64; // Default for curved edges (better detail)

              if (curveType === 'Circle') {
                // Angle step ~5.6 degrees (π/32) => ~64 samples for full circle
                const angleStep = Math.PI / 32;
                numSamples = Math.ceil(Math.abs(paramRange) / angleStep);
              } else if (curveType === 'Ellipse') {
                numSamples = 96;
              } else if (curveType === 'BSplineCurve') {
                numSamples = 96;
              }

              if (!isFinite(numSamples)) numSamples = 64;
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

            }
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

      wires.push(edges);

      wireExplorer.Next();
      wireIndex++;
    }

  } catch (e) {
    console.error(`[OCC] Error extracting edges for face ${faceIndex}:`, e);
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

  // Wire iteration order isn't reliable; pick the outer loop as the longest wire.
  let outerIdx = 0;
  let bestLen = -Infinity;
  for (let i = 0; i < wires.length; i++) {
    const len = wireApproxLength(wires[i]);
    if (len > bestLen) {
      bestLen = len;
      outerIdx = i;
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
        occSurface: surface,
        surfaceParams,
        color
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
function occEdgesToPolygon(edges: EdgeInfo[]): Vec3[] {
  if (edges.length === 0) return [];

  const polygon: Vec3[] = [];

  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];

    if (edge.sampledPoints && edge.sampledPoints.length > 0) {
      // For curved edges, use sampled points (skip last point to avoid duplicate with next edge's start)
      for (let j = 0; j < edge.sampledPoints.length - 1; j++) {
        const pt = edge.sampledPoints[j];
        polygon.push([pt.x, pt.y, pt.z]);
      }
    } else {
      // For line edges, just use the start point
      polygon.push([edge.startPoint.x, edge.startPoint.y, edge.startPoint.z]);
    }
  }

  return polygon;
}

/**
 * Tessellate a single planar face from OCC data using GPU ear-clipping
 * Each sub-function is profiled for performance analysis
 */
async function tessellatePlanarFaceFromOCC(face: FaceWithEdgesInfo): Promise<{
  vertices: Vec3[];
  triangles: number[][];
}> {
  const faceStart = performance.now();

  // 1. Convert edge data to polygon vertices
  let t0 = performance.now();
  const outer: Vec3[] = occEdgesToPolygon(face.outerLoop);
  const holes: Vec3[][] = face.innerLoops.map(loop => occEdgesToPolygon(loop));
  tessellationProfile.occEdgesToPolygon.total += performance.now() - t0;
  tessellationProfile.occEdgesToPolygon.calls++;

  if (outer.length < 3) {
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

  // 6. Bridge holes into outer polygon (required for GPU ear-clipping)
  t0 = performance.now();
  const mergedPolygon2d = bridgeAllHoles(normalized.outer2d, normalized.holes2d);
  tessellationProfile.bridgeAllHoles.total += performance.now() - t0;
  tessellationProfile.bridgeAllHoles.calls++;

  // Build 2D→3D lookup for merged polygon
  const lookup = new Map<string, Vec3>();
  for (let i = 0; i < normalized.outer2d.length; i++) {
    const key = `${normalized.outer2d[i][0].toFixed(9)},${normalized.outer2d[i][1].toFixed(9)}`;
    lookup.set(key, oriented3d.outer[i]);
  }
  for (let h = 0; h < normalized.holes2d.length; h++) {
    for (let i = 0; i < normalized.holes2d[h].length; i++) {
      const key = `${normalized.holes2d[h][i][0].toFixed(9)},${normalized.holes2d[h][i][1].toFixed(9)}`;
      lookup.set(key, oriented3d.holes[h][i]);
    }
  }

  // Build merged 3D vertices
  const merged3d: Vec3[] = [];
  for (const pt2d of mergedPolygon2d) {
    const key = `${pt2d[0].toFixed(9)},${pt2d[1].toFixed(9)}`;
    const pt3d = lookup.get(key);
    if (pt3d) {
      merged3d.push(pt3d);
    } else {
      merged3d.push([pt2d[0], pt2d[1], 0]);
    }
  }

  // 7. Run GPU ear clipping triangulation
  t0 = performance.now();
  const points2dAsVec3: Vec3[] = mergedPolygon2d.map(p => [p[0], p[1], 0]);
  const triangles = await earClipping(points2dAsVec3);
  tessellationProfile.earClipping.total += performance.now() - t0;
  tessellationProfile.earClipping.calls++;

  tessellationProfile.tessellatePlanarFace.total += performance.now() - faceStart;
  tessellationProfile.tessellatePlanarFace.calls++;

  return { vertices: merged3d, triangles };
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

type Vec2 = [number, number];

function wrapToPi(angleRad: number): number {
  const twoPi = Math.PI * 2;
  // Normalize into (-2π, 2π) first for numerical stability
  angleRad = angleRad % twoPi;
  if (angleRad > Math.PI) angleRad -= twoPi;
  if (angleRad < -Math.PI) angleRad += twoPi;
  return angleRad;
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

function getFaceTrimLoopsUV(
  oc: any,
  face: FaceWithEdgesInfo
): { uvOuter: Vec2[]; uvHoles: Vec2[][] } | null {
  if (!face.occSurface) return null;

  const sa = new oc.ShapeAnalysis_Surface(face.occSurface);

  const outer3d = occEdgesToPolygon(face.outerLoop);
  if (outer3d.length < 3) {
    sa.delete?.();
    return null;
  }

  const isUPeriodic = ['Cylinder', 'Sphere', 'Cone', 'Torus'].includes(face.surfaceType);
  const isVPeriodic = face.surfaceType === 'Torus';

  const uvOuter = projectPointsToUV(oc, sa, outer3d, { wrapU: isUPeriodic, wrapV: isVPeriodic });
  const uvHoles = face.innerLoops
    .map((loop) => occEdgesToPolygon(loop))
    .filter((loop3d) => loop3d.length >= 3)
    .map((loop3d) => projectPointsToUV(oc, sa, loop3d, { wrapU: isUPeriodic, wrapV: isVPeriodic }))
    .filter((loop2d) => loop2d.length >= 3);

  sa.delete?.();

  if (uvOuter.length < 3) return null;
  return { uvOuter, uvHoles };
}

function chooseTrimGridDensity(uvOuter: Vec2[], uvHoles: Vec2[][]): number {
  const totalPts = uvHoles.reduce((acc, h) => acc + h.length, uvOuter.length);
  // Heuristic: keep grid roughly proportional to boundary complexity.
  const base = Math.ceil(Math.sqrt(totalPts) * 4);
  return Math.max(16, Math.min(128, base));
}

/**
 * Tessellate a curved surface face using existing surface-tessellation functions
 */
async function tessellateCurvedFaceFromOCC(face: FaceWithEdgesInfo): Promise<{
  vertices: Vec3[];
  triangles: number[][];
}> {
  const faceStart = performance.now();

  if (!face.surfaceParams) {
    console.warn(`[Tessellate] No surface params for ${face.surfaceType} face ${face.faceIndex}`);
    return { vertices: [], triangles: [] };
  }

  const params = face.surfaceParams;

  // Prefer UV-trimmed tessellation using the actual face boundary wires.
  // The previous approach tessellated the whole (u,v) bounds rectangle, which drops trim details.
  try {
    const oc = await initOC();
    const loops = getFaceTrimLoopsUV(oc, face);
    if (loops) {
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
        const gridDensity = chooseTrimGridDensity(loops.uvOuter, loops.uvHoles);
        const mesh = await tessellateTrimmedSurface(surface, loops.uvOuter, gridDensity, loops.uvHoles);
        tessellationProfile.tessellateCurvedFace.total += performance.now() - faceStart;
        tessellationProfile.tessellateCurvedFace.calls++;
        return tessellatedMeshToVerticesAndTriangles(mesh);
      }
    }
  } catch (e) {
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
    const mesh = await tessellateCone(
      { type: 'CONICAL_SURFACE', placement: params.placement, radius: params.radius, semiAngle: params.semiAngle },
      uMin, uMax,
      vMin, vMax,
      64
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
async function tessellateOCCShape(faces: FaceWithEdgesInfo[]): Promise<Mesh> {
  const shapeStart = performance.now();
  console.log(`[Tessellate] Starting tessellation of ${faces.length} faces...`);

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

  for (const face of faces) {
    try {
      // Progress logging
      if (processedCount % progressInterval === 0 || processedCount < 10) {
        const elapsed = ((performance.now() - shapeStart) / 1000).toFixed(1);
        const pct = ((processedCount / faces.length) * 100).toFixed(1);
        console.log(`[Tessellate] Face ${processedCount}/${faces.length} (${pct}%) - ${elapsed}s elapsed - type: ${face.surfaceType}`);
      }
      let result: { vertices: Vec3[]; triangles: number[][] };
      const faceStart = performance.now();

      if (face.surfaceType === 'Plane') {
        result = await tessellatePlanarFaceFromOCC(face);
      } else if (['Cylinder', 'Sphere', 'Cone', 'Torus', 'BSplineSurface'].includes(face.surfaceType)) {
        result = await tessellateCurvedFaceFromOCC(face);
      } else {
        // Skip unsupported surface types
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
    } catch (e) {
      console.error(`[Tessellate] Error tessellating face ${face.faceIndex}:`, e);
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
export async function parseStepWithOCC(stepFileContent: string): Promise<Mesh> {
  console.log('[parseStepWithOCC] Starting...');
  const startTime = performance.now();

  // Step 1: Load STEP file with OCC (with color support)
  const { shape, colorTool, shapeTool, stepColors, shapeColorMap, faceIdOrder, geometryColorMap, solidMatchedColors, faceToSolid, solidToColor } = await loadStepFile(stepFileContent, 'input.step');

  // Log color extraction summary
  console.log(`[parseStepWithOCC] Color sources: shapeColorMap=${shapeColorMap.size}, stepColors=${stepColors.size}, faceIdOrder=${faceIdOrder.length}, geometryColorMap=${geometryColorMap.size}, solidMatchedColors=${solidMatchedColors.size}, faceToSolid=${faceToSolid.size}, solidToColor=${solidToColor.size}, colorTool=${!!colorTool}`);

  // Step 2: Extract faces with edges, surface parameters, and colors
  const faces = await extractFacesWithEdges(shape, colorTool, shapeTool, stepColors, shapeColorMap, faceIdOrder, geometryColorMap, solidMatchedColors, faceToSolid, solidToColor);

  // Count faces with colors
  const facesWithColor = faces.filter(f => f.color !== undefined).length;
  console.log(`[parseStepWithOCC] Extracted ${faces.length} faces (${facesWithColor} with colors)`);

  // Step 3: Tessellate all faces
  const mesh = await tessellateOCCShape(faces);

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

  // Initialize occt-import-js
  const initStart = performance.now();
  const occt = await occtimportjs();
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

// Export for use in browser
export { initOC, loadStepFile, countFaces, runCheckpoint1, extractSurfaces, runCheckpoint2, extractFacesWithEdges, runCheckpoint3, runCheckpoint4, runCheckpoint5 };
