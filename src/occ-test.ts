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
  bridgeAllHoles,
} from './step-parser';
import { earClipping } from './ear-clipping';
import { createThreeMeshFromTesselation } from './threejs-render';
import {
  tessellateCylinder,
  tessellateSphere,
  tessellateCone,
  tessellateTorus,
  tessellateBSplineSurface,
} from './surface-tessellation';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

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
  console.log('[OCC] Initializing OpenCascade.js...');
  const startTime = performance.now();

  // Load opencascade.js dynamically to avoid Vite bundling issues
  const opencascadeModule = await import('opencascade.js/dist/opencascade.wasm.js');
  const opencascade = opencascadeModule.default;

  // Get the WASM file URL
  const wasmUrl = new URL('opencascade.js/dist/opencascade.wasm.wasm', import.meta.url).href;

  oc = await opencascade({
    locateFile: (path: string) => {
      if (path.endsWith('.wasm')) {
        return wasmUrl;
      }
      return path;
    }
  });

  console.log(`[OCC] Initialized in ${(performance.now() - startTime).toFixed(0)}ms`);

  // Debug: Log B-spline related APIs
  console.log('[OCC] B-spline surface APIs:', Object.keys(oc).filter(k => k.includes('BSpline') && k.includes('Surface')).slice(0, 20));

  // Debug: Log color/styling related APIs
  const xcafApis = Object.keys(oc).filter(k => k.includes('XCAF') || k.includes('XDE'));
  const colorApis = Object.keys(oc).filter(k => k.includes('Color') && !k.includes('ColorScale'));
  const quantityApis = Object.keys(oc).filter(k => k.startsWith('Quantity_'));
  console.log('[OCC] XCAF/XDE APIs found:', xcafApis.length, xcafApis.slice(0, 10));
  console.log('[OCC] Color APIs found:', colorApis.length, colorApis.slice(0, 10));
  console.log('[OCC] Quantity APIs found:', quantityApis.length, quantityApis.slice(0, 10));

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
    console.log(`[StepColors] Found COLOUR_RGB #${id}: RGB(${r.toFixed(2)}, ${g.toFixed(2)}, ${b.toFixed(2)})`);
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
      console.log(`[StepColors] Found predefined color #${id}: ${colorName}`);
    }
  }

  console.log(`[StepColors] Total colors parsed: ${colors.size}`);
  return colors;
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
          console.log(`[StepColors] STYLED_ITEM #${styledItemId} applies color RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)}) to ${faceIds.length} faces via target #${targetId}`);
        } else {
          // Target might be the solid/shell itself, store the color for later use
          faceColors.set(targetId, color);
          console.log(`[StepColors] STYLED_ITEM #${styledItemId} applies color to target #${targetId} (will use as default)`);
        }
        break;
      }
    }
  }

  // Also look for a default color if we found colors but no face associations
  // This happens when STYLED_ITEM targets a solid but we couldn't resolve faces
  if (faceColors.size > 0) {
    console.log(`[StepColors] Face color map built: ${faceColors.size} entries with colors`);
  } else if (colorEntities.size > 0) {
    // No STYLED_ITEM found the faces, but we have colors - use the first one as default
    const firstColor = colorEntities.values().next().value;
    if (firstColor) {
      faceColors.set(-1, firstColor); // -1 as a sentinel for "default color"
      console.log(`[StepColors] No face associations found, using first color as default: RGB(${firstColor.r.toFixed(2)}, ${firstColor.g.toFixed(2)}, ${firstColor.b.toFixed(2)})`);
    }
  }

  return faceColors;
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
 * Load a STEP file and return the TopoDS_Shape with color information
 */
async function loadStepFile(fileContent: string, fileName: string): Promise<StepLoadResult> {
  const oc = await initOC();

  // Debug: log available APIs
  console.log('[OCC] Available STEPCAFControl_Reader constructors:',
    Object.keys(oc).filter(k => k.startsWith('STEPCAFControl_Reader')));
  console.log('[OCC] Available XCAFDoc APIs:',
    Object.keys(oc).filter(k => k.startsWith('XCAFDoc')).slice(0, 20));

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
      console.log('[OCC] Using STEPCAFControl_Reader for color support...');

      // Create XDE document
      const app = new oc.TDocStd_Application();
      doc = new oc.TDocStd_Document(new oc.TCollection_ExtendedString_1());
      app.NewDocument(new oc.TCollection_ExtendedString_2("MDTV-XCAF"), doc);

      // Create XCAF STEP reader
      let cafReader;
      if (oc.STEPCAFControl_Reader_1) {
        cafReader = new oc.STEPCAFControl_Reader_1();
      } else {
        cafReader = new oc.STEPCAFControl_Reader();
      }

      // Enable color reading
      if (cafReader.SetColorMode) {
        cafReader.SetColorMode(true);
      }

      // Read file
      const readResult = cafReader.ReadFile(fileName);
      console.log('[OCC] XCAF ReadFile result:', readResult);

      const isDone = (typeof readResult === 'object' && readResult.value === 1) || readResult === 1;

      if (isDone) {
        // Transfer to document
        if (cafReader.Transfer_1) {
          cafReader.Transfer_1(doc);
        } else if (cafReader.Transfer) {
          cafReader.Transfer(doc);
        }

        // Get tools
        shapeTool = oc.XCAFDoc_DocumentTool.ShapeTool(doc.Main());
        colorTool = oc.XCAFDoc_DocumentTool.ColorTool(doc.Main());

        console.log('[OCC] Got shapeTool:', !!shapeTool, 'colorTool:', !!colorTool);

        // Get all shapes from the document
        const labels = new oc.TDF_LabelSequence_1();
        shapeTool.GetFreeShapes(labels);

        console.log('[OCC] Free shapes count:', labels.Length());

        if (labels.Length() > 0) {
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
      console.log('[OCC] XCAF reader failed, falling back to basic reader:', xcafErr);
    }
  }

  // Fallback to basic reader if XCAF failed
  if (!shape || (shape.IsNull && shape.IsNull())) {
    console.log('[OCC] Using basic STEPControl_Reader (no color support)...');

    let reader;
    if (oc.STEPControl_Reader_1) {
      reader = new oc.STEPControl_Reader_1();
    } else if (oc.STEPControl_Reader) {
      reader = new oc.STEPControl_Reader();
    } else {
      throw new Error('STEPControl_Reader not found');
    }

    const readResult = reader.ReadFile(fileName);
    console.log('[OCC] ReadFile result:', readResult);

    const isDone = readResult === oc.IFSelect_ReturnStatus?.IFSelect_RetDone ||
                   readResult === 0 ||
                   (typeof readResult === 'object' && readResult.value === 1);
    if (!isDone) {
      oc.FS.unlink(fileName);
      throw new Error(`Failed to read STEP file: ${readResult}`);
    }

    console.log('[OCC] Transferring roots...');
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

  // Debug: Check if shape is null or empty
  console.log('[OCC] Getting shape...');
  if (!shape || shape.IsNull()) {
    console.error('[OCC] Shape is null or empty!');
  } else {
    console.log('[OCC] Shape type:', shape.ShapeType ? shape.ShapeType() : 'unknown');
    const countShapes = (shapeType: string, enumValue: any) => {
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

  // Clean up
  oc.FS.unlink(fileName);

  // Parse colors directly from STEP text as fallback
  console.log('[OCC] Parsing colors from STEP text...');
  const colorEntities = parseStepColors(fileContent);
  const stepColors = buildFaceColorMap(fileContent, colorEntities);

  return { shape, colorTool, shapeTool, doc, stepColors };
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
      console.log('[OCC] Trying GeomAdaptor_Surface_2 with handle...');
      const adaptor = new oc.GeomAdaptor_Surface_2(surfaceHandle);
      console.log('[OCC] Adaptor created, calling GetType()...');
      const surfType = adaptor.GetType();
      console.log('[OCC] GetType result:', surfType, typeof surfType);

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
      console.log('[OCC] Type value:', typeValue);
      return typeMap[typeValue] || `Unknown(${typeValue})`;
    }
  } catch (e) {
    console.log('[OCC] GeomAdaptor_Surface failed:', e);
  }

  // Fallback: Try to get the dynamic type name from the unwrapped surface
  try {
    const actualSurface = typeof surfaceHandle.get === 'function' ? surfaceHandle.get() : surfaceHandle;
    if (actualSurface && typeof actualSurface.DynamicType === 'function') {
      const typeHandle = actualSurface.DynamicType();
      // In opencascade.js, the type name might be stored differently
      // Try various ways to get the name
      if (typeHandle) {
        // Try .Name() method
        if (typeof typeHandle.Name === 'function') {
          const name = typeHandle.Name();
          console.log('[OCC] DynamicType.Name() result:', name);
          return name;
        }
        // Try accessing name from internal structure
        if (typeHandle.$$ && typeHandle.$$.ptrType && typeHandle.$$.ptrType.registeredClass) {
          const name = typeHandle.$$.ptrType.registeredClass.name;
          console.log('[OCC] Type from registeredClass:', name);
          return name;
        }
      }
    }
  } catch (e) {
    console.log('[OCC] DynamicType fallback failed:', e);
  }

  return `Unknown(${String(surfaceHandle)})`;
}

/**
 * Extract surface information from all faces in a shape
 */
async function extractSurfaces(shape: any): Promise<SurfaceInfo[]> {
  const oc = await initOC();
  const surfaces: SurfaceInfo[] = [];

  // Log available TopoDS APIs for debugging
  console.log('[OCC] Available TopoDS APIs:',
    Object.keys(oc).filter(k => k.startsWith('TopoDS')).slice(0, 30));
  console.log('[OCC] Available BRep_Tool APIs:',
    Object.keys(oc).filter(k => k.startsWith('BRep_Tool')));
  console.log('[OCC] Available BRepTools APIs:',
    Object.keys(oc).filter(k => k.startsWith('BRepTools')));

  const explorer = new oc.TopExp_Explorer_2(
    shape,
    oc.TopAbs_ShapeEnum.TopAbs_FACE,
    oc.TopAbs_ShapeEnum.TopAbs_SHAPE
  );

  let faceIndex = 0;
  while (explorer.More()) {
    const currentShape = explorer.Current();
    console.log(`[OCC] Face ${faceIndex} - currentShape type:`, typeof currentShape, Object.keys(currentShape).slice(0, 10));

    try {
      // Try different approaches to cast to face
      let face = currentShape;

      // In opencascade.js, TopoDS static methods might be accessed differently
      // Try: oc.TopoDS.Face_1 (instance method style)
      if (oc.TopoDS && typeof oc.TopoDS.Face_1 === 'function') {
        console.log('[OCC] Trying oc.TopoDS.Face_1...');
        face = oc.TopoDS.Face_1(currentShape);
      } else if (oc.TopoDS && typeof oc.TopoDS.Face === 'function') {
        console.log('[OCC] Trying oc.TopoDS.Face...');
        face = oc.TopoDS.Face(currentShape);
      } else if (typeof oc.TopoDS_Face === 'function') {
        // Maybe TopoDS_Face is a constructor that takes a shape
        console.log('[OCC] Trying new oc.TopoDS_Face...');
        // Don't construct, just use current shape directly
      }

      console.log(`[OCC] Face object type:`, typeof face, face ? Object.keys(face).slice(0, 10) : 'null');

      // Get the surface from the face using BRep_Tool::Surface
      // In opencascade.js, static methods are often flattened
      let surface = null;

      // Try different API patterns
      if (oc.BRep_Tool && typeof oc.BRep_Tool.Surface_2 === 'function') {
        console.log('[OCC] Trying oc.BRep_Tool.Surface_2...');
        surface = oc.BRep_Tool.Surface_2(face);
      } else if (oc.BRep_Tool && typeof oc.BRep_Tool.Surface === 'function') {
        console.log('[OCC] Trying oc.BRep_Tool.Surface...');
        surface = oc.BRep_Tool.Surface(face);
      } else if (typeof oc.BRep_Tool_Surface_2 === 'function') {
        console.log('[OCC] Trying oc.BRep_Tool_Surface_2...');
        surface = oc.BRep_Tool_Surface_2(face);
      } else if (typeof oc.BRep_Tool_Surface === 'function') {
        console.log('[OCC] Trying oc.BRep_Tool_Surface...');
        surface = oc.BRep_Tool_Surface(face);
      } else {
        console.log('[OCC] No BRep_Tool.Surface found, listing BRep_Tool keys:',
          oc.BRep_Tool ? Object.keys(oc.BRep_Tool) : 'BRep_Tool not found');
      }

      if (!surface) {
        console.warn(`[OCC] No surface found for face ${faceIndex}`);
        explorer.Next();
        faceIndex++;
        continue;
      }

      console.log(`[OCC] Surface object:`, typeof surface, surface ? Object.keys(surface).slice(0, 10) : 'null');

      // BRep_Tool.Surface_2 returns a Handle<Geom_Surface>
      // Pass the handle directly to getSurfaceTypeName - it expects a Handle, not the unwrapped surface
      const surfaceType = getSurfaceTypeName(oc, surface);

      // Get UV bounds using BRepAdaptor_Surface
      // This adapts the face and provides UV parameter ranges
      let uMin = 0, uMax = 0, vMin = 0, vMax = 0;

      try {
        // BRepAdaptor_Surface_2 constructor takes a TopoDS_Face
        if (oc.BRepAdaptor_Surface_2) {
          console.log('[OCC] Creating BRepAdaptor_Surface_2 for UV bounds...');
          const faceAdaptor = new oc.BRepAdaptor_Surface_2(face, true); // true = bounds restriction

          // Get UV parameter ranges
          uMin = faceAdaptor.FirstUParameter();
          uMax = faceAdaptor.LastUParameter();
          vMin = faceAdaptor.FirstVParameter();
          vMax = faceAdaptor.LastVParameter();

          console.log(`[OCC] UV bounds from adaptor: U=[${uMin}, ${uMax}], V=[${vMin}, ${vMax}]`);
        } else if (oc.BRepAdaptor_Surface_1) {
          console.log('[OCC] Trying BRepAdaptor_Surface_1...');
          const faceAdaptor = new oc.BRepAdaptor_Surface_1();
          faceAdaptor.Initialize_1(face, true);

          uMin = faceAdaptor.FirstUParameter();
          uMax = faceAdaptor.LastUParameter();
          vMin = faceAdaptor.FirstVParameter();
          vMax = faceAdaptor.LastVParameter();
        } else {
          console.log('[OCC] BRepAdaptor_Surface not found, available:',
            Object.keys(oc).filter(k => k.includes('BRepAdaptor')));
        }
      } catch (boundsErr) {
        console.log('[OCC] BRepAdaptor_Surface failed:', boundsErr);
      }

      surfaces.push({
        faceIndex,
        surfaceType,
        uvBounds: { uMin, uMax, vMin, vMax }
      });

      console.log(`[OCC] Face ${faceIndex}: ${surfaceType}, UV=[${uMin.toFixed(2)},${uMax.toFixed(2)}]x[${vMin.toFixed(2)},${vMax.toFixed(2)}]`);

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
    console.log('[OCC] GeomAdaptor_Curve failed:', e);
  }

  return 'Unknown';
}

/**
 * Extract boundary edges from a face
 */
async function extractFaceEdges(oc: any, face: any, faceIndex: number): Promise<{ outerLoop: EdgeInfo[]; innerLoops: EdgeInfo[][] }> {
  const outerLoop: EdgeInfo[] = [];
  const innerLoops: EdgeInfo[][] = [];

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
        console.log('[OCC] No WireExplorer available');
        wireExplorer.Next();
        wireIndex++;
        continue;
      }

      // Debug: log available methods on wire explorer
      console.log(`[OCC] WireExplorer methods:`, Object.keys(edgeExplorer).filter(k => typeof edgeExplorer[k] === 'function'));

      // Debug: check what TopExp functions are available for vertex extraction
      console.log(`[OCC] TopExp APIs:`, Object.keys(oc).filter(k => k.startsWith('TopExp')).slice(0, 20));

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

                console.log(`[OCC] Edge ${edgeIndex} from TopExp: start=(${startPoint.x.toFixed(2)}, ${startPoint.y.toFixed(2)}, ${startPoint.z.toFixed(2)}), end=(${endPoint.x.toFixed(2)}, ${endPoint.y.toFixed(2)}, ${endPoint.z.toFixed(2)})`);
              }
            } catch (topExpErr) {
              console.log('[OCC] TopExp vertex extraction failed:', topExpErr);
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

            console.log(`[OCC] Edge ${edgeIndex} from Adaptor: start=(${startPoint.x.toFixed(2)}, ${startPoint.y.toFixed(2)}, ${startPoint.z.toFixed(2)}), end=(${endPoint.x.toFixed(2)}, ${endPoint.y.toFixed(2)}, ${endPoint.z.toFixed(2)})`);
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
              let numSamples = 16; // Default

              if (curveType === 'Circle') {
                // More samples for larger arcs
                numSamples = Math.max(8, Math.ceil(paramRange / (Math.PI / 8)));
              }

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

              console.log(`[OCC] Edge ${edgeIndex} (${curveType}): sampled ${sampledPoints.length} points, param range [${first.toFixed(2)}, ${last.toFixed(2)}]`);
            }
          } else {
            console.log('[OCC] BRepAdaptor_Curve_2 not available');
          }

          edges.push({
            edgeIndex,
            curveType,
            startPoint,
            endPoint,
            sampledPoints
          });

        } catch (edgeErr) {
          console.log(`[OCC] Error processing edge ${edgeIndex}:`, edgeErr);
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

      // First wire is typically the outer loop, subsequent are inner loops (holes)
      if (wireIndex === 0) {
        outerLoop.push(...edges);
      } else {
        innerLoops.push(edges);
      }

      wireExplorer.Next();
      wireIndex++;
    }

  } catch (e) {
    console.error(`[OCC] Error extracting edges for face ${faceIndex}:`, e);
  }

  return { outerLoop, innerLoops };
}

/**
 * Get color for a face from the colorTool
 */
function getFaceColor(oc: any, face: any, colorTool: any, shapeTool: any): RGBColor | undefined {
  if (!colorTool || !shapeTool) {
    return undefined;
  }

  try {
    // Try to find the label for this face
    const label = new oc.TDF_Label();

    // Try to get color directly from the face
    const color = new oc.Quantity_Color_1();

    // XCAFDoc_ColorType: 0=XCAFDoc_ColorGen, 1=XCAFDoc_ColorSurf, 2=XCAFDoc_ColorCurv
    // Try surface color first (most common for faces)
    let hasColor = false;

    if (colorTool.GetColor_1) {
      // GetColor(shape, colorType, color)
      hasColor = colorTool.GetColor_1(face, 1, color); // 1 = XCAFDoc_ColorSurf
      if (!hasColor) {
        hasColor = colorTool.GetColor_1(face, 0, color); // 0 = XCAFDoc_ColorGen
      }
    } else if (colorTool.GetColor) {
      hasColor = colorTool.GetColor(face, 1, color);
      if (!hasColor) {
        hasColor = colorTool.GetColor(face, 0, color);
      }
    }

    if (hasColor) {
      const r = color.Red();
      const g = color.Green();
      const b = color.Blue();
      console.log(`[OCC] Found face color: RGB(${r.toFixed(2)}, ${g.toFixed(2)}, ${b.toFixed(2)})`);
      return { r, g, b };
    }
  } catch (e) {
    // Color extraction failed, return undefined
    console.log('[OCC] Color extraction failed:', e);
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
  stepColors?: Map<number, RGBColor>
): Promise<FaceWithEdgesInfo[]> {
  const oc = await initOC();
  const faces: FaceWithEdgesInfo[] = [];

  console.log('[OCC] extractFacesWithEdges: colorTool available:', !!colorTool);
  console.log('[OCC] extractFacesWithEdges: stepColors available:', !!stepColors, stepColors?.size || 0);

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

            console.log(`[OCC] Cylinder params: radius=${surfaceParams.radius}, location=(${surfaceParams.placement?.location.join(',')}), axis=(${surfaceParams.placement?.axis.join(',')})`);
          } catch (e) {
            console.log('[OCC] Failed to extract cylinder params:', e);
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

            console.log(`[OCC] Sphere params: radius=${surfaceParams.radius}`);
          } catch (e) {
            console.log('[OCC] Failed to extract sphere params:', e);
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

            console.log(`[OCC] Cone params: radius=${surfaceParams.radius}, semiAngle=${surfaceParams.semiAngle}`);
          } catch (e) {
            console.log('[OCC] Failed to extract cone params:', e);
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

            console.log(`[OCC] Torus params: majorRadius=${surfaceParams.majorRadius}, minorRadius=${surfaceParams.minorRadius}`);
          } catch (e) {
            console.log('[OCC] Failed to extract torus params:', e);
          }
        } else if (surfaceType === 'BSplineSurface') {
          try {
            console.log('[OCC] Attempting to extract B-spline surface...');

            // Try multiple approaches to get the B-spline surface
            let bspline = null;

            // Approach 1: BSpline() method on faceAdaptor
            if (faceAdaptor.BSpline) {
              console.log('[OCC] Trying faceAdaptor.BSpline()...');
              const bsplineSurf = faceAdaptor.BSpline();
              console.log('[OCC] BSpline() result:', bsplineSurf, typeof bsplineSurf);
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
              console.log('[OCC] Trying to get B-spline from surface handle...');
              // The surface variable is already a Handle<Geom_Surface> from BRep_Tool.Surface_2
              // For B-spline surfaces, we can try to access B-spline-specific methods
              const actualSurface = typeof surface.get === 'function' ? surface.get() : surface;
              console.log('[OCC] actualSurface:', actualSurface, typeof actualSurface);
              if (actualSurface) {
                console.log('[OCC] actualSurface methods:', Object.keys(actualSurface).filter(k => typeof actualSurface[k] === 'function').slice(0, 30));
              }
              if (actualSurface && typeof actualSurface.UDegree === 'function') {
                bspline = actualSurface;
              }
            }

            // Approach 3: Try DownCast if available
            if (!bspline && surface && oc.Geom_BSplineSurface) {
              console.log('[OCC] Trying DownCast to Geom_BSplineSurface...');
              try {
                // In opencascade.js, DownCast might be a static method
                if (oc.Geom_BSplineSurface.DownCast) {
                  const downcast = oc.Geom_BSplineSurface.DownCast(surface);
                  if (downcast && !downcast.IsNull()) {
                    bspline = downcast.get ? downcast.get() : downcast;
                    console.log('[OCC] DownCast succeeded:', bspline);
                  }
                }
              } catch (downcastErr) {
                console.log('[OCC] DownCast failed:', downcastErr);
              }
            }

            if (bspline) {
              console.log('[OCC] Got B-spline object:', bspline);
              console.log('[OCC] B-spline methods:', Object.keys(bspline).filter(k => typeof bspline[k] === 'function').slice(0, 20));

              // Get degrees
              const uDegree = bspline.UDegree();
              const vDegree = bspline.VDegree();
              console.log(`[OCC] Degrees: U=${uDegree}, V=${vDegree}`);

              // Get number of control points
              const numUPoles = bspline.NbUPoles();
              const numVPoles = bspline.NbVPoles();
              console.log(`[OCC] Poles: U=${numUPoles}, V=${numVPoles}`);

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
              console.log(`[OCC] Control points extracted: ${controlPoints.length} rows`);

              // Extract knots with multiplicities (OCC provides them separately)
              const numUKnots = bspline.NbUKnots();
              const numVKnots = bspline.NbVKnots();
              console.log(`[OCC] Knots: U=${numUKnots}, V=${numVKnots}`);

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

              console.log(`[OCC] BSpline params: degree=(${uDegree},${vDegree}), poles=(${numUPoles}x${numVPoles}), knots=(${uKnots.length},${vKnots.length})`);
            } else {
              console.log('[OCC] Could not get B-spline object from either approach');
            }
          } catch (e) {
            console.log('[OCC] Failed to extract B-spline params:', e);
            console.error('[OCC] B-spline extraction error stack:', e);
          }
        }
      }

      // Extract boundary edges
      const { outerLoop, innerLoops } = await extractFaceEdges(oc, face, faceIndex);

      // Extract face color - try XCAF first, then stepColors fallback
      let color = getFaceColor(oc, face, colorTool, shapeTool);

      // If XCAF didn't work, try to use parsed stepColors
      // Since we don't have a direct mapping from OCC faces to STEP entity IDs,
      // we use the first available color as a default for all faces
      if (!color && stepColors && stepColors.size > 0) {
        // Get the first color from stepColors
        // This handles both uniform-colored models and the -1 sentinel for default colors
        const firstColor = stepColors.values().next().value;
        if (firstColor) {
          color = firstColor;
          // Only log once for efficiency
          if (faceIndex === 0) {
            console.log(`[OCC] Using color from STEP text parsing: RGB(${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)})`);
          }
        }
      }

      faces.push({
        faceIndex,
        surfaceType,
        uvBounds: { uMin, uMax, vMin, vMax },
        outerLoop,
        innerLoops,
        surfaceParams,
        color
      });

      // Warn if we have a B-spline surface but couldn't extract params
      if (surfaceType === 'BSplineSurface' && (!surfaceParams || !surfaceParams.bspline)) {
        console.warn(`[OCC] Face ${faceIndex}: B-spline surface detected but params extraction FAILED`);
      }

      const colorStr = color ? `RGB(${color.r.toFixed(2)},${color.g.toFixed(2)},${color.b.toFixed(2)})` : 'none';
      console.log(`[OCC] Face ${faceIndex}: ${surfaceType}, ${outerLoop.length} outer edges, ${innerLoops.length} inner loops, hasParams=${!!surfaceParams}, color=${colorStr}`);

    } catch (e) {
      console.error(`[OCC] Error processing face ${faceIndex}:`, e);
    }

    explorer.Next();
    faceIndex++;
  }

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

  console.log(`[occEdgesToPolygon] Created polygon with ${polygon.length} vertices from ${edges.length} edges`);
  return polygon;
}

/**
 * Tessellate a single planar face from OCC data
 */
async function tessellatePlanarFaceFromOCC(face: FaceWithEdgesInfo): Promise<{
  vertices: Vec3[];
  triangles: number[][];
}> {
  // Convert edge data to polygon vertices
  const outer: Vec3[] = occEdgesToPolygon(face.outerLoop);
  const holes: Vec3[][] = face.innerLoops.map(loop => occEdgesToPolygon(loop));

  console.log(`[Tessellate] Face ${face.faceIndex}: ${outer.length} outer verts, ${holes.length} holes`);
  console.log(`[Tessellate] Outer 3D:`, JSON.stringify(outer));

  if (outer.length < 3) {
    return { vertices: [], triangles: [] };
  }

  // Compute face basis from outer loop
  const basis = computeFaceBasisFromLoop(outer);
  console.log(`[Tessellate] Basis origin:`, basis.origin, 'u:', basis.u, 'v:', basis.v);

  // Project to 2D
  const projected = projectFaceLoopsTo2D({ outer, holes }, basis);
  console.log(`[Tessellate] Projected 2D:`, projected.outer2d);

  // Normalize winding (CCW outer, CW holes)
  const normalized = normalizeWinding(projected);
  console.log(`[Tessellate] Normalized 2D:`, JSON.stringify(normalized.outer2d), 'reversed:', normalized.outerReversed);

  // Apply same winding changes to 3D
  const oriented3d = applyWindingTo3D(
    { outer, holes },
    normalized.outerReversed,
    normalized.holesReversed
  );
  console.log(`[Tessellate] Oriented 3D:`, oriented3d.outer);

  // Bridge holes into outer polygon
  const mergedPolygon2d = bridgeAllHoles(normalized.outer2d, normalized.holes2d);

  // Create 2D→3D lookup
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
      // Fallback - should not happen for valid geometry
      console.warn(`[Tessellate] No 3D match for 2D point:`, pt2d);
      merged3d.push([pt2d[0], pt2d[1], 0]);
    }
  }

  console.log(`[Tessellate] Merged 3D vertices:`, JSON.stringify(merged3d));

  // Convert to format expected by ear clipping
  const points2dAsVec3: Vec3[] = mergedPolygon2d.map(p => [p[0], p[1], 0]);

  // Run GPU ear clipping
  const triangles = await earClipping(points2dAsVec3);
  console.log(`[Tessellate] Triangles:`, JSON.stringify(triangles));

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

/**
 * Tessellate a curved surface face using existing surface-tessellation functions
 */
async function tessellateCurvedFaceFromOCC(face: FaceWithEdgesInfo): Promise<{
  vertices: Vec3[];
  triangles: number[][];
}> {
  if (!face.surfaceParams) {
    console.warn(`[Tessellate] No surface params for ${face.surfaceType} face ${face.faceIndex}`);
    return { vertices: [], triangles: [] };
  }

  const { uMin, uMax, vMin, vMax } = face.uvBounds;
  const params = face.surfaceParams;

  console.log(`[Tessellate] ${face.surfaceType} face ${face.faceIndex}: UV bounds [${uMin.toFixed(2)}, ${uMax.toFixed(2)}] x [${vMin.toFixed(2)}, ${vMax.toFixed(2)}]`);

  if (face.surfaceType === 'Cylinder' && params.radius !== undefined && params.placement) {
    const mesh = await tessellateCylinder(
      {
        type: 'CYLINDRICAL_SURFACE',
        placement: params.placement,
        radius: params.radius
      },
      uMin, uMax,  // angle range
      vMin, vMax,  // height range
      16, 2        // samples
    );
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  if (face.surfaceType === 'Sphere' && params.radius !== undefined && params.placement) {
    const mesh = await tessellateSphere(
      {
        type: 'SPHERICAL_SURFACE',
        placement: params.placement,
        radius: params.radius
      },
      uMin, uMax,  // longitude range
      vMin, vMax,  // latitude range
      16, 8        // samples
    );
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  if (face.surfaceType === 'Cone' && params.radius !== undefined && params.semiAngle !== undefined && params.placement) {
    const mesh = await tessellateCone(
      {
        type: 'CONICAL_SURFACE',
        placement: params.placement,
        radius: params.radius,
        semiAngle: params.semiAngle
      },
      uMin, uMax,
      vMin, vMax,
      16
    );
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  if (face.surfaceType === 'Torus' && params.majorRadius !== undefined && params.minorRadius !== undefined && params.placement) {
    const mesh = await tessellateTorus(
      {
        type: 'TOROIDAL_SURFACE',
        placement: params.placement,
        majorRadius: params.majorRadius,
        minorRadius: params.minorRadius
      },
      uMin, uMax,
      vMin, vMax,
      24, 12
    );
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  if (face.surfaceType === 'BSplineSurface' && params.bspline) {
    const { controlPoints, uDegree, vDegree, uKnots, vKnots, weights } = params.bspline;
    const mesh = await tessellateBSplineSurface(
      {
        type: 'B_SPLINE_SURFACE',
        controlPoints,
        uDegree,
        vDegree,
        uKnots,
        vKnots,
        weights
      },
      16, 16  // samples
    );
    return tessellatedMeshToVerticesAndTriangles(mesh);
  }

  console.warn(`[Tessellate] Unsupported curved surface type: ${face.surfaceType}`);
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
  const allVertices: Vec3[] = [];
  const allNormals: Vec3[] = [];
  const allColors: RGBColor[] = []; // Per-vertex colors
  const allIndices: number[] = [];
  let vertexOffset = 0;
  let hasAnyColor = false;

  for (const face of faces) {
    try {
      let result: { vertices: Vec3[]; triangles: number[][] };

      if (face.surfaceType === 'Plane') {
        result = await tessellatePlanarFaceFromOCC(face);
      } else if (['Cylinder', 'Sphere', 'Cone', 'Torus', 'BSplineSurface'].includes(face.surfaceType)) {
        result = await tessellateCurvedFaceFromOCC(face);
      } else {
        console.log(`[Tessellate] Skipping unsupported surface type: ${face.surfaceType}`);
        continue;
      }

      // Compute per-vertex normals by averaging face normals
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
    } catch (e) {
      console.error(`[Tessellate] Error tessellating face ${face.faceIndex}:`, e);
    }
  }

  // Build final mesh
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
  let vertexColors: Float32Array | undefined;
  if (hasAnyColor) {
    vertexColors = new Float32Array(allColors.length * 3);
    allColors.forEach((c, i) => {
      vertexColors![i * 3 + 0] = c.r;
      vertexColors![i * 3 + 1] = c.g;
      vertexColors![i * 3 + 2] = c.b;
    });
    console.log(`[Tessellate] Created vertex colors for ${allColors.length} vertices`);
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
  const { shape, colorTool, shapeTool, stepColors } = await loadStepFile(stepFileContent, 'input.step');

  // Step 2: Extract faces with edges, surface parameters, and colors
  const faces = await extractFacesWithEdges(shape, colorTool, shapeTool, stepColors);
  console.log(`[parseStepWithOCC] Extracted ${faces.length} faces`);

  // Step 3: Tessellate all faces
  const mesh = await tessellateOCCShape(faces);

  const endTime = performance.now();
  console.log(`[parseStepWithOCC] Complete in ${(endTime - startTime).toFixed(0)}ms: ${mesh.positions.length / 3} vertices, ${mesh.indices.length / 3} triangles`);

  return mesh;
}

// Export for use in browser
export { initOC, loadStepFile, countFaces, runCheckpoint1, extractSurfaces, runCheckpoint2, extractFacesWithEdges, runCheckpoint3, runCheckpoint4, runCheckpoint5 };
