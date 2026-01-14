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
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// OpenCascade instance type (using any for now since types aren't well-defined)
type OpenCascadeInstance = any;

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
  return oc;
}

/**
 * Load a STEP file and return the TopoDS_Shape
 */
async function loadStepFile(fileContent: string, fileName: string): Promise<any> {
  const oc = await initOC();

  // Debug: log available APIs
  console.log('[OCC] Available STEPControl_Reader constructors:',
    Object.keys(oc).filter(k => k.startsWith('STEPControl_Reader')));
  console.log('[OCC] Available Message_ProgressRange constructors:',
    Object.keys(oc).filter(k => k.startsWith('Message_ProgressRange')));

  // Write file to virtual filesystem
  oc.FS.createDataFile('/', fileName, fileContent, true, true, true);

  // Create STEP reader - try different constructor variants
  let reader;
  if (oc.STEPControl_Reader_1) {
    reader = new oc.STEPControl_Reader_1();
  } else if (oc.STEPControl_Reader) {
    reader = new oc.STEPControl_Reader();
  } else {
    throw new Error('STEPControl_Reader not found');
  }

  // Read the file
  const readResult = reader.ReadFile(fileName);
  console.log('[OCC] ReadFile result:', readResult);

  // Check result - handle both enum and number
  const isDone = readResult === oc.IFSelect_ReturnStatus?.IFSelect_RetDone ||
                 readResult === 0; // IFSelect_RetDone is often 0
  if (!isDone) {
    oc.FS.unlink(fileName);
    throw new Error(`Failed to read STEP file: ${readResult}`);
  }

  // Transfer roots - try without progress range first
  console.log('[OCC] Transferring roots...');
  if (reader.TransferRoots) {
    // Try without argument first
    try {
      reader.TransferRoots();
    } catch (e) {
      // If that fails, try with progress range
      if (oc.Message_ProgressRange_1) {
        reader.TransferRoots(new oc.Message_ProgressRange_1());
      } else if (oc.Message_ProgressRange) {
        reader.TransferRoots(new oc.Message_ProgressRange());
      }
    }
  } else if (reader.TransferRoot) {
    reader.TransferRoot();
  }

  // Get the shape
  console.log('[OCC] Getting shape...');
  const shape = reader.OneShape();

  // Clean up
  oc.FS.unlink(fileName);

  return shape;
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
    const shape = await loadStepFile(stepFileContent, 'test.step');

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
    const shape = await loadStepFile(stepFileContent, 'test.step');

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

          // Get curve type
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
          } else {
            console.log('[OCC] BRepAdaptor_Curve_2 not available');
          }

          edges.push({
            edgeIndex,
            curveType,
            startPoint,
            endPoint
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
 * Extract surfaces and boundary edges from all faces
 */
async function extractFacesWithEdges(shape: any): Promise<FaceWithEdgesInfo[]> {
  const oc = await initOC();
  const faces: FaceWithEdgesInfo[] = [];

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

      // Get UV bounds
      let uMin = 0, uMax = 0, vMin = 0, vMax = 0;
      if (oc.BRepAdaptor_Surface_2) {
        const faceAdaptor = new oc.BRepAdaptor_Surface_2(face, true);
        uMin = faceAdaptor.FirstUParameter();
        uMax = faceAdaptor.LastUParameter();
        vMin = faceAdaptor.FirstVParameter();
        vMax = faceAdaptor.LastVParameter();
      }

      // Extract boundary edges
      const { outerLoop, innerLoops } = await extractFaceEdges(oc, face, faceIndex);

      faces.push({
        faceIndex,
        surfaceType,
        uvBounds: { uMin, uMax, vMin, vMax },
        outerLoop,
        innerLoops
      });

      console.log(`[OCC] Face ${faceIndex}: ${surfaceType}, ${outerLoop.length} outer edges, ${innerLoops.length} inner loops`);

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
    const shape = await loadStepFile(stepFileContent, 'test.step');

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
 * We simply take the start point of each edge - they should form a connected loop.
 */
function occEdgesToPolygon(edges: EdgeInfo[]): Vec3[] {
  if (edges.length === 0) return [];

  // Simple approach: edges are in order, just take start point of each
  // The end point of each edge should equal the start point of the next edge
  const polygon: Vec3[] = [];

  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    polygon.push([edge.startPoint.x, edge.startPoint.y, edge.startPoint.z]);

    // Debug: verify connectivity
    const nextEdge = edges[(i + 1) % edges.length];
    const TOLERANCE = 1e-4;
    const connected =
      Math.abs(edge.endPoint.x - nextEdge.startPoint.x) < TOLERANCE &&
      Math.abs(edge.endPoint.y - nextEdge.startPoint.y) < TOLERANCE &&
      Math.abs(edge.endPoint.z - nextEdge.startPoint.z) < TOLERANCE;

    if (!connected) {
      console.warn(`[occEdgesToPolygon] Edge ${i} end (${edge.endPoint.x.toFixed(2)}, ${edge.endPoint.y.toFixed(2)}, ${edge.endPoint.z.toFixed(2)}) does not connect to edge ${(i + 1) % edges.length} start (${nextEdge.startPoint.x.toFixed(2)}, ${nextEdge.startPoint.y.toFixed(2)}, ${nextEdge.startPoint.z.toFixed(2)})`);
    }
  }

  console.log(`[occEdgesToPolygon] Created polygon with ${polygon.length} vertices`);
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
  console.log(`[Tessellate] Projected 2D:`, projected.outer);

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
 * Tessellate all faces from OCC and create a Mesh
 */
async function tessellateOCCShape(faces: FaceWithEdgesInfo[]): Promise<Mesh> {
  const allVertices: Vec3[] = [];
  const allIndices: number[] = [];
  let vertexOffset = 0;

  for (const face of faces) {
    if (face.surfaceType !== 'Plane') {
      console.log(`[Checkpoint 4] Skipping non-planar face ${face.faceIndex}: ${face.surfaceType}`);
      continue;
    }

    try {
      const { vertices, triangles } = await tessellatePlanarFaceFromOCC(face);

      for (const v of vertices) {
        allVertices.push(v);
      }

      for (const tri of triangles) {
        allIndices.push(
          tri[0] + vertexOffset,
          tri[1] + vertexOffset,
          tri[2] + vertexOffset
        );
      }

      vertexOffset += vertices.length;
    } catch (e) {
      console.error(`[Checkpoint 4] Error tessellating face ${face.faceIndex}:`, e);
    }
  }

  // Build final mesh
  const positions = new Float32Array(allVertices.length * 3);
  allVertices.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  const indices = new Uint32Array(allIndices);

  return { positions, indices };
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

  // Lights
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x888888, 1.5);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
  dirLight.position.set(center.x + cameraDistance, center.y + cameraDistance * 1.5, center.z + cameraDistance);
  scene.add(dirLight);

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
    const shape = await loadStepFile(stepFileContent, 'test.step');

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

// Export for use in browser
export { initOC, loadStepFile, countFaces, runCheckpoint1, extractSurfaces, runCheckpoint2, extractFacesWithEdges, runCheckpoint3, runCheckpoint4 };
