import {getGPUDevice} from "./lib";
import { gpuTesselate } from "./gpu-tesselate";
import { isCounterClockWise } from "./signed-area";
// Minimal STEP → mesh parser for the square face example
type Vec3 = [number, number, number];

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

interface FaceOuterBound {
  id: number;
  loopId: number;
  orientation: boolean;
}

interface AdvancedFace {
  id: number;
  boundIds: number[]; // here only one: outer bound
  surfaceId: number;
  sameSense: boolean;
}

// Simple container to hold all parsed entities
interface StepModel {
  points: Map<number, CartesianPoint>;
  vertices: Map<number, VertexPoint>;
  edgeCurves: Map<number, EdgeCurve>;
  orientedEdges: Map<number, OrientedEdge>;
  edgeLoops: Map<number, EdgeLoop>;
  faceBounds: Map<number, FaceOuterBound>;
  faces: Map<number, AdvancedFace>;
}

// --- Public API: parse STEP text into a Mesh (one face) ---

export async function parseStepToMesh(stepText: string): Mesh {
  // This function is browser-safe: it expects the STEP file contents as a string,
  // leaving file I/O (File API, fetch, Node fs, etc.) to the caller.

  try {
    const model = parseStep(stepText);
    console.log("MODEL");
    console.log(model);
    if (model.faces.size === 0) {
      throw new Error("No ADVANCED_FACE found in STEP file.");
    }

    // For this minimal example, just take the first face, because there is only one face. 
    const face = [...model.faces.values()][0];

    // Get its outer bound
    const outerBoundId = face.boundIds[0];
    const outerBound = model.faceBounds.get(outerBoundId);
    if (!outerBound) throw new Error(`FACE_OUTER_BOUND #${outerBoundId} not found`);

    const loop = model.edgeLoops.get(outerBound.loopId);
    if (!loop) throw new Error(`EDGE_LOOP #${outerBound.loopId} not found`);

    // Walk oriented edges in order to get vertex coordinates around the boundary
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

      // For building the polygon boundary, we can push the start point of each edge.
      // The last edge will end where the first one started, so we'll close the loop later.
      boundaryPoints.push(startPoint.coords);
    }

    // Close the loop explicitly if needed (ensure last != first before adding)
    const first = boundaryPoints[0];
    const last = boundaryPoints[boundaryPoints.length - 1];
    if (!vec3Equal(first, last)) {
      boundaryPoints.push(first);
    }

    let uniquePoints = boundaryPoints.slice(0, boundaryPoints.length - 1);
    const positions = new Float32Array(uniquePoints.length * 3);
    uniquePoints.forEach((p, i) => {
      positions[i * 3 + 0] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
    });

    if (!isCounterClockWise(uniquePoints))  {
      uniquePoints.reverse();
    }

    const indices = await gpuTesselate(uniquePoints);

    return { positions, indices};

  } catch(e) {
    throw e;
  }
  
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

// --- Helpers ---

function vec3Equal(a: Vec3, b: Vec3): boolean {
  return (
    Math.abs(a[0] - b[0]) < 1e-9 &&
    Math.abs(a[1] - b[1]) < 1e-9 &&
    Math.abs(a[2] - b[2]) < 1e-9
  );
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
    console.log("TRIMMED", trimmed);

    if (!trimmed.startsWith("#")) continue;
    console.log("DO WE MAKE IT HERE?")
    const match = trimmed.match(entityRegex);
    if (!match) continue;
    console.log("DO WE GET A MATCH?");

    const id = parseInt(match[1], 10);
    const type = match[2];
    const args = match[3]; // raw argument string inside (...)
    console.log("TYPE", type);

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
        parseFaceOuterBound(id, args, model);
        break;
      case "ADVANCED_FACE":
        parseAdvancedFace(id, args, model);
        break;
      // We ignore other entity types (PLANE, LINE, DIRECTION, etc.) for now
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
  console.log("PARSE ORIENTED EDGE", id);
  console.log("args", args);
  console.log("EDGE ELEMENT ID", edgeElementId);

  console.log("=====");


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

function parseFaceOuterBound(id: number, args: string, model: StepModel) {
  // FACE_OUTER_BOUND('', #70, .T.)
  const m = args.match(/'.*'\s*,\s*#(\d+)\s*,\s*(\.[TF]\.)/);
  if (!m) throw new Error(`Failed to parse FACE_OUTER_BOUND args: ${args}`);
  const loopId = parseInt(m[1], 10);
  const orientation = m[2] === ".T.";

  model.faceBounds.set(id, { id, loopId, orientation });
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

