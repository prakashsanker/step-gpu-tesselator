import {getGPUDevice} from "./lib";
// Minimal STEP → mesh parser for the square face example
type Vec3 = [number, number, number];

export interface Mesh {
  positions: Float32Array;
  indices: Uint16Array;
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
  const model = parseStep(stepText);

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
  console.log("BOUNDARY POINTS", boundaryPoints);


  // Now deduplicate last point for triangulation (we want N unique vertices)
  const uniquePoints = boundaryPoints.slice(0, boundaryPoints.length - 1);
  console.log("UNIQUE POINTS", uniquePoints);
  // unique points is the cartesian points, in an array of arrays. 

  // Build positions array -- this is the uniquePoints themselves flattened.
  const positions = new Float32Array(uniquePoints.length * 3);
  uniquePoints.forEach((p, i) => {
    positions[i * 3 + 0] = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p[2];
  });

  console.log("positions", positions);

  const gpuIndices = await tesselate(uniquePoints);
  console.log("GPU INDICES");
  console.log(gpuIndices);
  let indices = new Uint16Array(gpuIndices.length);
  for (let i =0; i < gpuIndices.length; i++) {
    indices[i] = gpuIndices[i];
  }

  return { positions, indices };
}

export async function tesselate(uniquePoints): Promise<{
  indicesU32: Uint32Array;
  triCount: number;
  vertexCount: number;
}>  {
    // 1. Assume you already have a WebGPU device
  const device = await getGPUDevice();

  // 2. Input: flat positions for N vertices (uniquePointsFlat.length === N * 3)

  const uniquePointsFlat = new Float32Array(uniquePoints.length * 3);
  uniquePoints.forEach((p, i) => {
    uniquePointsFlat[i * 3 + 0] = p[0];
    uniquePointsFlat[i * 3 + 1] = p[1];
    uniquePointsFlat[i * 3 + 2] = p[2];
  });
  const vertexCount = uniquePoints.length;

  // 3. Create input buffer (read-only in shader)
  const inputBuffer = device.createBuffer({
    size: uniquePointsFlat.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, uniquePointsFlat);


  // // 5. Compute triangle + index count
  const triCount = uniquePoints.length - 2;  // this is the number of triangles, this is the number of sides in the polygon - 2
  const indexCount = triCount * 3;
  const indicesBuffer = device.createBuffer({
    size: indexCount * 4, // Uint16 → 2 bytes each
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // 6. Uniform buffer for counts
  const uniformData = new Uint32Array([
    vertexCount,  // N
    triCount,     // triCount
  ]);
  const uniformBuffer = device.createBuffer({
    size: Math.ceil((uniformData.length * 4) / 16) * 16, // Uniform buffers have to be 16 byte aligned. We have 2 4 byte variables in vertex count and tri count, we need an extra 8 bytes. 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // 7. Create shader module
  const shaderModule = device.createShaderModule({
    code: /* wgsl */ `
          struct Positions {
        values: array<f32>,
      };

      struct Indices {
        values: array<u32>, // may reinterpret as u16 on CPU
      };

      struct Params {
        vertexCount: u32,  // N
        triCount: u32,     // N - 2
        padding0: u32,
        padding1: u32,
      };

      @group(0) @binding(0)
      var<storage, read> inputPositions : Positions;

      @group(0) @binding(1)
      var<storage, read_write> outIndices : Indices;

      @group(0) @binding(2)
      var<uniform> params : Params;


      @compute @workgroup_size(64)
      fn main_indices(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let t = global_id.x; // triangle index (0..triCount-1)

        if (t >= params.triCount) {
          return;
        }

        let i = t + 1u; // 1, 2, 3

        let base = t * 3u; // 0, 3, 6

        outIndices.values[base + 0u] = 0u; // 0
        outIndices.values[base + 1u] = i; // 
        outIndices.values[base + 2u] = i+1;
      }
    `,
  });

  const indicesPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "main_indices"
    }
  })

  // 9. Bind group
  const bindGroup = device.createBindGroup({
    layout: indicesPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: inputBuffer },
      },
      {
        binding: 1,
        resource: { buffer: indicesBuffer },
      },
      {
        binding: 2,
        resource: { buffer: uniformBuffer },
      },
    ],
  });

  // 10. Encode commands
  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(indicesPipeline);
  pass.setBindGroup(0, bindGroup);

  // // We’ll launch 1D workgroups, 1 thread per vertex for copying
  const workgroupSize = 64;
  const vertexWorkgroupCount = Math.ceil(triCount / workgroupSize);
  pass.dispatchWorkgroups(vertexWorkgroupCount);

  pass.end();
  const commandBuffer = commandEncoder.finish();
  device.queue.submit([commandBuffer]);

  await indicesBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = indicesBuffer.getMappedRange();

  // Copy into a typed array we can safely use after unmap
  const indicesU32 = new Uint32Array(arrayBuffer.slice(0));

  indicesBuffer.unmap();

  return { indicesU32, vertexCount, triCount };
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

  // Remove comments (/* ... */ and -- ... end-of-line)
  let text = stepText.replace(/\/\*[\s\S]*?\*\//g, "");     // block comments
  text = text.replace(/--.*$/gm, "");                       // line comments

  // Split into lines and process entity lines starting with '#'
  const lines = text.split(/\r?\n/);

  const entityRegex = /^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\);?$/;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("#")) continue;

    const match = trimmed.match(entityRegex);
    if (!match) continue;

    const id = parseInt(match[1], 10);
    const type = match[2];
    const args = match[3]; // raw argument string inside (...)

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

