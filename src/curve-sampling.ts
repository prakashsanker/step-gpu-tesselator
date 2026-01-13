/**
 * GPU-accelerated curve sampling for STEP file curves.
 *
 * This module samples CIRCLE, ELLIPSE, and B_SPLINE curves using WebGPU
 * compute shaders for parallel evaluation.
 */

import { getGPUDevice } from "./lib";
import type { Vec3, ResolvedCurve, ResolvedBSpline } from "./step-parser";

// =============================================================================
// Types
// =============================================================================

/** Options for curve sampling */
export interface CurveSamplingOptions {
  /** Angular tolerance in radians (default: 5 degrees = 0.0873) */
  angularTolerance?: number;
  /** Minimum samples per curve (default: 3) */
  minSamples?: number;
  /** Maximum samples per curve (default: 128) */
  maxSamples?: number;
}

/** A curve ready for GPU sampling with computed parameters */
interface CurveForSampling {
  curve: ResolvedCurve;
  startPoint: Vec3;
  endPoint: Vec3;
  startParam: number;  // Start angle or t parameter
  endParam: number;    // End angle or t parameter
  numSamples: number;
  reversed: boolean;   // Whether to traverse in reverse
}

// =============================================================================
// WGSL Compute Shaders
// =============================================================================

const SAMPLE_CONIC_SHADER = /* wgsl */`
// Curve types
const CURVE_CIRCLE: u32 = 1u;
const CURVE_ELLIPSE: u32 = 2u;

// Curve parameters structure (padded for alignment)
struct CurveParams {
  curveType: u32,
  numSamples: u32,
  startParam: f32,
  endParam: f32,

  // For CIRCLE/ELLIPSE:
  // center.xyz, padding
  // normal.xyz, padding
  // refDir.xyz, padding
  // radius, minorRadius (ellipse only), padding, padding
  center: vec4<f32>,
  normal: vec4<f32>,
  refDir: vec4<f32>,
  radii: vec4<f32>,  // x=radius (or major), y=minorRadius, z=0, w=0
}

struct OutputPoint {
  pos: vec3<f32>,
  curveIndex: u32,
}

@group(0) @binding(0) var<storage, read> curves: array<CurveParams>;
@group(0) @binding(1) var<storage, read_write> outputPoints: array<OutputPoint>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;  // x=numCurves, y=maxSamplesPerCurve

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let numCurves = params.x;
  let maxSamples = params.y;

  let curveIdx = gid.x / maxSamples;
  let sampleIdx = gid.x % maxSamples;

  if (curveIdx >= numCurves) {
    return;
  }

  let curve = curves[curveIdx];

  if (sampleIdx >= curve.numSamples) {
    return;
  }

  // Calculate parameter t (interpolate between start and end)
  let t = mix(curve.startParam, curve.endParam, f32(sampleIdx) / f32(curve.numSamples - 1u));

  var point: vec3<f32>;

  if (curve.curveType == CURVE_CIRCLE) {
    // Circle: center + radius * (cos(t) * refDir + sin(t) * perpDir)
    let center = curve.center.xyz;
    let normal = normalize(curve.normal.xyz);
    let refDir = normalize(curve.refDir.xyz);
    let perpDir = cross(normal, refDir);
    let radius = curve.radii.x;

    point = center + radius * (cos(t) * refDir + sin(t) * perpDir);
  } else if (curve.curveType == CURVE_ELLIPSE) {
    // Ellipse: center + majorR * cos(t) * refDir + minorR * sin(t) * perpDir
    let center = curve.center.xyz;
    let normal = normalize(curve.normal.xyz);
    let refDir = normalize(curve.refDir.xyz);
    let perpDir = cross(normal, refDir);
    let majorRadius = curve.radii.x;
    let minorRadius = curve.radii.y;

    point = center + majorRadius * cos(t) * refDir + minorRadius * sin(t) * perpDir;
  } else {
    // Unknown type - output zero
    point = vec3<f32>(0.0, 0.0, 0.0);
  }

  let outputIdx = curveIdx * maxSamples + sampleIdx;
  outputPoints[outputIdx].pos = point;
  outputPoints[outputIdx].curveIndex = curveIdx;
}
`;

const SAMPLE_BSPLINE_SHADER = /* wgsl */`
// B-Spline curve parameters
struct BSplineParams {
  degree: u32,
  numControlPoints: u32,
  numKnots: u32,
  numSamples: u32,
  startParam: f32,
  endParam: f32,
  controlPointOffset: u32,  // Offset into control points buffer
  knotOffset: u32,          // Offset into knots buffer
  weightOffset: u32,        // Offset into weights buffer (0 if non-rational)
  isRational: u32,          // 1 if NURBS, 0 if regular B-spline
  padding: vec2<u32>,
}

struct OutputPoint {
  pos: vec3<f32>,
  curveIndex: u32,
}

@group(0) @binding(0) var<storage, read> bsplines: array<BSplineParams>;
@group(0) @binding(1) var<storage, read> controlPoints: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> knots: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputPoints: array<OutputPoint>;
@group(0) @binding(5) var<uniform> params: vec4<u32>;  // x=numCurves, y=maxSamplesPerCurve

// Find knot span index (binary search)
fn findKnotSpan(n: u32, degree: u32, t: f32, knotOffset: u32) -> u32 {
  // n = number of control points - 1
  // Returns i such that knots[i] <= t < knots[i+1]

  let numKnots = n + degree + 2u;

  // Handle boundary cases
  if (t >= knots[knotOffset + n + 1u]) {
    return n;
  }
  if (t <= knots[knotOffset + degree]) {
    return degree;
  }

  // Binary search
  var low = degree;
  var high = n + 1u;
  var mid = (low + high) / 2u;

  while (t < knots[knotOffset + mid] || t >= knots[knotOffset + mid + 1u]) {
    if (t < knots[knotOffset + mid]) {
      high = mid;
    } else {
      low = mid;
    }
    mid = (low + high) / 2u;
  }

  return mid;
}

// De Boor's algorithm for B-spline evaluation
fn evaluateBSpline(
  curveIdx: u32,
  t: f32,
) -> vec3<f32> {
  let curve = bsplines[curveIdx];
  let n = curve.numControlPoints - 1u;
  let p = curve.degree;

  // Find knot span
  let span = findKnotSpan(n, p, t, curve.knotOffset);

  // Initialize with control points
  var d: array<vec4<f32>, 8>;  // Max degree 7
  for (var j = 0u; j <= p; j++) {
    let cpIdx = span - p + j;
    var cp = controlPoints[curve.controlPointOffset + cpIdx];

    // Apply weight for rational B-splines
    if (curve.isRational == 1u) {
      let w = weights[curve.weightOffset + cpIdx];
      cp = vec4<f32>(cp.xyz * w, w);
    }
    d[j] = cp;
  }

  // De Boor recursion
  for (var r = 1u; r <= p; r++) {
    for (var j = p; j >= r; j--) {
      let knotIdx = curve.knotOffset + span - p + j;
      let alpha = (t - knots[knotIdx]) / (knots[knotIdx + p - r + 1u] - knots[knotIdx]);
      d[j] = (1.0 - alpha) * d[j - 1u] + alpha * d[j];
    }
  }

  // For rational B-splines, divide by weight
  if (curve.isRational == 1u) {
    return d[p].xyz / d[p].w;
  }
  return d[p].xyz;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let numCurves = params.x;
  let maxSamples = params.y;

  let curveIdx = gid.x / maxSamples;
  let sampleIdx = gid.x % maxSamples;

  if (curveIdx >= numCurves) {
    return;
  }

  let curve = bsplines[curveIdx];

  if (sampleIdx >= curve.numSamples) {
    return;
  }

  // Calculate parameter t
  let t = mix(curve.startParam, curve.endParam, f32(sampleIdx) / f32(curve.numSamples - 1u));

  let point = evaluateBSpline(curveIdx, t);

  let outputIdx = curveIdx * maxSamples + sampleIdx;
  outputPoints[outputIdx].pos = point;
  outputPoints[outputIdx].curveIndex = curveIdx;
}
`;

// =============================================================================
// Helper Functions
// =============================================================================

/** Compute angle of a point on a circle relative to its center */
function pointToAngle(
  point: Vec3,
  center: Vec3,
  normal: Vec3,
  refDirection: Vec3
): number {
  // Vector from center to point
  const dx = point[0] - center[0];
  const dy = point[1] - center[1];
  const dz = point[2] - center[2];

  // Compute perpendicular direction (cross product of normal and refDirection)
  const perpX = normal[1] * refDirection[2] - normal[2] * refDirection[1];
  const perpY = normal[2] * refDirection[0] - normal[0] * refDirection[2];
  const perpZ = normal[0] * refDirection[1] - normal[1] * refDirection[0];

  // Project onto refDirection and perpendicular direction
  const x = dx * refDirection[0] + dy * refDirection[1] + dz * refDirection[2];
  const y = dx * perpX + dy * perpY + dz * perpZ;

  return Math.atan2(y, x);
}

/**
 * Compute parameter angle for an ellipse point.
 * For an ellipse: P(t) = center + majorR * cos(t) * refDir + minorR * sin(t) * perpDir
 * Given a point P, we need to find t such that:
 *   x = dot(P - center, refDir) = majorR * cos(t)
 *   y = dot(P - center, perpDir) = minorR * sin(t)
 * Therefore: t = atan2(y / minorR, x / majorR)
 */
function pointToEllipseAngle(
  point: Vec3,
  center: Vec3,
  normal: Vec3,
  refDirection: Vec3,
  majorRadius: number,
  minorRadius: number
): number {
  // Vector from center to point
  const dx = point[0] - center[0];
  const dy = point[1] - center[1];
  const dz = point[2] - center[2];

  // Compute perpendicular direction (cross product of normal and refDirection)
  const perpX = normal[1] * refDirection[2] - normal[2] * refDirection[1];
  const perpY = normal[2] * refDirection[0] - normal[0] * refDirection[2];
  const perpZ = normal[0] * refDirection[1] - normal[1] * refDirection[0];

  // Project onto refDirection and perpendicular direction
  const x = dx * refDirection[0] + dy * refDirection[1] + dz * refDirection[2];
  const y = dx * perpX + dy * perpY + dz * perpZ;

  // For ellipse: x = majorR * cos(t), y = minorR * sin(t)
  // So: cos(t) = x / majorR, sin(t) = y / minorR
  // Therefore: t = atan2(sin(t), cos(t)) = atan2(y/minorR, x/majorR)
  return Math.atan2(y / minorRadius, x / majorRadius);
}

/** Calculate number of samples for an arc based on angular tolerance */
function calculateArcSamples(
  startAngle: number,
  endAngle: number,
  options: CurveSamplingOptions
): number {
  // Reduced from 5° to 2.5° for smoother curve sampling
  const angularTolerance = options.angularTolerance ?? (2.5 * Math.PI / 180);
  const minSamples = options.minSamples ?? 3;
  const maxSamples = options.maxSamples ?? 128;

  let arcSpan = Math.abs(endAngle - startAngle);
  if (arcSpan > 2 * Math.PI) {
    arcSpan = 2 * Math.PI;
  }

  const samples = Math.ceil(arcSpan / angularTolerance) + 1;
  return Math.max(minSamples, Math.min(maxSamples, samples));
}

/** Normalize an angle difference to handle wraparound */
function normalizeAngleDiff(startAngle: number, endAngle: number, reversed: boolean): [number, number] {
  // Ensure we traverse in the correct direction based on the reversed flag
  if (reversed) {
    // Traverse backward (CW for positive angles)
    if (endAngle > startAngle) {
      endAngle -= 2 * Math.PI;
    }
  } else {
    // Traverse forward (CCW for positive angles)
    if (endAngle < startAngle) {
      endAngle += 2 * Math.PI;
    }
  }
  return [startAngle, endAngle];
}

// =============================================================================
// GPU Sampling Functions
// =============================================================================

let conicPipeline: GPUComputePipeline | null = null;
let bsplinePipeline: GPUComputePipeline | null = null;
let pipelinesDevice: GPUDevice | null = null;

/** Initialize GPU pipelines for curve sampling */
async function initPipelines(device: GPUDevice): Promise<void> {
  // Invalidate cached pipelines if device changed
  if (pipelinesDevice !== device) {
    conicPipeline = null;
    bsplinePipeline = null;
    pipelinesDevice = device;
  }

  if (conicPipeline && bsplinePipeline) {
    return;
  }

  // Conic (circle/ellipse) pipeline
  const conicModule = device.createShaderModule({
    code: SAMPLE_CONIC_SHADER,
  });
  conicPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: conicModule,
      entryPoint: "main",
    },
  });

  // B-Spline pipeline
  const bsplineModule = device.createShaderModule({
    code: SAMPLE_BSPLINE_SHADER,
  });
  bsplinePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: bsplineModule,
      entryPoint: "main",
    },
  });
}

/** Sample circles and ellipses on the GPU */
async function sampleConicsGPU(
  device: GPUDevice,
  curves: CurveForSampling[],
  maxSamplesPerCurve: number
): Promise<Vec3[][]> {
  if (curves.length === 0) {
    return [];
  }

  await initPipelines(device);
  if (!conicPipeline) {
    throw new Error("Conic pipeline not initialized");
  }

  const numCurves = curves.length;

  // Build curve parameters buffer using DataView for proper type handling
  // Each curve: curveType(u32), numSamples(u32), startParam(f32), endParam(f32),
  //             center(4xf32), normal(4xf32), refDir(4xf32), radii(4xf32)
  // Total: 20 x 4 bytes = 80 bytes per curve
  const curveParamsBuffer = new ArrayBuffer(numCurves * 80);
  const view = new DataView(curveParamsBuffer);

  for (let i = 0; i < numCurves; i++) {
    const { curve, startParam, endParam, numSamples } = curves[i];
    const byteOffset = i * 80;

    if (curve.type === 'CIRCLE') {
      view.setUint32(byteOffset + 0, 1, true);  // curveType = CIRCLE
      view.setUint32(byteOffset + 4, numSamples, true);
      view.setFloat32(byteOffset + 8, startParam, true);
      view.setFloat32(byteOffset + 12, endParam, true);
      view.setFloat32(byteOffset + 16, curve.center[0], true);
      view.setFloat32(byteOffset + 20, curve.center[1], true);
      view.setFloat32(byteOffset + 24, curve.center[2], true);
      view.setFloat32(byteOffset + 28, 0, true); // padding
      view.setFloat32(byteOffset + 32, curve.normal[0], true);
      view.setFloat32(byteOffset + 36, curve.normal[1], true);
      view.setFloat32(byteOffset + 40, curve.normal[2], true);
      view.setFloat32(byteOffset + 44, 0, true); // padding
      view.setFloat32(byteOffset + 48, curve.refDirection[0], true);
      view.setFloat32(byteOffset + 52, curve.refDirection[1], true);
      view.setFloat32(byteOffset + 56, curve.refDirection[2], true);
      view.setFloat32(byteOffset + 60, 0, true); // padding
      view.setFloat32(byteOffset + 64, curve.radius, true);
      view.setFloat32(byteOffset + 68, 0, true); // minorRadius
      view.setFloat32(byteOffset + 72, 0, true);
      view.setFloat32(byteOffset + 76, 0, true);
    } else if (curve.type === 'ELLIPSE') {
      view.setUint32(byteOffset + 0, 2, true);  // curveType = ELLIPSE
      view.setUint32(byteOffset + 4, numSamples, true);
      view.setFloat32(byteOffset + 8, startParam, true);
      view.setFloat32(byteOffset + 12, endParam, true);
      view.setFloat32(byteOffset + 16, curve.center[0], true);
      view.setFloat32(byteOffset + 20, curve.center[1], true);
      view.setFloat32(byteOffset + 24, curve.center[2], true);
      view.setFloat32(byteOffset + 28, 0, true);
      view.setFloat32(byteOffset + 32, curve.normal[0], true);
      view.setFloat32(byteOffset + 36, curve.normal[1], true);
      view.setFloat32(byteOffset + 40, curve.normal[2], true);
      view.setFloat32(byteOffset + 44, 0, true);
      view.setFloat32(byteOffset + 48, curve.refDirection[0], true);
      view.setFloat32(byteOffset + 52, curve.refDirection[1], true);
      view.setFloat32(byteOffset + 56, curve.refDirection[2], true);
      view.setFloat32(byteOffset + 60, 0, true);
      view.setFloat32(byteOffset + 64, curve.majorRadius, true);
      view.setFloat32(byteOffset + 68, curve.minorRadius, true);
      view.setFloat32(byteOffset + 72, 0, true);
      view.setFloat32(byteOffset + 76, 0, true);
    }
  }

  const curveParamsData = new Uint8Array(curveParamsBuffer);

  // Create buffers
  const curveParamsGPUBuffer = device.createBuffer({
    size: curveParamsData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(curveParamsGPUBuffer, 0, curveParamsData);

  const outputSize = numCurves * maxSamplesPerCurve * 16; // 4 floats per point
  const outputBuffer = device.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramsData = new Uint32Array([numCurves, maxSamplesPerCurve, 0, 0]);
  const paramsBuffer = device.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: conicPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: curveParamsGPUBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  // Run compute shader
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(conicPipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const workgroups = Math.ceil((numCurves * maxSamplesPerCurve) / 64);
  passEncoder.dispatchWorkgroups(workgroups);
  passEncoder.end();

  // Read back results
  const readBuffer = device.createBuffer({
    size: outputSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputSize);

  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await readBuffer.mapAsync(GPUMapMode.READ);
  const outputData = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();

  // Parse results
  const results: Vec3[][] = [];
  for (let i = 0; i < numCurves; i++) {
    const curvePoints: Vec3[] = [];
    const numSamples = curves[i].numSamples;
    for (let j = 0; j < numSamples; j++) {
      const offset = (i * maxSamplesPerCurve + j) * 4;
      curvePoints.push([outputData[offset], outputData[offset + 1], outputData[offset + 2]]);
    }
    results.push(curvePoints);
  }

  // Cleanup
  curveParamsGPUBuffer.destroy();
  outputBuffer.destroy();
  paramsBuffer.destroy();
  readBuffer.destroy();

  return results;
}

// =============================================================================
// Main Export
// =============================================================================

/**
 * Sample curves using GPU compute shaders.
 *
 * @param curves Array of resolved curves to sample
 * @param startPoints Start points for each curve (trim points)
 * @param endPoints End points for each curve (trim points)
 * @param reversed Whether each curve should be traversed in reverse
 * @param options Sampling options
 * @returns Array of sampled points for each curve
 */
export async function sampleCurvesGPU(
  curves: ResolvedCurve[],
  startPoints: Vec3[],
  endPoints: Vec3[],
  reversed: boolean[],
  options: CurveSamplingOptions = {}
): Promise<Vec3[][]> {
  if (curves.length === 0) {
    return [];
  }

  const device = await getGPUDevice();
  await initPipelines(device);

  // Separate curves by type
  const conicCurves: CurveForSampling[] = [];
  const bsplineCurves: CurveForSampling[] = [];
  const curveIndices: { type: 'conic' | 'bspline' | 'line'; index: number }[] = [];

  const maxSamplesPerCurve = options.maxSamples ?? 128;

  for (let i = 0; i < curves.length; i++) {
    const curve = curves[i];
    const startPoint = startPoints[i];
    const endPoint = endPoints[i];
    const rev = reversed[i];

    if (curve.type === 'CIRCLE' || curve.type === 'ELLIPSE') {
      // Calculate angles for circle/ellipse
      const center = curve.center;
      const normal = curve.normal;
      const refDir = curve.refDirection;

      // Check if this is a full circle (start == end point)
      const dx = endPoint[0] - startPoint[0];
      const dy = endPoint[1] - startPoint[1];
      const dz = endPoint[2] - startPoint[2];
      const isFullCircle = (dx*dx + dy*dy + dz*dz) < 1e-10;

      let startAngle: number;
      let endAngle: number;
      let numSamples: number;

      if (isFullCircle) {
        // Full circle/ellipse: use 0 to 2π
        startAngle = 0;
        endAngle = Math.PI * 2;
        numSamples = Math.max(options.minSamples ?? 8, 16);
      } else {
        // Use appropriate angle calculation based on curve type
        if (curve.type === 'ELLIPSE') {
          // For ellipse, we need to account for different radii
          startAngle = pointToEllipseAngle(startPoint, center, normal, refDir, curve.majorRadius, curve.minorRadius);
          endAngle = pointToEllipseAngle(endPoint, center, normal, refDir, curve.majorRadius, curve.minorRadius);
        } else {
          // For circle, use standard angle calculation
          startAngle = pointToAngle(startPoint, center, normal, refDir);
          endAngle = pointToAngle(endPoint, center, normal, refDir);
        }
        [startAngle, endAngle] = normalizeAngleDiff(startAngle, endAngle, rev);
        numSamples = calculateArcSamples(startAngle, endAngle, options);
      }

      conicCurves.push({
        curve,
        startPoint,
        endPoint,
        startParam: startAngle,
        endParam: endAngle,
        numSamples,
        reversed: rev,
      });
      curveIndices.push({ type: 'conic', index: conicCurves.length - 1 });
    } else if (curve.type === 'B_SPLINE') {
      // B-spline parameter calculation (TODO: implement properly)
      // For now, use 0-1 parameter range
      const numSamples = Math.min(maxSamplesPerCurve, curve.controlPoints.length * 4);
      bsplineCurves.push({
        curve,
        startPoint,
        endPoint,
        startParam: rev ? 1 : 0,
        endParam: rev ? 0 : 1,
        numSamples,
        reversed: rev,
      });
      curveIndices.push({ type: 'bspline', index: bsplineCurves.length - 1 });
    } else {
      // LINE - just return endpoints
      curveIndices.push({ type: 'line', index: i });
    }
  }

  // Sample conics on GPU
  let conicResults = await sampleConicsGPU(device, conicCurves, maxSamplesPerCurve);

  // Validate conic results - check that sampled points are close to original endpoints
  // If not, fall back to linear interpolation
  for (let i = 0; i < conicResults.length; i++) {
    const samples = conicResults[i];
    if (samples.length < 2) continue;

    const { startPoint, endPoint, curve } = conicCurves[i];
    const firstSample = samples[0];
    const lastSample = samples[samples.length - 1];

    // Distance from first/last sample to expected start/end
    const startDist = Math.sqrt(
      (firstSample[0] - startPoint[0]) ** 2 +
      (firstSample[1] - startPoint[1]) ** 2 +
      (firstSample[2] - startPoint[2]) ** 2
    );
    const endDist = Math.sqrt(
      (lastSample[0] - endPoint[0]) ** 2 +
      (lastSample[1] - endPoint[1]) ** 2 +
      (lastSample[2] - endPoint[2]) ** 2
    );

    // If either endpoint is way off, something is wrong
    // Use a tolerance based on curve size (e.g., 10% of radius for circles/ellipses)
    let tolerance = 1.0; // default 1 unit
    if (curve.type === 'CIRCLE') {
      tolerance = curve.radius * 0.1;
    } else if (curve.type === 'ELLIPSE') {
      tolerance = Math.max(curve.majorRadius, curve.minorRadius) * 0.1;
    }

    if (startDist > tolerance || endDist > tolerance) {
      console.warn(`[CURVE VALIDATION] ${curve.type} curve ${i}: endpoints don't match! startDist=${startDist.toFixed(2)}, endDist=${endDist.toFixed(2)}, tolerance=${tolerance.toFixed(2)}`);
      console.warn(`  Expected start: [${startPoint.map(v => v.toFixed(2)).join(', ')}], got: [${firstSample.map(v => v.toFixed(2)).join(', ')}]`);
      console.warn(`  Expected end: [${endPoint.map(v => v.toFixed(2)).join(', ')}], got: [${lastSample.map(v => v.toFixed(2)).join(', ')}]`);
      // Fall back to linear interpolation
      conicResults[i] = [startPoint, endPoint];
    }
  }

  // TODO: Sample B-splines on GPU (for now, fall back to CPU)
  const bsplineResults: Vec3[][] = bsplineCurves.map(({ curve, startParam, endParam, numSamples }) => {
    if (curve.type !== 'B_SPLINE') {
      return [];
    }
    // CPU fallback for B-splines
    return sampleBSplineCPU(curve, startParam, endParam, numSamples);
  });

  // Reconstruct results in original order
  const results: Vec3[][] = [];
  for (let i = 0; i < curves.length; i++) {
    const { type, index } = curveIndices[i];
    if (type === 'conic') {
      results.push(conicResults[index]);
    } else if (type === 'bspline') {
      results.push(bsplineResults[index]);
    } else {
      // LINE - return start and end points
      results.push([startPoints[i], endPoints[i]]);
    }
  }

  return results;
}

// =============================================================================
// CPU Fallback for B-Splines
// =============================================================================

/** De Boor's algorithm for B-spline evaluation (CPU version) */
function evaluateBSplineCPU(
  curve: ResolvedBSpline,
  t: number
): Vec3 {
  const { degree, controlPoints, knots, weights } = curve;
  const n = controlPoints.length - 1;
  const p = degree;

  // Find knot span
  let span = p;
  for (let i = p; i <= n; i++) {
    if (t >= knots[i] && t < knots[i + 1]) {
      span = i;
      break;
    }
  }
  if (t >= knots[n + 1]) {
    span = n;
  }

  // Initialize with control points (apply weights for NURBS)
  const d: [number, number, number, number][] = [];
  for (let j = 0; j <= p; j++) {
    const cpIdx = span - p + j;
    const cp = controlPoints[cpIdx];
    if (weights) {
      const w = weights[cpIdx];
      d.push([cp[0] * w, cp[1] * w, cp[2] * w, w]);
    } else {
      d.push([cp[0], cp[1], cp[2], 1]);
    }
  }

  // De Boor recursion
  for (let r = 1; r <= p; r++) {
    for (let j = p; j >= r; j--) {
      const knotIdx = span - p + j;
      const alpha = (t - knots[knotIdx]) / (knots[knotIdx + p - r + 1] - knots[knotIdx]);
      for (let k = 0; k < 4; k++) {
        d[j][k] = (1 - alpha) * d[j - 1][k] + alpha * d[j][k];
      }
    }
  }

  // Divide by weight for NURBS
  if (weights) {
    return [d[p][0] / d[p][3], d[p][1] / d[p][3], d[p][2] / d[p][3]];
  }
  return [d[p][0], d[p][1], d[p][2]];
}

/** Sample B-spline curve on CPU */
function sampleBSplineCPU(
  curve: ResolvedBSpline,
  startParam: number,
  endParam: number,
  numSamples: number
): Vec3[] {
  const points: Vec3[] = [];
  for (let i = 0; i < numSamples; i++) {
    const t = startParam + (endParam - startParam) * (i / (numSamples - 1));
    // Clamp t to valid knot range
    const tClamped = Math.max(curve.knots[curve.degree], Math.min(t, curve.knots[curve.knots.length - curve.degree - 1]));
    points.push(evaluateBSplineCPU(curve, tClamped));
  }
  return points;
}

export { pointToAngle, calculateArcSamples };
