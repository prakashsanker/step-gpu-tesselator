import { getGPUDevice } from "./lib";
import type { Axis2Placement3D, Surface } from "./surfaces";

type SupportedSurface =
    | Extract<Surface, { type: "PLANE" }>
    | Extract<Surface, { type: "CYLINDRICAL_SURFACE" }>
    | Extract<Surface, { type: "SPHERICAL_SURFACE" }>
    | Extract<Surface, { type: "CONICAL_SURFACE" }>
    | Extract<Surface, { type: "TOROIDAL_SURFACE" }>;

type SurfaceTypeCode = 0 | 1 | 2 | 3 | 4;

const SURFACE_CODES: Record<SupportedSurface["type"], SurfaceTypeCode> = {
    PLANE: 0,
    CYLINDRICAL_SURFACE: 1,
    SPHERICAL_SURFACE: 2,
    CONICAL_SURFACE: 3,
    TOROIDAL_SURFACE: 4,
};

const FLOATS_PER_SURFACE = 24; // 96 bytes

function computeYDirection(placement: Axis2Placement3D): [number, number, number] {
    const [ax, ay, az] = placement.axis;
    const [rx, ry, rz] = placement.refDirection;
    return [
        ay * rz - az * ry,
        az * rx - ax * rz,
        ax * ry - ay * rx,
    ];
}

function roundCapacity(size: number): number {
    const alignment = 256;
    return Math.max(alignment, Math.ceil(size / alignment) * alignment);
}

function isSupportedSurface(surface: Surface): surface is SupportedSurface {
    return (
        surface.type === "PLANE" ||
        surface.type === "CYLINDRICAL_SURFACE" ||
        surface.type === "SPHERICAL_SURFACE" ||
        surface.type === "CONICAL_SURFACE" ||
        surface.type === "TOROIDAL_SURFACE"
    );
}

function readGlobalBoolean(key: string, fallback: boolean): boolean {
    const raw = (globalThis as any)?.[key];
    return typeof raw === "boolean" ? raw : fallback;
}

function readGlobalNumber(key: string): number | undefined {
    const raw = (globalThis as any)?.[key];
    return typeof raw === "number" && Number.isFinite(raw) ? raw : undefined;
}

const SURFACE_EVAL_SHADER = /* wgsl */ `
struct DispatchParams {
    vertexCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct SurfaceParams {
    surfaceType: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    location: vec4f,
    axis: vec4f,
    refDir: vec4f,
    yDir: vec4f,
    scalar0: f32,
    scalar1: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0) var<storage, read> uvData: array<f32>;
@group(0) @binding(1) var<storage, read> vertexSurfaceIndex: array<u32>;
@group(0) @binding(2) var<storage, read> surfaces: array<SurfaceParams>;
@group(0) @binding(3) var<storage, read_write> positions: array<f32>;
@group(0) @binding(4) var<storage, read_write> normals: array<f32>;
@group(0) @binding(5) var<uniform> dispatch: DispatchParams;

fn safeNormalize(v: vec3f) -> vec3f {
    let len = length(v);
    if (len > 1e-10) {
        return v / len;
    }
    return vec3f(0.0, 0.0, 1.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= dispatch.vertexCount) {
        return;
    }

    let uvBase = idx * 2u;
    let outBase = idx * 3u;
    let u = uvData[uvBase + 0u];
    let v = uvData[uvBase + 1u];

    let surfaceIdx = vertexSurfaceIndex[idx];
    let surface = surfaces[surfaceIdx];

    let location = surface.location.xyz;
    let axis = surface.axis.xyz;
    let refDir = surface.refDir.xyz;
    let yDir = surface.yDir.xyz;

    let cosU = cos(u);
    let sinU = sin(u);
    var position = vec3f(0.0, 0.0, 0.0);
    var normal = vec3f(0.0, 0.0, 1.0);

    if (surface.surfaceType == 0u) {
        // PLANE
        position = location + u * refDir + v * yDir;
        normal = safeNormalize(axis);
    } else if (surface.surfaceType == 1u) {
        // CYLINDRICAL_SURFACE
        let radius = surface.scalar0;
        let radial = cosU * refDir + sinU * yDir;
        position = location + radius * radial + v * axis;
        normal = safeNormalize(radial);
    } else if (surface.surfaceType == 2u) {
        // SPHERICAL_SURFACE
        let radius = surface.scalar0;
        let cosV = cos(v);
        let sinV = sin(v);
        let radial = cosV * (cosU * refDir + sinU * yDir) + sinV * axis;
        position = location + radius * radial;
        normal = safeNormalize(radial);
    } else if (surface.surfaceType == 3u) {
        // CONICAL_SURFACE
        let radius = surface.scalar0;
        let semiAngle = surface.scalar1;
        let cosA = cos(semiAngle);
        let sinA = sin(semiAngle);
        let radial = cosU * refDir + sinU * yDir;
        let localRadius = radius + v * sinA;
        let z = v * cosA;
        position = location + localRadius * radial + z * axis;
        normal = safeNormalize(cosA * radial - sinA * axis);
    } else if (surface.surfaceType == 4u) {
        // TOROIDAL_SURFACE
        let majorRadius = surface.scalar0;
        let minorRadius = surface.scalar1;
        let cosV = cos(v);
        let sinV = sin(v);
        let radial = cosU * refDir + sinU * yDir;
        let tubeCenter = majorRadius + minorRadius * cosV;
        position = location + tubeCenter * radial + minorRadius * sinV * axis;
        normal = safeNormalize(cosV * radial + sinV * axis);
    }

    positions[outBase + 0u] = position.x;
    positions[outBase + 1u] = position.y;
    positions[outBase + 2u] = position.z;

    normals[outBase + 0u] = normal.x;
    normals[outBase + 1u] = normal.y;
    normals[outBase + 2u] = normal.z;
}
`;

const SURFACE_EVAL_GRID_SHADER = /* wgsl */ `
struct DispatchParams {
    vertexCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GridJob {
    gridU: u32,
    gridV: u32,
    startVertex: u32,
    surfaceIndex: u32,
    uMin: f32,
    vMin: f32,
    du: f32,
    dv: f32,
}

struct SurfaceParams {
    surfaceType: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    location: vec4f,
    axis: vec4f,
    refDir: vec4f,
    yDir: vec4f,
    scalar0: f32,
    scalar1: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0) var<storage, read> surfaces: array<SurfaceParams>;
@group(0) @binding(1) var<storage, read> jobs: array<GridJob>;
@group(0) @binding(2) var<storage, read> vertexJobIndex: array<u32>;
@group(0) @binding(3) var<uniform> dispatch: DispatchParams;
@group(0) @binding(4) var<storage, read_write> positions: array<f32>;
@group(0) @binding(5) var<storage, read_write> normals: array<f32>;

fn safeNormalize(v: vec3f) -> vec3f {
    let len = length(v);
    if (len > 1e-10) {
        return v / len;
    }
    return vec3f(0.0, 0.0, 1.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= dispatch.vertexCount) {
        return;
    }

    let jobIdx = vertexJobIndex[idx];
    let job = jobs[jobIdx];
    let cols = job.gridU + 1u;
    let localIdx = idx - job.startVertex;
    let i = localIdx % cols;
    let j = localIdx / cols;
    if (j > job.gridV) {
        return;
    }

    let u = job.uMin + f32(i) * job.du;
    let v = job.vMin + f32(j) * job.dv;
    let outBase = idx * 3u;
    let surface = surfaces[job.surfaceIndex];

    let location = surface.location.xyz;
    let axis = surface.axis.xyz;
    let refDir = surface.refDir.xyz;
    let yDir = surface.yDir.xyz;

    let cosU = cos(u);
    let sinU = sin(u);
    var position = vec3f(0.0, 0.0, 0.0);
    var normal = vec3f(0.0, 0.0, 1.0);

    if (surface.surfaceType == 0u) {
        position = location + u * refDir + v * yDir;
        normal = safeNormalize(axis);
    } else if (surface.surfaceType == 1u) {
        let radius = surface.scalar0;
        let radial = cosU * refDir + sinU * yDir;
        position = location + radius * radial + v * axis;
        normal = safeNormalize(radial);
    } else if (surface.surfaceType == 2u) {
        let radius = surface.scalar0;
        let cosV = cos(v);
        let sinV = sin(v);
        let radial = cosV * (cosU * refDir + sinU * yDir) + sinV * axis;
        position = location + radius * radial;
        normal = safeNormalize(radial);
    } else if (surface.surfaceType == 3u) {
        let radius = surface.scalar0;
        let semiAngle = surface.scalar1;
        let cosA = cos(semiAngle);
        let sinA = sin(semiAngle);
        let radial = cosU * refDir + sinU * yDir;
        let localRadius = radius + v * sinA;
        let z = v * cosA;
        position = location + localRadius * radial + z * axis;
        normal = safeNormalize(cosA * radial - sinA * axis);
    } else if (surface.surfaceType == 4u) {
        let majorRadius = surface.scalar0;
        let minorRadius = surface.scalar1;
        let cosV = cos(v);
        let sinV = sin(v);
        let radial = cosU * refDir + sinU * yDir;
        let tubeCenter = majorRadius + minorRadius * cosV;
        position = location + tubeCenter * radial + minorRadius * sinV * axis;
        normal = safeNormalize(cosV * radial + sinV * axis);
    }

    positions[outBase + 0u] = position.x;
    positions[outBase + 1u] = position.y;
    positions[outBase + 2u] = position.z;
    normals[outBase + 0u] = normal.x;
    normals[outBase + 1u] = normal.y;
    normals[outBase + 2u] = normal.z;
}
`;

const gpuEvalCache: {
    device: GPUDevice | null;
    pipeline: GPUComputePipeline | null;
    bindGroup: GPUBindGroup | null;
    bindGroupRefs: {
        uv: GPUBuffer | null;
        index: GPUBuffer | null;
        surfaces: GPUBuffer | null;
        positions: GPUBuffer | null;
        normals: GPUBuffer | null;
        dispatch: GPUBuffer | null;
    };
    uvBuffer: GPUBuffer | null;
    vertexSurfaceIndexBuffer: GPUBuffer | null;
    surfacesBuffer: GPUBuffer | null;
    positionsBuffer: GPUBuffer | null;
    normalsBuffer: GPUBuffer | null;
    dispatchParamsBuffer: GPUBuffer | null;
    stagingPositionsBuffer: GPUBuffer | null;
    stagingNormalsBuffer: GPUBuffer | null;
    uvCapacity: number;
    indexCapacity: number;
    surfacesCapacity: number;
    positionsCapacity: number;
    normalsCapacity: number;
    stagingPositionsCapacity: number;
    stagingNormalsCapacity: number;
} = {
    device: null,
    pipeline: null,
    bindGroup: null,
    bindGroupRefs: {
        uv: null,
        index: null,
        surfaces: null,
        positions: null,
        normals: null,
        dispatch: null,
    },
    uvBuffer: null,
    vertexSurfaceIndexBuffer: null,
    surfacesBuffer: null,
    positionsBuffer: null,
    normalsBuffer: null,
    dispatchParamsBuffer: null,
    stagingPositionsBuffer: null,
    stagingNormalsBuffer: null,
    uvCapacity: 0,
    indexCapacity: 0,
    surfacesCapacity: 0,
    positionsCapacity: 0,
    normalsCapacity: 0,
    stagingPositionsCapacity: 0,
    stagingNormalsCapacity: 0,
};

const gpuGridEvalCache: {
    device: GPUDevice | null;
    pipeline: GPUComputePipeline | null;
    bindGroup: GPUBindGroup | null;
    bindGroupRefs: {
        surfaces: GPUBuffer | null;
        jobs: GPUBuffer | null;
        vertexJobIndex: GPUBuffer | null;
        dispatch: GPUBuffer | null;
        positions: GPUBuffer | null;
        normals: GPUBuffer | null;
    };
    surfacesBuffer: GPUBuffer | null;
    jobsBuffer: GPUBuffer | null;
    vertexJobIndexBuffer: GPUBuffer | null;
    dispatchParamsBuffer: GPUBuffer | null;
    positionsBuffer: GPUBuffer | null;
    normalsBuffer: GPUBuffer | null;
    stagingPositionsBuffer: GPUBuffer | null;
    stagingNormalsBuffer: GPUBuffer | null;
    surfacesCapacity: number;
    jobsCapacity: number;
    vertexJobIndexCapacity: number;
    positionsCapacity: number;
    normalsCapacity: number;
    stagingPositionsCapacity: number;
    stagingNormalsCapacity: number;
} = {
    device: null,
    pipeline: null,
    bindGroup: null,
    bindGroupRefs: {
        surfaces: null,
        jobs: null,
        vertexJobIndex: null,
        dispatch: null,
        positions: null,
        normals: null,
    },
    surfacesBuffer: null,
    jobsBuffer: null,
    vertexJobIndexBuffer: null,
    dispatchParamsBuffer: null,
    positionsBuffer: null,
    normalsBuffer: null,
    stagingPositionsBuffer: null,
    stagingNormalsBuffer: null,
    surfacesCapacity: 0,
    jobsCapacity: 0,
    vertexJobIndexCapacity: 0,
    positionsCapacity: 0,
    normalsCapacity: 0,
    stagingPositionsCapacity: 0,
    stagingNormalsCapacity: 0,
};

function destroyCache() {
    gpuEvalCache.bindGroup = null;
    gpuEvalCache.bindGroupRefs.uv = null;
    gpuEvalCache.bindGroupRefs.index = null;
    gpuEvalCache.bindGroupRefs.surfaces = null;
    gpuEvalCache.bindGroupRefs.positions = null;
    gpuEvalCache.bindGroupRefs.normals = null;
    gpuEvalCache.bindGroupRefs.dispatch = null;
    gpuEvalCache.uvBuffer?.destroy();
    gpuEvalCache.vertexSurfaceIndexBuffer?.destroy();
    gpuEvalCache.surfacesBuffer?.destroy();
    gpuEvalCache.positionsBuffer?.destroy();
    gpuEvalCache.normalsBuffer?.destroy();
    gpuEvalCache.dispatchParamsBuffer?.destroy();
    gpuEvalCache.stagingPositionsBuffer?.destroy();
    gpuEvalCache.stagingNormalsBuffer?.destroy();
    gpuEvalCache.uvBuffer = null;
    gpuEvalCache.vertexSurfaceIndexBuffer = null;
    gpuEvalCache.surfacesBuffer = null;
    gpuEvalCache.positionsBuffer = null;
    gpuEvalCache.normalsBuffer = null;
    gpuEvalCache.dispatchParamsBuffer = null;
    gpuEvalCache.stagingPositionsBuffer = null;
    gpuEvalCache.stagingNormalsBuffer = null;
    gpuEvalCache.uvCapacity = 0;
    gpuEvalCache.indexCapacity = 0;
    gpuEvalCache.surfacesCapacity = 0;
    gpuEvalCache.positionsCapacity = 0;
    gpuEvalCache.normalsCapacity = 0;
    gpuEvalCache.stagingPositionsCapacity = 0;
    gpuEvalCache.stagingNormalsCapacity = 0;
}

function destroyGridCache() {
    gpuGridEvalCache.bindGroup = null;
    gpuGridEvalCache.bindGroupRefs.surfaces = null;
    gpuGridEvalCache.bindGroupRefs.jobs = null;
    gpuGridEvalCache.bindGroupRefs.vertexJobIndex = null;
    gpuGridEvalCache.bindGroupRefs.dispatch = null;
    gpuGridEvalCache.bindGroupRefs.positions = null;
    gpuGridEvalCache.bindGroupRefs.normals = null;
    gpuGridEvalCache.surfacesBuffer?.destroy();
    gpuGridEvalCache.jobsBuffer?.destroy();
    gpuGridEvalCache.vertexJobIndexBuffer?.destroy();
    gpuGridEvalCache.dispatchParamsBuffer?.destroy();
    gpuGridEvalCache.positionsBuffer?.destroy();
    gpuGridEvalCache.normalsBuffer?.destroy();
    gpuGridEvalCache.stagingPositionsBuffer?.destroy();
    gpuGridEvalCache.stagingNormalsBuffer?.destroy();
    gpuGridEvalCache.surfacesBuffer = null;
    gpuGridEvalCache.jobsBuffer = null;
    gpuGridEvalCache.vertexJobIndexBuffer = null;
    gpuGridEvalCache.dispatchParamsBuffer = null;
    gpuGridEvalCache.positionsBuffer = null;
    gpuGridEvalCache.normalsBuffer = null;
    gpuGridEvalCache.stagingPositionsBuffer = null;
    gpuGridEvalCache.stagingNormalsBuffer = null;
    gpuGridEvalCache.surfacesCapacity = 0;
    gpuGridEvalCache.jobsCapacity = 0;
    gpuGridEvalCache.vertexJobIndexCapacity = 0;
    gpuGridEvalCache.positionsCapacity = 0;
    gpuGridEvalCache.normalsCapacity = 0;
    gpuGridEvalCache.stagingPositionsCapacity = 0;
    gpuGridEvalCache.stagingNormalsCapacity = 0;
}

function ensureBuffer(
    device: GPUDevice,
    existing: GPUBuffer | null,
    currentCapacity: number,
    requiredSize: number,
    usage: GPUBufferUsageFlags,
    label: string
): { buffer: GPUBuffer; capacity: number } {
    const neededCapacity = roundCapacity(requiredSize);
    if (existing && currentCapacity >= neededCapacity) {
        return { buffer: existing, capacity: currentCapacity };
    }
    existing?.destroy();
    const buffer = device.createBuffer({
        label,
        size: neededCapacity,
        usage,
    });
    return { buffer, capacity: neededCapacity };
}

function writeSurfaceParams(
    f32: Float32Array,
    u32: Uint32Array,
    surfaceIndex: number,
    surface: SupportedSurface
) {
    const base = surfaceIndex * FLOATS_PER_SURFACE;
    const placement = surface.placement;
    const yDir = computeYDirection(placement);

    u32[base + 0] = SURFACE_CODES[surface.type];
    u32[base + 1] = 0;
    u32[base + 2] = 0;
    u32[base + 3] = 0;

    f32.set(
        [
            placement.location[0], placement.location[1], placement.location[2], 0,
            placement.axis[0], placement.axis[1], placement.axis[2], 0,
            placement.refDirection[0], placement.refDirection[1], placement.refDirection[2], 0,
            yDir[0], yDir[1], yDir[2], 0,
        ],
        base + 4
    );

    let scalar0 = 0;
    let scalar1 = 0;
    if (surface.type === "CYLINDRICAL_SURFACE" || surface.type === "SPHERICAL_SURFACE" || surface.type === "CONICAL_SURFACE") {
        scalar0 = surface.radius;
    } else if (surface.type === "TOROIDAL_SURFACE") {
        scalar0 = surface.majorRadius;
    }

    if (surface.type === "CONICAL_SURFACE") {
        scalar1 = surface.semiAngle;
    } else if (surface.type === "TOROIDAL_SURFACE") {
        scalar1 = surface.minorRadius;
    }

    f32[base + 20] = scalar0;
    f32[base + 21] = scalar1;
    f32[base + 22] = 0;
    f32[base + 23] = 0;
}

async function ensurePipeline(): Promise<{ device: GPUDevice; pipeline: GPUComputePipeline }> {
    const device = await getGPUDevice();
    if (gpuEvalCache.device !== device) {
        destroyCache();
        gpuEvalCache.pipeline = null;
        gpuEvalCache.device = device;
    }

    if (!gpuEvalCache.pipeline) {
        const module = device.createShaderModule({
            label: "surface-eval-shader",
            code: SURFACE_EVAL_SHADER,
        });
        gpuEvalCache.pipeline = device.createComputePipeline({
            label: "surface-eval-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "main",
            },
        });
    }

    if (!gpuEvalCache.dispatchParamsBuffer) {
        gpuEvalCache.dispatchParamsBuffer = device.createBuffer({
            label: "surface-eval-dispatch-params",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    return { device, pipeline: gpuEvalCache.pipeline };
}

async function ensureGridPipeline(): Promise<{ device: GPUDevice; pipeline: GPUComputePipeline }> {
    const device = await getGPUDevice();
    if (gpuGridEvalCache.device !== device) {
        destroyGridCache();
        gpuGridEvalCache.pipeline = null;
        gpuGridEvalCache.device = device;
    }

    if (!gpuGridEvalCache.pipeline) {
        const module = device.createShaderModule({
            label: "surface-eval-grid-shader",
            code: SURFACE_EVAL_GRID_SHADER,
        });
        gpuGridEvalCache.pipeline = device.createComputePipeline({
            label: "surface-eval-grid-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "main",
            },
        });
    }

    if (!gpuGridEvalCache.dispatchParamsBuffer) {
        gpuGridEvalCache.dispatchParamsBuffer = device.createBuffer({
            label: "surface-eval-grid-dispatch-params",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    return { device, pipeline: gpuGridEvalCache.pipeline };
}

let hasWarnedGpuEvalFailure = false;

interface PendingSurfaceEvalJob {
    surface: SupportedSurface;
    uvVertices: [number, number][];
    resolve: (result: { positions: Float32Array; normals: Float32Array } | null) => void;
}

let pendingJobs: PendingSurfaceEvalJob[] = [];
let flushInFlight = false;
let flushScheduled = false;

function getSurfaceEvalBatchDelayMs(): number {
    const explicitDelay = readGlobalNumber("__GPU_SURFACE_EVAL_BATCH_DELAY_MS__");
    if (explicitDelay !== undefined) {
        return Math.max(0, Math.min(32, explicitDelay));
    }
    return 0;
}

function scheduleFlush() {
    if (flushScheduled) return;
    flushScheduled = true;
    const delayMs = getSurfaceEvalBatchDelayMs();
    if (delayMs <= 0) {
        queueMicrotask(() => {
            flushScheduled = false;
            void flushPendingJobs();
        });
        return;
    }
    setTimeout(() => {
        flushScheduled = false;
        void flushPendingJobs();
    }, delayMs);
}

async function flushPendingJobs() {
    if (flushInFlight) return;
    flushInFlight = true;
    try {
        while (pendingJobs.length > 0) {
            const jobs = pendingJobs;
            pendingJobs = [];
            await runBatch(jobs);
        }
    } finally {
        flushInFlight = false;
    }
}

async function runBatch(jobs: PendingSurfaceEvalJob[]) {
    if (jobs.length === 0) return;

    try {
        const { device, pipeline } = await ensurePipeline();

        let totalVerts = 0;
        for (const job of jobs) totalVerts += job.uvVertices.length;

        const uvFlat = new Float32Array(totalVerts * 2);
        const vertexSurfaceIndex = new Uint32Array(totalVerts);
        const surfaceParamsF32 = new Float32Array(jobs.length * FLOATS_PER_SURFACE);
        const surfaceParamsU32 = new Uint32Array(surfaceParamsF32.buffer);

        const ranges: Array<{ start: number; count: number }> = [];
        let cursor = 0;
        for (let jobIndex = 0; jobIndex < jobs.length; jobIndex++) {
            const job = jobs[jobIndex];
            const count = job.uvVertices.length;
            ranges.push({ start: cursor, count });
            writeSurfaceParams(surfaceParamsF32, surfaceParamsU32, jobIndex, job.surface);
            for (let i = 0; i < count; i++) {
                const [u, v] = job.uvVertices[i];
                const outUvBase = (cursor + i) * 2;
                uvFlat[outUvBase + 0] = u;
                uvFlat[outUvBase + 1] = v;
                vertexSurfaceIndex[cursor + i] = jobIndex;
            }
            cursor += count;
        }

        const uvBufferInfo = ensureBuffer(
            device,
            gpuEvalCache.uvBuffer,
            gpuEvalCache.uvCapacity,
            uvFlat.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "surface-eval-uv"
        );
        gpuEvalCache.uvBuffer = uvBufferInfo.buffer;
        gpuEvalCache.uvCapacity = uvBufferInfo.capacity;

        const indexBufferInfo = ensureBuffer(
            device,
            gpuEvalCache.vertexSurfaceIndexBuffer,
            gpuEvalCache.indexCapacity,
            vertexSurfaceIndex.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "surface-eval-index"
        );
        gpuEvalCache.vertexSurfaceIndexBuffer = indexBufferInfo.buffer;
        gpuEvalCache.indexCapacity = indexBufferInfo.capacity;

        const surfaceBufferInfo = ensureBuffer(
            device,
            gpuEvalCache.surfacesBuffer,
            gpuEvalCache.surfacesCapacity,
            surfaceParamsF32.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "surface-eval-surfaces"
        );
        gpuEvalCache.surfacesBuffer = surfaceBufferInfo.buffer;
        gpuEvalCache.surfacesCapacity = surfaceBufferInfo.capacity;

        const outputBytes = totalVerts * 3 * 4;

        const positionsBufferInfo = ensureBuffer(
            device,
            gpuEvalCache.positionsBuffer,
            gpuEvalCache.positionsCapacity,
            outputBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "surface-eval-positions"
        );
        gpuEvalCache.positionsBuffer = positionsBufferInfo.buffer;
        gpuEvalCache.positionsCapacity = positionsBufferInfo.capacity;

        const normalsBufferInfo = ensureBuffer(
            device,
            gpuEvalCache.normalsBuffer,
            gpuEvalCache.normalsCapacity,
            outputBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "surface-eval-normals"
        );
        gpuEvalCache.normalsBuffer = normalsBufferInfo.buffer;
        gpuEvalCache.normalsCapacity = normalsBufferInfo.capacity;

        const stagingPosInfo = ensureBuffer(
            device,
            gpuEvalCache.stagingPositionsBuffer,
            gpuEvalCache.stagingPositionsCapacity,
            outputBytes,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "surface-eval-positions-staging"
        );
        gpuEvalCache.stagingPositionsBuffer = stagingPosInfo.buffer;
        gpuEvalCache.stagingPositionsCapacity = stagingPosInfo.capacity;

        const stagingNormInfo = ensureBuffer(
            device,
            gpuEvalCache.stagingNormalsBuffer,
            gpuEvalCache.stagingNormalsCapacity,
            outputBytes,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "surface-eval-normals-staging"
        );
        gpuEvalCache.stagingNormalsBuffer = stagingNormInfo.buffer;
        gpuEvalCache.stagingNormalsCapacity = stagingNormInfo.capacity;

        const dispatchData = new Uint32Array([totalVerts, 0, 0, 0]);

        device.queue.writeBuffer(gpuEvalCache.uvBuffer, 0, uvFlat);
        device.queue.writeBuffer(gpuEvalCache.vertexSurfaceIndexBuffer, 0, vertexSurfaceIndex);
        device.queue.writeBuffer(gpuEvalCache.surfacesBuffer, 0, surfaceParamsF32);
        device.queue.writeBuffer(gpuEvalCache.dispatchParamsBuffer!, 0, dispatchData);

        const needsBindGroupRefresh =
            !gpuEvalCache.bindGroup ||
            gpuEvalCache.bindGroupRefs.uv !== gpuEvalCache.uvBuffer ||
            gpuEvalCache.bindGroupRefs.index !== gpuEvalCache.vertexSurfaceIndexBuffer ||
            gpuEvalCache.bindGroupRefs.surfaces !== gpuEvalCache.surfacesBuffer ||
            gpuEvalCache.bindGroupRefs.positions !== gpuEvalCache.positionsBuffer ||
            gpuEvalCache.bindGroupRefs.normals !== gpuEvalCache.normalsBuffer ||
            gpuEvalCache.bindGroupRefs.dispatch !== gpuEvalCache.dispatchParamsBuffer;

        if (needsBindGroupRefresh) {
            gpuEvalCache.bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: gpuEvalCache.uvBuffer } },
                    { binding: 1, resource: { buffer: gpuEvalCache.vertexSurfaceIndexBuffer } },
                    { binding: 2, resource: { buffer: gpuEvalCache.surfacesBuffer } },
                    { binding: 3, resource: { buffer: gpuEvalCache.positionsBuffer } },
                    { binding: 4, resource: { buffer: gpuEvalCache.normalsBuffer } },
                    { binding: 5, resource: { buffer: gpuEvalCache.dispatchParamsBuffer! } },
                ],
            });
            gpuEvalCache.bindGroupRefs.uv = gpuEvalCache.uvBuffer;
            gpuEvalCache.bindGroupRefs.index = gpuEvalCache.vertexSurfaceIndexBuffer;
            gpuEvalCache.bindGroupRefs.surfaces = gpuEvalCache.surfacesBuffer;
            gpuEvalCache.bindGroupRefs.positions = gpuEvalCache.positionsBuffer;
            gpuEvalCache.bindGroupRefs.normals = gpuEvalCache.normalsBuffer;
            gpuEvalCache.bindGroupRefs.dispatch = gpuEvalCache.dispatchParamsBuffer;
        }

        const encoder = device.createCommandEncoder({ label: "surface-eval-encoder" });
        const pass = encoder.beginComputePass({ label: "surface-eval-pass" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, gpuEvalCache.bindGroup!);
        pass.dispatchWorkgroups(Math.ceil(totalVerts / 256));
        pass.end();

        encoder.copyBufferToBuffer(gpuEvalCache.positionsBuffer, 0, gpuEvalCache.stagingPositionsBuffer, 0, outputBytes);
        encoder.copyBufferToBuffer(gpuEvalCache.normalsBuffer, 0, gpuEvalCache.stagingNormalsBuffer, 0, outputBytes);
        device.queue.submit([encoder.finish()]);

        await Promise.all([
            gpuEvalCache.stagingPositionsBuffer.mapAsync(GPUMapMode.READ, 0, outputBytes),
            gpuEvalCache.stagingNormalsBuffer.mapAsync(GPUMapMode.READ, 0, outputBytes),
        ]);

        const positionsMapped = gpuEvalCache.stagingPositionsBuffer.getMappedRange(0, outputBytes);
        const normalsMapped = gpuEvalCache.stagingNormalsBuffer.getMappedRange(0, outputBytes);
        const positionsView = new Float32Array(positionsMapped);
        const normalsView = new Float32Array(normalsMapped);

        try {
            for (let i = 0; i < jobs.length; i++) {
                const { start, count } = ranges[i];
                const posBase = start * 3;
                const posEnd = posBase + count * 3;
                const positions = new Float32Array(count * 3);
                const normals = new Float32Array(count * 3);
                positions.set(positionsView.subarray(posBase, posEnd));
                normals.set(normalsView.subarray(posBase, posEnd));
                jobs[i].resolve({ positions, normals });
            }
        } finally {
            gpuEvalCache.stagingPositionsBuffer.unmap();
            gpuEvalCache.stagingNormalsBuffer.unmap();
        }
    } catch (err) {
        if (!hasWarnedGpuEvalFailure) {
            console.warn("[surface-eval-gpu] Batched GPU path failed; falling back to CPU.", err);
            hasWarnedGpuEvalFailure = true;
        }
        for (const job of jobs) {
            job.resolve(null);
        }
    }
}

export async function evaluateSurfaceMeshGPU(
    surface: Surface,
    uvVertices: [number, number][]
): Promise<{ positions: Float32Array; normals: Float32Array } | null> {
    if (!readGlobalBoolean("__ENABLE_GPU_SURFACE_EVAL__", true)) {
        return null;
    }
    if (!isSupportedSurface(surface)) {
        return null;
    }

    const vertexCount = uvVertices.length;
    const minVerts = Math.max(512, Math.floor(readGlobalNumber("__GPU_SURFACE_EVAL_MIN_VERTS__") ?? 2048));
    if (vertexCount < minVerts) {
        return null;
    }

    return await new Promise((resolve) => {
        pendingJobs.push({
            surface,
            uvVertices,
            resolve,
        });
        scheduleFlush();
    });
}

let hasWarnedGpuGridEvalFailure = false;

interface PendingGridEvalJob {
    surface: SupportedSurface;
    gridDensityU: number;
    gridDensityV: number;
    uMin: number;
    vMin: number;
    du: number;
    dv: number;
    resolve: (result: { positions: Float32Array; normals: Float32Array } | null) => void;
}

let pendingGridJobs: PendingGridEvalJob[] = [];
let gridFlushInFlight = false;
let gridFlushScheduled = false;

function getSurfaceGridEvalBatchDelayMs(): number {
    const explicitDelay = readGlobalNumber("__GPU_SURFACE_GRID_BATCH_DELAY_MS__");
    if (explicitDelay !== undefined) {
        return Math.max(0, Math.min(32, explicitDelay));
    }
    return 0;
}

function scheduleGridFlush() {
    if (gridFlushScheduled) return;
    gridFlushScheduled = true;
    const delayMs = getSurfaceGridEvalBatchDelayMs();
    if (delayMs <= 0) {
        queueMicrotask(() => {
            gridFlushScheduled = false;
            void flushPendingGridJobs();
        });
        return;
    }
    setTimeout(() => {
        gridFlushScheduled = false;
        void flushPendingGridJobs();
    }, delayMs);
}

async function flushPendingGridJobs() {
    if (gridFlushInFlight) return;
    gridFlushInFlight = true;
    try {
        while (pendingGridJobs.length > 0) {
            const jobs = pendingGridJobs;
            pendingGridJobs = [];
            await runDenseGridBatch(jobs);
        }
    } finally {
        gridFlushInFlight = false;
    }
}

async function runDenseGridBatch(jobs: PendingGridEvalJob[]) {
    if (jobs.length === 0) return;
    try {
        const { device, pipeline } = await ensureGridPipeline();

        const surfaceParamsF32 = new Float32Array(jobs.length * FLOATS_PER_SURFACE);
        const surfaceParamsU32 = new Uint32Array(surfaceParamsF32.buffer);
        const jobParamsU32 = new Uint32Array(jobs.length * 8);
        const jobParamsF32 = new Float32Array(jobParamsU32.buffer);

        let totalVerts = 0;
        for (const job of jobs) {
            totalVerts += (job.gridDensityU + 1) * (job.gridDensityV + 1);
        }
        const vertexJobIndex = new Uint32Array(totalVerts);
        const ranges: Array<{ start: number; count: number }> = [];

        let cursor = 0;
        for (let jobIndex = 0; jobIndex < jobs.length; jobIndex++) {
            const job = jobs[jobIndex];
            const count = (job.gridDensityU + 1) * (job.gridDensityV + 1);
            ranges.push({ start: cursor, count });
            vertexJobIndex.fill(jobIndex, cursor, cursor + count);

            writeSurfaceParams(surfaceParamsF32, surfaceParamsU32, jobIndex, job.surface);

            const base = jobIndex * 8;
            jobParamsU32[base + 0] = job.gridDensityU;
            jobParamsU32[base + 1] = job.gridDensityV;
            jobParamsU32[base + 2] = cursor;
            jobParamsU32[base + 3] = jobIndex;
            jobParamsF32[base + 4] = job.uMin;
            jobParamsF32[base + 5] = job.vMin;
            jobParamsF32[base + 6] = job.du;
            jobParamsF32[base + 7] = job.dv;

            cursor += count;
        }

        const outputBytes = totalVerts * 3 * 4;
        const dispatchParams = new Uint32Array([totalVerts, 0, 0, 0]);

        const surfacesInfo = ensureBuffer(
            device,
            gpuGridEvalCache.surfacesBuffer,
            gpuGridEvalCache.surfacesCapacity,
            surfaceParamsF32.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "surface-eval-grid-surfaces"
        );
        gpuGridEvalCache.surfacesBuffer = surfacesInfo.buffer;
        gpuGridEvalCache.surfacesCapacity = surfacesInfo.capacity;

        const jobsInfo = ensureBuffer(
            device,
            gpuGridEvalCache.jobsBuffer,
            gpuGridEvalCache.jobsCapacity,
            jobParamsU32.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "surface-eval-grid-jobs"
        );
        gpuGridEvalCache.jobsBuffer = jobsInfo.buffer;
        gpuGridEvalCache.jobsCapacity = jobsInfo.capacity;

        const vertexJobIndexInfo = ensureBuffer(
            device,
            gpuGridEvalCache.vertexJobIndexBuffer,
            gpuGridEvalCache.vertexJobIndexCapacity,
            vertexJobIndex.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "surface-eval-grid-vertex-job-index"
        );
        gpuGridEvalCache.vertexJobIndexBuffer = vertexJobIndexInfo.buffer;
        gpuGridEvalCache.vertexJobIndexCapacity = vertexJobIndexInfo.capacity;

        const positionsInfo = ensureBuffer(
            device,
            gpuGridEvalCache.positionsBuffer,
            gpuGridEvalCache.positionsCapacity,
            outputBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "surface-eval-grid-positions"
        );
        gpuGridEvalCache.positionsBuffer = positionsInfo.buffer;
        gpuGridEvalCache.positionsCapacity = positionsInfo.capacity;

        const normalsInfo = ensureBuffer(
            device,
            gpuGridEvalCache.normalsBuffer,
            gpuGridEvalCache.normalsCapacity,
            outputBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "surface-eval-grid-normals"
        );
        gpuGridEvalCache.normalsBuffer = normalsInfo.buffer;
        gpuGridEvalCache.normalsCapacity = normalsInfo.capacity;

        const stagingPositionsInfo = ensureBuffer(
            device,
            gpuGridEvalCache.stagingPositionsBuffer,
            gpuGridEvalCache.stagingPositionsCapacity,
            outputBytes,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "surface-eval-grid-positions-staging"
        );
        gpuGridEvalCache.stagingPositionsBuffer = stagingPositionsInfo.buffer;
        gpuGridEvalCache.stagingPositionsCapacity = stagingPositionsInfo.capacity;

        const stagingNormalsInfo = ensureBuffer(
            device,
            gpuGridEvalCache.stagingNormalsBuffer,
            gpuGridEvalCache.stagingNormalsCapacity,
            outputBytes,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "surface-eval-grid-normals-staging"
        );
        gpuGridEvalCache.stagingNormalsBuffer = stagingNormalsInfo.buffer;
        gpuGridEvalCache.stagingNormalsCapacity = stagingNormalsInfo.capacity;

        device.queue.writeBuffer(gpuGridEvalCache.surfacesBuffer, 0, surfaceParamsF32);
        device.queue.writeBuffer(gpuGridEvalCache.jobsBuffer, 0, jobParamsU32);
        device.queue.writeBuffer(gpuGridEvalCache.vertexJobIndexBuffer, 0, vertexJobIndex);
        device.queue.writeBuffer(gpuGridEvalCache.dispatchParamsBuffer!, 0, dispatchParams);

        const needsBindGroupRefresh =
            !gpuGridEvalCache.bindGroup ||
            gpuGridEvalCache.bindGroupRefs.surfaces !== gpuGridEvalCache.surfacesBuffer ||
            gpuGridEvalCache.bindGroupRefs.jobs !== gpuGridEvalCache.jobsBuffer ||
            gpuGridEvalCache.bindGroupRefs.vertexJobIndex !== gpuGridEvalCache.vertexJobIndexBuffer ||
            gpuGridEvalCache.bindGroupRefs.dispatch !== gpuGridEvalCache.dispatchParamsBuffer ||
            gpuGridEvalCache.bindGroupRefs.positions !== gpuGridEvalCache.positionsBuffer ||
            gpuGridEvalCache.bindGroupRefs.normals !== gpuGridEvalCache.normalsBuffer;

        if (needsBindGroupRefresh) {
            gpuGridEvalCache.bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: gpuGridEvalCache.surfacesBuffer } },
                    { binding: 1, resource: { buffer: gpuGridEvalCache.jobsBuffer } },
                    { binding: 2, resource: { buffer: gpuGridEvalCache.vertexJobIndexBuffer } },
                    { binding: 3, resource: { buffer: gpuGridEvalCache.dispatchParamsBuffer! } },
                    { binding: 4, resource: { buffer: gpuGridEvalCache.positionsBuffer } },
                    { binding: 5, resource: { buffer: gpuGridEvalCache.normalsBuffer } },
                ],
            });
            gpuGridEvalCache.bindGroupRefs.surfaces = gpuGridEvalCache.surfacesBuffer;
            gpuGridEvalCache.bindGroupRefs.jobs = gpuGridEvalCache.jobsBuffer;
            gpuGridEvalCache.bindGroupRefs.vertexJobIndex = gpuGridEvalCache.vertexJobIndexBuffer;
            gpuGridEvalCache.bindGroupRefs.dispatch = gpuGridEvalCache.dispatchParamsBuffer;
            gpuGridEvalCache.bindGroupRefs.positions = gpuGridEvalCache.positionsBuffer;
            gpuGridEvalCache.bindGroupRefs.normals = gpuGridEvalCache.normalsBuffer;
        }

        const encoder = device.createCommandEncoder({ label: "surface-eval-grid-batch-encoder" });
        const pass = encoder.beginComputePass({ label: "surface-eval-grid-batch-pass" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, gpuGridEvalCache.bindGroup!);
        pass.dispatchWorkgroups(Math.ceil(totalVerts / 256));
        pass.end();

        encoder.copyBufferToBuffer(gpuGridEvalCache.positionsBuffer, 0, gpuGridEvalCache.stagingPositionsBuffer, 0, outputBytes);
        encoder.copyBufferToBuffer(gpuGridEvalCache.normalsBuffer, 0, gpuGridEvalCache.stagingNormalsBuffer, 0, outputBytes);
        device.queue.submit([encoder.finish()]);

        await Promise.all([
            gpuGridEvalCache.stagingPositionsBuffer.mapAsync(GPUMapMode.READ, 0, outputBytes),
            gpuGridEvalCache.stagingNormalsBuffer.mapAsync(GPUMapMode.READ, 0, outputBytes),
        ]);

        const positionsMapped = gpuGridEvalCache.stagingPositionsBuffer.getMappedRange(0, outputBytes);
        const normalsMapped = gpuGridEvalCache.stagingNormalsBuffer.getMappedRange(0, outputBytes);
        const positionsView = new Float32Array(positionsMapped);
        const normalsView = new Float32Array(normalsMapped);

        try {
            for (let i = 0; i < jobs.length; i++) {
                const { start, count } = ranges[i];
                const base = start * 3;
                const end = base + count * 3;
                const positions = new Float32Array(count * 3);
                const normals = new Float32Array(count * 3);
                positions.set(positionsView.subarray(base, end));
                normals.set(normalsView.subarray(base, end));
                jobs[i].resolve({ positions, normals });
            }
        } finally {
            gpuGridEvalCache.stagingPositionsBuffer.unmap();
            gpuGridEvalCache.stagingNormalsBuffer.unmap();
        }
    } catch (err) {
        if (!hasWarnedGpuGridEvalFailure) {
            console.warn("[surface-eval-gpu] Dense grid GPU batch failed; falling back.", err);
            hasWarnedGpuGridEvalFailure = true;
        }
        for (const job of jobs) {
            job.resolve(null);
        }
    }
}

export async function evaluateSurfaceDenseGridGPU(
    surface: Surface,
    gridDensityU: number,
    gridDensityV: number,
    uMin: number,
    vMin: number,
    du: number,
    dv: number
): Promise<{ positions: Float32Array; normals: Float32Array } | null> {
    if (!readGlobalBoolean("__ENABLE_GPU_SURFACE_GRID_EVAL__", true)) {
        return null;
    }
    if (!isSupportedSurface(surface)) {
        return null;
    }

    const vertexCount = (gridDensityU + 1) * (gridDensityV + 1);
    const minVerts = Math.max(1024, Math.floor(readGlobalNumber("__GPU_SURFACE_GRID_EVAL_MIN_VERTS__") ?? 4096));
    if (vertexCount < minVerts) {
        return null;
    }

    return await new Promise((resolve) => {
        pendingGridJobs.push({
            surface,
            gridDensityU,
            gridDensityV,
            uMin,
            vMin,
            du,
            dv,
            resolve,
        });
        scheduleGridFlush();
    });
}
