import { getGPUDevice } from "./lib";

export interface MeshBatchFaceInput {
    positions: Float32Array;
    indices: Uint32Array;
    reverseWinding?: boolean;
}

export interface MeshBatchAssemblyResult {
    positions: Float32Array;
    indices: Uint32Array;
    vertexCounts: Uint32Array;
    indexCounts: Uint32Array;
}

const PREFIX_SUM_SHADER = /* wgsl */ `
struct Params {
    faceCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> vertexCounts: array<u32>;
@group(0) @binding(1) var<storage, read> indexCounts: array<u32>;
@group(0) @binding(2) var<storage, read_write> vertexOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> indexOffsets: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x != 0u) {
        return;
    }

    var vOffset = 0u;
    var iOffset = 0u;
    var i = 0u;
    loop {
        if (i >= params.faceCount) {
            break;
        }
        vertexOffsets[i] = vOffset;
        indexOffsets[i] = iOffset;
        vOffset = vOffset + vertexCounts[i];
        iOffset = iOffset + indexCounts[i];
        i = i + 1u;
    }
}
`;

const ASSEMBLY_SHADER = /* wgsl */ `
struct Params {
    faceCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> positionsIn: array<f32>;
@group(0) @binding(1) var<storage, read> indicesIn: array<u32>;
@group(0) @binding(2) var<storage, read> sourceVertexOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> sourceIndexOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> vertexOffsets: array<u32>;
@group(0) @binding(5) var<storage, read> indexOffsets: array<u32>;
@group(0) @binding(6) var<storage, read> vertexCounts: array<u32>;
@group(0) @binding(7) var<storage, read> indexCounts: array<u32>;
@group(0) @binding(8) var<storage, read> reverseFlags: array<u32>;
@group(0) @binding(9) var<storage, read_write> positionsOut: array<f32>;
@group(0) @binding(10) var<storage, read_write> indicesOut: array<u32>;
@group(0) @binding(11) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let faceIdx = gid.x;
    if (faceIdx >= params.faceCount) {
        return;
    }

    let srcVertexOffset = sourceVertexOffsets[faceIdx];
    let srcIndexOffset = sourceIndexOffsets[faceIdx];
    let dstVertexOffset = vertexOffsets[faceIdx];
    let dstIndexOffset = indexOffsets[faceIdx];
    let vertexCount = vertexCounts[faceIdx];
    let indexCount = indexCounts[faceIdx];
    let reverse = reverseFlags[faceIdx] != 0u;

    let srcPosBase = srcVertexOffset * 3u;
    let dstPosBase = dstVertexOffset * 3u;
    let positionCount = vertexCount * 3u;
    var p = 0u;
    loop {
        if (p >= positionCount) {
            break;
        }
        positionsOut[dstPosBase + p] = positionsIn[srcPosBase + p];
        p = p + 1u;
    }

    var t = 0u;
    loop {
        if (t + 2u >= indexCount) {
            break;
        }
        let localI0 = indicesIn[srcIndexOffset + t + 0u];
        let localI1 = indicesIn[srcIndexOffset + t + 1u];
        let localI2 = indicesIn[srcIndexOffset + t + 2u];
        let globalI0 = localI0 + dstVertexOffset;
        let globalI1 = localI1 + dstVertexOffset;
        let globalI2 = localI2 + dstVertexOffset;
        let outBase = dstIndexOffset + t;

        if (reverse) {
            indicesOut[outBase + 0u] = globalI0;
            indicesOut[outBase + 1u] = globalI2;
            indicesOut[outBase + 2u] = globalI1;
        } else {
            indicesOut[outBase + 0u] = globalI0;
            indicesOut[outBase + 1u] = globalI1;
            indicesOut[outBase + 2u] = globalI2;
        }
        t = t + 3u;
    }
}
`;

const cache: {
    device: GPUDevice | null;
    prefixPipeline: GPUComputePipeline | null;
    assemblyPipeline: GPUComputePipeline | null;
    paramsBuffer: GPUBuffer | null;
    vertexCountsBuffer: GPUBuffer | null;
    indexCountsBuffer: GPUBuffer | null;
    sourceVertexOffsetsBuffer: GPUBuffer | null;
    sourceIndexOffsetsBuffer: GPUBuffer | null;
    vertexOffsetsBuffer: GPUBuffer | null;
    indexOffsetsBuffer: GPUBuffer | null;
    reverseFlagsBuffer: GPUBuffer | null;
    positionsInBuffer: GPUBuffer | null;
    indicesInBuffer: GPUBuffer | null;
    positionsOutBuffer: GPUBuffer | null;
    indicesOutBuffer: GPUBuffer | null;
    stagingBuffer: GPUBuffer | null;
    vertexCountsCapacity: number;
    indexCountsCapacity: number;
    sourceVertexOffsetsCapacity: number;
    sourceIndexOffsetsCapacity: number;
    vertexOffsetsCapacity: number;
    indexOffsetsCapacity: number;
    reverseFlagsCapacity: number;
    positionsInCapacity: number;
    indicesInCapacity: number;
    positionsOutCapacity: number;
    indicesOutCapacity: number;
    stagingCapacity: number;
} = {
    device: null,
    prefixPipeline: null,
    assemblyPipeline: null,
    paramsBuffer: null,
    vertexCountsBuffer: null,
    indexCountsBuffer: null,
    sourceVertexOffsetsBuffer: null,
    sourceIndexOffsetsBuffer: null,
    vertexOffsetsBuffer: null,
    indexOffsetsBuffer: null,
    reverseFlagsBuffer: null,
    positionsInBuffer: null,
    indicesInBuffer: null,
    positionsOutBuffer: null,
    indicesOutBuffer: null,
    stagingBuffer: null,
    vertexCountsCapacity: 0,
    indexCountsCapacity: 0,
    sourceVertexOffsetsCapacity: 0,
    sourceIndexOffsetsCapacity: 0,
    vertexOffsetsCapacity: 0,
    indexOffsetsCapacity: 0,
    reverseFlagsCapacity: 0,
    positionsInCapacity: 0,
    indicesInCapacity: 0,
    positionsOutCapacity: 0,
    indicesOutCapacity: 0,
    stagingCapacity: 0,
};

function roundCapacity(size: number): number {
    const alignment = 256;
    return Math.max(alignment, Math.ceil(size / alignment) * alignment);
}

function ensureBuffer(
    device: GPUDevice,
    existing: GPUBuffer | null,
    existingCapacity: number,
    requiredBytes: number,
    usage: GPUBufferUsageFlags,
    label: string
): { buffer: GPUBuffer; capacity: number } {
    const needed = roundCapacity(requiredBytes);
    if (existing && existingCapacity >= needed) {
        return { buffer: existing, capacity: existingCapacity };
    }
    existing?.destroy();
    const buffer = device.createBuffer({
        label,
        size: needed,
        usage,
    });
    return { buffer, capacity: needed };
}

function destroyCache(): void {
    cache.paramsBuffer?.destroy();
    cache.vertexCountsBuffer?.destroy();
    cache.indexCountsBuffer?.destroy();
    cache.sourceVertexOffsetsBuffer?.destroy();
    cache.sourceIndexOffsetsBuffer?.destroy();
    cache.vertexOffsetsBuffer?.destroy();
    cache.indexOffsetsBuffer?.destroy();
    cache.reverseFlagsBuffer?.destroy();
    cache.positionsInBuffer?.destroy();
    cache.indicesInBuffer?.destroy();
    cache.positionsOutBuffer?.destroy();
    cache.indicesOutBuffer?.destroy();
    cache.stagingBuffer?.destroy();
    cache.paramsBuffer = null;
    cache.vertexCountsBuffer = null;
    cache.indexCountsBuffer = null;
    cache.sourceVertexOffsetsBuffer = null;
    cache.sourceIndexOffsetsBuffer = null;
    cache.vertexOffsetsBuffer = null;
    cache.indexOffsetsBuffer = null;
    cache.reverseFlagsBuffer = null;
    cache.positionsInBuffer = null;
    cache.indicesInBuffer = null;
    cache.positionsOutBuffer = null;
    cache.indicesOutBuffer = null;
    cache.stagingBuffer = null;
    cache.vertexCountsCapacity = 0;
    cache.indexCountsCapacity = 0;
    cache.sourceVertexOffsetsCapacity = 0;
    cache.sourceIndexOffsetsCapacity = 0;
    cache.vertexOffsetsCapacity = 0;
    cache.indexOffsetsCapacity = 0;
    cache.reverseFlagsCapacity = 0;
    cache.positionsInCapacity = 0;
    cache.indicesInCapacity = 0;
    cache.positionsOutCapacity = 0;
    cache.indicesOutCapacity = 0;
    cache.stagingCapacity = 0;
}

async function ensurePipelines(): Promise<{ device: GPUDevice; prefixPipeline: GPUComputePipeline; assemblyPipeline: GPUComputePipeline }> {
    const device = await getGPUDevice();
    if (cache.device !== device) {
        destroyCache();
        cache.prefixPipeline = null;
        cache.assemblyPipeline = null;
        cache.device = device;
    }

    if (!cache.prefixPipeline) {
        const module = device.createShaderModule({
            label: "mesh-batch-prefix-sum-shader",
            code: PREFIX_SUM_SHADER,
        });
        cache.prefixPipeline = device.createComputePipeline({
            label: "mesh-batch-prefix-sum-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "main",
            },
        });
    }

    if (!cache.assemblyPipeline) {
        const module = device.createShaderModule({
            label: "mesh-batch-assembly-shader",
            code: ASSEMBLY_SHADER,
        });
        cache.assemblyPipeline = device.createComputePipeline({
            label: "mesh-batch-assembly-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "main",
            },
        });
    }

    if (!cache.paramsBuffer) {
        cache.paramsBuffer = device.createBuffer({
            label: "mesh-batch-params",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    return { device, prefixPipeline: cache.prefixPipeline, assemblyPipeline: cache.assemblyPipeline };
}

let loggedBatchAssemblyFailure = false;

export async function assembleMeshBatchGPU(
    faces: MeshBatchFaceInput[]
): Promise<MeshBatchAssemblyResult | null> {
    if (faces.length === 0) {
        return {
            positions: new Float32Array(0),
            indices: new Uint32Array(0),
            vertexCounts: new Uint32Array(0),
            indexCounts: new Uint32Array(0),
        };
    }

    try {
        const { device, prefixPipeline, assemblyPipeline } = await ensurePipelines();
        const faceCount = faces.length;

        const vertexCounts = new Uint32Array(faceCount);
        const indexCounts = new Uint32Array(faceCount);
        const sourceVertexOffsets = new Uint32Array(faceCount);
        const sourceIndexOffsets = new Uint32Array(faceCount);
        const reverseFlags = new Uint32Array(faceCount);

        let totalVertices = 0;
        let totalIndices = 0;
        for (let i = 0; i < faceCount; i++) {
            const face = faces[i];
            const vertexCount = Math.floor(face.positions.length / 3);
            const indexCount = face.indices.length;
            vertexCounts[i] = vertexCount;
            indexCounts[i] = indexCount;
            sourceVertexOffsets[i] = totalVertices;
            sourceIndexOffsets[i] = totalIndices;
            reverseFlags[i] = face.reverseWinding ? 1 : 0;
            totalVertices += vertexCount;
            totalIndices += indexCount;
        }

        if (totalVertices === 0 || totalIndices === 0) {
            return {
                positions: new Float32Array(0),
                indices: new Uint32Array(0),
                vertexCounts,
                indexCounts,
            };
        }

        const positionsIn = new Float32Array(totalVertices * 3);
        const indicesIn = new Uint32Array(totalIndices);
        for (let i = 0; i < faceCount; i++) {
            const face = faces[i];
            positionsIn.set(face.positions, sourceVertexOffsets[i] * 3);
            indicesIn.set(face.indices, sourceIndexOffsets[i]);
        }

        const countsBytes = faceCount * 4;
        const positionsBytes = positionsIn.byteLength;
        const indicesBytes = indicesIn.byteLength;
        const stagingBytes = positionsBytes + indicesBytes;

        const vertexCountsInfo = ensureBuffer(
            device,
            cache.vertexCountsBuffer,
            cache.vertexCountsCapacity,
            countsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-vertex-counts"
        );
        cache.vertexCountsBuffer = vertexCountsInfo.buffer;
        cache.vertexCountsCapacity = vertexCountsInfo.capacity;

        const indexCountsInfo = ensureBuffer(
            device,
            cache.indexCountsBuffer,
            cache.indexCountsCapacity,
            countsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-index-counts"
        );
        cache.indexCountsBuffer = indexCountsInfo.buffer;
        cache.indexCountsCapacity = indexCountsInfo.capacity;

        const sourceVertexOffsetsInfo = ensureBuffer(
            device,
            cache.sourceVertexOffsetsBuffer,
            cache.sourceVertexOffsetsCapacity,
            countsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-source-vertex-offsets"
        );
        cache.sourceVertexOffsetsBuffer = sourceVertexOffsetsInfo.buffer;
        cache.sourceVertexOffsetsCapacity = sourceVertexOffsetsInfo.capacity;

        const sourceIndexOffsetsInfo = ensureBuffer(
            device,
            cache.sourceIndexOffsetsBuffer,
            cache.sourceIndexOffsetsCapacity,
            countsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-source-index-offsets"
        );
        cache.sourceIndexOffsetsBuffer = sourceIndexOffsetsInfo.buffer;
        cache.sourceIndexOffsetsCapacity = sourceIndexOffsetsInfo.capacity;

        const vertexOffsetsInfo = ensureBuffer(
            device,
            cache.vertexOffsetsBuffer,
            cache.vertexOffsetsCapacity,
            countsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-vertex-offsets"
        );
        cache.vertexOffsetsBuffer = vertexOffsetsInfo.buffer;
        cache.vertexOffsetsCapacity = vertexOffsetsInfo.capacity;

        const indexOffsetsInfo = ensureBuffer(
            device,
            cache.indexOffsetsBuffer,
            cache.indexOffsetsCapacity,
            countsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-index-offsets"
        );
        cache.indexOffsetsBuffer = indexOffsetsInfo.buffer;
        cache.indexOffsetsCapacity = indexOffsetsInfo.capacity;

        const reverseFlagsInfo = ensureBuffer(
            device,
            cache.reverseFlagsBuffer,
            cache.reverseFlagsCapacity,
            countsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-reverse-flags"
        );
        cache.reverseFlagsBuffer = reverseFlagsInfo.buffer;
        cache.reverseFlagsCapacity = reverseFlagsInfo.capacity;

        const positionsInInfo = ensureBuffer(
            device,
            cache.positionsInBuffer,
            cache.positionsInCapacity,
            positionsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-positions-in"
        );
        cache.positionsInBuffer = positionsInInfo.buffer;
        cache.positionsInCapacity = positionsInInfo.capacity;

        const indicesInInfo = ensureBuffer(
            device,
            cache.indicesInBuffer,
            cache.indicesInCapacity,
            indicesBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "mesh-batch-indices-in"
        );
        cache.indicesInBuffer = indicesInInfo.buffer;
        cache.indicesInCapacity = indicesInInfo.capacity;

        const positionsOutInfo = ensureBuffer(
            device,
            cache.positionsOutBuffer,
            cache.positionsOutCapacity,
            positionsBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "mesh-batch-positions-out"
        );
        cache.positionsOutBuffer = positionsOutInfo.buffer;
        cache.positionsOutCapacity = positionsOutInfo.capacity;

        const indicesOutInfo = ensureBuffer(
            device,
            cache.indicesOutBuffer,
            cache.indicesOutCapacity,
            indicesBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "mesh-batch-indices-out"
        );
        cache.indicesOutBuffer = indicesOutInfo.buffer;
        cache.indicesOutCapacity = indicesOutInfo.capacity;

        const stagingInfo = ensureBuffer(
            device,
            cache.stagingBuffer,
            cache.stagingCapacity,
            stagingBytes,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "mesh-batch-staging"
        );
        cache.stagingBuffer = stagingInfo.buffer;
        cache.stagingCapacity = stagingInfo.capacity;

        const params = new Uint32Array([faceCount, 0, 0, 0]);
        const zeroOffsets = new Uint32Array(faceCount);
        device.queue.writeBuffer(cache.paramsBuffer!, 0, params);
        device.queue.writeBuffer(cache.vertexCountsBuffer, 0, vertexCounts);
        device.queue.writeBuffer(cache.indexCountsBuffer, 0, indexCounts);
        device.queue.writeBuffer(cache.sourceVertexOffsetsBuffer, 0, sourceVertexOffsets);
        device.queue.writeBuffer(cache.sourceIndexOffsetsBuffer, 0, sourceIndexOffsets);
        device.queue.writeBuffer(cache.vertexOffsetsBuffer, 0, zeroOffsets);
        device.queue.writeBuffer(cache.indexOffsetsBuffer, 0, zeroOffsets);
        device.queue.writeBuffer(cache.reverseFlagsBuffer, 0, reverseFlags);
        device.queue.writeBuffer(cache.positionsInBuffer, 0, positionsIn);
        device.queue.writeBuffer(cache.indicesInBuffer, 0, indicesIn);

        const prefixBindGroup = device.createBindGroup({
            layout: prefixPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cache.vertexCountsBuffer } },
                { binding: 1, resource: { buffer: cache.indexCountsBuffer } },
                { binding: 2, resource: { buffer: cache.vertexOffsetsBuffer } },
                { binding: 3, resource: { buffer: cache.indexOffsetsBuffer } },
                { binding: 4, resource: { buffer: cache.paramsBuffer! } },
            ],
        });

        const assemblyBindGroup = device.createBindGroup({
            layout: assemblyPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cache.positionsInBuffer } },
                { binding: 1, resource: { buffer: cache.indicesInBuffer } },
                { binding: 2, resource: { buffer: cache.sourceVertexOffsetsBuffer } },
                { binding: 3, resource: { buffer: cache.sourceIndexOffsetsBuffer } },
                { binding: 4, resource: { buffer: cache.vertexOffsetsBuffer } },
                { binding: 5, resource: { buffer: cache.indexOffsetsBuffer } },
                { binding: 6, resource: { buffer: cache.vertexCountsBuffer } },
                { binding: 7, resource: { buffer: cache.indexCountsBuffer } },
                { binding: 8, resource: { buffer: cache.reverseFlagsBuffer } },
                { binding: 9, resource: { buffer: cache.positionsOutBuffer } },
                { binding: 10, resource: { buffer: cache.indicesOutBuffer } },
                { binding: 11, resource: { buffer: cache.paramsBuffer! } },
            ],
        });

        const encoder = device.createCommandEncoder({ label: "mesh-batch-assembly-encoder" });
        const prefixPass = encoder.beginComputePass({ label: "mesh-batch-prefix-sum-pass" });
        prefixPass.setPipeline(prefixPipeline);
        prefixPass.setBindGroup(0, prefixBindGroup);
        prefixPass.dispatchWorkgroups(1);
        prefixPass.end();

        const assemblyPass = encoder.beginComputePass({ label: "mesh-batch-assembly-pass" });
        assemblyPass.setPipeline(assemblyPipeline);
        assemblyPass.setBindGroup(0, assemblyBindGroup);
        assemblyPass.dispatchWorkgroups(Math.ceil(faceCount / 64));
        assemblyPass.end();

        encoder.copyBufferToBuffer(cache.positionsOutBuffer, 0, cache.stagingBuffer, 0, positionsBytes);
        encoder.copyBufferToBuffer(cache.indicesOutBuffer, 0, cache.stagingBuffer, positionsBytes, indicesBytes);
        device.queue.submit([encoder.finish()]);

        await cache.stagingBuffer.mapAsync(GPUMapMode.READ, 0, stagingBytes);
        try {
            const mapped = cache.stagingBuffer.getMappedRange(0, stagingBytes);
            const raw = mapped.slice(0);
            const positions = new Float32Array(raw, 0, positionsBytes / 4);
            const indices = new Uint32Array(raw, positionsBytes, indicesBytes / 4);
            const positionsCopy = new Float32Array(positions.length);
            const indicesCopy = new Uint32Array(indices.length);
            positionsCopy.set(positions);
            indicesCopy.set(indices);
            return {
                positions: positionsCopy,
                indices: indicesCopy,
                vertexCounts,
                indexCounts,
            };
        } finally {
            cache.stagingBuffer.unmap();
        }
    } catch (err) {
        if (!loggedBatchAssemblyFailure) {
            console.warn("[mesh-batch-assembly-gpu] GPU batch assembly failed; falling back.", err);
            loggedBatchAssemblyFailure = true;
        }
        return null;
    }
}
