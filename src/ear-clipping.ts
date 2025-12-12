import { BYTE_SIZE } from "./gpu-tesselate";
import { getGPUDevice, normalizePoints } from "./lib";

const VERTEX_PER_TRIANGLE = 3;

// The points are assumed to be in CCW.
export async function earClipping(points: number[][]) {
    try {
        const device = await getGPUDevice();
        const buffers = initializeBuffers(device, points);

        // now we have the buffers
        // we want to create the layout for the compute pass

        const bindGroupConfiguration = buffers.map((buffer,i) => {
            return {
                binding: i,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" as GPUBufferBindingType},
            }
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [ ...bindGroupConfiguration]
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [
                bindGroupLayout
            ]
        });

        const bindGroupEntries = buffers.map((buffer, i) => {
            return {
                binding: i,
                resource: {buffer: buffer}
            };
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: bindGroupEntries
        });

        // now I want to create the compute pipeline

        const convexityCheckShader = device.createShaderModule({
            code: `
                /* wgsl */
                struct Point {
                    position: vec3<f32>,
                    padding: f32
                }
                const REFLEX = 0u;
                const CONVEX = 1;
                const COLLINEAR = 2;

                const WORKGROUP_SIZE = 32u;
                @group(0) @binding(0) var<storage, read> pointsBuffer: array<Point>;
                @group(0) @binding(1) var<storage, read_write> vertexIsEarBuffer: array<u32>;
                @group(0) @binding(2) var<storage, read> previousVertexBuffer: array<u32>;
                @group(0) @binding(3) var<storage, read> nextVertexBuffer: array<u32>;
                @group(0) @binding(4) var<storage, read> activeCount: u32;
                @group(0) @binding(5) var<storage, read> triangleCount: u32;
                @group(0) @binding(6) var<storage, read> classifiedPointsBuffer: array<u32>;

                @compute @workgroup_size(WORKGROUP_SIZE) fn classifyPoints(
                    @builtin(global_invocation_id) id: vec3<u32>
                ) {
                    let i = id.x;
                    let pointsBufferLength = arrayLength(&pointsBuffer);
                    if (i >= pointsBufferLength) {
                        return;
                    }

                    var A: Point;
                    var B: Point;
                    var C: Point;

                    if (i == 0) {
                        A = pointsBuffer[pointsBufferLength - 1];
                        B = pointsBuffer[i];
                        C = pointsBuffer[i+1];

                    } else if (i == pointsBufferLength - 1) {
                        A = pointsBuffer[i-1];
                        B = pointsBuffer[i];
                        C = pointsBuffer[0];
                    } else {
                        A = pointsBuffer[i-1];
                        B = pointsBuffer[i];
                        C = pointsBuffer[i+1];
                    }

                    var E1 = B.position - A.position;
                    var E2 = C.position - B.position;
                    var crossProduct = E1.x * E2.y - E1.y * E2.x;
                    if (crossProduct > 0) {
                        classifiedPoints[i] = CONVEX;
                    } else if (crossProduct < 0) {
                        classifiedPoints[i] = REFLEX;
                    }
                }
            `
        });


        




    } catch (e) {
        throw e;
    }

}

function initializeBuffers(device: GPUDevice, points: number[][]) {
    try {
        const normalizedPoints = normalizePoints(points);
        const normalizedPointsFArray = new Float32Array(normalizedPoints.flat());
        const pointsBuffer = device.createBuffer({
            label: `EarClipping - pointsBuffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: normalizedPointsFArray.byteLength
        });
        // write into the buffer
        device.queue.writeBuffer(pointsBuffer, 0, normalizedPointsFArray);

        const outputIndicesBuffer = device.createBuffer({
            label: `EarClipping - outputIndices buffer`,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: VERTEX_PER_TRIANGLE*(points.length - 2)* BYTE_SIZE// this should be 3  * the number of triangles, so 
        });

        const vertexIsEarBuffer = device.createBuffer({
            label: `EarClipping - vertexIsEarBuffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE * points.length
        });

        const previousVertexBuffer = device.createBuffer({
            label: `EarClipping - previousVertexBuffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE*points.length
        });

        // we need to initialize this
        // we know that the the vertices are in CCW order.
        const prev = new Uint32Array(points.length);
        const next = new Uint32Array(points.length);

        for (let i = 0; i < points.length; i++) {
            prev[i] = (i === 0) ? (points.length - 1) : (i - 1);
            next[i] = (i === points.length - 1) ? 0 : (i + 1);
        }

        const nextVertexBuffer = device.createBuffer({
            label: `EarClipping - nextVertexBuffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE*points.length
        });

        device.queue.writeBuffer(previousVertexBuffer, 0, prev);
        device.queue.writeBuffer(nextVertexBuffer, 0, next);

        const activeCount = device.createBuffer({
            label: `EarClipping - activeCount`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE, // we need one 4 byte to keep track of the active count.
        });

        device.queue.writeBuffer(activeCount, 0, new Uint32Array(points.length));

        const triangleCount = device.createBuffer({
            label: `EarClipping - triangleCount`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE
        });

        device.queue.writeBuffer(triangleCount, 0, new Uint32Array(0)); // initialize to 0 because no active triangles

        const classifiedPointsBuffer = device.createBuffer({
            label: `EarClipping - classifiedPointsBuffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE * points.length
        });

        return [
            pointsBuffer, 
            outputIndicesBuffer,
            vertexIsEarBuffer,
            previousVertexBuffer,
            nextVertexBuffer,
            activeCount,
            triangleCount,
            classifiedPointsBuffer
        ];
    } catch (e) {
        throw e;
    }
}