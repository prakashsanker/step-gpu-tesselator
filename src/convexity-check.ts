import { getGPUDevice, normalizePoints } from "./lib";

// make sure that points is in counter clockwise order
export async function classifyPoints(points: number[][]) {
    try { 

        const device = await getGPUDevice();

        // Ensure all points are 3D (x, y, z) - pad with 0 if needed
        const normalizedPoints = normalizePoints(points);
        
        const numPoints = normalizedPoints.length;
        const flattenedPoints = new Float32Array(normalizedPoints.flat());
        
        
        const pointsBuffer = device.createBuffer({
            label: `isConvexityCheck - points buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            size: flattenedPoints.byteLength
        });
        
        device.queue.writeBuffer(pointsBuffer, 0, flattenedPoints);

        const classifiedPoints = device.createBuffer({
            label: `isConvexityCheck - classifiedPoints buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: points.length * 4
        });




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
                @group(0) @binding(1) var<storage, read_write> classifiedPoints: array<u32>;

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

        const convexityCheckBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage"}
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage"}
                }
            ]
        });

        const convexityCheckPipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [
                convexityCheckBindGroupLayout
            ]
        });

        const convexityCheckBindGroup = device.createBindGroup({
            layout: convexityCheckBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: pointsBuffer}
                },
                {
                    binding: 1,
                    resource: { buffer: classifiedPoints}
                }
            ]});

        const convexityCheckPipeline = device.createComputePipeline({
            layout: convexityCheckPipelineLayout,
            compute: {
                module: convexityCheckShader,
                entryPoint: "classifyPoints"
            }
        });

        const workgroupSize = 32;
        const numWorkgroups = Math.ceil(numPoints / workgroupSize);
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(convexityCheckPipeline);
        passEncoder.setBindGroup(0, convexityCheckBindGroup);
        passEncoder.dispatchWorkgroups(numWorkgroups);
        passEncoder.end();


        const readBuffer = device.createBuffer({
            label: `isConvexityCheck - read buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            size: points.length * 4
        });
        commandEncoder.copyBufferToBuffer(classifiedPoints, 0, readBuffer, 0, points.length * 4);
        device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const mappedRange = readBuffer.getMappedRange();
        const result = new Uint32Array(mappedRange.slice());
        readBuffer.unmap();

        return result;

    } catch (e) {
        throw e;
    }



}