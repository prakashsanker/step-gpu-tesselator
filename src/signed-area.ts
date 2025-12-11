import { BYTE_SIZE } from "./gpu-tesselate";
import { getGPUDevice } from "./lib";

export function isCounterClockWise(points): boolean{
    let sum = 0;
    for (let i = 0; i < points.length; i++) {
        const currentPoint = points[i];
        let nextPoint;
        if (i === points.length -1 ) {
            nextPoint = points[0];
        } else {
            nextPoint = points[i+1];
        }
        const x1 = currentPoint[0];
        const y1 = currentPoint[1];
        const x2 = nextPoint[0];
        const y2 = nextPoint[1];
        const intermediateSum = x1*y2 - x2*y1;
        sum+= intermediateSum;
    }
    if (sum === 0) {
        throw new Error("collinear or degenerate polygon");
    }

    if (sum > 0) {
        return true;
    }
    return false;
}


export async function isCounterClockWiseGPU(points): boolean {
    /*
        What do I need to do?

        1. I need to load up the points into a buffer
        2. I want the compute shader to run points.length time
    */

        try {
            const device = await getGPUDevice();

            // Extract only x and y coordinates from points (points may be 3D [x, y, z] or 2D [x, y])
            // Flatten to: [x1, y1, x2, y2, ...]
            // This matches WGSL struct layout: array<Point> where Point { x: i32, y: i32 }
            const xyPoints = points.map(p => [p[0], p[1]]);
            const flattenedPoints = xyPoints.flat();
            const intPointsArray = new Int32Array(flattenedPoints);
            
            const pointsBuffer = device.createBuffer({
                label: `isCounterClockWiseGPU - points buffer`,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
                size: intPointsArray.byteLength
            });

            device.queue.writeBuffer(pointsBuffer, 0, intPointsArray);

            const partialSumsBuffer = device.createBuffer({
                label: `isCounterClockWiseGPU - partialSums buffer`,
                usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
                size: points.length*BYTE_SIZE
            });

            const outputBuffer = device.createBuffer({
                label: `isCounterClockwiseGPU - output buffer`,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                size: BYTE_SIZE // I think this is 4 bytes because its only one integer
            });


            const isCCWShader = device.createShaderModule({
                code: `
                    /* wgsl */

                    @group(0) @binding(0) var<storage, read> partialSums: array<i32>;
                    @group(0) @binding(1) var<storage, read_write> output: i32;
                    @compute @workgroup_size(32) fn sum(
                        @builtin(global_invocation_id) id: vec3<u32>
                    ) {
                        let i = id.x;
                        let partialSumsLength = arrayLength(&partialSums);
                        if (i > partialSumsLength) {
                            return;
                        }

                        // now we want to do the sum as a for loop
                        var sum: i32 = 0;
                        for (var i: u32 = 0; i < partialSumsLength; i = i + 1) {
                            sum = partialSums[i] + sum;
                        }

                        output = sum;
                    }
                `
            });

            const partialSumsShader = device.createShaderModule({
                code: `
                /* wgsl */

                struct Point {
                    x: i32,
                    y: i32
                };

                const WORKGROUP_SIZE = 32u;
                @group(0) @binding(0) var<storage, read> pointsBuffer: array<Point>;
                @group(0) @binding(1) var<storage, read_write> partialSums: array<i32>;

                @compute @workgroup_size(WORKGROUP_SIZE) fn sum(
                    @builtin(global_invocation_id) id: vec3<u32>
                
                ) {
                    let pointsBufferLength = arrayLength(&pointsBuffer);
                    if (id.x > pointsBufferLength) {
                        return;
                    }

                    if (id.x < pointsBufferLength) {
                        let i = id.x;
                        var currentPoint = pointsBuffer[i];
                        var nextPoint: Point;
                        if (i == pointsBufferLength -1 ) {
                            nextPoint = pointsBuffer[0u];
                        } else {
                            nextPoint = pointsBuffer[i + 1u];
                        }

                        var x1 = currentPoint.x;
                        var y1 = currentPoint.y;
                        var x2 = nextPoint.x;
                        var y2 = nextPoint.y;

                        var intermediateSum = x1*y2 - x2*y1;
                        // now we want to store this sum in an array
                        partialSums[i] = intermediateSum;
                    }
                }
                `
            });

            const partialSumsBindGroupLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage"},
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {type: "storage"},
                    }
                ]
            });

            const isCCWBindGroupLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage"}                   
                    }, {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "storage"}
                    }
                ]
            });
            const partialSumsLayout = device.createPipelineLayout({
                bindGroupLayouts: [
                    partialSumsBindGroupLayout
                ]
            });

            const isCCWLayout = device.createPipelineLayout({
                bindGroupLayouts: [
                    isCCWBindGroupLayout
                ]
            });

            const partialSumsBindGroup = device.createBindGroup({
                layout: partialSumsBindGroupLayout,
                entries: [{
                        binding: 0,
                        resource: { buffer: pointsBuffer}
                    },
                    {
                        binding: 1,
                        resource: { buffer: partialSumsBuffer}
                    }
                ]
            })

            const isCCWBindGroup = device.createBindGroup({
                layout: isCCWBindGroupLayout,
                entries: [{
                    binding: 0,
                    resource: { buffer: partialSumsBuffer}
                }, {
                    binding: 1,
                    resource: { buffer: outputBuffer }
                }]
            });





            const partialSumsPipeline = device.createComputePipeline({
                layout: partialSumsLayout,
                compute: {
                    module: partialSumsShader,
                    entryPoint: "sum"
                }
            });

            const isCCWPipeline = device.createComputePipeline({
                layout: isCCWLayout,
                compute: {
                    module: isCCWShader,
                    entryPoint: "sum"
                }
            });

            // First compute pass: calculate partial sums
            const workgroupSize = 32;
            const numWorkgroups = Math.ceil(points.length / workgroupSize);

            const partialSumsCommandEncoder = device.createCommandEncoder();
            let passEncoder = partialSumsCommandEncoder.beginComputePass();
            passEncoder.setPipeline(partialSumsPipeline);
            passEncoder.setBindGroup(0, partialSumsBindGroup);
            passEncoder.dispatchWorkgroups(numWorkgroups);
            passEncoder.end();
            device.queue.submit([partialSumsCommandEncoder.finish()]);

            // Second compute pass: sum all partial sums
            const isCCWNumWorkgroups = Math.ceil(1 / workgroupSize); // Only need 1 thread, but minimum is 1 workgroup
            const isCCWCommandEncoder = device.createCommandEncoder();
            passEncoder = isCCWCommandEncoder.beginComputePass();
            passEncoder.setPipeline(isCCWPipeline);
            passEncoder.setBindGroup(0, isCCWBindGroup);
            passEncoder.dispatchWorkgroups(isCCWNumWorkgroups);
            passEncoder.end();
            device.queue.submit([isCCWCommandEncoder.finish()]);

            // Create a staging buffer to read the result
            const stagingBuffer = device.createBuffer({
                label: `isCounterClockWiseGPU - staging buffer`,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                size: BYTE_SIZE
            });

            // Copy the result from outputBuffer to stagingBuffer
            const readCommandEncoder = device.createCommandEncoder();
            readCommandEncoder.copyBufferToBuffer(
                outputBuffer,
                0,
                stagingBuffer,
                0,
                BYTE_SIZE
            );
            device.queue.submit([readCommandEncoder.finish()]);

            // Map the staging buffer and read the result
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const mappedRange = stagingBuffer.getMappedRange();
            const result = new Int32Array(mappedRange)[0]; // Read as signed integer
            stagingBuffer.unmap();

            // Clean up buffers
            // pointsBuffer.destroy(); // We need to reverse the memory 
            partialSumsBuffer.destroy();
            outputBuffer.destroy();
            stagingBuffer.destroy();

            // Return true if counter-clockwise (positive sum), false if clockwise (negative sum)
            if (result === 0) {
                throw new Error("collinear or degenerate polygon");
            }
            return result > 0;
        } catch (e) {
            throw e;
        }
}