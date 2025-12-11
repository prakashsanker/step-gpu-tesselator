import { getGPUDevice } from "./lib";


export async function pointInTriangle(point: number[], triangle: number[][]) {
    try {

        const device = await getGPUDevice();

        const normalizedTriangle = triangle.map(points => {
            if (points.length === 2) return [points[0], points[1], 0, 0];
            if (points.length === 3) return [points[0], points[1], points[2], 0];
            throw new Error(`Invalid point dimension: ${points.length}`);
        })

        const flattenedTriangle = normalizedTriangle.flat();
        const floatTriangle = new Float32Array(flattenedTriangle);

        // Log initial triangleBuffer data
        console.log("=== TRIANGLE BUFFER - INITIAL ===");
        console.log("Triangle array:", Array.from(floatTriangle));
        for (let i = 0; i < normalizedTriangle.length; i++) {
            const offset = i * 4;
            console.log(`Triangle point ${i}: (${floatTriangle[offset]}, ${floatTriangle[offset + 1]}, ${floatTriangle[offset + 2]})`);
        }
        console.log("=== END TRIANGLE BUFFER - INITIAL ===");

        const triangleBuffer = device.createBuffer({
            label: `pointInTriangle - triangle buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: floatTriangle.byteLength
        });

        device.queue.writeBuffer(triangleBuffer, 0, floatTriangle);

        // Normalize point to 4 components (x, y, z, padding)
        const normalizedPoint = point.length === 2 
            ? [point[0], point[1], 0, 0]
            : point.length === 3
            ? [point[0], point[1], point[2], 0]
            : (() => { throw new Error(`Invalid point dimension: ${point.length}`); })();

        const floatPoint = new Float32Array(normalizedPoint);

        const pointBuffer = device.createBuffer({
            label: `pointInTriangle - point buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: floatPoint.byteLength
        });

        device.queue.writeBuffer(pointBuffer, 0, floatPoint);

        const resultBooleanBuffer = device.createBuffer({
            label: `pointInTriangle - result boolean buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: 4
        });

        // Debug buffer: store a, b, c, p (each Point = 4 floats) + c1, c2, c3 (each f32) = 4*4 + 3 = 19 floats = 76 bytes
        const debugBufferSize = 76; // 19 floats * 4 bytes
        const debugBuffer = device.createBuffer({
            label: `pointInTriangle - debug buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: debugBufferSize
        });

        const pointInTriangleShader = device.createShaderModule({
            code: `/* wgsl */
                struct Point {
                    x: f32,
                    y: f32,
                    z: f32,
                    padding: f32
                }

                fn cross2d(v1: vec2<f32>, v2: vec2<f32>) -> f32 {
                    return v1.x * v2.y - v1.y * v2.x;
                }

                @group(0) @binding(0) var<storage, read> triangleBuffer: array<Point>;
                @group(0) @binding(1) var<storage, read> pointBuffer: Point;
                @group(0) @binding(2) var<storage, read_write> resultBooleanBuffer: array<u32>;
                @group(0) @binding(3) var<storage, read_write> debugBuffer: array<f32>;

                @compute @workgroup_size(32) fn pointInTriangle(
                    @builtin(global_invocation_id) id: vec3<u32>
                ) {
                    let i = id.x;
                    if (i > 0) {
                        // we only need this to run once, because we know there are 3 vertices in a triangle 
                        return;
                    }

                    var a: Point = triangleBuffer[0];
                    var b: Point = triangleBuffer[1];
                    var c: Point = triangleBuffer[2];
                    var p: Point = pointBuffer;

                    // Convert to vec2 for 2D cross product
                    var a2d = vec2<f32>(a.x, a.y);
                    var b2d = vec2<f32>(b.x, b.y);
                    var c2d = vec2<f32>(c.x, c.y);
                    var p2d = vec2<f32>(p.x, p.y);

                    var c1 = cross2d(b2d - a2d, p2d - a2d);
                    var c2 = cross2d(c2d - b2d, p2d - b2d);
                    var c3 = cross2d(a2d - c2d, p2d - c2d);

                    // Debug logging: store a, b, c, p, c1, c2, c3
                    // Layout: [a.x, a.y, a.z, a.padding, b.x, b.y, b.z, b.padding, c.x, c.y, c.z, c.padding, p.x, p.y, p.z, p.padding, c1, c2, c3]
                    // Offset: 0 (single invocation)
                    debugBuffer[0] = a.x;
                    debugBuffer[1] = a.y;
                    debugBuffer[2] = a.z;
                    debugBuffer[3] = 0.0; // padding
                    debugBuffer[4] = b.x;
                    debugBuffer[5] = b.y;
                    debugBuffer[6] = b.z;
                    debugBuffer[7] = 0.0; // padding
                    debugBuffer[8] = c.x;
                    debugBuffer[9] = c.y;
                    debugBuffer[10] = c.z;
                    debugBuffer[11] = 0.0; // padding
                    debugBuffer[12] = p.x;
                    debugBuffer[13] = p.y;
                    debugBuffer[14] = p.z;
                    debugBuffer[15] = 0.0; // padding
                    debugBuffer[16] = c1;
                    debugBuffer[17] = c2;
                    debugBuffer[18] = c3;

                    let epsilon = 1e-6;
                    if (c1 > epsilon && c2 > epsilon && c3 > epsilon) {
                        resultBooleanBuffer[0] = 1u;
                    } else {
                        resultBooleanBuffer[0] = 0u;
                    }
                    
                }
            `
        });

        const pointInTriangleBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage"}
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage"}
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage"}
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage"}
                }
            ]
        });
        const pointInTrianglePipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [
                pointInTriangleBindGroupLayout
            ]
        });

        const pointInTriangleBindGroup = device.createBindGroup({
            layout: pointInTriangleBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: triangleBuffer}
                },
                {
                    binding: 1,
                    resource: { buffer: pointBuffer}
                },
                {
                    binding: 2,
                    resource: { buffer: resultBooleanBuffer}
                },
                {
                    binding: 3,
                    resource: { buffer: debugBuffer}
                }
            ]
        });

        const pointInTrianglePipeline = device.createComputePipeline({
            layout: pointInTrianglePipelineLayout,
            compute: {
                module: pointInTriangleShader,
                entryPoint: "pointInTriangle"
            }
        });

        const workgroupSize = 32;
        const numWorkgroups = Math.ceil(normalizedTriangle.length / workgroupSize);

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pointInTrianglePipeline);
        passEncoder.setBindGroup(0, pointInTriangleBindGroup);
        passEncoder.dispatchWorkgroups(numWorkgroups);
        passEncoder.end();

        const readBuffer = device.createBuffer({
            label: `pointInTriangle - read buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            size: 4
        });
        commandEncoder.copyBufferToBuffer(resultBooleanBuffer, 0, readBuffer, 0, 4);
        
        // Create debug read buffer
        const debugReadBuffer = device.createBuffer({
            label: `pointInTriangle - debug read buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            size: debugBufferSize
        });
        commandEncoder.copyBufferToBuffer(debugBuffer, 0, debugReadBuffer, 0, debugBufferSize);
        
        device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const resultBooleanCopy = readBuffer.getMappedRange();
        const result = new Uint32Array(resultBooleanCopy.slice());
        readBuffer.unmap();

        // Read and log debug buffer
        await debugReadBuffer.mapAsync(GPUMapMode.READ);
        const debugMappedRange = debugReadBuffer.getMappedRange();
        const debugData = new Float32Array(debugMappedRange.slice());
        debugReadBuffer.unmap();

        // Log debug info
        console.log("=== DEBUG LOGGING ===");
        const a = { x: debugData[0], y: debugData[1], z: debugData[2] };
        const b = { x: debugData[4], y: debugData[5], z: debugData[6] };
        const c = { x: debugData[8], y: debugData[9], z: debugData[10] };
        const p = { x: debugData[12], y: debugData[13], z: debugData[14] };
        const c1 = debugData[16];
        const c2 = debugData[17];
        const c3 = debugData[18];
        console.log(`a=(${a.x}, ${a.y}, ${a.z})`);
        console.log(`b=(${b.x}, ${b.y}, ${b.z})`);
        console.log(`c=(${c.x}, ${c.y}, ${c.z})`);
        console.log(`p=(${p.x}, ${p.y}, ${p.z})`);
        console.log(`c1=${c1}`);
        console.log(`c2=${c2}`);
        console.log(`c3=${c3}`);
        console.log("=== END DEBUG LOGGING ===");

        return result;
    } catch (e) {
        throw e;
    }
}