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

        // Debug buffer: store a, b, c, p (each Point = 4 floats) + c1, c2, c3 (each f32) + 
        // For each cross product: v1 (2 floats), v2 (2 floats), v1.x*v2.y, v1.y*v2.x (2 floats)
        // Total: 16 + 3 + (6*3) = 16 + 3 + 18 = 37 floats = 148 bytes
        const debugBufferSize = 148; // 37 floats * 4 bytes
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

                    // Calculate c1: cross(b - a, p - a)
                    var v1_c1 = b2d - a2d;
                    var v2_c1 = p2d - a2d;
                    var term1_c1 = v1_c1.x * v2_c1.y;
                    var term2_c1 = v1_c1.y * v2_c1.x;
                    var c1 = term1_c1 - term2_c1;

                    // Calculate c2: cross(c - b, p - b)
                    var v1_c2 = c2d - b2d;
                    var v2_c2 = p2d - b2d;
                    var term1_c2 = v1_c2.x * v2_c2.y;
                    var term2_c2 = v1_c2.y * v2_c2.x;
                    var c2 = term1_c2 - term2_c2;

                    // Calculate c3: cross(a - c, p - c)
                    var v1_c3 = a2d - c2d;
                    var v2_c3 = p2d - c2d;
                    var term1_c3 = v1_c3.x * v2_c3.y;
                    var term2_c3 = v1_c3.y * v2_c3.x;
                    var c3 = term1_c3 - term2_c3;

                    // Debug logging: store a, b, c, p, c1, c2, c3, and intermediate values
                    // Layout: [a (4), b (4), c (4), p (4), c1, c2, c3,
                    //          v1_c1 (2), v2_c1 (2), term1_c1, term2_c1,
                    //          v1_c2 (2), v2_c2 (2), term1_c2, term2_c2,
                    //          v1_c3 (2), v2_c3 (2), term1_c3, term2_c3]
                    // Total: 16 + 3 + 6 + 6 + 6 = 37 floats
                    var offset = 0u;
                    // Points
                    debugBuffer[offset] = a.x; offset += 1u;
                    debugBuffer[offset] = a.y; offset += 1u;
                    debugBuffer[offset] = a.z; offset += 1u;
                    debugBuffer[offset] = 0.0; offset += 1u; // padding
                    debugBuffer[offset] = b.x; offset += 1u;
                    debugBuffer[offset] = b.y; offset += 1u;
                    debugBuffer[offset] = b.z; offset += 1u;
                    debugBuffer[offset] = 0.0; offset += 1u; // padding
                    debugBuffer[offset] = c.x; offset += 1u;
                    debugBuffer[offset] = c.y; offset += 1u;
                    debugBuffer[offset] = c.z; offset += 1u;
                    debugBuffer[offset] = 0.0; offset += 1u; // padding
                    debugBuffer[offset] = p.x; offset += 1u;
                    debugBuffer[offset] = p.y; offset += 1u;
                    debugBuffer[offset] = p.z; offset += 1u;
                    debugBuffer[offset] = 0.0; offset += 1u; // padding
                    // Cross products
                    debugBuffer[offset] = c1; offset += 1u;
                    debugBuffer[offset] = c2; offset += 1u;
                    debugBuffer[offset] = c3; offset += 1u;
                    // c1 intermediates
                    debugBuffer[offset] = v1_c1.x; offset += 1u;
                    debugBuffer[offset] = v1_c1.y; offset += 1u;
                    debugBuffer[offset] = v2_c1.x; offset += 1u;
                    debugBuffer[offset] = v2_c1.y; offset += 1u;
                    debugBuffer[offset] = term1_c1; offset += 1u;
                    debugBuffer[offset] = term2_c1; offset += 1u;
                    // c2 intermediates
                    debugBuffer[offset] = v1_c2.x; offset += 1u;
                    debugBuffer[offset] = v1_c2.y; offset += 1u;
                    debugBuffer[offset] = v2_c2.x; offset += 1u;
                    debugBuffer[offset] = v2_c2.y; offset += 1u;
                    debugBuffer[offset] = term1_c2; offset += 1u;
                    debugBuffer[offset] = term2_c2; offset += 1u;
                    // c3 intermediates
                    debugBuffer[offset] = v1_c3.x; offset += 1u;
                    debugBuffer[offset] = v1_c3.y; offset += 1u;
                    debugBuffer[offset] = v2_c3.x; offset += 1u;
                    debugBuffer[offset] = v2_c3.y; offset += 1u;
                    debugBuffer[offset] = term1_c3; offset += 1u;
                    debugBuffer[offset] = term2_c3; offset += 1u;

                    let epsilon = 1e-12;
                    // Point is inside if all cross products are >= 0 (allowing for floating point errors)
                    // This includes points on vertices (where one cross product = 0) and edges
                    if (c1 >= -epsilon && c2 >= -epsilon && c3 >= -epsilon) {
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
        let idx = 0;
        const a = { x: debugData[idx++], y: debugData[idx++], z: debugData[idx++], padding: debugData[idx++] };
        const b = { x: debugData[idx++], y: debugData[idx++], z: debugData[idx++], padding: debugData[idx++] };
        const c = { x: debugData[idx++], y: debugData[idx++], z: debugData[idx++], padding: debugData[idx++] };
        const p = { x: debugData[idx++], y: debugData[idx++], z: debugData[idx++], padding: debugData[idx++] };
        const c1 = debugData[idx++];
        const c2 = debugData[idx++];
        const c3 = debugData[idx++];
        
        console.log(`a=(${a.x}, ${a.y}, ${a.z})`);
        console.log(`b=(${b.x}, ${b.y}, ${b.z})`);
        console.log(`c=(${c.x}, ${c.y}, ${c.z})`);
        console.log(`p=(${p.x}, ${p.y}, ${p.z})`);
        console.log(`c1=${c1}`);
        console.log(`c2=${c2}`);
        console.log(`c3=${c3}`);
        
        // c1 intermediates: cross(b - a, p - a)
        const v1_c1 = { x: debugData[idx++], y: debugData[idx++] };
        const v2_c1 = { x: debugData[idx++], y: debugData[idx++] };
        const term1_c1 = debugData[idx++];
        const term2_c1 = debugData[idx++];
        console.log(`c1 calculation: cross(b - a, p - a)`);
        console.log(`  v1_c1 = b - a = (${v1_c1.x}, ${v1_c1.y})`);
        console.log(`  v2_c1 = p - a = (${v2_c1.x}, ${v2_c1.y})`);
        console.log(`  term1_c1 = v1_c1.x * v2_c1.y = ${v1_c1.x} * ${v2_c1.y} = ${term1_c1}`);
        console.log(`  term2_c1 = v1_c1.y * v2_c1.x = ${v1_c1.y} * ${v2_c1.x} = ${term2_c1}`);
        console.log(`  c1 = term1_c1 - term2_c1 = ${term1_c1} - ${term2_c1} = ${c1}`);
        
        // c2 intermediates: cross(c - b, p - b)
        const v1_c2 = { x: debugData[idx++], y: debugData[idx++] };
        const v2_c2 = { x: debugData[idx++], y: debugData[idx++] };
        const term1_c2 = debugData[idx++];
        const term2_c2 = debugData[idx++];
        console.log(`c2 calculation: cross(c - b, p - b)`);
        console.log(`  v1_c2 = c - b = (${v1_c2.x}, ${v1_c2.y})`);
        console.log(`  v2_c2 = p - b = (${v2_c2.x}, ${v2_c2.y})`);
        console.log(`  term1_c2 = v1_c2.x * v2_c2.y = ${v1_c2.x} * ${v2_c2.y} = ${term1_c2}`);
        console.log(`  term2_c2 = v1_c2.y * v2_c2.x = ${v1_c2.y} * ${v2_c2.x} = ${term2_c2}`);
        console.log(`  c2 = term1_c2 - term2_c2 = ${term1_c2} - ${term2_c2} = ${c2}`);
        
        // c3 intermediates: cross(a - c, p - c)
        const v1_c3 = { x: debugData[idx++], y: debugData[idx++] };
        const v2_c3 = { x: debugData[idx++], y: debugData[idx++] };
        const term1_c3 = debugData[idx++];
        const term2_c3 = debugData[idx++];
        console.log(`c3 calculation: cross(a - c, p - c)`);
        console.log(`  v1_c3 = a - c = (${v1_c3.x}, ${v1_c3.y})`);
        console.log(`  v2_c3 = p - c = (${v2_c3.x}, ${v2_c3.y})`);
        console.log(`  term1_c3 = v1_c3.x * v2_c3.y = ${v1_c3.x} * ${v2_c3.y} = ${term1_c3}`);
        console.log(`  term2_c3 = v1_c3.y * v2_c3.x = ${v1_c3.y} * ${v2_c3.x} = ${term2_c3}`);
        console.log(`  c3 = term1_c3 - term2_c3 = ${term1_c3} - ${term2_c3} = ${c3}`);
        
        console.log("=== END DEBUG LOGGING ===");

        return result;
    } catch (e) {
        throw e;
    }
}