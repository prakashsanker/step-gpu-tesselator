import { BYTE_SIZE } from "./gpu-tesselate";
import { getGPUDevice, normalizePoints } from "./lib";

const VERTEX_PER_TRIANGLE = 3;

export function createApplyShader(device: GPUDevice) {
    return device.createShaderModule({
        code: `
        /* wgsl */
        struct Point {
            x: f32,
            y: f32,
            z: f32,
            padding: f32
        }

        const WORKGROUP_SIZE = 32u;
        @group(0) @binding(0) var<storage, read> pointsBuffer: array<Point>;
        @group(0) @binding(1) var<storage, read> vertexIsEarBuffer: array<u32>;
        @group(0) @binding(2) var<storage, read_write> previousVertexBuffer: array<u32>;
        @group(0) @binding(3) var<storage, read_write> nextVertexBuffer: array<u32>;
        @group(0) @binding(4) var<storage, read_write> activeBuffer: array<u32>;
        @group(0) @binding(5) var<storage, read_write> outputIndicesBuffer: array<u32>;
        @group(0) @binding(6) var<storage, read_write> triangleCount: atomic<u32>;

        @compute @workgroup_size(WORKGROUP_SIZE) fn apply(
            @builtin(global_invocation_id) id: vec3<u32>
        ) {
            let i = id.x;
            let pointsBufferLength = arrayLength(&pointsBuffer);
            if (i >= pointsBufferLength) {
                return;
            }

            // Only the first thread finds and clips an ear
            if (i != 0u) {
                return;
            }

            // Find the lowest indexed ear
            var earIndex = 0xFFFFFFFFu; // Invalid index
            for (var j = 0u; j < pointsBufferLength; j++) {
                if (activeBuffer[j] == 1u && vertexIsEarBuffer[j] == 1u) {
                    earIndex = j;
                    break;
                }
            }

            // If no ear found, check if we have exactly 3 active vertices (final triangle)
            if (earIndex == 0xFFFFFFFFu) {
                // Count active vertices and collect their indices
                var activeCount = 0u;
                var idx0 = 0xFFFFFFFFu;
                var idx1 = 0xFFFFFFFFu;
                var idx2 = 0xFFFFFFFFu;
                
                for (var k = 0u; k < pointsBufferLength; k++) {
                    if (activeBuffer[k] == 1u) {
                        if (activeCount == 0u) {
                            idx0 = k;
                        } else if (activeCount == 1u) {
                            idx1 = k;
                        } else if (activeCount == 2u) {
                            idx2 = k;
                        }
                        activeCount = activeCount + 1u;
                    }
                }
                
                // If exactly 3 active vertices, add them as the final triangle
                if (activeCount == 3u) {
                    var triIdx = atomicLoad(&triangleCount);
                    outputIndicesBuffer[triIdx * 3u + 0u] = idx0;
                    outputIndicesBuffer[triIdx * 3u + 1u] = idx1;
                    outputIndicesBuffer[triIdx * 3u + 2u] = idx2;
                    atomicStore(&triangleCount, triIdx + 1u);
                }
                return;
            }

            // Get the triangle vertices
            var prevIdx = previousVertexBuffer[earIndex];
            var nextIdx = nextVertexBuffer[earIndex];

            // Find active neighbors
            while (activeBuffer[prevIdx] == 0u && prevIdx != earIndex) {
                prevIdx = previousVertexBuffer[prevIdx];
            }
            while (activeBuffer[nextIdx] == 0u && nextIdx != earIndex) {
                nextIdx = nextVertexBuffer[nextIdx];
            }

            // Write triangle to output buffer
            var triIdx = atomicLoad(&triangleCount);
            outputIndicesBuffer[triIdx * 3u + 0u] = prevIdx;
            outputIndicesBuffer[triIdx * 3u + 1u] = earIndex;
            outputIndicesBuffer[triIdx * 3u + 2u] = nextIdx;
            atomicStore(&triangleCount, triIdx + 1u);

            // Remove ear from polygon by updating prev/next pointers
            previousVertexBuffer[nextIdx] = prevIdx;
            nextVertexBuffer[prevIdx] = nextIdx;

            // Mark ear as inactive
            activeBuffer[earIndex] = 0u;
        }
        `
    });
}


export function createIsEarShader(device: GPUDevice) {
    const isEarShader = device.createShaderModule({
        code: `
            /* wgsl */
            struct Point {
                x: f32,
                y: f32,
                z: f32,
                padding: f32
            }

            const WORKGROUP_SIZE = 32u;
            const CONVEX = 1u;
            @group(0) @binding(0) var<storage, read> pointsBuffer: array<Point>;
            @group(0) @binding(1) var<storage, read_write> vertexIsEarBuffer: array<u32>;
            @group(0) @binding(2) var<storage, read> previousVertexBuffer: array<u32>;
            @group(0) @binding(3) var<storage, read> nextVertexBuffer: array<u32>;
            @group(0) @binding(4) var<storage, read> activeBuffer: array<u32>;
            @group(0) @binding(5) var<storage, read> classifiedPointsBuffer: array<u32>;

            @compute @workgroup_size(WORKGROUP_SIZE) fn isEar(
                @builtin(global_invocation_id) id: vec3<u32>
            ) {
                let i = id.x;
                let pointsBufferLength = arrayLength(&pointsBuffer);

                if (i >= pointsBufferLength) {
                    return;
                }

                // Skip inactive vertices
                if (activeBuffer[i] == 0u) {
                    return;
                }

                let vertexConvexity = classifiedPointsBuffer[i];
                if (vertexConvexity != CONVEX) {
                    vertexIsEarBuffer[i] = 0u;
                    return;
                }

                var currentVertexIndex = i;
                var previousVertexIndex = previousVertexBuffer[currentVertexIndex];
                var nextVertexIndex = nextVertexBuffer[currentVertexIndex];

                // Find the previous and next active vertices
                while (activeBuffer[previousVertexIndex] == 0u && previousVertexIndex != currentVertexIndex) {
                    previousVertexIndex = previousVertexBuffer[previousVertexIndex];
                }
                while (activeBuffer[nextVertexIndex] == 0u && nextVertexIndex != currentVertexIndex) {
                    nextVertexIndex = nextVertexBuffer[nextVertexIndex];
                }

                // If we only have 3 vertices left, it's definitely an ear
                if (previousVertexIndex == nextVertexIndex) {
                    vertexIsEarBuffer[i] = 1u;
                    return;
                }

                var a = pointsBuffer[previousVertexIndex];
                var b = pointsBuffer[currentVertexIndex];
                var c = pointsBuffer[nextVertexIndex];

                // Check all other active vertices to see if any are inside the triangle
                for (var j = 0u; j < pointsBufferLength; j++) {
                    if (j == currentVertexIndex || j == previousVertexIndex || j == nextVertexIndex) {
                        continue;
                    }
                    if (activeBuffer[j] == 0u) {
                        continue;
                    }
                    var p = pointsBuffer[j];

                    // Skip vertices that have the same coordinates as triangle vertices
                    // (can happen with bridge points in hole-merged polygons)
                    let sameAsA = abs(p.x - a.x) < 1e-9 && abs(p.y - a.y) < 1e-9;
                    let sameAsB = abs(p.x - b.x) < 1e-9 && abs(p.y - b.y) < 1e-9;
                    let sameAsC = abs(p.x - c.x) < 1e-9 && abs(p.y - c.y) < 1e-9;
                    if (sameAsA || sameAsB || sameAsC) {
                        continue;
                    }

                    var isInside = pointInTriangle(a, b, c, p);
                    if (isInside) {
                        vertexIsEarBuffer[i] = 0u;
                        return;
                    }
                }
                vertexIsEarBuffer[i] = 1u;
        }

        fn pointInTriangle(a: Point, b: Point, c: Point, p: Point) -> bool {
            var a2d = vec2<f32>(a.x, a.y);
            var b2d = vec2<f32>(b.x, b.y);
            var c2d = vec2<f32>(c.x, c.y);
            var p2d = vec2<f32>(p.x, p.y);

            var v1_c1 = b2d - a2d;
            var v2_c1 = p2d - a2d;
            var term1_c1 = v1_c1.x * v2_c1.y;
            var term2_c1 = v1_c1.y * v2_c1.x;
            var c1 = term1_c1 - term2_c1;

            var v1_c2 = c2d - b2d;
            var v2_c2 = p2d - b2d;
            var term1_c2 = v1_c2.x * v2_c2.y;
            var term2_c2 = v1_c2.y * v2_c2.x;
            var c2 = term1_c2 - term2_c2;
            
        
            var v1_c3 = a2d - c2d;
            var v2_c3 = p2d - c2d;
            var term1_c3 = v1_c3.x * v2_c3.y;
            var term2_c3 = v1_c3.y * v2_c3.x;
            var c3 = term1_c3 - term2_c3;


            let epsilon = 1e-12;
            // Point is inside if all cross products are >= 0 (allowing for floating point errors)
            // This includes points on vertices (where one cross product = 0) and edges
            if (c1 >= -epsilon && c2 >= -epsilon && c3 >= -epsilon) {
                return true;
            } else {
                return false;
            }
        }
        `
    });
    return isEarShader;
}

export function createClassifyPointsShader(device: GPUDevice) {
    const convexityCheckShader = device.createShaderModule({
        code: `
            /* wgsl */
            struct Point {
                x: f32,
                y: f32,
                z: f32,
                padding: f32
            }
            const REFLEX = 0u;
            const CONVEX = 1u;
            const COLLINEAR = 2u;

            const WORKGROUP_SIZE = 32u;
            @group(0) @binding(0) var<storage, read> pointsBuffer: array<Point>;
            @group(0) @binding(1) var<storage, read_write> classifiedPointsBuffer: array<u32>;
            @group(0) @binding(2) var<storage, read> previousVertexBuffer: array<u32>;
            @group(0) @binding(3) var<storage, read> nextVertexBuffer: array<u32>;
            @group(0) @binding(4) var<storage, read> activeBuffer: array<u32>;

            @compute @workgroup_size(WORKGROUP_SIZE) fn classifyPoints(
                @builtin(global_invocation_id) id: vec3<u32>
            ) {
                let i = id.x;
                let pointsBufferLength = arrayLength(&pointsBuffer);
                if (i >= pointsBufferLength) {
                    return;
                }

                // Skip inactive vertices
                if (activeBuffer[i] == 0u) {
                    return;
                }

                var prevIdx = previousVertexBuffer[i];
                var nextIdx = nextVertexBuffer[i];

                // Find the previous and next active vertices
                while (activeBuffer[prevIdx] == 0u && prevIdx != i) {
                    prevIdx = previousVertexBuffer[prevIdx];
                }
                while (activeBuffer[nextIdx] == 0u && nextIdx != i) {
                    nextIdx = nextVertexBuffer[nextIdx];
                }

                var A = pointsBuffer[prevIdx];
                var B = pointsBuffer[i];
                var C = pointsBuffer[nextIdx];

                var E1 = vec2<f32>(B.x - A.x, B.y - A.y);
                var E2 = vec2<f32>(C.x - B.x, C.y - B.y);
                var crossProduct = E1.x * E2.y - E1.y * E2.x;
                
                // Use a smaller epsilon to handle floating point precision
                // For 2D polygons, we want to be more lenient with near-zero values
                let epsilon = 1e-5;
                if (crossProduct > epsilon) {
                    classifiedPointsBuffer[i] = CONVEX;
                } else if (crossProduct < -epsilon) {
                    classifiedPointsBuffer[i] = REFLEX;
                } else {
                    // For very small cross products, treat as convex if positive, reflex if negative
                    // This handles cases where points are nearly collinear but should still be classified
                    if (crossProduct >= 0.0) {
                        classifiedPointsBuffer[i] = CONVEX;
                    } else {
                        classifiedPointsBuffer[i] = REFLEX;
                    }
                }
            }
        `
    });
    return convexityCheckShader;
}

// The points are assumed to be in CCW.
export async function earClipping(points: number[][]) {
    try {
        const device = await getGPUDevice();
        
        console.log("[Tessellator] Input points:", points);
        console.log("[Tessellator] Number of vertices:", points.length);
        
        if (points.length < 3) {
            throw new Error("Polygon must have at least 3 vertices");
        }

        // Initialize all buffers
        const buffers = initializeBuffers(device, points);
        const {
            pointsBuffer,
            outputIndicesBuffer,
            vertexIsEarBuffer,
            previousVertexBuffer,
            nextVertexBuffer,
            activeBuffer,
            triangleCount,
            classifiedPointsBuffer
        } = buffers;
        
        // Log normalized points to verify they're correct
        const normalizedPoints = normalizePoints(points);
        console.log("[Tessellator] Normalized points:", normalizedPoints);

        // Create shaders
        const classifyShader = createClassifyPointsShader(device);
        const isEarShader = createIsEarShader(device);
        const applyShader = createApplyShader(device);

        // Create bind group layouts
        const classifyBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ]
        });

        const isEarBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ]
        });

        const applyBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });

        // Create compute pipelines
        const classifyPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [classifyBGL] }),
            compute: { module: classifyShader, entryPoint: "classifyPoints" }
        });

        const isEarPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [isEarBGL] }),
            compute: { module: isEarShader, entryPoint: "isEar" }
        });

        const applyPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [applyBGL] }),
            compute: { module: applyShader, entryPoint: "apply" }
        });

        // Create bind groups
        const classifyBG = device.createBindGroup({
            layout: classifyBGL,
            entries: [
                { binding: 0, resource: { buffer: pointsBuffer } },
                { binding: 1, resource: { buffer: classifiedPointsBuffer } },
                { binding: 2, resource: { buffer: previousVertexBuffer } },
                { binding: 3, resource: { buffer: nextVertexBuffer } },
                { binding: 4, resource: { buffer: activeBuffer } },
            ]
        });

        const isEarBG = device.createBindGroup({
            layout: isEarBGL,
            entries: [
                { binding: 0, resource: { buffer: pointsBuffer } },
                { binding: 1, resource: { buffer: vertexIsEarBuffer } },
                { binding: 2, resource: { buffer: previousVertexBuffer } },
                { binding: 3, resource: { buffer: nextVertexBuffer } },
                { binding: 4, resource: { buffer: activeBuffer } },
                { binding: 5, resource: { buffer: classifiedPointsBuffer } },
            ]
        });

        const applyBG = device.createBindGroup({
            layout: applyBGL,
            entries: [
                { binding: 0, resource: { buffer: pointsBuffer } },
                { binding: 1, resource: { buffer: vertexIsEarBuffer } },
                { binding: 2, resource: { buffer: previousVertexBuffer } },
                { binding: 3, resource: { buffer: nextVertexBuffer } },
                { binding: 4, resource: { buffer: activeBuffer } },
                { binding: 5, resource: { buffer: outputIndicesBuffer } },
                { binding: 6, resource: { buffer: triangleCount } },
            ]
        });

        // Main algorithm loop
        const maxIterations = points.length - 2; // Maximum number of triangles
        let commandEncoder = device.createCommandEncoder();

        console.log("[Tessellator] Starting algorithm loop, max iterations:", maxIterations);

        for (let iteration = 0; iteration < maxIterations; iteration++) {
            console.log(`[Tessellator] Iteration ${iteration + 1}/${maxIterations}`);
            
            // Reset vertexIsEarBuffer at the start of each iteration
            const resetEarArray = new Uint32Array(points.length);
            resetEarArray.fill(0);
            device.queue.writeBuffer(vertexIsEarBuffer, 0, resetEarArray);
            
            // Classify points (convex/reflex)
            const classifyPass = commandEncoder.beginComputePass();
            classifyPass.setPipeline(classifyPipeline);
            classifyPass.setBindGroup(0, classifyBG);
            classifyPass.dispatchWorkgroups(Math.ceil(points.length / 32));
            classifyPass.end();

            // Check which vertices are ears
            const isEarPass = commandEncoder.beginComputePass();
            isEarPass.setPipeline(isEarPipeline);
            isEarPass.setBindGroup(0, isEarBG);
            isEarPass.dispatchWorkgroups(Math.ceil(points.length / 32));
            isEarPass.end();

            // Apply ear clipping (remove one ear)
            const applyPass = commandEncoder.beginComputePass();
            applyPass.setPipeline(applyPipeline);
            applyPass.setBindGroup(0, applyBG);
            applyPass.dispatchWorkgroups(1); // Only needs to run once
            applyPass.end();

            // Submit commands
            device.queue.submit([commandEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            
            // Read back buffers for debugging (after commands complete)
            if (iteration === 0) {
                // Read back vertexIsEarBuffer to debug
                const earReadbackBuffer = device.createBuffer({
                    size: BYTE_SIZE * points.length,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                const earReadEncoder = device.createCommandEncoder();
                earReadEncoder.copyBufferToBuffer(vertexIsEarBuffer, 0, earReadbackBuffer, 0, BYTE_SIZE * points.length);
                device.queue.submit([earReadEncoder.finish()]);
                await earReadbackBuffer.mapAsync(GPUMapMode.READ);
                const earMapped = earReadbackBuffer.getMappedRange();
                const earArray = new Uint32Array(earMapped);
                const earArrayCopy = new Uint32Array(earArray);
                earReadbackBuffer.unmap();
                console.log(`[Tessellator] Iteration ${iteration + 1} - vertexIsEarBuffer:`, Array.from(earArrayCopy));
                
                // Read back classifiedPointsBuffer
                const classifyReadbackBuffer = device.createBuffer({
                    size: BYTE_SIZE * points.length,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                const classifyReadEncoder = device.createCommandEncoder();
                classifyReadEncoder.copyBufferToBuffer(classifiedPointsBuffer, 0, classifyReadbackBuffer, 0, BYTE_SIZE * points.length);
                device.queue.submit([classifyReadEncoder.finish()]);
                await classifyReadbackBuffer.mapAsync(GPUMapMode.READ);
                const classifyMapped = classifyReadbackBuffer.getMappedRange();
                const classifyArray = new Uint32Array(classifyMapped);
                const classifyArrayCopy = new Uint32Array(classifyArray);
                classifyReadbackBuffer.unmap();
                console.log(`[Tessellator] Iteration ${iteration + 1} - classifiedPointsBuffer:`, Array.from(classifyArrayCopy));
                
                // Read back triangleCount
                const triCountReadback = device.createBuffer({
                    size: BYTE_SIZE,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                const triCountEncoder = device.createCommandEncoder();
                triCountEncoder.copyBufferToBuffer(triangleCount, 0, triCountReadback, 0, BYTE_SIZE);
                device.queue.submit([triCountEncoder.finish()]);
                await triCountReadback.mapAsync(GPUMapMode.READ);
                const triCountMapped = triCountReadback.getMappedRange();
                const triCountValue = new Uint32Array(triCountMapped)[0];
                triCountReadback.unmap();
                console.log(`[Tessellator] Iteration ${iteration + 1} - triangleCount:`, triCountValue);
            }

            // Check if we're done (only 3 vertices left)
            // Read back active count to check
            const readbackBuffer = device.createBuffer({
                size: BYTE_SIZE * points.length,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            const readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(activeBuffer, 0, readbackBuffer, 0, BYTE_SIZE * points.length);
            device.queue.submit([readEncoder.finish()]);
            await readbackBuffer.mapAsync(GPUMapMode.READ);
            const mappedRange = readbackBuffer.getMappedRange();
            const activeArray = new Uint32Array(mappedRange);
            // Copy the data before unmapping
            const activeArrayCopy = new Uint32Array(activeArray);
            readbackBuffer.unmap();

            const activeCount = Array.from(activeArrayCopy).filter(x => x === 1).length;
            console.log(`[Tessellator] After iteration ${iteration + 1}, active vertices: ${activeCount}`);
            
            if (activeCount <= 3) {
                console.log("[Tessellator] Algorithm complete, only 3 or fewer vertices remaining");
                // If we have exactly 3 vertices, they form the final triangle - add it
                if (activeCount === 3) {
                    // Find the 3 active vertices
                    const activeIndices: number[] = [];
                    for (let i = 0; i < points.length; i++) {
                        if (activeArrayCopy[i] === 1) {
                            activeIndices.push(i);
                        }
                    }
                    console.log("[Tessellator] Adding final triangle with vertices:", activeIndices);
                    
                    // Read current triangle count
                    const finalTriCountReadback = device.createBuffer({
                        size: BYTE_SIZE,
                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                    });
                    const finalTriCountEncoder = device.createCommandEncoder();
                    finalTriCountEncoder.copyBufferToBuffer(triangleCount, 0, finalTriCountReadback, 0, BYTE_SIZE);
                    device.queue.submit([finalTriCountEncoder.finish()]);
                    await finalTriCountReadback.mapAsync(GPUMapMode.READ);
                    const finalTriCountMapped = finalTriCountReadback.getMappedRange();
                    const finalTriIdx = new Uint32Array(finalTriCountMapped)[0];
                    finalTriCountReadback.unmap();
                    
                    // Create a buffer with the final triangle indices
                    const finalTriIndices = new Uint32Array([activeIndices[0], activeIndices[1], activeIndices[2]]);
                    const finalTriBuf = device.createBuffer({
                        size: BYTE_SIZE * 3,
                        usage: GPUBufferUsage.COPY_SRC,
                        mappedAtCreation: true
                    });
                    new Uint32Array(finalTriBuf.getMappedRange()).set(finalTriIndices);
                    finalTriBuf.unmap();
                    
                    // Write final triangle to output buffer and update count
                    const finalTriWriteEncoder = device.createCommandEncoder();
                    finalTriWriteEncoder.copyBufferToBuffer(
                        finalTriBuf,
                        0,
                        outputIndicesBuffer,
                        finalTriIdx * 3 * BYTE_SIZE,
                        BYTE_SIZE * 3
                    );
                    
                    // Update triangle count
                    const finalCountUpdate = new Uint32Array([finalTriIdx + 1]);
                    const finalCountBuf = device.createBuffer({
                        size: BYTE_SIZE,
                        usage: GPUBufferUsage.COPY_SRC,
                        mappedAtCreation: true
                    });
                    new Uint32Array(finalCountBuf.getMappedRange()).set(finalCountUpdate);
                    finalCountBuf.unmap();
                    finalTriWriteEncoder.copyBufferToBuffer(
                        finalCountBuf,
                        0,
                        triangleCount,
                        0,
                        BYTE_SIZE
                    );
                    device.queue.submit([finalTriWriteEncoder.finish()]);
                    await device.queue.onSubmittedWorkDone();
                }
                break;
            }

            // Create new command encoder for next iteration
            commandEncoder = device.createCommandEncoder();
        }

        // Read back triangle indices
        const readbackIndices = device.createBuffer({
            size: outputIndicesBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const finalEncoder = device.createCommandEncoder();
        finalEncoder.copyBufferToBuffer(outputIndicesBuffer, 0, readbackIndices, 0, outputIndicesBuffer.size);
        device.queue.submit([finalEncoder.finish()]);
        await readbackIndices.mapAsync(GPUMapMode.READ);
        
        const triangleCountReadback = device.createBuffer({
            size: BYTE_SIZE,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const countEncoder = device.createCommandEncoder();
        countEncoder.copyBufferToBuffer(triangleCount, 0, triangleCountReadback, 0, BYTE_SIZE);
        device.queue.submit([countEncoder.finish()]);
        await triangleCountReadback.mapAsync(GPUMapMode.READ);
        
        // Copy data before unmapping
        const triangleCountMapped = triangleCountReadback.getMappedRange();
        const numTriangles = new Uint32Array(triangleCountMapped)[0];
        triangleCountReadback.unmap();

        const indicesMapped = readbackIndices.getMappedRange();
        const indicesArray = new Uint32Array(indicesMapped);
        // Copy the data before unmapping
        const indicesArrayCopy = new Uint32Array(indicesArray);
        readbackIndices.unmap();

        // Convert to array of triangles
        const triangles: number[][] = [];
        for (let i = 0; i < numTriangles; i++) {
            triangles.push([
                indicesArrayCopy[i * 3],
                indicesArrayCopy[i * 3 + 1],
                indicesArrayCopy[i * 3 + 2]
            ]);
        }

        console.log("[Tessellator] Number of triangles generated:", numTriangles);
        console.log("[Tessellator] Triangles (indices):", triangles);
        console.log("[Tessellator] Full indices array:", Array.from(indicesArrayCopy));
        
        // Log triangle vertices with actual coordinates
        console.log("[Tessellator] Triangles with coordinates:");
        for (let i = 0; i < triangles.length; i++) {
            const tri = triangles[i];
            console.log(`  Triangle ${i}:`, {
                indices: tri,
                vertices: [
                    points[tri[0]],
                    points[tri[1]],
                    points[tri[2]]
                ]
            });
        }

        return triangles;

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
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: VERTEX_PER_TRIANGLE*(points.length - 2)* BYTE_SIZE// this should be 3  * the number of triangles, so 
        });

        const vertexIsEarBuffer = device.createBuffer({
            label: `EarClipping - vertexIsEarBuffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE * points.length
        });
        
        // Initialize vertexIsEarBuffer to all zeros
        const initialEarArray = new Uint32Array(points.length);
        initialEarArray.fill(0);
        device.queue.writeBuffer(vertexIsEarBuffer, 0, initialEarArray);

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

        const activeBuffer = device.createBuffer({
            label: `EarClipping - activeBuffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE * points.length
        });

        // Initialize all vertices as active (1)
        const activeArray = new Uint32Array(points.length);
        activeArray.fill(1);
        device.queue.writeBuffer(activeBuffer, 0, activeArray);

        const triangleCount = device.createBuffer({
            label: `EarClipping - triangleCount`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE
        });

        device.queue.writeBuffer(triangleCount, 0, new Uint32Array([0])); // initialize to 0

        const classifiedPointsBuffer = device.createBuffer({
            label: `EarClipping - classifiedPointsBuffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: BYTE_SIZE * points.length
        });
        
        // Initialize classifiedPointsBuffer to all zeros
        const initialClassifyArray = new Uint32Array(points.length);
        initialClassifyArray.fill(0);
        device.queue.writeBuffer(classifiedPointsBuffer, 0, initialClassifyArray);

        return {
            pointsBuffer, 
            outputIndicesBuffer,
            vertexIsEarBuffer,
            previousVertexBuffer,
            nextVertexBuffer,
            activeBuffer,
            triangleCount,
            classifiedPointsBuffer
        };
    } catch (e) {
        throw e;
    }
}