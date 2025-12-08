import { getGPUDevice } from "./lib"; 

export const BYTE_SIZE = 4;


export async function gpuTesselate(uniquePoints) {
    console.log("DO WE MAKE IT HERE?");
    try {
        const device = await getGPUDevice();
        // now we want to flatten the uniquePoints

        const flattenedUniquePoints  = uniquePoints.flat();
        const flattenedUnsignedIntegerUniquePoints = new Uint32Array(flattenedUniquePoints.length);
        flattenedUniquePoints.forEach((point, index) => {
            flattenedUnsignedIntegerUniquePoints[index] = flattenedUniquePoints[index];
        });
        // now let's create the input buffer

        const vertexBuffer = device.createBuffer({
            label: `vertexes`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: flattenedUnsignedIntegerUniquePoints.byteLength
        });

        device.queue.writeBuffer(vertexBuffer, 0, flattenedUnsignedIntegerUniquePoints, 0, flattenedUnsignedIntegerUniquePoints.length);

        // now let's create the output buffer

        const triangleCount = uniquePoints.length - 2;
        const indexCount = triangleCount * 3; //  for each triangle, there are three vertexes, and for each vertex, there are three points?
        const indicesBuffer = device.createBuffer({
            label: `indices`, 
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
            size: indexCount * 4
        });

        // this is 1 4 byte, with 3 bytes of padding 
        const uniformStorageSize = 16;
        const uint32array = new Uint32Array([triangleCount]);




        const uniformStorage = device.createBuffer({
            label: `triangle count storage`,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            size: uniformStorageSize
        });

        // ok now I need to use these buffers

        device.queue.writeBuffer(uniformStorage, 0, uint32array);

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: "storage"}
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform"},
                }
            ],
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [
              bindGroupLayout, // @group(0)
            ]
          });

        // ok I think I have the stuff laid out, I now need to actually do the triangulation

        const computeShader = device.createShaderModule({
            code: `
                struct Uniforms { 
                    triangleCount: u32
                }

                @group(0) @binding(0) var<storage, read> vertexBuffer: array<u32>;
                @group(0) @binding(1) var<storage, read_write> indicesBuffer: array<u32>;
                @group(0) @binding(2) var<uniform> uniforms: Uniforms;

                @compute @workgroup_size(64) fn create_indices(
                    @builtin(global_invocation_id) id: vec3<u32>
                ) {
                
                    if (id.x >= uniforms.triangleCount) {
                        return;
                    }

                    let i  = id.x + 1u; // 1, 2, 3
                    let base = id.x *3u; // 0, 3, 6

                    indicesBuffer[base + 0u] = 0u;
                    indicesBuffer[base + 1u] = i;
                    indicesBuffer[base + 2u] = i + 1;
                }
            `
        })


        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0, 
                resource: { buffer: vertexBuffer}
            }, {
                binding: 1,
                resource: {buffer: indicesBuffer}
            }, {
                binding: 2,
                resource: { buffer: uniformStorage}
            }]
        });

        const computePipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: computeShader,
                entryPoint: "create_indices"
            }
        });

        const readBuffer = device.createBuffer({
            size: indexCount*4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });



        const commandEncoder = device.createCommandEncoder();

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(64);
        passEncoder.end();



        commandEncoder.copyBufferToBuffer(
            indicesBuffer,
            0,
            readBuffer,
            0,
            indexCount * 4
        );

        device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const indicesCopy  = readBuffer.getMappedRange();
        const result = new Uint32Array(indicesCopy.slice());
        readBuffer.unmap();

        console.log("RESULT");
        console.log(result);

        return result;

    } catch (e) {
        throw e;
    }
    /*
        Steps

            1. We first need to initialize the GPU device
            2. We then need to load everything into memory. Specifically
                a. Iniitally I have 
                    [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
                b. We want this to become
                    [x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3]
            3. We then pass this into a compute shader. The compute shader should create an indices output buffer, which is basically vertex points fan triangulated
                a. My confusion is basically, what does the indices have to look like? 
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 9, 10, 11, 6, 7, 8]
                    I think it's this.
                b. This is what the compute shader needs to output. 


            4. We also need to tell the compute shader threads how many to run
                a. So we need an uniform buffer that can store the triangle count --> this triangle count is N - 2 = 2 (this is a general rule for convex polygons)


            5. The indices output buffer then needs to be copied to another buffer and the data has to be extracted.
            6. Then the {positions, indices} object is extracted and returned. 
    */


}