import type { Mesh } from "./step-parser";
import {getGPUDevice} from "./lib";

export async function render(mesh: Mesh, canvas: HTMLCanvasElement) {
    try {

        const device = await getGPUDevice();

        const indices = mesh.indices;
        const positions = mesh.positions;
        const indexCount = indices.length;

        // first thing is to load up the positions and indices and counts

        const indicesBuffer = device.createBuffer({
            label: `indices buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: indices.byteLength
        });

        device.queue.writeBuffer(indicesBuffer, 0, indices, 0, indices.length);

        const positionsBuffer = device.createBuffer({
            label: `positions buffer`,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
            size: positions.byteLength 
        });

        device.queue.writeBuffer(positionsBuffer, 0, positions);
        const uniformStorageSize = 16; // magic number for now
        const indexCountArr = new Uint32Array([indexCount]);
        console.log("BYTE LENGTH");
        console.log(indexCountArr.byteLength);

        const uniformStorage = device.createBuffer({
            label: `uniform storage - index count`,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            size: uniformStorageSize
        });

        device.queue.writeBuffer(uniformStorage, 0, indexCountArr);

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage"}
                }, 
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage"}
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
                    buffer: {type: "uniform"}
                }
            ]
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [
                bindGroupLayout
            ]
        });

        const shaderModule = device.createShaderModule({
            code: `            /* wgsl */

            struct Uniforms {
                indexCount: u32
            }

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) color: vec4f,
          };
            
            @group(0) @binding(0) var<storage, read> indicesBuffer: array<u32>;
            @group(0) @binding(1) var<storage, read> positionsBuffer: array<u32>;
            @group(0) @binding(2) var<uniform> uniforms: Uniforms;

            @vertex fn draw(
                @builtin(vertex_index) vertexIndex: u32
            ) -> VertexOutput {
                // I have it wrong
                // [x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3]
                // [ 0, 1, 2, 0, 2, 3]
                // I'm going through each index? 
                let index = indicesBuffer[vertexIndex]; // 2
                let x0 = positionsBuffer[3u* index]; // 6
                let y0 = positionsBuffer[3u*index + 1u]; // 7
                let x0_f32 = f32(x0);
                let y0_f32 = f32(y0);


                let ndc_x = (x0_f32 * 2.0) - 1.0;
                // For Y, WebGPU typically uses Y-up coordinates if you want
                // standard mathematical orientation, but simple flip might be needed:
                let ndc_y = (y0_f32 * 2.0) - 1.0; 
                // --------------------------------

                var vsOutput: VertexOutput;
                vsOutput.position = vec4f(ndc_x, ndc_y, 0.0, 1.0);
                vsOutput.color = vec4f(1,0,0,1);
                return vsOutput;
            }

            @fragment fn fs(fsInput: VertexOutput) -> @location(0) vec4f {
                return vec4f(1,0,0,1);
            }
            `
        });

        console.log("INDICES BUFFER", indicesBuffer);
        console.log("POSITIONS BUFFER", positionsBuffer);
        console.log("UNIFORM STORAGE", uniformStorage);
        console.log("PIPELINE LAYOUT", pipelineLayout);

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: indicesBuffer}
            }, {
                binding: 1, 
                resource: { buffer: positionsBuffer}
            }, {
                binding: 2,
                resource: { buffer: uniformStorage}
            }]
        });

        const context = canvas.getContext('webgpu') as GPUCanvasContext;
        if (!context) {
            throw new Error("Failed to get WebGPU context from canvas");
        }
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        context.configure({
            device,
            format: presentationFormat
        });

        const currentTextureView = context.getCurrentTexture().createView();

        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: currentTextureView,
                    // Defines what to do at the start of the pass (clear the screen)
                    loadOp: 'clear', 
                    clearValue: { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }, 
                    // Defines what to do at the end of the pass (save the results)
                    storeOp: 'store',
                },
            ],
            // If using a depth buffer, you would add depthStencilAttachment here
        };

        const renderPipeline = device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: "draw"
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fs",
                targets: [{
                    format: presentationFormat
                }]
            },
            primitive: {
                topology: "triangle-list", // interpret indices as independent triangles
                cullMode: "back",
              },
        });

        const commandEncoder = device.createCommandEncoder();

        const pass = commandEncoder.beginRenderPass(renderPassDescriptor);
        pass.setPipeline(renderPipeline);
        pass.setBindGroup(0, bindGroup);
        console.log("IDNEX COUNT", indexCount);
        pass.draw(indexCount);
        pass.end();
        device.queue.submit([commandEncoder.finish()]);


    } catch(e) {
        throw e;
    }

}


/*
    What do I have?

    1. [x0, y0, z0....] --> basically the vertexes defined in the STEP file. 
    2. [0, 1, 2, 0, 2, 3] --> the indices which tell the GPU the triangle. 


    What do I want?

        1. First I need to load up the positions and the indices into buffers. 
        2. I then need to create a vertex shader 
        3. 




*/