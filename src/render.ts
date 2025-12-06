type Mesh = {
    positions: Float32Array;
    indices: Uint32Array;
  };
  
  export async function renderMeshWithWebGPU(mesh: Mesh, canvas: HTMLCanvasElement) {
    // -------------------------------
    // 1. Get WebGPU device + context
    // -------------------------------
    if (!('gpu' in navigator)) {
      throw new Error("WebGPU not supported in this browser.");
    }
  
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("Failed to get GPU adapter.");
    }
  
    const device = await adapter.requestDevice();
  
    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    if (!context) {
      throw new Error("Failed to get WebGPU context from canvas.");
    }
  
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  
    context.configure({
      device,
      format: presentationFormat,
      alphaMode: "opaque",
    });
  
    // ---------------------------------
    // 2. Create GPU buffers for the mesh
    // ---------------------------------
  
    // Vertex buffer: holds positions (x,y,z)
    const vertexBuffer = device.createBuffer({
      size: mesh.positions.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
    });
    device.queue.writeBuffer(vertexBuffer, 0, mesh.positions);
  
    // Index buffer: holds triangle indices
    const indexBuffer = device.createBuffer({
      size: mesh.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
    });
    device.queue.writeBuffer(indexBuffer, 0, mesh.indices);
  
    const indexCount = mesh.indices.length;
  
    // --------------------------
    // 3. Create shader module
    // --------------------------
  
    const shaderModule = device.createShaderModule({
      code: /* wgsl */ `
        struct VertexOutput {
          @builtin(position) position : vec4<f32>,
          @location(0) v_color : vec3<f32>,
        };
  
        @vertex
        fn vs_main(@location(0) in_pos : vec3<f32>) -> VertexOutput {
          var out : VertexOutput;
  
          // STEP coords are in [0,1] range. Convert to clip space [-1,1]:
          var pos = in_pos;
          let clipPos = vec2<f32>(
            pos.x * 2.0 - 1.0,
            pos.y * 2.0 - 1.0
          );
  
          out.position = vec4<f32>(clipPos, pos.z, 1.0);
  
          // Simple constant color (light green)
          out.v_color = vec3<f32>(0.3, 0.9, 0.4);
          return out;
        }
  
        @fragment
        fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
          return vec4<f32>(in.v_color, 1.0);
        }
      `,
    });
  
    // --------------------------------------
    // 4. Describe vertex buffer layout
    // --------------------------------------
  
    // Our mesh.positions is [x,y,z, x,y,z, ...] → 3 floats per vertex
    const vertexBuffers: GPUVertexBufferLayout[] = [
      {
        arrayStride: 3 * 4, // 3 floats * 4 bytes = 12 bytes per vertex
        attributes: [
          {
            shaderLocation: 0,  // matches @location(0) in vs_main
            offset: 0,
            format: "float32x3",
          },
        ],
      },
    ];
  
    // ------------------------------
    // 5. Create render pipeline
    // ------------------------------
  
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: vertexBuffers,
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: presentationFormat,
          },
        ],
      },
      primitive: {
        topology: "triangle-list", // interpret indices as independent triangles
        cullMode: "back",
      },
    });
  
    // ------------------------------
    // 6. Render loop (one frame)
    // ------------------------------
  
    function frame() {
      // Get the current canvas texture to render into
      const textureView = context.getCurrentTexture().createView();
  
      // Create a command encoder (record GPU commands)
      const commandEncoder = device.createCommandEncoder();
  
      // Begin a render pass
      const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: textureView,
            clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }, // dark gray background
            loadOp: "clear",  // clear each frame
            storeOp: "store", // store the result so it shows on screen
          },
        ],
      });
  
      // Bind pipeline + buffers
      renderPass.setPipeline(pipeline);
      renderPass.setVertexBuffer(0, vertexBuffer);
      renderPass.setIndexBuffer(indexBuffer, "uint16");
  
      // Draw the indexed geometry
      renderPass.drawIndexed(indexCount, 1, 0, 0, 0);
  
      // End render pass and submit commands
      renderPass.end();
      device.queue.submit([commandEncoder.finish()]);
  
      // Schedule next frame
      requestAnimationFrame(frame);
    }
  
    // Kick off the first frame
    requestAnimationFrame(frame);
  }