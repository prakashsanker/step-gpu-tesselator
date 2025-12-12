import { BYTE_SIZE } from "./gpu-tesselate";
import { getGPUDevice, normalizePoints } from "./lib";

const VERTEX_PER_TRIANGLE = 3;
const G0_POINTS       = 0; // read-only storage
const G0_PREV         = 1; // storage (we will mutate during apply)
const G0_NEXT         = 2; // storage
const G0_ACTIVE       = 3; // storage (u32 per vertex)
const G0_ACTIVE_COUNT = 4; // storage (atomic<u32>)
const G0_TRI_COUNT    = 5; // storage (atomic<u32>)

function createConvexityBGL(device: GPUDevice) {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"}
            }
        ]
    });
}

function createIsEarBGL(device: GPUDevice) {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage"},
            },
            {
                binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage"}, 
            },
        ]
    });
}

function creatApplyBGL(device: GPUDevice) {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage"},
            },
            {
                binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage"},
            },
            {
                binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage"},
            },
        ]
    });
}


function createPolygonStateBGL(device: GPUDevice) {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: G0_POINTS,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage"}
            },
            
            {
                binding: G0_PREV,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"}
            },
            {
                binding: G0_NEXT,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"}
            },
            {
                binding: G0_ACTIVE,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"}
            },
            {
                binding: G0_ACTIVE_COUNT,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"}
            },
            {
                binding: G0_TRI_COUNT,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"}
            }
        ]
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
            @group(0) @binding(0) var<storage, read> pointsBuffer: array<Point>;
            @group(0) @binding(1) var<storage, read_write> vertexIsEarBuffer: array<u32>;
            @group(0) @binding(2) var<storage, read> previousVertexBuffer: array<u32>;
            @group(0) @binding(3) var<storage, read> nextVertexBuffer: array<u32>;
            @group(0) @binding(4) var<storage, read> activeCount: u32;
            @group(0) @binding(5) var<storage, read> triangleCount: u32;
            @group(0) @binding(6) var<storage, read> classifiedPointsBuffer: array<u32>;

            @compute @workgroup_size(WORKGROUP_SIZE) fn isEar(
                @builtin(global_invocation_id) id: vec3<u32>
            ) {
                let i = id.x;
                let pointsBufferLength = arrayLength(&pointsBuffer);


            if (i >= pointsBufferLength) {
                return;
            }

            const vertexConvexity = classifiedPointsBuffer[i];
            if (vertexConvexity != CONVEX) {
                return;
            }

            
            var currentVertexIndex = i;
            var previousVertexIndex = previousVertexBuffer[currentVertexIndex];
            var nextVertexIndex = nextVertexBuffer[currentVertexIndex];
            // we want to now loop through all the other vertices and see if any of them are inside the triangle.
            for (let j = 0; j < pointsBufferLength; j++) {
                if (j != currentVertexIndex && j != previousVertexIndex && j != nextVertexIndex) {
                    var point = pointsBuffer[j];
                    var isInside = pointInTriangle(previousVertexIndex, currentVertexIndex, nextVertexIndex, point);
                    if (isInside) {
                      // if it is inside, then we need to set the vertex is ear buffer to false.
                      // we have our answer, we should return
                      vertexIsEarBuffer[j] = 0u;
                      return;
                    }
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
    return convexityCheckShader;
}

export function createPointInTriangleShader(device: GPUDevice) {
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
    return pointInTriangleShader;
}

// The points are assumed to be in CCW.
export async function earClipping(points: number[][]) {
    try {
        const device = await getGPUDevice();
        
        
        




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