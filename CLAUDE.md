# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a WebGPU-accelerated STEP file parser and 3D renderer. The project parses STEP CAD files (ISO 10303-21), triangulates polygonal faces using GPU-accelerated ear clipping, and renders the results with Three.js.

## Build and Development Commands

```bash
# Start development server with hot reload
yarn dev

# Build for production (runs TypeScript compiler, then Vite build)
yarn build

# Preview production build
yarn preview
```

The project uses Vite as the build tool and TypeScript for type checking.

## High-Level Architecture

### Pipeline Flow

The application follows this data flow:
1. **File Input** (`main.ts`) - User selects a STEP file via browser file input
2. **STEP Parsing** (`step-parser.ts`) - Parses STEP entities into boundary representation
3. **Winding Order** (`signed-area.ts`) - Determines if polygon vertices are CCW using GPU
4. **Triangulation** (`ear-clipping.ts`) - GPU-accelerated ear clipping algorithm
5. **Rendering** (`threejs-render.ts`) - Displays triangulated mesh with Three.js

### WebGPU Compute Shaders

The project uses WebGPU compute shaders (WGSL) for performance-critical geometry operations:

- **Signed Area Calculation** (`signed-area.ts:isCounterClockWiseGPU`) - Determines polygon winding order by computing signed area on GPU
- **Ear Clipping Triangulation** (`ear-clipping.ts`) - GPU-accelerated polygon triangulation with three shader stages:
  - `classifyPoints` shader: Classifies vertices as convex/reflex/collinear based on cross products
  - `isEar` shader: Determines if a convex vertex is an "ear" (no other vertices inside its triangle)
  - `apply` shader: Clips the lowest-indexed ear, updates vertex connectivity

### Key GPU Buffers

The ear clipping algorithm maintains several GPU buffers (see `ear-clipping.ts:initializeBuffers`):
- `pointsBuffer` - Vertex positions with padding `[x, y, z, 0]`
- `outputIndicesBuffer` - Triangle indices output
- `vertexIsEarBuffer` - Boolean flags for ear vertices
- `previousVertexBuffer` / `nextVertexBuffer` - Circular linked list of vertices
- `activeBuffer` - Tracks which vertices are still part of the polygon
- `triangleCount` - Atomic counter for triangles generated
- `classifiedPointsBuffer` - Convexity classification results

### STEP File Parsing

The parser (`step-parser.ts`) extracts these STEP entities:
- `CARTESIAN_POINT` - 3D coordinates
- `VERTEX_POINT` - References to points
- `EDGE_CURVE` - Edges between vertices
- `ORIENTED_EDGE` - Directed edges with orientation flags
- `EDGE_LOOP` - Ordered sequence of oriented edges
- `FACE_OUTER_BOUND` - Outer boundary of a face
- `ADVANCED_FACE` - Topological faces

The parser walks the oriented edges to build boundary polygon vertices in order, handles edge orientation reversals, and ensures CCW winding before triangulation.

### GPU Device Initialization

All GPU operations use a shared WebGPU device obtained via `lib.ts:getGPUDevice()`. This function checks for WebGPU support and requests the GPU adapter/device.

## TypeScript Configuration

- **Target**: ES2022 with ESNext modules
- **Strict mode**: Enabled with `noUnusedLocals` and `noUnusedParameters`
- **Module resolution**: Bundler mode for Vite compatibility
- **Type checking**: `tsc` runs before build, but uses `noEmit` (Vite handles transpilation)

## File Structure Notes

- `src/main.ts` - Browser entry point that wires up file input handler
- `src/step-parser.ts` - STEP file parsing and mesh conversion
- `src/ear-clipping.ts` - GPU ear clipping implementation (~770 lines, includes WGSL shaders)
- `src/signed-area.ts` - Winding order detection (CPU and GPU versions)
- `src/threejs-render.ts` - Three.js mesh creation and scene setup
- `src/lib.ts` - Shared utilities (GPU device, point normalization)
- `step-examples/` - Sample STEP files for testing (includes basic shapes and complex models)
- `src/README.md` - Development notes on ear clipping algorithm implementation

## WebGPU Considerations

- The project requires a WebGPU-capable browser
- Shaders are written in WGSL (WebGPU Shading Language)
- GPU buffer operations are asynchronous - use `await device.queue.onSubmittedWorkDone()` between dependent operations
- Point coordinates are normalized to 4-component vectors `[x, y, z, padding]` for GPU alignment
- The ear clipping algorithm runs iteratively on CPU but executes compute shaders on GPU each iteration


## Critical Dependencies

**IMPORTANT - OpenCascade.js Version**: This project requires `opencascade.js@2.0.0-beta.fdece36` (NOT the stable 1.1.1 version). The beta version includes:
- XCAF support for colors and document structure
- Dynamic module loading (`loadDynamicLibrary`)
- All the `module.TK*.wasm` files needed for STEP parsing

If you ever need to reinstall node_modules:
1. Make sure `package.json` has `"opencascade.js": "2.0.0-beta.fdece36"`
2. Run `npm install` (not yarn, to avoid lock file conflicts)
3. Verify the install by checking that `node_modules/opencascade.js/dist/` contains many `module.*.wasm` files

The stable npm version (1.1.1) is a minimal build that does NOT support STEP file parsing properly.

## Testing

**IMPORTANT**: When testing STEP file rendering, always use `tests/visual-validation.html` instead of `occ-test.html` or `occ-test.js`. The visual validation harness provides proper screenshot comparison and regression testing.

## Style

1. Prefer descriptive variable names.
2. Instead of using ternary operators, prefer using if statements.

## Debugging Approach

When debugging tessellation issues, **always explain problems visually using ASCII diagrams** before attempting a fix. This helps ensure the problem and proposed solution are clearly understood. For example:

```
Problem: UV boundary crosses seam
┌─────────────────┐
│ B               │
│ │               │  Point A at V=+π
│ │   JUMP!       │  Point B at V=-π
│ └───────────────┤ A  (they're neighbors in 3D but far apart in UV)
└─────────────────┘

Fix: Shift V to [0, 2π] range
┌─────────────────┐
│                 │
│    ┌───────┐    │  Now A and B are adjacent
│    │ A───B │    │  (continuous boundary)
│    └───────┘    │
└─────────────────┘
``` 
