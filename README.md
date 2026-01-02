# WebGPU STEP File Renderer

A WebGPU-accelerated STEP file parser and 3D renderer that triangulates polygonal faces using GPU compute shaders.

> **Beta Notice**: This project is currently in beta. It only supports **ear clipping triangulation** for simple polygonal faces. Features like curved surfaces, holes, and complex BREP operations are not yet implemented.

## Overview

This project parses STEP CAD files (ISO 10303-21), extracts boundary representation geometry, and triangulates faces using a GPU-accelerated ear clipping algorithm. The triangulated mesh is then rendered with Three.js.

### Current Capabilities

- Parse STEP files with simple polygonal faces
- GPU-accelerated ear clipping triangulation via WebGPU compute shaders
- Automatic winding order detection and correction
- Three.js rendering of triangulated meshes

### Limitations

- Only supports planar polygonal faces (no curves or NURBS)
- No support for faces with holes
- Limited to single-shell geometry

## Requirements

- Node.js 18+
- A WebGPU-capable browser (Chrome 113+, Edge 113+, or Firefox Nightly with flags)

## Getting Started

```bash
# Install dependencies
yarn install

# Start development server
yarn dev
```

Then open `http://localhost:5173` in a WebGPU-capable browser.

## Running Tests

Tests use Puppeteer to run WebGPU compute shaders in a headless Chrome browser.

```bash
# Run the full test suite
yarn test

# Run tests in watch mode
yarn test:watch
```

The test suite covers:
- Convex polygon triangulation
- Concave polygon triangulation
- Winding order detection
- STEP file parsing and integration
- Triangle output validity

## Project Structure

```
src/
  main.ts           # Browser entry point
  step-parser.ts    # STEP file parsing
  ear-clipping.ts   # GPU ear clipping implementation
  signed-area.ts    # Winding order detection
  threejs-render.ts # Three.js rendering
  lib.ts            # Shared utilities

tests/
  run-tests.js      # Puppeteer test runner
  test-harness.html # Browser test environment

step-examples/      # Sample STEP files for testing
```

## License

MIT
