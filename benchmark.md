# WebGPU STEP Parser - Benchmark Results

This document tracks performance benchmarks comparing our GPU-accelerated STEP parser against occt-import-js (WebAssembly).

---

## Current Results (2026-01-04)

### Summary

| Test | Hybrid GPU | OCCT | Speedup | Winner |
|------|-----------|------|---------|--------|
| Simple Square (no holes) | 3.37ms | 3.60ms | **1.07x faster** | GPU |
| Small (4 holes) | 2.73ms | 4.83ms | **1.77x faster** | GPU |
| Medium (25 holes) | 7.20ms | 15.17ms | **2.11x faster** | GPU |
| Large (100 holes) | 3.17ms | 63.90ms | **20.18x faster** | GPU |
| XLarge (400 holes) | 35.93ms | 555.67ms | **15.46x faster** | GPU |

**Average: 8.12x faster than OCCT**
**GPU wins: 5/5 benchmarks**

### Timing Breakdown (XLarge - 400 holes)

| Phase | Time |
|-------|------|
| STEP Parsing | 2.50ms |
| Face Extraction | 9.48ms |
| Hole Bridging | 9.48ms |
| GPU Triangulation | 6.70ms |
| GPU Smooth Normals | ~2ms |
| **Total** | **35.93ms** |

---

## Optimizations Applied

### 1. Hybrid GPU/CPU Triangulation
- **File:** `src/triangulate-hybrid.ts`
- **Description:** Uses GPU for polygons ≤256 vertices, CPU earcut.js for larger
- **Impact:** Handles arbitrary polygon sizes while maximizing GPU usage

### 2. Parallel Face Processing
- **File:** `src/step-parser.ts` (parseStepToMesh)
- **Description:** Process all faces in parallel using `Promise.all`
- **Impact:** ~8x speedup on large files (26.67ms → 3.10ms on 100 holes)

### 3. Console.log Removal
- **Description:** Removed 78 console.log statements from hot paths
- **Impact:** Reduced overhead in tight loops

### 4. GPU-Accelerated Smooth Normals
- **File:** `src/smooth-normals-gpu.ts`
- **Description:** WebGPU compute shader for angle-weighted vertex normals
- **Technique:** Atomic integer accumulation (scale floats to i32, atomicAdd, normalize)
- **Impact:** 1.5-2x faster than CPU for normal computation

### 5. Batched GPU Ear Clipping
- **File:** `src/ear-clipping-batched.ts`
- **Description:** Process multiple small polygons in single GPU dispatch
- **Limitation:** Max 256 vertices per polygon

---

## Benchmark Files

Located in `step-examples/benchmark/`:

| File | Vertices | Holes | Triangles |
|------|----------|-------|-----------|
| `simple-square.step` | 4 | 0 | 2 |
| `small-4-holes.step` | ~20 | 4 | 22 |
| `medium-25-holes.step` | ~100 | 25 | 127 |
| `large-100-holes.step` | ~500 | 100 | 502 |
| `xlarge-400-holes.step` | ~2000 | 400 | 2002 |

---

## Running Benchmarks

```bash
# Run full benchmark suite
node tests/benchmark.js

# Run tests
yarn test
```

---

## Architecture

```
STEP File
    ↓
parseStep() ─────────────────── STEP entity parsing
    ↓
extractFaceBoundsWithCurves() ─ Face/hole extraction + curve sampling
    ↓
projectFaceLoopsTo2D() ──────── 3D → 2D projection
    ↓
normalizeWinding() ───────────── CCW outer, CW holes
    ↓
bridgeAllHoles() ─────────────── Merge holes into single polygon
    ↓
triangulateHybrid() ──────────── GPU/CPU ear clipping
    ↓
computeSmoothNormalsGPU() ────── GPU vertex normals
    ↓
Final Mesh (positions, indices, normals)
```

---

## Potential Future Optimizations

| Optimization | Expected Impact | Complexity | Status |
|--------------|-----------------|------------|--------|
| GPU hole bridging | 2-3x on bridging phase | High | Not started |
| WebAssembly for CPU fallback | 1.5-2x on large polygons | Medium | Not started |
| Shared GPU buffers (reuse) | Reduce allocation overhead | Low | Not started |
| Web Workers for multi-face | True parallelism on many faces | Medium | Tested - slower |
| Spatial index (quadtree) | O(n log n) visibility | Medium | Tested - no benefit |

---

## Historical Results

### 2026-01-04 (Spatial Index Experiment - Reverted)
- Implemented quadtree spatial index for hole bridging visibility queries
- Expected O(n log n) improvement over O(n²) linear scan
- **Result: Made performance WORSE** (overhead outweighed benefits at current sizes)
- XLarge with spatial index: 40.57ms (12.47x faster) vs 35.93ms without (15.46x faster)
- Reverted changes - quadtree only beneficial for much larger polygons (>1000+ vertices per face)

### 2026-01-04 (GPU Normals)
- Added GPU-accelerated smooth normals
- Restored performance to pre-normals levels while including normal computation
- Large: 3.27ms (19.83x faster than OCCT)
- XLarge: 35.27ms (16.54x faster than OCCT)

### 2026-01-04 (Pre-GPU Normals)
- CPU smooth normals added overhead
- Large: 5.87ms
- XLarge: 81.07ms

### 2026-01-04 (Parallel Processing)
- Added Promise.all for face processing
- Large: 3.10ms (from 26.67ms)
- 8.6x improvement on large files
