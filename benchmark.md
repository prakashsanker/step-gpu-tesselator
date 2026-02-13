# WebGPU STEP Parser - Benchmark Results

This document tracks performance benchmarks comparing our GPU-accelerated STEP parser against occt-import-js (WebAssembly).

---

## Latest Results (2026-02-11, Beat-OCCT M0 Baseline)

### Harness Update (Representative Suites)

Benchmark harness has been updated to use representative model suites instead of mostly synthetic-only defaults.

New commands:

- `npm run -s bench:canary` (fast dev loop)
- `npm run -s bench:representative` (real-world gate, includes Electronic Enclosure)
- `npm run -s bench:full` (broader milestone sweep)

Canary policy (guarded in code):

- `VM-001` is always included in canary as a required large-file sentinel.
- `rocky_house*` and `rotor-201nal.step` stay excluded from routine canary/perf loops.

`tests/benchmark-comprehensive.js` now supports:

- `--suite canary|representative|full`
- `--runs N --warmup N --timeout-ms N`
- `--filter PATTERN`

Known long/pathological files are explicitly excluded from routine suites:

- `step-examples/complex/nissan.step`
- `step-examples/complex/rocky_house.step`
- `step-examples/complex/rotor-201nal.step`

Baseline runner default suite is now `representative` via:

- `npm run -s baseline:beat` (uses `--bench-suite representative`)

### Baseline Run Metadata

- Commit: `c59b8a6`
- Branch: `beat-occt-import-js`
- Command:
  - `npm run -s baseline:beat -- --date 2026-02-11 --skip-ai`
- Artifacts:
  - `diagnostics/beat-occt-import-js/2026-02-11/2026-02-11T07-30-04-101Z-c59b8a6/`

### Correctness Gate

- `tests/run-tests.js`: **FAIL**
  - Cause: `testVisualHoleRendering` timeout (`Waiting failed: 60000ms exceeded`)
  - Note: all earlier non-visual suites in the same run passed before the timeout.

### Performance A: `tests/benchmark-comprehensive.js` (Fast Set, 8 Models)

| Metric | Value |
|------|------|
| Successful | 8 / 8 |
| Average speedup vs `occt-import-js` | **1.20x faster** |
| Wins vs `occt-import-js` | **1 / 8** |

### Performance B: `tests/benchmark.js` (Core Synthetic Set)

| Test | Hybrid GPU vs OCCT |
|------|---------------------|
| Simple Square (no holes) | **2.14x faster** |
| Small (4 holes) | **4.91x faster** |
| Medium (25 holes) | **6.69x faster** |
| Large (100 holes) | **4.51x faster** |
| XLarge (400 holes) | **1.08x slower** |

**Average: 3.84x faster than OCCT**  
**GPU wins: 4/5 benchmarks**

### Comparison vs Previous Published Baseline

Compared to the previous `tests/benchmark.js` snapshot (2026-01-04):

| Metric | 2026-01-04 | 2026-02-11 | Delta |
|------|-------------|-------------|-------|
| Average speedup vs OCCT | 8.12x faster | 3.84x faster | -4.28x |
| Wins vs OCCT | 5/5 | 4/5 | -1 |

For the Beat-OCCT effort, treat the 2026-02-11 run as the fresh baseline reference point.

---

## M1.1 Incremental Tuning (2026-02-11)

### Changes

1. Planar no-hole triangulation switched from direct `earClipping()` to `triangulateFast()` with safe fallback.
2. Curve sampling defaults reduced (runtime-tunable):
   - `__EDGE_DEFAULT_SAMPLES__`: `24 -> 16`
   - `__EDGE_BSPLINE_SAMPLES__`: `32 -> 16`
   - `__EDGE_ELLIPSE_SAMPLES__`: `32 -> 20`
   - circle step: `π/16 -> π/12`

### Canary Results (single-run, direction-only)

Command:
- `npm run -s bench:canary`

Before sampling tune (after planar dispatch):

| Model | Ours (ms) | Ref (ms) | Speed |
|------|-----------:|---------:|------:|
| Plate XLarge | 4191.2 | 405.6 | 10.33x slower |
| Cone | 71.3 | 399.3 | 5.60x faster |
| Cylinder With Hole | 20.1 | 37.7 | 1.88x faster |
| BSpline Bowl | 43.2 | 96.7 | 2.24x faster |
| Conical Surface | 121.6 | 86.4 | 1.41x slower |
| VM-001 | 481.3 | 386.4 | 1.25x slower |

After sampling tune:

| Model | Ours (ms) | Ref (ms) | Speed | Ours Tris | Ref Tris |
|------|-----------:|---------:|------:|----------:|---------:|
| Plate XLarge | 4177.5 | 367.7 | 11.36x slower | 172 | 172 |
| Cone | 51.1 | 73.1 | 1.43x faster | 1078 | 1841 |
| Cylinder With Hole | 14.8 | 38.2 | 2.59x faster | 92 | 308 |
| BSpline Bowl | 49.7 | 69.1 | 1.39x faster | 338 | 116 |
| Conical Surface | 84.2 | 69.1 | 1.22x slower | 1728 | 1318 |
| VM-001 | 396.6 | 293.0 | 1.35x slower | 22348 | 3116 |

Aggregate (after sampling tune):
- Wins vs `occt-import-js`: `3/6`
- Median speedup: `1.10x` (single-run)
- Remaining major gap: cold-start `loadStepFile` and triangle inflation on `VM-001`.

### Harness Stability/Prewarm Update

Additional harness changes:

1. Fixed multi-run aggregation bug in `tests/benchmark-comprehensive.js` (it previously returned after first successful run).
2. Added one-time prewarm phase (default on, `--no-prewarm` to disable) to separate cold-start cost from steady-state comparison.

Command:
- `npm run -s bench:canary`

Warm-state canary result (prewarm on):

| Model | Ours (ms) | Ref (ms) | Speed | Ours Tris | Ref Tris |
|------|-----------:|---------:|------:|----------:|---------:|
| Plate XLarge | 126.9 | 229.0 | 1.80x faster | 172 | 172 |
| Cone | 70.3 | 139.4 | 1.98x faster | 1078 | 1841 |
| Cylinder With Hole | 24.1 | 31.3 | 1.30x faster | 92 | 308 |
| BSpline Bowl | 42.5 | 70.0 | 1.65x faster | 338 | 116 |
| Conical Surface | 88.8 | 47.2 | 1.88x slower | 1728 | 1318 |
| VM-001 | 366.5 | 305.4 | 1.20x slower | 22348 | 3116 |

Aggregate:
- Wins vs `occt-import-js`: `4/6`
- Speedup median: `1.47x`
- Ours avg runtime: `119.9ms` (p90 `246.7ms`)
- Ref avg runtime: `137.0ms` (p90 `267.2ms`)

### Correctness Gate

Command:
- `npm test`

Result:
- Non-visual suites passed through curved-surface tests.
- Runner timed out in visual stage:
  - `testVisualHoleRendering`: `Waiting failed: 60000ms exceeded`.

---

## M1.1.2 Trim Density Tuning + Rollback (2026-02-11)

### Why

The initial M1.1.2 experiment introduced a global trim-loop simplification pass that caused instability on cone-heavy cases. We rolled that back and kept cone-specific simplification only, then tuned default trim-density caps to reduce triangle inflation and runtime.

### Changes

- Removed global trim-loop simplification pass in `getFaceTrimLoopsUV()` (kept cone-only simplification).
- Lowered default trim grid knobs:
  - `__TRIM_GRID_SCALE__`: `1.0 -> 0.85`
  - `__TRIM_MIN_GRID_DENSITY__`: `12 -> 10`
  - `__TRIM_MAX_GRID_DENSITY__`: `40 -> 32`
  - `__TRIM_MAX_GRID_DENSITY_NO_HOLES__`: `24 -> 20`
  - `__TRIM_MAX_GRID_DENSITY_WITH_HOLES__`: `40 -> 24`
  - `__CONE_TRIM_MAX_GRID_DENSITY__`: `36 -> 24`
  - `__CONE_TRIM_MAX_GRID_DENSITY_NO_HOLES__`: `16 -> 14`
  - `__TRIM_HIGH_COMPLEXITY_POINT_THRESHOLD__`: `900 -> 600`
- Lowered OCCT-inspired cone trim-domain defaults:
  - `__OCCT_INSPIRED_TRIM_GRID_SCALE__`: `1.35 -> 0.9`
  - `__OCCT_INSPIRED_TRIM_MIN_GRID__`: `24 -> 14`
  - `__OCCT_INSPIRED_TRIM_MAX_GRID__`: `64 -> 32`

### Canary Result

Command:
- `npm run -s bench:canary`

| Model | Ours (ms) | Ref (ms) | Speed | Ours Tris | Ref Tris |
|------|-----------:|---------:|------:|----------:|---------:|
| Plate XLarge | 162.8 | 257.3 | 1.58x faster | 172 | 172 |
| Cone | 54.6 | 107.3 | 1.97x faster | 358 | 1841 |
| Cylinder With Hole | 11.5 | 34.9 | 3.04x faster | 92 | 308 |
| BSpline Bowl | 42.4 | 84.7 | 2.00x faster | 200 | 116 |
| Conical Surface | 91.0 | 62.0 | 1.47x slower | 832 | 1318 |
| VM-001 | 372.3 | 365.3 | 1.02x slower | 15812 | 3116 |

Aggregate:
- Wins vs `occt-import-js`: `4/6`
- Speedup median: `1.77x`
- Ours avg runtime: `122.4ms` (p90 `267.6ms`)
- Ref avg runtime: `151.9ms` (p90 `311.3ms`)

### Correctness Gate

Command:
- `npm test`

Result:
- Non-visual suites pass.
- Visual stage still times out at `testVisualHoleRendering` (`Waiting failed: 60000ms exceeded`).

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
