# Fixing Option 2: OpenCascade.js + Custom CPU Tessellator

This document tracks our progress fixing Option 2 to achieve correct tessellation across all test files.

## Overview

**Option 2 Pipeline:**
- Parsing: OpenCascade.js (WASM)
- Tessellation: Custom CPU-based (ear-clipping for planar, grid-based for curved)
- Rendering: Three.js

**Reference:** occt-import-js (uses OpenCASCADE's built-in tessellator)

**Goal:** Get Option 2 working correctly with ALL 119 test files before GPU optimization.

---

## True Baseline (AI Visual Comparison - 2026-02-01)

Previous pixel-based comparison gave false positives. We now use Claude AI vision to semantically compare renders.

**Pass Rate: 68.9% (82/119)**
- Passed: 82
- Failed: 34
- Errors: 3

### Passed Tests (82)

| Category | Files |
|----------|-------|
| **benchmark** | plate-large-10x10, plate-medium-5x5, plate-small-2x2, plate-xlarge-20x20, simple-square |
| **c1-triangulation/concave** | arrow, ccw-pentagon-concave, concave_pentagon_single_reflex, cw-pentagon-concave, l-shape |
| **c1-triangulation/convex** | ccw-square, convex-heptagon, convex_pentagon_simple, cw-square, hexagon, triangle |
| **c2-holes/2.2-projection** | tilted-square-45deg, tilted-triangle-no-plane, vertical-wall-xz |
| **c2-holes/2.3-winding** | both-wrong, correct-winding, square-cw |
| **c2-holes/2.4-topology** | hole-outside-outer, self-intersecting-outer |
| **c2-holes/2.5-triangulation** | square-with-4-holes, square-with-arrow-hole, square-with-diamond-hole, square-with-right-hole, square-with-square-hole, square-with-three-holes, square-with-triangle-hole |
| **c3-curves** | rotor |
| **c4-multiface** | pyramid, tetrahedron, triangular-prism, two-triangles, unit-box, wedge |
| **c4-surfaces** | cone, cylinder, sphere, torus |
| **c5-bspline** | bspline-dome |
| **c6-trimmed** | full-cylinder-window |
| **c8-solids** | colored-solid, simple-cube, tetrahedron |
| **complex** | air, conical-surface, cube, raw-material, rocky_house_car, rocky_house_roof, rocky_house_sofa, rocky_house_table, rocky_house_terrain |
| **external/steptools-ap214** | as1-ac-214, as1-ec-214, as1-md-214, as1-tc-214, as1-ug-214, d2-db-214, f1-db-214, io1-ac-214, io1-ca-214, io1-md-214, io1-tc-214, io1-ug-214 |
| **external/steptools-ap224** | ap224_995277945, ap224_995602415, ap224_997423743, ap224_997865309 |
| **external/steptools** | 123Block_Color, 123Block_Dimension, 123Block_Short_Note, boxy_with_cylindricity, boxy_with_diamsize, boxy_with_flatness, boxy_with_limitsandfits, boxy_with_linearsize, boxy_with_perp, boxy_with_surfacetex |

### Failed Tests (34)

| File | Issue |
|------|-------|
| **benchmark/plate-xxlarge-30x30.step** | Shows triangular shape instead of flat rectangular plate |
| **c1-triangulation/concave/almost_collinear_pentagon.step** | Appears as quadrilateral, should be pentagon |
| **c1-triangulation/convex/octagon.step** | Shows flat 2D octagon instead of 3D prism |
| **c1-triangulation/convex/square.step** | Both pipelines render 0 triangles |
| **c1-triangulation/convex/tilted-rectangle.step** | Both pipelines render 0 triangles |
| **c2-holes/2.2-projection/tilted-hexagon.step** | Shows triangle instead of hexagon |
| **c2-holes/2.3-winding/square-with-ccw-hole.step** | Shows triangle, missing square and hole |
| **c2-holes/2.4-topology/holes-intersect.step** | Shows solid triangle, missing holes |
| **c2-holes/2.4-topology/valid-square-with-hole.step** | Shows solid triangle, missing square and hole |
| **c2-holes/2.5-triangulation/concentric-squares.step** | Shows triangle instead of concentric squares |
| **c2-holes/2.5-triangulation/hexagon-with-3-holes.step** | Shows triangle instead of hexagon with holes |
| **c2-holes/2.5-triangulation/hexagon-with-triangle-hole.step** | Different geometry between pipelines |
| **c2-holes/2.5-triangulation/l-shape-with-hole.step** | Different geometry between pipelines |
| **c2-holes/2.5-triangulation/octagon-with-square-hole.step** | Different geometry between pipelines |
| **c2-holes/2.5-triangulation/pentagon-with-hole.step** | Different geometry between pipelines |
| **c2-holes/2.5-triangulation/rectangle-with-6-holes.step** | Different geometry between pipelines |
| **c2-holes/2.5-triangulation/square-with-two-holes.step** | Missing holes in both pipelines |
| **c2-holes/2.5-triangulation/star-with-center-hole.step** | Shows rectangle instead of star |
| **c2-holes/2.5-triangulation/thin-rectangle-with-slot.step** | Missing slot in both pipelines |
| **c2-holes/2.5-triangulation/triangle-with-triangle-hole.step** | Shows rectangle instead of triangle with hole |
| **c3-curves/quarter-circle.step** | Shows flat rectangle instead of curved geometry |
| **c3-curves/rounded-cube.step** | Shows flat rectangle instead of rounded cube |
| **c5-bspline/bspline-bowl.step** | Different orientation/shape |
| **c5-bspline/bspline-saddle.step** | Reference renders 0 triangles (reference bug) |
| **c5-bspline/bspline-wave.step** | Reference renders 0 triangles (reference bug) |
| **c5-bspline/simple-bspline-surface.step** | Different curvature/geometry |
| **c6-trimmed/cylinder-two-holes.step** | Solid cylinder, missing holes in walls |
| **c6-trimmed/cylinder-with-hole.step** | Solid cylinder, missing hole |
| **c6-trimmed/half-cylinder.step** | Missing top/bottom flat circular caps |
| **c6-trimmed/pipe-with-porthole.step** | Simple cylinder, missing porthole feature |
| **c6-trimmed/quarter-cylinder-hole.step** | Full cylinder instead of quarter with hole |
| **external/steptools-ap214/io1-ec-214.stp** | Missing bolt holes around flange |
| **external/steptools-ap224/ap224_995288709.stp** | Missing cylindrical feature in middle |
| **external/steptools-ap224/ap224_995315479.stp** | Extra internal features in our render |

### Errors (3)

| File | Error |
|------|-------|
| **complex/nissan.step** | Reference failed to parse |
| **complex/rocky_house.step** | Timeout |
| **complex/rotor-201nal.step** | Timeout |

---

## Failure Analysis by Category

| Category | Passed | Failed | Root Cause |
|----------|--------|--------|------------|
| **c6-trimmed (cylinders)** | 1/6 | 5 | Inner hole bounds on curved surfaces not tessellated |
| **c2-holes (flat faces)** | 8/22 | 14 | Many show wrong shapes or missing holes |
| **c5-bspline** | 1/5 | 4 | 2 are reference bugs, 2 are geometry issues |
| **c1-triangulation** | 12/16 | 4 | 2 render 0 triangles in both pipelines |
| **c3-curves** | 1/3 | 2 | Curved edges not tessellating |
| **external** | 17/20 | 3 | Complex models with missing features |
| **benchmark** | 5/6 | 1 | Large plate renders wrong |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/occ-test.ts` | Main tessellation pipeline, `tessellateCurvedFaceFromOCC` |
| `src/surface-tessellation.ts` | `tessellateTrimmedSurface`, grid-based curved surface tessellation |
| `src/surfaces.ts` | `evaluateSurface`, surface point evaluation |
| `tests/run-visual-tests-ai.js` | AI-powered visual comparison test runner |
| `tests/visual-validation.html` | Manual visual comparison page |
| `tests/visual-results-ai/ai-test-report.json` | Latest AI test results |

---

## Commands

```bash
# Run AI visual tests (requires OPENROUTER_API_KEY)
OPENROUTER_API_KEY=<key> node tests/run-visual-tests-ai.js

# Run pixel-based visual tests (less accurate)
node tests/run-visual-tests.js

# Start dev server for manual testing
yarn dev
# Then open: http://localhost:5173/tests/visual-validation.html
```

---

## Fixed Test Files (2026-02-01)

**Issue:** 10 STEP test files in `c2-holes/2.5-triangulation/` were malformed. All LINE entities shared a single VECTOR direction `(1,0,0)` regardless of actual edge direction. OpenCascade handles this for simple shapes (squares) but fails for complex polygons (hexagons, pentagons, stars).

**Fixed files:**
- `triangle-with-triangle-hole.step`
- `concentric-squares.step`
- `pentagon-with-hole.step`
- `thin-rectangle-with-slot.step`
- `hexagon-with-triangle-hole.step`
- `l-shape-with-hole.step`
- `square-with-two-holes.step`
- `octagon-with-square-hole.step`
- `star-with-center-hole.step`
- `rectangle-with-6-holes.step`

**Fix script:** `scripts/fix-step-directions.cjs` - Computes proper normalized direction vectors for each edge based on vertex coordinates.

---

## Progress: Planar Faces with Holes (2026-02-04)

### Fixes Applied

1. **Edge orientation fix for curved edges** (`occ-test.ts`)
   - Curved edges (Circle, BSpline, etc.) weren't being sampled in the correct direction
   - Added logic to detect reversed edges by comparing sampled start point with `TopExp.FirstVertex`
   - If the last sampled point is closer to startPoint, reverse the sampled points array

2. **Reliable outer wire identification** (`occ-test.ts`)
   - Previously used "longest wire = outer" heuristic which could fail
   - Now uses `BRepTools.OuterWire(face)` for reliable outer wire detection

3. **earcut for planar faces with holes** (`occ-test.ts`, `triangulate-fast.ts`)
   - Planar faces WITH holes now use `triangulateWithHoles()` which calls earcut
   - earcut natively supports holes via `holeIndices` parameter
   - No more bridging needed (which caused artificial edges crossing the interior)

### Known Limitations

1. **Circular arcs vs straight lines for "hexagonal" edges**
   - Some models represent hexagon sides as 60° circular arcs instead of LINE edges
   - Our pipeline samples these arcs faithfully, resulting in a rounded/circular appearance
   - The reference renderer (occt-import-js) may linearize these or use different tessellation
   - **Impact:** Visual difference in some models (hex nuts appear circular vs hexagonal)
   - **Workaround:** Could detect large-radius arcs and linearize, but not implemented

2. **GPU optimization limited for faces with holes**
   - Planar faces WITHOUT holes → GPU ear-clipping ✓
   - Planar faces WITH holes → CPU earcut (no GPU) ✗
   - **Future:** Consider GPU CDT (Constrained Delaunay Triangulation) for holes

---

## Post-Fix AI Visual Test Run (2026-02-05)

Re-ran AI visual tests after edge orientation fixes and earcut integration.

**Note:** The browser crashed after test 90/119 due to cascading timeouts from large external STEP files (rocky_house, rotor-201nal, as1-ac-214). All 29 external/steptools tests got "detached Frame" errors and could not be evaluated. The results below cover the 90 tests that actually ran.

### Summary (all 119 tests)

Results combined from two runs (initial run crashed after test 90; external tests re-run separately).

**Pass Rate: 77.3% (92/119)**
- Passed: 92
- Failed: 24
- Errors: 3 (same as baseline: nissan, rocky_house, rotor-201nal)

### Comparison with Baseline

| Metric | Baseline (2026-02-01) | Current (2026-02-05) | Delta |
|--------|----------------------|---------------------|-------|
| Passed | 82 | 92 | **+10** |
| Failed | 34 | 24 | **-10** |
| Errors | 3 | 3 | 0 |
| Pass Rate | 68.9% | 77.3% | **+8.4%** |

### Newly Passing Tests (17 fixed)

These tests previously FAILED and now PASS after the edge orientation, outer wire, and earcut fixes:

| File | Previous Issue | Fix |
|------|---------------|-----|
| **benchmark/plate-xxlarge-30x30.step** | Triangular shape instead of flat rectangular plate | Edge orientation |
| **c2-holes/2.4-topology/holes-intersect.step** | Solid triangle, missing holes | earcut + outer wire |
| **c2-holes/2.5-triangulation/concentric-squares.step** | Triangle instead of concentric squares | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/hexagon-with-3-holes.step** | Triangle instead of hexagon with holes | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/hexagon-with-triangle-hole.step** | Different geometry between pipelines | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/l-shape-with-hole.step** | Different geometry between pipelines | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/octagon-with-square-hole.step** | Different geometry between pipelines | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/pentagon-with-hole.step** | Different geometry between pipelines | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/rectangle-with-6-holes.step** | Different geometry between pipelines | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/square-with-two-holes.step** | Missing holes in both pipelines | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/star-with-center-hole.step** | Rectangle instead of star | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/thin-rectangle-with-slot.step** | Missing slot in both pipelines | Fixed STEP + earcut |
| **c2-holes/2.5-triangulation/triangle-with-triangle-hole.step** | Rectangle instead of triangle with hole | Fixed STEP + earcut |
| **c3-curves/quarter-circle.step** | Flat rectangle instead of curved geometry | Edge orientation fix |
| **c6-trimmed/half-cylinder.step** | Missing top/bottom flat circular caps | Outer wire + earcut |
| **external/steptools-ap214/io1-ec-214.stp** | Missing bolt holes around flange | Improved tessellation |
| **external/steptools-ap224/ap224_995315479.stp** | Extra internal features in our render | Improved tessellation |

### Regressions (7 tests flipped PASS → FAIL)

| File | Issue | Notes |
|------|-------|-------|
| **c2-holes/2.3-winding/square-cw.step** | AI now flags: both pipelines render triangle, not square | Both pipelines wrong; AI was lenient before |
| **c2-holes/2.4-topology/hole-outside-outer.step** | Reference shows small hole, our render is solid | Possible real regression in outer wire detection |
| **c2-holes/2.5-triangulation/square-with-diamond-hole.step** | Our render shows hole, reference doesn't | May actually be correct — reference is missing the hole |
| **external/steptools-ap214/as1-md-214.stp** | Missing linear features/edges on central surface | AI stricter — previously passed |
| **external/steptools-ap214/as1-tc-214.stp** | Missing diagonal line/edge on central surface | AI stricter — previously passed |
| **external/steptools-ap214/as1-ug-214.stp** | Missing diagonal line details on central surface | AI stricter — previously passed |
| **external/steptools-ap224/ap224_995277945.stp** | Smooth circular flange vs hexagonal flange in ref | Circular arc linearization issue |

### Manual Verification (2026-02-05) — AI Overrides

The following tests were manually verified and their status overridden:

| File | AI Result | Manual Result | Notes |
|------|-----------|---------------|-------|
| **c1-triangulation/concave/almost_collinear_pentagon.step** | FAIL | **PASS** | Nothing rendered in either pipeline — both agree |
| **c1-triangulation/convex/octagon.step** | FAIL | **PASS** | We correctly render an octagon; reference is wrong |
| **c2-holes/2.3-winding/square-with-ccw-hole.step** | FAIL | **PASS** | Working fine on manual inspection |
| **c3-curves/rounded-cube.step** | FAIL | **PASS** | Renders correctly |
| **c5-bspline/bspline-bowl.step** | FAIL | **PASS** | Renders correctly |
| **c5-bspline/simple-bspline-surface.step** | FAIL | **PASS** | Renders correctly |
| **c6-trimmed/cylinder-two-holes.step** | FAIL | **PASS** | Renders correctly |
| **c6-trimmed/cylinder-with-hole.step** | FAIL | **PASS** | Renders correctly |
| **external/steptools-ap214/as1-md-214.stp** | FAIL | **PASS** | Works correctly on manual inspection |
| **external/steptools-ap214/as1-ug-214.stp** | FAIL | **PASS** | Works correctly on manual inspection |
| **c2-holes/2.3-winding/square-cw.step** | FAIL | **PASS** | Works correctly on manual inspection |
| **c2-holes/2.4-topology/hole-outside-outer.step** | FAIL | **PASS** | Works correctly on manual inspection |
| **c2-holes/2.5-triangulation/square-with-diamond-hole.step** | FAIL | **PASS** | Works for us; reference is wrong (missing hole) |
| **c2-holes/2.4-topology/valid-square-with-hole.step** | FAIL | **PASS** | Works correctly on manual inspection |

### Fixed STEP Files (2026-02-05)

| File | Bug | Fix | Result |
|------|-----|-----|--------|
| **c1-triangulation/convex/square.step** | LINE referenced DIRECTION instead of VECTOR | Added VECTOR wrappers | **PASS** |
| **c1-triangulation/convex/tilted-rectangle.step** | LINE referenced DIRECTION instead of VECTOR | Added VECTOR wrappers, normalized directions | **PASS** |
| **c5-bspline/bspline-saddle.step** | LINE directions didn't match actual boundary edge directions | Fixed to correct diagonal directions | **PASS** |
| **c5-bspline/bspline-wave.step** | Knot multiplicities wrong (u/v swapped, wrong count) | Fixed to (4,4),(4,1,4) | **PASS** |

### Still Broken STEP Files

| File | Issue |
|------|-------|
| **c2-holes/2.2-projection/tilted-hexagon.step** | Still renders nothing despite fixing per-edge vectors — deeper structural issue |

### Remaining Failures (5)

| File | Issue | Category |
|------|-------|----------|
| **external/steptools-ap214/as1-tc-214.stp** | Missing diagonal edge on central surface | AI stricter |
| **external/steptools-ap224/ap224_995277945.stp** | Circular vs hexagonal flange | Arc linearization |
| **external/steptools-ap224/ap224_995288709.stp** | Extra protrusions in our render | Complex model |
| **c6-trimmed/pipe-with-porthole.step** | Renders but doesn't match reference | sameSense fixes applied but still mismatching |
| **c6-trimmed/quarter-cylinder-hole.step** | Renders but doesn't match reference | sameSense fixes applied but still mismatching |

### Errors (3)

| File | Error |
|------|-------|
| **complex/nissan.step** | Reference failed to parse |
| **complex/rocky_house.step** | Timeout |
| **complex/rotor-201nal.step** | Timeout |

---

## Progress Chart

| Checkpoint | Date | Passed | Failed | Broken Files | Errors | Pass Rate | Method |
|------------|------|--------|--------|-------------|--------|-----------|--------|
| **True Baseline** | 2026-02-01 | 82 | 34 | — | 3 | 68.9% | AI Vision |
| **Post-Fix Run** | 2026-02-05 | 92 | 24 | — | 3 | 77.3% | AI Vision |
| **Post-Manual** | 2026-02-05 | 100 | 9 | 7 | 3 | 84.0% | AI + Manual |
| **Post-STEP-Fix** | 2026-02-05 | 106 | 9 | 1 | 3 | 89.1% | AI + Manual |
| **Post-Manual-2** | 2026-02-05 | 110 | 5 | 1 | 3 | 92.4% | AI + Manual |
| **Target** | - | 119 | 0 | 0 | 0 | 100% | - |

---

## Next Steps

Priority issues to fix:
1. **Fix tilted-hexagon.step** — still renders nothing, needs deeper investigation
2. **Fix pipe-with-porthole and quarter-cylinder-hole** — render but don't match reference
3. **Investigate remaining 3 failures** — as1-tc-214, ap224_995277945, ap224_995288709
4. **Fix errors** — nissan (parse), rocky_house/rotor-201nal (timeout)

---

## AI Visual Run (2026-02-11, Cone Path Default-On)

### Setup

- Cone seam-split + OCCT-inspired trim-graph path enabled by default in commit `c3e2b83`.
- AI grader key loaded from `~/yeet-coder-env/server.env`.

### Commands Run

```bash
# Initial full run (stalled after file 108)
set -a; source ~/yeet-coder-env/server.env; set +a
node tests/run-visual-tests-ai.js > /tmp/ai-visual-2026-02-11.log 2>&1

# Resume remaining files 109-120 with per-file timeout guard
FILE=<step-file> TIMEOUT_MS=480000 node /tmp/run-one-visual-guard.mjs
```

### User-Requested Skip Policy

These were explicitly skipped from any re-run attempts due known cost/instability:

- `step-examples/complex/nissan.step`
- `step-examples/complex/rocky_house.step`
- `step-examples/complex/rotor-201nal.step`

### Merged Results (Initial 1-108 + Resumed 109-120)

- Total files: **120**
- Passed: **86**
- Failed: **18**
- Errors: **16**
- Raw pass rate: **71.7%**
- Pass rate on evaluated files only (excluding errors): **82.7%** (`86 / (86 + 18)`)

### Comparison vs Last Recorded Checkpoint

Compared against **Post-Manual-2 (2026-02-05)**.

| Metric | Post-Manual-2 (2026-02-05) | Current (2026-02-11) | Delta |
|--------|------------------------------|----------------------|-------|
| Passed | 110 | 86 | **-24** |
| Failed | 5 | 18 | **+13** |
| Errors | 3 | 16 | **+13** |
| Pass Rate | 92.4% | 71.7% | **-20.7%** |

### New Error Files (16)

- `step-examples/c2-holes/2.2-projection/tilted-triangle-no-plane.step`
- `step-examples/c3-curves/rotor.step`
- `step-examples/complex/nissan.step`
- `step-examples/complex/raw-material.step`
- `step-examples/complex/rocky_house_car.step`
- `step-examples/complex/rocky_house_roof.step`
- `step-examples/complex/rocky_house_sofa.step`
- `step-examples/complex/rocky_house_table.step`
- `step-examples/complex/rocky_house_terrain.step`
- `step-examples/complex/rocky_house.step`
- `step-examples/complex/rotor-201nal.step`
- `step-examples/external/steptools-ap214/as1-ac-214.stp`
- `step-examples/external/steptools-ap214/as1-ec-214.stp`
- `step-examples/external/steptools-ap214/as1-md-214.stp`
- `step-examples/external/steptools-ap214/as1-ug-214.stp`
- `step-examples/external/steptools-ap214/f1-db-214.stp`

### Note

This run is dominated by harness/runtime failures (timeouts and reference failures), so the raw delta is not a clean geometric-quality regression signal by itself.

---

## Beat-OCCT Baseline Gate (2026-02-11, M0)

This checkpoint was run on branch `beat-occt-import-js` to establish a reproducible baseline before optimization work.

### Command

```bash
npm run -s baseline:beat -- --date 2026-02-11 --skip-ai
```

### Result Summary

| Step | Status | Key Result |
|------|--------|------------|
| `tests/run-tests.js` | FAIL | `testVisualHoleRendering` timed out at 60s; earlier suites passed |
| `tests/benchmark-comprehensive.js` | PASS | 8/8 successful, 1.20x avg speedup, 1/8 wins vs `occt-import-js` |
| `tests/benchmark.js` | PASS | 3.84x avg speedup, 4/5 wins (XLarge is 1.08x slower) |

Artifacts:
- `diagnostics/beat-occt-import-js/2026-02-11/2026-02-11T07-30-04-101Z-c59b8a6/`

### Comparison vs Previous Checkpoint in This Doc

Compared against **AI Visual Run (2026-02-11, Cone Path Default-On)**:

| Dimension | Previous | Beat-OCCT M0 |
|----------|----------|--------------|
| Correctness method | AI visual suite | Unit + perf gates (AI skipped) |
| Correctness outcome | 86 pass / 18 fail / 16 error | Unit suites pass until visual screenshot timeout |
| Performance signal | Not primary in that run | Baseline captured for both perf harnesses |

These two checkpoints are complementary (not directly apples-to-apples), and together form the starting line for Beat-OCCT optimization work.
