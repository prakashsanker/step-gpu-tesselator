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

## Progress Chart

| Checkpoint | Date | Passed | Failed | Errors | Pass Rate | Method |
|------------|------|--------|--------|--------|-----------|--------|
| **True Baseline** | 2026-02-01 | 82 | 34 | 3 | 68.9% | AI Vision |
| **Target** | - | 119 | 0 | 0 | 100% | - |

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

## Next Steps

Priority issues to fix:
1. **Re-run AI visual tests** - Verify c2-holes pass rate after edge orientation fixes
2. **Cylinder trimming (5 failures)** - Inner hole bounds on curved surfaces
3. **BSpline surfaces (2 real failures)** - Different geometry/orientation
4. **Curved edges (2 failures)** - quarter-circle, rounded-cube
