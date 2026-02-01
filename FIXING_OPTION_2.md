# Fixing Option 2: OpenCascade.js + Custom CPU Tessellator

This document tracks our progress fixing Option 2 to achieve correct tessellation across all test files.

## Overview

**Option 2 Pipeline:**
- Parsing: OpenCascade.js (WASM)
- Tessellation: Custom CPU-based (ear-clipping for planar, grid-based for curved)
- Rendering: Three.js

**Goal:** Get Option 2 working correctly with ALL 119 test files before GPU optimization.

---

## Baseline Status (2026-02-01)

**Visual Test Results:**
- **Passed:** 70 (58.8%)
- **Failed:** 48
- **Errors:** 1
- **Total:** 119 files

---

## Issues Identified

### Priority 1: Curved Surface Tessellation (Critical)

| Issue | File | Our Tris | Ref Tris | Diff | Status |
|-------|------|----------|----------|------|--------|
| Cone 93% missing | cone.step | 2366 | 1841 | +29% | ✅ FIXED |
| Sphere 2x over | sphere.step | 6212 | 3004 | +107% | ✅ PASSES |
| Torus 29% missing | torus.step | 4096 | 5760 | -29% | ✅ PASSES |
| Cylinder 11% extra | cylinder.step | 312 | 280 | +11% | ✅ PASSES |

### Priority 2: Complex Files Missing Geometry

| Issue | Files | Our Tris | Ref Tris | Diff | Status |
|-------|-------|----------|----------|------|--------|
| Missing curved geometry | nissan, rocky_house, cube, etc. | 630 | 1318 | -52% | TODO |

### Priority 3: Multi-Face Coordinate Issues

| Issue | File | Visual Diff | Status |
|-------|------|-------------|--------|
| Position error | unit-box.step | 16.7% | TODO |
| Position error | triangular-prism.step | 13.4% | TODO |
| Position error | two-triangles.step | 13.4% | TODO |
| Position error | pyramid.step | 11.5% | TODO |

---

## Fix Log

### Fix #0: Baseline Established
**Date:** 2026-02-01
**Changes:**
- Created visual comparison test suite (`tests/run-visual-tests.js`)
- Uses existing `visual-validation.html` with pixelmatch comparison
- Tests all 119 STEP files automatically

**Test Results:**
- Passed: 70
- Failed: 48
- Errors: 1
- Pass Rate: 58.8%

---

### Fix #1: Cone Tessellation + Color Matching
**Date:** 2026-02-01
**Problem:** Cone produces only 130 triangles vs 1841 expected (93% missing)

**Investigation:**
- [x] Check UV boundary extraction for conical surfaces
- [x] Check point-in-polygon test in `tessellateTrimmedSurface`
- [x] Check grid density calculation
- [x] Check `evaluateSurface` for cone surface type

**Root Cause:**
1. **Degenerate UV boundary for full cones**: The UV boundary for a full 360° cone forms a "lollipop" shape - the apex is a single point at v=0 connected to the circular base at v=height. The point-in-polygon test fails because the polygon doesn't properly enclose interior points.
2. **Insufficient height samples**: The fallback `tessellateCone` was using only 2 height samples (default), resulting in very few triangles.
3. **Color mismatch**: Visual validation used different colors for our pipeline (blue `0x6699ff`) vs reference (orange `0xff9966`), causing pixel comparison to fail even when geometry was correct.

**Changes Made:**
1. **Added cone center check** (`occ-test.ts:4388-4396`): Detect when cone UV boundary doesn't enclose the center point and fall back to rectangular tessellation
2. **Added sphere center check** (`occ-test.ts:4398-4406`): Same fix for spheres
3. **Increased cone height samples** (`occ-test.ts:4473-4475`): Calculate height samples proportional to height range (`heightRange * 8`)
4. **Fixed color matching** (`visual-validation.html:398`): Changed reference color to match our pipeline for accurate pixel comparison

**Benchmark - Curved Surface Results:**
| Surface | Visual Diff | Status |
|---------|-------------|--------|
| cone.step | 2.6% | ✅ PASS |
| cylinder.step | 2.3% | ✅ PASS |
| sphere.step | 0.0% | ✅ PASS |
| torus.step | 0.0% | ✅ PASS |

**Test Results After Fix:**
- Passed: 92
- Failed: 24
- Errors: 3
- Pass Rate: 77.3% (+18.5pp improvement)

**Summary:**
The core issue was that full 360° cones and spheres have degenerate UV boundaries - the UV polygon forms a "lollipop" shape where the apex is a single point connected to the circular base. The point-in-polygon test fails for interior points because the polygon doesn't properly enclose them. The fix detects this by checking if the UV center point is inside the boundary; if not, it falls back to rectangular tessellation that covers the full UV range. Additionally, cone height sampling was increased from 2 samples to `heightRange * 8` for proper tessellation density. The color matching fix ensures visual tests compare identical geometries without false failures due to different render colors.

---

### Fix #1b: Cone V-Parameter Interpretation
**Date:** 2026-02-01
**Problem:** Cone base cap appeared "sunken" below the top rim of the conical surface

**Root Cause:**
Our `evaluateCone` function treated V as **height along axis**, but OpenCascade uses V as **distance along the cone generator (slant line)**.

For a cone with height h=2 and base radius r=1:
- Our interpretation: vMax = 2.0 (height)
- OpenCascade's interpretation: vMax = √(h² + r²) = √5 ≈ 2.236 (slant distance)

**Fix (`surfaces.ts:evaluateCone`):**
```javascript
// OLD (wrong): V as height
localRadius = radius + v * tan(semiAngle)
z = v

// NEW (correct): V as slant distance
localRadius = radius + v * sin(semiAngle)
z = v * cos(semiAngle)
```

**Result:** Cone visual diff improved from 2.6% to **0.0%**

---

### Fix #2: Sphere/Torus - RESOLVED BY FIX #1
**Date:** 2026-02-01
**Problem:** Sphere produces 2x triangles, Torus 29% missing

**Status:** ✅ Both now pass visual tests (0.0% visual diff) after color matching fix.
- Sphere: 6212 vs 3004 tris (2x more) - visually correct
- Torus: 4096 vs 5760 tris (29% less) - visually correct

The triangle count differences don't matter if visual output is correct.

---

### Fix #4: [TODO] Complex Files Missing Geometry
**Date:** TBD
**Problem:** Complex files (nissan, rocky_house, etc.) missing 52% of triangles

**Investigation:**
- [ ] Check which face types are being skipped
- [ ] Check for exceptions in `tessellateCurvedFaceFromOCC`
- [ ] Check BSpline surface handling

**Root Cause:** TBD

**Changes Made:**
- TBD

**Test Results After Fix:**
- Passed: TBD
- Failed: TBD
- Pass Rate: TBD

---

### Fix #5: [TODO] Multi-Face Coordinate Transforms
**Date:** TBD
**Problem:** Multi-face models have correct triangles but wrong positions (10-17% visual diff)

**Investigation:**
- [ ] Check coordinate system between our pipeline and reference
- [ ] Check Y/Z axis handling
- [ ] Check mesh assembly in `tessellateOCCShape`

**Root Cause:** TBD

**Changes Made:**
- TBD

**Test Results After Fix:**
- Passed: TBD
- Failed: TBD
- Pass Rate: TBD

---

## Key Files

| File | Purpose |
|------|---------|
| `src/occ-test.ts` | Main tessellation pipeline, `tessellateCurvedFaceFromOCC` |
| `src/surface-tessellation.ts` | `tessellateTrimmedSurface`, grid-based curved surface tessellation |
| `src/surfaces.ts` | `evaluateSurface`, surface point evaluation |
| `tests/run-visual-tests.js` | Automated visual comparison test runner |
| `tests/visual-validation.html` | Manual visual comparison page |
| `tests/visual-results/visual-test-report.json` | Latest test results |

---

## Commands

```bash
# Run visual tests
npm run test:visual

# Run visual tests and save screenshots for failures
npm run test:visual:save

# Run visual tests for specific pattern
node tests/run-visual-tests.js cone

# Start dev server for manual testing
yarn dev
# Then open: http://localhost:5173/tests/visual-validation.html
```

---

## Progress Chart

| Checkpoint | Date | Passed | Failed | Errors | Pass Rate |
|------------|------|--------|--------|--------|-----------|
| Baseline | 2026-02-01 | 70 | 48 | 1 | 58.8% |
| Fix #1 (Cone/Color) | 2026-02-01 | 92 | 24 | 3 | 77.3% |
| Fix #1b (Cone V-param) | 2026-02-01 | 92 | 24 | 3 | 77.3% |
| Fix #4 | TBD | TBD | TBD | TBD | TBD |
| **Target** | - | **119** | **0** | **0** | **100%** |

**Note:** Fix #1b improved cone visual quality (2.6% → 0.0%) but didn't change pass count since cone was already passing.

---

## Remaining Issues (24 failures, 3 errors)

### BSpline Failures (2)
- `bspline-saddle.step` - Reference XCAF transfer fails (0 triangles)
- `bspline-wave.step` - Reference XCAF transfer fails (0 triangles)

### External AP214 Files (13)
Various steptools AP214 files with 6.8-17.7% visual diff - likely color or position issues.

### c8-solids and complex (7)
Files showing identical triangle counts (7.1-8.0% diff) suggest rendering or camera issues, not tessellation.

### Timeouts (3 errors)
- `nissan.step` - Reference failed
- `rocky_house.step` - Timeout
- `rotor-201nal.step` - Timeout
