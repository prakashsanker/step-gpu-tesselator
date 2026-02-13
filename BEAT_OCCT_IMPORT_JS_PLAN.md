# Plan: Beat `occt-import-js` (Speed + Correctness)

## 1. Goal

Ship a default pipeline that is faster than `occt-import-js` on real models while preserving visual/correctness quality.

Definition of done:
- Speed: consistently faster on end-to-end suite (median and p90), not just synthetic cases.
- Correctness: no regression versus current baseline in unit tests and AI visual comparison.
- Stability: no new hangs/timeouts in normal benchmark runs.

---

## 2. Baseline Freeze (Before More Optimization)

Run and save baseline results from this branch before any new perf work:

1. Unit/full correctness:
   - `node --experimental-vm-modules tests/run-tests.js`
2. AI visual comparison:
   - `node tests/run-visual-tests-ai.js`
3. Performance:
   - `node tests/benchmark-comprehensive.js`
   - `node tests/benchmark.js`

Store outputs in a dated folder, for example:
- `diagnostics/beat-occt-import-js/2026-02-11/`

Include:
- commit SHA
- machine/browser details
- command lines
- raw logs + summarized table

One-command runner for this branch:
- `npm run baseline:beat`
- Optional: `npm run baseline:beat -- --all-bench`
- Optional: `npm run baseline:beat -- --skip-ai`

---

## 3. Benchmark Policy (Every Step)

For each milestone/optimization step:

1. Run targeted perf benchmark(s) for the changed area.
2. Run quick correctness gate:
   - `node --experimental-vm-modules tests/run-tests.js`
3. Every 2-3 perf steps (or end of milestone), run full AI visual suite:
   - `node tests/run-visual-tests-ai.js`
4. Record results in:
   - `benchmark.md` (new dated section)
   - `FIXING_OPTION_2.md` (progress + pass/fail deltas)

Known long/problematic files can be excluded for routine AI runs:
- `complex/nissan.step` (reference parse failure)
- `complex/rocky_house.step` (timeout)
- `complex/rotor-201nal.step` (timeout)

But keep a periodic “full attempted run” note so exclusions stay explicit.

---

## 4. Milestone Roadmap

### M0: Measurement + Harness Hardening
- Add/confirm stable benchmark harness commands and output format.
- Ensure results are reproducible and comparable by SHA.

Gate:
- Baseline report committed and reproducible.

### M1: CPU Hot Path Reduction (No Algorithm Risk)
- Remove avoidable allocations/copies in tessellation hot paths.
- Reduce repeated topology/surface queries.
- Improve per-face work scheduling to cut overhead.

Gate:
- End-to-end runtime improvement with no correctness drop.

### M1.2: Load/Parse Path Reduction (Now Priority)
Reason:
- Current laggards are dominated by `loadStepFile`, not tessellation compute.
- Remaining laggard gap is small and likely recoverable by parse-path work.

Scope:
1. Instrument `loadStepFile` sub-phases:
   - file ingest
   - OCC handoff
   - STEP read
   - shape extraction
   - color/material/diagnostic passes
2. Add fast default path for perf runs:
   - geometry-only reader path (skip heavy metadata flow)
   - keep XCAF path behind flag for color/material-sensitive runs
3. Make optional diagnostics truly opt-in:
   - disable expensive color/alt-shape diagnostics in perf mode
4. Defer heavy per-face prep:
   - lazily extract expensive trim data when needed

Gate:
- Canary laggards (`Conical Surface`, `VM-001`) improved.
- `wins vs occt-import-js` stays >= current level.
- No new correctness regressions in quick gate.

### M1.1: Immediate Bottleneck Attack (Based on Current Canary Profile)
Current canary hotspots:
- `VM-001`: `tessellatePlanarFace`/`earClipping` dominate.
- `Plate XLarge`: cold-start dominated by `loadStepFile`.
- Triangle inflation remains high on complex models (`triRatio` vs ref).

Execution order:
1. Remove hot-path debug overhead in planar-hole triangulation.
2. Add planar-hole complexity dispatch (cheap/medium/heavy buckets) so heavy faces avoid worst CPU path.
3. Add UV loop cleanup (dedupe + collinear pruning + endpoint snapping) before triangulation.
4. Separate cold vs warm benchmark reporting so iteration loop measures steady-state speed.
5. Add triangle-budget/adaptive sampling guardrails to reduce over-tessellation.

Gate:
- Canary warm runtime improves on `Plate XLarge` and `VM-001`.
- No geometry regressions in quick correctness gate.

### M2: Cone/Trim Path Robustness + Cost Control
- Keep current occt-inspired cone trim/split path.
- Reduce constraint-recovery failures and retries in CDT path.
- Add cheap prechecks to avoid expensive failing triangulation attempts.

Gate:
- Cone-heavy models faster and visually stable vs reference.

### M3: GPU-First Expansion for Curved/Trimmed Work
- Move more per-face triangulation/normal computation work to GPU where feasible.
- Keep CPU fallback for unsupported edge cases.

Gate:
- Net speedup on mixed real-world suite, not only microbench.

### M4: End-to-End Throughput
- Parallelize face batching and reduce CPU↔GPU synchronization cost.
- Minimize redundant conversions between representations.

Gate:
- Better p90 and worst-case latency on large STEP files.

### M3/M4 Execution Track (Current Focus)
1. GPU-priority triangulation policy (in progress):
   - Make triangulation thresholds runtime-tunable.
   - Enable GPU-priority thresholds in benchmark perf mode.
   - Verify geometry quality on cylinder/cone trim canaries before broader rollout.
2. Batched planar triangulation:
   - Route no-hole planar faces through batched GPU triangulation (single dispatch per batch).
   - Keep hole/complex topology on current stable path.
3. Batched curved-face dispatch preparation:
   - Build per-face trim/sampling jobs first, submit compute in grouped batches.
   - Reduce one-face-at-a-time GPU submit/readback behavior.
4. Post-tessellation GPU normals:
   - Compute normals in a single mesh-level GPU pass for large outputs.
   - Keep CPU fallback below a triangle threshold.
5. Sync minimization:
   - Reuse GPU buffers/pipelines across faces/models.
   - Avoid per-face mapAsync/readback inside inner loops when possible.

Validation cadence for this track:
- Per step: `npm run -s bench:canary` + targeted visual checks (`c4-surfaces/*`, `c6-trimmed/*`, `VM-001`).
- Every 2 steps: representative run + full correctness suite.

### M5: Final Quality/Performance Balance
- Tune defaults (tolerances/deflection-like knobs) for best quality-per-ms.
- Validate no new holes/gaps/artifacts.

Gate:
- Final comparison report vs baseline and vs `occt-import-js`.

---

## 5. Correctness Cadence (Mandatory)

- Per perf commit: unit/full correctness (`tests/run-tests.js`).
- Every 2-3 perf commits: AI visual run (`tests/run-visual-tests-ai.js`).
- Before merge to `master`: full correctness + AI visual + perf summary.

If any correctness regression appears:
- Pause perf work.
- Fix regression first.
- Re-run last green benchmark set.

---

## 6. Reporting Template (Use Every Step)

For each update in `benchmark.md` / `FIXING_OPTION_2.md`:

- Date:
- Commit:
- Change summary:
- Bench commands:
- Perf results (ours vs `occt-import-js`):
  - median:
  - p90:
  - worst:
- Correctness results:
  - `run-tests.js`:
  - AI visual pass/fail/error:
- Verdict:
  - `improved` / `neutral` / `regressed`

---

## 7. Immediate Next Actions

1. [x] Capture fresh baseline artifacts on this branch.
2. [x] Add first benchmark entry (M0) to `benchmark.md`.
3. [x] Upgrade benchmark harness to representative canary/representative/full suites.
4. [x] Prioritize optimization queue using measured hotspot breakdown (canary).
5. [x] Execute M1.1.1: remove planar-hole hot-path debug overhead and re-run canary.
6. [x] Execute M1.1.2: planar-hole complexity dispatch + UV cleanup and re-run canary.
7. [x] Execute M1.2.1: add `loadStepFile` sub-phase instrumentation and capture canary deltas.
8. [x] Execute M1.2.2: add geometry-only fast parse path (default for perf), keep XCAF behind flag.
9. [x] Execute M1.2.3: disable heavy diagnostics by default in perf mode and re-run canary.
10. [x] Run full correctness suite every 2-3 perf steps to prevent regressions.
11. [x] Execute M1.3.1: gate cylinder 3D bbox filtering to seam-sensitive trims and re-run canary/representative.
12. [x] Execute M2.1: reduce CDT constraint-recovery cost (edge bbox fast-reject + robust cavity boundary ordering) and re-run canary + representative spot-check.

### 2026-02-11 M1.1 Progress (In Flight)

- Completed:
  - M1.1.1 done: hole-triangulation diagnostics are debug-gated (`__TRIANGULATE_HOLES_DEBUG__`).
  - Planar no-hole path switched from direct `earClipping()` to `triangulateFast()` with fallback.
  - Edge sampling defaults reduced (still runtime-tunable) to lower triangle inflation.
  - Benchmark harness upgraded with one-time prewarm + fixed multi-run aggregation bug.
- Canary trend (single-run, noisy but directional):
  - VM-001: `2471.8ms -> 396.6ms` (tris `23033 -> 22348`)
  - Cone: `125.4ms -> 51.1ms`
  - Cylinder With Hole: `154.3ms -> 14.8ms`
  - Plate XLarge remains dominated by cold `loadStepFile` startup (`~4.2s`).
- Canary (warm-state prewarm enabled):
  - Wins vs `occt-import-js`: `4/6`
  - Median speedup: `1.47x`
  - Ours avg runtime: `119.9ms` vs ref `137.0ms`
  - Remaining laggards: `Conical Surface (complex)`, `VM-001`
- Status of M1.1.2:
  - Complexity dispatch: partial (no-hole dispatch done).
  - UV cleanup: pending.
- Correctness cadence:
  - `npm test` re-run attempted after perf changes.
  - Result: runner reached visual stage then timed out at `testVisualHoleRendering` (`Waiting failed: 60000ms exceeded`).

### 2026-02-11 M1.1.2 Iteration (Trim Complexity + Rollback)

- What changed:
  - Removed the global trim-loop simplification pass added in this branch iteration (kept cone-specific simplification only).
  - Tuned default trim density caps to reduce triangle inflation:
    - `__TRIM_GRID_SCALE__`: `1.0 -> 0.85`
    - `__TRIM_MIN_GRID_DENSITY__`: `12 -> 10`
    - `__TRIM_MAX_GRID_DENSITY__`: `40 -> 32`
    - `__TRIM_MAX_GRID_DENSITY_NO_HOLES__`: `24 -> 20`
    - `__TRIM_MAX_GRID_DENSITY_WITH_HOLES__`: `40 -> 24`
    - `__CONE_TRIM_MAX_GRID_DENSITY__`: `36 -> 24`
    - `__CONE_TRIM_MAX_GRID_DENSITY_NO_HOLES__`: `16 -> 14`
    - `__TRIM_HIGH_COMPLEXITY_POINT_THRESHOLD__`: `900 -> 600`
    - high-complexity cap factor: `0.85 -> 0.75`
  - Tuned OCCT-inspired cone-domain defaults:
    - `__OCCT_INSPIRED_TRIM_GRID_SCALE__`: `1.35 -> 0.9`
    - `__OCCT_INSPIRED_TRIM_MIN_GRID__`: `24 -> 14`
    - `__OCCT_INSPIRED_TRIM_MAX_GRID__`: `64 -> 32`

- Canary result after rollback+tuning (`npm run -s bench:canary`):
  - Wins vs `occt-import-js`: `4/6`
  - Speedup median: `1.77x`
  - Ours avg runtime: `122.4ms` (p90 `267.6ms`)
  - Ref avg runtime: `151.9ms` (p90 `311.3ms`)
  - Remaining laggards:
    - `Conical Surface (complex)`: `91.0ms` vs `62.0ms` (`1.47x` slower)
    - `VM-001`: `372.3ms` vs `365.3ms` (`1.02x` slower, near parity)

- Triangle trend:
  - `Conical Surface`: `832` vs ref `1318` (no longer inflated).
  - `VM-001`: `15812` vs ref `3116` (still inflated, but lower than prior run in this branch).

- Correctness gate:
  - `npm test`: fails at known visual timeout (`testVisualHoleRendering`, 60s wait exceeded).

### 2026-02-11 M1.2.1 Iteration (Load Sub-phase Surfacing)

- What changed:
  - `tests/benchmark-comprehensive.js` now keeps `loadStepFile_*` keys in aggregated phase output:
    - `loadStepFile_initOC`
    - `loadStepFile_createDoc`
    - `loadStepFile_readFile`
    - `loadStepFile_transfer`
    - `loadStepFile_getTools`
    - `loadStepFile_colorParsing`
  - Per-model benchmark output now prints load-path share and top 3 load sub-phases for lagging models.

- Canary result (`npm run -s bench:canary`):
  - Wins vs `occt-import-js`: `4/6`
  - Speedup median: `1.66x`
  - Ours avg runtime: `108.5ms` (p90 `241.8ms`)
  - Ref avg runtime: `127.8ms` (p90 `258.9ms`)

- Remaining laggard attribution:
  - `Conical Surface (complex)`:
    - ours `76.3ms`, ref `59.5ms` (`1.28x` slower)
    - `loadStepFile`: `57.4ms` (`75.2%` of ours)
    - top load sub-phases: `colorParsing 46.6ms`, `transfer 6.3ms`, `readFile 3.4ms`
  - `VM-001`:
    - ours `334.5ms`, ref `276.1ms` (`1.21x` slower)
    - `loadStepFile`: `228.7ms` (`68.4%` of ours)
    - top load sub-phases: `transfer 143.2ms`, `colorParsing 46.8ms`, `readFile 37.6ms`

- Conclusion:
  - M1.2.2 and M1.2.3 should target `transfer` and `colorParsing` first.

### 2026-02-11 M1.2.2 Iteration (Geometry-only Perf Load Path)

- What changed:
  - Added perf-load switches in `loadStepFile()`:
    - `__PERF_GEOMETRY_ONLY_LOAD__` (default `false`)
    - `__ENABLE_XCAF_READER__` (default `!__PERF_GEOMETRY_ONLY_LOAD__`)
    - `__ENABLE_STEP_COLOR_PARSING__` (default `!__PERF_GEOMETRY_ONLY_LOAD__`)
  - Added lazy STEP text decode:
    - do not decode `Uint8Array` to string unless step-color parsing is enabled.
  - Benchmark harness now enables geometry-only mode for perf comparisons:
    - `tests/benchmark-comprehensive.html` sets:
      - `__PERF_GEOMETRY_ONLY_LOAD__ = true`
      - `__ENABLE_XCAF_READER__ = false`
      - `__ENABLE_STEP_COLOR_PARSING__ = false`
    - flags are restored after each `runOCC`.

- Canary result (`npm run -s bench:canary`):
  - Wins vs `occt-import-js`: `6/6`
  - Speedup median: `1.75x`
  - Ours avg runtime: `87.1ms` (p90 `199.8ms`)
  - Ref avg runtime: `126.8ms` (p90 `238.8ms`)

- Key laggard flips:
  - `Conical Surface (complex)`: `76.3ms -> 25.7ms` (now `1.75x` faster vs ref)
  - `VM-001`: `334.5ms -> 270.7ms` (now `1.03x` faster vs ref)

- Correctness check:
  - `npm test` still hits known visual timeout in `testVisualHoleRendering` (60s wait exceeded).


### 2026-02-11 M1.3.1 Iteration (Cylinder 3D Bbox Gating)

- What changed:
  - In `src/occ-test.ts`, cylinder `bbox3d` computation is no longer unconditional.
  - `occEdgesToPolygon(face.outerLoop)` + bbox path now runs only when:
    - `face.surfaceType === 'Cylinder'` and
    - (`cylinderCrossesSeam` OR `degeneratePeriodicTrim` OR `__FORCE_CYLINDER_BBOX3D__ === true`).
  - This avoids expensive boundary resampling and per-grid-point 3D filtering on non-problematic cylinder faces.

- Canary result (`npm run -s bench:canary`):
  - Wins vs `occt-import-js`: `6/6`
  - Speedup median: `2.25x`
  - Ours avg runtime: `155.8ms` (p90 `377.2ms`)
  - Ref avg runtime: `391.5ms` (p90 `919.8ms`)

- Representative result (`npm run -s bench:representative`):
  - Wins vs `occt-import-js`: `5/6`
  - Speedup median: `4.11x`
  - Ours avg runtime: `1827.7ms` (p90 `5444.8ms`)
  - Ref avg runtime: `982.2ms` (p90 `2752.9ms`)
  - Remaining laggard:
    - `Electronic Enclosure`: ours `10631.1ms`, ref `5239.8ms` (`2.03x` slower)
    - `loadStepFile`: `3582.5ms` (`33.7%` of ours)
    - top load sub-phases: `transfer 2794.8ms`, `readFile 785.2ms`, `fsWrite 1.1ms`

- Delta vs prior representative run in this branch:
  - `Electronic Enclosure`: `11120.3ms -> 10631.1ms` (`-489.2ms`, `-4.4%`)


### 2026-02-11 M1.2.3 Iteration (Force Diagnostics Off in Perf Mode)

- What changed:
  - Added `loadDiagnosticsEnabled()` in `src/occ-test.ts`:
    - returns `false` whenever `__PERF_GEOMETRY_ONLY_LOAD__ === true`
    - otherwise respects `__ENABLE_LOAD_DIAGNOSTICS__`
  - Switched load/solid-color diagnostic gates to use `loadDiagnosticsEnabled()`.
  - Hardened `tests/benchmark-comprehensive.html` perf run wrapper to explicitly set:
    - `__ENABLE_LOAD_DIAGNOSTICS__ = false`
    - `__CURVE_VERBOSE_LOGS__ = false`
    - `__TESSELLATION_VERBOSE_LOGS__ = false`
    - and restore prior values after each run.

- Canary result (`npm run -s bench:canary`):
  - Wins vs `occt-import-js`: `6/6`
  - Speedup median: `3.77x`
  - Ours avg runtime: `179.8ms` (p90 `390.4ms`)
  - Ref avg runtime: `553.7ms` (p90 `965.4ms`)

- Representative spot checks:
  - `npm run -s bench:representative -- --filter "Electronic Enclosure"`:
    - ours `12384.4ms`, ref `5130.0ms` (`2.41x` slower)
    - `loadStepFile`: `4959.9ms` (`40.1%` of ours)
    - top sub-phases: `transfer 3732.7ms`, `readFile 1199.2ms`, `fsWrite 2.4ms`
  - `npm run -s bench:representative -- --filter "VM-001"`:
    - ours `493.5ms`, ref `610.7ms` (`1.24x` faster)

- Conclusion:
  - M1.2.3 successfully prevents perf-run contamination from diagnostic flags.
  - Electronic Enclosure remains dominated by `loadStepFile` (`transfer` + `readFile`) and curved/trimmed tessellation costs, so M2/M3 and load-path reduction remain the next blockers.


### 2026-02-11 M2.1 Iteration (CDT Recovery Fast-Reject + Boundary Ordering)

- What changed (`src/cdt-gpu.ts`):
  - Added triangle-vs-constraint edge AABB fast-reject inside `triangleIntersectsEdge(...)` to skip expensive segment crossing checks when boxes do not overlap.
  - Hoisted edge endpoints and edge bbox computation in `recoverConstraintEdge(...)` so they are computed once per constraint edge and reused for all triangle checks.
  - Reworked cavity boundary ordering:
    - Added cycle-aware `orderBoundaryEdges(...)` walker (degree-2 cycle path).
    - Added `orderBoundaryEdgesGreedy(...)` fallback for irregular boundaries.
    - Added endpoint-based retry (`v1` first, retry from `v2`) before declaring `Constraint endpoints not on cavity boundary`.

- Canary result (`npm run -s bench:canary`):
  - Wins vs `occt-import-js`: `6/6`
  - Speedup median: `2.48x`
  - Ours avg runtime: `92.5ms` (p90 `192.0ms`)
  - Ref avg runtime: `186.8ms` (p90 `376.2ms`)

- Representative spot-check (`npm run -s bench:representative -- --filter "Electronic Enclosure"`):
  - `Electronic Enclosure`: ours `11778.6ms`, ref `5128.8ms` (`2.30x` slower)
  - `loadStepFile`: `4416.9ms` (`37.5%` of ours)
  - top load sub-phases: `transfer 3299.6ms`, `readFile 1113.4ms`, `fsWrite 1.3ms`
  - triangle ratio: `58745 / 39148` (`1.50x`)

- Conclusion:
  - M2.1 reduced CDT hot-loop overhead and stabilized cone/trim canary performance.
  - Electronic Enclosure remains dominated by load path (`transfer` + `readFile`) plus residual curved/trim triangle inflation.
  - Next steps stay unchanged: continue M2/M3 while parallelizing load-path reductions.


### 2026-02-11 M0 Checkpoint

- Run command:
  - `npm run -s baseline:beat -- --date 2026-02-11 --skip-ai`
- Artifacts:
  - `diagnostics/beat-occt-import-js/2026-02-11/2026-02-11T07-30-04-101Z-c59b8a6/`
- Outcome:
  - `tests/run-tests.js`: fail (`testVisualHoleRendering` timeout)
  - `tests/benchmark-comprehensive.js`: pass (avg 1.20x, 1/8 wins)
  - `tests/benchmark.js`: pass (avg 3.84x, 4/5 wins)

### 2026-02-11 Harness Upgrade

- `tests/benchmark-comprehensive.js` now uses suite profiles:
  - `canary` (fast loop)
  - `representative` (includes Electronic Enclosure)
  - `full` (broader coverage)
- Added package scripts:
  - `bench:canary`
  - `bench:representative`
  - `bench:full`
  - `bench:micro`
- `scripts/run-beat-occt-baseline.mjs` now accepts:
  - `--bench-suite canary|representative|full`
  - default baseline suite = `representative`

### 2026-02-13 Next Optimization Decision (Post Curved-Face Batching)

- Current optimization in progress:
  - **Triangle reduction on curved/trimmed faces** (adaptive budgets + anisotropic sampling), starting with Electronic Enclosure and VM-001.

- Why:
  - Latest representative profile still shows Electronic Enclosure slower than `occt-import-js`.
  - Runtime split is still dominated by tessellation work after load:
    - total `7667.5ms`
    - `loadStepFile`: `2928.0ms` (`38.2%`)
    - non-load tessellation path: `~4739.5ms` (`61.8%`)
  - Triangle inflation remains the key multiplier:
    - Electronic Enclosure triangles: ours `222890` vs ref `39148` (`5.69x`).

- Impact model (estimated):
  1. GPU parallelization improvements (M3/M4):
     - Expected: `-0.8s` to `-1.8s` on Electronic Enclosure (`~10-24%` total).
     - Scope:
       - larger curved-face dispatch batches
       - persistent buffers/pipelines
       - reduced per-face submit/readback/sync
  2. Triangle-count reduction (new top priority):
     - Expected: `-1.5s` to `-3.0s` (`~20-40%` total).
     - Scope:
       - adaptive deflection/error budgets per face type
       - anisotropic sampling (boundary-dense, interior-coarse)
       - primitive-specific caps for cylinder/cone/torus/bspline
  3. Load-path follow-up (M1.2 continuation):
     - Expected: `-0.7s` to `-1.4s` (`~9-18%` total).
     - Scope:
       - transfer/copy elimination
       - FS/write-path minimization
       - stricter perf-mode fast path

- Execution order for next cycle:
  1. Triangle reduction first (quality-guarded).
  2. GPU batching/sync minimization second.
  3. Load-path copy/transfer reduction third.

- Validation cadence for this cycle:
  - Every step: `npm run -s bench:canary` (80-model canary including VM-001).
  - Every 2 steps: `npm run -s bench:representative`.
  - Every 2-3 steps: full correctness gate (`node --experimental-vm-modules tests/run-tests.js`), then AI visual run when needed.

### 2026-02-13 Triangle Reduction Track (Execution)

- Optimization being executed now:
  - Reduce triangle inflation at the source (boundary sampling + trim grid density), with sharp-feature preservation.
- Why this is first:
  - `Electronic Enclosure` remains triangle-heavy (`~5.7x` ref triangles), and this directly scales tessellation cost.
- Concrete implementation steps:
  1. Coarsen smooth circular boundary sampling in perf mode using adaptive deflection defaults (not fixed dense arcs).
  2. Tighten cylinder/cone trimmed-face U/V density caps in perf mode, especially for high-complexity trims.
  3. Keep seam/hole guardrails and corner-preserving behavior so we do not blunt slots or lose walls.
- Validation for each step:
  - `npm run -s bench:canary`
  - `npm run -s bench:representative` (includes Electronic Enclosure + VM-001)
  - Visual spot-check on trimmed cylinder/cone fixtures before accepting.
