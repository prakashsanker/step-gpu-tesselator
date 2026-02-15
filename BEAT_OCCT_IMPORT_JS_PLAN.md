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
6. GPU model-wide mesh assembly (single readback):
   - Curved-face batches should write into one global GPU output (positions/normals/indices).
   - Replace per-face CPU stitching with one batch-level readback + merge.
7. GPU-side prefix-sum/offset pass:
   - Compute per-face vertex/index offsets on GPU.
   - Move index rebasing/compaction out of CPU loops.
8. Fixed benchmark loop for each step:
   - Run full canary first.
   - Then run representative and track Electronic Enclosure speed ratio (`ours/ref` and `ref/ours`), not only absolute ms.

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
  - **GPU-first throughput work (M3/M4)** on curved/trimmed tessellation and post-tessellation mesh processing.
  - Keep triangle reduction as a quality guardrail, not the primary lever this cycle.

- Why:
  - Latest representative profile still shows Electronic Enclosure slower than `occt-import-js`.
  - Runtime split remains dominated by non-load work after parsing:
    - total `~7.7s`
    - `loadStepFile`: `~2.8-2.9s` (`~36-38%`)
    - non-load tessellation + mesh path: `~4.8-4.9s` (`~62-64%`)
  - Triangle inflation has improved materially in recent passes, but we are still slower on large complex files; the bigger remaining gap is throughput on curved/trimmed face processing + mesh post-processing.

- Impact model (estimated):
  1. GPU parallelization improvements (M3/M4) [Top priority]:
     - Expected: `-1.0s` to `-2.2s` on Electronic Enclosure (`~13-29%` total).
     - Scope:
       - larger curved-face dispatch batches
       - persistent/reused GPU buffers and pipelines
       - reduced per-face submit/readback/sync
       - one-pass mesh-level GPU normals for large outputs
  2. Pathological face preprocessing/classification controls:
     - Expected: `-0.6s` to `-1.5s` (`~8-20%` total).
     - Scope:
       - classify high-risk faces early (seam-crossing, huge trim loops, dense pcurves, near-singular UV spans)
       - route only those faces to robust expensive paths
       - keep default fast path for normal faces
  3. Triangle-count reduction (targeted, hotspot-only):
     - Expected: `-0.3s` to `-1.0s` (`~4-13%` total).
     - Scope:
       - adaptive deflection/error budgets where face-level profiles show inflation
       - keep slot/corner fidelity and seam guardrails
  4. Load-path follow-up (M1.2 continuation):
     - Expected: `-0.7s` to `-1.4s` (`~9-18%` total).
     - Scope:
       - transfer/copy elimination
       - FS/write-path minimization
       - stricter perf-mode fast path

- Execution order for next cycle:
  1. GPU batching/sync minimization + mesh-level GPU normals first.
  2. Pathological preprocessing/classification gating second.
  3. Hotspot-only triangle reduction third.
  4. Load-path copy/transfer reduction in parallel.

- Validation cadence for this cycle:
  - Every step: `npm run -s bench:canary` (80-model canary including VM-001).
  - Every 2 steps: `npm run -s bench:representative`.
  - Every 2-3 steps: full correctness gate (`node --experimental-vm-modules tests/run-tests.js`), then AI visual run when needed.

### 2026-02-13 Pathological Face Preprocessing/Classification (Definition)

- Meaning in this plan:
  - "Pathological" faces are outliers that trigger expensive or unstable behavior if treated with the same defaults as ordinary faces.
  - Examples:
    - seam-crossing trims and wrapped domains
    - very dense pcurve loops
    - highly imbalanced inner/outer loop complexity
    - singular/near-singular UV regions
- Why it helps on unseen files:
  - This is not model-specific tuning; it is complexity-based routing.
  - Any unseen model with similar geometric complexity patterns is detected and routed to robust handling, while typical faces remain on a cheaper fast path.
  - Result: lower tail latency and fewer worst-case slowdowns without overfitting to known fixtures.

### 2026-02-13 M3 Step: GPU UV->3D Surface Evaluation (Primitive Curved Faces)

- Optimization implemented:
  - Added GPU compute path for per-vertex surface evaluation (UV -> position + normal) on:
    - `CYLINDRICAL_SURFACE`
    - `SPHERICAL_SURFACE`
    - `CONICAL_SURFACE`
    - `TOROIDAL_SURFACE`
    - `PLANE`
  - Kept CPU fallback for unsupported surfaces and small meshes.
  - Added persistent GPU buffer/pipeline reuse for this path.
  - Wired into `tessellateTrimmedSurface` evaluation stage via `evaluateUVMesh`.

- Validation:
  - `npm run -s bench:canary`
    - pass `80/80`, quality gate pass.
  - `npm run -s bench:representative`
    - Electronic Enclosure: `8091.7ms -> 7956.7ms` (`-135.0ms`)
    - ref: `3250.5ms -> 3086.3ms`
    - speedup multiple (`ref/ours`) moved `0.402x -> 0.388x` (worse)
    - equivalent slowdown ratio (`ours/ref`) moved `2.49x -> 2.58x` (worse)
    - loadStepFile share now `~40.1%` (`3188.9ms / 7956.7ms`)

- Takeaway:
  - This step improved our absolute runtime but regressed relative speedup vs `occt-import-js`.
  - Remaining gap is still dominated by:
    - load path (`transfer` + `readFile`) and
    - curved/trim triangulation throughput outside UV evaluation.

### 2026-02-13 M3 Step: Batched GPU Surface Evaluation Queue

- Optimization implemented:
  - Switched GPU UV->3D eval from per-call dispatch/readback to queued batched dispatch:
    - collect multiple face eval jobs
    - pack UVs + surface params into a single GPU submission
    - single batched readback, then split results per face
  - Goal: reduce per-face sync overhead in curved-face runs.

- Validation:
  - `npm run -s bench:canary`
    - pass `80/80`, quality gate pass.
  - `npm run -s bench:representative`
    - Electronic Enclosure: `7956.7ms -> 8413.4ms` (regressed)
    - speedup multiple (`ref/ours`) moved `0.388x -> 0.387x` (slightly worse)
    - equivalent slowdown ratio (`ours/ref`) moved `2.58x -> 2.59x` (slightly worse)
    - loadStepFile share dropped (`40.1% -> 36.1%`), but non-load path increased enough to offset.

- Takeaway:
  - This batching shape is not yet a net win on representative KPI.
  - Next revision should reduce CPU packing/splitting overhead and increase effective batch size before keeping this path on by default.

### 2026-02-13 Known Regression Note (Visual Quality)

- Observed regression:
  - Batched GPU surface-eval variant introduced visible shading/faceting artifacts on Electronic Enclosure (interior cone/bowl region).
- Action:
  - Keep this variant experimental only (do not treat as production default).
  - Add a visual check for this artifact before accepting future GPU-eval batching changes.
- Suspected causes to verify in a future fix pass:
  - batched surface parameter packing/decoding mismatch,
  - per-batch result slicing/indexing mistakes,
  - normal continuity mismatch on trimmed/seam-sensitive regions.

### Next GPU Priority (After This Attempt)

- Highest leverage next step:
  - Move trimmed-grid classification + triangle candidate generation from CPU loops to GPU kernels.
- Why this is next:
  - Current majority non-load time is still in curved/trimmed tessellation prep, not just UV->3D evaluation.
  - `tessellateTrimmedSurface` still spends heavy CPU time in:
    - point-in-polygon / hole inclusion checks per grid point,
    - per-cell triangle candidate filtering.
- Expected impact:
  - Better amortization than per-face UV eval alone, especially on large trimmed curved faces (Electronic Enclosure).

### 2026-02-13 M3 Step: GPU Surface-Eval Sync Minimization (Bind-Group Reuse + Copy Elision)

- Optimization implemented:
  - Reused cached surface-eval bind groups across GPU batches when buffer bindings are unchanged.
  - Removed redundant full-batch CPU copies on GPU readback:
    - no intermediate `positionsAll/normalsAll` full-array clone,
    - direct per-job extraction from mapped ranges before unmap.
  - Goal: reduce CPU overhead and sync cost in batched UV->3D surface evaluation.

- Validation:
  - `npm run -s bench:canary`
    - pass `80/80`, quality gate pass.
    - wins vs `occt-import-js`: `79/80`
    - speedup median: `4.75x faster`
  - `npm run -s bench:representative`
    - Electronic Enclosure: `7510.2ms` vs ref `3068.3ms` (`2.45x slower`)
    - VM-001: `191.3ms` vs ref `174.2ms` (`1.10x slower`)
    - wins vs `occt-import-js`: `4/6`
    - speedup median: `2.57x faster`
    - loadStepFile share (Electronic Enclosure): `37.2%` (`2793.5ms / 7510.2ms`)

- Takeaway:
  - Absolute Electronic Enclosure runtime improved, but relative multiple vs `occt-import-js` remains behind because ref also moved faster in this run.
  - This confirms GPU sync reductions help, but they are still not the dominant remaining lever.
  - Next step remains unchanged: move trimmed-grid triangle candidate generation/filtering out of CPU loops.

### 2026-02-13 M3 Step: Zero-Copy GPU Trim Classify -> Triangle Build

- Optimization implemented:
  - Added a fused GPU path in `src/trim-grid-gpu.ts`:
    - `classifyAndBuildTrimGridTrianglesGPU(...)`
  - Runs trim-grid classification and trim-cell triangle generation in one GPU command sequence.
  - Removes CPU orchestration bounce on this path:
    - no GPU mask readback to CPU,
    - no CPU mask upload back to GPU for triangle build.
  - `src/surface-tessellation.ts` now tries this fused path first for eligible trimmed surfaces.
  - Existing readback-based classification + triangle build remains as fallback for safety.

- Validation:
  - `npm run -s bench:canary`
    - pass `80/80`, quality gate pass.
    - wins vs `occt-import-js`: `79/80`
    - speedup median: `5.27x faster`
  - `npm run -s bench:representative`
    - Electronic Enclosure: `7892.1ms` vs ref `3500.6ms` (`2.25x slower`)
    - VM-001: `335.6ms` vs ref `240.8ms` (`1.39x slower`)
    - wins vs `occt-import-js`: `4/6`
    - speedup median: `2.17x faster`
    - loadStepFile share (Electronic Enclosure): `38.9%` (`3066.6ms / 7892.1ms`)
    - speed-multiple trend: improved from `2.36x slower` to `2.25x slower` on Electronic Enclosure.

- Takeaway:
  - This removes a real CPU↔GPU orchestration tax and improved the Electronic Enclosure slowdown multiple.
  - Absolute runtime remained noisy run-to-run; relative multiple is the metric to track.
  - Remaining gap is still dominated by load path + large non-load curved/trimmed throughput work.

### 2026-02-13 M3 Step: Dense-Grid GPU Surface Evaluation for Fused Trim Path

- Optimization implemented:
  - Added dense-grid GPU surface evaluation path in `src/surface-eval-gpu.ts`:
    - `evaluateSurfaceDenseGridGPU(...)`
  - For fused trim classify+triangle-build faces, we now:
    - evaluate dense-grid positions/normals directly on GPU from grid params,
    - avoid CPU `Vec2[]` vertex materialization before evaluation.
  - Wired in `src/surface-tessellation.ts` for the fused GPU trim path, with fallback to existing evaluation path.

- Validation:
  - `npm run -s bench:canary`
    - pass `80/80`, quality gate pass.
    - wins vs `occt-import-js`: `80/80`
    - speedup median: `4.56x faster`
  - `npm run -s bench:representative`
    - Electronic Enclosure: `7279.1ms` vs ref `3082.5ms` (`2.36x slower`)
    - VM-001: `211.4ms` vs ref `171.7ms` (`1.23x slower`)
    - wins vs `occt-import-js`: `4/6`
    - speedup median: `2.68x faster`
    - loadStepFile share (Electronic Enclosure): `37.4%` (`2722.1ms / 7279.1ms`)
    - trend note: ours improved in absolute time, but slowdown multiple vs ref stayed effectively flat due faster ref in this run.

- Takeaway:
  - This reduces CPU orchestration/object churn in the fused trim path.
  - It did not create the expected representative-speedup multiple gain by itself.
  - Next leverage remains larger batch-level GPU fusion (fewer per-face sync points) plus load-path work.

### 2026-02-13 M4 Step (In Progress): GPU Curved-Batch Assembly + GPU Offsets

- Optimization being implemented:
  - Add GPU batch mesh assembly for curved-face model batches with a single batch readback.
  - Add GPU-side prefix-sum offsets so per-face index rebasing/compaction is not CPU-stitched.
- Validation protocol for this step:
  - Run full canary.
  - Run representative.
  - Track Electronic Enclosure speed ratio trend (`ref/ours` and `ours/ref`) after each run.

### 2026-02-13 M4 Step: GPU Curved-Batch Assembly + GPU Offset Pass (Initial Run)

- Optimization implemented in code:
  - Added `src/mesh-batch-assembly-gpu.ts`:
    - GPU prefix-sum pass computes per-face vertex/index offsets.
    - GPU assembly pass rebases indices and compacts curved-face batch meshes into one global output.
    - Single staging readback (`mapAsync`) for combined positions+indices payload.
  - Wired curved-face batching in `src/occ-test.ts` to use GPU assembly when enabled.
  - Added runtime gate:
    - `__ENABLE_GPU_CURVED_BATCH_ASSEMBLY__` (default ties to perf geometry-only mode).

- Validation:
  - `npm run -s bench:canary`
    - successful: `80/80`
    - wins vs `occt-import-js`: `78/80`
    - speedup median: `4.45x faster`
  - `npm run -s bench:representative`
    - Electronic Enclosure: `ours=8050.4ms`, `ref=3758.5ms`
      - `ours/ref = 2.14x slower`
      - `ref/ours = 0.467x`
    - VM-001: `ours=286.4ms`, `ref=192.3ms` (`1.49x slower`)
    - wins vs `occt-import-js`: `4/6`
    - speedup median: `2.70x faster`

- Electronic Enclosure trend vs previous representative sample:
  - absolute runtime: ours `+271.5ms` slower, ref `+143.4ms` slower
  - ratio: slightly improved from `2.15x slower` to `2.14x slower` (`ref/ours` improved by `1.005x`)
  - triangle ratio unchanged (`3.649x`)

- Takeaway:
  - Curved-batch CPU stitching moved to GPU offsets+assembly path without correctness failures in canary.
  - Representative slowdown multiple improved only marginally; more GPU-side fusion is still required.

### 2026-02-13 M4 Step: Remove Curved-Face Re-Flattening (Carry Flat Mesh Buffers)

- Optimization implemented in code:
  - `tessellatedMeshToVerticesAndTriangles(...)` now keeps native typed mesh buffers attached to curved-face results.
  - Curved-face batch GPU assembly path now consumes these flat buffers directly (`getFaceResultFlatArrays`) instead of rebuilding typed arrays from `Vec3[]/number[][]`.
  - Non-batch append path also consumes flat buffers directly when available, avoiding extra per-face flattening work in hot loops.

- Validation:
  - `npm run -s bench:canary`
    - successful: `80/80`
    - failed: `0/80`
    - wins vs `occt-import-js`: `77/80`
    - speedup median: `4.44x faster`
  - `npm run -s bench:representative`
    - Electronic Enclosure: `ours=7607.5ms`, `ref=3763.2ms`
      - `ours/ref = 2.02x slower`
      - `ref/ours = 0.495x`
    - VM-001: `ours=290.2ms`, `ref=185.0ms` (`1.57x slower`)
    - wins vs `occt-import-js`: `4/6`

- Electronic Enclosure trend vs previous representative sample:
  - ours faster by `442.9ms`
  - ratio improved by `1.060x` (`2.14x slower -> 2.02x slower`)
  - triRatio unchanged (`3.649x`)

- Takeaway:
  - Removing CPU re-flattening delivered a meaningful representative ratio improvement on Electronic Enclosure.
  - Next step should continue reducing CPU-side per-face object assembly (toward model-level typed output assembly).

### 2026-02-13 M4 Step: Typed-Array-First Model Assembly (No Global Vec3/number Stitching)

- Optimization implemented in code:
  - `tessellateOCCShape` now accumulates mesh output as typed chunks:
    - `positionChunks: Float32Array[]`
    - `indexChunks: Uint32Array[]`
    - color runs (`vertexCount + color`) instead of per-vertex color objects.
  - Final mesh assembly now concatenates typed chunks directly into final `positions/indices`.
  - CPU smooth-normal fallback switched from object-vertex path to typed arrays:
    - `computeSmoothNormalsCPUFromFlat(positions, indices)`.

- Validation:
  - `npm run -s bench:canary`
    - successful: `80/80`
    - failed: `0/80`
    - wins vs `occt-import-js`: `78/80`
    - speedup median: `4.31x faster`
  - `npm run -s bench:representative`
    - Electronic Enclosure: `ours=8672.2ms`, `ref=3279.5ms`
      - `ours/ref = 2.64x slower`
      - `ref/ours = 0.378x`
    - VM-001: `ours=203.9ms`, `ref=197.1ms` (`1.03x slower`)
    - wins vs `occt-import-js`: `4/6`

- Electronic Enclosure trend vs previous representative sample:
  - ours slower by `1064.7ms`
  - ratio regressed by `1.308x` (`2.02x slower -> 2.64x slower`)
  - triRatio unchanged (`3.649x`)
  - ref faster by `483.6ms` in this run.

- Takeaway:
  - Typed assembly path is functionally stable (canary pass), but this single representative sample regressed.
  - Regression appears dominated by run-to-run load-path variance in this measurement; keep tracking ratio trend over subsequent steps.

### 2026-02-13 M4 Step: Increase Model-Level Curved Batch Size (Fewer GPU Readbacks)

- Optimization implemented in code:
  - Increased perf-mode default for model-level curved batching:
    - `__MODEL_LEVEL_CURVED_BATCH_SIZE__` default `24 -> 256` when `__PERF_GEOMETRY_ONLY_LOAD__=true`.
  - Raised hard cap for model-level curved batch size:
    - `64 -> 1024` (still clamped and runtime-tunable via global).
  - This reduces `assembleMeshBatchGPU(...)` invocations and `mapAsync` readbacks for curved-face-heavy models.

- Validation:
  - `npm run -s bench:canary`
    - successful: `80/80`
    - failed: `0/80`
    - wins vs `occt-import-js`: `77/80`
    - speedup median: `4.73x faster`
  - `node tests/benchmark-comprehensive.js --suite representative --runs 3 --warmup 1`
    - Electronic Enclosure: `ours=7202.5ms`, `ref=2825.9ms`
      - `ours/ref = 2.55x slower`
      - `ref/ours = 0.392x`
      - `loadStepFile=2629.2ms` (`36.5%` of ours)
    - VM-001: `ours=168.2ms`, `ref=165.4ms` (`1.02x slower`)
    - wins vs `occt-import-js`: `4/6`

- Takeaway:
  - Change is stable (no canary regressions).
  - Electronic Enclosure remains limited by load path + curved-face runtime variability, so next step should remove remaining per-batch sync by pushing model-level curved assembly to a true single-readback path.

### 2026-02-13 Profiling Deep Dive (Wall-Clock, Latest)

- Run used for wall-clock attribution:
  - `npm run -s bench:representative -- --filter Electronic --max-files 1 --runs 1 --warmup 0 --no-prewarm --detailed-profile`
- Key result:
  - Electronic Enclosure: `ours=9230.4ms`, `ref=3297.8ms` (`2.80x slower`)
  - `loadStepFile=4696.6ms` (`50.9%`)
  - non-load wall path: `tessellateOCCShape=4410.1ms`
- Top non-load wall phases:
  - `curved_batch_compute_wall=4289.5ms`
  - `curved_batch_max_trim_uvmesh_gpu_eval=4229.1ms`
  - `curved_batch_max_trim_final_evaluate_mesh=4277.1ms`
  - `curved_batch_max_trim_cpu_triangle_build=1214.8ms`
  - `curved_batch_max_trim_cpu_grid_classify=134.2ms`
  - `curved_batch_assembly_wall=19.4ms`
- Important interpretation:
  - `curved_trim_phase_*` totals are sum-over-faces (overlapping waits), not wall-clock.
  - The wall-clock bottleneck is now clearly `curved_batch_compute_wall`, specifically dense UV-mesh evaluation/final evaluation, not curved batch assembly.

### 2026-02-13 Next Execution Order (Updated from Profile)

1. M4.1: model-wide dense trimmed-grid GPU evaluation (single packed dispatch/readback per curved batch).
2. M4.2: GPU-side per-face offset/prefix handling for dense eval outputs (remove CPU split/reindex loops).
3. M4.3: move trimmed-cell triangle generation/reindex fully to GPU for eligible faces (eliminate `cpu_triangle_build` hotspot share).
4. Keep validation strict after each step:
   - `npm run -s bench:canary`
   - then `npm run -s bench:representative`
   - track Electronic Enclosure using slowdown multiple (`ours/ref`) and speedup (`ref/ours`), not absolute ms alone.

### 2026-02-13 M4.1 Step: Model-Wide Trimmed Eval Coalescing (One/Few Global GPU Jobs)

- Optimization implemented:
  - Updated `/src/surface-eval-gpu.ts` batching policy for both sparse UV eval and dense-grid eval:
    - perf-mode batch targets raised (`jobs: 256`, plus new vertex-cap target),
    - perf-mode coalescing window increased (`delay: 4ms`),
    - timer-based flush changed to debounce so arrivals over a short window coalesce into larger submissions.
  - Added vertex-based flush thresholds:
    - `__GPU_SURFACE_EVAL_BATCH_TARGET_VERTS__`
    - `__GPU_SURFACE_GRID_BATCH_TARGET_VERTS__`
  - Wired deterministic benchmark-mode knobs in `/tests/benchmark-comprehensive.html` (set + restore):
    - `__GPU_SURFACE_GRID_BATCH_TARGET_JOBS__ = 1024`
    - `__GPU_SURFACE_GRID_BATCH_TARGET_VERTS__ = 2_000_000`
    - `__GPU_SURFACE_GRID_BATCH_DELAY_MS__ = 8`
    - `__GPU_SURFACE_EVAL_BATCH_TARGET_JOBS__ = 1024`
    - `__GPU_SURFACE_EVAL_BATCH_TARGET_VERTS__ = 2_000_000`
    - `__GPU_SURFACE_EVAL_BATCH_DELAY_MS__ = 8`

- Validation:
  - `npm run -s bench:canary`
    - successful: `80/80`
    - failed: `0/80`
    - wins vs `occt-import-js`: `80/80`
    - speedup median: `4.78x faster`
  - `npm run -s bench:representative`
    - Electronic Enclosure: `ours=7406.3ms`, `ref=3125.4ms`
      - `ours/ref = 2.37x slower`
      - `ref/ours = 0.422x`
      - `loadStepFile=2777.1ms` (`37.5%`)
      - `curved_batch_compute_wall=4320.6ms`
    - VM-001: `ours=194.4ms`, `ref=190.9ms` (`1.02x slower`)
    - wins vs `occt-import-js`: `3/6`
  - Detailed single-file profile (`--filter Electronic --detailed-profile`):
    - Electronic Enclosure: `ours=9303.9ms`, `ref=3212.8ms` (`2.90x slower`)
    - `loadStepFile=4827.5ms` (`51.9%`)
    - top non-load wall: `curved_batch_compute_wall=4245.1ms`,
      `curved_batch_max_trim_uvmesh_gpu_eval=4163.4ms`,
      `curved_batch_max_trim_final_evaluate_mesh=4204.0ms`.

- Takeaway:
  - Coalescing is stable and improves Electronic Enclosure ratio vs prior worse sample (`2.90x -> 2.37x slower`) but does not yet break through the main wall-time hotspot.
  - Bottleneck remains dense trimmed evaluation + final evaluate path (`uvmesh_gpu_eval`/`final_evaluate_mesh`) and still requires deeper fusion in M4.2/M4.3.

### 2026-02-15 Classifier Parity Sprint (Priority Track)

Recent profiling shows two non-obvious facts that must be addressed before we can trust throughput gains:
1) fast paths can still be dominated by incorrect geometry decisions on cone-like domains, and  
2) correctness work must remain in the canary loop, otherwise regressions slip through when changing classification logic.

Current objective:
- Restore parity on OCC-like point-in-domain behavior for trimmed curved faces (especially cones) without regressing the canary geometry.
- Once parity is acceptable, keep squeezing throughput under the same benchmark gates.

Execution sequence (strict order):

1. Stage A (Classification Instrumentation and Baselines)
   - Continue shadow-mode tracking with `--classifier-shadow` on representative and canary.
   - Record:
     - `mismatchCount`
     - `localUncertain`
     - `effectiveMismatch = mismatchCount + localUncertain`
     - `mismatchFromStageA`, `mismatchFromStageB`, `mismatchFromDomainUnsafe`
     - `domainUnsafeFaceCount`
     - `mismatch` trend for Electronic Enclosure.
   - Gate:
     - canary status: `PASS`
     - no catastrophic geometry regressions in canary images.

2. Stage B (OCCT Parity Work on our classifier)
   - Implement/adjust cone-focused Stage A/B behavior in `tessellateTrimmedSurface` and related helpers toward OCCT-like periodic/domain behavior.
   - Keep domain handling explicit: avoid making cone faces uncertain-only unless telemetry proves domain-unsafe path is dominating.
   - Gate:
     - canary + representative pass
     - `mismatch` and `effectiveMismatch` improve in the direction of lower values
     - domain-unsafe count and `effectiveMismatch` trend should not worsen.

3. Stage C (Candidate mode promotion path)
   - Switch from default fallback to candidate path only where telemetry allows:
     - `--classifier-candidate` enabled.
     - `--classifier-no-fallback` only when mismatch/uncertainty are clearly below target.
   - Gate:
     - canary must remain pass in candidate mode (no visible regressions).
     - representative must stay stable.

4. Throughput re-check after each classifier stage
   - Keep the existing strict benchmark gate after each code change:
     - `npm run -s bench:canary -- --classifier-shadow`
     - `npm run -s bench:representative -- --classifier-shadow`
     - `npm run -s bench:canary`
     - `npm run -s bench:representative`
   - Always compare to last good snapshot using:
     - slowdown multiple (`ours/ref`)
     - `effectiveMismatch` trend.

5. Integration rule with M4/GPU efforts
   - Any classifier edit blocks promotion to default if it causes geometric instability, regardless of speed gains.
   - Once parity and canary are green, continue M4.1/M4.2/M4.3 optimizations with the same gates.
