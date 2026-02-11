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
6. [ ] Execute M1.1.2: planar-hole complexity dispatch + UV cleanup and re-run canary.
7. [x] Run full correctness suite every 2-3 perf steps to prevent regressions.

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
