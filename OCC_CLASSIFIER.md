# OCC Classifier Replacement Plan (Correctness First)

Last updated: 2026-02-14
Owner: Tessellation pipeline team

## Goal

Replace per-point OCC classifier dependency (`BRepTopAdaptor_FClass2d.Perform`) with our own classifier while preserving visual correctness and mesh validity.

This document is the source of truth for:
- replacement strategy
- correctness gate criteria
- benchmark/correctness history
- what to do after context compaction

## Why This Exists

Current measurements indicate OCC point classification is a major cost center on Electronic Enclosure-class models.

Validated diagnosis summary:
- 156 OCC-classified faces (154 cones + 2 lids) dominate trim classification/triangle gating CPU time.
- 448 polygon-classified faces are comparatively cheap.
- OCC path incurs heavy WASM round-trip volume due to per-point calls from:
  - grid classification
  - triangle gate sampling

## Non-Negotiable Rule

Correctness comes before speed.

No performance optimization is accepted unless:
1. Canary correctness passes (expanded canary suite).
2. Representative correctness is visually acceptable on critical files.
3. No catastrophic geometry regressions (missing faces, severe seam tears, inverted regions).

## Deferred Visual Regression Note

User-confirmed visual regression to track (do not forget while mismatch work continues):
- Files:
  - `step-examples/c4-surfaces/cone.step`
  - `step-examples/complex/conical-surface.step`
- Observation: slight visual regression in cone/conical rendering under current local-classifier-focused configuration.
- Policy:
  - Keep mismatch-reduction as the active priority now.
  - Do not treat this as closed until we run a dedicated visual re-check pass after the next parity milestone.
  - Before any promotion/merge, this visual regression must be explicitly re-tested and documented in this file.

## Active Benchmark Rules

These rules are now mandatory for routine iteration:

1. Always run with local classifier output path in strict mode (no OCC fallback).
2. Routine suites are:
   - full canary (`npm run -s bench:canary`)
   - Electronic Enclosure focused representative (`npm run -s bench:electronic`)
3. OCC-backed shadow runs are diagnostic-only and require explicit opt-in.

## Execution Strategy

OCCT-spec alignment update:
- We will not do a wholesale OCCT kernel port.
- We will treat `BRepTopAdaptor_FClass2d::Perform` semantics as the spec and implement a TS classifier that mirrors:
  - decision ordering (outer/hole/on-boundary precedence)
  - periodic-parameter handling
  - tolerance policy and uncertainty handling
- We will validate in shadow mode against OCC on target faces before promotion.

## OCCT Source Breakdown (Authoritative)

Primary files analyzed (OCCT master):
- `/tmp/occt-repo/src/ModelingAlgorithms/TKTopAlgo/BRepTopAdaptor/BRepTopAdaptor_FClass2d.cxx`
- `/tmp/occt-repo/src/ModelingAlgorithms/TKTopAlgo/BRepClass/BRepClass_FaceClassifier.cxx`
- `/tmp/occt-repo/src/ModelingAlgorithms/TKGeomAlgo/TopClass/TopClass_FaceClassifier.gxx`
- `/tmp/occt-repo/src/ModelingAlgorithms/TKGeomAlgo/TopClass/TopClass_Classifier2d.gxx`
- `/tmp/occt-repo/src/ModelingAlgorithms/TKTopAlgo/BRepClass/BRepClass_FaceExplorer.cxx`
- `/tmp/occt-repo/src/ModelingAlgorithms/TKTopAlgo/BRepClass/BRepClass_Intersector.cxx`

### 1) `BRepTopAdaptor_FClass2d` is not a simple point-in-polygon

It does all of this before answering a point query:
- Builds per-wire sampled UV polylines from edge pcurves (orientation-aware).
- Detects degenerate/closed/invalid edges and marks wire quality.
- Tracks per-wire deflection (`FlecheU/FlecheV`) and refines sampling if polygon quality is poor.
- Computes wire orientation from signed area proxy (`square`) with special cases.
- Stores wire classifiers (`CSLib_Class2d`) plus orientation bits.

Then `Perform()`:
- Re-frames periodic UV (`RecadreOnPeriodic`) into wire domain.
- Evaluates each wire via `SiDans`.
- Combines wire result with orientation rules to infer in/out.
- If result is ambiguous (`cur == 0`) or wire quality is bad, falls back to `BRepClass_FaceClassifier`.
- For periodic surfaces, retries shifted periodic images and accepts early on `IN`/`ON`.

### 2) Fallback (`BRepClass_FaceClassifier`) uses ray/edge-transition topology logic

It is a topological classifier, not pure polygon math:
- Chooses a robust probing segment (`FaceExplorer::Segment`) away from tangency.
- Intersects probe with each edge (`Intersector::Perform`) with tolerance handling.
- Handles direct `ON` by minimum-distance-to-curve check.
- Handles high-tolerance vertices via skip-bridge (`CheckSkip`) to avoid missed crossings.
- Uses transition states + edge orientation to determine `IN/OUT/ON`.
- Resolves head/end vertex events with transition accumulator (`TopTrans_CurveTransition`).

This explains why naive UV polygon tests are systematically wrong on many cone faces.

### 3) Why our current local classifier diverges

Observed mismatch pattern is dominated by cone `false-inside`, interior (not boundary):
- `mismatch=23836/113207`, `falseInside=23835`, `falseOutside=1`
- Large face families repeat `inside=625, outside=0, uncertain=104` on 27x27 grids.

Interpretation:
- We are still approximating with simplified geometric loop logic.
- OCCT decision path includes edge-transition and tolerance semantics we do not yet emulate.
- Periodic candidate heuristics alone do not solve this gap.

## Port Plan (from OCCT semantics)

### Stage A: TS `FClass2d` equivalent (wire-level classifier)
- Build per-wire UV polylines from OCC pcurves with minimal simplification.
- Carry per-wire orientation and quality flags.
- Implement `SiDans`/`SiDans_OnMode`-equivalent with OCCT-like boundary treatment.
- Implement periodic reframe/retry loop order equivalent to `Perform()`.

### Stage B: TS fallback classifier equivalent (`FaceClassifier` subset)
- Build robust probe line generation similar to `FaceExplorer::Segment`.
- Intersect probe against wire edges with tolerance and closest-hit selection.
- Implement transition-state mapping (`In/Out/Touch`) with edge orientation.
- Implement head/end complex-transition accumulation behavior.
- Include high-tolerance-vertex skip bridge equivalent (`CheckSkip`) for problematic vertices.

### Stage C: Promotion criteria for OCCT replacement
- Shadow metrics on Electronic Enclosure:
  - overall mismatch < 1%
  - interior mismatch < 0.2%
  - cone false-inside near zero
- Candidate mode with OCC fallback:
  - no visual regressions on canary + representative
  - fallback confined to measured pathological faces

1. Implement custom classifier as candidate primary.
- Seam-aware UV unwrap handling.
- Hole-aware point-in-polygon behavior.
- Boundary tolerance band handling.

2. Shadow mode validation (no behavior change yet).
- Compute both decisions:
  - custom classifier decision
  - OCC classifier decision
- Render/output still uses OCC decisions.
- Record mismatch metrics by:
  - face
  - surface type
  - near-boundary vs interior
  - seam-proximate vs non-seam

3. Define and enforce promotion criteria.
- Required before switching to custom-primary:
  - zero catastrophic failures on canary + representative
  - low mismatch rate overall
  - low mismatch rate in boundary/seam zones

4. Controlled rollout.
- Switch to custom-primary with OCC fallback.
- Fallback triggers:
  - low confidence
  - high mismatch in sampled probes
  - known risky face ids/types

5. Optimize only after correctness lock.
- Caching/memoization on custom classifier.
- Vectorized CPU paths or GPU batching.
- Reduce OCC fallback footprint iteratively.

## Required Reporting Per Iteration

Every iteration that touches classifier/tessellation logic must append:
- commit hash
- canary status
- representative status
- mismatch metrics (if shadow mode active)
- fallback usage metrics
- speed ratio vs `occt-import-js` on Electronic Enclosure

Electronic shadow reporting rule (mandatory on every run):
- Always report the latest-run value and the lowest-seen value for:
  - `effectiveMismatchCount`
  - `mismatchFromStageA`
  - `mismatchFromStageB`
  - `mismatchFromDomainUnsafeStageA`
  - `mismatchFromDomainUnsafeStageB`
- For each metric, also report:
  - delta vs lowest
  - delta vs previous run
- Use explicit labels (`current`, `lowest`, `delta_vs_lowest`, `delta_vs_previous`) to avoid ambiguity.

## Post-Compaction Rule (Mandatory)

After any context compaction/restart:
1. Read this file first.
2. Continue from the latest entry in "Correctness Results Log".
3. Run correctness checks before new optimization work.
4. Append new correctness results to this file before moving on.

## Correctness Results Log

Use this format for each new entry:

### YYYY-MM-DD HH:MM (local) - <short label>
- Commit: `<hash>`
- Change summary: `<1-2 lines>`
- Canary: `PASS/FAIL` (`x/y`)
- Representative: `PASS/FAIL` (`x/y`)
- Visual notes: `<key observations>`
- Shadow mismatch (if enabled):
  - overall: `<value>`
  - boundary band: `<value>`
  - seam-proximate: `<value>`
- OCC fallback usage:
  - faces: `<count>`
  - classify calls: `<count>`
- Electronic Enclosure:
  - ours: `<ms>`
  - reference: `<ms>`
  - speed ratio: `<ours/reference>x slower` or `<reference/ours>x faster`
- Decision: `promote / hold / rollback`

## 2026-02-15 Targeted Lever Execution (Mismatch Reduction)

Objective:
- Reduce Electronic Enclosure shadow mismatch from the current baseline by applying targeted classifier levers one at a time.
- After each lever:
  - run `npm run -s bench:representative -- --filter Electronic --classifier-shadow`
  - capture `mismatchCount`, `localUncertain`, `effectiveMismatchCount`
  - run `npm run -s bench:canary -- --classifier-shadow` as regression gate when behavior meaningfully changes

Execution order:
1. Face-level branch policy prepass (pathological cone-family routing)
2. Deterministic disagreement routing map (remove ad-hoc tie behavior)
3. Stage-B parity pass on residual bucket
4. Stage-A semantics improvement with edge-level pcurve chains
5. Remove any remaining order-dependent arbitration logic
6. Re-run strict gate (Electronic + canary shadow) and record final delta

Tracking table:

| Step | Lever | Electronic mismatch | Electronic uncertain | Electronic effectiveMismatch | Delta vs previous | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 0 | Baseline before lever execution | 10131 | 40 | 10171 | n/a | commit `2bddb4e` baseline |
| 1 | Face-level branch policy prepass | 10131 | 40 | 10171 | `0 / 0 / 0` | No measurable mismatch change on Electronic shadow run |
| 2 | Deterministic disagreement routing map | 27708 | 1953 | 29661 | `+17577 / +1913 / +19490` | Regression when disagreement map was enabled; default switched back OFF |
| 3 | Stage-B parity pass on residual bucket | 10131 | 40 | 10171 | `-17577 / -1913 / -19490` | Crossing-accumulation default changed; no net gain beyond returning to baseline |
| 4 | Stage-A edge-level semantics | 10131 | 40 | 10171 | `0 / 0 / 0` | Edge-loop Stage-A wiring landed; mismatch unchanged in shadow metric |
| 5 | Remove order-dependent arbitration | 10131 | 40 | 10171 | `0 / 0 / 0` | Refactor to deterministic helper; mismatch unchanged |
| 6 | Final gate rerun (Electronic + canary shadow) | 10131 | 40 | 10171 | `0 / 0 / 0` | Canary shadow gate PASS (`80/80`), Electronic shadow stable at baseline |

### 2026-02-15 Experiment: Domain-Unsafe Stage-B-Only (Rejected)

- Change: Force domain-unsafe path to return forced Stage-B directly (`__LOCAL_UV_DOMAIN_UNSAFE_STAGEB_ONLY__=true` default), bypassing Stage-A arbitration/provisional in that branch.
- Electronic shadow result:
  - mismatch: `15105` (was `10131`, +`4974`)
  - uncertain: `4071` (was `40`, +`4031`)
  - effectiveMismatch: `19176` (was `10171`, +`9005`)
- Source/bucket shift:
  - mismatchFromDomainUnsafeStageA: `0` (was `8856`)
  - mismatchFromDomainUnsafeStageB: `15105` (was `1275`)
  - falseInside: `7168`, falseOutside: `7937`
  - mismatchBoundaryBand: `0`, mismatchInterior: `15105`
- Perf note: Electronic runtime improved modestly (`~8916ms`, `2.71x` slower vs ref), but parity regression is too large.
- Decision: `reject as default` (keep as experimental toggle only).

## 2026-02-15 OCCT-Inspired Parity Execution (Next)

Goal:
- Reduce domain-unsafe cone mismatch by matching OCCT decision ordering and transition semantics more directly.

Current baseline snapshot (Electronic shadow):
- mismatch: `10131`
- uncertain: `40`
- effectiveMismatch: `10171`
- mismatch split: `domain_stageA=8856` (mostly boundary), `domain_stageB=1275` (interior)

Execution order:
1. Tighten domain-unsafe Stage-A confidence gate to OCCT-like boundary tolerance behavior.
2. Port closest-hit transition ownership semantics from `TopClass_Classifier2d` into Stage-B.
3. Add `CheckOn` + `CheckSkip` equivalents in Stage-B intersector path.
4. Replace Stage-B wire voting with OCCT-style global closest-hit compare across all wires for each probe segment.

Tracking:

| Step | Change | Electronic mismatch | Electronic uncertain | Electronic effectiveMismatch | Delta vs previous | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 0 | Baseline before OCCT-inspired pass | 10131 | 40 | 10171 | n/a | split: stageA=8856, stageB=1275 |
| 1 | Stage-A confidence gate tightened | 16086 | 40 | 16126 | `+5955 / +0 / +5955` | Strict gate regressed parity (stageB mismatch ballooned); kept behind flag `__LOCAL_UV_DOMAIN_UNSAFE_STRICT_STAGEA_CONFIDENCE__`, default OFF |
| 2 | Stage-B closest-hit transition parity | 10131 | 40 | 10171 | `-5955 / +0 / -5955` | Closest-hit-only bundle ownership did not change mismatch buckets after reverting strict Stage-A gate |
| 3 | Stage-B CheckOn/CheckSkip parity | 10131 | 40 | 10171 | `0 / 0 / 0` | Expanded unstable-bundle skip now triggers (`stageBBundleSkips=4`) but parity unchanged |
| 4 | Targeted inside-vs-outside Stage-A guard near boundary | 19441 | 40 | 19481 | `+9310 / +0 / +9310` | Reduced stageA mismatches but exploded stageB false-outside; flag `__LOCAL_UV_DOMAIN_UNSAFE_GUARD_INSIDE_VS_OUTSIDE__` kept OFF by default |
| 5 | Stage-B multi-probe consensus (Segment/OtherSegment-inspired) | 10131 | 40 | 10171 | `-9310 / +0 / -9310` | No parity gain; consensus path kept behind flag `__LOCAL_UV_STAGEB_MULTI_PROBE_CONSENSUS__` (default OFF) |
| 6 | Stage-B face-wide probe traversal (single direction across wires) | 10131 | 40 | 10171 | `0 / 0 / 0` | No Electronic parity gain. Strict candidate canary still fails cone/conical triangle coverage gates; kept behind `__LOCAL_UV_STAGEB_FACE_WIDE_PROBE_TRAVERSAL__` (default OFF) |
| 7 | Stage-B OCCT-style global closest-hit compare (single nearest intersection across wires) | 10131 | 40 | 10171 | `0 / 0 / 0` | Implemented face-level nearest-intersection ownership (`__LOCAL_UV_STAGEB_FACE_WIDE_PROBE_TRAVERSAL__` now default ON). No Electronic parity movement; transition-state exactness still missing |
| 8 | Stage-B TopTrans tie accumulator at closest-hit band | 10131 | 40 | 10171 | `0 / 0 / 0` | Added event-sign accumulator for equal-distance ties (`__LOCAL_UV_STAGEB_TOPTRANS_TIE_ACCUMULATOR__`), but no parity movement; left default OFF due extra CPU overhead |

## Current Baseline Snapshot (for reference)

Working diagnosis we are preserving:
- OCC-classified faces are the main runtime driver in trim classification/gating.
- Polygon-only classification is fast and not the bottleneck.
- Main near-term goal: prove custom-classifier correctness in shadow mode, then promote with fallback.

### 2026-02-14 11:40 (local) - Stage 1 candidate classifier wiring
- Commit: `<pending>`
- Change summary: Added seam-aware local UV classifier candidate path in `tessellateTrimmedSurface` with hole-aware classification and boundary-band uncertainty handling. OCC remains available as fallback for uncertain points.
- Canary: `PASS` (`80/80`)
- Representative: `PASS` (`6/6`)
- Visual notes: No catastrophic regressions observed in benchmark runs; candidate path is default-off and safety-preserving.
- Shadow mismatch (if enabled):
  - overall: `n/a`
  - boundary band: `n/a`
  - seam-proximate: `n/a`
- OCC fallback usage:
  - faces: `n/a (not instrumented in this step)`
  - classify calls: `n/a (not instrumented in this step)`
- Electronic Enclosure:
  - ours: `8386.7ms`
  - reference: `3736.7ms`
  - speed ratio: `2.24x slower`
- Decision: `hold` (proceed to Stage 2 shadow mismatch instrumentation)

### 2026-02-14 12:30 (local) - Stage 2 shadow instrumentation + benchmark plumbing
- Commit: `<pending>`
- Change summary: Added local-vs-OCC shadow summary reporting in `tessellateTrimmedSurface` and per-face/per-surface telemetry aggregation in `occ-test`; benchmark harness now supports `--classifier-shadow` and writes classifier telemetry to `benchmark-results.json`.
- Canary: `PASS` (`80/80`) [default benchmark mode]
- Representative: `PASS` (`6/6`) [shadow mode: `--classifier-shadow`]
- Visual notes: Shadow mode keeps OCC decision as source of truth for trim inclusion; no catastrophic geometry failures observed in benchmark runs.
- Shadow mismatch (if enabled):
  - overall: `38126 / 113207` grid samples (`33.68%`) on Electronic Enclosure
  - boundary band: `0 / 38126` mismatches (`0.00%`)
  - seam-proximate: `0 / 38126` mismatches (`0.00%`)
- OCC fallback usage:
  - faces: `156 patches` instrumented (Electronic Enclosure shadow run)
  - classify calls: `140299 total` (`113363` grid, `26936` triangle-gate)
- Electronic Enclosure:
  - ours: `8429.5ms`
  - reference: `3249.8ms`
  - speed ratio: `2.59x slower`
- Decision: `hold` (Stage 3 promotion criteria not met; large mismatch rate requires classifier correctness work before promotion)

### 2026-02-14 12:45 (local) - Stage 1 implementation: robust local classifier core
- Commit: `<pending>`
- Change summary: Upgraded local classifier internals with loop normalization (dedupe/closure cleanup + orientation), explicit on-edge classification, winding-number inclusion test, and interval-based periodic U remap for seam-aware point placement.
- Canary: `PASS` (`80/80`) [default benchmark mode]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Visual notes: Shadow mode kept OCC as source-of-truth output; no catastrophic geometry regressions in canary.
- Shadow mismatch (if enabled):
  - overall: `23836 / 113207` grid samples (`21.06%`) on Electronic Enclosure
  - boundary band: `0 / 23836` mismatches (`0.00%`)
  - seam-proximate: `0 / 23836` mismatches (`0.00%`)
- OCC fallback usage:
  - faces: `156 patches` instrumented
  - classify calls: `140299 total` (`113363` grid, `26936` triangle-gate)
- Electronic Enclosure:
  - ours: `8060.6ms`
  - reference: `3165.8ms`
  - speed ratio: `2.55x slower`
- Decision: `hold` (mismatch improved vs prior shadow run, but still too high for promotion)

### 2026-02-14 13:05 (local) - Stage 1 follow-up: confusion metrics + branch-consistent periodic candidate
- Commit: `<pending>`
- Change summary: Added explicit local-vs-OCC confusion counters (`falseInsideCount`, `falseOutsideCount`) and switched local periodic classification to branch-consistent multi-candidate evaluation (`u-2π`, `u`, `u+2π`) with hole-aware candidate ranking.
- Canary: `PASS` (`80/80`) [shadow mode]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Visual notes: No catastrophic regressions observed; canary shadow remains clean.
- Shadow mismatch (if enabled):
  - overall: `23836 / 113207` grid samples (`21.06%`) on Electronic Enclosure
  - boundary band: `0 / 23836` mismatches (`0.00%`)
  - seam-proximate: `0 / 23836` mismatches (`0.00%`)
  - confusion: `false-inside=23835`, `false-outside=1`
- OCC fallback usage:
  - faces: `156 patches` instrumented
  - classify calls: `140299 total` (`113363` grid, `26936` triangle-gate)
- Electronic Enclosure:
  - ours: `8887.8ms`
  - reference: `3473.9ms`
  - speed ratio: `2.56x slower`
- Decision: `hold` (new metrics show dominant cone false-inside behavior; next step is cone-loop fidelity/outer-domain correction before promotion)

### 2026-02-14 15:42 (local) - Big lever follow-up: cone polarity calibration on domain-unsafe path
- Commit: `<pending>`
- Change summary: Enabled cone local-UV polarity calibration even when classifier domain is marked unsafe; this lets Stage-B-driven cone faces auto-select direct vs inverted polarity from OCC shadow samples.
- Canary: `PASS` (`80/80`) [candidate mode, `--classifier-candidate`]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Visual notes: Candidate canary stayed stable (no catastrophic cone/cylinder regressions after this patch).
- Shadow mismatch (if enabled):
  - overall: `12808 / 113207` mismatches (`11.31%`)
  - uncertain: `4291 / 113207` (`3.79%`)
  - effective mismatch: `17099 / 113207` (`15.10%`)
  - confusion: `false-inside=2268`, `false-outside=10540`
- OCC fallback usage:
  - faces: `156 patches` instrumented
  - classify calls: `144067 total` (`117131` grid, `26936` triangle-gate)
- Electronic Enclosure:
  - ours: `8192.8ms`
  - reference: `2919.7ms`
  - speed ratio: `2.81x slower`
- Delta vs prior big-lever baseline (`mismatch=13759, uncertain=4291, effective=18050`):
  - mismatch: `-951`
  - uncertain: `unchanged`
  - effective mismatch: `-951`
- Decision: `hold` (improved parity but still cone-dominated mismatch; continue OCCT-style transition/topology alignment)

### 2026-02-14 16:05 (local) - Candidate safety gate for domain-unsafe faces
- Commit: `<pending>`
- Change summary: Kept the Stage-A-first domain-unsafe shadow logic (parity win), but in candidate mode forced domain-unsafe faces to return `uncertain` so candidate always falls back to OCC there. This preserves candidate output quality while shadow parity work continues.
- Canary: `PASS` (`80/80`) [candidate mode, `--classifier-candidate`]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Visual notes: Prior cone under-triangulation regression in candidate mode was removed; cone/conical quality gate is green again.
- Shadow mismatch (if enabled):
  - overall: `12518 / 113207` mismatches (`11.06%`)
  - uncertain: `4277 / 113207` (`3.78%`)
  - effective mismatch: `16795 / 113207` (`14.84%`)
  - confusion: `false-inside=2231`, `false-outside=10287`
- OCC fallback usage:
  - faces: domain-unsafe candidate faces now deliberately OCC-backed (safety gate)
  - classify calls: shadow run unchanged in structure; candidate run shifts domain-unsafe points to OCC fallback
- Electronic Enclosure:
  - ours: `7980.0ms`
  - reference: `3083.9ms`
  - speed ratio: `2.59x slower`
- Delta vs previous logged step (`mismatch=12808, uncertain=4291, effective=17099`):
  - mismatch: `-290`
  - uncertain: `-14`
  - effective mismatch: `-304`
- Decision: `hold` (better parity and restored candidate safety; continue reducing cone mismatches before widening candidate coverage)

### 2026-02-15 06:55 (local) - Stage 2/3 execution: cone telemetry expansion + Stage-B bad-wire ownership
- Commit: `<pending>`
- Change summary: Added cone-focused Stage-A/Stage-B telemetry (usage, trigger reason, transition-skip stats, decision-source mismatch attribution) and upgraded Stage-B traversal to edge-chain probing with OCCT-style ambiguous/bad-wire ownership in periodic evaluation.
- Canary: `skipped` (by request; fallback mode already known-good)
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Visual notes: No catastrophic geometry changes in this targeted run; shadow still uses OCC output.
- Shadow mismatch (if enabled):
  - overall: `12460 / 113207` mismatches (`11.01%`)
  - uncertain: `4232 / 113207` (`3.74%`)
  - effective mismatch: `16692 / 113207` (`14.74%`)
  - mismatch source: `stageA=550`, `stageB=4057`, `stageBForced=0`, `domain=7853`
  - Stage flow: `stageA=117057`, `stageB=33636`, `stageBForced=19936`, `badWireTriggers=0`, `stageBResolvedBadWire=0`
- OCC fallback usage:
  - classifier shadow run still compares against OCC for parity metrics
  - classify calls reported via existing telemetry path
- Electronic Enclosure:
  - ours: `8213.3ms`
  - reference: `3155.3ms`
  - speed ratio: `2.60x slower`
- Decision: `hold` (telemetry now isolates dominant mismatch source to domain-stage cone paths; next step is reducing domain-stage dependence and improving Stage-B parity there)

### 2026-02-15 (local) - Domain-unsafe path: force Stage-B ownership by default
- Commit: `<pending>`
- Change summary: Updated `classifyWithLocalUvClassifierRaw` so domain-unsafe faces now default to Stage-B forced periodic traversal (`forceStageB`) rather than Stage-A-first short-circuiting. Added runtime flag `__LOCAL_UV_DOMAIN_UNSAFE_FORCE_STAGEB__` (default `true`) to keep this behavior controllable.
- Canary: `not run` (by request)
- Representative: `not run` (by request)
- Visual notes: `n/a` (implementation-only step)
- Shadow mismatch (if enabled):
  - overall: `pending`
  - uncertain: `pending`
  - effective mismatch: `pending`
  - mismatch source: `pending`
- OCC fallback usage:
  - faces: `pending`
  - classify calls: `pending`
- Electronic Enclosure:
  - ours: `pending`
  - reference: `pending`
  - speed ratio: `pending`
- Decision: `hold` (next run should validate whether domain-stage mismatch drops without increasing uncertainty too much)

### 2026-02-15 (local) - Stage-B probe policy closer to OCCT segment selection
- Commit: `<pending>`
- Change summary: Reworked `classifyPointAgainstWireByTransitions` to prioritize a single high-quality probe direction (best non-tangent segment) with limited alternates, instead of majority voting across many directions. This better matches OCCT's robust-segment-first fallback behavior.
- Canary: `not run` (by request)
- Representative: `not run` (by request)
- Visual notes: `n/a` (implementation-only step)
- Shadow mismatch (if enabled):
  - overall: `pending`
  - uncertain: `pending`
  - effective mismatch: `pending`
  - stageB mismatch: `pending`
- OCC fallback usage:
  - faces: `pending`
  - classify calls: `pending`
- Electronic Enclosure:
  - ours: `pending`
  - reference: `pending`
  - speed ratio: `pending`
- Decision: `hold` (validate whether Stage-B false-inside drops without increasing uncertain-heavy outcomes)

### 2026-02-15 (local) - Wire-local periodic recadre for Stage A/B
- Commit: `<pending>`
- Change summary: Added wire-local periodic recadre (`__LOCAL_UV_WIRE_LOCAL_RECADRE__`, default `true`) so Stage-A and Stage-B classification tests are run in each wire's own U-band instead of a single global periodic frame. This targets domain-stage cone mismatches where wires sit in shifted periodic images.
- Canary: `not run` (by request)
- Representative: `not run` (by request)
- Visual notes: `n/a` (implementation-only step)
- Shadow mismatch (if enabled):
  - overall: `pending`
  - uncertain: `pending`
  - effective mismatch: `pending`
  - mismatchFromDomainUnsafe: `pending`
- OCC fallback usage:
  - faces: `pending`
  - classify calls: `pending`
- Electronic Enclosure:
  - ours: `pending`
  - reference: `pending`
  - speed ratio: `pending`
- Decision: `hold` (next validation should confirm whether domain-stage mismatch drops without boundary regressions)

### 2026-02-15 (local) - Stage-B complex transition tie handling
- Commit: `<working tree>`
- Change summary: Added OCCT-inspired complex transition handling in `classifyPointAgainstWireByDirection`:
  - bundle transition accumulator helper
  - disambiguation across subsequent hit bundles when closest bundle has canceling in/out transitions
  - robust bundle-wise fallback accumulation before parity fallback
- Canary: `PASS` (`80/80`) [shadow mode]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Visual notes: No catastrophic regressions observed in benchmark runs.
- Shadow mismatch (if enabled):
  - canary hotspot models:
    - `c4-surfaces/cone.step`: `mismatch=348`, `uncertain=25`, `effective=373`, source `domain=348`
    - `complex/conical-surface.step`: `mismatch=350`, `uncertain=25`, `effective=375`, source `domain=350`
    - `complex/cube.step`: `mismatch=350`, `uncertain=25`, `effective=375`, source `domain=350`
  - representative (Electronic Enclosure):
    - overall: `11917 / 113207` mismatches (`10.53%`)
    - uncertain: `4071 / 113207` (`3.60%`)
    - effective mismatch: `15988 / 113207` (`14.12%`)
    - mismatch source: `stageA=550`, `stageB=3971`, `stageBForced=0`, `domain=7396`
- OCC fallback usage:
  - faces: shadow mode still OCC-backed for source-of-truth output
  - classify calls: tracked in benchmark console output path
- Electronic Enclosure:
  - ours: `8127.9ms`
  - reference: `3162.9ms`
  - speed ratio: `2.57x slower`
- Decision: `hold` (domain-stage mismatch remains dominant; continue Stage-B/Stage-A parity alignment)

### 2026-02-15 (local) - Domain-unsafe arbitration (Stage-B primary, Stage-A confident override)
- Commit: `<working tree>`
- Change summary: Added domain-unsafe arbitration flags:
  - `__LOCAL_UV_DOMAIN_UNSAFE_ARBITRATE_STAGEA__` (default `true`)
  - `__LOCAL_UV_DOMAIN_UNSAFE_ARBITRATE_TIES_ONLY__` (default `true`)
  Forced Stage-B remains primary, but confident Stage-A (`non-uncertain`, not bad-wire, not boundary-band) can override when Stage-B is ambiguous/tie-prone (uncertain/near-boundary/transition-tie/probe-fallback).
- Canary: `PASS` (`80/80`) [shadow mode]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Visual notes: No canary regressions; conical mismatch hotspots unchanged.
- Shadow mismatch (if enabled):
  - canary hotspot models unchanged:
    - `c4-surfaces/cone.step`: `mismatch=348`, `uncertain=25`, `effective=373`
    - `complex/conical-surface.step`: `mismatch=350`, `uncertain=25`, `effective=375`
    - `complex/cube.step`: `mismatch=350`, `uncertain=25`, `effective=375`
  - representative (Electronic Enclosure):
    - overall: `11917 / 113207` mismatches (`10.53%`)
    - uncertain: `4071 / 113207` (`3.60%`)
    - effective mismatch: `15988 / 113207` (`14.12%`)
    - mismatch source: `stageA=550`, `stageB=3971`, `stageBForced=0`, `domain=7396`
- OCC fallback usage:
  - faces: shadow mode still OCC-backed for source-of-truth output
  - classify calls: tracked in benchmark console output path
- Electronic Enclosure:
  - ours: `8610.4ms`
  - reference: `2941.5ms`
  - speed ratio: `2.93x slower`
- Decision: `hold` (no parity gain observed; this patch likely needs redesign or rollback)

### 2026-02-15 (local) - Stage-B near-vertex endpoint-side transition resolution
- Commit: `<working tree>`
- Change summary: Extended Stage-B ray-hit bundles with endpoint-side metadata (`vertexRole`, `vertexOtherY`) and added `resolveComplexVertexBundle()` to classify vertex-touch vs crossing using neighboring endpoint side tests before generic transition accumulation.
- Canary: `PASS` (`80/80`) [shadow mode]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Shadow mismatch (if enabled):
  - representative (Electronic Enclosure):
    - overall: `11917 / 113207` mismatches (`10.53%`)
    - uncertain: `4071 / 113207` (`3.60%`)
    - effective mismatch: `15988 / 113207` (`14.12%`)
    - mismatch source: `stageA=550`, `stageB=3971`, `stageBForced=0`, `domain=7396`
- Electronic Enclosure:
  - ours: `8542.8ms`
  - reference: `2859.4ms`
  - speed ratio: `2.99x slower`
- Decision: `hold` (no measurable parity movement; next step must target Stage-B event semantics closer to OCCT TopTrans/transition state flow)

### 2026-02-15 (local) - Stage-B TopTrans-style event-state scan (cross/touch/ambiguous bundles)
- Commit: `<working tree>`
- Change summary: Replaced Stage-B bundle `sum+parity` fallback with explicit per-bundle event resolution:
  - `cross`: immediate in/out from transition sign
  - `touch`: deferred (continue scanning later bundles)
  - `ambiguous`: fallback pass without high-tol skip, then deferred
  - removed unconditional skip of all near-vertex-only bundles (only high-tolerance unstable bundles are skipped)
- Canary: `PASS` (`80/80`) [shadow mode]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Shadow mismatch (if enabled):
  - representative (Electronic Enclosure):
    - overall: `11953 / 113207` mismatches (`10.56%`)
    - uncertain: `4071 / 113207` (`3.60%`)
    - effective mismatch: `16024 / 113207` (`14.15%`)
    - mismatch source: `stageA=550`, `stageB=4041`, `stageBForced=0`, `domain=7362`
- Electronic Enclosure:
  - ours: `8495.5ms`
  - reference: `2931.4ms`
  - speed ratio: `2.90x slower`
- Delta vs previous shadow checkpoint (`11917/4071/15988`):
  - mismatch: `+36`
  - uncertain: `+0`
  - effective mismatch: `+36`
  - source shift: `stageB +70`, `domain -34`
- Decision: `hold` (slight parity regression despite modest runtime gain; next change should improve Stage-B sign selection on mixed-transition bundles)

### 2026-02-15 (local) - Stage-B weighted mixed-sign arbitration + Stage-A boundary/bad-wire tightening
- Commit: `<working tree>`
- Change summary:
  - Stage-B: added weighted sign arbitration for mixed-sign bundles when transition sum is non-zero (`__LOCAL_UV_STAGEB_WEIGHTED_MIXED_SIGN__`, default `true`), preferring stable/non-high-tolerance crossings.
  - Stage-A: added tighter boundary-band scale (`__LOCAL_UV_STAGEA_BOUNDARY_BAND_SCALE__`, default `0.5`) and contradictory same-orientation wire classification -> `badWire` routing to Stage-B.
- Canary: `PASS` (`80/80`) [shadow mode]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Shadow mismatch (if enabled):
  - representative (Electronic Enclosure):
    - overall: `11953 / 113207` mismatches (`10.56%`)
    - uncertain: `4071 / 113207` (`3.60%`)
    - effective mismatch: `16024 / 113207` (`14.15%`)
    - mismatch source: `stageA=550`, `stageB=4041`, `stageBForced=0`, `domain=7362`
- Electronic Enclosure:
  - ours: `8791.2ms`
  - reference: `3376.3ms`
  - speed ratio: `2.60x slower`
- Delta vs previous checkpoint (`11953/4071/16024`):
  - mismatch: `0`
  - uncertain: `0`
  - effective mismatch: `0`
  - source split: unchanged
- Decision: `hold` (no measurable parity gain; stageA bad-wire trigger still effectively dormant and stageB mixed-sign arbitration did not shift mismatch buckets)

### 2026-02-15 (local) - Domain-unsafe arbitration widened (Stage-A confident override enabled)
- Commit: `<working tree>`
- Change summary:
  - widened domain arbitration so Stage-A can override forced Stage-B beyond tie-only cases:
    - `__LOCAL_UV_DOMAIN_UNSAFE_ARBITRATE_TIES_ONLY__` default changed to `false`
  - relaxed Stage-A confidence gate from `!nearBoundaryBand` to `minBoundaryDistance > classifierPointEpsilon` (still requires non-uncertain + non-badWire)
- Canary: `PASS` (`80/80`) [shadow mode]
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Shadow mismatch (if enabled):
  - representative (Electronic Enclosure):
    - overall: `11682 / 113207` mismatches (`10.32%`)
    - uncertain: `4071 / 113207` (`3.60%`)
    - effective mismatch: `15753 / 113207` (`13.92%`)
    - mismatch source: `stageA=550`, `stageB=4041`, `stageBForced=0`, `domain=7091`
    - decision source: `domainStageA=1268`, `domainStageB=27238`
- Electronic Enclosure:
  - ours: `8568.4ms`
  - reference: `3056.2ms`
  - speed ratio: `2.80x slower`
- Delta vs previous checkpoint (`11953/4071/16024`):
  - mismatch: `-271`
  - uncertain: `+0`
  - effective mismatch: `-271`
  - source shift: `domain -271` (stageA/stageB unchanged)
- Decision: `keep` (first measurable drop in dominant domain mismatch bucket; next step is reducing Stage-B mismatch bucket)

### 2026-02-16 (local) - Stage-B TopTrans tie-accumulator experiment (no parity gain)
- Commit: `<working tree>`
- Change summary:
  - Added closest-hit tie accumulator in Stage-B face-wide traversal:
    - new flag `__LOCAL_UV_STAGEB_TOPTRANS_TIE_ACCUMULATOR__`
    - tie resolution now attempts event-sign accumulation (`cross`) before falling back to vote/uncertain
  - Result: parity metrics did not move; flag left default `false` to avoid extra CPU overhead in hot path.
- Canary: `FAIL` (`80/80` pass, quality-gate fail unchanged)
  - quality-gate failures unchanged from baseline:
    - `step-examples/c4-surfaces/cone.step` triangles/triRatio below thresholds
    - `step-examples/complex/conical-surface.step` triangles/triRatio below thresholds
- Representative: `PASS` (`1/1`) [Electronic Enclosure only, shadow mode]
- Shadow mismatch (Electronic Enclosure):
  - overall: `10131 / 126913` mismatches
  - uncertain: `40 / 126913`
  - effective mismatch: `10171 / 126913`
  - mismatch source: `stageA=0`, `stageB=0`, `stageBForced=0`, `domain=10131`
- Electronic Enclosure:
  - run A: ours `10893.2ms`, ref `4238.4ms`, speed ratio `2.57x slower`
  - run B: ours `9157.9ms`, ref `3241.7ms`, speed ratio `2.83x slower`
- Decision: `hold` (no correctness gain; keep the experiment switch available but disabled by default)
