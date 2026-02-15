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
