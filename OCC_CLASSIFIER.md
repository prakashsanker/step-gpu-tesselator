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
