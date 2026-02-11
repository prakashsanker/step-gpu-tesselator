# Cone Seam Face Diff Checkpoint (2026-02-11)

This checkpoint captures the targeted seam-face validation for cone faces `63,64,65,66` on branch `codex/occt-face-diff-harness`.

## Goal

Verify whether the current seam-loop labeling/split fix is converging to reference behavior (`occt-import-js`) on the problematic cone seam faces.

## Harness used

- Script: `tests/run-face-diff-harness.mjs`
- Browser API: `window.visualValidation.runFaceDiff(...)`
- STEP file: `step-examples/complex/electronicEnclosure.step`
- Faces: `63,64,65,66`
- Feature flags during run:
  - `__ENABLE_CONE_SEAM_SPLIT__ = true`
  - `__CONE_SEAM_SPLIT_FACE_IDS__ = [63,64,65,66]`
  - `__FACE_DEBUG_MODE__ = 'only'`
  - `__FACE_DEBUG_IDS__ = [63,64,65,66]`

## Raw report

- JSON: `diagnostics/face-diff/face-diff-report-face63-66-2026-02-11.json`

## Result summary

The seam fix **did not work** for these faces. Triangle counts remain far above reference:

- Face `63`: ours `6996` vs ref `725` (`+864.97%`)
- Face `64`: ours `3328` vs ref `397` (`+738.29%`)
- Face `65`: ours `32312` vs ref `339` (`+9431.56%`)
- Face `66`: ours `6750` vs ref `690` (`+878.26%`)

Totals over target faces:

- Ours: `49386` triangles
- Reference: `2151` triangles

## Observations from logs

- Hole-heavy loop sets are still present on seam faces (`63` and `66`).
- Face `63` split produced a pathological patch (`outer=7`, `holes=11`) on one side.
- Face `65` skipped seam split (`crossesSeam=false`) but still exploded in triangle count.

## Decision

Move to an OCCT-inspired path, using native OCC face triangulation behavior as the baseline reference to port before GPU-specific optimization.
