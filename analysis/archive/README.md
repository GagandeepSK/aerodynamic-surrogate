# Experiment Archive

This directory preserves model-development evidence that is not the active release baseline.

| Directory | Status | Purpose |
|---|---|---|
| `pre_glauert_v2/` | Historical baseline | Original v2 external-reference and calibration artefacts, plus the associated model manifest. |
| `rejected_glauert_v3/` | Rejected experiment | Global finite-thickness lift-slope trial, rejected because it worsened NASA TM 4074 attached-flow lift MAE from 0.0462 to 0.0961. |

The active selected-baseline outputs are one level above this directory:

- `analysis/naca0012_tm4074_metrics.json`
- `analysis/naca0012_tm4074_drag_calibration.json`

Archived checksums are intentionally local to each experiment directory, so they verify the preserved artefacts rather than the active release files.
