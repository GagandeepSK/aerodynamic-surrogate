# Rejected global thickness experiment

## Change tested

Applied `Cl = 2π(1 + 0.77t)(α − α0)` globally in source generation, physics loss, and dashboard fallback.

## External result

NACA 0012, NASA TM 4074, Re=6e6, grit-fixed transition:

| Regime | Selected 2π baseline Cl MAE | Global thickness experiment Cl MAE |
|---|---:|---:|
| Attached flow, 12 points | 0.0462 | 0.0961 |
| Overall, 14 points | 0.0670 | 0.1111 |

The trial improved the zero-lift point but over-predicted lift throughout most of attached flow. Apparent recovery near 14° was caused by the stall blend rather than an improved lift slope.

## Decision

Reject the global multiplier and retain the classical `2π(α − α0)` source slope. This experiment is archived for audit traceability, not release. Before source transfer, archive the exact EC2 checkpoint, model JSON, dashboard, metrics, and calibration JSON in this directory.
