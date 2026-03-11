# Part 3: Training diagnostics summary

**Run ID:** `smolvla_libero_task0_2026-03-10_15-08-37`

## Training loss curve

- Plot: `diagnostics_plots.png` (left panel).
- Logged points: 100 (every `log_freq` steps).
- Final loss (last logged): 0.0730

## Validation metrics

- LeRobot offline training does not run env evaluation by default (no eval env set).
- To add validation loss or success rate, configure `--env.type` and `--eval.freq` or inspect checkpoint eval logs if present.

## GPU memory usage

- Plot: `diagnostics_plots.png` (right panel).
- Peak memory used: 4851 MB
- Total GPU memory: 24564 MB

## Throughput

- **Total wall time (s):** 2787.1
- **Steps per second:** 3.5880

## Convergence / overfitting / underfitting

- *Fill in after reviewing the loss curve: Did loss decrease and stabilize? Any rise in loss (overfitting) or plateau at high loss (underfitting)?*
