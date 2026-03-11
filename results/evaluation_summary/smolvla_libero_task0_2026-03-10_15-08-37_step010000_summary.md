# Part 4: Policy Evaluation — ID vs OOD

**Checkpoint:** `/home/metamobility2/Vaidehi/ndimensions/outputs/train/smolvla_libero_task0_2026-03-10_15-08-37/checkpoints/010000/pretrained_model`
**Episodes per condition:** 10

## Task success rates

| Condition | Description | Success rate (%) | Notes |
|-----------|-------------|------------------|-------|
| id | In-distribution (Task 0) | 40.0 | - |
| ood1 | OOD: libero_spatial Task 1 | 80.0 | - |
| ood2 | OOD: libero_spatial Task 2 | 10.0 | - |

## Comparison

In-distribution (ID) uses the same task as training (libero_spatial Task 0). OOD1 and OOD2 use unseen tasks (Task 1 and Task 2) with different object positions and goals. Lower success on OOD conditions indicates limited generalization to new scenes.

- **ID vs ood1:** 40.0% → 80.0% (Δ = -40.0%).
- **ID vs ood2:** 40.0% → 10.0% (Δ = +30.0%).
