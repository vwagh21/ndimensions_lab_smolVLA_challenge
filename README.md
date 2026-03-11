# ndimensions lab: take home

Fine-tune SmolVLA on LIBERO Spatial Task 0 with LoRA, then evaluate in-distribution and out-of-distribution.
**Checkpoint:** [vwagh21/ndim_smolVLA](https://huggingface.co/vwagh21/ndim_smolVLA)

Summary of evaluation results, training plots, and videos demonstrating successful executions and failure cases in the results folder.

### Demo videos

<table>
  <tr>
    <td><b>Run 19-23-30, episode 4</b></td>
    <td><b>Run 20-04-44, episode 0</b></td>
  </tr>
  <tr>
    <td><video src="outputs/eval/smolvla_libero_task0_2026-03-10_19-23-30_step010000_id/videos/libero_spatial_0/eval_episode_4.mp4" controls width="320"></video></td>
    <td><video src="outputs/eval/smolvla_libero_task0_2026-03-10_20-04-44_step010000_id/videos/libero_spatial_0/eval_episode_0.mp4" controls width="320"></video></td>
  </tr>
</table>

## Setup

Requires Python 3.12 (not 3.14+) and a CUDA GPU.

```bash
conda create -n ndim python=3.12 pip -y
conda activate ndim
pip install -r requirements.txt
pip install robomimic==0.2.0 robosuite==1.4.0 hf-libero==0.1.3 --no-deps
pip install hf-egl-probe --no-build-isolation
```

The `--no-deps` / `--no-build-isolation` flags avoid a C extension build failure in `egl_probe`. See `requirements.txt` header for details.

## Part 1 : Dataset analysis

```bash
python scripts/analyze_dataset.py
```

Loads `lerobot/libero_spatial_image` Task 0, writes observation/action space info to `results/dataset_analysis.md`, saves the train/val split to `episode_split.json` (38 train, 5 val).

## Part 2 : Fine-tuning

```bash
python scripts/train.py
```

Fine-tunes `HuggingFaceVLA/smolvla_libero` with LoRA (rank 8) for 10k steps. Checkpoints go to `outputs/train/<run_id>/`. Training loss, GPU memory, and throughput are logged to `results/diagnostics/<run_id>/`. Validation loss is computed automatically after training.

Config (batch size, steps, LoRA rank, etc.) is set at the top of `scripts/train.py`.

## Part 3 : Diagnostics

```bash
python scripts/plot_diagnostics.py [run_id]
```

Generates training loss curve (train + val), GPU memory plot, and a summary. Omit `run_id` to use the latest run.

To backfill validation loss for an existing run:

```bash
python scripts/validation_loss.py outputs/train/<run_id>
```

## Part 4 : Evaluation

```bash
python scripts/eval_policy.py [checkpoint_path]
```

Runs the fine-tuned policy in LIBERO simulation across three conditions:

- **ID** : `libero_spatial` Task 0 (same as training)
- **OOD1** : `libero_spatial` Task 1
- **OOD2** : `libero_spatial` Task 2

Omit `checkpoint_path` to auto-detect the latest checkpoint. Results and a summary go to `results/evaluation_summary/`.

For headless/SSH environments, install `xvfb` (`sudo apt install xvfb`) : the script wraps itself automatically.

## Project structure

```
requirements.txt          # all dependencies (single file)
episode_split.json        # train/val episode IDs
scripts/
  analyze_dataset.py      # Part 1
  train.py                # Part 2
  plot_diagnostics.py     # Part 3
  validation_loss.py      # Part 3 (val loss helper)
  eval_policy.py          # Part 4
outputs/                  # checkpoints + eval videos
results/                  # analysis, diagnostics, eval summaries
```
