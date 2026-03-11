"""
Part 3: Training diagnostics – plot training loss, GPU usage, and report throughput.
Usage: python scripts/plot_diagnostics.py [run_id]
  If run_id is omitted, uses the latest run in results/diagnostics/.
Outputs: results/diagnostics/<run_id>/train_loss.png, gpu_memory.png, diagnostics_summary.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS_BASE = REPO_ROOT / "results" / "diagnostics"


def get_run_dir(run_id: str | None) -> Path | None:
    if run_id:
        d = DIAGNOSTICS_BASE / run_id
        return d if d.is_dir() else None
    if not DIAGNOSTICS_BASE.is_dir():
        return None
    runs = sorted(DIAGNOSTICS_BASE.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def main() -> int:
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = get_run_dir(run_id)
    if not run_dir:
        print("No diagnostics run found. Run training first: python scripts/train.py")
        return 1

    print("Using run:", run_dir.name)

    # Load train log
    train_log_path = run_dir / "train_log.csv"
    steps, losses, lrs, wall_times = [], [], [], []
    if train_log_path.exists():
        with open(train_log_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    steps.append(int(row["step"]))
                    losses.append(float(row["loss"]))
                    lrs.append(float(row["lr"]) if row.get("lr") else None)
                    wall_times.append(float(row["wall_s"]) if row.get("wall_s") else None)
                except (KeyError, ValueError):
                    continue

    # Load GPU log
    gpu_path = run_dir / "gpu_usage.csv"
    gpu_times, gpu_used, gpu_total = [], [], []
    if gpu_path.exists():
        with open(gpu_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    gpu_used.append(float(row["memory_used_mb"]))
                    gpu_total.append(float(row["memory_total_mb"]))
                    gpu_times.append(row.get("timestamp", ""))
                except (KeyError, ValueError):
                    continue
        if gpu_times and not gpu_times[0].replace(" ", "").replace(":", "").isdigit():
            gpu_times = list(range(len(gpu_used)))

    # Load validation loss log (step, val_loss per checkpoint)
    val_steps, val_losses = [], []
    val_log_path = run_dir / "val_log.csv"
    if val_log_path.exists():
        with open(val_log_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    val_steps.append(int(row["step"]))
                    val_losses.append(float(row["val_loss"]))
                except (KeyError, ValueError):
                    continue

    # Load throughput summary
    throughput_path = run_dir / "throughput_summary.txt"
    throughput_lines = []
    if throughput_path.exists():
        with open(throughput_path, encoding="utf-8") as f:
            throughput_lines = f.read().strip().split("\n")

    # Plots
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if steps and losses:
        axes[0].plot(steps, losses, "b-", alpha=0.8, label="Train loss")
        if val_steps and val_losses:
            axes[0].plot(val_steps, val_losses, "r-o", alpha=0.8, markersize=6, label="Val loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and validation loss")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(True, alpha=0.3)
    else:
        if val_steps and val_losses:
            axes[0].plot(val_steps, val_losses, "r-o", alpha=0.8, markersize=6, label="Val loss")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Val loss")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No train_log.csv or val_log.csv data", ha="center", va="center")
        axes[0].set_title("Training and validation loss")

    if gpu_used:
        x = range(len(gpu_used))
        axes[1].plot(x, gpu_used, "g-", label="Used (MB)", alpha=0.8)
        if gpu_total:
            axes[1].plot(x, gpu_total, "gray", linestyle="--", label="Total (MB)", alpha=0.6)
        axes[1].set_xlabel("Sample (every 60s)")
        axes[1].set_ylabel("GPU memory (MB)")
        axes[1].set_title("GPU memory usage")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No gpu_usage.csv data", ha="center", va="center")
        axes[1].set_title("GPU memory usage")

    plt.tight_layout()
    plot_path = run_dir / "diagnostics_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print("Saved", plot_path)

    # Summary markdown for Part 3 writeup
    summary_md = run_dir / "diagnostics_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Part 3: Training diagnostics summary\n\n")
        f.write(f"**Run ID:** `{run_dir.name}`\n\n")
        f.write("## Training loss curve\n\n")
        if steps and losses:
            f.write("- Plot: `diagnostics_plots.png` (left panel).\n")
            f.write(f"- Logged points: {len(steps)} (every `log_freq` steps).\n")
            f.write(f"- Final loss (last logged): {losses[-1]:.4f}\n\n")
        else:
            f.write("- No train log data.\n\n")
        f.write("## Validation loss\n\n")
        if val_steps and val_losses:
            f.write("- Plot: `diagnostics_plots.png` (left panel, red points).\n")
            f.write(f"- Checkpoints evaluated: {len(val_steps)} (steps: {val_steps}).\n")
            f.write(f"- Final val loss (last checkpoint): {val_losses[-1]:.4f}\n\n")
        else:
            f.write("- No `val_log.csv` (validation loss is computed after training on the val split).\n\n")
        f.write("## GPU memory usage\n\n")
        if gpu_used:
            f.write(f"- Plot: `diagnostics_plots.png` (right panel).\n")
            f.write(f"- Peak memory used: {max(gpu_used):.0f} MB\n")
            if gpu_total:
                f.write(f"- Total GPU memory: {gpu_total[0]:.0f} MB\n\n")
        else:
            f.write("- No GPU log (nvidia-smi may be unavailable or not run).\n\n")
        f.write("## Throughput\n\n")
        for line in throughput_lines:
            if "steps_per_sec" in line:
                _, val = line.split(",", 1)
                f.write(f"- **Steps per second:** {val.strip()}\n")
            elif "total_wall_s" in line:
                _, val = line.split(",", 1)
                f.write(f"- **Total wall time (s):** {val.strip()}\n")
        f.write("\n## Convergence / overfitting / underfitting\n\n")
        f.write("- *Fill in after reviewing the loss curve: Did loss decrease and stabilize? Any rise in loss (overfitting) or plateau at high loss (underfitting)?*\n")
    print("Saved", summary_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
