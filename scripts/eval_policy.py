"""
Part 4: Policy evaluation — ID and OOD success rates.

Runs the fine-tuned policy in LIBERO simulation under:
1. In-distribution (ID): libero_spatial Task 0 (same as training).
2. Out-of-distribution (OOD1): libero_spatial Task 1 (different object/scene).
3. Out-of-distribution (OOD2): libero_spatial Task 2 (another unseen task).

Usage:
  python scripts/eval_policy.py [CHECKPOINT_PATH]
  python scripts/eval_policy.py --checkpoint outputs/train/<run_id>/checkpoints/010000/pretrained_model

If CHECKPOINT_PATH is omitted, uses the latest training run's latest checkpoint under outputs/train/.
Writes eval outputs to outputs/eval/<eval_id>_id, _ood1, _ood2 and a summary to results/evaluation_summary/.

Headless (SSH, no display): When DISPLAY is unset, the script re-runs itself under xvfb-run so the
simulator can render to a virtual display and save outputs. Install xvfb: sudo apt-get install xvfb.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_TRAIN = REPO_ROOT / "outputs" / "train"
OUTPUTS_EVAL = REPO_ROOT / "outputs" / "eval"
RESULTS_PART4 = REPO_ROOT / "results" / "evaluation_summary"

CHECKPOINTS_DIR = "checkpoints"
PRETRAINED_MODEL_DIR = "pretrained_model"

# Eval settings (light by default for quick runs; increase n_episodes for reporting)
N_EPISODES = 10
EVAL_BATCH_SIZE = 1


def find_latest_checkpoint() -> Path | None:
    """Return path to pretrained_model of the latest step in the latest training run."""
    if not OUTPUTS_TRAIN.exists():
        return None
    run_dirs = sorted(OUTPUTS_TRAIN.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        ckpt_base = run_dir / CHECKPOINTS_DIR
        if not ckpt_base.exists():
            continue
        step_dirs = sorted(
            (d for d in ckpt_base.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
            reverse=True,
        )
        for step_dir in step_dirs:
            pretrained = step_dir / PRETRAINED_MODEL_DIR
            if pretrained.exists() and (pretrained / "config.json").exists():
                return pretrained
    return None


def run_eval(
    policy_path: Path,
    task_ids: list[int],
    output_dir: Path,
    n_episodes: int = N_EPISODES,
    batch_size: int = EVAL_BATCH_SIZE,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run lerobot-eval for libero_spatial with given task_ids."""
    task_ids_str = "[" + ",".join(str(i) for i in task_ids) + "]"
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={policy_path}",
        "--env.type=libero",
        "--env.task=libero_spatial",
        f"--env.task_ids={task_ids_str}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--output_dir={output_dir}",
    ]
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=False, env=run_env)


def load_eval_info(output_dir: Path) -> dict | None:
    """Load eval_info.json from an eval output dir."""
    path = output_dir / "eval_info.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ensure_display() -> None:
    """If no display (e.g. SSH), re-exec under xvfb-run so the sim can render and save outputs."""
    if os.environ.get("DISPLAY"):
        return
    xvfb = subprocess.run(
        ["which", "xvfb-run"],
        capture_output=True,
        text=True,
    )
    if xvfb.returncode != 0 or not xvfb.stdout.strip():
        print("No DISPLAY set and xvfb-run not found. Install: sudo apt-get install xvfb", file=sys.stderr)
        print("Then run: xvfb-run -a python scripts/eval_policy.py ...", file=sys.stderr)
        sys.exit(1)
    xvfb_path = xvfb.stdout.strip().split("\n")[0]
    os.execv(xvfb_path, [xvfb_path, "-a", "-s", "-screen 0 1024x768x24", sys.executable] + sys.argv)


def main() -> int:
    _ensure_display()

    parser = argparse.ArgumentParser(description="Part 4: Evaluate policy (ID + 2 OOD conditions).")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to pretrained_model dir (e.g. outputs/train/.../checkpoints/010000/pretrained_model). Default: latest.",
    )
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES, help=f"Episodes per condition (default {N_EPISODES}).")
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE, help=f"Eval batch size (default {EVAL_BATCH_SIZE}).")
    args = parser.parse_args()

    if args.checkpoint:
        policy_path = Path(args.checkpoint)
        if not policy_path.is_dir() or not (policy_path / "config.json").exists():
            print(f"Invalid checkpoint: {policy_path} (expected dir with config.json)")
            return 1
    else:
        policy_path = find_latest_checkpoint()
        if policy_path is None:
            print("No checkpoint found under outputs/train/. Train first or pass CHECKPOINT_PATH.")
            return 1
        print("Using latest checkpoint:", policy_path)

    OUTPUTS_EVAL.mkdir(parents=True, exist_ok=True)
    RESULTS_PART4.mkdir(parents=True, exist_ok=True)

    # Unique eval run id (from checkpoint path: .../train/<run_id>/checkpoints/<step>/pretrained_model)
    try:
        parts = policy_path.resolve().parts
        run_id = "unknown"
        for i, p in enumerate(parts):
            if p == "train" and i + 1 < len(parts):
                run_id = parts[i + 1]
                break
        step = policy_path.parent.name if policy_path.parent.name.isdigit() else "?"
        eval_stem = f"{run_id}_step{step}"
    except Exception:
        eval_stem = "eval_run"

    conditions = [
        ("id", [0], "In-distribution (Task 0)"),
        ("ood1", [1], "OOD: libero_spatial Task 1"),
        ("ood2", [2], "OOD: libero_spatial Task 2"),
    ]

    results = {}
    for key, task_ids, label in conditions:
        output_dir = OUTPUTS_EVAL / f"{eval_stem}_{key}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {label} (task_ids={task_ids}) -> {output_dir}")
        ret = run_eval(
            policy_path,
            task_ids,
            output_dir,
            n_episodes=args.n_episodes,
            batch_size=args.batch_size,
        )
        if ret.returncode != 0:
            print(f"Eval failed for {key} (exit {ret.returncode}). Continuing.")
        info = load_eval_info(output_dir)
        results[key] = {
            "label": label,
            "task_ids": task_ids,
            "output_dir": str(output_dir),
            "eval_info": info,
            "exit_code": ret.returncode,
        }

    # Summary report
    summary_lines = [
        "# Part 4: Policy Evaluation — ID vs OOD",
        "",
        f"**Checkpoint:** `{policy_path}`",
        f"**Episodes per condition:** {args.n_episodes}",
        "",
        "## Task success rates",
        "",
        "| Condition | Description | Success rate (%) | Notes |",
        "|-----------|-------------|------------------|-------|",
    ]

    for key, task_ids, label in conditions:
        r = results[key]
        info = r["eval_info"]
        if info and "overall" in info:
            pc = info["overall"].get("pc_success")
            pc_str = f"{pc:.1f}" if pc is not None and not (isinstance(pc, float) and (pc != pc)) else "N/A"
        else:
            pc_str = "N/A (run failed or no eval_info)"
        summary_lines.append(f"| {key} | {label} | {pc_str} | - |")

    summary_lines.extend([
        "",
        "## Comparison",
        "",
        "In-distribution (ID) uses the same task as training (libero_spatial Task 0). "
        "OOD1 and OOD2 use unseen tasks (Task 1 and Task 2) with different object positions and goals. "
        "Lower success on OOD conditions indicates limited generalization to new scenes.",
        "",
    ])

    # Add numeric comparison if we have both ID and OOD rates
    id_info = results.get("id", {}).get("eval_info")
    if id_info and id_info.get("overall") is not None:
        id_pc = id_info["overall"].get("pc_success")
        for key in ("ood1", "ood2"):
            ood_info = results.get(key, {}).get("eval_info")
            if ood_info and ood_info.get("overall") is not None and id_pc is not None:
                ood_pc = ood_info["overall"].get("pc_success")
                if ood_pc is not None:
                    drop = id_pc - ood_pc
                    summary_lines.append(f"- **ID vs {key}:** {id_pc:.1f}% → {ood_pc:.1f}% (Δ = {drop:+.1f}%).")
        summary_lines.append("")

    summary_md = "\n".join(summary_lines)
    summary_path = RESULTS_PART4 / f"{eval_stem}_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print("\nSummary written to:", summary_path)

    # Persist raw results JSON for scripting
    results_path = RESULTS_PART4 / f"{eval_stem}_results.json"
    out = {}
    for k, v in results.items():
        out[k] = {
            "label": v["label"],
            "task_ids": v["task_ids"],
            "output_dir": v["output_dir"],
            "exit_code": v["exit_code"],
            "pc_success": None,
        }
        if v.get("eval_info") and v["eval_info"].get("overall"):
            out[k]["pc_success"] = v["eval_info"]["overall"].get("pc_success")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Results JSON:", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
