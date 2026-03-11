"""
Part 2: Fine-tune SmolVLA on LIBERO Spatial Task 0 with PEFT (LoRA/QLoRA).
Reads episode_split.json, invokes lerobot-train with train episodes and PEFT flags.
Tracks Part 3 diagnostics: training loss, GPU memory, throughput (see results/diagnostics/<run_id>/).
After training, computes validation loss on the val split for each saved checkpoint and writes
results/diagnostics/<run_id>/val_log.csv; plot_diagnostics.py includes val loss in the curve.
Single command: python scripts/train.py (from ndimensions/).
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS_BASE = REPO_ROOT / "results" / "diagnostics"


def make_output_dir() -> Path:
    base = REPO_ROOT / "outputs" / "train"
    base.mkdir(parents=True, exist_ok=True)
    stem = f"smolvla_libero_task0_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out = base / stem
    if out.exists():
        out = base / f"{stem}_{int(time.time() * 1000)}"
    return out


EPISODE_SPLIT_PATH = REPO_ROOT / "episode_split.json"

# PEFT (required): LoRA config for 8–12 GB GPU

PEFT_METHOD = "LORA"
PEFT_R = 8

# Training: consumer GPU friendly
BATCH_SIZE = 48
STEPS = 10_000
SAVE_FREQ = 2_000
LOG_FREQ = 100
EVAL_FREQ = 2_000

# Part 3: diagnostic logging interval (seconds)
GPU_LOG_INTERVAL_S = 60


def _parse_step(s: str) -> int | None:
    """Parse step from LeRobot log line, e.g. step:100 or step:10.5K."""
    m = re.search(r"step:([\d.KM]+)", s)
    if not m:
        return None
    val = m.group(1).strip().upper().replace("K", "e3").replace("M", "e6")
    try:
        return int(float(val))
    except ValueError:
        return None


def _parse_loss(s: str) -> float | None:
    m = re.search(r"loss:([\d.]+)", s)
    return float(m.group(1)) if m else None


def _parse_lr(s: str) -> float | None:
    m = re.search(r"lr:([\d.e-]+)", s)
    return float(m.group(1)) if m else None


def _gpu_logger_loop(diagnostics_dir: Path, stop_event: threading.Event) -> None:
    """Background thread: log GPU memory usage to diagnostics_dir/gpu_usage.csv."""
    gpu_path = diagnostics_dir / "gpu_usage.csv"
    header_written = False
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,timestamp",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                with open(gpu_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if not header_written:
                        w.writerow(["memory_used_mb", "memory_total_mb", "timestamp"])
                        header_written = True
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            w.writerow(parts)
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        stop_event.wait(GPU_LOG_INTERVAL_S)


def main() -> int:
    if not EPISODE_SPLIT_PATH.exists():
        print(f"Missing {EPISODE_SPLIT_PATH}. Run: python scripts/analyze_dataset.py")
        return 1

    with open(EPISODE_SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)
    train_episodes = split["train"]
    episodes_arg = json.dumps(train_episodes)

    output_dir = make_output_dir()
    run_id = output_dir.name
    diagnostics_dir = DIAGNOSTICS_BASE / run_id
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        "--policy.path=HuggingFaceVLA/smolvla_libero",
        f"--dataset.repo_id=lerobot/libero_spatial_image",
        f"--dataset.episodes={episodes_arg}",
        '--rename_map={"observation.images.wrist_image": "observation.images.image2"}',
        f"--output_dir={output_dir}",
        "--job_name=smolvla_libero_task0",
        "--policy.device=cuda",
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        f"--save_freq={SAVE_FREQ}",
        f"--log_freq={LOG_FREQ}",
        f"--eval_freq={EVAL_FREQ}",
        "--save_checkpoint=true",
        f"--peft.method_type={PEFT_METHOD}",
        f"--peft.r={PEFT_R}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]

    print("Running:", " ".join(cmd[:6]), "...", cmd[6:])
    print("Part 3 diagnostics:", diagnostics_dir)

    train_log_path = diagnostics_dir / "train_log.csv"
    stop_gpu = threading.Event()
    gpu_thread = threading.Thread(target=_gpu_logger_loop, args=(diagnostics_dir, stop_gpu))
    gpu_thread.start()

    start_wall = time.perf_counter()
    with open(train_log_path, "w", newline="", encoding="utf-8") as log_f:
        writer = csv.writer(log_f)
        writer.writerow(["step", "loss", "lr", "wall_s"])

        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            step = _parse_step(line)
            loss = _parse_loss(line)
            lr = _parse_lr(line)
            if step is not None and loss is not None:
                writer.writerow([step, loss, lr if lr is not None else "", time.perf_counter() - start_wall])
                log_f.flush()

    exit_code = proc.wait()
    stop_gpu.set()
    gpu_thread.join(timeout=GPU_LOG_INTERVAL_S + 2)

    total_wall = time.perf_counter() - start_wall
    summary_path = diagnostics_dir / "throughput_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"total_wall_s,{total_wall:.1f}\n")
        f.write(f"steps,{STEPS}\n")
        f.write(f"steps_per_sec,{STEPS / total_wall:.4f}\n")
        f.write(f"exit_code,{exit_code}\n")

    # Validation loss: compute on val split for each saved checkpoint
    val_episodes = split.get("val", [])
    if val_episodes and exit_code == 0:
        try:
            # Import after training so lerobot/deps are only needed when validation runs
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "validation_loss", REPO_ROOT / "scripts" / "validation_loss.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.run_validation_on_run(
                output_dir,
                val_episodes,
                diagnostics_dir,
                batch_size=min(8, max(1, len(val_episodes))),
                max_batches_per_ckpt=25,
            )
            val_log = diagnostics_dir / "val_log.csv"
            if val_log.exists():
                print("Validation loss log:", val_log)
        except Exception as e:
            print("Validation loss tracking failed:", e, file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
