"""
Compute validation loss for a trained checkpoint on the val split.

Used by the training pipeline to log validation loss per checkpoint (see train.py).
Can also be run standalone to backfill val_log.csv for an existing run:

  python scripts/validation_loss.py outputs/train/<run_id> [--episode-split episode_split.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EPISODE_SPLIT_PATH = REPO_ROOT / "episode_split.json"
CHECKPOINTS_DIR = "checkpoints"
PRETRAINED_MODEL_DIR = "pretrained_model"


def compute_validation_loss(
    checkpoint_pretrained_dir: Path,
    val_episodes: list[int],
    *,
    device: str = "cuda",
    batch_size: int = 8,
    max_batches: int | None = None,
) -> float:
    """
    Load the policy and config from checkpoint_pretrained_dir, build val dataset,
    run forward passes in eval mode, return mean loss.
    """
    import torch

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint_pretrained_dir = Path(checkpoint_pretrained_dir)
    if not (checkpoint_pretrained_dir / "train_config.json").exists():
        raise FileNotFoundError(f"No train_config.json in {checkpoint_pretrained_dir}")

    # Load config from checkpoint (do not run full validate() to avoid CLI dependency)
    cfg = TrainPipelineConfig.from_pretrained(checkpoint_pretrained_dir)
    cfg.dataset.episodes = val_episodes
    cfg.policy.pretrained_path = checkpoint_pretrained_dir
    cfg.policy.device = device

    dataset = make_dataset(cfg)
    if len(dataset) == 0:
        return float("nan")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    policy = policy.to(device)
    policy.eval()

    processor_kwargs: dict = {"dataset_stats": dataset.meta.stats}
    postprocessor_kwargs: dict = {}
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            total_loss += loss.item()
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

    return total_loss / num_batches if num_batches else float("nan")


def run_validation_on_run(
    output_dir: Path,
    val_episodes: list[int],
    diagnostics_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    max_batches_per_ckpt: int | None = 20,
) -> None:
    """
    For each checkpoint in output_dir/checkpoints/, compute validation loss and append
    (step, val_loss) to diagnostics_dir/val_log.csv.
    """
    import csv

    checkpoints_base = output_dir / CHECKPOINTS_DIR
    if not checkpoints_base.exists():
        return

    step_dirs = sorted(
        (d for d in checkpoints_base.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    val_log_path = diagnostics_dir / "val_log.csv"
    existing_steps = set()
    if val_log_path.exists():
        with open(val_log_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_steps.add(int(row["step"]))

    rows = []
    for step_dir in step_dirs:
        step = int(step_dir.name)
        if step in existing_steps:
            continue
        pretrained = step_dir / PRETRAINED_MODEL_DIR
        if not (pretrained / "train_config.json").exists():
            continue
        try:
            val_loss = compute_validation_loss(
                pretrained,
                val_episodes,
                device=device,
                batch_size=batch_size,
                max_batches=max_batches_per_ckpt,
            )
            rows.append((step, val_loss))
        except Exception as e:
            print(f"Validation loss at step {step} failed: {e}", file=sys.stderr)

    if not rows:
        return

    write_header = not val_log_path.exists()
    with open(val_log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["step", "val_loss"])
        for step, val_loss in rows:
            w.writerow([step, f"{val_loss:.6f}"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute validation loss for checkpoints.")
    parser.add_argument("output_dir", type=Path, help="Training output dir (e.g. outputs/train/<run_id>)")
    parser.add_argument(
        "--episode-split",
        type=Path,
        default=EPISODE_SPLIT_PATH,
        help="Path to episode_split.json",
    )
    parser.add_argument("--diagnostics-dir", type=Path, default=None, help="Defaults to results/diagnostics/<run_id>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=20, help="Max batches per checkpoint (for speed)")
    args = parser.parse_args()

    if not args.episode_split.exists():
        print(f"Missing {args.episode_split}", file=sys.stderr)
        return 1

    with open(args.episode_split, encoding="utf-8") as f:
        split = json.load(f)
    val_episodes = split.get("val", [])
    if not val_episodes:
        print("No val episodes in split.", file=sys.stderr)
        return 0

    diagnostics_dir = args.diagnostics_dir or (REPO_ROOT / "results" / "diagnostics" / args.output_dir.name)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    run_validation_on_run(
        args.output_dir,
        val_episodes,
        diagnostics_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_batches_per_ckpt=args.max_batches,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
