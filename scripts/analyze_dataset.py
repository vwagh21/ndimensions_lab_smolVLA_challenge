"""
Part 1: Dataset analysis for ndimensions SmolVLA challenge.
Loads libero_spatial_image Task 0, describes observation/action space,
trajectory counts, task descriptions, and writes train/val split.
Run from repo root: python scripts/analyze_dataset.py
"""
from pathlib import Path
import json

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Task 0 = first 43 episodes in libero_spatial_image
TASK0_EPISODES = list(range(43))
N_TRAIN = 38
N_VAL = 5

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("Loading lerobot/libero_spatial_image (Task 0)...")
    dataset = LeRobotDataset(
        "lerobot/libero_spatial_image",
        episodes=TASK0_EPISODES,
    )

    # --- Episode lengths ---
    lengths = []
    for i in range(dataset.num_episodes):
        ep = dataset.meta.episodes[i]
        length = ep["dataset_to_index"] - ep["dataset_from_index"]
        lengths.append(length)

    n_episodes = dataset.num_episodes
    n_frames = dataset.num_frames
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    # --- Observation space (from meta.features / meta.shapes) ---
    features = dataset.meta.features
    shapes = dataset.meta.shapes
    observation_keys = [k for k in features if k.startswith("observation.")]
    action_keys = [k for k in features if k.startswith("action") or k == "action"]

    def fmt_shape(shape):
        if shape == "?":
            return shape
        if hasattr(shape, "__iter__") and not isinstance(shape, str):
            return tuple(shape)
        return shape

    obs_lines = []
    for key in sorted(observation_keys):
        shape = fmt_shape(shapes.get(key, features[key].get("shape", "?")))
        dtype = features[key].get("dtype", "?")
        obs_lines.append(f"- **{key}**: shape `{shape}`, dtype `{dtype}`")

    action_lines = []
    for key in sorted(action_keys):
        shape = fmt_shape(shapes.get(key, features[key].get("shape", "?")))
        dtype = features[key].get("dtype", "?")
        action_lines.append(f"- **{key}**: shape `{shape}`, dtype `{dtype}`")

    # --- Task descriptions ---
    task_descriptions = []
    if dataset.meta.tasks is not None:
        for task_name in dataset.meta.tasks.index:
            task_idx = dataset.meta.tasks.loc[task_name, "task_index"]
            task_descriptions.append(f"- **Task {task_idx}**: {task_name}")
    else:
        task_descriptions.append("- (no task metadata; using Task 0 subset of LIBERO Spatial)")

    # --- Train/val split ---
    train_episodes = TASK0_EPISODES[:N_TRAIN]
    val_episodes = TASK0_EPISODES[N_TRAIN : N_TRAIN + N_VAL]
    split = {"train": train_episodes, "val": val_episodes}

    # --- Write dataset_analysis.md ---
    md = f"""# Dataset analysis (Part 1)

**Dataset:** `lerobot/libero_spatial_image`  
**Subset:** Task 0 only (episodes 0–{TASK0_EPISODES[-1]}).

## Observation space

{chr(10).join(obs_lines)}

(Images are 256×256 RGB; state is 8-D: x, y, z, quaternion rx, ry, rz, rw, gripper.)

## Action space

{chr(10).join(action_lines)}

(7-D continuous: x, y, z, roll, pitch, yaw, gripper; typically in [-1, 1].)

## Number of trajectories

| Metric | Value |
|--------|-------|
| Episodes (trajectories) | {n_episodes} |
| Total frames | {n_frames} |
| Mean episode length (frames) | {avg_len:.1f} |
| Min / max episode length | {min_len} / {max_len} |

## Task descriptions

{chr(10).join(task_descriptions)}

We use **Task 0** only for training and validation.

## Train/validation split

| Split | Episode indices | Count |
|-------|-----------------|-------|
| Train | 0–{N_TRAIN - 1} | {N_TRAIN} |
| Val   | {N_TRAIN}–{N_TRAIN + N_VAL - 1} | {N_VAL} |

Split is saved in `episode_split.json` for use in training and evaluation.
"""
    out_md = RESULTS_DIR / "dataset_analysis.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_md}")

    # --- Write episode_split.json ---
    split_path = REPO_ROOT / "episode_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"Wrote {split_path}")

    # --- Episode length plot ---
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(lengths, bins=20)
        plt.xlabel("Episode length (frames)")
        plt.ylabel("Count")
        plt.title("Episode length distribution - Task 0")
        plot_path = RESULTS_DIR / "episode_lengths.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Wrote {plot_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")

    # --- Console summary ---
    print("\n--- Summary ---")
    print(f"Episodes: {n_episodes}  Frames: {n_frames}")
    print(f"Train: {len(train_episodes)}  Val: {len(val_episodes)}")
    print(f"Observation keys: {observation_keys}")
    print(f"Action keys: {action_keys}")


if __name__ == "__main__":
    main()
