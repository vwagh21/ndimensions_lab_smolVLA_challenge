# Dataset analysis (Part 1)

**Dataset:** `lerobot/libero_spatial_image`  
**Subset:** Task 0 only (episodes 0–42).

## Observation space

- **observation.images.image**: shape `(256, 256, 3)`, dtype `image`
- **observation.images.wrist_image**: shape `(256, 256, 3)`, dtype `image`
- **observation.state**: shape `(8,)`, dtype `float32`

(Images are 256×256 RGB; state is 8-D: x, y, z, quaternion rx, ry, rz, rw, gripper.)

## Action space

- **action**: shape `(7,)`, dtype `float32`

(7-D continuous: x, y, z, roll, pitch, yaw, gripper; typically in [-1, 1].)

## Number of trajectories

| Metric | Value |
|--------|-------|
| Episodes (trajectories) | 43 |
| Total frames | 5192 |
| Mean episode length (frames) | 120.7 |
| Min / max episode length | 84 / 164 |

## Task descriptions

- **Task 0**: pick up the black bowl next to the cookie box and place it on the plate
- **Task 1**: pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
- **Task 2**: pick up the black bowl on the ramekin and place it on the plate
- **Task 3**: pick up the black bowl on the stove and place it on the plate
- **Task 4**: pick up the black bowl between the plate and the ramekin and place it on the plate
- **Task 5**: pick up the black bowl on the cookie box and place it on the plate
- **Task 6**: pick up the black bowl next to the plate and place it on the plate
- **Task 7**: pick up the black bowl next to the ramekin and place it on the plate
- **Task 8**: pick up the black bowl from table center and place it on the plate
- **Task 9**: pick up the black bowl on the wooden cabinet and place it on the plate

We use **Task 0** only for training and validation.

## Train/validation split

| Split | Episode indices | Count |
|-------|-----------------|-------|
| Train | 0–37 | 38 |
| Val   | 38–42 | 5 |

Split is saved in `episode_split.json` for use in training and evaluation.
