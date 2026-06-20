# Quest 3 Hand Tracking -> AeroHand MuJoCo Retargeting / Pinch Control Pipeline

This subproject is an initial research scaffold for Quest 3 hand tracking to AeroHand retargeting in MuJoCo. It is intentionally lightweight: no complex optimizer, no neural network training, and no dependency on a live Quest device.

The goal is to make the pipeline, module boundaries, data contracts, baselines, and TODOs explicit before adding heavier algorithms.

## Pipeline

```text
Quest 3 / Hand Tracking Streamer
        ↓
Raw hand landmarks
        ↓
Preprocessing: filtering + coordinate normalization + pinch features
        ↓
Human ghost skeleton visualization
        ↓
Baseline 1 / Baseline 2 / Baseline 3 retargeting
        ↓
AeroHand MuJoCo simulation
        ↓
Sim residual correction
        ↓
Evaluation and plots
```

## Modules

- `quest_io`: frame schema, mock HTS receiver, JSONL logger, JSONL loader.
- `preprocessing`: coordinate normalization, low-pass filtering, pinch features.
- `visualization`: Quest 3 21-point hand skeleton and MuJoCo AeroHand landmark skeleton plotting.
- `retargeting`: three baseline interfaces for direct, vector-optimized, and pinch-aware retargeting.
- `sim`: AeroHand MuJoCo wrapper that applies 7D AeroHand actions and reads real fingertip and 21-point landmark sites, with a placeholder fallback when no model path is configured.
- `residual`: zero-residual scaffold for simulation residual correction.
- `evaluation`: metrics and plotting placeholders.
- `utils`: config and math helpers.

## Retargeting Methods

- **Baseline 1: Direct Pose Mapping** maps Quest landmarks/features to a compact 7D AeroHand action. With full 21 wrist-local landmarks, it now reuses the main repo's existing `aero_quest.retargeting.quest_points_to_action_7d`; compact pinch/curl heuristics are only a fallback for minimal frames.
- **Baseline 2: Optimized Vector Retargeting** samples 7D AeroHand actions through MuJoCo and minimizes `L_vector + L_smooth + L_limit` over five-fingertip shape.
- **Baseline 3: Contact-aware / Pinch-aware Retargeting** extends Baseline 2 with `lambda_pinch * L_pinch` plus a close-distance term when Quest thumb/index pinch strength is high.
- **Proposed: Baseline 3 + Sim Residual Correction** adds `delta_u` from simulated fingertip/contact feedback. The current residual model returns zeros and documents the future observation contract.

## Current Version

Implemented now:

- A minimal hand frame schema with the requested fields.
- Optional full 21-point `landmarks_wrist` on hand frames.
- Mock thumb-index pinch trajectory generation.
- JSONL record/replay utilities.
- Coordinate, filtering, and pinch feature function skeletons.
- Three retargeting baseline classes; Baseline 1 uses the existing AeroHand 7D geometric retargeter when full landmarks are present.
- AeroHand simulation wrapper using MuJoCo fingertip sites and full `aero_*_lm` 21-point hand landmark sites when a model path is configured.
- Offline Quest 3 vs MuJoCo AeroHand skeleton comparison plots.
- Zero residual correction class.
- Metrics for pinch distance, robot distance, fingertip error, success rates, smoothness, and latency.
- A runnable placeholder demo that saves metrics.

Not implemented yet:

- Real Quest 3 / Hand Tracking Streamer connection in this subproject.
- Real optimized vector retargeting.
- Real contact-aware loss using MuJoCo contact pairs.
- Neural residual model training.
- Object slip metrics from task objects.

## Run Placeholder Demo

From the repository root:

```bash
python quest_aerohand_retargeting/scripts/evaluate_baselines.py --config quest_aerohand_retargeting/configs/default.yaml
```

The demo generates a mock pinch trajectory, runs the three baselines plus the zero-residual proposed method, prints a metric summary, and writes JSON outputs under `quest_aerohand_retargeting/data/processed/`.

Run Baseline 1 on a recorded Quest dual-channel JSONL file:

```bash
python quest_aerohand_retargeting/scripts/evaluate_recorded_baseline1.py --input data/teleop_episodes/baseline1_test.jsonl
```

This evaluates Baseline 1 and Baseline 1b by default, then writes per-frame actions, metrics, and quick plots under `quest_aerohand_retargeting/data/processed/baseline1_recorded/`. Baseline 2 and Baseline 3 are slower because they evaluate sampled actions through MuJoCo; use `--method baseline3 --max-frames 200` for quick pinch-aware tuning.

By default, the scaffold now uses `models/piper_aero_hand/Piper_aerohand.xml` for recorded evaluation, so `robot_thumb_index_distance` is measured from MuJoCo fingertip sites instead of the old placeholder distance formula.

Save a Quest 3 21-point skeleton vs MuJoCo AeroHand 21-site skeleton comparison:

```bash
python quest_aerohand_retargeting/scripts/visualize_quest_aero_skeletons.py \
  --input data/teleop_episodes/baseline1_test.jsonl \
  --method baseline1b \
  --num-frames 6
```

The output defaults to `quest_aerohand_retargeting/data/processed/skeleton_compare/baseline1b_skeletons.png`. Quest landmarks remain wrist-local, while AeroHand sites are read from MuJoCo/world coordinates; the plot normalizes each skeleton separately to compare hand shape rather than pretending the frames are identical.

## Relation To Existing Repo

The main repository already contains a more mature live teleoperation path in `aero_quest/` and `scripts/teleop/`. This scaffold is a research-facing layer for baseline comparison and future pinch/contact/residual experiments. Later integration points:

- Replace `quest_io.hts_receiver.MockHTSReceiver` with `aero_quest.quest_receiver.QuestTelemetryReceiver`.
- Replace compact placeholder action logic with `aero_quest.retargeting.quest_points_to_action_7d` where full 21 landmarks are available.
- Replace `sim.aerohand_env.AeroHandSimEnv` placeholder with the existing AeroHand MuJoCo model wrapper.

## More Detail

For the exact Baseline action -> MuJoCo ctrl -> fingertip site distance flow, see:

```text
quest_aerohand_retargeting/MUJOCO_SIM_INTEGRATION.zh-CN.md
```
