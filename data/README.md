# Data

This directory stores recorded Quest to Aero Hand demonstration data.

The default recorder output is:

```text
demo_quest_aero_sim_001.jsonl
```

## JSONL Format

Each line is one independent JSON object and can be read with `json.loads`.

Expected top-level fields:

```text
episode_id
frame_id
wall_time
t
quest
action
sim
```

`quest` contains:

```text
side
sequence_id
wrist
landmarks
```

`action` contains:

```text
aero_action_7d
mujoco_ctrl
```

`sim` contains:

```text
qpos
qvel
ctrl
```

## Quick Validation

Check that every line is valid JSON:

```bash
python -c "import json, pathlib; p=pathlib.Path('data/demo_quest_aero_sim_001.jsonl'); [json.loads(line) for line in p.open()] ; print('ok')"
```

Print the first frame:

```bash
python -c "import json; print(json.dumps(json.loads(open('data/demo_quest_aero_sim_001.jsonl').readline()), indent=2)[:2000])"
```

## Notes

The recorded `aero_action_7d` is normalized `[0, 1]` and should remain the main policy-learning interface. `mujoco_ctrl` is the model-specific control vector after applying the current Aero Hand mapping.
