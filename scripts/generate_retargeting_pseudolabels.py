import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("mujoco is required. Install it with: pip install mujoco") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.action_optimizer import ActionOptimizer
from aero_quest.mujoco_control import apply_normalized_aero_action
from aero_quest.mujoco_landmarks import get_missing_robot_landmark_sites
from aero_quest.quality_filter import build_quality_mask
from aero_quest.retargeting import quest_points_to_action_7d


DEFAULT_XML = PROJECT_ROOT / "mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MuJoCo-optimized retargeting pseudo-labels.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--w-closure", type=float, default=1.0)
    parser.add_argument("--w-prior", type=float, default=1.0)
    parser.add_argument("--w-smooth", type=float, default=0.2)
    parser.add_argument("--w-bend", type=float, default=5.0)
    parser.add_argument("--w-tip", type=float, default=0.5)
    parser.add_argument("--w-dir", type=float, default=1.0)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--finite-diff-eps", type=float, default=1e-3)
    parser.add_argument("--quality-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-warmup-frames", type=int, default=0)
    parser.add_argument("--min-palm-scale", type=float, default=0.03)
    parser.add_argument("--max-palm-scale", type=float, default=0.25)
    parser.add_argument("--min-scale-ratio", type=float, default=0.55)
    parser.add_argument("--max-scale-ratio", type=float, default=1.80)
    parser.add_argument("--min-segment-length", type=float, default=0.005)
    parser.add_argument("--min-segment-ratio", type=float, default=0.35)
    parser.add_argument("--max-segment-ratio", type=float, default=2.50)
    parser.add_argument("--max-wrist-step", type=float, default=0.15)
    parser.add_argument("--max-point-step", type=float, default=0.12)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_landmarks(path):
    data = np.load(path)
    if "P_human" in data:
        landmarks = data["P_human"]
    elif "landmarks" in data:
        landmarks = data["landmarks"]
    else:
        raise SystemExit(f"{path} must contain key 'P_human' or 'landmarks'")
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise SystemExit(f"Expected landmarks shape (T, 21, 3), got {landmarks.shape}")
    return landmarks


def compute_formula_action(p_human):
    return quest_points_to_action_7d(p_human)


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    xml_path = resolve_path(args.xml)

    raw_landmarks = load_landmarks(input_path)
    raw_count = len(raw_landmarks)
    raw_indices = np.arange(raw_count, dtype=np.int64)
    if args.quality_filter:
        quality_keep_mask, quality_reasons, quality_summary = build_quality_mask(
            raw_landmarks,
            warmup_frames=args.quality_warmup_frames,
            min_palm_scale=args.min_palm_scale,
            max_palm_scale=args.max_palm_scale,
            min_scale_ratio=args.min_scale_ratio,
            max_scale_ratio=args.max_scale_ratio,
            min_segment_length=args.min_segment_length,
            min_segment_ratio=args.min_segment_ratio,
            max_segment_ratio=args.max_segment_ratio,
            max_wrist_step=args.max_wrist_step,
            max_point_step=args.max_point_step,
        )
        landmarks = raw_landmarks[quality_keep_mask]
        source_indices = raw_indices[quality_keep_mask]
        print(
            "Quality filter: "
            f"kept={quality_summary['kept']}/{quality_summary['total']} "
            f"dropped={quality_summary['dropped']} "
            f"reference_scale={quality_summary['reference_scale']:.6f}"
        )
        for reason, count in sorted(quality_summary.items()):
            if reason in {"total", "kept", "dropped", "reference_scale", "ok"}:
                continue
            print(f"  dropped {reason}: {count}")
    else:
        quality_keep_mask = np.ones(raw_count, dtype=bool)
        quality_reasons = np.asarray(["ok"] * raw_count, dtype=object)
        landmarks = raw_landmarks
        source_indices = raw_indices
        print("Quality filter disabled.")

    if args.max_frames is not None:
        max_frames = max(0, int(args.max_frames))
        landmarks = landmarks[:max_frames]
        source_indices = source_indices[:max_frames]

    if len(landmarks) == 0:
        raise SystemExit("No landmarks remain after quality filtering. Relax thresholds or disable --quality-filter.")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    missing = get_missing_robot_landmark_sites(model)
    if missing:
        print("Missing robot landmark sites:")
        for name in missing:
            print(f"  - {name}")
        raise SystemExit(1)

    optimizer = ActionOptimizer(
        model=model,
        data=data,
        apply_action_fn=apply_normalized_aero_action,
        w_closure=args.w_closure,
        w_prior=args.w_prior,
        w_smooth=args.w_smooth,
        w_bend=args.w_bend,
        w_tip=args.w_tip,
        w_dir=args.w_dir,
        maxiter=args.maxiter,
        settle_steps=args.settle_steps,
        finite_diff_eps=args.finite_diff_eps,
    )

    a_formula_list = []
    a_opt_list = []
    delta_list = []
    loss_before_list = []
    loss_after_list = []
    success_list = []
    a_prev = None

    total = len(landmarks)
    for idx, p_human in enumerate(landmarks, start=1):
        a_formula = compute_formula_action(p_human)
        result = optimizer.optimize(p_human, a_formula, a_prev)
        a_opt = result["a_opt"]
        a_formula_list.append(result["a_formula"])
        a_opt_list.append(a_opt)
        delta_list.append(result["delta"])
        loss_before_list.append(float(result["loss_before"]["total"]))
        loss_after_list.append(float(result["loss_after"]["total"]))
        success_list.append(bool(result["success"]))
        a_prev = a_opt

        if idx % 50 == 0 or idx == total:
            success_rate = float(np.mean(success_list)) if success_list else 0.0
            print(
                f"[{idx}/{total}] "
                f"loss_before={loss_before_list[-1]:.6f} "
                f"loss_after={loss_after_list[-1]:.6f} "
                f"success_rate={success_rate:.3f}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        P_human=landmarks.astype(np.float32),
        source_index=source_indices.astype(np.int64),
        quality_keep_mask=quality_keep_mask.astype(bool),
        quality_reason=quality_reasons.astype(str),
        raw_num_frames=np.asarray(raw_count, dtype=np.int64),
        a_formula=np.asarray(a_formula_list, dtype=np.float32),
        a_opt=np.asarray(a_opt_list, dtype=np.float32),
        delta=np.asarray(delta_list, dtype=np.float32),
        loss_before=np.asarray(loss_before_list, dtype=np.float32),
        loss_after=np.asarray(loss_after_list, dtype=np.float32),
        success=np.asarray(success_list, dtype=bool),
        w_closure=np.asarray(args.w_closure, dtype=np.float32),
        w_prior=np.asarray(args.w_prior, dtype=np.float32),
        w_smooth=np.asarray(args.w_smooth, dtype=np.float32),
        w_bend=np.asarray(args.w_bend, dtype=np.float32),
        w_tip=np.asarray(args.w_tip, dtype=np.float32),
        w_dir=np.asarray(args.w_dir, dtype=np.float32),
        maxiter=np.asarray(args.maxiter, dtype=np.int32),
        settle_steps=np.asarray(args.settle_steps, dtype=np.int32),
        finite_diff_eps=np.asarray(args.finite_diff_eps, dtype=np.float32),
        quality_filter=np.asarray(args.quality_filter, dtype=bool),
        quality_warmup_frames=np.asarray(args.quality_warmup_frames, dtype=np.int32),
        min_palm_scale=np.asarray(args.min_palm_scale, dtype=np.float32),
        max_palm_scale=np.asarray(args.max_palm_scale, dtype=np.float32),
        min_scale_ratio=np.asarray(args.min_scale_ratio, dtype=np.float32),
        max_scale_ratio=np.asarray(args.max_scale_ratio, dtype=np.float32),
        min_segment_length=np.asarray(args.min_segment_length, dtype=np.float32),
        min_segment_ratio=np.asarray(args.min_segment_ratio, dtype=np.float32),
        max_segment_ratio=np.asarray(args.max_segment_ratio, dtype=np.float32),
        max_wrist_step=np.asarray(args.max_wrist_step, dtype=np.float32),
        max_point_step=np.asarray(args.max_point_step, dtype=np.float32),
    )
    print(f"Saved pseudo-labels to {output_path}")


if __name__ == "__main__":
    main()
