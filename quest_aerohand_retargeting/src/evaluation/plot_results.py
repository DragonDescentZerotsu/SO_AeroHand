from __future__ import annotations

import os


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def plot_pinch_distances(records: list[dict], output_path: str) -> None:
    """Plot human and robot pinch distances."""
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("No records to plot")
    t = [record["timestamp"] for record in records]
    t0 = t[0]
    t = [value - t0 for value in t]
    human = [record["human_pinch_distance"] for record in records]
    robot = [record["robot_pinch_distance"] for record in records]
    plt.figure(figsize=(9, 4))
    plt.plot(t, human, label="human pinch distance")
    plt.plot(t, robot, label="robot pinch distance")
    plt.xlabel("time (s)")
    plt.ylabel("distance")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_action_curves(records: list[dict], output_path: str, action_names: list[str] | tuple[str, ...]) -> None:
    """Plot 7D AeroHand action values over time."""
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("No records to plot")
    t = [record["timestamp"] for record in records]
    t0 = t[0]
    t = [value - t0 for value in t]
    actions = [record["action"] for record in records]
    plt.figure(figsize=(10, 5))
    for index, name in enumerate(action_names):
        plt.plot(t, [action[index] for action in actions], label=name)
    plt.xlabel("time (s)")
    plt.ylabel("normalized action")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
