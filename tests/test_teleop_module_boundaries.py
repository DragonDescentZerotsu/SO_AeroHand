import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_aero_hand_channel_does_not_depend_on_arm_control():
    modules = imported_modules(PROJECT_ROOT / "aero_quest/aero_hand_teleop.py")
    assert "aero_quest.arm_teleop" not in modules
    assert "aero_quest.osqp_ik" not in modules
    assert "aero_quest.quest_hand_frame" not in modules


def test_arm_control_does_not_depend_on_hand_retargeting():
    modules = imported_modules(PROJECT_ROOT / "aero_quest/arm_teleop.py")
    assert "aero_quest.retargeting" not in modules
    assert "aero_quest.aero_hand_teleop" not in modules
