import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.planning.plan_piper_gripper_pipette_handoff import (  # noqa: E402
    HANDOFF_AXIS_TOLERANCE_M,
    HANDOFF_CONTACT_TOP_TOLERANCE_M,
    HANDOFF_HOOK_TARGET_TOLERANCE_M,
    HANDOFF_TOP_TOLERANCE_M,
    hook_handoff_success_flags,
)


def test_hook_handoff_accepts_contact_confirmed_small_top_margin():
    strict, contact, functional, reached = hook_handoff_success_flags(
        final_hook_target_error=HANDOFF_HOOK_TARGET_TOLERANCE_M - 0.001,
        final_hook_top_offset=HANDOFF_TOP_TOLERANCE_M + 0.0001,
        final_hook_axis_offset=HANDOFF_AXIS_TOLERANCE_M - 0.001,
        target_top_offset=0.0,
        target_contact_during_settle=True,
        release_target_contact_fraction=1.0,
    )

    assert not strict
    assert contact
    assert functional
    assert reached


def test_hook_handoff_rejects_relaxed_top_without_contact():
    strict, contact, functional, reached = hook_handoff_success_flags(
        final_hook_target_error=HANDOFF_HOOK_TARGET_TOLERANCE_M - 0.001,
        final_hook_top_offset=HANDOFF_TOP_TOLERANCE_M + 0.0001,
        final_hook_axis_offset=HANDOFF_AXIS_TOLERANCE_M - 0.001,
        target_top_offset=0.0,
        target_contact_during_settle=False,
        release_target_contact_fraction=0.0,
    )

    assert not strict
    assert not contact
    assert not functional
    assert not reached


def test_hook_handoff_rejects_large_top_error_even_with_contact():
    strict, contact, functional, reached = hook_handoff_success_flags(
        final_hook_target_error=HANDOFF_HOOK_TARGET_TOLERANCE_M - 0.001,
        final_hook_top_offset=HANDOFF_CONTACT_TOP_TOLERANCE_M + 0.001,
        final_hook_axis_offset=HANDOFF_AXIS_TOLERANCE_M - 0.001,
        target_top_offset=0.0,
        target_contact_during_settle=True,
        release_target_contact_fraction=1.0,
    )

    assert not strict
    assert contact
    assert not functional
    assert not reached
