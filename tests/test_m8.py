"""Operator console tests - all offline (mock/oracle), no Viser server and no
LLM spend, so this stays part of the fast CI suite. Live-model behavior was
verified manually against the live console app."""
import copy

import pybullet as p
import pytest

from agents.config import load_config
from agents.llm_client import LLMClient, UsageMeter
from agents.planner import Planner
from agents.recovery import MockRecovery
from agents.schema import Plan, Verdict
from agents.verifier import OracleVerifier
from console.driver import apply_place_on_top, apply_sabotage, run_episode
from env.observe import object_poses, robot_joint_angles
from env.scene import ARM_JOINTS, build_scene
from env.skills import Skills

INSTR = "Put the red cube in the red bin and put the blue cube in the blue bin."


@pytest.fixture
def world():
    cfg = copy.deepcopy(load_config())
    cfg["sim"]["gui"] = False
    scene = build_scene(cfg)
    skills = Skills(scene, cfg)
    yield cfg, scene, skills
    p.disconnect(scene.client)


def test_object_poses_has_all_cubes_with_position_and_orientation(world):
    _, scene, _ = world
    poses = object_poses(scene)
    assert set(poses) == set(scene.cubes)
    for pos, orn in poses.values():
        assert len(pos) == 3 and len(orn) == 4


def test_robot_joint_angles_matches_rest_pose(world):
    _, scene, _ = world
    angles = robot_joint_angles(scene)
    assert angles.shape == (len(ARM_JOINTS) + 1,)          # 7 arm + 1 finger
    # rest pose from env/scene.py; finger starts open (~0.04)
    assert angles[1] == pytest.approx(-0.785, abs=1e-2)
    assert angles[-1] == pytest.approx(0.04, abs=1e-2)


def test_usage_meter_counts_calls_and_is_free_in_mock():
    meter = UsageMeter()
    client = LLMClient(usage_meter=meter)
    client.call("planner", "x", Plan)
    client.call("verifier", "x", Verdict)
    assert meter.calls_by_role == {"planner": 1, "verifier": 1}
    assert meter.total_calls == 2
    assert meter.cost_usd == 0.0                            # mock never spends


def test_usage_meter_records_real_usage():
    meter = UsageMeter()
    meter.record_usage("deepseek/deepseek-v4-flash",
                       type("U", (), {"prompt_tokens": 1000, "completion_tokens": 500})())
    assert meter.prompt_tokens == 1000 and meter.completion_tokens == 500
    assert meter.cost_usd == pytest.approx((1000 * 0.09 + 500 * 0.18) / 1_000_000)


def test_apply_sabotage_knock_off_moves_cube_out_of_bin(world):
    _, scene, skills = world
    skills.pick("red_cube")
    skills.place("red_cube", "red_bin")
    from env.state import in_bin
    assert in_bin(scene, "red_cube", "red_bin")
    desc = apply_sabotage(scene, skills, "knock_off", "red_cube")
    assert "knocked off" in desc
    assert not in_bin(scene, "red_cube", "red_bin")


def test_apply_sabotage_steal_releases_held_cube(world):
    _, scene, skills = world
    skills.pick("red_cube")
    assert skills.held == "red_cube"
    desc = apply_sabotage(scene, skills, "steal", "red_cube")
    assert "stolen" in desc
    assert skills.held is None


def test_apply_sabotage_steal_noop_when_not_held(world):
    _, scene, skills = world
    desc = apply_sabotage(scene, skills, "steal", "red_cube")
    assert "nothing to steal" in desc


def test_apply_sabotage_topple_noop_when_not_stacked(world):
    """Regression: knock_off and topple used to be IDENTICAL code (same
    direction, no precondition). Topple must now require the cube to
    actually be on top of something, mirroring how 'steal' requires holding."""
    _, scene, skills = world
    desc = apply_sabotage(scene, skills, "topple", "green_cube")
    assert "nothing to topple" in desc


def test_apply_sabotage_topple_moves_cube_off_stack(world):
    _, scene, skills = world
    skills.pick("red_cube")
    skills.place("red_cube", "red_bin")
    skills.pick("green_cube")
    skills.stack("green_cube", "red_cube")
    from env.state import on_top_of
    assert on_top_of(scene, "green_cube", "red_cube")
    desc = apply_sabotage(scene, skills, "topple", "green_cube", direction="left")
    assert "toppled off red_cube" in desc
    assert not on_top_of(scene, "green_cube", "red_cube")


def test_apply_sabotage_direction_changes_the_outcome(world):
    """Regression: knock_off previously always pushed +x regardless of which
    button/cube was chosen. Different directions must now produce genuinely
    different resulting positions."""
    _, scene, skills = world
    x0, y0, _ = p.getBasePositionAndOrientation(scene.cubes["blue_cube"])[0]

    apply_sabotage(scene, skills, "knock_off", "blue_cube", direction="left")
    _, y_left, _ = p.getBasePositionAndOrientation(scene.cubes["blue_cube"])[0]

    skills._teleport(scene.cubes["blue_cube"], [x0, y0, scene.cfg["scene"]["cube_half_extent"]])
    apply_sabotage(scene, skills, "knock_off", "blue_cube", direction="right")
    _, y_right, _ = p.getBasePositionAndOrientation(scene.cubes["blue_cube"])[0]

    assert y_left < y0 < y_right         # left and right move opposite ways
    assert y_left != y_right              # the reported bug (identical outcome) is fixed


def test_apply_sabotage_repeated_pushes_stay_within_the_workspace(world):
    """Regression: repeated clicks used to compound an unbounded offset from
    the cube's current position, walking it off the rendered floor."""
    _, scene, skills = world
    for _ in range(30):                  # far more than needed to hit the clamp
        apply_sabotage(scene, skills, "knock_off", "blue_cube", direction="right")
    _, y, _ = p.getBasePositionAndOrientation(scene.cubes["blue_cube"])[0]
    # a few mm of physics-settle drift past the clamp boundary is fine; a 2.7m
    # runaway (30 * 0.09) is what the old unbounded-offset bug would produce
    assert y <= 0.35 + 0.01


def test_apply_sabotage_refuses_to_teleport_into_another_cube(world):
    """Regression (real bug found live): teleporting a cube to overlap another
    one causes PyBullet's solver to violently eject them apart during
    settle() -- a launch, not a nudge. Reproduced live: a "right" push landed
    4cm from another cube (needs >=5cm to avoid overlap) and both cubes flew
    across the table. The push must now be refused instead of executed.

    Set up the exact "4cm away after one more push" geometry directly (rather
    than replaying clicks) so this test is independent of which cube happens
    to be in the way first."""
    _, scene, skills = world
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    skills._teleport(scene.cubes["green_cube"], [0.45, -0.9, cube_h])  # out of the way
    bx, by, _ = p.getBasePositionAndOrientation(scene.cubes["blue_cube"])[0]
    skills._teleport(scene.cubes["red_cube"], [bx, by - 0.13, cube_h])  # one 0.09 push -> 4cm gap

    desc = apply_sabotage(scene, skills, "knock_off", "red_cube", direction="right")
    assert "is in the way" in desc and "blue_cube" in desc
    # red_cube must not have moved -- refused, not launched
    xr, yr, _ = p.getBasePositionAndOrientation(scene.cubes["red_cube"])[0]
    assert (xr, yr) == pytest.approx((bx, by - 0.13))


def test_apply_sabotage_never_creates_a_cube_overlap(world):
    """Broader invariant: no sequence of pushes should ever land one cube
    within the minimum safe gap of another, regardless of direction."""
    _, scene, skills = world
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    min_gap = 2 * cube_h + 0.02
    directions = ["left", "right", "forward", "backward"]
    for i in range(40):
        apply_sabotage(scene, skills, "knock_off", "red_cube", direction=directions[i % 4])
        positions = {n: p.getBasePositionAndOrientation(cid)[0] for n, cid in scene.cubes.items()}
        names = list(positions)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                pa, pb = positions[names[a]], positions[names[b]]
                dist = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                assert dist >= min_gap - 1e-6, f"{names[a]}/{names[b]} overlap at step {i}"


def test_apply_sabotage_unknown_direction_raises(world):
    _, scene, skills = world
    with pytest.raises(ValueError):
        apply_sabotage(scene, skills, "knock_off", "red_cube", direction="sideways")


def test_apply_sabotage_unknown_mode_raises(world):
    _, scene, skills = world
    with pytest.raises(ValueError):
        apply_sabotage(scene, skills, "bogus_mode", "red_cube")


def test_run_episode_clean_success_emits_full_event_sequence(world):
    cfg, scene, skills = world
    events = []
    result = run_episode(INSTR, scene, skills, Planner(cfg), OracleVerifier(cfg),
                         MockRecovery(cfg), emit=lambda e: events.append(e["type"]))
    assert result["success"] is True
    assert events[0] == "planning" and events[1] == "plan_ready" and events[-1] == "done"
    assert "step_verified" in events


def test_run_episode_sabotage_mid_grasp_triggers_recovery(world):
    """The headline console interaction: steal the held cube right after pick,
    before place -> the precondition gate catches it -> replan -> recovers."""
    cfg, scene, skills = world
    calls = {"n": 0}

    def poll():
        calls["n"] += 1
        if calls["n"] == 2:                 # 2nd poll = top of the "place" iteration,
            return [{"type": "sabotage", "mode": "steal", "object": "red_cube"}]
        return []                          # i.e. right after pick succeeded (mid-grasp)

    events = []
    result = run_episode(INSTR, scene, skills, Planner(cfg), OracleVerifier(cfg),
                         MockRecovery(cfg), recovery_enabled=True,
                         emit=lambda e: events.append(e), poll_commands=poll)
    assert result["success"] is True
    assert result["n_recover"] >= 1
    assert any(e["type"] == "sabotage" for e in events)
    assert any(e["type"] == "recover" for e in events)


def test_run_episode_recovery_off_fails_after_sabotage(world):
    cfg, scene, skills = world
    calls = {"n": 0}

    def poll():
        calls["n"] += 1
        if calls["n"] == 2:                # mid-grasp, same timing as the ON test
            return [{"type": "sabotage", "mode": "steal", "object": "red_cube"}]
        return []

    result = run_episode(INSTR, scene, skills, Planner(cfg), OracleVerifier(cfg),
                         MockRecovery(cfg), recovery_enabled=False, poll_commands=poll)
    assert result["success"] is False


def test_run_episode_recovers_a_late_sabotage_via_goal_recheck(world):
    """Regression (real gap the user hit): a cube knocked out of its bin AFTER
    its own step already passed is NOT caught by per-step verification (later
    steps never re-check it), so the episode ended FAILED with 0 recoveries.
    The end-of-plan goal re-check must now catch it, re-plan from the current
    scene, and fix it -> SUCCESS."""
    from env.state import in_bin
    cfg, scene, skills = world
    calls = {"n": 0}

    def poll():
        calls["n"] += 1
        if calls["n"] == 3:      # after place red_cube (step 2) passed; later steps ignore red
            return [{"type": "sabotage", "mode": "knock_off", "object": "red_cube"}]
        return []

    result = run_episode(INSTR, scene, skills, Planner(cfg), OracleVerifier(cfg),
                         MockRecovery(cfg), recovery_enabled=True, poll_commands=poll)
    assert result["success"] is True           # the goal re-check recovered it
    assert result["n_recover"] >= 1
    assert in_bin(scene, "red_cube", "red_bin")   # red is genuinely back in its bin


def test_run_episode_bad_instruction_emits_error_not_crash(world):
    cfg, scene, skills = world

    class _BrokenPlanner:
        def plan(self, *a, **k):
            raise RuntimeError("boom")

    events = []
    result = run_episode(INSTR, scene, skills, _BrokenPlanner(), OracleVerifier(cfg),
                         MockRecovery(cfg), emit=lambda e: events.append(e))
    assert result["success"] is False
    assert any(e["type"] == "error" for e in events)


def test_apply_place_on_top_works_for_any_pair(world):
    """The manual 'Place on top' button must be generic, not hardcoded to one
    pair -- verify with 3 different combinations, matching real usage."""
    from env.state import on_top_of
    for obj, on in [("green_cube", "red_cube"), ("red_cube", "blue_cube"),
                   ("blue_cube", "green_cube")]:
        cfg = copy.deepcopy(load_config()); cfg["sim"]["gui"] = False
        scene = build_scene(cfg); skills = Skills(scene, cfg)
        desc = apply_place_on_top(scene, skills, obj, on)
        assert f"{obj} placed on top of {on}" == desc
        assert on_top_of(scene, obj, on)
        p.disconnect(scene.client)


def test_apply_place_on_top_releases_a_held_cube_first(world):
    _, scene, skills = world
    skills.pick("green_cube")
    assert skills.held == "green_cube"
    desc = apply_place_on_top(scene, skills, "green_cube", "red_cube")
    assert desc == "green_cube placed on top of red_cube"
    assert skills.held is None
    from env.state import on_top_of
    assert on_top_of(scene, "green_cube", "red_cube")


def test_apply_place_on_top_rejects_self_and_unknown(world):
    _, scene, skills = world
    assert "itself" in apply_place_on_top(scene, skills, "red_cube", "red_cube")
    assert "unknown" in apply_place_on_top(scene, skills, "purple_cube", "red_cube")


def test_apply_place_on_top_refuses_to_overlap_a_third_cube(world):
    """Same collision-safety principle as sabotage: placing obj on top of
    `on` must not be allowed to land on top of a THIRD cube that's already
    sitting there."""
    _, scene, skills = world
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    # put blue directly where "green on top of red" would land
    rx, ry, rz = p.getBasePositionAndOrientation(scene.cubes["red_cube"])[0]
    skills._teleport(scene.cubes["blue_cube"], [rx, ry, rz + 2 * cube_h])
    desc = apply_place_on_top(scene, skills, "green_cube", "red_cube")
    assert "is in the way" in desc
    from env.state import on_top_of
    assert not on_top_of(scene, "green_cube", "red_cube")


def test_planner_prompt_reflects_a_manually_placed_scene(world):
    """The user's core requirement: manual placement done BEFORE calling the
    planner must be visible to it -- scene_state() (what the prompt is built
    from) must reflect live state, not a cached/default snapshot."""
    import json

    from agents.planner import build_prompt
    from env.state import scene_state

    cfg, scene, skills = world
    default_pos = scene_state(scene, skills)["objects"]["green_cube"]["position"]

    apply_place_on_top(scene, skills, "green_cube", "red_cube")
    new_pos = scene_state(scene, skills)["objects"]["green_cube"]["position"]
    assert new_pos[2] > default_pos[2] + 0.03   # sanity: really elevated (on top of red)
    assert new_pos != default_pos

    prompt = build_prompt("put the blue cube in the blue bin", scene, skills)
    # the live scene_state JSON embedded in the prompt must carry green_cube's
    # CURRENT (post-placement) position, not its original start position
    assert json.dumps(new_pos) in prompt
    assert json.dumps(default_pos) not in prompt


def test_apply_place_on_top_builds_a_three_high_tower(world):
    """Regression (real bug found live): the collision check used to compare
    only (x, y), ignoring height -- so stacking a 3rd cube on a 2-high tower
    falsely reported the BOTTOM cube as "in the way" of a target that's
    actually a safe height above it. Building bottom-up must work."""
    from env.state import on_top_of
    _, scene, skills = world
    assert apply_place_on_top(scene, skills, "blue_cube", "green_cube") == \
        "blue_cube placed on top of green_cube"
    assert apply_place_on_top(scene, skills, "red_cube", "blue_cube") == \
        "red_cube placed on top of blue_cube"
    assert on_top_of(scene, "blue_cube", "green_cube")
    assert on_top_of(scene, "red_cube", "blue_cube")
    gz = p.getBasePositionAndOrientation(scene.cubes["green_cube"])[0][2]
    bz = p.getBasePositionAndOrientation(scene.cubes["blue_cube"])[0][2]
    rz = p.getBasePositionAndOrientation(scene.cubes["red_cube"])[0][2]
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    assert bz == pytest.approx(gz + 2 * cube_h, abs=1e-3)
    assert rz == pytest.approx(bz + 2 * cube_h, abs=1e-3)


def test_apply_place_on_top_refuses_to_move_a_cube_with_a_rider(world):
    """Regression (real bug found live): moving a cube that has another cube
    resting on it used to leave the rider floating in place, then falling
    once physics settled, instead of refusing or moving it along."""
    _, scene, skills = world
    apply_place_on_top(scene, skills, "red_cube", "blue_cube")
    rz_before = p.getBasePositionAndOrientation(scene.cubes["red_cube"])[0]

    desc = apply_place_on_top(scene, skills, "blue_cube", "green_cube")
    assert "is stacked on top of it" in desc and "red_cube" in desc

    from env.state import on_top_of
    assert on_top_of(scene, "red_cube", "blue_cube")   # untouched, still riding blue
    assert p.getBasePositionAndOrientation(scene.cubes["red_cube"])[0] == pytest.approx(rz_before)
    assert not on_top_of(scene, "blue_cube", "green_cube")  # refused move never happened


def test_apply_sabotage_knock_off_refuses_to_move_a_cube_with_a_rider(world):
    """Same rider-safety guarantee applies to sabotage's knock_off/topple."""
    _, scene, skills = world
    apply_place_on_top(scene, skills, "green_cube", "red_cube")
    desc = apply_sabotage(scene, skills, "knock_off", "red_cube", direction="right")
    assert "is stacked on top of" in desc and "green_cube" in desc
    from env.state import on_top_of
    assert on_top_of(scene, "green_cube", "red_cube")   # nothing moved


def test_steal_safe_drop_avoids_overlap(world):
    """Regression (audit C-steal): steal dropped the cube at the gripper xy with
    NO overlap check, so stealing a cube picked off a stack dropped it onto the
    cube below and PyBullet launched them. _safe_drop must nudge to a clear spot."""
    from console.driver import _safe_drop
    from env.state import on_top_of
    _, scene, skills = world
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    bx, by, _ = p.getBasePositionAndOrientation(scene.cubes["blue_cube"])[0]
    _safe_drop(scene, skills, scene.cubes["red_cube"], bx, by, cube_h, 60)  # drop onto blue's spot
    rx, ry, _ = p.getBasePositionAndOrientation(scene.cubes["red_cube"])[0]
    assert ((rx - bx) ** 2 + (ry - by) ** 2) ** 0.5 >= 2 * cube_h     # nudged clear, not on top
    assert not on_top_of(scene, "red_cube", "blue_cube")


def test_run_episode_emits_plan_update_on_replan(world):
    """Regression (audit C-plan-stale): a recovery replan mutates the plan but
    used to emit no event, so the GUI plan panel went stale. It must now emit a
    plan_update carrying the new tail."""
    cfg, scene, skills = world
    calls = {"n": 0}

    def poll():
        calls["n"] += 1
        if calls["n"] == 2:                 # steal mid-grasp -> gate replan on the place step
            return [{"type": "sabotage", "mode": "steal", "object": "red_cube"}]
        return []

    events = []
    run_episode(INSTR, scene, skills, Planner(cfg), OracleVerifier(cfg),
                MockRecovery(cfg), recovery_enabled=True,
                emit=lambda e: events.append(e), poll_commands=poll)
    assert any(e["type"] == "plan_update" for e in events)


def test_run_episode_accepts_callable_recovery_enabled(world):
    """Regression (audit C-toggle): recovery_enabled may be a live callable so a
    mid-episode Recovery toggle actually takes effect on the running episode."""
    cfg, scene, skills = world
    flag = {"on": True}
    result = run_episode(INSTR, scene, skills, Planner(cfg), OracleVerifier(cfg),
                         MockRecovery(cfg), recovery_enabled=lambda: flag["on"])
    assert result["success"] is True


def test_run_episode_gives_up_on_persistent_verifier_disagreement(world):
    """Audit regression: the console driver's GLOBAL replan cap must stop a run
    where the verifier keeps disagreeing while execution succeeds - each replan
    shifts the failing index, so the per-step cap alone never fires (this is the
    'stack green on a red cube inside the red tray' loop the user hit)."""
    from agents.schema import Verdict
    from agents.verifier import is_visual
    cfg, scene, skills = world

    class _AlwaysDisagrees:
        def verify_subtask(self, scene, subtask, skills=None):
            return {c: Verdict(satisfied=False, observed="stub", reason="disagrees")
                    for c in subtask.postconditions if is_visual(c)}

    result = run_episode(INSTR, scene, skills, Planner(cfg), _AlwaysDisagrees(),
                         MockRecovery(cfg), recovery_enabled=True, max_steps=40)
    assert result["steps"] < 40
    assert result["success"] is True
    assert result["n_recover"] <= 6       # R2: recovery not called after the replan cap is spent


def test_console_gui_modules_import():
    """CI coverage (audit D-console-untested): the Viser GUI modules import
    viser/yourdfpy and are never exercised by the offline driver tests, so at
    least assert they import cleanly and expose their entry classes - this
    catches an import/API break that the driver-only tests would miss."""
    import console.app
    import console.scene3d
    assert hasattr(console.app, "ConsoleApp")
    assert hasattr(console.scene3d, "Console3D")
