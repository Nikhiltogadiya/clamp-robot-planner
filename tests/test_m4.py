"""The closed loop recovers a scripted failure, and success with
recovery ON beats OFF. Uses the OracleVerifier + MockRecovery so the loop logic
is tested deterministically and offline."""
import copy

import pybullet as p
import pytest

from agents.config import load_config
from agents.planner import Planner
from agents.recovery import MockRecovery
from agents.schema import Plan, Subtask
from agents.verifier import OracleVerifier
from env.failure import FailureInjector
from env.scene import build_scene
from env.skills import Skills
from env.state import in_bin
from loop.controller import preconditions_met, run

INSTR = "Put the red cube in the red bin and put the blue cube in the blue bin."


def _cfg_with_scripted_miss():
    cfg = copy.deepcopy(load_config())
    cfg["sim"]["gui"] = False
    cfg["failure"] = {**cfg["failure"], "p_fail": 0.0,
                      "scripted": [None, "missed_placement"] + [None] * 10}
    return cfg


def _run(enabled: bool):
    cfg = _cfg_with_scripted_miss()
    scene = build_scene(cfg)
    skills = Skills(scene, cfg)
    result = run(INSTR, scene, skills, Planner(cfg), OracleVerifier(cfg),
                 MockRecovery(cfg), recovery_enabled=enabled,
                 failure_injector=FailureInjector(cfg))
    p.disconnect(scene.client)
    return result


def test_recovery_on_succeeds_off_fails():
    assert _run(enabled=False)["success"] is False     # baseline can't recover
    on = _run(enabled=True)
    assert on["success"] is True                        # closed loop recovers
    assert any(e["phase"] == "recover" for e in on["events"])   # it actually recovered


def test_loop_terminates_under_persistent_failure():
    """Every placement fails forever -> the loop must still stop (max_steps)."""
    cfg = copy.deepcopy(load_config())
    cfg["sim"]["gui"] = False
    cfg["failure"] = {**cfg["failure"], "p_fail": 1.0, "scripted": []}
    scene = build_scene(cfg)
    skills = Skills(scene, cfg)
    result = run(INSTR, scene, skills, Planner(cfg), OracleVerifier(cfg),
                 MockRecovery(cfg), recovery_enabled=True,
                 failure_injector=FailureInjector(cfg), max_steps=12)
    p.disconnect(scene.client)
    assert result["steps"] <= 12
    assert result["success"] is False


def test_loop_gives_up_when_verifier_persistently_disagrees():
    """Audit regression: if the VERIFIER keeps saying 'not satisfied' while the
    task actually succeeds (a scene it can't perceive - e.g. a cube stacked
    inside a same-coloured tray), each replan shifts the failing step to a NEW
    index, so the per-step cap never fires. The GLOBAL replan cap must make the
    loop give up well before max_steps; ground truth then reports the real
    (successful) outcome."""
    from agents.schema import Verdict
    from agents.verifier import is_visual

    class _AlwaysDisagrees:
        def verify_subtask(self, scene, subtask, skills=None):
            return {c: Verdict(satisfied=False, observed="stub", reason="always disagrees")
                    for c in subtask.postconditions if is_visual(c)}

    cfg = copy.deepcopy(load_config()); cfg["sim"]["gui"] = False
    scene = build_scene(cfg); skills = Skills(scene, cfg)
    result = run(INSTR, scene, skills, Planner(cfg), _AlwaysDisagrees(),
                 MockRecovery(cfg), recovery_enabled=True, max_steps=40)
    p.disconnect(scene.client)
    assert result["steps"] < 40           # gave up; did NOT churn to the step budget
    assert result["success"] is True      # ground truth: the plan physically executed fine
    assert result["n_recover"] <= 6       # R2: no wasted recovery calls after the cap is spent


def test_mock_recovery_replans_a_place_failure():
    plan = Plan(goal="g", subtasks=[
        Subtask(id=1, action="pick", object="red_cube", target=None,
                preconditions=[], postconditions=["holding red_cube"]),
        Subtask(id=2, action="place", object="red_cube", target="red_bin",
                preconditions=[], postconditions=["red_cube in red_bin"]),
    ])
    dec = MockRecovery().decide("g", plan, failed_index=1, verdicts={},
                                scene=None, skills=None)
    assert dec.strategy == "replan"
    assert dec.new_tail[0].action == "pick" and dec.new_tail[0].object == "red_cube"
    assert dec.new_tail[1].action == "place" and dec.new_tail[1].target == "red_bin"


def test_oracle_verifier_matches_ground_truth():
    cfg = copy.deepcopy(load_config())
    cfg["sim"]["gui"] = False
    scene = build_scene(cfg)
    skills = Skills(scene, cfg)
    st = Subtask(id=1, action="place", object="red_cube", target="red_bin",
                 preconditions=[], postconditions=["red_cube in red_bin"])
    v = OracleVerifier(cfg)
    # before placing: not satisfied
    assert v.verify_subtask(scene, st, skills)["red_cube in red_bin"].satisfied is False
    skills.pick("red_cube"); skills.place("red_cube", "red_bin")
    assert in_bin(scene, "red_cube", "red_bin")
    assert v.verify_subtask(scene, st, skills)["red_cube in red_bin"].satisfied is True
    p.disconnect(scene.client)


def test_preconditions_gate():
    cfg = copy.deepcopy(load_config())
    cfg["sim"]["gui"] = False
    scene = build_scene(cfg)
    skills = Skills(scene, cfg)
    need_hold = Subtask(id=1, action="place", object="red_cube", target="red_bin",
                        preconditions=["holding red_cube"], postconditions=[])
    assert preconditions_met(scene, skills, need_hold) is False   # nothing held
    skills.pick("red_cube")
    assert preconditions_met(scene, skills, need_hold) is True
    p.disconnect(scene.client)
