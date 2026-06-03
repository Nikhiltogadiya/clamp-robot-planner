"""The planner turns 3 sample instructions into valid, ordered,
scene-consistent plans, and open-loop execution drives each to success."""
import pytest

from agents.config import load_config
from agents.planner import PlanValidationError, Planner, validate_plan
from agents.schema import Plan, Subtask
from env.scene import build_scene
from env.skills import Skills
from loop.controller import SAMPLE_INSTRUCTIONS, run_open_loop
import pybullet as p


@pytest.fixture
def world():
    cfg = load_config()
    cfg["sim"]["gui"] = False
    scene = build_scene(cfg)
    skills = Skills(scene, cfg)
    yield cfg, scene, skills
    p.disconnect(scene.client)


def test_three_instructions_are_distinct_valid_ordered_plans(world):
    cfg, scene, skills = world
    planner = Planner(cfg)
    sizes = []
    for instr in SAMPLE_INSTRUCTIONS:
        plan = planner.plan(instr, scene, skills)
        assert isinstance(plan, Plan) and len(plan.subtasks) >= 2
        ids = [st.id for st in plan.subtasks]
        assert ids == list(range(1, len(ids) + 1))          # sequential, ordered
        validate_plan(plan, scene)                          # references are real
        sizes.append(len(plan.subtasks))
    assert sizes == [4, 4, 6]                               # three genuinely different plans


@pytest.mark.parametrize("instruction", SAMPLE_INSTRUCTIONS)
def test_open_loop_reaches_goal(world, instruction):
    cfg, scene, skills = world
    planner = Planner(cfg)
    result = run_open_loop(instruction, scene, skills, planner)
    assert result["success"] is True
    assert result["events"] == []          # a correct plan trips no precondition gate


def test_validate_plan_rejects_unknown_object(world):
    _, scene, _ = world
    bad = Plan(goal="x", subtasks=[
        Subtask(id=1, action="pick", object="purple_cube", target=None,
                preconditions=[], postconditions=[]),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(bad, scene)


def test_validate_plan_rejects_place_into_a_cube(world):
    _, scene, _ = world
    bad = Plan(goal="x", subtasks=[
        Subtask(id=1, action="place", object="red_cube", target="blue_cube",
                preconditions=[], postconditions=[]),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(bad, scene)


def test_validate_plan_accepts_pick_then_put_down(world):
    """put_down (put a cube on the table / outside a bin) is a valid release
    action; a pick -> put_down plan must validate."""
    _, scene, _ = world
    plan = Plan(goal="x", subtasks=[
        Subtask(id=1, action="pick", object="red_cube", target=None,
                preconditions=[], postconditions=[]),
        Subtask(id=2, action="put_down", object="red_cube", target=None,
                preconditions=[], postconditions=["gripper empty"]),
    ])
    validate_plan(plan, scene)                    # must not raise


def test_validate_plan_rejects_put_down_with_a_target(world):
    _, scene, _ = world
    bad = Plan(goal="x", subtasks=[
        Subtask(id=1, action="pick", object="red_cube", target=None,
                preconditions=[], postconditions=[]),
        Subtask(id=2, action="put_down", object="red_cube", target="red_bin",
                preconditions=[], postconditions=[]),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(bad, scene)


def test_validate_plan_rejects_stack_on_itself(world):
    _, scene, _ = world
    bad = Plan(goal="x", subtasks=[
        Subtask(id=1, action="pick", object="green_cube", target=None,
                preconditions=[], postconditions=[]),
        Subtask(id=2, action="stack", object="green_cube", target="green_cube",
                preconditions=[], postconditions=[]),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(bad, scene)


def test_validate_plan_rejects_place_without_a_preceding_pick(world):
    _, scene, _ = world
    bad = Plan(goal="x", subtasks=[            # place with nothing held
        Subtask(id=1, action="place", object="red_cube", target="red_bin",
                preconditions=[], postconditions=[]),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(bad, scene)


def test_validate_plan_rejects_pick_while_already_holding(world):
    _, scene, _ = world
    bad = Plan(goal="x", subtasks=[
        Subtask(id=1, action="pick", object="red_cube", target=None,
                preconditions=[], postconditions=[]),
        Subtask(id=2, action="pick", object="blue_cube", target=None,
                preconditions=[], postconditions=[]),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(bad, scene)


def test_validate_plan_accepts_leaving_a_cube_outside(world):
    """A 'put red+blue in bins, leave green outside' plan (green simply gets no
    subtask) must be VALID - the shape the planner now produces for
    'others outside the box' instead of an invalid place-with-no-bin."""
    _, scene, _ = world
    plan = Plan(goal="x", subtasks=[
        Subtask(id=1, action="pick", object="red_cube", target=None,
                preconditions=[], postconditions=[]),
        Subtask(id=2, action="place", object="red_cube", target="red_bin",
                preconditions=[], postconditions=[]),
        Subtask(id=3, action="pick", object="blue_cube", target=None,
                preconditions=[], postconditions=[]),
        Subtask(id=4, action="place", object="blue_cube", target="blue_bin",
                preconditions=[], postconditions=[]),
    ])
    validate_plan(plan, scene)          # must not raise; green_cube is left alone


def test_validate_plan_rejects_duplicate_ids(world):
    _, scene, _ = world
    bad = Plan(goal="x", subtasks=[
        Subtask(id=1, action="pick", object="red_cube", target=None,
                preconditions=[], postconditions=[]),
        Subtask(id=1, action="place", object="red_cube", target="red_bin",
                preconditions=[], postconditions=[]),
    ])
    with pytest.raises(PlanValidationError):
        validate_plan(bad, scene)


def test_precondition_gate_flags_unmet(world):
    """If a cube is already held, a plan that starts with another pick trips
    the 'gripper empty' gate - proving the gate actually evaluates state."""
    cfg, scene, skills = world
    skills.pick("red_cube")                # gripper now full
    from env.state import postcondition_true
    assert postcondition_true(scene, skills, "gripper empty") is False
