"""Planner agent: natural-language instruction -> structured Plan.

Builds a prompt from the scene state + skill API description, calls the LLM
(mockable), and validates that the returned Plan only references
objects and bins that actually exist.
"""
import json
from pathlib import Path

from agents.config import load_config
from agents.llm_client import LLMClient
from agents.schema import Plan
from env.state import scene_state

_PROMPT_TEMPLATE = (Path(__file__).resolve().parent.parent / "prompts" / "planner.txt").read_text()

_VALID_ACTIONS = {"pick", "place", "stack", "put_down"}


class PlanValidationError(ValueError):
    """The Plan is schema-valid but references things not in the scene, or is
    otherwise inconsistent with the skill API."""


def build_prompt(instruction: str, scene, skills) -> str:
    state = scene_state(scene, skills)
    return (
        _PROMPT_TEMPLATE
        .replace("__OBJECTS__", ", ".join(state["objects"]))
        .replace("__BINS__", ", ".join(state["bins"]))
        .replace("__STATE__", json.dumps(state))
        .replace("__INSTRUCTION__", instruction)
    )


def validate_plan(plan: Plan, scene) -> None:
    """Raise PlanValidationError if the plan references unknown cubes/bins, uses a
    target inconsistent with its action, has duplicate ids, or violates the
    pick-before-manipulate / one-cube-at-a-time invariant (a static gripper-state
    trace catches place/stack with nothing held and pick while already holding)."""
    cubes, bins = set(scene.cubes), set(scene.bins)
    seen_ids: set[int] = set()
    for st in plan.subtasks:
        if st.id in seen_ids:
            raise PlanValidationError(f"duplicate subtask id {st.id}")
        seen_ids.add(st.id)
        if st.action not in _VALID_ACTIONS:
            raise PlanValidationError(f"subtask {st.id}: unknown action {st.action!r}")
        if st.object not in cubes:
            raise PlanValidationError(f"subtask {st.id}: unknown object {st.object!r}")
        if st.action == "pick":
            if st.target is not None:
                raise PlanValidationError(f"subtask {st.id}: pick must have target=None")
        elif st.action == "place":
            if st.target not in bins:
                raise PlanValidationError(
                    f"subtask {st.id}: place target {st.target!r} is not a bin - place only puts a "
                    f"cube INTO a bin ({sorted(bins)}); there is no place-on-table action, so a "
                    f"cube meant to stay outside a bin should be left alone (no subtask)")
        elif st.action == "stack":
            if st.target not in cubes:
                raise PlanValidationError(f"subtask {st.id}: stack target {st.target!r} is not a cube")
            if st.target == st.object:
                raise PlanValidationError(f"subtask {st.id}: cannot stack {st.object!r} on itself")
        elif st.action == "put_down":
            if st.target is not None:
                raise PlanValidationError(f"subtask {st.id}: put_down must have target=None")

    held: str | None = None
    for st in plan.subtasks:
        if st.action == "pick":
            if held is not None:
                raise PlanValidationError(
                    f"subtask {st.id}: pick {st.object!r} while already holding {held!r}")
            held = st.object
        else:  # place / stack must act on the currently-held cube
            if held != st.object:
                raise PlanValidationError(
                    f"subtask {st.id}: {st.action} {st.object!r} but gripper holds {held!r}")
            held = None


class Planner:
    def __init__(self, cfg: dict | None = None, client: LLMClient | None = None):
        self.cfg = cfg or load_config()
        self.client = client or LLMClient(self.cfg)

    def plan(self, instruction: str, scene, skills) -> Plan:
        prompt = build_prompt(instruction, scene, skills)
        plan = self.client.call("planner", prompt, Plan)
        validate_plan(plan, scene)
        return plan
