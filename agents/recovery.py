"""Recovery agent: decide how to recover a failed step.

Given the goal, the full plan, the failed step index, the verifier's Verdict,
and the current scene state, returns a RecoveryDecision: retry the same step, or
replan the remaining steps (new_tail) from the current world state. The
controller applies the decision.
"""
import json
from pathlib import Path

from agents.config import load_config
from agents.llm_client import LLMClient
from agents.schema import Plan, RecoveryDecision, Subtask, Verdict
from env.state import scene_state

_PROMPT_TEMPLATE = (Path(__file__).resolve().parent.parent / "prompts" / "recovery.txt").read_text()


def _plan_lines(plan: Plan) -> str:
    return "\n".join(
        f"  [{st.id}] {st.action} {st.object}" + (f" -> {st.target}" if st.target else "")
        for st in plan.subtasks
    )


class Recovery:
    def __init__(self, cfg: dict | None = None, client: LLMClient | None = None):
        self.cfg = cfg or load_config()
        self.client = client or LLMClient(self.cfg)

    def decide(self, goal: str, plan: Plan, failed_index: int,
               verdicts: dict[str, Verdict], scene, skills) -> RecoveryDecision:
        st = plan.subtasks[failed_index]
        verdict_txt = "; ".join(f"{c}: {v.reason}" for c, v in verdicts.items()) or "(unsatisfied)"
        prompt = (
            _PROMPT_TEMPLATE
            .replace("__GOAL__", goal)
            .replace("__PLAN__", _plan_lines(plan))
            .replace("__INDEX__", str(failed_index))
            .replace("__SUBTASK_ID__", str(st.id))
            .replace("__ACTION__", st.action)
            .replace("__OBJECT__", st.object)
            .replace("__TARGET__", str(st.target))
            .replace("__POSTCONDITIONS__", ", ".join(st.postconditions))
            .replace("__VERDICT__", verdict_txt)
            .replace("__STATE__", json.dumps(scene_state(scene, skills)))
        )
        return self.client.call("recovery", prompt, RecoveryDecision)


class MockRecovery:
    """Deterministic, context-aware recovery for offline loop testing (parallel
    to OracleVerifier). Uses the real plan objects it already has - the generic
    llm_client mock only ever returns "retry", which cannot fix a displaced cube.

    Policy: a failed place/stack means the cube is no longer held (it was
    released, then displaced), so the fix is to REPLAN - re-pick the object,
    redo the action, then continue with the rest of the plan. Anything else
    (e.g. a precondition gate with no verdict) → retry.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()

    def decide(self, goal: str, plan: Plan, failed_index: int,
               verdicts: dict[str, Verdict], scene, skills) -> RecoveryDecision:
        st = plan.subtasks[failed_index]
        if st.action in ("place", "stack"):
            base = st.id
            repick = Subtask(id=base * 100 + 1, action="pick", object=st.object,
                             target=None, preconditions=["gripper empty"],
                             postconditions=[f"holding {st.object}"])
            redo = Subtask(id=base * 100 + 2, action=st.action, object=st.object,
                           target=st.target, preconditions=[f"holding {st.object}"],
                           postconditions=st.postconditions)
            new_tail = [repick, redo, *plan.subtasks[failed_index + 1:]]
            return RecoveryDecision(strategy="replan", new_tail=new_tail,
                                    rationale=f"{st.object} was displaced and is not held; "
                                              f"re-pick and redo {st.action}")
        return RecoveryDecision(strategy="retry", new_tail=None,
                                rationale="retry the step")
