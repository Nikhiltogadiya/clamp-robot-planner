"""Schema round-trips + MOCK LLM client, zero network calls."""
import os

import pytest

from agents.llm_client import LLMClient
from agents.schema import Plan, RecoveryDecision, Subtask, Verdict


def test_subtask_roundtrip():
    st = Subtask(
        id=1, action="pick", object="red_cube", target=None,
        preconditions=["gripper empty"], postconditions=["holding red_cube"],
    )
    st2 = Subtask.model_validate_json(st.model_dump_json())
    assert st2 == st
    assert st2.assigned_agent == "arm_0"


def test_plan_roundtrip():
    plan = Plan(goal="test goal", subtasks=[
        Subtask(id=1, action="place", object="blue_cube", target="blue_bin",
                preconditions=[], postconditions=["blue_cube in blue_bin"]),
    ])
    assert Plan.model_validate_json(plan.model_dump_json()) == plan


def test_verdict_and_recovery_decision_construct():
    Verdict(satisfied=False, observed="cube missing", reason="not visible")
    RecoveryDecision(strategy="retry", rationale="try again")
    RecoveryDecision(strategy="replan", new_tail=[], rationale="reorder")


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setattr(
        "agents.llm_client.load_config",
        lambda: {"llm": {"mock": True, "models": {}, "base_url": "unused",
                          "api_key_env": "OPENROUTER_API_KEY"}},
    )


def test_mock_client_returns_valid_plan(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = LLMClient()
    plan = client.call("planner", "any instruction", Plan)
    assert isinstance(plan, Plan)
    assert len(plan.subtasks) == 6
    ids = [st.id for st in plan.subtasks]
    assert ids == sorted(ids)  # ordering intact


def test_mock_client_returns_valid_verdict_and_recovery():
    client = LLMClient()
    verdict = client.call("verifier", "check this", Verdict)
    assert isinstance(verdict, Verdict) and verdict.satisfied is True
    decision = client.call("recovery", "recover this", RecoveryDecision)
    assert isinstance(decision, RecoveryDecision) and decision.strategy == "retry"


def test_mock_client_never_touches_network(monkeypatch):
    # If MOCK ever tried to build a real client, this would raise KeyError
    # since the env var is intentionally absent here.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = LLMClient()
    client.call("planner", "any instruction", Plan)
    assert client._client is None  # real OpenAI client was never constructed


class _FakeCreate:
    """Minimal stand-in for openai chat.completions.create returning canned
    message contents in order."""
    def __init__(self, contents):
        self._contents, self._i = list(contents), 0

    def create(self, **_):
        content = self._contents[self._i]
        self._i += 1
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def test_real_client_repairs_empty_content(monkeypatch):
    """A model that returns empty content on the first call must trigger the
    repair retry, not crash on None.strip() (regression: a real eval run hit this)."""
    cfg = {"llm": {"mock": False, "base_url": "x", "api_key_env": "K",
                   "models": {"verifier": "m"}, "temperature": 0.0, "max_repair_retries": 1}}
    client = LLMClient(cfg)
    fake = type("Client", (), {})()
    fake.chat = type("Chat", (), {})()
    fake.chat.completions = _FakeCreate(
        [None, '{"satisfied": true, "observed": "ok", "reason": "fixed on retry"}'])
    monkeypatch.setattr(client, "_real_client", lambda: fake)
    verdict = client.call("verifier", "prompt", Verdict)
    assert verdict.satisfied is True and "retry" in verdict.reason
