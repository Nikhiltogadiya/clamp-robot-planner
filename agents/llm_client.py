"""OpenRouter text+vision wrapper with a MOCK mode.

In MOCK mode (config.yaml: llm.mock: true) this never touches the network -
it returns canned, schema-valid responses so the rest of the pipeline can be
built and tested without spending a token. The canned Plan below is the
single source of truth for the default demo scene (env/scene.py's default cube
and bin layout): red_cube -> red_bin, blue_cube -> blue_bin, green_cube
stacked on red_cube.
"""
import base64
import io
import json
import os
from typing import Callable, TypeVar

import imageio.v3 as iio
import numpy as np
from pydantic import BaseModel, ValidationError

from agents.config import load_config
from agents.schema import Plan, RecoveryDecision, Subtask, Verdict

T = TypeVar("T", bound=BaseModel)

# Approximate OpenRouter prices ($ per 1M tokens) for the console's live cost
# meter only - NOT used for billing; update here if config.yaml's model ids change.
_PRICING_PER_M = {
    "deepseek/deepseek-v4-flash": (0.09, 0.18),
    "qwen/qwen3.5-flash-02-23": (0.065, 0.26),
}


class UsageMeter:
    """Tracks LLM call counts, tokens, and estimated cost. Attach one instance
    to an LLMClient (shared across Planner/Verifier/Recovery via the `client`
    constructor arg) to get a running total for a whole console session.
    Counts calls even in MOCK mode; tokens/cost only accrue on real calls."""

    def __init__(self):
        self.calls_by_role: dict[str, int] = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def record_call(self, role: str) -> None:
        self.calls_by_role[role] = self.calls_by_role.get(role, 0) + 1

    def record_usage(self, model: str, usage) -> None:
        if usage is None:
            return
        p_tok = getattr(usage, "prompt_tokens", 0) or 0
        c_tok = getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        in_rate, out_rate = _PRICING_PER_M.get(model, (0.0, 0.0))
        self.cost_usd += (p_tok * in_rate + c_tok * out_rate) / 1_000_000

    @property
    def total_calls(self) -> int:
        return sum(self.calls_by_role.values())

    def summary(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.calls_by_role.items()))
        return f"{self.total_calls} calls ({parts})  ~${self.cost_usd:.4f}"


def _pick(i: int, obj: str, extra_pre: list[str] | None = None) -> Subtask:
    return Subtask(id=i, action="pick", object=obj, target=None,
                   preconditions=["gripper empty", *(extra_pre or [])],
                   postconditions=[f"holding {obj}"])


def _place(i: int, obj: str, bin_: str) -> Subtask:
    return Subtask(id=i, action="place", object=obj, target=bin_,
                   preconditions=[f"holding {obj}"],
                   postconditions=[f"{obj} in {bin_}", "gripper empty"])


def _stack(i: int, obj: str, on: str) -> Subtask:
    return Subtask(id=i, action="stack", object=obj, target=on,
                   preconditions=[f"holding {obj}"],
                   postconditions=[f"{obj} on {on}", "gripper empty"])


def _plan_s1() -> Plan:
    """Sort: red -> red_bin, blue -> blue_bin (no stacking)."""
    return Plan(
        goal="Put the red cube in the red bin and put the blue cube in the blue bin.",
        subtasks=[_pick(1, "red_cube"), _place(2, "red_cube", "red_bin"),
                  _pick(3, "blue_cube"), _place(4, "blue_cube", "blue_bin")],
    )


def _plan_s2() -> Plan:
    """Order + stack: blue -> blue_bin, then green stacked on blue."""
    return Plan(
        goal="Put the blue cube in the blue bin, then stack the green cube on the blue cube.",
        subtasks=[_pick(1, "blue_cube"), _place(2, "blue_cube", "blue_bin"),
                  _pick(3, "green_cube", ["blue_cube in blue_bin"]),
                  _stack(4, "green_cube", "blue_cube")],
    )


def _plan_s3() -> Plan:
    """Full pipeline (also the demo default): sort red+blue, then green on red."""
    return Plan(
        goal="Put the red cube in the red bin, put the blue cube in the blue "
             "bin, then stack the green cube on the red cube.",
        subtasks=[_pick(1, "red_cube"), _place(2, "red_cube", "red_bin"),
                  _pick(3, "blue_cube"), _place(4, "blue_cube", "blue_bin"),
                  _pick(5, "green_cube", ["red_cube in red_bin"]),
                  _stack(6, "green_cube", "red_cube")],
    )


def _mock_plan_for(prompt: str) -> Plan:
    """Pick a canned plan by matching instruction phrases in the prompt.

    Keeps planning fully offline while still producing DISTINCT valid ordered plans
    per instruction. Matches on POSITIVE instruction-specific phrases (the
    prompt template itself always mentions "stack" as a skill, so absence-based
    matching is unreliable). Unmatched input falls back to the full S3 plan (so
    the demo instruction and generic test inputs keep returning 6 subtasks).
    """
    t = prompt.lower()
    if "stack the green cube on the blue cube" in t:
        return _plan_s2()
    if "and put the blue cube in the blue bin" in t:
        return _plan_s1()
    return _plan_s3()


# Back-compat alias: the default mock plan.
_mock_plan = _plan_s3


def _mock_verdict() -> Verdict:
    return Verdict(satisfied=True, observed="mock observation", reason="MOCK mode")


def _mock_retry() -> RecoveryDecision:
    return RecoveryDecision(strategy="retry", new_tail=None, rationale="MOCK mode")


# Prompt-independent mocks (Plan is handled separately by _mock_plan_for).
_MOCK: dict[type[BaseModel], Callable[[], BaseModel]] = {
    Verdict: _mock_verdict,
    RecoveryDecision: _mock_retry,
}


def _extract_json(text: str) -> str:
    """Strip markdown code fences and slice to the outermost JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in response: {text!r}")
    return text[start:end + 1]


def _image_to_data_url(img: np.ndarray) -> str:
    buf = io.BytesIO()
    iio.imwrite(buf, img, extension=".png")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class LLMClient:
    """call(role, prompt, response_model, images) -> response_model instance.

    MOCK mode (default): returns a canned instance of response_model, no
    network access. Real mode: calls OpenRouter via the openai SDK, embeds
    the target JSON schema in the prompt (OpenRouter's response_format=
    json_schema is provider-dependent and hard-fails on unsupported cheap
    models, so we don't rely on it), and does ONE repair retry on a
    validation failure.
    """

    def __init__(self, cfg: dict | None = None, usage_meter: "UsageMeter | None" = None):
        self.cfg = (cfg or load_config())["llm"]
        self._client = None  # created lazily, only if a real call happens
        self.usage_meter = usage_meter

    def _real_client(self):
        if self._client is None:
            from openai import OpenAI

            base_url = os.environ.get("OPENROUTER_BASE_URL", self.cfg["base_url"])
            api_key = os.environ[self.cfg["api_key_env"]]  # never logged
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        return self._client

    def call(
        self,
        role: str,
        prompt: str,
        response_model: type[T],
        images: list[np.ndarray] | None = None,
    ) -> T:
        if self.usage_meter is not None:
            self.usage_meter.record_call(role)
        if self.cfg.get("mock", True):
            if response_model is Plan:
                return _mock_plan_for(prompt).model_copy(deep=True)  # type: ignore[return-value]
            factory = _MOCK.get(response_model)
            if factory is None:
                raise ValueError(f"no mock registered for {response_model}")
            return factory().model_copy(deep=True)  # type: ignore[return-value]
        return self._call_real(role, prompt, response_model, images)

    def _call_real(
        self,
        role: str,
        prompt: str,
        response_model: type[T],
        images: list[np.ndarray] | None,
    ) -> T:
        client = self._real_client()
        model = self.cfg["models"][role]
        schema_hint = (
            f"{prompt}\n\nReturn ONLY a JSON object matching this schema:\n"
            f"{json.dumps(response_model.model_json_schema())}"
        )
        content: list[dict] = [{"type": "text", "text": schema_hint}]
        for img in images or []:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

        messages = [{"role": "user", "content": content}]
        max_retries = self.cfg.get("max_repair_retries", 1)

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=self.cfg.get("temperature", 0.0),
                # response_format intentionally omitted by default: OpenRouter
                # structured-output support is model-dependent and hard-fails
                # on unsupported (cheap) routes. Uncomment if the chosen model
                # is verified to support it:
                # response_format={"type": "json_object"},
            )
            if self.usage_meter is not None:
                self.usage_meter.record_usage(model, getattr(resp, "usage", None))
            text = resp.choices[0].message.content
            try:
                if not text:                    # reasoning models sometimes return empty content
                    raise ValueError("empty response content")
                return response_model.model_validate_json(_extract_json(text))
            except (ValidationError, ValueError) as err:
                last_err = err
                messages.append({"role": "assistant", "content": text or ""})
                messages.append({
                    "role": "user",
                    "content": f"Your previous reply failed validation: {err}. "
                               f"Return ONLY corrected JSON matching the schema.",
                })
        raise ValueError(f"LLM response failed validation after {max_retries + 1} attempts: {last_err}")


if __name__ == "__main__":
    client = LLMClient()
    plan = client.call(
        "planner",
        "Put the red cube in the red bin, put the blue cube in the blue "
        "bin, then stack the green cube on the red cube.",
        Plan,
    )
    print(plan.model_dump_json(indent=2))
