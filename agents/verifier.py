"""Verifier agent: does the rendered scene match the expected postcondition?

Renders the current scene to RGB and asks a vision model to judge each of the
just-executed subtask's postconditions from the image alone. Returns one
Verdict per postcondition. The ground-truth check in env/state.py is logged
alongside (by the probe) to measure how good the VLM actually is - the verifier
itself never sees ground truth.
"""
from pathlib import Path

from agents.config import load_config
from agents.llm_client import LLMClient
from agents.schema import Verdict
from env.render import render_rgb

_PROMPT_TEMPLATE = (Path(__file__).resolve().parent.parent / "prompts" / "verifier.txt").read_text()

# Only spatial relations are reliably judgeable from a single still frame.
# "gripper empty" / "holding X" describe transient arm state the camera can't
# see cleanly (the arm sits at rest pose after a magic-grasp teleport), so the
# verifier skips them.
_VISUAL_PREFIXES = (" in ", " on ")


def is_visual(postcondition: str) -> bool:
    return any(k in postcondition for k in _VISUAL_PREFIXES)


class Verifier:
    def __init__(self, cfg: dict | None = None, client: LLMClient | None = None):
        self.cfg = cfg or load_config()
        self.client = client or LLMClient(self.cfg)

    def verify(self, scene, postcondition: str, rgb=None) -> Verdict:
        """Judge a single postcondition string from a rendered image."""
        if rgb is None:
            rgb = render_rgb(scene)
        prompt = _PROMPT_TEMPLATE.replace("__POSTCONDITION__", postcondition)
        return self.client.call("verifier", prompt, Verdict, images=[rgb])

    def verify_subtask(self, scene, subtask, skills=None) -> dict[str, Verdict]:
        """Verify every visually-checkable postcondition of a subtask.

        Renders once and reuses the frame for all of the subtask's conditions.
        `skills` is accepted for a uniform interface with OracleVerifier.
        """
        rgb = render_rgb(scene)
        return {c: self.verify(scene, c, rgb=rgb)
                for c in subtask.postconditions if is_visual(c)}


class OracleVerifier:
    """A perfect verifier backed by geometric ground truth instead of the VLM.

    Used to develop and test the closed-loop CONTROLLER offline/deterministically
    (no tokens, no image noise), and as the "perfect verifier" reference. The
    real VLM Verifier is the honest end-to-end verifier. Same duck-typed
    interface (verify_subtask) so the controller is agnostic to which is used.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()

    def verify_subtask(self, scene, subtask, skills=None) -> dict[str, Verdict]:
        from env.state import postcondition_true
        out: dict[str, Verdict] = {}
        for c in subtask.postconditions:
            if not is_visual(c):
                continue
            ok = bool(postcondition_true(scene, skills, c))
            out[c] = Verdict(satisfied=ok, observed="ground truth",
                             reason="oracle" if ok else "postcondition false in sim")
        return out
