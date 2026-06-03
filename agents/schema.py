# agents/schema.py
from pydantic import BaseModel
from typing import Literal, Optional

Action = Literal["pick", "place", "stack", "put_down"]

class Subtask(BaseModel):
    id: int
    action: Action
    object: str                 # e.g. "red_cube"
    target: Optional[str]       # bin/zone or object to stack on
    preconditions: list[str]    # e.g. ["blue_cube in left_bin", "gripper empty"]
    postconditions: list[str]   # e.g. ["red_cube on blue_cube"]
    assigned_agent: str = "arm_0"   # future-proofs multi-robot; single value for now

class Plan(BaseModel):
    goal: str
    subtasks: list[Subtask]

class Verdict(BaseModel):
    satisfied: bool
    observed: str               # what the VLM saw
    reason: str

class RecoveryDecision(BaseModel):
    strategy: Literal["retry", "replan"]
    new_tail: Optional[list[Subtask]] = None   # required if strategy == "replan"
    rationale: str
