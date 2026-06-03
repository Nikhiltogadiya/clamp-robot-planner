"""Ground-truth scene state and geometric predicate checks.

This is the oracle: it reads true object positions straight from the physics
server and decides whether human-readable predicate strings (the same ones the
planner writes into pre/postconditions) actually hold. Used for scoring and,
later, to measure how good the VLM verifier is. It is NEVER fed to the agents.

Predicate grammar (see postcondition_true):
  "<obj> in <bin>"    -> obj resting inside the tray's inner bounds
  "<obj> on <obj2>"   -> obj resting one cube-height above obj2, xy-aligned
  "gripper empty"     -> nothing grasped
  "holding <obj>"     -> obj currently grasped
Anything else parses to None (unknown / not machine-checkable).

Contract: predicates assume a settled world. skills.py always settles after a
detach, so callers should check right after a skill returns.
"""
import re

import pybullet as p

# Tolerances for 5cm cubes (half-extent 0.025). Centralized so tests and
# checks agree.
IN_BIN_XY_MARGIN = 0.0        # added to the tray inner_half; 0 = strictly inside rim
IN_BIN_Z_TOL = 0.02
ON_XY_TOL = 0.03              # <=60% of cube width center-to-center
ON_Z_TOL = 0.015

_PRED_RE = re.compile(r"^(\w+)\s+(in|on)\s+(\w+)$")


def _pos(scene, name: str):
    return p.getBasePositionAndOrientation(scene.cubes[name], physicsClientId=scene.client)[0]


def in_bin(scene, cube: str, bin_name: str) -> bool:
    x, y, z = _pos(scene, cube)
    b = scene.bins[bin_name]
    cx, cy = b["center"]
    half = b["inner_half"]
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    resting_z = b["base_top"] + cube_h            # cube center when seated on the pad
    return (
        abs(x - cx) <= half + IN_BIN_XY_MARGIN
        and abs(y - cy) <= half + IN_BIN_XY_MARGIN
        and abs(z - resting_z) <= IN_BIN_Z_TOL
    )


def on_top_of(scene, a: str, b: str) -> bool:
    xa, ya, za = _pos(scene, a)
    xb, yb, zb = _pos(scene, b)
    cube_h = 2 * scene.cfg["scene"]["cube_half_extent"]
    return (
        abs(xa - xb) <= ON_XY_TOL
        and abs(ya - yb) <= ON_XY_TOL
        and abs((za - zb) - cube_h) <= ON_Z_TOL
    )


def postcondition_true(scene, skills, cond: str):
    """Return True/False if the predicate is machine-checkable, else None."""
    cond = cond.strip()
    if cond == "gripper empty":
        return skills.held is None
    if cond.startswith("holding "):
        return skills.held == cond[len("holding "):].strip()
    m = _PRED_RE.match(cond)
    if not m:
        return None
    obj, rel, target = m.groups()
    if obj not in scene.cubes:
        return None
    if rel == "in" and target in scene.bins:
        return in_bin(scene, obj, target)
    if rel == "on" and target in scene.cubes:
        return on_top_of(scene, obj, target)
    return None


def scene_state(scene, skills) -> dict:
    """A JSON-friendly snapshot for prompting and precondition gating.

    Each object also carries its stacking relations so the planner can reason
    about order without inferring it from raw coordinates:
      on_top_of  : the cube this one is resting ON (or null)
      blocked_by : the cube resting ON TOP of this one - it must be moved first
                   before this cube can be picked (or null)
    """
    objects = {}
    for name in scene.cubes:
        x, y, z = _pos(scene, name)
        color = next(c["color"] for c in scene.cfg["scene"]["cubes"] if c["name"] == name)
        on = next((o for o in scene.cubes if o != name and on_top_of(scene, name, o)), None)
        blocker = next((o for o in scene.cubes if o != name and on_top_of(scene, o, name)), None)
        objects[name] = {"color": color, "position": [round(x, 3), round(y, 3), round(z, 3)],
                         "on_top_of": on, "blocked_by": blocker}
    bins = {}
    for name, b in scene.bins.items():
        occupants = [c for c in scene.cubes if in_bin(scene, c, name)]
        bins[name] = {"color": b["color"], "center": b["center"], "occupants": occupants}
    return {"objects": objects, "bins": bins, "holding": skills.held}


def goal_satisfied(scene, skills, plan) -> bool:
    """True iff every machine-checkable postcondition of the plan holds.

    Uses the last action touching each object (its final intended state), so a
    pick-then-place chain is judged on the place, not the transient hold.
    Unparseable predicates (e.g. "holding X" mid-plan) are ignored.
    """
    final_conds: dict[str, list[str]] = {}
    for st in plan.subtasks:
        final_conds[st.object] = st.postconditions
    for conds in final_conds.values():
        for cond in conds:
            result = postcondition_true(scene, skills, cond)
            if result is False:
                return False
    return True


def goal_predicates_satisfied(scene, skills, predicates: list[str]) -> bool:
    """True iff every predicate in `predicates` holds in the current sim state.

    This is the INDEPENDENT task-goal check used for eval scoring: the goal is
    defined by the task, not by the plan the (fallible) planner produced, so a
    planner that writes the wrong postconditions cannot score itself correct.

    A predicate that isn't machine-checkable (a typo'd cube/bin name or unknown
    grammar -> None) is a bug in the TASK definition; we raise rather than treat
    it as False, which would silently score the task unsatisfiable forever.
    """
    results = []
    for pred in predicates:
        r = postcondition_true(scene, skills, pred)
        if r is None:
            raise ValueError(f"goal predicate is not machine-checkable "
                             f"(typo or unknown cube/bin?): {pred!r}")
        results.append(r)
    return all(results)
