"""Cooperative episode driver for the operator console.

Reimplements loop.controller.run's closed-loop logic (plan -> precondition
gate -> execute -> verify -> recover) as a step-emitting driver that (a) can
be sabotaged BETWEEN steps via an externally-supplied command queue, and
(b) is UI-agnostic - it takes plain callables (`emit`, `poll_commands`), so
it is testable without Viser and reusable by both the console and tests.
The canonical loop.controller.run is untouched and remains the tested/eval
path; nothing here changes it or is imported by it.
"""
import pybullet as p

from env.scene import settle
from env.state import goal_satisfied, on_top_of
from loop.controller import preconditions_met

SABOTAGE_MODES = ("knock_off", "steal", "topple")
SABOTAGE_DIRECTIONS = ("left", "right", "forward", "backward")

# Horizontal-only: the workspace is a flat tabletop, so every cube always
# rests at a fixed height (floor / tray / another cube). "Down" would clip
# into whatever it's resting on; "up" would just fall back to nearly the same
# spot once physics settles. Four sideways directions are the physically
# meaningful ones. left/right = -y/+y (matches the camera calibration: +y
# reads image-right); forward/backward = +x/-x (away from / toward the arm
# base, per env/scene.py's "Panda at origin facing +x" convention).
_DIRECTION_VECTORS = {
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
}
_SABOTAGE_STEP = 0.09
# Clamp bounds keep a displaced cube on the rendered floor and within the
# arm's real reach (env/scene.py's documented workspace is x in [0.3, 0.8],
# y in [-0.3, 0.3]; a little slack here still stays reachable).
_WORKSPACE_X = (0.25, 0.85)
_WORKSPACE_Y = (-0.35, 0.35)
# Minimum center-to-center gap to keep between cubes. Teleporting a cube to
# overlap another one doesn't just look wrong - PyBullet's solver resolves
# the interpenetration with a violent separating impulse during settle(),
# launching one or both cubes a large, unpredictable distance (confirmed by
# reproducing a live bug report: a "right" push landed 4cm from another cube
# - under the 5cm needed to avoid overlap - and both cubes flew across the
# table). Anything closer than this is refused rather than teleported into.
_MIN_CUBE_GAP_MARGIN = 0.02
# Give up replanning a step after this many attempts (matches loop.controller).
_MAX_REPLANS = 3
# Global cap: each replan shifts the failing step to a new index, so the
# per-step cap alone never fires on a persistent verifier disagreement (e.g. a
# scene the VLM can't perceive). Stop after this many total replans per episode.
_MAX_TOTAL_REPLANS = 4
# Outer loop: per-step verification only checks each step as it runs, so a cube
# sabotaged AFTER its step already passed is never re-examined and the episode
# ends FAILED with 0 recoveries. After the plan finishes we re-check the whole
# goal and, if it's broken, re-plan from the current scene and run that - up to
# this many times. This is what makes recovery robust to sabotage at any moment.
_MAX_GOAL_ATTEMPTS = 3


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _blocking_cube(scene, exclude_ids, nx: float, ny: float, nz: float, cube_h: float) -> str | None:
    """Name of the first OTHER cube (not in `exclude_ids`) that (nx, ny, nz)
    would overlap, or None if the spot is clear. Uses full 3D distance -
    height matters: a cube two levels below a tower's top shares the same
    (x, y) as everything above it, so an (x, y)-only check would wrongly call
    it "in the way" of a target that's actually a safe height above it. Shared
    by every action that teleports a cube somewhere, so nothing can create the
    interpenetration that made PyBullet's solver violently launch cubes."""
    min_gap = 2 * cube_h + _MIN_CUBE_GAP_MARGIN
    for other_name, other_id in scene.cubes.items():
        if other_id in exclude_ids:
            continue
        ox, oy, oz = p.getBasePositionAndOrientation(other_id, physicsClientId=scene.client)[0]
        if ((nx - ox) ** 2 + (ny - oy) ** 2 + (nz - oz) ** 2) ** 0.5 < min_gap:
            return other_name
    return None


def _cube_resting_on(scene, obj: str) -> str | None:
    """Name of whatever cube is CURRENTLY resting on top of `obj`, if any.
    Moving `obj` horizontally without also moving its rider would leave the
    rider floating in place, then falling once physics settles - exactly the
    "half worked" bug this guards against."""
    return next((other for other in scene.cubes
                if other != obj and on_top_of(scene, other, obj)), None)


def _push(scene, skills, obj: str, direction: str, cube_h: float, settle_steps: int) -> str | None:
    """Move `obj` one step in `direction`, clamped to the workspace. Returns a
    human-readable reason if the move was refused (and does NOT move it in
    that case), else None on a normal move."""
    if direction not in _DIRECTION_VECTORS:
        raise ValueError(f"unknown sabotage direction {direction!r}")

    rider = _cube_resting_on(scene, obj)
    if rider:
        return f"{rider} is stacked on top of {obj} - move it out of the way first"

    cube_id = scene.cubes[obj]
    dx, dy = _DIRECTION_VECTORS[direction]
    x, y, z = p.getBasePositionAndOrientation(cube_id, physicsClientId=scene.client)[0]
    nx = _clamp(x + dx * _SABOTAGE_STEP, *_WORKSPACE_X)
    ny = _clamp(y + dy * _SABOTAGE_STEP, *_WORKSPACE_Y)

    blocker = _blocking_cube(scene, {cube_id}, nx, ny, z, cube_h)   # z unchanged: a horizontal push
    if blocker:
        return f"{blocker} is in the way"

    skills._teleport(cube_id, [nx, ny, z])
    settle(scene.client, settle_steps)
    return None


def apply_place_on_top(scene, skills, obj: str, on: str) -> str:
    """Manually place `obj` directly on top of `on`, right now - a
    constructive counterpart to the sabotage actions, for setting up a
    custom scene before asking the planner to solve it. Works for ANY pair
    of cubes, and for towers of any height: `on`'s CURRENT position is read
    live (same as skills.stack()), and the collision check is full 3D, so
    stacking on top of an already-stacked cube works correctly. If `obj` is
    currently held, it's released first, same as sabotage's "steal"."""
    if obj not in scene.cubes or on not in scene.cubes:
        return f"unknown object(s): {obj!r}, {on!r}"
    if obj == on:
        return f"{obj} can't be placed on top of itself"

    rider = _cube_resting_on(scene, obj)
    if rider:
        return f"can't move {obj} - {rider} is stacked on top of it (move it first)"

    cube_id, on_id = scene.cubes[obj], scene.cubes[on]
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    settle_steps = scene.cfg["sim"].get("settle_steps", 240)

    if skills.held == obj and skills.grasp_cid is not None:
        p.removeConstraint(skills.grasp_cid, physicsClientId=scene.client)
        skills.grasp_cid = None
        skills.held = None

    ox, oy, oz = p.getBasePositionAndOrientation(on_id, physicsClientId=scene.client)[0]
    nx, ny, nz = ox, oy, oz + 2 * cube_h

    blocker = _blocking_cube(scene, {cube_id, on_id}, nx, ny, nz, cube_h)
    if blocker:
        return f"can't place {obj} on {on} - {blocker} is in the way"

    skills._teleport(cube_id, [nx, ny, nz])
    settle(scene.client, settle_steps)
    return f"{obj} placed on top of {on}"


def _safe_drop(scene, skills, cube_id: int, x: float, y: float,
               cube_h: float, settle_steps: int) -> None:
    """Drop a released cube on the table at (x, y), nudged to the nearest clear
    spot if that would overlap another cube. Teleporting a cube into an overlap
    makes PyBullet's solver launch both across the table - the same failure the
    _push / place-on-top guards prevent; steal must not reintroduce it (it can:
    steal a cube that was just picked off a stack and the gripper sits right over
    the cube below)."""
    nx, ny = _clamp(x, *_WORKSPACE_X), _clamp(y, *_WORKSPACE_Y)
    if _blocking_cube(scene, {cube_id}, nx, ny, cube_h, cube_h) is not None:
        step = 2 * cube_h + _MIN_CUBE_GAP_MARGIN
        for r in range(1, 6):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)):
                cx = _clamp(nx + dx * step * r, *_WORKSPACE_X)
                cy = _clamp(ny + dy * step * r, *_WORKSPACE_Y)
                if _blocking_cube(scene, {cube_id}, cx, cy, cube_h, cube_h) is None:
                    nx, ny = cx, cy
                    break
            else:
                continue
            break
    skills._teleport(cube_id, [nx, ny, cube_h])
    settle(scene.client, settle_steps)


def apply_sabotage(scene, skills, mode: str, obj: str, direction: str = "right") -> str:
    """Displace `obj` right now, regardless of what the plan is doing. Returns
    a human-readable description for the event feed. Uses the same
    `skills._teleport` primitive env/failure.py's scripted failures use.

    `direction` (left/right/forward/backward) controls knock_off/topple;
    steal ignores it (it drops the cube near the gripper, not a chosen way).
    Each mode has its own precondition, mirroring env/failure.py's three
    failure archetypes:
      - steal:     only if `obj` is currently held (like a failed grasp)
      - topple:    only if `obj` is currently stacked on another cube
      - knock_off: always applicable (a generic displacement)
    """
    if obj not in scene.cubes:
        return f"unknown object {obj!r}"
    cube_id = scene.cubes[obj]
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    settle_steps = scene.cfg["sim"].get("settle_steps", 240)

    if mode == "steal":
        if skills.held == obj and skills.grasp_cid is not None:
            p.removeConstraint(skills.grasp_cid, physicsClientId=scene.client)
            skills.grasp_cid = None
            skills.held = None
            ee = p.getLinkState(scene.robot, scene.ee_link, physicsClientId=scene.client)[4]
            _safe_drop(scene, skills, cube_id, ee[0], ee[1], cube_h, settle_steps)
            return f"{obj} stolen out of the gripper mid-grasp"
        return f"{obj} isn't currently held - nothing to steal"

    if mode == "topple":
        support = next((other for other in scene.cubes
                        if other != obj and on_top_of(scene, obj, other)), None)
        if support is None:
            return f"{obj} isn't stacked on anything - nothing to topple"
        reason = _push(scene, skills, obj, direction, cube_h, settle_steps)
        if reason:
            return f"can't topple {obj} ({direction}) - {reason}"
        return f"{obj} toppled off {support} ({direction})"

    if mode == "knock_off":
        reason = _push(scene, skills, obj, direction, cube_h, settle_steps)
        if reason:
            return f"can't knock {obj} off ({direction}) - {reason}"
        return f"{obj} knocked off ({direction})"

    raise ValueError(f"unknown sabotage mode {mode!r}")


def _subtask_dict(st) -> dict:
    return {"id": st.id, "action": st.action, "object": st.object, "target": st.target}


def _execute_plan(plan, scene, skills, verifier, recovery, recovery_on, max_steps,
                  emit, poll_commands, events) -> tuple[int, int, int]:
    """Run one plan to completion: for each subtask drain sabotage commands ->
    precondition gate (+ recover) -> execute -> verify (+ recover). Mutates
    plan.subtasks on an inner replan. Returns (n_verify, n_recover, steps)."""
    i, steps = 0, 0
    n_verify = n_recover = 0
    retries: dict[int, int] = {}
    replans: dict[int, int] = {}
    total_replans = 0

    while i < len(plan.subtasks) and steps < max_steps:
        for cmd in poll_commands():
            if cmd.get("type") == "sabotage":
                desc = apply_sabotage(scene, skills, cmd["mode"], cmd["object"],
                                     cmd.get("direction", "right"))
                emit({"type": "sabotage", "description": desc})
            elif cmd.get("type") == "place_on_top":
                desc = apply_place_on_top(scene, skills, cmd["object"], cmd["on"])
                emit({"type": "place_on_top", "description": desc})

        st = plan.subtasks[i]
        steps += 1
        emit({"type": "step_start", "index": i, "subtask": _subtask_dict(st)})

        if not preconditions_met(scene, skills, st):
            if recovery_on() and recovery is not None and total_replans < _MAX_TOTAL_REPLANS:
                dec = recovery.decide(plan.goal, plan, i, {}, scene, skills)
                n_recover += 1
                events.append({"step": i, "phase": "precondition", "strategy": dec.strategy})
                emit({"type": "recover", "index": i, "phase": "precondition",
                      "strategy": dec.strategy, "rationale": dec.rationale})
                if (dec.strategy == "replan" and dec.new_tail
                        and replans.get(i, 0) < _MAX_REPLANS
                        and total_replans < _MAX_TOTAL_REPLANS):
                    replans[i] = replans.get(i, 0) + 1
                    total_replans += 1
                    plan.subtasks[i:] = dec.new_tail
                    emit({"type": "plan_update", "goal": plan.goal,
                          "subtasks": [_subtask_dict(s) for s in plan.subtasks]})
                    continue

        skills.execute(st)
        emit({"type": "executed", "index": i, "subtask": _subtask_dict(st)})

        verdicts = verifier.verify_subtask(scene, st, skills)
        if verdicts:                     # empty => nothing visual to check (e.g. a pick)
            n_verify += 1
        satisfied = all(v.satisfied for v in verdicts.values()) if verdicts else True
        reasons = {c: v.reason for c, v in verdicts.items()}
        emit({"type": "step_verified", "index": i, "subtask": _subtask_dict(st),
              "satisfied": satisfied, "reasons": reasons})

        if satisfied:
            i += 1
            continue
        if not recovery_on() or recovery is None or total_replans >= _MAX_TOTAL_REPLANS:
            i += 1                   # recovery-off, or global replan budget spent: barrel on
            continue

        dec = recovery.decide(plan.goal, plan, i, verdicts, scene, skills)
        n_recover += 1
        retries[i] = retries.get(i, 0) + 1
        events.append({"step": i, "phase": "recover", "strategy": dec.strategy,
                       "attempt": retries[i]})
        emit({"type": "recover", "index": i, "phase": "verify",
              "strategy": dec.strategy, "rationale": dec.rationale})
        if (dec.strategy == "replan" and dec.new_tail
                and replans.get(i, 0) < _MAX_REPLANS
                and total_replans < _MAX_TOTAL_REPLANS):
            replans[i] = replans.get(i, 0) + 1
            total_replans += 1
            plan.subtasks[i:] = dec.new_tail
            retries.pop(i, None)
            emit({"type": "plan_update", "goal": plan.goal,
                  "subtasks": [_subtask_dict(s) for s in plan.subtasks]})
        elif (retries[i] > 2 or replans.get(i, 0) >= _MAX_REPLANS
              or total_replans >= _MAX_TOTAL_REPLANS):
            i += 1
    return n_verify, n_recover, steps


def run_episode(instruction, scene, skills, planner, verifier, recovery, *,
                recovery_enabled=True, max_steps=40, emit=None, poll_commands=None) -> dict:
    """Plan + execute an instruction, emitting a rich event after every
    meaningful moment, and draining `poll_commands()` for live sabotage between
    steps. Never raises: a planner/execution error is emitted as an "error"
    event and returned as a failed result, so a bad instruction can't crash the
    console.

    Two nested loops: an INNER per-step loop (execute -> verify -> recover, in
    `_execute_plan`) and an OUTER goal loop here. Per-step verification only
    checks each step as it runs, so a cube sabotaged AFTER its step already
    passed is never re-examined; the outer loop re-checks the whole goal when
    the plan finishes and, if it's broken, re-plans from the current scene and
    runs that - so recovery is robust to sabotage at any moment."""
    emit = emit or (lambda e: None)
    poll_commands = poll_commands or (lambda: [])
    # recovery_enabled may be a live callable so a mid-episode toggle takes effect
    recovery_on = recovery_enabled if callable(recovery_enabled) else (lambda: recovery_enabled)

    try:
        emit({"type": "planning", "instruction": instruction})
        plan = planner.plan(instruction, scene, skills)
        emit({"type": "plan_ready", "goal": plan.goal,
              "subtasks": [_subtask_dict(st) for st in plan.subtasks]})
    except Exception as e:
        emit({"type": "error", "phase": "planning", "message": f"{type(e).__name__}: {e}"})
        return {"instruction": instruction, "success": False, "steps": 0, "events": [],
                "plan": None, "n_verify": 0, "n_recover": 0}

    goal_plan = plan                 # the ORIGINAL plan defines the goal (its postconditions)
    n_verify = n_recover = steps = 0
    events: list[dict] = []

    try:
        for attempt in range(_MAX_GOAL_ATTEMPTS):
            nv, nr, ns = _execute_plan(plan, scene, skills, verifier, recovery, recovery_on,
                                       max_steps, emit, poll_commands, events)
            n_verify += nv; n_recover += nr; steps += ns
            if goal_satisfied(scene, skills, goal_plan):
                break
            if attempt == _MAX_GOAL_ATTEMPTS - 1 or not recovery_on() or recovery is None:
                break
            # goal still broken (e.g. a cube sabotaged after its step passed, so no
            # per-step check re-examined it) -> re-plan from the CURRENT scene to
            # fix whatever is now wrong, and run that plan.
            n_recover += 1
            events.append({"phase": "goal", "strategy": "replan"})
            emit({"type": "recover", "index": -1, "phase": "goal", "strategy": "replan",
                  "rationale": "the goal isn't satisfied yet - re-planning from the current "
                               "scene to fix what's still wrong"})
            try:
                plan = planner.plan(instruction, scene, skills)
            except Exception as e:
                emit({"type": "error", "phase": "replan", "message": f"{type(e).__name__}: {e}"})
                break
            emit({"type": "plan_update", "goal": plan.goal,
                  "subtasks": [_subtask_dict(s) for s in plan.subtasks]})
    except Exception as e:
        emit({"type": "error", "phase": "execution", "message": f"{type(e).__name__}: {e}"})

    success = goal_satisfied(scene, skills, goal_plan)
    result = {"instruction": instruction, "success": success, "steps": steps,
              "events": events, "plan": plan, "n_verify": n_verify, "n_recover": n_recover}
    emit({"type": "done", "success": success, "steps": steps,
          "n_verify": n_verify, "n_recover": n_recover})
    return result
