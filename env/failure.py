"""Failure injection - this is what CREATES the problem the recovery loop
later solves. After a skill runs, with probability p_fail, perturb the outcome
so the subtask's postcondition no longer holds:

  failed_grasp     (after pick):  object left at its source, not held
  missed_placement (after place): object lands just outside the target tray
  toppled_stack    (after stack): stacked object knocked off to the side

Each action has one natural failure mode. A deterministic `scripted` list
(config: failure.scripted) overrides the probabilistic path for reproducible
demos and tests: entry i is a mode string (or null) applied at execution step i.
"""
import random

import pybullet as p

from env.scene import settle

# Which failure each action produces when it fails.
_NATURAL_MODE = {"pick": "failed_grasp", "place": "missed_placement", "stack": "toppled_stack"}


class FailureInjector:
    def __init__(self, cfg: dict, rng: random.Random | None = None):
        fcfg = cfg["failure"]
        self.p_fail = fcfg.get("p_fail", 0.0)
        self.scripted = list(fcfg.get("scripted") or [])
        self.rng = rng or random.Random(cfg.get("seed", 0))
        self.cube_h = cfg["scene"]["cube_half_extent"]
        self.settle_steps = cfg["sim"].get("settle_steps", 240)
        self.step = 0

    def _decide(self, action: str, idx: int) -> str | None:
        if self.scripted:                       # deterministic override
            return (self.scripted[idx] or None) if idx < len(self.scripted) else None
        if self.rng.random() >= self.p_fail:
            return None
        return _NATURAL_MODE.get(action)

    def maybe_perturb(self, action: str, obj: str, target, scene, skills) -> str | None:
        """Call right AFTER a skill runs. Returns the injected mode, or None."""
        idx = self.step
        self.step += 1
        mode = self._decide(action, idx)
        if mode is None:
            return None
        self._apply(mode, obj, target, scene, skills)
        settle(scene.client, self.settle_steps)
        return mode

    def _apply(self, mode: str, obj: str, target, scene, skills) -> None:
        cube = scene.cubes[obj]
        if mode == "failed_grasp":
            # The grasp slipped: release if held and let the cube drop straight
            # down to the table from wherever the gripper currently holds it -
            # NOT back to its original config start (that was an impossible jump
            # once the cube had already been relocated by an earlier step).
            if skills.held == obj and skills.grasp_cid is not None:
                p.removeConstraint(skills.grasp_cid, physicsClientId=scene.client)
                skills.grasp_cid = None
                skills.held = None
            x, y, _ = p.getBasePositionAndOrientation(cube, physicsClientId=scene.client)[0]
            skills._teleport(cube, [x, y, self.cube_h])
        elif mode == "missed_placement":
            # Drop the cube on open floor OUTSIDE the target tray, along the ray
            # from the tray centre toward the workspace front-centre. The clearance
            # is derived from the tray geometry (inner_half + cube + margin) so the
            # miss is guaranteed outside the rim regardless of bin layout - both
            # the geometry check and the camera read it unambiguously as "missed".
            b = scene.bins[target]
            cx, cy = b["center"]
            dx, dy = 0.50 - cx, 0.0 - cy               # toward workspace front-centre
            d = (dx * dx + dy * dy) ** 0.5 or 1.0
            clear = b["inner_half"] + self.cube_h + 0.06
            skills._teleport(cube, [cx + dx / d * clear, cy + dy / d * clear, self.cube_h])
        elif mode == "toppled_stack":
            # Knocked off the stack onto open floor toward the arm (-x), far enough
            # to clear any tray the stack sat over (a bin-stack topple must travel
            # past the tray rim; a plain +x nudge landed inside the wall).
            x, y, _ = p.getBasePositionAndOrientation(cube, physicsClientId=scene.client)[0]
            skills._teleport(cube, [max(0.30, x - 0.15), y, self.cube_h])
        else:
            raise ValueError(f"unknown failure mode {mode!r}")
