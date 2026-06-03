"""Manipulation skills via "magic grasp" - no trajectory planning, no grasp
physics. Each skill is a state transition:

  pick(obj):   teleport obj to the end-effector, attach a fixed constraint
  place(obj):  detach, drop obj inside the target tray, let it settle
  stack(obj):  detach, drop obj on top of another cube, let it settle

With sim.use_ik: true, pick/place also drive the arm to a single IK pose over
the object first, purely for nicer visuals (one IK call, still no trajectory).
The controller and agents only ever see the resulting state transitions.
"""
import math

import pybullet as p

from env.scene import ARM_JOINTS, settle

IDENTITY_QUAT = [0, 0, 0, 1]


class Skills:
    def __init__(self, scene, cfg: dict):
        self.scene = scene
        self.client = scene.client
        self.cfg = cfg
        self.held: str | None = None
        self.grasp_cid: int | None = None
        self.drop_height = cfg["skills"]["drop_height"]
        self.settle_steps = cfg["sim"].get("settle_steps", 240)
        self.use_ik = cfg["sim"].get("use_ik", False)
        self.cube_h = cfg["scene"]["cube_half_extent"]

    # --- helpers -----------------------------------------------------------
    def _ee_pos(self):
        return p.getLinkState(self.scene.robot, self.scene.ee_link,
                              physicsClientId=self.client)[4]

    def _cube_pos(self, name: str):
        return p.getBasePositionAndOrientation(self.scene.cubes[name],
                                               physicsClientId=self.client)[0]

    def _teleport(self, cube_id: int, pos):
        p.resetBasePositionAndOrientation(cube_id, pos, IDENTITY_QUAT,
                                          physicsClientId=self.client)
        # resetBasePositionAndOrientation does NOT zero velocity - do it explicitly
        # or the cube keeps any momentum and bounces out of the tray.
        p.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

    def _move_ee_above(self, pos, hover=0.10):
        """IK-mode only: drive the arm to a single down-facing pose above pos."""
        target = [pos[0], pos[1], pos[2] + hover]
        down = p.getQuaternionFromEuler([math.pi, 0, 0])
        poses = p.calculateInverseKinematics(
            self.scene.robot, self.scene.ee_link, target, down,
            maxNumIterations=50, residualThreshold=1e-4, physicsClientId=self.client)
        for i in ARM_JOINTS:
            p.setJointMotorControl2(self.scene.robot, i, p.POSITION_CONTROL,
                                    poses[i], force=240, maxVelocity=2,
                                    physicsClientId=self.client)
        settle(self.client, 120)

    # --- skills ------------------------------------------------------------
    def pick(self, obj: str, target=None) -> None:
        # Graceful degradation: if the gripper is unexpectedly full (the world
        # diverged from the plan, e.g. after an injected failure), the open-loop
        # baseline just skips rather than crashing - the bad state then stands
        # for the verifier / ground truth to catch ("barrel on").
        if self.held is not None:
            return
        cube_id = self.scene.cubes[obj]
        if self.use_ik:
            self._move_ee_above(self._cube_pos(obj))
        # teleport-then-attach: move the cube onto the EE first so the fixed
        # constraint has a zero-length offset and never yanks the cube.
        self._teleport(cube_id, self._ee_pos())
        self.grasp_cid = p.createConstraint(
            parentBodyUniqueId=self.scene.robot, parentLinkIndex=self.scene.ee_link,
            childBodyUniqueId=cube_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
            parentFrameOrientation=IDENTITY_QUAT, childFrameOrientation=IDENTITY_QUAT,
            physicsClientId=self.client)
        p.changeConstraint(self.grasp_cid, maxForce=200, physicsClientId=self.client)
        settle(self.client, 24)
        self.held = obj

    def _release_to(self, obj: str, pos) -> None:
        # Graceful no-op if we aren't actually holding obj (e.g. a prior grasp
        # failed): the cube stays where it is and the postcondition simply won't
        # hold, which is exactly what the verifier should detect.
        if self.held != obj:
            return
        p.removeConstraint(self.grasp_cid, physicsClientId=self.client)
        self.grasp_cid = None
        self._teleport(self.scene.cubes[obj], pos)
        settle(self.client, self.settle_steps)
        self.held = None

    def place(self, obj: str, target: str) -> None:
        b = self.scene.bins[target]
        pos = self._bin_drop(obj, target, b)
        if self.use_ik:
            self._move_ee_above(pos)
        self._release_to(obj, pos)

    def _bin_drop(self, obj: str, target: str, b: dict):
        """Full drop position [x, y, z] for placing obj into bin `target`, chosen
        so cubes never pack into a horizontal overlap. An empty bin -> centre. If
        the bin already holds cube(s), spread all of them across evenly-spaced,
        non-overlapping slots so they genuinely REST side-by-side (a stacked cube
        sits a cube-height up and correctly reads as NOT in the bin - this fixed a
        silent mis-score of a 'both cubes in one bin' goal). The tray only fits a
        few cubes; BEYOND that, drop this one STACKED on an existing occupant (a
        stable vertical rest) rather than into a horizontal overlap, which the
        physics solver resolves violently. Occupants are nudged aside to fit."""
        from env.state import in_bin
        cx, cy = b["center"]
        floor_z = b["base_top"] + self.cube_h + self.drop_height
        others = [c for c in self.scene.cubes
                  if c != obj and in_bin(self.scene, c, target)]
        if not others:
            return [cx, cy, floor_z]
        max_off = b["inner_half"] - self.cube_h            # keep every cube inside the rim
        min_gap = 2 * self.cube_h + 0.005
        max_slots = max(1, int(2 * max_off / min_gap) + 1)  # cubes that fit side-by-side
        members = [*others, obj]
        if len(members) <= max_slots:
            n = len(members)
            ys = [cy - max_off + (2 * max_off) * k / (n - 1) for k in range(n)]
            resting_z = b["base_top"] + self.cube_h
            for c, y in zip(others, ys):                    # nudge existing occupants aside
                self._teleport(self.scene.cubes[c], [cx, y, resting_z])
            settle(self.client, self.settle_steps)
            return [cx, ys[-1], floor_z]                    # obj takes the remaining slot
        # bin full: stack this cube on an occupant (stable) instead of overlapping
        support = others[0]
        sx, sy, sz = p.getBasePositionAndOrientation(
            self.scene.cubes[support], physicsClientId=self.client)[0]
        return [sx, sy, sz + 2 * self.cube_h + self.drop_height]

    def stack(self, obj: str, on: str) -> None:
        ox, oy, oz = self._cube_pos(on)
        drop_z = oz + 2 * self.cube_h + self.drop_height
        if self.use_ik:
            self._move_ee_above([ox, oy, drop_z])
        self._release_to(obj, [ox, oy, drop_z])

    def put_down(self, obj: str) -> None:
        """Put the held cube down on the open table, OUTSIDE any bin - the only
        way to place a cube where it's not in a bin or on another cube (for
        'take it out' / 'put it on the table' instructions)."""
        pos = self._free_table_spot(obj)
        if self.use_ik:
            self._move_ee_above(pos)
        self._release_to(obj, pos)

    def _free_table_spot(self, obj: str):
        """A clear spot on the open table for obj: not overlapping another cube,
        and outside every bin's footprint. Tries the cube's config home first,
        then a small grid across the front workspace; falls back to the home."""
        home = next(c["start_pos"] for c in self.cfg["scene"]["cubes"] if c["name"] == obj)
        candidates = [(home[0], home[1])]
        for x in (0.40, 0.48, 0.56):
            for y in (0.0, -0.12, 0.12, -0.24, 0.24):
                candidates.append((x, y))
        min_gap = 2 * self.cube_h + 0.02
        for x, y in candidates:
            if self._table_spot_clear(x, y, obj, min_gap):
                return [x, y, self.cube_h]
        return [home[0], home[1], self.cube_h]

    def _table_spot_clear(self, x: float, y: float, obj: str, min_gap: float) -> bool:
        for name in self.scene.cubes:
            if name == obj:
                continue
            ox, oy, _ = self._cube_pos(name)
            if ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 < min_gap:
                return False
        for b in self.scene.bins.values():
            bx, by = b["center"]
            half = b["inner_half"] + b["wall_thickness"] + self.cube_h
            if abs(x - bx) <= half and abs(y - by) <= half:
                return False
        return True

    def execute(self, subtask) -> None:
        """Dispatch a Subtask to the matching skill."""
        if subtask.action == "pick":
            self.pick(subtask.object)
        elif subtask.action == "place":
            self.place(subtask.object, subtask.target)
        elif subtask.action == "stack":
            self.stack(subtask.object, subtask.target)
        elif subtask.action == "put_down":
            self.put_down(subtask.object)
        else:
            raise ValueError(f"unknown action {subtask.action!r}")
