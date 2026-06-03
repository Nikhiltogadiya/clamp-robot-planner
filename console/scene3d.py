"""Mirrors the PyBullet CLAMP scene into a Viser browser 3D view.

PyBullet itself never renders to the browser (it stays headless/DIRECT for the
console - see console/driver.py); this module rebuilds the same geometry as
env/scene.py in Viser and repositions it every step to match the live sim.

Quaternion convention: pybullet returns (x, y, z, w) scalar-last; Viser's
`wxyz` fields are scalar-first - every conversion here reorders explicitly.
"""
import os
import time
from functools import partial

import numpy as np
import pybullet_data
import viser
import yourdfpy
from viser.extras import ViserUrdf

from env.observe import object_poses, robot_joint_angles

_ANIM_FPS = 30           # tween frames per second pushed to the browser


def _xyzw_to_wxyz(xyzw) -> tuple[float, float, float, float]:
    x, y, z, w = xyzw
    return (w, x, y, z)


def _smoothstep(t: float) -> float:
    """Ease-in-out so tweened motion starts/stops gently, not linearly."""
    return t * t * (3.0 - 2.0 * t)


def _resolve_panda_mesh(fname: str, urdf_dir: str) -> str:
    """The bundled panda.urdf uses `package://meshes/...` as a bare directive
    prefix (not real ROS package semantics) - yourdfpy's built-in package
    handler strips a package NAME segment and looks in the wrong place.
    Stripping the literal prefix and joining with the URDF's own directory
    resolves the real files (verified: 41 meshes load with real geometry)."""
    if fname.startswith("package://"):
        fname = fname[len("package://"):]
    return os.path.join(urdf_dir, fname)


class Console3D:
    """Owns every Viser scene node and keeps them in sync with a Scene."""

    def __init__(self, server: viser.ViserServer, scene, skills):
        self.server = server
        self.cube_handles: dict[str, viser.SceneNodeHandle] = {}
        self._disp_cubes: dict[str, np.ndarray] = {}   # positions currently shown
        self._disp_joints: np.ndarray | None = None    # joint angles currently shown
        self._build_static(scene)
        self._build_cubes(scene)
        self._build_robot(scene)
        self.sync(scene, skills)

    def _build_static(self, scene) -> None:
        cfg = scene.cfg
        floor_rgba = cfg["scene"].get("floor_rgba", [0.82, 0.82, 0.85, 1.0])
        self.server.scene.add_box(
            "/floor", color=_rgb255(floor_rgba), dimensions=(2.0, 2.0, 0.02),
            position=(0.0, 0.0, -0.02))
        for b in cfg["scene"]["bins"]:
            self._add_tray(b)

    def _add_tray(self, spec: dict) -> None:
        cx, cy = spec["center"]
        inner, thick, wall_h, base_top = (spec["inner_half"], spec["wall_thickness"],
                                          spec["wall_height"], spec["base_top"])
        color = _rgb255(spec["rgba"])
        outer = inner + thick
        name = spec["name"]

        def box(suffix, half_extents, pos):
            self.server.scene.add_box(f"/bins/{name}/{suffix}", color=color,
                                      dimensions=tuple(2 * h for h in half_extents),
                                      position=pos)

        box("pad", [outer, outer, base_top / 2], (cx, cy, base_top / 2))
        wall_cz = base_top + wall_h / 2
        box("wall_px", [thick / 2, outer, wall_h / 2], (cx + inner + thick / 2, cy, wall_cz))
        box("wall_nx", [thick / 2, outer, wall_h / 2], (cx - inner - thick / 2, cy, wall_cz))
        box("wall_py", [inner, thick / 2, wall_h / 2], (cx, cy + inner + thick / 2, wall_cz))
        box("wall_ny", [inner, thick / 2, wall_h / 2], (cx, cy - inner - thick / 2, wall_cz))

    def _build_cubes(self, scene) -> None:
        half = scene.cfg["scene"]["cube_half_extent"]
        rgba_by_name = {c["name"]: c["rgba"] for c in scene.cfg["scene"]["cubes"]}
        for name in scene.cubes:
            self.cube_handles[name] = self.server.scene.add_box(
                f"/cubes/{name}", color=_rgb255(rgba_by_name[name]),
                dimensions=(2 * half, 2 * half, 2 * half))

    def _build_robot(self, scene) -> None:
        urdf_path = os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")
        urdf = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=partial(_resolve_panda_mesh, urdf_dir=os.path.dirname(urdf_path)))
        self.robot = ViserUrdf(self.server, urdf, root_node_name="/robot")

    def sync(self, scene, skills) -> None:
        """Snap the browser view to the current PyBullet state instantly. Cheap
        (a handful of websocket messages)."""
        poses = object_poses(scene)
        joints = np.asarray(robot_joint_angles(scene), dtype=float)
        for name, (pos, orn) in poses.items():
            handle = self.cube_handles[name]
            handle.position = pos
            handle.wxyz = _xyzw_to_wxyz(orn)
        self.robot.update_cfg(joints)
        self._disp_cubes = {n: np.asarray(pos, dtype=float) for n, (pos, _) in poses.items()}
        self._disp_joints = joints

    def animate_to(self, scene, skills, duration: float = 0.9) -> None:
        """Smoothly tween the browser view from what's currently shown to the live
        sim state over `duration` seconds. VISUAL ONLY - the sim is already at the
        final state; the real skills teleport ("magic grasp") so a step would
        otherwise snap instantly. This makes each step watchable: the cube glides
        to the gripper / into the bin and the arm sweeps, instead of jumping.
        Falls back to an instant snap if there's no prior state to tween from."""
        target_poses = object_poses(scene)
        target_joints = np.asarray(robot_joint_angles(scene), dtype=float)
        if self._disp_joints is None or not self._disp_cubes:
            self.sync(scene, skills)
            return
        start_cubes, start_joints = self._disp_cubes, self._disp_joints
        target_cubes = {n: np.asarray(pos, dtype=float) for n, (pos, _) in target_poses.items()}
        orns = {n: orn for n, (_, orn) in target_poses.items()}

        frames = max(1, int(duration * _ANIM_FPS))
        for f in range(1, frames + 1):
            a = _smoothstep(f / frames)
            for name, tgt in target_cubes.items():
                src = start_cubes.get(name, tgt)
                pos = src + (tgt - src) * a
                handle = self.cube_handles[name]
                handle.position = tuple(float(v) for v in pos)
                handle.wxyz = _xyzw_to_wxyz(orns[name])
            self.robot.update_cfg(start_joints + (target_joints - start_joints) * a)
            time.sleep(duration / frames)

        self._disp_cubes, self._disp_joints = target_cubes, target_joints


def _rgb255(rgba) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgba[:3])
