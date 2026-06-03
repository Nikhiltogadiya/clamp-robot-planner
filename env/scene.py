"""Build the PyBullet scene: ground plane + Franka Panda + colored cubes +
shallow-walled trays + a fixed camera view. No motion planning here - this
just constructs the world and holds handles to every body.

Coordinate frame: the Panda base sits at the origin facing +x. The tabletop
workspace is x in [0.3, 0.8], y in [-0.3, 0.3]. All poses come from
config.yaml (scene.cubes / scene.bins), so the layout is data, not code.
"""
from dataclasses import dataclass, field

import pybullet as p
import pybullet_data

# Franka Panda facts (verified against bullet3 panda_sim_grasp.py):
EE_LINK = 11                                   # panda_grasptarget, between the fingertips
ARM_JOINTS = list(range(7))                    # revolute joints 0-6
FINGER_JOINTS = [9, 10]                         # prismatic, range 0-0.04
REST_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
FINGER_OPEN = 0.04


@dataclass
class Scene:
    client: int
    robot: int
    plane: int
    ee_link: int
    cubes: dict[str, int]              # name -> body id
    bins: dict[str, dict] = field(default_factory=dict)  # name -> geom dict (+ "id")
    cfg: dict = field(default_factory=dict)


def connect(cfg: dict) -> int:
    """Connect to a physics server (GUI or DIRECT) and configure it."""
    mode = p.GUI if cfg["sim"].get("gui", False) else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=client)
    hz = cfg["sim"].get("hz", 240)
    if hz != 240:
        p.setTimeStep(1.0 / hz, physicsClientId=client)
    if mode == p.GUI:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    return client


def settle(client: int, steps: int) -> None:
    for _ in range(steps):
        p.stepSimulation(physicsClientId=client)


def _add_cube(client: int, half: float, rgba, pos) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=rgba,
                              physicsClientId=client)
    return p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=pos,
                             physicsClientId=client)


def _add_tray(client: int, spec: dict) -> list[int]:
    """A shallow-walled tray = 1 static base pad + 4 static walls.

    Five plain static bodies (mass 0) are simpler and better-behaved than a
    single multibody with links. Returns all body ids so the scene can track
    them (ids[0] is the base pad). The in-bin check itself uses the config
    `base_top` scalar, not these body ids - the ids are only bookkeeping.
    """
    cx, cy = spec["center"]
    inner = spec["inner_half"]
    thick = spec["wall_thickness"]
    wall_h = spec["wall_height"]
    base_top = spec["base_top"]
    rgba = spec["rgba"]
    ids: list[int] = []

    def box(half_extents, pos):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba,
                                  physicsClientId=client)
        bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis, basePosition=pos,
                                physicsClientId=client)
        ids.append(bid)
        return bid

    outer = inner + thick
    # base pad: top surface at base_top
    box([outer, outer, base_top / 2], [cx, cy, base_top / 2])
    wall_cz = base_top + wall_h / 2
    # +x / -x walls span the full outer width in y
    box([thick / 2, outer, wall_h / 2], [cx + inner + thick / 2, cy, wall_cz])
    box([thick / 2, outer, wall_h / 2], [cx - inner - thick / 2, cy, wall_cz])
    # +y / -y walls span only the inner width in x (corners already covered)
    box([inner, thick / 2, wall_h / 2], [cx, cy + inner + thick / 2, wall_cz])
    box([inner, thick / 2, wall_h / 2], [cx, cy - inner - thick / 2, wall_cz])
    return ids


def _add_ground(client: int, cfg: dict) -> int:
    """Ground the scene sits on. Default is a plain matte floor (a large flat
    box) so the VLM sees an uncluttered background; set scene.plain_floor:false
    to fall back to the textured checkerboard plane.urdf."""
    if not cfg["scene"].get("plain_floor", True):
        return p.loadURDF("plane.urdf", physicsClientId=client)
    rgba = cfg["scene"].get("floor_rgba", [0.82, 0.82, 0.85, 1.0])
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01], physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01], rgbaColor=rgba,
                              physicsClientId=client)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=[0, 0, -0.01],
                             physicsClientId=client)


def build_scene(cfg: dict) -> Scene:
    client = connect(cfg)
    try:
        plane = _add_ground(client, cfg)
        robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True,
                           physicsClientId=client)
        for j, q in zip(ARM_JOINTS, REST_POSE):
            p.resetJointState(robot, j, q, physicsClientId=client)
        for j in FINGER_JOINTS:
            p.resetJointState(robot, j, FINGER_OPEN, physicsClientId=client)

        half = cfg["scene"]["cube_half_extent"]
        cubes: dict[str, int] = {}
        for c in cfg["scene"]["cubes"]:
            cubes[c["name"]] = _add_cube(client, half, c["rgba"], c["start_pos"])

        bins: dict[str, dict] = {}
        for b in cfg["scene"]["bins"]:
            ids = _add_tray(client, b)
            bins[b["name"]] = {**b, "ids": ids, "base_id": ids[0]}

        settle(client, cfg["sim"].get("settle_steps", 240))
        return Scene(client=client, robot=robot, plane=plane, ee_link=EE_LINK,
                     cubes=cubes, bins=bins, cfg=cfg)
    except Exception:
        p.disconnect(client)     # don't leak the physics client if construction fails
        raise


if __name__ == "__main__":
    from agents.config import load_config

    scene = build_scene(load_config())
    print("robot id:", scene.robot)
    print("cubes:", scene.cubes)
    print("bins:", {k: v["base_id"] for k, v in scene.bins.items()})
    ee = p.getLinkState(scene.robot, scene.ee_link)[4]
    print("ee world pos:", tuple(round(x, 3) for x in ee))
    p.disconnect(scene.client)
