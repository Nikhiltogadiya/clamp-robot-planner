"""Runtime observation helpers: full object poses (position + orientation)
and robot joint angles, for mirroring the PyBullet simulation into a separate
3D renderer (the Viser operator console). Nothing here changes sim behavior -
these are read-only queries layered on top of env/scene.py's existing bodies.
"""
import numpy as np
import pybullet as p

from env.scene import ARM_JOINTS, FINGER_JOINTS


def object_poses(scene) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    """name -> (position_xyz, orientation_xyzw) for every cube in the scene."""
    poses = {}
    for name, body_id in scene.cubes.items():
        pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=scene.client)
        poses[name] = (tuple(pos), tuple(orn))
    return poses


def robot_joint_angles(scene) -> np.ndarray:
    """Joint angles in the order Viser's ViserUrdf.get_actuated_joint_names()
    expects for this Panda URDF: 7 arm joints, then ONE finger value (the
    second finger is a mimic joint in the URDF and is driven automatically)."""
    arm = [p.getJointState(scene.robot, j, physicsClientId=scene.client)[0] for j in ARM_JOINTS]
    finger = p.getJointState(scene.robot, FINGER_JOINTS[0], physicsClientId=scene.client)[0]
    return np.array(arm + [finger], dtype=np.float64)
