"""RGB capture from the fixed camera. Uses the software TinyRenderer so it
works headless (p.DIRECT, no display/GL needed)."""
import imageio.v3 as iio
import numpy as np
import pybullet as p


def render_rgb(scene) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB frame from the config's camera."""
    cam = scene.cfg["camera"]
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam["target"],
        distance=cam["distance"],
        yaw=cam["yaw"],
        pitch=cam["pitch"],
        roll=0,
        upAxisIndex=2,
        physicsClientId=scene.client,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=cam["fov"],
        aspect=cam["width"] / cam["height"],
        nearVal=cam["near"],
        farVal=cam["far"],
        physicsClientId=scene.client,
    )
    w, h, px, _, _ = p.getCameraImage(
        cam["width"], cam["height"],
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_NO_SEGMENTATION_MASK,
        shadow=0,
        lightDirection=[0.4, -0.4, 1.0],
        physicsClientId=scene.client,
    )
    # np.asarray handles both the ndarray return (numpy-enabled builds) and
    # the flat-tuple return (numpy-disabled builds).
    rgba = np.asarray(px, dtype=np.uint8).reshape(h, w, 4)
    return rgba[:, :, :3].copy()


def save_png(rgb: np.ndarray, path: str) -> None:
    iio.imwrite(path, rgb)


if __name__ == "__main__":
    from agents.config import load_config
    from env.scene import build_scene

    scene = build_scene(load_config())
    save_png(render_rgb(scene), "results/scene_check.png")
    print("wrote results/scene_check.png")
    p.disconnect(scene.client)
