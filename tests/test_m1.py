"""Scene builds, predicates are correct, the hardcoded plan
runs to goal_satisfied, and a non-trivial PNG is produced. All headless."""
import os

import numpy as np
import pybullet as p
import pytest

from agents.config import load_config
from agents.llm_client import LLMClient
from agents.schema import Plan
from env.render import render_rgb
from env.scene import build_scene
from env.skills import Skills
from env.state import goal_satisfied, in_bin, on_top_of, postcondition_true


@pytest.fixture
def scene_and_skills():
    cfg = load_config()
    cfg["sim"]["gui"] = False
    scene = build_scene(cfg)
    skills = Skills(scene, cfg)
    yield scene, skills
    p.disconnect(scene.client)


def test_scene_has_all_bodies(scene_and_skills):
    scene, _ = scene_and_skills
    assert set(scene.cubes) == {"red_cube", "green_cube", "blue_cube"}
    assert set(scene.bins) == {"red_bin", "blue_bin"}
    assert scene.ee_link == 11
    # robot + plane + 3 cubes + 2 trays (5 bodies each) = 15 bodies
    assert p.getNumBodies(physicsClientId=scene.client) == 2 + 3 + 10


def test_put_down_places_a_cube_on_the_open_table(scene_and_skills):
    """put_down releases the held cube onto the table, OUTSIDE every bin and
    not overlapping another cube - the 'take it out / put it on the table' skill."""
    scene, skills = scene_and_skills
    skills.pick("red_cube"); skills.place("red_cube", "red_bin")
    assert in_bin(scene, "red_cube", "red_bin")
    skills.pick("red_cube")
    assert skills.held == "red_cube"
    skills.put_down("red_cube")
    assert skills.held is None
    assert not in_bin(scene, "red_cube", "red_bin")          # no longer in a bin
    assert not any(in_bin(scene, "red_cube", b) for b in scene.bins)   # in NO bin
    x, y, z = p.getBasePositionAndOrientation(scene.cubes["red_cube"])[0]
    assert abs(z - scene.cfg["scene"]["cube_half_extent"]) < 1e-2       # resting on the table
    for other in ("green_cube", "blue_cube"):                # not overlapping another cube
        ox, oy, _ = p.getBasePositionAndOrientation(scene.cubes[other])[0]
        assert ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 >= 2 * scene.cfg["scene"]["cube_half_extent"]


def test_scene_state_reports_stacking(scene_and_skills):
    """scene_state must expose on_top_of / blocked_by so the planner can unstack
    before moving a blocked cube."""
    from env.state import scene_state
    scene, skills = scene_and_skills
    skills.pick("red_cube"); skills.place("red_cube", "red_bin")
    skills.pick("green_cube"); skills.stack("green_cube", "red_cube")
    st = scene_state(scene, skills)["objects"]
    assert st["green_cube"]["on_top_of"] == "red_cube"
    assert st["red_cube"]["blocked_by"] == "green_cube"
    assert st["blue_cube"]["on_top_of"] is None and st["blue_cube"]["blocked_by"] is None


def test_in_bin_predicate_flips(scene_and_skills):
    scene, skills = scene_and_skills
    # a cube at its start pose is in no bin
    assert not in_bin(scene, "red_cube", "red_bin")
    # teleport it to the red bin's resting spot -> in_bin true
    b = scene.bins["red_bin"]
    resting_z = b["base_top"] + scene.cfg["scene"]["cube_half_extent"]
    skills._teleport(scene.cubes["red_cube"], [b["center"][0], b["center"][1], resting_z])
    assert in_bin(scene, "red_cube", "red_bin")
    assert not in_bin(scene, "red_cube", "blue_bin")


def test_on_top_of_predicate(scene_and_skills):
    scene, skills = scene_and_skills
    cube_h = scene.cfg["scene"]["cube_half_extent"]
    skills._teleport(scene.cubes["red_cube"], [0.5, 0.0, cube_h])
    # not stacked initially
    assert not on_top_of(scene, "green_cube", "red_cube")
    # place green exactly one cube-height above red
    skills._teleport(scene.cubes["green_cube"], [0.5, 0.0, cube_h + 2 * cube_h])
    assert on_top_of(scene, "green_cube", "red_cube")


def test_postcondition_unparseable_returns_none(scene_and_skills):
    scene, skills = scene_and_skills
    assert postcondition_true(scene, skills, "some vague thing") is None
    assert postcondition_true(scene, skills, "gripper empty") is True


def test_full_plan_reaches_goal(scene_and_skills):
    scene, skills = scene_and_skills
    plan = LLMClient(scene.cfg).call("planner", "demo", Plan)
    for st in plan.subtasks:
        skills.execute(st)
    assert goal_satisfied(scene, skills, plan) is True


def test_render_is_nontrivial(scene_and_skills, tmp_path):
    scene, _ = scene_and_skills
    rgb = render_rgb(scene)
    assert rgb.shape == (scene.cfg["camera"]["height"], scene.cfg["camera"]["width"], 3)
    assert rgb.dtype == np.uint8
    # real rendered content, not a flat monochrome frame (a blank PNG would
    # have std==0 and compress to a few hundred bytes)
    assert rgb.std() > 5
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 10
    from env.render import save_png
    out = tmp_path / "frame.png"
    save_png(rgb, str(out))
    assert out.stat().st_size > 3_000
