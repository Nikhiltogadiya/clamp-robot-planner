# CLAMP Console User Guide

The whole project is a **robot-arm simulation you drive from a browser**: you type an instruction,
an LLM plans it, a simulated Franka arm executes it in live 3D, a vision model checks each step from
a camera image, and if something fails a recovery agent re-plans and fixes it. You can also sabotage
the scene mid-task to trigger that recovery yourself.

---

## 1. Setup (one time)

```bash
uv venv -p 3.11 .venv
uv pip install -r requirements.txt
```

Python 3.11 because PyBullet ships a prebuilt cp311 wheel (no compiler needed). For the live LLM+VLM
brain, export an OpenRouter key (a session costs a few cents; the key is never stored in the repo):

```bash
export OPENROUTER_API_KEY=sk-or-...
```

No key? The console still runs in **MOCK mode** (offline, instant, free): fully interactive, just
limited to a few pre-written instructions and canned verdicts.

---

## 2. Run the console

```bash
.venv/bin/python -m console.app
```

It prints a URL (`http://localhost:8080`); open it in your browser. This is a **live control room**,
not a recording:

- A **3D view** of the actual simulation (the Panda arm, cubes, trays) that updates live, with each
  step animated so you can follow the motion.
- A text box for **typing any instruction in your own words** ("stack the blue cube on the green
  one", "put the red and green cubes on the table, then put blue in the red bin"), then click
  **Run**. With a live key, a real LLM plans it.
- A **plan panel** that lights up step by step: pending → running → OK / FAILED.
- **Sabotage buttons.** Pick an **Object** and a **Direction** (left / right / forward / backward,
  sideways only; the tabletop is flat), then click one *while the arm is mid-task* to break it:
  - **Knock off:** always works; pushes the cube a step in the chosen direction.
  - **Steal from gripper:** only if the cube is currently held (otherwise: "nothing to steal").
    Most dramatic right after you see a `pick` step succeed.
  - **Topple:** only if the cube is currently stacked on another cube; pushes it off the stack.

  If a push would land the cube on top of another one, it's refused ("X is in the way"). Teleporting
  a cube into another would otherwise make the physics engine violently eject them apart, so a small
  safe gap is always kept.
- **Manual placement.** Pick an **Object** and **Place on top of** another cube, then **Place on
  top**: it instantly stacks it (no LLM call), for any pair, including a 3-cube tower. Use this to
  build a custom scenario *before* you Run, so the planner sees the exact state you set up. If a cube
  has another resting on it, moving it is refused ("X is stacked on top of it, move it first") rather
  than stranding the top cube.
- An **event feed** with the real verifier's and recovery agent's own reasoning in plain English,
  e.g. *"the robot is not holding the red cube... we must re-pick it, place it, then proceed."*
- **Recovery ON/OFF** and **Live/Mock brain** toggles, a **Reset** button, and a running LLM
  call-count and cost meter.

**Try this:** type the default instruction, click Run, and the instant you see `pick red_cube -> OK`
in the feed, click **Steal from gripper**. Watch the recovery agent notice, explain itself, and fix
it, live.

---

## 3. What the verifier actually sees (and a known limit)

The VLM verifier judges the **camera render**: the images saved to `results/console_frames/*.png`
each step. Open them to see *exactly* what the model was given; that's different from the clean
browser 3D view.

> **Known perception limit (not a bug).** If you **stack a cube on another cube that's inside a
> same-coloured tray** (e.g. green on the red cube while the red cube is in the red bin), the bottom
> cube is camouflaged (same colour as the tray) and occluded (by the top cube plus the tray walls),
> so the verifier may report *"not on a cube, that's a tray"* even though it's geometrically correct.
> The ground truth is right and the closed loop **recovers to success**: a realistic single-view
> perception limitation, not a code fault. (Stacking on the open table verifies fine, because the
> supporting cube is clearly visible.)

---

## 4. Tests

```bash
.venv/bin/python -m pytest -q        # all offline / mock, no API key needed (~7s)
```

Covers the schema, the scene plus magic-grasp skills plus ground-truth predicates, the planner and
its validation, the closed loop with recovery, and the console driver (sabotage, manual placement,
episode logic).

---

## 5. If something doesn't work

| Symptom | Fix |
|---|---|
| `No module named ...` | Activate / point at the venv: `uv venv -p 3.11 .venv && uv pip install -r requirements.txt` |
| Browser shows nothing at the URL | Give it a couple seconds (it's building the scene and loading the Panda mesh); refresh. Check the terminal for a Python traceback. |
| Sabotage button seems to do nothing | Timing: it visibly triggers recovery only if the target cube is mid-manipulation (just picked, not yet placed). Click right after a `pick` line appears in the feed. |
| Live brain shows `401 User not found` | The OpenRouter key is invalid or expired; check <https://openrouter.ai/keys>. Without a key it falls back to MOCK. |

---

## 6. Code map

```
console/               the operator console:
                         scene3d.py  Viser 3D mirror of the PyBullet scene (+ step animation)
                         driver.py   step-emitting episode loop + sabotage / manual placement
                         app.py      the browser GUI (python -m console.app)
agents/                schema, llm_client (+ mock + usage meter), planner, verifier, recovery
env/                   scene, magic-grasp skills, failure injection, render, ground-truth state
loop/controller.py     the closed loop (plan → execute → verify → recover)
prompts/               planner / verifier / recovery prompt templates
config.yaml            scene, camera, models, and skill parameters
```

Simulation only: grasping is a "magic" rigid attach, not real finger physics; one arm; a small fixed
set of cubes and trays. The contribution is the **agentic loop** (plan → verify → recover), which you
drive live from the console.
