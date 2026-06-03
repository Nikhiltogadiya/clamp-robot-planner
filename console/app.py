"""Operator console: a browser-based control room for CLAMP.

    python -m console.app [--port 8080]

Opens a Viser server (prints the local URL to open in a browser). Type any
instruction, watch the Panda arm execute it live in 3D, sabotage the scene
mid-task with a click, and watch the real VLM verifier / recovery LLM's own
reasoning stream into the event feed. Live LLM+VLM brain by default; falls
back to MOCK (+ a visible warning) if OPENROUTER_API_KEY isn't set.

All PyBullet calls happen on this one process's main loop (`run_forever`);
Viser GUI callbacks (run on Viser's own server thread) only ever enqueue a
command - never touch PyBullet directly. See docs/USER_GUIDE.md.
"""
import argparse
import copy
import os
import queue

import pybullet as p
import viser

from agents.config import load_config
from agents.llm_client import LLMClient, UsageMeter
from agents.planner import Planner
from agents.recovery import MockRecovery, Recovery
from agents.verifier import OracleVerifier, Verifier
from console.driver import apply_place_on_top, apply_sabotage, run_episode
from console.scene3d import Console3D
from env.render import render_rgb, save_png
from env.scene import build_scene
from env.skills import Skills

MAX_FEED_LINES = 200
# Where the exact frames sent to the VLM verifier are saved, so a user can open
# them and see precisely what the model was judging (answers "what does the LLM
# actually see?"). Cleared at the start of each run.
_FRAME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "console_frames")


def _build_cfg() -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["sim"]["gui"] = False      # PyBullet stays headless; Viser renders the scene
    cfg["sim"]["use_ik"] = True    # console-only: the arm visibly reaches for objects
    return cfg


def _live_key_available(cfg: dict) -> bool:
    return bool(os.environ.get(cfg["llm"]["api_key_env"]))


class ConsoleApp:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.base_cfg = _build_cfg()
        self.recovery_enabled = True
        self.live_brain = _live_key_available(self.base_cfg)
        self.usage_meter = UsageMeter()
        self.command_queue: queue.Queue = queue.Queue()
        self.feed_lines: list[str] = []
        os.makedirs(_FRAME_DIR, exist_ok=True)
        self._frame_seq = 0

        self.scene = build_scene(self.base_cfg)
        self.skills = Skills(self.scene, self.base_cfg)
        self.console3d = Console3D(self.server, self.scene, self.skills)

        self._build_gui()
        if not self.live_brain:
            key_name = self.base_cfg["llm"]["api_key_env"]
            self._log(f"**No {key_name} found - starting in MOCK mode.** "
                     f"Export it and toggle 'Live brain' to plan any instruction "
                     f"with real models.")
        self._log(f"Ready. Brain: **{'LIVE' if self.live_brain else 'MOCK'}**.")

    # --- gui construction ----------------------------------------------------
    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.add_markdown("# CLAMP Operator Console")
        self.status_md = gui.add_markdown("_idle_")

        with gui.add_folder("Task"):
            self.instruction = gui.add_text(
                "Instruction",
                initial_value="Put the red cube in the red bin, put the blue cube in "
                              "the blue bin, then stack the green cube on the red cube.")
            gui.add_markdown("_MOCK brain only understands 3 canned phrasings - turn "
                             "on Live brain below to plan ANY instruction._")
            self.run_btn = gui.add_button("Run")
            self.run_btn.on_click(lambda _: self._enqueue(
                {"type": "run_task", "instruction": self.instruction.value}))

        with gui.add_folder("Sabotage (works best mid-task)"):
            self.sabotage_obj = gui.add_dropdown("Object", options=tuple(self.scene.cubes))
            self.sabotage_dir = gui.add_dropdown(
                "Direction", options=("left", "right", "forward", "backward"),
                initial_value="right")
            gui.add_markdown("_Direction is used by Knock off / Topple (left/right = "
                             "sideways, forward/backward = away from / toward the arm). "
                             "Topple only does something if the cube is on top of another one._")
            for mode, label in (("knock_off", "Knock off"),
                               ("steal", "Steal from gripper"),
                               ("topple", "Topple")):
                btn = gui.add_button(label)
                btn.on_click(lambda _, m=mode: self._enqueue(
                    {"type": "sabotage", "mode": m, "object": self.sabotage_obj.value,
                     "direction": self.sabotage_dir.value}))

        with gui.add_folder("Manual placement (set up a scene before Run)"):
            self.place_obj = gui.add_dropdown("Object", options=tuple(self.scene.cubes))
            self.place_on = gui.add_dropdown("Place on top of", options=tuple(self.scene.cubes))
            gui.add_markdown("_Instantly places Object on top of the other cube - no LLM call. "
                             "Works for any pair; use this to build a custom scenario, then type "
                             "an instruction and click Run - the planner sees this exact state._")
            place_btn = gui.add_button("Place on top")
            place_btn.on_click(lambda _: self._enqueue(
                {"type": "place_on_top", "object": self.place_obj.value, "on": self.place_on.value}))

        with gui.add_folder("Controls"):
            recovery_cb = gui.add_checkbox("Recovery ON", initial_value=True)
            recovery_cb.on_update(lambda _: self._enqueue(
                {"type": "set_recovery", "value": recovery_cb.value}))
            brain_cb = gui.add_checkbox("Live brain (real LLM+VLM)",
                                       initial_value=self.live_brain)
            brain_cb.on_update(lambda _: self._enqueue(
                {"type": "set_brain", "value": brain_cb.value}))
            reset_btn = gui.add_button("Reset scene")
            reset_btn.on_click(lambda _: self._enqueue({"type": "reset"}))

        with gui.add_folder("Plan"):
            self.plan_md = gui.add_markdown("_no plan yet_")
        with gui.add_folder("Event feed"):
            self.feed_md = gui.add_markdown("")
        with gui.add_folder("Usage"):
            self.usage_md = gui.add_markdown(self.usage_meter.summary())

    def _enqueue(self, cmd: dict) -> None:
        self.command_queue.put(cmd)

    def _log(self, line: str) -> None:
        self.feed_lines.append(line)
        del self.feed_lines[:-MAX_FEED_LINES]
        self.feed_md.content = "\n\n".join(f"- {l}" for l in reversed(self.feed_lines))

    def _set_status(self, text: str) -> None:
        self.status_md.content = f"**{text}**"

    def _render_plan(self, subtasks: list[dict], current_index: int | None,
                     satisfied_by_index: dict[int, bool]) -> None:
        lines = []
        for idx, st in enumerate(subtasks):
            tgt = f" -> {st['target']}" if st["target"] else ""
            if idx in satisfied_by_index:
                mark = "OK" if satisfied_by_index[idx] else "FAILED"
            elif idx == current_index:
                mark = "running"
            else:
                mark = "pending"
            lines.append(f"{idx + 1}. `{st['action']} {st['object']}{tgt}` - {mark}")
        self.plan_md.content = "\n".join(lines) if lines else "_no plan yet_"

    # --- driver event -> GUI bridge -------------------------------------------
    def _make_emit(self):
        plan_state = {"subtasks": [], "satisfied": {}}

        def emit(event: dict) -> None:
            etype = event["type"]
            if etype == "planning":
                self._set_status(f"🤔 LLM is planning… ({event['instruction'][:50]})")
                self._log(f"**Planning:** {event['instruction']}")
            elif etype == "plan_ready":
                plan_state["subtasks"] = event["subtasks"]
                plan_state["satisfied"] = {}
                self._render_plan(plan_state["subtasks"], None, {})
                self._log(f"Plan ready: {len(event['subtasks'])} subtasks")
            elif etype == "plan_update":                       # recovery replanned the tail
                plan_state["subtasks"] = event["subtasks"]
                self._render_plan(plan_state["subtasks"], None, plan_state["satisfied"])
                self._log(f"_plan updated by recovery -> {len(event['subtasks'])} subtasks_")
            elif etype == "step_start":
                st = event["subtask"]
                self._set_status(f"step {event['index'] + 1}: {st['action']} {st['object']}")
                self._render_plan(plan_state["subtasks"], event["index"], plan_state["satisfied"])
            elif etype == "executed":
                self.console3d.animate_to(self.scene, self.skills)   # watchable glide, not a snap
                self._set_status(f"👁 verifying step {event['index'] + 1}…")
            elif etype == "step_verified":
                plan_state["satisfied"][event["index"]] = event["satisfied"]
                self._render_plan(plan_state["subtasks"], event["index"], plan_state["satisfied"])
                st = event["subtask"]
                tag = "OK" if event["satisfied"] else "NOT SATISFIED"
                reason = next(iter(event["reasons"].values()), "")
                line = (f"`{st['action']} {st['object']}` -> **{tag}**"
                        + (f" - _{reason}_" if reason else ""))
                if event["reasons"]:                       # a visual VLM check happened here
                    path = self._save_vlm_frame(st)        # save exactly what it saw
                    if path:
                        line += f"  📷 `{path}`"
                self._log(line)
            elif etype == "recover":
                self._log(f"**Recovery [{event['phase']}]:** {event['strategy']} - "
                         f"_{event['rationale']}_")
            elif etype == "sabotage":
                self.console3d.animate_to(self.scene, self.skills, duration=0.5)
                self._log(f"**SABOTAGE:** {event['description']}")
            elif etype == "place_on_top":
                self.console3d.animate_to(self.scene, self.skills, duration=0.6)
                self._log(f"**PLACE:** {event['description']}")
            elif etype == "error":
                self._log(f"**ERROR ({event['phase']}):** {event['message']}")
            elif etype == "done":
                self.console3d.sync(self.scene, self.skills)
                tag = "SUCCESS" if event["success"] else "FAILED"
                self._log(f"**Result: {tag}** - {event['steps']} steps, "
                         f"{event['n_recover']} recoveries")
                self._set_status(f"idle - last run: {tag}")
                self.usage_md.content = self.usage_meter.summary()

        return emit

    def _poll_live_commands(self) -> list[dict]:
        """Non-blocking drain of sabotage/toggle commands only, called between
        steps by run_episode. Any queued run_task/reset is put back for the
        idle loop to handle once the current episode finishes."""
        keep, apply_now = [], []
        while True:
            try:
                cmd = self.command_queue.get_nowait()
            except queue.Empty:
                break
            if cmd["type"] in ("sabotage", "place_on_top"):
                apply_now.append(cmd)
            elif cmd["type"] == "set_recovery":
                self.recovery_enabled = cmd["value"]         # read live by the running episode
                self._log(f"Recovery set to {'ON' if cmd['value'] else 'OFF'} (applied live)")
            elif cmd["type"] == "set_brain":
                self.live_brain = cmd["value"]               # agents are built per-run
                self._log(f"Brain -> {'LIVE' if cmd['value'] else 'MOCK'} "
                          f"(takes effect on the next Run)")
            else:
                keep.append(cmd)
        for cmd in keep:
            self.command_queue.put(cmd)
        return apply_now

    def _save_vlm_frame(self, subtask: dict) -> str | None:
        """Render + save the exact frame the VLM verifier just judged (the sim
        state is unchanged between verify and this call), returning a repo-
        relative path for the feed. Best-effort - never breaks the run."""
        try:
            self._frame_seq += 1
            fname = f"{self._frame_seq:03d}_{subtask['action']}_{subtask['object']}.png"
            path = os.path.join(_FRAME_DIR, fname)
            save_png(render_rgb(self.scene), path)
            return os.path.relpath(path)
        except Exception:
            return None

    def _rebuild_agents(self):
        cfg = copy.deepcopy(self.base_cfg)
        cfg["llm"] = {**cfg["llm"], "mock": not self.live_brain}
        client = LLMClient(cfg, usage_meter=self.usage_meter)
        planner = Planner(cfg, client=client)
        if self.live_brain:
            verifier, recovery = Verifier(cfg, client=client), Recovery(cfg, client=client)
        else:
            verifier, recovery = OracleVerifier(cfg), MockRecovery(cfg)
        return planner, verifier, recovery

    # --- command handlers ------------------------------------------------------
    def _handle_run_task(self, instruction: str) -> None:
        if not instruction.strip():
            self._log("(empty instruction, ignored)")
            return
        self._frame_seq = 0     # per-run frame numbering restarts at 001 (overwrites by name;
                                # the folder isn't wiped, so committed example frames persist)
        if self.live_brain:
            self._log(f"📷 saving the VLM's camera frames to `{os.path.relpath(_FRAME_DIR)}/` "
                     f"- open them to see exactly what the verifier judges.")

        planner, verifier, recovery = self._rebuild_agents()
        run_episode(instruction, self.scene, self.skills, planner, verifier, recovery,
                   recovery_enabled=lambda: self.recovery_enabled,   # live: mid-run toggle applies
                   emit=self._make_emit(), poll_commands=self._poll_live_commands)

    def _handle_reset(self) -> None:
        p.disconnect(self.scene.client)
        self.scene = build_scene(self.base_cfg)
        self.skills = Skills(self.scene, self.base_cfg)
        self.console3d.sync(self.scene, self.skills)
        self.plan_md.content = "_no plan yet_"
        self._log("**Scene reset.**")
        self._set_status("idle")

    # --- main loop -------------------------------------------------------------
    def run_forever(self) -> None:
        while True:
            try:
                cmd = self.command_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            # Idle handlers touch PyBullet directly; unlike run_episode (which
            # catches its own errors) an exception here would escape run_forever
            # and tear down the whole console. Guard it: log and keep serving.
            try:
                self._dispatch_idle(cmd)
            except Exception as e:
                self._log(f"**ERROR (idle):** {type(e).__name__}: {e}")
                self._set_status("idle - recovered from an error")

    def _dispatch_idle(self, cmd: dict) -> None:
        if cmd["type"] == "run_task":
            self._handle_run_task(cmd["instruction"])
        elif cmd["type"] == "reset":
            self._handle_reset()
        elif cmd["type"] == "sabotage":
            desc = apply_sabotage(self.scene, self.skills, cmd["mode"], cmd["object"],
                                  cmd.get("direction", "right"))
            self.console3d.animate_to(self.scene, self.skills, duration=0.5)
            self._log(f"**SABOTAGE (idle):** {desc}")
        elif cmd["type"] == "place_on_top":
            desc = apply_place_on_top(self.scene, self.skills, cmd["object"], cmd["on"])
            self.console3d.animate_to(self.scene, self.skills, duration=0.6)
            self._log(f"**PLACE (idle):** {desc}")
        elif cmd["type"] == "set_recovery":
            self.recovery_enabled = cmd["value"]
            self._log(f"Recovery set to {'ON' if cmd['value'] else 'OFF'}")
        elif cmd["type"] == "set_brain":
            self.live_brain = cmd["value"]
            self._log(f"Brain set to {'LIVE' if cmd['value'] else 'MOCK'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    app = ConsoleApp(server)
    print(f"\nCLAMP operator console: http://localhost:{args.port}\n")
    try:
        app.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect(app.scene.client)
        server.stop()


if __name__ == "__main__":
    main()
