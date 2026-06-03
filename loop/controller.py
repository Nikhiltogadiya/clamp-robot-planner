"""Control loop.

- run_open_loop: plan -> execute every subtask -> ground-truth success.
  No verification, no recovery. This is the ablation's recovery-OFF baseline.
- run: the full closed loop - precondition gate -> execute
  (a failure_injector may perturb) -> render+verify -> on mismatch, recover
  (retry, or replan the tail). The `recovery_enabled` flag toggles the loop
  between the closed (ON) and open (OFF) behaviours for the headline ablation.
"""
import pybullet as p

from agents.config import load_config
from agents.planner import Planner
from env.scene import build_scene
from env.skills import Skills
from env.state import goal_satisfied, postcondition_true


# Give up replanning a given step after this many attempts. Retries are already
# capped (>2); without this, a recovery that keeps returning "replan" is bounded
# only by max_steps, so a misbehaving agent could burn the whole step budget
# churning replans at one position instead of giving up cleanly.
_MAX_REPLANS_PER_STEP = 3
# ...and a GLOBAL cap across the whole episode: each replan grows the plan and
# shifts the failing step to a NEW index, so the per-step cap alone never fires
# on a persistent verifier disagreement (e.g. a scene the VLM can't perceive,
# like a cube stacked inside a same-coloured tray). After this many total
# replans, stop and let the ground-truth check decide the outcome.
_MAX_TOTAL_REPLANS = 4


def preconditions_met(scene, skills, subtask) -> bool:
    """Cheap gate: are all machine-checkable preconditions currently true?"""
    return all(postcondition_true(scene, skills, c) is not False
               for c in subtask.preconditions)


def run(instruction, scene, skills, planner, verifier, recovery, *,
        plan=None, recovery_enabled=True, failure_injector=None, max_steps=40,
        on_step=None) -> dict:
    """Full closed loop. Returns success (plan-based), events, steps,
    and LLM-call counts. Pass a pre-made `plan` to reuse one plan across the
    ablation arms (it is deep-copied, so the caller's plan is not mutated).

    on_step(index, subtask, satisfied) is called after each executed subtask is
    verified - used by the ROS 2 bridge to stream per-step feedback + a camera
    frame. Optional; does not affect control flow."""
    import copy as _copy
    if plan is None:
        plan = planner.plan(instruction, scene, skills)
    else:
        plan = _copy.deepcopy(plan)             # run() mutates plan on replan
    i, steps = 0, 0
    n_verify = n_recover = 0
    retries: dict[int, int] = {}
    replans: dict[int, int] = {}
    total_replans = 0
    events: list[dict] = []

    while i < len(plan.subtasks) and steps < max_steps:
        st = plan.subtasks[i]
        steps += 1

        # cheap precondition gate: only recovery acts on a violation
        if not preconditions_met(scene, skills, st):
            if recovery_enabled and recovery is not None and total_replans < _MAX_TOTAL_REPLANS:
                dec = recovery.decide(plan.goal, plan, i, {}, scene, skills)
                n_recover += 1
                events.append({"step": i, "phase": "precondition", "strategy": dec.strategy})
                if (dec.strategy == "replan" and dec.new_tail
                        and replans.get(i, 0) < _MAX_REPLANS_PER_STEP
                        and total_replans < _MAX_TOTAL_REPLANS):
                    replans[i] = replans.get(i, 0) + 1
                    total_replans += 1
                    plan.subtasks[i:] = dec.new_tail
                    continue
            # open-loop / no fix available (or replan budget spent): barrel on

        skills.execute(st)
        if failure_injector is not None:
            mode = failure_injector.maybe_perturb(st.action, st.object, st.target, scene, skills)
            if mode:
                events.append({"step": i, "phase": "injected", "mode": mode})

        verdicts = verifier.verify_subtask(scene, st, skills)
        if verdicts:                     # empty => nothing visual to check (e.g. a pick);
            n_verify += 1                # don't count a no-op as a verification
        satisfied = all(v.satisfied for v in verdicts.values()) if verdicts else True

        if on_step is not None:
            on_step(i, st, satisfied)

        if satisfied:
            i += 1
            continue
        if not recovery_enabled or recovery is None or total_replans >= _MAX_TOTAL_REPLANS:
            i += 1                       # recovery-OFF, or global replan budget spent: barrel on
            continue

        dec = recovery.decide(plan.goal, plan, i, verdicts, scene, skills)
        n_recover += 1
        retries[i] = retries.get(i, 0) + 1
        events.append({"step": i, "phase": "recover", "strategy": dec.strategy,
                       "attempt": retries[i]})
        if (dec.strategy == "replan" and dec.new_tail
                and replans.get(i, 0) < _MAX_REPLANS_PER_STEP
                and total_replans < _MAX_TOTAL_REPLANS):
            replans[i] = replans.get(i, 0) + 1
            total_replans += 1
            plan.subtasks[i:] = dec.new_tail        # i stays; new step at i runs next
            retries.pop(i, None)
        elif (retries[i] > 2 or replans.get(i, 0) >= _MAX_REPLANS_PER_STEP
              or total_replans >= _MAX_TOTAL_REPLANS):
            i += 1                                   # give up: retries/replans (this step or total) spent
        # else retry: i unchanged, loop re-executes subtask i

    return {"instruction": instruction, "success": goal_satisfied(scene, skills, plan),
            "steps": steps, "events": events, "plan": plan,
            "n_verify": n_verify, "n_recover": n_recover}


def run_open_loop(instruction: str, scene, skills, planner: Planner,
                  *, max_steps: int = 40) -> dict:
    """Plan the instruction, execute it straight through, report success."""
    plan = planner.plan(instruction, scene, skills)
    events: list[dict] = []
    for st in plan.subtasks[:max_steps]:
        unmet = [c for c in st.preconditions
                 if postcondition_true(scene, skills, c) is False]
        if unmet:
            events.append({"step": st.id, "unmet_preconditions": unmet})
        skills.execute(st)
    return {
        "instruction": instruction,
        "success": goal_satisfied(scene, skills, plan),
        "steps": len(plan.subtasks),
        "events": events,
        "plan": plan,
    }


SAMPLE_INSTRUCTIONS = [
    "Put the red cube in the red bin and put the blue cube in the blue bin.",
    "Put the blue cube in the blue bin, then stack the green cube on the blue cube.",
    "Put the red cube in the red bin, put the blue cube in the blue bin, "
    "then stack the green cube on the red cube.",
]


def _run_samples() -> None:
    cfg = load_config()
    planner = Planner(cfg)
    for instr in SAMPLE_INSTRUCTIONS:
        scene = build_scene(cfg)          # fresh world per instruction
        skills = Skills(scene, cfg)
        result = run_open_loop(instr, scene, skills, planner)
        print(f"\ninstruction: {instr}")
        print(f"  subtasks: {result['steps']}  success: {result['success']}"
              f"  unmet-precondition events: {len(result['events'])}")
        for st in result["plan"].subtasks:
            tgt = f"->{st.target}" if st.target else ""
            print(f"    [{st.id}] {st.action} {st.object}{tgt}")
        p.disconnect(scene.client)


def _run_recovery_ablation(real: bool = False) -> None:
    """Scripted failure (missed placement on step 2) run with recovery ON vs OFF
    on identical worlds - the recovery acceptance: success ON should beat OFF."""
    import copy

    from agents.recovery import MockRecovery, Recovery
    from agents.verifier import OracleVerifier, Verifier
    from env.failure import FailureInjector

    instr = SAMPLE_INSTRUCTIONS[0]     # S1: pick/place red, pick/place blue
    base = copy.deepcopy(load_config())
    # deterministic one-shot failure: miss the first placement (step index 1)
    base["failure"] = {**base["failure"], "p_fail": 0.0,
                       "scripted": [None, "missed_placement"] + [None] * 10}
    if real:
        base["llm"] = {**base["llm"], "mock": False}   # actually hit the live models

    verifier = Verifier(base) if real else OracleVerifier(base)
    recovery_agent = Recovery(base) if real else MockRecovery(base)
    # Planner stays MOCKED even in the real ablation, so the scripted failure
    # always lands on the known step and the demo is deterministic; only the
    # verifier + recovery are live.
    planner_cfg = copy.deepcopy(base)
    planner_cfg["llm"] = {**base["llm"], "mock": True}
    planner = Planner(planner_cfg)
    print(f"recovery ablation ({'REAL VLM+LLM' if real else 'oracle+mock'}), "
          f"scripted missed_placement on step 2:\n  instruction: {instr}")

    outcomes: dict[bool, bool] = {}
    for enabled in (False, True):
        cfg = copy.deepcopy(base)
        scene = build_scene(cfg)
        skills = Skills(scene, cfg)
        injector = FailureInjector(cfg)
        result = run(instr, scene, skills, planner, verifier,
                     recovery_agent, recovery_enabled=enabled, failure_injector=injector)
        recs = [e for e in result["events"] if e["phase"] == "recover"]
        outcomes[enabled] = result["success"]
        print(f"  recovery {'ON ' if enabled else 'OFF'}: success={result['success']}"
              f"  steps={result['steps']}  recover-actions={len(recs)}"
              f"  events={[e.get('mode') or e.get('strategy') for e in result['events']]}")
        p.disconnect(scene.client)

    # The whole point of the ablation is ON > OFF. Assert it so this doubles as a
    # real CI smoke check (the offline oracle+mock path is deterministic; the
    # live path can be noisy, so only the offline path is treated as a gate).
    if not real:
        if outcomes[False] or not outcomes[True]:
            print(f"\nABLATION CHECK FAILED: expected OFF=False, ON=True; "
                  f"got OFF={outcomes[False]}, ON={outcomes[True]}")
            raise SystemExit(1)
        print("\nablation check OK: recovery ON succeeds where OFF fails.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="samples",
                    choices=["samples", "ablation", "ablation-real"])
    args = ap.parse_args()
    if args.mode == "samples":
        _run_samples()
    else:
        _run_recovery_ablation(real=(args.mode == "ablation-real"))
