"""
Closed-loop evaluation in the action-dependent simulator (src/sim_env.py).

Each system is run over a whole held-out stream (test by default, or validation),
file by file, and its own actions change which flows it sees next. Systems:
  none        - always Allow / No action (damage if nothing is done)
  rule        - static-threshold baseline (default hand-set thresholds)
  rule_tuned  - the same rules with thresholds tuned on validation (src/tune_rule.py)
  v2_marl     - the 3 agents trained interactively in the simulator
  v2_single   - single-agent baseline trained in the simulator

    python -m src.evaluate_sim                                   # legacy: models_v2/, test split
    python -m src.evaluate_sim --models-dir runs/seed1 --split val --out runs/seed1/eval_val.json
    python -m src.evaluate_sim --models-dir runs/seed1 --split test --out runs/seed1/eval_test.json \
        --tuned-rule runs/tuned_rule.json
    python -m src.evaluate_sim --models-dir runs/seed1 --sensitivity --out runs/seed1/sensitivity.json
"""

import argparse
import dataclasses
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from stable_baselines3 import PPO

from src.baseline_rule import rule_based_firewall, rule_based_response
from src.env import fit_scaler
from src.features import CLASSES, N_CLASSES, REPEAT_OFFENDER_THRESHOLD
from src.sim_env import (
    DEFAULT_RULE_PARAMS,
    DefenseSim,
    SimConfig,
    det_obs,
    fw_obs,
    load_split_cached,
    load_splits_cached,
    make_fast_predict,
    resp_obs,
    rule_detect_array,
    single_obs,
)
from src.single_agent_env import decode_action

V2_DIR = "models_v2"
OUTPUTS_DIR = "outputs"

SENSITIVITY_VARIANTS = {
    "default": {},
    "ttl_x0.5": dict(ttl_block=100, ttl_throttle=60, ttl_isolate=200, ttl_alert=150, alert_delay=20),
    "ttl_x2": dict(ttl_block=400, ttl_throttle=240, ttl_isolate=800, ttl_alert=600, alert_delay=80),
    "no_escalation": dict(age_cap=0.0),
    "strong_escalation": dict(age_cap=4.0, age_scale=30.0),
    "disruption_x2": dict(disruption_block=2.0, disruption_rl=0.6, disruption_isolate=6.0, outage_cost=1.0),
    "equal_damage": dict(damage_base=(0.0, 1.0, 1.0, 1.0)),
}


def _load(path):
    return make_fast_predict(PPO.load(path, device="cpu")) if os.path.exists(path + ".zip") else None


def policy_none(sim):
    return 0, 0, 0


def make_rule_policy(resp_thr=REPEAT_OFFENDER_THRESHOLD):
    def policy(sim):
        d = int(sim.rule_det[sim.pos])
        f = rule_based_firewall(d)
        return d, f, rule_based_response(f, sim.cur_count, resp_thr)

    return policy


policy_rule = make_rule_policy()


def make_policy_v2(models_dir=V2_DIR):
    det, fw, resp = (_load(f"{models_dir}/{n}") for n in ("detection", "firewall", "response"))
    if not all((det, fw, resp)):
        return None

    def policy(sim):
        d = det(det_obs(sim))
        f = fw(fw_obs(sim, d))
        return d, f, resp(resp_obs(sim, f))

    return policy


def make_policy_single(models_dir=V2_DIR):
    single = _load(f"{models_dir}/single_agent")
    if single is None:
        return None
    return lambda sim: decode_action(single(single_obs(sim)))


def run_closed_loop(sim, policy):
    sim.reset_stats()
    trues, dets = [], []
    for start in sim.file_starts:
        sim.reset(start=int(start), max_steps=None)
        while not sim.done:
            d, f, r = policy(sim)
            true = sim.apply(d, f, r)[4]
            trues.append(true)
            dets.append(d)
        sim.s["uncontained"] += sim.open_onsets()
    s = sim.s
    lat = s["latencies"]
    trues, dets = np.array(trues), np.array(dets)
    f1 = f1_score(trues, dets, labels=range(N_CLASSES), average=None, zero_division=0)
    return {
        "damage": s["damage"],
        "disruption": s["disruption"],
        "operating_cost": s["cost"],
        "total_loss": s["damage"] + s["disruption"] + s["cost"],
        "benign_disrupted_rate": s["benign_disrupted"] / max(1, s["benign_presented"]),
        "containment_events": len(lat),
        "uncontained_attacks": s["uncontained"],
        "mean_latency_flows": float(np.mean(lat)) if lat else None,
        "attack_flows_dropped": s["dropped"],
        "detection_macro_f1": float(f1_score(trues, dets, average="macro", zero_division=0)),
        "detection_f1": dict(zip(CLASSES, [float(x) for x in f1])),
    }


def quick_total_loss(sim, policy):
    """Closed-loop total loss only (no F1 bookkeeping); used for threshold search."""
    sim.reset_stats()
    for start in sim.file_starts:
        sim.reset(start=int(start), max_steps=None)
        while not sim.done:
            sim.apply(*policy(sim))
    s = sim.s
    return s["damage"] + s["disruption"] + s["cost"]


def get_split(split):
    if split == "val":
        return load_splits_cached()[1]
    return load_split_cached()[1]


def load_scaler(models_dir):
    path = f"{models_dir}/scaler.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    return fit_scaler(load_splits_cached()[0])


def evaluate(df, scaler, models_dir, cfg=None, tuned_rule=None, label=""):
    sim = DefenseSim(df, scaler, cfg=cfg, seed=0)
    default_det = sim.rule_det
    systems = {"none": (policy_none, None), "rule": (policy_rule, None)}
    if tuned_rule:
        arr = rule_detect_array(df, tuned_rule)
        systems["rule_tuned"] = (make_rule_policy(tuned_rule.get("resp_thr", REPEAT_OFFENDER_THRESHOLD)), arr)
    for name, make in (("v2_marl", make_policy_v2), ("v2_single", make_policy_single)):
        pol = make(models_dir)
        if pol is None:
            print(f"(skipping {name}: models not found in {models_dir})")
        else:
            systems[name] = (pol, None)

    results = {}
    for name, (pol, det_arr) in systems.items():
        print(f"running {name} {label}...", flush=True)
        sim.rule_det = det_arr if det_arr is not None else default_det
        results[name] = run_closed_loop(sim, pol)
    sim.rule_det = default_det
    base = results["none"]["damage"]
    for r in results.values():
        r["damage_reduction"] = 1 - r["damage"] / base if base else 0.0
    return results


def print_table(results):
    print(f"\n{'system':11s} {'damage':>9s} {'disrupt':>9s} {'cost':>7s} {'TOTAL':>9s} {'dmg red.':>9s} "
          f"{'benign FP':>10s} {'latency':>8s} {'uncont.':>8s} {'macroF1':>8s}")
    for name, r in results.items():
        lat = f"{r['mean_latency_flows']:.1f}" if r["mean_latency_flows"] is not None else "n/a"
        print(f"{name:11s} {r['damage']:9.0f} {r['disruption']:9.0f} {r['operating_cost']:7.0f} "
              f"{r['total_loss']:9.0f} {r['damage_reduction']:9.1%} {r['benign_disrupted_rate']:10.2%} {lat:>8s} "
              f"{r['uncontained_attacks']:8d} {r['detection_macro_f1']:8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=V2_DIR)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--out", default=None, help="write results JSON here (default: legacy outputs/ files)")
    ap.add_argument("--tuned-rule", default=None, help="JSON with tuned rule thresholds (from src.tune_rule)")
    ap.add_argument("--sensitivity", action="store_true", help="re-evaluate under varied simulator settings")
    ap.add_argument("--variants", default=None, help="comma-separated subset of sensitivity variants to run")
    args = ap.parse_args()

    df = get_split(args.split)
    scaler = load_scaler(args.models_dir)
    tuned = None
    if args.tuned_rule:
        tuned = json.load(open(args.tuned_rule))["params"]

    if args.sensitivity:
        out = {}
        chosen = args.variants.split(",") if args.variants else list(SENSITIVITY_VARIANTS)
        for vname in chosen:
            overrides = SENSITIVITY_VARIANTS[vname]
            cfg = dataclasses.replace(SimConfig(), **overrides)
            out[vname] = evaluate(df, scaler, args.models_dir, cfg=cfg, tuned_rule=tuned, label=f"[{vname}] ")
            print(f"\n=== variant: {vname} ===")
            print_table(out[vname])
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(out, open(args.out, "w"), indent=2)
            print("saved", args.out)
        return

    results = evaluate(df, scaler, args.models_dir, tuned_rule=tuned)
    print_table(results)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print("saved", args.out)
        return

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(f"{OUTPUTS_DIR}/eval_sim_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    names = list(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(names))
    for key, label in (("damage", "attack damage"), ("disruption", "benign disruption"), ("operating_cost", "operating cost")):
        vals = np.array([results[n][key] for n in names])
        ax.bar(names, vals, bottom=bottom, label=label)
        bottom += vals
    ax.set_ylabel("Total loss on held-out test stream (lower is better)")
    ax.set_title("Closed-loop simulator: total loss by system")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUTS_DIR}/eval_sim_loss.png", dpi=150)
    print(f"Saved {OUTPUTS_DIR}/eval_sim_summary.json and eval_sim_loss.png")


if __name__ == "__main__":
    main()
