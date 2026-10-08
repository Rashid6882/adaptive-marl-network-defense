"""
Closed-loop evaluation in the action-dependent simulator (src/sim_env.py).

Each system is run over the whole held-out test stream, file by file, and its
own actions change which flows it sees next. Systems compared:
  none       - always Allow / No action (damage if nothing is done)
  rule       - static-threshold baseline
  v1_marl    - the 3 agents trained offline on replayed flows (models/)
  v2_marl    - the 3 agents trained interactively in the simulator (models_v2/)
  v2_single  - single-agent baseline trained in the simulator (models_v2/)

    python -m src.evaluate_sim
"""

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
from src.env import DPORT_IDX, PKTS_IDX
from src.features import CLASSES, N_CLASSES, REPEAT_OFFENDER_THRESHOLD
from src.sim_env import (
    DefenseSim,
    det_obs,
    fw_obs,
    load_split_cached,
    make_fast_predict,
    resp_obs,
    single_obs,
)
from src.single_agent_env import decode_action

V1_DIR = "models"
V2_DIR = "models_v2"
OUTPUTS_DIR = "outputs"


def _load(path):
    return make_fast_predict(PPO.load(path, device="cpu")) if os.path.exists(path + ".zip") else None


def policy_none(sim):
    return 0, 0, 0


def policy_rule(sim):
    d = int(sim.rule_det[sim.pos])
    f = rule_based_firewall(d)
    return d, f, rule_based_response(f, sim.cur_count, REPEAT_OFFENDER_THRESHOLD)


def make_policy_v1():
    det, fw, resp = (_load(f"{V1_DIR}/{n}") for n in ("detection", "firewall", "response"))
    if not all((det, fw, resp)):
        return None

    def policy(sim):
        x = sim.X[sim.pos]
        d = det(x)
        oh = np.zeros(N_CLASSES, dtype=np.float32)
        oh[d] = 1.0
        f = fw(np.concatenate([oh, [x[DPORT_IDX], x[PKTS_IDX]]]).astype(np.float32))
        oh3 = np.zeros(3, dtype=np.float32)
        oh3[f] = 1.0
        r = resp(np.concatenate([oh3, [min(sim.cur_count, 20) / 20.0]]).astype(np.float32))
        return d, f, r

    return policy


def make_policy_v2():
    det, fw, resp = (_load(f"{V2_DIR}/{n}") for n in ("detection", "firewall", "response"))
    if not all((det, fw, resp)):
        return None

    def policy(sim):
        d = det(det_obs(sim))
        f = fw(fw_obs(sim, d))
        return d, f, resp(resp_obs(sim, f))

    return policy


def make_policy_single():
    single = _load(f"{V2_DIR}/single_agent")
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


def main():
    _, test_df = load_split_cached()
    scaler = joblib.load(f"{V1_DIR}/scaler.joblib")
    sim = DefenseSim(test_df, scaler, seed=0)

    systems = {"none": policy_none, "rule": policy_rule}
    for name, make in (("v1_marl", make_policy_v1), ("v2_marl", make_policy_v2), ("v2_single", make_policy_single)):
        pol = make()
        if pol is None:
            print(f"(skipping {name}: models not found)")
        else:
            systems[name] = pol

    results = {}
    for name, pol in systems.items():
        print(f"running {name} ...", flush=True)
        results[name] = run_closed_loop(sim, pol)

    base = results["none"]["damage"]
    print(f"\n{'system':10s} {'damage':>9s} {'disrupt':>9s} {'cost':>7s} {'TOTAL':>9s} {'dmg red.':>9s} "
          f"{'benign FP':>10s} {'latency':>8s} {'uncont.':>8s} {'macroF1':>8s}")
    for name, r in results.items():
        red = 1 - r["damage"] / base if base else 0.0
        lat = f"{r['mean_latency_flows']:.1f}" if r["mean_latency_flows"] is not None else "n/a"
        print(f"{name:10s} {r['damage']:9.0f} {r['disruption']:9.0f} {r['operating_cost']:7.0f} "
              f"{r['total_loss']:9.0f} {red:9.1%} {r['benign_disrupted_rate']:10.2%} {lat:>8s} "
              f"{r['uncontained_attacks']:8d} {r['detection_macro_f1']:8.3f}")
        r["damage_reduction"] = red

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
