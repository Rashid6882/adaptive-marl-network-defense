"""
Experiment 2: when the 3-agent system causes harm, which agent is responsible, and why?

For every harmful step (an attack let through, or a benign flow disrupted) we fix ONE agent at a time
with the correct decision and check whether the harm disappears. Then SHAP shows what the Firewall's
decision depended on.

    python -m src.explain_agents --models-dir runs/seed1 --out runs/seed1/explain.json
"""

import argparse
import json
from collections import Counter

import numpy as np
import shap
import torch
from stable_baselines3 import PPO

from src.evaluate_sim import get_split, load_scaler
from src.sim_env import DefenseSim, det_obs, fw_obs, make_fast_predict, resp_obs

ap = argparse.ArgumentParser()
ap.add_argument("--models-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-category", type=int, default=60, help="max cases explained with SHAP per blame category")
args = ap.parse_args()

# the three trained agents
models = {n: PPO.load(f"{args.models_dir}/{n}", device="cpu") for n in ("detection", "firewall", "response")}
det, fw, resp = (make_fast_predict(models[n]) for n in ("detection", "firewall", "response"))
sim = DefenseSim(get_split("val"), load_scaler(args.models_dir), seed=0)


def harm(d, f, r):
    damage, disruption, _ = sim.preview(d, f, r)  # what this joint action would cost right now
    return damage + disruption


# pass 1: run the validation stream and look at every harmful step
cases, background = [], []
for start in sim.file_starts:
    sim.reset(start=int(start), max_steps=None)
    while not sim.done:
        d = det(det_obs(sim))
        f = fw(fw_obs(sim, d))
        r = resp(resp_obs(sim, f))
        if sim.s["steps"] % 500 == 0:
            background.append(fw_obs(sim, d))
        if harm(d, f, r) > 0:
            true = int(sim.labels[sim.pos])
            attack = true != 0
            # fix Detection only: give the Firewall the true class, let Firewall and Response decide again
            f2 = fw(fw_obs(sim, true))
            fixed_detection = harm(true, f2, resp(resp_obs(sim, f2))) == 0
            # fix Firewall only: correct firewall action (block attacks, allow benign), Response decides again
            f3 = 2 if attack else 0
            fixed_firewall = harm(d, f3, resp(resp_obs(sim, f3))) == 0
            # fix Response only: correct response (isolate attacks, do nothing for benign)
            fixed_response = harm(d, f, 2 if attack else 0) == 0
            fixers = [n for n, ok in (("Detection", fixed_detection), ("Firewall", fixed_firewall),
                                      ("Response", fixed_response)) if ok]  # in pipeline order
            cases.append({"fixers": fixers, "obs": fw_obs(sim, d)})
        sim.apply(d, f, r)

print(f"harmful steps: {len(cases)}")
exactly_one = sum(len(c["fixers"]) == 1 for c in cases)  # pre-registered H2a
print(f"[pre-registered H2a] fixable by exactly one agent: {exactly_one} ({exactly_one / len(cases):.1%})")
# exploratory: root cause = the earliest agent in the pipeline whose correction alone removes the harm
for c in cases:
    c["blame"] = c["fixers"][0] if c["fixers"] else "none"
counts = Counter(c["blame"] for c in cases)
print("[exploratory] root cause (earliest agent that could have prevented the harm):")
for label, n in counts.most_common():
    print(f"  {label:10s} {n:6d}  ({n / len(cases):.1%})")

# pass 2: SHAP on the Firewall's probability of choosing Block
def p_block(x):
    with torch.no_grad():
        dist = models["firewall"].policy.get_distribution(torch.as_tensor(x, dtype=torch.float32))
    return dist.distribution.probs[:, 2].numpy()


explainer = shap.KernelExplainer(p_block, np.array(background[:50]))
groups = {"Detection verdict": [0, 1, 2, 3], "repeat count": [4], "packet rate": [5], "context": [6, 7, 8, 9, 10]}
shares = {}
for label in counts:
    chosen = [c["obs"] for c in cases if c["blame"] == label][: args.per_category]
    values = np.abs(explainer.shap_values(np.array(chosen), nsamples=200, silent=True)).sum(axis=0)
    shares[label] = {g: float(values[idx].sum() / values.sum()) for g, idx in groups.items()}
    print(f"SHAP share of the Firewall's decision, root cause = {label} ({len(chosen)} cases):",
          {g: f"{v:.0%}" for g, v in shares[label].items()})

json.dump({"harmful_steps": len(cases), "exactly_one_agent": exactly_one, "root_cause_counts": counts,
           "shap_shares_by_root_cause": shares},
          open(args.out, "w"), indent=2)
