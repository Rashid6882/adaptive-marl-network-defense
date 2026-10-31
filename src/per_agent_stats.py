"""
What each trained agent actually does on the test stream (descriptive, over the flows it saw;
attack flows are dropped once an attacker is contained).

    python -m src.per_agent_stats 1        # reads runs/seed1, writes runs/seed1/per_agent_test.json
"""

import collections
import json
import sys

import numpy as np
from stable_baselines3 import PPO

from src.evaluate_sim import get_split, load_scaler
from src.features import CLASSES
from src.sim_env import DefenseSim, det_obs, fw_obs, make_fast_predict, resp_obs

seed = sys.argv[1]
d_dir = f"runs/seed{seed}"
preds = {n: make_fast_predict(PPO.load(f"{d_dir}/{n}", device="cpu")) for n in ("detection", "firewall", "response")}
sim = DefenseSim(get_split("test"), load_scaler(d_dir), seed=0)

det_tab = np.zeros((4, 4), int)  # rows: true class, columns: Detection's verdict
fw_tab = collections.Counter()  # (flow is attack, verdict is attack, firewall action)
resp_tab = collections.Counter()  # (flow is attack, response action)
for start in sim.file_starts:
    sim.reset(start=int(start), max_steps=None)
    while not sim.done:
        d = preds["detection"](det_obs(sim))
        f = preds["firewall"](fw_obs(sim, d))
        r = preds["response"](resp_obs(sim, f))
        t = int(sim.labels[sim.pos])
        det_tab[t][d] += 1
        fw_tab[(t != 0, d != 0, f)] += 1
        resp_tab[(t != 0, r)] += 1
        sim.apply(d, f, r)

out = {"seed": seed}
# Detection: per attack class, share of flows given the right class / given any attack verdict; benign false alarms
out["detection"] = {
    CLASSES[c]: {
        "flows": int(det_tab[c].sum()),
        "exact_class_recall": float(det_tab[c][c] / det_tab[c].sum()),
        "flagged_as_any_attack": float(1 - det_tab[c][0] / det_tab[c].sum()),
        "precision": float(det_tab[c][c] / max(1, det_tab[:, c].sum())),
    }
    for c in range(1, 4)
}
out["detection"]["BENIGN"] = {"flows": int(det_tab[0].sum()), "false_alarm_rate": float(1 - det_tab[0][0] / det_tab[0].sum())}


def share(num, den):
    return float(sum(num) / max(1, sum(den)))


keys = list(fw_tab)
atk = [k for k in keys if k[0]]
ben = [k for k in keys if not k[0]]
out["firewall"] = {
    "block_rate_on_attack_flows": share([fw_tab[k] for k in atk if k[2] == 2], [fw_tab[k] for k in atk]),
    "block_rate_on_benign_flows": share([fw_tab[k] for k in ben if k[2] == 2], [fw_tab[k] for k in ben]),
    "block_rate_when_verdict_is_attack": share([fw_tab[k] for k in keys if k[1] and k[2] == 2], [fw_tab[k] for k in keys if k[1]]),
    "block_rate_when_verdict_benign_but_flow_is_attack": share(
        [fw_tab[k] for k in atk if not k[1] and k[2] == 2], [fw_tab[k] for k in atk if not k[1]]),
}
names = ["No action", "Alert", "Isolate"]
out["response"] = {
    ("attack_flows" if a else "benign_flows"): {
        names[r]: float(resp_tab[(a, r)] / max(1, sum(resp_tab[(a, x)] for x in range(3)))) for r in range(3)
    }
    for a in (True, False)
}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{d_dir}/per_agent_test.json", "w"), indent=1)
