# Module docstring: describes what each trained agent actually does on the test stream (counts over the flows
# it saw; attack flows are dropped once the attacker is contained).
# Example command.
# Ends the docstring.
"""
What each trained agent actually does on the test stream (descriptive, over the flows it saw;
attack flows are dropped once an attacker is contained).

    python -m src.per_agent_stats 1        # reads runs/seed1, writes runs/seed1/per_agent_test.json
"""

# Imports collections (for Counter).
import collections
# Imports json.
import json
# Imports sys to read the command-line argument.
import sys

# Imports numpy.
import numpy as np
# Imports PPO to load saved agents.
from stable_baselines3 import PPO

# Imports the split loader and scaler loader.
from src.evaluate_sim import get_split, load_scaler
# Imports the class names.
from src.features import CLASSES
# Imports the simulator, observation builders and fast-predict wrapper.
from src.sim_env import DefenseSim, det_obs, fw_obs, make_fast_predict, resp_obs

# Reads the seed number from the command line.
seed = sys.argv[1]
# Folder of that seed's models.
d_dir = f"runs/seed{seed}"
# Loads the three models and wraps each in a fast prediction function.
preds = {n: make_fast_predict(PPO.load(f"{d_dir}/{n}", device="cpu")) for n in ("detection", "firewall", "response")}
# Builds a simulator on the test stream.
sim = DefenseSim(get_split("test"), load_scaler(d_dir), seed=0)

# 4 x 4 table of counts: rows = true class, columns = Detection's verdict.
det_tab = np.zeros((4, 4), int)  # rows: true class, columns: Detection's verdict
# Counter of (flow is an attack, verdict is an attack, Firewall action).
fw_tab = collections.Counter()  # (flow is attack, verdict is attack, firewall action)
# Counter of (flow is an attack, Response action).
resp_tab = collections.Counter()  # (flow is attack, response action)
# Loops over each file's start position.
for start in sim.file_starts:
    # Starts a run through the file.
    sim.reset(start=int(start), max_steps=None)
    # Keeps going until the file ends.
    while not sim.done:
        # Detection's decision.
        d = preds["detection"](det_obs(sim))
        # Firewall's decision.
        f = preds["firewall"](fw_obs(sim, d))
        # Response's decision.
        r = preds["response"](resp_obs(sim, f))
        # True class of the flow.
        t = int(sim.labels[sim.pos])
        # Counts this (true class, verdict) pair.
        det_tab[t][d] += 1
        # Counts this (attack?, verdict attack?, Firewall action) combination.
        fw_tab[(t != 0, d != 0, f)] += 1
        # Counts this (attack?, Response action) combination.
        resp_tab[(t != 0, r)] += 1
        # Applies the decision and moves on.
        sim.apply(d, f, r)

# Starts the output dictionary with the seed.
out = {"seed": seed}
# Detection: per attack class, share of flows given the right class / given any attack verdict; benign false alarms
# Starts the Detection section.
out["detection"] = {
    # One entry per attack class.
    CLASSES[c]: {
        # Number of flows of this class that the agent saw.
        "flows": int(det_tab[c].sum()),
        # Exact-class recall: correct verdicts divided by flows of this class.
        "exact_class_recall": float(det_tab[c][c] / det_tab[c].sum()),
        # Share of flows of this class flagged as any attack.
        "flagged_as_any_attack": float(1 - det_tab[c][0] / det_tab[c].sum()),
        # Precision: correct verdicts divided by all flows given this verdict.
        "precision": float(det_tab[c][c] / max(1, det_tab[:, c].sum())),
    # Ends the entry.
    }
    # Loops over the three attack classes (1 to 3).
    for c in range(1, 4)
# Ends the dictionary.
}
# Adds benign: the number of flows and the false-alarm rate (benign flows not called benign).
out["detection"]["BENIGN"] = {"flows": int(det_tab[0].sum()), "false_alarm_rate": float(1 - det_tab[0][0] / det_tab[0].sum())}


# Helper function for a share.
def share(num, den):
    # Returns the sum of the numerators divided by the sum of the denominators (denominator at least 1).
    return float(sum(num) / max(1, sum(den)))


# List of all combinations seen by the Firewall.
keys = list(fw_tab)
# Combinations where the flow is an attack.
atk = [k for k in keys if k[0]]
# Combinations where the flow is benign.
ben = [k for k in keys if not k[0]]
# Starts the Firewall section.
out["firewall"] = {
    # Block rate on attack flows.
    "block_rate_on_attack_flows": share([fw_tab[k] for k in atk if k[2] == 2], [fw_tab[k] for k in atk]),
    # Block rate on benign flows.
    "block_rate_on_benign_flows": share([fw_tab[k] for k in ben if k[2] == 2], [fw_tab[k] for k in ben]),
    # Block rate when Detection's verdict was an attack.
    "block_rate_when_verdict_is_attack": share([fw_tab[k] for k in keys if k[1] and k[2] == 2], [fw_tab[k] for k in keys if k[1]]),
    # Block rate on attack flows that Detection called benign.
    "block_rate_when_verdict_benign_but_flow_is_attack": share(
        [fw_tab[k] for k in atk if not k[1] and k[2] == 2], [fw_tab[k] for k in atk if not k[1]]),
# Ends the Firewall section.
}
# Names of the Response actions.
names = ["No action", "Alert", "Isolate"]
# Starts the Response section.
out["response"] = {
    # Key: "attack_flows" or "benign_flows".
    ("attack_flows" if a else "benign_flows"): {
        # For each action, its share of that group's flows (the denominator is at least 1).
        names[r]: float(resp_tab[(a, r)] / max(1, sum(resp_tab[(a, x)] for x in range(3)))) for r in range(3)
    # Ends the inner dictionary.
    }
    # Loops over attack (True) and benign (False).
    for a in (True, False)
# Ends the Response section.
}
# Prints the results as JSON.
print(json.dumps(out, indent=1))
# Saves the results to per_agent_test.json in that seed's folder.
json.dump(out, open(f"{d_dir}/per_agent_test.json", "w"), indent=1)
