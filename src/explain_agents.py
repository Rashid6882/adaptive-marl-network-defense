# Module docstring title: when the 3-agent system causes harm, which agent is responsible, and why?
# Method: for every harmful step (an attack let through or a benign flow disrupted) fix ONE agent at a time
# with the correct decision and see whether the harm disappears; then SHAP shows what the Firewall's decision
# depended on.
# Example command.
# Ends the docstring.
"""
Experiment 2: when the 3-agent system causes harm, which agent is responsible, and why?

For every harmful step (an attack let through, or a benign flow disrupted) we fix ONE agent at a time
with the correct decision and check whether the harm disappears. Then SHAP shows what the Firewall's
decision depended on.

    python -m src.explain_agents --models-dir runs/seed1 --out runs/seed1/explain.json
"""

# Imports argparse for command-line options.
import argparse
# Imports json to save results.
import json
# Imports Counter to count blame categories.
from collections import Counter

# Imports numpy.
import numpy as np
# Imports shap (explanation library).
import shap
# Imports PyTorch.
import torch
# Imports PPO to load saved agents.
from stable_baselines3 import PPO

# Imports the split loader and scaler loader from the evaluation file.
from src.evaluate_sim import get_split, load_scaler
# Imports the simulator, the three observation builders and the fast-predict wrapper.
from src.sim_env import DefenseSim, det_obs, fw_obs, make_fast_predict, resp_obs

# Creates the command-line parser.
ap = argparse.ArgumentParser()
# Required option: the folder with the trained models.
ap.add_argument("--models-dir", required=True)
# Required option: where to write the results JSON.
ap.add_argument("--out", required=True)
# Option: maximum number of cases explained with SHAP for each blame category (default 60).
ap.add_argument("--per-category", type=int, default=60, help="max cases explained with SHAP per blame category")
# Reads the options.
args = ap.parse_args()

# the three trained agents
# Loads the detection, firewall and response models on the CPU.
models = {n: PPO.load(f"{args.models_dir}/{n}", device="cpu") for n in ("detection", "firewall", "response")}
# Wraps each model in a fast prediction function.
det, fw, resp = (make_fast_predict(models[n]) for n in ("detection", "firewall", "response"))
# Builds a simulator on the validation stream.
sim = DefenseSim(get_split("val"), load_scaler(args.models_dir), seed=0)


# Function that measures the harm of a joint action on the current flow.
def harm(d, f, r):
    # Asks the simulator for the immediate damage and disruption of that action without changing state.
    damage, disruption, _ = sim.preview(d, f, r)  # what this joint action would cost right now
    # Harm = damage + disruption (operating cost is ignored).
    return damage + disruption


# pass 1: run the validation stream and look at every harmful step
# Lists for the harmful cases and for background Firewall inputs (used by SHAP).
cases, background = [], []
# Loops over each file's start position.
for start in sim.file_starts:
    # Starts a run through the file with no length limit.
    sim.reset(start=int(start), max_steps=None)
    # Keeps going until the file ends.
    while not sim.done:
        # Detection's actual decision.
        d = det(det_obs(sim))
        # Firewall's actual decision given Detection's verdict.
        f = fw(fw_obs(sim, d))
        # Response's actual decision given the Firewall's action.
        r = resp(resp_obs(sim, f))
        # Every 500th step...
        if sim.s["steps"] % 500 == 0:
            # ...saves the Firewall input as a SHAP background sample.
            background.append(fw_obs(sim, d))
        # Checks whether this joint decision causes harm.
        if harm(d, f, r) > 0:
            # The true class of the flow.
            true = int(sim.labels[sim.pos])
            # True if the flow is an attack.
            attack = true != 0
            # fix Detection only: give the Firewall the true class, let Firewall and Response decide again
            # Gives the Firewall the TRUE class as if Detection were perfect, and lets the Firewall decide
            # again.
            f2 = fw(fw_obs(sim, true))
            # Checks whether the harm disappears when Detection is corrected (Response also decides again).
            fixed_detection = harm(true, f2, resp(resp_obs(sim, f2))) == 0
            # fix Firewall only: correct firewall action (block attacks, allow benign), Response decides again
            # The correct Firewall action: Block for attacks, Allow for benign.
            f3 = 2 if attack else 0
            # Checks whether the harm disappears with the correct Firewall action (Response decides again).
            fixed_firewall = harm(d, f3, resp(resp_obs(sim, f3))) == 0
            # fix Response only: correct response (isolate attacks, do nothing for benign)
            # Checks whether the harm disappears with the correct Response (Isolate for attacks, No action for
            # benign).
            fixed_response = harm(d, f, 2 if attack else 0) == 0
            # Builds the list of agents whose correction alone removes the harm, in pipeline order.
            fixers = [n for n, ok in (("Detection", fixed_detection), ("Firewall", fixed_firewall),
                                      ("Response", fixed_response)) if ok]  # in pipeline order
            # Saves the case with its fixers and the Firewall input (for SHAP).
            cases.append({"fixers": fixers, "obs": fw_obs(sim, d)})
        # Applies the original decision and moves on to the next flow.
        sim.apply(d, f, r)

# Prints the number of harmful steps.
print(f"harmful steps: {len(cases)}")
# Counts harmful steps fixable by exactly one agent (hypothesis H2a, defined before the runs).
exactly_one = sum(len(c["fixers"]) == 1 for c in cases)  # pre-registered H2a
# Prints that count and its share.
print(f"[pre-registered H2a] fixable by exactly one agent: {exactly_one} ({exactly_one / len(cases):.1%})")
# exploratory: root cause = the earliest agent in the pipeline whose correction alone removes the harm
# Loops over the cases.
for c in cases:
    # Blame = the first fixer in pipeline order, or "none" if no single fix works.
    c["blame"] = c["fixers"][0] if c["fixers"] else "none"
# Counts how many cases each agent is blamed for.
counts = Counter(c["blame"] for c in cases)
# Prints a heading for the blame table.
print("[exploratory] root cause (earliest agent that could have prevented the harm):")
# Loops over categories from most to least common.
for label, n in counts.most_common():
    # Prints each category's count and share.
    print(f"  {label:10s} {n:6d}  ({n / len(cases):.1%})")

# pass 2: SHAP on the Firewall's probability of choosing Block
# Function that returns the Firewall's probability of Block for a batch of inputs.
def p_block(x):
    # Turns off gradient tracking.
    with torch.no_grad():
        # Gets the Firewall policy's action distribution for the inputs.
        dist = models["firewall"].policy.get_distribution(torch.as_tensor(x, dtype=torch.float32))
    # Returns the probability of action 2 (Block) for each input.
    return dist.distribution.probs[:, 2].numpy()


# Creates a Kernel SHAP explainer using up to 50 background inputs.
explainer = shap.KernelExplainer(p_block, np.array(background[:50]))
# Groups the Firewall's 11 inputs: the four verdict entries, repeat count, packet rate and five context
# values.
groups = {"Detection verdict": [0, 1, 2, 3], "repeat count": [4], "packet rate": [5], "context": [6, 7, 8, 9, 10]}
# Dictionary that will hold the SHAP shares.
shares = {}
# Loops over each blame category.
for label in counts:
    # Takes the Firewall inputs of the cases in that category (up to the per-category limit).
    chosen = [c["obs"] for c in cases if c["blame"] == label][: args.per_category]
    # Computes SHAP values (200 samples each) and sums their absolute values per input.
    values = np.abs(explainer.shap_values(np.array(chosen), nsamples=200, silent=True)).sum(axis=0)
    # Computes each input group's share of the total.
    shares[label] = {g: float(values[idx].sum() / values.sum()) for g, idx in groups.items()}
    # Prints the shares for this blame category.
    print(f"SHAP share of the Firewall's decision, root cause = {label} ({len(chosen)} cases):",
          {g: f"{v:.0%}" for g, v in shares[label].items()})

# Saves the harmful-step count, the exactly-one count, the blame counts and the SHAP shares as JSON.
json.dump({"harmful_steps": len(cases), "exactly_one_agent": exactly_one, "root_cause_counts": counts,
           "shap_shares_by_root_cause": shares},
          open(args.out, "w"), indent=2)
