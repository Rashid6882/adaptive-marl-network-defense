"""
SHAP explanations for the 3-agent system: for a flow, which inputs pushed each agent
(Detection, Firewall, Response) toward the action it chose?

    python -m src.explain_shap --models-dir runs/seed1
"""

import argparse

import numpy as np
import shap
import torch
from stable_baselines3 import PPO

from src.evaluate_sim import get_split, load_scaler
from src.features import CLASSES, DETECTION_FEATURES, FIREWALL_ACTIONS, RESPONSE_ACTIONS
from src.sim_env import DefenseSim, det_obs, fw_obs, make_fast_predict, resp_obs

CONTEXT = ["recent damage", "recent block rate", "outage active", "alert pending", "recent attack verdicts"]
INPUT_NAMES = {  # what each number in an agent's input means
    "detection": DETECTION_FEATURES + CONTEXT,
    "firewall": [f"verdict={c}" for c in CLASSES] + ["repeat count", "packet rate"] + CONTEXT,
    "response": [f"firewall={a}" for a in FIREWALL_ACTIONS] + ["repeat-offender counter"] + CONTEXT,
}
ACTION_NAMES = {"detection": CLASSES, "firewall": FIREWALL_ACTIONS, "response": RESPONSE_ACTIONS}


class AgentExplainer:
    def __init__(self, name, model, background):
        self.name, self.model, self.background = name, model, background

    def probs(self, x):
        """Probability the agent gives to each of its actions, for a batch of inputs."""
        with torch.no_grad():
            dist = self.model.policy.get_distribution(torch.as_tensor(x, dtype=torch.float32))
        return dist.distribution.probs.numpy()

    def explain(self, obs, action, top=3):
        """SHAP values of the probability of the chosen action; returns the top inputs and a check."""
        explainer = shap.KernelExplainer(lambda x: self.probs(x)[:, action], self.background)
        phi = explainer.shap_values(obs[None, :], nsamples=200, silent=True)[0]
        prob = self.probs(obs[None, :])[0, action]
        # SHAP values must add up: base value + sum(phi) = the model's own probability
        error = abs(float(np.ravel(explainer.expected_value)[0]) + phi.sum() - prob)
        order = np.argsort(-np.abs(phi))[:top]
        names = INPUT_NAMES[self.name]
        return {"prob": float(prob), "error": float(error),
                "top": [(names[i], float(phi[i]), float(obs[i])) for i in order]}


def decide(sim, preds):
    """The three agents' inputs and actions for the current flow."""
    o_det = det_obs(sim)
    d = preds["detection"](o_det)
    o_fw = fw_obs(sim, d)
    f = preds["firewall"](o_fw)
    o_resp = resp_obs(sim, f)
    r = preds["response"](o_resp)
    return {"detection": (o_det, d), "firewall": (o_fw, f), "response": (o_resp, r)}


ap = argparse.ArgumentParser()
ap.add_argument("--models-dir", required=True)
ap.add_argument("--every", type=int, default=700, help="explain every N-th decision for the coverage check")
args = ap.parse_args()

models = {n: PPO.load(f"{args.models_dir}/{n}", device="cpu") for n in ("detection", "firewall", "response")}
preds = {n: make_fast_predict(m) for n, m in models.items()}
sim = DefenseSim(get_split("val"), load_scaler(args.models_dir), seed=0)

# one pass over the validation stream: collect background inputs, sampled decisions and example flows
background = {n: [] for n in models}
sampled = {n: [] for n in models}
examples = {}
for start in sim.file_starts:
    sim.reset(start=int(start), max_steps=None)
    while not sim.done:
        step = decide(sim, preds)
        d, f, r = step["detection"][1], step["firewall"][1], step["response"][1]
        true = int(sim.labels[sim.pos])
        damage, disruption, _ = sim.preview(d, f, r)
        kind = ("attack stopped" if true and not damage else "attack missed" if true else
                "false alarm" if disruption else "benign allowed")
        if kind not in examples:
            examples[kind] = (sim.pos, true, step)
        for n in models:
            if sim.s["steps"] % 1500 == 0:
                background[n].append(step[n][0])
            if sim.s["steps"] % args.every == 0:
                sampled[n].append(step[n])
        sim.apply(d, f, r)

explainers = {n: AgentExplainer(n, models[n], np.array(background[n][:50])) for n in models}

print("=== Example decisions: top 3 inputs behind each agent's choice (SHAP, scaled input value) ===")
for kind, (pos, true, step) in examples.items():
    print(f"\n[{kind}] flow {pos}, true class: {CLASSES[true]}")
    for n in ("detection", "firewall", "response"):
        obs, action = step[n]
        e = explainers[n].explain(obs, action)
        why = ", ".join(f"{name}={val:.2f} ({phi:+.2f})" for name, phi, val in e["top"])
        print(f"  {n.capitalize():9s} -> {ACTION_NAMES[n][action]} (p={e['prob']:.2f}): {why}")

print("\n=== Explainability coverage: share of sampled decisions with a faithful explanation ===")
for n in models:
    errors = [explainers[n].explain(obs, action)["error"] for obs, action in sampled[n]]
    faithful = np.mean(np.array(errors) < 0.02)
    print(f"  {n.capitalize():9s} {faithful:.0%} of {len(errors)} decisions (SHAP adds up to the model's probability within 0.02)")
