# Module docstring: SHAP explanations for the 3 agents: for a flow, which inputs pushed each agent toward the
# action it chose?
# Example command.
# Ends the docstring.
"""
SHAP explanations for the 3-agent system: for a flow, which inputs pushed each agent
(Detection, Firewall, Response) toward the action it chose?

    python -m src.explain_shap --models-dir runs/seed1
"""

# Imports argparse.
import argparse

# Imports numpy.
import numpy as np
# Imports shap (explanation library).
import shap
# Imports PyTorch.
import torch
# Imports PPO to load saved agents.
from stable_baselines3 import PPO

# Imports the split loader and scaler loader.
from src.evaluate_sim import get_split, load_scaler
# Imports class names, feature names and action names.
from src.features import CLASSES, DETECTION_FEATURES, FIREWALL_ACTIONS, RESPONSE_ACTIONS
# Imports the simulator, the observation builders and the fast-predict wrapper.
from src.sim_env import DefenseSim, det_obs, fw_obs, make_fast_predict, resp_obs

# Names of the five context inputs.
CONTEXT = ["recent damage", "recent block rate", "outage active", "alert pending", "recent attack verdicts"]
# Dictionary naming every number in each agent's input.
INPUT_NAMES = {  # what each number in an agent's input means
    # Detection inputs: the 21 feature names plus the context names.
    "detection": DETECTION_FEATURES + CONTEXT,
    # Firewall inputs: four "verdict=class" names, repeat count, packet rate, plus context.
    "firewall": [f"verdict={c}" for c in CLASSES] + ["repeat count", "packet rate"] + CONTEXT,
    # Response inputs: three "firewall=action" names, the repeat-offender counter, plus context.
    "response": [f"firewall={a}" for a in FIREWALL_ACTIONS] + ["repeat-offender counter"] + CONTEXT,
# Closes the dictionary.
}
# Action names for each agent (used when printing).
ACTION_NAMES = {"detection": CLASSES, "firewall": FIREWALL_ACTIONS, "response": RESPONSE_ACTIONS}


# Class that explains one agent's decisions.
class AgentExplainer:
    # Constructor taking the agent's name, its model and background samples.
    def __init__(self, name, model, background):
        # Stores those three values.
        self.name, self.model, self.background = name, model, background

    # Method that returns the agent's probability for each action.
    def probs(self, x):
        # Docstring for that method.
        """Probability the agent gives to each of its actions, for a batch of inputs."""
        # Turns off gradient tracking.
        with torch.no_grad():
            # Gets the policy's action distribution for the inputs.
            dist = self.model.policy.get_distribution(torch.as_tensor(x, dtype=torch.float32))
        # Returns the probabilities as a numpy array.
        return dist.distribution.probs.numpy()

    # Method that explains one decision (default: top 3 inputs).
    def explain(self, obs, action, top=3):
        # Docstring: SHAP values for the probability of the chosen action, plus a faithfulness check.
        """SHAP values of the probability of the chosen action; returns the top inputs and a check."""
        # Creates a Kernel SHAP explainer whose output is the probability of the chosen action.
        explainer = shap.KernelExplainer(lambda x: self.probs(x)[:, action], self.background)
        # Computes each input's SHAP value for this one observation (200 samples).
        phi = explainer.shap_values(obs[None, :], nsamples=200, silent=True)[0]
        # The model's own probability for the chosen action.
        prob = self.probs(obs[None, :])[0, action]
        # SHAP values must add up: base value + sum(phi) = the model's own probability
        # The additivity error: how far that sum is from the model's probability.
        error = abs(float(np.ravel(explainer.expected_value)[0]) + phi.sum() - prob)
        # Indexes of the inputs with the largest absolute SHAP values (top 3).
        order = np.argsort(-np.abs(phi))[:top]
        # Names of this agent's inputs.
        names = INPUT_NAMES[self.name]
        # Returns the probability, the error and the top inputs with their SHAP value and scaled input value.
        return {"prob": float(prob), "error": float(error),
                "top": [(names[i], float(phi[i]), float(obs[i])) for i in order]}


# Function that gets all three agents' inputs and actions for the current flow.
def decide(sim, preds):
    # Docstring.
    """The three agents' inputs and actions for the current flow."""
    # Detection's input.
    o_det = det_obs(sim)
    # Detection's action.
    d = preds["detection"](o_det)
    # Firewall's input (uses Detection's verdict).
    o_fw = fw_obs(sim, d)
    # Firewall's action.
    f = preds["firewall"](o_fw)
    # Response's input (uses the Firewall's action).
    o_resp = resp_obs(sim, f)
    # Response's action.
    r = preds["response"](o_resp)
    # Returns each agent's (input, action) pair.
    return {"detection": (o_det, d), "firewall": (o_fw, f), "response": (o_resp, r)}


# Creates the command-line parser.
ap = argparse.ArgumentParser()
# Required option: models folder.
ap.add_argument("--models-dir", required=True)
# Option: explain every N-th decision for the coverage check (default 700).
ap.add_argument("--every", type=int, default=700, help="explain every N-th decision for the coverage check")
# Reads the options.
args = ap.parse_args()

# Loads the three trained models.
models = {n: PPO.load(f"{args.models_dir}/{n}", device="cpu") for n in ("detection", "firewall", "response")}
# Wraps each in a fast prediction function.
preds = {n: make_fast_predict(m) for n, m in models.items()}
# Builds a simulator on the validation stream.
sim = DefenseSim(get_split("val"), load_scaler(args.models_dir), seed=0)

# one pass over the validation stream: collect background inputs, sampled decisions and example flows
# Empty background lists per agent.
background = {n: [] for n in models}
# Empty lists of sampled decisions per agent.
sampled = {n: [] for n in models}
# Empty dictionary of example flows by kind.
examples = {}
# Loops over each file's start position.
for start in sim.file_starts:
    # Starts a run through the file.
    sim.reset(start=int(start), max_steps=None)
    # Keeps going until the file ends.
    while not sim.done:
        # Gets all three agents' inputs and actions.
        step = decide(sim, preds)
        # Pulls out the three actions.
        d, f, r = step["detection"][1], step["firewall"][1], step["response"][1]
        # True class of the flow.
        true = int(sim.labels[sim.pos])
        # Previews the damage and disruption of that decision.
        damage, disruption, _ = sim.preview(d, f, r)
        # Labels the flow: "attack stopped", "attack missed", "false alarm" or "benign allowed".
        kind = ("attack stopped" if true and not damage else "attack missed" if true else
                "false alarm" if disruption else "benign allowed")
        # Checks whether this kind has no example yet.
        if kind not in examples:
            # Saves the first flow of each kind as the example.
            examples[kind] = (sim.pos, true, step)
        # Loops over the three agents.
        for n in models:
            # Every 1,500th step...
            if sim.s["steps"] % 1500 == 0:
                # ...saves the agent's input as a SHAP background sample.
                background[n].append(step[n][0])
            # Every N-th step (default 700)...
            if sim.s["steps"] % args.every == 0:
                # ...saves the decision for the coverage check.
                sampled[n].append(step[n])
        # Applies the decision and moves on.
        sim.apply(d, f, r)

# Creates an explainer for each agent using up to 50 background inputs.
explainers = {n: AgentExplainer(n, models[n], np.array(background[n][:50])) for n in models}

# Prints a heading for the example decisions.
print("=== Example decisions: top 3 inputs behind each agent's choice (SHAP, scaled input value) ===")
# Loops over the example flows.
for kind, (pos, true, step) in examples.items():
    # Prints the kind, flow position and true class.
    print(f"\n[{kind}] flow {pos}, true class: {CLASSES[true]}")
    # Loops over the three agents.
    for n in ("detection", "firewall", "response"):
        # Gets that agent's input and chosen action.
        obs, action = step[n]
        # Explains the decision with SHAP.
        e = explainers[n].explain(obs, action)
        # Builds a text of the top inputs with their value and SHAP contribution.
        why = ", ".join(f"{name}={val:.2f} ({phi:+.2f})" for name, phi, val in e["top"])
        # Prints the agent, its action, the probability and the reasons.
        print(f"  {n.capitalize():9s} -> {ACTION_NAMES[n][action]} (p={e['prob']:.2f}): {why}")

# Prints a heading for the coverage check.
print("\n=== Explainability coverage: share of sampled decisions with a faithful explanation ===")
# Loops over the agents.
for n in models:
    # Explains each sampled decision and collects the additivity errors.
    errors = [explainers[n].explain(obs, action)["error"] for obs, action in sampled[n]]
    # Share of explanations whose error is below 0.02 (the faithful ones).
    faithful = np.mean(np.array(errors) < 0.02)
    # Prints the share and the number of decisions checked.
    print(f"  {n.capitalize():9s} {faithful:.0%} of {len(errors)} decisions (SHAP adds up to the model's probability within 0.02)")
