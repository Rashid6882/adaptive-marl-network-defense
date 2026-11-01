# Module docstring title: closed-loop evaluation in the action-dependent simulator; each system runs over a
# whole held-out stream and its own actions change what it sees next.
# Lists the five systems compared: none (always allow), rule (default thresholds), rule_tuned (thresholds
# tuned on validation), v2_marl (the 3 agents) and v2_single (single-agent baseline).
# Example commands: default run, validation run, test run with tuned rule, and sensitivity run.
# Ends the docstring.
"""
Closed-loop evaluation in the action-dependent simulator (src/sim_env.py).

Each system is run over a whole held-out stream (test by default, or validation),
file by file, and its own actions change which flows it sees next. Systems:
  none        - always Allow / No action (damage if nothing is done)
  rule        - static-threshold baseline (default hand-set thresholds)
  rule_tuned  - the same rules with thresholds tuned on validation (src/tune_rule.py)
  v2_marl     - the 3 agents trained interactively in the simulator
  v2_single   - single-agent baseline trained in the simulator

    python -m src.evaluate_sim                                   # default: runs/seed1, test split
    python -m src.evaluate_sim --models-dir runs/seed1 --split val --out runs/seed1/eval_val.json
    python -m src.evaluate_sim --models-dir runs/seed1 --split test --out runs/seed1/eval_test.json \
        --tuned-rule runs/tuned_rule.json
    python -m src.evaluate_sim --models-dir runs/seed1 --sensitivity --out runs/seed1/sensitivity.json
"""

# Imports argparse for command-line options.
import argparse
# Imports dataclasses (used to copy the simulator settings with changes).
import dataclasses
# Imports json.
import json
# Imports os.
import os

# Imports joblib to load the saved scaler.
import joblib
# Imports matplotlib for the chart.
import matplotlib

# Selects a non-interactive drawing backend (writes image files without a window).
matplotlib.use("Agg")
# Imports pyplot.
import matplotlib.pyplot as plt
# Imports numpy.
import numpy as np
# Imports f1_score to measure Detection quality.
from sklearn.metrics import f1_score
# Imports PPO to load trained agents.
from stable_baselines3 import PPO

# Imports the rule-based Firewall and Response rules.
from src.baseline_rule import rule_based_firewall, rule_based_response
# Imports the function that fits a scaler.
from src.env import fit_scaler
# Imports class names, class count and the repeat-offender threshold.
from src.features import CLASSES, N_CLASSES, REPEAT_OFFENDER_THRESHOLD
# Imports the simulator, its settings, the observation builders, the data loaders, the fast-predict wrapper
# and the rule detector.
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
# Imports the function that splits a joint action into three actions.
from src.single_agent_env import decode_action

# Default folder of models (seed 1).
V2_DIR = "runs/seed1"
# Folder for output files.
OUTPUTS_DIR = "outputs"

# Starts the dictionary of simulator variants for the sensitivity check.
SENSITIVITY_VARIANTS = {
    # Default settings, no changes.
    "default": {},
    # Shorter containment durations (half).
    "ttl_x0.5": dict(ttl_block=100, ttl_throttle=60, ttl_isolate=200, ttl_alert=150, alert_delay=20),
    # Longer containment durations (double).
    "ttl_x2": dict(ttl_block=400, ttl_throttle=240, ttl_isolate=800, ttl_alert=600, alert_delay=80),
    # No damage escalation (age cap of 0).
    "no_escalation": dict(age_cap=0.0),
    # Stronger escalation (cap 4, faster growth).
    "strong_escalation": dict(age_cap=4.0, age_scale=30.0),
    # Benign disruption costs doubled.
    "disruption_x2": dict(disruption_block=2.0, disruption_rl=0.6, disruption_isolate=6.0, outage_cost=1.0),
    # Equal base damage for all three attack classes.
    "equal_damage": dict(damage_base=(0.0, 1.0, 1.0, 1.0)),
# Ends the dictionary.
}


# Helper that loads a trained model as a fast predictor.
def _load(path):
    # Returns the predictor if the model file exists, otherwise None.
    return make_fast_predict(PPO.load(path, device="cpu")) if os.path.exists(path + ".zip") else None


# Policy for the "no defense" system.
def policy_none(sim):
    # Always returns (Allow-benign verdict, Allow, No action) = (0, 0, 0).
    return 0, 0, 0


# Function that builds a rule-based policy with a given Response threshold.
def make_rule_policy(resp_thr=REPEAT_OFFENDER_THRESHOLD):
    # Defines the policy function.
    def policy(sim):
        # Detection verdict from the pre-computed rule detector.
        d = int(sim.rule_det[sim.pos])
        # Firewall action from the rule.
        f = rule_based_firewall(d)
        # Returns the three decisions, with Response from the rule.
        return d, f, rule_based_response(f, sim.cur_count, resp_thr)

    # Returns the policy function.
    return policy


# The default rule-based policy.
policy_rule = make_rule_policy()


# Function that builds the 3-agent policy from saved models.
def make_policy_v2(models_dir=V2_DIR):
    # Loads the three models (detection, firewall, response).
    det, fw, resp = (_load(f"{models_dir}/{n}") for n in ("detection", "firewall", "response"))
    # Checks that all three exist.
    if not all((det, fw, resp)):
        # Returns None if any is missing.
        return None

    # Defines the policy function.
    def policy(sim):
        # Detection decides from its input.
        d = det(det_obs(sim))
        # Firewall decides from the verdict.
        f = fw(fw_obs(sim, d))
        # Response decides from the Firewall's action; returns all three decisions.
        return d, f, resp(resp_obs(sim, f))

    # Returns the policy function.
    return policy


# Function that builds the single-agent policy.
def make_policy_single(models_dir=V2_DIR):
    # Loads the single-agent model.
    single = _load(f"{models_dir}/single_agent")
    # Checks whether it exists.
    if single is None:
        # Returns None if it does not.
        return None
    # Returns a function that picks the joint action and splits it into three decisions.
    return lambda sim: decode_action(single(single_obs(sim)))


# Function that runs one policy over the whole stream and returns the metrics.
def run_closed_loop(sim, policy):
    # Clears the simulator's statistics.
    sim.reset_stats()
    # Lists to collect the true classes and Detection's verdicts.
    trues, dets = [], []
    # Loops over each file in the stream.
    for start in sim.file_starts:
        # Starts a run at the beginning of that file with no length limit.
        sim.reset(start=int(start), max_steps=None)
        # Keeps going until the file ends.
        while not sim.done:
            # The policy makes its three decisions.
            d, f, r = policy(sim)
            # Applies them in the simulator and gets the true class.
            true = sim.apply(d, f, r)[4]
            # Records the true class.
            trues.append(true)
            # Records Detection's verdict.
            dets.append(d)
        # At the end of each file, adds attacks that never got contained to the "uncontained" count.
        sim.s["uncontained"] += sim.open_onsets()
    # Short name for the statistics.
    s = sim.s
    # List of containment latencies.
    lat = s["latencies"]
    # Converts the lists to arrays.
    trues, dets = np.array(trues), np.array(dets)
    # Computes the F1 score of each class.
    f1 = f1_score(trues, dets, labels=range(N_CLASSES), average=None, zero_division=0)
    # Starts the results dictionary.
    return {
        # Total attack damage.
        "damage": s["damage"],
        # Total benign disruption.
        "disruption": s["disruption"],
        # Total operating cost.
        "operating_cost": s["cost"],
        # Total loss = damage + disruption + cost.
        "total_loss": s["damage"] + s["disruption"] + s["cost"],
        # Share of benign flows that were disrupted.
        "benign_disrupted_rate": s["benign_disrupted"] / max(1, s["benign_presented"]),
        # Number of containment events.
        "containment_events": len(lat),
        # Number of attacks never contained.
        "uncontained_attacks": s["uncontained"],
        # Mean containment latency in flows (None if no containments).
        "mean_latency_flows": float(np.mean(lat)) if lat else None,
        # Number of attacker flows dropped after containment.
        "attack_flows_dropped": s["dropped"],
        # Macro F1 of Detection's verdicts.
        "detection_macro_f1": float(f1_score(trues, dets, average="macro", zero_division=0)),
        # Per-class F1 scores.
        "detection_f1": dict(zip(CLASSES, [float(x) for x in f1])),
    # Ends the dictionary.
    }


# Faster version that returns only the total loss (used by the rule-tuning search).
def quick_total_loss(sim, policy):
    # Docstring.
    """Closed-loop total loss only (no F1 bookkeeping); used for threshold search."""
    # Clears statistics.
    sim.reset_stats()
    # Loops over each file.
    for start in sim.file_starts:
        # Starts a run at the beginning of the file.
        sim.reset(start=int(start), max_steps=None)
        # Keeps going until the file ends.
        while not sim.done:
            # Applies the policy's decisions.
            sim.apply(*policy(sim))
    # Short name for the statistics.
    s = sim.s
    # Returns damage + disruption + cost.
    return s["damage"] + s["disruption"] + s["cost"]


# Function that returns the validation or the test table.
def get_split(split):
    # If validation was asked for...
    if split == "val":
        # ...return the validation table (index 1)...
        return load_splits_cached()[1]
    # ...otherwise return the test table.
    return load_split_cached()[1]


# Function that loads the scaler of a models folder.
def load_scaler(models_dir):
    # Path of the scaler file.
    path = f"{models_dir}/scaler.joblib"
    # If it exists...
    if os.path.exists(path):
        # ...load it...
        return joblib.load(path)
    # ...otherwise fit a new one on the training data.
    return fit_scaler(load_splits_cached()[0])


# Function that evaluates all systems on a data table.
def evaluate(df, scaler, models_dir, cfg=None, tuned_rule=None, label=""):
    # Creates the simulator on that data.
    sim = DefenseSim(df, scaler, cfg=cfg, seed=0)
    # Remembers the default rule detections.
    default_det = sim.rule_det
    # Starts the systems dictionary with "none" and "rule".
    systems = {"none": (policy_none, None), "rule": (policy_rule, None)}
    # If tuned thresholds were given...
    if tuned_rule:
        # ...compute the rule detections with them...
        arr = rule_detect_array(df, tuned_rule)
        # ...and add the "rule_tuned" system.
        systems["rule_tuned"] = (make_rule_policy(tuned_rule.get("resp_thr", REPEAT_OFFENDER_THRESHOLD)), arr)
    # Loops over the two learned systems.
    for name, make in (("v2_marl", make_policy_v2), ("v2_single", make_policy_single)):
        # Builds the policy from saved models.
        pol = make(models_dir)
        # If the models are missing...
        if pol is None:
            # ...print a message and skip that system.
            print(f"(skipping {name}: models not found in {models_dir})")
        # Otherwise...
        else:
            # ...add the system.
            systems[name] = (pol, None)

    # Dictionary for the results.
    results = {}
    # Loops over the systems.
    for name, (pol, det_arr) in systems.items():
        # Prints which system is running.
        print(f"running {name} {label}...", flush=True)
        # Uses the system's rule detections (or the default ones).
        sim.rule_det = det_arr if det_arr is not None else default_det
        # Runs the closed loop and stores the metrics.
        results[name] = run_closed_loop(sim, pol)
    # Restores the default rule detections.
    sim.rule_det = default_det
    # Damage of the no-defense system (baseline).
    base = results["none"]["damage"]
    # Loops over all results.
    for r in results.values():
        # Damage reduction = 1 - damage / no-defense damage.
        r["damage_reduction"] = 1 - r["damage"] / base if base else 0.0
    # Returns the results.
    return results


# Function that prints a results table.
def print_table(results):
    # Prints the header row with the metric names.
    print(f"\n{'system':11s} {'damage':>9s} {'disrupt':>9s} {'cost':>7s} {'TOTAL':>9s} {'dmg red.':>9s} "
          f"{'benign FP':>10s} {'latency':>8s} {'uncont.':>8s} {'macroF1':>8s}")
    # Loops over the systems.
    for name, r in results.items():
        # Formats the latency (or "n/a").
        lat = f"{r['mean_latency_flows']:.1f}" if r["mean_latency_flows"] is not None else "n/a"
        # Prints one table row with damage, disruption, cost, total loss, damage reduction, benign false-
        # positive rate, latency, uncontained attacks and macro F1.
        print(f"{name:11s} {r['damage']:9.0f} {r['disruption']:9.0f} {r['operating_cost']:7.0f} "
              f"{r['total_loss']:9.0f} {r['damage_reduction']:9.1%} {r['benign_disrupted_rate']:10.2%} {lat:>8s} "
              f"{r['uncontained_attacks']:8d} {r['detection_macro_f1']:8.3f}")


# Main function.
def main():
    # Creates the command-line parser.
    ap = argparse.ArgumentParser()
    # Option: models folder.
    ap.add_argument("--models-dir", default=V2_DIR)
    # Option: test or val split.
    ap.add_argument("--split", choices=["test", "val"], default="test")
    # Option: where to write results JSON.
    ap.add_argument("--out", default=None, help="write results JSON here (default: legacy outputs/ files)")
    # Option: tuned rule thresholds file.
    ap.add_argument("--tuned-rule", default=None, help="JSON with tuned rule thresholds (from src.tune_rule)")
    # Flag: run the sensitivity check.
    ap.add_argument("--sensitivity", action="store_true", help="re-evaluate under varied simulator settings")
    # Option: subset of sensitivity variants.
    ap.add_argument("--variants", default=None, help="comma-separated subset of sensitivity variants to run")
    # Reads the options.
    args = ap.parse_args()

    # Gets the chosen data split.
    df = get_split(args.split)
    # Loads the scaler.
    scaler = load_scaler(args.models_dir)
    # Starts with no tuned thresholds.
    tuned = None
    # If a tuned-rule file was given...
    if args.tuned_rule:
        # ...load its parameters.
        tuned = json.load(open(args.tuned_rule))["params"]

    # If the sensitivity check was requested...
    if args.sensitivity:
        # ...create the results dictionary...
        out = {}
        # ...choose which variants to run...
        chosen = args.variants.split(",") if args.variants else list(SENSITIVITY_VARIANTS)
        # ...loop over them...
        for vname in chosen:
            # ...get the variant's setting changes...
            overrides = SENSITIVITY_VARIANTS[vname]
            # ...copy the default settings with those changes...
            cfg = dataclasses.replace(SimConfig(), **overrides)
            # ...evaluate all systems under the variant...
            out[vname] = evaluate(df, scaler, args.models_dir, cfg=cfg, tuned_rule=tuned, label=f"[{vname}] ")
            # ...print the variant name...
            print(f"\n=== variant: {vname} ===")
            # ...and its table.
            print_table(out[vname])
        # If an output file was given...
        if args.out:
            # ...create its folder...
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            # ...save all variants as JSON...
            json.dump(out, open(args.out, "w"), indent=2)
            # ...print where it was saved.
            print("saved", args.out)
        # Stops after the sensitivity run.
        return

    # Evaluates all systems with the default settings.
    results = evaluate(df, scaler, args.models_dir, tuned_rule=tuned)
    # Prints the table.
    print_table(results)

    # If an output file was given...
    if args.out:
        # ...create its folder...
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # ...save the results...
        json.dump(results, open(args.out, "w"), indent=2)
        # ...print where they were saved...
        print("saved", args.out)
        # ...and stop.
        return

    # Otherwise creates the outputs folder.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Saves the summary JSON in outputs/.
    with open(f"{OUTPUTS_DIR}/eval_sim_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    # List of system names for the chart.
    names = list(results)
    # Creates a figure.
    fig, ax = plt.subplots(figsize=(8, 5))
    # Starts the stacked bars at zero.
    bottom = np.zeros(len(names))
    # Loops over damage, disruption and operating cost.
    for key, label in (("damage", "attack damage"), ("disruption", "benign disruption"), ("operating_cost", "operating cost")):
        # Values of that component for each system.
        vals = np.array([results[n][key] for n in names])
        # Draws the bars stacked on the previous ones.
        ax.bar(names, vals, bottom=bottom, label=label)
        # Raises the stacking base.
        bottom += vals
    # Y-axis label.
    ax.set_ylabel("Total loss on held-out test stream (lower is better)")
    # Chart title.
    ax.set_title("Closed-loop simulator: total loss by system")
    # Adds the legend.
    ax.legend()
    # Adjusts the layout.
    fig.tight_layout()
    # Saves the chart as a PNG.
    fig.savefig(f"{OUTPUTS_DIR}/eval_sim_loss.png", dpi=150)
    # Prints where the files were saved.
    print(f"Saved {OUTPUTS_DIR}/eval_sim_summary.json and eval_sim_loss.png")


# Runs main() only when the file is run directly.
if __name__ == "__main__":
    # Calls main().
    main()
