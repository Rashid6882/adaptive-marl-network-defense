# Module docstring title: tunes the rule-based baseline's thresholds so the comparison is not against a weak
# strawman.
# Random search plus local refinement, minimizing closed-loop total loss on the validation stream only; test
# flows are never used.
# Example command (1,200 trials).
# Ends the docstring.
"""
Tune the rule-based baseline's thresholds so the comparison is not against a strawman.

Random search plus local refinement over the detection and response thresholds,
minimizing closed-loop total loss on the VALIDATION stream only. Test flows are
never used. Evaluate the result on test with:
    python -m src.evaluate_sim --tuned-rule runs/tuned_rule.json ...

    python -m src.tune_rule --out runs/tuned_rule.json --trials 1200
"""

# Imports argparse for command-line options.
import argparse
# Imports json to save results.
import json
# Imports os for folders.
import os
# Imports time to enforce a time budget.
import time

# Imports numpy for random numbers.
import numpy as np

# Imports helpers from the evaluation file: scaler loading, the rule policy builder and a fast total-loss
# function.
from src.evaluate_sim import load_scaler, make_rule_policy, quick_total_loss
# Imports the default thresholds, the simulator, the validation loader and the vectorized rule detector.
from src.sim_env import DEFAULT_RULE_PARAMS, DefenseSim, load_splits_cached, rule_detect_array


# Function that draws one random set of thresholds.
def sample(rng):
    # Returns a dictionary of parameters.
    return dict(
        # DDoS packet-rate threshold: log-uniform between 100 and 30,000.
        ddos_pkt=float(np.exp(rng.uniform(np.log(100), np.log(30000)))),
        # DDoS forward-packet threshold: whole number from 1 to 59.
        ddos_fwd=int(rng.integers(1, 60)),
        # PortScan duration threshold: log-uniform between 50 and 200,000.
        ps_dur=float(np.exp(rng.uniform(np.log(50), np.log(2e5)))),
        # PortScan repeat-count threshold: whole number from 2 to 39.
        ps_rep=int(rng.integers(2, 40)),
        # BruteForce repeat-count threshold: whole number from 1 to 29.
        bf_rep=int(rng.integers(1, 30)),
        # BruteForce duration threshold: log-uniform between 100,000 and 100,000,000.
        bf_dur=float(np.exp(rng.uniform(np.log(1e5), np.log(1e8)))),
        # Response repeat-offender threshold: whole number from 2 to 19.
        resp_thr=int(rng.integers(2, 20)),
    # Ends the dictionary.
    )


# Function that makes a small random change to a set of thresholds (local refinement).
def perturb(p, rng):
    # Copies the parameters.
    q = dict(p)
    # Loops over the three continuous thresholds.
    for k in ("ddos_pkt", "ps_dur", "bf_dur"):
        # Multiplies each by a random factor around 1 (about plus or minus 30%).
        q[k] = float(p[k] * np.exp(rng.normal(0, 0.3)))
    # Loops over the four whole-number thresholds with their allowed ranges.
    for k, lo, hi in (("ddos_fwd", 1, 60), ("ps_rep", 2, 40), ("bf_rep", 1, 30), ("resp_thr", 2, 20)):
        # Adds a random whole number from -3 to +3 and keeps the result inside the allowed range.
        q[k] = int(np.clip(p[k] + rng.integers(-3, 4), lo, hi))
    # Returns the changed parameters.
    return q


# Main function.
def main():
    # Creates the command-line parser.
    ap = argparse.ArgumentParser()
    # Required option: where to write the tuned thresholds.
    ap.add_argument("--out", required=True)
    # Option: folder used only to find a feature scaler.
    ap.add_argument("--models-dir", default="runs/seed1", help="only used to find a feature scaler")
    # Option: number of search trials (default 1,200).
    ap.add_argument("--trials", type=int, default=1200)
    # Option: time limit in minutes (default 8).
    ap.add_argument("--max-minutes", type=float, default=8.0)
    # Option: random seed.
    ap.add_argument("--seed", type=int, default=0)
    # Reads the options.
    args = ap.parse_args()

    # Loads the validation table (index 1 of the three-way split).
    val_df = load_splits_cached()[1]
    # Builds a simulator on the validation data.
    sim = DefenseSim(val_df, load_scaler(args.models_dir), seed=0)
    # Creates the random generator.
    rng = np.random.default_rng(args.seed)

    # Inner function that scores a set of thresholds.
    def loss_of(params):
        # Recomputes the rule-based detection of every validation flow with those thresholds.
        sim.rule_det = rule_detect_array(val_df, params)
        # Runs the whole validation stream and returns the total loss.
        return quick_total_loss(sim, make_rule_policy(params["resp_thr"]))

    # Loss of the default thresholds (the starting point).
    default_loss = loss_of(DEFAULT_RULE_PARAMS)
    # Starts with the defaults as the best set found so far.
    best, best_loss = dict(DEFAULT_RULE_PARAMS), default_loss
    # Prints the default validation loss.
    print(f"validation total loss with default thresholds: {default_loss:.0f}", flush=True)

    # Records the start time.
    t0 = time.time()
    # Counts completed trials.
    done = 0
    # The first 75% of trials are fully random; the rest are local refinements.
    n_random = int(args.trials * 0.75)
    # Loops over the trials.
    for i in range(args.trials):
        # Checks whether the time limit has passed.
        if time.time() - t0 > args.max_minutes * 60:
            # If so, prints how many trials ran...
            print(f"time budget reached after {done} trials", flush=True)
            # ...and stops.
            break
        # Picks a random candidate early on, or a perturbed copy of the best set later.
        cand = sample(rng) if i < n_random else perturb(best, rng)
        # Computes the candidate's validation loss.
        loss = loss_of(cand)
        # Counts the trial.
        done += 1
        # If this candidate is better than the best so far...
        if loss < best_loss:
            # ...it becomes the new best...
            best, best_loss = cand, loss
            # ...and the improvement is printed.
            print(f"  trial {i}: new best validation loss {best_loss:.0f}", flush=True)

    # Creates the output folder if needed.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Opens the output file for writing.
    with open(args.out, "w") as f:
        # Saves the best parameters, their validation loss, the default loss and the number of trials as JSON.
        json.dump(
            {"params": best, "val_loss": best_loss, "default_val_loss": default_loss, "trials": done},
            f, indent=2,
        )
    # Prints the best and default losses.
    print(f"best validation loss {best_loss:.0f} (default {default_loss:.0f}) after {done} trials")
    # Prints the best parameters.
    print("params:", json.dumps(best))


# Runs main() only when the file is run directly.
if __name__ == "__main__":
    # Calls main().
    main()
