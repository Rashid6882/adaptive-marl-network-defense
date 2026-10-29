"""
Tune the rule-based baseline's thresholds so the comparison is not against a strawman.

Random search plus local refinement over the detection and response thresholds,
minimizing closed-loop total loss on the VALIDATION stream only. Test flows are
never used. Evaluate the result on test with:
    python -m src.evaluate_sim --tuned-rule runs/tuned_rule.json ...

    python -m src.tune_rule --out runs/tuned_rule.json --trials 1200
"""

import argparse
import json
import os
import time

import numpy as np

from src.evaluate_sim import load_scaler, make_rule_policy, quick_total_loss
from src.sim_env import DEFAULT_RULE_PARAMS, DefenseSim, load_splits_cached, rule_detect_array


def sample(rng):
    return dict(
        ddos_pkt=float(np.exp(rng.uniform(np.log(100), np.log(30000)))),
        ddos_fwd=int(rng.integers(1, 60)),
        ps_dur=float(np.exp(rng.uniform(np.log(50), np.log(2e5)))),
        ps_rep=int(rng.integers(2, 40)),
        bf_rep=int(rng.integers(1, 30)),
        bf_dur=float(np.exp(rng.uniform(np.log(1e5), np.log(1e8)))),
        resp_thr=int(rng.integers(2, 20)),
    )


def perturb(p, rng):
    q = dict(p)
    for k in ("ddos_pkt", "ps_dur", "bf_dur"):
        q[k] = float(p[k] * np.exp(rng.normal(0, 0.3)))
    for k, lo, hi in (("ddos_fwd", 1, 60), ("ps_rep", 2, 40), ("bf_rep", 1, 30), ("resp_thr", 2, 20)):
        q[k] = int(np.clip(p[k] + rng.integers(-3, 4), lo, hi))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default="models_v2", help="only used to find a feature scaler")
    ap.add_argument("--trials", type=int, default=1200)
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    val_df = load_splits_cached()[1]
    sim = DefenseSim(val_df, load_scaler(args.models_dir), seed=0)
    rng = np.random.default_rng(args.seed)

    def loss_of(params):
        sim.rule_det = rule_detect_array(val_df, params)
        return quick_total_loss(sim, make_rule_policy(params["resp_thr"]))

    default_loss = loss_of(DEFAULT_RULE_PARAMS)
    best, best_loss = dict(DEFAULT_RULE_PARAMS), default_loss
    print(f"validation total loss with default thresholds: {default_loss:.0f}", flush=True)

    t0 = time.time()
    done = 0
    n_random = int(args.trials * 0.75)
    for i in range(args.trials):
        if time.time() - t0 > args.max_minutes * 60:
            print(f"time budget reached after {done} trials", flush=True)
            break
        cand = sample(rng) if i < n_random else perturb(best, rng)
        loss = loss_of(cand)
        done += 1
        if loss < best_loss:
            best, best_loss = cand, loss
            print(f"  trial {i}: new best validation loss {best_loss:.0f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"params": best, "val_loss": best_loss, "default_val_loss": default_loss, "trials": done},
            f, indent=2,
        )
    print(f"best validation loss {best_loss:.0f} (default {default_loss:.0f}) after {done} trials")
    print("params:", json.dumps(best))


if __name__ == "__main__":
    main()
