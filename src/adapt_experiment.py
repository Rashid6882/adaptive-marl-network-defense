"""
Experiment 1: how many training steps does it take to adapt to a new ("slow") attack variant?
  marl   : only the Detection agent retrains; Firewall and Response stay frozen.
  single : the whole single-agent policy retrains.

    python -m src.adapt_experiment --base-dir runs/seed1 --system marl --variant slow_ddos --out runs/adapt/s1_ddos_marl.json
    (use --checkpoints "" to only measure the base models on the shifted traffic)
"""

import argparse
import json

import joblib
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.features import CLASSES
from src.shift import VARIANTS, apply_variant
from src.single_agent_env import decode_action
from src.sim_env import (
    DefenseSim, RoleEnv, SimConfig, SingleSimEnv, det_obs, fw_obs, load_splits_cached,
    make_fast_predict, resp_obs, single_obs,
)


def window_loss(sim, policy, starts):
    """Average total loss over short windows of 1024 flows starting at the given rows."""
    total = 0.0
    for s in starts:
        sim.reset_stats()
        sim.reset(start=int(s), max_steps=1024)
        while not sim.done:
            sim.apply(*policy(sim))
        total += sim.s["damage"] + sim.s["disruption"] + sim.s["cost"]
    return total / len(starts)


ap = argparse.ArgumentParser()
ap.add_argument("--base-dir", required=True)
ap.add_argument("--system", choices=["marl", "single"], required=True)
ap.add_argument("--variant", choices=list(VARIANTS), required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--checkpoints", default="25000,50000,100000,200000,400000")
ap.add_argument("--n-windows", type=int, default=24)
args = ap.parse_args()
checkpoints = [int(x) for x in args.checkpoints.split(",") if x]

# data: the attack variant is applied to the train split (to adapt) and the validation split (to evaluate)
train_df, val_df, _ = load_splits_cached()
scaler = joblib.load(f"{args.base_dir}/scaler.joblib")
shifted_train, shifted_val = apply_variant(train_df, args.variant), apply_variant(val_df, args.variant)
sim_shifted, sim_normal = DefenseSim(shifted_val, scaler), DefenseSim(val_df, scaler)

# fixed evaluation windows: around the shifted attack class, and random "general" ones to check forgetting
cls = CLASSES.index(VARIANTS[args.variant][0])
rows = np.where(sim_normal.labels == cls)[0]
anchors = rows[np.linspace(0, len(rows) - 1, args.n_windows).astype(int)]
class_starts = [max(sim_normal.file_start[a], a - 30) for a in anchors]
general_starts = np.random.default_rng(0).integers(0, sim_normal.n - 1024, args.n_windows)


def load(name):
    return PPO.load(f"{args.base_dir}/{name}", device="cpu")


# the model that keeps learning, the environment it learns in, and how to turn it into a policy
cfg = SimConfig(anchor_class_probs=(0.25, 0.25, 0.5))
if args.system == "marl":
    fw, resp = make_fast_predict(load("firewall")), make_fast_predict(load("response"))
    env = RoleEnv(shifted_train, scaler, 0, {"det": None, "fw": fw, "resp": resp}, cfg=cfg, seed=7)
    model = PPO.load(f"{args.base_dir}/detection", env=Monitor(env), device="cpu")

    def make_policy():
        det = make_fast_predict(model)

        def policy(sim):
            d = det(det_obs(sim))
            f = fw(fw_obs(sim, d))
            return d, f, resp(resp_obs(sim, f))

        return policy
else:
    env = SingleSimEnv(shifted_train, scaler, cfg=cfg, seed=7)
    model = PPO.load(f"{args.base_dir}/single_agent", env=Monitor(env), device="cpu")

    def make_policy():
        single = make_fast_predict(model)
        return lambda sim: decode_action(single(single_obs(sim)))

# train in chunks; after each chunk, measure the loss on shifted traffic and on unshifted traffic
results, steps_done = [], 0
for target in [0] + checkpoints:
    if target > steps_done:
        model.learn(target - steps_done, reset_num_timesteps=False)
        steps_done = target
    policy = make_policy()
    if target == 0:
        reference = window_loss(sim_normal, policy, class_starts)  # loss on the SAME windows before the shift
    row = {
        "steps": steps_done,
        "shifted_loss": window_loss(sim_shifted, policy, class_starts),
        "general_loss": window_loss(sim_normal, policy, general_starts),
    }
    results.append(row)
    print(row, flush=True)

# steps needed to get 90% of the way back from the shifted loss to the pre-shift loss
start_loss = results[0]["shifted_loss"]
goal = start_loss - 0.9 * (start_loss - reference)
reached = [r["steps"] for r in results if r["shifted_loss"] <= goal and r["steps"] > 0]
print(f"pre-shift loss {reference:.1f} | shifted loss {start_loss:.1f} (x{start_loss / max(reference, 1.0):.1f})"
      f" | steps to 90% recovery: {reached[0] if reached else 'not reached'}")
json.dump({"reference": reference, "results": results, "steps_to_90pct": reached[0] if reached else None},
          open(args.out, "w"), indent=2)
