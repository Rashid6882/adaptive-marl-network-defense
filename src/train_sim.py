"""
Train one agent inside the closed-loop simulator (src/sim_env.py).

Round-robin co-training: run the stages repeatedly, e.g.
    round 0:  --role detection, --role firewall, --role response
    round 1+: the same three again (each warm-starts from its saved model, with
              the other two now being the learned policies from the last round).
A partner whose model isn't saved yet falls back to the rule-based policy.

    python -m src.train_sim --role detection --steps 150000
    python -m src.train_sim --role single --steps 200000     # single-agent baseline

Experiment protocol (multiple seeds, validation split):
    python -m src.train_sim --role detection --steps 300000 --fresh \
        --use-val-split --seed-id 1 --models-dir runs/seed1 --bf-anchor 0.5
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.env import fit_scaler
from src.sim_env import (
    ROLE_IDX, RoleEnv, SimConfig, SingleSimEnv, load_split_cached, load_splits_cached,
    make_fast_predict,
)

MODELS_DIR = "models_v2"
OUTPUTS_DIR = "outputs"
SEEDS = {"detection": 11, "firewall": 12, "response": 13, "single": 14}


class RewardCurve(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.rewards.append(ep["r"])
        return True


def load_partner(models_dir, name):
    path = f"{models_dir}/{name}.zip"
    if os.path.exists(path):
        return make_fast_predict(PPO.load(path, device="cpu"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["detection", "firewall", "response", "single"])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--fresh", action="store_true", help="ignore any saved model for this role")
    ap.add_argument(
        "--bf-anchor", type=float, default=None,
        help="share of attack-anchored training episodes that start on BruteForce (rest split evenly)",
    )
    ap.add_argument("--models-dir", default=MODELS_DIR, help="where models, scaler and reward curves are stored")
    ap.add_argument("--seed-id", type=int, default=0, help="independent training run id (shifts all RNG seeds)")
    ap.add_argument(
        "--use-val-split", action="store_true",
        help="train only on the first 70%% of each file (validation and test stay unseen)",
    )
    args = ap.parse_args()
    models_dir = args.models_dir
    cfg = None
    if args.bf_anchor is not None:
        rest = (1.0 - args.bf_anchor) / 2.0
        cfg = SimConfig(anchor_class_probs=(rest, rest, args.bf_anchor))

    train_df = load_splits_cached()[0] if args.use_val_split else load_split_cached()[0]
    scaler_path = f"{models_dir}/scaler.joblib"
    os.makedirs(models_dir, exist_ok=True)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = fit_scaler(train_df)
        joblib.dump(scaler, scaler_path)

    seed = SEEDS[args.role] + 1000 * args.seed_id
    if args.role == "single":
        env = SingleSimEnv(train_df, scaler, cfg=cfg, seed=seed)
        name = "single_agent"
    else:
        partners = {"det": None, "fw": None, "resp": None}
        for key, partner_name in (("det", "detection"), ("fw", "firewall"), ("resp", "response")):
            if partner_name != args.role:
                partners[key] = load_partner(models_dir, partner_name)
        print("partners:", {k: ("learned" if v else "rule-based") for k, v in partners.items()}, flush=True)
        env = RoleEnv(train_df, scaler, ROLE_IDX[args.role], partners, cfg=cfg, seed=seed)
        name = args.role

    env = Monitor(env)
    path = f"{models_dir}/{name}.zip"
    if os.path.exists(path) and not args.fresh:
        print("warm-starting from", path, flush=True)
        model = PPO.load(path, env=env, device="cpu")
        reset_ts = False
    else:
        model = PPO(
            "MlpPolicy", env, verbose=0, seed=seed, n_steps=2048, batch_size=256,
            ent_coef=0.01, device="cpu",
        )
        reset_ts = True

    cb = RewardCurve()
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb, reset_num_timesteps=reset_ts)
    dt = time.time() - t0
    model.save(path)

    curve_dir = OUTPUTS_DIR if models_dir == MODELS_DIR else models_dir
    curve_path = f"{curve_dir}/sim_{name}_reward_curve.json"
    prev = json.load(open(curve_path)) if (os.path.exists(curve_path) and not args.fresh) else []
    with open(curve_path, "w") as f:
        json.dump(prev + cb.rewards, f)
    last = np.mean(cb.rewards[-10:]) if cb.rewards else float("nan")
    print(f"[{name}] {args.steps} steps in {dt:.0f}s ({args.steps/dt:.0f} steps/s), "
          f"{len(cb.rewards)} episodes, last-10 mean reward = {last:.2f}", flush=True)


if __name__ == "__main__":
    main()
