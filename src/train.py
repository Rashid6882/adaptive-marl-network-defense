"""
Train the three agents in sequence (Detection -> Firewall -> Response), each
via PPO (Stable-Baselines3), each downstream agent conditioned on a frozen
snapshot of its upstream agent(s) -- see src/env.py docstring for why this is
equivalent to the spec's sequential per-flow conditioning.

Usage:
    python -m src.train --detection-steps 60000 --firewall-steps 40000 --response-steps 40000
"""

import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.data import chronological_split, load_clean_dataset
from src.env import DetectionEnv, FirewallEnv, ResponseEnv, fit_scaler, make_predict_fn

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


class RewardCurveCallback(BaseCallback):
    """Records per-episode total reward as training progresses."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])
        return True


def train_agent(env, name, total_timesteps, seed=0):
    env = Monitor(env)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, n_steps=2048, batch_size=256, device="cpu")
    cb = RewardCurveCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(f"{MODELS_DIR}/{name}")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(f"{OUTPUTS_DIR}/{name}_reward_curve.json", "w") as f:
        json.dump(cb.episode_rewards, f)
    print(f"[{name}] trained for {total_timesteps} steps, "
          f"{len(cb.episode_rewards)} episodes, "
          f"last 10-episode mean reward = {np.mean(cb.episode_rewards[-10:]) if cb.episode_rewards else float('nan'):.3f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection-steps", type=int, default=60_000)
    parser.add_argument("--firewall-steps", type=int, default=40_000)
    parser.add_argument("--response-steps", type=int, default=40_000)
    parser.add_argument("--episode-length", type=int, default=2048)
    parser.add_argument(
        "--only",
        choices=["detection", "firewall", "response"],
        help="train a single stage, loading upstream agents from models/",
    )
    args = parser.parse_args()

    print("Loading and cleaning dataset...")
    df = load_clean_dataset()
    train_df, test_df = chronological_split(df)
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    scaler = fit_scaler(train_df)
    os.makedirs(MODELS_DIR, exist_ok=True)
    import joblib

    joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")

    run = lambda stage: args.only in (None, stage)

    if run("detection"):
        print("\n=== Training Detection agent ===", flush=True)
        det_env = DetectionEnv(train_df, scaler, episode_length=args.episode_length, seed=1)
        det_model = train_agent(det_env, "detection", args.detection_steps, seed=1)
    else:
        det_model = PPO.load(f"{MODELS_DIR}/detection", device="cpu")
    det_predict = make_predict_fn(det_model)

    if run("firewall"):
        print("\n=== Training Firewall agent (conditioned on frozen Detection) ===", flush=True)
        fw_env = FirewallEnv(train_df, scaler, det_predict, episode_length=args.episode_length, seed=2)
        fw_model = train_agent(fw_env, "firewall", args.firewall_steps, seed=2)
    elif args.only == "response":
        fw_model = PPO.load(f"{MODELS_DIR}/firewall", device="cpu")
    fw_predict = make_predict_fn(fw_model) if run("firewall") or args.only == "response" else None

    if run("response"):
        print("\n=== Training Response agent (conditioned on frozen Detection + Firewall) ===", flush=True)
        resp_env = ResponseEnv(
            train_df, scaler, det_predict, fw_predict, episode_length=args.episode_length, seed=3
        )
        train_agent(resp_env, "response", args.response_steps, seed=3)

    print("\nAll agents trained and saved to models/. Reward curves saved to outputs/.")


if __name__ == "__main__":
    main()
