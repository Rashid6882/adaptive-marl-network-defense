# Module docstring title: trains one agent inside the closed-loop simulator.
# Explains round-robin co-training: train Detection, then Firewall, then Response, and repeat; in later rounds
# each agent starts from its saved model and the other two are the learned policies from the last round. A
# partner with no saved model uses the rule-based policy.
# Example commands: train one role, or the single-agent baseline.
# Example command for the real experiment protocol (multiple seeds, a models folder per seed, validation
# split, half of attack episodes anchored on BruteForce).
# Ends the docstring.
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

# Imports argparse to read command-line options.
import argparse
# Imports json to save reward curves.
import json
# Imports os for file paths.
import os
# Imports time to measure training time.
import time

# Imports joblib to save and load the feature scaler.
import joblib
# Imports numpy.
import numpy as np
# Imports PPO from Stable-Baselines3 (the learning algorithm).
from stable_baselines3 import PPO
# Imports the callback base class (a hook that runs during training).
from stable_baselines3.common.callbacks import BaseCallback
# Imports Monitor, which records the total reward of each episode.
from stable_baselines3.common.monitor import Monitor

# Imports the function that fits the feature scaler.
from src.env import fit_scaler
# Imports the role numbers, the training environments, the simulator settings, the data loaders and the fast-
# predict wrapper.
from src.sim_env import (
    ROLE_IDX, RoleEnv, SimConfig, SingleSimEnv, load_split_cached, load_splits_cached,
    make_fast_predict,
)

# Default folder for models.
MODELS_DIR = "models_v2"
# Default folder for outputs.
OUTPUTS_DIR = "outputs"
# Base random seed for each role (so each role gets a different seed).
SEEDS = {"detection": 11, "firewall": 12, "response": 13, "single": 14}


# Callback class that records episode rewards for the reward-curve plots.
class RewardCurve(BaseCallback):
    # Constructor.
    def __init__(self):
        # Initializes the parent class.
        super().__init__()
        # List that will hold the episode rewards.
        self.rewards = []

    # Method called by PPO after every environment step.
    def _on_step(self):
        # Loops over the info dictionaries from the environments.
        for info in self.locals.get("infos", []):
            # Gets the episode summary (Monitor adds it only when an episode ends).
            ep = info.get("episode")
            # Checks that an episode just finished.
            if ep is not None:
                # Saves that episode's total reward.
                self.rewards.append(ep["r"])
        # Returns True so training continues.
        return True


# Function that loads a frozen partner agent.
def load_partner(models_dir, name):
    # Path of the saved model file.
    path = f"{models_dir}/{name}.zip"
    # Checks whether the file exists.
    if os.path.exists(path):
        # Loads the model on the CPU and wraps it in the fast prediction function.
        return make_fast_predict(PPO.load(path, device="cpu"))
    # Returns None if there is no saved model (the rule-based policy will be used instead).
    return None


# Main function that runs one training stage.
def main():
    # Creates the command-line parser.
    ap = argparse.ArgumentParser()
    # Required option: which agent to train (detection, firewall, response or single).
    ap.add_argument("--role", required=True, choices=["detection", "firewall", "response", "single"])
    # Number of training steps (default 100,000).
    ap.add_argument("--steps", type=int, default=100_000)
    # Flag: ignore any saved model and start from scratch.
    ap.add_argument("--fresh", action="store_true", help="ignore any saved model for this role")
    # Option: share of attack-anchored episodes that start on BruteForce (the rest split evenly).
    ap.add_argument(
        "--bf-anchor", type=float, default=None,
        help="share of attack-anchored training episodes that start on BruteForce (rest split evenly)",
    )
    # Option: folder where models, scaler and curves are stored.
    ap.add_argument("--models-dir", default=MODELS_DIR, help="where models, scaler and reward curves are stored")
    # Option: id of the independent training run (changes all random seeds).
    ap.add_argument("--seed-id", type=int, default=0, help="independent training run id (shifts all RNG seeds)")
    # Flag: train only on the first 70% of each file so validation and test stay unseen.
    ap.add_argument(
        "--use-val-split", action="store_true",
        help="train only on the first 70%% of each file (validation and test stay unseen)",
    )
    # Reads the command-line options.
    args = ap.parse_args()
    # Stores the models folder.
    models_dir = args.models_dir
    # Default simulator settings (None means use SimConfig defaults).
    cfg = None
    # If a BruteForce share was given...
    if args.bf_anchor is not None:
        # ...the remaining share is split equally between DDoS and PortScan...
        rest = (1.0 - args.bf_anchor) / 2.0
        # ...and a SimConfig is built with those anchor probabilities (DDoS, PortScan, BruteForce).
        cfg = SimConfig(anchor_class_probs=(rest, rest, args.bf_anchor))

    # Chooses training data: the first 70% of each file with --use-val-split, otherwise the first 80%.
    train_df = load_splits_cached()[0] if args.use_val_split else load_split_cached()[0]
    # Path where the scaler is stored.
    scaler_path = f"{models_dir}/scaler.joblib"
    # Creates the models folder if it does not exist.
    os.makedirs(models_dir, exist_ok=True)
    # If a scaler already exists...
    if os.path.exists(scaler_path):
        # ...load it (so all stages use the same scaling)...
        scaler = joblib.load(scaler_path)
    # ...otherwise...
    else:
        # ...fit a new scaler on the training data...
        scaler = fit_scaler(train_df)
        # ...and save it.
        joblib.dump(scaler, scaler_path)

    # Random seed = role's base seed + 1000 x run id.
    seed = SEEDS[args.role] + 1000 * args.seed_id
    # If training the single-agent baseline...
    if args.role == "single":
        # ...create the single-agent environment...
        env = SingleSimEnv(train_df, scaler, cfg=cfg, seed=seed)
        # ...and name the model "single_agent".
        name = "single_agent"
    # Otherwise (training one of the three roles)...
    else:
        # Start with no partners (all rule-based).
        partners = {"det": None, "fw": None, "resp": None}
        # Loops over the three possible partners.
        for key, partner_name in (("det", "detection"), ("fw", "firewall"), ("resp", "response")):
            # Skips the agent that is being trained.
            if partner_name != args.role:
                # Loads that partner's saved model if one exists.
                partners[key] = load_partner(models_dir, partner_name)
        # Prints which partners are learned and which are rule-based.
        print("partners:", {k: ("learned" if v else "rule-based") for k, v in partners.items()}, flush=True)
        # Creates the environment for the role being trained, with the frozen partners.
        env = RoleEnv(train_df, scaler, ROLE_IDX[args.role], partners, cfg=cfg, seed=seed)
        # Names the model after the role.
        name = args.role

    # Wraps the environment in Monitor to record episode rewards.
    env = Monitor(env)
    # Path of this agent's model file.
    path = f"{models_dir}/{name}.zip"
    # If a saved model exists and --fresh was not given...
    if os.path.exists(path) and not args.fresh:
        # ...print that training continues from it...
        print("warm-starting from", path, flush=True)
        # ...load it and attach the environment (warm start)...
        model = PPO.load(path, env=env, device="cpu")
        # ...keep the step counter going.
        reset_ts = False
    # Otherwise...
    else:
        # ...create a new PPO with a standard MLP network, seed, 2,048 steps per update, batch size 256 and
        # entropy bonus 0.01, on CPU.
        model = PPO(
            "MlpPolicy", env, verbose=0, seed=seed, n_steps=2048, batch_size=256,
            ent_coef=0.01, device="cpu",
        )
        # Reset the step counter for a new model.
        reset_ts = True

    # Creates the reward-curve recorder.
    cb = RewardCurve()
    # Records the start time.
    t0 = time.time()
    # Trains the agent for the given number of steps.
    model.learn(total_timesteps=args.steps, callback=cb, reset_num_timesteps=reset_ts)
    # Computes how long training took.
    dt = time.time() - t0
    # Saves the trained model.
    model.save(path)

    # Curves go to outputs/ for the default folder, otherwise into the models folder.
    curve_dir = OUTPUTS_DIR if models_dir == MODELS_DIR else models_dir
    # File name of this agent's reward curve.
    curve_path = f"{curve_dir}/sim_{name}_reward_curve.json"
    # Loads earlier rewards if the file exists and training continued; otherwise starts empty.
    prev = json.load(open(curve_path)) if (os.path.exists(curve_path) and not args.fresh) else []
    # Opens the curve file for writing.
    with open(curve_path, "w") as f:
        # Saves old plus new episode rewards.
        json.dump(prev + cb.rewards, f)
    # Average of the last 10 episode rewards (NaN if none).
    last = np.mean(cb.rewards[-10:]) if cb.rewards else float("nan")
    # Prints a summary: steps, time, speed, episode count and recent reward.
    print(f"[{name}] {args.steps} steps in {dt:.0f}s ({args.steps/dt:.0f} steps/s), "
          f"{len(cb.rewards)} episodes, last-10 mean reward = {last:.2f}", flush=True)


# Runs main() only when the file is run directly.
if __name__ == "__main__":
    # Calls main().
    main()
