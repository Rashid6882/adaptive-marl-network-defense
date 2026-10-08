"""Train the single-agent RL baseline (no role separation/coordination)."""

import argparse

from src.data import chronological_split, load_clean_dataset
from src.env import fit_scaler
from src.single_agent_env import SingleAgentEnv
from src.train import train_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--episode-length", type=int, default=2048)
    args = parser.parse_args()

    df = load_clean_dataset()
    train_df, _ = chronological_split(df)
    scaler = fit_scaler(train_df)

    env = SingleAgentEnv(train_df, scaler, episode_length=args.episode_length, seed=4)
    train_agent(env, "single_agent", args.steps, seed=4)


if __name__ == "__main__":
    main()
