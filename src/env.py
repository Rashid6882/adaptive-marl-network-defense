"""
Sequential 3-agent network-defense environment.

Each network flow = one timestep. Detection acts first, Firewall acts
conditioned on Detection's action, Response acts conditioned on Firewall's
action. Because the flows themselves are a fixed dataset replay (not causally
affected by agent actions), the only cross-agent coupling that matters is
*within the same timestep*: a downstream agent's observation depends on the
upstream agent's action for the *current* flow. This lets each agent be
trained with Stable-Baselines3 PPO independently, using the latest frozen
snapshot of its upstream agent(s) to build its observations (a standard
independent-learning MARL setup), while still respecting the sequential
conditioning the project spec requires.

Three Gymnasium environments are provided:
  - DetectionEnv:  standalone, no upstream dependency.
  - FirewallEnv:   needs a frozen Detection policy.
  - ResponseEnv:   needs frozen Detection and Firewall policies.
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

from src.features import (
    CLASSES,
    DETECTION_FEATURES,
    N_CLASSES,
    REPEAT_OFFENDER_THRESHOLD,
    detection_reward,
    firewall_reward,
    response_reward,
)

DPORT_IDX = DETECTION_FEATURES.index("dport_repeat_count")
PKTS_IDX = DETECTION_FEATURES.index("Flow Packets/s")


def fit_scaler(train_df) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[DETECTION_FEATURES].values)
    return scaler


def _prep_arrays(df, scaler: StandardScaler):
    X = scaler.transform(df[DETECTION_FEATURES].values)
    X = np.clip(X, -8.0, 8.0).astype(np.float32)
    labels = df["Label"].map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
    return X, labels


class _BaseFlowEnv(gym.Env):
    """Shared episode/windowing logic for all three agent envs."""

    metadata = {"render_modes": []}

    # Each row's features (including dport_repeat_count) are already fully
    # computed in src/data.py independent of visit order, so Detection and
    # Firewall don't need chronological replay -- only Response does, for its
    # within-episode repeat-offender counter. Subclasses that don't need
    # chronological order set balanced_sampling=True to draw each step's row
    # via class-balanced sampling instead of a contiguous windowed replay.
    balanced_sampling = False
    # Fraction of steps drawn uniformly-by-class (vs. the dataset's natural
    # distribution) when balanced_sampling is on. 1.0 would give every class
    # equal 1/N_CLASSES exposure regardless of rarity; mixing in some natural-
    # distribution draws keeps a realistic prior without starving BruteForce
    # (~1.1% of rows) of training signal the way pure chronological replay did.
    balance_fraction = 0.7

    def __init__(self, df, scaler: StandardScaler, episode_length: int = 2048, seed: int = 0):
        super().__init__()
        self.X, self.labels = _prep_arrays(df, scaler)
        self.n = len(df)
        self.episode_length = min(episode_length, self.n)
        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._ep_end = 0
        self._dport_counts = {}
        self._indices_by_class = [np.where(self.labels == c)[0] for c in range(N_CLASSES)]

    def _sample_index(self) -> int:
        nonempty_classes = [c for c in range(N_CLASSES) if len(self._indices_by_class[c]) > 0]
        if nonempty_classes and self._rng.random() < self.balance_fraction:
            cls = int(self._rng.choice(nonempty_classes))
            return int(self._rng.choice(self._indices_by_class[cls]))
        return int(self._rng.integers(0, self.n))

    def _start_episode(self):
        if self.balanced_sampling:
            # No notion of a contiguous window needed -- each step below
            # draws its own class-balanced index. Episode length only bounds
            # how many steps elapse before PPO truncates the rollout.
            self._idx = self._sample_index()
            self._steps_taken = 0
            self._dport_counts = {}
            return
        max_start = max(1, self.n - self.episode_length)
        nonempty_classes = [c for c in range(N_CLASSES) if len(self._indices_by_class[c]) > 0 and c != 0]
        if nonempty_classes and self._rng.random() < 0.6:
            cls = int(self._rng.choice(nonempty_classes))
            anchor = int(self._rng.choice(self._indices_by_class[cls]))
            offset = int(self._rng.integers(0, self.episode_length))
            self._idx = int(np.clip(anchor - offset, 0, max_start))
        else:
            self._idx = int(self._rng.integers(0, max_start))
        self._ep_end = self._idx + self.episode_length
        self._dport_counts = {}

    def _advance(self):
        """Move to the next row and report whether the episode is done."""
        if self.balanced_sampling:
            self._steps_taken += 1
            truncated = self._steps_taken >= self.episode_length
            self._idx = self._sample_index()
            return truncated
        self._idx += 1
        return self._idx >= self._ep_end

    def _episode_repeat_count(self, row_idx: int, dport_raw) -> int:
        self._dport_counts[dport_raw] = self._dport_counts.get(dport_raw, 0) + 1
        return self._dport_counts[dport_raw]


class DetectionEnv(_BaseFlowEnv):
    balanced_sampling = True

    def __init__(self, df, scaler, episode_length=2048, seed=0):
        super().__init__(df, scaler, episode_length, seed)
        self.action_space = spaces.Discrete(N_CLASSES)
        self.observation_space = spaces.Box(
            low=-8.0, high=8.0, shape=(len(DETECTION_FEATURES),), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._start_episode()
        return self.X[self._idx], {}

    def step(self, action):
        true_idx = self.labels[self._idx]
        reward = detection_reward(int(action), int(true_idx))
        info = {"true_idx": int(true_idx), "pred_idx": int(action)}
        terminated = False
        truncated = self._advance()
        obs = self.X[self._idx]
        return obs, reward, terminated, truncated, info


class FirewallEnv(_BaseFlowEnv):
    balanced_sampling = True

    def __init__(self, df, scaler, detection_predict_fn, episode_length=2048, seed=0):
        super().__init__(df, scaler, episode_length, seed)
        self.detection_predict_fn = detection_predict_fn
        self.action_space = spaces.Discrete(3)  # Allow, Rate-limit, Block
        self.observation_space = spaces.Box(
            low=-8.0, high=8.0, shape=(N_CLASSES + 2,), dtype=np.float32
        )

    def _build_obs(self, row_idx):
        det_action = int(self.detection_predict_fn(self.X[row_idx]))
        onehot = np.zeros(N_CLASSES, dtype=np.float32)
        onehot[det_action] = 1.0
        repeat = self.X[row_idx, DPORT_IDX]
        pkt_rate = self.X[row_idx, PKTS_IDX]
        return np.concatenate([onehot, [repeat, pkt_rate]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._start_episode()
        return self._build_obs(self._idx), {}

    def step(self, action):
        true_idx = self.labels[self._idx]
        reward = firewall_reward(int(action), int(true_idx))
        info = {"true_idx": int(true_idx), "fw_action": int(action)}
        terminated = False
        truncated = self._advance()
        obs = self._build_obs(self._idx)
        return obs, reward, terminated, truncated, info


class ResponseEnv(_BaseFlowEnv):
    def __init__(
        self,
        df,
        scaler,
        detection_predict_fn,
        firewall_predict_fn,
        episode_length=2048,
        seed=0,
    ):
        super().__init__(df, scaler, episode_length, seed)
        self.detection_predict_fn = detection_predict_fn
        self.firewall_predict_fn = firewall_predict_fn
        self._dport_raw = df["Destination Port"].values
        self.action_space = spaces.Discrete(3)  # No action, Alert, Isolate
        self.observation_space = spaces.Box(low=-8.0, high=8.0, shape=(3 + 1,), dtype=np.float32)

    def _fw_obs(self, row_idx):
        det_action = int(self.detection_predict_fn(self.X[row_idx]))
        onehot = np.zeros(N_CLASSES, dtype=np.float32)
        onehot[det_action] = 1.0
        repeat = self.X[row_idx, DPORT_IDX]
        pkt_rate = self.X[row_idx, PKTS_IDX]
        return np.concatenate([onehot, [repeat, pkt_rate]]).astype(np.float32)

    def _build_obs(self, row_idx):
        fw_action = int(self.firewall_predict_fn(self._fw_obs(row_idx)))
        onehot = np.zeros(3, dtype=np.float32)
        onehot[fw_action] = 1.0
        ep_count = self._episode_repeat_count(row_idx, self._dport_raw[row_idx])
        ep_count_scaled = min(ep_count, 20) / 20.0
        self._last_is_repeat = ep_count > REPEAT_OFFENDER_THRESHOLD
        return np.concatenate([onehot, [ep_count_scaled]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._start_episode()
        self._last_is_repeat = False
        return self._build_obs(self._idx), {}

    def step(self, action):
        true_idx = self.labels[self._idx]
        reward = response_reward(int(action), int(true_idx), self._last_is_repeat)
        info = {"true_idx": int(true_idx), "resp_action": int(action)}
        self._idx += 1
        terminated = False
        truncated = self._idx >= self._ep_end
        obs = self._build_obs(self._idx if not truncated else self._idx - 1)
        return obs, reward, terminated, truncated, info


def make_predict_fn(model):
    """Wrap a trained SB3 model as a frozen obs -> action function."""

    def predict(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return predict
