"""
Single-agent RL baseline (Section 12): one PPO agent handles detection,
blocking, and escalation together, with no role separation or coordination.
Used to isolate what the multi-agent design actually buys over a monolithic
policy. Action space is the flattened product of all three agents' actions
(4 x 3 x 3 = 36); reward is the sum of the three role-specific reward
functions evaluated on the decoded sub-actions, using the same state the
Detection agent sees (this agent must infer firewall/response-relevant
context, such as "is this a repeat offender", from the same raw input, since
it does not get to condition on anyone else's action).
"""

import numpy as np
from gymnasium import spaces

from src.env import DPORT_IDX, _BaseFlowEnv
from src.features import (
    N_CLASSES,
    REPEAT_OFFENDER_THRESHOLD,
    detection_reward,
    firewall_reward,
    response_reward,
)

N_FW = 3
N_RESP = 3
N_ACTIONS = N_CLASSES * N_FW * N_RESP


def decode_action(a: int):
    det = a // (N_FW * N_RESP)
    rem = a % (N_FW * N_RESP)
    fw = rem // N_RESP
    resp = rem % N_RESP
    return det, fw, resp


class SingleAgentEnv(_BaseFlowEnv):
    def __init__(self, df, scaler, episode_length=2048, seed=0):
        super().__init__(df, scaler, episode_length, seed)
        self._dport_raw = df["Destination Port"].values
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-8.0, high=8.0, shape=(self.X.shape[1],), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._start_episode()
        return self.X[self._idx], {}

    def step(self, action):
        det, fw, resp = decode_action(int(action))
        true_idx = int(self.labels[self._idx])
        ep_count = self._episode_repeat_count(self._idx, self._dport_raw[self._idx])
        is_repeat = ep_count > REPEAT_OFFENDER_THRESHOLD

        reward = (
            detection_reward(det, true_idx)
            + firewall_reward(fw, true_idx)
            + response_reward(resp, true_idx, is_repeat)
        )
        info = {"true_idx": true_idx, "det": det, "fw": fw, "resp": resp}
        self._idx += 1
        terminated = False
        truncated = self._idx >= self._ep_end
        obs = self.X[self._idx] if not truncated else self.X[self._idx - 1]
        return obs, reward, terminated, truncated, info
