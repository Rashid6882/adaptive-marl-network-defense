"""
Closed-loop, trace-driven network-defense simulator (v2).

The v1 environments replay recorded flows independently of what the agents do,
so each step's reward depends only on that step (a contextual bandit). Here the
recorded trace is still the source of traffic, but the agents' actions change
what happens next:

  - Block / Isolate on an attack flow contains that attacker (identified by its
    attack class within the episode, since this CSV release has no Source IP):
    its later flows are dropped from the stream until the block expires (TTL).
  - Rate-limit throttles the attacker (damage x throttle_factor) until it expires.
  - Alert schedules an analyst containment after a delay (cheap, but slow).
  - Uncontained attack flows cause damage that escalates the longer the attack
    runs, so reaction time matters (latency).
  - Isolating a benign host causes an outage that disrupts the next benign flows.

Rewards are outcome-based (damage, disruption, operating cost) shared by all
three agents, plus a small role-specific shaping term (the v1 reward tables),
so Detection - whose output is only a message to Firewall - still gets signal.
Ground-truth labels are used only by the simulator to compute outcomes; the
agents never observe them.
"""

import os
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from src.baseline_rule import rule_based_firewall, rule_based_response
from src.data import chronological_split, load_clean_dataset
from src.env import DPORT_IDX, PKTS_IDX, _prep_arrays
from src.features import (
    DETECTION_FEATURES,
    N_CLASSES,
    REPEAT_OFFENDER_THRESHOLD,
    detection_reward,
    firewall_reward,
    response_reward,
)

CTX_DIM = 5
CACHE_PATH = "outputs/_clean_cache.pkl"
ROLE_IDX = {"detection": 0, "firewall": 1, "response": 2}


@dataclass
class SimConfig:
    max_steps: int = 1024
    anchor_prob: float = 0.7
    anchor_lead: int = 300
    anchor_class_probs: tuple = (1 / 3, 1 / 3, 1 / 3)  # DDoS, PortScan, BruteForce
    damage_base: tuple = (0.0, 1.0, 0.5, 1.5)  # indexed by true class
    age_scale: float = 60.0
    age_cap: float = 2.0
    throttle_factor: float = 0.3
    ttl_block: int = 200
    ttl_throttle: int = 120
    ttl_isolate: int = 400
    ttl_alert: int = 300
    alert_delay: int = 40
    onset_gap: int = 300
    disruption_block: float = 1.0
    disruption_rl: float = 0.3
    disruption_isolate: float = 3.0
    outage_len: int = 15
    outage_cost: float = 0.5
    isolate_cost: float = 0.3
    alert_cost: float = 0.05
    false_alert_cost: float = 0.1
    shaping: float = 0.5


def load_split_cached():
    """Cleaned train/test split with only the columns the sim needs, cached on
    disk so every training stage doesn't redo the slow cleaning pass."""
    cols = DETECTION_FEATURES + ["Label", "source_file"]
    if os.path.exists(CACHE_PATH):
        try:
            tr, te = pd.read_pickle(CACHE_PATH)
            if all(c in tr.columns for c in cols):
                return tr, te
        except Exception:
            pass
    df = load_clean_dataset()
    tr, te = chronological_split(df)
    tr, te = tr[cols].copy(), te[cols].copy()
    del df
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    pd.to_pickle((tr, te), CACHE_PATH)
    return tr, te


def rule_detect_array(df):
    """Vectorized equivalent of baseline_rule.rule_based_detect (DDoS > PortScan > BruteForce)."""
    pkt = df["Flow Packets/s"].values
    dur = df["Flow Duration"].values
    syn = df["SYN Flag Count"].values
    dport = df["Destination Port"].values
    rep = df["dport_repeat_count"].values
    fwd = df["Total Fwd Packets"].values
    out = np.zeros(len(df), dtype=np.int64)
    out[np.isin(dport, (21, 22)) & (rep >= 5) & (dur < 2_000_000)] = 3
    out[(syn >= 1) & (dur < 1000) & (rep >= 10)] = 2
    out[(pkt > 2000) & (fwd > 10)] = 1
    return out


def make_fast_predict(model):
    """Deterministic obs -> action for a frozen SB3 policy, without predict() overhead."""
    policy = model.policy
    policy.set_training_mode(False)

    @torch.no_grad()
    def predict(obs):
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(policy._predict(t, deterministic=True).item())

    return predict


class DefenseSim:
    def __init__(self, df, scaler, cfg=None, seed=0):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.X, self.labels = _prep_arrays(df, scaler)
        self.n = len(df)
        self.dport = df["Destination Port"].values
        self.rule_det = rule_detect_array(df)
        files = df["source_file"].values
        change = np.flatnonzero(files[1:] != files[:-1]) + 1
        starts = np.concatenate([[0], change]).astype(np.int64)
        ends = np.concatenate([change, [self.n]]).astype(np.int64)
        self.file_starts = starts
        self.file_start = np.repeat(starts, ends - starts)
        self.file_end = np.repeat(ends, ends - starts)
        self._attack_idx = [np.flatnonzero(self.labels == k) for k in range(1, N_CLASSES)]
        self.done = True
        self.reset_stats()

    def reset_stats(self):
        self.s = dict(
            damage=0.0, disruption=0.0, cost=0.0, steps=0, dropped=0,
            attack_presented=0, benign_presented=0, benign_disrupted=0,
            latencies=[], uncontained=0,
        )

    def open_onsets(self):
        return sum(1 for o in self.onset if o >= 0)

    # ---- episode control -------------------------------------------------
    def _sample_start(self):
        c = self.cfg
        if self.rng.random() < c.anchor_prob:
            idxs = self._attack_idx[int(self.rng.choice(N_CLASSES - 1, p=c.anchor_class_probs))]
            if len(idxs):
                anchor = int(self.rng.choice(idxs))
                start = anchor - int(self.rng.integers(0, c.anchor_lead))
                return max(start, int(self.file_start[anchor]))
        return int(self.rng.integers(0, self.n))

    def reset(self, start=None, max_steps="default"):
        c = self.cfg
        limit = c.max_steps if max_steps == "default" else max_steps
        for _ in range(20):
            s = self._sample_start() if start is None else int(start)
            self._init_episode(s, limit)
            if not self.done:
                break

    def _init_episode(self, start, limit):
        K = N_CLASSES
        self.pos = start
        self.end = int(self.file_end[start])
        self.limit = limit
        self.steps = 0
        self.block_until = [-1] * K
        self.throttle_until = [-1] * K
        self.alert_at = [-1] * K
        self.onset = [-1] * K
        self.last_attack_pos = [-10**9] * K
        self.age = [0.0] * K
        self.outage_left = 0
        self.ema_damage = 0.0
        self.fw_hist = deque([0] * 20, maxlen=20)
        self.det_hist = deque([0] * 20, maxlen=20)
        self.dport_counts = {}
        self.cur_count = 0
        self.cur_repeat = False
        self.done = False
        self._settle()

    def _contain(self, k, ttl):
        self.block_until[k] = self.pos + ttl
        self.throttle_until[k] = -1
        self.alert_at[k] = -1
        self.age[k] = 0.0
        if self.onset[k] >= 0:
            self.s["latencies"].append(self.pos - self.onset[k])
            self.onset[k] = -1

    def _settle(self):
        """Advance to the next flow the agents must decide on, applying alert
        maturation and dropping flows from already-contained attackers."""
        c = self.cfg
        while True:
            if self.pos >= self.end or (self.limit is not None and self.steps >= self.limit):
                self.done = True
                return
            for k in range(1, N_CLASSES):
                if self.alert_at[k] >= 0 and self.pos >= self.alert_at[k]:
                    self._contain(k, c.ttl_alert)
            lab = int(self.labels[self.pos])
            if lab != 0 and self.block_until[lab] > self.pos:
                self.s["dropped"] += 1
                self.pos += 1
                continue
            d = self.dport[self.pos]
            self.cur_count = self.dport_counts.get(d, 0) + 1
            self.dport_counts[d] = self.cur_count
            self.cur_repeat = self.cur_count > REPEAT_OFFENDER_THRESHOLD
            return

    def ctx(self):
        return np.array(
            [
                min(self.ema_damage, 1.0),
                sum(self.fw_hist) / 20.0,
                1.0 if self.outage_left > 0 else 0.0,
                1.0 if max(self.alert_at) >= 0 else 0.0,
                sum(self.det_hist) / 20.0,
            ],
            dtype=np.float32,
        )

    # ---- one joint decision ---------------------------------------------
    def apply(self, det_a, fw_a, resp_a):
        c = self.cfg
        pos = self.pos
        true = int(self.labels[pos])
        damage = disruption = cost = 0.0

        if true != 0:
            self.s["attack_presented"] += 1
            if self.onset[true] < 0 or pos - self.last_attack_pos[true] > c.onset_gap:
                if self.onset[true] >= 0:
                    self.s["uncontained"] += 1
                self.onset[true] = pos
            self.last_attack_pos[true] = pos
            base = c.damage_base[true] * (1.0 + min(self.age[true] / c.age_scale, c.age_cap))
            factor = c.throttle_factor if self.throttle_until[true] > pos else 1.0
            if fw_a == 2 or resp_a == 2:
                factor = 0.0
                self._contain(true, c.ttl_isolate if resp_a == 2 else c.ttl_block)
            else:
                if fw_a == 1:
                    factor = min(factor, c.throttle_factor)
                    self.throttle_until[true] = pos + c.ttl_throttle
                if resp_a == 1 and self.alert_at[true] < 0:
                    self.alert_at[true] = pos + c.alert_delay
                self.age[true] += factor
            damage = base * factor
        else:
            self.s["benign_presented"] += 1
            if fw_a == 2:
                disruption += c.disruption_block
            elif fw_a == 1:
                disruption += c.disruption_rl
            if resp_a == 2:
                disruption += c.disruption_isolate
                self.outage_left = c.outage_len
            elif self.outage_left > 0:
                disruption += c.outage_cost
                self.outage_left -= 1
            if disruption > 0:
                self.s["benign_disrupted"] += 1
            if resp_a == 1:
                cost += c.false_alert_cost

        if resp_a == 1:
            cost += c.alert_cost
        elif resp_a == 2:
            cost += c.isolate_cost

        outcome = -(damage + disruption + cost)
        shape_det = c.shaping * detection_reward(det_a, true)
        shape_fw = c.shaping * firewall_reward(fw_a, true)
        shape_resp = c.shaping * response_reward(resp_a, true, self.cur_repeat)

        self.s["damage"] += damage
        self.s["disruption"] += disruption
        self.s["cost"] += cost
        self.s["steps"] += 1

        self.ema_damage = 0.9 * self.ema_damage + 0.1 * min(damage, 3.0)
        self.fw_hist.append(1 if fw_a == 2 else 0)
        self.det_hist.append(1 if det_a != 0 else 0)
        self.steps += 1
        self.pos += 1
        self._settle()
        return outcome, shape_det, shape_fw, shape_resp, true


# ---- observation builders (shared by training envs and evaluation) --------
def det_obs(sim):
    return np.concatenate([sim.X[sim.pos], sim.ctx()]).astype(np.float32)


def fw_obs(sim, det_a):
    onehot = np.zeros(N_CLASSES, dtype=np.float32)
    onehot[det_a] = 1.0
    extra = [sim.X[sim.pos, DPORT_IDX], sim.X[sim.pos, PKTS_IDX]]
    return np.concatenate([onehot, extra, sim.ctx()]).astype(np.float32)


def resp_obs(sim, fw_a):
    onehot = np.zeros(3, dtype=np.float32)
    onehot[fw_a] = 1.0
    return np.concatenate([onehot, [min(sim.cur_count, 20) / 20.0], sim.ctx()]).astype(np.float32)


def single_obs(sim):
    return np.concatenate(
        [sim.X[sim.pos], [min(sim.cur_count, 20) / 20.0], sim.ctx()]
    ).astype(np.float32)


OBS_DIM = {
    0: len(DETECTION_FEATURES) + CTX_DIM,
    1: N_CLASSES + 2 + CTX_DIM,
    2: 3 + 1 + CTX_DIM,
}
SINGLE_OBS_DIM = len(DETECTION_FEATURES) + 1 + CTX_DIM


class RoleEnv(gym.Env):
    """Trains one agent while the other two are frozen partners (a learned
    policy callable, or None for the rule-based fallback)."""

    metadata = {"render_modes": []}

    def __init__(self, df, scaler, role, partners, cfg=None, seed=0):
        super().__init__()
        self.sim = DefenseSim(df, scaler, cfg, seed)
        self.role = role
        self.partners = partners
        self.action_space = spaces.Discrete(N_CLASSES if role == 0 else 3)
        self.observation_space = spaces.Box(-8.0, 8.0, (OBS_DIM[role],), np.float32)
        self._obs = np.zeros(OBS_DIM[role], dtype=np.float32)

    def _partner_det(self):
        p = self.partners.get("det")
        return int(self.sim.rule_det[self.sim.pos]) if p is None else p(det_obs(self.sim))

    def _partner_fw(self, det_a):
        p = self.partners.get("fw")
        return rule_based_firewall(det_a) if p is None else p(fw_obs(self.sim, det_a))

    def _partner_resp(self, fw_a):
        p = self.partners.get("resp")
        if p is None:
            return rule_based_response(fw_a, self.sim.cur_count, REPEAT_OFFENDER_THRESHOLD)
        return p(resp_obs(self.sim, fw_a))

    def _prepare(self):
        if self.role == 0:
            return det_obs(self.sim)
        self._det_a = self._partner_det()
        if self.role == 1:
            return fw_obs(self.sim, self._det_a)
        self._fw_a = self._partner_fw(self._det_a)
        return resp_obs(self.sim, self._fw_a)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self._obs = self._prepare()
        return self._obs, {}

    def step(self, action):
        a = int(action)
        if self.role == 0:
            det_a = a
            fw_a = self._partner_fw(det_a)
            resp_a = self._partner_resp(fw_a)
        elif self.role == 1:
            det_a, fw_a = self._det_a, a
            resp_a = self._partner_resp(fw_a)
        else:
            det_a, fw_a, resp_a = self._det_a, self._fw_a, a
        outcome, sd, sf, sr, true = self.sim.apply(det_a, fw_a, resp_a)
        reward = outcome + (sd, sf, sr)[self.role]
        done = self.sim.done
        if not done:
            self._obs = self._prepare()
        return self._obs, reward, False, done, {"true_idx": true}


class SingleSimEnv(gym.Env):
    """Single-agent baseline in the same simulator: one policy picks the joint
    (detection, firewall, response) action, with the same information and the
    same outcome reward the three MARL agents collectively get."""

    metadata = {"render_modes": []}

    def __init__(self, df, scaler, cfg=None, seed=0):
        super().__init__()
        from src.single_agent_env import N_ACTIONS

        self.sim = DefenseSim(df, scaler, cfg, seed)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-8.0, 8.0, (SINGLE_OBS_DIM,), np.float32)
        self._obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self._obs = single_obs(self.sim)
        return self._obs, {}

    def step(self, action):
        from src.single_agent_env import decode_action

        det_a, fw_a, resp_a = decode_action(int(action))
        outcome, sd, sf, sr, true = self.sim.apply(det_a, fw_a, resp_a)
        reward = outcome + sd + sf + sr
        done = self.sim.done
        if not done:
            self._obs = single_obs(self.sim)
        return self._obs, reward, False, done, {"true_idx": true}
