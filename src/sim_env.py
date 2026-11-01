# Module docstring title: a closed-loop simulator that replays recorded network traffic.
# Why it exists: replaying traffic without reacting to the agents would make each step independent (a
# contextual bandit); here the agents' actions change what happens next.
# Block or Isolate on an attack flow contains that attacker (identified by its attack class, since there is no
# Source IP); its later flows are dropped until the block expires.
# Rate-limit throttles the attacker (damage multiplied by a throttle factor) until it expires.
# Alert makes an analyst contain the attacker after a delay (cheap but slow).
# Attack flows that get through cause damage that grows the longer the attack runs, so reaction time matters.
# Isolating a benign host causes an outage that disrupts the next benign flows.
# Rewards are outcomes (damage, disruption, cost) shared by all three agents, plus a small role-specific
# shaping term so Detection also gets a learning signal.
# Labels are used only inside the simulator to compute outcomes; agents never observe them.
# Ends the docstring.
"""
Closed-loop, trace-driven network-defense simulator.

Replaying recorded flows independently of what the agents do would make each
step's reward depend only on that step (a contextual bandit). Here the recorded
trace is still the source of traffic, but the agents' actions change what
happens next:

  - Block / Isolate on an attack flow contains that attacker (identified by its
    attack class within the episode, since this CSV release has no Source IP):
    its later flows are dropped from the stream until the block expires (TTL).
  - Rate-limit throttles the attacker (damage x throttle_factor) until it expires.
  - Alert schedules an analyst containment after a delay (cheap, but slow).
  - Uncontained attack flows cause damage that escalates the longer the attack
    runs, so reaction time matters (latency).
  - Isolating a benign host causes an outage that disrupts the next benign flows.

Rewards are outcome-based (damage, disruption, operating cost) shared by all
three agents, plus a small role-specific shaping term (the reward tables in features.py),
so Detection - whose output is only a message to Firewall - still gets signal.
Ground-truth labels are used only by the simulator to compute outcomes; the
agents never observe them.
"""

# Imports os for file paths and folder creation.
import os
# Imports deque (fixed-length history of recent decisions).
from collections import deque
# Imports dataclass, used to define the settings class.
from dataclasses import dataclass

# Imports gymnasium, the standard environment interface used by Stable-Baselines3.
import gymnasium as gym
# Imports numpy.
import numpy as np
# Imports pandas.
import pandas as pd
# Imports PyTorch, used to run trained policies quickly.
import torch
# Imports gymnasium's space types (Discrete, Box).
from gymnasium import spaces

# Imports the rule-based Firewall and Response rules (used as default partner agents).
from src.baseline_rule import rule_based_firewall, rule_based_response
# Imports the data loading and chronological split functions.
from src.data import chronological_split, load_clean_dataset
# Imports the scaler helpers and the two column positions the Firewall reads.
from src.env import DPORT_IDX, PKTS_IDX, _prep_arrays
# Imports class count, feature list, repeat-offender threshold and the three label-based shaping reward
# functions.
from src.features import (
    DETECTION_FEATURES,
    N_CLASSES,
    REPEAT_OFFENDER_THRESHOLD,
    detection_reward,
    firewall_reward,
    response_reward,
)

# Number of extra "context" inputs each agent receives (5).
CTX_DIM = 5
# Where the cleaned train/test data is cached on disk.
CACHE_PATH = "outputs/_clean_cache.pkl"
# Maps role names to numbers (not used elsewhere in this file).
ROLE_IDX = {"detection": 0, "firewall": 1, "response": 2}


# Decorator: automatically creates the constructor for the settings class below.
@dataclass
# Settings class holding every simulator constant.
class SimConfig:
    # Episode length: 1,024 flows.
    max_steps: int = 1024
    # Probability (0.7) that an episode starts just before an attack.
    anchor_prob: float = 0.7
    # The start is up to 300 flows before the chosen attack flow.
    anchor_lead: int = 300
    # Which attack class anchors an episode: equal chance for DDoS, PortScan and BruteForce.
    anchor_class_probs: tuple = (1 / 3, 1 / 3, 1 / 3)  # DDoS, PortScan, BruteForce
    # Base damage per attack flow by true class: benign 0, DDoS 1.0, PortScan 0.5, BruteForce 1.5.
    damage_base: tuple = (0.0, 1.0, 0.5, 1.5)  # indexed by true class
    # Damage grows with attack age; this sets how many flows are needed to add one unit of escalation (60).
    age_scale: float = 60.0
    # Maximum escalation: damage can grow to at most 3 times the base (1 + 2).
    age_cap: float = 2.0
    # Throttled attackers cause only 30% of normal damage.
    throttle_factor: float = 0.3
    # A Block lasts 200 flows.
    ttl_block: int = 200
    # A rate-limit lasts 120 flows.
    ttl_throttle: int = 120
    # An Isolate lasts 400 flows.
    ttl_isolate: int = 400
    # Containment started by an alert lasts 300 flows.
    ttl_alert: int = 300
    # An alert takes 40 flows before the analyst acts.
    alert_delay: int = 40
    # If 300 flows pass with no attack of a class, the next one counts as a new attack onset.
    onset_gap: int = 300
    # Benign disruption cost for a Block: 1.0.
    disruption_block: float = 1.0
    # Benign disruption cost for a rate-limit: 0.3.
    disruption_rl: float = 0.3
    # Benign disruption cost for an Isolate: 3.0.
    disruption_isolate: float = 3.0
    # An isolation causes a 15-flow outage for benign traffic.
    outage_len: int = 15
    # Each benign flow hit by the outage costs 0.5.
    outage_cost: float = 0.5
    # Operating cost of an Isolate: 0.3.
    isolate_cost: float = 0.3
    # Operating cost of an Alert: 0.05.
    alert_cost: float = 0.05
    # Extra cost for an Alert on a benign flow: 0.1.
    false_alert_cost: float = 0.1
    # Weight of the shaping reward (0.5).
    shaping: float = 0.5


# Function that returns the cleaned train and test tables (using a cache).
def load_split_cached():
    # Docstring: keeps only needed columns and caches them so the slow cleaning is done once.
    """Cleaned train/test split with only the columns the sim needs, cached on
    disk so every training stage doesn't redo the slow cleaning pass."""
    # Columns the simulator needs: the 21 features, the label and the source file.
    cols = DETECTION_FEATURES + ["Label", "source_file"]
    # Checks whether the cache file exists.
    if os.path.exists(CACHE_PATH):
        # Starts a try block in case the cache is unreadable.
        try:
            # Loads the cached train and test tables.
            tr, te = pd.read_pickle(CACHE_PATH)
            # Checks that all needed columns are present.
            if all(c in tr.columns for c in cols):
                # If so, returns the cached data.
                return tr, te
        # If reading failed...
        except Exception:
            # ...ignore the error and rebuild the data.
            pass
    # Loads and cleans the full dataset (slow).
    df = load_clean_dataset()
    # Splits it chronologically into train and test.
    tr, te = chronological_split(df)
    # Keeps only the needed columns in each part.
    tr, te = tr[cols].copy(), te[cols].copy()
    # Frees the large full table from memory.
    del df
    # Creates the cache folder if missing.
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    # Saves the two tables to the cache.
    pd.to_pickle((tr, te), CACHE_PATH)
    # Returns the train and test tables.
    return tr, te


# Function that returns train, validation and test tables.
def load_splits_cached(val_share=0.125):
    # Docstring: per file the first 70% is train, the next 10% is validation and the last 20% is test;
    # validation is taken from the end of the training part so tuning never touches test flows.
    """Chronological train / validation / test: per file, the first 70% / next 10% / last 20%.
    The test part is identical to load_split_cached()'s test split; validation is carved
    from the end of its training part so tuning never touches test flows."""
    # Gets the train (80%) and test (20%) tables.
    tr, te = load_split_cached()
    # Two lists for the kept training pieces and the validation pieces.
    keep, val = [], []
    # Loops over each file's rows.
    for _, g in tr.groupby("source_file", sort=False):
        # Cut position: keep 87.5% of the 80% (which is 70% of the file).
        cut = int(len(g) * (1 - val_share))
        # Earlier rows stay as training.
        keep.append(g.iloc[:cut])
        # Later rows become validation (10% of the file).
        val.append(g.iloc[cut:])
    # Starts the return value (a tuple of three tables).
    return (
        # Training table, rows renumbered.
        pd.concat(keep, axis=0).reset_index(drop=True),
        # Validation table, rows renumbered.
        pd.concat(val, axis=0).reset_index(drop=True),
        # Test table, unchanged.
        te,
    # Closes the tuple.
    )


# Default thresholds of the rule-based baseline: DDoS packet rate 2000 and forward packets 10, PortScan
# duration 1000 and repeat 10, BruteForce repeat 5 and duration 2,000,000, Response threshold 5.
DEFAULT_RULE_PARAMS = dict(
    ddos_pkt=2000.0, ddos_fwd=10, ps_dur=1000.0, ps_rep=10, bf_rep=5, bf_dur=2_000_000.0, resp_thr=5,
)


# Function that applies the rule-based detector to all rows at once (fast, vectorized).
def rule_detect_array(df, params=None):
    # Docstring: same rules as baseline_rule.py with priority DDoS > PortScan > BruteForce; thresholds can be
    # overridden for tuning.
    """Vectorized equivalent of baseline_rule.rule_based_detect (DDoS > PortScan > BruteForce).
    `params` overrides the default thresholds (used to tune the rule baseline)."""
    # Merges default thresholds with any overrides.
    p = {**DEFAULT_RULE_PARAMS, **(params or {})}
    # Packet rate column as an array.
    pkt = df["Flow Packets/s"].values
    # Duration column.
    dur = df["Flow Duration"].values
    # SYN flag count column.
    syn = df["SYN Flag Count"].values
    # Destination port column.
    dport = df["Destination Port"].values
    # Repeat count column.
    rep = df["dport_repeat_count"].values
    # Forward packet count column.
    fwd = df["Total Fwd Packets"].values
    # Starts with every flow classed as benign (0).
    out = np.zeros(len(df), dtype=np.int64)
    # Sets class 3 (BruteForce) where port is 21/22, repeat count and duration rules match.
    out[np.isin(dport, (21, 22)) & (rep >= p["bf_rep"]) & (dur < p["bf_dur"])] = 3
    # Sets class 2 (PortScan) where SYN, short-duration and repeat rules match (overrides BruteForce).
    out[(syn >= 1) & (dur < p["ps_dur"]) & (rep >= p["ps_rep"])] = 2
    # Sets class 1 (DDoS) where high packet rate and forward packets match (highest priority).
    out[(pkt > p["ddos_pkt"]) & (fwd > p["ddos_fwd"])] = 1
    # Returns the class number for every flow.
    return out


# Function that wraps a trained PPO model into a fast "observation to action" function.
def make_fast_predict(model):
    # Docstring.
    """Deterministic obs -> action for a frozen SB3 policy, without predict() overhead."""
    # Takes the neural network policy out of the model.
    policy = model.policy
    # Puts it in evaluation mode (no training behaviour).
    policy.set_training_mode(False)

    # Decorator: turn off gradient tracking for speed.
    @torch.no_grad()
    # The fast prediction function.
    def predict(obs):
        # Converts the observation to a tensor with a batch dimension of 1.
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        # Gets the most likely action (deterministic) and returns it as an int.
        return int(policy._predict(t, deterministic=True).item())

    # Returns the prediction function.
    return predict


# The simulator class.
class DefenseSim:
    # Constructor: takes a data table, a scaler, optional settings and a random seed.
    def __init__(self, df, scaler, cfg=None, seed=0):
        # Uses the given settings or the defaults.
        self.cfg = cfg or SimConfig()
        # Random generator seeded for repeatability.
        self.rng = np.random.default_rng(seed)
        # Builds the scaled feature array and the label array.
        self.X, self.labels = _prep_arrays(df, scaler)
        # Number of flows.
        self.n = len(df)
        # Destination port of every flow.
        self.dport = df["Destination Port"].values
        # Pre-computes the rule-based detection of every flow (for the rule partner).
        self.rule_det = rule_detect_array(df)
        # Which source file each flow came from.
        files = df["source_file"].values
        # Positions where the file name changes (file boundaries).
        change = np.flatnonzero(files[1:] != files[:-1]) + 1
        # Start positions of each file (0 plus the change points).
        starts = np.concatenate([[0], change]).astype(np.int64)
        # End positions of each file.
        ends = np.concatenate([change, [self.n]]).astype(np.int64)
        # Stores the file start positions (used to run evaluation file by file).
        self.file_starts = starts
        # For every flow, the start of its own file.
        self.file_start = np.repeat(starts, ends - starts)
        # For every flow, the end of its own file.
        self.file_end = np.repeat(ends, ends - starts)
        # For each attack class, the list of positions of its attack flows (used to start episodes near
        # attacks).
        self._attack_idx = [np.flatnonzero(self.labels == k) for k in range(1, N_CLASSES)]
        # Marks the simulator as "done" until reset is called.
        self.done = True
        # Creates the statistics counters.
        self.reset_stats()

    # Method that resets the counters.
    def reset_stats(self):
        # Dictionary of running totals: damage, disruption, cost, steps, dropped flows, attack and benign
        # flows presented, benign flows disrupted, containment latencies and uncontained attacks.
        self.s = dict(
            damage=0.0, disruption=0.0, cost=0.0, steps=0, dropped=0,
            attack_presented=0, benign_presented=0, benign_disrupted=0,
            latencies=[], uncontained=0,
        )

    # Method counting attacks that started but were never contained.
    def open_onsets(self):
        # Counts onset entries that are still set (>= 0).
        return sum(1 for o in self.onset if o >= 0)

    # ---- episode control -------------------------------------------------
    # Picks where an episode starts.
    def _sample_start(self):
        # Short name for the settings.
        c = self.cfg
        # With probability 0.7 (anchor_prob) start near an attack.
        if self.rng.random() < c.anchor_prob:
            # Picks an attack class at random and gets its flow positions.
            idxs = self._attack_idx[int(self.rng.choice(N_CLASSES - 1, p=c.anchor_class_probs))]
            # Only if that class has flows.
            if len(idxs):
                # Picks a random attack flow as the anchor.
                anchor = int(self.rng.choice(idxs))
                # Moves the start back by a random 0-299 flows.
                start = anchor - int(self.rng.integers(0, c.anchor_lead))
                # Returns the start, but not before the beginning of that file.
                return max(start, int(self.file_start[anchor]))
        # Otherwise returns a random start anywhere in the data.
        return int(self.rng.integers(0, self.n))

    # Method that starts a new episode (optionally from a given start and length).
    def reset(self, start=None, max_steps="default"):
        # Short name for the settings.
        c = self.cfg
        # Episode length limit: the default, or the value given (None means run to the end of the file).
        limit = c.max_steps if max_steps == "default" else max_steps
        # Tries up to 20 times to get a valid episode.
        for _ in range(20):
            # Uses a random start or the one provided.
            s = self._sample_start() if start is None else int(start)
            # Sets up the episode state.
            self._init_episode(s, limit)
            # Checks whether the episode is already finished (no usable flow).
            if not self.done:
                # If it is usable, stop retrying.
                break

    # Method that sets up all episode state.
    def _init_episode(self, start, limit):
        # Number of classes.
        K = N_CLASSES
        # Current position in the data.
        self.pos = start
        # Last flow of this file (episodes never cross files).
        self.end = int(self.file_end[start])
        # Episode length limit.
        self.limit = limit
        # Number of decisions made so far.
        self.steps = 0
        # For each class, the position until which that attacker is blocked.
        self.block_until = [-1] * K
        # For each class, until when it is throttled.
        self.throttle_until = [-1] * K
        # For each class, when a pending alert will trigger containment.
        self.alert_at = [-1] * K
        # For each class, where the current attack began.
        self.onset = [-1] * K
        # For each class, position of its last attack flow.
        self.last_attack_pos = [-10**9] * K
        # For each class, the "age" of the attack (grows when it is not stopped).
        self.age = [0.0] * K
        # Remaining benign flows hit by an isolation outage.
        self.outage_left = 0
        # Exponential moving average of recent damage (used as context).
        self.ema_damage = 0.0
        # History of the last 20 Firewall decisions (1 if Block).
        self.fw_hist = deque([0] * 20, maxlen=20)
        # History of the last 20 Detection decisions (1 if flagged as attack).
        self.det_hist = deque([0] * 20, maxlen=20)
        # Dictionary counting destination ports seen in this episode.
        self.dport_counts = {}
        # Repeat count of the current flow's port.
        self.cur_count = 0
        # Whether the current flow's port is a repeat offender.
        self.cur_repeat = False
        # Marks the episode as running.
        self.done = False
        # Moves to the first flow that needs a decision.
        self._settle()

    # Method that contains an attacker class k for ttl flows.
    def _contain(self, k, ttl):
        # Blocks the attacker until the current position plus ttl.
        self.block_until[k] = self.pos + ttl
        # Cancels any throttle.
        self.throttle_until[k] = -1
        # Cancels any pending alert.
        self.alert_at[k] = -1
        # Resets the attack's age to zero.
        self.age[k] = 0.0
        # Checks whether an attack onset was open.
        if self.onset[k] >= 0:
            # Records the containment latency: flows between attack start and containment.
            self.s["latencies"].append(self.pos - self.onset[k])
            # Clears the onset.
            self.onset[k] = -1

    # Method that moves to the next flow the agents must decide on.
    def _settle(self):
        # Docstring: applies matured alerts and skips flows from already-contained attackers.
        """Advance to the next flow the agents must decide on, applying alert
        maturation and dropping flows from already-contained attackers."""
        # Short name for the settings.
        c = self.cfg
        # Loops until a decidable flow is found.
        while True:
            # Checks whether the data or the episode length limit has been reached.
            if self.pos >= self.end or (self.limit is not None and self.steps >= self.limit):
                # Marks the episode as finished.
                self.done = True
                # Stops.
                return
            # Loops over each attack class.
            for k in range(1, N_CLASSES):
                # Checks whether an alert is pending and its delay has passed.
                if self.alert_at[k] >= 0 and self.pos >= self.alert_at[k]:
                    # If so, the analyst contains that attacker for ttl_alert flows.
                    self._contain(k, c.ttl_alert)
            # True class of the current flow.
            lab = int(self.labels[self.pos])
            # If the flow is an attack from a currently blocked attacker...
            if lab != 0 and self.block_until[lab] > self.pos:
                # ...count it as dropped (never reaches the agents)...
                self.s["dropped"] += 1
                # ...move to the next flow...
                self.pos += 1
                # ...and keep looping.
                continue
            # Destination port of the current flow.
            d = self.dport[self.pos]
            # Updates the repeat count for that port (previous count plus one).
            self.cur_count = self.dport_counts.get(d, 0) + 1
            # Stores the count.
            self.dport_counts[d] = self.cur_count
            # Marks the flow as a repeat offender if the count exceeds the threshold.
            self.cur_repeat = self.cur_count > REPEAT_OFFENDER_THRESHOLD
            # Returns: the current flow now needs a decision.
            return

    # Method that builds the 5 context inputs shared by all agents.
    def ctx(self):
        # Builds a numpy array from the list below.
        return np.array(
            # Opens the list of the five context values.
            [
                # Recent damage level (capped at 1).
                min(self.ema_damage, 1.0),
                # Fraction of the last 20 flows that the Firewall blocked.
                sum(self.fw_hist) / 20.0,
                # 1 if a benign outage is active, else 0.
                1.0 if self.outage_left > 0 else 0.0,
                # 1 if an alert is pending, else 0.
                1.0 if max(self.alert_at) >= 0 else 0.0,
                # Fraction of the last 20 flows that Detection flagged as attacks.
                sum(self.det_hist) / 20.0,
            # Ends the list.
            ],
            # Uses 32-bit floats.
            dtype=np.float32,
        # Ends the call.
        )

    # ---- one joint decision ---------------------------------------------
    # Method that applies one joint decision (detection, firewall, response) to the current flow and returns
    # the rewards.
    def apply(self, det_a, fw_a, resp_a):
        # Short name for the settings.
        c = self.cfg
        # Current position.
        pos = self.pos
        # True class of the current flow (used only for outcomes).
        true = int(self.labels[pos])
        # Starts damage, disruption and cost at zero.
        damage = disruption = cost = 0.0

        # Branch: the flow is an attack.
        if true != 0:
            # Counts attack flows presented to the agents.
            self.s["attack_presented"] += 1
            # Checks whether this starts a new attack: no open onset, or the last attack flow was too long
            # ago.
            if self.onset[true] < 0 or pos - self.last_attack_pos[true] > c.onset_gap:
                # If an older onset was still open...
                if self.onset[true] >= 0:
                    # ...count it as an uncontained attack.
                    self.s["uncontained"] += 1
                # Records this flow as the new onset.
                self.onset[true] = pos
            # Remembers this attack flow's position.
            self.last_attack_pos[true] = pos
            # Base damage grows with the attack's age, capped at 3 times the base.
            base = c.damage_base[true] * (1.0 + min(self.age[true] / c.age_scale, c.age_cap))
            # Damage factor is the throttle factor if the attacker is throttled, else 1.
            factor = c.throttle_factor if self.throttle_until[true] > pos else 1.0
            # Checks whether the Firewall blocked or Response isolated.
            if fw_a == 2 or resp_a == 2:
                # Then the flow causes no damage.
                factor = 0.0
                # Contains the attacker for the isolate duration if isolated, otherwise the block duration.
                self._contain(true, c.ttl_isolate if resp_a == 2 else c.ttl_block)
            # Otherwise (not blocked or isolated):
            else:
                # If the Firewall rate-limited...
                if fw_a == 1:
                    # ...damage is reduced to the throttle factor...
                    factor = min(factor, c.throttle_factor)
                    # ...and the throttle lasts for ttl_throttle flows.
                    self.throttle_until[true] = pos + c.ttl_throttle
                # If Response alerted and no alert is pending...
                if resp_a == 1 and self.alert_at[true] < 0:
                    # ...schedule an analyst containment after the alert delay.
                    self.alert_at[true] = pos + c.alert_delay
                # The attack ages by the damage factor (it gets older when it is not stopped).
                self.age[true] += factor
            # Damage for this flow = base damage x factor.
            damage = base * factor
        # Branch: the flow is benign.
        else:
            # Counts benign flows presented.
            self.s["benign_presented"] += 1
            # If the Firewall blocked a benign flow...
            if fw_a == 2:
                # ...add the block disruption cost.
                disruption += c.disruption_block
            # Else if it rate-limited...
            elif fw_a == 1:
                # ...add the smaller rate-limit disruption.
                disruption += c.disruption_rl
            # If Response isolated...
            if resp_a == 2:
                # ...add the larger isolate disruption...
                disruption += c.disruption_isolate
                # ...and start an outage that hits the next benign flows.
                self.outage_left = c.outage_len
            # Else if an outage is already running...
            elif self.outage_left > 0:
                # ...add the outage cost...
                disruption += c.outage_cost
                # ...and use up one outage flow.
                self.outage_left -= 1
            # If this flow was disrupted at all...
            if disruption > 0:
                # ...count it as a disrupted benign flow.
                self.s["benign_disrupted"] += 1
            # If Response alerted on benign traffic...
            if resp_a == 1:
                # ...add the false-alert cost.
                cost += c.false_alert_cost

        # If Response alerted (any flow)...
        if resp_a == 1:
            # ...add the alert operating cost.
            cost += c.alert_cost
        # Else if Response isolated...
        elif resp_a == 2:
            # ...add the isolate operating cost.
            cost += c.isolate_cost

        # Outcome reward (shared by all agents): minus damage, disruption and cost.
        outcome = -(damage + disruption + cost)
        # Detection's shaping reward, scaled by 0.5.
        shape_det = c.shaping * detection_reward(det_a, true)
        # Firewall's shaping reward, scaled by 0.5.
        shape_fw = c.shaping * firewall_reward(fw_a, true)
        # Response's shaping reward, scaled by 0.5 (uses the repeat-offender flag).
        shape_resp = c.shaping * response_reward(resp_a, true, self.cur_repeat)

        # Adds damage to the running total.
        self.s["damage"] += damage
        # Adds disruption to the running total.
        self.s["disruption"] += disruption
        # Adds cost to the running total.
        self.s["cost"] += cost
        # Counts the step.
        self.s["steps"] += 1

        # Updates the damage moving average (this step's damage capped at 3).
        self.ema_damage = 0.9 * self.ema_damage + 0.1 * min(damage, 3.0)
        # Records whether the Firewall blocked.
        self.fw_hist.append(1 if fw_a == 2 else 0)
        # Records whether Detection flagged an attack.
        self.det_hist.append(1 if det_a != 0 else 0)
        # Increments the episode step counter.
        self.steps += 1
        # Moves to the next flow.
        self.pos += 1
        # Advances to the next flow needing a decision (skipping contained attackers).
        self._settle()
        # Returns the outcome reward, the three shaping rewards and the true class.
        return outcome, shape_det, shape_fw, shape_resp, true

    # Method that predicts the immediate damage, disruption and cost of an action without changing anything.
    def preview(self, det_a, fw_a, resp_a):
        # Docstring: used for counterfactual (blame) analysis.
        """Immediate (damage, disruption, cost) of a joint action on the current flow,
        without changing any simulator state. Used for counterfactual analysis."""
        # Short name for the settings.
        c = self.cfg
        # Current position.
        pos = self.pos
        # True class of the current flow.
        true = int(self.labels[pos])
        # Starts damage, disruption and cost at zero.
        damage = disruption = cost = 0.0
        # Branch: attack flow.
        if true != 0:
            # Base damage with age escalation.
            base = c.damage_base[true] * (1.0 + min(self.age[true] / c.age_scale, c.age_cap))
            # Throttle factor if throttled.
            factor = c.throttle_factor if self.throttle_until[true] > pos else 1.0
            # If blocked or isolated...
            if fw_a == 2 or resp_a == 2:
                # ...no damage.
                factor = 0.0
            # Else if rate-limited...
            elif fw_a == 1:
                # ...reduced damage.
                factor = min(factor, c.throttle_factor)
            # Damage for this flow.
            damage = base * factor
        # Branch: benign flow.
        else:
            # If blocked...
            if fw_a == 2:
                # ...block disruption.
                disruption += c.disruption_block
            # Else if rate-limited...
            elif fw_a == 1:
                # ...rate-limit disruption.
                disruption += c.disruption_rl
            # If isolated...
            if resp_a == 2:
                # ...isolate disruption.
                disruption += c.disruption_isolate
            # Else if an outage is running...
            elif self.outage_left > 0:
                # ...outage cost.
                disruption += c.outage_cost
            # If alerted on benign traffic...
            if resp_a == 1:
                # ...false-alert cost.
                cost += c.false_alert_cost
        # If alerted...
        if resp_a == 1:
            # ...alert cost.
            cost += c.alert_cost
        # Else if isolated...
        elif resp_a == 2:
            # ...isolate cost.
            cost += c.isolate_cost
        # Returns the three values.
        return damage, disruption, cost


# ---- observation builders (shared by training envs and evaluation) --------
# Detection's input builder.
def det_obs(sim):
    # Detection sees the 21 scaled features plus the 5 context values (26 numbers).
    return np.concatenate([sim.X[sim.pos], sim.ctx()]).astype(np.float32)


# Firewall's input builder (takes Detection's verdict).
def fw_obs(sim, det_a):
    # Creates a 4-long zero vector.
    onehot = np.zeros(N_CLASSES, dtype=np.float32)
    # Sets the entry of Detection's verdict to 1 (one-hot).
    onehot[det_a] = 1.0
    # Two extra values: the scaled repeat count and scaled packet rate of the flow.
    extra = [sim.X[sim.pos, DPORT_IDX], sim.X[sim.pos, PKTS_IDX]]
    # Firewall input = verdict one-hot + two features + context (11 numbers).
    return np.concatenate([onehot, extra, sim.ctx()]).astype(np.float32)


# Response's input builder (takes the Firewall's action).
def resp_obs(sim, fw_a):
    # Creates a 3-long zero vector.
    onehot = np.zeros(3, dtype=np.float32)
    # Sets the entry of the Firewall action to 1.
    onehot[fw_a] = 1.0
    # Response input = action one-hot + repeat count (capped at 20, scaled to 0-1) + context (9 numbers).
    return np.concatenate([onehot, [min(sim.cur_count, 20) / 20.0], sim.ctx()]).astype(np.float32)


# Single-agent input builder.
def single_obs(sim):
    # Single agent sees all 21 features, the repeat count and the context (27 numbers).
    return np.concatenate(
        [sim.X[sim.pos], [min(sim.cur_count, 20) / 20.0], sim.ctx()]
    ).astype(np.float32)


# Dictionary of input sizes per agent.
OBS_DIM = {
    # Detection: 21 features + 5 context = 26.
    0: len(DETECTION_FEATURES) + CTX_DIM,
    # Firewall: 4 verdict + 2 features + 5 context = 11.
    1: N_CLASSES + 2 + CTX_DIM,
    # Response: 3 actions + 1 repeat count + 5 context = 9.
    2: 3 + 1 + CTX_DIM,
# Closes the dictionary.
}
# Single agent input size: 21 + 1 + 5 = 27.
SINGLE_OBS_DIM = len(DETECTION_FEATURES) + 1 + CTX_DIM


# Environment class that trains one agent while the other two are fixed partners.
class RoleEnv(gym.Env):
    # Docstring: partners are learned policies, or None to use the rule-based fallback.
    """Trains one agent while the other two are frozen partners (a learned
    policy callable, or None for the rule-based fallback)."""

    # Gymnasium metadata (no rendering).
    metadata = {"render_modes": []}

    # Constructor: data, scaler, role (0, 1, 2), partners, settings, seed.
    def __init__(self, df, scaler, role, partners, cfg=None, seed=0):
        # Initializes the parent class.
        super().__init__()
        # Creates the simulator.
        self.sim = DefenseSim(df, scaler, cfg, seed)
        # Stores which agent is learning.
        self.role = role
        # Stores the frozen partner policies.
        self.partners = partners
        # Action space: 4 choices for Detection, 3 for the others.
        self.action_space = spaces.Discrete(N_CLASSES if role == 0 else 3)
        # Observation space: a vector with values from -8 to 8 of that agent's input size.
        self.observation_space = spaces.Box(-8.0, 8.0, (OBS_DIM[role],), np.float32)
        # Placeholder observation.
        self._obs = np.zeros(OBS_DIM[role], dtype=np.float32)

    # Method that gets Detection's decision when Detection is a partner.
    def _partner_det(self):
        # Looks up the Detection partner.
        p = self.partners.get("det")
        # If none, use the rule-based detection; otherwise ask the policy.
        return int(self.sim.rule_det[self.sim.pos]) if p is None else p(det_obs(self.sim))

    # Method that gets the Firewall partner's decision.
    def _partner_fw(self, det_a):
        # Looks up the Firewall partner.
        p = self.partners.get("fw")
        # If none, use the rule-based Firewall; otherwise ask the policy.
        return rule_based_firewall(det_a) if p is None else p(fw_obs(self.sim, det_a))

    # Method that gets the Response partner's decision.
    def _partner_resp(self, fw_a):
        # Looks up the Response partner.
        p = self.partners.get("resp")
        # If there is none...
        if p is None:
            # ...use the rule-based Response rule.
            return rule_based_response(fw_a, self.sim.cur_count, REPEAT_OFFENDER_THRESHOLD)
        # Otherwise ask the policy.
        return p(resp_obs(self.sim, fw_a))

    # Method that builds the observation for the learning agent.
    def _prepare(self):
        # If Detection is learning...
        if self.role == 0:
            # ...its input is just the Detection observation.
            return det_obs(self.sim)
        # Otherwise first get Detection's decision from its partner.
        self._det_a = self._partner_det()
        # If Firewall is learning...
        if self.role == 1:
            # ...its input uses that verdict.
            return fw_obs(self.sim, self._det_a)
        # Otherwise get the Firewall partner's decision.
        self._fw_a = self._partner_fw(self._det_a)
        # Response's input uses the Firewall action.
        return resp_obs(self.sim, self._fw_a)

    # Gymnasium reset method.
    def reset(self, *, seed=None, options=None):
        # Seeds the parent's random generator.
        super().reset(seed=seed)
        # Starts a new simulator episode.
        self.sim.reset()
        # Builds the first observation.
        self._obs = self._prepare()
        # Returns the observation and an empty info dictionary.
        return self._obs, {}

    # Gymnasium step method: receives the learning agent's action.
    def step(self, action):
        # Converts the action to an int.
        a = int(action)
        # If Detection is learning...
        if self.role == 0:
            # ...its action is the Detection verdict...
            det_a = a
            # ...the Firewall partner reacts to it...
            fw_a = self._partner_fw(det_a)
            # ...and the Response partner reacts to the Firewall.
            resp_a = self._partner_resp(fw_a)
        # Else if Firewall is learning...
        elif self.role == 1:
            # ...Detection's decision was stored earlier; the action is the Firewall action...
            det_a, fw_a = self._det_a, a
            # ...and the Response partner reacts.
            resp_a = self._partner_resp(fw_a)
        # Else (Response is learning)...
        else:
            # ...the earlier decisions are stored; the action is the Response action.
            det_a, fw_a, resp_a = self._det_a, self._fw_a, a
        # Applies the joint decision in the simulator and gets rewards.
        outcome, sd, sf, sr, true = self.sim.apply(det_a, fw_a, resp_a)
        # Reward = shared outcome + this agent's own shaping term.
        reward = outcome + (sd, sf, sr)[self.role]
        # Whether the episode is over.
        done = self.sim.done
        # If not over...
        if not done:
            # ...prepare the next observation.
            self._obs = self._prepare()
        # Returns observation, reward, "terminated" False, "truncated" = episode over, and the true class.
        return self._obs, reward, False, done, {"true_idx": true}


# Environment class for the single-agent baseline.
class SingleSimEnv(gym.Env):
    # Docstring: one policy picks all three decisions with the same information and the same outcome reward.
    """Single-agent baseline in the same simulator: one policy picks the joint
    (detection, firewall, response) action, with the same information and the
    same outcome reward the three MARL agents collectively get."""

    # Gymnasium metadata.
    metadata = {"render_modes": []}

    # Constructor.
    def __init__(self, df, scaler, cfg=None, seed=0):
        # Initializes the parent class.
        super().__init__()
        # Imports the number of joint actions here (to avoid a circular import).
        from src.single_agent_env import N_ACTIONS

        # Creates the simulator.
        self.sim = DefenseSim(df, scaler, cfg, seed)
        # Action space: 36 joint actions.
        self.action_space = spaces.Discrete(N_ACTIONS)
        # Observation space: 27 numbers between -8 and 8.
        self.observation_space = spaces.Box(-8.0, 8.0, (SINGLE_OBS_DIM,), np.float32)
        # Placeholder observation.
        self._obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)

    # Gymnasium reset method.
    def reset(self, *, seed=None, options=None):
        # Seeds the parent's random generator.
        super().reset(seed=seed)
        # Starts a new simulator episode.
        self.sim.reset()
        # Builds the single-agent observation.
        self._obs = single_obs(self.sim)
        # Returns the observation and an empty info dictionary.
        return self._obs, {}

    # Gymnasium step method.
    def step(self, action):
        # Imports the joint-action decoder.
        from src.single_agent_env import decode_action

        # Splits the joint action into detection, firewall and response actions.
        det_a, fw_a, resp_a = decode_action(int(action))
        # Applies them in the simulator.
        outcome, sd, sf, sr, true = self.sim.apply(det_a, fw_a, resp_a)
        # Reward = outcome + all three shaping terms (the single agent gets the sum).
        reward = outcome + sd + sf + sr
        # Whether the episode is over.
        done = self.sim.done
        # If not over...
        if not done:
            # ...build the next observation.
            self._obs = single_obs(self.sim)
        # Returns observation, reward, terminated False, truncated flag and the true class.
        return self._obs, reward, False, done, {"true_idx": true}
