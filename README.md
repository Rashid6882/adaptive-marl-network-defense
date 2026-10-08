# Adaptive Multi-Agent RL for Intelligent Network Defense

Three-agent MARL pipeline (Detection -> Firewall -> Response) trained on real
CICIDS2017 traffic, with a rule-based static-threshold baseline and an
optional single-agent RL baseline for comparison. See the project doc
(Sections 8-9) for the full state/action/reward spec.

## Known dataset limitation

The CSVs in `DATASET/` are the standard "MachineLearningCVE" release of
CICIDS2017, which does **not** include a Source IP column (only
`Destination Port` onward). The spec's "source IP repeat count" / "rolling
count of flows from the same source IP" features are therefore approximated
with a **Destination Port repeat-count proxy** (`dport_repeat_count` in
`src/data.py`, `src/features.py`), which captures the same underlying signal
(bursty/repeated behavior from one actor) without literal IP identity. This
is called out everywhere the feature is used.

## Setup

```bash
pip install -r requirements.txt
```

**Dataset:** the CICIDS2017 CSVs (~844 MB) are not included in this repository. Download the MachineLearningCVE release and place the three files used (`Tuesday-WorkingHours.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`) in a `DATASET/` folder.

## Pipeline

```bash
# 1. Sanity-check data loading/cleaning
python -m src.data

# 2. Train the 3 MARL agents (Detection -> Firewall -> Response)
python -m src.train --detection-steps 120000 --firewall-steps 60000 --response-steps 60000

# 3. (optional, stretch goal) Train the single-agent RL baseline
python -m src.train_single_agent --steps 120000

# 4. Evaluate everything on the held-out chronological test split
python -m src.evaluate
```

Outputs land in `outputs/`: reward curves (`reward_curves.png`), a detection
F1 comparison chart (`detection_f1_comparison.png`), and a full metrics dump
(`eval_summary.json`). Trained models land in `models/`.

## v2: interactive closed-loop training (`src/sim_env.py`)

v1 replays recorded flows independently of the agents' actions (each step's
reward depends only on that step). v2 keeps the recorded trace but makes
actions change what happens next: Block/Isolate contains an attacker (its later
flows are dropped until a TTL expires), Rate-limit throttles it, Alert triggers
a delayed analyst containment, uncontained attacks cause escalating damage,
and isolating a benign host causes an outage. Rewards are outcome-based
(damage, disruption, cost) shared by all agents plus a small role-specific
shaping term. Labels are used only by the simulator, never observed.

```bash
# round 0 (rule-based partners are used until a partner model exists)
python -m src.train_sim --role detection --steps 300000 --fresh
python -m src.train_sim --role firewall  --steps 200000 --fresh
python -m src.train_sim --role response  --steps 150000 --fresh
# round 1+ (warm-starts; each agent trains against the others' learned policies)
python -m src.train_sim --role detection --steps 250000
python -m src.train_sim --role firewall  --steps 150000
python -m src.train_sim --role response  --steps 150000
# BruteForce fix: v2 Detection was fine, but Firewall let ~94% of BruteForce through
# (rare in its training episodes). Two extra Firewall passes with more BruteForce exposure:
python -m src.train_sim --role firewall --steps 250000 --bf-anchor 0.6
python -m src.train_sim --role firewall --steps 200000 --bf-anchor 0.7
python -m src.train_sim --role single --steps 400000 --fresh   # baseline
python -m src.evaluate_sim
```

Assumptions: the attacker is identified by attack class within an episode (no
Source IP in this CSV release), and dropping flows after a block is a modeling
assumption, not a live network. Closed-loop results are in
`outputs/eval_sim_summary.json`.

## Design notes

- **Sequential conditioning**: each flow is one timestep. Detection acts
  first; Firewall's observation includes Detection's action for that same
  flow; Response's observation includes Firewall's action. Implemented via
  independent PPO (Stable-Baselines3) learners where each downstream agent's
  env calls a *frozen* snapshot of its upstream agent(s) to build
  observations (see `src/env.py` docstring) -- this reproduces the spec's
  per-flow sequential dependency without requiring a single joint policy.
- **Chronological split**: 80/20 split per source file (not shuffled), so
  test traffic is always later-captured than train traffic, per Section 11.
- **Class imbalance**: BruteForce is ~1.1% of rows, PortScan ~10.6%, DDoS
  ~14.9%, BENIGN ~73.5%. Early runs showed Detection collapsing to near-zero
  F1 on BruteForce/PortScan: PPO's cumulative-reward objective lets a policy
  that never predicts rare classes still score well overall, since nailing
  the BENIGN/DDoS majority dominates total return. Two fixes address this:
  (1) `DETECTION_CLASS_WEIGHT` in `src/features.py` scales the
  correct-prediction reward and false-negative penalty by inverse class
  frequency (BruteForce x2.5, PortScan x1.3), amplifying the gradient signal
  minority classes get per occurrence; (2) Detection and Firewall now use
  **class-balanced per-step sampling** (`_BaseFlowEnv._sample_index`,
  `balanced_sampling=True`) instead of chronological-window replay -- each
  step draws its row from a uniformly-chosen class 70% of the time (the
  remaining 30% from the natural distribution), since their features
  (including `dport_repeat_count`) are precomputed independent of visit
  order and don't require chronological replay. Response keeps the original
  chronological-window + attack-anchored sampling, since its within-episode
  repeat-offender counter genuinely needs sequential order.
- **BruteForce precision**: after the fixes above BruteForce recall was ~95%
  but precision was poor (~14k benign flows mislabeled). Only ~0.6% of benign
  flows use ports 21/22 (the Patator targets), so `is_ftp_ssh_port` (a flag
  derived from `Destination Port`) was added to `DETECTION_FEATURES`.
  Training is CPU-only (`device="cpu"`); stages can be run one at a time with
  `python -m src.train --only {detection,firewall,response}`.
- **Rewards**: asymmetric, role-specific costs per Section 8/9 -- see
  `src/features.py` for the exact reward tables.
