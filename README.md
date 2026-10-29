# Adaptive Multi-Agent RL for Intelligent Network Defense

Three reinforcement-learning agents (Detection -> Firewall -> Response) defend a simulated
network. The traffic is real recorded CICIDS2017 flows, but the **agents' actions change what
happens next**: blocking an attacker removes its later flows, letting an attack through causes
damage that keeps growing, and wrongly blocking normal traffic disrupts service. The agents learn
from these outcomes with PPO and never see the true labels.

The earlier offline-replay version of this project (agents scored directly against labels on
independently replayed flows) is preserved on the `v1` branch of the repository.

## Dataset and setup

```bash
pip install -r requirements.txt
```

The CICIDS2017 CSVs (~844 MB) are not included in this repository. Download the
MachineLearningCVE release and place the three files used
(`Tuesday-WorkingHours.pcap_ISCX.csv`,
`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`,
`Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`) in a `DATASET/` folder.

**Known limitation:** this CSV release has no Source IP column. A destination-port repeat count
(`dport_repeat_count`) stands in for "repeat offender" signals, and in the simulator an attacker is
identified by its attack class within an episode.

## How the simulator works (`src/sim_env.py`)

Each step the agents decide on one flow, in order: Detection -> Firewall -> Response. The
simulator then applies the joint action.

| Event | Effect |
|---|---|
| **Block** or **Isolate** on an attack flow | That attacker is contained: its later flows are dropped from the stream until a block expires (TTL: 200 rows for Block, 400 for Isolate) |
| **Rate-limit** on an attack flow | Attacker is throttled for 120 rows: damage x 0.3 per flow |
| **Alert** on an attack flow | An analyst contains the attacker 40 rows later (TTL 300): cheap but slow |
| Attack flow let through | Damage = base x (1 + min(age/60, 2)), so it escalates the longer the attack runs. Base damage: DDoS 1.0, PortScan 0.5, BruteForce 1.5 |
| **Block** on a benign flow | Disruption 1.0 (Rate-limit: 0.3) |
| **Isolate** on a benign flow | Disruption 3.0 + an outage: the next 15 benign flows each add 0.5 |
| Action costs | Alert 0.05 (+0.1 if the flow was benign), Isolate 0.3 |

**Reward** = `-(damage + disruption + cost)` (shared outcome) + `0.5 x` a role-specific shaping term
(label-based reward tables in `src/features.py`, so Detection, whose output is only a message to
Firewall, still gets signal). Ground-truth labels are used **only inside the simulator** to compute
outcomes; the agents never observe them.

**Observations** (all agents also get a 5-value context: recent damage, recent block rate, outage
flag, alert-pending flag, recent attack-verdict rate):

| Agent | Observation | Dim | Actions |
|---|---|---|---|
| Detection | 21 flow features + context | 26 | Benign / DDoS / PortScan / BruteForce |
| Firewall | Detection verdict (one-hot) + repeat count + packet rate + context | 11 | Allow / Rate-limit / Block |
| Response | Firewall action (one-hot) + repeat-offender counter + context | 9 | No action / Alert / Isolate |
| Single-agent baseline | 21 features + repeat counter + context | 27 | 36 joint actions (4x3x3) |

Episodes are 1024-flow stretches of the trace, 70% anchored just before a random attack, the rest
random. Data: 858,485 cleaned flows, split 80/20 chronologically per file.

## Training (`src/train_sim.py`)

PPO (Stable-Baselines3, MlpPolicy, CPU, `n_steps=2048`, `batch_size=256`, `ent_coef=0.01`). One agent
trains at a time while the other two are frozen partners (a rule-based policy stands in for any
partner that has no trained model yet). Running the stages repeatedly gives **round-robin
co-training**: each agent warm-starts and trains against the others' latest learned policies. The
first run builds a cached, trimmed copy of the cleaned data (`outputs/_clean_cache.pkl`) and a
feature scaler (`models_v2/scaler.joblib`).

```bash
# Round 0
python -m src.train_sim --role detection --steps 300000 --fresh
python -m src.train_sim --role firewall  --steps 200000 --fresh
python -m src.train_sim --role response  --steps 150000 --fresh

# Round 1 (warm start; partners are now learned policies)
python -m src.train_sim --role detection --steps 250000
python -m src.train_sim --role firewall  --steps 150000
python -m src.train_sim --role response  --steps 150000

# BruteForce fix: two extra Firewall passes with more BruteForce-anchored episodes
python -m src.train_sim --role firewall --steps 250000 --bf-anchor 0.6
python -m src.train_sim --role firewall --steps 200000 --bf-anchor 0.7

# Single-agent baseline in the same simulator
python -m src.train_sim --role single --steps 400000 --fresh
```

**Why the BruteForce fix:** after round 1, Detection labeled all BruteForce flows correctly, but
Firewall let ~94% of them through (BruteForce was rare in its training episodes and ~38% of its
BruteForce verdicts were false alarms). More BruteForce exposure raised Firewall's block rate on
BruteForce flows from 6% to 92%.

## Evaluation (`src/evaluate_sim.py`)

Each system is run closed-loop over the whole held-out chronological test stream (171,699 flows,
file by file), and its own actions determine which flows it sees.

```bash
python -m src.evaluate_sim     # writes outputs/eval_sim_summary.json and eval_sim_loss.png
```

| System | Total loss | Damage reduction | Benign flows disrupted | Mean latency (flows) |
|---|---|---|---|---|
| No defense | 52,160 | 0% | 0% | n/a |
| Rule-based | 54,238 | 0% | 1.65% | n/a |
| **3-agent MARL** | **361** | **99.7%** | **0.12%** | 8.5 |
| Single-agent RL | 34,169 | 34.5% | 0.00% | 19.8 |

The rule-based baseline never flags a real attack, so it equals "no defense" plus the cost of its
false alarms. Per-class F1 is not a headline metric here: once an attacker is contained its flows
are dropped, so F1 is computed on very few attack flows.

## Dashboard

```bash
python -m streamlit run src/dashboard.py
```

The **Interactive RL** tab shows the results table and loss charts, and lets you pick an attack
type, burst and system to watch cumulative damage with and without a defense, plus training
curves. The first load takes a minute or two.

## Files

| Path | Purpose |
|---|---|
| `src/sim_env.py` | Simulator (`DefenseSim`), role environments, single-agent environment, config |
| `src/train_sim.py` | Per-role PPO training with warm start and frozen partners |
| `src/evaluate_sim.py` | Closed-loop evaluation of all systems |
| `src/data.py`, `src/features.py` | Data cleaning and features, reward tables |
| `src/env.py`, `src/single_agent_env.py` | Shared feature helpers and joint-action encoding |
| `src/baseline_rule.py` | Rule-based static-threshold baseline |
| `models_v2/` | Trained agents and scaler. `models_v2_round1/` is the pre-BruteForce-fix backup |
| `outputs/` | Closed-loop results (`eval_sim_summary*.json`), charts, training reward curves |

## Honest limitations

- **Simulated environment.** The dynamics above (TTLs, escalation, outages) are modeling
  assumptions on a recorded trace, not a live network. Dropping flows after a block is a
  counterfactual assumption.
- **Not purely RL.** Part of the reward is a label-based shaping term, and training is round-robin
  (one agent at a time with frozen partners) rather than simultaneous.
- **Possible test-set tuning.** The BruteForce diagnosis and the choice of `--bf-anchor` values were
  made by looking at test-stream results; there is no separate validation split.
- **Unequal baseline training.** The single-agent baseline got one 400k-step run, while the 3-agent
  system got several rounds, so the gap overstates the benefit of coordination.
- **Weak rule baseline.** Its thresholds are hand-set heuristics that were never tuned on this data.
- **One seed**, and attackers are identified by attack class (no Source IP).
- **Not yet addressed:** adaptive or unseen attack variants and the planned SHAP explainability layer.
