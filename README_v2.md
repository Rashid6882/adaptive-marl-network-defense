# v2: Interactive Closed-Loop RL for Network Defense

v2 extends the three-agent pipeline (Detection -> Firewall -> Response) from
[README.md](README.md) so that the agents **learn by interacting with an environment that reacts
to their actions**, instead of replaying recorded flows independently of what they do.

- **v1 (offline replay):** each recorded flow is scored right away against its label. The next
  flow never depends on the agents' actions, so the problem behaves like a contextual bandit.
- **v2 (closed loop):** the traffic is still a recorded trace, but blocking, throttling and
  alerting change which flows the agents see next. Rewards come from outcomes (damage,
  disruption, cost), so decisions have delayed consequences.

v1 is untouched and still available (`models/`, `src/train.py`, `src/evaluate.py`).

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
(the v1 reward tables, so Detection, whose output is only a message to Firewall, still gets
signal). Ground-truth labels are used **only inside the simulator** to compute outcomes. The
agents never observe them.

**Observations** (all agents also get a 5-value context: recent damage, recent block rate,
outage flag, alert-pending flag, recent attack-verdict rate):

| Agent | Observation | Dim | Actions |
|---|---|---|---|
| Detection | 21 flow features + context | 26 | Benign / DDoS / PortScan / BruteForce |
| Firewall | Detection verdict (one-hot) + repeat count + packet rate + context | 11 | Allow / Rate-limit / Block |
| Response | Firewall action (one-hot) + repeat-offender counter + context | 9 | No action / Alert / Isolate |
| Single-agent baseline | 21 features + repeat counter + context | 27 | 36 joint actions (4x3x3) |

Episodes are 1024-flow stretches of the trace, 70% anchored just before a random attack
(class-weighted, see `--bf-anchor`), the rest random. The attacker is identified by its attack
class within an episode, because this CSV release has no Source IP column.

## Training (`src/train_sim.py`)

PPO (Stable-Baselines3, MlpPolicy, CPU, `n_steps=2048`, `batch_size=256`, `ent_coef=0.01`).
One agent trains at a time while the other two are frozen partners. Running the stages
repeatedly gives **round-robin co-training**: each agent warm-starts and then trains against the
others' latest learned policies. A partner whose model doesn't exist yet falls back to the
rule-based policy.

```bash
pip install -r requirements.txt

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

Models are saved to `models_v2/`; reward curves to `outputs/sim_<role>_reward_curve.json`.
The first run builds a cached, trimmed copy of the cleaned dataset
(`outputs/_clean_cache.pkl`) so later stages start quickly and use much less memory.

**Why the BruteForce fix:** after round 1, Detection labeled all BruteForce flows correctly, but
Firewall let ~94% of them through (BruteForce was rare in its training episodes and ~38% of its
BruteForce verdicts were false alarms). More BruteForce exposure raised Firewall's block rate on
BruteForce flows from 6% to 92%.

## Evaluation (`src/evaluate_sim.py`)

Each system is run closed-loop over the whole held-out chronological test stream (171,699
flows, file by file), and its own actions determine which flows it sees.

```bash
python -m src.evaluate_sim     # writes outputs/eval_sim_summary.json and eval_sim_loss.png
```

| System | Total loss | Damage reduction | Benign flows disrupted | Mean latency (flows) |
|---|---|---|---|---|
| No defense | 52,160 | 0% | 0% | n/a |
| Rule-based | 54,238 | 0% | 1.65% | n/a |
| v1 MARL (offline-trained) | 1,679 | 99.8% | 0.89% | 1.8 |
| **v2 MARL (interactive)** | **361** | **99.7%** | **0.12%** | 8.5 |
| v2 single-agent | 34,169 | 34.5% | 0.00% | 19.8 |

v2 reaches about the same damage reduction as v1 with far fewer false alarms. Per-class F1 under
closed loop is **not** reported as a headline metric: once an attacker is contained its flows are
dropped, so F1 is computed on very few attack flows. Open-loop F1 for v1 is in the main README.

## Dashboard

```bash
python -m streamlit run src/dashboard.py
```

The **"v2: Interactive RL"** tab shows the results table and loss charts, and lets you pick an
attack type, burst and system to watch cumulative damage with and without a defense, plus v2
training curves. The first load takes a minute or two. Close other heavy apps first if RAM is tight.

## Files

| Path | Purpose |
|---|---|
| `src/sim_env.py` | Simulator (`DefenseSim`), role environments, single-agent environment, config |
| `src/train_sim.py` | Per-role PPO training with warm start and frozen partners |
| `src/evaluate_sim.py` | Closed-loop evaluation of all systems |
| `models_v2/` | Trained v2 agents. `models_v2_round1/` holds the pre-BruteForce-fix backup |
| `outputs/eval_sim_summary*.json` | Closed-loop results (`_round1` = before the BruteForce fix) |

## Honest limitations

- **Simulated environment.** The dynamics above (TTLs, escalation, outages) are modeling
  assumptions on a recorded trace, not a live network. Dropping flows after a block is a
  counterfactual assumption.
- **Not purely RL.** Part of the reward is a label-based shaping term, training is round-robin
  (one agent at a time with frozen partners) rather than simultaneous, and round 0 used
  rule-based or under-trained partners.
- **Possible test-set tuning.** The BruteForce diagnosis and the choice of `--bf-anchor` values
  were made by looking at test-stream results; there is no separate validation split. Treat the
  fix as exploratory.
- **Unequal baseline training.** The single-agent baseline got one 400k-step run, while the
  3-agent system got several rounds (roughly 0.8M Firewall, 0.55M Detection, 0.3M Response
  steps), so the gap overstates the benefit of coordination.
- **One seed**, and attackers are identified by attack class (no Source IP).
- **Not yet addressed:** adaptive or unseen attack variants, response-latency evaluation beyond
  the simple flow-count metric, and the SHAP explainability layer.
