# Adaptive Multi-Agent RL for Intelligent Network Defense

Three reinforcement-learning agents (Detection -> Firewall -> Response) defend a simulated
network. The traffic is real recorded CICIDS2017 flows, but the **agents' actions change what
happens next**: blocking an attacker removes its later flows, letting an attack through causes
damage that keeps growing, and wrongly blocking normal traffic disrupts service. The agents learn
from these outcomes with PPO and never see the true labels.

## Main findings

- **Learned agents beat rule baselines by a large margin.** On the held-out test stream the 3-agent
  system cuts attack damage by 98.9% (a tuned rule baseline: 77.9%, while disrupting 19.5% of benign flows).
- **The 3-agent system does not beat a single agent at equal budget.** Over 4 seeds with the same
  1.2M training steps the two are tied (total loss 803 +/- 738 vs 778 +/- 1,000). We do not claim
  a performance advantage for MARL.
- **What the multi-agent design does give us is traceability.** Because the agents are separate we
  can fix one agent at a time and see which one caused a failure, and SHAP shows what each agent
  relied on. This found why one seed failed (see "Why did it fail?" below) and led to a fix.
- **Limits:** the simulator is our own model of consequences on a recorded trace, results come from
  4 seeds with large spread, and the Response agent mostly stays passive.

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

**What each agent sees.** Only Detection sees the flow. Firewall sees Detection's verdict plus two
flow values, and Response sees only Firewall's action plus history. All agents also get a 5-value
context: recent damage, recent block rate, outage flag, alert-pending flag, recent attack-verdict rate.

| Agent | Observation | Dim | Actions |
|---|---|---|---|
| Detection | 21 flow features + context | 26 | Benign / DDoS / PortScan / BruteForce |
| Firewall | Detection verdict (one-hot) + repeat count + packet rate + context | 11 | Allow / Rate-limit / Block |
| Response | Firewall action (one-hot) + repeat-offender counter + context | 9 | No action / Alert / Isolate |
| Single-agent baseline | 21 features + repeat counter + context | 27 | 36 joint actions (4x3x3) |

Episodes are 1024-flow stretches of the trace, 70% anchored just before a random attack (half of
them on BruteForce, the rest split between DDoS and PortScan), the rest random. Data: 858,485
cleaned flows, split **chronologically per file into 70% train / 10% validation / 20% test**.
Tuning and diagnosis use validation only; the test stream is evaluated once, at the end.

## Training (`src/train_sim.py`)

PPO (Stable-Baselines3, MlpPolicy, CPU, `n_steps=2048`, `batch_size=256`, `ent_coef=0.01`). One agent
trains at a time while the other two are frozen partners (a rule-based policy stands in for any
partner that has no trained model yet). Running the stages repeatedly gives **round-robin
co-training**: each agent warm-starts and trains against the others' latest learned policies.

Both systems get the **same total budget of 1.2M environment steps**, per seed. Seeds 1 to 4:

```bash
D=runs/seed1; S=1      # repeat for seeds 2, 3, 4 (change both)
C="--use-val-split --seed-id $S --models-dir $D --bf-anchor 0.5"

# 3-agent system. Round 0, then round 1 (warm start; partners are now learned policies)
python -m src.train_sim --role detection --steps 300000 --fresh $C
python -m src.train_sim --role firewall  --steps 200000 --fresh $C
python -m src.train_sim --role response  --steps 150000 --fresh $C
python -m src.train_sim --role detection --steps 250000 $C
python -m src.train_sim --role firewall  --steps 150000 $C
python -m src.train_sim --role response  --steps 150000 $C

# Single-agent baseline, same budget
python -m src.train_sim --role single --steps 1200000 --fresh $C

# Rule baseline thresholds, tuned on validation only
python -m src.tune_rule --out runs/tuned_rule.json --trials 1200
```

## Evaluation (`src/evaluate_sim.py`)

Each system is run closed-loop over the whole held-out stream (test: 171,699 flows, file by file),
and its own actions determine which flows it sees.

```bash
python -m src.evaluate_sim --models-dir runs/seed1 --split val  --out runs/seed1/eval_val.json  --tuned-rule runs/tuned_rule.json
python -m src.evaluate_sim --models-dir runs/seed1 --split test --out runs/seed1/eval_test.json --tuned-rule runs/tuned_rule.json
python -m src.aggregate_seeds --glob "runs/seed*/eval_test.json" --out runs/summary_test
```

### Results on the held-out test stream (mean +/- std over 4 seeds)

| System | Total loss | Damage reduction | Benign flows disrupted | Mean latency (flows) |
|---|---|---|---|---|
| No defense | 52,160 | 0% | 0% | n/a |
| Rule-based (default thresholds) | 54,238 | 0% | 1.65% | n/a |
| Rule-based (tuned on validation) | 38,004 | 77.9% | 19.51% | 148.6 |
| **3-agent MARL** | **803 +/- 738** | 98.9% +/- 1.5% | 0.15% +/- 0.28% | 50.4 +/- 86.0 |
| **Single agent** | **778 +/- 1,000** | 98.6% +/- 1.9% | 0.02% +/- 0.02% | 23.1 +/- 28.8 |

Per seed (total loss, lower is better):

| | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|
| Test: 3-agent | 996 | 1,764 | 163 | 290 |
| Test: single | 508 | 2,260 | 170 | 175 |
| Validation: 3-agent | 92 | 870 | 131 | 327 |
| Validation: single | 427 | 6,643 | 164 | 69 |

Validation, mean +/- std: 3-agent 355 +/- 358, single 1,826 +/- 3,215 (tuned rule 22,349). That gap is
almost entirely one failed single-agent seed (6,643); on the test stream the two are tied.

- The 3-agent system wins 2 of 4 seeds on test and the single agent wins 2. The difference in means
  (25) is far smaller than the spread (roughly 700 to 1,000), so we treat it as a tie.
- The 3-agent system disrupts more benign flows (0.15% vs 0.02%), driven mainly by seed 1.
- The default rule flags no real attack, so it equals "no defense" plus the cost of its false alarms.
  The tuned rule is the fair baseline.
- Per-class F1 is not a headline metric: once an attacker is contained its flows are dropped, so F1 is
  computed on very few attack flows.

### What each agent does (`src/per_agent_stats.py`, test stream, descriptive)

Counted over flows each agent actually saw.

| | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|
| Detection: DDoS exact-class recall | 74% | 69% | 20% | 2% |
| Detection: PortScan exact-class recall | 53% | 39% | 44% | 15% |
| Detection: BruteForce exact-class recall | 92% | **0%** | 92% | 93% |
| Detection: benign false-alarm rate | 0.83% | 0.03% | 0.01% | 0.19% |
| Firewall: blocks attack flows | 80% | 13% | 72% | 54% |
| Firewall: blocks when verdict is "attack" | 73% | 64% | 99% | 55% |
| Response: No action / Alert / Isolate on attacks | 77/23/0% | 84/16/0% | 28/72/0% | 90/1/9% |

- Seed 2 now misses BruteForce on the test stream (0% recall). Its DDoS failure is fixed, but the
  agents are not uniformly reliable across seeds. This is a generalization failure, not under-learning:
  with the allow-only check used in our diagnosis, seed 2's Detection flags 95% of BruteForce flows on its
  training stream but only 6% on validation and 4% on test, while seed 1 flags 100% / 96% / 99%. Why the
  later BruteForce flows differ from the early ones is untested.
- The Firewall almost never blocks an attack that Detection called Benign (0% in seeds 1 to 3, 8% in
  seed 4), so a Detection miss passes straight through. It does not always block when told "attack"
  (55% to 99% of attack verdicts), which is the Firewall's own failure mode (see the diagnosis below).
- **Response is mostly passive:** it isolates 0% of attacks in three seeds and 9% in seed 4, and
  never acts on benign flows. In practice the system works mainly as Detection plus Firewall.

## Why did it fail? (the diagnosis, `src/explain_agents.py`, validation stream)

In the first protocol-compliant run (3 seeds, old class weights) one seed failed badly (validation loss
10,574). Per-agent analysis showed that Detection's recall by attack class followed its reward weights
(DDoS 1.0, PortScan 1.3, BruteForce 2.5 gave recall of about 20%, 50% and 92%). Setting all three attack
weights to 2.5 (`DETECTION_CLASS_WEIGHT` in `src/features.py`) fixed that seed (validation loss 10,574 -> 870)
and left the other seeds about the same (103 -> 92 and 133 -> 131). The weights were chosen after seeing this validation failure, not on test.

For the final models, a step is "harmful" if it lets an attack through or disrupts a benign flow. We fix
each agent alone with the correct decision and check whether the harm disappears. The **root cause** is the
earliest agent whose fix alone removes the harm.

| | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|
| Harmful steps | 80 | 489 | 107 | 413 |
| Fixable by exactly one agent (pre-registered H2a) | 0% | 0% | 0% | 1.5% |
| Root cause: Detection | 92.5% | 9.8% | 89.7% | 67.1% |
| Root cause: Firewall | 7.5% | **90.2%** | 10.3% | 8.7% |
| Root cause: Response | 0% | 0% | 0% | 0% |
| No single agent fix removes the harm | 0% | 0% | 0% | 24.2% |
| SHAP share of the verdict inputs in Firewall's decision, Detection root cause | 91% | 0% | 90% | 50% |

- **The failure type differs by seed.** In seeds 1 and 3 the harm starts in Detection, and Firewall
  followed a wrong verdict (SHAP: about 90% of the Firewall's decision came from the verdict inputs). In
  seed 2 the harm starts in the Firewall, which relied on packet rate (88% of its SHAP mass) rather than
  the verdict. A single agent's three decisions come from one network, so this per-stage blame cannot be
  computed for it.
- **H2a (pre-registered: most harm fixable by exactly one agent) is not supported.** Several agents can
  often prevent the same harm; the root-cause rule above was added after seeing the first output and is
  exploratory (`archive/runs_partA/PREREGISTRATION.md`). It favors earlier agents by construction: with a
  downstream-first rule Response would take most of the blame.

### SHAP explanations (`src/explain_shap.py`, seed 1, validation stream)

For any flow, Kernel SHAP gives the top inputs behind each agent's chosen action. Examples are the first of
each kind in the stream, chosen for illustration:

| Example | Detection | Firewall | Response |
|---|---|---|---|
| Attack stopped (BruteForce) | BruteForce, driven by `is_ftp_ssh_port` (+0.50) and PSH flag count (+0.33) | Block, driven by the verdict (+0.48) | Alert |
| Attack missed (BruteForce) | Benign (p=1.00); the port flag pushes against it (-0.12) | Allow (follows the Benign verdict) | No action |
| False alarm (benign flow) | PortScan (p=0.59), driven by `Bwd Packets/s` (+0.75) | Block, driven by the verdict (+0.47) | Alert |

For 73 sampled decisions per agent, 100% had a faithful explanation (SHAP values add up to the model's own
probability within 0.02). This checks faithfulness to the model, not human understandability. For routine
"Benign -> Allow" decisions SHAP values are near 0, because they are measured against an average of mostly
routine traffic.

## Results that did not go our way

- **Adaptation to a new attack variant** (a synthetic "slow" DDoS) was a pilot on one seed and did not
  support our hypothesis: the single agent recovered, while the 3-agent system (only Detection retrained)
  did not. Details are in `archive/runs_partA/EXPERIMENT_RESULTS.md`; the code was removed to keep the
  project small and is in git commit `d4afdf7`.
- **MARL vs single agent:** a tie on test (above). The earlier exploratory run that showed MARL far ahead
  used unequal training and tuning on the test stream (`archive/`); it should not be cited.

## Dashboard

```bash
python -m streamlit run src/dashboard.py
```

The **Interactive RL** tab shows the 4-seed results table and lets you pick an attack type, burst and
system to watch cumulative damage with and without a defense (it uses the seed-1 models), plus training
curves. The first load takes a minute or two.

## Files

| Path | Purpose |
|---|---|
| `src/sim_env.py` | Simulator (`DefenseSim`), role environments, single-agent environment, config |
| `src/train_sim.py` | Per-role PPO training with warm start and frozen partners |
| `src/evaluate_sim.py` | Closed-loop evaluation of all systems |
| `src/tune_rule.py`, `src/baseline_rule.py` | Rule baseline and its validation-tuned thresholds |
| `src/aggregate_seeds.py` | Mean +/- std tables across seeds |
| `src/explain_agents.py` | Counterfactual blame and SHAP shares per root cause |
| `src/explain_shap.py` | Per-decision SHAP explanations and the coverage check |
| `src/per_agent_stats.py` | What each agent actually does on the test stream |
| `src/data.py`, `src/features.py` | Data cleaning and features, reward tables |
| `src/env.py`, `src/single_agent_env.py` | Shared feature helpers and joint-action encoding |
| `runs/seed{1..4}/` | Trained agents, scaler, evaluation and explanation outputs per seed |
| `runs/summary_{val,test}.md` | Aggregated results |
| `archive/` | Earlier protocol results, pre-registration and the adaptation pilot (kept as a record) |

## Honest limitations

- **Simulated environment.** The dynamics above (TTLs, escalation, outages) are modeling
  assumptions on a recorded trace, not a live network. Dropping flows after a block is a
  counterfactual assumption. Results do not show performance on live traffic.
- **Not purely RL.** Part of the reward is a label-based shaping term, and training is round-robin
  (one agent at a time with frozen partners) rather than simultaneous.
- **Design choices were made during exploration.** The simulator settings and `--bf-anchor 0.5` come from
  an earlier exploratory phase that looked at test results. The class weights were set after a validation
  failure. The final test stream was evaluated once, with nothing changed afterwards.
- **No MARL performance advantage** over the single agent at equal budget, and the 3-agent system
  disrupts more benign flows.
- **Four seeds, large spread.** The test standard deviations (738 and 1,000) are as large as the
  difference we would want to detect. Individual seeds can fail on a class (seed 2 on BruteForce).
- **Response is mostly passive**, so the third agent adds little to the measured outcome.
- **Attackers are identified by attack class** (no Source IP), and only three CICIDS2017 attack types
  and one dataset are used.
- **The blame analysis is exploratory** and depends on the "earliest fixer" convention. SHAP explains the
  model, not the traffic.
- **Not addressed:** adaptive or unseen attack variants (the pilot was negative) and adversarial attackers.
