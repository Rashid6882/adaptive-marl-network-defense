# Experiments on why a decomposed design could help (status)

Hypotheses and criteria are in `PREREGISTRATION.md` (written before the runs, with one dated amendment).
Code: `src/shift.py`, `src/adapt_experiment.py` (Experiment 1), `src/explain_agents.py` (Experiment 2),
plus `DefenseSim.preview()` in `src/sim_env.py` (checked: 9,903 steps, 0 mismatches against the real simulator).

## Experiment 2: who is responsible for a harmful step, and why (DONE, validation stream, 3 seeds)

A step is harmful if it lets an attack through or disrupts a benign flow. Each agent is fixed alone with the
correct decision and we check whether the harm disappears.

| | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| Harmful steps | 91 | 5,159 | 124 |
| Fixable by exactly one agent (pre-registered H2a) | 8 (8.8%) | 0 (0%) | 0 (0%) |
| Root cause = Detection (earliest agent that could have prevented it) | 85% | 67% | 82% |
| Root cause = Firewall | 15% | 33% | 18% |
| Root cause = Response | 0% | 0% | 0% |
| SHAP share of Detection-verdict inputs in the Firewall's decision, Detection root cause (H2b) | 49% | 72% | 65% |

- **H2a (pre-registered): not supported.** Harm was almost never fixable by exactly one agent, because several
  agents (notably Response, which can always isolate) can each prevent it on their own.
- **Root-cause attribution is exploratory (added after seeing seed 1).** Most harm is rooted in Detection
  (67-85%). H2b (at least 50% of SHAP on the verdict inputs) holds in seeds 2 and 3 and narrowly misses in seed 1 (49%).
- **Caveat on the attribution rule:** blaming the earliest fixable agent favors Detection by construction.
  With a downstream-first rule, Response would take most of the blame. The split depends on this convention.
- Seed 2's count is dominated by the DDoS detection failure found earlier.
- Not shown: the single agent cannot be analyzed this way, because its three decisions come from one input
  and nothing downstream depends on its "detection" output (a structural fact, not a measured result).

## Experiment 1: cost of adapting to a new attack variant (option A run on seed 1 only; see end of section)

Pre-registered selection rule: use a variant only if it raises loss at least 3x for BOTH systems (seed 1).

| Variant | System | Pre-shift loss | Shifted loss | Ratio |
|---|---|---|---|---|
| slow DDoS | 3-agent | 15.1 | 81.4 | 5.4x |
| slow DDoS | single | 10.0 | 89.5 | 8.9x |
| slow PortScan | 3-agent | 9.4 | 39.8 | 4.2x |
| slow PortScan | single | 10.7 | 44.0 | 4.1x |

Both variants pass. (Loss = mean total loss over windows of 1,024 flows anchored on the shifted class.)

Measured cost of a small run (seed 1, 25k adaptation steps + 3 evaluations of 12 windows): 233 s for the
3-agent system, 138 s for the single agent. Scaling this up, a 200k-step adaptation run takes roughly 17 min
(3-agent) and 9 min (single); one variant across 3 seeds is about 80 min, and two variants about 2.7 hours.
A single run can exceed the 10-minute per-command limit, so larger budgets need either a smaller design or
a save/resume option.

### Option A run: seed 1, slow DDoS, up to 100k adaptation steps, 8 windows per evaluation

Loss on windows of 1,024 flows anchored on the shifted class (lower is better). Pre-shift reference: 16.1
(3-agent) and 9.9 (single).

| Adaptation steps | 3-agent (only Detection retrains): shifted / unshifted loss | Single agent (whole policy retrains): shifted / unshifted loss |
|---|---|---|
| 0 | 101.0 / 7.7 | 112.5 / 5.3 |
| 25k | 93.8 / 9.2 | 108.1 / 4.9 |
| 50k | 100.8 / 7.4 | 92.5 / 5.7 |
| 100k | 100.3 / 7.8 | 24.3 / 5.8 |

- **H1 (pre-registered) is NOT supported on this run.** The 3-agent system did not recover at all within 100k
  steps, while the single agent recovered most of the way (24.3 vs. a 90%-recovery target of 20.2, so about 86%,
  and it was still improving).
- Neither system forgot unshifted traffic.
- Limits: one seed, one variant, 100k steps (the single agent was still improving, the 3-agent system flat).
- **Possible explanations (untested):** (1) the Firewall and Response agents are frozen, but their inputs
  (packet rate, repeat count) also changed, so retraining only Detection cannot fix a Firewall problem;
  Experiment 2 found the Firewall to be the root cause in 15-33% of harmful steps. (2) Detection learns through
  a frozen Firewall, so its reward signal is indirect. (3) 100k steps may be too short for it.

## SHAP explainability layer (`src/explain_shap.py`; run on seed 1, validation stream)

For any flow, SHAP (Kernel SHAP, 50 background inputs, 200 samples) gives the top inputs behind each agent's
chosen action. Example flows are the first of each kind in the stream, chosen for illustration, not at random.

| Example | Detection | Firewall | Response |
|---|---|---|---|
| Attack stopped (BruteForce) | BruteForce, driven by `is_ftp_ssh_port` (+0.76) | Block, driven by "verdict is not Benign" (+0.80) | Alert |
| Attack missed (BruteForce) | Benign (p=1.00); port flag pushes against it (-0.29) | Allow (follows the Benign verdict) | No action |
| False alarm (benign flow) | BruteForce (p=0.92), driven by `is_ftp_ssh_port` (+0.73) | Block, driven by the verdict (+0.80) | No action |

- **Coverage:** for 73 sampled decisions per agent, 100% had a faithful explanation (SHAP values add up to the
  model's own probability within 0.02). This checks faithfulness to the model, not human understandability.
- **Finding:** Detection leans heavily on the FTP/SSH port flag, and the false-alarm example shows the
  consequence: a benign flow on port 21/22 is flagged as BruteForce, and Firewall then follows the verdict.
- **Reading routine decisions:** for the common "Benign -> Allow" case SHAP values are about 0, because
  SHAP explains the difference from the average decision and the background is mostly routine traffic.
- Limits: one seed; KernelSHAP is an approximation; explanations describe what the model uses, not why
  the traffic is an attack.
