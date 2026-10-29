# Pre-registration: two experiments on why a decomposed (3-agent) design could help

Written BEFORE running either experiment. Predictions can fail; whatever happens is reported.
Evidence base: `PART_A_RESULTS.md` (equal-budget, 3 seeds, validation split).

## Experiment 1: cost of adapting to a new attack variant

**Question.** When attack traffic changes, is it cheaper to adapt the 3-agent system by retraining
only Detection (Firewall and Response frozen) than to retrain the single agent?

**Setup**
- Starting point: the already-trained seed 1-3 models (`runs/seed{1,2,3}`).
- Shift (`src/shift.py`): a synthetic "slow" variant of one attack class. Rates (`Flow Bytes/s`,
  `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`) are multiplied by 0.05; times (`Flow Duration`,
  `Flow IAT Mean`, `Flow IAT Std`) are divided by 0.05. This is a simulated variant, not real captured
  traffic, and the rolling repeat-count feature is left unchanged.
- Candidate variants: `slow_ddos`, `slow_portscan`.
- **Variant selection rule (applied before looking at adaptation results):** a variant is used only if,
  for seed 1 base models, mean loss on windows of the shifted class is at least 3x higher than on the
  same windows unshifted, for BOTH systems. Otherwise the shift is not a meaningful challenge.
- Adaptation: PPO fine-tuning on the shifted train split, same anchoring as original training
  (`--bf-anchor 0.5`). 3-agent system: only Detection learns. Single agent: the whole policy learns.
  Cumulative environment steps at checkpoints: 0, 25k, 50k, 100k, 200k, 400k.
- Evaluation at each checkpoint (validation split only, fixed windows): 24 windows of 1024 flows
  anchored on the shifted class (shifted traffic), and 24 general windows (unshifted traffic) to
  measure forgetting.

**Metrics.** (a) Environment steps to recover 90% of the way from the shifted-base loss back to the
pre-shift loss, with "not reached" if never. (b) Loss at 400k steps. (c) Loss on unshifted windows
(forgetting).

**Hypothesis H1.** The 3-agent system (Detection-only retraining) recovers in fewer environment steps
than the single agent, in at least 2 of 3 seeds.

**What would count against it.** The single agent recovers as fast or faster, or only one seed shows
the effect. That would be reported as a null result.

**Limits (stated now).** 3 seeds, synthetic shift, steps measured in environment interactions (not
wall-clock), agents' base quality differs by seed (a seed that already failed on the class may "recover"
by fixing the original failure rather than the shift).

## Experiment 2: who is responsible for a bad outcome, and why

**Question.** For the 3-agent system's bad outcomes (attack flow let through, or benign flow
disrupted), can we assign responsibility to a specific agent, and does SHAP show the reason?

**Method** (`src/explain_agents.py`, validation stream, seeds 1-3)
- A step is "bad" if it causes damage or benign disruption.
- Counterfactual blame by replacing ONE agent's decision with the oracle decision and re-running the
  downstream agents: fixing Detection (true class as verdict, Firewall and Response re-decide), fixing
  Firewall (Block if attack else Allow, Response re-decides), fixing Response (Isolate if attack else
  No action). An agent is "responsible" if fixing it alone removes all damage and disruption.
- SHAP (Kernel SHAP) on the Firewall's P(Block), grouped into: Detection-verdict inputs, repeat count,
  packet rate, context.

**Hypothesis H2a.** Most bad outcomes (more than 50%) are attributable to exactly one agent.
**Hypothesis H2b.** In cases where Detection is responsible, the Firewall's decision is explained mostly
(at least 50% of SHAP mass) by the Detection-verdict inputs, i.e. the Firewall followed a wrong verdict.

**Structural note (not an empirical claim).** The single agent chooses all three decisions from the
same input, so there is no downstream dependence to intervene on. Fixing its "detection" component never
changes the outcome by construction. This capability is specific to a sequential pipeline.

**Limits (stated now).** Counterfactuals use the oracle decision as a reference and treat each step in
isolation (no changes to later flows); SHAP explains one output (P(Block)); 3 seeds.

## Amendment (written after seeing the first Experiment 2 output, seed 1)

- **H2a as pre-registered failed on seed 1:** only 8 of 91 harmful steps (8.8%) were fixable by exactly one
  agent. 83 (91.2%) could be prevented by correcting any of several agents, because Response can always
  prevent damage by isolating. "Fixable by exactly one agent" is therefore not a useful notion of
  responsibility. The pre-registered tally is still reported unchanged.
- **Added (exploratory, post-hoc, not pre-registered):** a root-cause attribution, which blames the
  EARLIEST agent in the pipeline order Detection -> Firewall -> Response whose correction alone removes
  the harm. SHAP shares are reported per root-cause group. H2b is evaluated on the root-cause group
  "Detection" (post-hoc), since the pre-registered "Detection only" group was empty on seed 1.
