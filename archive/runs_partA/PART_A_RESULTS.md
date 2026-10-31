# Part A: Making the evidence hold up

Protocol, results and conclusions for the rigor fixes (equal-budget baseline, validation split,
multiple seeds, simulator sensitivity, tuned rule baseline). All numbers come from files in `runs/`.

## Protocol (fixed before the test set was touched)

- **Splits** (per file, chronological): first 70% train, next 10% validation, last 20% test. The test
  set is identical to the one used in earlier experiments.
- **Training:** train split only. BruteForce gets 50% of attack-anchored episodes in every stage
  (`--bf-anchor 0.5`), chosen up front, with no after-the-fact fixes.
- **3 seeds** (`runs/seed1..3`), each trained independently.
- **3-agent budget:** Detection 550k + Firewall 350k + Response 300k = 1.2M steps (round 0 then round 1).
- **Single-agent budget:** 1.2M steps, the same total, trained under the same simulator and anchoring.
- **Rule baseline:** thresholds tuned by random search on the **validation** stream only
  (`runs/tuned_rule.json`); the default hand-set version is kept for reference.
- **Reporting:** all three seeds are reported, with no seed selection. Test was evaluated once per seed.

## Test stream (171,699 flows): mean +/- std over 3 seeds

| System | Total loss | Attack damage | Benign disruption | Damage reduction | Benign flows disrupted |
|---|---|---|---|---|---|
| No defense | 52,160 | 52,160 | 0 | 0.0% | 0.00% |
| Rule-based (default) | 54,238 | 52,160 | 1,990 | 0.0% | 1.65% |
| Rule-based (tuned on validation) | 38,004 | 11,518 | 25,276 | 77.9% | 19.51% |
| 3-agent MARL | 1,934 +/- 2,939 | 1,856 +/- 2,938 | 66 +/- 61 | 96.4% +/- 5.6% | 0.04% +/- 0.04% |
| Single agent (equal budget) | 2,869 +/- 4,583 | 1,874 +/- 3,010 | 926 +/- 1,490 | 96.4% +/- 5.8% | 0.54% +/- 0.85% |

Per seed (test total loss):

| | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| 3-agent MARL | 242 | 5,328 | 234 |
| Single agent | 275 | 172 | 8,161 |

## Validation stream (85,850 flows): mean +/- std over 3 seeds

| System | Total loss | Damage reduction | Benign flows disrupted |
|---|---|---|---|
| No defense | 76,430 | 0.0% | 0.00% |
| Rule-based (default) | 76,840 | 0.0% | 0.99% |
| Rule-based (tuned) | 22,349 | 79.6% | 16.27% |
| 3-agent MARL | 3,603 +/- 6,037 | 95.3% +/- 7.9% | 0.03% +/- 0.03% |
| Single agent | 3,099 +/- 5,036 | 96.9% +/- 5.0% | 1.19% +/- 2.06% |

Per seed (validation total loss): MARL 103 / 10,574 / 133; single 136 / 247 / 8,914.

## Where the failed seeds failed (validation, diagnostic only)

- **MARL seed 2:** Detection labeled 4,588 of 5,160 DDoS flows as benign, which caused 10,518 of the
  10,538 total damage. PortScan and BruteForce were handled well.
- **Single seed 3:** it labeled most DDoS and BruteForce flows as benign (DDoS damage 5,877,
  BruteForce 930).
- The pattern suggests DDoS detection does not always generalize to later-in-time traffic. A possible
  contributing factor is that BruteForce-heavy anchoring leaves DDoS with less exposure (25% of
  attack-anchored episodes vs. one third before). **This is a hypothesis; it has not been tested.**

## Simulator sensitivity (seed 1, validation stream, agents trained under default settings)

| Variant | None | Rule | Tuned rule | MARL | Single |
|---|---|---|---|---|---|
| default | 76,430 | 76,840 | 22,349 | 103 | 136 |
| TTLs x0.5 | 76,430 | 76,840 | 28,511 | 154 | 214 |
| TTLs x2 | 76,430 | 76,840 | 20,083 | 78 | 98 |
| no escalation | 25,598 | 26,008 | 13,295 | 169 | 134 |
| strong escalation | 127,262 | 127,672 | 31,404 | 107 | 145 |
| disruption cost x2 | 76,430 | 77,231 | 28,779 | 119 | 141 |
| equal damage weights | 105,297 | 105,708 | 24,666 | 102 | 142 |

The ordering "learned systems << tuned rule << no defense" holds in all 7 variants.
MARL vs. single agent flips in one variant (no escalation), by a small margin.

## Tuned rule thresholds (validation-tuned)

`ddos_pkt=199, ddos_fwd=2, ps_dur=332, ps_rep=31, bf_rep=16, bf_dur=5.16e6, resp_thr=5` over 1,200
trials. It cut validation loss from 76,840 to 22,349 but transferred imperfectly to test (38,004).

## What these results support, and what they do not

**Supported**
- Learned agents, MARL or single, beat both rule baselines by a large margin, including a tuned one
  (mean test loss about 20x lower for the 3-agent system and about 13x lower for the single agent
  than the tuned rule), and this ranking survives all simulator variants tried.
- Both learned systems are highly seed-dependent: one of three seeds failed badly for each.

**Not supported**
- "Coordination gives a large benefit." At equal budget the 3-agent and single-agent systems have the
  same mean damage reduction (96.4%) and overlapping total loss (1,934 +/- 2,939 vs. 2,869 +/- 4,583).
  The earlier large gap came from unequal training.
- A suggestive but unproven difference: MARL disrupts fewer benign flows (0.04% vs. 0.54%), but the
  single agent's number is dominated by one seed, and n = 3 is too small to claim significance.

## Caveats

- n = 3 seeds; the high variance means more seeds are needed before any ranking between MARL and the
  single agent can be claimed.
- Sensitivity covers one seed and the validation stream; agents were not retrained under each variant.
- The rule baseline was tuned only on validation; it still uses hand-coded FTP/SSH ports.
- The earlier exploratory results (models_v2/, the BruteForce fix) were tuned looking at test results and
  should not be presented as final numbers.
