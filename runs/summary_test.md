Seeds aggregated: 4 (runs\seed1\eval_test.json, runs\seed2\eval_test.json, runs\seed3\eval_test.json, runs\seed4\eval_test.json)

| System | Total loss | Attack damage | Benign disruption | Damage reduction | Benign flows disrupted | Mean latency (flows) |
|---|---|---|---|---|---|---|
| none | 52,160 +/- 0 | 52,160 +/- 0 | 0 +/- 0 | 0.0% +/- 0.0% | 0.00% +/- 0.00% | n/a |
| rule | 54,238 +/- 0 | 52,160 +/- 0 | 1,990 +/- 0 | 0.0% +/- 0.0% | 1.65% +/- 0.00% | n/a |
| rule_tuned | 38,004 +/- 0 | 11,518 +/- 0 | 25,276 +/- 0 | 77.9% +/- 0.0% | 19.51% +/- 0.00% | 148.6 +/- 0.0 |
| v2_marl | 803 +/- 738 | 549 +/- 806 | 220 +/- 400 | 98.9% +/- 1.5% | 0.15% +/- 0.28% | 50.4 +/- 86.0 |
| v2_single | 778 +/- 1,000 | 736 +/- 1,008 | 15 +/- 20 | 98.6% +/- 1.9% | 0.02% +/- 0.02% | 23.1 +/- 28.8 |
