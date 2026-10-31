Seeds aggregated: 3 (runs\seed1\eval_test.json, runs\seed2\eval_test.json, runs\seed3\eval_test.json)

| System | Total loss | Attack damage | Benign disruption | Damage reduction | Benign flows disrupted | Mean latency (flows) |
|---|---|---|---|---|---|---|
| none | 52,160 +/- 0 | 52,160 +/- 0 | 0 +/- 0 | 0.0% +/- 0.0% | 0.00% +/- 0.00% | n/a |
| rule | 54,238 +/- 0 | 52,160 +/- 0 | 1,990 +/- 0 | 0.0% +/- 0.0% | 1.65% +/- 0.00% | n/a |
| rule_tuned | 38,004 +/- 0 | 11,518 +/- 0 | 25,276 +/- 0 | 77.9% +/- 0.0% | 19.51% +/- 0.00% | 148.6 +/- 0.0 |
| v2_marl | 1,934 +/- 2,939 | 1,856 +/- 2,938 | 66 +/- 61 | 96.4% +/- 5.6% | 0.04% +/- 0.04% | 10.7 +/- 6.5 |
| v2_single | 2,869 +/- 4,583 | 1,874 +/- 3,010 | 926 +/- 1,490 | 96.4% +/- 5.8% | 0.54% +/- 0.85% | 80.4 +/- 125.6 |
