Seeds aggregated: 3 (runs\seed1\eval_val.json, runs\seed2\eval_val.json, runs\seed3\eval_val.json)

| System | Total loss | Attack damage | Benign disruption | Damage reduction | Benign flows disrupted | Mean latency (flows) |
|---|---|---|---|---|---|---|
| none | 76,430 +/- 0 | 76,430 +/- 0 | 0 +/- 0 | 0.0% +/- 0.0% | 0.00% +/- 0.00% | n/a |
| rule | 76,840 +/- 0 | 76,430 +/- 0 | 390 +/- 0 | 0.0% +/- 0.0% | 0.99% +/- 0.00% | n/a |
| rule_tuned | 22,349 +/- 0 | 15,626 +/- 0 | 6,430 +/- 0 | 79.6% +/- 0.0% | 16.27% +/- 0.00% | 93.7 +/- 0.0 |
| v2_marl | 3,603 +/- 6,037 | 3,577 +/- 6,028 | 15 +/- 13 | 95.3% +/- 7.9% | 0.03% +/- 0.03% | 8.1 +/- 9.4 |
| v2_single | 3,099 +/- 5,036 | 2,382 +/- 3,842 | 660 +/- 1,139 | 96.9% +/- 5.0% | 1.19% +/- 2.06% | 70.4 +/- 115.6 |
