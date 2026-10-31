Seeds aggregated: 4 (runs\seed1\eval_val.json, runs\seed2\eval_val.json, runs\seed3\eval_val.json, runs\seed4\eval_val.json)

| System | Total loss | Attack damage | Benign disruption | Damage reduction | Benign flows disrupted | Mean latency (flows) |
|---|---|---|---|---|---|---|
| none | 76,430 +/- 0 | 76,430 +/- 0 | 0 +/- 0 | 0.0% +/- 0.0% | 0.00% +/- 0.00% | n/a |
| rule | 76,840 +/- 0 | 76,430 +/- 0 | 390 +/- 0 | 0.0% +/- 0.0% | 0.99% +/- 0.00% | n/a |
| rule_tuned | 22,349 +/- 0 | 15,626 +/- 0 | 6,430 +/- 0 | 79.6% +/- 0.0% | 16.27% +/- 0.00% | 93.7 +/- 0.0 |
| v2_marl | 355 +/- 358 | 304 +/- 376 | 33 +/- 42 | 99.6% +/- 0.5% | 0.08% +/- 0.11% | 26.0 +/- 45.8 |
| v2_single | 1,826 +/- 3,215 | 1,796 +/- 3,218 | 1 +/- 2 | 97.7% +/- 4.2% | 0.01% +/- 0.01% | 21.6 +/- 31.0 |
