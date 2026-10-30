# Multi-benchmark CRC study

Frozen real pairs; historical modeled costs, not bills. Target regret: 5%.

| Task | Policy | Quality | Regret | USD/1k | Savings |
|---|---|---:|---:|---:|---:|
| gsm8k | cheap-only | 63.83% | 29.109% | 0.261 | 98.03% |
| gsm8k | strong-only | 85.83% | 0.000% | 13.266 | 0.00% |
| gsm8k | CertiRoute | 83.92% | 4.728% | 10.445 | 21.27% |
| arc | cheap-only | 83.69% | 13.146% | 0.021 | 97.65% |
| arc | strong-only | 95.60% | 0.000% | 0.911 | 0.00% |
| arc | CertiRoute | 90.74% | 5.030% | 0.530 | 41.78% |
| mbpp | cheap-only | 82.08% | 5.392% | 0.280 | 91.61% |
| mbpp | strong-only | 81.17% | 0.000% | 3.343 | 0.00% |
| mbpp | CertiRoute | 81.81% | 4.549% | 0.805 | 75.93% |
| equal-task portfolio | cheap-only | 76.53% | 15.882% | 0.188 | 96.79% |
| equal-task portfolio | strong-only | 87.53% | 0.000% | 5.840 | 0.00% |
| equal-task portfolio | CertiRoute | 85.49% | 4.769% | 3.927 | 32.77% |

40% training; remaining 60% split equally into calibration/test on each of 1,000 seeded draws. Full epsilon grid, random-ranking CRC, fixed threshold and random mixtures are in results.json. Costs per request are averaged before portfolio savings are calculated. See docs/MULTIBENCH.md for provenance, scoring, split rules and limitations.
