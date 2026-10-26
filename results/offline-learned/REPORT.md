# Offline fixture quality–cost comparison

Synthetic outputs, invented token counts and hypothetical prices. No external API calls.

Seed 42; train/calibration/test = 96/96/48.

| Policy | Epsilon | Quality | Mean regret | Hypothetical USD | Savings vs strong | Strong fraction |
|---|---:|---:|---:|---:|---:|---:|
| cheap-only | — | 58.33% | 0.4167 | 0.00006840 | 90.45% | 0.00% |
| strong-only | — | 100.00% | 0.0000 | 0.00071600 | 0.00% | 100.00% |
| CertiRoute | 0.01 | 100.00% | 0.0000 | 0.00071600 | 0.00% | 100.00% |
| CertiRoute | 0.02 | 100.00% | 0.0000 | 0.00043880 | 38.72% | 54.17% |
| CertiRoute | 0.05 | 97.92% | 0.0208 | 0.00042620 | 40.47% | 52.08% |
| CertiRoute | 0.1 | 97.92% | 0.0208 | 0.00042620 | 40.47% | 52.08% |

One small held-out split is descriptive; it does not validate the CRC guarantee.
The epsilon grid was fixed before fitting. All operating points are reported.
With n=96, epsilons below 0.010309 force always-strong.
Costs cover the selected generation only; simulated paired collection cost is not serving cost.
No real provider cost, latency, or quality was measured. See results.json for full provenance.
