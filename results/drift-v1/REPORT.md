# Drift detection and recalibration replay

| Shift | Detection | No-shift alarms | Before regret | Recalibrated regret |
|---|---:|---:|---:|---:|
| long-to-short | 100.0% | 0.0% | 20.215% | 4.799% |
| short-to-long | 100.0% | 0.1% | 0.039% | 4.874% |

Target: 5%. An alert is evaluated after 100 requests; acquire 100 further paired audits while serving strong, then evaluate on 100 disjoint target requests. All splits, costs, detector-policy outcomes and adjusted intervals are saved. See docs/DRIFT.md; a missed alert does not trigger recalibration.
