# Drift detection and recalibration

Implemented components: `certiroute.monitor.drift.score_shift`, the offline replay
`eval.drift`, existing immutable calibration tools, and strong-only gateway override.
This is an executable replay and an operator workflow, not an autonomous production monitor.

## Detect → contain → recalibrate → validate → activate

1. Freeze a reference score sample from the source distribution and declare a monitoring
   horizon. Compare non-overlapping windows with `score_shift(reference, current,
   alpha=0.05, windows=2)`. The statistic is the maximum empirical-CDF difference.
   The alert threshold is `sqrt(log(4*W/alpha)/(2*n_ref)) + sqrt(log(4*W/alpha)/(2*n_window))`.
   This uses two DKW bands and a union bound over W windows. It is conservative with ties.
2. On alert, serve `model: certiroute/strong` (or `x-certiroute-mode: strong`). Record the
   alert and artifact ID. A score-distribution alert is not proof of quality degradation.
   Keep periodic **random labeled audits**, since score drift cannot detect every label shift.
3. Acquire a **new, independent target audit batch** with both models while serving strong.
   Do not reuse the alarm window or choose rows based on failures. Separate acquisition
   cost from serving cost; the cheap shadow calls add cost even during strong fallback.
4. Freeze a `PairedDataset` JSON with current provider/model identities, scope, quality
   metric, source hashes and collection date. Keep the existing predictor frozen unless
   retraining uses another disjoint batch. Build a new immutable calibration artifact:

   ```bash
   python -m eval.calibrate --config configs/deployment.yaml --input data/target-audits.json --out artifacts/target-calibration-v2.json
   ```

   The paths above are operator-supplied production files, not bundled live credentials
   or data. The command validates identity/scope and rejects overlaps with router training.
5. Evaluate on a further held-out target batch. Do not repeatedly tune the threshold on
   this holdout. Record uncertainty and revert to strong-only if the deployment's acceptance
   rule fails. An observed mean below epsilon alone does not prove deployment validity.
6. Point the deployment config at the new artifact and restart the gateway; check `/readyz`
   and response calibration-ID headers. There is no hot-reload/admin activation endpoint
   in this version. Preserve the old artifact/config for rollback. Start a **new, declared
   monitoring horizon**; do not extend a finite-horizon significance claim indefinitely.

## Reproducible abrupt-shift replay

```bash
python -m eval.drift --out results/reproduced/drift
```

The replay uses checksum-validated scores from the existing GSM8K length-shift study.
The router training set and 227-character question-length boundary remain frozen. Both
directions run 1,000 times using seed 42. On each draw, all six 100-row batches are disjoint:
source calibration, source reference, source no-shift control, target detection window,
target labeled audit batch, and target held-out test batch.

| Shift | Detection rate | No-shift alarm rate | Unmitigated regret | After recalibration |
|---|---:|---:|---:|---:|
| Long → short | 100% | 0% | 20.215% | 4.799% |
| Short → long | 100% | 0.1% | 0.039% | 4.874% |

Target regret is **5%**. Detection occurs after one 100-request window. Another 100
paired audits are acquired before reactivation; during that phase the operational
workflow serves strong. Before/after regret is measured on the later disjoint holdout.
The saved report includes conditional intervals, the actual detector-triggered policy
(including missed alerts), and incremental/total audit costs. Serving savings in the
post-recalibration holdout do not include recovery acquisition cost. Delays and costs
are offline replay quantities, not live monitoring measurements.

[Results](../results/drift-v1/results.json) · [Report](../results/drift-v1/REPORT.md)

## Assumptions and remaining work

The DKW bound assumes IID samples within windows/reference; real traffic can be
autocorrelated. Repeated replay partitions reuse a finite pool. The normal Monte Carlo
intervals condition on that pool and use a six-comparison Bonferroni family for two
directions × three arms. They do not capture new-population uncertainty. The target audit
batch is independent of the alarm window; selecting difficult/failing prompts instead
would invalidate the exchangeability argument. Gradual drift, label-only drift, automatic
audit scheduling, long-running monitoring and cross-dataset transfer remain unvalidated.
