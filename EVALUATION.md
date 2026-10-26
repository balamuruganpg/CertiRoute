# CertiRoute — Evaluation Plan

> This document defines the evaluation plan. The [expanded GSM8K study](docs/GSM8K_REPEATED.md) implements G1 on the available finite pool; the [sensitivity study](docs/GSM8K_SENSITIVITY.md) adds G2/G6 comparisons; and the [within-GSM8K shift study](docs/GSM8K_SHIFT.md) tests transfer and target recalibration. Larger-data, cross-dataset shift and live-serving studies remain incomplete. Every reported number must come from a saved, reproducible result file. Where this plan states a *target*, it is a pass/fail criterion we set in advance, not a prediction.

## 1. Claims under test

| ID | Claim | How it is tested |
|---|---|---|
| C1 | **Validity:** with CRC-calibrated λ̂, expected regret vs always-strong is ≤ ε | Repeated random calibration/test splits (§6, G1–G4) |
| C2 | **Efficiency:** at the same ε, CertiRoute saves more cost than baselines | Cost–quality Pareto and cost-at-ε tables (§7) |
| C3 | **Robustness:** we know when and how the guarantee fails under shift, and mitigation helps | Shift experiments (§8) |
| C4 | **Production cost:** the gateway adds small, bounded overhead and fails safe | Load and failure tests (§9) |
| C5 | **Monitorability:** drift is detected and recalibration restores risk control | Simulated drift with audits (§8, S3) |

## 2. Setup

- **Model pair:** one cheap, one strong model, chosen in `configs/models.yaml`. Record exact provider, model id, date, and decoding params in every result file. Model identity is part of the artifact.
- **Decoding:** temperature 0, fixed `max_tokens` and prompt templates for calibration and evaluation. If the provider is non-deterministic at temperature 0, run each prompt once and treat residual noise as part of the data (document it); optionally measure it by re-running a 5% subsample.
- **Quality `q ∈ [0,1]`:**
  - Verifiable tasks: exact match / normalized numeric match (GSM8K, MMLU, ARC), unit-test execution (HumanEval/MBPP-style, in a sandbox).
  - Open-ended tasks (stretch): pairwise judge. Judge is validated against ground-truth tasks first (agreement, position bias, length bias), and a second judge family is used for a robustness check.
- **Loss:** `regret = max(0, q_strong − q_cheap)`; `L(λ) = regret · 1{s > λ}`; `B = 1`.
- **Cost accounting:** tokens × `pricing.yaml` (with `as_of` date). Report the cost of the *routed* policy relative to always-strong. Use token-based cost so results are reproducible; report billed spend only for the live demo.

## 3. Datasets

| Dataset | Type | Quality metric | Role |
|---|---|---|---|
| MMLU (multi-subject) | Multiple choice | Exact match | Router train, calibration, test; shift source |
| GSM8K | Grade-school math | Numeric match | Router train, calibration, test |
| ARC-Challenge | Science MCQ | Exact match | Shift target (easy→hard science) |
| MATH (subset) | Competition math | Numeric/symbolic match | Shift target (harder) |
| HumanEval / MBPP (subset) | Code | Test execution | Shift target (new modality) |
| Open-ended chat set (stretch) | Real-style prompts | Pairwise judge | "Realistic traffic" experiment; mind dataset licence/gating |

Notes:

- **Leakage control:** deduplicate across router/calibration/test splits with exact + near-duplicate (n-gram/fuzzy/embedding) matching. Verascan can be reused for this check, which doubles as a cross-project integration story.
- **Benchmark contamination of the LLMs** can inflate absolute accuracy, but the guarantee is about *relative* regret; we still report it as a threat to validity (§12).
- **Size planning (assumption, adjust to budget):** several thousand prompts total. As a sizing rule, token volume ≈ (#prompts × avg input tokens) and (#prompts × avg output tokens) *per model*; multiply by current prices before committing. Calibration needs `n ≳ B/ε` at minimum; aim for ≥ 1,000 calibration prompts per distribution of interest.

## 4. Data collection protocol

1. Build the prompt pool; apply the dedup/split guard.
2. For each prompt run **both** models; store raw outputs, token counts, latency, finish reason, and `q` for each. Cache on `hash(prompt, model, params)`; make the job resumable.
3. Compute `regret_i`.
4. Freeze the dataset (hash it); record the hash in all downstream artifacts.

## 5. Splits

- Split by prompt hash into `D_router` (≈ 40%) and `D_pool` (≈ 60%).
- The router is trained **only** on `D_router`.
- Each guarantee experiment draws many random `(D_cal, D_test)` partitions from `D_pool`, scores them with the frozen router, and applies CRC on `D_cal`.
- For shift experiments, calibration comes from source domain(s) and test from held-out domain(s) (§8).

## 6. Guarantee testing methodology

Because all outcomes (`s_i`, `regret_i`) are precomputed, thousands of splits cost seconds.

### G1 — Main validity test

```
for r in 1..R (R = 1000+):
    shuffle D_pool with seed r
    D_cal  <- first n_cal points; D_test <- next n_test points
    lam    <- crc_lambda(s[D_cal], regret[D_cal], eps)
    risk_r <- mean(regret[D_test] * (s[D_test] > lam))
    cost_r <- routed cost on D_test
report mean(risk_r) with 95% CI, quantiles of risk_r, mean(cost_r)
```

- **Pass criterion:** for each ε in the grid, the lower end of the 95% CI of mean risk is ≤ ε (we cannot reject validity). With several ε values, apply a Bonferroni correction to the CI level.
- CRC controls the **mean** over calibration draws, so individual splits may exceed ε. That is expected and is *not* a failure. We report the fraction of splits with `risk_r > ε` as an honest descriptive statistic.
- What G1 validates: the implementation and the finite-sample behavior under exchangeability-by-construction. It does **not** validate real-world exchangeability; §8 and the live audits address that.

Implemented finite-pool study: `python -m eval.repeated --out results/new-study`.
The router is trained once and fixed; 1,000 independent partitions reuse the
remaining finite pool. Normal Monte Carlo intervals treat whole partitions as
observations, with Bonferroni adjustment across the epsilon grid. These approximate
intervals are conditional on this pool and training set, not confidence intervals
for a newly sampled population. See the linked study for the exact protocol,
limitations, risk quantiles and per-split outcomes.

### G2 — Sample-size sweep

The [implemented sensitivity study](docs/GSM8K_SENSITIVITY.md) covers the feasible
50/100/200/500 sizes on this dataset, with a fixed test set within each draw,
multiple training/ranking seeds and the empirical-threshold G6 negative control.
Larger calibration sizes below require more paired data.

`n_cal ∈ {50, 100, 200, 500, 1000, 2000}`, ε fixed. Show: mean risk stays ≤ ε; variance of realized risk and of λ̂ shrinks with `n_cal`; savings rise with `n_cal` as the `B/(n+1)` slack and the variance penalty fall. Also show the minimum-n behavior: for tiny `n`, λ̂ → 1 (always strong), i.e. safe but no savings.

### G3 — ε sweep (reliability diagram)

ε ∈ {0.005, 0.01, 0.02, 0.05, 0.10, 0.20}. Plot target ε (x) vs mean realized risk (y) with CI and the `y = x` line. Expect points on or below the line. A large gap below the line quantifies conservativeness.

### G4 — Tightness

Compare mean realized risk to ε. CRC is typically close to ε from below for continuous-ish losses; with binary regret and discrete scores it can be more conservative. Report the gap and have the theory partner relate it to the known `O(1/n)` slack, noting the continuity assumptions under which tightness results hold.

### G5 — High-probability comparison (stretch)

Implement a Learn-then-Test / Hoeffding–Bentkus-style bound `P(R(λ̂) ≤ ε) ≥ 1 − δ`. Report the fraction of splits with realized risk ≤ ε (should be ≥ 1 − δ), and the extra cost (lost savings) compared with CRC. This is a nice "expectation vs high-probability" figure for the report.

### G6 — Negative controls (the guarantee is not magic)

| Control | Expected behavior |
|---|---|
| Fixed threshold `s > 0.5`, no calibration | No guarantee; risk varies with dataset and model pair |
| Threshold tuned on validation to hit ε empirically (no conformal correction) | Realized risk exceeds ε in a substantial fraction of splits, especially at small `n` |
| Calibrate on one domain, test on another (no mitigation) | Violation when the target is harder than the source |
| Router trained on `D_cal` (leakage) | Optimistic λ̂ and violation: demonstrates why D4 in `ARCHITECTURE.md` matters |

### G7 — Loss variants

Binary regret vs continuous (judge score gap) vs asymmetric weighting. Check monotonicity and boundedness hold; compare savings.

### G8 — Label-noise sensitivity

Inject noise into `q` that mimics judge error (symmetric flips with rate η; or judge-specific confusion estimated on a labeled subset). Plot *true* risk (ground truth) vs ε for calibration done on noisy labels. This quantifies how judge error erodes the guarantee, and is a good theory question (`SPLIT.md`, T3).

### G9 — End-to-end live replay

Replay a held-out prompt stream through the **running gateway** with audits at a high rate (for the experiment). Compare the audited realized regret with the offline estimate and with ε. This ties the offline statistics to the deployed system. Cache disabled.

## 7. Efficiency evaluation

### Baselines

| Baseline | Description | Gets a guarantee? |
|---|---|---|
| Always-strong | Upper-quality, highest cost | n/a |
| Always-cheap | Lowest cost, lowest quality | n/a |
| Random mix | Route cheap with probability `p`; traces a straight line between endpoints. The router must beat this clearly | No |
| Confidence threshold | Route on a simple confidence proxy (prompt length or cheap-model logprob if available) with threshold tuned on validation | No |
| RouteLLM-style router | Preference/outcome-trained classifier with threshold tuned on validation; reimplemented on our data or the public implementation if licences/versions allow | No |
| Uncalibrated own router | Our router with a validation-tuned threshold (no CRC) | No |
| Oracle router | Uses true regret; routes cheap iff regret = 0. Upper bound on savings | n/a |
| FrugalGPT-style cascade (stretch) | Cheap first, escalate on low confidence; non-streaming | No |

### Metrics

- **Realized risk** (mean regret vs always-strong) and **absolute quality** (mean `q` of the routed policy).
- **Cost savings** vs always-strong, and **cost at fixed ε**.
- **Escalation rate.**
- **Router quality:** AUC/PR of `s(x)` predicting `regret = 0`; ECE of `s(x)`; and, more relevant, the *savings–risk curve* it induces.
- **Pareto area:** area under the cost–quality frontier for each method.
- For each method with a tunable threshold, trace the whole curve, then mark the operating points chosen for ε ∈ {0.01, 0.02, 0.05}.

Report **quality–cost frontier** (figure) and **cost at risk ≤ ε** with its realized risk across splits (table) so baselines that miss ε are visibly penalized.

### Ablations

- Features: embedding only vs + surface features vs + cheap-model probe (stretch).
- Predictor: logistic vs LightGBM vs small MLP.
- Embedding model size.
- `n_cal`, `n_router`.
- Domain mix of calibration data.
- Audit rate vs recalibration quality (§8).

## 8. Shift and monitoring experiments

Implemented first increment: [question-length covariate shift within GSM8K](docs/GSM8K_SHIFT.md).
The boundary is fixed from training prompts; both directions compare source
controls, unmitigated transfer and disjoint target recalibration over 1,000 draws.
The report separates observed mean regret <= epsilon from the adjusted
non-rejection criterion. Long-to-short transfer rejects at all four epsilons;
the reverse direction is conservative. This supports G6 and C3 but does **not**
complete S1's cross-dataset scope or C5/S3's online detection/audit requirements.

| ID | Scenario | Setup | Measures |
|---|---|---|---|
| S1 | Domain shift | Calibrate on MMLU+GSM8K; test on MATH, ARC, code | Realized risk vs ε, direction of failure (harder target ⇒ violation) |
| S2 | Mixture shift | Same domains, changed proportions | Risk vs mixture; effect of global λ |
| S3 | Gradual drift | Stream whose difficulty ramps up over time; audits at rate `p` | Time-to-detect, risk over time, cost of mitigation |
| S4 | Model update | Swap cheap-model version | Artifact invalidation works; risk after recalibration |

**Mitigation arms (compare on S1–S3):**

1. None (negative control).
2. Detect drift → recalibrate on recent audits (MVP).
3. Weighted CRC with density-ratio weights from a domain classifier (stretch).
4. Online threshold update targeting long-run risk ε (stretch).

**Drift detection metrics:** detection delay, false-alarm rate on no-shift streams, power of PSI/KS on `s(X)` vs embedding-space tests vs rolling audited-regret CI.

**Audit-rate study:** vary `p ∈ {0.5%, 1%, 2%, 5%}`; report recalibration quality, extra spend, and detection delay.

## 9. Systems evaluation

| Test | Method | Reported |
|---|---|---|
| Router overhead | Microbenchmark features + predictor, cold/warm, CPU | p50/p95/p99 ms |
| Gateway overhead | Locust against a **mock upstream** with fixed latency/streaming cadence | Added latency p50/p95/p99, TTFT overhead, throughput at concurrency 1/16/64/256 |
| Real-upstream run | Same profile against real providers | End-to-end TTFT/latency by tier, error rates |
| Streaming correctness | Compare chunk sequence and `[DONE]` through gateway vs direct | Byte-level equality of content deltas |
| Failure injection | Upstream timeout, 429, 5xx, mid-stream disconnect; corrupt/missing artifact; DB unavailable | Correct fallback tier, tags, no crashes, readiness state |
| Soak | 30–60 min constant load | Memory growth, connection leaks, queue backlog |

## 10. Reproducibility

- `make reproduce` runs from the frozen dataset to all figures/tables. Collection from APIs is separate (`make collect`) because it costs money and may not be bit-reproducible; the frozen paired-output dataset (or its hash and a download script, subject to licences) is the reproducibility anchor.
- All random seeds are in configs; each result JSON stores config, code git SHA, dataset hash, artifact id.
- Figures are generated by scripts, never by hand.

## 11. Success criteria (targets set in advance)

**Must pass (MVP):**

1. G1: mean realized risk consistent with ≤ ε at every ε in the grid (CI criterion above), on real model outputs.
2. G6: at least two negative controls demonstrably violate ε, showing the test can fail.
3. C2: CertiRoute's cost at ε dominates the random-mix baseline at every ε, and is reported honestly against the RouteLLM-style and confidence baselines.
4. C4: streaming passthrough correct; failure-injection suite passes; overhead numbers published with methodology.
5. S1 run and reported even if (especially if) the guarantee fails.

**Should pass:**

6. Recalibration (S3) restores risk ≤ ε after detected drift with a stated audit budget.
7. G8 shows a measured relationship between judge noise and true-risk inflation.

**Could pass (stretch):** G5 high-probability variant; weighted/online mitigation arms; open-ended-task experiment with judge.

If savings turn out small on some model pair, we report that. A small-savings result with a clean guarantee is a legitimate finding, and we also show the oracle bound so readers can see how much headroom the router leaves.

## 12. Hypotheses (to be tested, not claimed)

- Savings should be largest on benchmarks where the cheap model is strong relative to the strong model (for example easier MCQ/math), and near zero where it is weak (harder math, code).
- A better-ranking router should increase savings at the same ε without affecting validity.
- At small `n_cal` or small ε, CRC should be visibly conservative (λ̂ near 1).
- Calibrating on easy domains and testing on harder ones should violate ε unless mitigated.
- Judge noise should usually push true risk above the nominal ε in a measurable way.

## 13. Threats to validity

- **Exchangeability** in real traffic is an assumption; we test violations but cannot prove it.
- **Quality-metric validity:** exact match can penalize correct but differently formatted answers; judges have biases. Normalize outputs, validate judges, report both.
- **Benchmark contamination** of the evaluated models.
- **Provider nondeterminism** and silent model updates; record dates and ids.
- **Finite pool:** offline splits resample a fixed pool, so they measure finite-population behavior; the live replay (G9) and shift tests complement this.
- **Single model pair:** results may not transfer; a second pair is a stretch goal.
- **Selection of ε:** the grid is fixed before looking at results.

## 14. Deliverables from this plan

- Figures: F1 cost–quality Pareto; F2 target-vs-realized risk (reliability); F3 distribution of risk across splits; F4 `n_cal` sweep; F5 shift timeline with/without mitigation; F6 latency CDFs.
- Tables: T1 cost at ε vs baselines (with realized risk and violation fraction); T2 shift results; T3 overhead; T4 ablations.
- A results section in `README.md` and a 6–8 page technical report built from the same result files.
