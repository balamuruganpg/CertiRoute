# Expanded GSM8K and repeated-split study

The follow-up [calibration-size and seed sensitivity study](GSM8K_SENSITIVITY.md)
adds four calibration sizes, five training partitions, five random rankings,
the no-CRC negative control and plots. This page preserves the original G1 run.

The [saved report](../results/gsm8k-repeated/REPORT.md) covers all 1,319 usable frozen
RouteLLM GSM8K pairs. It implements the G1 repeated-partition protocol on the
available finite benchmark pool, with random-routing and fixed-threshold baselines.
It does not complete EVALUATION.md's full guarantee, shift or live-serving suite.

## Reproduce entirely offline

From the repository root with the existing environment:

```powershell
.\.venv\Scripts\python -m eval.repeated --out results/my-gsm8k-study
```

Use a new output directory. Defaults are the included `data/eval/gsm8k-full/paired.jsonl`,
its `gateway-template.yaml`, `configs/pricing.gsm8k.yaml` and
`configs/study.gsm8k.yaml`. No source downloads, keys, model calls, embedding
dependencies or tokenizer downloads are needed. The same pinned environment in
`requirements-eval.lock` applies. Use `--input`, `--config`, `--pricing` and `--study`
to explicitly select other saved inputs. The new single-split path also works:

```powershell
.\.venv\Scripts\python -m eval.offline --input data/eval/gsm8k-full/paired.jsonl --config data/eval/gsm8k-full/gateway-template.yaml --pricing configs/pricing.gsm8k.yaml --out results/my-full-single-split
```

## What changed

The old 100-example pilot compared full prompts for duplicates. Their shared
eight-example prefix made unrelated short questions look alike. The expanded
snapshot removes only that exact prefix when making duplicate fingerprints,
then applies the same token-trigram Jaccard threshold of 0.9 to the test question.
All 1,319 questions passed; none were removed. Questions are processed in a
stable hash order, with no selection based on model outputs or correctness.

The prefix is explicit in the dataset manifest and `router.signature_prefix`.
It is also stored in and hashed with learned weights; the loader rejects a
mismatched rule. Fixture loading, fitting, calibration and test isolation all use
the same rule. Missing or altered prefixes fail closed. Existing artifacts with
no prefix retain their original hashing and full-prompt duplicate checks.

Only duplicate checking changes: the full original prompt, few-shot examples,
surface features, logistic learner, training settings, decoding, scoring and price
scenario stay the same. No predictor features or hyperparameters were tuned using
this study's outcomes. The larger snapshot reuses some previously inspected pilot
rows, so it is not a pristine new holdout for future model selection.

## Protocol fixed before the run

- Seed 42; assign 40% by the existing prompt-hash rule to router training: 527 rows.
- Fit the seven-feature logistic predictor once. Freeze it before scoring the 792-row pool.
- Generate 1,000 independent pseudorandom permutations of that pool. Each uses
  528 calibration and 264 test rows, with no overlap inside a draw or with training.
- Select CRC thresholds using calibration rows only, for epsilon 0.01, 0.02, 0.05, 0.10.
- Evaluate frozen scores and paired outcomes; repeat cheaply without provider calls.
- Run the entire first draw through the real FastAPI gateway using its replay
  transport. Verify identical CRC thresholds and routing decisions, and aggregate
  quality, regret, cost and escalation agreement within absolute tolerance 1e-12.

Every partition is reconstructed from the saved ordered pool IDs, seed, draw
index and algorithm; a checksum verifies each reconstruction. Test rows may recur
in different draws by design. The training set never changes across draws.

## Baselines and uncertainty

Always-cheap, always-strong and the learned score's fixed threshold `s > 0.5`
are evaluated on every test partition. Random mixes use cheap probabilities
0.10, 0.25, 0.50, 0.75 and 0.90, fixed before evaluation. Their quality, regret and
cost integrate the Bernoulli routing coins analytically. Their split exceedance
rates therefore concern conditional expected loss, not realized coin flips.

A further baseline uses a label-independent hash ranking with the same CRC
calibration rule. Comparing it with CertiRoute at the same epsilon isolates the
value of the learned ranking more directly. It uses one fixed random ranking;
this study does not average over ranking seeds. Equal target epsilon does not
imply exactly equal attained risk.

The sampling unit for confidence intervals is the **whole partition**. Intervals
use the mean plus/minus a normal quantile times the standard deviation across
1,000 draws divided by sqrt(1000). Both ordinary 95% and Bonferroni-adjusted
intervals are saved. Bonferroni adjustment uses four epsilons, separately for each
CRC policy's risk family and for the paired savings-comparison family. These are
approximate Monte Carlo intervals conditional on the fixed training set, finite
pool and random ranking. They do not quantify uncertainty over newly collected
datasets, retraining, alternative random rankings or production traffic.

G1's non-rejection criterion is the adjusted lower risk bound <= epsilon.
Non-rejection is not proof of the risk bound. A confidence interval extending above
epsilon also prevents claiming that its upper bound confirms the target.

## Results and interpretation

| Epsilon | Mean regret | Adjusted interval | Splits exceeding epsilon | Estimated savings |
|---|---:|---|---:|---:|
| 0.01 | 0.00951 | [0.00894, 0.01008] | 42.4% | 4.32% |
| 0.02 | 0.01859 | [0.01780, 0.01939] | 39.2% | 7.57% |
| 0.05 | 0.04881 | [0.04752, 0.05011] | 42.5% | 21.69% |
| 0.10 | 0.09722 | [0.09543, 0.09902] | 42.7% | 38.22% |

All four settings meet the specified non-rejection criterion on this pool.
Individual splits exceeding epsilon are common: CRC controls an expectation,
not the loss on every test partition. The earlier 20-question pilot exceeding its
target is therefore not, by itself, evidence of a CRC implementation bug. The
first-draw gateway comparison finds no disagreement with the precomputed path.

At epsilon 0.05, random-score CRC saves 16.78%, compared with 21.69% for the
learned ranking: a paired advantage of 4.910 percentage points, with adjusted
interval [4.523, 5.297]. This supports the value of the current predictor on this
fixed pool against this random ranking; it does not establish general superiority.
The fixed `s > 0.5` policy has mean regret 0.28258 despite high savings and does
not achieve any tested risk target. Full baseline curves, all risk quantiles,
exceedance fractions, mean costs and individual draws are saved.

Absolute quality and clipped regret need not add up to strong-only quality:
cheap sometimes answers correctly when strong does not, and clipped regret does
not credit those gains. Mean savings is the mean of per-draw cost-saving ratios.
Costs cover the selected generation on each 264-question test set; they are
estimated, not bills, and exclude hosting and paired-data collection.

Models, reconstructed full prompts, source revisions, source hashes, licenses,
scoring limitations and unavailable generation metadata are preserved from the
[public pilot](GSM8K_PILOT.md). Costs still use character-count token estimates
and the dated RouteLLM paper price scenario. Only 528 calibration examples are
available per draw, below the plan's larger-data target of at least 1,000.
Benchmark contamination and semantic duplicates remain possible. Live traffic
exchangeability, drift, latency and actual billed savings remain untested.

## Saved artifacts and snapshot rebuilding

`results.json` stores aggregates, ordinary/adjusted intervals, paired comparisons,
full configuration and provenance. `draws.jsonl` stores every draw's partition
checksum, thresholds and measurements. `pool-scores.json` holds all frozen pool
scores and outcomes; `split-manifest.json` identifies the fixed training set and
pool order. `router.json` is the frozen predictor. `calibration-artifact.json`,
`calibration.json`, `test.json`, `gateway.yaml` and `gateway-check.json` document
the first draw's integration check; that artifact represents **one draw**, not all
1,000. Timestamps and artifact IDs change on rerun; weights, partition checksums,
draw measurements and aggregate results are deterministic for identical inputs.

To reconstruct the full snapshot from the already cached pinned source files:

```powershell
.\.venv\Scripts\python -m eval.import_gsm8k --all --out data/eval/gsm8k-full-rebuilt
```

Only add `--download` if the source cache is missing and public network access is
wanted. The included full snapshot is sufficient for ordinary offline runs.
