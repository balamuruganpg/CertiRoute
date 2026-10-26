# Calibration-size and seed sensitivity

The [consolidated report](../results/gsm8k-sensitivity/REPORT.md) extends the
single-training-partition study with calibration sizes 50, 100, 200 and 500,
five training partitions, five independent random rankings and a no-CRC
negative control. It uses the same 1,319 frozen model-response pairs and the
same seven-feature logistic predictor. No features, optimization settings,
epsilon values, pricing assumptions or quality labels were tuned for this run.

## Observed results

At epsilon 0.05, learned CRC has the following mean results across training seeds:

| Calibration examples | Mean regret | Estimated savings vs strong | Random CRC savings |
|---:|---:|---:|---:|
| 50 | 0.03900 | 15.78% | 12.68% |
| 100 | 0.05001 | 20.49% | 16.01% |
| 200 | 0.04976 | 20.71% | 15.99% |
| 500 | 0.05008 | 21.06% | 16.09% |

At size 500, the learned savings advantage averages 4.977 percentage points,
with a range of 0.333 to 8.531 across all 25 training/ranking-seed pairs.
All 25 are positive; 23 have an adjusted paired interval entirely above zero.
The result is not uniform across settings: at size 200 and epsilon 0.05 one
pair loses by 0.053 points, and tighter epsilon settings have more losing pairs.
These compare the same target epsilon, not exactly equal attained risk.

All 80 learned-CRC cells meet the adjusted non-rejection criterion; the no-CRC
control is rejected in 70/80 cells. At size 50 and epsilon 0.01, CRC routes
everything to strong and saves nothing. Larger calibration sets reduce this
conservatism, but do not imply monotonic savings at every epsilon or seed.
These finite-pool results do not prove a deployment guarantee. Individual
splits can exceed epsilon, costs remain estimates, and the overlapping seed
partitions are not independent new datasets.

## Reproduce offline

From the repository root, with dependencies already installed:

```powershell
.\.venv\Scripts\python -m eval.sensitivity --out results/my-sensitivity
```

Then use a Python environment with Matplotlib to render the saved results:

```powershell
python -m eval.plot_sensitivity --input results/my-sensitivity/results.json --out results/my-sensitivity/plots
```

The analysis uses the existing project environment and no new dependencies.
Plotting is separate so Matplotlib is not a gateway/runtime dependency. Its
versions are pinned in `requirements-plot.lock` and the optional `plots` extra.
Both stages are fully offline once their dependencies are available. This run
used the existing project Python 3.13.14 for analysis and the already installed
Python 3.11 plotting environment, Matplotlib 3.10.8 and NumPy 2.2.6. No packages,
models, embeddings or data were downloaded. Plot provenance records the actual
interpreter, libraries, script and input hash.

Choose new output directories: saved results and plots are not overwritten.
The study settings live in `configs/study.sensitivity.yaml`; `--study` selects
another explicit configuration. Dataset, gateway config and price paths also
have CLI overrides. Run from the repository root so the `eval` modules resolve.

## Predeclared experiment

| Setting | Values |
|---|---|
| Training-partition seeds | 17, 42, 73, 101, 137 |
| Random-ranking seeds | 0, 1, 2, 3, 4 |
| Calibration sizes | 50, 100, 200, 500 |
| Test questions per draw | 264 |
| Draws per training partition | 1,000 |
| Epsilons | 0.01, 0.02, 0.05, 0.10 |
| Learner | Existing logistic classifier, 1,500 iterations, learning rate 0.1, L2 0.01 |

For each training seed, the existing prompt-hash assignment selects 527 examples
for training. That seed changes the data partition, not a stochastic optimizer:
the learner uses deterministic full-batch gradient descent. Its feature scaler
and weights are fitted once on those examples, before scoring the remaining
792-question pool. Training never uses calibration/test outcomes.

Each draw independently permutes the pool. The last 264 questions form the test
set; the first 528 are calibration candidates. The four calibration sets use
nested prefixes of lengths 50, 100, 200 and 500. Unused candidates are excluded
from that cell, rather than added to test. Thus test questions and test-set size
remain identical across all sizes and policies within a draw, supporting paired
comparisons without confounding calibration size with test size. The same
random ranking of a question is reused across draws and training seeds.

The total is 5,000 partition draws, 20,000 training-seed/size/draw combinations
and 560,000 policy/epsilon measurements. Those measurements are **not** 560,000
independent data samples: policies, nested calibration sizes and training
partitions reuse data.

## Policies

- **CRC:** the learned score and the production CRC function.
- **No-CRC control:** the same learned score, candidates and strict routing rule,
  selecting the smallest threshold satisfying `sum(regret)/n <= epsilon`.
  This removes the `+1` and `n+1` correction and carries no risk guarantee.
- **Random CRC:** five label-independent hash rankings, each calibrated with the
  production CRC function on exactly the same calibration questions.

The negative control exists only in evaluation code. It does not change the
gateway or generate a calibrated serving artifact. Always-cheap, always-strong
and fixed `score > 0.5` baselines are also recorded. A cached sorted test curve
computes metrics efficiently while preserving strict `>` and tied scores;
tests compare it against direct request-by-request accounting.

## Statistics and interpretation

Per-cell results include mean regret, quality, estimated cost, savings,
escalation, risk quantiles, risk variance, threshold variance and the fraction
of draws exceeding epsilon. Approximate normal Monte Carlo intervals use
**whole draws** as observations, conditional on each frozen predictor, pool
and ranking. They do not quantify uncertainty about new datasets or providers.

The risk-interval Bonferroni family includes all training seeds, ranking seeds,
calibration sizes, epsilon values and policies: 560 cells in this run. A
separate family of 400 paired savings intervals compares learned CRC with each
random ranking on identical draws. Non-rejection means the adjusted lower
risk bound is at most epsilon; it is not proof that the bound holds.

The consolidated tables average training-seed means equally. Random-CRC rows
first average the five rankings within each training seed. Reported ranges
are descriptive minima/maxima of seed means, **not pooled confidence intervals**.
The learned-versus-random comparison reports all 25 training/ranking-seed pairs
per calibration-size/epsilon combination, including negative differences and
intervals crossing zero. Equal epsilon need not mean equal attained regret.

The plots show:

1. Mean regret versus calibration size, with the epsilon reference and shaded
   ranges of training-seed means.
2. Risk versus estimated cost relative to always-strong; all predeclared epsilon
   operating points are shown, plus the random-mixture reference line.
3. Savings advantage of the learned ranking for every training seed and size,
   averaged over the five random rankings, separately for each epsilon.

The no-CRC control may spend more risk and therefore save more cost. Its savings
must be read alongside its realized regret. The study does not choose a new
configuration based on favorable outcomes or promote any result to production.

## Artifacts and limitations

`results.json` contains configurations, provenance, all conditional intervals,
per-cell and aggregate statistics and paired comparisons. `draws.jsonl.gz`
records every partition checksum, threshold and metric in the order specified
by `draw_columns`; deterministic compression permits checksum comparison.
Each `seed-*/` directory contains the training rows, frozen learned weights,
pool scores, random rankings and ordered train/pool IDs. These are research
artifacts, not newly deployed gateway calibrations.

Plots are saved in PNG, SVG and PDF, with checksums in `plots/provenance.json`.
Timestamps and absolute output paths can change; fitted weights, partitions,
compressed measurements and aggregate metrics reproduce under the same inputs,
code and environment. The old public pilot and G1 result directories remain
unchanged.

The benchmark has been used in earlier experiments; this is sensitivity
analysis rather than a pristine new holdout. Overlapping seed partitions do not
create new independent datasets. Five training partitions and five rankings do
not establish universal robustness. Model generation metadata remains partly
unknown, token counts use character-based estimates, and costs use the same
historical scenario. Benchmark contamination and semantic duplicates are still
possible. No live traffic, drift, provider latency or billed cost is measured.
