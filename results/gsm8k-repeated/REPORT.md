# GSM8K repeated-split study

1000 independent partitions; sizes: {'train': 527, 'pool': 792, 'calibration': 528, 'test': 264}.
One frozen logistic surface-feature predictor; no tuning on calibration/test outcomes.
Real cached outputs; estimated tokens and historical modeled costs; zero API calls.

## Calibrated policies

CI is Bonferroni-adjusted Monte Carlo uncertainty for mean regret, conditional on the pool.
Non-rejection follows EVALUATION.md; it is not evidence of deployment safety.

| Policy | Epsilon | Mean regret | Adjusted CI | Splits > epsilon | Accuracy | Savings | Strong fraction | Non-rejection |
|---|---:|---:|---|---:|---:|---:|---:|---|
| CertiRoute | 0.01 | 0.00951 | [0.00894, 0.01008] | 42.4% | 85.90% | 4.32% | 95.12% | True |
| CertiRoute | 0.02 | 0.01859 | [0.01780, 0.01939] | 39.2% | 85.53% | 7.57% | 91.61% | True |
| CertiRoute | 0.05 | 0.04881 | [0.04752, 0.05011] | 42.5% | 83.86% | 21.69% | 76.40% | True |
| CertiRoute | 0.1 | 0.09722 | [0.09543, 0.09902] | 42.7% | 80.15% | 38.22% | 58.78% | True |
| random-score-CRC | 0.01 | 0.00952 | [0.00894, 0.01011] | 43.2% | 85.21% | 3.79% | 96.16% | True |
| random-score-CRC | 0.02 | 0.01900 | [0.01818, 0.01982] | 39.2% | 84.59% | 6.84% | 93.03% | True |
| random-score-CRC | 0.05 | 0.04916 | [0.04792, 0.05040] | 42.5% | 82.44% | 16.78% | 83.00% | True |
| random-score-CRC | 0.1 | 0.09789 | [0.09607, 0.09972] | 44.2% | 79.18% | 32.49% | 67.23% | True |

## Uncalibrated baselines

Random mixes below are exact expectations over routing coins, not calibrated policies.
The fixed-threshold baseline routes cheap iff score > 0.5.

| Policy | Cheap probability | Mean regret | Accuracy | Savings |
|---|---:|---:|---:|---:|
| cheap-only | — | 0.29147 | 63.80% | 98.03% |
| strong-only | — | 0.00000 | 85.87% | 0.00% |
| fixed-threshold | — | 0.28258 | 64.69% | 96.63% |
| random-mix-expected | 0.1 | 0.02915 | 83.66% | 9.80% |
| random-mix-expected | 0.25 | 0.07287 | 80.35% | 24.51% |
| random-mix-expected | 0.5 | 0.14574 | 74.83% | 49.02% |
| random-mix-expected | 0.75 | 0.21861 | 69.32% | 73.52% |
| random-mix-expected | 0.9 | 0.26233 | 66.00% | 88.23% |

## Learned versus random ranking

Both rankings use CRC at the same target epsilon, not necessarily the same attained risk.
Random-score CRC uses a fixed label-independent hash ranking; it does not fit labels.
Positive advantage means greater savings for the learned router on paired partitions.

| Epsilon | Savings advantage (percentage points) | Adjusted paired CI (pp) |
|---|---:|---|
| 0.01 | 0.532 | [0.366, 0.697] |
| 0.02 | 0.732 | [0.480, 0.984] |
| 0.05 | 4.910 | [4.523, 5.297] |
| 0.1 | 5.735 | [5.296, 6.175] |

## Provenance and interpretation

Approximate normal Monte Carlo confidence intervals over independent partitions conditional on this fixed training set and finite pool. Bonferroni across the epsilon grid separately for each CRC policy and paired-comparison family. These intervals exclude uncertainty from sampling new datasets or retraining.

All first-draw CRC thresholds and routing decisions agree exactly; baseline/CertiRoute metrics agree within 1e-12.

Every draw, threshold and partition checksum is in draws.jsonl; risk quantiles, exceedance fractions, mean total test costs, configurations and hashes are in results.json.
Partitions can be reconstructed from split-manifest.json and the pinned algorithm.

Models: https://replay-cheap.invalid/v1::mistralai/Mixtral-8x7B-Instruct-v0.1; https://replay-strong.invalid/v1::gpt-4-1106-preview.
Prompt: Single user message: first 8 GSM8K training examples formatted as 'Question: {question}\nAnswer: {answer}\n\n', then 'Question: {test_question}\nAnswer:'. Full text is in every row.
Decoding metadata: `{"temperature": 0, "max_tokens": 1024, "stop": ["Question"], "top_p": "Not explicit upstream; replay uses 1 (assumed default).", "provider": "Upstream SGLang OpenAI-compatible router; original host not recorded.", "generated_at": null, "finish_reason": null, "latency": null, "provenance": "Settings/model IDs from pinned generation script; original wire requests, provider revisions and generation dates unavailable."}`.
Pricing: Historical RouteLLM paper scenario. Mixtral input/output assumption $0.24/M; GPT-4-1106 input $10/M, output $30/M. Token usage is estimated from text; these are modeled generation costs, not measured bills or current prices.
Pricing source: https://arxiv.org/html/2406.18665v1#A4

Limitations:

- Real frozen model outputs and recomputed correctness; estimated tokens and costs.
- Original generation date, provider deployment and exact top_p are not recorded. These artifacts are bound to .invalid replay endpoints, not live deployment.
- Question-only lexical deduplication does not detect all semantic duplicates. This is one finite public benchmark, not a representative sample of live traffic.
- Metric copies RouteLLM's last unsigned integer rule, including its limitations for decimals, signs and trailing numbers; it is not a general math verifier.
- GSM8K may be in LLM pretraining data; benchmark contamination is unmeasured.
- Fewer than 1000 calibration examples; this implements G1 on the available pool, not the planned larger-data study or the other G2-G9 experiments.
- Non-rejection (adjusted lower CI <= epsilon) is not proof of validity. No live exchangeability or production guarantee is established.
- Random-mix rows integrate Bernoulli routing analytically: their exceedance rates refer to conditional expected regret on each split, not sampled routing coins.
- The old pilot has already exposed some of these benchmark rows; new features and hyperparameters were not tuned here. This is not a pristine new holdout.
- The random-score ranking uses one fixed seed; intervals do not include variation across alternative random rankings. Normal Monte Carlo CIs are approximate.
