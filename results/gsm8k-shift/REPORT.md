# GSM8K question-length distribution shift

1000 draws, seed 42, 100 calibration and 100 test examples per domain.
Sizes: {'train': 527, 'short': 428, 'long': 364}. Training-only median question length: 227 characters.
Short questions are at or below that boundary; long questions are above it. The eight shared demonstrations are excluded from length, but remain in router features.
One frozen predictor. Both directions, all epsilons and all arms were fixed before inspecting this experiment's outcomes. No threshold uses test labels.

Source-control calibrates and tests on disjoint source examples. Shifted reuses that threshold on target examples. Target-recalibrated calibrates on separate target examples and evaluates on the identical target test set. Unused domain rows remain excluded.

## Empirical results

Regret is clipped per question before averaging. Within target reports the measured mean, without a confidence allowance. Rejected means the adjusted lower CI is above epsilon; this is distinct from one empirical mean or split exceeding epsilon.

| Direction | Arm | Epsilon | Mean regret | Adjusted CI | Within target? | Rejected? | Splits > target | Estimated savings |
|---|---|---:|---:|---|---|---|---:|---:|
| short-to-long | source-control | 0.01 | 0.00999 | [0.00862, 0.01136] | yes | no | 24.6% | 4.70% |
| short-to-long | shifted | 0.01 | 0.00000 | [0.00000, 0.00000] | yes | no | 0.0% | 0.04% |
| short-to-long | target-recalibrated | 0.01 | 0.00939 | [0.00809, 0.01069] | yes | no | 23.2% | 3.74% |
| short-to-long | source-control | 0.02 | 0.01923 | [0.01734, 0.02112] | yes | no | 27.9% | 9.01% |
| short-to-long | shifted | 0.02 | 0.00000 | [0.00000, 0.00000] | yes | no | 0.0% | 0.15% |
| short-to-long | target-recalibrated | 0.02 | 0.01985 | [0.01795, 0.02175] | yes | no | 31.6% | 6.77% |
| short-to-long | source-control | 0.05 | 0.04848 | [0.04551, 0.05145] | yes | no | 35.0% | 22.24% |
| short-to-long | shifted | 0.05 | 0.00039 | [0.00020, 0.00058] | yes | no | 0.0% | 0.95% |
| short-to-long | target-recalibrated | 0.05 | 0.05022 | [0.04719, 0.05325] | no | no | 36.3% | 16.51% |
| short-to-long | source-control | 0.1 | 0.09738 | [0.09324, 0.10152] | yes | no | 38.9% | 44.22% |
| short-to-long | shifted | 0.1 | 0.00422 | [0.00342, 0.00502] | yes | no | 0.2% | 2.64% |
| short-to-long | target-recalibrated | 0.1 | 0.09944 | [0.09536, 0.10352] | yes | no | 42.0% | 28.66% |
| long-to-short | source-control | 0.01 | 0.00939 | [0.00809, 0.01069] | yes | no | 23.2% | 3.74% |
| long-to-short | shifted | 0.01 | 0.11822 | [0.11337, 0.12307] | no | yes | 100.0% | 52.17% |
| long-to-short | target-recalibrated | 0.01 | 0.00999 | [0.00862, 0.01136] | yes | no | 24.6% | 4.70% |
| long-to-short | source-control | 0.02 | 0.01985 | [0.01795, 0.02175] | yes | no | 31.6% | 6.77% |
| long-to-short | shifted | 0.02 | 0.16083 | [0.15659, 0.16507] | no | yes | 99.9% | 68.50% |
| long-to-short | target-recalibrated | 0.02 | 0.01923 | [0.01734, 0.02112] | yes | no | 27.9% | 9.01% |
| long-to-short | source-control | 0.05 | 0.05022 | [0.04719, 0.05325] | no | no | 36.3% | 16.51% |
| long-to-short | shifted | 0.05 | 0.20015 | [0.19659, 0.20371] | no | yes | 100.0% | 86.60% |
| long-to-short | target-recalibrated | 0.05 | 0.04848 | [0.04551, 0.05145] | yes | no | 35.0% | 22.24% |
| long-to-short | source-control | 0.1 | 0.09944 | [0.09536, 0.10352] | yes | no | 42.0% | 28.66% |
| long-to-short | shifted | 0.1 | 0.21471 | [0.21113, 0.21829] | no | yes | 100.0% | 93.44% |
| long-to-short | target-recalibrated | 0.1 | 0.09738 | [0.09324, 0.10152] | yes | no | 38.9% | 44.22% |

## Domain diagnostics

Computed after fixing the split rule; these labels did not select the domains.

| Domain | Mean question characters | Cheap-only regret | Cheap accuracy |
|---|---:|---:|---:|
| short | 170.6 | 0.23832 | 69.63% |
| long | 321.5 | 0.35440 | 56.87% |

## Interpretation and reproducibility

Approximate normal Monte Carlo CIs over independent partition draws, conditional on one fixed training set and finite pools. Bonferroni across both directions, three arms and all epsilons. Arms share rows; they are not independent datasets. Mean <= target is descriptive; lower adjusted CI <= target is the plan's non-rejection criterion, not proof of validity. Individual splits can exceed epsilon.

Adjusted risk family: 24 cells. Per-draw partitions, thresholds and metrics are in draws.jsonl; ordered IDs, lengths and group membership in split-manifest.json; scores and weights are frozen.

Model identities: https://replay-cheap.invalid/v1::mistralai/Mixtral-8x7B-Instruct-v0.1; https://replay-strong.invalid/v1::gpt-4-1106-preview
Prompt: Single user message: first 8 GSM8K training examples formatted as 'Question: {question}\nAnswer: {answer}\n\n', then 'Question: {test_question}\nAnswer:'. Full text is in every row.
Decoding metadata: `{"temperature": 0, "max_tokens": 1024, "stop": ["Question"], "top_p": "Not explicit upstream; replay uses 1 (assumed default).", "provider": "Upstream SGLang OpenAI-compatible router; original host not recorded.", "generated_at": null, "finish_reason": null, "latency": null, "provenance": "Settings/model IDs from pinned generation script; original wire requests, provider revisions and generation dates unavailable."}`.
Tokens: ESTIMATED independently for prompt and each completion: ceil(Unicode character count / 4), including all 8 few-shot examples. Same approximation for both models; excludes chat framing. Not model-tokenizer counts or billed provider usage.
Pricing: Historical RouteLLM paper scenario. Mixtral input/output assumption $0.24/M; GPT-4-1106 input $10/M, output $30/M. Token usage is estimated from text; these are modeled generation costs, not measured bills or current prices.

## Limitations

- Real frozen model outputs and recomputed correctness; estimated tokens and costs.
- Original generation date, provider deployment and exact top_p are not recorded. These artifacts are bound to .invalid replay endpoints, not live deployment.
- Question-only lexical deduplication does not detect all semantic duplicates. This is one finite public benchmark, not a representative sample of live traffic.
- Metric copies RouteLLM's last unsigned integer rule, including its limitations for decimals, signs and trailing numbers; it is not a general math verifier.
- GSM8K may be in LLM pretraining data; benchmark contamination is unmeasured.
- Question length is a label-free covariate, not a verified difficulty label. This within-GSM8K stress test does not complete S1's cross-dataset domain study.
- One training seed and previously analyzed benchmark rows; not a new untouched holdout.
- Target recalibration uses a separate, fully labeled target sample; no online drift detector, audit-delay simulation or audit budget is evaluated. Calibration acquisition cost is excluded from reported routed-generation savings.
- Real frozen outputs; estimated token counts and historical modeled prices, not bills. No live production traffic, provider calls or production guarantee.
