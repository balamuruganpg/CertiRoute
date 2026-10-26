# GSM8K calibration and seed sensitivity

Training seeds [17, 42, 73, 101, 137]; random-ranking seeds [0, 1, 2, 3, 4]; 1000 draws per training seed; fixed test size 264.
Same full prompts, learner, seven features, hyperparameters and epsilon grid. Each predictor is fitted once before all calibration/test draws.

Calibration prefixes are nested and test questions stay identical across sizes in a draw.
Tables average training-seed means equally; random CRC also averages ranking seeds equally.
Ranges show sensitivity across seed means, not confidence intervals for new data.

## Calibration-size sweep and no-CRC control

| n_cal | Epsilon | Policy | Mean regret | Training-seed regret range | Estimated savings | Splits > epsilon |
|---:|---:|---|---:|---|---:|---:|
| 50 | 0.01 | CRC | 0.00000 | [0.00000, 0.00000] | 0.00% | 0.0% |
| 50 | 0.01 | no-CRC | 0.01931 | [0.01817, 0.01993] | 7.64% | 57.9% |
| 50 | 0.01 | random-CRC | 0.00000 | [0.00000, 0.00000] | 0.00% | 0.0% |
| 50 | 0.02 | CRC | 0.01931 | [0.01817, 0.01993] | 7.64% | 34.2% |
| 50 | 0.02 | no-CRC | 0.03900 | [0.03659, 0.04006] | 15.78% | 68.4% |
| 50 | 0.02 | random-CRC | 0.01979 | [0.01940, 0.02021] | 6.35% | 35.6% |
| 50 | 0.05 | CRC | 0.03900 | [0.03659, 0.04006] | 15.78% | 27.2% |
| 50 | 0.05 | no-CRC | 0.05855 | [0.05733, 0.06025] | 23.84% | 51.9% |
| 50 | 0.05 | random-CRC | 0.03922 | [0.03866, 0.03959] | 12.68% | 27.9% |
| 50 | 0.1 | CRC | 0.09777 | [0.09554, 0.09917] | 39.08% | 42.1% |
| 50 | 0.1 | no-CRC | 0.11710 | [0.11515, 0.11821] | 46.14% | 59.1% |
| 50 | 0.1 | random-CRC | 0.09799 | [0.09658, 0.09885] | 32.40% | 43.0% |
| 100 | 0.01 | CRC | 0.00976 | [0.00933, 0.01007] | 3.72% | 37.0% |
| 100 | 0.01 | no-CRC | 0.01988 | [0.01952, 0.02010] | 7.73% | 70.4% |
| 100 | 0.01 | random-CRC | 0.01000 | [0.00983, 0.01018] | 3.20% | 38.6% |
| 100 | 0.02 | CRC | 0.01988 | [0.01952, 0.02010] | 7.73% | 38.6% |
| 100 | 0.02 | no-CRC | 0.02976 | [0.02897, 0.03080] | 11.93% | 61.0% |
| 100 | 0.02 | random-CRC | 0.01981 | [0.01946, 0.02024] | 6.34% | 38.8% |
| 100 | 0.05 | CRC | 0.05001 | [0.04898, 0.05102] | 20.49% | 42.6% |
| 100 | 0.05 | no-CRC | 0.06011 | [0.05947, 0.06047] | 24.64% | 58.3% |
| 100 | 0.05 | random-CRC | 0.04964 | [0.04909, 0.05008] | 16.01% | 42.3% |
| 100 | 0.1 | CRC | 0.09937 | [0.09770, 0.10117] | 39.96% | 44.9% |
| 100 | 0.1 | no-CRC | 0.10926 | [0.10777, 0.11106] | 43.70% | 56.5% |
| 100 | 0.1 | random-CRC | 0.09936 | [0.09832, 0.10051] | 32.82% | 44.9% |
| 200 | 0.01 | CRC | 0.00995 | [0.00969, 0.01032] | 3.74% | 42.4% |
| 200 | 0.01 | no-CRC | 0.01489 | [0.01436, 0.01532] | 5.60% | 63.0% |
| 200 | 0.01 | random-CRC | 0.01002 | [0.00979, 0.01021] | 3.19% | 42.2% |
| 200 | 0.02 | CRC | 0.01977 | [0.01923, 0.02028] | 7.50% | 39.9% |
| 200 | 0.02 | no-CRC | 0.02483 | [0.02436, 0.02555] | 9.64% | 54.9% |
| 200 | 0.02 | random-CRC | 0.02001 | [0.01969, 0.02028] | 6.38% | 40.9% |
| 200 | 0.05 | CRC | 0.04976 | [0.04948, 0.05025] | 20.71% | 43.3% |
| 200 | 0.05 | no-CRC | 0.05485 | [0.05446, 0.05519] | 22.76% | 52.9% |
| 200 | 0.05 | random-CRC | 0.04970 | [0.04929, 0.04994] | 15.99% | 43.2% |
| 200 | 0.1 | CRC | 0.09921 | [0.09853, 0.10003] | 40.15% | 46.5% |
| 200 | 0.1 | no-CRC | 0.10411 | [0.10350, 0.10491] | 42.04% | 53.3% |
| 200 | 0.1 | random-CRC | 0.09958 | [0.09900, 0.10022] | 32.87% | 46.0% |
| 500 | 0.01 | CRC | 0.01008 | [0.00983, 0.01039] | 3.82% | 46.4% |
| 500 | 0.01 | no-CRC | 0.01203 | [0.01161, 0.01251] | 4.49% | 56.9% |
| 500 | 0.01 | random-CRC | 0.01005 | [0.00989, 0.01021] | 3.23% | 45.9% |
| 500 | 0.02 | CRC | 0.01999 | [0.01956, 0.02042] | 7.23% | 42.1% |
| 500 | 0.02 | no-CRC | 0.02207 | [0.02163, 0.02248] | 8.08% | 49.5% |
| 500 | 0.02 | random-CRC | 0.02001 | [0.01974, 0.02030] | 6.33% | 42.3% |
| 500 | 0.05 | CRC | 0.05008 | [0.04955, 0.05113] | 21.06% | 44.4% |
| 500 | 0.05 | no-CRC | 0.05207 | [0.05170, 0.05297] | 21.85% | 49.1% |
| 500 | 0.05 | random-CRC | 0.05004 | [0.04969, 0.05038] | 16.09% | 44.3% |
| 500 | 0.1 | CRC | 0.09980 | [0.09866, 0.10047] | 40.48% | 46.8% |
| 500 | 0.1 | no-CRC | 0.10175 | [0.10059, 0.10248] | 41.27% | 50.3% |
| 500 | 0.1 | random-CRC | 0.10007 | [0.09933, 0.10062] | 33.00% | 47.5% |

## Learned ranking versus random rankings

Positive values favor the learned ranking at the same target epsilon; attained risks differ.
The range covers every training/ranking-seed pair, after averaging its draws.

| n_cal | Epsilon | Mean savings advantage (pp) | Seed-pair range (pp) | Positive pairs | Positive adjusted-CI pairs |
|---:|---:|---:|---|---:|---:|
| 50 | 0.01 | 0.000 | [0.000, 0.000] | 0/25 | 0/25 |
| 50 | 0.02 | 1.294 | [-0.772, 3.216] | 22/25 | 13/25 |
| 50 | 0.05 | 3.099 | [0.054, 5.741] | 25/25 | 18/25 |
| 50 | 0.1 | 6.678 | [2.165, 9.521] | 25/25 | 24/25 |
| 100 | 0.01 | 0.516 | [-0.858, 1.688] | 20/25 | 12/25 |
| 100 | 0.02 | 1.393 | [-0.633, 3.291] | 21/25 | 18/25 |
| 100 | 0.05 | 4.483 | [0.407, 7.371] | 25/25 | 24/25 |
| 100 | 0.1 | 7.148 | [2.774, 10.545] | 25/25 | 25/25 |
| 200 | 0.01 | 0.547 | [-0.790, 1.810] | 20/25 | 14/25 |
| 200 | 0.02 | 1.120 | [-1.158, 3.063] | 20/25 | 17/25 |
| 200 | 0.05 | 4.721 | [-0.053, 8.129] | 24/25 | 24/25 |
| 200 | 0.1 | 7.282 | [2.667, 10.988] | 25/25 | 25/25 |
| 500 | 0.01 | 0.587 | [-0.975, 2.037] | 19/25 | 15/25 |
| 500 | 0.02 | 0.899 | [-1.464, 2.720] | 18/25 | 16/25 |
| 500 | 0.05 | 4.977 | [0.333, 8.531] | 25/25 | 23/25 |
| 500 | 0.1 | 7.479 | [2.251, 12.455] | 25/25 | 25/25 |

## Interpretation and plots

CRC non-rejections: 80/80 cells. No-CRC rejections: 70/80 cells. The criterion is adjusted lower mean-regret CI <= epsilon; non-rejection is not proof.
The negative control chooses the smallest lambda with sum(loss)/n <= epsilon. CRC uses (sum(loss)+1)/(n+1) <= epsilon. Both route cheap strictly when score > lambda.

![Calibration-size sensitivity](plots/calibration-sweep.png)

![Risk versus estimated cost](plots/risk-cost.png)

![Ranking advantage across training seeds](plots/seed-advantage.png)

## Reproducibility and limitations

Per-cell normal Monte Carlo CIs use independent draws conditional on that training set, finite pool and ranking. Bonferroni risk family includes all seeds, rankings, sizes, epsilons and three policies. Paired savings CIs adjust across all seed pairs, sizes and epsilons. Seed ranges are descriptive, not population confidence intervals. Overlapping training/pool sets and nested sizes are never counted as independent samples.

Risk CI family: 560 cells; paired family: 400 comparisons. Normal intervals are approximate.
Every threshold/metric is in draws.jsonl.gz, using draw_columns from results.json. Partition checksums, seed-specific ordered IDs, frozen scores and weights permit replay.
Unused calibration candidates are not added to the fixed test set for smaller sizes.
Means, variances, quantiles, per-cell CIs and pair comparisons are in results.json.
Plots use the saved aggregates; PNG/SVG/PDF and plotting provenance are in plots/.

Model identities: https://replay-cheap.invalid/v1::mistralai/Mixtral-8x7B-Instruct-v0.1; https://replay-strong.invalid/v1::gpt-4-1106-preview
Prompt: Single user message: first 8 GSM8K training examples formatted as 'Question: {question}\nAnswer: {answer}\n\n', then 'Question: {test_question}\nAnswer:'. Full text is in every row.
Decoding metadata: `{"temperature": 0, "max_tokens": 1024, "stop": ["Question"], "top_p": "Not explicit upstream; replay uses 1 (assumed default).", "provider": "Upstream SGLang OpenAI-compatible router; original host not recorded.", "generated_at": null, "finish_reason": null, "latency": null, "provenance": "Settings/model IDs from pinned generation script; original wire requests, provider revisions and generation dates unavailable."}`.
Tokens: ESTIMATED independently for prompt and each completion: ceil(Unicode character count / 4), including all 8 few-shot examples. Same approximation for both models; excludes chat framing. Not model-tokenizer counts or billed provider usage.
Pricing: Historical RouteLLM paper scenario. Mixtral input/output assumption $0.24/M; GPT-4-1106 input $10/M, output $30/M. Token usage is estimated from text; these are modeled generation costs, not measured bills or current prices.

- Real frozen model outputs and recomputed correctness; estimated tokens and costs.
- Original generation date, provider deployment and exact top_p are not recorded. These artifacts are bound to .invalid replay endpoints, not live deployment.
- Question-only lexical deduplication does not detect all semantic duplicates. This is one finite public benchmark, not a representative sample of live traffic.
- Metric copies RouteLLM's last unsigned integer rule, including its limitations for decimals, signs and trailing numbers; it is not a general math verifier.
- GSM8K may be in LLM pretraining data; benchmark contamination is unmeasured.
- Benchmark rows were used in prior experiments; this is sensitivity analysis, not a new holdout.
- Only five configured training partitions and five rankings in the default study; no generalization claim across arbitrary datasets or model families.
- The no-CRC policy is an evaluation-only negative control, not a deployable calibrated artifact.
- Risk bounds are marginal expectations, not guarantees on every split. Non-rejection is not proof.
- Token counts and historical costs remain estimates. No live provider calls or billed spend.
