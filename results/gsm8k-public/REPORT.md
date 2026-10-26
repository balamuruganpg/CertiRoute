# Offline paired quality–cost comparison

Real frozen GSM8K model outputs; estimated token usage and historical modeled costs. No external API calls.

Seed 42; train/calibration/test = 40/40/20.
Costs are totals over the held-out set, not per-request costs.

| Policy | Epsilon | Quality | Mean regret | Estimated USD | Savings vs strong | Strong fraction |
|---|---:|---:|---:|---:|---:|---:|
| cheap-only | — | 45.00% | 0.5000 | 0.00545832 | 98.09% | 0.00% |
| strong-only | — | 95.00% | 0.0000 | 0.28532000 | 0.00% | 100.00% |
| CertiRoute | 0.01 | 95.00% | 0.0000 | 0.28532000 | 0.00% | 100.00% |
| CertiRoute | 0.02 | 95.00% | 0.0000 | 0.28532000 | 0.00% | 100.00% |
| CertiRoute | 0.05 | 85.00% | 0.1000 | 0.22626440 | 20.70% | 80.00% |
| CertiRoute | 0.1 | 60.00% | 0.3500 | 0.11251424 | 60.57% | 40.00% |

One small held-out split is descriptive; it does not validate the CRC guarantee.
The epsilon grid was fixed before fitting. All operating points are reported.
With n=40, epsilons below 0.024390 force always-strong.

Cheap model: `mistralai/Mixtral-8x7B-Instruct-v0.1`; strong model: `gpt-4-1106-preview`.
Quality metric: `routellm_gsm8k_last_integer` v1.
Pricing as of 2024-06-26: Historical RouteLLM paper scenario. Mixtral input/output assumption $0.24/M; GPT-4-1106 input $10/M, output $30/M. Token usage is estimated from text; these are modeled generation costs, not measured bills or current prices.
Pricing source: https://arxiv.org/html/2406.18665v1#A4

| Model | Input USD / million tokens | Output USD / million tokens |
|---|---:|---:|
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 0.24 | 0.24 |
| gpt-4-1106-preview | 10.0 | 30.0 |

Dataset SHA256: `3a6fec941886362c7f778c8e6887e89449773c4e5fa9be073d168738d1e401c9`.
See results.json for configuration, split, code, dependency and per-request provenance.

Paired-output source: https://raw.githubusercontent.com/lm-sys/RouteLLM/0b64fdafe049e596a3f5657c219329f24af24198/routellm/evals/gsm8k/gsm8k_responses.csv
Generation script: https://raw.githubusercontent.com/lm-sys/RouteLLM/0b64fdafe049e596a3f5657c219329f24af24198/routellm/evals/gsm8k/generate_responses.py

Single user message: first 8 GSM8K training examples formatted as 'Question: {question}\nAnswer: {answer}\n\n', then 'Question: {test_question}\nAnswer:'. Full text is in every row.
Full frozen requests and responses: data/eval/gsm8k/paired.jsonl.
Generation metadata: `{"finish_reason": null, "generated_at": null, "latency": null, "max_tokens": 1024, "provenance": "Settings/model IDs from pinned generation script; original wire requests, provider revisions and generation dates unavailable.", "provider": "Upstream SGLang OpenAI-compatible router; original host not recorded.", "stop": ["Question"], "temperature": 0, "top_p": "Not explicit upstream; replay uses 1 (assumed default)."}`.
ESTIMATED independently for prompt and each completion: ceil(Unicode character count / 4), including all 8 few-shot examples. Same approximation for both models; excludes chat framing. Not model-tokenizer counts or billed provider usage.
Sort all 1319 prompts by SHA256(UTF-8 question); greedily retain first 100 with full-prompt token-trigram Jaccard < 0.9 against retained prompts. No selection on labels or outputs.

Held-out regret exceeded epsilon at epsilon 0.05: observed regret 0.1000; epsilon 0.1: observed regret 0.3500. CRC controls an expectation over calibration/test draws, not every split. These observations do not establish validity or cost savings at the target risk.

Limitations:

- Real frozen model outputs and recomputed correctness; estimated tokens and costs.
- Original generation date, provider deployment and exact top_p are not recorded. These artifacts are bound to .invalid replay endpoints, not live deployment.
- Full-prompt deduplication includes a long common few-shot prefix; selection can favor longer or lexically distinctive questions. This is not a random GSM8K sample.
- Metric copies RouteLLM's last unsigned integer rule, including its limitations for decimals, signs and trailing numbers; it is not a general math verifier.
- GSM8K may be in LLM pretraining data; benchmark contamination is unmeasured.
- Single held-out split, not G1 repeated-split validity or live G9.
- Zero billed API spend; cost covers the selected generation only, not collection or router hosting. No live latency was measured.
