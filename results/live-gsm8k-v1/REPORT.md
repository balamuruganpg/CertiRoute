# Live GSM8K: Groq GPT-OSS 20B + xAI Grok

**Completed 2026-10-30. The observed 12.12% CertiRoute regret exceeds the 5% target.**
The 15.96% recorded test-cost saving is not a saving at validated 5% live risk.

| Policy | Test requests | Quality | Regret | Recorded USD | p50 | p95 | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cheap-only | 100 | 40% | 55.56% | 0.01355250 | 0.883 s | 2.208 s | 0% |
| Strong-only | 100 | 95% | 0% | 0.07832080 | 1.541 s | 2.310 s | 1% |
| CertiRoute | 100 | 84% | 12.12% | 0.06582315 | 1.471 s | 2.329 s | 0% |

Regret is mean `max(0, q_strong - q_policy)` over **99** prompts with successful
strong-only reference calls. Quality includes all 100 requests; failed requests score
zero. One strong-only request has unknown usage and retains its conservative cost
reservation, so the cost comparison is not a reconciled invoice.

## Protocol and accounting

- Fixed public GSM8K prompts, ordered by seed **20261026**: **60 train / 80 calibration /
  100 test**. Frozen split IDs and source checksum are in `splits.json`; full prompt
  messages are in `data/eval/gsm8k-full/paired.jsonl` (see dataset manifest).
- Cheap: Groq `openai/gpt-oss-20b`. Strong: xAI `grok-4.20-0309-non-reasoning`.
- Temperature 0, top_p 1, streaming off, max output 256 tokens. xAI receives
  `max_completion_tokens`; Groq receives `max_tokens`. No search or tools.
- Router and CRC are fit/calibrated from new live pairs, never historical output labels.
  Three policies run on each held-out prompt in seeded randomized order. Acquisition
  was resumed twice; successful outputs were reused regardless of correctness.
- Completion latency excludes deliberate pacing; gateway runs through in-process ASGI.
  The resumed invocation used 15-second pacing; early acquisition used 2 seconds.
  Serving throughput and production network overhead are not measured.
- Rates dated **2026-10-30**, USD per million tokens: Groq input **0.075**, cached input
  **0.0375**, output **0.30**; xAI input **1.25**, cached input **0.20**, output **2.50**.
  Sources: [Groq model page](https://console.groq.com/docs/model/openai/gpt-oss-20b),
  [xAI model table](https://docs.x.ai/developers/models.md).
  xAI-reported USD ticks take precedence; Groq uses actual usage times recorded prices.
- Acquisition: **281 attempts**, **$0.1430889075** including one failed attempt's
  reservation. Test: **300 policy calls**, **$0.15769645**. This full study totals
  **$0.3007853575**. Cumulative ledger including earlier runs/pilots: **$0.3613503325**.

## What this establishes

Real provider connectivity, token/cost metering, timing, fresh calibration, and an
end-to-end three-policy comparison all ran. This single draw **missed the empirical
risk target**. It does not validate a live guarantee, nor by itself disprove a marginal
expected-risk theorem. Cheap-model truncation, small calibration size, model behavior
and sampling variation need investigation before any cause is claimed.

This is benchmark traffic, not production traffic. The offline Mixtral/GPT-4 studies
use different models, costs and protocols and must not be conflated with these results.

## Recheck without keys

```bash
python -m eval.publish_live --verify results/live-gsm8k-v1
python -m eval.plot_portfolio --out results/reproduced/portfolio-plots
```

The first command recomputes policy metrics and checks test-output quality against
the public references. `observations.jsonl` preserves outputs, usage, quality, timing
and cost; provider request IDs and local account paths are omitted. The original
private result and published observations are bound by SHA-256 hashes in `results.json`.
Re-rendering is offline; recollecting provider outputs is paid and not bit-reproducible.

[Collection and recovery commands](../../docs/LIVE_MEASUREMENT.md)
