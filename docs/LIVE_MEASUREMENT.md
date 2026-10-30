# Live provider measurement

**Status: first live comparison completed, 2026-10-30.** Groq GPT-OSS 20B + direct xAI
Grok achieved **15.96% recorded test-cost savings**, but **12.12% empirical regret**
exceeded the 5% target. [Full report and frozen evidence](../results/live-gsm8k-v1/REPORT.md).
This is not a validated live risk guarantee. Earlier OpenRouter/AgentRouter attempts
were incomplete. Mock measurements are never presented as provider results.

## First xAI pilot: 2026-10-30

Saved locally in `results/xai-pilot-1/results.json`: 10 paired GSM8K prompts,
20 requests, no errors. Groq costs use recorded rates and actual tokens;
xAI costs come from its reported USD ticks.

| Provider | Requests | Quality (10 prompts) | Cost USD | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| Groq GPT-OSS 20B | 10 | 40% | 0.00161310 | 1.613 s | 2.868 s |
| xAI Grok 4.20 non-reasoning | 10 | 100% | 0.01251425 | 2.260 s | 3.187 s |

Pilot cost: **$0.01412735**. Shared journal including earlier attempts:
**$0.060564975**. The pilot projects **$0.4803299** for full collection at its
observed mean rates. This small sample does not establish a CRC guarantee or
production savings. The 256-token cap can truncate the cheap model's reasoning.
Full collection initially stopped on a Groq HTTP 429 after 25 complete pairs and
was subsequently resumed. It finished with 140 acquisition pairs and 300 test calls.
Full-study cost/reservations: **$0.3007853575**; cumulative journal including earlier
attempts and pilots: **$0.3613503325**. One strong-only test request failed; regret
uses 99 valid strong references. Preserve the journal for any further collection.

## Approved configuration

[`configs/live.study.json`](../configs/live.study.json) selects:

| Tier | Endpoint | Model | USD/M input | USD/M cached input | USD/M output |
|---|---|---|---:|---:|---:|
| Cheap | `https://api.groq.com/openai/v1` | `openai/gpt-oss-20b` | 0.075 | 0.0375 | 0.30 |
| Strong | `https://api.x.ai/v1` | `grok-4.20-0309-non-reasoning` | 1.25 | 0.20 | 2.50 |

Prices recorded **2026-10-30** from [Groq](https://console.groq.com/docs/model/openai/gpt-oss-20b)
and [xAI's full model table](https://docs.x.ai/developers/models.md).
The dated non-reasoning model is deliberate: xAI's `max_completion_tokens` caps
visible output, not hidden reasoning. The metered transport converts the common
256-token cap to this parameter and refuses other xAI models. No tools, search,
multiple completions, priority service or long-context pricing is enabled.
The selected model must be present in the authenticated catalog before completions.
No model substitution is automatic; the old AgentRouter/Codex config is not used.

## Run

Put your Groq key in `CHEAP_API_KEY` and your **xAI (Grok)** key in `XAI_API_KEY`
in `.env` or process environment variables. These names are selected by each tier's
`api_key_env` field in `configs/live.study.json`.
Never commit `.env`, print keys, or pass them as command arguments.

```bash
# Local plan only: no credentials needed, no requests, no spend
python -m eval.live_study

# Limited paid pilot: 10 prompts x 2 providers; at most $0.10 additional spend
python -m eval.live_study --mode pilot --execute --out results/xai-pilot-1
```

Pilot is the default mode. It stops after **20 provider calls** (10 per provider),
or sooner on error/budget exhaustion, with no retries, fallback or automatic continuation.
It writes `results.json` with each provider's quality, actual token usage, estimated
cost, p50/p95 latency and errors. It does **not** fit a router or claim a CRC guarantee.
Its deterministic prompts are drawn from the eventual training partition; full
execution recollects them, keeping the full held-out test set out of pilot selection.
Set `pilot_requests` to an even number from 10 to 20 to change the pilot size.

Review `projected_full_study_usd`, provider errors, and actual account debits before
continuing. Projection uses mean pilot costs and assumes both models are called for
every routed test request to allow for fallback. This is a planning estimate, not a
bound or a guarantee of savings. Full execution requires a complete successful pilot
with known token/cost accounting, an identical config hash, the same budget journal,
and a projection that fits the remaining allowance. The additional
`--confirm-account-pricing` flag records your confirmation that account rates and
pilot debits do not exceed the recorded estimates, and fees fit the $0.49 reserve.
If actual rates are higher, update the config and run a new pilot; do not confirm:

```bash
# Only after reviewing the pilot's rate and account charges:
python -m eval.live_study --mode full --execute --confirm-account-pricing --pilot-report results/xai-pilot-1/results.json --out results/xai-full-1
```

The full study makes **580 provider calls**, plus possible CertiRoute fallback
attempts: 60 training prompts × 2 models; 80 calibration prompts × 2 models; 100
held-out prompts × three policies. The cumulative API allowance is **$2.50**, including
earlier attempts and all pilots in the same journal, retaining **$0.49 within a $2.99 total**
for fees. Each pilot also has a separate $0.10 additional-spend limit. The script
never purchases or auto-refills credits. Account funding, fees and unexpected billing
adjustments are not controlled by a client script: configure provider-side limits and
keep these within the total authorization. These guards apply to the live study;
running the standalone gateway does not use the study's budget journal.

The CLI now runs sequentially with **15-second pacing** before each logical request and a
256-token completion cap. GPT-OSS reasoning consumes part of that cap; truncation
is part of the measured result, not silently retried. No temperature/model/scope
changes are allowed after collection without a fresh calibration.
`--interval-seconds` overrides operational pacing without changing the model,
generation settings, split or pilot-config hash. Each invocation records the effective
pacing in `execution.jsonl`; the final report also includes it. Old collected responses
retain their original measured latency, so resumed acquisition may span pacing regimes.

## Measurement design

1. Deduplicate the frozen GSM8K prompt pool, sort with seed 20261026, and freeze three
   disjoint splits. **Historical output labels are never used for live calibration.**
2. Collect fresh cheap and strong outputs on training and calibration prompts with
   identical generation settings. Score last-integer answers against GSM8K references.
   Stop on acquisition failures rather than selectively dropping failed pairs.
3. Fit fresh weights on training only; produce a new hash-bound 5% CRC artifact from
   calibration only. Missing label diversity stops the run rather than inventing a router.
4. On each test prompt, randomize the order of cheap-only, strong-only and CertiRoute.
   Cheap-only and strong-only are direct calls without fallback. CertiRoute goes through
   the real FastAPI application in-process, including routing and upward fallback.
5. Meter every upstream attempt, including failed cheap attempts before fallback.
   Record actual provider usage, returned model identifier, HTTP status and wall time.
   Multiply actual tokens by recorded current prices, including validated cached-input
   counts and discounts. xAI's documented `usage.cost_in_usd_ticks` divided by 10^10
   overrides token estimates when present. Arbitrary `usage.cost` values are ignored
   for xAI. See [cost tracking](https://docs.x.ai/developers/cost-tracking).
   `cost_known` denotes computable accounting, not a reconciled invoice.
   If cache details are missing, label the full input charge as an uncached upper bound.
   Unknown usage retains the full reserved cost. Provider caching remains enabled;
   [Groq's cache behavior](https://console.groq.com/docs/prompt-caching) can affect both
   cost and latency, so policy order is randomized and cache metadata is retained.
6. Save raw responses, usage, per-request observations, quality, clipped regret, p50/p95
   latency and error rates. Report acquisition spend separately from test policy spend.

All request latencies include errors and fallback time; they exclude the intentional
pacing delay. Error rate counts logical policy requests; a separate upstream-attempt
summary counts provider errors. Regret uses only test prompts with a successful strong
reference response and reports its denominator. Failed policy requests score zero quality.
This is one descriptive calibration draw, not a replacement for repeated CRC validity tests.

## Budget and interruption behavior

`build/live-budget.jsonl` is a persistent append-only journal, independent of output
directories. Each request reserves a conservative token/price bound **before sending**;
each reservation and settlement is flushed to disk. Unknown or interrupted usage keeps
the whole reservation. Failed runs and new output directories share the same allowance.
An exclusive `.lock` prevents concurrent writers. Do not change the journal path or delete
it between retries: that would reset the local accounting. Remove a stale lock only after
confirming no study process is running; preserve the journal.
If any provider reports a charge above its reservation, persist the actual charge and
halt subsequent calls, including after restarting the command, until the discrepancy
is reviewed. Provider-side spend limits remain the final billing backstop.

A failed run saves `status: incomplete` and partial observations. It does not publish
incomplete samples as a completed study. Full acquisition retries HTTP 429 up to three
times, waiting at least 60, 120 and 240 seconds, respectively, or the numeric
`Retry-After` value if longer. A delay above 300 seconds stops collection for quota
review. All attempts remain metered, including unknown-cost reservations. Other
failures, truncated answers and wrong answers are not retried. Pilots and held-out
test requests are not retried; the gateway's existing upward fallback still applies.

To continue the interrupted acquisition in the **same output folder**, use:

```powershell
.\.venv\Scripts\python.exe -m eval.live_study --mode full --execute --resume --interval-seconds 15 --confirm-account-pricing --pilot-report results/xai-pilot-1/results.json --out results/xai-full-1
```

Resume validates the data/splits/config and checks each saved attempt against the
original budget journal. It reuses every successful acquisition response (including
one half of a pair) and retries the first missing response, preserving failed attempts.
It never selects responses based on correctness. Prior incomplete reports are archived
before updating `results.json`. Resume is deliberately limited to acquisition before
router/calibration artifacts or any test observations exist; it rejects completed runs.
Progress prints every ten acquisition pairs, announces fitting/calibration and the test
phase, then prints after every test policy request. Acquisition `140/140` is not the
end of the study: another 100 prompts × 3 policies follow. At 15-second pacing alone,
that test phase takes at least 75 minutes, plus provider response time. Logging changes
apply to newly launched processes; do not restart an active test phase just to get them.
Do not start a second study simultaneously.

Groq limits can apply to requests or tokens over different time windows; check your
account's [limits](https://console.groq.com/settings/limits). Slower pacing helps minute
limits but cannot remove a daily quota. See [Groq's rate-limit documentation](https://console.groq.com/docs/rate-limits).
Numeric `Retry-After` is recorded; raw response headers and error bodies are not logged.
Acquisition failures display the provider and HTTP status, without raw error bodies or
credentials. HTTP 402 means to check provider credits and key spending limits before
retrying. A missing HTTP status means no response was received. Fix the cause first,
then resume eligible acquisition using the original output folder and budget journal.

## Limits

Current-price estimates and provider-reported costs are not reconciled invoices.
The $0.49 reserve is not measured overhead. In-process gateway latency excludes network
ingress/proxy overhead; no streaming TTFT or load throughput is measured. A completed
pilot is cost/connectivity evidence only; a full run is needed for a live CRC comparison.
`configs/gateway.yaml` selects the new pair, but old calibration artifacts remain invalid
for it. The full study creates its own gateway config and fresh model-pair artifacts.
