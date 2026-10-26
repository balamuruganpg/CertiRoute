# CertiRoute — Roadmap

This plan implements [ARCHITECTURE.md](ARCHITECTURE.md) and uses the gates in
[EVALUATION.md](EVALUATION.md). Milestones are dependency-based, not calendar
promises. The runnable scaffold is a starting point for M0, not a completed MVP.
Owners and handoffs are defined in [SPLIT.md](SPLIT.md).

## Scope and invariants

The MVP uses one cheap/strong model pair, text-only chat at calibrated generation
settings, prompt-side routing, a global CRC threshold per predeclared epsilon,
and verifiable quality metrics. Tools, multimodal inputs, unsupported languages,
long contexts and uncalibrated settings go strong through the scope guard.

- Route cheap only when `s(X) > lambda`; ties go strong.
- Use `regret = max(0, q_strong - q_cheap)` with `q` in `[0,1]` and `B = 1`.
- Train the router on `D_router`, separate from calibration and test data.
- Bind artifacts to the router, models, metric, data and generation settings.
- Missing, stale or mismatched artifacts mean unready and strong-only serving.
- CRC controls marginal expected regret under exchangeability; it does not give
  per-request, per-group or high-probability guarantees.
- Keep prompt text out of operational logs by default; never log API keys.

## M0 — Decisions, runnable skeleton and CI

**Deliver:** Python package under `src/certiroute`, FastAPI entry point, typed
configuration, `.env.example`, package metadata, stub interfaces, local run
instructions, smoke checks and then CI for lint/type/test checks. Record model
pair, decoding parameters, metric version, dataset licences, epsilon grids and
budget assumptions in configs and ADRs before collection.

**Exit:** a fresh environment can install the project and start the server.
Liveness works; readiness is false without validated calibration. Unimplemented
chat returns an explicit error, never a fabricated model output or guarantee.
Resolve artifact expiry, canonical hashing, scope metadata and alias/header
precedence before M3. CI remains outstanding until added and run.

## M1 — Paired data, passthrough and CRC reference

**Depends on:** M0 decisions and interfaces.

Systems builds a pooled async OpenAI-compatible adapter and strong passthrough,
including SSE ordering, `[DONE]`, cancellation, timeouts and normalized errors.
Collection runs both tiers at fixed settings, resumes from cached completed work,
stores usage/latency/finish reasons and scores, then freezes a hashed dataset.
Start with MMLU/GSM8K verifiable tasks; estimate cost before a full collection.
Deduplicate before splitting by prompt hash, including near duplicates.

Theory implements and reviews the CRC reference and bounded regret, including
ties, scores at 0/1, epsilon boundaries, invalid inputs, empty candidate sets and
the always-strong fallback. Test monotonicity, small samples and synthetic data
with known risk. No serving guarantee is enabled by the reference alone.

**Exit:** resumable paired collection, a frozen pilot dataset, streaming and
non-streaming mock-provider contract tests, CRC unit/property tests and repeated
synthetic calibration experiments pass. Specify failure accounting before real
collection so failed generations cannot be silently dropped to improve results.

## M2 — Router, artifacts and first efficiency curve

**Depends on:** M1 paired outcomes and verified CRC.

Use approximately 40% of prompts for `D_router` and 60% for `D_pool`; freeze the
router before repeatedly partitioning `D_pool` into calibration and test sets.
Start with prompt embeddings plus surface features and logistic regression;
LightGBM is a planned comparison. No generated-answer features in the MVP path.

Produce immutable, content-addressed artifacts with thresholds for the fixed
serving grid `[0.01, 0.02, 0.05, 0.10]`. Validate schema, finite numeric ranges,
router hash, model identities, metric, decoding/scope compatibility and expiry.
Do not use the architecture's illustrative thresholds as calibration data.

Implement always-strong, always-cheap, random mix, confidence threshold,
RouteLLM-style, uncalibrated own-router and oracle baselines. Tune eligible
baselines on validation data, never final test outcomes. Publish a first F1
Pareto curve and T1 cost-at-epsilon table, with achieved risk and oracle headroom.

**Exit:** reproducible fit/calibrate commands on frozen data; artifact rejection
tests; results include config, seeds, git SHA, dataset hash and artifact id.
No savings claim is made until measured.

## M3 — Calibrated gateway MVP

**Depends on:** M2 artifacts and M1 passthrough.

Integrate scope checks, inference, exact epsilon-grid lookup, routing headers,
static bearer authentication, separate admin credentials, rate limits, metrics,
safe artifact reload and readiness. Complete `/v1/chat/completions`, `/v1/models`,
`/healthz`, `/readyz`, `/metrics` and `/admin/reload-calibration`.
Implement cheap-to-strong fallback before the first token; never reroute a partial
stream. Any permitted strong-to-cheap downgrade requires explicit client opt-in,
is tagged degraded and is outside the guarantee.

Run G1–G4, G6 and the MVP-relevant loss checks in G7 on real paired outputs. Use
at least 1,000 random splits, the planned sample-size and epsilon sweeps, and
Bonferroni-adjusted intervals for the predeclared grid. Preserve the evaluation
plan's G1 criterion: the lower confidence endpoint for mean risk is at most
epsilon. Passing means validity was not rejected; it is not statistical proof.
Report individual-split violation fractions as descriptive statistics.

**MVP exit gates (EVALUATION §11):**

1. G1 meets the stated CI criterion at every evaluated epsilon.
2. At least two G6 negative controls demonstrate violations; report failure to
   meet this gate without choosing new controls after seeing results.
3. Cost at epsilon beats random mix at every epsilon; honestly compare other
   baselines and report any unmet efficiency target.
4. Streaming contracts and failure injection pass; publish overhead methodology
   and measured numbers against a mock upstream.
5. Run and report S1 domain shift, including failures of the guarantee.

These are acceptance targets, not predictions. A weak model pair may miss the
efficiency gate; publish the outcome and oracle headroom instead of hiding it.

## M4 — Monitoring, shift experiments and operational evidence

**Depends on:** M3 serving and traceable outcomes. S1 already runs at M3.

Add SQLite-backed requests/audits/calibrations, bounded asynchronous work queues,
uniform audits over all in-scope traffic regardless of chosen tier, and a
recalibration path. Add the MVP-lite dashboard for cost, escalation, audited
regret with intervals, drift and artifact history.

Complete S1–S4, G9 live replay with cache disabled, and G8 noise sensitivity.
Compare no mitigation with drift detection plus recalibration. Evaluate PSI/KS,
embedding-space tests and rolling audited regret; measure false alarms and delay.
Sweep audit rates `[0.005, 0.01, 0.02, 0.05]` and include audit spend. Newly pooled
data still needs exchangeability justification; recency weighting is a stretch
method and cannot inherit vanilla CRC's claim automatically.

Run concurrency 1/16/64/256, cold/warm router benchmarks, failure injection and
a 30–60 minute soak. Record p50/p95/p99 latency, TTFT, throughput, backlog and
memory/connection behavior. Recalibration recovery and G8 are the evaluation
plan's should-pass criteria; document limitations when they are unmet.

**Exit:** F5/F6 and T2/T3 plus relevant T4 ablations reproduced from result files;
drift response, artifact rollback/reload and audit budgets are documented.

## M5 — Reproducibility, report and v0.1

**Depends on:** M3 gates and M4 evidence, with remaining limitations explicit.

Deliver `make reproduce` from frozen paired data to F1–F6 and T1–T4, separate
paid collection (`make collect`), lockfile, Docker/Compose demo, operational docs,
and a 6–8 page report. Confirm licence and dataset redistribution terms; verify
citations before publication. Update README results only from committed result
JSONs. Document hardware, dependency versions, model revisions, collection dates,
provider nondeterminism, contamination risks and finite-pool limitations.

**Exit:** another person reproduces the reported outputs from the documented
inputs, and all unimplemented features and unmet targets are clearly labelled.

## Stretch — separate experiments after the baseline

| Track | Prerequisite | Evidence required before a claim |
|---|---|---|
| LTT / high-probability risk control (G5) | CRC baseline, fixed candidate policy family | Valid bound and extra-cost comparison; account for finite test-set estimation error |
| Weighted CRC under shift | S1–S3, theory T4 | Weight assumptions, diagnostics and robustness to estimated ratios |
| Online CRC | Audits, theory T5 | Long-run bound, delayed-label handling and simulation |
| Group-conditional calibration | Adequate group sample sizes | Per-group assumptions, multiplicity and sparse-group fallback |
| Open-ended judge scoring | G8 and validated judges | Agreement, position/length bias, second judge family |
| Three-model or post-generation cascade | Two-tier baseline | New monotone policy/loss analysis and streaming trade-offs |
| Semantic cache | Established uncached guarantee | Calibration-aware policy and separate evaluation; disabled for core guarantee tests |
| ONNX, Postgres, Redis, Grafana, second model pair | Measured need or evaluation budget | Measured benefit and regression checks |

## Working cadence and risk management

At each milestone, systems demonstrates a runnable increment and theory reviews
its statistical assumptions against frozen inputs. Track work as not started,
in progress or verified; the existence of a file does not finish a milestone.
Review spend after the pilot, leakage before fitting, scope/identity changes
before serving, and test outcomes before publication. Escalate unresolved
decisions through the owners in SPLIT rather than silently changing the claim.
