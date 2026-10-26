# CertiRoute — Work Division

Two primary owners share one codebase: **Systems** owns implementation and
operations; **Theory** owns statistical validity and experimental interpretation.
Each reviews changes at the boundary. This plan follows [ARCHITECTURE.md](ARCHITECTURE.md),
[EVALUATION.md](EVALUATION.md) and [ROADMAP.md](ROADMAP.md).

## Systems work packages

| ID | Work | Deliverable and completion evidence | Milestone |
|---|---|---|---|
| S1 | Packaging, configuration, CI | Installable `src` package, startup guide, checks, secret-free example config | M0 |
| S2 | Gateway and provider adapter | Auth, schemas, aliases, SSE passthrough, pooled HTTP, timeout/retry/circuit-breaker behavior; mock SDK contract tests | M1–M3 |
| S3 | Paired data pipeline | Licensed prompt pool, dedup/split manifest, resumable paired outputs, quality/usage/latency rows, frozen dataset hash | M1 |
| S4 | Prompt features and predictor | CPU embeddings/surface features, fitted predictor and hash, inference benchmark; training only on `D_router` | M2 |
| S5 | Artifact serving and policy | Schema/identity/expiry validation, scope guard, strict threshold comparison, atomic reload, readiness and fallback tests | M2–M3 |
| S6 | Storage, audits and monitoring | SQLite schema, bounded queues, uniform audit sampling, drift job, recalibration integration, dashboard | M4 |
| S7 | Evaluation execution and release | Baseline runs, reproducible result JSONs/plots, Locust and failure/soak tests, Docker demo, release docs | M2–M5 |

The S identifiers in this table are work-package ids. Evaluation scenarios S1–S4
remain defined in EVALUATION.md; use the prefix “evaluation” when referring to them.

## Theory/statistics work packages

| ID | Work | Deliverable and completion evidence | Milestone |
|---|---|---|---|
| T1 | Loss and CRC reference | Formal assumptions and guarantee note; bounded regret; reference CRC; ties/endpoints/empty-set/property tests and known-risk simulation | M0–M1 |
| T2 | Evaluation and epsilon selection | Predeclared grids/splits/CI method; G1–G4 and G6 design; baseline tuning protocol; correct marginal-risk interpretation | M0–M3 |
| T3 | Quality metrics and label noise | Metric versions and scorer fixtures; G7/G8; judge validation and bias/noise analysis before open-ended claims | M1–M4 |
| T4 | Distribution shift and weighted CRC | Evaluation S1–S4 design; drift power/false-alarm comparison; assumptions and analysis for density-ratio weighting (stretch) | M3–M4, stretch |
| T5 | Online risk control | Bound and simulation for threshold updates, delayed and sampled labels, step-size and clipping rules (stretch) | After M4 |
| T6 | LTT and group guarantees | G5 analysis; population-risk versus finite-test-risk distinction; group sample-size/multiplicity requirements (stretch) | After M3 |
| T7 | Statistical reporting | Review result interpretation, tightness assumptions and threats to validity; coauthor report and review all guarantee claims | M2–M5 |

T3, T4 and T5 preserve the references from the architecture and evaluation docs.
Theory owns the numerical reference; systems owns serving integration. Optimize
the reference only after parity checks on shared fixtures.

## Shared contracts and handoffs

| Contract | Producer → consumer | Required contents / review |
|---|---|---|
| Paired-output dataset | Systems → Theory | Prompt id/hash, source/group, split, exact model/provider revisions, decoding/template versions, both outputs and scores, token counts, latency, finish/error status, dataset hash; Theory approves exclusion rules |
| Split manifest | Systems → both | Exact and near-duplicate guard, seeds, `D_router` disjoint from `D_pool`, repeated cal/test partition seeds; Theory checks leakage |
| Router package | Systems → calibration/serving | Feature/embedding version, predictor weights/hash, score orientation `P(regret=0)`, training data hash, supported scope; Theory checks training separation |
| CRC API | Theory → Systems | Scores/regrets in range, fixed epsilon, `B=1`, threshold in `[0,1]`, strict `>` semantics, invalid-input policy, always-strong fallback fixtures |
| Calibration artifact | Theory + Systems → gateway | ARCHITECTURE §5.5 fields, threshold grid, metric/data/router/model provenance; jointly define schema version, decoding/scope metadata, expiry and canonical content hash |
| Decision/event | Systems → monitor/evaluation | Request id, tier, scope, score when available, epsilon, artifact id, degraded/bypass/fallback reason, usage and timing; no raw prompt by default |
| Results bundle | Systems → Theory → report | Config/seeds/git SHA/dataset hash/artifact id, raw per-split risk/cost, CI procedure and plots; Theory signs off on interpretation |

Resolve API and artifact changes together before merging either producer or
consumer. Calibration metadata must cover fixed generation settings even though
the example artifact in ARCHITECTURE omits those fields. The initial scaffold
exposes interfaces only and must not accept a placeholder artifact as valid.

## Milestone synchronization

1. **M0:** agree model pair, metric, scope, epsilon grids, budget, licences and
   data/error schemas. Record choices as ADRs; unresolved values stay placeholders.
2. **M1:** exchange a tiny deterministic fixture and pilot paired dataset. Systems
   demonstrates passthrough; Theory demonstrates CRC boundary and simulation checks.
3. **M2:** freeze the predictor and dataset. Theory reviews leakage and artifacts;
   Systems demonstrates repeatable fitting/calibration and baseline plots.
4. **M3:** jointly review readiness, scope, fallback, streaming and G1/G6 results.
   Each MVP gate needs evidence; file completion alone is insufficient.
5. **M4:** agree audit budgets, drift thresholds and action policy before shift runs.
   Review observed recovery, false alarms and unsupported exchangeability claims.
6. **M5:** Systems reproduces the bundle; Theory checks every reported claim and
   interval; both approve limitations, licence and release documentation.

## Ownership of open questions

| Question from ARCHITECTURE §14 | Lead | Decision artifact |
|---|---|---|
| Binary versus soft open-ended loss | Theory T3; Systems implements scorer | Boundedness/monotonicity note and G7 comparison |
| Judge noise and guarantee degradation | Theory T3 | G8 protocol/results plus judge agreement and bias checks |
| Epsilon-grid selection validity | Theory T2 | Fixed-grid and caller-selection ADR; no result-dependent epsilon choice |
| One global threshold versus groups | Theory T6; Systems supplies group data | MVP global threshold retained; separate group proposal and sample-size analysis |
| Drift statistic and alarm trade-off | Theory T4; Systems S6 | No-shift false alarms, shifted power/delay, audit cost and chosen response |

Systems also leads alias/header precedence, request limits, artifact reload
atomicity, error mapping and deployment decisions; Theory reviews any effect on
loss, scope or sampling. Both approve model/metric/decoding changes that require
new calibration.

## Review boundaries and blocked work

Neither owner changes the loss, threshold direction, sampling scheme or split
policy without the other's review. Serving problems fall back toward strong;
cheap downgrade is explicit and excluded from the guarantee. Mid-stream errors
are counted as failures, not retried into a mixed answer.

If provider access or collection budget is unavailable, Systems proceeds with
mock upstreams while Theory uses synthetic fixtures; neither labels those as
real-output evaluation. If no valid artifact exists, readiness remains false.
If an acceptance target fails, preserve results and document the unmet gate.
