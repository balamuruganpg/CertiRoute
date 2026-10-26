# CertiRoute — Required Knowledge, Libraries and Tools

This is the team capability and dependency plan, not agent instructions.
The stack follows [ARCHITECTURE.md](ARCHITECTURE.md); verification requirements
come from [EVALUATION.md](EVALUATION.md). Only the small runtime and development
set is needed for the scaffold. Add research/operations dependencies when their
milestones land, then resolve and record exact versions in the lockfile.

## Required knowledge

| Area | Systems person | Theory/statistics person | Practical readiness check |
|---|---|---|---|
| Python 3.12, typing, packaging | Own async code, `src` packaging and config | Write typed numerical code and reproducible scripts | Install package, run server and checks from a fresh environment |
| HTTP and OpenAI-compatible chat | Own auth, validation, connection pools, SSE, cancellation and retries | Understand how timeouts/scope change measured outcomes | Mock streaming/non-streaming contract preserves chunks and `[DONE]` |
| Conformal risk control | Understand loss, strict comparator and fallback | Own theorem assumptions, bounded monotone loss and marginal expectation | Explain ties, finite-sample correction and always-strong fallback |
| Statistical evaluation | Capture provenance and execute runs | Own resampling, uncertainty, multiplicity and negative controls | Interpret G1 without confusing non-rejection with proof |
| ML routing | Own embeddings/features, fitting and inference | Review separation, ranking and savings–risk behavior | Freeze a router fitted only on `D_router` |
| Quality measurement | Implement normalized scorers and sandbox execution | Own metric validity, noise, bias and sensitivity | Version scorer fixtures; evaluate judge agreement before use |
| Shift and monitoring | Implement audits, queues, storage and metrics | Own detection power, false alarms and recalibration assumptions | Uniformly sample both routed tiers and measure delay/cost |
| Reproducibility and privacy | Own hashes, resumability, secret handling and redaction | Review data leakage, exclusions and claims | Trace a figure to config, code SHA, dataset and artifact |

Both people should be able to explain why a better router improves savings but
does not replace calibration; why exchangeability can fail; why a marginal
guarantee is not per-user; and why predictor probability calibration is distinct
from conformal threshold calibration.

## Scaffold dependencies

| Library/tool | Purpose | Scope |
|---|---|---|
| Python 3.12 | Reference interpreter from the architecture | Runtime; newer interpreters require verification |
| FastAPI + Uvicorn | ASGI app, routes and local server | Runtime |
| Pydantic v2 + pydantic-settings | Typed config, environment variables and `.env` loading | Runtime |
| PyYAML | Parse checked-in gateway configuration | Runtime |
| HTTPX | Async provider transport and TestClient support | Runtime/development |
| python-dotenv | Resolve provider key names from `.env`, with environment precedence | Runtime |
| setuptools | Build the installable `src` package | Build |
| pytest | Small startup/API checks, later contract/failure tests | Development |
| Ruff + mypy + PyYAML stubs | Lint, formatting and type checking | Development |
| uv (preferred), pip/venv (alternative) | Environment and dependency management | Tooling |

`pyproject.toml` is the dependency source of truth; `requirements.txt` delegates
to that package. The runtime does not install embedding models, download datasets,
contact providers or run calibration at startup; chat requests contact providers. Uvicorn's
default event loop is portable; optional uvloop acceleration belongs on supported
platforms, not as a Windows requirement.

## Libraries to add by milestone

| Milestone | Library/tool | Intended use |
|---|---|---|
| M1 | NumPy, SciPy | CRC arrays, known-risk simulation and statistical calculations |
| M1 | pandas, dataset loaders as needed | Paired data preparation and analysis; respect original dataset licences |
| M1 | Hypothesis, respx, OpenAI Python SDK | CRC properties, mocked HTTPX failures, client compatibility contracts |
| M1–M2 | scikit-learn, sentence-transformers | Logistic baseline, metrics and CPU prompt embeddings |
| M2 | LightGBM | Alternative predictor/ablation |
| M2 | matplotlib / seaborn | Scripted F1–F6 and reproducible tables/figures |
| M3–M4 | structlog, prometheus-client | Redacted structured logs and gateway/risk metrics |
| M4 | SQLAlchemy + SQLite | Requests, audits and calibration metadata |
| M4 | Streamlit | MVP-lite dashboard reading stored outcomes |
| M3–M4 | Locust | Mock/real upstream load, TTFT and overhead experiments |
| M0–M5 | pre-commit, GitHub Actions, Docker/Compose, Make | Checks, CI, demo packaging and reproduction commands |

Stretch-only options include ONNX Runtime, FAISS, PyTorch MLP experiments,
Postgres, Redis, OpenTelemetry and Grafana. Install these only for an accepted
experiment or measured operational need. LTT, weighted CRC and online updates
need statistical analysis as well as software; a library does not establish
their assumptions.

## Learning and implementation order

1. Read the existing architecture/evaluation together. Reproduce the strict `>`
   loss on a tiny hand-calculated dataset, including tied scores.
2. Systems builds the ASGI/provider contract; Theory builds CRC/property fixtures.
   Both agree the paired-data and artifact contracts in [SPLIT.md](SPLIT.md).
3. Practice deduplication, disjoint splits and fitting on a small frozen pilot.
   Fix seeds, versions, generation settings and epsilon grids before results.
4. Reproduce mean-risk estimates across repeated splits, adjusted intervals and
   negative controls. Understand finite-pool uncertainty and test-set noise.
5. Add serving integration and failure injection, then audit/drift experiments.
   Delay weighted/online/group claims until the corresponding analysis is ready.

## Statistical and operational pitfalls to know

- CRC uses `n/(n+1) * empirical_risk + B/(n+1)`; tiny samples may force
  `lambda = 1`. With bounded scores and strict `>`, that policy has zero regret
  versus the same strong model even when no corrected candidate is feasible.
- The grid is predetermined. Choosing epsilon based on attractive calibration
  outcomes is a different procedure requiring separate validation.
- G1 concerns mean risk over calibration draws. Individual test splits may exceed
  epsilon. The evaluation's CI criterion is non-rejection, not proof of validity.
- Repeated splits reuse a finite dataset; state the scope of the resulting
  uncertainty. LTT's population-risk probability is not identical to the fraction
  of finite test estimates below epsilon.
- Auditing only cheap decisions biases the reusable calibration sample. Sample
  all in-scope requests uniformly and retain selection metadata.
- New model versions, metrics, router weights, generation settings or traffic
  distributions can invalidate the calibration claim.
- Scoring generated code needs a real sandbox with resource limits. Prompt and
  answer retention for datasets must be explicit and separate from redacted logs.
- Prices require a dated config; API collection is separate from offline
  reproduction and should be budgeted using both tiers' token volume.

## Evidence of readiness to release

Use the ROADMAP gates, not self-assessed familiarity: numerical fixtures and
properties for CRC; mock-provider and failure contracts for serving; real-output
G1/G6, baseline and shift results for the MVP claim; and a reproducible result
bundle for publication. Verify bibliographic references listed in README before
the report; this plan introduces no new benchmark results or citations.
