# Running the minimal gateway

Run from the repository root with Python 3.12 or newer. Python 3.12 remains the
architecture's reference version. The current implementation forwards real
OpenAI-compatible responses and supports a learned local-feature router with CRC
thresholds. It is not the completed evaluation or production gateway.

For the fully local path, start with [OFFLINE_EVALUATION.md](OFFLINE_EVALUATION.md).
It needs neither provider configuration nor API keys and includes the learned
router, calibration and a held-out quality/cost comparison. The provider setup
below applies when you later switch to real model outputs.

## Install and configure

With uv:

```powershell
uv sync --extra dev
Copy-Item .env.example .env
```

Or on Windows with Python/venv:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Skip the copy if you already have a configured .env. On POSIX use
`python3.12 -m venv .venv`, `.venv/bin/python` and `cp .env.example .env`.

Edit `configs/gateway.yaml`:

- Set `models.cheap.base_url` and `models.strong.base_url` to the provider API
  roots ending in `/v1` (not `/chat/completions`).
- Set the exact cheap and strong model IDs. Both tiers may use the same provider
  endpoint or different OpenAI-compatible providers/local servers.
- Set `CHEAP_API_KEY` and `STRONG_API_KEY` in .env. The adapter resolves each
  configured `api_key_env` name; process environment wins over .env. Blank keys
  omit Authorization, useful for local servers.
- Replace the gateway development key before exposing the service. Gateway
  credentials are never forwarded upstream.
- Choose models supporting the configured chat generation parameters:
  temperature 0, max_tokens 256 and top_p 1 by default. No model IDs/prices are
  assumed or recommended by the implementation.

Start:

```powershell
.\.venv\Scripts\python -m uvicorn certiroute.gateway.app:app --host 127.0.0.1 --port 8080
```

The uv equivalent is `uv run uvicorn certiroute.gateway.app:app --port 8080`.
Open `http://127.0.0.1:8080/docs` for the API schema.

## First real request: strong passthrough

No artifact is needed for forwarding. With no valid artifact, auto routes to
strong and readiness is 503. Health is 200. In a second PowerShell terminal:

```powershell
$body = @{
    model = "certiroute/auto"
    messages = @(@{role = "user"; content = "What is 17 * 23?"})
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType application/json -Headers @{Authorization = "Bearer certiroute-local"} -Body $body
```

This contacts the configured upstream and can incur provider costs. The gateway
replaces the alias with that tier's model ID and returns the upstream completion,
including its choices, model ID and usage. It does not manufacture an answer.
Provider HTTP/connection/timeout errors return sanitized error responses.

`stream: true` forwards SSE content and `[DONE]` unchanged. The adapter reads the
first chunk before committing downstream headers so initial failures can fall
back. Once bytes have been sent, failures emit an SSE error event and close the
stream; they never splice in another model.

## Build a calibration artifact

1. Freeze model revisions/endpoints, metric, feature version, hyperparameters,
   scope and epsilon grid before examining calibration outcomes. The learned
   score is a logistic estimate of zero regret using prompt-side features only.
   Scaling statistics and weights are trained exclusively on `D_router`.
2. Collect both models' outputs at the exact configured settings, score each
   using the versioned quality metric in [EVALUATION.md](../EVALUATION.md), and
   isolate the calibration split. Preserve raw outputs, usage and collection
   metadata separately. Do not calibrate on final test data or tune the score on
   calibration outcomes. Exact and lexical near duplicates are rejected by the
   training/calibration scripts; semantic leakage checks remain a responsibility
   of the collection and splitting pipeline.
3. Save a frozen paired-output JSON using the format below. Model identities are
   `<base_url_without_trailing_slash>::<model_id>`. The loader requires the dataset
   model identities, metric and scope to match the config.
4. Fit the router on the separate training JSON, then calibrate on the calibration
   JSON. Both commands run offline and refuse to overwrite existing artifacts:

```powershell
.\.venv\Scripts\python -m eval.fit_router --config configs/gateway.yaml --input data/router-train.json --out artifacts/router.json
.\.venv\Scripts\python -m eval.calibrate --config configs/gateway.yaml --input data/calibration-pairs.json --out artifacts/calibration.json
```

The following is a **schema illustration**, not real calibration data:

```json
{
  "models": {
    "cheap": "http://127.0.0.1:8001/v1::replace-with-cheap-model-id",
    "strong": "http://127.0.0.1:8002/v1::replace-with-strong-model-id"
  },
  "quality_metric": {"name": "exact_match", "version": "1"},
  "scope": {"max_chars": 16000, "temperature": 0, "max_tokens": 256, "top_p": 1},
  "sources": ["replace-with-collected-source"],
  "collected_at": "2026-10-23T00:00:00Z",
  "rows": [
    {
      "request": {
        "messages": [{"role": "user", "content": "What is 17 * 23?"}],
        "temperature": 0,
        "max_tokens": 256,
        "top_p": 1
      },
      "q_strong": 1.0,
      "q_cheap": 0.0
    }
  ]
}
```

Use actual paired scores in [0,1]. The CLI trusts their correctness and provenance;
hashing detects content changes, not truthful labels. One row will yield
always-strong at the default grid. The finite-sample correction needs roughly
`n >= 1/epsilon - 1` even for a nontrivial candidate; aim for at least 1,000
representative calibration prompts as planned in EVALUATION.md.

Artifacts contain the corrected thresholds, sample count, dataset hash, scorer
identity/hash, provider/model identities, metric, scope/decoding settings, creation
and expiry times. Their id hashes canonical content excluding the id itself.
The CLI refuses to overwrite an existing file; use a new artifact filename for
each recalibration, update the config path and restart the gateway. Default
validity is seven days (`--valid-days` changes it); this is an operational expiry
policy, not a guarantee that traffic stays exchangeable for that duration.

## Calibrated routing and API behavior

After creating an artifact, restart the gateway. Readiness becomes 200 when its
schema, hash, identities, scope, epsilon grid and expiry validate. It does not
probe provider availability or certify the dataset's representativeness.
Expiry is checked on every request as well as at startup.

Auto routes cheap iff `score > threshold`; ties route strong. Missing, malformed,
expired or mismatched artifacts preserve strong passthrough and readiness 503.
There is no interpolation between epsilons.

The calibrated scope accepts one user text message, optionally preceded by one
system text message, at the configured generation settings and length limit.
Omitted temperature/max_tokens/top_p are filled with configured values for both
tiers. Other generation parameters, multi-turn messages, tools, multimodal
content, n > 1 and excess length go strong. Unknown options are forwarded to
strong rather than silently dropped. This initial guard does not detect language
or embedding-based distribution shift; exchangeability remains an assumption.

| API/header | Behavior |
|---|---|
| GET /healthz | Process liveness |
| GET /readyz | 200 with a valid artifact; otherwise 503 |
| GET /v1/models | Authenticated list of auto/strong/cheap aliases |
| POST /v1/chat/completions | Authenticated JSON or SSE forwarding |
| x-certiroute-epsilon | Must be on the configured/calibrated grid; default from config |
| x-certiroute-mode | auto/strong/cheap; may override auto alias; conflicts with explicit tier aliases return 400 |
| x-certiroute-allow-downgrade | Only literal true enables strong-to-cheap failure fallback |

Explicit strong/cheap aliases bypass calibration; explicit cheap is tagged
degraded and excluded from the claim. Cheap-provider failure before output falls
back once to strong. Strong-provider failure does not silently downgrade; an
opted-in downgrade is tagged degraded with no calibration id. A failed fallback
returns an error without cycling between providers.

Routing headers report request id, final tier, score (or unavailable), epsilon,
calibration id (or none), degraded flag, scope and reason. A validated artifact
supports marginal expected regret only under the assumptions in ARCHITECTURE.md.
Neither readiness nor a happy-path test establishes real-traffic validity.

## Checks and current limits

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m mypy
```

Tests use synthetic calibration fixtures and mocked HTTPX upstreams, including
both tiers, SSE ordering, strict ties, invalid/expired artifacts, fallback and
the calibration CLI. They incur no provider costs. Run the real-output repeated
split, baseline, negative-control and shift experiments before claiming quality
or cost guarantees; none have been measured by this increment.

The minimal adapter pools connections and bounds connect/read/total timeouts
(defaults 5/60/120 seconds per provider attempt; configurable under each model).
Automatic same-tier retries/backoff, circuit breakers, rate limits, metrics,
live pricing, audit/drift jobs, embedding features, language/OOD detection, admin
reload, Docker/Compose and the full evaluation suite remain future increments.
The existing README offline/Makefile commands are still target interfaces except
for the calibration command documented here.
