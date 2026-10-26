# Public paired GSM8K pilot

The newer [expanded repeated-split study](GSM8K_REPEATED.md) uses all 1,319 pairs
and fixes duplicate checking to exclude the shared prefix. This document preserves
the original 100-example pilot and its historical results.

The repository now includes 100 real frozen pairs from
[LMSYS RouteLLM](https://github.com/lm-sys/RouteLLM/tree/0b64fdafe049e596a3f5657c219329f24af24198/routellm/evals/gsm8k).
The cheap model is `mistralai/Mixtral-8x7B-Instruct-v0.1`; the strong model is
`gpt-4-1106-preview`. Responses are unchanged from the public CSV. No API keys,
LLM calls, embedding downloads, or additional libraries are needed to evaluate them.

## Reproduce offline

From the repository root, using the existing installed environment:

```powershell
.\.venv\Scripts\python -m eval.offline --input data/eval/gsm8k/paired.jsonl --config configs/offline.gsm8k.yaml --pricing configs/pricing.gsm8k.yaml --out results/my-gsm8k-run
```

Choose a new output directory. This fits the existing seven-feature logistic
predictor on 40 examples, calibrates CRC on 40 disjoint examples, and compares
all six policies on the remaining 20 through the actual FastAPI gateway.
The epsilon grid, seed 42, training procedure and features are unchanged from
the synthetic experiment. No settings were tuned on the held-out scores.

The [saved report](../results/gsm8k-public/REPORT.md) contains the complete table;
`results.json` records every decision, model/config/price/data/code hashes,
Python/package versions and the dataset manifest. Regenerating changes artifact
timestamps/IDs and absolute paths; splits, weights, thresholds and aggregate
metrics are deterministic for the same code, inputs, seed and environment.

The recorded environment is CPython 3.13.14 on Windows, with runtime and test-tool
versions in [`requirements-eval.lock`](../requirements-eval.lock). For a fresh
environment, install that file, then install this project with `pip install -e .
--no-deps`. Package installation requires available wheels; the evaluation itself
uses no network and needs no source cache or model downloads.

The existing standalone commands also work on the generated splits:

```powershell
.\.venv\Scripts\python -m eval.calibrate --config results/my-gsm8k-run/gateway.yaml --input results/my-gsm8k-run/calibration.json --out results/my-gsm8k-run/recalibrated.json
```

Use `eval.fit_router` with the generated `train.json` to fit separately. Never
train on the entire snapshot before calibration. Artifacts expire after seven
days; rerun locally to create fresh ones. Archived results remain inspectable.

## What is measured and what is estimated

| Field | Provenance |
|---|---|
| Questions/reference answers | GSM8K test set, pinned RouteLLM repository copy |
| Cheap/strong output text | Real frozen public model responses |
| Model IDs | CSV columns and pinned generation script |
| Prompt | Reconstructed exactly from the script: first eight training examples plus test question, one user message; full text in each row |
| Decoding | Script specifies temperature 0, max_tokens 1024, stop `["Question"]` |
| top_p | Not explicit in source; 1 is an assumed replay default |
| Quality | Recomputed using RouteLLM's last unsigned integer after comma removal; verified against both cached correctness columns |
| Token usage | **Estimated** as `ceil(len(text)/4)` separately for the full prompt and each output; no chat framing or model-specific tokenizer |
| Pricing | Historical scenario from RouteLLM v1 Appendix D, dated June 26, 2024 |
| Original generation date/provider deployment/latency/finish reason | Unavailable, explicitly recorded as unknown |

Pricing per million input/output tokens is $0.24/$0.24 for Mixtral (the paper's
assumption) and $10/$30 for GPT-4-1106. See
[Appendix D](https://arxiv.org/html/2406.18665v1#A4) and
[`pricing.gsm8k.yaml`](../configs/pricing.gsm8k.yaml). These are not current
quotes or measured bills. The USD table models selected generation costs only;
it excludes paired collection and router hosting. Actual API spend for this run is zero.

The scorer intentionally preserves the source's limitations with decimals,
negative signs, and trailing numbers. It is versioned as
`routellm_gsm8k_last_integer:1`, not generic exact match or a general math verifier.

## Selection and scientific limits

Sort all 1,319 source questions by SHA256 of their UTF-8 text. Greedily retain the
first 100 passing the existing full-prompt token-trigram Jaccard < 0.9 guard.
Selection never examines outputs or correctness. The eight-example common prefix
makes this guard conservative and favors some longer or distinctive questions;
this is **not a random representative GSM8K sample**. The manifest records source
indices. The ordinary seed-based 40/40/20 split and train/cal/test overlap checks
then run unchanged. Lexical deduplication does not exclude semantic duplicates.

The held-out results are cheap-only 45%, strong-only 95%, and CertiRoute 85% at
epsilon 0.05 with 20.70% estimated savings. Its regret is 0.10, above the target.
At epsilon 0.10, quality is 60% and regret 0.35. The stricter 0.01/0.02 settings
route everything strong because the finite-sample correction is 1/41.
All operating points are reported, including these unfavorable outcomes.

This is a reproducible public-output pilot aligned with EVALUATION.md's GSM8K
metric, paired regret and split protocol. It is not G1's repeated-split guarantee
study, a live G9 run, or evidence of savings at a validated risk target. CRC bounds
expected regret over exchangeable draws, not realized regret on each split.
Benchmark contamination, the small sample, selection bias, approximate usage,
and unavailable generation metadata limit interpretation. The replay endpoints
use `.invalid` domains; their artifacts cannot be applied unchanged to live
providers. Collect actual usage and fully specified generations before live claims.

## Rebuild the snapshot (optional)

Normal evaluation uses only the included JSONL and its checksum-validated
manifest. To independently reconstruct them from upstream, run:

```powershell
.\.venv\Scripts\python -m eval.import_gsm8k --download --out data/eval/gsm8k-rebuilt
```

Only this explicit `--download` operation uses the network, fetching public files
at pinned Git revisions. All inputs are checked against hardcoded SHA256 values;
corrupt or changed inputs fail closed. With the source files already cached under
`data/eval/routellm-source`, omit `--download` to rebuild offline. The source cache
is ignored; it is not required for evaluation. The frozen JSONL is deterministic
and includes full prompts, outputs, ground truths, derived scores and estimated
usage. The adjacent manifest binds model IDs, scope, source URLs/checksums and
the snapshot checksum. RouteLLM's Apache-2.0 and GSM8K's MIT notices are included.
