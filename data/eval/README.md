# Evaluation datasets

`gsm8k-full/` is the current expanded snapshot: all 1,319 public paired outputs,
with question-only deduplication and a bound prefix in `gateway-template.yaml`.
See [the repeated-split study](../../docs/GSM8K_REPEATED.md) for offline commands.
The `gsm8k/` directory preserves the original 100-example pilot.

## Synthetic fixture

`sample.jsonl` contains 240 deterministic arithmetic prompts with paired **simulated**
cheap/strong outputs, exact-match quality labels, and **invented** token counts.
No LLM was called. Regenerate an identical file with:

```powershell
python -m eval.make_fixture --out data/eval/new-sample.jsonl
```

The simulated cheap tier handles addition and a subset of multiplication; the
simulated strong tier has occasional off-by-one errors. These rules generate the
fixture labels only. The router learns coefficients from its training split and
cannot read output strings, answers, labels, record IDs or generation rules at
inference time. See `eval/make_fixture.py` for the complete deterministic generator.

The fixture is intentionally learnable and tests plumbing, not model capabilities.
The hypothetical prices in `configs/pricing.fixture.yaml` are not actual provider
prices. Report fixture accuracy and hypothetical costs as such.

The offline runner deduplicates by hashed token trigrams, then uses a fixed seed
and prompt hashes to split 40% router training, 40% calibration and 20% held-out
test. All splits, hashes, model weights, thresholds and metrics are written into
the run directory. Learned scaling/coefficients never use calibration/test labels.

## Public paired outputs

`gsm8k/paired.jsonl` contains 100 real frozen Mixtral/GPT-4 GSM8K pairs.
`gsm8k/paired.manifest.json` records exact source revisions, full prompt construction,
quality scoring, model IDs and the data checksum. Token usage is estimated from
text length; historical pricing is an explicit scenario. Included license files
and `NOTICE.md` document attribution and modifications.

See [reproduction instructions](../../docs/GSM8K_PILOT.md) and the
[saved report](../../results/gsm8k-public/REPORT.md). No download is required to run it.

