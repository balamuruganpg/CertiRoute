# Multi-benchmark evidence

The version-one expansion evaluates **GSM8K, ARC-Challenge and MBPP** with the
existing seven-feature logistic router and bounded-regret CRC. It adds evidence,
not a new routing algorithm. Results: [saved report](../results/multibench-v1/REPORT.md)
and [machine-readable metrics](../results/multibench-v1/results.json).

## Sources and scoring

| Task | Frozen pairs | Cheap / strong | Quality |
|---|---:|---|---|
| GSM8K | 1,319 | Mixtral-8x7B-Instruct / GPT-4-1106-preview | Existing RouteLLM last-integer scorer |
| ARC-Challenge | 1,470 | `mistralai/mixtral-8x7b-chat` / `gpt-4-1106-preview` | RouterBench published exact-match correctness, with raw responses retained |
| MBPP | 399 | `gpt-3.5-turbo` / `gpt-4-1106-preview` | All sanitized MBPP base assertions pass in Docker |

ARC is taken from the **entire ARC-Challenge portion** of
[RouterBench](https://huggingface.co/datasets/withmartian/routerbench) revision
`784021482c3f320c6619ed4b3bb3b41a21424fcb`. We preserve its original zero-shot
prompts, responses and correctness labels. We do not reconstruct answers from
correctness labels or claim to have independently regraded ARC. The full source
pickle is decoded into inert symbolic records; no downloaded reducers execute.

MBPP pairs are the intersection of both public
[EvalPlus v0.2.0 generation archives](https://github.com/evalplus/evalplus/releases/tag/v0.2.0)
and [Google's sanitized MBPP](https://github.com/google-research/google-research/tree/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp).
The complete raw output, base assertions and setup imports are frozen. Extract
all fenced Python blocks (or the whole output if unfenced), append the base
assertions, and score one only if the process exits successfully within fifteen
seconds. Syntax errors, assertion failures and timeouts score zero.

This is **not EvalPlus extended-test accuracy**. Multiple-code-block extraction
and example code in outputs can affect results. The archive's GPT-3.5 alias does
not identify an exact dated snapshot; its original chat wrapper is not recovered.
The frozen task text is used for router features and input-cost estimates.

Execution uses a digest-pinned Python 3.12 container: no network, read-only mounts
and root filesystem, non-root UID, no capabilities, no-new-privileges, one CPU,
512 MB memory and a 64 MB temporary filesystem. Each program has a fresh temporary
directory and subprocess; timed-out process groups are killed. This is a benchmark
runner, not a public arbitrary-code execution service. The worker refuses host execution.

Reproduction found one slow amicable-number program crossing the original three-second
timeout between runs. Before freezing the final report, the timeout was uniformly
increased to fifteen seconds for every program and the complete execution was repeated
twice. This is a scorer stability correction, not a change to CRC or its target grid.

All source URLs and SHA-256 digests are in
[`sources.lock.json`](../data/eval/multibench/sources.lock.json). Dataset manifests
bind the final JSONL bytes. MBPP additionally records every execution verdict,
the container digest, worker hash and unscored dataset hash. See
[data notices](../data/eval/multibench/NOTICE.md).

## Predeclared evaluation protocol

- Seed 42; epsilon grid **1%, 2%, 5%, 10%**; **1,000** independent partition draws.
- Remove lexical duplicates before splitting, using the existing token-trigram
  Jaccard threshold 0.9. GSM8K strips its bound common demonstration prefix for
  duplicate detection. Three near-duplicate MBPP records are removed, leaving 396.
- Sort by `SHA256([seed, full messages])`; the first floor(40% N) train the router.
  Freeze its scaler, weights and feature definitions. The remaining 60% is the pool.
- In each draw, shuffle the pool using the existing `g1-shuffle-v1` seeded procedure;
  half calibrates and the rest tests. No calibration or test label trains the router.
- Calibration/test sizes: GSM8K 396/396; ARC 441/441; MBPP 119/120.
- Compare cheap-only, strong-only, fixed threshold 0.5, five analytically integrated
  random mixtures, random-ranking CRC, and learned CRC. Strict `score > threshold`.
- Save all scores, weights, first-draw calibration artifacts and split IDs. Verify
  first-draw artifact thresholds and gateway routing decisions against offline evaluation.
- Report conditional Monte Carlo intervals over whole partitions. The headline CRC
  family uses Bonferroni across four task/portfolio rows × four epsilons (16 tests).
  Lower CI ≤ epsilon means **non-rejection**, not proof of validity.

The aggregate is an **equal-task portfolio**, with a separate router and threshold
for each known task. Average per-request quality, regret and cost across tasks first;
compute savings from the ratio of aggregated costs. It is not a single global
threshold, an automatic task classifier or a guarantee for arbitrary traffic mixes.
GSM8K and ARC also have different upstream provider identifiers despite their shared
Mixtral family. Do not interchange their calibration artifacts.

## Findings at 5%

The portfolio has 85.49% quality, 4.769% regret and 32.77% estimated savings versus
its always-strong portfolio. ARC's 5.030% mean is slightly above target, with an
adjusted interval approximately [4.893%, 5.168%]; report this as non-rejection.
MBPP's cheap model outperforms the strong model in aggregate on this fixed training
split/pool. Clipped regret still counts cases where only the strong answer passes;
quality differences are not a substitute for that loss.

These figures use a different calibration-size protocol from the original
21.06% GSM8K sensitivity result. Neither result supersedes or is averaged into the other.

## Costs and limitations

All offline costs are **historical modeled scenarios**, not observed tokens or bills.
Tokens use `ceil(characters/4)`. Rates in USD per million input/output tokens are
Mixtral 0.24/0.24, GPT-3.5 0.50/1.50, GPT-4-1106-preview 10/30, labeled 2024-06-26.
GPT-3.5's dated generation snapshot is unknown: those rates are a declared scenario,
not a claim about the original generation bill. Missing MBPP chat wrappers and
hidden framing make input estimates incomplete. Acquisition and audit costs are excluded
from serving savings. Prices are recorded in each manifest, never inferred from tier names.

Additional limits: public benchmark contamination; small coding pool; lexical rather
than semantic deduplication; different model pair for code; archived responses may
be sanitized by the upstream publisher; finite pools reused across draws; a single
training seed here; ARC labels trusted from the source; no new cross-dataset transfer
claim. Current live provider models require fresh calibration, not these artifacts.

## Reproduce offline

```bash
python -m eval.multibench --out results/reproduced/multibench
python -m eval.plot_expansion --input results/reproduced/multibench/results.json --out results/reproduced/multibench/plots
```

Only the optional plot command needs `requirements-plot.lock`. No network or Docker
is needed to reevaluate the frozen scored pairs. To rebuild source imports and execute
coding tests again (source downloads and first container pull require network):

```bash
python -m eval.import_benchmarks --mbpp-revision f82046ba5aabbbb427dbfd38a254d26bff08b533 --out build/reimported
python -m eval.score_code --data build/reimported
python -m eval.multibench --data build/reimported --out results/reproduced/from-reimport
```

Use fresh output directories. Generated per-draw logs are reproducible and not committed.
