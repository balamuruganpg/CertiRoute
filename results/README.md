# Published evaluation snapshots

This directory includes the reports, aggregate metrics, figures, frozen weights
and split metadata referenced by the main README. Paired model outputs are under
`data/eval/`; no provider calls are needed to reproduce the studies.

Detailed generated `draws.jsonl` and `draws.jsonl.gz` logs are excluded from Git
to keep clones and uploads manageable. The original local logs are retained,
their checksums remain in each `results.json`, and rerunning the documented
evaluation commands produces them in the new output directory.

Use a new output directory for every run. Unpublished reruns and draft plot
directories are also excluded from Git. See the main README for commands.
