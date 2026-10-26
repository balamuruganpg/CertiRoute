# Calibration artifacts

Create an artifact from frozen paired observations with `python -m eval.calibrate`;
see [DEVELOPMENT.md](../docs/DEVELOPMENT.md). No real calibration data or thresholds
are bundled. Do not use the architecture's illustrative values for serving.

Artifacts are content-addressed and cannot be overwritten by the writer. Their
model endpoints/IDs, router hash, metric, scope, epsilon grid and expiry must
match serving configuration. Missing or invalid artifacts make readiness false
and auto traffic strong-only. Restart after pointing config at a new artifact.

Learned logistic weights are plain JSON with a content hash and training prompt
fingerprints. The gateway loads them once at startup and checks the calibration
against that model identity; weight changes require a new calibration and restart.

The offline run saves fixture-trained weights and calibration under its results
directory, bound to fixture model identities. They cannot validate the real-model
config. Synthetic inputs live in `data/eval/sample.jsonl`; the saved report and
artifacts are in `results/offline-learned`. These do not substantiate real-model
risk or cost claims. Keep large future embedding weights in release assets or Git LFS.
