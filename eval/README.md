# Offline pipeline

Run the complete local fixture pipeline with:

    python -m eval.offline --out results/my-offline-run

It learns a logistic predictor on local prompt features, builds CRC calibration,
then replays held-out requests through the gateway for cheap-only, strong-only
and every configured CertiRoute epsilon. No external calls or keys are needed.

See [OFFLINE_EVALUATION.md](../docs/OFFLINE_EVALUATION.md) for input formats,
split isolation, provenance, output files and limitations.

Individual commands:
- python -m eval.make_fixture: generate simulated paired arithmetic observations.
- python -m eval.fit_router: fit JSON weights on the disjoint training dataset.
- python -m eval.calibrate: calibrate the loaded learned predictor on calibration data.
- python -m eval.offline: fit, calibrate, compare and report all policies locally.

Synthetic outputs/token counts/prices do not establish real-provider quality
or savings. Real collection, the G1 guarantee suite, shift experiments, broader
baselines and plotting remain planned in EVALUATION.md.
