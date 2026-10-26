# Offline evaluation dashboard

A standalone FastAPI app with plain HTML/CSS/JavaScript. It uses the existing
runtime dependencies, serves all assets locally, and makes no provider calls.
The gateway's serving endpoints are unchanged. An operational audit/drift
dashboard remains future work.

From the repository root, with the project environment activated:

```bash
python -m uvicorn certiroute.dashboard.app:app --host 127.0.0.1 --port 8765
```

Open [localhost:8765](http://127.0.0.1:8765). The desktop layout shows three
side-by-side cards; on mobile they stack. Expand **Explore the saved studies**
for plots. Browser screenshots need no external fonts or services.

## Data and interpretation

The default published snapshots are:

- `results/gsm8k-sensitivity/results.json`: all three policy cards, 16 selectable
  operating points and the headline savings. Cheap/strong baselines average the
  same training-seed means as CertiRoute. Cost per 1,000 is mean test-set cost
  divided by the fixed test-set size, times 1,000.
- `results/gsm8k-shift/results.json`: separate shift panel, with the selected
  epsilon but a fixed calibration size of 100. Both directions are reported.
- Their `plots/` subdirectories: saved PNG figures served through an allowlist.

The app reloads the JSON files on each page load and verifies matching dataset,
model, metric and pricing metadata. It does not scan experimental/draft folders
or silently select a newer incompatible run. Missing/malformed files produce
an explicit error, not demo numbers. `/api/results` exposes only presentation
metrics and public provenance, with source-file hashes shown in the footer.

Quality is the saved numeric-answer accuracy. Clipped regret is averaged per
question; it is not strong accuracy minus routed accuracy. Estimated savings
average the saved per-seed/draw ratios, whereas cost bars show the ratio of mean
costs; minor rounding differences are expected. Non-rejection is the study's
adjusted lower-CI criterion, not a production guarantee.

## View a reproduced run

Keep the same directory names below a new result root:

```text
results/reproduced/
  gsm8k-sensitivity/results.json
  gsm8k-sensitivity/plots/*.png
  gsm8k-shift/results.json
  gsm8k-shift/plots/*.png
```

macOS/Linux:

```bash
CERTIROUTE_RESULTS_DIR=results/reproduced python -m uvicorn certiroute.dashboard.app:app --host 127.0.0.1 --port 8765
```

Windows PowerShell:

```powershell
$env:CERTIROUTE_RESULTS_DIR = "results/reproduced"
python -m uvicorn certiroute.dashboard.app:app --host 127.0.0.1 --port 8765
```

Remove that variable to return to the included snapshots. Use a different port
if a dashboard is already running. The static HTML/CSS/JS is included as Python
package data; result files remain repository artifacts.

## Screenshot provenance

The README images in `docs/assets/` capture the real app at epsilon 0.05 and
calibration size 500. They are presentation artifacts, not new experiments.
See `docs/assets/provenance.json` for the source-result hashes and image hashes.
After changing the UI or default data, recapture them at that same operating
point. Scientific plots are reproduced with the commands in the main README.
