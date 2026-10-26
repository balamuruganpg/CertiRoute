"""Standalone read-only dashboard. Run from the repository root with uvicorn."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from certiroute.dashboard.data import load_results

STATIC = Path(__file__).with_name("static")


def create_app(main_path: Path | None = None, shift_path: Path | None = None) -> FastAPI:
    root = Path(os.environ.get("CERTIROUTE_RESULTS_DIR", "results"))
    main_path = main_path or root / "gsm8k-sensitivity/results.json"
    shift_path = shift_path or root / "gsm8k-shift/results.json"
    application = FastAPI(title="CertiRoute · Offline results", docs_url=None, redoc_url=None)
    application.mount("/assets", StaticFiles(directory=STATIC), name="assets")

    @application.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @application.get("/api/results")
    def results() -> JSONResponse:
        try:
            data = load_results(main_path, shift_path)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Evaluation results unavailable or incompatible. Check the saved "
                    "sensitivity and shift results under CERTIROUTE_RESULTS_DIR, then reload.",
                },
            )
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @application.get("/plots/{name}", response_model=None)
    def plot(name: str) -> FileResponse | JSONResponse:
        plots = {
            key: main_path.parent / "plots" / f"{key}.png"
            for key in (
                "risk-cost",
                "calibration-sweep",
                "seed-advantage",
            )
        }
        plots["shift-risk"] = shift_path.parent / "plots/shift-risk.png"
        path = plots.get(name)
        if path is None or not path.is_file():
            return JSONResponse(status_code=404, content={"error": "Plot unavailable."})
        return FileResponse(path)

    return application


app = create_app()
