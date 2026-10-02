"""Web server for the LLM Disagreement Analysis System (standard library only).

    python web/server.py            ->  http://127.0.0.1:8000
Endpoints: GET /  |  GET /api/sample  |  POST /api/run
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Config, ConfigError          # noqa: E402
from src.data_collection import DatasetError        # noqa: E402
from src.llm_clients import LLMUnavailable          # noqa: E402
from src.pipeline import run_pipeline               # noqa: E402

MAX_BODY = 1_000_000


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.states: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("[STATE]"):
            self.states.append(msg)


def run_analysis(p: dict) -> tuple[int, dict]:
    """Run the real pipeline for one web request. Returns (http status, json body)."""
    tmp = Path(tempfile.mkdtemp(prefix="llmd_"))
    data = ROOT / "data" / "sample_dataset.csv"
    if p.get("csv"):                                   # STD: submitInput(dataset, ...)
        data = tmp / "upload.csv"
        data.write_text(p["csv"], encoding="utf-8")
    try:
        models = [m.strip() for m in p.get("models", []) if m.strip()][:6]
        cfg = Config(data_path=data, output_dir=tmp / "out", backend=p.get("backend", "mock"),
                     models=models or Config().models, n_samples=min(int(p.get("samples", 5)), 20),
                     agreement_threshold=float(p.get("threshold", 0.75)), use_cache=False)
    except (TypeError, ValueError) as exc:
        return 400, {"error": f"bad parameters: {exc}"}
    cap = _Capture()
    log = logging.getLogger("pipeline")
    log.addHandler(cap)
    try:
        res = run_pipeline(cfg)
    except (DatasetError, ConfigError) as exc:          # STD: configDone [invalid]
        return 400, {"error": str(exc), "states": cap.states}
    except LLMUnavailable as exc:                       # STD: queryFailed
        return 502, {"error": f"LLM Model Pool unavailable: {exc}", "states": cap.states}
    finally:
        log.removeHandler(cap)
    text = {i.item_id: i.text for i in res["items"]}
    m = res["metrics"].astype(object).where(res["metrics"].notna(), None)
    rows = m.to_dict("records")
    for r in rows:
        r["text"] = text[r["item_id"]]
    return 200, {"summary": res["summary"], "items": rows, "states": cap.states,
                 "config": {"backend": cfg.backend, "models": cfg.models,
                            "samples": cfg.n_samples, "threshold": cfg.agreement_threshold}}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode(),
                   "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/sample":
            self._send(200, (ROOT / "data" / "sample_dataset.csv").read_bytes(), "text/csv; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/run":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._json(413, {"error": "upload too large (max 1 MB)"})
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "request body is not valid JSON"})
        code, body = run_analysis(payload)
        self._json(code, body)

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.address_string(), fmt % args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
