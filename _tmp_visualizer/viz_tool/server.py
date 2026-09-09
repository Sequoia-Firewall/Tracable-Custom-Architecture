"""
viz_tool/server.py
--------------------
Browser-based viewer for TCA training/inference output: node graphs,
per-segment confidence, signal paths, and logs — made easy to read without
digging through raw JSON/log files by hand.

Design goals (in priority order, per the actual ask):
  1. Minimal system load. This is stdlib-only (http.server) — no Flask, no
     dependencies to install, no background threads, no file-watchers, no
     polling loop running server-side. Every request does a small, bounded
     amount of file I/O + JSON parsing and returns — nothing runs when
     nobody's looking at the browser. If you want "live" updates, the
     frontend polls on an interval YOU control (default: off, manual
     refresh button) — never a tight loop hammering the server, which is
     what would actually compete with training for CPU.
  2. Reads existing artifacts, not a live hook into training. Segment
     graphs come from .nexseg files (already written every time a segment
     improves), JudgeNode's routing state from judge_node.judgestate, logs
     from the .log files RichConsole already writes. The only thing that
     didn't already exist is trace.jsonl (opt-in, see SystemHandler.runInfer
     (trace=...) / settings.infer.trace_enabled) — appended once per
     INFERENCE call, never per training sample.
  3. Works with any TCA run directory — point --dir at wherever a run's
     .nexseg/.judgestate/logs/trace.jsonl files live. Nothing here imports
     TCA's own Python modules; it only reads the JSON/text files they write.

Usage
-----
    python3 server.py --dir /path/to/a/tca/run/dir --port 8642

Then open http://localhost:8642 in a browser.
"""
import argparse
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Visualization is disabled outright above this many dimensions — geometric
# rendering (2D canvas, or the simple 3D projection) stops being meaningful
# past 3D, and nobody asked for a 4D+ projection scheme.
MAX_VISUALIZABLE_DIMENSIONS = 3


class DataStore:
    """All file reads live here — one place to reason about I/O cost and
    keep every read bounded (no full-file loads of unbounded-size logs)."""

    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)

    def _path(self, *parts) -> str:
        return os.path.join(self.run_dir, *parts)

    # ── Segments ────────────────────────────────────────────────────────

    def list_segments(self) -> list[dict]:
        out = []
        if not os.path.isdir(self.run_dir):
            return out
        for fname in sorted(os.listdir(self.run_dir)):
            m = re.match(r"^segment_(\d+)\.nexseg$", fname)
            if not m:
                continue
            seg_id = int(m.group(1))
            try:
                with open(self._path(fname)) as f:
                    data = json.load(f)
            except Exception:
                continue
            out.append({
                "segment_id": seg_id,
                "max_x": data.get("max_x"),
                "dimensions": data.get("dimensions"),
                "n_nodes": len(data.get("processing_nodes", [])),
                "n_reviewers": len(data.get("reviewers", [])),
            })
        return out

    def get_segment(self, seg_id: int, include_weights: bool = False) -> dict | None:
        path = self._path(f"segment_{seg_id}.nexseg")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        nodes = []
        for n in data.get("processing_nodes", []):
            entry = {
                "position": n.get("position"),
                "connected_positions": n.get("connected_positions", []),
            }
            if include_weights:
                entry["weights"] = n.get("weights", {})
            nodes.append(entry)
        return {
            "segment_id": data.get("segment_id"),
            "max_x": data.get("max_x"),
            "dimensions": data.get("dimensions"),
            "pred_min": data.get("pred_min"),
            "pred_max": data.get("pred_max"),
            "splitter": {
                "position": data.get("splitter", {}).get("position"),
                "connected_positions": data.get("splitter", {}).get("connected_positions", []),
            },
            "reviewers": data.get("reviewers", []),
            "processing_nodes": nodes,
        }

    # ── JudgeNode routing state ────────────────────────────────────────

    def get_judge_state(self) -> dict | None:
        path = self._path("judge_node.judgestate")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    # ── Overall status (drives the >3D disable) ────────────────────────

    def get_dimensions(self) -> int | None:
        judge = self.get_judge_state()
        segs = self.list_segments()
        if segs and segs[0].get("dimensions") is not None:
            return segs[0]["dimensions"]
        # Fall back to inferring from a centroid's length isn't reliable
        # (centroids are full feature-length, not geometric dims) — if no
        # segment file is available yet, dimensions is simply unknown.
        return None

    # ── Logs ────────────────────────────────────────────────────────────

    def list_log_files(self) -> list[str]:
        logs_dir = self._path("logs")
        if not os.path.isdir(logs_dir):
            return []
        return sorted(
            [f for f in os.listdir(logs_dir) if f.endswith(".log")],
            key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)),
            reverse=True,
        )

    def tail_log(self, filename: str, limit: int = 500, min_level: int | None = None) -> list[dict]:
        """Read at most the last `limit` lines of a log file — bounded I/O
        regardless of how large the file has grown, using a chunked seek-
        from-end instead of loading the whole file into memory."""
        # filename comes straight from a query param — reject any path
        # separators outright (log files are always flat inside logs/) and
        # resolve symlinks/".." before the containment check so a value like
        # "../../../etc/passwd" can't walk out of logs_dir.
        if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
            return []
        logs_dir = os.path.realpath(self._path("logs"))
        path = os.path.realpath(os.path.join(logs_dir, filename))
        if os.path.commonpath([path, logs_dir]) != logs_dir:
            return []
        if not os.path.exists(path):
            return []

        chunk_size = 65536
        lines: list[str] = []
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            pos = file_size
            buf = b""
            while pos > 0 and len(lines) <= limit:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                buf = f.read(read_size) + buf
                lines = buf.split(b"\n")
        if lines and lines[-1] == b"":
            lines = lines[:-1]  # trailing newline produces a bogus empty "line"
        tail = lines[-limit:] if len(lines) > limit else lines

        _LEVEL_RE = re.compile(r"^\[\s*(INFO|WARNING|ERROR|DEBUG)\s*\]:\s*(.*)$")
        _LEVEL_NUM = {"DEBUG": 1, "ERROR": 2, "WARNING": 3, "INFO": 4}
        entries = []
        for raw in tail:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            if not text.strip():
                continue
            m = _LEVEL_RE.match(text)
            if m:
                level_name, message = m.group(1), m.group(2)
                level_num = _LEVEL_NUM.get(level_name, 4)
            else:
                level_name, level_num, message = "INFO", 4, text
            if min_level is not None and level_num < min_level:
                continue
            entries.append({"level": level_name, "level_num": level_num, "message": message})
        return entries

    # ── Trace (signal paths + confidence over time) ────────────────────

    def read_trace(self, limit: int = 100, trace_filename: str = "trace.jsonl") -> list[dict]:
        if not trace_filename or "/" in trace_filename or "\\" in trace_filename or trace_filename in (".", ".."):
            return []
        run_dir_real = os.path.realpath(self.run_dir)
        path = os.path.realpath(os.path.join(run_dir_real, trace_filename))
        if os.path.commonpath([path, run_dir_real]) != run_dir_real:
            return []
        if not os.path.exists(path):
            return []
        chunk_size = 65536
        lines: list[bytes] = []
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            pos = file_size
            buf = b""
            while pos > 0 and len(lines) <= limit:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                buf = f.read(read_size) + buf
                lines = buf.split(b"\n")
        if lines and lines[-1] == b"":
            lines = lines[:-1]  # trailing newline produces a bogus empty "line"
        tail = [l for l in (lines[-limit:] if len(lines) > limit else lines) if l.strip()]
        records = []
        for raw in tail:
            try:
                records.append(json.loads(raw))
            except Exception:
                continue
        return records

    # ── Epoch metrics (training curves) ────────────────────────────────

    def read_epoch_metrics(self, limit: int = 2000) -> list[dict]:
        path = self._path("error-epoch.csv")
        if not os.path.exists(path):
            return []
        import csv
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-limit:]


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, default=str).encode("utf-8")


def make_handler(store: DataStore):
    class Handler(BaseHTTPRequestHandler):
        # Quiet the default per-request stderr logging — one more small
        # step toward "minimal load", and keeps the terminal usable.
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, obj, status=200):
            body = _json_bytes(obj)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path, content_type):
            if not os.path.exists(path):
                self.send_error(404, "Not found")
                return
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
                return
            if path == "/app.js":
                self._send_file(os.path.join(STATIC_DIR, "app.js"), "application/javascript")
                return
            if path == "/style.css":
                self._send_file(os.path.join(STATIC_DIR, "style.css"), "text/css")
                return

            if path == "/api/status":
                dims = store.get_dimensions()
                self._send_json({
                    "run_dir": store.run_dir,
                    "dimensions": dims,
                    "visualization_enabled": (dims is not None and dims <= MAX_VISUALIZABLE_DIMENSIONS),
                    "max_visualizable_dimensions": MAX_VISUALIZABLE_DIMENSIONS,
                    "has_judge_state": store.get_judge_state() is not None,
                })
                return

            if path == "/api/segments":
                self._send_json(store.list_segments())
                return

            m = re.match(r"^/api/segment/(\d+)$", path)
            if m:
                include_weights = qs.get("full", ["0"])[0] == "1"
                seg = store.get_segment(int(m.group(1)), include_weights=include_weights)
                if seg is None:
                    self.send_error(404, "Segment not found")
                    return
                self._send_json(seg)
                return

            if path == "/api/judge":
                judge = store.get_judge_state()
                if judge is None:
                    self._send_json({})
                    return
                self._send_json(judge)
                return

            if path == "/api/log_files":
                self._send_json(store.list_log_files())
                return

            if path == "/api/logs":
                filename = qs.get("file", [None])[0]
                if not filename:
                    files = store.list_log_files()
                    filename = files[0] if files else None
                if not filename:
                    self._send_json([])
                    return
                limit = int(qs.get("limit", ["500"])[0])
                min_level = qs.get("min_level", [None])[0]
                min_level = int(min_level) if min_level is not None else None
                self._send_json(store.tail_log(filename, limit=limit, min_level=min_level))
                return

            if path == "/api/trace":
                limit = int(qs.get("limit", ["100"])[0])
                trace_filename = qs.get("file", ["trace.jsonl"])[0]
                self._send_json(store.read_trace(limit=limit, trace_filename=trace_filename))
                return

            if path == "/api/epoch_metrics":
                limit = int(qs.get("limit", ["2000"])[0])
                self._send_json(store.read_epoch_metrics(limit=limit))
                return

            self.send_error(404, "Not found")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="TCA visualization/log viewer (minimal-load, stdlib-only)")
    parser.add_argument("--dir", default=".", help="TCA run directory to serve data from (default: cwd)")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()

    store = DataStore(args.dir)
    handler = make_handler(store)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving TCA run data from: {store.run_dir}")
    print(f"Open http://localhost:{args.port} in a browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
