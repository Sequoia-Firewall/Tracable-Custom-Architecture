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
     .nexseg/.judgestate/logs/trace.jsonl files live. Everything above is
     read-only and never imports TCA's own Python modules.

One deliberate exception: POST /api/query. It loads the trained segments
from --dir (SystemHandler.load_segments(), the same restore path hot-swap
uses) and runs ONE live runInfer() call on whatever feature values the
browser form submits — this does import TCA's modules and does execute
code, unlike everything else here. Still cheap (a single inference, not
training) and still local-only (binds to 127.0.0.1 only, no --host flag
offered on purpose) — this is meant for a developer poking at their own
run on their own machine, not a multi-user service.

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
import threading
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
        self._tca_ready = False   # lazily puts run_dir on sys.path — only needed for /api/query
        self._query_logger = None
        self._query_lock = threading.Lock()  # serialize live inference calls: the TCA codebase
                                               # seeds/consumes a process-global random.random()
                                               # stream with no locking of its own, so two
                                               # concurrent queries could interleave draws.

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

    # ── Custom query (live inference — the one non-read-only feature) ──

    MAX_CATEGORY_OPTIONS = 25
    MAX_SCHEMA_ROWS_SCANNED = 20000  # bounded — one manual button click, not a poll loop

    def get_input_schema(self) -> dict:
        """Derive the form the browser should show for a custom query: every
        dataset column except the target and any ignored_columns, tagged as
        numeric (min/max) or categorical (observed value list) by scanning
        the dataset CSV once. Falls back to 'text' for high-cardinality
        non-numeric columns rather than dumping hundreds of <option>s."""
        settings_path = self._path("settings.json")
        if not os.path.exists(settings_path):
            return {"columns": [], "error": "settings.json not found in run_dir"}
        with open(settings_path) as f:
            settings = json.load(f)
        d = settings.get("dataset", {})
        csv_path = self._path(d.get("csv_path", "dataset.csv"))
        target = d.get("target_column")
        ignored = set(d.get("ignored_columns") or [])
        if not os.path.exists(csv_path):
            return {"columns": [], "error": f"dataset csv not found: {csv_path}"}

        import csv as csv_mod
        with open(csv_path, newline="") as f:
            reader = csv_mod.reader(f)
            header = next(reader, [])
            columns = [c for c in header if c != target and c not in ignored]
            col_idx = {c: header.index(c) for c in columns}
            stats = {c: {"is_numeric": True, "values": set(), "min": None, "max": None, "overflow": False}
                     for c in columns}
            for n, row in enumerate(reader):
                if n >= self.MAX_SCHEMA_ROWS_SCANNED:
                    break
                for c in columns:
                    idx = col_idx[c]
                    raw = row[idx] if idx < len(row) else ""
                    st = stats[c]
                    try:
                        v = float(raw)
                        if st["min"] is None or v < st["min"]:
                            st["min"] = v
                        if st["max"] is None or v > st["max"]:
                            st["max"] = v
                    except ValueError:
                        st["is_numeric"] = False
                    if not st["overflow"]:
                        st["values"].add(raw)
                        if len(st["values"]) > self.MAX_CATEGORY_OPTIONS:
                            st["overflow"] = True

        result = []
        for c in columns:
            st = stats[c]
            if st["is_numeric"]:
                result.append({"name": c, "type": "numeric", "min": st["min"], "max": st["max"]})
            elif not st["overflow"]:
                result.append({"name": c, "type": "categorical", "options": sorted(st["values"])})
            else:
                result.append({"name": c, "type": "text"})
        return {"columns": result, "target_column": target}

    def _ensure_tca_importable(self) -> None:
        if not self._tca_ready:
            if self.run_dir not in sys.path:
                sys.path.insert(0, self.run_dir)
            self._tca_ready = True

    def _get_query_logger(self):
        if self._query_logger is None:
            self._ensure_tca_importable()
            import Components.RichConsole as RC
            # log_level=0 / console_level=5: below every real classification
            # (1-4), so this writes nothing to file or console. Every query's
            # result is returned directly in the HTTP response — there's
            # nothing here worth a persisted per-query log file.
            self._query_logger = RC.RichLogger(filename="viz_tool_query.log", log_level=0, console_level=5)
        return self._query_logger

    def run_query(self, feature_values: dict, aggregation_mode: str = "bma",
                  selection_percentage: float = 0.5) -> dict:
        """Load the trained segments fresh from disk (same restore path
        hot_swap_segment() uses) and run ONE live inference call on
        feature_values, trace=True so the full breakdown/signal-path
        structure comes back — same shape as any other trace.jsonl record,
        just sourced from a live call instead of a past one.

        Serialized behind self._query_lock: TCA's stochastic routing reads
        the process-global random.random() stream with no locking of its
        own, so two concurrent queries in different request threads could
        otherwise interleave draws."""
        with self._query_lock:
            self._ensure_tca_importable()
            from Settings import Settings
            from SystemHandler import SystemHandler

            settings = Settings(self._path("settings.json"))
            logger = self._get_query_logger()
            system = SystemHandler.from_settings(settings, logger)
            system.load_segments(self.run_dir)

            result = system.runInfer(
                dict(feature_values), loud=False,
                aggregation_mode=aggregation_mode, selection_percentage=selection_percentage,
                trace=True, trace_path=self._path("trace.jsonl"),
            )
            if result is None:
                raise ValueError(
                    "runInfer returned no result — check feature_values match the "
                    "columns from /api/schema."
                )
            records = self.read_trace(limit=1)
            return records[-1] if records else result


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

            if path == "/api/schema":
                self._send_json(store.get_input_schema())
                return

            self.send_error(404, "Not found")

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/query":
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length <= 0 or length > 1_000_000:  # a form of ~a dozen fields is bytes, not MB
                    self._send_json({"error": "missing or oversized request body"}, status=400)
                    return
                try:
                    body = json.loads(self.rfile.read(length))
                except Exception:
                    self._send_json({"error": "request body must be valid JSON"}, status=400)
                    return

                feature_values = body.get("feature_values")
                if not isinstance(feature_values, dict) or not feature_values:
                    self._send_json({"error": "feature_values must be a non-empty object"}, status=400)
                    return
                aggregation_mode = body.get("aggregation_mode", "bma")
                try:
                    selection_percentage = float(body.get("selection_percentage", 0.5))
                except (TypeError, ValueError):
                    self._send_json({"error": "selection_percentage must be a number"}, status=400)
                    return

                try:
                    result = store.run_query(feature_values, aggregation_mode=aggregation_mode,
                                             selection_percentage=selection_percentage)
                except FileNotFoundError as e:
                    self._send_json({"error": f"no trained segments found: {e}"}, status=400)
                    return
                except Exception as e:
                    print(f"[/api/query] {type(e).__name__}: {e}", file=sys.stderr)
                    self._send_json({"error": f"{type(e).__name__}: {e}"}, status=500)
                    return
                self._send_json(result)
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
