"""
comparisons/ComparisonManager.py
----------------------------------
Orchestrates all comparison models on a single shared train/test split.
Collects every model's metrics dict, writes comparison_results.csv, and
renders a Rich comparison table to the console.

Fairness guarantees:
  - PreProcessingNode runs ONCE; the same preprocessed records go to all
    sklearn / CNN models.
  - The 80/20 split is performed once; all models see identical train/test sets.
  - SegmentHandler receives the full DataFrame reconstructed from those same
    records (train first, test second) so its internal split reproduces ours.
  - Metrics are computed by shared_metrics.compute_metrics for every model
    except SegmentHandler (which recomputes from its own inference pass using
    the same function).
  - Timing uses time.perf_counter for sub-millisecond accuracy.
"""

import os
import re
import sys
import csv
import time
import math
import json
import hashlib
import datetime as _dt
import importlib
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from Components.PreProcessingNode import PreProcesingNode
from comparisons.shared_metrics import (
    compute_metrics, append_comparison_row,
    COMPARISON_CSV, CSV_FIELDNAMES,
)


class ComparisonManager:
    """Run all models on the same data and generate a unified comparison report."""

    # ── tiny tqdm adapter so callers use the same (prog, task) protocol ──
    class _TqdmAdapter:
        """Wraps a tqdm bar behind the (prog, task_id) interface used internally."""
        def __init__(self, bar):
            self._bar = bar

        def update(self, _task, total=None, completed=None, advance=None, description=None):
            if total is not None and self._bar.total != total:
                self._bar.total = total
                self._bar.refresh()
            if completed is not None:
                self._bar.n = completed
                self._bar.refresh()
            elif advance is not None:
                self._bar.update(advance)
            if description is not None:
                self._bar.set_description(ComparisonManager._strip_markup(description))

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _strip_markup(text: str) -> str:
        """Remove Rich markup tags (e.g. [bold cyan]) from a string."""
        return re.sub(r'\[/?[^\[\]]*\]', '', text)

    def __init__(self, logger=None):
        self.logger           = logger
        self.results          = []   # list of metrics dicts, one per model run
        self._checkpoint      = {}   # loaded/active checkpoint data
        self._checkpoint_path = None # path to current checkpoint JSON
        self._output_csv      = COMPARISON_CSV # overridable per run_all() call

    # ── logging ──────────────────────────────────────────────────────────

    def _log(self, msg: str, level: int = 4):
        full = f"[ComparisonManager]: {msg}"
        if self.logger:
            self.logger.log(full, level, True)
        else:
            print(self._strip_markup(full))

    def _rule(self, title: str):
        console = getattr(self.logger, 'console', None)
        if console is not None:
            from rich.rule import Rule
            console.print(Rule(f"[bold yellow]{title}[/bold yellow]"))
        else:
            plain = self._strip_markup(title)
            width = max(60, len(plain) + 4)
            print(f"\n{'═' * width}")
            print(f"  {plain}")
            print('═' * width)

    @contextmanager
    def _progress(self, desc: str, total=None, transient: bool = True):
        """Yield (progress, task_id).

        - If a logger with make_progress is available, use that (Rich).
        - Else if tqdm is installed, wrap a tqdm bar in _TqdmAdapter.
        - Otherwise yield (None, None) silently.
        """
        mp = getattr(self.logger, 'make_progress', None)
        if mp:
            with mp(transient=transient) as prog:
                task = prog.add_task(desc, total=total)
                yield prog, task
            return

        try:
            from tqdm import tqdm
            plain_desc = self._strip_markup(desc)
            bar = tqdm(total=total, desc=plain_desc, leave=not transient,
                       dynamic_ncols=True, unit='item')
            adapter = self._TqdmAdapter(bar)
            try:
                yield adapter, 0
            finally:
                bar.close()
        except ImportError:
            yield None, None

    # ── checkpointing ────────────────────────────────────────────────────

    def _job_hash(self, dataset_path: str, target: str,
                  epoch_count: int, max_x_list: list, system_max_x_list: list,
                  output_csv: str) -> str:
        key = json.dumps({
            "dataset":    dataset_path,
            "target":     target,
            "epochs":     epoch_count,
            "max_x":      sorted(max_x_list),
            "system_max_x": sorted(system_max_x_list),
            # Folded into the hash so two notebooks writing to different CSVs
            # (e.g. comparison_results.csv vs segment_comparison_results.csv)
            # never share a checkpoint and silently reuse each other's results.
            "output_csv": output_csv,
        }, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _init_checkpoint(self, dataset_path: str, target: str,
                         epoch_count: int, max_x_list: list, system_max_x_list: list,
                         output_csv: str):
        """Load existing checkpoint if job params match; otherwise start fresh."""
        job_hash  = self._job_hash(dataset_path, target, epoch_count, max_x_list,
                                    system_max_x_list, output_csv)
        ckpt_path = f"comparison_checkpoint_{job_hash}.json"
        self._checkpoint_path = ckpt_path

        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path) as fh:
                    data = json.load(fh)
                completed = data.get("completed_models", {})
                self._checkpoint = data
                self._log(
                    f"Checkpoint found: {ckpt_path}  "
                    f"({len(completed)} model(s) already done — will skip them)"
                )
                # Reload already-completed results so final report is complete
                for name, result in completed.items():
                    self.results.append(result)
                return
            except Exception as e:
                self._log(f"Could not load checkpoint ({e}); starting fresh.", level=3)

        self._checkpoint = {
            "job": {
                "dataset":    dataset_path,
                "target":     target,
                "epoch_count":epoch_count,
                "max_x":      sorted(max_x_list),
                "system_max_x": sorted(system_max_x_list),
            },
            "completed_models": {},
        }
        self._log(f"New checkpoint: {ckpt_path}")

    def _save_checkpoint(self, model_key: str, result: dict):
        """Record model result in checkpoint JSON immediately after completion."""
        if not self._checkpoint_path:
            return
        self._checkpoint.setdefault("completed_models", {})[model_key] = result
        tmp = self._checkpoint_path + ".tmp"
        try:
            with open(tmp, 'w') as fh:
                json.dump(self._checkpoint, fh, indent=2, default=str)
            os.replace(tmp, self._checkpoint_path)
        except Exception as e:
            self._log(f"Warning: could not write checkpoint — {e}", level=3)

    def _is_checkpointed(self, model_key: str) -> bool:
        return model_key in self._checkpoint.get("completed_models", {})

    # ── main entry point ─────────────────────────────────────────────────

    def run_all(
        self,
        dataset,
        target:         str,
        epoch_count:    int        = 20,
        segment_max_x             = 20,   # int or list[int]
        run_segment:    bool       = True,
        run_cnn:        bool  = True,
        run_linear:     bool  = True,
        run_knn:        bool  = True,
        run_rf:         bool  = True,
        run_system:     bool  = True,
        system_max_x              = 10,   # int or list[int]; one run per value
        system_dimensions:  int   = 2,
        system_training_mode: str = "partitioned",
        system_judge_iterations: int = 20,
        system_judge_min_clusters: int | None = None,
        system_judge_max_clusters: int | None = None,
        system_aggregation_modes      = None,  # list[str] | str | None → defaults to all 3
        system_selection_percentage: float = 0.5,
        run_xgb:        bool  = True,
        run_mlp:        bool  = True,
        output_csv:     str | None = None,
    ):
        """
        Preprocess once, split once, run every enabled model, save CSV,
        then render the comparison report.

        Parameters
        ----------
        dataset       : path to CSV or pandas DataFrame
        target        : name of the target column
        epoch_count   : epochs for SegmentHandler and CNN
        segment_max_x : SegmentHandler max_x hyperparameter
        run_*         : toggle individual models on/off
        output_csv    : path to write results to (default: comparison_results.csv).
                         Pass a distinct path when running this manager from a
                         notebook other than the main system comparison, so its
                         runs never append to / collide with that file or its
                         checkpoint.
        """
        self._output_csv = output_csv or COMPARISON_CSV
        run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rule(f"Comparison Run  {run_id}")

        # ── Resolve max_x lists early so checkpoint hash is stable ────
        max_x_values = (
            segment_max_x
            if isinstance(segment_max_x, (list, tuple))
            else [segment_max_x]
        )
        system_max_x_values = (
            system_max_x
            if isinstance(system_max_x, (list, tuple))
            else [system_max_x]
        )
        _ALL_AGG_MODES = ["bma", "simple_mean", "relevance_weighted"]
        if system_aggregation_modes is None:
            agg_modes = _ALL_AGG_MODES
        elif isinstance(system_aggregation_modes, str):
            agg_modes = [system_aggregation_modes]
        else:
            agg_modes = list(system_aggregation_modes)
        dataset_path = dataset if isinstance(dataset, str) else "<DataFrame>"
        self._init_checkpoint(dataset_path, target, epoch_count, list(max_x_values),
                               list(system_max_x_values), self._output_csv)

        # ── Count total models for [X/N] tracking ─────────────────────
        self._model_total = (
            (1 if run_linear else 0) +
            (2 if run_knn    else 0) +
            (1 if run_rf     else 0) +
            (1 if run_xgb    else 0) +
            (1 if run_mlp    else 0) +
            (1 if run_cnn    else 0) +
            (len(max_x_values) if run_segment else 0) +
            (len(system_max_x_values) * len(agg_modes) if run_system else 0)
        )
        self._model_idx = 0
        self._log(f"{self._model_total} models queued for this run.")

        # ── Step 1: Preprocess ────────────────────────────────────────
        self._rule("Step 1 — Shared Preprocessing")
        preprocessor = PreProcesingNode(Logger=self.logger, logger_classification=4)

        if isinstance(dataset, str):
            self._log(f"Loading dataset: {dataset}")
            with self._progress("Loading CSV…", transient=True) as (prog, task):
                dataset = pd.read_csv(dataset)
                if prog:
                    prog.update(task, total=1, completed=1)

        n_rows = len(dataset)
        self._log(f"Running PreProcessingNode on {n_rows} rows (shared across all models)…")
        with self._progress(f"Preprocessing {n_rows} rows…", transient=True) as (prog, task):
            processed_df = preprocessor.process_dataset(dataset.copy()).fillna(0)
            if prog:
                prog.update(task, total=1, completed=1)
        all_cols = list(processed_df.columns)

        if target not in all_cols:
            raise ValueError(f"Target '{target}' not found after preprocessing.")

        feature_cols = [c for c in all_cols if c != target]
        n_features   = len(feature_cols)
        self._log(f"Preprocessing complete — {n_features} features, target='{target}'")

        # ── Step 2: Single shared split ───────────────────────────────
        self._rule("Step 2 — Single Train / Test Split  (80 / 20)")
        records = processed_df.to_dict(orient='records')
        split   = max(1, int(len(records) * 0.8))
        train_  = records[:split]
        test_   = records[split:]
        self._log(
            f"{len(records)} samples  →  "
            f"train={len(train_)} ({100*len(train_)//len(records)}%)   "
            f"test={len(test_)} ({100*len(test_)//len(records)}%)"
        )
        self._log("All models will receive this identical split.")

        # ── Step 3: Run models ────────────────────────────────────────
        self._rule("Step 3 — Running Models")

        common = dict(
            train_records=train_,
            test_records=test_,
            feature_cols=feature_cols,
            target=target,
            logger=self.logger,
        )
        # Linear Regression
        if run_linear:
            self._run_model("LinearRegression", lambda: self._run_linear(**common))

        # KNN k=5
        if run_knn:
            self._run_model("KNN(k=5)",  lambda: self._run_knn(k=5,  **common))
            self._run_model("KNN(k=10)", lambda: self._run_knn(k=10, **common))

        
        # Random Forest
        if run_rf:
            self._run_model("RandomForest", lambda: self._run_rf(**common))

        # XGBoost / GBR
        if run_xgb:
            self._run_model("XGBoost", lambda: self._run_xgb(**common))

        # MLP (sklearn, properly implemented)
        if run_mlp:
            self._run_model("MLP(sklearn)", lambda: self._run_mlp(**common))

        # Pure-Python CNN
        if run_cnn:
            self._run_model("CNN", lambda: self._run_cnn(
                epoch_count=epoch_count, **common
            ))

        # SegmentHandler — one run per max_x value
        if run_segment:
            for mx in max_x_values:
                self._run_model(
                    f"SegmentHandler(max_x={mx})",
                    lambda mx=mx: self._run_segment(
                        epoch_count=epoch_count,
                        max_x=mx,
                        **common,
                    ),
                )

        # SystemHandler (full nexus) — one run per (max_x × aggregation_mode)
        if run_system:
            for mx in system_max_x_values:
                for agg in agg_modes:
                    self._run_model(
                        f"SystemHandler(max_x={mx},agg={agg})",
                        lambda mx=mx, agg=agg: self._run_system(
                            epoch_count=epoch_count,
                            max_x=mx,
                            dimensions=system_dimensions,
                            training_mode=system_training_mode,
                            judge_iterations=system_judge_iterations,
                            judge_min_clusters=system_judge_min_clusters,
                            judge_max_clusters=system_judge_max_clusters,
                            aggregation_mode=agg,
                            selection_percentage=system_selection_percentage,
                            **common,
                        ),
                    )

        # ── Step 4: Tag results with run metadata and save ────────────
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in self.results:
            r.setdefault('timestamp', ts)
            r.setdefault('run_id',    run_id)
        self._save_comparison_csv(run_id)

        # ── Step 5: Report ────────────────────────────────────────────
        self.generate_report()

    # ── model runners ────────────────────────────────────────────────────

    def _run_model(self, name: str, fn):
        """Call fn(), catch errors, append result to self.results.
        Skips execution if a checkpoint result already exists for this model.
        """
        self._model_idx = getattr(self, '_model_idx', 0) + 1
        idx_tag = f"[{self._model_idx}/{getattr(self, '_model_total', '?')}]"

        # ── Checkpoint resume ─────────────────────────────────────────
        if self._is_checkpointed(name):
            self._log(f"{idx_tag} {name} — already in checkpoint, skipping.")
            return

        self._rule(f"  {idx_tag} Model: {name}")
        try:
            result = fn()
            self.results.append(result)
            self._save_checkpoint(name, result)
            self._log(
                f"{idx_tag} {name} done — "
                f"R²={result.get('r2', float('nan')):.4f}  "
                f"MAPE={result.get('mape_pct', float('nan')):.2f}%  "
                f"F1={result.get('f1', float('nan')):.4f}  "
                f"train={result.get('train_time_sec', '?')}s"
            )
        except Exception as e:
            self._log(f"{name} FAILED — {e}", level=2)
            import traceback
            self._log(traceback.format_exc(), level=2)

    def _run_linear(self, **kw):
        from comparisons.LinearRegressionModel import LinearRegressionModel
        return LinearRegressionModel().run(**kw)

    def _run_knn(self, k: int, **kw):
        from comparisons.KNNModel import KNNModel
        return KNNModel(k=k).run(**kw)

    def _run_rf(self, **kw):
        from comparisons.RandomForestModel import RandomForestModel
        return RandomForestModel(n_estimators=100).run(**kw)

    def _run_xgb(self, **kw):
        from comparisons.XGBoostModel import XGBoostModel
        return XGBoostModel(n_estimators=300).run(**kw)

    def _run_mlp(self, **kw):
        from comparisons.MLPModel import MLPModel
        return MLPModel(hidden_layer_sizes=(128, 64, 32), max_iter=500).run(**kw)

    def _run_cnn(self, epoch_count: int, **kw):
        from comparisons.CNN import CNN
        cnn = CNN(target=kw['target'], logger=kw['logger'])
        return cnn.run(
            train_records=kw['train_records'],
            test_records=kw['test_records'],
            feature_cols=kw['feature_cols'],
            target=kw['target'],
            epoch_count=epoch_count,
        )

    def _run_segment(self, epoch_count: int, max_x: int, **kw):
        from comparisons.SegmentHandlerWrapper import SegmentHandlerWrapper
        wrapper = SegmentHandlerWrapper(max_x=max_x)
        return wrapper.run(
            train_records=kw['train_records'],
            test_records=kw['test_records'],
            feature_cols=kw['feature_cols'],
            target=kw['target'],
            epoch_count=epoch_count,
            logger=kw['logger'],
        )

    def _run_system(
        self,
        epoch_count:           int,
        max_x:                 int,
        dimensions:            int   = 2,
        training_mode:         str   = "partitioned",
        judge_iterations:      int   = 20,
        judge_min_clusters:    int | None = None,
        judge_max_clusters:    int | None = None,
        aggregation_mode:      str   = "bma",
        selection_percentage:  float = 0.5,
        **kw,
    ):
        from comparisons.SystemHandlerWrapper import SystemHandlerWrapper
        wrapper = SystemHandlerWrapper(
            max_x=max_x,
            dimensions=dimensions,
            training_mode=training_mode,
            judge_iterations=judge_iterations,
            judge_min_clusters=judge_min_clusters,
            judge_max_clusters=judge_max_clusters,
            aggregation_mode=aggregation_mode,
            selection_percentage=selection_percentage,
        )
        return wrapper.run(
            train_records=kw['train_records'],
            test_records=kw['test_records'],
            feature_cols=kw['feature_cols'],
            target=kw['target'],
            epoch_count=epoch_count,
            logger=kw['logger'],
        )

    # ── CSV persistence ───────────────────────────────────────────────────

    def _save_comparison_csv(self, run_id: str):
        """Append all results from this run to self._output_csv."""
        n = len(self.results)
        self._log(f"Writing {n} rows to {self._output_csv}…")
        with self._progress(
            f"Saving {n} rows → {self._output_csv}",
            total=n, transient=True,
        ) as (prog, task):
            write_header = not os.path.exists(self._output_csv)
            with open(self._output_csv, 'a', newline='') as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=CSV_FIELDNAMES, extrasaction='ignore'
                )
                if write_header:
                    writer.writeheader()
                for r in self.results:
                    row = {}
                    for field in CSV_FIELDNAMES:
                        v = r.get(field, '')
                        if isinstance(v, float) and math.isnan(v):
                            row[field] = ''
                        elif isinstance(v, float):
                            row[field] = f"{v:.6f}"
                        else:
                            row[field] = v if v is not None else ''
                    writer.writerow(row)
                    if prog:
                        prog.update(task, advance=1)
        self._log(f"Saved → {self._output_csv}")

    def merge_existing_csvs(self):
        """
        Read historical rows from error-epoch.csv (SegmentHandler) and
        cnn_error_epoch.csv (CNN standalone) and add them to self.results
        so they appear in the comparison report.  Useful for including
        previously completed runs without re-training.
        """
        self._log("Merging existing model CSVs into comparison results…")

        # SegmentHandler historical runs
        self._merge_csv(
            path="error-epoch.csv",
            model_name_field='segment_id',
            model_name_prefix='SegmentHandler',
            config_fields=['max_x', 'dimensions', 'connection_pct', 'density'],
        )

        # CNN standalone runs
        self._merge_csv(
            path="cnn_error_epoch.csv",
            model_name_field='model_type',
            model_name_prefix='CNN',
            config_fields=['n_filters', 'kernel_size', 'n_conv_layers',
                           'dense_hidden_sizes', 'pool_size'],
        )

    def _merge_csv(self, path, model_name_field, model_name_prefix, config_fields):
        if not os.path.exists(path):
            self._log(f"  {path} not found — skipping.", level=3)
            return
        count = 0
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                cfg = ','.join(
                    f"{f}={row.get(f, '')}"
                    for f in config_fields if row.get(f)
                )
                merged = {
                    'timestamp':   row.get('timestamp', ''),
                    'run_id':      f"historical_{path}",
                    'model_name':  f"{model_name_prefix}(historical)",
                    'model_config':cfg,
                    'target':      row.get('target', ''),
                    'n_features':  row.get('n_features', ''),
                    'n_train':     row.get('n_train', ''),
                    'n_test':      row.get('n_test', ''),
                    'epoch_count': row.get('epoch_count', ''),
                    'train_time_sec':    '',
                    'inference_time_sec':'',
                    'mae':         row.get('mae', ''),
                    'rmse':        row.get('rmse', ''),
                    'r2':          row.get('r2', ''),
                    'mape_pct':    row.get('mape_pct', ''),
                    'acc_5_pct':   row.get('acc_5_pct', ''),
                    'acc_10_pct':  row.get('acc_10_pct', ''),
                    'acc_20_pct':  row.get('acc_20_pct', ''),
                    'direction_acc_pct': row.get('direction_acc_pct', ''),
                    'precision':   row.get('precision', ''),
                    'recall':      row.get('recall', ''),
                    'f1':          row.get('f1', ''),
                    'tp': row.get('tp', ''), 'fp': row.get('fp', ''),
                    'fn': row.get('fn', ''), 'tn': row.get('tn', ''),
                    'best_epoch':             row.get('best_epoch', ''),
                    'best_test_error_pct':    row.get('best_test_error_pct', ''),
                    'related_train_error_pct':row.get('related_train_error_pct', ''),
                    'notes': f'from {path}',
                }
                self.results.append(merged)
                count += 1
        self._log(f"  Merged {count} row(s) from {path}")

    # ── report ────────────────────────────────────────────────────────────

    def generate_report(self):
        """Print a Rich comparison table; falls back to plain text."""
        if not self.results:
            self._log("No results to report.", level=3)
            return

        self._rule("Comparison Report")

        console = getattr(self.logger, 'console', None)
        if console is not None:
            self._rich_report(console)
        else:
            self._plain_report()

    def _rich_report(self, console):
        from rich.table import Table
        from rich import box

        # Sort by R² descending (best model first)
        def _r2(r):
            try:
                return float(r.get('r2', -999) or -999)
            except (ValueError, TypeError):
                return -999.0

        sorted_results = sorted(self.results, key=_r2, reverse=True)

        t = Table(
            title="Model Comparison",
            box=box.ROUNDED, border_style="yellow",
            show_lines=True,
        )
        cols = [
            ("Model",       "bold white",   "left"),
            ("R²",          "bright_green", "right"),
            ("MAPE %",      "cyan",         "right"),
            ("Acc@10%",     "cyan",         "right"),
            ("Acc@20%",     "cyan",         "right"),
            ("F1",          "magenta",      "right"),
            ("MAE",         "white",        "right"),
            ("RMSE",        "white",        "right"),
            ("Train (s)",   "yellow",       "right"),
            ("Infer (s)",   "yellow",       "right"),
            ("Epochs",      "white",        "right"),
            ("Config",      "dim white",    "left"),
        ]
        for name, style, justify in cols:
            t.add_column(name, style=style, justify=justify)

        def _f(v, dec=4):
            try:
                fv = float(v)
                return f"{fv:.{dec}f}" if not math.isnan(fv) else "—"
            except (TypeError, ValueError):
                return str(v) if v else "—"

        best_r2 = _r2(sorted_results[0]) if sorted_results else -999

        for r in sorted_results:
            r2_val = _r2(r)
            r2_str = f"[bold bright_green]{r2_val:.4f}[/]" if r2_val == best_r2 else _f(r.get('r2'))
            t.add_row(
                str(r.get('model_name', '?')),
                r2_str,
                _f(r.get('mape_pct'), 2),
                _f(r.get('acc_10_pct'), 1),
                _f(r.get('acc_20_pct'), 1),
                _f(r.get('f1')),
                _f(r.get('mae')),
                _f(r.get('rmse')),
                _f(r.get('train_time_sec'), 2),
                _f(r.get('inference_time_sec'), 4),
                str(r.get('epoch_count', '—')),
                str(r.get('model_config', ''))[:48],
            )
        console.print(t)
        console.print(
            f"\n[dim]Full results saved to [bold]{self._output_csv}[/bold][/dim]\n"
        )

    def _plain_report(self):
        headers = ["Model", "R²", "MAPE%", "Acc@10%", "F1", "MAE", "Train(s)"]
        rows = []
        for r in sorted(
            self.results,
            key=lambda x: float(x.get('r2') or -999),
            reverse=True,
        ):
            def _f(k, d=4):
                try:
                    return f"{float(r.get(k, 'nan')):.{d}f}"
                except Exception:
                    return '—'
            rows.append([
                str(r.get('model_name', '?')),
                _f('r2'), _f('mape_pct', 2), _f('acc_10_pct', 1),
                _f('f1'), _f('mae'), _f('train_time_sec', 2),
            ])
        col_w = [max(len(h), max((len(row[i]) for row in rows), default=0))
                 for i, h in enumerate(headers)]
        sep = '+-' + '-+-'.join('-' * w for w in col_w) + '-+'
        fmt = '| ' + ' | '.join(f'{{:<{w}}}' for w in col_w) + ' |'
        print(sep)
        print(fmt.format(*headers))
        print(sep)
        for row in rows:
            print(fmt.format(*row))
        print(sep)
        print(f"\nResults saved to {self._output_csv}")
