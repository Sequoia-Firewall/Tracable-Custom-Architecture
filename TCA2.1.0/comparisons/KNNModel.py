"""
comparisons/KNNModel.py
------------------------
sklearn KNeighborsRegressor.  Geometrically motivated like SegmentHandler —
both use proximity in feature space to make predictions.  Run at k=5 and k=10
to bracket the sensitivity to the k parameter.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comparisons.shared_metrics import compute_metrics

try:
    from sklearn.neighbors import KNeighborsRegressor
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


class KNNModel:
    """K-Nearest Neighbours regression."""

    MODEL_NAME = "KNN"

    def __init__(self, k: int = 5, weights: str = 'distance'):
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn is required for KNNModel.")
        self.k       = k
        self.weights = weights
        self.model   = KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=-1)

    @property
    def config_str(self) -> str:
        return f"k={self.k},weights={self.weights}"

    def run(self, train_records: list, test_records: list,
            feature_cols: list, target: str, logger=None) -> dict:
        from contextlib import contextmanager

        @contextmanager
        def _progress(desc, total=None, transient=True):
            mp = getattr(logger, 'make_progress', None)
            if mp:
                with mp(transient=transient) as prog:
                    task = prog.add_task(desc, total=total)
                    yield prog, task
            else:
                yield None, None

        def _log(msg):
            full = f"[KNN(k={self.k})]: {msg}"
            if logger:
                logger.log(full, 4, True)
            else:
                print(full)

        # ── Build feature arrays ──────────────────────────────────────
        n_total = len(train_records) + len(test_records)
        _log(f"Building arrays — {n_total} samples, {len(feature_cols)} features…")
        with _progress(f"[cyan]Building arrays…", total=n_total) as (prog, task):
            X_train, y_train = [], []
            for r in train_records:
                X_train.append([float(r.get(f, 0)) for f in feature_cols])
                y_train.append(float(r.get(target, 0)))
                if prog: prog.update(task, advance=1)
            X_test, y_test = [], []
            for r in test_records:
                X_test.append([float(r.get(f, 0)) for f in feature_cols])
                y_test.append(float(r.get(target, 0)))
                if prog: prog.update(task, advance=1)

        # ── Fit ───────────────────────────────────────────────────────
        _log(f"Fitting KNN(k={self.k}) on {len(X_train)} samples…")
        t0 = time.time()
        with _progress(f"[yellow]Fitting KNN(k={self.k})…") as _:
            self.model.fit(X_train, y_train)
        train_time = time.time() - t0
        _log(f"Training complete in {train_time:.3f}s")

        # ── Inference ─────────────────────────────────────────────────
        _log(f"Running inference on {len(X_test)} test samples…")
        t1 = time.time()
        with _progress("[green]Inference…") as _:
            preds = list(self.model.predict(X_test))
        inference_time = time.time() - t1

        pa = [(p, a) for p, a in zip(preds, y_test) if a != 0]
        predictions = [p for p, _ in pa]
        actuals     = [a for _, a in pa]

        metrics = compute_metrics(predictions, actuals)
        metrics.update({
            'model_name':             f"{self.MODEL_NAME}(k={self.k})",
            'model_config':           self.config_str,
            'n_train':                len(train_records),
            'n_test':                 len(actuals),
            'n_features':             len(feature_cols),
            'target':                 target,
            'epoch_count':            1,
            'train_time_sec':         round(train_time, 4),
            'inference_time_sec':     round(inference_time, 6),
            'best_epoch':             '',
            'best_test_error_pct':    '',
            'related_train_error_pct':'',
            'notes':                  '',
        })
        _log(
            f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"R²={metrics['r2']:.4f}  MAPE={metrics['mape_pct']:.2f}%  "
            f"Acc@10%={metrics['acc_10_pct']:.1f}%  F1={metrics['f1']:.4f}"
        )
        return metrics
