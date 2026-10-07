"""
comparisons/XGBoostModel.py
-----------------------------
Gradient boosted trees.  Attempts to import xgboost; falls back to
sklearn.ensemble.GradientBoostingRegressor if unavailable.  The model_name
field in the CSV records which implementation was used so results are not
misattributed.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comparisons.shared_metrics import compute_metrics

# Try XGBoost first, fall back to sklearn GBR
try:
    import xgboost as xgb
    _BACKEND = 'xgboost'
except ImportError:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        _BACKEND = 'sklearn_gbr'
    except ImportError:
        _BACKEND = None


class XGBoostModel:
    """
    Gradient-boosted trees.
    Uses xgboost if installed, otherwise sklearn GradientBoostingRegressor.
    The active backend is recorded in model_name so results are unambiguous.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 random_state: int = 42):
        if _BACKEND is None:
            raise ImportError(
                "Neither xgboost nor scikit-learn is available. "
                "Install one to use XGBoostModel."
            )
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.random_state  = random_state

        if _BACKEND == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
            self.MODEL_NAME = "XGBoost"
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state,
            )
            self.MODEL_NAME = "GradientBoosting(sklearn)"

    @property
    def config_str(self) -> str:
        return (f"backend={_BACKEND},"
                f"n_estimators={self.n_estimators},"
                f"max_depth={self.max_depth},"
                f"lr={self.learning_rate},"
                f"subsample={self.subsample}")

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
            full = f"[{self.MODEL_NAME}]: {msg}"
            if logger:
                logger.log(full, 4, True)
            else:
                print(full)

        # ── Build feature arrays ──────────────────────────────────────
        n_total = len(train_records) + len(test_records)
        _log(f"Building arrays — {n_total} samples, {len(feature_cols)} features…")
        with _progress("[cyan]Building arrays…", total=n_total) as (prog, task):
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
        _log(f"Fitting {self.n_estimators} boosting rounds…")
        t0 = time.time()
        with _progress(
            f"[yellow]Fitting {self.MODEL_NAME} ({self.n_estimators} rounds)…"
        ) as _:
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
            'model_name':             self.MODEL_NAME,
            'model_config':           self.config_str,
            'n_train':                len(train_records),
            'n_test':                 len(actuals),
            'n_features':             len(feature_cols),
            'target':                 target,
            'epoch_count':            self.n_estimators,
            'train_time_sec':         round(train_time, 4),
            'inference_time_sec':     round(inference_time, 6),
            'best_epoch':             '',
            'best_test_error_pct':    '',
            'related_train_error_pct':'',
            'notes':                  f'backend={_BACKEND}',
        })
        _log(
            f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"R²={metrics['r2']:.4f}  MAPE={metrics['mape_pct']:.2f}%  "
            f"Acc@10%={metrics['acc_10_pct']:.1f}%  F1={metrics['f1']:.4f}"
        )
        return metrics
