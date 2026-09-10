"""
comparisons/MLPModel.py
------------------------
sklearn MLPRegressor — a properly implemented dense neural network.
This is the fair neural-network comparison against SegmentHandler, correcting
the pure-Python CNN's implementation weaknesses (no gradient instability,
proper weight initialisation, Adam-style adaptive learning).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from comparisons.shared_metrics import compute_metrics

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


class MLPModel:
    """
    Multi-Layer Perceptron regression via sklearn.
    Uses StandardScaler internally (fit on train only) to ensure fair
    gradient behaviour — equivalent to what batch normalisation achieves
    in PyTorch networks.
    """

    MODEL_NAME = "MLP"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (128, 64, 32),
        activation:         str   = 'relu',
        max_iter:           int   = 500,
        learning_rate_init: float = 0.001,
        early_stopping:     bool  = True,
        validation_fraction:float = 0.1,
        random_state:       int   = 42,
    ):
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn is required for MLPModel.")
        self.hidden_layer_sizes  = hidden_layer_sizes
        self.activation          = activation
        self.max_iter            = max_iter
        self.learning_rate_init  = learning_rate_init
        self.early_stopping      = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state        = random_state

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver='adam',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
            verbose=False,
        )
        self.scaler = StandardScaler()

    @property
    def config_str(self) -> str:
        return (f"layers={list(self.hidden_layer_sizes)},"
                f"act={self.activation},"
                f"lr={self.learning_rate_init},"
                f"max_iter={self.max_iter},"
                f"early_stopping={self.early_stopping}")

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
            full = f"[MLP]: {msg}"
            if logger:
                logger.log(full, 4, True)
            else:
                print(full)

        # ── Build feature arrays ──────────────────────────────────────
        n_total = len(train_records) + len(test_records)
        _log(f"Building arrays — {n_total} samples, {len(feature_cols)} features…")
        with _progress("[cyan]Building arrays…", total=n_total) as (prog, task):
            X_train_raw, y_train = [], []
            for r in train_records:
                X_train_raw.append([float(r.get(f, 0)) for f in feature_cols])
                y_train.append(float(r.get(target, 0)))
                if prog: prog.update(task, advance=1)
            X_test_raw, y_test = [], []
            for r in test_records:
                X_test_raw.append([float(r.get(f, 0)) for f in feature_cols])
                y_test.append(float(r.get(target, 0)))
                if prog: prog.update(task, advance=1)

        # ── Scale (fit on train only — prevents data leakage) ─────────
        _log("Fitting StandardScaler on training data…")
        with _progress("[cyan]Fitting StandardScaler…") as _:
            X_train = self.scaler.fit_transform(X_train_raw)
            X_test  = self.scaler.transform(X_test_raw)

        # ── Fit MLP ───────────────────────────────────────────────────
        _log(
            f"Fitting MLP {list(self.hidden_layer_sizes)} on {len(X_train)} samples "
            f"(max_iter={self.max_iter}, early_stopping={self.early_stopping})…"
        )
        t0 = time.time()
        with _progress(
            f"[yellow]Fitting MLP {list(self.hidden_layer_sizes)} "
            f"(up to {self.max_iter} iters)…"
        ) as _:
            self.model.fit(X_train, y_train)
        train_time   = time.time() - t0
        actual_iters = self.model.n_iter_
        _log(f"Training complete in {train_time:.3f}s  ({actual_iters} iterations)")

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
            'epoch_count':            actual_iters,
            'train_time_sec':         round(train_time, 4),
            'inference_time_sec':     round(inference_time, 6),
            'best_epoch':             '',
            'best_test_error_pct':    '',
            'related_train_error_pct':'',
            'notes':                  f'actual_iters={actual_iters}',
        })
        _log(
            f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"R²={metrics['r2']:.4f}  MAPE={metrics['mape_pct']:.2f}%  "
            f"Acc@10%={metrics['acc_10_pct']:.1f}%  F1={metrics['f1']:.4f}"
        )
        return metrics
