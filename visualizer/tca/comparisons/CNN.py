"""
comparisons/CNN.py
------------------
1-D Convolutional Neural Network baseline for tabular regression.

Architecture per run:
    [Conv1D → ReLU → MaxPool1D] × n_conv_layers
    → Flatten
    → [Dense → ReLU] × len(dense_hidden_sizes)
    → Dense (linear, 1 output)

Uses the same PreProcessingNode as SegmentHandler so both models receive
identical input representations.  All metrics are written to the shared
error-epoch.csv in the working directory so runs from both models can be
compared row-by-row.
"""

import math
import os
import csv
import sys
import random
import datetime as _dt

# Allow imports from the parent v10/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from Components.PreProcessingNode import PreProcesingNode


# ── Pure-Python Layer Primitives ─────────────────────────────────────────────

class _Conv1DLayer:
    """Multi-channel 1-D convolution (no activation — applied separately).

    Input shape  : [in_channels][length]
    Output shape : [n_filters][length - kernel_size + 1]   (valid padding)
    """

    def __init__(self, n_filters: int, kernel_size: int, in_channels: int = 1):
        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        scale = math.sqrt(2.0 / (in_channels * kernel_size))
        # kernels[filter][in_channel][k_pos]
        self.kernels = [
            [[random.gauss(0, scale) for _ in range(kernel_size)]
             for _ in range(in_channels)]
            for _ in range(n_filters)
        ]
        self.biases   = [0.0] * n_filters
        self._inp     = None   # [in_ch][L]  — saved for backward
        self._out_len = 0
        self._eff_k   = 0      # effective kernel size (clamped when L < kernel_size)

    def forward(self, x):
        """x: [in_ch][L].  Returns [n_filters][out_len] (pre-activation)."""
        in_ch = len(x)
        L     = len(x[0]) if x else 0
        k     = min(self.kernel_size, L)
        out_len = max(1, L - k + 1)
        self._inp = x;  self._out_len = out_len;  self._eff_k = k

        pre = []
        for f in range(self.n_filters):
            row = []
            for i in range(out_len):
                v = self.biases[f]
                for c in range(in_ch):
                    kf = self.kernels[f][c % self.in_channels]
                    for j in range(k):
                        if i + j < L:
                            v += kf[j] * x[c][i + j]
                row.append(v)
            pre.append(row)
        return pre

    def backward(self, grad, lr):
        """grad: [n_filters][out_len] (ReLU mask already applied).
        Returns grad_input: [in_ch][L]."""
        in_ch   = len(self._inp)
        L       = len(self._inp[0]) if self._inp else 0
        k       = self._eff_k
        out_len = self._out_len
        grad_in = [[0.0] * L for _ in range(in_ch)]

        for f in range(self.n_filters):
            self.biases[f] -= lr * sum(grad[f])
            for c in range(in_ch):
                kf = self.kernels[f][c % self.in_channels]
                # kernel gradient
                for j in range(k):
                    gk = sum(
                        grad[f][i] * (self._inp[c][i + j] if i + j < L else 0.0)
                        for i in range(out_len)
                    )
                    kf[j] -= lr * gk
                # input gradient
                for i in range(out_len):
                    for j in range(k):
                        if i + j < L:
                            grad_in[c][i + j] += kf[j] * grad[f][i]
        return grad_in

    def param_count(self):
        return self.n_filters * self.in_channels * self.kernel_size + self.n_filters


class _ReLULayer:
    """Stateful ReLU — stores mask for backprop.  Works on nested or flat lists."""

    def __init__(self):
        self._mask = None

    def forward(self, x):
        nested = x and isinstance(x[0], list)
        if nested:
            self._mask = [[1.0 if v > 0 else 0.0 for v in row] for row in x]
            return [[max(0.0, v) for v in row] for row in x]
        self._mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        nested = grad and isinstance(grad[0], list)
        if nested:
            return [[g * m for g, m in zip(gr, mr)]
                    for gr, mr in zip(grad, self._mask)]
        return [g * m for g, m in zip(grad, self._mask)]


class _MaxPool1DLayer:
    """Max-pool along the length axis: [n_filters][L] → [n_filters][ceil(L/pool)]."""

    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self._shape    = (0, 0)
        self._max_idx  = []    # [n_filters][n_windows]

    def forward(self, x):
        """x: [n_filters][L].  Returns [n_filters][windows]."""
        p  = self.pool_size
        nf = len(x)
        L  = len(x[0]) if x else 0
        self._shape = (nf, L)
        pooled, max_idx = [], []
        for f in range(nf):
            pr, pi = [], []
            i = 0
            while i < L:
                window = x[f][i:i + p]
                mx_v   = max(window)
                mx_i   = i + window.index(mx_v)
                pr.append(mx_v);  pi.append(mx_i)
                i += p
            pooled.append(pr);  max_idx.append(pi)
        self._max_idx = max_idx
        return pooled

    def backward(self, grad):
        """grad: [n_filters][windows].  Returns [n_filters][L]."""
        nf, L = self._shape
        g_in  = [[0.0] * L for _ in range(nf)]
        for f in range(nf):
            for j, idx in enumerate(self._max_idx[f]):
                if idx < L:
                    g_in[f][idx] += grad[f][j]
        return g_in


class _DenseLayer:
    """Fully-connected layer with relu or linear activation."""

    def __init__(self, in_size: int, out_size: int, activation: str = 'relu'):
        self.activation = activation
        scale = math.sqrt(2.0 / in_size)
        self.W    = [[random.gauss(0, scale) for _ in range(in_size)]
                     for _ in range(out_size)]
        self.b    = [0.0] * out_size
        self._inp = None
        self._pre = None   # pre-activation values

    def forward(self, x):
        self._inp = x
        pre = [sum(self.W[i][j] * x[j] for j in range(len(x))) + self.b[i]
               for i in range(len(self.W))]
        self._pre = pre
        if self.activation == 'relu':
            return [max(0.0, v) for v in pre]
        return list(pre)

    def backward(self, grad_out, lr):
        out_size = len(self.W)
        in_size  = len(self._inp)
        if self.activation == 'relu':
            delta = [grad_out[i] * (1.0 if self._pre[i] > 0 else 0.0)
                     for i in range(out_size)]
        else:
            delta = list(grad_out)
        grad_in = [
            sum(self.W[i][j] * delta[i] for i in range(out_size))
            for j in range(in_size)
        ]
        for i in range(out_size):
            self.b[i] -= lr * delta[i]
            for j in range(in_size):
                self.W[i][j] -= lr * delta[i] * self._inp[j]
        return grad_in

    def param_count(self):
        return len(self.W) * len(self.W[0]) + len(self.b)


# ── CNN Model (layer chain) ───────────────────────────────────────────────────

class _CNNModel:
    """Chains conv blocks and dense layers; handles flatten at the boundary."""

    def __init__(self, n_features: int, n_filters: int, kernel_size: int,
                 n_conv_layers: int, dense_hidden_sizes: list, pool_size: int):
        self.layers = []
        self.n_conv_layers = n_conv_layers

        # Conv blocks
        in_ch   = 1
        seq_len = n_features
        for _ in range(n_conv_layers):
            eff_k   = min(kernel_size, seq_len)
            out_len = max(1, seq_len - eff_k + 1)
            pooled  = max(1, (out_len + pool_size - 1) // pool_size)
            self.layers.append(_Conv1DLayer(n_filters, kernel_size, in_ch))
            self.layers.append(_ReLULayer())
            self.layers.append(_MaxPool1DLayer(pool_size))
            in_ch   = n_filters
            seq_len = pooled

        # Size after flatten
        flat_size = in_ch * seq_len if n_conv_layers > 0 else n_features

        # Dense blocks
        prev = flat_size
        for h in dense_hidden_sizes:
            self.layers.append(_DenseLayer(prev, h, activation='relu'))
            prev = h
        self.layers.append(_DenseLayer(prev, 1, activation='linear'))

        self._pool_shape = None   # set during forward; used in backward

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _flatten(x):
        out = []
        for row in x:
            out.extend(row)
        return out

    @staticmethod
    def _unflatten(g, shape):
        nf, L = shape
        return [g[f * L:(f + 1) * L] for f in range(nf)]

    # ── forward pass ─────────────────────────────────────────────────────

    def forward(self, x: list) -> float:
        """x: flat feature vector.  Returns scalar prediction."""
        h          = [list(x)]    # [1][n_features] — single-channel 1D input
        hit_dense  = False
        pool_shape = None

        for layer in self.layers:
            if isinstance(layer, _MaxPool1DLayer):
                h          = layer.forward(h)
                pool_shape = (len(h), len(h[0]) if h else 0)
            elif isinstance(layer, (_Conv1DLayer, _ReLULayer)):
                h = layer.forward(h)
            else:  # _DenseLayer
                if not hit_dense:
                    self._pool_shape = pool_shape
                    h = self._flatten(h)
                    hit_dense = True
                h = layer.forward(h)

        return h[0]

    # ── backward pass ────────────────────────────────────────────────────

    def backward(self, grad_scalar: float, lr: float):
        """Backpropagate scalar MSE gradient through all layers."""
        grad        = [grad_scalar]
        unflattened = False
        has_conv    = self.n_conv_layers > 0

        for layer in reversed(self.layers):
            if isinstance(layer, _DenseLayer):
                grad = layer.backward(grad, lr)
                # Unflatten once we pass back through the flatten boundary
                if has_conv and not unflattened and self._pool_shape:
                    nf, L = self._pool_shape
                    if nf > 0 and L > 0 and nf * L == len(grad):
                        grad        = self._unflatten(grad, (nf, L))
                        unflattened = True
            elif isinstance(layer, _MaxPool1DLayer):
                grad = layer.backward(grad)
            elif isinstance(layer, _ReLULayer):
                grad = layer.backward(grad)
            elif isinstance(layer, _Conv1DLayer):
                grad = layer.backward(grad, lr)

    # ── utilities ────────────────────────────────────────────────────────

    def count_params(self) -> int:
        return sum(
            l.param_count()
            for l in self.layers
            if isinstance(l, (_Conv1DLayer, _DenseLayer))
        )

    def save(self, path: str):
        import json
        state = []
        for layer in self.layers:
            if isinstance(layer, _Conv1DLayer):
                state.append({'type': 'conv',
                               'kernels': layer.kernels, 'biases': layer.biases})
            elif isinstance(layer, _DenseLayer):
                state.append({'type': 'dense', 'activation': layer.activation,
                               'W': layer.W, 'b': layer.b})
            elif isinstance(layer, _ReLULayer):
                state.append({'type': 'relu'})
            elif isinstance(layer, _MaxPool1DLayer):
                state.append({'type': 'pool', 'pool_size': layer.pool_size})
        with open(path, 'w') as fh:
            json.dump(state, fh)

    def load(self, path: str):
        import json
        with open(path) as fh:
            state = json.load(fh)
        si = 0
        for layer in self.layers:
            if si >= len(state):
                break
            if isinstance(layer, _Conv1DLayer) and state[si]['type'] == 'conv':
                layer.kernels = state[si]['kernels']
                layer.biases  = state[si]['biases']
                si += 1
            elif isinstance(layer, _DenseLayer) and state[si]['type'] == 'dense':
                layer.W = state[si]['W']
                layer.b = state[si]['b']
                si += 1
            elif isinstance(layer, _ReLULayer) and state[si]['type'] == 'relu':
                si += 1
            elif isinstance(layer, _MaxPool1DLayer) and state[si]['type'] == 'pool':
                si += 1


# ── Public CNN Trainer ────────────────────────────────────────────────────────

# CNN writes its own CSV; column names match SegmentHandler's error-epoch.csv
# where possible so both files can be compared column-by-column.
CSV_PATH  = "cnn_error_epoch.csv"
SAVE_PATH = "cnn_best.json"

# Column order kept identical to SegmentHandler so both models append cleanly
CSV_FIELDNAMES = [
    # Identity
    "timestamp", "model_type", "model_id", "target",
    # Data split
    "n_features", "n_train", "n_test", "epoch_count",
    # SegmentHandler-specific (blank for CNN)
    "max_x", "dimensions", "connection_pct", "density",
    # CNN-specific (blank for SegmentHandler)
    "n_filters", "kernel_size", "n_conv_layers",
    "dense_hidden_sizes", "n_dense_layers", "pool_size",
    "total_hidden_layers",   # n_conv_layers + n_dense_layers
    "total_nodes",           # input nodes + all hidden dense nodes + output node
    "total_params", "learning_rate", "lr_decay",
    # Best-epoch
    "best_epoch", "best_test_error_pct", "related_train_error_pct",
    # Regression metrics
    "mae", "rmse", "r2", "mape_pct",
    # Threshold accuracy
    "acc_5_pct", "acc_10_pct", "acc_20_pct",
    # Direction-based classification proxies
    "direction_acc_pct", "precision", "recall", "f1",
    "tp", "fp", "fn", "tn",
]


class CNN:
    """1-D CNN trainer — mirrors the SegmentHandler training interface."""

    def __init__(
        self,
        target:             str,
        logger              = None,
        n_filters:          int   = 8,
        kernel_size:        int   = 3,
        n_conv_layers:      int   = 1,
        dense_hidden_sizes: list  = None,
        pool_size:          int   = 2,
        learning_rate:      float = 0.01,
        lr_decay:           float = 0.95,
        model_id:           int   = 0,
    ):
        self.target             = target
        self.logger             = logger
        self.n_filters          = n_filters
        self.kernel_size        = kernel_size
        self.n_conv_layers      = n_conv_layers
        self.dense_hidden_sizes = dense_hidden_sizes if dense_hidden_sizes is not None else [64, 32]
        self.pool_size          = pool_size
        self.lr                 = learning_rate
        self.lr_decay           = lr_decay
        self.model_id           = model_id
        self.model              = None
        self.feature_cols       = None

    # ── display helpers ──────────────────────────────────────────────────

    def _log(self, msg: str, level: int = 4):
        """level: 4=INFO  3=WARNING  2=ERROR  1=DEBUG"""
        full = f"[CNN]: {msg}"
        if self.logger is not None:
            self.logger.log(full, level, True)
        else:
            print(full)

    def _rule(self, title: str):
        """Print a Rich section separator, or a plain divider if no console."""
        console = getattr(self.logger, 'console', None)
        if console is not None:
            from rich.rule import Rule
            console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
        else:
            print(f"\n{'─' * 10} {title} {'─' * 10}")

    # ── sample helpers ───────────────────────────────────────────────────

    def _xy(self, sample: dict):
        x = [float(sample.get(c, 0)) for c in self.feature_cols]
        y = float(sample.get(self.target, 0))
        return x, y

    def _test_avg_err(self, test_list: list) -> float:
        errors = []
        for s in test_list:
            x, y = self._xy(s)
            if y == 0:
                continue
            pred = self.model.forward(x)
            errors.append(abs(pred - y) / abs(y) * 100.0)
        return sum(errors) / len(errors) if errors else float('nan')

    # ── main training entry ──────────────────────────────────────────────

    def train(self, dataset, epoch_count: int = 20):
        """Train the CNN on dataset (DataFrame or CSV path)."""

        has_progress = hasattr(self.logger, 'make_progress')

        # ── Step 1: Preprocess ────────────────────────────────────────
        self._rule("Step 1 — Preprocessing")
        self._log("Loading and preprocessing dataset via PreProcessingNode…")
        preprocessor = PreProcesingNode(Logger=self.logger, logger_classification=4)
        if isinstance(dataset, str):
            self._log(f"Reading CSV: {dataset}")
            dataset = pd.read_csv(dataset)
        processed_df = preprocessor.process_dataset(dataset.copy()).fillna(0)

        all_cols = list(processed_df.columns)
        if self.target not in all_cols:
            raise ValueError(f"Target '{self.target}' not found after preprocessing.")

        self.feature_cols = [c for c in all_cols if c != self.target]
        n_features        = len(self.feature_cols)
        self._log(f"Preprocessing complete — {n_features} features, target='{self.target}'")

        # ── Step 2: Split ─────────────────────────────────────────────
        self._rule("Step 2 — Train / Test Split")
        records = processed_df.to_dict(orient='records')
        split   = max(1, int(len(records) * 0.8))
        train_  = records[:split]
        test_   = records[split:]
        self._log(
            f"{len(records)} total samples  →  "
            f"train={len(train_)} (80%)   test={len(test_)} (20%)"
        )

        # ── Step 3: Build model ───────────────────────────────────────
        self._rule("Step 3 — Building Model")
        self.model   = _CNNModel(
            n_features, self.n_filters, self.kernel_size,
            self.n_conv_layers, self.dense_hidden_sizes, self.pool_size,
        )
        total_params = self.model.count_params()
        self._log(f"Input nodes   : {n_features}")
        self._log(
            f"Conv block(s) : {self.n_conv_layers}× "
            f"Conv1D(filters={self.n_filters}, kernel={self.kernel_size}) "
            f"→ ReLU → MaxPool(pool={self.pool_size})"
        )
        self._log(f"Dense layers  : {self.dense_hidden_sizes}  ({len(self.dense_hidden_sizes)} hidden)")
        self._log(f"Output nodes  : 1 (linear regression)")
        self._log(
            f"Total layers  : {self.n_conv_layers + len(self.dense_hidden_sizes) + 1}  "
            f"hidden={self.n_conv_layers + len(self.dense_hidden_sizes)}  "
            f"total params={total_params:,}"
        )

        # ── Step 4: Epoch training loop ───────────────────────────────
        self._rule("Step 4 — Training")
        lr            = self.lr
        best_epoch    = None
        best_test_err = float('inf')
        epoch_history = []

        def _run_epochs(progress=None, epoch_task=None):
            nonlocal lr, best_epoch, best_test_err

            for epoch in range(epoch_count):
                random.shuffle(train_)
                train_errs = []

                # Inner progress bar — samples within this epoch (transient)
                if has_progress:
                    with self.logger.make_progress(transient=True) as inner:
                        sample_task = inner.add_task(
                            f"Epoch {epoch + 1}/{epoch_count} — training samples",
                            total=len(train_),
                        )
                        for sample in train_:
                            x, y = self._xy(sample)
                            pred  = self.model.forward(x)
                            self.model.backward(2.0 * (pred - y), lr)
                            if y != 0:
                                train_errs.append(abs(pred - y) / abs(y) * 100.0)
                            inner.update(sample_task, advance=1)
                else:
                    for sample in train_:
                        x, y = self._xy(sample)
                        pred  = self.model.forward(x)
                        self.model.backward(2.0 * (pred - y), lr)
                        if y != 0:
                            train_errs.append(abs(pred - y) / abs(y) * 100.0)

                train_avg = (sum(train_errs) / len(train_errs)
                             if train_errs else float('nan'))

                # Evaluate on test set
                if has_progress:
                    with self.logger.make_progress(transient=True) as inner:
                        eval_task = inner.add_task(
                            f"Epoch {epoch + 1}/{epoch_count} — evaluating test set",
                            total=len(test_),
                        )
                        errors = []
                        for s in test_:
                            x, y = self._xy(s)
                            if y != 0:
                                errors.append(
                                    abs(self.model.forward(x) - y) / abs(y) * 100.0
                                )
                            inner.update(eval_task, advance=1)
                        test_avg = sum(errors) / len(errors) if errors else float('nan')
                else:
                    test_avg = self._test_avg_err(test_)

                is_best = not math.isnan(test_avg) and test_avg < best_test_err
                epoch_history.append({
                    'epoch': epoch + 1,
                    'train_err': train_avg,
                    'test_err':  test_avg,
                })

                status = "★ NEW BEST" if is_best else ""
                self._log(
                    f"Epoch {epoch + 1:>3}/{epoch_count}  |  "
                    f"train={train_avg:6.2f}%   test={test_avg:6.2f}%   "
                    f"lr={lr:.5f}   best={best_test_err:.2f}%  {status}"
                )

                if is_best:
                    best_test_err = test_avg
                    best_epoch    = epoch + 1
                    self.model.save(SAVE_PATH)
                    self._log(
                        f"  → Saved new best model (epoch {best_epoch}, "
                        f"test={best_test_err:.2f}%) → {SAVE_PATH}"
                    )

                lr *= self.lr_decay

                if progress and epoch_task is not None:
                    progress.update(
                        epoch_task,
                        advance=1,
                        description=(
                            f"Training epochs  "
                            f"[train={train_avg:.1f}%  test={test_avg:.1f}%  "
                            f"best={best_test_err:.1f}%]"
                        ),
                    )

        # Outer epoch progress bar (persistent across all epochs)
        if has_progress:
            with self.logger.make_progress() as progress:
                epoch_task = progress.add_task("Training epochs", total=epoch_count)
                _run_epochs(progress, epoch_task)
        else:
            _run_epochs()

        # ── Step 5: Restore best weights ──────────────────────────────
        self._rule("Step 5 — Restoring Best Model")
        if best_epoch and os.path.exists(SAVE_PATH):
            self.model.load(SAVE_PATH)
            self._log(
                f"Restored weights from epoch {best_epoch}  "
                f"(best test error: {best_test_err:.2f}%)"
            )
        else:
            self._log("No saved model found — using final epoch weights.", level=3)

        # ── Step 6: Final evaluation + CSV + plots ────────────────────
        self._run_final_eval(
            best_epoch    = best_epoch,
            best_test_err = best_test_err,
            test_list     = test_,
            epoch_history = epoch_history,
            n_train       = len(train_),
            n_features    = n_features,
            epoch_count   = epoch_count,
            total_params  = total_params,
        )
        self._rule("Step 7 — Generating Plots")
        self._plot(epoch_history, best_epoch)

    # ── final evaluation ─────────────────────────────────────────────────

    def _run_final_eval(
        self, best_epoch, best_test_err, test_list, epoch_history,
        n_train, n_features, epoch_count, total_params,
    ):
        self._rule("Step 6 — Final Evaluation on Test Set")
        self._log(
            f"Running inference on {len(test_list)} test samples "
            f"using best model (epoch {best_epoch})…"
        )

        has_progress = hasattr(self.logger, 'make_progress')
        predictions, actuals = [], []

        if has_progress:
            with self.logger.make_progress(transient=True) as progress:
                task = progress.add_task(
                    "Final test inference", total=len(test_list)
                )
                for s in test_list:
                    x, y = self._xy(s)
                    if y != 0:
                        predictions.append(self.model.forward(x))
                        actuals.append(y)
                    progress.update(task, advance=1)
        else:
            for s in test_list:
                x, y = self._xy(s)
                if y == 0:
                    continue
                predictions.append(self.model.forward(x))
                actuals.append(y)

        n = len(predictions)
        if n == 0:
            self._log("No test predictions produced — skipping final eval.", level=3)
            return

        self._log(f"Inference complete — {n} valid predictions collected.")

        # ── Regression metrics ────────────────────────────────────────
        mae    = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n
        rmse   = math.sqrt(sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n)
        mean_a = sum(actuals) / n
        ss_tot = sum((a - mean_a) ** 2 for a in actuals)
        ss_res = sum((p - a) ** 2 for p, a in zip(predictions, actuals))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        mape   = (
            sum(abs(p - a) / abs(a) * 100.0 for p, a in zip(predictions, actuals)) / n
        )

        def _within(pct):
            return (
                sum(1 for p, a in zip(predictions, actuals)
                    if a != 0 and abs(p - a) / abs(a) <= pct) / n * 100.0
            )

        acc_5  = _within(0.05)
        acc_10 = _within(0.10)
        acc_20 = _within(0.20)

        # ── Direction-based precision / recall / F1 (median split) ────
        median_a   = sorted(actuals)[n // 2]
        actual_pos = [1 if a >= median_a else 0 for a in actuals]
        pred_pos   = [1 if p >= median_a else 0 for p in predictions]
        tp = sum(1 for ap, pp in zip(actual_pos, pred_pos) if ap == 1 and pp == 1)
        fp = sum(1 for ap, pp in zip(actual_pos, pred_pos) if ap == 0 and pp == 1)
        fn = sum(1 for ap, pp in zip(actual_pos, pred_pos) if ap == 1 and pp == 0)
        tn = sum(1 for ap, pp in zip(actual_pos, pred_pos) if ap == 0 and pp == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = (
            2 * precision * recall / (precision + recall)
            if not (math.isnan(precision) or math.isnan(recall))
            and (precision + recall) > 0
            else float('nan')
        )
        dir_acc = (tp + tn) / n * 100.0

        # ── Related train error at best epoch ─────────────────────────
        related_train = float('nan')
        for e in epoch_history:
            if e['epoch'] == best_epoch:
                related_train = e['train_err']
                break

        # ── Console display ───────────────────────────────────────────
        console = getattr(self.logger, 'console', None) if self.logger else None
        if console is not None:
            from rich.table import Table
            from rich import box
            from rich.rule import Rule
            console.print(Rule("[bold green]CNN — Final Test Evaluation[/bold green]"))
            t = Table(
                title=f"CNN Test Metrics  (n={n}, best epoch={best_epoch})",
                box=box.ROUNDED, border_style="green", show_lines=True,
            )
            t.add_column("Metric", style="bold white")
            t.add_column("Value",  style="bright_green", justify="right")
            t.add_row("MAE",                  f"{mae:.4f}")
            t.add_row("RMSE",                 f"{rmse:.4f}")
            t.add_row("R²",                   f"{r2:.4f}" if not math.isnan(r2) else "—")
            t.add_row("MAPE",                 f"{mape:.2f}%")
            t.add_section()
            t.add_row("Within  5% accuracy",  f"{acc_5:.1f}%")
            t.add_row("Within 10% accuracy",  f"{acc_10:.1f}%")
            t.add_row("Within 20% accuracy",  f"{acc_20:.1f}%")
            t.add_section()
            t.add_row("Direction accuracy",   f"{dir_acc:.1f}%")
            t.add_row("Precision (median)",   f"{precision:.4f}" if not math.isnan(precision) else "—")
            t.add_row("Recall    (median)",   f"{recall:.4f}"    if not math.isnan(recall)    else "—")
            t.add_row("F1        (median)",   f"{f1:.4f}"        if not math.isnan(f1)        else "—")
            console.print(t)
        else:
            self._log(
                f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²="
                f"{'—' if math.isnan(r2) else f'{r2:.4f}'}  "
                f"MAPE={mape:.2f}%  Acc@10%={acc_10:.1f}%  "
                f"F1={'—' if math.isnan(f1) else f'{f1:.4f}'}"
            )

        # ── Append to cnn_error_epoch.csv ─────────────────────────────
        self._log(f"Writing metrics to {CSV_PATH}…")
        def _fmt(v, decimals=6):
            return f"{v:.{decimals}f}" if not math.isnan(v) else ''

        row = {
            "timestamp":               _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type":              "CNN",
            "model_id":                self.model_id,
            "target":                  self.target,
            "n_features":              n_features,
            "n_train":                 n_train,
            "n_test":                  n,
            "epoch_count":             epoch_count,
            # SegmentHandler-specific (blank)
            "max_x":                   "",
            "dimensions":              "",
            "connection_pct":          "",
            "density":                 "",
            # CNN-specific
            "n_filters":               self.n_filters,
            "kernel_size":             self.kernel_size,
            "n_conv_layers":           self.n_conv_layers,
            "dense_hidden_sizes":      str(self.dense_hidden_sizes),
            "n_dense_layers":          len(self.dense_hidden_sizes),
            "pool_size":               self.pool_size,
            "total_hidden_layers":     self.n_conv_layers + len(self.dense_hidden_sizes),
            "total_nodes":             (n_features                        # input layer
                                        + sum(self.dense_hidden_sizes)     # hidden dense layers
                                        + 1),                              # output node
            "total_params":            total_params,
            "learning_rate":           self.lr,
            "lr_decay":                self.lr_decay,
            # Best-epoch
            "best_epoch":              best_epoch if best_epoch is not None else '',
            "best_test_error_pct":     _fmt(best_test_err, 4),
            "related_train_error_pct": _fmt(related_train, 4),
            # Regression
            "mae":                     _fmt(mae),
            "rmse":                    _fmt(rmse),
            "r2":                      _fmt(r2),
            "mape_pct":                _fmt(mape, 4),
            # Accuracy
            "acc_5_pct":               _fmt(acc_5, 2),
            "acc_10_pct":              _fmt(acc_10, 2),
            "acc_20_pct":              _fmt(acc_20, 2),
            # Classification proxies
            "direction_acc_pct":       _fmt(dir_acc, 2),
            "precision":               _fmt(precision),
            "recall":                  _fmt(recall),
            "f1":                      _fmt(f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._log(f"Run metrics appended to {CSV_PATH}.")

    # ── comparison interface ─────────────────────────────────────────────

    def run(self, train_records: list, test_records: list,
            feature_cols: list, target: str, epoch_count: int = 20) -> dict:
        """
        Standard comparison interface — accepts pre-split records from
        ComparisonManager, bypasses internal preprocessing and splitting.
        Returns a metrics dict compatible with ComparisonManager.
        """
        import time
        from comparisons.shared_metrics import compute_metrics

        has_progress = hasattr(self.logger, 'make_progress')

        self.target       = target
        self.feature_cols = feature_cols
        n_features        = len(feature_cols)

        # ── Build model ───────────────────────────────────────────────
        self._rule("CNN — Building Model")
        self.model = _CNNModel(
            n_features, self.n_filters, self.kernel_size,
            self.n_conv_layers, self.dense_hidden_sizes, self.pool_size,
        )
        total_params = self.model.count_params()
        self._log(
            f"Architecture: {self.n_conv_layers}×Conv1D(f={self.n_filters},"
            f"k={self.kernel_size}) + dense={self.dense_hidden_sizes}  "
            f"params={total_params:,}"
        )

        lr            = self.lr
        best_epoch    = None
        best_test_err = float('inf')
        epoch_history = []

        # ── Epoch loop ────────────────────────────────────────────────
        self._rule("CNN — Training")

        def _run_epochs(outer_prog=None, outer_task=None):
            nonlocal lr, best_epoch, best_test_err

            for epoch in range(epoch_count):
                random.shuffle(train_records)
                train_errs = []

                # Inner: per-sample training bar (transient)
                if has_progress:
                    with self.logger.make_progress(transient=True) as inner:
                        stask = inner.add_task(
                            f"Epoch {epoch + 1}/{epoch_count} — training",
                            total=len(train_records),
                        )
                        for sample in train_records:
                            x, y = self._xy(sample)
                            pred  = self.model.forward(x)
                            self.model.backward(2.0 * (pred - y), lr)
                            if y != 0:
                                train_errs.append(abs(pred - y) / abs(y) * 100.0)
                            inner.update(stask, advance=1)
                else:
                    for sample in train_records:
                        x, y = self._xy(sample)
                        pred  = self.model.forward(x)
                        self.model.backward(2.0 * (pred - y), lr)
                        if y != 0:
                            train_errs.append(abs(pred - y) / abs(y) * 100.0)

                train_avg = (sum(train_errs) / len(train_errs)
                             if train_errs else float('nan'))

                # Inner: test-set eval bar (transient)
                if has_progress:
                    with self.logger.make_progress(transient=True) as inner:
                        etask = inner.add_task(
                            f"Epoch {epoch + 1}/{epoch_count} — evaluating",
                            total=len(test_records),
                        )
                        errors = []
                        for s in test_records:
                            x, y = self._xy(s)
                            if y != 0:
                                errors.append(
                                    abs(self.model.forward(x) - y) / abs(y) * 100.0
                                )
                            inner.update(etask, advance=1)
                        test_avg = sum(errors) / len(errors) if errors else float('nan')
                else:
                    test_avg = self._test_avg_err(test_records)

                is_best = not math.isnan(test_avg) and test_avg < best_test_err
                epoch_history.append({
                    'epoch': epoch + 1,
                    'train_err': train_avg,
                    'test_err':  test_avg,
                })
                status = "★ BEST" if is_best else ""
                self._log(
                    f"Epoch {epoch + 1:>3}/{epoch_count}  |  "
                    f"train={train_avg:6.2f}%   test={test_avg:6.2f}%   "
                    f"lr={lr:.5f}   best={best_test_err:.2f}%  {status}"
                )

                if is_best:
                    best_test_err = test_avg
                    best_epoch    = epoch + 1
                    self.model.save(SAVE_PATH)

                lr *= self.lr_decay

                if outer_prog and outer_task is not None:
                    outer_prog.update(
                        outer_task, advance=1,
                        description=(
                            f"CNN epochs  "
                            f"[train={train_avg:.1f}%  test={test_avg:.1f}%  "
                            f"best={best_test_err:.1f}%]"
                        ),
                    )

        t0 = time.time()
        if has_progress:
            with self.logger.make_progress() as outer:
                epoch_task = outer.add_task("CNN epochs", total=epoch_count)
                _run_epochs(outer, epoch_task)
        else:
            _run_epochs()
        train_time = time.time() - t0

        # ── Restore best weights ──────────────────────────────────────
        if best_epoch and os.path.exists(SAVE_PATH):
            self.model.load(SAVE_PATH)
            self._log(f"Restored best weights from epoch {best_epoch} "
                      f"(test err={best_test_err:.2f}%)")

        # ── Inference timing pass ─────────────────────────────────────
        self._rule("CNN — Final Inference")
        self._log(f"Timing inference on {len(test_records)} test samples…")
        t1 = time.time()
        predictions, actuals = [], []
        if has_progress:
            with self.logger.make_progress(transient=True) as inner:
                itask = inner.add_task(
                    "CNN inference", total=len(test_records)
                )
                for s in test_records:
                    x, y = self._xy(s)
                    if y != 0:
                        predictions.append(self.model.forward(x))
                        actuals.append(y)
                    inner.update(itask, advance=1)
        else:
            for s in test_records:
                x, y = self._xy(s)
                if y != 0:
                    predictions.append(self.model.forward(x))
                    actuals.append(y)
        inference_time = time.time() - t1
        self._log(f"Inference complete — {len(predictions)} valid predictions.")

        # Related train error at best epoch
        related_train = float('nan')
        for e in epoch_history:
            if e['epoch'] == best_epoch:
                related_train = e['train_err']
                break

        metrics = compute_metrics(predictions, actuals)
        metrics.update({
            'model_name':   'CNN',
            'model_config': (
                f"filters={self.n_filters},kernel={self.kernel_size},"
                f"conv_layers={self.n_conv_layers},"
                f"dense={self.dense_hidden_sizes},pool={self.pool_size}"
            ),
            'n_train':                len(train_records),
            'n_test':                 len(actuals),
            'n_features':             n_features,
            'target':                 target,
            'epoch_count':            epoch_count,
            'train_time_sec':         round(train_time, 4),
            'inference_time_sec':     round(inference_time, 6),
            'best_epoch':             best_epoch or '',
            'best_test_error_pct':    best_test_err if not math.isnan(best_test_err) else '',
            'related_train_error_pct':related_train if not math.isnan(related_train) else '',
            'notes':                  'pure-python backprop',
        })
        return metrics

    # ── plots ────────────────────────────────────────────────────────────

    def _plot(self, epoch_history: list, best_epoch):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self._log("matplotlib not available — skipping plots.")
            return

        epochs     = [e['epoch']     for e in epoch_history]
        train_errs = [e['train_err'] for e in epoch_history]
        test_errs  = [e['test_err']  for e in epoch_history]

        # ── Train vs Test graph (mirrors SegmentHandler layout) ───────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_errs, marker='s', linewidth=2,
                color='steelblue', label='Train Avg Error %')
        ax.plot(epochs, test_errs,  marker='^', linewidth=2,
                color='tomato',    label='Test Avg Error %')

        valid_test = [(i, v) for i, v in enumerate(test_errs)
                      if not math.isnan(v)]
        if valid_test:
            bi, bv = min(valid_test, key=lambda t: t[1])
            ax.axvline(x=epochs[bi], color='gold', linestyle='--', linewidth=1.5,
                       label=f"Best Epoch ({epochs[bi]})")
            ax.annotate(f"{bv:.1f}%", xy=(epochs[bi], bv),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=9, color='tomato')

        ax.set_title("CNN — Train vs Test Average Error % per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Error %")
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig("cnn_epoch_train_test_error.png")
        plt.close(fig)
        self._log("Saved cnn_epoch_train_test_error.png.")

        # ── Per-epoch error graph (mirrors epoch_error.png) ───────────
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(epochs, test_errs, marker='o', linewidth=2,
                 color='tomato',    label='Test Avg Error %')
        ax2.plot(epochs, train_errs, marker='s', linewidth=1.5,
                 color='steelblue', linestyle='--', alpha=0.6,
                 label='Train Avg Error %')
        if best_epoch is not None:
            ax2.axvline(x=best_epoch, color='gold', linestyle='--',
                        linewidth=1.5, label=f"Best Epoch ({best_epoch})")
        ax2.set_title("CNN — Epoch vs Error %")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Error %")
        ax2.set_xticks(epochs)
        ax2.legend()
        ax2.grid(True, alpha=0.4)
        fig2.tight_layout()
        fig2.savefig("cnn_epoch_error.png")
        plt.close(fig2)
        self._log("Saved cnn_epoch_error.png.")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from Components.RichConsole import RichLogger

    dataset_path = input("Path to CSV dataset: ").strip()
    target_col   = input("Target column name:  ").strip()

    logger = RichLogger(f"cnn_{int(__import__('time').time())}.log", log_level=4)

    cnn = CNN(
        target             = target_col,
        logger             = logger,
        n_filters          = 8,
        kernel_size        = 3,
        n_conv_layers      = 1,
        dense_hidden_sizes = [64, 32],
        pool_size          = 2,
        learning_rate      = 0.01,
        lr_decay           = 0.95,
        model_id           = 0,
    )
    cnn.train(dataset_path, epoch_count=20)
