"""
comparisons/shared_metrics.py
------------------------------
Single source of truth for all metric computation used by every comparison
model.  Ensures identical calculations across LinearRegression, KNN,
RandomForest, XGBoost, MLP, CNN, and SegmentHandler so results are directly
comparable.
"""

import math
import csv
import os
import datetime as _dt


# ── Unified CSV schema for comparison_results.csv ────────────────────────────

COMPARISON_CSV = "comparison_results.csv"

CSV_FIELDNAMES = [
    # Identity
    "timestamp", "run_id", "model_name", "model_config",
    # Data
    "target", "n_features", "n_train", "n_test", "epoch_count",
    # Timing
    "train_time_sec", "inference_time_sec",
    # Regression
    "mae", "rmse", "r2", "mape_pct",
    # Threshold accuracy
    "acc_5_pct", "acc_10_pct", "acc_20_pct",
    # Classification proxies
    "direction_acc_pct", "precision", "recall", "f1",
    "tp", "fp", "fn", "tn",
    # Best epoch (where applicable)
    "best_epoch", "best_test_error_pct", "related_train_error_pct",
    # Notes
    "notes",
]


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(predictions: list, actuals: list) -> dict:
    """
    Compute a full suite of regression and classification-proxy metrics.

    Parameters
    ----------
    predictions : list of float
    actuals     : list of float  (ground-truth target values, non-zero)

    Returns
    -------
    dict with keys: mae, rmse, r2, mape_pct,
                    acc_5_pct, acc_10_pct, acc_20_pct,
                    direction_acc_pct, precision, recall, f1,
                    tp, fp, fn, tn
    """
    n = len(predictions)
    if n == 0:
        return {k: float('nan') for k in [
            'mae', 'rmse', 'r2', 'mape_pct',
            'acc_5_pct', 'acc_10_pct', 'acc_20_pct',
            'direction_acc_pct', 'precision', 'recall', 'f1',
            'tp', 'fp', 'fn', 'tn',
        ]}

    # ── Regression metrics ────────────────────────────────────────────
    mae    = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n
    rmse   = math.sqrt(sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n)
    mean_a = sum(actuals) / n
    ss_tot = sum((a - mean_a) ** 2 for a in actuals)
    ss_res = sum((p - a) ** 2 for p, a in zip(predictions, actuals))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    mape   = (sum(abs(p - a) / abs(a) * 100.0
               for p, a in zip(predictions, actuals) if a != 0) / n)

    # ── Within-threshold accuracy ─────────────────────────────────────
    def _within(pct):
        return (sum(1 for p, a in zip(predictions, actuals)
                    if a != 0 and abs(p - a) / abs(a) <= pct) / n * 100.0)

    acc_5  = _within(0.05)
    acc_10 = _within(0.10)
    acc_20 = _within(0.20)

    # ── Direction-based precision / recall / F1 (median split) ────────
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

    return {
        'mae':               mae,
        'rmse':              rmse,
        'r2':                r2,
        'mape_pct':          mape,
        'acc_5_pct':         acc_5,
        'acc_10_pct':        acc_10,
        'acc_20_pct':        acc_20,
        'direction_acc_pct': dir_acc,
        'precision':         precision,
        'recall':            recall,
        'f1':                f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def _fmt(v, decimals=4):
    """Format a float for CSV output; empty string for NaN."""
    if isinstance(v, float) and math.isnan(v):
        return ''
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v) if v is not None else ''


def append_comparison_row(metrics: dict, run_id: str):
    """Append one result row to comparison_results.csv."""
    row = {f: _fmt(metrics.get(f, '')) for f in CSV_FIELDNAMES}
    # Non-float fields pass through as-is
    for key in ('timestamp', 'run_id', 'model_name', 'model_config',
                 'target', 'n_features', 'n_train', 'n_test', 'epoch_count',
                 'best_epoch', 'notes'):
        row[key] = metrics.get(key, '')
    row['run_id'] = run_id

    write_header = not os.path.exists(COMPARISON_CSV)
    with open(COMPARISON_CSV, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
