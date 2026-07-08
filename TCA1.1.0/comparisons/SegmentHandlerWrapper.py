"""
comparisons/SegmentHandlerWrapper.py
--------------------------------------
Thin wrapper around SegmentHandler that exposes the same run() interface
as all other comparison models.  Zero modifications to SegmentHandler itself.

Strategy for fair comparison:
- Reconstructs a DataFrame from the pre-split records (train first, test
  second) in the exact order ComparisonManager produced them.
- SegmentHandler's 80/20 split on this DataFrame reproduces the same split.
- SegmentHandler runs its own PreProcessingNode internally (harmless re-run
  since all values are already numeric floats after ComparisonManager's pass).
- After train() completes, reads the last row of error-epoch.csv for metrics
  (SegmentHandler already computed them all).
- Times a fresh inference pass on the test set for inference_time_sec.
"""

import os
import sys
import csv
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from comparisons.shared_metrics import compute_metrics


class SegmentHandlerWrapper:
    """Wraps SegmentHandler with the standard comparison run() interface."""

    MODEL_NAME = "SegmentHandler"

    def __init__(
        self,
        max_x:                int   = 20,
        dimensions:           int   = 2,
        connection_percentage:float = 0.1,
        density:              float = 0.8,
    ):
        self.max_x                 = max_x
        self.dimensions            = dimensions
        self.connection_percentage = connection_percentage
        self.density               = density

    @property
    def config_str(self) -> str:
        return (f"max_x={self.max_x},"
                f"dim={self.dimensions},"
                f"conn={self.connection_percentage},"
                f"density={self.density}")

    def run(self, train_records: list, test_records: list,
            feature_cols: list, target: str,
            epoch_count: int = 20, logger=None) -> dict:
        """
        Reconstruct DataFrame → call SegmentHandler.train() → read metrics.
        Returns a dict compatible with ComparisonManager.
        """
        from SegmentHandler import SegmentHandler

        def _log(msg):
            full = f"[SegmentHandlerWrapper(max_x={self.max_x})]: {msg}"
            if logger:
                logger.log(full, 4, True)
            else:
                print(full)

        # ── Reconstruct DataFrame in split order ──────────────────────
        # train records first, test records second so SegmentHandler's
        # internal 80/20 slice reproduces the same split exactly.
        all_records = train_records + test_records
        df = pd.DataFrame(all_records)

        # ── Build and initialise segment ──────────────────────────────
        handler = SegmentHandler(
            maxX=self.max_x,
            target=target,
            logger=logger,
            connection_percentage=self.connection_percentage,
            density=self.density,
            dimensions=self.dimensions,
            classification=4,
            segment_id=0,
        )
        handler.initializeSegment()

        # ── Train (timed) ─────────────────────────────────────────────
        _log(f"Starting train() — max_x={self.max_x}, epochs={epoch_count}…")
        t0 = time.time()
        handler.train(df, epoch_count=epoch_count)
        train_time = time.time() - t0
        _log(f"train() finished in {train_time:.1f}s")

        # ── Read metrics from error-epoch.csv (last row) ──────────────
        seg_metrics = self._read_last_csv_row("error-epoch.csv")

        # ── Time a fresh inference pass on test records ───────────────
        _log(f"Timing inference on {len(test_records)} test samples…")
        t1 = time.time()
        inf_predictions, inf_actuals = [], []
        for s in test_records:
            target_val = s.get(target)
            if not target_val or target_val == 0:
                continue
            _, reviewer_data = handler._forward_segment(s)
            preds = [pred for _, _, pred in reviewer_data if pred is not None]
            if preds:
                avg = sum(preds) / len(preds)
                inf_predictions.append(avg)
                inf_actuals.append(float(target_val))
        inference_time = time.time() - t1
        _log(f"Inference complete in {inference_time:.4f}s")

        # ── Recompute metrics from fresh inference pass ───────────────
        # (gives consistent metric source across all models)
        recomputed = compute_metrics(inf_predictions, inf_actuals)

        result = {**recomputed}
        result.update({
            'model_name':    self.MODEL_NAME,
            'model_config':  self.config_str,
            'n_train':       len(train_records),
            'n_test':        len(inf_actuals),
            'n_features':    len(feature_cols),
            'target':        target,
            'epoch_count':   epoch_count,
            'train_time_sec':     round(train_time, 4),
            'inference_time_sec': round(inference_time, 6),
            # Pull best_epoch fields from SegmentHandler's own CSV if available
            'best_epoch':             seg_metrics.get('best_epoch', ''),
            'best_test_error_pct':    seg_metrics.get('best_test_error_pct', ''),
            'related_train_error_pct':seg_metrics.get('related_train_error_pct', ''),
            'notes': f'max_x={self.max_x},dim={self.dimensions}',
        })
        return result

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _read_last_csv_row(path: str) -> dict:
        """Return the last data row of a CSV as a dict, or {} if unavailable."""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, newline='') as fh:
                rows = list(csv.DictReader(fh))
            return rows[-1] if rows else {}
        except Exception:
            return {}
