"""
comparisons/SystemHandlerWrapper.py
--------------------------------------
Thin wrapper around the full SystemHandler (multi-segment nexus) that exposes
the same run() interface as all other comparison models.  Zero modifications
to SystemHandler itself.

Strategy for fair comparison:
- Training receives ONLY the train split (train_records) produced by
  ComparisonManager's shared 80/20 split, so the system never sees test data.
- SystemHandler's internal PreProcessingNode runs on already-numeric records
  (ComparisonManager preprocessed them first) — effectively a harmless no-op
  re-run that also warms up the preprocessor's fitted state for inference.
- After train() completes, inference is timed on the test split (test_records),
  calling system.runInfer() row-by-row exactly as the production code does.
- Metrics are computed by shared_metrics.compute_metrics, identical to all
  other wrappers.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from comparisons.shared_metrics import compute_metrics


class SystemHandlerWrapper:
    """Wraps SystemHandler (full nexus) with the standard comparison run() interface."""

    MODEL_NAME = "SystemHandler"

    def __init__(
        self,
        max_x:                  int   = 10,
        dimensions:             int   = 2,
        connection_percentage:  float = 0.1,
        density:                float = 0.8,
        training_mode:          str   = "partitioned",  # "partitioned" | "full"
        judge_iterations:       int   = 20,
        judge_min_clusters:     int | None = None,
        judge_max_clusters:     int | None = None,
        aggregation_mode:       str   = "bma",
        selection_percentage:   float = 0.5,
    ):
        self.max_x                 = max_x
        self.dimensions            = dimensions
        self.connection_percentage = connection_percentage
        self.density               = density
        self.training_mode         = training_mode
        self.judge_iterations      = judge_iterations
        self.judge_min_clusters    = judge_min_clusters
        self.judge_max_clusters    = judge_max_clusters
        self.aggregation_mode      = aggregation_mode
        self.selection_percentage  = selection_percentage

    @property
    def config_str(self) -> str:
        return (
            f"max_x={self.max_x},"
            f"dim={self.dimensions},"
            f"conn={self.connection_percentage},"
            f"density={self.density},"
            f"mode={self.training_mode},"
            f"agg={self.aggregation_mode}"
        )

    def run(
        self,
        train_records: list,
        test_records:  list,
        feature_cols:  list,
        target:        str,
        epoch_count:   int  = 20,
        logger=None,
    ) -> dict:
        """
        Build SystemHandler → initializeAllSegments → train (train split only)
        → time inference on test split → compute metrics.
        Returns a dict compatible with ComparisonManager.

        Notes
        -----
        logger must be non-None — SystemHandler requires a logger for internal
        display calls, matching the same constraint as SegmentHandlerWrapper.
        """
        from SystemHandler import SystemHandler

        def _log(msg):
            full = f"[SystemHandlerWrapper(max_x={self.max_x})]: {msg}"
            if logger:
                logger.log(full, 4, True)
            else:
                print(full)

        n_segs = 2 ** self.dimensions

        # ── Reconstruct training DataFrame from train split only ──────
        # Test records are withheld entirely — SystemHandler must not see them.
        train_df = pd.DataFrame(train_records)

        # ── Build and initialise system ───────────────────────────────
        _log(
            f"Building SystemHandler — max_x={self.max_x}, "
            f"dim={self.dimensions}, segments={n_segs}, "
            f"mode={self.training_mode}"
        )
        system = SystemHandler(
            maxX=self.max_x,
            target=target,
            logger=logger,
            connection_percentage=self.connection_percentage,
            density=self.density,
            dimensions=self.dimensions,
            classification=4,
        )
        system.initializeAllSegments(Loud=False)
        _log(f"Segments initialised: {[s.segment_id for s in system.segments]}")

        # ── Train (timed) ─────────────────────────────────────────────
        _log(f"Starting training — mode={self.training_mode}, epochs={epoch_count}…")
        t0 = time.perf_counter()
        if self.training_mode == "full":
            system.train_full(train_df, epoch_count=epoch_count, loud=False)
        else:
            system.train(
                train_df,
                epoch_count=epoch_count,
                loud=False,
                judge_iterations=self.judge_iterations,
                judge_min_clusters=self.judge_min_clusters,
                judge_max_clusters=self.judge_max_clusters,
            )
        train_time = time.perf_counter() - t0
        _log(f"Training finished in {train_time:.2f}s")

        # ── Inference pass on test split (timed) ──────────────────────
        _log(f"Timing inference on {len(test_records)} test samples…")
        t1 = time.perf_counter()
        predictions, actuals = [], []
        failed = 0
        for sample in test_records:
            target_val = sample.get(target)
            if target_val is None or target_val == 0:
                continue
            try:
                pred = system.runInfer(
                    sample.copy(),
                    loud=False,
                    aggregation_mode=self.aggregation_mode,
                    selection_percentage=self.selection_percentage,
                )
                if pred is not None:
                    predictions.append(float(pred))
                    actuals.append(float(target_val))
            except Exception:
                failed += 1
        inference_time = time.perf_counter() - t1

        _log(
            f"Inference complete in {inference_time:.4f}s — "
            f"{len(predictions)}/{len(test_records)} valid predictions"
            + (f", {failed} errors" if failed else "")
        )

        # ── Compute unified metrics ───────────────────────────────────
        metrics = compute_metrics(predictions, actuals)

        result = {**metrics}
        result.update({
            "model_name":         self.MODEL_NAME,
            "model_config":       self.config_str,
            "n_train":            len(train_records),
            "n_test":             len(actuals),
            "n_features":         len(feature_cols),
            "target":             target,
            "epoch_count":        epoch_count,
            "train_time_sec":     round(train_time, 4),
            "inference_time_sec": round(inference_time, 6),
            "notes": (
                f"max_x={self.max_x},"
                f"dim={self.dimensions},"
                f"segments={n_segs},"
                f"mode={self.training_mode}"
            ),
        })
        return result
