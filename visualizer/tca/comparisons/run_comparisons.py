"""
comparisons/run_comparisons.py
--------------------------------
Single entry point — "the main button".

Run this file to:
  1. Preprocess the dataset once (shared across all models)
  2. Perform a single 80/20 train/test split (shared across all models)
  3. Train and evaluate every model on identical data
  4. Save comparison_results.csv
  5. Print a ranked comparison table

Usage:
    python comparisons/run_comparisons.py

Or with arguments (skips interactive prompts):
    python comparisons/run_comparisons.py \
        --dataset path/to/data.csv \
        --target  exam_score \
        --epochs  20 \
        --max_x   20
"""

import os
import sys
import argparse

# Allow imports from parent v10/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from Components.RichConsole import RichLogger
from comparisons.ComparisonManager import ComparisonManager


def main():
    parser = argparse.ArgumentParser(
        description="Run all comparison models on a single dataset."
    )
    parser.add_argument("--dataset", default=None, help="Path to CSV dataset")
    parser.add_argument("--target",  default=None, help="Target column name")
    parser.add_argument("--epochs",  type=int, default=None, help="Epoch count")
    parser.add_argument("--max_x",   type=int, nargs='+', default=None,
                        help="SegmentHandler max_x value(s), e.g. --max_x 5 10 15 20 25")
    # Model toggles
    parser.add_argument("--no-segment", action="store_true")
    parser.add_argument("--no-cnn",     action="store_true")
    parser.add_argument("--no-linear",  action="store_true")
    parser.add_argument("--no-knn",     action="store_true")
    parser.add_argument("--no-rf",      action="store_true")
    parser.add_argument("--no-xgb",     action="store_true")
    parser.add_argument("--no-mlp",     action="store_true")
    parser.add_argument("--no-system",  action="store_true",
                        help="Skip SystemHandler (full nexus) comparison")
    parser.add_argument("--system-max-x", type=int, nargs='+', default=None,
                        help="SystemHandler max_x value(s), e.g. --system-max-x 10 15 20")
    parser.add_argument("--system-training-mode", default="partitioned",
                        choices=["partitioned", "full"],
                        help="Training mode for SystemHandler (default: partitioned)")
    parser.add_argument("--system-dimensions", type=int, default=2,
                        help="Dimensions for SystemHandler segments (default: 2)")
    parser.add_argument("--system-agg-mode", nargs='+',
                        choices=["bma", "simple_mean", "relevance_weighted"],
                        default=None,
                        help="Aggregation mode(s) for SystemHandler inference. "
                             "Pass one or more of: bma simple_mean relevance_weighted. "
                             "Omit to run all three.")
    # Include historical runs from existing CSVs
    parser.add_argument("--merge-history", action="store_true",
                        help="Include historical rows from error-epoch.csv "
                             "and cnn_error_epoch.csv in the report")
    args = parser.parse_args()

    # ── Interactive prompts for missing args ──────────────────────────
    dataset = args.dataset or input("Path to CSV dataset: ").strip()
    target  = args.target  or input("Target column name:  ").strip()

    epochs_str = str(args.epochs) if args.epochs else input("Epoch count        [20]: ").strip()
    epochs     = int(epochs_str) if epochs_str else 20

    if args.max_x:
        max_x = args.max_x   # already a list from nargs='+'
    else:
        raw = input("SegmentHandler max_x value(s) [5 10 15 20 25]: ").strip().strip('[]')
        max_x = [int(v) for v in raw.replace(',', ' ').split()] if raw else [5, 10, 15, 20, 25]

    if args.system_max_x:
        system_max_x = args.system_max_x
    else:
        raw = input("SystemHandler max_x value(s)  [10]:           ").strip().strip('[]')
        system_max_x = [int(v) for v in raw.replace(',', ' ').split()] if raw else [10]

    if args.system_agg_mode:
        system_agg_modes = args.system_agg_mode
    else:
        _choices = "bma, simple_mean, relevance_weighted"
        raw = input(f"SystemHandler agg mode(s) [{_choices}] (blank=all): ").strip()
        system_agg_modes = [m.strip() for m in raw.split(',')] if raw else None

    # ── Logger ────────────────────────────────────────────────────────
    import time
    logger = RichLogger(f"comparison_{int(time.time())}.log", log_level=4)

    # ── Run ───────────────────────────────────────────────────────────
    manager = ComparisonManager(logger=logger)

    if args.merge_history:
        manager.merge_existing_csvs()

    manager.run_all(
        dataset        = dataset,
        target         = target,
        epoch_count    = epochs,
        segment_max_x  = max_x,
        run_segment    = not args.no_segment,
        run_cnn        = not args.no_cnn,
        run_linear     = not args.no_linear,
        run_knn        = not args.no_knn,
        run_rf         = not args.no_rf,
        run_xgb        = not args.no_xgb,
        run_mlp        = not args.no_mlp,
        run_system     = not args.no_system,
        system_max_x   = system_max_x,
        system_dimensions       = args.system_dimensions,
        system_training_mode    = args.system_training_mode,
        system_aggregation_modes= system_agg_modes,
    )


if __name__ == "__main__":
    main()
