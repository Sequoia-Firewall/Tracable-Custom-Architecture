"""
RandomForest + XGBoost baselines on YearPredictionMSD, same 40,000-row
sample/split used throughout the x-scale and load-balancing tests, so
these numbers are directly comparable to TCA's own results:

  TCA auto mode (best, max_x=20):  R2=-0.038  MAE=8.01   train=15755s (4.4hr)
  TCA auto mode (balanced, max_x=20): R2=-0.235  MAE=9.85  train=10074s (2.8hr)

comparisons/RandomForestModel.py and XGBoostModel.py already have exactly
the interface needed (run(train_records, test_records, feature_cols,
target, logger)) -- fully dataset-agnostic, no adapter required.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import Components.RichConsole as RC
from comparisons.RandomForestModel import RandomForestModel
from comparisons.XGBoostModel import XGBoostModel

TARGET = "year"
SAMPLE_SIZE = 40000
TEST_SPLIT = 0.2
CHECKPOINT_PATH = "yearpred_baselines_report.json"


def main():
    t_start = time.time()
    print("=== Loading YearPredictionMSD sample (same seed as TCA tests) ===")
    cols = [TARGET] + [f"feat_{i}" for i in range(90)]
    full_df = pd.read_csv("../datasets/yearpredictionmsd/YearPredictionMSD.txt", header=None, names=cols)
    dataset = full_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    n_test = int(len(dataset) * TEST_SPLIT)
    train_records = dataset.iloc[:-n_test].to_dict(orient="records")
    test_records = dataset.iloc[-n_test:].to_dict(orient="records")
    feature_cols = [f"feat_{i}" for i in range(90)]
    print(f"total={len(dataset)}  train={len(train_records)}  test={len(test_records)}")

    logger = RC.RichLogger(filename=f"yearpred_baselines_{int(time.time())}.log", log_level=4, console_level=2)

    results = {}

    print("\n=== RandomForest ===")
    rf = RandomForestModel(n_estimators=100, random_state=42)
    rf_metrics = rf.run(train_records, test_records, feature_cols, TARGET, logger=logger)
    results["RandomForest"] = rf_metrics
    print(f"RandomForest: R2={rf_metrics['r2']:.4f}  MAE={rf_metrics['mae']:.3f}  "
          f"train_time={rf_metrics['train_time_sec']:.1f}s")

    print("\n=== XGBoost ===")
    xgb = XGBoostModel(n_estimators=300, random_state=42)
    xgb_metrics = xgb.run(train_records, test_records, feature_cols, TARGET, logger=logger)
    results["XGBoost"] = xgb_metrics
    print(f"XGBoost: R2={xgb_metrics['r2']:.4f}  MAE={xgb_metrics['mae']:.3f}  "
          f"train_time={xgb_metrics['train_time_sec']:.1f}s")

    results["tca_reference"] = {
        "auto_max_x20_unbalanced": {"r2": -0.0379, "mae": 8.009, "train_wall_time_sec": 15755.0},
        "auto_max_x20_balanced":   {"r2": -0.2351, "mae": 9.847, "train_wall_time_sec": 10074.0},
    }
    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Summary ===")
    print(f"RandomForest: R2={rf_metrics['r2']:.4f}  MAE={rf_metrics['mae']:.3f}  time={rf_metrics['train_time_sec']:.1f}s")
    print(f"XGBoost:      R2={xgb_metrics['r2']:.4f}  MAE={xgb_metrics['mae']:.3f}  time={xgb_metrics['train_time_sec']:.1f}s")
    print(f"TCA (auto, max_x=20, unbalanced): R2=-0.0379  MAE=8.009  time=15755.0s")
    print(f"TCA (auto, max_x=20, balanced):   R2=-0.2351  MAE=9.847  time=10074.0s")
    print(f"\nTotal elapsed: {results['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
