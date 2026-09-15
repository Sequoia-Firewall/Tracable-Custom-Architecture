"""Smoke test for train_parallel_balanced() before committing to the full
comparison runs -- small scale, checks: runs without error, shard plan
looks sane, merged segments are structurally valid, and accuracy is in
the same ballpark as train_parallel_partitioned() on identical data."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from SystemHandler import SystemHandler
import Components.RichConsole as RC
from comparisons.shared_metrics import compute_metrics

TARGET = "exam_score"


def evaluate(system, test_rows):
    preds, actuals = [], []
    for row in test_rows:
        actual = row.get(TARGET)
        if actual is None:
            continue
        sample = {k: v for k, v in row.items() if k != TARGET}
        result = system.runInfer(sample, loud=False, aggregation_mode="bma", selection_percentage=0.5)
        if result is None:
            continue
        preds.append(float(result["score"]))
        actuals.append(float(actual))
    m = compute_metrics(preds, actuals)
    m["n"] = len(preds)
    return m


def main():
    df = pd.read_csv("Exam_Score_Prediction.csv").drop(columns=["student_id"])
    df = df.sample(n=1500, random_state=42).reset_index(drop=True)
    n_test = int(len(df) * 0.2)
    train_df = df.iloc[:-n_test].reset_index(drop=True)
    test_rows = df.iloc[-n_test:].to_dict(orient="records")
    print(f"train={len(train_df)} test={len(test_rows)}")

    lr_cfg = {"mode": "manual-full", "max_lr_scale": 0.25, "decay": 0.85}
    grad_cfg = {"mode": "manual", "value": 1}
    delta_cfg = {"mode": "manual", "value": 10}
    pred_range = {"mode": "manual", "min_value": 0, "max_value": 100}

    print("\n=== train_parallel_balanced ===")
    logger = RC.RichLogger(filename=f"smoke_balanced_{int(time.time())}.log", log_level=0, console_level=5)
    system = SystemHandler(maxX=10, target=TARGET, logger=logger, connection_percentage=0.1,
                           density=0.8, dimensions=2, classification=4)
    system.initializeAllSegments(Loud=False)
    t0 = time.time()
    results = system.train_parallel_balanced(
        train_df, epoch_count=5, judge_iterations=5, judge_min_clusters=4, judge_max_clusters=8,
        loud=False, lr_scale_cfg=lr_cfg, prediction_range_cfg=pred_range,
        grad_clip_cfg=grad_cfg, delta_clip_cfg=delta_cfg, reconnect_pct=0.0, max_workers=4,
    )
    dt = time.time() - t0
    print(f"train_parallel_balanced: {dt:.1f}s")
    print("shard/merge results:", results)
    m = evaluate(system, test_rows)
    print("balanced test metrics:", m)

    print("\n=== train_parallel_partitioned (baseline, same data/config) ===")
    logger2 = RC.RichLogger(filename=f"smoke_partitioned_{int(time.time())}.log", log_level=0, console_level=5)
    system2 = SystemHandler(maxX=10, target=TARGET, logger=logger2, connection_percentage=0.1,
                            density=0.8, dimensions=2, classification=4)
    system2.initializeAllSegments(Loud=False)
    t0 = time.time()
    results2 = system2.train_parallel_partitioned(
        train_df, epoch_count=5, judge_iterations=5, judge_min_clusters=4, judge_max_clusters=8,
        loud=False, lr_scale_cfg=lr_cfg, prediction_range_cfg=pred_range,
        grad_clip_cfg=grad_cfg, delta_clip_cfg=delta_cfg, reconnect_pct=0.0, max_workers=4,
    )
    dt2 = time.time() - t0
    print(f"train_parallel_partitioned: {dt2:.1f}s")
    m2 = evaluate(system2, test_rows)
    print("partitioned (baseline) test metrics:", m2)

    print(f"\nSanity check: balanced R2={m['r2']:.4f} vs partitioned R2={m2['r2']:.4f}")


if __name__ == "__main__":
    main()
