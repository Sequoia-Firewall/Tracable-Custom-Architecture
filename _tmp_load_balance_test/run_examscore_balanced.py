"""
No-regression check: train_parallel_balanced() on the full 20,000-row
exam-score dataset, matching the existing best-known full run's config
exactly (max_x=20, dimensions=2, max_lr_scale=0.25, manual grad_clip=1,
manual delta_clip=10 -- see TCA1.1.4/full_confirmation_multi_report.json,
R2=0.705, MAE=8.35, the best result found on this dataset all session).

Exam-score's clusters come out much more evenly sized than
YearPredictionMSD's (recall ~3200-3216 rows/segment in every prior full
run here), so proportional sharding should mostly decide shard_count=1
per segment -- meaning this exercises the merge path rarely or not at
all, which is exactly the point: does train_parallel_balanced reproduce
the same result as train_parallel_partitioned when balancing isn't
actually needed?
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from SystemHandler import SystemHandler
import Components.RichConsole as RC
from comparisons.shared_metrics import compute_metrics

TARGET = "exam_score"
TEST_SPLIT = 0.2
CHECKPOINT_PATH = "examscore_balanced_report.json"

REFERENCE = {"r2": 0.7052, "mae": 8.354, "source": "TCA1.1.4/full_confirmation_multi_report.json (max_lr_scale=0.25)"}


def evaluate(system, test_rows, target, aggregation_mode, selection_percentage):
    preds, actuals = [], []
    for row in test_rows:
        actual = row.get(target)
        if actual is None:
            continue
        sample = {k: v for k, v in row.items() if k != target}
        result = system.runInfer(sample, loud=False, aggregation_mode=aggregation_mode,
                                 selection_percentage=selection_percentage)
        if result is None:
            continue
        preds.append(float(result["score"]))
        actuals.append(float(actual))
    m = compute_metrics(preds, actuals)
    m["n"] = len(preds)
    return m


def main():
    t_start = time.time()
    print("=== Loading full exam-score dataset ===")
    dataset = pd.read_csv("Exam_Score_Prediction.csv").drop(columns=["student_id"])
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    n_test = int(len(dataset) * TEST_SPLIT)
    train_df = dataset.iloc[:-n_test].reset_index(drop=True)
    test_rows = dataset.iloc[-n_test:].to_dict(orient="records")
    print(f"total={len(dataset)}  train={len(train_df)}  test={len(test_rows)}")

    lr_cfg = {"mode": "manual-full", "max_lr_scale": 0.25, "decay": 0.85}
    grad_cfg = {"mode": "manual", "value": 1}
    delta_cfg = {"mode": "manual", "value": 10}
    pred_range = {"mode": "manual", "min_value": 0, "max_value": 100}

    logger = RC.RichLogger(filename=f"examscore_balanced_{int(time.time())}.log", log_level=4, console_level=2)
    system = SystemHandler(maxX=20, target=TARGET, logger=logger, connection_percentage=0.1,
                           density=0.8, dimensions=2, classification=4)
    system.initializeAllSegments(Loud=False)

    print("\n=== train_parallel_balanced (max_x=20, full 20k exam-score) ===")
    t0 = time.time()
    shard_results = system.train_parallel_balanced(
        train_df, epoch_count=20, judge_iterations=20, judge_min_clusters=4, judge_max_clusters=20,
        loud=True, lr_scale_cfg=lr_cfg, prediction_range_cfg=pred_range,
        grad_clip_cfg=grad_cfg, delta_clip_cfg=delta_cfg, reconnect_pct=0.0, position_momentum=0.0,
        max_workers=None,
    )
    train_time = time.time() - t0
    print(f"train wall time: {train_time:.1f}s")
    print("shard/merge results:", shard_results)

    t0 = time.time()
    metrics = evaluate(system, test_rows, TARGET, "bma", 0.5)
    eval_time = time.time() - t0
    print(f"eval wall time: {eval_time:.1f}s  ->  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.3f}")

    results = {
        "reference_unbalanced": REFERENCE,
        "balanced": {
            "train_wall_time_sec": round(train_time, 1),
            "eval_wall_time_sec": round(eval_time, 1),
            "shard_results": shard_results,
            "test_metrics": metrics,
        },
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    n_multishard = sum(1 for r in shard_results if r["n_shards"] > 1)
    print(f"\n=== Summary ===")
    print(f"Segments that needed multi-sharding: {n_multishard}/{len(shard_results)}")
    print(f"balanced R2={metrics['r2']:.4f}  MAE={metrics['mae']:.3f}")
    print(f"reference R2={REFERENCE['r2']:.4f}  MAE={REFERENCE['mae']:.3f}")
    print(f"delta R2: {metrics['r2'] - REFERENCE['r2']:+.4f}")
    print(f"Total elapsed: {results['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
