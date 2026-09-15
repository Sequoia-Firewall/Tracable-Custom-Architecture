"""
Speed comparison: train_parallel_balanced() vs the already-measured
train_parallel_partitioned() baseline, same 40,000-row YearPredictionMSD
sample/config as _tmp_yearpred_xscale_test/run_yearpred_xscale.py.

Only testing max_x=15 and max_x=20 -- the two most expensive stages, where
the baseline's imbalance (8.0x and 9.35x respectively) cost the most wall-
clock in absolute terms. max_x=5/10 finished fast enough in the baseline
that imbalance there isn't worth re-measuring.

Baseline reference (train_parallel_partitioned, already measured):
  max_x=15: R2=-0.0921  MAE=8.752   train_time=6514s   worker_times=[6285.74, 5350.74, 1627.76, 783.29]
  max_x=20: R2=-0.0379  MAE=8.009   train_time=15755s  worker_times=[15525.88, 11824.96, 3569.09, 1659.98]

If proportional sharding achieves close to ideal 4-way balance, expect
wall-clock near (sum of worker times)/4 rather than max(worker times):
  max_x=15: ideal ~3512s (vs 6514s baseline)
  max_x=20: ideal ~8145s (vs 15755s baseline)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from SystemHandler import SystemHandler
import Components.RichConsole as RC
from comparisons.shared_metrics import compute_metrics

MAX_X_VALUES = [15, 20]
SAMPLE_SIZE = 40000
TEST_SPLIT = 0.2
DIMENSIONS = 2
EPOCH_COUNT = 10
JUDGE_ITERATIONS = 10
TARGET = "year"
PRED_MIN, PRED_MAX = 1922, 2011

AUTO = {"mode": "auto"}
CHECKPOINT_PATH = "yearpred_balanced_report.json"


def evaluate(system, test_rows, aggregation_mode="bma", selection_percentage=0.5):
    preds, actuals = [], []
    for row in test_rows:
        actual = row.get(TARGET)
        if actual is None:
            continue
        sample = {k: v for k, v in row.items() if k != TARGET}
        result = system.runInfer(sample, loud=False, aggregation_mode=aggregation_mode,
                                 selection_percentage=selection_percentage)
        if result is None:
            continue
        preds.append(float(result["score"]))
        actuals.append(float(actual))
    m = compute_metrics(preds, actuals)
    m["n"] = len(preds)
    return m


def run_one_max_x(max_x, train_df, test_rows):
    logger = RC.RichLogger(filename=f"yearpred_balanced_maxx{max_x}_{int(time.time())}.log",
                            log_level=4, console_level=2)
    system = SystemHandler(maxX=max_x, target=TARGET, logger=logger,
                           connection_percentage=0.1, density=0.8,
                           dimensions=DIMENSIONS, classification=4)
    system.initializeAllSegments(Loud=False)

    print(f"\n=== max_x={max_x} (balanced) ===")
    t0 = time.time()
    shard_results = system.train_parallel_balanced(
        train_df, epoch_count=EPOCH_COUNT, judge_iterations=JUDGE_ITERATIONS,
        judge_min_clusters=4, judge_max_clusters=8, loud=True,
        lr_scale_cfg=AUTO, prediction_range_cfg={"mode": "manual", "min_value": PRED_MIN, "max_value": PRED_MAX},
        grad_clip_cfg=AUTO, delta_clip_cfg=AUTO,
        reconnect_pct=0.0, position_momentum=0.0, max_workers=None,
    )
    train_time = time.time() - t0
    print(f"train wall time: {train_time:.1f}s")
    print("shard/merge results:", shard_results)

    t0 = time.time()
    metrics = evaluate(system, test_rows)
    eval_time = time.time() - t0
    print(f"eval wall time: {eval_time:.1f}s  ->  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.3f}")

    return {
        "max_x": max_x,
        "train_wall_time_sec": round(train_time, 1),
        "eval_wall_time_sec": round(eval_time, 1),
        "shard_results": shard_results,
        "test_metrics": metrics,
    }


def main():
    t_start = time.time()
    print("=== Loading YearPredictionMSD sample (same seed as xscale baseline) ===")
    cols = [TARGET] + [f"feat_{i}" for i in range(90)]
    full_df = pd.read_csv("../datasets/yearpredictionmsd/YearPredictionMSD.txt", header=None, names=cols)
    dataset = full_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    n_test = int(len(dataset) * TEST_SPLIT)
    train_df = dataset.iloc[:-n_test].reset_index(drop=True)
    test_rows = dataset.iloc[-n_test:].to_dict(orient="records")
    print(f"total={len(dataset)}  train={len(train_df)}  test={len(test_rows)}")

    baseline_reference = {
        15: {"r2": -0.0921, "mae": 8.752, "train_wall_time_sec": 6514.0,
             "worker_times": [6285.74, 5350.74, 1627.76, 783.29]},
        20: {"r2": -0.0379, "mae": 8.009, "train_wall_time_sec": 15755.0,
             "worker_times": [15525.88, 11824.96, 3569.09, 1659.98]},
    }

    results = {"config": {"sample_size": SAMPLE_SIZE, "n_train": len(train_df), "n_test": len(test_rows),
                          "dimensions": DIMENSIONS, "epoch_count": EPOCH_COUNT, "lr_mode": "auto"},
               "baseline_reference_unbalanced": baseline_reference,
               "runs": []}
    for max_x in MAX_X_VALUES:
        r = run_one_max_x(max_x, train_df, test_rows)
        results["runs"].append(r)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Summary: balanced vs unbalanced ===")
    for r in results["runs"]:
        base = baseline_reference[r["max_x"]]
        speedup = base["train_wall_time_sec"] / r["train_wall_time_sec"]
        print(f"  max_x={r['max_x']:3d}: balanced_time={r['train_wall_time_sec']:.0f}s  "
              f"baseline_time={base['train_wall_time_sec']:.0f}s  speedup={speedup:.2f}x  "
              f"balanced_R2={r['test_metrics']['r2']:.4f}  baseline_R2={base['r2']:.4f}")
    print(f"\nTotal elapsed: {results['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
