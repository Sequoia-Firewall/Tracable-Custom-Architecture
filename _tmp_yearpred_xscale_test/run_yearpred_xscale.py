"""
Full-auto (learning_rate/grad_clip/delta_clip all "auto", parallel
partitioned training) run on YearPredictionMSD -- a structurally different
dataset from the exam-score one every other test in this project is based
on (515,345 rows, 90 purely numeric audio features, no categoricals,
target = release year 1922-2011) -- with max_x ("x scale") varied across
several values, to check whether the now-fixed auto LR formula (clamp
bounds 0.1-1.0) behaves reasonably across graph sizes on genuinely
different data, rather than assuming what worked on exam-score generalizes.

Explicitly NOT changing settings.json's shipped default based on this or
any single-dataset result -- see conversation: TCA needs to stay a
general-purpose architecture, this is a generalization check, not a
hyperparameter search to hardcode into defaults.

Sizing, from direct calibration on this dataset (calibrate.py):
  max_x=15: 0.050 s/(row*epoch) measured (~2.5x exam-score's rate at the
            same depth -- consistent with ~90 features vs ~30)
  max_x=20: 0.105 s/(row*epoch) measured directly (not extrapolated)
40,000-row sample (2x the exam-score "full" test's 20,000 rows -- this
dataset has 515k available, but going further increases wall-clock
substantially for a generalization check, not worth it here), dimensions=2
(4 segments), so JudgeNode splits ~32,000 train rows across 4 parallel
workers -- wall-clock is bounded by the slowest worker's share, not the
sum. epoch_count=10 (not this project's usual 20) to keep total wall-clock
in the few-hours range across 4 max_x values -- a generalization check,
not a maximal-accuracy campaign.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from SystemHandler import SystemHandler
import Components.RichConsole as RC
from comparisons.shared_metrics import compute_metrics

MAX_X_VALUES = [5, 10, 15, 20]
SAMPLE_SIZE = 40000
TEST_SPLIT = 0.2
DIMENSIONS = 2
EPOCH_COUNT = 10
JUDGE_ITERATIONS = 10
TARGET = "year"
PRED_MIN, PRED_MAX = 1922, 2011

AUTO = {"mode": "auto"}
CHECKPOINT_PATH = "yearpred_xscale_report.json"


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
    logger = RC.RichLogger(filename=f"yearpred_xscale_maxx{max_x}_{int(time.time())}.log",
                            log_level=4, console_level=2)
    system = SystemHandler(maxX=max_x, target=TARGET, logger=logger,
                           connection_percentage=0.1, density=0.8,
                           dimensions=DIMENSIONS, classification=4)
    system.initializeAllSegments(Loud=False)

    print(f"\n=== max_x={max_x} ===")
    t0 = time.time()
    worker_results = system.train_parallel_partitioned(
        train_df, epoch_count=EPOCH_COUNT, judge_iterations=JUDGE_ITERATIONS,
        judge_min_clusters=4, judge_max_clusters=8, loud=True,
        lr_scale_cfg=AUTO, prediction_range_cfg={"mode": "manual", "min_value": PRED_MIN, "max_value": PRED_MAX},
        grad_clip_cfg=AUTO, delta_clip_cfg=AUTO,
        reconnect_pct=0.0, position_momentum=0.0, max_workers=None,
    )
    train_time = time.time() - t0
    print(f"train wall time: {train_time:.1f}s")

    t0 = time.time()
    metrics = evaluate(system, test_rows)
    eval_time = time.time() - t0
    print(f"eval wall time: {eval_time:.1f}s  ->  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.3f}")

    return {
        "max_x": max_x,
        "train_wall_time_sec": round(train_time, 1),
        "eval_wall_time_sec": round(eval_time, 1),
        "worker_results": worker_results,
        "test_metrics": metrics,
    }


def main():
    t_start = time.time()
    print("=== Loading YearPredictionMSD sample ===")
    cols = [TARGET] + [f"feat_{i}" for i in range(90)]
    full_df = pd.read_csv("../datasets/yearpredictionmsd/YearPredictionMSD.txt", header=None, names=cols)
    dataset = full_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    n_test = int(len(dataset) * TEST_SPLIT)
    train_df = dataset.iloc[:-n_test].reset_index(drop=True)
    test_rows = dataset.iloc[-n_test:].to_dict(orient="records")
    print(f"total={len(dataset)}  train={len(train_df)}  test={len(test_rows)}")

    results = {"config": {"sample_size": SAMPLE_SIZE, "n_train": len(train_df), "n_test": len(test_rows),
                          "dimensions": DIMENSIONS, "epoch_count": EPOCH_COUNT,
                          "lr_mode": "auto", "grad_clip_mode": "auto", "delta_clip_mode": "auto",
                          "parallel_enabled": True},
               "runs": []}
    for max_x in MAX_X_VALUES:
        r = run_one_max_x(max_x, train_df, test_rows)
        results["runs"].append(r)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Summary: auto mode across max_x on YearPredictionMSD ===")
    for r in results["runs"]:
        tm = r["test_metrics"]
        print(f"  max_x={r['max_x']:3d}: R2={tm['r2']:.4f}  MAE={tm['mae']:.3f}  "
              f"train_time={r['train_wall_time_sec']:.0f}s")
    print(f"\nTotal elapsed: {results['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
