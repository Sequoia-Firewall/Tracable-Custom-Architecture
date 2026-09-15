"""
Threat-detection derisking test: full auto mode (learning_rate/grad_clip/
delta_clip all auto, parallel partitioned training) across several max_x
values on NSL-KDD, binary threat target (0=normal, 1=any attack).

Uses the STANDARD NSL-KDD train/test split (KDDTrain+.txt / KDDTest+.txt)
rather than a random split of one file -- the official test set deliberately
includes attack patterns not present in training, which is a much closer
match to "can this generalize to a threat it hasn't seen" than a random
split would be. Train set is sampled down to keep cost bounded (same
discipline as the YearPredictionMSD tests); full official test set used
for evaluation.

Same max_x range [5,10,15,20] as the YearPredictionMSD x-scale test for
direct comparability.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from SystemHandler import SystemHandler
import Components.RichConsole as RC
from comparisons.shared_metrics import compute_metrics

COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label", "difficulty",
]

MAX_X_VALUES = [5, 10, 15, 20]
TRAIN_SAMPLE_SIZE = 40000
DIMENSIONS = 2
EPOCH_COUNT = 10
JUDGE_ITERATIONS = 10
TARGET = "threat"

AUTO = {"mode": "auto"}
CHECKPOINT_PATH = "nslkdd_xscale_report.json"


def load(path):
    df = pd.read_csv(path, header=None, names=COLS)
    df["threat"] = (df["label"] != "normal").astype(int)
    return df.drop(columns=["label", "difficulty"])


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
    logger = RC.RichLogger(filename=f"nslkdd_xscale_maxx{max_x}_{int(time.time())}.log",
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
        lr_scale_cfg=AUTO, prediction_range_cfg={"mode": "manual", "min_value": 0, "max_value": 1},
        grad_clip_cfg=AUTO, delta_clip_cfg=AUTO,
        reconnect_pct=0.0, position_momentum=0.0, max_workers=None,
    )
    train_time = time.time() - t0
    print(f"train wall time: {train_time:.1f}s")

    t0 = time.time()
    metrics = evaluate(system, test_rows)
    eval_time = time.time() - t0
    print(f"eval wall time: {eval_time:.1f}s  ->  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.3f}  "
          f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}")

    return {
        "max_x": max_x,
        "train_wall_time_sec": round(train_time, 1),
        "eval_wall_time_sec": round(eval_time, 1),
        "worker_results": worker_results,
        "test_metrics": metrics,
    }


def main():
    t_start = time.time()
    print("=== Loading NSL-KDD (standard train/test split) ===")
    train_full = load("../datasets/nslkdd/KDDTrain+.txt")
    test_full = load("../datasets/nslkdd/KDDTest+.txt")
    train_df = train_full.sample(n=min(TRAIN_SAMPLE_SIZE, len(train_full)), random_state=42).reset_index(drop=True)
    test_rows = test_full.to_dict(orient="records")
    print(f"train_sample={len(train_df)}  test(official)={len(test_rows)}  "
          f"train_threat_rate={train_df['threat'].mean():.3f}  test_threat_rate={test_full['threat'].mean():.3f}")

    results = {"config": {"train_sample_size": len(train_df), "n_test": len(test_rows),
                          "dimensions": DIMENSIONS, "epoch_count": EPOCH_COUNT, "lr_mode": "auto"},
               "runs": []}
    for max_x in MAX_X_VALUES:
        r = run_one_max_x(max_x, train_df, test_rows)
        results["runs"].append(r)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Summary ===")
    for r in results["runs"]:
        tm = r["test_metrics"]
        print(f"  max_x={r['max_x']:3d}: R2={tm['r2']:.4f}  MAE={tm['mae']:.3f}  "
              f"precision={tm['precision']:.3f}  recall={tm['recall']:.3f}  f1={tm['f1']:.3f}  "
              f"train_time={r['train_wall_time_sec']:.0f}s")
    print(f"\nTotal elapsed: {results['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
