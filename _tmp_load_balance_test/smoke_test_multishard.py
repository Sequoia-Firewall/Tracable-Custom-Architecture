"""The first smoke test's clusters came out too balanced to trigger
multi-sharding at all (n_shards=1 everywhere -> byte-identical to the
baseline, which is correct but doesn't exercise _merge_segment_shards'
actual averaging/reconnect path). Force it by setting max_workers high
relative to data size (target_shard_size = total_rows / max_workers
shrinks, so shard_count = round(segment_rows / target) exceeds 1) --
oversubscription on a 4-core box just queues extra tasks, doesn't fail."""
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

    print("\n=== train_parallel_balanced, max_workers=16 (forces multi-sharding) ===")
    logger = RC.RichLogger(filename=f"smoke_multishard_{int(time.time())}.log", log_level=0, console_level=5)
    system = SystemHandler(maxX=10, target=TARGET, logger=logger, connection_percentage=0.1,
                           density=0.8, dimensions=2, classification=4)
    system.initializeAllSegments(Loud=False)
    t0 = time.time()
    results = system.train_parallel_balanced(
        train_df, epoch_count=5, judge_iterations=5, judge_min_clusters=4, judge_max_clusters=8,
        loud=False, lr_scale_cfg=lr_cfg, prediction_range_cfg=pred_range,
        grad_clip_cfg=grad_cfg, delta_clip_cfg=delta_cfg, reconnect_pct=0.0, max_workers=16,
    )
    dt = time.time() - t0
    print(f"train_parallel_balanced (forced multi-shard): {dt:.1f}s")
    print("shard/merge results:", results)
    assert any(r["n_shards"] > 1 for r in results), "FAILED: still no multi-sharding triggered"
    print("CONFIRMED: multi-sharding triggered for at least one segment")

    m = evaluate(system, test_rows)
    print("multi-shard-merged test metrics:", m)
    print(f"\nR2={m['r2']:.4f} MAE={m['mae']:.3f}  (sanity: not NaN, not wildly broken)")

    # Structural sanity checks on the merged segments themselves
    import json
    for seg in system.segments:
        with open(f"segment_{seg.segment_id}.nexseg") as f:
            data = json.load(f)
        n_nodes = len(data["processing_nodes"])
        n_conns = sum(len(n["connected_positions"]) for n in data["processing_nodes"])
        print(f"segment {seg.segment_id}: {n_nodes} nodes, {n_conns} total connections "
              f"(0 would mean reconnect after merge silently failed)")
        assert n_conns > 0, f"segment {seg.segment_id} has zero connections after merge!"


if __name__ == "__main__":
    main()
