"""
Full production-scale confirmation of the LR formula investigation:
tests several max_lr_scale values on the COMPLETE 20,000-row dataset
(not a sample sweep) to see actual accuracy behavior at real scale,
where node_activations~=219 (max_x=20, dimensions=2, ~3200 train
rows/segment) -- the smaller-scale sweeps (_tmp_lr_formula_sweep/) never
tested a real full run, only max_x up to 20 in a single (n=600) sample.

Every candidate uses the EXACT same setup as the already-completed manual
reference run (mode='manual-full', decay=0.85 fixed, manual grad_clip=1,
manual delta_clip=10, same train/test split via random_state=42) so the
comparison isolates ONLY max_lr_scale -- matching how the original
auto-vs-manual comparison was structured.

Candidates chosen to cover the untested range between the current floor
(0.1) and what's already known:
  - max_lr_scale=1.0 already ran (full_manual_parallel_report.json):
    R2=0.556 -- the current "known good" reference, not rerun here.
  - max_lr_scale~=3.0 (the OLD auto ceiling, via the decayed auto curve)
    already ran (full_auto_parallel_report.json): R2~=-0.004 -- already
    characterizes the high end as bad.
  - This script fills in the gap: 0.05 (what the sweep-fit formula
    extrapolated to, well beyond its tested row-count range -- this is
    the main thing being confirmed or refuted), 0.1 (the new shipped
    floor), 0.25, 0.5 (values the smaller-scale sweeps found optimal at
    shallower depths / lower row counts, worth checking whether they
    transfer to production scale).

Each candidate is a full sequential run through the whole pipeline --
NOT run concurrently with each other (each one's parallel training
already saturates all 4 CPU cores via ProcessPoolExecutor; running
candidates concurrently would just contend with itself for no benefit).
Checkpoints to disk after every candidate so a partial result is never
lost if interrupted.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from Settings import Settings
from SystemHandler import SystemHandler
import Components.RichConsole as RC
from comparisons.shared_metrics import compute_metrics

CANDIDATES = [0.05, 0.1, 0.25, 0.5]
TEST_SPLIT = 0.2
CHECKPOINT_PATH = "full_confirmation_multi_report.json"


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


def run_candidate(max_lr_scale, dataset, train_df, test_rows, d, m):
    settings = Settings("settings.json")
    settings.override("training.learning_rate.mode", "manual-full")
    settings.override("training.learning_rate.max_lr_scale", max_lr_scale)
    settings.override("training.learning_rate.decay", 0.85)
    settings.override("training.grad_clip.mode", "manual")
    settings.override("training.grad_clip.value", 1)
    settings.override("training.delta_clip.mode", "manual")
    settings.override("training.delta_clip.value", 10)
    settings.override("training.parallel_enabled", True)
    settings.override("training.max_workers", None)

    t = settings.training
    logger = RC.RichLogger(filename=f"full_confirmation_lr{max_lr_scale}_{int(time.time())}.log",
                            log_level=4, console_level=2)
    system = SystemHandler.from_settings(settings, logger)
    system.initializeAllSegments(Loud=False)

    print(f"\n=== max_lr_scale={max_lr_scale} ===")
    t0 = time.time()
    worker_results = system.train_parallel_partitioned(
        train_df, epoch_count=t["epoch_count"], judge_iterations=t["judge_iterations"],
        judge_min_clusters=t.get("judge_min_clusters"), judge_max_clusters=t.get("judge_max_clusters"),
        loud=True, lr_scale_cfg=t.get("learning_rate"), prediction_range_cfg=d.get("prediction_range"),
        grad_clip_cfg=t.get("grad_clip"), delta_clip_cfg=t.get("delta_clip"),
        reconnect_pct=t.get("reconnect_pct", 0.005), position_momentum=t.get("position_momentum", 0.0),
        max_workers=t.get("max_workers"),
    )
    train_time = time.time() - t0
    print(f"train wall time: {train_time:.1f}s")

    t0 = time.time()
    metrics = evaluate(system, test_rows, d["target_column"], m["aggregation_mode"], m["selection_percentage"])
    eval_time = time.time() - t0
    print(f"eval wall time: {eval_time:.1f}s  ->  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.3f}")

    return {
        "max_lr_scale": max_lr_scale,
        "train_wall_time_sec": round(train_time, 1),
        "eval_wall_time_sec": round(eval_time, 1),
        "worker_results": worker_results,
        "test_metrics": metrics,
    }


def main():
    t_start = time.time()
    settings = Settings("settings.json")
    d, m = settings.dataset, settings.model

    print("=== Loading full dataset ===")
    dataset = pd.read_csv(d["csv_path"])
    if d.get("shuffle", True):
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    n_test = int(len(dataset) * settings.training.get("test_split", TEST_SPLIT))
    train_df = dataset.iloc[:-n_test].reset_index(drop=True)
    test_rows = dataset.iloc[-n_test:].to_dict(orient="records")
    print(f"total={len(dataset)}  train={len(train_df)}  test={len(test_rows)}")

    results = {
        "reference_points_already_measured": {
            "max_lr_scale=1.0 (manual-full, decay=0.85)": {"r2": 0.5564663911964134, "mae": 10.261075859053616,
                                                             "source": "full_manual_parallel_report.json"},
            "max_lr_scale~=3.0 (old auto ceiling)": {"r2": -0.004268064013699835, "mae": 15.805567172893674,
                                                       "source": "full_auto_parallel_report.json"},
        },
        "candidates": [],
    }
    for lr in CANDIDATES:
        result = run_candidate(lr, dataset, train_df, test_rows, d, m)
        results["candidates"].append(result)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Summary ===")
    print("max_lr_scale=1.0 (reference): R2=0.5565")
    print("max_lr_scale~=3.0 (old auto, reference): R2=-0.0043")
    for c in results["candidates"]:
        print(f"max_lr_scale={c['max_lr_scale']}: R2={c['test_metrics']['r2']:.4f}  MAE={c['test_metrics']['mae']:.3f}")
    print(f"\nTotal elapsed: {results['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
