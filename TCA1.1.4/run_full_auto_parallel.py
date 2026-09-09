"""
One-off full-dataset production run: entire Exam_Score_Prediction.csv
(20,000 rows), auto-derived learning rate / grad clip / delta clip /
prediction range (rather than the manually-tuned settings.json defaults),
and parallel partitioned training (training.parallel_enabled=True).

Non-interactive equivalent of `python3 main.py --mode train` with those
settings overrides, run directly against SystemHandler so it doesn't block
on main.py's confirmation prompts. Held-out test metrics computed the same
way as the earlier parallel-training validation scripts (disjoint test
split the system never trains on).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from Settings import Settings
from SystemHandler import SystemHandler
import Components.RichConsole as RC
from comparisons.shared_metrics import compute_metrics

TEST_SPLIT = 0.2


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
    settings = Settings("settings.json")

    # "full auto features": switch the precision knobs from this repo's
    # manually-tuned settings.json values to the auto-derived formulas.
    settings.override("dataset.prediction_range.mode", "auto")
    settings.override("training.learning_rate.mode", "auto")
    settings.override("training.learning_rate.min_lr_scale", 0.5)
    settings.override("training.learning_rate.max_lr_scale", 3.0)
    settings.override("training.grad_clip.mode", "auto")
    settings.override("training.delta_clip.mode", "auto")

    # Parallel training, opt-in flag added in TCA1.1.4.
    settings.override("training.parallel_enabled", True)
    settings.override("training.max_workers", None)

    d, m, t = settings.dataset, settings.model, settings.training
    print(f"model: max_x={m['max_x']} dimensions={m['dimensions']} training_mode={m['training_mode']}")
    print(f"training: epoch_count={t['epoch_count']} judge_iterations={t['judge_iterations']} "
          f"lr_mode={t['learning_rate']['mode']} grad_clip={t['grad_clip']['mode']} "
          f"delta_clip={t['delta_clip']['mode']} parallel_enabled={t['parallel_enabled']}")

    logger = RC.RichLogger(filename=f"full_auto_parallel_{int(time.time())}.log", log_level=4, console_level=2)

    print("\n=== Loading full dataset ===")
    dataset = pd.read_csv(d["csv_path"])
    if d.get("shuffle", True):
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    n_test = int(len(dataset) * t.get("test_split", TEST_SPLIT))
    train_df = dataset.iloc[:-n_test].reset_index(drop=True)
    test_rows = dataset.iloc[-n_test:].to_dict(orient="records")
    print(f"total={len(dataset)}  train={len(train_df)}  test={len(test_rows)}")

    system = SystemHandler.from_settings(settings, logger)
    system.initializeAllSegments(Loud=False)

    print("\n=== Training (parallel partitioned, auto-derived config) ===")
    t0 = time.time()
    worker_results = system.train_parallel_partitioned(
        train_df,
        epoch_count=t["epoch_count"],
        judge_iterations=t["judge_iterations"],
        judge_min_clusters=t.get("judge_min_clusters"),
        judge_max_clusters=t.get("judge_max_clusters"),
        loud=True,
        lr_scale_cfg=t.get("learning_rate"),
        prediction_range_cfg=d.get("prediction_range"),
        grad_clip_cfg=t.get("grad_clip"),
        delta_clip_cfg=t.get("delta_clip"),
        reconnect_pct=t.get("reconnect_pct", 0.005),
        position_momentum=t.get("position_momentum", 0.0),
        max_workers=t.get("max_workers"),
    )
    train_time = time.time() - t0
    print(f"\nTraining wall time: {train_time:.1f}s")
    print("worker breakdown:", worker_results)

    print("\n=== Evaluating on held-out test split ===")
    t0 = time.time()
    metrics = evaluate(system, test_rows, d["target_column"], m["aggregation_mode"], m["selection_percentage"])
    eval_time = time.time() - t0
    print(f"Eval wall time: {eval_time:.1f}s")
    print("test metrics:", metrics)

    report = {
        "config": {
            "n_rows_total": len(dataset), "n_train": len(train_df), "n_test": len(test_rows),
            "max_x": m["max_x"], "dimensions": m["dimensions"], "training_mode": m["training_mode"],
            "epoch_count": t["epoch_count"], "judge_iterations": t["judge_iterations"],
            "lr_mode": "auto", "grad_clip_mode": "auto", "delta_clip_mode": "auto",
            "parallel_enabled": True, "max_workers": t.get("max_workers"),
        },
        "train_wall_time_sec": round(train_time, 1),
        "eval_wall_time_sec": round(eval_time, 1),
        "worker_results": worker_results,
        "test_metrics": metrics,
        "total_elapsed_sec": round(time.time() - t_start, 1),
    }
    with open("full_auto_parallel_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n=== DONE (total elapsed {report['total_elapsed_sec']:.0f}s) ===")
    print(f"Report written -> full_auto_parallel_report.json")


if __name__ == "__main__":
    main()
