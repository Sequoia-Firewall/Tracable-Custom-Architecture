"""
_tmp_lr_formula_sweep/sweep_confirmation.py
------------------------------------------------
The "real test" sweep_hops_rows.py's trend-sniffing pass was building
toward: 3 repeats per cell (statistical rigor, like sweep v2), finer LR
candidates concentrated in the range everything so far has pointed to
(0.05-0.9, not the original 0.02-3.0), sample sizes chosen per-depth to
bracket each depth's own observed plateau-then-drop transition from
sweep_hops_rows, PLUS a max_x=20 point for direct production-scale
relevance (never swept before -- the two full-dataset runs used a single
fixed config, not a search).

Per-(max_x, n) sample sizes were chosen from sweep_hops_rows_results.json's
already-observed transitions:
  max_x=5  (hops=4):  plateaus at 1.0 through n=600, drops to 0.5 by n=1200
                        -> bracket with [400, 800, 1400]
  max_x=10 (hops=8):  drops earliest, between n=300 and n=600
                        -> bracket with [200, 450, 800]
  max_x=15 (hops=12): drops between n=600 and n=1200 (like max_x=5, despite
                        being deeper -- the part sweep_hops_rows couldn't
                        resolve with only 1 repeat)
                        -> bracket with [400, 800, 1400]
  max_x=20: no prior sweep data at all (production scale) -- one point
                        (n=600) as a direct confirmation/sanity check
                        against whatever the max_x=5/10/15 curve predicts,
                        not a full mini-grid (cost: this is the most
                        expensive max_x tested, ~0.36s/row at epoch=8 based
                        on extrapolating the real max_x=10->15 node*hops
                        scaling relationship, which matched observed data
                        well).

Real per-max_x times measured across sweep v2 + sweep_hops_rows (used to
size this grid, see conversation): max_x=5 ~0.0073s/row, max_x=10
~0.050s/row, max_x=15 ~0.160s/row (all at epoch_count=8).
"""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from SegmentHandler import SegmentHandler
import Components.RichConsole as RC

TARGET = "exam_score"
DIMENSIONS = 2
CONNECTION_PCT = 0.1
DENSITY = 0.8
EPOCH_COUNT = 8
PRED_RANGE = {"mode": "manual", "min_value": 0, "max_value": 100}
AUTO_CFG = {"mode": "auto"}

MAX_X_SAMPLES = {
    5: [400, 800, 1400],
    10: [200, 450, 800],
    15: [400, 800, 1400],
    20: [600],
}
LR_CANDIDATES = [0.05, 0.15, 0.3, 0.5, 0.9]
N_REPEATS = 3

CSV_PATH = "Exam_Score_Prediction.csv"
CHECKPOINT_PATH = "sweep_confirmation_results.json"


def run_one(df, max_x, epoch_count, lr_scale, logger):
    handler = SegmentHandler(maxX=max_x, target=TARGET, logger=logger,
                             connection_percentage=CONNECTION_PCT, density=DENSITY,
                             dimensions=DIMENSIONS, classification=4, segment_id=0)
    handler.initializeSegment()
    num_nodes = len(handler.segmentComponents['processing_nodes'])
    hops = max(1, math.ceil(num_nodes ** (1.0 / DIMENSIONS)))

    lr_cfg = {"mode": "manual-scale", "max_lr_scale": lr_scale, "min_lr_scale": 0.05 * lr_scale}
    handler.train(df, epoch_count=epoch_count, lr_scale_cfg=lr_cfg,
                  pred_min=PRED_RANGE["min_value"], pred_max=PRED_RANGE["max_value"],
                  grad_clip_cfg=AUTO_CFG, delta_clip_cfg=AUTO_CFG,
                  reconnect_pct=0.0, position_momentum=0.0)

    m = handler.best_epoch_metrics or {}
    return {"r2": m.get("r2"), "n_train": m.get("n_train"), "num_nodes": num_nodes, "hops": hops}


def mean_std_stderr(values):
    v = [x for x in values if x is not None and not math.isnan(x)]
    if not v:
        return None, None, None
    arr = np.array(v, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    stderr = std / math.sqrt(len(arr)) if len(arr) > 1 else float("nan")
    return mean, std, stderr


def main():
    t_start = time.time()
    logger = RC.RichLogger(filename=f"sweep_confirmation_{int(time.time())}.log", log_level=0, console_level=5)
    full_df = pd.read_csv(CSV_PATH).drop(columns=["student_id"])

    cells = [(max_x, n) for max_x, ns in MAX_X_SAMPLES.items() for n in ns]
    n_runs_total = len(cells) * len(LR_CANDIDATES) * N_REPEATS
    print(f"=== {len(cells)} (max_x, n) cells x {len(LR_CANDIDATES)} LR candidates x "
          f"{N_REPEATS} repeats = {n_runs_total} runs ===")

    results = {"cells": []}
    best_per_cell = {}
    for max_x, n in cells:
        df = full_df.sample(n=n, random_state=42).reset_index(drop=True)
        cell_summaries = []
        for lr in LR_CANDIDATES:
            trial_r2s, trial_meta = [], None
            for rep in range(N_REPEATS):
                t0 = time.time()
                r = run_one(df, max_x, EPOCH_COUNT, lr, logger)
                dt = time.time() - t0
                trial_r2s.append(r["r2"])
                trial_meta = r
                results["cells"].append({
                    "max_x": max_x, "sample_size": n, "lr_scale": lr, "repeat": rep,
                    "r2": r["r2"], "n_train": r["n_train"], "num_nodes": r["num_nodes"],
                    "hops": r["hops"], "wall_time_sec": round(dt, 1),
                })
            mean, std, stderr = mean_std_stderr(trial_r2s)
            cell_summaries.append({
                "max_x": max_x, "sample_size": n, "lr_scale": lr,
                "hops": trial_meta["hops"], "n_train": trial_meta["n_train"],
                "r2_mean": mean, "r2_std": std, "r2_stderr": stderr, "r2_trials": trial_r2s,
            })
            print(f"  max_x={max_x:3d} n={n:5d} lr={lr:5.3f}  ->  "
                  f"R2 mean={mean:.4f} std={std:.4f} (trials={[round(x,3) for x in trial_r2s]})")
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2, default=str)

        valid = [c for c in cell_summaries if c["r2_mean"] is not None]
        if valid:
            best = max(valid, key=lambda c: c["r2_mean"])
            others = [c for c in valid if c is not best]
            if others:
                runner_up = max(others, key=lambda c: c["r2_mean"])
                gap = best["r2_mean"] - runner_up["r2_mean"]
                combined_stderr = math.sqrt((best["r2_stderr"] or 0) ** 2 + (runner_up["r2_stderr"] or 0) ** 2)
                best["distinguishable_from_runner_up"] = bool(combined_stderr > 0 and gap > combined_stderr)
                best["runner_up_lr"] = runner_up["lr_scale"]
                best["gap_vs_runner_up"] = gap
            best_per_cell[(max_x, n)] = best
            print(f"  -> best: lr={best['lr_scale']} R2={best['r2_mean']:.4f} "
                  f"distinguishable={best.get('distinguishable_from_runner_up')}")

    results["best_per_cell"] = [{"max_x": k[0], "sample_size": k[1], **v} for k, v in best_per_cell.items()]

    print("\n=== Summary: best LR per (max_x, n) cell ===")
    for (max_x, n), b in sorted(best_per_cell.items()):
        print(f"  max_x={max_x:3d} n={n:5d} hops={b['hops']:3d}  best_lr={b['lr_scale']:.3f}  "
              f"R2={b['r2_mean']:.4f}  distinguishable={b.get('distinguishable_from_runner_up')}")

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== DONE (total elapsed {results['elapsed_sec']:.0f}s) ===")


if __name__ == "__main__":
    main()
