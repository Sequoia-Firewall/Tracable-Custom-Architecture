"""
_tmp_lr_formula_sweep/sweep_hops_rows.py
--------------------------------------------
Quick trend-sniffing pass, NOT the final confirmatory test. sweep v2 found
that at fixed max_x, more training rows lowers the optimal LR, but the
SIZE of that effect depends on graph depth (hops) -- max_x=5 (hops=4)
showed zero sensitivity to rows, while max_x=10/15 (hops=8/12) both showed
a ~4x LR drop for the same ~2.67x row increase. The current formula
conflates hops and rows into one ratio (node_activations = rows*hops/nodes)
and that single-variable fit was WORSE than the raw data (non-monotonic
even restricted to the statistically-solid points).

This sweep treats hops (via max_x, still the only way to vary it in this
architecture) and train_rows as separate axes with several rows values per
hops level, to see the SLOPE of rows-sensitivity at each depth -- not just
2 points per depth like before. Deliberately cheap: 1 repeat (not 3), a
narrower LR candidate range (0.1-1.0, informed by everything sweep v1/v2
already ruled out), same short epoch budget. This is meant to show whether
a depth-dependent rows-sensitivity is real and worth designing a proper
confirmatory sweep around -- not to produce a final formula.
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

MAX_X_LEVELS = [5, 10, 15]
SAMPLE_SIZES = [150, 300, 600, 1200]
LR_CANDIDATES = [0.1, 0.25, 0.5, 1.0]
N_REPEATS = 1  # quick pass -- rougher than sweep v2's 3 repeats on purpose

CSV_PATH = "Exam_Score_Prediction.csv"
CHECKPOINT_PATH = "sweep_hops_rows_results.json"


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


def main():
    t_start = time.time()
    logger = RC.RichLogger(filename=f"sweep_hops_rows_{int(time.time())}.log", log_level=0, console_level=5)
    full_df = pd.read_csv(CSV_PATH).drop(columns=["student_id"])

    n_runs = len(MAX_X_LEVELS) * len(SAMPLE_SIZES) * len(LR_CANDIDATES) * N_REPEATS
    print(f"=== {len(MAX_X_LEVELS)} hops levels x {len(SAMPLE_SIZES)} sample sizes x "
          f"{len(LR_CANDIDATES)} LR candidates x {N_REPEATS} repeat = {n_runs} runs ===")

    results = []
    best_per_cell = {}
    for max_x in MAX_X_LEVELS:
        for n in SAMPLE_SIZES:
            df = full_df.sample(n=n, random_state=42).reset_index(drop=True)
            cell_r2s = {}
            for lr in LR_CANDIDATES:
                for rep in range(N_REPEATS):
                    t0 = time.time()
                    r = run_one(df, max_x, EPOCH_COUNT, lr, logger)
                    dt = time.time() - t0
                    results.append({"max_x": max_x, "sample_size": n, "lr_scale": lr, "repeat": rep,
                                    "r2": r["r2"], "n_train": r["n_train"], "hops": r["hops"],
                                    "num_nodes": r["num_nodes"], "wall_time_sec": round(dt, 1)})
                    cell_r2s.setdefault(lr, []).append(r["r2"])
            means = {lr: (sum(v) / len(v)) for lr, v in cell_r2s.items() if v}
            if means:
                best_lr = max(means, key=means.get)
                best_per_cell[(max_x, n)] = {"best_lr": best_lr, "r2": means[best_lr], "all_means": means}
                print(f"  max_x={max_x:3d} n={n:5d}  ->  best_lr={best_lr:.2f} R2={means[best_lr]:.4f}  "
                      f"(all: {[(k, round(v,3)) for k,v in sorted(means.items())]})")
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump({"runs": results, "best_per_cell": [
                    {"max_x": k[0], "sample_size": k[1], **v} for k, v in best_per_cell.items()
                ]}, f, indent=2, default=str)

    # ── Trend summary: does rows-sensitivity scale with hops? ─────────────
    print("\n=== Rows-sensitivity by depth (best_lr at smallest vs largest sample size) ===")
    trend = []
    for max_x in MAX_X_LEVELS:
        pts = [(n, best_per_cell[(max_x, n)]["best_lr"]) for n in SAMPLE_SIZES if (max_x, n) in best_per_cell]
        if len(pts) >= 2:
            pts.sort()
            lo_n, lo_lr = pts[0]
            hi_n, hi_lr = pts[-1]
            ratio = (lo_lr / hi_lr) if hi_lr else float("nan")
            trend.append({"max_x": max_x, "points": pts, "lr_drop_ratio_low_to_high_n": ratio})
            print(f"  max_x={max_x:3d}: {pts}  ->  LR dropped {ratio:.2f}x from n={lo_n} to n={hi_n}")

    elapsed = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"runs": results, "best_per_cell": [
            {"max_x": k[0], "sample_size": k[1], **v} for k, v in best_per_cell.items()
        ], "trend": trend, "elapsed_sec": elapsed}, f, indent=2, default=str)
    print(f"\n=== DONE (total elapsed {elapsed:.0f}s) ===")


if __name__ == "__main__":
    main()
