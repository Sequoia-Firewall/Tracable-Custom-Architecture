"""
_tmp_lr_formula_sweep/sweep_lr_formula.py
-------------------------------------------
Empirically calibrates SegmentHandler's "auto" learning-rate formula.

Background: the current formula is
    max_lr_scale = clip(LR_SCALE_REFERENCE_ROWS / node_activations, floor, ceiling)
    where node_activations = train_rows * hops / num_nodes
    (LR_SCALE_REFERENCE_ROWS=2000, floor=0.5, ceiling=3.0 by default)
This was hand-calibrated against a single reference config. A real
full-dataset run at max_x=20/dimensions=2 (node_activations ~= 219) hit the
formula's ceiling (3.0, i.e. 3x the validated manual max_lr_scale=1.0) and
scored R2 ~= 0 -- essentially failed to learn. grad_clip and delta_clip's
auto values came out close to the manual reference for that same run, so
LR is the isolated culprit there -- see full_auto_parallel_report.json /
full_manual_parallel_report.json in TCA1.1.4/ for the run this diagnosis
came from.

Phase 1: sweep (max_x, sample_size) to vary node_activations across a wide
range (holding grad_clip/delta_clip at 'auto', dimensions=2 fixed), and at
each capacity point grid-search which max_lr_scale actually gives the best
held-out R2. Fit a power law (log-log linear regression) to the resulting
(node_activations, best_lr_scale) pairs and compare it against the current
hardcoded formula.

Phase 2: at two representative capacity points, layer in a small grad_clip
multiplier sweep around the phase-1 best LR, to check whether the joint
optimum shifts -- i.e. whether LR alone is a safe thing to calibrate in
isolation, or whether grad_clip needs recalibrating too.

Deliberately small-scale (short epoch counts, modest max_x, small samples)
so this sweep is cheap to run -- it only needs to rank LR candidates
relative to each other at each capacity point, not produce a
production-quality final model. Any promising fitted formula should be
confirmed with a couple of full-scale runs afterward, not by sweeping at
full scale here.
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
EPOCH_COUNT = 8          # short -- only need relative ranking, not a converged model
PRED_RANGE = {"mode": "manual", "min_value": 0, "max_value": 100}
AUTO_CFG = {"mode": "auto"}

# Phase 1 grid: varies node_activations via graph size (max_x) and data
# volume (sample_size) independently -- these are the two inputs the
# formula actually multiplies together.
CAPACITY_GRID = [(max_x, n) for max_x in (5, 10, 15) for n in (300, 800)]
LR_CANDIDATES = [0.25, 0.5, 1.0, 1.5, 2.5, 4.0]

# Phase 2: representative low/high capacity points + a small grad_clip
# multiplier grid layered on top of phase 1's best LR (and neighbors).
GRAD_CLIP_MULTIPLIERS = [0.5, 1.0, 2.0]
LR_OFFSET_FACTORS = [0.7, 1.0, 1.4]  # relative to phase-1 best at that point

CSV_PATH = "Exam_Score_Prediction.csv"
CHECKPOINT_PATH = "sweep_results.json"


def measure_capacity(max_x: int, logger) -> dict:
    """Build a throwaway segment just to measure num_nodes/hops at this max_x
    -- node count depends on density/connection_percentage in a way that
    isn't closed-form, so this is cheaper than re-deriving it analytically."""
    probe = SegmentHandler(maxX=max_x, target=TARGET, logger=logger,
                           connection_percentage=CONNECTION_PCT, density=DENSITY,
                           dimensions=DIMENSIONS, classification=4, segment_id=0)
    probe.initializeSegment()
    num_nodes = len(probe.segmentComponents['processing_nodes'])
    hops = max(1, math.ceil(num_nodes ** (1.0 / DIMENSIONS)))
    return {"num_nodes": num_nodes, "hops": hops}


def run_one(df, max_x, epoch_count, lr_scale, grad_clip_multiplier, logger):
    """Train one SegmentHandler with a forced manual-scale LR (reproducing
    auto's floor-ratio/decay-solving shape, just with the max_lr_scale
    candidate forced in) and return (r2, n_train, num_nodes, hops)."""
    handler = SegmentHandler(maxX=max_x, target=TARGET, logger=logger,
                             connection_percentage=CONNECTION_PCT, density=DENSITY,
                             dimensions=DIMENSIONS, classification=4, segment_id=0)
    handler.initializeSegment()
    num_nodes = len(handler.segmentComponents['processing_nodes'])
    hops = max(1, math.ceil(num_nodes ** (1.0 / DIMENSIONS)))

    lr_cfg = {"mode": "manual-scale", "max_lr_scale": lr_scale, "min_lr_scale": 0.05 * lr_scale}
    grad_cfg = dict(AUTO_CFG)
    if grad_clip_multiplier != 1.0:
        # _auto_grad_clip has no multiplier knob -- resolve auto's own value
        # first, then apply the multiplier manually via 'manual' mode so
        # phase 2 can test grad_clip values the formula itself can't reach.
        pred_min, pred_max = PRED_RANGE["min_value"], PRED_RANGE["max_value"]
        base = SegmentHandler._auto_grad_clip(pred_min, pred_max)
        grad_cfg = {"mode": "manual", "value": base * grad_clip_multiplier}

    handler.train(df, epoch_count=epoch_count, lr_scale_cfg=lr_cfg,
                  pred_min=PRED_RANGE["min_value"], pred_max=PRED_RANGE["max_value"],
                  grad_clip_cfg=grad_cfg, delta_clip_cfg=AUTO_CFG,
                  reconnect_pct=0.0, position_momentum=0.0)

    m = handler.best_epoch_metrics or {}
    return {
        "r2": m.get("r2"), "n_train": m.get("n_train"),
        "num_nodes": num_nodes, "hops": hops,
    }


def node_activations(n_train, hops, num_nodes):
    return n_train * hops / num_nodes


def fit_power_law(xs, ys):
    """lr_scale = a * node_activations^b, via log-log least squares."""
    log_x = np.log(np.array(xs, dtype=float))
    log_y = np.log(np.array(ys, dtype=float))
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = math.exp(log_a)
    return a, b


def current_formula(node_act, floor=0.5, ceiling=3.0, reference=2000.0):
    return min(max(reference / node_act, floor), ceiling)


def main():
    t_start = time.time()
    logger = RC.RichLogger(filename=f"sweep_{int(time.time())}.log", log_level=0, console_level=5)
    full_df = pd.read_csv(CSV_PATH).drop(columns=["student_id"])

    results = {"phase1": [], "phase2": []}

    # ── Phase 1 ──────────────────────────────────────────────────────────
    print(f"=== Phase 1: {len(CAPACITY_GRID)} capacity points x {len(LR_CANDIDATES)} LR candidates "
          f"= {len(CAPACITY_GRID) * len(LR_CANDIDATES)} runs ===")
    phase1_best = {}  # (max_x, n) -> {"node_activations":, "best_lr":, "best_r2":}
    for max_x, n in CAPACITY_GRID:
        df = full_df.sample(n=n, random_state=42).reset_index(drop=True)
        point_results = []
        for lr in LR_CANDIDATES:
            t0 = time.time()
            r = run_one(df, max_x, EPOCH_COUNT, lr, 1.0, logger)
            dt = time.time() - t0
            act = node_activations(r["n_train"], r["hops"], r["num_nodes"])
            entry = {"max_x": max_x, "sample_size": n, "lr_scale": lr,
                     "r2": r["r2"], "n_train": r["n_train"], "num_nodes": r["num_nodes"],
                     "hops": r["hops"], "node_activations": act, "wall_time_sec": round(dt, 1)}
            point_results.append(entry)
            print(f"  max_x={max_x:3d} n={n:4d} lr={lr:4.2f}  ->  "
                  f"R2={r['r2']:.4f}  node_act={act:.1f}  ({dt:.1f}s)")
        results["phase1"].extend(point_results)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

        valid = [e for e in point_results if e["r2"] is not None and not math.isnan(e["r2"])]
        if valid:
            best = max(valid, key=lambda e: e["r2"])
            phase1_best[(max_x, n)] = best
            print(f"  -> best at this point: lr={best['lr_scale']} R2={best['r2']:.4f} "
                  f"node_act={best['node_activations']:.1f}")

    # ── Fit curve ────────────────────────────────────────────────────────
    xs = [b["node_activations"] for b in phase1_best.values()]
    ys = [b["lr_scale"] for b in phase1_best.values()]
    fit = {}
    if len(xs) >= 2:
        a, b_exp = fit_power_law(xs, ys)
        fit = {"a": a, "b": b_exp, "form": f"lr_scale = {a:.4g} * node_activations^{b_exp:.4g}"}
        print(f"\nFitted: {fit['form']}")
        print("Comparison at swept points (fitted vs current clip(2000/x, 0.5, 3.0)):")
        for x, y in sorted(zip(xs, ys)):
            fitted_y = a * (x ** b_exp)
            current_y = current_formula(x)
            print(f"  node_act={x:8.1f}  actual_best={y:.3f}  fitted={fitted_y:.3f}  current_formula={current_y:.3f}")
    results["fit"] = fit
    results["phase1_best"] = [{"max_x": k[0], "sample_size": k[1], **v} for k, v in phase1_best.items()]

    # ── Phase 2: joint grad_clip check at low/high capacity points ────────
    if phase1_best:
        sorted_points = sorted(phase1_best.items(), key=lambda kv: kv[1]["node_activations"])
        check_points = [sorted_points[0], sorted_points[-1]] if len(sorted_points) > 1 else sorted_points
        print(f"\n=== Phase 2: joint grad_clip check at {len(check_points)} points "
              f"x {len(LR_OFFSET_FACTORS)} LR offsets x {len(GRAD_CLIP_MULTIPLIERS)} grad_clip multipliers ===")
        for (max_x, n), best in check_points:
            df = full_df.sample(n=n, random_state=42).reset_index(drop=True)
            base_lr = best["lr_scale"]
            for lr_factor in LR_OFFSET_FACTORS:
                lr = base_lr * lr_factor
                for gc_mult in GRAD_CLIP_MULTIPLIERS:
                    t0 = time.time()
                    r = run_one(df, max_x, EPOCH_COUNT, lr, gc_mult, logger)
                    dt = time.time() - t0
                    entry = {"max_x": max_x, "sample_size": n, "lr_scale": lr,
                             "grad_clip_multiplier": gc_mult, "r2": r["r2"],
                             "n_train": r["n_train"], "wall_time_sec": round(dt, 1)}
                    results["phase2"].append(entry)
                    print(f"  max_x={max_x:3d} n={n:4d} lr={lr:5.3f} gc_mult={gc_mult:4.2f}  "
                          f"->  R2={r['r2']:.4f}  ({dt:.1f}s)")
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2, default=str)

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== DONE (total elapsed {results['elapsed_sec']:.0f}s) ===")
    print(f"Results written -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
