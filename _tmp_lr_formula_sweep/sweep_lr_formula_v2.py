"""
_tmp_lr_formula_sweep/sweep_lr_formula_v2.py
------------------------------------------------
Follow-up to sweep_lr_formula.py. That first pass found every max_x=10/15
capacity point picked the SMALLEST LR candidate (0.25) as best -- the floor
of the tested range -- meaning the true optimum was never actually located.
This version:

  1. Widens LR_CANDIDATES down to 0.02 (12x lower than before) so the
     bottom can actually be found for deep graphs, while still covering
     the higher range max_x=5 preferred (up to 3.0).
  2. Repeats each (capacity, lr) cell N_REPEATS times and reports
     mean/stdev/stderr, so "best LR" is a statistically grounded claim
     (a candidate has to beat its neighbors by more than noise) rather
     than a single noisy trial. Repeats are NOT re-seeded -- SegmentHandler
     training draws from one continuously-advancing global random.random()
     stream (random.seed(42) fires once at ProcessingNode.py's import
     time), so successive run_one() calls on the same data naturally see
     different stochastic routing/init -- exactly the run-to-run training
     variance we want to characterize, without conflating it with
     different held-out data per repeat.

Code-correctness note (see conversation): traced apply_weight_gradient /
apply_position_gradient / the lr_w=WEIGHT_LR*cur_scale chain in
SegmentHandler._resolve_lr_schedule -- confirmed a larger lr_scale really
does produce a larger effective learning rate everywhere, consistently,
no sign flips. The "backwards" result from sweep v1 (deep graphs wanting
SMALLER lr_scale) has a real structural explanation, not a bug:
ProcessingNode.train_process_signal() scales each node's forward
contribution by 1/(1+distance_to_origin), and accumulate_weight_gradient/
accumulate_position_gradient reuse that same distance to scale gradients.
Nodes further from the splitter (i.e. later-hop nodes in deep graphs) are
therefore already geometrically down-weighted in both directions --
applying a BIGGER lr_scale for deep/diluted graphs (as the current auto
formula does) fights that existing damping instead of complementing it.
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

CAPACITY_GRID = [(max_x, n) for max_x in (5, 10, 15) for n in (300, 800)]
LR_CANDIDATES = [0.02, 0.05, 0.1, 0.2, 0.4, 0.75, 1.5, 3.0]
N_REPEATS = 3

GRAD_CLIP_MULTIPLIERS = [0.5, 1.0, 2.0]
LR_OFFSET_FACTORS = [0.7, 1.0, 1.4]
PHASE2_REPEATS = 2

CSV_PATH = "Exam_Score_Prediction.csv"
CHECKPOINT_PATH = "sweep_v2_results.json"


def run_one(df, max_x, epoch_count, lr_scale, grad_clip_multiplier, logger):
    handler = SegmentHandler(maxX=max_x, target=TARGET, logger=logger,
                             connection_percentage=CONNECTION_PCT, density=DENSITY,
                             dimensions=DIMENSIONS, classification=4, segment_id=0)
    handler.initializeSegment()
    num_nodes = len(handler.segmentComponents['processing_nodes'])
    hops = max(1, math.ceil(num_nodes ** (1.0 / DIMENSIONS)))

    lr_cfg = {"mode": "manual-scale", "max_lr_scale": lr_scale, "min_lr_scale": 0.05 * lr_scale}
    grad_cfg = dict(AUTO_CFG)
    if grad_clip_multiplier != 1.0:
        pred_min, pred_max = PRED_RANGE["min_value"], PRED_RANGE["max_value"]
        base = SegmentHandler._auto_grad_clip(pred_min, pred_max)
        grad_cfg = {"mode": "manual", "value": base * grad_clip_multiplier}

    handler.train(df, epoch_count=epoch_count, lr_scale_cfg=lr_cfg,
                  pred_min=PRED_RANGE["min_value"], pred_max=PRED_RANGE["max_value"],
                  grad_clip_cfg=grad_cfg, delta_clip_cfg=AUTO_CFG,
                  reconnect_pct=0.0, position_momentum=0.0)

    m = handler.best_epoch_metrics or {}
    return {"r2": m.get("r2"), "n_train": m.get("n_train"), "num_nodes": num_nodes, "hops": hops}


def node_activations(n_train, hops, num_nodes):
    return n_train * hops / num_nodes


def mean_std_stderr(values):
    v = [x for x in values if x is not None and not math.isnan(x)]
    if not v:
        return None, None, None
    arr = np.array(v, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    stderr = std / math.sqrt(len(arr)) if len(arr) > 1 else float("nan")
    return mean, std, stderr


def fit_power_law(xs, ys):
    log_x = np.log(np.array(xs, dtype=float))
    log_y = np.log(np.array(ys, dtype=float))
    b, log_a = np.polyfit(log_x, log_y, 1)
    return math.exp(log_a), b


def current_formula(node_act, floor=0.5, ceiling=3.0, reference=2000.0):
    return min(max(reference / node_act, floor), ceiling)


def main():
    t_start = time.time()
    logger = RC.RichLogger(filename=f"sweep_v2_{int(time.time())}.log", log_level=0, console_level=5)
    full_df = pd.read_csv(CSV_PATH).drop(columns=["student_id"])

    results = {"phase1": [], "phase2": []}
    n_runs_total = len(CAPACITY_GRID) * len(LR_CANDIDATES) * N_REPEATS
    print(f"=== Phase 1: {len(CAPACITY_GRID)} capacity points x {len(LR_CANDIDATES)} LR candidates "
          f"x {N_REPEATS} repeats = {n_runs_total} runs ===")

    phase1_best = {}
    for max_x, n in CAPACITY_GRID:
        df = full_df.sample(n=n, random_state=42).reset_index(drop=True)
        cell_summaries = []
        for lr in LR_CANDIDATES:
            trial_r2s, trial_meta = [], None
            for rep in range(N_REPEATS):
                t0 = time.time()
                r = run_one(df, max_x, EPOCH_COUNT, lr, 1.0, logger)
                dt = time.time() - t0
                trial_r2s.append(r["r2"])
                trial_meta = r
                results["phase1"].append({
                    "max_x": max_x, "sample_size": n, "lr_scale": lr, "repeat": rep,
                    "r2": r["r2"], "n_train": r["n_train"], "num_nodes": r["num_nodes"],
                    "hops": r["hops"], "wall_time_sec": round(dt, 1),
                })
            mean, std, stderr = mean_std_stderr(trial_r2s)
            act = node_activations(trial_meta["n_train"], trial_meta["hops"], trial_meta["num_nodes"])
            cell_summaries.append({
                "max_x": max_x, "sample_size": n, "lr_scale": lr, "node_activations": act,
                "r2_mean": mean, "r2_std": std, "r2_stderr": stderr, "r2_trials": trial_r2s,
            })
            print(f"  max_x={max_x:3d} n={n:4d} lr={lr:5.3f}  ->  "
                  f"R2 mean={mean:.4f} std={std:.4f} (trials={[round(x,3) for x in trial_r2s]})  "
                  f"node_act={act:.1f}")
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2, default=str)

        valid = [c for c in cell_summaries if c["r2_mean"] is not None]
        if valid:
            best = max(valid, key=lambda c: c["r2_mean"])
            # Flag whether "best" is statistically distinguishable from the
            # runner-up (means within 1 combined stderr of each other = tied).
            others = [c for c in valid if c is not best]
            if others:
                runner_up = max(others, key=lambda c: c["r2_mean"])
                gap = best["r2_mean"] - runner_up["r2_mean"]
                combined_stderr = math.sqrt(
                    (best["r2_stderr"] or 0) ** 2 + (runner_up["r2_stderr"] or 0) ** 2
                )
                best["distinguishable_from_runner_up"] = bool(
                    combined_stderr > 0 and gap > combined_stderr
                )
                best["runner_up_lr"] = runner_up["lr_scale"]
                best["gap_vs_runner_up"] = gap
            phase1_best[(max_x, n)] = best
            hit_floor = best["lr_scale"] == min(LR_CANDIDATES)
            print(f"  -> best: lr={best['lr_scale']} R2={best['r2_mean']:.4f} "
                  f"node_act={best['node_activations']:.1f} "
                  f"distinguishable={best.get('distinguishable_from_runner_up')} "
                  f"{'*** HIT FLOOR AGAIN ***' if hit_floor else ''}")

    xs = [b["node_activations"] for b in phase1_best.values()]
    ys = [b["lr_scale"] for b in phase1_best.values()]
    fit = {}
    if len(xs) >= 2:
        a, b_exp = fit_power_law(xs, ys)
        fit = {"a": a, "b": b_exp, "form": f"lr_scale = {a:.4g} * node_activations^{b_exp:.4g}"}
        print(f"\nFitted: {fit['form']}")
        for x, y in sorted(zip(xs, ys)):
            print(f"  node_act={x:8.1f}  actual_best={y:.3f}  fitted={a*(x**b_exp):.3f}  "
                  f"current_formula={current_formula(x):.3f}")
    results["fit"] = fit
    results["phase1_best"] = [{"max_x": k[0], "sample_size": k[1], **v} for k, v in phase1_best.items()]

    if phase1_best:
        sorted_points = sorted(phase1_best.items(), key=lambda kv: kv[1]["node_activations"])
        check_points = [sorted_points[0], sorted_points[-1]] if len(sorted_points) > 1 else sorted_points
        print(f"\n=== Phase 2: joint grad_clip check at {len(check_points)} points "
              f"x {len(LR_OFFSET_FACTORS)} LR offsets x {len(GRAD_CLIP_MULTIPLIERS)} grad_clip "
              f"multipliers x {PHASE2_REPEATS} repeats ===")
        for (max_x, n), best in check_points:
            df = full_df.sample(n=n, random_state=42).reset_index(drop=True)
            base_lr = best["lr_scale"]
            for lr_factor in LR_OFFSET_FACTORS:
                lr = base_lr * lr_factor
                for gc_mult in GRAD_CLIP_MULTIPLIERS:
                    trial_r2s = []
                    for rep in range(PHASE2_REPEATS):
                        t0 = time.time()
                        r = run_one(df, max_x, EPOCH_COUNT, lr, gc_mult, logger)
                        dt = time.time() - t0
                        trial_r2s.append(r["r2"])
                        results["phase2"].append({
                            "max_x": max_x, "sample_size": n, "lr_scale": lr,
                            "grad_clip_multiplier": gc_mult, "repeat": rep, "r2": r["r2"],
                            "n_train": r["n_train"], "wall_time_sec": round(dt, 1),
                        })
                    mean, std, _ = mean_std_stderr(trial_r2s)
                    print(f"  max_x={max_x:3d} n={n:4d} lr={lr:5.3f} gc_mult={gc_mult:4.2f}  "
                          f"->  R2 mean={mean:.4f} std={std:.4f}")
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f, indent=2, default=str)

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== DONE (total elapsed {results['elapsed_sec']:.0f}s) ===")


if __name__ == "__main__":
    main()
