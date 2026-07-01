import comparisons.ComparisonManager as CompMgr
import Components.NotebookLogger as LoggerModule
import Components.RichConsole as RichConsole
import time
import pandas as pd
logger = RichConsole.RichLogger(log_level=4, console_level=4, filename=f"segment_analysis_notebook_{int(time.time())}.log")

manager = CompMgr.ComparisonManager(logger=logger)
dataset = pd.read_csv("Exam_Score_Prediction.csv")
target  = "exam_score"


JUDGE_ITERS          = 20
JUDGE_MIN_CLUSTERS   = 4
JUDGE_MAX_CLUSTERS   = 20
SELECTION_PERCENTAGE = 0.5
TRAINING_MODE        = "partitioned"

manager.run_all(
    dataset                     = dataset,
    target                      = target,
    epoch_count                 = 20,

    # ── Classical baselines ────────────────────────────────────────────────
    run_linear                  = True,
    run_knn                     = True,
    run_rf                      = True,
    run_xgb                     = True,
    run_mlp                     = True,
    run_cnn                     = True,

    # ── Single-segment Nexus (isolated segment baseline) ──────────────────
    # Runs the segment handler alone (no JudgeNode/HandlerNode) at two
    # graph sizes so the multi-segment system can be compared against a
    # single-segment configuration of similar node count.
    run_segment                 = True,
    segment_max_x               = [5, 10, 15, 20, 25, 30, 35],

    # ── Full multi-segment Nexus system ───────────────────────────────────
    # Runs SystemHandler with the same settings used in this notebook demo.
    # All three aggregation modes are tested to show their relative effect.
    run_system                  = True,
    system_max_x                = [10, 15, 20, 25, 30],
    system_dimensions           = 2,
    system_training_mode        = TRAINING_MODE,
    system_judge_iterations     = JUDGE_ITERS,
    system_judge_min_clusters   = JUDGE_MIN_CLUSTERS,
    system_judge_max_clusters   = JUDGE_MAX_CLUSTERS,
    system_aggregation_modes    = ["bma", "simple_mean", "relevance_weighted"],
    system_selection_percentage = SELECTION_PERCENTAGE,
)
