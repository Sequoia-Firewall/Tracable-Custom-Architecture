import comparisons.ComparisonManager as CompMgr
import Components.NotebookLogger as LoggerModule
import Components.RichConsole as RichConsole
import time
import pandas as pd
logger = RichConsole.RichLogger(log_level=4, console_level=4, filename=f"segment_analysis_notebook_{int(time.time())}.log")

manager = CompMgr.ComparisonManager(logger=logger)
dataset = pd.read_csv("Exam_Score_Prediction.csv")
target  = "exam_score"
epochs  = 20
max_x   = [5, 10, 15, 20, 25, 30, 35, 40]

# Dedicated output file + run_system=False: this notebook only evaluates the
# standalone SegmentHandler, so it must never write into comparison_results.csv
# (the file v12Analysis.ipynb reads for the full SystemHandler comparison) and
# must never spend hours retraining the system. output_csv also changes the
# checkpoint job-hash, so this run can't pick up a stale/incompatible checkpoint
# from a previous (pre-leak-fix) run of this same cell.
manager.run_all(
    dataset        = dataset,
    target         = target,
    epoch_count    = epochs,
    segment_max_x  = max_x,
    run_segment    = True,
    run_cnn        = True,
    run_linear     = True,
    run_knn        = True,
    run_rf         = True,
    run_xgb        = True,
    run_mlp        = True,
    run_system     = False,
    output_csv     = "segment_comparison_results.csv",
)



