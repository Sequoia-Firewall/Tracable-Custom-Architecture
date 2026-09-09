"""One-off, non-interactive smoke run to populate real artifacts (.nexseg,
judge_node.judgestate, log files, trace.jsonl, error-epoch.csv) so the viz
tool can be tested end-to-end. Small scale on purpose — this is only for
exercising the viewer, not a real training run."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from Settings import Settings
from SystemHandler import SystemHandler
import Components.RichConsole as RC

settings = Settings("settings.json")
settings.override("model.max_x", 6)
settings.override("model.dimensions", 2)
settings.override("training.epoch_count", 4)
settings.override("training.judge_iterations", 5)
settings.override("training.judge_min_clusters", 4)
settings.override("training.judge_max_clusters", 4)

logger = RC.RichLogger(filename=f"smoke_{int(time.time())}.log", log_level=4, console_level=1)

d = settings.dataset
dataset = pd.read_csv(d["csv_path"]).sample(n=300, random_state=42).reset_index(drop=True)

system = SystemHandler.from_settings(settings, logger)
system.initializeAllSegments(Loud=False, visualization_enabled=False)

t = settings.training
system.train(dataset,
              epoch_count=t["epoch_count"],
              judge_iterations=t["judge_iterations"],
              loud=True,
              judge_min_clusters=t.get("judge_min_clusters"),
              judge_max_clusters=t.get("judge_max_clusters"),
              lr_scale_cfg=t.get("learning_rate"),
              prediction_range_cfg=d.get("prediction_range"),
              grad_clip_cfg=t.get("grad_clip"),
              delta_clip_cfg=t.get("delta_clip"),
              reconnect_pct=t.get("reconnect_pct", 0.005),
              position_momentum=t.get("position_momentum", 0.0),
              feature_pruning_enabled=t.get("feature_pruning_enabled", False))

print("Training complete. Running traced inference calls...")
m = settings.model
for i in range(8):
    sample = dataset.iloc[i].to_dict()
    result = system.runInfer(sample.copy(), loud=False,
                              aggregation_mode=m["aggregation_mode"],
                              selection_percentage=m["selection_percentage"],
                              trace=True, trace_path="trace.jsonl")
    print(i, result)

print("Done.")
