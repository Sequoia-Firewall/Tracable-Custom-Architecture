"""Populate trace.jsonl with a diverse batch of real inference calls so the
viz tool's Confidence/Signal Paths tabs have data spanning all 4 segments,
not just whichever one or two a single earlier test query happened to hit."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from collections import Counter
from Settings import Settings
from SystemHandler import SystemHandler
import Components.RichConsole as RC

settings = Settings("settings.json")
logger = RC.RichLogger(filename=f"trace_batch_{int(time.time())}.log", log_level=0, console_level=5)

d, m = settings.dataset, settings.model
dataset = pd.read_csv(d["csv_path"]).sample(n=60, random_state=7).reset_index(drop=True)

system = SystemHandler.from_settings(settings, logger)
system.load_segments(".")

selected_counter = Counter()
for i in range(len(dataset)):
    sample = {k: v for k, v in dataset.iloc[i].to_dict().items() if k != d["target_column"]}
    result = system.runInfer(sample, loud=False, aggregation_mode=m["aggregation_mode"],
                              selection_percentage=m["selection_percentage"],
                              trace=True, trace_path="trace.jsonl")
    if result:
        selected_counter[result["segment_id"]] += 1

print(f"Ran {len(dataset)} traced queries. Dominant-segment distribution: {dict(selected_counter)}")
