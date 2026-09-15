"""
RandomForest + XGBoost baselines on NSL-KDD, same train sample / official
test split as run_nslkdd_xscale.py, for a fair comparison against TCA's
numbers on this dataset.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import Components.RichConsole as RC
from comparisons.RandomForestModel import RandomForestModel
from comparisons.XGBoostModel import XGBoostModel

COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label", "difficulty",
]

TARGET = "threat"
TRAIN_SAMPLE_SIZE = 40000
CHECKPOINT_PATH = "nslkdd_baselines_report.json"


def load(path):
    df = pd.read_csv(path, header=None, names=COLS)
    df["threat"] = (df["label"] != "normal").astype(int)
    return df.drop(columns=["label", "difficulty"])


def main():
    t_start = time.time()
    print("=== Loading NSL-KDD (standard train/test split, same sample seed as xscale test) ===")
    train_full = load("../datasets/nslkdd/KDDTrain+.txt")
    test_full = load("../datasets/nslkdd/KDDTest+.txt")
    train_df = train_full.sample(n=min(TRAIN_SAMPLE_SIZE, len(train_full)), random_state=42).reset_index(drop=True)

    # RandomForest/XGBoost need numeric-only features -- one-hot the three
    # symbolic columns (protocol_type, service, flag) exactly like TCA's own
    # PreProcessingNode does internally, so both sides see equivalent input.
    cat_cols = ["protocol_type", "service", "flag"]
    combined = pd.concat([train_df, test_full], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=cat_cols)
    train_enc = combined.xs("train")
    test_enc = combined.xs("test")

    feature_cols = [c for c in train_enc.columns if c != TARGET]
    train_records = train_enc.to_dict(orient="records")
    test_records = test_enc.to_dict(orient="records")
    print(f"train={len(train_records)}  test={len(test_records)}  n_features={len(feature_cols)}")

    logger = RC.RichLogger(filename=f"nslkdd_baselines_{int(time.time())}.log", log_level=4, console_level=2)

    results = {}

    print("\n=== RandomForest ===")
    rf = RandomForestModel(n_estimators=100, random_state=42)
    rf_metrics = rf.run(train_records, test_records, feature_cols, TARGET, logger=logger)
    results["RandomForest"] = rf_metrics
    print(f"RandomForest: R2={rf_metrics['r2']:.4f}  precision={rf_metrics['precision']:.3f}  "
          f"recall={rf_metrics['recall']:.3f}  f1={rf_metrics['f1']:.3f}  train_time={rf_metrics['train_time_sec']:.1f}s")

    print("\n=== XGBoost ===")
    xgb = XGBoostModel(n_estimators=300, random_state=42)
    xgb_metrics = xgb.run(train_records, test_records, feature_cols, TARGET, logger=logger)
    results["XGBoost"] = xgb_metrics
    print(f"XGBoost: R2={xgb_metrics['r2']:.4f}  precision={xgb_metrics['precision']:.3f}  "
          f"recall={xgb_metrics['recall']:.3f}  f1={xgb_metrics['f1']:.3f}  train_time={xgb_metrics['train_time_sec']:.1f}s")

    results["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Summary ===")
    print(f"RandomForest: R2={rf_metrics['r2']:.4f}  F1={rf_metrics['f1']:.3f}  time={rf_metrics['train_time_sec']:.1f}s")
    print(f"XGBoost:      R2={xgb_metrics['r2']:.4f}  F1={xgb_metrics['f1']:.3f}  time={xgb_metrics['train_time_sec']:.1f}s")
    print(f"\nTotal elapsed: {results['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
