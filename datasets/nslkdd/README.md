# NSL-KDD

Classic labeled network intrusion detection benchmark, pulled to derisk
whether TCA has any usable signal for a threat-detection use case before
investing in anything more ambitious (localization, temporal patterns,
calibrated confidence -- see the conversation this came from).

**Committed directly** (unlike `datasets/yearpredictionmsd/`, which is
gitignored for size) — `KDDTrain+.txt` + `KDDTest+.txt` together are only
~22.5MB, well within a reasonable commit.

## Source

- Mirror used (direct download, no auth): <https://github.com/defcom17/NSL_KDD>
- Canonical dataset page: <https://www.unb.ca/cic/datasets/nsl.html> (University of New Brunswick)
- Improved version of the original 1999 KDD Cup dataset — removes duplicate
  records and rebalances difficulty, standard benchmark in intrusion-detection
  ML literature so RF/XGBoost comparisons here are literature-comparable.

## What it is

- **Task (as pulled)**: multi-class — `label` names the specific attack type
  (`normal`, `neptune`, `satan`, `ipsweep`, `smurf`, `portsweep`, `nmap`,
  `back`, `teardrop`, `warezclient`, ... long tail of rare types).
- **Rows**: 125,973 train / 22,544 test (fixed split, matches the file split
  as distributed — don't reshuffle across files if comparing to published
  benchmarks).
- **Columns**: 43 total, no header row in the raw `.txt` files — `Field
  Names.csv` lists the 41 feature names + types (continuous/symbolic); the
  raw files append `label` (attack type string) and `difficulty` (an
  original-paper classification-difficulty score, not a feature) as columns
  42-43.
- **Feature mix**: continuous (e.g. `duration`, `src_bytes`, `dst_bytes`,
  various `*_rate` fields) and symbolic/categorical (`protocol_type`,
  `service`, `flag`) — TCA's existing PreProcessingNode already one-hot
  encodes categoricals automatically, same as it does for exam-score's
  categorical columns.
- **Label balance**: ~53.5% `normal` in train, rest split across attack
  types with a long tail (many attack types have well under 100 examples) —
  genuinely imbalanced once you look past the binary normal/attack split,
  worth keeping in mind given JudgeNode's k-means clustering has no special
  handling for rare classes.

## Target construction for TCA (binary threat framing)

TCA's native mode is regression, not classification, so `label` gets
collapsed to a binary `threat` column (0 = normal, 1 = any attack type)
rather than trying to regress toward a multi-class target -- this matches
the actual ask ("gauge if there's a threat"), sidesteps the severe
multi-class imbalance in the attack-type tail, and keeps the evaluation
methodology identical to every other TCA test this session (R2/MAE against
a 0/1 -- or optionally 0/100-scaled -- continuous target, thresholded for
accuracy/precision/recall). `difficulty` is dropped (not a feature, an
artifact of the original paper's own evaluation methodology).

## Using it with TCA

```python
import pandas as pd
cols = [
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
df = pd.read_csv("KDDTrain+.txt", header=None, names=cols)
df["threat"] = (df["label"] != "normal").astype(int)
df = df.drop(columns=["label", "difficulty"])
```
