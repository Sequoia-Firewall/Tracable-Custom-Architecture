# YearPredictionMSD

A large-scale regression benchmark, pulled for testing TCA at a scale well
beyond the ~20,000-row `Exam_Score_Prediction.csv` dataset used everywhere
else in this repo.

**The data file itself (`YearPredictionMSD.txt`, ~428MB) is intentionally
NOT committed to git** — it's listed in `.gitignore`. Re-download it with
the command below before using it.

## Source

- UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/203/yearpredictionmsd>
- Direct download (no auth/API key required):
  ```bash
  curl -sL -o yearpredictionmsd.zip https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip
  unzip yearpredictionmsd.zip -d .
  rm yearpredictionmsd.zip
  ```
- Subset of the Million Song Dataset (Bertin-Mahieux et al., 2011). Cite the
  Million Song Dataset paper if this is used in any published work — see the
  UCI page for the full citation.

## What it is

- **Task**: regression — predict a song's release year from audio features.
- **Rows**: 515,345 (confirmed by direct count after download).
- **Columns**: 91 total, comma-separated, **no header row**.
  - Column 1: target — release year (integer, observed range 1922-2011).
  - Columns 2-91: 90 numeric audio features — 12 timbre averages followed
    by 78 timbre covariances, extracted from the Echo Nest analysis of each
    track.
- **File size**: ~428MB uncompressed (`YearPredictionMSD.txt`), ~211MB
  zipped.

## Recommended train/test split

The UCI page specifies a **fixed** split to keep results comparable across
papers/experiments: the **first 463,715 examples are train**, the
**last 51,630 are test**. Do NOT shuffle before splitting if comparing
against published benchmarks — artist overlap between train/test is
avoided only by respecting this exact boundary.

## Using it with TCA

This is plain CSV with no header and the target in column 1 (TCA's
dataset loader expects the target as a named column) — load it with
pandas and assign column names before handing it to TCA's preprocessing:

```python
import pandas as pd
cols = ["year"] + [f"feat_{i}" for i in range(90)]
df = pd.read_csv("YearPredictionMSD.txt", header=None, names=cols)
```

At 515k rows and 90 features, expect training wall-clock to be
**substantially** longer than anything run against the exam-score dataset
so far — start with a small random sample (a few thousand rows) to smoke-test
config before committing to a full-scale run, same discipline used
throughout this project's other large-scale tests.
