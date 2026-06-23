# DragonChild

## Abstract

DragonChild is a research project exploring whether a neural network organized like a **map** — instead of like a stack of layers — can make predictions that are both accurate and easy to explain.

Most neural networks are black boxes: data goes in, a prediction comes out, and the "why" is buried in thousands of opaque weights. DragonChild instead places nodes at positions in a coordinate space and routes data between them based on geometric distance, the way a signal might hop across a network of relay stations. Along the way, each step tracks its own uncertainty, so the system can say "I'm confident" or "I'm not sure" rather than just guessing. The result is a model whose internal structure can be visualized, traced, and audited step by step.

The project is tested against a concrete benchmark — predicting student exam scores from a tabular dataset — and compared head-to-head against standard machine learning models (linear regression, random forest, gradient boosting, a small MLP, and a CNN) to see whether the added structure is actually worth its cost. The current version (v12 and its evaluation suite) does not yet outperform a plain linear regression on this benchmark, but it shows the architecture's strengths — interpretability, modularity, and built-in uncertainty — clearly enough to be a promising foundation for tasks where those properties matter more than squeezing out the last bit of accuracy (e.g. anomaly or intrusion detection, sketched out in `v12/simulatedattackplan.txt`).

**Status:** `v11` and `v12` (plus `v12 analysis`) are the current, actively developed line. Everything before `v11` (`v1`–`v10`, `v10Experimental`) is historical and kept for reference only.

**License:** [MIT](LICENSE)

## Setup

```bash
pip install -r requirements.txt
```

Run a model directly (from inside `v12/` or `v12 analysis/`):

```python
from SystemHandler import SystemHandler
# or, for a single segment in isolation:
from SegmentHandler import SegmentHandler
```

Or open the evaluation notebooks in `v12 analysis/` (`v12Analysis.ipynb`, `SegmentHandlerAnalysis.ipynb`) with Jupyter to reproduce the benchmark comparisons.

---

## How it works

v12 is the current reference implementation. A prediction flows through five stages:

1. **PreProcessingNode** — tokenizes/normalizes the raw dataset.
2. **JudgeNode** — k-means clusters the input space into segments and routes each sample to the most relevant segment(s).
3. **SplitterNode → ProcessingNode** — within a segment, signals propagate stochastically between geometrically-positioned processing nodes, each applying a learned weight and accumulating variance.
4. **ReviewerNode** — aggregates a segment's signals into one prediction via inverse-variance weighting.
5. **HandlerNode** — combines all active segments' predictions into the final output, via `bma`, `simple_mean`, or `relevance_weighted` aggregation.

`SegmentHandler` exercises steps 3–4 alone (one segment, no routing); `SystemHandler` runs the full pipeline.

## Analysis

The `v12 analysis/` folder is the benchmarking and evaluation harness for v12 and is the source of truth for any performance claims:

- **`v12Analysis.ipynb`** — full `SystemHandler` study against the sklearn/CNN baselines, including a "Claude analysis of results" section covering headline comparisons, an ablation by `max_x`, aggregation-mode comparison, and inference-cost scaling.
- **`SegmentHandlerAnalysis.ipynb`** — standalone `SegmentHandler` study. Its cell 62 narrative still reflects pre-leak-fix (v10-era) numbers and needs to be rerun and rewritten; the notebook was recently made safe to rerun in isolation (separate `output_csv`/checkpoint namespace via `segment_comparison_results.csv`, `run_system=False`) without disturbing `comparison_results.csv` or the System Handler's cached run.
- **`comparison_results.csv`** — the shared results table both notebooks read from/append to.

## Strengths and weaknesses

**Strengths**
- Interpretable, inspectable topology — node positions, signal paths, and variance are all traceable, unlike a typical NN's weight matrices.
- Built-in uncertainty quantification: high inter-reviewer variance is a usable "I'm not sure" signal, not just a wrong-but-confident prediction.
- Unsupervised regime discovery via `JudgeNode` k-means means segments specialize without manual labeling.
- Modular by construction, which makes it a plausible fit for tasks that want per-regime experts plus a confidence signal (e.g. anomaly/intrusion detection, as sketched in `v12/simulatedattackplan.txt`).

**Weaknesses**
- On the exam-score benchmark, the best `SystemHandler` configuration (MAE ≈ 9.3, R² ≈ 0.64) still trails plain Linear Regression (MAE ≈ 7.97, R² ≈ 0.73) and MLP — the architecture has not yet beaten simple baselines on this task.
- Training cost is dramatically higher than the sklearn baselines (hours vs. seconds), which is a hard sell unless the interpretability/uncertainty properties are specifically needed.
- `JudgeNode`'s clustering has no fixed random seed, so repeated runs show real variance in train time and results — a reproducibility gap relative to the rest of the codebase, which is seeded.
- The codebase has a history of subtle correctness bugs that quietly inflate results (the v10 target-leak below being the clearest example), which argues for treating any single notebook run's numbers with caution until cross-checked.

## Evolution (detailed history)

| Version | Summary |
|---|---|
| **v1 – v2** *(historical)* | Origin as "BrainNexus": a general spatially-organized neural network with LLM-style tokenization (Mistral integration) and flat-text training. Exploratory, not yet tied to a specific prediction task. |
| **v3 – v4** *(historical)* | Introduced `BrainSegment` (the segment concept) and modular/reinforcement-learning training guides. Added hypercube positioning and dynamic node allocation experiments. Complexity grew faster than stability here. |
| **v5** *(historical)* | First architectural refactor into the role-based node types that define the project since: `Judge`, `Splitter`, `Computational` (processing), `Reviewer`, `Handler`, under a `Nodes/` package, plus pluggable `LearningMethods` (supervised, reinforcement, classification). |
| **v6** *(historical)* | Architecture re-targeted at a concrete benchmark task: **exam score prediction**. `plan.txt` lays out the Judge → Splitter → Processing → Reviewer → Handler pipeline and the `Signal` data structure for the first time in this form. |
| **v7 / v7.1 / v7.2** *(historical)* | Formalized the math: Judge Node computes segment relevance, Splitter computes feature relevance and routes signals by Euclidean distance, Processing nodes apply a Bayesian-style update and forward signals stochastically (distance-weighted random routing), Reviewer/Handler aggregate without learning. Added experiment-tracking (`stats/`, `logs.txt`). |
| **v8 – v9** *(historical)* | Migrated to the current `Components/` package structure (one class per file) with visualizations of the generated node graph (`nexus_structure.png`) and a dedicated `Train.py`. Still single-system, no baseline comparison or standalone segment evaluation yet. |
| **v10 / v10Experimental / v10 analysis** *(historical)* | Split the system into **`SegmentHandler`** (train/evaluate one segment in isolation) and **`SystemHandler`** (full Judge-routed, multi-segment system), and introduced the `comparisons/` benchmarking framework (`ComparisonManager`) to score the Nexus against Linear Regression, KNN, Random Forest, Gradient Boosting, MLP, and a CNN baseline on the same data. **Contained a target-leakage bug**: `SegmentHandler` passed the full preprocessed row — including the target column — into node weight initialization, letting a node learn the target's own column directly. This made v10's standalone segment results look artificially strong. |
| **v11** *(current)* | First fully-wired System Handler with `JudgeNode` k-means clustering for unsupervised segment/cluster discovery and BMA (Bayesian model averaging) aggregation across segments, with a rich `Reports.txt` output (MAE 11.86, R² 0.373 on the exam-score task). |
| **v12** *(current)* | Same System Handler design as v11, **with the v10 target-leakage bug fixed** (`SegmentHandler` now strips the target column before the forward pass). Modest accuracy improvement over v11 (MAE 10.26, R² 0.546, see `v12/v12 reports.txt`). Also includes `v12/simulatedattackplan.txt`, a forward-looking design doc for repurposing the architecture as an adversarial network-intrusion detector (not yet implemented). |
| **v12 analysis** *(current)* | A self-contained analysis copy of v12 (own `Components/`, `SegmentHandler.py`, `SystemHandler.py`, `comparisons/`) used purely for evaluation — see Analysis above. |

## Other folders

- `v6`–`v9`, `v10Experimental` — historical/exploratory snapshots, kept for reference, not under active development.
- `brainnexus_workspace/` — early workspace artifacts from the v1–v4 "BrainNexus" naming era.
