"""
Settings.py — configuration loader for DragonChild v13.

Reads settings.json, fills in any missing keys with defaults, and exposes
typed section accessors so the rest of the codebase can do:

    s = Settings()
    s.model["max_x"]          # int
    s.training["epoch_count"] # int
    s.run_mode                # str
"""
import json
import os

_DEFAULTS = {
    "run_mode": "train",
    "dataset": {
        "csv_path": "Exam_Score_Prediction.csv",
        "target_column": "exam_score",
        "ignored_columns": [],
        "shuffle": True,
        "prediction_range": {
            "mode": "auto",       # "auto" (scan dataset target column) or "manual"
            "min_value": None,    # used when mode == "manual"
            "max_value": None,    # used when mode == "manual"
        },
    },
    "model": {
        "max_x": 20,
        "dimensions": 2,
        "connection_percentage": 0.1,
        "density": 0.8,
        "training_mode": "partitioned",
        "aggregation_mode": "bma",
        "selection_percentage": 0.5,
    },
    "training": {
        "epoch_count": 20,
        "judge_iterations": 20,
        "judge_min_clusters": 4,
        "judge_max_clusters": 20,
        "test_split": 0.2,
        "learning_rate": {
            "mode": "auto",        # "auto" | "manual-scale" | "manual-full"
            "min_lr_scale": 0.5,   # auto: clamp FLOOR for the row-count formula (2000/train_rows)
                                    # manual-scale: literal min endpoint of the decay curve
                                    # manual-full: unused
            "max_lr_scale": 3.0,   # auto: clamp CEILING for the row-count formula
                                    # manual-scale: literal max endpoint (decay solved from epoch_count)
                                    # manual-full: literal starting scale (decay set explicitly below)
            "decay": None,         # manual-full only: literal per-epoch decay factor (e.g. 0.85)
                                    # ignored / auto-solved in "auto" and "manual-scale"
        },
        "grad_clip": {
            "mode": "auto",   # "auto" (derive from dataset.prediction_range span) or "manual"
            "value": None,    # used when mode == "manual"
        },
        "delta_clip": {
            "mode": "auto",   # "auto" (derive from segment hop topology + prediction range span) or "manual"
            "value": None,    # used when mode == "manual"
        },
        "reconnect_pct": 0.005,  # fraction of an epoch's samples between mid-epoch
                                  # topology reconnects (0 < value <= 1). A node's
                                  # connected_nodes list is only refreshed every
                                  # round(reconnect_pct * train_rows) samples instead
                                  # of every sample. Set to 0 to reconnect after every
                                  # sample (pre-throttling behavior) — useful for
                                  # isolating throttling's effect in an ablation test.
        "position_momentum": 0.0,  # EMA coefficient for position-gradient steps
                                     # (velocity = position_momentum * velocity + gradient).
                                     # 0.0 (default) reduces to the original raw-gradient
                                     # step exactly. Opt-in — 1.1.3 ablation testing found
                                     # it helps mid-size graphs but hurts small ones (max_x=5),
                                     # so it's off by default rather than baked in.
        "feature_pruning_enabled": False,  # experimental: JudgeNode screens features by
                                     # cluster relevance, each segment confirms candidates
                                     # against real learned weight magnitude before
                                     # freezing (stop updating) then removing (drop from
                                     # the forward pass) any of them. SystemHandler /
                                     # training_mode='partitioned' only — no effect on
                                     # SegmentHandler or training_mode='full' (neither
                                     # uses JudgeNode clustering). Off by default: a
                                     # full-scale ablation on this dataset never found a
                                     # feature worth pruning at the default threshold
                                     # (SegmentHandler.FEATURE_FREEZE_WEIGHT_THRESHOLD) —
                                     # may behave differently on larger/higher-dimensional
                                     # datasets with more genuinely redundant columns.
    },
    "logging": {
        "log_level": 4,
        "console_level": 4,
        "filename_prefix": "v13_run",
    },
    "output": {
        "save_pretrain_graph": True,
        "pretrain_graph_path": "nexus_pretrain.png",
        "save_posttrain_graph": True,
        "posttrain_graph_path": "nexus_posttrain.png",
        # Per-segment structure graphs: one PNG at initializeSegment() and one
        # more per training epoch (dimensions==2 only). Off by default — the
        # per-epoch renders run on every SegmentHandler.train() call and are
        # pure debugging/visualization output, not needed for correctness.
        "visualization_enabled": False,
    },
    "infer": {
        "nexseg_dir": ".",
        "sample_index": 0,
        "loud": True,
    },
    "comparison": {
        "run_segment": True,
        "run_system": True,
        "run_linear": True,
        "run_knn": True,
        "run_rf": True,
        "run_xgb": True,
        "run_mlp": True,
        "run_cnn": False,
        "segment_max_x": [5, 10, 15, 20, 25],
        "system_max_x": [5, 10, 15, 20, 25],
        "epoch_count": 20,
        "output_csv": "comparison_results.csv",
    },
}

_VALID_MODES         = {"train", "infer", "compare"}
_VALID_TRAINING_MODE = {"partitioned", "full"}
_VALID_AGG_MODE      = {"bma", "simple_mean", "relevance_weighted"}
_VALID_PRED_RANGE_MODE = {"auto", "manual"}
_VALID_GRAD_CLIP_MODE = {"auto", "manual"}
_VALID_DELTA_CLIP_MODE = {"auto", "manual"}
_VALID_LR_MODE = {"auto", "manual-scale", "manual-full"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on conflicts."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


class Settings:
    def __init__(self, path: str = "settings.json"):
        self.path = path
        self._data: dict = {}
        self.load()

    # ── I/O ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load settings from file, filling missing keys with defaults."""
        import copy
        merged = copy.deepcopy(_DEFAULTS)
        if os.path.exists(self.path):
            with open(self.path) as f:
                user = json.load(f)
            _deep_merge(merged, user)
        self._data = merged
        self._validate()

    def save(self) -> None:
        """Write current settings back to file."""
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def override(self, key_path: str, value) -> None:
        """Set a nested key using dot notation, e.g. 'model.max_x'."""
        keys = key_path.split(".")
        node = self._data
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value

    # ── Validation ───────────────────────────────────────────────────────

    def _validate(self) -> None:
        mode = self._data.get("run_mode", "train")
        if mode not in _VALID_MODES:
            raise ValueError(f"settings.json: run_mode '{mode}' not in {_VALID_MODES}")

        tm = self._data["model"].get("training_mode", "partitioned")
        if tm not in _VALID_TRAINING_MODE:
            raise ValueError(f"settings.json: model.training_mode '{tm}' not in {_VALID_TRAINING_MODE}")

        am = self._data["model"].get("aggregation_mode", "bma")
        if am not in _VALID_AGG_MODE:
            raise ValueError(f"settings.json: model.aggregation_mode '{am}' not in {_VALID_AGG_MODE}")

        if not os.path.exists(self._data["dataset"]["csv_path"]):
            # Warn rather than crash — path may be valid at runtime from a different cwd
            pass

        lr = self._data["training"].get("learning_rate", {})
        lr_mode = lr.get("mode", "auto")
        if lr_mode not in _VALID_LR_MODE:
            raise ValueError(f"settings.json: training.learning_rate.mode '{lr_mode}' not in {_VALID_LR_MODE}")
        if lr_mode in ("auto", "manual-scale"):
            if lr.get("min_lr_scale", 0) > lr.get("max_lr_scale", 0):
                raise ValueError(
                    "settings.json: training.learning_rate.min_lr_scale must be <= max_lr_scale"
                )
        if lr_mode == "manual-full":
            if lr.get("decay") is None:
                raise ValueError(
                    "settings.json: training.learning_rate.mode is 'manual-full' but decay is not set"
                )
            if not (0 < lr["decay"] <= 1):
                raise ValueError("settings.json: training.learning_rate.decay must be in (0, 1]")
            if lr.get("max_lr_scale") is None or lr["max_lr_scale"] <= 0:
                raise ValueError("settings.json: training.learning_rate.max_lr_scale must be > 0")

        reconnect_pct = self._data["training"].get("reconnect_pct", 0.005)
        if reconnect_pct < 0 or reconnect_pct > 1:
            raise ValueError("settings.json: training.reconnect_pct must be in [0, 1]")

        position_momentum = self._data["training"].get("position_momentum", 0.0)
        if position_momentum < 0 or position_momentum >= 1:
            raise ValueError("settings.json: training.position_momentum must be in [0, 1)")

        pr = self._data["dataset"].get("prediction_range", {})
        pr_mode = pr.get("mode", "auto")
        if pr_mode not in _VALID_PRED_RANGE_MODE:
            raise ValueError(f"settings.json: dataset.prediction_range.mode '{pr_mode}' not in {_VALID_PRED_RANGE_MODE}")
        if pr_mode == "manual":
            if pr.get("min_value") is None or pr.get("max_value") is None:
                raise ValueError(
                    "settings.json: dataset.prediction_range.mode is 'manual' but "
                    "min_value/max_value are not both set"
                )
            if pr["min_value"] > pr["max_value"]:
                raise ValueError("settings.json: dataset.prediction_range.min_value must be <= max_value")

        gc = self._data["training"].get("grad_clip", {})
        gc_mode = gc.get("mode", "auto")
        if gc_mode not in _VALID_GRAD_CLIP_MODE:
            raise ValueError(f"settings.json: training.grad_clip.mode '{gc_mode}' not in {_VALID_GRAD_CLIP_MODE}")
        if gc_mode == "manual":
            if gc.get("value") is None:
                raise ValueError(
                    "settings.json: training.grad_clip.mode is 'manual' but value is not set"
                )
            if gc["value"] <= 0:
                raise ValueError("settings.json: training.grad_clip.value must be > 0")

        dc = self._data["training"].get("delta_clip", {})
        dc_mode = dc.get("mode", "auto")
        if dc_mode not in _VALID_DELTA_CLIP_MODE:
            raise ValueError(f"settings.json: training.delta_clip.mode '{dc_mode}' not in {_VALID_DELTA_CLIP_MODE}")
        if dc_mode == "manual":
            if dc.get("value") is None:
                raise ValueError(
                    "settings.json: training.delta_clip.mode is 'manual' but value is not set"
                )
            if dc["value"] <= 0:
                raise ValueError("settings.json: training.delta_clip.value must be > 0")

    # ── Section accessors ────────────────────────────────────────────────

    @property
    def run_mode(self) -> str:
        return self._data["run_mode"]

    @property
    def dataset(self) -> dict:
        return self._data["dataset"]

    @property
    def model(self) -> dict:
        return self._data["model"]

    @property
    def training(self) -> dict:
        return self._data["training"]

    @property
    def logging(self) -> dict:
        return self._data["logging"]

    @property
    def output(self) -> dict:
        return self._data["output"]

    @property
    def infer(self) -> dict:
        return self._data["infer"]

    @property
    def comparison(self) -> dict:
        return self._data["comparison"]

    # ── Display ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"  run_mode          : {self.run_mode}"]
        lines += [f"  dataset.csv       : {self.dataset['csv_path']}"]
        lines += [f"  dataset.target    : {self.dataset['target_column']}"]
        lines += [f"  model.max_x       : {self.model['max_x']}"]
        lines += [f"  model.dimensions  : {self.model['dimensions']}"]
        lines += [f"  model.agg_mode    : {self.model['aggregation_mode']}"]
        lines += [f"  model.train_mode  : {self.model['training_mode']}"]
        lines += [f"  training.epochs   : {self.training['epoch_count']}"]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Settings(path={self.path!r}, run_mode={self.run_mode!r})"
