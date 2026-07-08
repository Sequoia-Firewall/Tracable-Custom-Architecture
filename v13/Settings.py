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
