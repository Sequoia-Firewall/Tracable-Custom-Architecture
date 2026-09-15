# Code required for the nexus system handler
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Components"))

from SegmentHandler import SegmentHandler
from Components.JudgeNode import JudgeNode
from Components.Logger import Logger
from Components.PreProcessingNode import PreProcesingNode
from Components.HandlerNode import HandlerNode

class SystemHandler:
    # Bottom fraction of features (by JudgeNode.compute_feature_relevance()'s
    # between/within cluster-variance ratio, ascending) passed to each segment
    # as freeze/remove screening candidates when feature_pruning_enabled=True.
    # This is only a cheap unsupervised PRE-FILTER — the actual freeze/remove
    # decision is made per-segment from real learned weight magnitude (see
    # SegmentHandler._update_feature_pruning). Experimental — off by default
    # (training.feature_pruning_enabled): a full-scale ablation on this
    # dataset never found a feature worth pruning at the default threshold
    # (SegmentHandler.FEATURE_FREEZE_WEIGHT_THRESHOLD), so it's shipped as an
    # opt-in rather than a default-on behavior change. May behave differently
    # on larger/higher-dimensional datasets with more genuinely redundant
    # columns — worth revisiting there.
    CANDIDATE_FEATURE_FRACTION = 0.3

    def __init__(self, maxX, target='exam_score', logger = None, connection_percentage=.08, density = .95, dimensions = 2, classification = 1, removable_columns=None):
        self.dimensions = dimensions
        self.max_x = maxX
        self.target = target
        self.logger = logger
        self.dimensions = dimensions
        self.connection_percentage = connection_percentage
        self.density = density
        self.classification = classification
        self.segments = []
        self.JudgeNode = JudgeNode(logger=self.logger, target=self.target, classification=self.classification)
        self.HandlerNode = HandlerNode(logger=self.logger, classification=self.classification)
        self.preprocessor = PreProcesingNode(Logger=self.logger, logger_classification=4, removable_columns=removable_columns)

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings, logger):
        """Construct a SystemHandler from a Settings instance."""
        m = settings.model
        d = settings.dataset
        return cls(
            maxX=m["max_x"],
            target=d["target_column"],
            logger=logger,
            connection_percentage=m["connection_percentage"],
            density=m["density"],
            dimensions=m["dimensions"],
            classification=settings.logging.get("log_level", 4),
            removable_columns=d.get("ignored_columns") or None,
        )

    # ── Segment loading ──────────────────────────────────────────────────

    def load_segments(self, nexseg_dir: str = ".") -> None:
        """Restore all segments from .nexseg files saved by a previous training run.

        Also restores JudgeNode's routing state from the judge_node.judgestate
        file saved alongside them (see JudgeNode.save_state()). If that file
        isn't present — e.g. .nexseg files produced before this fix existed —
        this falls back to the original behavior: JudgeNode stays untrained
        and runInfer() weights every segment equally.
        """
        self.segments = []
        seg_count = 2 ** self.dimensions
        for i in range(seg_count):
            path = os.path.join(nexseg_dir, f"segment_{i}.nexseg")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Segment file not found: {path}  "
                    f"(run 'train' mode first to produce .nexseg files)"
                )
            seg = SegmentHandler.load_nexseg(
                path,
                logger=self.logger,
                connection_percentage=self.connection_percentage,
                classification=self.classification,
            )
            self.segments.append(seg)
        self.display(f"Loaded {len(self.segments)} segments from '{nexseg_dir}'.")

        judge_state_path = os.path.join(nexseg_dir, "judge_node.judgestate")
        if self.JudgeNode.load_state(judge_state_path):
            self.display(f"JudgeNode routing state restored from '{judge_state_path}'.")
        else:
            self.display(
                f"No JudgeNode routing state found at '{judge_state_path}' — "
                f"falling back to equal relevance for every segment.",
                classification=3
            )

    def hot_swap_segment(self, segment_id: int, nexseg_dir: str = ".") -> None:
        """
        Replace ONE segment in-place with a freshly retrained/saved .nexseg,
        leaving JudgeNode's routing state and every other segment untouched.

        Safe as long as JudgeNode's own state has already been restored (see
        load_segments()/JudgeNode.load_state()) — JudgeNode routes purely by
        segment_id, never by a segment's internal weights/positions, so a
        swapped-in segment is immediately routable as long as it keeps the
        segment_id it's replacing. This does NOT re-validate that the new
        segment was trained on data compatible with what JudgeNode already
        clustered for this segment_id — that's on the caller (e.g. retrain
        only on rows JudgeNode already assigned to this segment_id).
        """
        path = os.path.join(nexseg_dir, f"segment_{segment_id}.nexseg")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Segment file not found: {path}")
        new_segment = SegmentHandler.load_nexseg(
            path,
            logger=self.logger,
            connection_percentage=self.connection_percentage,
            classification=self.classification,
        )
        if new_segment.segment_id != segment_id:
            raise ValueError(
                f"Loaded segment's segment_id ({new_segment.segment_id}) does not "
                f"match requested segment_id ({segment_id}) — refusing to hot-swap."
            )
        for i, seg in enumerate(self.segments):
            if seg.segment_id == segment_id:
                self.segments[i] = new_segment
                self.display(f"Hot-swapped segment {segment_id} from '{path}'.")
                return
        raise ValueError(f"No existing segment with segment_id={segment_id} to swap.")

    # ── Internals ────────────────────────────────────────────────────────

    def display(self, message, classification = None, Loud = True):
        message = f"[Main]: {message}"
        if self.logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.logger.log(message, classification, Loud)

    def _resolve_prediction_bounds(self, dataset, prediction_range_cfg):
        """Resolve settings.dataset.prediction_range into concrete (min, max) bounds.
        'auto' scans this run's full dataset target column; 'manual' uses the
        configured values as-is. Returns (None, None) when cfg is absent."""
        if not prediction_range_cfg:
            return None, None
        if prediction_range_cfg.get("mode") == "manual":
            return prediction_range_cfg.get("min_value"), prediction_range_cfg.get("max_value")
        col = dataset[self.target]
        return float(col.min()), float(col.max())

    def initializeAllSegments(self, Loud = False, visualization_enabled = False):
        segmentCount = 2 ** self.dimensions
        for i in range(segmentCount):
            self.segments.append(SegmentHandler(maxX=self.max_x, target=self.target, logger=self.logger, connection_percentage=self.connection_percentage, density=self.density, dimensions=self.dimensions, classification=self.classification, segment_id=i))
        for segment in self.segments:
            # Derive axis signs from segment_id bits: bit d=0 → +1, bit d=1 → -1
            loc = [1 - 2 * ((segment.segment_id >> d) & 1) for d in range(self.dimensions)]
            segment.initializeSegment(loc=loc, visualization_enabled=visualization_enabled)

        self.display(f"Initialized {segmentCount} segments", Loud=Loud)

    def train(self, dataset, epoch_count: int = 5, judge_iterations: int = 10, loud: bool = True,
              judge_min_clusters: int | None = None, judge_max_clusters: int | None = None,
              lr_scale_cfg: dict | None = None, prediction_range_cfg: dict | None = None,
              grad_clip_cfg: dict | None = None, delta_clip_cfg: dict | None = None,
              visualization_enabled: bool = False, reconnect_pct: float = 0.005,
              position_momentum: float = 0.0, feature_pruning_enabled: bool = False) -> None:
        """
        feature_pruning_enabled : experimental, off by default (settings.
                       training.feature_pruning_enabled). When True, JudgeNode
                       screens features by cluster relevance and each segment
                       independently confirms candidates against real learned
                       weight magnitude before freezing/removing anything (see
                       SegmentHandler._update_feature_pruning). A full-scale
                       ablation on this dataset never found a feature worth
                       pruning at the default threshold — kept opt-in rather
                       than default-on; may behave differently on larger or
                       higher-dimensional datasets.
        """
        from collections import defaultdict
        if not self.segments:
            raise ValueError("Segments must be initialized before training. Call initializeAllSegments() first.")

        pred_min, pred_max = self._resolve_prediction_bounds(dataset, prediction_range_cfg)

        # Step 1: Cluster the full dataset and assign clusters to segments.
        # Drop the target column before clustering — JudgeNode must partition on
        # input features only. Including the target would cause clusters to reflect
        # output range (e.g. all low-score students) rather than input structure,
        # making segments degenerate (constant output = cluster mean).
        self.display("Training JudgeNode — clustering full dataset...", Loud=loud)
        preprocessed = self.preprocessor.process_dataset(dataset.copy())
        judge_input = preprocessed.drop(columns=[self.target], errors='ignore')
        self.JudgeNode.train(judge_input, judge_iterations, segments=self.segments,
                             min_clusters=judge_min_clusters, max_clusters=judge_max_clusters)
        self.display("JudgeNode training complete. Proceeding to segment training...", Loud=loud)

        # Persist routing state alongside the per-segment .nexseg files so a
        # later load_segments() can restore real cluster-based routing
        # instead of falling back to equal-relevance for every segment —
        # see JudgeNode.save_state()/load_state().
        judge_state_path = self.JudgeNode.save_state()
        self.display(f"JudgeNode routing state saved -> {judge_state_path}", Loud=loud)

        # Step 1b: Screen for low cluster-relevance features. Cheap, unsupervised
        # pre-filter only — each segment independently confirms (or rejects) these
        # candidates against its own actual learned weight magnitude before ever
        # freezing/removing anything (see SegmentHandler._update_feature_pruning).
        # Off by default — see feature_pruning_enabled docstring above.
        candidate_features = []
        if feature_pruning_enabled:
            feature_relevance = self.JudgeNode.compute_feature_relevance()
            if feature_relevance:
                n_candidates = max(1, int(len(feature_relevance) * self.CANDIDATE_FEATURE_FRACTION))
                candidate_features = list(feature_relevance.keys())[:n_candidates]
                self.display(
                    f"JudgeNode flagged {len(candidate_features)}/{len(feature_relevance)} "
                    f"low cluster-relevance candidate feature(s) for freeze/remove "
                    f"screening: {candidate_features}",
                    Loud=loud
                )

        # Step 2: Map each cluster's points back to original dataset row indices.
        # Use the same target-dropped view for the lookup so tuple keys match cluster points.
        judge_input = judge_input.reset_index(drop=True)
        n_rows = len(dataset)
        vector_to_indices: dict[tuple, list[int]] = defaultdict(list)
        for i, rec in enumerate(judge_input.to_dict(orient='records')):
            if i < n_rows:
                vector_to_indices[tuple(rec.values())].append(i)

        segment_indices: dict[int, set[int]] = defaultdict(set)
        for cluster in self.JudgeNode.segment_weights['clusters']:
            sid = cluster.get('segment_id')
            if sid is None:
                continue
            for point in cluster['points']:
                for idx in vector_to_indices.get(tuple(point), []):
                    segment_indices[sid].add(idx)

        # Step 3: Train each segment only on its assigned rows
        for segment in self.segments:
            indices = sorted(idx for idx in segment_indices.get(segment.segment_id, []) if idx < n_rows)
            if not indices:
                self.display(f"Segment {segment.segment_id} has no assigned data — skipping.", Loud=loud)
                continue
            subset = dataset.iloc[indices].reset_index(drop=True)
            self.display(f"Training segment {segment.segment_id} on {len(subset)}/{len(dataset)} rows...", Loud=loud)
            segment.train(subset, epoch_count=epoch_count, preprocessor=self.preprocessor,
                          lr_scale_cfg=lr_scale_cfg, pred_min=pred_min, pred_max=pred_max,
                          grad_clip_cfg=grad_clip_cfg, delta_clip_cfg=delta_clip_cfg,
                          visualization_enabled=visualization_enabled,
                          reconnect_pct=reconnect_pct, position_momentum=position_momentum,
                          candidate_features=candidate_features)

    def train_full(self, dataset, epoch_count: int = 5, loud: bool = True,
                    lr_scale_cfg: dict | None = None, prediction_range_cfg: dict | None = None,
                    grad_clip_cfg: dict | None = None, delta_clip_cfg: dict | None = None,
                    visualization_enabled: bool = False, reconnect_pct: float = 0.005,
                    position_momentum: float = 0.0) -> None:
        """Train every segment on the complete dataset (no JudgeNode partitioning).
        JudgeNode routing still works at inference — clusters are built on the
        full dataset so all segments see the same data distribution during training."""
        if not self.segments:
            raise ValueError("Segments must be initialized before training. Call initializeAllSegments() first.")

        pred_min, pred_max = self._resolve_prediction_bounds(dataset, prediction_range_cfg)
        self.display("Full-dataset training mode — all segments train on complete dataset.", Loud=loud)
        for segment in self.segments:
            self.display(f"Training segment {segment.segment_id} on {len(dataset)} rows...", Loud=loud)
            segment.train(dataset, epoch_count=epoch_count, preprocessor=self.preprocessor,
                          lr_scale_cfg=lr_scale_cfg, pred_min=pred_min, pred_max=pred_max,
                          grad_clip_cfg=grad_clip_cfg, delta_clip_cfg=delta_clip_cfg,
                          visualization_enabled=visualization_enabled,
                          reconnect_pct=reconnect_pct, position_momentum=position_momentum)

    def runInfer(self, input, loud = True, aggregation_mode: str = "bma", selection_percentage: float = .5,
                 trace: bool = False, trace_path: str = "trace.jsonl"):
        """
        Run a single inference. Returns a dict:
            {
                'score':       float — the aggregated prediction (was the
                               entire return value before this fix),
                'confidence':  float — cross-segment agreement, see
                               HandlerNode.process_reports(),
                'segment_id':  int | None — the segment that most shaped the
                               final score (highest final aggregation weight),
                'archetype':   str | None — "cluster_<id>", the JudgeNode
                               cluster whose centroid is nearest this input
                               (see JudgeNode.find_nearest_cluster) — a
                               semantic "what kind of row is this" label,
                               distinct from segment_id (compute routing),
                'breakdown':   dict — full per-segment mean/weight/relevance
                               detail; same object as self.HandlerNode.last_breakdown.
            }
        Returns None if no segment produced a usable prediction (matches the
        prior bare-float behavior's None case).

        trace : off by default — zero added cost on the normal path (no new
                       computation, just an `if` around the block below and a
                       collect_paths=False no-op flag passed to segmentInfer).
                       When True, appends one JSON line per call to trace_path
                       containing the full breakdown plus, per contributing
                       segment, each collected signal's hop-by-hop node-
                       position path — this is the raw data source for the
                       visualization tool's signal-path tracing view. A
                       single file append per INFERENCE call, never per
                       training sample — this must never be turned on inside
                       a training loop's hot path.
        """
        if self.JudgeNode is None or self.HandlerNode is None:
            raise ValueError("JudgeNode or HandlerNode not assigned")

        self.display("Running inference on JudgeNode", Loud=loud)
        pre_input = self.preprocessor.process_data(input)

        # JudgeNode was trained on target-excluded vectors; strip it here so the
        # routing distance calculation aligns with the cluster centroid dimensions.
        judge_input = {k: v for k, v in pre_input.items() if k != self.target}

        # If JudgeNode has not been trained, activate all segments equally
        if not self.JudgeNode.segment_weights['segment']:
            self.display("JudgeNode not trained — using all segments at equal relevance.", Loud=loud)
            selected_segments = [(s.segment_id, 1.0) for s in self.segments]
        else:
            relevance_scores  = self.JudgeNode.calculate_input_segment_relevance(judge_input, Loud=loud)
            selected_segments = self.JudgeNode.find_relevant_segments(relevance_scores, selection_percentage=selection_percentage, Loud=loud)

        self.display(f"Selected segments: {[sid for sid, _ in selected_segments]}", Loud=loud)
        self.NumberSegsUsed = len(selected_segments)

        # Strip target from segment input — segments must not see the answer at inference time.
        seg_input = {k: v for k, v in pre_input.items() if k != self.target}

        segment_map = {s.segment_id: s for s in self.segments}
        trace_paths = {} if trace else None
        for segment_id, relevance in selected_segments:
            segment = segment_map[segment_id]
            reports = segment.segmentInfer(seg_input, loud=loud, collect_paths=trace)
            for report in reports:
                self.HandlerNode.receive_report(segment_id, relevance, report['prediction'],
                                                confidence=report.get('confidence', 1.0))
                if trace and 'paths' in report:
                    trace_paths.setdefault(segment_id, []).extend(report['paths'])

        breakdown = self.HandlerNode.process_reports(loud, aggregation_mode=aggregation_mode)
        if breakdown is None:
            return None

        archetype_match = self.JudgeNode.find_nearest_cluster(judge_input)
        archetype = f"cluster_{archetype_match['archetype_id']}" if archetype_match else None

        result = {
            'score':      breakdown['score'],
            'confidence': breakdown['confidence'],
            'segment_id': breakdown['dominant_segment_id'],
            'archetype':  archetype,
            'breakdown':  breakdown,
        }

        if trace:
            import json as _json, time as _time
            record = {
                'timestamp': _time.time(),
                'dimensions': self.dimensions,
                'max_x': self.max_x,
                'selected_segments': [sid for sid, _ in selected_segments],
                'paths': trace_paths,
                **result,
            }
            try:
                with open(trace_path, 'a') as f:
                    f.write(_json.dumps(record, default=str) + '\n')
            except Exception:
                pass  # tracing must never break inference itself

        return result

    def getNumberSegmentsUsed(self):
        return self.NumberSegsUsed if hasattr(self, 'NumberSegsUsed') else None
