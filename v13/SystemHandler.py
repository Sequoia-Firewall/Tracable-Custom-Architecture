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

        JudgeNode state is not persisted in .nexseg files; inference will fall back
        to weighting all segments equally, which is the existing graceful fallback.
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

    # ── Internals ────────────────────────────────────────────────────────

    def display(self, message, classification = None, Loud = True):
        message = f"[Main]: {message}"
        if self.logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.logger.log(message, classification, Loud)

    def initializeAllSegments(self, Loud = False):
        segmentCount = 2 ** self.dimensions
        for i in range(segmentCount):
            self.segments.append(SegmentHandler(maxX=self.max_x, target=self.target, logger=self.logger, connection_percentage=self.connection_percentage, density=self.density, dimensions=self.dimensions, classification=self.classification, segment_id=i))
        for segment in self.segments:
            # Derive axis signs from segment_id bits: bit d=0 → +1, bit d=1 → -1
            loc = [1 - 2 * ((segment.segment_id >> d) & 1) for d in range(self.dimensions)]
            segment.initializeSegment(loc=loc)

        self.display(f"Initialized {segmentCount} segments", Loud=Loud)

    def train(self, dataset, epoch_count: int = 5, judge_iterations: int = 10, loud: bool = True,
              judge_min_clusters: int | None = None, judge_max_clusters: int | None = None) -> None:
        from collections import defaultdict
        if not self.segments:
            raise ValueError("Segments must be initialized before training. Call initializeAllSegments() first.")

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
            segment.train(subset, epoch_count=epoch_count, preprocessor=self.preprocessor)

    def train_full(self, dataset, epoch_count: int = 5, loud: bool = True) -> None:
        """Train every segment on the complete dataset (no JudgeNode partitioning).
        JudgeNode routing still works at inference — clusters are built on the
        full dataset so all segments see the same data distribution during training."""
        if not self.segments:
            raise ValueError("Segments must be initialized before training. Call initializeAllSegments() first.")

        self.display("Full-dataset training mode — all segments train on complete dataset.", Loud=loud)
        for segment in self.segments:
            self.display(f"Training segment {segment.segment_id} on {len(dataset)} rows...", Loud=loud)
            segment.train(dataset, epoch_count=epoch_count, preprocessor=self.preprocessor)

    def runInfer(self, input, loud = True, aggregation_mode: str = "bma", selection_percentage: float = .5):
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
        for segment_id, relevance in selected_segments:
            segment = segment_map[segment_id]
            reports = segment.segmentInfer(seg_input, loud=loud)
            for report in reports:
                self.HandlerNode.receive_report(segment_id, relevance, report['prediction'])

        return self.HandlerNode.process_reports(loud, aggregation_mode=aggregation_mode)

    def getNumberSegmentsUsed(self):
        return self.NumberSegsUsed if hasattr(self, 'NumberSegsUsed') else None
