# Code required for the nexus system handler
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Components"))

from SegmentHandler import SegmentHandler
from Components.JudgeNode import JudgeNode
from Components.Logger import Logger
from Components.PreProcessingNode import PreProcesingNode
from Components.HandlerNode import HandlerNode


def _train_segment_worker(args: dict) -> dict:
    """
    Module-level (picklable) worker for SystemHandler.train_parallel_ensemble().
    Runs in its own OS process — builds ONE segment, trains it on the FULL
    dataset, saves it to its own .nexseg, and returns only small, plain-
    picklable summary info.

    Deliberately does NOT return the live SegmentHandler object: its Logger
    holds an open file handle (RichConsole._fh, added for the earlier
    logging-performance fix), and open file handles cannot be pickled across
    a process boundary. Each worker gets its own Logger/log file rather than
    sharing the parent's — a shared file handle isn't picklable to hand to a
    child in the first place, and multiple OS processes writing the same
    file concurrently would interleave garbage even if it were.

    random.seed(42 + segment_id): each worker process is a fresh Python
    interpreter, so the module-level random.seed(42) in ProcessingNode.py
    runs again independently in every worker — without this line every
    segment would replay the IDENTICAL sequence of random draws (weight
    init noise, stochastic hop routing) and differ from each other only by
    their geometric quadrant (loc), not by any real stochastic diversity.
    In the existing sequential train()/train_full(), segments get different
    randomness "for free" because they share one continuously-advancing
    random stream — parallel processes lose that property unless restored
    explicitly, which is what this line does (still fully deterministic and
    reproducible: the same segment_id always gets the same seed).
    """
    import random
    # seed_id vs segment_id: for sharded/load-balanced dispatch, several
    # shards of the SAME real segment need IDENTICAL initial topology (so
    # merging shards back together later is averaging corresponding nodes,
    # not garbage) but must each write to a DIFFERENT output file (else
    # concurrent shards racing on the same segment_N.nexseg would corrupt
    # it). seed_id (defaults to segment_id for every existing non-sharded
    # caller) controls the former; segment_id alone controls the latter via
    # SegmentHandler's own filename convention.
    random.seed(42 + args.get('seed_id', args['segment_id']))

    import time as _time
    import pandas as pd
    import Components.RichConsole as RC
    from SegmentHandler import SegmentHandler

    logger = RC.RichLogger(
        filename=f"parallel_ensemble_segment{args['segment_id']}_{int(_time.time())}.log",
        log_level=4, console_level=1,
    )
    segment = SegmentHandler(
        maxX=args['max_x'], target=args['target'], logger=logger,
        connection_percentage=args['connection_percentage'], density=args['density'],
        dimensions=args['dimensions'], classification=args['classification'],
        segment_id=args['segment_id'],
    )
    segment.initializeSegment(loc=args['loc'], visualization_enabled=False)

    dataset = pd.DataFrame(args['dataset_records'])
    t0 = _time.time()
    segment.train(
        dataset, epoch_count=args['epoch_count'],
        lr_scale_cfg=args['lr_scale_cfg'], pred_min=args['pred_min'], pred_max=args['pred_max'],
        grad_clip_cfg=args['grad_clip_cfg'], delta_clip_cfg=args['delta_clip_cfg'],
        reconnect_pct=args['reconnect_pct'], position_momentum=args['position_momentum'],
    )
    return {
        'segment_id': args['segment_id'],
        'nexseg_path': f"segment_{args['segment_id']}.nexseg",
        'train_time_sec': round(_time.time() - t0, 2),
    }


def _merge_segment_shards(shard_paths: list, real_segment_id: int, logger,
                          connection_percentage: float, classification: int):
    """Average N shards of the SAME real segment (identical init topology --
    see seed_id in _train_segment_worker -- trained on disjoint row subsets)
    into one merged SegmentHandler: local-SGD / federated-averaging style
    periodic merge, not anything JudgeNode-related.

    Loads real SegmentHandler objects via load_nexseg() (reusing proven
    deserialization rather than hand-rolling raw JSON merging), averages
    each processing node's position and weights element-wise, averages the
    splitter's signal_weights, then REBUILDS connectivity from the merged
    (now-diverged-then-averaged) positions -- each shard's saved
    connections reflect its own pre-merge layout and are no longer valid
    once positions move. Reviewer positions are untouched (reviewers never
    move during training in this codebase, so they're identical across
    shards already -- kept from the first shard as-is).
    """
    from SegmentHandler import SegmentHandler

    segments = [SegmentHandler.load_nexseg(p, logger=logger,
                connection_percentage=connection_percentage, classification=classification)
                for p in shard_paths]
    if len(segments) == 1:
        segments[0].segment_id = real_segment_id
        return segments[0]

    n_shards = len(segments)
    base = segments[0]
    base_nodes = base.segmentComponents['processing_nodes']
    other_node_lists = [s.segmentComponents['processing_nodes'] for s in segments[1:]]

    for i, node in enumerate(base_nodes):
        other_nodes = [nodes[i] for nodes in other_node_lists]
        # Rebuild + reassign rather than mutate elements in place -- position
        # comes back as a tuple from load_nexseg() (immutable), not a list.
        node.position = [
            (node.position[j] + sum(o.position[j] for o in other_nodes)) / n_shards
            for j in range(len(node.position))
        ]
        # distance_to_origin is computed once at construction from position
        # and used directly in the forward-pass delta scaling and routing
        # bias -- ProcessingNode itself recomputes it after every position
        # gradient step (see apply_position_gradient), so the merge must
        # too, or every merged node silently uses a stale pre-merge distance.
        node.distance_to_origin = sum(p ** 2 for p in node.position) ** 0.5
        all_features = set(node.weights.keys())
        for o in other_nodes:
            all_features |= set(o.weights.keys())
        for feat in all_features:
            vals = [node.weights.get(feat, 0.0)] + [o.weights.get(feat, 0.0) for o in other_nodes]
            node.weights[feat] = sum(vals) / n_shards

    base_splitter = base.segmentComponents['splitter']
    other_splitters = [s.segmentComponents['splitter'] for s in segments[1:]]
    all_sfeatures = set(base_splitter.signal_weights.keys())
    for o in other_splitters:
        all_sfeatures |= set(o.signal_weights.keys())
    for feat in all_sfeatures:
        vals = [base_splitter.signal_weights.get(feat, 0.0)] + [o.signal_weights.get(feat, 0.0) for o in other_splitters]
        base_splitter.signal_weights[feat] = sum(vals) / n_shards

    # Rebuild connectivity from the merged positions -- reuses the exact
    # same reconnect calls _epoch() makes mid-training, not new logic.
    reviewers = base.segmentComponents['reviewer']
    node_list = base_nodes + reviewers
    for node in base_nodes:
        node.connected_nodes = []
        node.connect_nearest_nodes(node_list, connection_percentage)
    base_splitter.calculate_nearest_neighbors(base_nodes)

    base.segment_id = real_segment_id
    return base


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

    def train_parallel_ensemble(self, dataset, epoch_count: int = 5, judge_iterations: int = 10,
                                loud: bool = True,
                                judge_min_clusters: int | None = None, judge_max_clusters: int | None = None,
                                lr_scale_cfg: dict | None = None, prediction_range_cfg: dict | None = None,
                                grad_clip_cfg: dict | None = None, delta_clip_cfg: dict | None = None,
                                reconnect_pct: float = 0.005, position_momentum: float = 0.0,
                                max_workers: int | None = None) -> list[dict]:
        """
        Cluster the data with JudgeNode (for INFERENCE-time segment
        selection/weighting only), then train every segment as a generalist
        on the FULL dataset, in parallel across OS processes.

        This is a third training mode alongside train() (partitioned: each
        segment sees only its own cluster's rows) and train_full() (every
        segment sees the full dataset, but JudgeNode never actually gets
        trained despite its docstring's claim — see its code). This method
        does what train_full()'s docstring already promises, for real, plus
        real parallelism.

        Why generalist segments instead of cluster-specialists — hot-swap
        implications: hot_swap_segment() replaces one segment_id's model
        without touching JudgeNode's routing or any other segment. Under the
        normal partitioned train(), a freshly retrained replacement is only
        really valid if it was trained on the SAME rows JudgeNode originally
        assigned to that segment_id — swap in a segment trained on a
        different data slice and you've introduced a distribution mismatch
        the system has no way to detect. Under this method, every segment_id
        is trained on the identical full dataset, so that constraint simply
        doesn't exist: ANY freshly retrained segment (on this same dataset,
        or a superset/updated version of it) is compatible with ANY
        segment_id by construction. Hot-swapping becomes safe by
        construction instead of safe-if-you-remembered-to-match-the-data.

        Scoring/combination at inference is unchanged and needs no new code:
        runInfer() -> HandlerNode.process_reports() already combines each
        selected segment's own prediction (weighted by JudgeNode relevance
        and reviewer-confidence/inter-segment variance per aggregation_mode)
        into one system-level score — that "final judge system based on
        selected segments" already exists; this method just changes how the
        segments feeding it get trained.

        max_workers : passed straight to ProcessPoolExecutor — None uses
                       os.cpu_count(). Training is CPU-bound and Python's GIL
                       blocks real threading speedup, so this uses real OS
                       processes, not threads.

        Returns the list of per-segment worker result dicts (segment_id,
        nexseg_path, train_time_sec) for whichever caller wants visibility
        into what actually got trained/timed.
        """
        import concurrent.futures as cf

        if not self.segments:
            raise ValueError("Segments must be initialized before training. Call initializeAllSegments() first.")

        pred_min, pred_max = self._resolve_prediction_bounds(dataset, prediction_range_cfg)

        self.display("Training JudgeNode — clustering full dataset (inference-time selection only, "
                     "not used to partition training rows)...", Loud=loud)
        preprocessed = self.preprocessor.process_dataset(dataset.copy())
        judge_input = preprocessed.drop(columns=[self.target], errors='ignore')
        self.JudgeNode.train(judge_input, judge_iterations, segments=self.segments,
                             min_clusters=judge_min_clusters, max_clusters=judge_max_clusters)
        judge_state_path = self.JudgeNode.save_state()
        self.display(f"JudgeNode routing state saved -> {judge_state_path}", Loud=loud)

        dataset_records = dataset.to_dict(orient='records')
        worker_args = []
        for segment in self.segments:
            loc = [1 - 2 * ((segment.segment_id >> d) & 1) for d in range(self.dimensions)]
            worker_args.append({
                'segment_id': segment.segment_id, 'loc': loc,
                'max_x': self.max_x, 'dimensions': self.dimensions,
                'connection_percentage': self.connection_percentage, 'density': self.density,
                'classification': self.classification, 'target': self.target,
                'dataset_records': dataset_records, 'epoch_count': epoch_count,
                'lr_scale_cfg': lr_scale_cfg, 'pred_min': pred_min, 'pred_max': pred_max,
                'grad_clip_cfg': grad_clip_cfg, 'delta_clip_cfg': delta_clip_cfg,
                'reconnect_pct': reconnect_pct, 'position_momentum': position_momentum,
            })

        self.display(
            f"Training {len(worker_args)} segments in parallel "
            f"(max_workers={max_workers or 'auto/os.cpu_count()'})...",
            Loud=loud
        )
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_train_segment_worker, worker_args))

        # Reload every freshly-trained segment from disk — the exact same
        # mechanism hot_swap_segment() already uses for a single segment, run
        # once per segment here. Nothing new to trust: if hot-swap is
        # correct, this is correct.
        for r in sorted(results, key=lambda r: r['segment_id']):
            self.display(
                f"Segment {r['segment_id']} trained in {r['train_time_sec']}s -> {r['nexseg_path']}",
                Loud=loud
            )
            self.hot_swap_segment(r['segment_id'], ".")

        self.display("Parallel ensemble training complete.", Loud=loud)
        return results

    def train_parallel_partitioned(self, dataset, epoch_count: int = 5, judge_iterations: int = 10,
                                   loud: bool = True,
                                   judge_min_clusters: int | None = None, judge_max_clusters: int | None = None,
                                   lr_scale_cfg: dict | None = None, prediction_range_cfg: dict | None = None,
                                   grad_clip_cfg: dict | None = None, delta_clip_cfg: dict | None = None,
                                   reconnect_pct: float = 0.005, position_momentum: float = 0.0,
                                   max_workers: int | None = None) -> list[dict]:
        """
        Parallel version of the normal partitioned train(): JudgeNode still
        clusters and assigns each row to exactly ONE segment (cluster-
        specialist segments, identical partitioning logic to train()) — the
        only difference is that each segment's own (smaller, disjoint)
        subset trains in its own OS process instead of one after another.

        Reuses the exact same _train_segment_worker() as
        train_parallel_ensemble() — the worker doesn't know or care whether
        the rows it receives are the full dataset or one cluster's slice of
        it, it just trains on whatever dataset_records it's handed.

        Hot-swap implications differ from train_parallel_ensemble(): since
        each segment_id is trained on a DIFFERENT subset here (same as
        sequential train()), hot_swap_segment()'s original caveat still
        applies — a replacement is only valid if trained on the SAME rows
        JudgeNode assigned to that segment_id. This method does NOT remove
        that constraint the way train_parallel_ensemble() does; it only
        tests whether cluster-specialist accuracy holds up when the same
        four segments train in parallel instead of in sequence.

        feature_pruning_enabled is intentionally not threaded through here —
        each worker is a fresh process with no visibility into the others,
        so the existing feature-pruning design (freeze/remove state tracked
        per SegmentHandler instance across its own epochs) still works fine
        per-worker, but wiring the settings-level toggle through was out of
        scope for this speed/accuracy comparison specifically.
        """
        import concurrent.futures as cf
        from collections import defaultdict

        if not self.segments:
            raise ValueError("Segments must be initialized before training. Call initializeAllSegments() first.")

        pred_min, pred_max = self._resolve_prediction_bounds(dataset, prediction_range_cfg)

        self.display("Training JudgeNode — clustering full dataset...", Loud=loud)
        preprocessed = self.preprocessor.process_dataset(dataset.copy())
        judge_input = preprocessed.drop(columns=[self.target], errors='ignore')
        self.JudgeNode.train(judge_input, judge_iterations, segments=self.segments,
                             min_clusters=judge_min_clusters, max_clusters=judge_max_clusters)
        judge_state_path = self.JudgeNode.save_state()
        self.display(f"JudgeNode routing state saved -> {judge_state_path}", Loud=loud)

        # Map each cluster's points back to original dataset row indices —
        # identical logic to train()'s own Step 2.
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

        worker_args = []
        skipped = []
        for segment in self.segments:
            indices = sorted(idx for idx in segment_indices.get(segment.segment_id, []) if idx < n_rows)
            if not indices:
                skipped.append(segment.segment_id)
                continue
            subset_records = dataset.iloc[indices].reset_index(drop=True).to_dict(orient='records')
            loc = [1 - 2 * ((segment.segment_id >> d) & 1) for d in range(self.dimensions)]
            worker_args.append({
                'segment_id': segment.segment_id, 'loc': loc,
                'max_x': self.max_x, 'dimensions': self.dimensions,
                'connection_percentage': self.connection_percentage, 'density': self.density,
                'classification': self.classification, 'target': self.target,
                'dataset_records': subset_records, 'epoch_count': epoch_count,
                'lr_scale_cfg': lr_scale_cfg, 'pred_min': pred_min, 'pred_max': pred_max,
                'grad_clip_cfg': grad_clip_cfg, 'delta_clip_cfg': delta_clip_cfg,
                'reconnect_pct': reconnect_pct, 'position_momentum': position_momentum,
            })
        if skipped:
            self.display(f"Segments with no assigned data — skipping: {skipped}", Loud=loud)

        self.display(
            f"Training {len(worker_args)} segments in parallel, each on its own "
            f"JudgeNode-assigned subset (max_workers={max_workers or 'auto/os.cpu_count()'})...",
            Loud=loud
        )
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_train_segment_worker, worker_args))

        for r in sorted(results, key=lambda r: r['segment_id']):
            self.display(
                f"Segment {r['segment_id']} trained in {r['train_time_sec']}s -> {r['nexseg_path']}",
                Loud=loud
            )
            self.hot_swap_segment(r['segment_id'], ".")

        self.display("Parallel partitioned training complete.", Loud=loud)
        return results

    def train_parallel_balanced(self, dataset, epoch_count: int = 5, judge_iterations: int = 10,
                                loud: bool = True,
                                judge_min_clusters: int | None = None, judge_max_clusters: int | None = None,
                                lr_scale_cfg: dict | None = None, prediction_range_cfg: dict | None = None,
                                grad_clip_cfg: dict | None = None, delta_clip_cfg: dict | None = None,
                                reconnect_pct: float = 0.005, position_momentum: float = 0.0,
                                max_workers: int | None = None) -> list[dict]:
        """
        Load-balanced variant of train_parallel_partitioned(). Identical
        JudgeNode clustering and row-assignment (deliberately -- see the
        conversation this came from: JudgeNode's clustering must stay driven
        purely by task/data nature, never reshaped for compute convenience,
        since it's meant to eventually route between fundamentally different
        task types, not just balance load for one). The only difference is
        HOW an already-decided, possibly very uneven set of per-segment row
        assignments gets executed.

        Observed problem this addresses: on datasets where JudgeNode's
        clusters come out very unevenly sized (e.g. YearPredictionMSD's 90
        continuous audio features clustered ~8.5x unevenly vs the much more
        balanced exam-score dataset), train_parallel_partitioned() dispatches
        one big blocking task per segment -- the largest segment's worker
        keeps 3 other cores idle for most of the run once they finish early.

        Fix (execution layer only, never touches clustering): split each
        segment's ALREADY-ASSIGNED rows into multiple shards sized so every
        dispatched task takes roughly the same time, submit all shards
        (across all segments) to ONE shared worker pool sorted largest-first,
        then merge each segment's shards back into one final segment via
        weight/position averaging (local-SGD / federated-averaging style) +
        rebuilt connectivity, since positions genuinely diverge during
        training even from an identical start.

        Shard count per segment is proportional to its row count relative to
        a target shard size (total_rows / max_workers), so total shard count
        roughly matches available cores regardless of how imbalanced the
        underlying clusters are.
        """
        import concurrent.futures as cf
        import os as _os
        from collections import defaultdict

        if not self.segments:
            raise ValueError("Segments must be initialized before training. Call initializeAllSegments() first.")

        pred_min, pred_max = self._resolve_prediction_bounds(dataset, prediction_range_cfg)

        self.display("Training JudgeNode — clustering full dataset...", Loud=loud)
        preprocessed = self.preprocessor.process_dataset(dataset.copy())
        judge_input = preprocessed.drop(columns=[self.target], errors='ignore')
        self.JudgeNode.train(judge_input, judge_iterations, segments=self.segments,
                             min_clusters=judge_min_clusters, max_clusters=judge_max_clusters)
        judge_state_path = self.JudgeNode.save_state()
        self.display(f"JudgeNode routing state saved -> {judge_state_path}", Loud=loud)

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

        # ── Proportional sharding: decide shard count per segment ─────────
        n_workers = max_workers or _os.cpu_count() or 4
        per_segment_rows = {}
        skipped = []
        for segment in self.segments:
            indices = sorted(idx for idx in segment_indices.get(segment.segment_id, []) if idx < n_rows)
            if not indices:
                skipped.append(segment.segment_id)
                continue
            per_segment_rows[segment.segment_id] = indices
        if skipped:
            self.display(f"Segments with no assigned data — skipping: {skipped}", Loud=loud)

        total_rows = sum(len(v) for v in per_segment_rows.values())
        target_shard_size = max(1, total_rows // n_workers) if total_rows else 1

        worker_args = []
        shard_plan: dict[int, int] = {}  # real segment_id -> shard count, for the report
        for segment in self.segments:
            indices = per_segment_rows.get(segment.segment_id)
            if not indices:
                continue
            shard_count = max(1, round(len(indices) / target_shard_size))
            shard_plan[segment.segment_id] = shard_count
            loc = [1 - 2 * ((segment.segment_id >> d) & 1) for d in range(self.dimensions)]
            chunk_size = max(1, -(-len(indices) // shard_count))  # ceil div
            for shard_idx in range(shard_count):
                shard_indices = indices[shard_idx * chunk_size: (shard_idx + 1) * chunk_size]
                if not shard_indices:
                    continue
                subset_records = dataset.iloc[shard_indices].reset_index(drop=True).to_dict(orient='records')
                # segment_id here is a SYNTHETIC per-shard id (unique output
                # filename) -- seed_id carries the REAL segment identity so
                # every shard of the same real segment gets identical init
                # topology (required for the merge to average corresponding
                # nodes, not garbage), while loc (quadrant) is also derived
                # from the real segment_id, not the synthetic one.
                synthetic_id = segment.segment_id * 1000 + shard_idx
                worker_args.append({
                    'segment_id': synthetic_id, 'seed_id': segment.segment_id, 'loc': loc,
                    'max_x': self.max_x, 'dimensions': self.dimensions,
                    'connection_percentage': self.connection_percentage, 'density': self.density,
                    'classification': self.classification, 'target': self.target,
                    'dataset_records': subset_records, 'epoch_count': epoch_count,
                    'lr_scale_cfg': lr_scale_cfg, 'pred_min': pred_min, 'pred_max': pred_max,
                    'grad_clip_cfg': grad_clip_cfg, 'delta_clip_cfg': delta_clip_cfg,
                    'reconnect_pct': reconnect_pct, 'position_momentum': position_momentum,
                    '_real_segment_id': segment.segment_id, '_row_count': len(shard_indices),
                })

        # Dispatch-order optimization: largest shards first, so if there are
        # ever more shards than workers, the long poles start earliest
        # rather than being queued behind short ones.
        worker_args.sort(key=lambda a: -a['_row_count'])
        self.display(
            f"Training {len(self.segments) - len(skipped)} segments as {len(worker_args)} shards "
            f"(plan={shard_plan}, max_workers={n_workers})...",
            Loud=loud
        )
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            shard_results = list(executor.map(_train_segment_worker, worker_args))

        # ── Merge shards back into one segment per real segment_id ────────
        synthetic_to_args = {a['segment_id']: a for a in worker_args}
        by_real_id: dict[int, list[str]] = defaultdict(list)
        time_by_real_id: dict[int, float] = defaultdict(float)
        for r in shard_results:
            real_id = synthetic_to_args[r['segment_id']]['_real_segment_id']
            by_real_id[real_id].append(r['nexseg_path'])
            time_by_real_id[real_id] += r['train_time_sec']

        results = []
        for real_id, shard_paths in sorted(by_real_id.items()):
            merged = _merge_segment_shards(shard_paths, real_id, self.logger,
                                           self.connection_percentage, self.classification)
            merged_path = merged._save_nexseg()
            results.append({'segment_id': real_id, 'nexseg_path': merged_path,
                            'n_shards': len(shard_paths),
                            'total_shard_train_time_sec': round(time_by_real_id[real_id], 2)})
            self.display(f"Segment {real_id}: merged {len(shard_paths)} shard(s) -> {merged_path}", Loud=loud)
            self.hot_swap_segment(real_id, ".")

        self.display("Parallel balanced training complete.", Loud=loud)
        return results

    def runInfer(self, input, loud = True, aggregation_mode: str = "bma", selection_percentage: float = .5):
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
        for segment_id, relevance in selected_segments:
            segment = segment_map[segment_id]
            reports = segment.segmentInfer(seg_input, loud=loud)
            for report in reports:
                self.HandlerNode.receive_report(segment_id, relevance, report['prediction'],
                                                confidence=report.get('confidence', 1.0))

        breakdown = self.HandlerNode.process_reports(loud, aggregation_mode=aggregation_mode)
        if breakdown is None:
            return None

        archetype_match = self.JudgeNode.find_nearest_cluster(judge_input)
        archetype = f"cluster_{archetype_match['archetype_id']}" if archetype_match else None

        return {
            'score':      breakdown['score'],
            'confidence': breakdown['confidence'],
            'segment_id': breakdown['dominant_segment_id'],
            'archetype':  archetype,
            'breakdown':  breakdown,
        }

    def getNumberSegmentsUsed(self):
        return self.NumberSegsUsed if hasattr(self, 'NumberSegsUsed') else None
