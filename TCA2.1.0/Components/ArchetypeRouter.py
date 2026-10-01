import numpy as np
from sklearn.cluster import KMeans


class ArchetypeRouter:
    """
    Discovers cheap, intra-segment "kinds of questions" (via KMeans -- fast,
    deterministic, no epsilon-tuning fragility like DBSCAN would need here)
    and learns a per-(cluster, edge) routing preference from real observed
    error, via a one-time exploration phase (see run_exploration() in the
    test harness that drives this) rather than a routing-time heuristic.

    Deliberately proportional, not winner-take-all: every exploratory trial
    nudges every edge on its path by an amount proportional to how much
    better/worse than baseline that trial's error was, so a single lucky
    trial can't lock in a path, and the resulting factor is a smooth
    multiplier on top of whatever base routing heuristic is in use --
    "even if the assignment is granular it can follow the connections that
    have been trained" was the design constraint this satisfies.

    Does not touch JudgeNode -- this clustering is entirely intra-segment
    and separate from JudgeNode's cross-segment clustering objective.
    """

    def __init__(self, n_clusters=4, lr=0.05, factor_floor=0.1):
        self.n_clusters = n_clusters
        self.lr = lr
        self.factor_floor = factor_floor
        self.kmeans = None
        self.feat_order = None
        # {cluster_id: {(pos_a_tuple, pos_b_tuple): accumulated_score}}
        self.routing_factor = {}

    def fit(self, samples, feat_order):
        """samples : list of feature dicts (no target). Disables itself
        (falls back to a no-op factor of 1.0 everywhere) if there isn't
        enough data to support n_clusters -- a segment too small for this
        shouldn't crash, just skip the mechanism."""
        self.feat_order = list(feat_order)
        if len(samples) < self.n_clusters:
            self.kmeans = None
            return
        X = np.array([[s.get(f, 0.0) for f in self.feat_order] for s in samples], dtype=np.float64)
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=5, random_state=42).fit(X)

    def cluster_id(self, input_dict):
        if self.kmeans is None or self.feat_order is None:
            return 0
        x = np.array([[input_dict.get(f, 0.0) for f in self.feat_order]], dtype=np.float64)
        return int(self.kmeans.predict(x)[0])

    def factor(self, cluster_id, pos_a, pos_b):
        table = self.routing_factor.get(cluster_id)
        if not table:
            return 1.0
        score = table.get((tuple(pos_a), tuple(pos_b)), 0.0)
        # Accumulated score is roughly centered at 0 (positive = this edge
        # has historically beaten baseline for this cluster, negative =
        # historically worse) -- convert to a positive multiplier, floored
        # so a bad edge is disfavored rather than made unreachable.
        return max(self.factor_floor, 1.0 + score)

    def update(self, cluster_id, edge_path, trial_error, baseline_error):
        """edge_path : list of (pos_a, pos_b) position-tuple pairs, one per
        processing-node-to-processing-node hop this trial actually took
        (the final hop into a reviewer is intentionally excluded -- factor()
        is never consulted for reviewer candidates, so crediting that edge
        would be bookkeeping nothing ever reads back)."""
        table = self.routing_factor.setdefault(cluster_id, {})
        delta = self.lr * (baseline_error - trial_error)
        for pos_a, pos_b in edge_path:
            key = (tuple(pos_a), tuple(pos_b))
            table[key] = table.get(key, 0.0) + delta

    def stats(self):
        return {
            "enabled": self.kmeans is not None,
            "n_clusters": self.n_clusters,
            "n_clusters_with_data": len(self.routing_factor),
            "total_edges_scored": sum(len(t) for t in self.routing_factor.values()),
        }
