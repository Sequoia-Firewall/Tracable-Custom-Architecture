"""
DistributionJudge: TCA's JudgeNode concept, repurposed. TCA's JudgeNode
clusters the input feature space to pick which SEGMENT handles a sample --
a router. This clusters the model's own last-token final hidden states
(the exact vector the output head predicts from) across the training set,
and repurposes cluster membership as a NOVELTY signal at inference time:
how far is this input from anything the model actually trained on, not
which sub-component should handle it.

Fit once, post-training (mirroring ArchetypeRouter's fit-after-the-fact
pattern in TCA) -- representations drift during training, so fitting
mid-training would describe a model that no longer exists by the time
anything queries it.

Three things this is meant to support (see judge_grad_check/judge_smoke
tests, and TransformerArchitectureCustom/README.md):
  1. Bias visibility: cluster_population_summary() is a direct report of
     what the training set was concentrated on in activation space.
  2. Attribution: cluster_members() / nearest_training_examples() let you
     ask "what training inputs does this one's activation most resemble."
  3. A novelty FEATURE (distance + cluster/noise flag) to feed a trainable
     confidence estimator -- NOT assumed to help, tested in
     judge_confidence_test.py.
"""
import numpy as np
from sklearn.cluster import DBSCAN


class DistributionJudge:
    def __init__(self, eps=1.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.training_vectors = None   # (N, dim)
        self.labels_ = None            # (N,) int, -1 = noise
        self.sample_meta = None        # optional list, len N, caller-defined
        self._fitted = False

    def fit(self, vectors, sample_meta=None):
        """
        vectors: (N, dim) array -- one last-token final-hidden-state vector
        per training sequence. sample_meta: optional list of length N
        (e.g. the raw token sequences, or dataset indices) purely for
        attribution lookups -- never used by DBSCAN itself.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(vectors)

        self.training_vectors = vectors
        self.labels_ = labels
        self.sample_meta = sample_meta
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("DistributionJudge.fit() must be called before query()")

    def query(self, vector):
        """
        vector: (dim,) -- a single inference-time last-token hidden state.
        Returns dict: nearest_index, nearest_label (-1 if that training
        point was itself noise), distance (to the nearest training point).
        This is deliberately nearest-NEIGHBOR distance, not
        distance-to-centroid -- DBSCAN clusters aren't guaranteed convex,
        so "distance to centroid" can understate novelty for a point near
        a cluster's concave edge.
        """
        self._check_fitted()
        vector = np.asarray(vector, dtype=np.float64)
        diffs = self.training_vectors - vector[None, :]
        dists = np.linalg.norm(diffs, axis=-1)
        nearest_index = int(np.argmin(dists))
        return {
            "nearest_index": nearest_index,
            "nearest_label": int(self.labels_[nearest_index]),
            "distance": float(dists[nearest_index]),
        }

    def query_batch(self, vectors):
        """Same as query() but vectorized over (batch, dim) -- used when
        tracing/evaluating many inputs at once rather than one at a time."""
        self._check_fitted()
        vectors = np.asarray(vectors, dtype=np.float64)
        # (batch, N) pairwise distances -- fine at toy/test scale; a real
        # deployment with a large training set would want a KD-tree here.
        diffs = vectors[:, None, :] - self.training_vectors[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        nearest_idx = np.argmin(dists, axis=-1)
        nearest_dist = dists[np.arange(len(vectors)), nearest_idx]
        nearest_label = self.labels_[nearest_idx]
        return nearest_idx, nearest_label, nearest_dist

    def cluster_population_summary(self):
        """{cluster_id: count}, including -1 for noise -- a direct report
        of what the training set's activations were concentrated on."""
        self._check_fitted()
        ids, counts = np.unique(self.labels_, return_counts=True)
        return dict(zip(ids.tolist(), counts.tolist()))

    def cluster_members(self, cluster_id, max_n=5):
        """Indices (into the fitted training set) of up to max_n members of
        the given cluster -- attribution: 'what does training data in this
        cluster actually look like.'"""
        self._check_fitted()
        idx = np.where(self.labels_ == cluster_id)[0]
        return idx[:max_n].tolist()

    def nearest_training_examples(self, vector, n=3):
        """n nearest training points to `vector` (any label), with
        distances -- the direct 'what does this inference input's
        activation most resemble' attribution query."""
        self._check_fitted()
        vector = np.asarray(vector, dtype=np.float64)
        dists = np.linalg.norm(self.training_vectors - vector[None, :], axis=-1)
        order = np.argsort(dists)[:n]
        return [
            {
                "index": int(i),
                "label": int(self.labels_[i]),
                "distance": float(dists[i]),
                "meta": self.sample_meta[i] if self.sample_meta is not None else None,
            }
            for i in order
        ]
