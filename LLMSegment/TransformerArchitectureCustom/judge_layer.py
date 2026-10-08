"""
Formalizes TCA's JudgeNode as an explicit, traceable layer-level step,
rather than a helper buried inside whichever consumer (e.g.
SparseMoEFeedForward) happens to use it. Computes "which training-
distribution cluster does this token belong to" ONCE per block, so
multiple consumers (the FFN-gate today, an attention-gate later --
see README "Next steps") can share the same read instead of each
maintaining its own redundant DistributionJudge fit on the same
vectors.

Deliberately holds ONLY the novelty-detection machinery (querying the
attached DistributionJudge). It does NOT hold any learned/credit-updated
preference state itself -- that stays with each CONSUMER (e.g.
SparseMoEFeedForward.judge_bias_table), since different consumers may
want different preferences for the same cluster (an FFN-expert
preference for cluster 3 isn't necessarily the same as a future
attention-expert preference for cluster 3).
"""
import numpy as np


class JudgeLayer:
    def __init__(self):
        self.judge = None

    def attach(self, judge):
        self.judge = judge

    def forward(self, x, trace=None):
        """
        x: (batch, seq, dim). Returns cluster_ids: (batch, seq) int array
        (-1 = noise/outlier), or None if no judge attached/fitted yet --
        callers (e.g. SparseMoEFeedForward) treat None as "no judge
        signal available", identical to today's behavior with nothing
        attached.
        """
        if self.judge is None or not self.judge._fitted:
            return None
        b, s, d = x.shape
        flat_x = x.reshape(-1, d)
        _, nearest_label, nearest_dist = self.judge.query_batch(flat_x)
        cluster_ids = nearest_label.reshape(b, s)

        if trace is not None:
            unique, counts = np.unique(cluster_ids, return_counts=True)
            trace.record("judge", {
                "cluster_histogram": {int(c): int(n) for c, n in zip(unique, counts)},
                "mean_novelty_distance": float(nearest_dist.mean()),
            })

        return cluster_ids
