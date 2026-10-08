"""
Periodic reclustering for a JudgeLayer, adapted from DCST section 4.7 --
scoped deliberately to the MoE-bias Judge, not the confidence Judge (see
README: the confidence Judge wants to describe the FINAL, stable model;
the MoE-bias Judge actively shapes training, same genre as DCST's
problem, so it's the one that should move).

Operates on a JudgeLayer (shared cluster-computation infrastructure, see
judge_layer.py) plus a list of "bias consumers" -- objects with their own
judge_bias_table/cluster_running_loss/n_experts (today just a single
SparseMoEFeedForward; the split exists so a future attention-expert
consumer sharing the same JudgeLayer gets its OWN warm-started table
rather than clobbering the FFN's).

Two explicit simplifications from DCST's literal recipe, stated here
rather than silently assumed:
  - Nearest-CENTROID matching between old and new clusters, not the
    Hungarian algorithm (optimal assignment) -- cheaper, same spirit
    (carry bias values to whichever new cluster most resembles each old
    one), not guaranteed optimal for ambiguous cases. Known limitation,
    confirmed in judge_reclustering_test.py: there's no distance cutoff,
    so a genuinely NOVEL cluster (nothing old really corresponds to it)
    still gets matched to whichever old centroid happens to be nearest,
    inheriting a stale/irrelevant bias rather than starting fresh. The
    credit-update mechanism keeps adjusting from there regardless, so
    this is a quality-of-warm-start issue, not a correctness one -- a
    distance-thresholded match would fix it if it matters in practice.
  - No literal learning-rate warmup after an applied refit (we don't have
    an LR schedule to hook into) -- instead, cluster_running_loss (the
    credit-update baseline) is reset for every touched cluster, so the
    credit system re-learns a fresh baseline under the new mapping rather
    than comparing against a stale one. Judge_bias_table values ARE
    carried over (warm-started), matching DCST's intent.
"""
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from judge import DistributionJudge


def _centroids(judge):
    """{label: mean vector} for every real (non-noise) cluster."""
    out = {}
    for label in judge.cluster_population_summary():
        if label == -1:
            continue
        mask = judge.labels_ == label
        out[label] = judge.training_vectors[mask].mean(axis=0)
    return out


def _nearest_centroid_match(old_centroids, new_centroids):
    """new_cid -> old_cid, each new cluster matched to its closest old one
    by centroid distance. Not necessarily a bijection."""
    if not old_centroids:
        return {}
    old_ids = list(old_centroids.keys())
    old_mat = np.stack([old_centroids[i] for i in old_ids])
    match = {}
    for new_cid, new_centroid in new_centroids.items():
        dists = np.linalg.norm(old_mat - new_centroid[None, :], axis=-1)
        match[new_cid] = old_ids[int(np.argmin(dists))]
    return match


def maybe_refit_judge(judge_layer, bias_consumers, probe_vectors, eps, min_samples,
                       min_ami_drop=0.05, rng_seed=None):
    """
    Fits a candidate Judge on probe_vectors; applies it (replacing
    judge_layer.judge, warm-starting each consumer's judge_bias_table via
    nearest-centroid matching) only if the new clustering's structure has
    actually diverged from the old one -- DCST's stability guard against
    thrashing on noise, adapted to detect real structural change rather
    than relabeling.

    judge_layer: a JudgeLayer (see judge_layer.py). bias_consumers: list
    of objects each with judge_bias_table/cluster_running_loss/n_experts
    (e.g. a block's SparseMoEFeedForward) -- each gets its OWN warm-started
    table, since different consumers may prefer different experts for the
    same cluster.

    Change detection uses Adjusted Mutual Information between the OLD
    judge's labels and the NEW judge's labels, both queried against the
    SAME probe_vectors -- NOT a naive "does old_judge's nearest-neighbor
    label match new_judge's centroid-matched label" comparison. That
    naive version has a real blind spot: when a genuinely NEW cluster
    appears, old_judge's nearest-neighbor fallback for those points and
    the new cluster's nearest-OLD-centroid match can both independently
    land on the same nearby old cluster, making a real structural change
    look like zero change. AMI compares the two full label PARTITIONS
    (independent of the specific integers used), which doesn't have that
    asymmetry -- caught via judge_reclustering_test.py's "new 4th blob
    appears" case, which the naive version silently failed to detect.

    Returns a dict describing what happened: {'applied': bool,
    'ami': float or None (None on the very first fit, where there's
    nothing to compare against -- AMI near 1.0 means "structurally the
    same partition", lower means real divergence), 'n_clusters': int}.
    """
    candidate = DistributionJudge(eps=eps, min_samples=min_samples)
    candidate.fit(probe_vectors)

    if judge_layer.judge is None or not judge_layer.judge._fitted:
        judge_layer.attach(candidate)
        return {"applied": True, "ami": None, "n_clusters": len(candidate.cluster_population_summary())}

    old_judge = judge_layer.judge
    _, old_labels, _ = old_judge.query_batch(probe_vectors)
    _, new_labels, _ = candidate.query_batch(probe_vectors)

    ami = float(adjusted_mutual_info_score(old_labels, new_labels))
    if ami > (1.0 - min_ami_drop):
        return {"applied": False, "ami": ami,
                "n_clusters": len(candidate.cluster_population_summary())}

    old_centroids = _centroids(old_judge)
    new_centroids = _centroids(candidate)
    match = _nearest_centroid_match(old_centroids, new_centroids)  # new_cid -> old_cid, used ONLY
    # for warm-starting bias values below -- a heuristic carry-over, not the change-detection signal.

    for consumer in bias_consumers:
        old_table = consumer.judge_bias_table
        carried_table = {}
        for new_cid in candidate.cluster_population_summary():
            old_cid = match.get(new_cid, None)
            carried_table[new_cid] = old_table.get(old_cid, np.zeros(consumer.n_experts)).copy() \
                if old_cid is not None else np.zeros(consumer.n_experts)
            # cluster_running_loss is intentionally NOT carried (see module docstring) --
            # fresh baseline under the new mapping, bias values ARE carried.
        consumer.judge_bias_table = carried_table
        consumer.cluster_running_loss = {}

    judge_layer.judge = candidate

    return {"applied": True, "ami": ami,
            "n_clusters": len(candidate.cluster_population_summary())}
