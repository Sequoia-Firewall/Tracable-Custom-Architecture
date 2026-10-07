"""
Standalone test for DistributionJudge -- isolated from the transformer
entirely, synthetic data with a known ground truth. Confirms the
clustering/novelty/attribution logic is sane BEFORE wiring it to real
model activations, where "is this behaving correctly" would be much
harder to tell apart from "is the model just doing something unexpected."
"""
import numpy as np
from judge import DistributionJudge

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    all_pass = True

    # Three well-separated, tight 2D blobs -- ground truth: DBSCAN should
    # find exactly 3 clusters, no noise, since points are tight (std=0.1)
    # relative to the gap between blobs (10 units).
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    n_per_blob = 40
    blobs = np.concatenate([
        c + rng.normal(0, 0.1, size=(n_per_blob, 2)) for c in centers
    ])

    judge = DistributionJudge(eps=1.0, min_samples=5)
    judge.fit(blobs)

    summary = judge.cluster_population_summary()
    print(f"cluster population: {summary}")
    n_real_clusters = sum(1 for k in summary if k != -1)
    noise_count = summary.get(-1, 0)
    ok = n_real_clusters == 3 and noise_count == 0
    print(f"[{'PASS' if ok else 'FAIL'}] found 3 clean clusters, no noise")
    all_pass &= ok

    # A query point right at the center of blob 0 -- should match blob 0's
    # label with a small distance.
    in_dist_point = np.array([0.0, 0.05])
    result = judge.query(in_dist_point)
    label0 = judge.labels_[0]  # first blob's cluster label
    ok = result["nearest_label"] == label0 and result["distance"] < 0.5
    print(f"[{'PASS' if ok else 'FAIL'}] in-distribution point: {result} (expected label={label0}, small distance)")
    all_pass &= ok

    # A query point FAR from every blob -- the whole point of this module:
    # confirm large distance is reported (DBSCAN doesn't have to label the
    # NEAREST TRAINING POINT as noise for this to work -- novelty is read
    # from the distance, not solely the label).
    ood_point = np.array([100.0, 100.0])
    result = judge.query(ood_point)
    in_dist_dist = judge.query(in_dist_point)["distance"]
    ok = result["distance"] > 50 * max(in_dist_dist, 1e-6)
    print(f"[{'PASS' if ok else 'FAIL'}] out-of-distribution point: distance={result['distance']:.2f} "
          f"(vs in-distribution ~{in_dist_dist:.4f}) -- novelty signal separates cleanly")
    all_pass &= ok

    # Attribution: nearest_training_examples should return points genuinely
    # near the query, in increasing distance order.
    neighbors = judge.nearest_training_examples(in_dist_point, n=3)
    dists = [n["distance"] for n in neighbors]
    ok = dists == sorted(dists) and all(d < 0.5 for d in dists)
    print(f"[{'PASS' if ok else 'FAIL'}] nearest_training_examples sorted & close: {[f'{d:.3f}' for d in dists]}")
    all_pass &= ok

    # cluster_members should only return members actually labeled with that cluster.
    members = judge.cluster_members(label0, max_n=5)
    ok = all(judge.labels_[i] == label0 for i in members) and len(members) == 5
    print(f"[{'PASS' if ok else 'FAIL'}] cluster_members returns only matching-label points: {members}")
    all_pass &= ok

    # A point genuinely between two blobs, far from both (sparse region) --
    # expect DBSCAN's nearest point to still resolve, but at a distance that
    # reflects real sparsity, bigger than a true in-cluster point's.
    between_point = np.array([5.0, 5.0])
    result = judge.query(between_point)
    print(f"[INFO] between-blobs point: nearest_label={result['nearest_label']}, "
          f"distance={result['distance']:.2f}")

    # query_batch should agree with calling query() one at a time.
    batch = np.stack([in_dist_point, ood_point, between_point])
    idx_b, label_b, dist_b = judge.query_batch(batch)
    single_results = [judge.query(v) for v in batch]
    ok = (
        list(idx_b) == [r["nearest_index"] for r in single_results]
        and list(label_b) == [r["nearest_label"] for r in single_results]
        and np.allclose(dist_b, [r["distance"] for r in single_results])
    )
    print(f"[{'PASS' if ok else 'FAIL'}] query_batch agrees with per-point query()")
    all_pass &= ok

    print("\n" + ("ALL JUDGE SMOKE TESTS PASSED" if all_pass else "SOME JUDGE SMOKE TESTS FAILED"))
    assert all_pass
