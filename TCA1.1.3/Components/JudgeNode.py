import random
random.seed(42)  # For reproducibility in clustering
import math
import numpy as np
import Logger
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class JudgeNode:
    def __init__(self, ignored_features=None, logger=None, classification=None, target=None):
        self.segments = []
        random.seed(42)  # For reproducibility in clustering
        self.segment_weights = {
            'segment': [],
            'clusters': [],
        }
        self.Logger = logger  if logger else Logger.Logger('JudgeNode.log', 4)
        self.classification = classification
        self.ignored_features = ignored_features if ignored_features else []
        self.features = []
        self.mode = "" # special modes can be assigned later
        self.target = target

    def display(self, message, Loud, classification=None):
        message = f"[JudgeNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        self.Logger.log(message, classification if classification is not None else self.classification, Loud)

    def filter_features(self, vector):
        """
        Supports:
        - list / tuple vectors → ignores by index
        - dict vectors        → ignores by feature name
        """

        # Dict-based vector (preprocessing / pandas style)
        if isinstance(vector, dict):
            return [
                v for k, v in vector.items()
                if k not in self.ignored_features
            ]

        # List / tuple vector (numeric pipeline)
        return [
            v for i, v in enumerate(vector)
            if i not in self.ignored_features
        ]
    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def generate_clusters(self, dataset, cluster_count):
        if len(dataset) < cluster_count:
            raise ValueError("Cluster count cannot exceed dataset size.")

        # Initialize centroids randomly
        centroids = random.sample(dataset, cluster_count)

        # Points never change across EM iterations, so their filtered/numeric
        # view can be built once instead of on every one of up to 100 passes —
        # this (plus the vectorized distance+assignment below) is what removes
        # the O(iterations * n_points * n_clusters) pure-Python loop that
        # dominated JudgeNode.train()'s clustering sweep.
        dataset_f = np.array([self.filter_features(p) for p in dataset], dtype=float)

        # Run up to 100 iterations; stop early when no centroid moves more than tol.
        _KMEANS_MAX_ITER = 100
        _KMEANS_TOL = 1e-6
        for _ in range(_KMEANS_MAX_ITER):
            # Centroids are fixed for this whole pass over dataset.
            centroids_f = np.array([self.filter_features(c) for c in centroids], dtype=float)

            # Vectorized (n_points, n_clusters) distance matrix + nearest-
            # centroid assignment, replacing the per-point/per-centroid pure
            # Python loop. argmin ties break the same way as the original
            # list.index(min(...)) — both take the first occurrence.
            diffs         = dataset_f[:, None, :] - centroids_f[None, :, :]
            sq_distances  = np.sum(diffs * diffs, axis=2)
            assignments   = np.argmin(sq_distances, axis=1)

            clusters = {i: [] for i in range(cluster_count)}
            for idx, point in zip(assignments.tolist(), dataset):
                clusters[idx].append(point)

            new_centroids = []
            for points in clusters.values():
                if not points:
                    new_centroids.append(random.choice(dataset))
                    continue
                dim = len(points[0])
                centroid = [
                    sum(p[i] for p in points) / len(points)
                    for i in range(dim)
                ]
                new_centroids.append(centroid)

            # Early stop: all centroids stationary within tolerance
            max_shift = max(
                self.euclidean_distance(
                    self.filter_features(old), self.filter_features(new)
                )
                for old, new in zip(centroids, new_centroids)
            )
            centroids = new_centroids
            if max_shift < _KMEANS_TOL:
                break

        self.segment_weights['clusters'] = [
            self.cluster_create(c, clusters[i])
            for i, c in enumerate(centroids)
        ]

    def cluster_create(self, centroid: list, points: list) -> dict:
        return {
            'centroid':   centroid,
            'points':     points,
            'segment_id': None,
        }
    
    def calculate_input_segment_relevance(self, input_vectorized, Loud: bool = False) -> dict:
        relevance_scores: dict[str, list] = {
            'clusters': [],
            'scores': []
        }
        # If we have stored feature column names from training, extract values in
        # that exact order so distances align with the cluster centroid dimensions.
        # Without this, tokenize_categorical's dict ordering (categorical features
        # grouped first, then numerical) can differ from the DataFrame column order
        # used to build centroids, causing completely wrong routing.
        if self.features and isinstance(input_vectorized, dict):
            input_filtered = [
                input_vectorized.get(f, 0.0)
                for f in self.features
                if f not in self.ignored_features
            ]
        else:
            input_filtered = self.filter_features(input_vectorized)
        if not self.segment_weights['clusters']:
            self.display("No clusters available for relevance calculation. Using default override with all segments of relevance 1", Loud)
            for sid in self.segment_weights['segment']:
                relevance_scores['clusters'].append({'segment_id': sid, 'centroid': [], 'points': []})
                relevance_scores['scores'].append(1.0)
            return relevance_scores
        for cluster in self.segment_weights['clusters']:
            centroid = cluster['centroid']
            distance = self.euclidean_distance(input_filtered, centroid)
            relevance = 1 / (1 + distance)
            relevance_scores['clusters'].append(cluster)
            relevance_scores['scores'].append(relevance)
        return relevance_scores
    
    def find_relevant_segments(self, relevance_scores: dict, selection_percentage: float = 0.5, Loud: bool = False) -> list[tuple[int, float]]:
        if not relevance_scores['scores']:
            self.display("No relevant segments found. Using all segments as fallback.", Loud=Loud)
            return [(sid, 1.0) for sid in self.segment_weights['segment']]

        # Aggregate to segment level: best (max) cluster score per segment
        segment_best: dict[int, float] = {}
        for cluster, score in zip(relevance_scores['clusters'], relevance_scores['scores']):
            sid = cluster.get('segment_id')
            if sid is None:
                self.display("Cluster has no assigned segment_id. Skipping.", Loud=Loud)
                continue
            if sid not in segment_best or score > segment_best[sid]:
                segment_best[sid] = score

        if not segment_best:
            self.display("No valid segment assignments found. Using all segments as fallback.", Loud=Loud)
            return [(sid, 1.0) for sid in self.segment_weights['segment']]

        # Select top selection_percentage of SEGMENTS (not clusters)
        sorted_segs = sorted(segment_best.items(), key=lambda x: x[1], reverse=True)
        n_select = max(1, math.ceil(len(sorted_segs) * selection_percentage))
        return sorted_segs[:n_select]
    
    def geometric_uniqueness(self, cluster, all_clusters, eps=1e-6):
        centroid = self.filter_features(cluster['centroid'])

        # Distance to nearest other centroid
        distances = []
        for other in all_clusters:
            if other is cluster:
                continue
            d = self.euclidean_distance(
                centroid,
                self.filter_features(other['centroid'])
            )
            distances.append(d)

        if not distances:
            return 0.0

        nearest = min(distances)

        # Cluster radius
        if not cluster['points']:
            return 0.0

        radius = sum(
            self.euclidean_distance(
                centroid,
                self.filter_features(p)
            )
            for p in cluster['points']
        ) / len(cluster['points'])

        return nearest / (radius + eps)

    def behavioral_diversity(self, cluster, all_clusters, dataset):
        if not dataset:
            return 0.0

        X = np.array([self.filter_features(p) for p in dataset], dtype=float)
        c1 = np.array(self.filter_features(cluster['centroid']), dtype=float)
        r1 = 1.0 / (1.0 + np.linalg.norm(X - c1, axis=1))

        diffs = []
        for other in all_clusters:
            if other is cluster:
                continue
            c2 = np.array(self.filter_features(other['centroid']), dtype=float)
            r2 = 1.0 / (1.0 + np.linalg.norm(X - c2, axis=1))
            diffs.append(float(np.mean(np.abs(r1 - r2))))

        return sum(diffs) / len(diffs) if diffs else 0.0

    def information_gain(self, cluster, dataset):
        if not dataset:
            return 0.0

        size_factor = len(cluster['points']) / len(dataset)
        X = np.array([self.filter_features(p) for p in dataset], dtype=float)
        c = np.array(self.filter_features(cluster['centroid']), dtype=float)
        mean_relevance = float(np.mean(1.0 / (1.0 + np.linalg.norm(X - c, axis=1))))

        return size_factor * mean_relevance



    def assign_clusters_to_segments(self, segments: list, balance_weight: float = 1.0) -> None:
        if not segments:
            raise ValueError("Segments list is empty.")

        self.segment_weights['segment'] = [s.segment_id for s in segments]
        clusters = self.segment_weights['clusters']

        total_samples = sum(len(c['points']) for c in clusters)
        target_per_seg = total_samples / len(segments)
        centroids_f = [self.filter_features(c['centroid']) for c in clusters]

        # Normalisation factor for centroid distances
        all_dists = [
            self.euclidean_distance(centroids_f[i], centroids_f[j])
            for i in range(len(centroids_f))
            for j in range(i + 1, len(centroids_f))
        ]
        max_dist = max(all_dists) if all_dists else 1.0

        # Greedy assignment: largest clusters first so balance is easiest to achieve
        seg_state = {seg.segment_id: {'centroid': None, 'samples': 0} for seg in segments}
        for ci in sorted(range(len(clusters)), key=lambda x: len(clusters[x]['points']), reverse=True):
            c_centroid = centroids_f[ci]
            n_c = len(clusters[ci]['points'])
            best_sid, best_score = None, float('inf')
            for seg in segments:
                sid = seg.segment_id
                state = seg_state[sid]
                dist = (
                    self.euclidean_distance(c_centroid, state['centroid']) / max_dist
                    if state['centroid'] is not None else 0.0
                )
                excess_ratio = max(0.0, state['samples'] - target_per_seg) / (target_per_seg + 1e-9)
                score = dist + balance_weight * excess_ratio
                if score < best_score:
                    best_score, best_sid = score, sid
            clusters[ci]['segment_id'] = best_sid
            state = seg_state[best_sid]
            n_old = state['samples']
            n_total = n_old + n_c
            state['centroid'] = (
                list(c_centroid) if state['centroid'] is None
                else [(o * n_old + n * n_c) / n_total for o, n in zip(state['centroid'], c_centroid)]
            )
            state['samples'] = n_total
    
    def calculate_cluster_scoring(
        self,
        dataset,
        w1=1.0, #weight for geometric uniqueness
        w2=1.0, #weight for behavioral diversity
        w3=1.0  #weight for information gain
    ):
        scores = []

        clusters = self.segment_weights['clusters']

        for cluster in clusters:
            g = self.geometric_uniqueness(cluster, clusters)
            b = self.behavioral_diversity(cluster, clusters, dataset)
            i = self.information_gain(cluster, dataset)

            score = w1 * g + w2 * b + w3 * i

            cluster['metrics'] = {
                'GeometricUniqueness': g,
                'BehavioralDiversity': b,
                'InformationGain': i,
                'ClusterScore': score
            }

            scores.append(cluster)

        return scores
    def summarize_clusters(self, clusters):
        summary = []
        for i, c in enumerate(clusters):
            m = c.get('metrics', {})
            summary.append({
                "cluster": i,
                "size": len(c['points']),
                "centroid": c['centroid'],
                "GeometricUniqueness": m.get('GeometricUniqueness', 0),
                "BehavioralDiversity": m.get('BehavioralDiversity', 0),
                "InformationGain": m.get('InformationGain', 0),
                "ClusterScore": m.get('ClusterScore', 0),
            })
        return summary

    def train(self, preprocessed_dataset, iterations: int, segments: list | None = None,
              min_clusters: int | None = None, max_clusters: int | None = None):
        # Store column order so calculate_input_segment_relevance can extract
        # inference dict values in the same dimension order as these training vectors.
        self.features = list(preprocessed_dataset.columns)
        dataset_vectors = [
            list(data_point.values())
            for data_point in preprocessed_dataset.to_dict(orient='records')
        ]
        scores = []
        seg_count = len(segments) if segments is not None else 1
        _min = min_clusters if min_clusters is not None else seg_count
        _max = max_clusters if max_clusters is not None else seg_count * 5
        for i in range(iterations):
            cluster_count = _min + i  # increment by 1 each iteration
            if cluster_count > _max:
                break
            self.display(f"Iteration {i+1}/{min(iterations, _max - _min + 1)} | clusters={cluster_count} ...", True, classification=1)
            self.generate_clusters(dataset_vectors, cluster_count)
            scored_clusters = self.calculate_cluster_scoring(dataset_vectors)

            # Balance bonus: reward even distribution across clusters
            sizes = [len(c['points']) for c in scored_clusters]
            mean_sz = sum(sizes) / len(sizes)
            std_sz = math.sqrt(sum((s - mean_sz) ** 2 for s in sizes) / len(sizes))
            cv = std_sz / (mean_sz + 1e-9)
            balance_bonus = 1.0 / (1.0 + cv)
            raw_score = sum(c['metrics']['ClusterScore'] for c in scored_clusters) / len(scored_clusters)
            combined_score = raw_score * (1.0 + balance_bonus)
            self.display(f"  -> score={combined_score:.4f} (raw={raw_score:.4f}, balance_bonus={balance_bonus:.4f})", True, classification=1)

            scores.append({
                "cluster_count": cluster_count,
                "avg_score": combined_score,
                "raw_score": raw_score,
                "balance_bonus": balance_bonus,
                "summary": self.summarize_clusters(scored_clusters)
            })
        scores.sort(key=lambda x: x["avg_score"], reverse=True)

        best = scores[0]
        self.display(
            f"Optimal clusters: {best['cluster_count']} | score={best['avg_score']:.4f} (raw={best['raw_score']:.4f}, balance={best['balance_bonus']:.4f})",
            True, classification=1
        )
        self.display("Cluster summaries:", True)
        for c in best['summary']:
            self.display(
                f"Cluster {c['cluster']}: size={c['size']} | GeometricUniqueness={c['GeometricUniqueness']:.4f} | BehavioralDiversity={c['BehavioralDiversity']:.4f} | InformationGain={c['InformationGain']:.4f} | ClusterScore={c['ClusterScore']:.4f}",
                True
            )
        self.display("Summary for each cluster amount:", True)
        for s in scores:
            self.display(
                f"Clusters: {s['cluster_count']} | Avg ClusterScore: {s['avg_score']:.4f} (raw={s['raw_score']:.4f}, balance={s['balance_bonus']:.4f})",
                True
            )
            for c in s['summary']:
                self.display(
                    f"  Cluster {c['cluster']}: size={c['size']} | GeometricUniqueness={c['GeometricUniqueness']:.4f} | BehavioralDiversity={c['BehavioralDiversity']:.4f} | InformationGain={c['InformationGain']:.4f} | ClusterScore={c['ClusterScore']:.4f}",
                    True
                )
        self.display("Generating final clusters based on optimal cluster count...", True, classification=1)
        self.generate_clusters(dataset_vectors, best['cluster_count'])
        self.calculate_cluster_scoring(dataset_vectors)
        if segments is not None:
            self.assign_clusters_to_segments(segments)
        

def plot_clusters_2d(clusters):
    # Collect all points
    all_points: list[list[float]] = []
    labels: list[int] = []

    for i, c in enumerate(clusters):
        for p in c['points']:
            all_points.append(p)
            labels.append(i)

    if not all_points:
        return

    # Reduce to 2D for visualization
    pca = PCA(n_components=2)

    X = np.asarray(all_points, dtype=float)
    reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for i in set(labels):
        xs = [reduced[j, 0] for j in range(len(labels)) if labels[j] == i]
        ys = [reduced[j, 1] for j in range(len(labels)) if labels[j] == i]
        plt.scatter(xs, ys, label=f"Cluster {i}", alpha=0.6)

    plt.legend()
    plt.title("Cluster Geometry (PCA projection)")
    plt.show()

def plot_cluster_scores(clusters):
    names = [f"C{i}" for i in range(len(clusters))]
    g = [c['metrics']['GeometricUniqueness'] for c in clusters]
    b = [c['metrics']['BehavioralDiversity'] for c in clusters]
    i = [c['metrics']['InformationGain'] for c in clusters]

    x = range(len(clusters))
    plt.figure(figsize=(10, 5))
    plt.bar(x, g, label="Geometric")
    plt.bar(x, b, bottom=g, label="Behavioral")
    plt.bar(x, i, bottom=[g[j]+b[j] for j in x], label="Info")

    plt.xticks(x, names)
    plt.legend()
    plt.title("ClusterScore Composition")
    plt.show()