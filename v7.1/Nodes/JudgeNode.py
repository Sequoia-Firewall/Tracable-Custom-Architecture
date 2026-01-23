"""
- Judge Node:
 -- will compute 1 different types of predictions. 
 -- will predict segment relevance: 
 --- using values of input data, will use an algorithm to compute the relevance of different segments
 --- Will likely use some kind of unsupervised learning or clustering algorithm to determine which segments are most relevant to the input data
 ---- likely some kind of K means clustering
 --- will activate the top 50% of relevant segments
 --- segment relevance scores to be used later in handler to compute final prediction
 -- will route data to the relevant segments and their splitter for further processing
 -- integrated ML for segment relevance calculations
"""

from random import sample
import math

class JudgeNode:
    def __init__(self, dimensions, logging_enabled=False, logger=None):
        self.logging_enabled = logging_enabled
        self.segments = []  # connected segments (SplitterNodes)
        self.segment_weights = []  # segment_id → relevance score
        self.features = []  # list of feature names
        self.dimensions = dimensions # amount of clusters needed
        self.clusters = []  # clusters generated from training data
        self.cluster_centroids = []
        self.segment_cluster_map = {}  # segment_id → cluster_id
        self.labels_ = []  # cluster labels for training data points
        self.logger = logger

    def display(self, message, Loud):
        if self.logging_enabled:
            if self.logger is None:
                raise ValueError("Logger is not set for JudgeNode.")
            message = (f"[Judge Node] {message}")
            self.logger.log(message, Loud= Loud)
    def vector_to_feature_dict(self, vector):
        return dict(zip(self.features, vector))
    def calculate_segment_relevance(self, input_data, Loud):
        self.display("Calculating segment relevance.", Loud= Loud)


        vectorized_input = self.vectorize_input(input_data, Loud = Loud)

        segment_relevance = {}
        for segment in self.segments:
            seg_id = segment['id']

            vectorized_input = self.vectorize_input(input_data, Loud = Loud)
            cluster_relevance = self.compute_cluster_relevance(vectorized_input, Loud = Loud)

            segment_relevance = {}
            for segment in self.segments:
                seg_id = segment['id']
                cid = self.segment_cluster_map.get(seg_id, None)
                if cid is not None:
                    # If cid is a list, use the first element
                    if isinstance(cid, list):
                        if len(cid) > 0:
                            segment_relevance[seg_id] = cluster_relevance.get(cid[0], 0.0)
                        else:
                            segment_relevance[seg_id] = 0.0
                    else:
                        segment_relevance[seg_id] = cluster_relevance.get(cid, 0.0)
                else:
                    segment_relevance[seg_id] = 0.0

        self.segment_weights = segment_relevance
        self.display(f"Segment relevance: {segment_relevance}", Loud= Loud)
        return segment_relevance
    
    
    def select_segments(self, segment_relevance, Loud):
        self.display("Selecting top 50% relevant segments.", Loud= Loud)
        if not segment_relevance:
            self.display("No segment weights calculated; cannot select segments.", Loud= Loud)
            return []
        
        sorted_segments = sorted(segment_relevance.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, len(sorted_segments) // 2)
        selected_segments = [self.segments[idx] for idx, _ in sorted_segments[:top_count]]
        
        self.display(f"Selected segments: {[segment['splitter'].position for segment in selected_segments]}", Loud= Loud)
        return selected_segments
    
    def vectorize_input(self, input_data, Loud):
        self.display(f"Vectorizing input data: {input_data}.", Loud= Loud)
        if not self.features:
            self.features = [k for k in input_data.keys() if k != "exam_score" and k != "student_id"]
        vectored = [input_data[f] if not isinstance(input_data[f], tuple)
                else input_data[f][0]  # take mean from evidence triple
                for f in self.features]
        self.display(f"Vectorized input: {vectored}.", Loud= Loud)
        return vectored
    
    def generate_clusters(self, training_vectors, Loud, k = None, ):
        self.display("Generating clusters from training data.", Loud= Loud)
        
        if k is None:
            k = 2 ** self.dimensions
        self.display(f"Generating {k} clusters.", Loud= Loud)
        # --- Initialize centroids (classic k-means) ---
        centroids = sample(training_vectors, k)

        for _ in range(20):  # fixed small iterations
            clusters = {i: [] for i in range(k)}
            labels = []
            for vec in training_vectors:
                distances = [
                    math.dist(vec, c) for c in centroids
                ]
                cid = distances.index(min(distances))
                clusters[cid].append(vec)
                labels.append(cid)

            for cid in clusters:
                if clusters[cid]:
                    centroids[cid] = [
                        sum(dim) / len(dim)
                        for dim in zip(*clusters[cid])
                    ]

        self.cluster_centroids = centroids
        self.labels_ = labels
        
        
        self.display("Clusters generated.", Loud= Loud)

    def assign_segments_to_clusters(self, Loud):
        self.display("Assigning segments to clusters.", Loud= Loud)

        if not self.cluster_centroids or not self.segments:
            self.display("No clusters or segments available.", Loud= Loud)
            return

        num_clusters = len(self.cluster_centroids)
        num_segments = len(self.segments)

        self.segment_cluster_map = {}

        # --- Pass 1: guarantee every cluster has at least one segment ---
        for cluster_id in range(num_clusters):
            seg_index = cluster_id % num_segments
            seg_id = self.segments[seg_index]['id']
            self.segment_cluster_map.setdefault(seg_id, []).append(cluster_id)

        # --- Pass 2: assign remaining segments (if any) ---
        for i, segment in enumerate(self.segments):
            seg_id = segment['id']
            if seg_id not in self.segment_cluster_map:
                cluster_id = i % num_clusters
                self.segment_cluster_map[seg_id] = [cluster_id]

        self.display(f"Segment-cluster map: {self.segment_cluster_map}", Loud= Loud)
        
    
    def compute_cluster_relevance(self, input_vector, Loud, alpha=1.0, ):
        if not self.cluster_centroids:
            self.display("No cluster centroids available; cannot compute relevance.", Loud= Loud)
            return {}
        distances = [
            math.dist(input_vector, c)
            for c in self.cluster_centroids
        ]

        scores = [
            math.exp(-alpha * d) for d in distances
        ]

        total = sum(scores)
        return {
            i: s / total for i, s in enumerate(scores)
        }
    
    def compute_inertia(self, training_vectors, Loud):
        inertia = 0.0
        if not self.cluster_centroids or not self.labels_:
            self.display("No cluster centroids or labels available; cannot compute inertia.", Loud= Loud)
            return float("inf")
        for vec, label in zip(training_vectors, self.labels_):
            centroid = self.cluster_centroids[label]
            inertia += math.dist(vec, centroid) ** 2
        return inertia
    
    def train(self, training_vectors, Loud, max_amount_scaleups=5):
        self.display("Training JudgeNode to determine best cluster scale.", Loud= Loud)

        best_inertia = float("inf")
        best_centroids = None
        best_labels = None
        best_k = None

        base_k = 2 ** self.dimensions

        for scaleup in range(max_amount_scaleups):
            k = base_k * (scaleup + 1)
            self.display(f"Training with {k} clusters.", Loud= Loud)

            self.generate_clusters(training_vectors, k=k, Loud= Loud)
            inertia = self.compute_inertia(training_vectors, Loud= Loud)

            self.display(f"Inertia for k={k}: {inertia}", Loud= Loud)

            if inertia < best_inertia and self.cluster_centroids is not None and self.labels_ is not None:
                best_inertia = inertia
                best_centroids = list(self.cluster_centroids)
                best_labels = list(self.labels_)
                best_k = k

        # Lock in best configuration
        self.cluster_centroids = best_centroids
        self.labels_ = best_labels

        self.display(f"Selected best k={best_k} with inertia={best_inertia}", Loud= Loud)

    
    
    def save_cluster_map(self, training_vectors, segment_assignments, Loud):
        """
        training_vectors: list[list[float]]
        segment_assignments: dict {segment_id: cluster_index}
        """

        import os
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from sklearn.decomposition import PCA

        os.makedirs("stats", exist_ok=True)

        # --- Reduce to 2D for visualization ---
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(training_vectors)

        # --- Build cluster → points mapping ---
        cluster_points = defaultdict(list)
        if not self.labels_:
            self.display("No cluster labels available; cannot save cluster map.", Loud= Loud)
            return
        for point, label in zip(points_2d, self.labels_):
            cluster_points[label].append(point)

        # --- Plot ---
        plt.figure(figsize=(10, 8))
        for cluster_id, pts in cluster_points.items():
            pts = list(zip(*pts))
            plt.scatter(
                pts[0],
                pts[1],
                label=f"Cluster {cluster_id}",
                alpha=0.6
            )

        # --- Legend: cluster → segment ---
        legend_lines = []
        for seg_id, cluster_id in segment_assignments.items():
            legend_lines.append(f"Segment {seg_id} → Cluster {cluster_id}")

        legend_text = "\n".join(legend_lines)
        plt.gca().text(
            1.02,
            0.5,
            legend_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="center"
        )

        plt.title("Cluster Map (Training Data)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(loc="upper left")
        plt.tight_layout()

        path = "stats/cluster_map.jpg"
        plt.savefig(path, dpi=300)
        plt.close()

        self.display(f"Cluster map saved to {path}", Loud= Loud)