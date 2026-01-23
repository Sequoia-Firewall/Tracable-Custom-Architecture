"""
- splitter node:
 -- 2 primary tasks
 -- will predict feature relevance:
 --- to be used like attention
 --- impacts processing node calculations
 -- will route new Signals to nearest 5% of processing nodes. (each get a unique signal)
 --- euclidean distance formula based on dimensions
 -- Integrated ML for feature relevance calculations

"""

import math
from .Signal import Signal as Signal
class SplitterNode:
    def __init__(self, position, logging_enabled=False, logger = None):
        self.logging_enabled = logging_enabled
        self.position = position
        self.processing_nodes = []  # connected ProcessingNodes
        self.feature_relevance = []
        self.signal = None
        self.logger = logger
        self.feature_relations = {}
        self.relation_lr = .01

    def __repr__(self) -> str:
        return f"SplitterNode(pos={self.position})"
    
    def display(self, message, Loud):
        if self.logging_enabled:
            if self.logger is None:
                raise ValueError("Logger is not set for SplitterNode.")
            message = (f"[Splitter Node] {message}")
            self.logger.log(message, Loud)

    def connect_processing_nodes(self, all_nodes, Loud):
        self.display("Connecting to nearest processing nodes.", Loud= Loud)
        distances = []
        for node in all_nodes:
            dist = sum([(a - b) ** 2 for a, b in zip(self.position, node.position)]) ** 0.5
            distances.append((dist, node))
        distances.sort(key=lambda x: x[0])
        num_neighbors = max(1, int(len(distances) * 0.05))
        neighbors = [node for _, node in distances[:num_neighbors]]
        self.processing_nodes = neighbors
        self.display(f"Connected processing nodes: {self.processing_nodes}", Loud= Loud)

    def generate_signal(self, segment_relevance, feature_relevance, node_count, inputdata, Loud, mean = .5):
        self.display("Generating new signal for processing nodes.", Loud= Loud)
        signal = None

        signal = Signal(
            segment_relevance_score=segment_relevance,
            feature_relevance_scores=feature_relevance,
            life = round(node_count * .9),
            input_data=inputdata,
            mean=mean
        )
           
        self.display(f"Generated signal for nodes: {signal}", Loud= Loud)
        return signal
    
    def reset(self, Loud):
        self.display("Resetting SplitterNode state.", Loud= Loud)
        self.signal = None
    
    def calculate_feature_relevance(self, input, Loud):
        self.display("Calculating feature relevance with relational attention.", Loud= Loud)
        feature_relevance = {}

        for fi in input:
            xi = input[fi]
            xi = xi[0] if isinstance(xi, tuple) else xi

            # Base relevance
            score = 1.0

            # Relational modulation
            for fj in input:
                if fi == fj:
                    continue

                xj = input[fj]
                xj = xj[0] if isinstance(xj, tuple) else xj

                relation = self.feature_relations.get((fi, fj), 0.0)
                score += relation * xj

            feature_relevance[fi] = max(0.0, score)

        # Normalize (soft attention)
        total = sum(feature_relevance.values()) + 1e-8
        for f in feature_relevance:
            feature_relevance[f] /= total

        self.feature_relevance = feature_relevance
        self.display(f"Calculated feature relevance: {feature_relevance}", Loud= Loud)
        return feature_relevance
    
    def forward_signals(self, Loud):
        if self.signal is None:
            self.display("No signal to forward.", Loud= Loud)
            return False
        node_count = 0
        signal_clones = []
        for node in self.processing_nodes:
            signal_clone = self.signal.clone()
            received = node.receive_signal(signal_clone, Loud= Loud)
            if received:
                node_count += 1
                signal_clones.append(signal_clone)

        self.display(f"Signals forwarded to {node_count} processing nodes.", Loud= Loud)
        self.reset(Loud= Loud)
        return node_count, signal_clones
    
    def update_feature_relations(self, input_data, residual, Loud):
        features = list(input_data.keys())

        for i, fi in enumerate(features):
            xi = input_data[fi]
            xi = xi[0] if isinstance(xi, tuple) else xi

            for j, fj in enumerate(features):
                if i == j:
                    continue

                xj = input_data[fj]
                xj = xj[0] if isinstance(xj, tuple) else xj

                key = (fi, fj)

                # Hebbian × residual correlation
                delta = self.relation_lr * residual * xi * xj

                self.feature_relations[key] = (
                    self.feature_relations.get(key, 0.0) + delta
                )

    