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
    def __init__(self, position, logging_enabled=False):
        self.logging_enabled = logging_enabled
        self.position = position
        self.processing_nodes = []  # connected ProcessingNodes
        self.feature_relevance = []
        self.signal = None
    def __repr__(self) -> str:
        return f"SplitterNode(pos={self.position})"
    
    def display(self, message):
        if self.logging_enabled:
            print(f"[SplitterNode] {message}")

    def connect_processing_nodes(self, all_nodes):
        self.display("Connecting to nearest processing nodes.")
        distances = []
        for node in all_nodes:
            dist = sum([(a - b) ** 2 for a, b in zip(self.position, node.position)]) ** 0.5
            distances.append((dist, node))
        distances.sort(key=lambda x: x[0])
        num_neighbors = max(1, int(len(distances) * 0.05))
        neighbors = [node for _, node in distances[:num_neighbors]]
        self.processing_nodes = neighbors
        self.display(f"Connected processing nodes: {self.processing_nodes}")

    def generate_signal(self, segment_relevance, feature_relevance, node_count, inputdata):
        self.display("Generating new signal for processing nodes.")
        signal = None

        signal = Signal(
            segment_relevance_score=segment_relevance,
            feature_relevance_scores=feature_relevance,
            life = round(node_count * .8),
            input_data=inputdata
        )
           
        self.display(f"Generated signal for nodes: {signal}")
        return signal
    
    def reset(self):
        self.display("Resetting SplitterNode state.")
        self.signal = None
    
    def calculate_feature_relevance(self, input):
        self.display("Calculating feature relevance.")
        # Placeholder: In real implementation, use ML model to calculate feature relevance
        # Ensure output is a dict: {feature_name: relevance_score}
        feature_relevance = {}
        for feature in input:
            # Dummy relevance score, replace with ML output as needed
            feature_relevance[feature] = 1.0
        self.feature_relevance = feature_relevance
        self.display(f"Calculated feature relevance: {feature_relevance}")
        return feature_relevance
    
    def forward_signals(self):
        if self.signal is None:
            self.display("No signal to forward.")
            return False
        node_count = 0
        signal_clones = []
        for node in self.processing_nodes:
            signal_clone = self.signal.clone()
            received = node.receive_signal(signal_clone)
            if received:
                node_count += 1
                signal_clones.append(signal_clone)

        self.display(f"Signals forwarded to {node_count} processing nodes.")
        self.reset()
        return node_count, signal_clones
    