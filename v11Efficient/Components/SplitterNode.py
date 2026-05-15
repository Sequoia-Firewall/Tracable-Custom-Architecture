from Components.Signal import Signal
class SplitterNode:
    def __init__(self, position, connection_percentage, Logger=None, classification=4):
        self.position = position
        self.connection_percentage = connection_percentage
        self.signal_weights = {}
        self.connected_nodes = []
        self.Logger = Logger
        self.classification = classification
        self.feature_relevance_gradients = {}
    
    def __repr__(self) -> str:
        return f"Splitter(pos={self.position})"
    
    def display(self, message):
        message = f"[SplitterNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        self.Logger.log(message, self.classification, True)
        

    def calculate_nearest_neighbors(self, nodes):
        distances = {}
        self.connected_nodes = []
        for node in nodes:
            for _ in self.position:
                distance = sum((p1 - p2) ** 2 for p1, p2 in zip(self.position, node.position)) ** 0.5
            distances[node] = distance
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        
        while len(self.connected_nodes) < len(nodes) * self.connection_percentage:
            nearest_node = next(iter(sorted_distances))
            self.connected_nodes.append(nearest_node)
            del sorted_distances[nearest_node]
        return self.connected_nodes
    
    def initialize_signal_weights(self, input_data):
        for feature in input_data:
            self.signal_weights[feature] = 1.0  # Initial weight of 1.0 for each feature

    
    def generate_signals(self, input_data, max_x, feature_relevance):
        signals = []
        for _ in self.connected_nodes:
            signal = Signal(
                position=self.position,
                prediction=0.0,
                input=input_data,
                variance=0.0,
                feature_relevance=feature_relevance,
                max_x=max_x
            )
            signals.append(signal)
        
        return signals
    
    def calculate_feature_relevance(self, input_data):
        feature_relevance = {}
        for feature in input_data:
            feature_relevance[feature] = self.signal_weights.get(feature, 1.0)
        return feature_relevance

    def accumulate_feature_relevance_gradient(self, dL_dpred_i, signal):
        """
        Accumulate feature relevance gradients from one signal's path.

        The feature relevance rel_j flows into every processing node k on the path:
            scaled_delta_k = (... + value_j * w_j_k * rel_j + ...) / (1 + dist_k)
        So:
            d(scaled_delta_k)/d(rel_j) = value_j * w_j_k / (1 + dist_k)
            dL/d(rel_j) += dL_dpred_i * value_j * w_j_k / (1 + dist_k)
        """
        GRAD_CLIP = 1.0
        for contrib in signal.path_contributions.values():
            scale = 1.0 / (1.0 + contrib['distance'])
            for feature, fd in contrib['feature_details'].items():
                grad = dL_dpred_i * fd['value'] * fd['weight'] * scale
                grad = max(-GRAD_CLIP, min(GRAD_CLIP, grad))
                self.feature_relevance_gradients[feature] = (
                    self.feature_relevance_gradients.get(feature, 0.0) + grad
                )

    def apply_feature_relevance_gradient(self, learning_rate):
        """Apply accumulated feature relevance gradients to signal_weights."""
        WEIGHT_CLIP = 5.0
        for feature, grad in self.feature_relevance_gradients.items():
            current = self.signal_weights.get(feature, 1.0)
            updated = current - learning_rate * grad
            self.signal_weights[feature] = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, updated))
        self.feature_relevance_gradients = {}

    def reset_feature_relevance_gradients(self):
        """Reset gradient accumulators."""
        self.feature_relevance_gradients = {}

    def process(self, input_data, max_x):
        feature_relevance = self.calculate_feature_relevance(input_data)
        signals = self.generate_signals(input_data, max_x, feature_relevance)
        return signals
    
