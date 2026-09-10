class Signal:
    def __init__(self, position, prediction, input, variance, feature_relevance, max_x):
        # Signal info which must be defined
        self.position = position
        self.prediction = prediction
        self.input = input
        self.feature_relevance = feature_relevance
        self.max_x = max_x
        self.visited_nodes = []  # Nodes that have processed this signal (for path_contributions)
        self.recent_visited = []  # Last 3 visited nodes (for routing exclusion)
        self.collected = False
        self.path_contributions = {}  # {node_id: contribution_dict} for gradient computation

        # Additional calculated attributes
        self.signal_life = (max_x ** 2) * .8 # Life of signal based on max_x
        self.variance = variance

    def __repr__(self):
        return f"Signal(pos={self.position}, pred={self.prediction}, input={self.input}, var={self.variance}, relevance={self.feature_relevance}, max_x={self.max_x})"

    def is_active(self):
        return self.signal_life > 0

    