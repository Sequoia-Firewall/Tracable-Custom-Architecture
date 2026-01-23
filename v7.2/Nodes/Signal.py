"""
- signal:
 -- a class simply to contain all info as it is passed allowing

"""
import copy
class Signal:
    def __init__(self, segment_relevance_score, feature_relevance_scores, mean, life=50, input_data=None, id=None):
        id = id
        self.prediction = mean
        self.accumulated_variance = 1.0
        self.segment_relevance = segment_relevance_score  # to be set later
        self.feature_relevance = feature_relevance_scores  # feature_name → relevance score structured as {feature_name: [], relevance_score: []}
        self.alive = True
        self.position = None  # to be set later
        self.life = life  # default life
        self.input_data = input_data  # Placeholder for input data
        self.visited_nodes = set()  # track visited nodes to avoid cycles

    def clone(self):
        return copy.deepcopy(self)
    def __repr__(self):
        return f"Signal(prediction={self.prediction}, accumulated_variance={self.accumulated_variance}, segment_relevance={self.segment_relevance}, feature_relevance={self.feature_relevance})"
    