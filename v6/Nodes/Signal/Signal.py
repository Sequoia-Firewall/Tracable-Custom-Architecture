import sys
LOGGING_ENABLED = '--debug' in sys.argv
"""
from notes
- signal : The representation of the data traveling between nodes 
    -- in a 2d system there should be 4 handlers and in total 4 * (.05 * total_processing_node_count) signals
    -- can be killed partway based on variance
    -- hop based weights to emphasize progression to end
    -- carries the position, segment weight, active prediction, accumulated variance, and signal life

"""

class Signal:
    def __init__ (self, position, segment_weight, feature_relevance, active_prediction, accumulated_variance, life, input_data):
        if LOGGING_ENABLED:
            print(f'[DEBUG] Signal initialized at position {position}, segment_weight={segment_weight}')
        self.position = position  # Current position of the signal in the nexus
        self.segment_weight = segment_weight  # Weight assigned to the current segment
        self.feature_relevance = feature_relevance  # Feature relevance of the signal
        self.active_prediction = active_prediction  # Current prediction value
        self.prediction = active_prediction if active_prediction is not None else 0.0  # For compatibility with downstream code
        self.accumulated_variance = accumulated_variance  # Accumulated variance of the signal
        self.life = life  # Remaining life of the signal
        self.input_data = input_data  # Placeholder for input data
        self.alive = True  # Signal is alive by default
        self.visited_nodes = set()  # Track visited nodes to prevent cycles

        # Aliases for compatibility with ProcessingNode
        self.input = input_data
        self.feature_weights = feature_relevance
        self._thread = None  # Reference to thread running this signal, if any

    def identify_next_node(self, connected_nodes):
        # Implement logic to identify the next node for the signal
        pass

    def subtract_life(self, amount):
        self.life -= amount
        if self.life <= 0:
            self.kill_signal()

    def kill_signal(self):
        # Mark as not alive and attempt to stop thread if possible
        self.alive = False
        # No direct way to kill a thread in Python, but this flag can be checked in threaded logic
        # If thread reference is set, join with timeout for cleanup
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.1)
        self._thread = None

    