import sys
import threading
LOGGING_ENABLED = '--debug' in sys.argv
"""
from notes:
 - splitter : takes data from judge and passes to closest 5% of processing nodes
    -- splitters are divided based on 2d quadrant or 3d octant
    -- currently no logic to how splitters get different values - judge manages that
    -- gets a feature and passes to processing nodes
    -- intent is generic routing as they create signals
    -- splitters must not be purely feature based
"""

from .BaseNode import BaseNode
import numpy as np
from .Signal.Signal import Signal

class SplitterNode(BaseNode):
    def __init__(self, position, nodes_in_segment, dimenstions, segment_id):
        if LOGGING_ENABLED:
            print(f'[DEBUG] SplitterNode initialized at position {position}, segment_id={segment_id}')
        self.position = position  # Position in the nexus (likely 1 from origin (1,1) or (-1, -1))
        self.closest_nodes = []  # List of closest nodes for routing
        self.nodes_in_segment = nodes_in_segment  # Nodes in the same segment
        self.all_node_count = len(nodes_in_segment)
        self.dimenstions = dimenstions  # Number of dimensions (2D or 3D)
        self.segment_id = segment_id  # Segment identifier
        self.signal_threads = []  # Track threads for signals


    def compute_closest_nodes(self, all_node_count, nodes_in_segment, percent=0.05):
        # Compute Euclidean distance to each node
        distances = []
        for node in nodes_in_segment:
            node_pos = np.array(getattr(node, 'position', (0, 0)))
            splitter_pos = np.array(self.position)
            dist = np.linalg.norm(splitter_pos - node_pos)
            distances.append((node, dist))
        # Sort by distance and select closest N%
        distances.sort(key=lambda x: x[1])
        n_closest = max(1, int(len(nodes_in_segment) * percent))
        self.closest_nodes = [node for node, _ in distances[:n_closest]]

    def handle_carrier(self, carrier_data):
        segment_features = carrier_data.get('segment_feature_relevance', {})
        feature_relevance = segment_features.get(
            self.segment_id,
            carrier_data.get('feature_relevance', {})
        )
        segment_relevance = carrier_data.get('segment_relevance', {}).get(self.segment_id, 1.0)
        return segment_relevance, feature_relevance

    def process(self, carrier_data):
        # For each node, create a new signal using the carrier data
        segment_relevance, feature_relevance = self.handle_carrier(carrier_data)
        # Ensure feature_relevance is a dict
        if isinstance(feature_relevance, list):
            # Convert list of tuples to dict
            feature_relevance = dict(feature_relevance)
        created_signals = []
        self.signal_threads = []  # Reset threads for each process call

        def signal_thread_func(node, signal):
            node.receive_signal(signal)

        for i in self.closest_nodes:
            input_data = carrier_data.get('input_data', None)
            # Convert numpy array to dict if needed
            if isinstance(input_data, np.ndarray):
                input_data = {str(idx): val for idx, val in enumerate(input_data)}
            signal = Signal(
                position=self.position,
                segment_weight=segment_relevance,
                feature_relevance=feature_relevance,
                active_prediction=None,
                accumulated_variance=0.0,
                life=100,  # Arbitrary initial life value
                input_data=input_data
            )
            t = threading.Thread(target=signal_thread_func, args=(i, signal))
            t.daemon = True
            created_signals.append(signal)
            t.start()
            self.signal_threads.append(t)
        return created_signals

    def cleanup_threads(self):
        # Wait for all signal threads to finish (if needed)
        for t in self.signal_threads:
            if t.is_alive():
                t.join(timeout=0.1)
        self.signal_threads = []
    

