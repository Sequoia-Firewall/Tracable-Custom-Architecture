import sys
LOGGING_ENABLED = '--debug' in sys.argv
"""
From notes:
- reviewer : applies final weights based on possible nearest 5% of nodes return
  --located at furthest corners
  -- will attempt to reduce variance 
  -- akin to a critic which wont add weights but will stabilize final returns for each segment

"""

from .BaseNode import BaseNode

class ReviewerNode(BaseNode):
    def __init__(self, position, splitter):
        if LOGGING_ENABLED:
            print(f'[DEBUG] ReviewerNode initialized at position {position}')
        self.position = position  # Position in the nexus
        self.connected_nodes = []  # List of connected nodes
        self.signals = []  # List of signals being reviewed
        self.amount_live_signals = 0  # Count of live signals
        self.handler = None  # Connected handler node
        self.prepped = False  # Flag to indicate if reviewing is done
        self.signal_arrival_count = 0  # Count signals received
        self.dead_signal_count = 0
        self.splitter = splitter  # Associated splitter node, if any
        self.expected_signals = 0  # Expected number of signals to process

    def process(self):
        if self.prepped != True:
            return
        total_received = self.signal_arrival_count + self.dead_signal_count
        if total_received < self.expected_signals:
            return  # Still waiting for more signals
        if not self.signals:
            return None

        weighted_sum = 0.0
        precision_sum = 0.0

        live_signals = []
        for sig in self.signals:
            if sig is None:
                continue
            if not getattr(sig, "alive", True):
                continue
            pred = getattr(sig, 'prediction', None)
            if pred is None:
                continue
            live_signals.append(sig)
        
        if not live_signals:
            self.signals = []
            return None
        for sig in live_signals:
            pred = sig.prediction
            var = max(getattr(sig, 'accumulated_variance', 1e-6), 1e-6)
            precision = 1.0 / var
            weighted_sum += pred * precision
            precision_sum += precision

        final_prediction = weighted_sum / precision_sum if precision_sum > 0 else 0.0
        final_variance = 1.0 / precision_sum if precision_sum > 0 else 1.0

        reviewed_signal = live_signals[0]
        reviewed_signal.prediction = final_prediction
        reviewed_signal.accumulated_variance = final_variance
        reviewed_signal.segment_id = self.splitter.segment_id  # Make sure this is set
        self.signals = [reviewed_signal]
        self.prepped = True

    def set_connected_nodes(self, nodes):
        self.connected_nodes = nodes

    def receive_signal(self, signal):
        if signal is None:
            return
        
        # Check if signal is dead/alive
        if not getattr(signal, 'alive', True) or getattr(signal, 'life', 0) <= 0:
            self.dead_signal_count += 1  # COUNT DEAD SIGNALS
        else:
            self.signals.append(signal)
            self.signal_arrival_count += 1

        if self.signal_arrival_count + self.dead_signal_count >= self.expected_signals:
            self.prepped = True

        

    def reset(self):
        self.signals = []
        self.amount_live_signals = 0
        self.signal_arrival_count = 0
        self.dead_signal_count = 0  # RESET THIS TOO
        self.expected_signals = 0
        self.prepped = False

    def kill_all_signals_and_threads(self):
            # Kill all signals and join their threads if possible
            for sig in self.signals:
                if hasattr(sig, 'kill_signal'):
                    sig.kill_signal()
            self.signals = []
            # Also attempt to clean up threads from associated splitter if available
            if hasattr(self, 'splitter') and hasattr(self.splitter, 'cleanup_threads'):
                self.splitter.cleanup_threads()