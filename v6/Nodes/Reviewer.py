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
        self.splitter = splitter  # Associated splitter node, if any

    def process(self):
        if not self.signals:
            return None

        weighted_sum = 0.0
        precision_sum = 0.0
        live_signals = [sig for sig in self.signals if getattr(sig, "alive", True)]
        if not live_signals:
            self.signals = []
            return None
        for sig in live_signals:
            var = max(sig.accumulated_variance, 1e-6)
            precision = 1.0 / var
            weighted_sum += sig.prediction * precision
            precision_sum += precision

        final_prediction = weighted_sum / precision_sum if precision_sum > 0 else 0.0
        final_variance = 1.0 / precision_sum if precision_sum > 0 else 1.0

        # Use the first live signal and update its values
        reviewed_signal = live_signals[0]
        reviewed_signal.prediction = final_prediction
        reviewed_signal.accumulated_variance = final_variance
        self.signals = [reviewed_signal]
        self.prepped = True

    def set_connected_nodes(self, nodes):
        self.connected_nodes = nodes

    def receive_signal(self, signal):
        # Receive and process the signal
        self.signals.append(signal)
        self.signal_arrival_count += 1
        live_count = sum(1 for sig in self.signals if getattr(sig, "alive", True))
        if self.amount_live_signals == 0:
            self.amount_live_signals = live_count
        if live_count == self.amount_live_signals:
            self.process()
            self.forward_review()

    def reset_signals(self):
        self.signals = []
        self.amount_live_signals = 0
    
    def forward_review(self):
        # Forward the reviewed signal to a random connected node or handler
        import random
        if self.signals:
            reviewed_signal = self.signals[0]
            targets = []
            if self.handler:
                targets.append(self.handler)
            targets += [node for node in getattr(self, 'connected_nodes', []) if hasattr(node, 'receive_signal')]
            if targets:
                random.choice(targets).receive_signal(reviewed_signal)
        self.reset_signals()
        self.prepped = False
    
    def reset(self):
        self.reset_signals()
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