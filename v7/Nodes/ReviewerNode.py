"""
 - reviewer:
 -- will be located at the extreme corner of the dimensional object. to incentivise signals to move there
 -- will take all calculated predictions and their variance scores
 -- will use variance and predictions to create a final prediction for that segment
 -- will forward this to handler
 -- purely mathematical
"""

class ReviewerNode:
    def __init__(self, max_x, position, logging_enabled=False, dimensions=2):
        self.logging_enabled = logging_enabled
        self.signals = []
        self.distance_to_origin = ((max_x ** 2) * dimensions) ** 0.5
        self.position = position
        self.prepped = False
        self.expected_signals = 0
        self.accumulated_signals = 0
        self.signal = None #always none for compatibility with forwarding functions

    def __repr__(self) -> str:
        return f"ReviewerNode(pos={self.position})"
    
    def display(self, message):
        if self.logging_enabled:
            print(f"[ReviewerNode] {message}")

    def receive_signal(self, signal):
        self.display(f"Received signal: {signal}")
        self.signals.append(signal)
        signal.alive = False  # Mark signal as consumed
        if len(self.signals) >= self.expected_signals:
            self.prepped = True
        self.display(f"Total signals received: {len(self.signals)}/{self.expected_signals}")

    def reset(self):
        self.display("Resetting ReviewerNode state.")
        self.signals.clear()
        self.prepped = False
    
    def process(self):
        if not self.prepped:
            self.display("Reviewer not prepped.")
            return None
        if not self.signals:
            self.display("No signals to process.")
            return None
        self.display(f"Processing {len(self.signals)} signals.")
        weighted_sum = 0.0
        precision_sum = 0.0

        for signal in self.signals:
            var = max(signal.accumulated_variance, 1e-6)
            precision = 1.0 / var
            weighted_sum += signal.prediction * precision
            precision_sum += precision

        final_prediction = weighted_sum / precision_sum if precision_sum > 0 else None

        self.display(f"Final prediction after variance adjustment: {final_prediction}")

        segment_relevance = self.signals[0].segment_relevance if self.signals else 1.0

        self.reset()
        return final_prediction, segment_relevance
        
