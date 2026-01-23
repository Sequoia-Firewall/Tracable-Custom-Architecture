import sys
LOGGING_ENABLED = '--debug' in sys.argv
"""
From notes:
 - handler : applies final weights from reviewers and returns answer while applying emphasis based on judge relevance scores
  -- final estimation node
  -- will utilize segment weights to find final values
  -- purely mathematical - no ML
  -- handlers must not be trainable
"""

from .BaseNode import BaseNode

class HandlerNode(BaseNode):
    def __init__(self, position):
        if LOGGING_ENABLED:
            print(f'[DEBUG] HandlerNode initialized at position {position}')
        self.position = position
        self.reviewer_reports = []
        self.reviewers = []
        self.segment_weights = {}  # segment_id → weight

    def set_reviewers(self, reviewers):
        self.reviewers = reviewers
    def receive_signal(self, signal):
            # Allow ReviewerNode to forward a signal directly
            # Wrap as a ReviewerReport-like object for downstream process()
            report = type('ReviewerReport', (object,), {
                'segment_id': getattr(signal, 'segment_id', None),
                'prediction': getattr(signal, 'prediction', None),
                'variance': getattr(signal, 'accumulated_variance', None)
            })()
            self.reviewer_reports.append(report)
    def set_segment_weights(self, weights):
        self.segment_weights = weights  # externally provided

    def receive_reports(self):
        for reviewer in self.reviewers:
            if reviewer.prepped and reviewer.signals:
                self.receive_signal(reviewer.signals[0])
                reviewer.reset()


    def process(self):
        if not self.reviewer_reports:
            print(f"[DEBUG][HandlerNode] No reviewer reports to process.")
            return None

        weighted_sum = 0.0
        weight_total = 0.0

        for report in self.reviewer_reports:
            print(f"[DEBUG][HandlerNode] Processing report from segment {report.segment_id} with prediction {report.prediction} and variance {report.variance}")
            if not self.segment_weights:
                self.segment_weights = {r.segment_id: 1.0 for r in self.reviewer_reports}
            seg_weight = self.segment_weights.get(report.segment_id, 1.0)
            precision = 1.0 / max(report.variance, 1e-6)

            contribution = report.prediction * precision * seg_weight
            weighted_sum += contribution
            weight_total += precision * seg_weight
        self.reviewer_reports.clear()
        return weighted_sum / weight_total if weight_total > 0 else 0.0
