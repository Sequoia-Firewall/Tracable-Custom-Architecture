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
        self.position = position  # Position in the nexus
        self.reviewers = []  # List of connected reviewer nodes

    def process(self, data):
        # Implement the handling logic for the HandlerNode
        reviewer_reports = []
        segment_weights = []
        pass

    def set_reviewers(self, reviewers):
        self.reviewers = reviewers
    
    def apply_segment_weights(self, final_reports, segment_weights):
        # Implement logic to apply weights from reviewers
        pass

    