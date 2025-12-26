"""
From notes:
- reviewer : applies final weights based on possible nearest 5% of nodes return
  --located at furthest corners
  -- will attempt to reduce variance 
  -- akin to a critic which wont add weights but will stabilize final returns for each segment

"""

from .BaseNode import BaseNode

class ReviewerNode(BaseNode):
    def __init__(self, position):
        self.position = position  # Position in the nexus
        self.connected_nodes = []  # List of connected nodes

    def process(self, data):
        # Implement the review logic for the ReviewerNode
        pass

    def set_connected_nodes(self, nodes):
        self.connected_nodes = nodes
