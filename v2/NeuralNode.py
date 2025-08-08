import random
import numpy as np
from computations import Judge, Controller, Splitter, Computational, Repeater, Reviewer, Retainer, Handler
class NeuralNode:
    def __new__(cls, node_id, node_type, node_position, demo=False):
        # Factory: instantiate the correct subclass based on node_type
        if node_type == 'Controller':
            # Provide a default for num_branches in demo mode
            return Controller(node_id, node_position, num_branches=4, demo=demo)
        elif node_type == 'Judge':
            return Judge(node_id, node_position, demo=demo)
        elif node_type == 'Splitter':
            return Splitter(node_id, node_position, num_branches=1, demo=demo)
        elif node_type == 'Computational':
            return Computational(node_id, node_position, demo=demo)
        elif node_type == 'Repeater':
            return Repeater(node_id, node_position, demo=demo)
        elif node_type == 'Retainer':
            return Retainer(node_id, node_position, expected_nodes=1, demo=demo)
        elif node_type == 'Review' or node_type == 'Reviewer':
            return Reviewer(node_id, node_position, num_comps=1, demo=demo)
        elif node_type == "Handler":
            return Handler(node_id, node_position, num_reviewers=4, demo=demo)
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def __init__(self, node_id, node_type, node_position, demo=False):
        # This will only be called for base NeuralNode, which should not happen
        self.node_id = node_id
        self.node_type = node_type
        self.node_position = node_position
        self.demo = demo
        # No setup_node_type needed

    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
    def update_node_weights(self, max_random=None, min_random=None, constant=None):
        if max_random is not None:
            self.weights['Max_random'] = max_random
        if min_random is not None:
            self.weights['Min_random'] = min_random
        if constant is not None:
            self.weights['constant'] = constant
    
    def setup_node_type(self, node_type):
        self.node_type = node_type
        if node_type == 'Judge':
            self.node_class = Judge
        elif node_type == 'Controller':
            self.node_class = Controller
        elif node_type == 'Splitter':
            self.node_class = Splitter
        elif node_type == 'Computational':
            self.node_class = Computational
        elif node_type == 'Repeater':
            self.node_class = Repeater
        elif node_type == 'Retainer':
            self.node_class = Retainer
        elif node_type == 'Review':
            self.node_class = Reviewer
        else:
            raise ValueError("Unknown node type")
    
    def process(self, token_embeddings):
        """
        Process the input token embeddings.
        This method should be overridden by subclasses to implement specific processing logic.
        """
        raise NotImplementedError("Subclasses must implement this method")