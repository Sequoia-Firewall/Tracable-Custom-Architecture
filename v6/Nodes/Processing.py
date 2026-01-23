import sys
LOGGING_ENABLED = '--debug' in sys.argv
"""
from notes:
- processing : takes in some input and applies some kind of weight to it
  -- connections based on distance formula
  -- maybe bayes theorem or gauss
  -- the local modelto produce the prediciton and variance
  -- kills the signal if variance exceeds limits
"""

from .BaseNode import BaseNode
import numpy as np

class ProcessingNode(BaseNode):
    def __init__(self, position, reviewer_position=None):
        if LOGGING_ENABLED:
            print(f'[DEBUG] ProcessingNode initialized at position {position}')
        self.position = position  # Position in the nexus
        self.signal = None  # Current signal being processed
        self.weights = {}  # Weights for processing
        self.neighbors = []  # Neighboring nodes for forwarding
        self.reviewer_position = reviewer_position
        

    def create_weights(self, data):
        if not data:
            self.weights = {}
            return

        # Normalize input format
        if isinstance(data, list):
            data = {k: v for k, v in data}

        # Use actual relevance values instead of normalized uniform weights
        self.weights = {k: float(v) for k, v in data.items()}

    def update_weights(self, new_weights):
        # Update weights by blending with new incoming weights (simple moving average)
        if not self.weights:
            self.weights = new_weights.copy()
            return
        for k in new_weights:
            if k in self.weights:
                self.weights[k] = 0.5 * self.weights[k] + 0.5 * new_weights[k]
            else:
                self.weights[k] = new_weights[k]
    
    def process(self):
        # Extract input and feature relevance from the signal
        if not self.signal:
            return
        input_data = getattr(self.signal, 'input_data', None)
        feature_relevances = getattr(self.signal, 'feature_relevance', None)
        if input_data is None or feature_relevances is None:
            return
        # Initialize weights from feature relevance if not set
        if not self.weights:
            self.create_weights(feature_relevances)
        # Apply local model (Bayesian logic)
        prediction, variance = self.apply_local_model(input_data)
        # Update signal variance
        self.signal.accumulated_variance = getattr(self.signal, 'accumulated_variance', 0) + variance
        self.signal.prediction = prediction
        # Mark this node as visited
        if hasattr(self.signal, 'visited_nodes'):
            self.signal.visited_nodes.add(id(self))
            
                
    def receive_signal(self, signal):
        # Receive and process the signal
        if signal is None:
            return
        if getattr(signal, "life", 0) <= 0:
            return
        
        self.signal = signal
        if not self.weights:
          self.create_weights(self.signal.feature_relevance)
          dist = np.linalg.norm(np.array(self.signal.position) - np.array(self.position))
          scale = 1.0 / (1.0 + dist)
          self.weights = {k: v * scale for k, v in self.weights.items()}
        
    
    def apply_local_model(self, input_data):
        # Bayesian-style weighted sum using learned weights
        pred = 0.0
        var = 0.0
        for f, w in self.weights.items():
            x = input_data.get(f, 0)
            pred += w * x
            var += w ** 2
        return pred, max(var, 1e-6)

    def learn(self, input_data, target, lr=0.01):
        """Update weights using a simple gradient step (Bayesian update style)."""
        # Predict with current weights
        pred, _ = self.apply_local_model(input_data)
        error = target - pred
        # Gradient step for each weight
        for f in self.weights:
            x = input_data.get(f, 0)
            self.weights[f] += lr * error * x
    
    def check_signal_variance(self, variance_threshold):
        # Kill the signal if accumulated variance exceeds threshold
        if self.signal and getattr(self.signal, 'accumulated_variance', 0) > variance_threshold:
            if hasattr(self.signal, 'kill_signal'):
                self.signal.kill_signal()
            self.signal = None
    
    def forward_signal(self, next_node=None):
        
        if self.signal is None:
            return None, None
        signal_clone = self.signal
        if self.signal.life <= 0 or not self.neighbors:
            self.signal.alive = False
            self.signal.kill_signal()
            self.signal = None
            return None, signal_clone
        # Only after all processing, forward the signal
        if hasattr(self.signal, 'life') and self.signal.life > 0 and self.neighbors and self.reviewer_position is not None:
            # Sort neighbors by distance to reviewer
            neighbors_sorted = sorted(self.neighbors, key=lambda n: np.linalg.norm(np.array(n.position) - np.array(self.reviewer_position)))
            # Filter out already visited neighbors
            unvisited_neighbors = [n for n in neighbors_sorted if not hasattr(self.signal, 'visited_nodes') or id(n) not in self.signal.visited_nodes]
            if not unvisited_neighbors:
                self.signal = None
                return None, signal_clone  # No unvisited neighbors left
            # Assign custom weights: 20% for closest, 10% for next, 5% for next, 3% for next, 2% for next, 1% for the rest
            base_weights = [20, 10, 5, 3, 2]
            weights = []
            for i in range(len(unvisited_neighbors)):
                if i < len(base_weights):
                    weights.append(base_weights[i])
                else:
                    weights.append(1)
            # Scale weights to sum to 100
            total = sum(weights)
            scaled_weights = [w / total for w in weights]
            import random
            chosen_idx = random.choices(range(len(unvisited_neighbors)), weights=scaled_weights, k=1)[0]
            chosen_neighbor = unvisited_neighbors[chosen_idx]
            self.signal.life -= 1
              # In real scenario, you might want to deep copy
            self.signal = None  # Clear current signal after forwarding
            return chosen_neighbor, signal_clone
        # If no forwarding occurs, return None, None
        return None, signal_clone

    def compute_neighbors(self, all_nodes, percent=0.05):
      distances = []
      for node in all_nodes:
          if node is self:
              continue
          d = np.linalg.norm(
              np.array(self.position) - np.array(node.position)
          )
          distances.append((node, d))

      distances.sort(key=lambda x: x[1])
      n = max(1, int(len(distances) * percent))
      self.neighbors = [n for n, _ in distances[:n]]


