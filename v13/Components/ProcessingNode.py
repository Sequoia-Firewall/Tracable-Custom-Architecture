import math
import random
random.seed(42)

class ProcessingNode:
    # Stability constants
    DELTA_CLIP = 10.0      # Max absolute value for any single node's delta
    PRED_CLIP = 1e6        # Max absolute prediction value (generous, prevents inf)
    GRAD_CLIP = 1.0        # Max absolute value per accumulated gradient element
    WEIGHT_CLIP = 5.0      # Max absolute weight value after update

    # Connectivity floor: every node is guaranteed at least this many outward
    # connections after connect_nearest_nodes(), regardless of connection_percentage.
    # Raises the floor so signals are less likely to reach a dead end mid-graph.
    MIN_CONNECTIONS = 3

    # Displacement penalty: soft L2 regulariser that penalises moving far from
    # the node's original position.  Gradient = POSITION_PENALTY * displacement,
    # so the further a node drifts, the harder it is pulled back.  Tune this to
    # balance topology freedom vs. collapse prevention.
    POSITION_PENALTY = 0.1

    def __init__(self, position, Logger=None, classification=4):
        self.position          = position
        self.original_position = list(position)   # anchor for displacement penalty
        self.Logger = Logger
        self.classification = classification
        self.signal = None
        self.signal_queue = []
        self.connected_nodes = []
        self.weights = {}
        self.distance_to_origin = sum(p ** 2 for p in position) ** 0.5
        self.activation_count = 0
        self.weight_gradients = {}
        self.position_gradient = [0.0] * len(position)

    def __repr__(self) -> str:
        return f"ProcessingNode(pos={self.position})"
    
    def display(self, message, classification = None, Loud = True):
        message = f"[ProcessingNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.Logger.log(message, classification, Loud)
    
    def initialize_weights(self, input_data):
        # Scale initial weight by 1/num_features so weighted_sum starts at a
        # reasonable magnitude regardless of how many features exist.
        # Small random perturbation breaks the initial symmetry between nodes
        # so they differentiate faster during early training.
        n = max(len(input_data), 1)
        init_w = 1.0 / n
        for feature in input_data:
            self.weights[feature] = init_w + random.uniform(-0.01, 0.01)

        # input_prediction weight kept small to dampen the feedback loop
        # (prediction gets multiplied by this and re-added every node)
        self.weights['input_prediction'] = init_w + random.uniform(-0.01, 0.01)

    def receive_signal(self, signal):
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
                self.signal_queue.append(signal)
            else:
                self.signal = signal
                
        else:
            self.signal_queue.append(signal)
        
        if self.signal is not None:
            self.signal.position = self.position
            self.signal.signal_life -= 1

        for signal in self.signal_queue:
            signal.position = self.position
            signal.signal_life -= 1
        return True
    
    def connect_nearest_nodes(self, node_list, connection_percentage):
        candidates = [
            n for n in node_list
            if n is not self and
            getattr(n, "distance_to_origin", None) > self.distance_to_origin
        ]

        if not candidates:
            return []

        # Compute distances and sort nearest first.
        distances = [(n, math.dist(self.position, n.position)) for n in candidates]
        distances.sort(key=lambda x: x[1])

        # Honour connection_percentage but never go below MIN_CONNECTIONS,
        # capped at the number of available candidates.
        target_count = max(self.MIN_CONNECTIONS, int(math.ceil(connection_percentage * len(candidates))))
        target_count = min(target_count, len(candidates))
        selected = [node for node, _ in distances[:target_count]]

        for node in selected:
            if node not in self.connected_nodes:
                self.connected_nodes.append(node)

        return self.connected_nodes
    
    def forward_signal(self):
        if self.signal is None:
            self.display("No signal to forward. Checking queue", 1, Loud=False)

        if self.signal is None:
            return False
        self.signal.visited_nodes.append(self)

        # Track recent 3 for routing exclusion
        self.signal.recent_visited.append(self)
        if len(self.signal.recent_visited) > 3:
            self.signal.recent_visited.pop(0)

        viable_nodes = []
        for node in self.connected_nodes:
            if node in self.signal.recent_visited:
                continue
            viable_nodes.append(node)

        if not viable_nodes:
            # Dead end: clear recent_visited but keep this node on it so signal
            # can backtrack without immediately returning here.
            self.display("No viable connected nodes to forward the signal.", 1, Loud=False)
            self.signal.recent_visited = [self]
            # Allow all connected nodes except this one as candidates
            viable_nodes = [n for n in self.connected_nodes if n is not self]
            if not viable_nodes:
                self.signal.signal_life = 0
                self.signal = None
                return False

        # Outward-alignment bias: reward movement away from origin.
        # For each candidate, compute how well the movement direction aligns
        # with the away-from-origin direction at this node.
        origin_dist = self.distance_to_origin + 1e-9
        self_norm   = [p / origin_dist for p in self.position]   # unit vec pointing outward

        REVIEWER_BONUS = 3.0   # reviewers are preferred terminal targets
        WEIGHT_FLOOR   = 1e-3  # minimum routing weight — nodes near the origin
                               # still get a non-zero chance so random.choices
                               # never sees an all-zero weight vector

        def _weight(node):
            # Reviewer nodes skip the alignment penalty so all reviewers remain
            # equally reachable regardless of their direction from this node.
            if hasattr(node, 'review_signals'):
                return max(WEIGHT_FLOOR, node.distance_to_origin * REVIEWER_BONUS)
            move = [b - a for a, b in zip(self.position, node.position)]
            move_len = math.sqrt(sum(v * v for v in move)) + 1e-9
            alignment = sum(s * m / move_len for s, m in zip(self_norm, move))
            outward   = max(0.0, alignment)
            return max(WEIGHT_FLOOR, node.distance_to_origin * (1.0 + outward))

        weights = [_weight(n) for n in viable_nodes]
        selected_node = random.choices(viable_nodes, weights=weights, k=1)[0]
        selected_node.receive_signal(self.signal)
        #self.display(f"Forwarded signal to node at position {selected_node.position}.", 4)
        self.signal = None
            
        return True

    def process_signal(self):
        """Inference forward pass — same math as train_process_signal but without gradient recording."""
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
            else:
                return None

        # 1. Input prediction as a feature
        pred_w = self.weights.get('input_prediction', 1.0)
        weighted_sum = self.signal.prediction * pred_w

        # 2. Weighted feature contributions (same formula as train_process_signal)
        for feature, value in self.signal.input.items():
            w = self.weights.get(feature, 1.0)
            rel = self.signal.feature_relevance.get(feature, 1.0)
            weighted_sum += value * w * rel

        # 3. Distance-based precision scaling
        distance = self.distance_to_origin + 1e-6
        scaled_delta = weighted_sum / (1.0 + distance)

        # Clamp delta to prevent explosion
        scaled_delta = max(-self.DELTA_CLIP, min(self.DELTA_CLIP, scaled_delta))

        # 4. Update prediction
        self.signal.prediction += scaled_delta
        self.signal.prediction = max(-self.PRED_CLIP, min(self.PRED_CLIP, self.signal.prediction))

        if hasattr(self.signal, "variance"):
            self.signal.variance += abs(scaled_delta)

        return scaled_delta

    def train_process_signal(self):
        """Forward pass for training - computes delta correctly and records contribution for gradients"""
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
            else:
                return None

        self.activation_count += 1

        # 1. Input prediction as a feature
        pred_w = self.weights.get('input_prediction', 1.0)
        prev_prediction = self.signal.prediction
        weighted_sum = self.signal.prediction * pred_w

        # 2. Weighted feature contributions
        feature_details = {}
        for feature, value in self.signal.input.items():
            w = self.weights.get(feature, 1.0)
            rel = self.signal.feature_relevance.get(feature, 1.0)
            contrib = value * w * rel
            weighted_sum += contrib
            feature_details[feature] = {'value': value, 'weight': w, 'relevance': rel}

        # 3. Distance-based scaling
        distance = self.distance_to_origin + 1e-6
        scaled_delta = weighted_sum / (1.0 + distance)

        # Clamp delta to prevent explosion
        scaled_delta = max(-self.DELTA_CLIP, min(self.DELTA_CLIP, scaled_delta))

        # 4. Update prediction
        self.signal.prediction += scaled_delta
        self.signal.prediction = max(-self.PRED_CLIP, min(self.PRED_CLIP, self.signal.prediction))

        if hasattr(self.signal, "variance"):
            self.signal.variance += abs(scaled_delta)

        # 5. Record contribution for gradient computation
        self.signal.path_contributions[id(self)] = {
            'node': self,
            'scaled_delta': scaled_delta,
            'raw_delta': weighted_sum,
            'distance': distance,
            'feature_details': feature_details,
            'pred_weight': pred_w,
            'prev_prediction': prev_prediction
        }

        return scaled_delta

    def accumulate_weight_gradient(self, dL_dpred, signal):
        """Accumulate weight gradients from one signal path"""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        distance = contrib['distance']
        scale = 1.0 / (1.0 + distance)

        for feature, fd in contrib['feature_details'].items():
            dL_dw = dL_dpred * fd['value'] * fd['relevance'] * scale
            dL_dw = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, dL_dw))  # Clip per-element gradient
            self.weight_gradients[feature] = self.weight_gradients.get(feature, 0.0) + dL_dw

        dL_dw_pred = dL_dpred * contrib['prev_prediction'] * scale
        dL_dw_pred = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, dL_dw_pred))
        self.weight_gradients['input_prediction'] = self.weight_gradients.get('input_prediction', 0.0) + dL_dw_pred

    def accumulate_position_gradient(self, dL_dpred, signal):
        """Accumulate position gradients from one signal path"""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        raw_delta = contrib['raw_delta']
        distance = contrib['distance']

        for j, p in enumerate(self.position):
            if distance < 1e-9:
                continue
            dscale_dpos = -p / (distance * (1.0 + distance) ** 2)
            grad_j = dL_dpred * raw_delta * dscale_dpos
            grad_j = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, grad_j))  # Clip
            self.position_gradient[j] += grad_j

    def apply_weight_gradient(self, learning_rate):
        """Apply accumulated weight gradients"""
        for feature in self.weight_gradients:
            current = self.weights.get(feature, 1.0)
            updated = current - learning_rate * self.weight_gradients[feature]
            self.weights[feature] = max(-self.WEIGHT_CLIP, min(self.WEIGHT_CLIP, updated))  # Clamp weights
        self.weight_gradients = {}

    def apply_position_gradient(self, learning_rate, max_step, max_x=None):
        """Apply position gradient, clamp step, then enforce quarter-circle bounds.

        Before stepping, the displacement penalty gradient is injected:
            grad_j += POSITION_PENALTY * (pos_j - original_pos_j)
        This is the gradient of λ||pos - pos_original||², pulling the node back
        toward its starting position proportionally to how far it has drifted.
        """
        # Inject displacement penalty (grows with drift, no hard limit)
        for j in range(len(self.position)):
            displacement = float(self.position[j]) - self.original_position[j]
            self.position_gradient[j] += self.POSITION_PENALTY * displacement

        new_position = list(self.position)
        for j in range(len(self.position)):
            step = -learning_rate * self.position_gradient[j]
            step = max(-max_step, min(max_step, step))
            new_position[j] = self.position[j] + step

        if max_x is not None:
            # Clamp each axis to the node's natural quadrant, determined by
            # the sign of its original position.  Using max(0, ...) for all
            # axes only works for the positive quadrant (segment 0); other
            # segments have negative coordinates and would collapse to 0.
            for j, orig in enumerate(self.original_position):
                if orig < 0:
                    new_position[j] = max(-float(max_x), min(0.0, new_position[j]))
                else:
                    new_position[j] = max(0.0, min(float(max_x), new_position[j]))
            # Project back onto the arc if the move pushed outside the radius
            dist = math.sqrt(sum(c ** 2 for c in new_position))
            if dist > max_x:
                scale = max_x / dist
                new_position = [c * scale for c in new_position]

        # Exact origin is forbidden: a node at (0,0,...) has distance_to_origin=0
        # which collapses its routing weight to zero.  If gradient drift pushed
        # every coordinate to zero, snap back to the original position instead.
        if not any(abs(c) > 1e-9 for c in new_position):
            new_position = list(self.original_position)

        self.position = tuple(new_position) if isinstance(self.position, tuple) else new_position
        self.distance_to_origin = sum(p ** 2 for p in self.position) ** 0.5
        self.position_gradient = [0.0] * len(self.position)

    def reset_gradients(self):
        """Reset all gradient accumulators"""
        self.weight_gradients = {}
        self.position_gradient = [0.0] * len(self.position)

    def clear_signals(self):
        """Clear signal state between training samples"""
        self.signal = None
        self.signal_queue = []

