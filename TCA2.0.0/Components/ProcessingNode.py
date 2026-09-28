import heapq
import math
import random
import numpy as np
random.seed(42)

class ProcessingNode:
    """
    Geometric/routing substrate shared by every TCA2.0.0 processing-node
    type: position, connectivity, hop-by-hop signal routing, and the
    position-gradient step. Task-specific computation (how a node turns an
    incoming signal into a prediction delta, and how it learns) is NOT
    defined here -- concrete node types (e.g. RegressionProcessingNode
    below) implement initialize_weights/process_signal/train_process_signal/
    accumulate_weight_gradient/apply_weight_gradient. This split is what
    lets a segment mix different node "specialities" on the same map later
    without touching routing at all.
    """
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
        self.position_velocity = [0.0] * len(position)  # EMA of position gradients (momentum)
        self.pred_min = None   # per-instance override; falls back to -PRED_CLIP if unset
        self.pred_max = None   # per-instance override; falls back to  PRED_CLIP if unset
        self.grad_clip = None  # per-instance override; falls back to GRAD_CLIP if unset
        self.delta_clip = None # per-instance override; falls back to DELTA_CLIP if unset

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pos={self.position})"

    def set_grad_clip(self, value):
        """Override the per-element gradient clip (defaults to GRAD_CLIP)."""
        self.grad_clip = value

    def _grad_clip_bound(self):
        return self.grad_clip if self.grad_clip is not None else self.GRAD_CLIP

    def set_delta_clip(self, value):
        """Override the per-hop prediction-delta clip (defaults to DELTA_CLIP)."""
        self.delta_clip = value

    def _delta_clip_bound(self):
        return self.delta_clip if self.delta_clip is not None else self.DELTA_CLIP

    def set_prediction_range(self, min_value, max_value):
        """Override the propagated-prediction clip bounds (defaults to +/-PRED_CLIP)."""
        self.pred_min = min_value
        self.pred_max = max_value

    def _prediction_clip_bounds(self):
        lo = self.pred_min if self.pred_min is not None else -self.PRED_CLIP
        hi = self.pred_max if self.pred_max is not None else self.PRED_CLIP
        return lo, hi

    def display(self, message, classification = None, Loud = True):
        message = f"[{type(self).__name__}]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.Logger.log(message, classification, Loud)

    # ---- Task-specific compute interface -- implemented by concrete node types ----

    def initialize_weights(self, input_data):
        raise NotImplementedError(f"{type(self).__name__} must implement initialize_weights()")

    def process_signal(self):
        raise NotImplementedError(f"{type(self).__name__} must implement process_signal()")

    def train_process_signal(self):
        raise NotImplementedError(f"{type(self).__name__} must implement train_process_signal()")

    def accumulate_weight_gradient(self, dL_dpred, signal):
        raise NotImplementedError(f"{type(self).__name__} must implement accumulate_weight_gradient()")

    def apply_weight_gradient(self, learning_rate, frozen_features=None):
        raise NotImplementedError(f"{type(self).__name__} must implement apply_weight_gradient()")

    # ---- Geometric substrate -- shared by every node type ----

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

        # Honour connection_percentage but never go below MIN_CONNECTIONS,
        # capped at the number of available candidates.
        target_count = max(self.MIN_CONNECTIONS, int(math.ceil(connection_percentage * len(candidates))))
        target_count = min(target_count, len(candidates))

        # Only the target_count nearest are ever used, so a partial selection
        # (O(n log k)) is enough — this runs once per node per training
        # sample whenever positions change, so avoiding a full sort of every
        # candidate adds up. Same result as sorting fully and slicing.
        selected = heapq.nsmallest(target_count, candidates,
                                    key=lambda n: math.dist(self.position, n.position))

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

    def accumulate_position_gradient(self, dL_dpred, signal):
        """Accumulate position gradients from one signal path"""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        raw_delta = contrib['raw_delta']
        distance = contrib['distance']
        gc = self._grad_clip_bound()

        for j, p in enumerate(self.position):
            if distance < 1e-9:
                continue
            dscale_dpos = -p / (distance * (1.0 + distance) ** 2)
            grad_j = dL_dpred * raw_delta * dscale_dpos
            grad_j = max(-gc, min(gc, grad_j))  # Clip
            self.position_gradient[j] += grad_j

    def apply_position_gradient(self, learning_rate, max_step, max_x=None, momentum=0.0):
        """Apply position gradient, clamp step, then enforce quarter-circle bounds.

        Before stepping, the displacement penalty gradient is injected:
            grad_j += POSITION_PENALTY * (pos_j - original_pos_j)
        This is the gradient of λ||pos - pos_original||², pulling the node back
        toward its starting position proportionally to how far it has drifted.

        momentum : EMA coefficient for the position step (velocity = momentum *
                   velocity + gradient, step = -lr * velocity). 0.0 (default)
                   reduces to the original raw-gradient step exactly — momentum
                   is opt-in via settings.training.position_momentum.
        """
        # Inject displacement penalty (grows with drift, no hard limit)
        for j in range(len(self.position)):
            displacement = float(self.position[j]) - self.original_position[j]
            self.position_gradient[j] += self.POSITION_PENALTY * displacement

        new_position = list(self.position)
        for j in range(len(self.position)):
            self.position_velocity[j] = momentum * self.position_velocity[j] + self.position_gradient[j]
            step = -learning_rate * self.position_velocity[j]
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


class RegressionProcessingNode(ProcessingNode):
    """
    First concrete TCA2.0.0 node type: continuous regression compute
    (weighted-sum-of-features + prediction feedback, distance-scaled) --
    the same formula TCA1.1.x's ProcessingNode always used.

    `weights` stays a plain dict externally (SegmentHandler's feature-
    pruning reads, .nexseg save/load, hot_swap) via a property, so nothing
    outside this class needs to know the internal representation changed.
    Internally, a fixed-order numpy vector is the real working copy used by
    process_signal/train_process_signal/gradient methods -- a single
    np.dot() replaces the old per-feature Python loop over dict items. The
    dict is only rebuilt from the array lazily, on the next external read
    of `.weights`, not on every signal.
    """

    def __init__(self, position, Logger=None, classification=4):
        self._feat_order   = None   # fixed feature order, established on first init/load
        self._w_arr        = None   # numpy weight vector, aligned with _feat_order
        self._pred_w       = 1.0    # input_prediction weight (kept as a scalar, not in the array)
        self._grad_arr     = None   # numpy weight-gradient accumulator
        self._pred_grad    = 0.0
        self._weights_dict = {}     # backing store for the `weights` property
        self._dict_stale   = False  # True: array moved ahead of dict, dict needs a refresh on read
        self._array_stale  = True   # True: dict moved ahead of array (or array unset)
        super().__init__(position, Logger, classification)

    @property
    def weights(self):
        if self._dict_stale:
            self._sync_dict_from_array()
        return self._weights_dict

    @weights.setter
    def weights(self, value):
        self._weights_dict = value
        self._array_stale = True
        self._dict_stale = False

    def _sync_array_from_dict(self):
        if self._feat_order is None:
            self._feat_order = sorted(f for f in self._weights_dict.keys() if f != 'input_prediction')
        self._w_arr = np.array([self._weights_dict.get(f, 1.0) for f in self._feat_order], dtype=np.float64)
        self._pred_w = float(self._weights_dict.get('input_prediction', 1.0))
        if self._grad_arr is None or len(self._grad_arr) != len(self._feat_order):
            self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._array_stale = False

    def _sync_dict_from_array(self):
        if self._feat_order is not None and self._w_arr is not None:
            self._weights_dict = {f: float(v) for f, v in zip(self._feat_order, self._w_arr)}
            self._weights_dict['input_prediction'] = float(self._pred_w)
        self._dict_stale = False

    def _ensure_array(self):
        if self._array_stale or self._w_arr is None:
            self._sync_array_from_dict()

    def _active_selection(self, signal):
        """
        (feat_order, mask, weights) for this signal's pass through this node.

        mask is None (weights = full self._w_arr) unless a CompressionNode
        already filtered this signal (signal.active_mask set) -- in that
        case feat_order/weights are the smaller kept-feature subset, so the
        forward pass's dot product and this pass's gradient contribution
        are both over fewer elements, not just zeroed ones.
        """
        mask = getattr(signal, 'active_mask', None)
        if mask is None:
            return self._feat_order, None, self._w_arr
        return signal.active_feat_order, mask, self._w_arr[mask]

    def _signal_arrays(self, signal, feat_order):
        """Convert a signal's per-feature dicts into arrays matching feat_order."""
        values = np.fromiter((signal.input.get(f, 0.0) for f in feat_order),
                              dtype=np.float64, count=len(feat_order))
        rel = np.fromiter((signal.feature_relevance.get(f, 1.0) for f in feat_order),
                           dtype=np.float64, count=len(feat_order))
        return values, rel

    def initialize_weights(self, input_data):
        # Scale initial weight by 1/num_features so weighted_sum starts at a
        # reasonable magnitude regardless of how many features exist.
        # Small random perturbation breaks the initial symmetry between nodes
        # so they differentiate faster during early training.
        # Fixed alphabetical feature order (rather than input_data's incidental
        # dict order) so the node's internal layout doesn't depend on how the
        # caller happened to build the sample dict.
        self._feat_order = sorted(input_data.keys())
        n = max(len(self._feat_order), 1)
        init_w = 1.0 / n
        self._w_arr = np.array(
            [init_w + random.uniform(-0.01, 0.01) for _ in self._feat_order], dtype=np.float64
        )
        # input_prediction weight kept small to dampen the feedback loop
        # (prediction gets multiplied by this and re-added every node)
        self._pred_w = init_w + random.uniform(-0.01, 0.01)
        self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._pred_grad = 0.0
        self._array_stale = False
        self._dict_stale = True

    def process_signal(self):
        """Inference forward pass — same math as train_process_signal but without gradient recording."""
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
            else:
                return None

        self._ensure_array()
        feat_order, _mask, w = self._active_selection(self.signal)
        values, rel = self._signal_arrays(self.signal, feat_order)

        # 1+2. Prediction feedback + weighted feature contributions, vectorized.
        weighted_sum = self.signal.prediction * self._pred_w + float(np.dot(w, values * rel))

        # 3. Distance-based precision scaling
        distance = self.distance_to_origin + 1e-6
        scaled_delta = weighted_sum / (1.0 + distance)

        # Clamp delta to prevent explosion
        dc = self._delta_clip_bound()
        scaled_delta = max(-dc, min(dc, scaled_delta))

        # 4. Update prediction
        self.signal.prediction += scaled_delta
        lo, hi = self._prediction_clip_bounds()
        self.signal.prediction = max(lo, min(hi, self.signal.prediction))

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
        self._ensure_array()

        # 1. Input prediction as a feature
        prev_prediction = self.signal.prediction

        # 2. Weighted feature contributions, vectorized
        feat_order, mask, w = self._active_selection(self.signal)
        values, rel = self._signal_arrays(self.signal, feat_order)
        weighted_sum = prev_prediction * self._pred_w + float(np.dot(w, values * rel))

        # 3. Distance-based scaling
        distance = self.distance_to_origin + 1e-6
        scaled_delta = weighted_sum / (1.0 + distance)

        # Clamp delta to prevent explosion
        dc = self._delta_clip_bound()
        scaled_delta = max(-dc, min(dc, scaled_delta))

        # 4. Update prediction
        self.signal.prediction += scaled_delta
        lo, hi = self._prediction_clip_bounds()
        self.signal.prediction = max(lo, min(hi, self.signal.prediction))

        if hasattr(self.signal, "variance"):
            self.signal.variance += abs(scaled_delta)

        # 5. Record contribution for gradient computation. `weights` is a
        # snapshot (not a live reference to self._w_arr) because the weight
        # phase of backprop mutates self._w_arr for this sample BEFORE the
        # splitter's feature-relevance phase reads this contrib -- gradients
        # for this forward pass must see the weights as they were when this
        # pass actually ran, not weights already updated by a later phase.
        self.signal.path_contributions[id(self)] = {
            'node': self,
            'scaled_delta': scaled_delta,
            'raw_delta': weighted_sum,
            'distance': distance,
            'values': values,
            'relevance': rel,
            'weights': w.copy(),
            'feat_order': feat_order,     # list aligned with values/relevance/weights above
            'active_mask': mask,          # None (full node) or bool array aligned to self._feat_order
            'pred_weight': self._pred_w,
            'prev_prediction': prev_prediction,
        }

        return scaled_delta

    def accumulate_weight_gradient(self, dL_dpred, signal):
        """Accumulate weight gradients from one signal path"""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        self._ensure_array()
        distance = contrib['distance']
        scale = 1.0 / (1.0 + distance)
        gc = self._grad_clip_bound()

        dL_dw = dL_dpred * contrib['values'] * contrib['relevance'] * scale
        np.clip(dL_dw, -gc, gc, out=dL_dw)
        mask = contrib.get('active_mask')
        if mask is not None:
            self._grad_arr[mask] += dL_dw
        else:
            self._grad_arr += dL_dw

        dL_dw_pred = dL_dpred * contrib['prev_prediction'] * scale
        dL_dw_pred = max(-gc, min(gc, dL_dw_pred))
        self._pred_grad += dL_dw_pred

    def apply_weight_gradient(self, learning_rate, frozen_features=None):
        """Apply accumulated weight gradients.

        frozen_features : optional set of feature names to skip updating —
                       used by SegmentHandler's feature-pruning to stop
                       learning on features whose weight has stayed near
                       zero for several epochs, without discarding their
                       current (frozen) contribution to the forward pass.
        """
        self._ensure_array()

        update = learning_rate * self._grad_arr
        if frozen_features:
            frozen_mask = np.array([f in frozen_features for f in self._feat_order], dtype=bool)
            update = np.where(frozen_mask, 0.0, update)
        self._w_arr = np.clip(self._w_arr - update, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)

        if not (frozen_features and 'input_prediction' in frozen_features):
            self._pred_w = max(-self.WEIGHT_CLIP, min(self.WEIGHT_CLIP,
                                self._pred_w - learning_rate * self._pred_grad))

        self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._pred_grad = 0.0
        self.weight_gradients = {}   # vestigial dict, cleared for any external reader
        self._dict_stale = True

    def reset_gradients(self):
        """Reset all gradient accumulators"""
        super().reset_gradients()
        if self._feat_order is not None:
            self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._pred_grad = 0.0


class BayesianProcessingNode(ProcessingNode):
    """
    Second concrete TCA2.0.0 node type: same distance-scaled weighted-sum
    forward math as RegressionProcessingNode, but each weight is a Gaussian
    belief (mu, var) instead of a bare point value, updated with a
    Kalman-filter-style rule instead of a flat-learning-rate gradient step:

        kalman_gain = var / (var + OBS_NOISE_VAR)   # in [0, 1)
        mu  -= kalman_gain * dL_dw                  # bigger step while uncertain
        var *= (1 - kalman_gain)                    # uncertainty shrinks with evidence

    This is a genuinely different learning paradigm, not gradient descent
    with a different label: step size is self-regulating (large while var is
    high, naturally decaying as evidence accumulates -- no externally
    imposed LR schedule needed) and every node carries a real, propagated
    uncertainty estimate instead of SignalNode.variance's old
    `+= abs(scaled_delta)` proxy (which conflated "this node made a big
    correction" with "this node is unsure" -- a confident node correcting a
    genuinely large error scored as if it were LESS confident than a lazy,
    barely-updating one).

    Predictive variance for a weighted sum of independent Gaussian weights
    against fixed/known inputs: Var(sum_j w_j x_j) = sum_j Var(w_j) x_j^2 --
    that's what feeds signal.variance here, not a magnitude proxy.

    `weights` (dict of posterior means) and `posterior_var` (dict of
    posterior variances) both stay plain-dict properties for the same
    reasons as RegressionProcessingNode: SegmentHandler's pruning reads,
    .nexseg save/load (see SegmentHandler._save_nexseg/load_nexseg, which
    persist/restore posterior_var only for node types that define it).
    """

    PRIOR_VAR     = 1.0   # initial per-weight uncertainty
    OBS_NOISE_VAR = 0.5   # assumed noise in each single sample's gradient signal --
                          # lower = trust each sample more (faster, less stable updates),
                          # higher = trust each sample less (slower, steadier updates)

    def __init__(self, position, Logger=None, classification=4):
        self._feat_order   = None
        self._mu            = None   # posterior mean weight vector
        self._var           = None   # posterior variance per weight
        self._pred_mu        = 1.0
        self._pred_var       = self.PRIOR_VAR
        self._grad_arr       = None   # accumulated dL/dw per weight (an "innovation", not applied raw)
        self._pred_grad      = 0.0
        self._weights_dict   = {}     # backing store for the `weights` property (posterior means)
        self._var_dict       = {}     # backing store for the `posterior_var` property
        self._dict_stale     = False
        self._var_dict_stale = False
        self._array_stale    = True
        super().__init__(position, Logger, classification)

    @property
    def weights(self):
        if self._dict_stale:
            self._sync_dict_from_array()
        return self._weights_dict

    @weights.setter
    def weights(self, value):
        self._weights_dict = value
        self._array_stale = True
        self._dict_stale = False

    @property
    def posterior_var(self):
        if self._var_dict_stale:
            self._sync_dict_from_array()
        return self._var_dict

    @posterior_var.setter
    def posterior_var(self, value):
        self._var_dict = value
        self._array_stale = True
        self._var_dict_stale = False

    def _sync_array_from_dict(self):
        if self._feat_order is None:
            self._feat_order = sorted(f for f in self._weights_dict.keys() if f != 'input_prediction')
        self._mu = np.array([self._weights_dict.get(f, 1.0) for f in self._feat_order], dtype=np.float64)
        self._pred_mu = float(self._weights_dict.get('input_prediction', 1.0))
        self._var = np.array([self._var_dict.get(f, self.PRIOR_VAR) for f in self._feat_order], dtype=np.float64)
        self._pred_var = float(self._var_dict.get('input_prediction', self.PRIOR_VAR))
        if self._grad_arr is None or len(self._grad_arr) != len(self._feat_order):
            self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._array_stale = False

    def _sync_dict_from_array(self):
        if self._feat_order is not None and self._mu is not None:
            self._weights_dict = {f: float(v) for f, v in zip(self._feat_order, self._mu)}
            self._weights_dict['input_prediction'] = float(self._pred_mu)
            self._var_dict = {f: float(v) for f, v in zip(self._feat_order, self._var)}
            self._var_dict['input_prediction'] = float(self._pred_var)
        self._dict_stale = False
        self._var_dict_stale = False

    def _ensure_array(self):
        if self._array_stale or self._mu is None:
            self._sync_array_from_dict()

    def _active_selection(self, signal):
        """(feat_order, mask, mu, var) for this signal's pass through this node."""
        mask = getattr(signal, 'active_mask', None)
        if mask is None:
            return self._feat_order, None, self._mu, self._var
        return signal.active_feat_order, mask, self._mu[mask], self._var[mask]

    def _signal_arrays(self, signal, feat_order):
        values = np.fromiter((signal.input.get(f, 0.0) for f in feat_order),
                              dtype=np.float64, count=len(feat_order))
        rel = np.fromiter((signal.feature_relevance.get(f, 1.0) for f in feat_order),
                           dtype=np.float64, count=len(feat_order))
        return values, rel

    def initialize_weights(self, input_data):
        self._feat_order = sorted(input_data.keys())
        n = max(len(self._feat_order), 1)
        init_w = 1.0 / n
        self._mu = np.array(
            [init_w + random.uniform(-0.01, 0.01) for _ in self._feat_order], dtype=np.float64
        )
        self._var = np.full(len(self._feat_order), self.PRIOR_VAR, dtype=np.float64)
        self._pred_mu = init_w + random.uniform(-0.01, 0.01)
        self._pred_var = self.PRIOR_VAR
        self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._pred_grad = 0.0
        self._array_stale = False
        self._dict_stale = True
        self._var_dict_stale = True

    def _forward(self, signal):
        """Shared math for process_signal/train_process_signal. Returns
        (scaled_delta, weighted_sum, distance, feat_order, mask, mu, var, values, rel)."""
        self._ensure_array()
        feat_order, mask, mu, var = self._active_selection(signal)
        values, rel = self._signal_arrays(signal, feat_order)
        contrib = values * rel

        weighted_sum = signal.prediction * self._pred_mu + float(np.dot(mu, contrib))

        # Predictive variance of a sum of independent Gaussian weights against
        # fixed inputs: Var(sum w_j x_j) = sum Var(w_j) x_j^2.
        pred_var = float(np.dot(var, contrib ** 2)) + self._pred_var * (signal.prediction ** 2)

        distance = self.distance_to_origin + 1e-6
        scaled_delta = weighted_sum / (1.0 + distance)
        scaled_var = pred_var / (1.0 + distance) ** 2

        dc = self._delta_clip_bound()
        scaled_delta = max(-dc, min(dc, scaled_delta))

        return scaled_delta, weighted_sum, scaled_var, distance, feat_order, mask, values, rel

    def process_signal(self):
        """Inference forward pass — same math as train_process_signal but without gradient recording."""
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
            else:
                return None

        scaled_delta, _wsum, scaled_var, _dist, *_ = self._forward(self.signal)

        self.signal.prediction += scaled_delta
        lo, hi = self._prediction_clip_bounds()
        self.signal.prediction = max(lo, min(hi, self.signal.prediction))

        if hasattr(self.signal, "variance"):
            self.signal.variance += scaled_var

        return scaled_delta

    def train_process_signal(self):
        """Forward pass for training - computes delta correctly and records contribution for gradients"""
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
            else:
                return None

        self.activation_count += 1
        prev_prediction = self.signal.prediction

        scaled_delta, weighted_sum, scaled_var, distance, feat_order, mask, values, rel = \
            self._forward(self.signal)

        self.signal.prediction += scaled_delta
        lo, hi = self._prediction_clip_bounds()
        self.signal.prediction = max(lo, min(hi, self.signal.prediction))

        if hasattr(self.signal, "variance"):
            self.signal.variance += scaled_var

        self.signal.path_contributions[id(self)] = {
            'node': self,
            'scaled_delta': scaled_delta,
            'raw_delta': weighted_sum,
            'distance': distance,
            'values': values,
            'relevance': rel,
            'weights': (self._mu[mask] if mask is not None else self._mu).copy(),
            'feat_order': feat_order,
            'active_mask': mask,
            'pred_weight': self._pred_mu,
            'prev_prediction': prev_prediction,
        }

        return scaled_delta

    def accumulate_weight_gradient(self, dL_dpred, signal):
        """Accumulate an "innovation" signal per weight from one signal path
        (same dL/dw quantity a gradient-descent node would use -- the Bayesian
        node just applies it through a Kalman gain instead of a flat LR)."""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        self._ensure_array()
        distance = contrib['distance']
        scale = 1.0 / (1.0 + distance)
        gc = self._grad_clip_bound()

        dL_dw = dL_dpred * contrib['values'] * contrib['relevance'] * scale
        np.clip(dL_dw, -gc, gc, out=dL_dw)
        mask = contrib.get('active_mask')
        if mask is not None:
            self._grad_arr[mask] += dL_dw
        else:
            self._grad_arr += dL_dw

        dL_dw_pred = dL_dpred * contrib['prev_prediction'] * scale
        dL_dw_pred = max(-gc, min(gc, dL_dw_pred))
        self._pred_grad += dL_dw_pred

    def apply_weight_gradient(self, learning_rate, frozen_features=None):
        """Bayesian (Kalman-style) posterior update.

        learning_rate is accepted for interface compatibility with
        SegmentHandler's generic per-node call (every node type gets the
        same call signature regardless of learning paradigm) but is NOT
        used here -- the Kalman gain (var / (var + OBS_NOISE_VAR)) already
        supplies a self-regulating step size that shrinks on its own as
        evidence accumulates, which is the entire point of a Bayesian
        update over a flat-LR gradient step.

        frozen_features : as in RegressionProcessingNode -- skip both the
                       mean update AND the variance shrinkage for these
                       features (freezing means "stop learning about this
                       feature" entirely, not just "stop moving its point
                       estimate").
        """
        self._ensure_array()

        kalman_gain = self._var / (self._var + self.OBS_NOISE_VAR)
        if frozen_features:
            frozen_mask = np.array([f in frozen_features for f in self._feat_order], dtype=bool)
            kalman_gain = np.where(frozen_mask, 0.0, kalman_gain)
        self._mu = np.clip(self._mu - kalman_gain * self._grad_arr, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        self._var = (1.0 - kalman_gain) * self._var

        if not (frozen_features and 'input_prediction' in frozen_features):
            pred_kalman_gain = self._pred_var / (self._pred_var + self.OBS_NOISE_VAR)
            self._pred_mu = max(-self.WEIGHT_CLIP, min(self.WEIGHT_CLIP,
                                 self._pred_mu - pred_kalman_gain * self._pred_grad))
            self._pred_var = (1.0 - pred_kalman_gain) * self._pred_var

        self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._pred_grad = 0.0
        self.weight_gradients = {}   # vestigial dict, cleared for any external reader
        self._dict_stale = True
        self._var_dict_stale = True

    def reset_gradients(self):
        """Reset all gradient accumulators"""
        super().reset_gradients()
        if self._feat_order is not None:
            self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._pred_grad = 0.0
