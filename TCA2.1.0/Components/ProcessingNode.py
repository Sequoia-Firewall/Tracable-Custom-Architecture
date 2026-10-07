import heapq
import math
import random
import numpy as np
random.seed(42)


def _cosine_similarity(dict_a, dict_b):
    """Cosine similarity between two feature-name-keyed dicts, over their
    shared keys only. Used by ProcessingNode.forward_signal()'s
    routing_mode='similarity' to compare a signal's raw feature values
    against a candidate node's learned weight vector -- different kinds of
    vectors (data vs. parameters), but well-defined since they share the
    same feature-name axes. Degrades gracefully (returns 0.0, a neutral/
    orthogonal similarity) when the two dicts share no keys at all, e.g. a
    CompressionNode-filtered signal missing some of a node's features, or a
    reviewer node with no weights attribute at all.
    """
    keys = dict_a.keys() & dict_b.keys()
    if not keys:
        return 0.0
    dot = sum(dict_a[k] * dict_b[k] for k in keys)
    norm_a = math.sqrt(sum(dict_a[k] ** 2 for k in keys))
    norm_b = math.sqrt(sum(dict_b[k] ** 2 for k in keys))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


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

    def __init__(self, position, Logger=None, classification=4, routing_mode='geometric'):
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
        # 'geometric' (default): forward_signal() weights candidates by
        # distance_to_origin * outward-alignment, exactly as before --
        # blind to the signal's actual content. 'similarity': processing-
        # node candidates are weighted by cosine similarity between the
        # signal's feature vector and the candidate's own learned weight
        # vector instead (reviewers unaffected -- see forward_signal()).
        # Piggybacks on weight training that already happens for prediction
        # accuracy, so routing quality improves for free as weights train,
        # with no separate routing-specific gradient needed.
        self.routing_mode = routing_mode

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

        REVIEWER_BONUS = 3.0   # reviewers are preferred terminal targets
        WEIGHT_FLOOR   = 1e-3  # minimum routing weight — nodes near the origin
                               # still get a non-zero chance so random.choices
                               # never sees an all-zero weight vector

        def _geometric_weight(node):
            # The original, always-100%-collection-rate heuristic (confirmed
            # empirically). Reused both as the default routing mode AND as
            # the lingering-escape fallback below, since it's the one
            # mechanism already proven never to strand a signal.
            if hasattr(node, 'review_signals'):
                return max(WEIGHT_FLOOR, node.distance_to_origin * REVIEWER_BONUS)
            origin_dist = self.distance_to_origin + 1e-9
            self_norm = [p / origin_dist for p in self.position]
            move = [b - a for a, b in zip(self.position, node.position)]
            move_len = math.sqrt(sum(v * v for v in move)) + 1e-9
            alignment = sum(s * m / move_len for s, m in zip(self_norm, move))
            outward = max(0.0, alignment)
            return max(WEIGHT_FLOOR, node.distance_to_origin * (1.0 + outward))

        if self.routing_mode == 'similarity':
            # Content-aware routing: candidates whose own learned weight
            # vector is more cosine-similar to THIS signal's feature vector
            # are preferred. Reviewers keep the original distance-based
            # bonus unchanged (they have no comparable weight vector, and
            # keeping SOME distance-based pull toward them is what stops
            # signals wandering indefinitely among similarly-scored
            # processing nodes now that the outward-progress guarantee is
            # gone for processing-node-to-processing-node hops -- see the
            # signal-collection-rate test this mode was validated against).
            def _base_weight(node):
                if hasattr(node, 'review_signals'):
                    return max(WEIGHT_FLOOR, node.distance_to_origin * REVIEWER_BONUS)
                sim = _cosine_similarity(self.signal.input, getattr(node, 'weights', {}))
                return max(WEIGHT_FLOOR, (sim + 1.0) / 2.0)  # map [-1,1] -> [0,1]
        else:
            _base_weight = _geometric_weight

        # ---- optional: archetype routing factor (Components/ArchetypeRouter.py) ----
        # Multiplies the base weight by a trained per-(cluster, edge) score,
        # discovered via a one-time exploration phase (see ArchetypeRouter),
        # not by anything computed here. Defaults to a no-op (factor 1.0
        # for every edge) until that exploration phase has actually run.
        archetype_router = getattr(self, 'archetype_router', None)
        if archetype_router is not None:
            # Cached on the SIGNAL, not recomputed every hop -- signal.input
            # never changes during a signal's journey (only .prediction/
            # .variance do), so the KMeans-predict call behind cluster_id()
            # was previously repeated once per hop for an answer that can't
            # change. Confirmed as a real cost at full-system scale: eval
            # time was ~2.8x baseline (43s vs 15s on 2000 rows) before this
            # fix, entirely attributable to this redundant recomputation
            # compounding across hops x rows x the 2 segments runInfer()
            # typically queries per row.
            cluster_id = getattr(self.signal, '_archetype_cluster_id', None)
            if cluster_id is None:
                cluster_id = archetype_router.cluster_id(self.signal.input)
                self.signal._archetype_cluster_id = cluster_id
            def _weight(node, _base=_base_weight):
                w = _base(node)
                if not hasattr(node, 'review_signals'):
                    w *= archetype_router.factor(cluster_id, self.position, node.position)
                return max(WEIGHT_FLOOR, w)
        else:
            _weight = _base_weight

        weights = [_weight(n) for n in viable_nodes]

        # ---- optional: confidence-gated exploration temperature ----
        # Uses ConfidenceEstimator.confidence(signal) on the signal's state
        # SO FAR (valid mid-route: only needs variance/hop-count, both
        # already meaningful before collection). Low confidence flattens
        # the distribution (more exploration); high confidence sharpens it
        # (more greedy, follow the base/archetype-favored path).
        conf_est = getattr(self, 'confidence_estimator_for_routing', None)
        if conf_est is not None:
            confidence = conf_est.confidence(self.signal)
            t_min = getattr(self, 'temp_min', 0.3)
            t_max = getattr(self, 'temp_max', 3.0)
            temperature = t_min + (t_max - t_min) * (1.0 - confidence)
            weights = [max(WEIGHT_FLOOR, w ** (1.0 / temperature)) for w in weights]

        # ---- optional: lingering-time escape ----
        # Flagged risk this exists to fix: confidence-gated exploration can
        # feed back on itself -- more exploration -> more hops -> higher
        # accumulated variance -> lower confidence next hop -> even more
        # exploration. As a signal's remaining life shrinks, blend the
        # (possibly exploratory) weight distribution toward the proven
        # geometric/reviewer-seeking one with growing "urgency," so a
        # signal that would otherwise wander forever gets pulled out
        # before expiry regardless of what any other mechanism wants.
        if getattr(self, 'use_lingering_escape', False):
            # Calibrated against hop count vs. max_x, NOT signal_life vs. its
            # theoretical max -- signal_life starts at (max_x**2)*0.8 (e.g.
            # 51.2 at max_x=8), but real signals get collected in ~5-6 hops,
            # so elapsed_fraction relative to THAT budget never rises above
            # ~0.1-0.15 even for a signal that has genuinely finished its
            # journey, and urgency=elapsed_fraction**4 stays negligible
            # (<0.0002) for the entire normal operating range -- confirmed
            # directly: an earlier version of this used signal_life and
            # produced byte-identical results with the escape on vs. off,
            # because it never actually engaged. hop_count/(max_x*hop_scale)
            # reaches a meaningful fraction within a realistic path length
            # instead, so urgency stays negligible for normally-quick
            # signals but actually ramps up for ones taking unusually long.
            hop_scale = getattr(self, 'lingering_hop_scale', 1.5)
            reference_hops = max(1.0, self.signal.max_x * hop_scale)
            elapsed_fraction = min(1.0, len(self.signal.visited_nodes) / reference_hops)
            urgency = elapsed_fraction ** getattr(self, 'lingering_exponent', 4.0)
            if urgency > 1e-6:
                escape_weights = [_geometric_weight(n) for n in viable_nodes]
                w_sum = sum(weights) or 1.0
                e_sum = sum(escape_weights) or 1.0
                weights = [
                    (1.0 - urgency) * (w / w_sum) + urgency * (e / e_sum)
                    for w, e in zip(weights, escape_weights)
                ]

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

    def __init__(self, position, Logger=None, classification=4, routing_mode='geometric'):
        self._feat_order   = None   # fixed feature order, established on first init/load
        self._w_arr        = None   # numpy weight vector, aligned with _feat_order
        self._pred_w       = 1.0    # input_prediction weight (kept as a scalar, not in the array)
        self._grad_arr     = None   # numpy weight-gradient accumulator
        self._pred_grad    = 0.0
        self._weights_dict = {}     # backing store for the `weights` property
        self._dict_stale   = False  # True: array moved ahead of dict, dict needs a refresh on read
        self._array_stale  = True   # True: dict moved ahead of array (or array unset)
        super().__init__(position, Logger, classification, routing_mode=routing_mode)

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

    def __init__(self, position, Logger=None, classification=4, routing_mode='geometric'):
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
        super().__init__(position, Logger, classification, routing_mode=routing_mode)

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


class DistanceValueProcessingNode(ProcessingNode):
    """
    Third concrete TCA2.1.0/2.2.0 node type, and a genuinely different idea
    from Regression/Bayesian: those both ADD a correction to the running
    prediction. This node instead PULLS the prediction toward a value the
    node's own POSITION represents:

        node_value = pred_min + (distance_to_origin / max_x) * (pred_max - pred_min)

    i.e. distance from origin directly encodes a point on the answer's
    range, not just a routing/damping signal the way it does for every
    other node type. A learned, input-dependent "pull strength" (sigmoid
    of a weighted combination of features, same weight-vector machinery as
    RegressionProcessingNode) controls how much of the gap between the
    current running prediction and this node's own value gets closed on
    this hop:

        pull         = sigmoid(dot(weights, features))
        scaled_delta = pull * (node_value - prev_prediction)

    The reason this doesn't need a new training mechanism (no LVQ-style
    update, no exploration phase): node_value is a direct, differentiable
    function of distance_to_origin, which is a direct, differentiable
    function of position -- so the EXISTING position-gradient machinery
    (apply_position_gradient, inherited unchanged from the base class)
    already has what it needs to learn to move a node's distance -- and
    therefore the value it represents -- toward wherever it's actually
    useful, purely by extending the chain rule through node_value. Only
    the gradient of scaled_delta w.r.t. position differs from Regression/
    Bayesian nodes (derived below); weight-gradient bookkeeping, dict/array
    duality for `weights`, and save/load all reuse the same pattern as
    RegressionProcessingNode.

    Deliberately NOT including a prediction-feedback term (no
    'input_prediction' weight) -- pull strength is meant to represent "how
    well does this node's specialization match the input," which should
    depend only on the input's own features, not on the (noisy,
    accumulating) running prediction other nodes have produced so far.
    Nothing else in this codebase requires 'input_prediction' to be present
    in a node's weights dict (checked directly), so omitting it is safe.

    EXPERIMENTAL -- correct but weak; not recommended for default use.
    Gradients verified analytically (numerical finite-difference check
    passes to ~1e-8) and the full train/save/load pipeline round-trips
    correctly, but characterization against RegressionProcessingNode /
    BayesianProcessingNode on matched synthetic tasks (3000 rows, single
    segment, same epoch/node budget unless noted) found:
      - Plateaus around R^2~=0.5-0.6 on a plain linear-regression task even
        at 10x the baseline epoch budget (100 vs 10 epochs), vs ~0.74-0.90
        for Regression/Bayesian at a FRACTION of that budget. Returns
        diminish sharply (50->100 epochs only gained +0.08 R^2) -- this
        reads as a real representational ceiling, not just slow convergence.
      - More nodes without proportionally more epochs HURTS (dilutes
        per-node training exposure); doesn't scale for free the way
        Regression does.
      - A WEIGHT_LR_MULTIPLIER lever (see below) gives a small, consistent
        boost at a low epoch budget (2x mult, R^2 0.26->0.33 at 20 epochs)
        but REVERSES at a realistic budget (2x mult at 100 epochs scores
        WORSE than mult=1.0: 0.30 vs 0.51) -- bigger steps that help early
        prevent fine convergence later. No multiplier tried raised the
        ceiling; it only shifted when diminishing returns kick in.
      - No task-shape tried changed this materially: worse than linear on
        a piecewise/regime-switch target, and while it was the *least bad*
        of the three node types on a smooth sinusoidal target, all three
        were still underwater (negative R^2) there within a 10-epoch budget.
    Root cause: each node can only express "pull toward one scalar value"
    (determined by its own position), vs. Regression's per-node weighted
    sum of features -- a much narrower function class per node that more
    epochs/nodes/LR tuning compensates for only partially. Kept as a
    documented, working, third node type for future ensemble-mixing or
    non-regression-shaped segments (e.g. the LLM/transformer work), not
    because it currently wins anything on its own.
    """

    def __init__(self, position, Logger=None, classification=4, routing_mode='geometric'):
        self._feat_order   = None
        self._w_arr        = None
        self._grad_arr     = None
        self._weights_dict = {}
        self._dict_stale   = False
        self._array_stale  = True
        super().__init__(position, Logger, classification, routing_mode=routing_mode)

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
            self._feat_order = sorted(self._weights_dict.keys())
        self._w_arr = np.array([self._weights_dict.get(f, 0.0) for f in self._feat_order], dtype=np.float64)
        if self._grad_arr is None or len(self._grad_arr) != len(self._feat_order):
            self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._array_stale = False

    def _sync_dict_from_array(self):
        if self._feat_order is not None and self._w_arr is not None:
            self._weights_dict = {f: float(v) for f, v in zip(self._feat_order, self._w_arr)}
        self._dict_stale = False

    def _ensure_array(self):
        if self._array_stale or self._w_arr is None:
            self._sync_array_from_dict()

    def _signal_arrays(self, signal, feat_order):
        values = np.fromiter((signal.input.get(f, 0.0) for f in feat_order),
                              dtype=np.float64, count=len(feat_order))
        rel = np.fromiter((signal.feature_relevance.get(f, 1.0) for f in feat_order),
                           dtype=np.float64, count=len(feat_order))
        return values, rel

    def initialize_weights(self, input_data):
        # Unlike RegressionProcessingNode (weighted-sum-of-raw-features, where
        # init_w = 1/n keeps the initial delta at a reasonable magnitude),
        # these weights feed a SIGMOID gate (pull = sigmoid(dot(w, features))).
        # Real-world features are rarely normalized to O(1) (normalize_numerical
        # is a passthrough for scalars -- see PreProcessingNode), so a 1/n-scale
        # init against even modest feature magnitudes (e.g. three features
        # around 5-10 each) pushes z into the sigmoid's saturated tail
        # immediately: pull~0.99, pull*(1-pull)~0.006, and every weight
        # gradient (which is proportional to that factor) starts crippled.
        # Small, feature-count-independent init keeps z near 0 and pull near
        # 0.5 -- the sigmoid's region of maximum gradient -- regardless of
        # how many features exist or what scale they happen to be on.
        self._feat_order = sorted(input_data.keys())
        self._w_arr = np.array(
            [random.uniform(-0.02, 0.02) for _ in self._feat_order], dtype=np.float64
        )
        self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self._array_stale = False
        self._dict_stale = True

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _node_value(self, max_x):
        lo, hi = self._prediction_clip_bounds()
        frac = 0.0 if max_x <= 0 else max(0.0, min(1.0, self.distance_to_origin / max_x))
        return lo + frac * (hi - lo), frac

    def _forward(self, signal):
        self._ensure_array()
        feat_order, mask, w = self._active_selection(signal)
        values, rel = self._signal_arrays(signal, feat_order)
        contrib = values * rel

        z = float(np.dot(w, contrib))
        pull = self._sigmoid(z)
        node_value, frac = self._node_value(signal.max_x)
        prev_prediction = signal.prediction
        raw_delta = pull * (node_value - prev_prediction)

        dc = self._delta_clip_bound()
        scaled_delta = max(-dc, min(dc, raw_delta))

        return scaled_delta, raw_delta, pull, node_value, frac, feat_order, mask, values, rel, prev_prediction

    def _active_selection(self, signal):
        mask = getattr(signal, 'active_mask', None)
        if mask is None:
            return self._feat_order, None, self._w_arr
        return signal.active_feat_order, mask, self._w_arr[mask]

    def process_signal(self):
        """Inference forward pass — same math as train_process_signal but without gradient recording."""
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
            else:
                return None

        scaled_delta, *_ = self._forward(self.signal)

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
        (scaled_delta, raw_delta, pull, node_value, frac,
         feat_order, mask, values, rel, prev_prediction) = self._forward(self.signal)

        self.signal.prediction += scaled_delta
        lo, hi = self._prediction_clip_bounds()
        self.signal.prediction = max(lo, min(hi, self.signal.prediction))

        if hasattr(self.signal, "variance"):
            self.signal.variance += abs(scaled_delta)

        self.signal.path_contributions[id(self)] = {
            'node': self,
            'scaled_delta': scaled_delta,
            'raw_delta': raw_delta,
            'distance': self.distance_to_origin,
            'pull': pull,
            'node_value': node_value,
            'frac': frac,
            'prev_prediction': prev_prediction,
            'values': values,
            'relevance': rel,
            'weights': (self._w_arr[mask] if mask is not None else self._w_arr).copy(),
            'feat_order': feat_order,
            'active_mask': mask,
            'max_x': self.signal.max_x,
        }

        return scaled_delta

    def accumulate_weight_gradient(self, dL_dpred, signal):
        """dL/dw_i = dL_dpred * (node_value - prev_prediction) * pull*(1-pull) * value_i*rel_i
        -- chain rule through pull = sigmoid(dot(w, values*rel))."""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        self._ensure_array()
        gc = self._grad_clip_bound()
        pull = contrib['pull']
        d_pull_dz = pull * (1.0 - pull)
        d_delta_d_pull = contrib['node_value'] - contrib['prev_prediction']

        dL_dw = dL_dpred * d_delta_d_pull * d_pull_dz * contrib['values'] * contrib['relevance']
        np.clip(dL_dw, -gc, gc, out=dL_dw)

        mask = contrib.get('active_mask')
        if mask is not None:
            self._grad_arr[mask] += dL_dw
        else:
            self._grad_arr += dL_dw

    def accumulate_position_gradient(self, dL_dpred, signal):
        """dL/d(position_j) = dL_dpred * pull * (pred_max-pred_min) * (1/max_x) * (position_j/distance)
        -- chain rule through node_value = f(distance_to_origin) = f(||position||). Zero at the
        frac clamp boundary (node_value is locally constant w.r.t. position there) and at distance=0
        (undefined norm gradient)."""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        frac = contrib['frac']
        if frac <= 0.0 or frac >= 1.0:
            return  # clamped -- node_value doesn't locally respond to position here
        distance = contrib['distance']
        if distance < 1e-9:
            return

        lo, hi = self._prediction_clip_bounds()
        gc = self._grad_clip_bound()
        pull = contrib['pull']
        max_x = contrib['max_x']

        d_delta_d_distance = pull * (hi - lo) / max_x
        for j, p in enumerate(self.position):
            d_distance_d_pj = p / distance
            grad_j = dL_dpred * d_delta_d_distance * d_distance_d_pj
            grad_j = max(-gc, min(gc, grad_j))
            self.position_gradient[j] += grad_j

    # Per-node-type weight-LR scale -- NOT a per-dataset tuning knob, a
    # property of this node type's own gradient shape: pull = sigmoid(dot(w,
    # features)) has a strictly flatter gradient (max 0.25x at pull=0.5, far
    # less away from it) than RegressionProcessingNode's direct linear sum,
    # so the same raw learning_rate under-drives this node type by
    # construction. Default 1.0 (no behavior change) until deliberately
    # overridden -- analogous to BayesianProcessingNode already having its
    # own distinct (Kalman-gain) update rule instead of RegressionProcessingNode's.
    WEIGHT_LR_MULTIPLIER = 1.0

    def apply_weight_gradient(self, learning_rate, frozen_features=None):
        self._ensure_array()
        update = learning_rate * self.WEIGHT_LR_MULTIPLIER * self._grad_arr
        if frozen_features:
            frozen_mask = np.array([f in frozen_features for f in self._feat_order], dtype=bool)
            update = np.where(frozen_mask, 0.0, update)
        self._w_arr = np.clip(self._w_arr - update, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
        self.weight_gradients = {}
        self._dict_stale = True

    def reset_gradients(self):
        super().reset_gradients()
        if self._feat_order is not None:
            self._grad_arr = np.zeros(len(self._feat_order), dtype=np.float64)
