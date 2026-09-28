import math


class ConfidenceEstimator:
    """
    Trainable replacement for the fixed `1.0 / max(signal.variance, 1e-9)`
    confidence heuristic used throughout this codebase (ReviewerNode,
    SegmentHandler's reviewer aggregation, segmentInfer's reported
    per-reviewer confidence). That heuristic was never validated against
    actual prediction error -- it assumes "low accumulated variance along
    a signal's path" IS "low error," which the earlier Bayesian-node
    calibration tests found to be close to uncorrelated with real error in
    practice.

    This estimator instead directly predicts a signal's actual error from
    a small set of features available at collection time (before the
    target is known, so it works identically at training and inference):
        f1 = log(signal.variance + eps)   -- the old heuristic's raw signal
        f2 = len(signal.visited_nodes)    -- hop count taken to reach the
                                              reviewer; not used by the old
                                              heuristic at all
    and is trained via ordinary gradient descent against actual ABSOLUTE
    error (target IS available during training), so "does low reported
    confidence actually mean high error" becomes a learned, checkable
    relationship rather than an assumption. Absolute (not squared) error is
    the training target deliberately: squared error's dynamic range blows
    up quadratically with the target's span (MAE~17 on a 0-100 range means
    TYPICAL squared error is already ~289), which either saturates a fixed
    clip on nearly every sample (destroying the variance the estimator
    needs to learn from -- this happened during testing with an
    unscaled ERROR_CLIP=25) or requires clip tuning per target scale.
    Absolute error's range grows linearly with target span instead, and
    matches the same "error" quantity this codebase's own calibration
    checks (confidence-vs-error correlation) already use throughout.

    predicted_error = softplus(w1*f1 + w2*f2 + bias)   -- softplus keeps
                       this non-negative, required for the confidence
                       transform below to stay in (0, 1].
    confidence       = 1.0 / (1.0 + predicted_error)   -- same 1/(1+x) shape
                       already used elsewhere in this codebase (HandlerNode's
                       cross-segment confidence), for consistency.

    Deliberately NOT touching SegmentHandler._backprop()'s inverse-variance
    weighting used for weight/position gradient credit assignment across
    multiple signals reaching one reviewer -- that's a different concern
    (how to distribute blame across signal paths when updating shared
    node weights) from "how much should the FINAL reported prediction be
    trusted." Changing both at once would make it impossible to tell which
    change caused any observed effect.
    """

    EPS = 1e-9
    # actual_error is unbounded -- an occasional very bad prediction can
    # still produce a large absolute error, which without a clip produces a
    # huge dL_dpred and blows the weights out (observed directly during
    # testing: with squared error and an unscaled clip, w1 reached -1892
    # and confidence saturated at 1.0 for every signal). Clipping the
    # TARGET before computing loss, plus the usual per-element gradient
    # clip and post-update weight clip, mirrors the stability discipline
    # every other trainable quantity in this codebase already has
    # (ProcessingNode.GRAD_CLIP/WEIGHT_CLIP, SplitterNode's clip in
    # apply_feature_relevance_gradient). ERROR_CLIP's default (25.0) is a
    # fallback for when no target-range info is available; SegmentHandler.
    # train() calls set_error_clip() with a value derived from the actual
    # pred_min/pred_max span whenever those bounds are known, the same
    # auto-derivation convention _auto_grad_clip/_auto_delta_clip already
    # use for their own clips.
    ERROR_CLIP  = 25.0
    GRAD_CLIP   = 5.0
    WEIGHT_CLIP = 10.0

    def __init__(self, hop_scale=0.1, error_clip=None):
        # Small random-free init (0.0) rather than the small-random-noise
        # init used for ProcessingNode weights -- there's no symmetry to
        # break here (only 2 input features, not many parallel identical
        # units), so starting at 0 gives predicted_error=softplus(0)=log(2)
        # for every signal until training differentiates them, a neutral
        # starting confidence rather than an arbitrary random one.
        self.w1 = 0.0     # weight on f1 = log(variance + eps)
        self.w2 = 0.0     # weight on f2 = hop_count * hop_scale
        self.bias = 0.0
        # hop_count is O(5-30) while log(variance) is typically O(-5..5) --
        # this scales hop_count into a comparable range so gradient descent
        # doesn't have to discover two wildly different natural step sizes
        # for w1 vs w2 on its own.
        self.hop_scale = hop_scale
        # Per-instance override of the class-level ERROR_CLIP default,
        # mirroring ProcessingNode's set_grad_clip/set_delta_clip pattern.
        self.error_clip = error_clip if error_clip is not None else self.ERROR_CLIP

        self._grad_w1 = 0.0
        self._grad_w2 = 0.0
        self._grad_bias = 0.0
        self._accum_count = 0

    def _features(self, signal):
        f1 = math.log(max(signal.variance, 0.0) + self.EPS)
        f2 = len(signal.visited_nodes) * self.hop_scale
        return f1, f2

    @staticmethod
    def _softplus(z):
        # Numerically stable: softplus(z) = max(z,0) + log(1+exp(-|z|))
        return max(z, 0.0) + math.log1p(math.exp(-abs(z)))

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict_error(self, signal):
        f1, f2 = self._features(signal)
        z = self.w1 * f1 + self.w2 * f2 + self.bias
        return self._softplus(z)

    def confidence(self, signal):
        return 1.0 / (1.0 + self.predict_error(signal))

    def set_error_clip(self, value):
        """Override the per-sample actual_error clip (defaults to ERROR_CLIP)."""
        self.error_clip = value

    def accumulate_gradient(self, signal, actual_error):
        """actual_error : non-negative real error for this signal's final
        prediction (absolute error against the known target -- see class
        docstring for why absolute rather than squared) -- only available
        during training. dL/dw for L=(predicted_error-actual_error)^2."""
        actual_error = min(actual_error, self.error_clip)
        f1, f2 = self._features(signal)
        z = self.w1 * f1 + self.w2 * f2 + self.bias
        predicted_error = self._softplus(z)

        dL_dpred = 2.0 * (predicted_error - actual_error)
        dpred_dz = self._sigmoid(z)
        dL_dz = dL_dpred * dpred_dz

        gc = self.GRAD_CLIP
        self._grad_w1 += max(-gc, min(gc, dL_dz * f1))
        self._grad_w2 += max(-gc, min(gc, dL_dz * f2))
        self._grad_bias += max(-gc, min(gc, dL_dz))
        self._accum_count += 1

    def apply_gradient(self, learning_rate):
        if self._accum_count == 0:
            return
        # Average accumulated gradient over however many signals contributed
        # this sample/epoch, matching the convention used elsewhere in this
        # codebase (e.g. SplitterNode's feature-relevance update) of a
        # single step per apply call rather than per-signal steps.
        n = self._accum_count
        wc = self.WEIGHT_CLIP
        self.w1 = max(-wc, min(wc, self.w1 - learning_rate * (self._grad_w1 / n)))
        self.w2 = max(-wc, min(wc, self.w2 - learning_rate * (self._grad_w2 / n)))
        self.bias = max(-wc, min(wc, self.bias - learning_rate * (self._grad_bias / n)))
        self.reset_gradients()

    def reset_gradients(self):
        self._grad_w1 = 0.0
        self._grad_w2 = 0.0
        self._grad_bias = 0.0
        self._accum_count = 0

    def to_dict(self):
        return {"w1": self.w1, "w2": self.w2, "bias": self.bias,
                "hop_scale": self.hop_scale, "error_clip": self.error_clip}

    @classmethod
    def from_dict(cls, state):
        est = cls(hop_scale=state.get("hop_scale", 0.1), error_clip=state.get("error_clip"))
        est.w1 = state.get("w1", 0.0)
        est.w2 = state.get("w2", 0.0)
        est.bias = state.get("bias", 0.0)
        return est
