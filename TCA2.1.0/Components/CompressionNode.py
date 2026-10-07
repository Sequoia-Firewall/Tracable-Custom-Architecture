import numpy as np


class CompressionNode:
    """
    Feature-compression stage: sits between the splitter and the processing-
    node graph. Reuses the splitter's already-learned per-feature relevance
    (SplitterNode.signal_weights) to keep only the top `keep_fraction`
    features for a signal's onward journey -- every RegressionProcessingNode
    that signal visits afterward does a smaller dot product for its entire
    remaining path, not just a smaller sum (filtered features are dropped,
    not zeroed, so the downstream node's *array itself* shrinks).

    Deliberately NOT part of the position-based routing graph: it doesn't
    do a stochastic hop. Every signal passes through it exactly once,
    deterministically, right after the splitter generates the signal --
    that sidesteps needing the stochastic router to guarantee a specific
    node is visited first.

    keep_fraction : fraction of the segment's features to keep (top-N by
                   |signal_weight|, the same score the splitter already
                   learns via feature-relevance gradient descent).
    warmup_epochs : compression stays off for this many epochs so
                   signal_weights (all initialized to 1.0) get a chance to
                   actually diverge before being used to decide what to
                   drop -- filtering by relevance before any relevance has
                   been learned would just be an arbitrary tie-broken cut.
    """

    def __init__(self, feat_order, keep_fraction=0.7, warmup_epochs=1, Logger=None, classification=4):
        self.feat_order = list(feat_order)   # canonical, segment-wide feature order
        self.keep_fraction = keep_fraction
        self.warmup_epochs = warmup_epochs
        self.Logger = Logger
        self.classification = classification
        self.enabled = warmup_epochs <= 0

    def display(self, message, classification=None, Loud=True):
        message = f"[CompressionNode]: {message}"
        if self.Logger is None:
            return
        if classification is None:
            classification = self.classification
        self.Logger.log(message, classification, Loud)

    def set_epoch(self, epoch_index):
        self.enabled = epoch_index >= self.warmup_epochs

    def filter(self, signal, signal_weights):
        """Mutates signal.input/feature_relevance in place to the kept subset,
        and attaches signal.active_mask/active_feat_order for downstream
        RegressionProcessingNodes to slice their own weight vectors with."""
        if not self.enabled or self.keep_fraction >= 1.0:
            signal.active_mask = None
            signal.active_feat_order = None
            return signal

        n_keep = max(1, int(round(len(self.feat_order) * self.keep_fraction)))
        ranked = sorted(self.feat_order, key=lambda f: abs(signal_weights.get(f, 1.0)), reverse=True)
        keep_set = set(ranked[:n_keep])

        signal.input = {k: v for k, v in signal.input.items() if k in keep_set}
        signal.feature_relevance = {k: v for k, v in signal.feature_relevance.items() if k in keep_set}
        signal.active_mask = np.array([f in keep_set for f in self.feat_order], dtype=bool)
        signal.active_feat_order = [f for f in self.feat_order if f in keep_set]
        return signal
