"""
SparseMoEFeedForward -- a drop-in replacement for FeedForward inside a
TransformerBlock (same forward(x, trace=None)/backward(dout) interface,
same sublayers()/named_sublayers()/zero_grad() contract), so the rest of
the architecture doesn't need to know which one it's holding.

Top-k gating with a Switch-Transformer-style load-balancing auxiliary
loss (the standard fix for gate collapse -- see README for the mechanism
explanation). On top of that: an OPTIONAL per-cluster bias added to gate
logits, sourced from cluster_ids computed by a JudgeLayer (see
judge_layer.py) and passed into forward() -- this layer no longer holds
its own DistributionJudge; cluster COMPUTATION is shared infrastructure
(JudgeLayer), but the learned PREFERENCE for each cluster
(judge_bias_table) stays here, since a future attention-expert consumer
might want a different preference for the same cluster. That bias is NOT
gradient-trained -- it's updated separately via update_judge_bias_credit(),
a proportional-credit rule matching ArchetypeRouter's (nudge toward
whatever empirically reduced error relative to a per-cluster running
baseline, not a hard assignment).

Two additions pulled from Data-Clustered Sparse Training (DCST, see
top-level "Data-Clustered Sparse Training Across All Layers.md") after
comparing it to this mechanism -- see TransformerArchitectureCustom/
README.md for the full comparison and what was deliberately NOT adopted
(real sparse compute, attention/conv partitioning, a single shared
cluster map across all layers):
  - n_shared_experts: always-active experts (DCST's "shared group",
    also DeepSeekMoE's design) alongside the top-k ROUTED experts.
  - route_noise_to_shared_only: DCST's dense-fallback-for-outliers idea,
    adapted -- a token the Judge calls noise (-1, doesn't belong to any
    dense training-data region) gets ZERO routed-expert weight, relying
    only on the shared expert(s), instead of silently taking whatever
    the unbiased gate happens to prefer.

A third addition, ReviewerLayer, ports TCA's ReviewerNode concept: once
the gate (+ judge bias) decides WHO is selected, the Reviewer decides
HOW MUCH each selected expert's output counts, via confidence (inverse
accumulated-deviation) rather than the gate's own softmax score --
starting with TCA's ORIGINAL crude 1/variance heuristic (variance here
proxied by output norm), matching TCA's own evolution: ship the
heuristic first, only replace with a trained ConfidenceEstimator if it
proves out (see README "Next steps"). Deliberately MODULATES the gate's
softmax weight rather than replacing it outright: full replacement would
remove the gate's only quality-linked gradient signal (its gradient
currently flows entirely through how much weight it assigns to the
output combination) -- it would then be trained only by the aux
load-balancing loss, which pushes toward EVEN usage, not GOOD usage.
Confidence is treated as a stop-gradient quantity throughout (TCA's own
variance is bookkeeping, not a trained/differentiable signal either).

Honest simplification, unchanged: this computes EVERY expert's output
for EVERY token (dense compute), then zeros out non-selected/non-shared
contributions via gating weights. The real MATH of sparse gating, not
the compute saving -- see README.
"""
import numpy as np
from layers import Linear, softmax, softmax_backward
from transformer import FeedForward


class ReviewerLayer:
    """
    TCA's ReviewerNode, ported: combines the gate's selected experts by
    confidence (1/variance, variance proxied by output norm -- TCA's
    ORIGINAL heuristic, before ConfidenceEstimator replaced it), MODULATING
    the gate's own softmax weight rather than replacing it (see module
    docstring for why). Stop-gradient throughout: confidence influences
    forward computation but no gradient flows back through it into
    expert_outs via this path (only the direct linear-scaling path,
    exactly like TCA's variance bookkeeping is not itself trained).
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def combine(self, gate_weights, expert_outs, keep_mask):
        """
        gate_weights: (b,s,n_experts), already zero outside top-k.
        expert_outs: (b,s,n_experts,dim).
        keep_mask: (b,s), 1.0/0.0 outlier zeroing (applied after combination).

        Returns (effective_weights, cache) where effective_weights is what
        actually multiplies expert_outs in the output sum, and cache holds
        what backward() needs (reviewer_weights r, the g*r sum S, and the
        combined-before-keep_mask weights, all under the Reviewer's own key
        so moe.py's backward can call Reviewer.backward separately).
        """
        variance_proxy = np.linalg.norm(expert_outs, axis=-1) + self.eps  # (b,s,n_experts)
        raw_confidence = 1.0 / variance_proxy
        raw_confidence = raw_confidence * (gate_weights > 0)  # only among selected experts

        conf_sum = raw_confidence.sum(axis=-1, keepdims=True)
        conf_sum = np.where(conf_sum > 0, conf_sum, 1.0)  # no selection (e.g. outlier token) -> avoid /0
        r = raw_confidence / conf_sum  # (b,s,n_experts), sums to 1 over the selected set

        S = (gate_weights * r).sum(axis=-1, keepdims=True)
        S = np.where(S > 0, S, 1.0)
        combined = gate_weights * r / S  # modulated weights, before outlier keep_mask

        effective_weights = combined * keep_mask[..., None]
        cache = {"r": r, "S": S, "combined": combined, "gate_weights": gate_weights}
        return effective_weights, cache

    def backward(self, d_effective_weights, cache, keep_mask):
        """
        Returns d_gate_weights -- the gradient w.r.t. the gate's OWN
        softmax output, to be fed into the existing softmax_backward(.,
        gate_weights) call exactly where d_gate_weights used to go
        directly. Derived in full in the README (closed form for
        combined_e = g_e r_e / S, r stop-gradient): with
        dot = sum_e(d_combined_e * combined_e),
            d_g = (r / S) * (d_combined - dot)
        """
        r, S, combined = cache["r"], cache["S"], cache["combined"]
        d_combined = d_effective_weights * keep_mask[..., None]  # chain through keep_mask (constant scale)
        dot = (d_combined * combined).sum(axis=-1, keepdims=True)
        d_gate_weights = (r / S) * (d_combined - dot)
        return d_gate_weights


class SparseMoEFeedForward:
    def __init__(self, dim, hidden_dim, n_experts, rng, top_k=1, aux_loss_weight=0.01,
                 n_shared_experts=0, route_noise_to_shared_only=True,
                 use_reviewer=False, prior_strength=0.0):
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.n_shared_experts = n_shared_experts
        self.route_noise_to_shared_only = route_noise_to_shared_only
        self.prior_strength = prior_strength

        self.experts = [FeedForward(dim, hidden_dim, rng) for _ in range(n_experts)]
        self.shared_experts = [FeedForward(dim, hidden_dim, rng) for _ in range(n_shared_experts)]
        self.gate = Linear(dim, n_experts, rng)
        self.reviewer = ReviewerLayer() if use_reviewer else None

        self.judge_bias_table = {}       # cluster_id -> (n_experts,) array, NOT gradient-trained
        self.cluster_running_loss = {}   # cluster_id -> EMA of per-token loss, the credit-update baseline

        self._cache = None
        self.last_aux_loss = 0.0

    # ---------------------------------------------------------------- judge-bias credit (consumer-owned state) ----
    def _bias_row(self, cid):
        """Lazy per-cluster bias row: zero (or DCST-style prior toward
        expert cid % n_experts, if prior_strength > 0) the first time a
        cluster id is seen, then whatever credit updates have moved it to
        since. No explicit 'attach' step needed -- the cluster COMPUTATION
        lives in JudgeLayer now; this layer only needs cluster ids handed
        to it (see forward())."""
        if cid not in self.judge_bias_table:
            bias = np.zeros(self.n_experts)
            if self.prior_strength > 0.0 and cid != -1:
                bias[cid % self.n_experts] = self.prior_strength
            self.judge_bias_table[cid] = bias
        return self.judge_bias_table[cid]

    def update_judge_bias_credit(self, cluster_ids, chosen_experts, token_losses, lr=0.05, ema=0.9):
        """
        Proportional-credit update, same spirit as ArchetypeRouter.update():
        nudge judge_bias_table[cluster][expert] toward whatever empirically
        beat that cluster's running-average loss, rather than hard-assigning
        cluster->expert. Call this AFTER a training step, using that step's
        real per-token loss -- not part of the gradient-trained path at all.

        cluster_ids, chosen_experts, token_losses: flat arrays, same length
        (one entry per token this step).
        """
        for cid, expert, loss in zip(cluster_ids, chosen_experts, token_losses):
            cid = int(cid)
            self._bias_row(cid)  # ensure it exists
            baseline = self.cluster_running_loss.get(cid, loss)
            advantage = baseline - loss  # positive = did BETTER than this cluster's usual
            self.judge_bias_table[cid][expert] += lr * advantage
            self.cluster_running_loss[cid] = ema * baseline + (1 - ema) * loss

    # ---------------------------------------------------------------- forward/backward ----
    def forward(self, x, trace=None, cluster_ids=None):
        """
        cluster_ids: (batch, seq) int array from a JudgeLayer.forward()
        call (-1 = noise), or None -- same "no judge signal" semantics as
        before, just supplied externally now instead of this layer
        querying its own attached Judge.
        """
        b, s, d = x.shape
        raw_logits = self.gate.forward(x)  # (b, s, n_experts)

        judge_bias = None
        if cluster_ids is not None:
            bias_rows = np.stack([self._bias_row(int(c)) for c in cluster_ids.reshape(-1)], axis=0)
            judge_bias = bias_rows.reshape(b, s, self.n_experts)
        logits = raw_logits if judge_bias is None else raw_logits + judge_bias

        # top-k mask (unaffected by outlier handling -- the gate still makes
        # a real choice for every token; only how much weight that choice
        # gets APPLIED with changes for outliers, see keep_mask below)
        topk_idx = np.argsort(-logits, axis=-1)[..., :self.top_k]  # (b, s, top_k)
        mask = np.zeros_like(logits, dtype=bool)
        np.put_along_axis(mask, topk_idx, True, axis=-1)

        neg_inf = np.finfo(logits.dtype).min
        masked_logits = np.where(mask, logits, neg_inf)
        gate_weights = softmax(masked_logits, axis=-1)          # zero outside top-k

        full_probs = softmax(logits, axis=-1)                   # used ONLY by the aux loss
        frac_tokens = mask.astype(np.float64).mean(axis=(0, 1))  # (n_experts,) fraction routed to each
        avg_prob = full_probs.mean(axis=(0, 1))                  # (n_experts,)
        aux_loss = self.n_experts * float(np.sum(frac_tokens * avg_prob))
        self.last_aux_loss = aux_loss

        # Outlier handling: a noise-labeled token's routed-expert weight is
        # zeroed entirely (not renormalized/redistributed -- just dropped),
        # relying on the shared expert(s) alone. keep_mask is 1.0 everywhere
        # when outlier handling doesn't apply (no judge, no shared experts,
        # or disabled) -- fully backward compatible.
        keep_mask = np.ones((b, s), dtype=np.float64)
        if self.route_noise_to_shared_only and cluster_ids is not None and self.n_shared_experts > 0:
            keep_mask = np.where(cluster_ids == -1, 0.0, 1.0)

        expert_outs = np.stack([e.forward(x, trace=trace) for e in self.experts], axis=-2)  # (b,s,n_experts,d)

        reviewer_cache = None
        if self.reviewer is not None:
            effective_gate_weights, reviewer_cache = self.reviewer.combine(gate_weights, expert_outs, keep_mask)
        else:
            effective_gate_weights = gate_weights * keep_mask[..., None]

        routed_output = (effective_gate_weights[..., :, None] * expert_outs).sum(axis=-2)  # (b, s, d)

        shared_output = np.zeros((b, s, d))
        for se in self.shared_experts:
            shared_output = shared_output + se.forward(x, trace=trace)
        output = shared_output + routed_output

        self._cache = (x, logits, mask, gate_weights, keep_mask, full_probs, frac_tokens,
                       expert_outs, reviewer_cache)

        if trace is not None:
            trace.record("moe_gate", {
                "expert_usage_frac": frac_tokens.tolist(),
                "aux_loss": aux_loss,
                "mean_top_gate_weight": float(gate_weights.max(axis=-1).mean()),
                "outlier_frac": float(1.0 - keep_mask.mean()),
                "reviewer_active": self.reviewer is not None,
            })

        return output

    def backward(self, dout):
        (x, logits, mask, gate_weights, keep_mask, full_probs, frac_tokens,
         expert_outs, reviewer_cache) = self._cache
        b, s, d = x.shape

        dx = np.zeros_like(x)

        # --- shared experts: unconditional, unscaled -- every one gets dout directly ---
        for se in self.shared_experts:
            dx += se.backward(dout)

        # --- routed experts, through the (outlier-zeroed, possibly Reviewer-modulated) weights ---
        # reuse the cached pre-keep_mask 'combined' from forward() rather than recomputing it
        if self.reviewer is not None:
            effective_gate_weights = reviewer_cache["combined"] * keep_mask[..., None]
        else:
            effective_gate_weights = gate_weights * keep_mask[..., None]

        d_effective_gate_weights = np.einsum("bsd,bsed->bse", dout, expert_outs)  # (b, s, n_experts)
        d_expert_outs = effective_gate_weights[..., :, None] * dout[..., None, :]  # (b, s, n_experts, d)

        for e, expert in enumerate(self.experts):
            dx += expert.backward(d_expert_outs[..., e, :])

        if self.reviewer is not None:
            # Reviewer's own backward gives d_gate_weights directly (handles the
            # keep_mask chain internally, since combined/keep_mask interact there).
            d_gate_weights = self.reviewer.backward(d_effective_gate_weights, reviewer_cache, keep_mask)
        else:
            # chain through the outlier keep_mask (a constant scale, same reasoning as the
            # top-k mask itself: cluster_ids come from a non-differentiable nearest-neighbor
            # lookup, so keep_mask is treated as locally constant, not backpropagated through)
            d_gate_weights = d_effective_gate_weights * keep_mask[..., None]

        # --- gradient through top-k softmax (primary path: only selected experts get a nonzero term) ---
        d_logits_primary = softmax_backward(d_gate_weights, gate_weights)

        # --- gradient through the auxiliary load-balancing loss (the ONLY path reaching
        # unselected experts' logits -- frac_tokens is a hard count, zero gradient; only
        # avg_prob's dependence on the FULL softmax is differentiable) ---
        n = b * s
        d_full_probs = self.aux_loss_weight * self.n_experts * (frac_tokens / n)
        d_full_probs = np.broadcast_to(d_full_probs, logits.shape)
        d_logits_aux = softmax_backward(d_full_probs, full_probs)

        d_logits = d_logits_primary + d_logits_aux
        dx += self.gate.backward(d_logits)
        return dx

    # ---------------------------------------------------------------- plumbing (matches FeedForward's contract) ----
    def sublayers(self):
        layers = [self.gate]
        for e in self.experts:
            layers += e.sublayers()
        for se in self.shared_experts:
            layers += se.sublayers()
        return layers

    def named_sublayers(self):
        named = [("gate", self.gate)]
        for i, e in enumerate(self.experts):
            named += [(f"expert{i}.{n}", l) for n, l in e.named_sublayers()]
        for i, se in enumerate(self.shared_experts):
            named += [(f"shared{i}.{n}", l) for n, l in se.named_sublayers()]
        return named

    def zero_grad(self):
        for layer in self.sublayers():
            layer.zero_grad()
