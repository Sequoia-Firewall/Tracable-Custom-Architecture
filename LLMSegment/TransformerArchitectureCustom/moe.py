"""
SparseMoEFeedForward -- a drop-in replacement for FeedForward inside a
TransformerBlock (same forward(x, trace=None)/backward(dout) interface,
same sublayers()/named_sublayers()/zero_grad() contract), so the rest of
the architecture doesn't need to know which one it's holding.

Top-k gating with a Switch-Transformer-style load-balancing auxiliary
loss (the standard fix for gate collapse -- see README for the mechanism
explanation). On top of that: an OPTIONAL per-cluster bias added to gate
logits, sourced from a DistributionJudge attached via attach_judge().
That bias is NOT gradient-trained -- it's updated separately via
update_judge_bias_credit(), a proportional-credit rule matching
ArchetypeRouter's (nudge toward whatever empirically reduced error
relative to a per-cluster running baseline, not a hard assignment).

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

Honest simplification, unchanged: this computes EVERY expert's output
for EVERY token (dense compute), then zeros out non-selected/non-shared
contributions via gating weights. The real MATH of sparse gating, not
the compute saving -- see README.
"""
import numpy as np
from layers import Linear, softmax, softmax_backward
from transformer import FeedForward


class SparseMoEFeedForward:
    def __init__(self, dim, hidden_dim, n_experts, rng, top_k=1, aux_loss_weight=0.01,
                 n_shared_experts=0, route_noise_to_shared_only=True):
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.n_shared_experts = n_shared_experts
        self.route_noise_to_shared_only = route_noise_to_shared_only

        self.experts = [FeedForward(dim, hidden_dim, rng) for _ in range(n_experts)]
        self.shared_experts = [FeedForward(dim, hidden_dim, rng) for _ in range(n_shared_experts)]
        self.gate = Linear(dim, n_experts, rng)

        self.judge = None
        self.judge_bias_table = {}       # cluster_id -> (n_experts,) array, NOT gradient-trained
        self.cluster_running_loss = {}   # cluster_id -> EMA of per-token loss, the credit-update baseline

        self._cache = None
        self.last_aux_loss = 0.0

    # ---------------------------------------------------------------- judge wiring ----
    def attach_judge(self, judge, prior_strength=0.0):
        """
        judge: a fitted DistributionJudge, queried per-token against this
        layer's FFN-input vectors (not necessarily the same Judge instance
        used for confidence -- a Judge fit on THIS layer's input
        distribution, since that's a different vector than the model's
        final hidden state).

        prior_strength: 0.0 (default) -- judge_bias_table starts at zero,
        matching the original design (specialization must emerge purely
        from credit updates). > 0.0 -- DCST's "arbitrary assignment"
        idea (4.3.1 in the proposal): cluster k starts with a strong bias
        toward expert (k mod n_experts), and credit updates are free to
        drift away from it if that's actually better. This tests DCST's
        hypothesis directly inside this mechanism: does starting from a
        hard cluster->expert prior and letting it adapt beat starting
        from nothing?
        """
        self.judge = judge
        self.judge_bias_table = {}
        for cid in judge.cluster_population_summary():
            bias = np.zeros(self.n_experts)
            if prior_strength > 0.0 and cid != -1:
                bias[cid % self.n_experts] = prior_strength
            self.judge_bias_table[cid] = bias

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
            if cid not in self.judge_bias_table:
                self.judge_bias_table[cid] = np.zeros(self.n_experts)
            baseline = self.cluster_running_loss.get(cid, loss)
            advantage = baseline - loss  # positive = did BETTER than this cluster's usual
            self.judge_bias_table[cid][expert] += lr * advantage
            self.cluster_running_loss[cid] = ema * baseline + (1 - ema) * loss

    # ---------------------------------------------------------------- forward/backward ----
    def _judge_bias(self, x):
        """x: (batch, seq, dim) -> (bias: (batch, seq, n_experts) or None, cluster_ids: (batch, seq) or None)."""
        if self.judge is None or not self.judge._fitted:
            return None, None
        b, s, d = x.shape
        flat_x = x.reshape(-1, d)
        _, nearest_label, _ = self.judge.query_batch(flat_x)
        bias_rows = np.stack([
            self.judge_bias_table.get(int(c), np.zeros(self.n_experts)) for c in nearest_label
        ], axis=0)
        return bias_rows.reshape(b, s, self.n_experts), nearest_label.reshape(b, s)

    def forward(self, x, trace=None):
        b, s, d = x.shape
        raw_logits = self.gate.forward(x)  # (b, s, n_experts)

        judge_bias, cluster_ids = self._judge_bias(x)
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
        effective_gate_weights = gate_weights * keep_mask[..., None]

        expert_outs = np.stack([e.forward(x, trace=trace) for e in self.experts], axis=-2)  # (b,s,n_experts,d)
        routed_output = (effective_gate_weights[..., :, None] * expert_outs).sum(axis=-2)  # (b, s, d)

        shared_output = np.zeros((b, s, d))
        for se in self.shared_experts:
            shared_output = shared_output + se.forward(x, trace=trace)
        output = shared_output + routed_output

        self._cache = (x, logits, mask, gate_weights, keep_mask, full_probs, frac_tokens, expert_outs)

        if trace is not None:
            trace.record("moe_gate", {
                "expert_usage_frac": frac_tokens.tolist(),
                "aux_loss": aux_loss,
                "mean_top_gate_weight": float(gate_weights.max(axis=-1).mean()),
                "outlier_frac": float(1.0 - keep_mask.mean()),
            })

        return output

    def backward(self, dout):
        x, logits, mask, gate_weights, keep_mask, full_probs, frac_tokens, expert_outs = self._cache
        b, s, d = x.shape

        dx = np.zeros_like(x)

        # --- shared experts: unconditional, unscaled -- every one gets dout directly ---
        for se in self.shared_experts:
            dx += se.backward(dout)

        # --- routed experts, through the (outlier-zeroed) effective gate weights ---
        effective_gate_weights = gate_weights * keep_mask[..., None]
        d_effective_gate_weights = np.einsum("bsd,bsed->bse", dout, expert_outs)  # (b, s, n_experts)
        d_expert_outs = effective_gate_weights[..., :, None] * dout[..., None, :]  # (b, s, n_experts, d)

        for e, expert in enumerate(self.experts):
            dx += expert.backward(d_expert_outs[..., e, :])

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
