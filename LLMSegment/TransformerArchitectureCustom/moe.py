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

Honest simplification: this computes EVERY expert's output for EVERY
token (dense compute), then zeros out non-selected experts' contribution
via the gating weights. A real production SMoE's speed win comes from
only running the SELECTED experts per token -- that needs genuine
per-token dynamic dispatch, which doesn't fit numpy's batched-array model
without real engineering effort disproportionate to this being a small
research/educational implementation. What's implemented here is the
exact MATH of sparse gating (top-k selection, the same backward-pass
asymmetry between selected/unselected experts, the same load-balancing
loss) -- just not the compute saving.
"""
import numpy as np
from layers import Linear, softmax, softmax_backward
from transformer import FeedForward


class SparseMoEFeedForward:
    def __init__(self, dim, hidden_dim, n_experts, rng, top_k=1, aux_loss_weight=0.01):
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        self.experts = [FeedForward(dim, hidden_dim, rng) for _ in range(n_experts)]
        self.gate = Linear(dim, n_experts, rng)

        self.judge = None
        self.judge_bias_table = {}       # cluster_id -> (n_experts,) array, NOT gradient-trained
        self.cluster_running_loss = {}   # cluster_id -> EMA of per-token loss, the credit-update baseline

        self._cache = None
        self.last_aux_loss = 0.0

    # ---------------------------------------------------------------- judge wiring ----
    def attach_judge(self, judge):
        """judge: a fitted DistributionJudge, queried per-token against
        this layer's FFN-input vectors (not necessarily the same Judge
        instance used for confidence -- a Judge fit on THIS layer's input
        distribution, since that's a different vector than the model's
        final hidden state)."""
        self.judge = judge
        self.judge_bias_table = {
            cid: np.zeros(self.n_experts) for cid in judge.cluster_population_summary()
        }

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

        # top-k mask
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

        expert_outs = np.stack([e.forward(x, trace=trace) for e in self.experts], axis=-2)  # (b,s,n_experts,d)
        output = (gate_weights[..., :, None] * expert_outs).sum(axis=-2)  # (b, s, d)

        self._cache = (x, logits, mask, gate_weights, full_probs, frac_tokens, expert_outs)

        if trace is not None:
            trace.record("moe_gate", {
                "expert_usage_frac": frac_tokens.tolist(),
                "aux_loss": aux_loss,
                "mean_top_gate_weight": float(gate_weights.max(axis=-1).mean()),
            })

        return output

    def backward(self, dout):
        x, logits, mask, gate_weights, full_probs, frac_tokens, expert_outs = self._cache
        b, s, d = x.shape

        # --- gradient through the weighted sum ---
        d_gate_weights = np.einsum("bsd,bsed->bse", dout, expert_outs)  # (b, s, n_experts)
        d_expert_outs = gate_weights[..., :, None] * dout[..., None, :]  # (b, s, n_experts, d)

        dx = np.zeros_like(x)
        for e, expert in enumerate(self.experts):
            dx += expert.backward(d_expert_outs[..., e, :])

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
        return layers

    def named_sublayers(self):
        named = [("gate", self.gate)]
        for i, e in enumerate(self.experts):
            named += [(f"expert{i}.{n}", l) for n, l in e.named_sublayers()]
        return named

    def zero_grad(self):
        for layer in self.sublayers():
            layer.zero_grad()
