"""
Multi-head causal self-attention, built on the Linear primitive from
layers.py. This is the main "architecture internals" surface meant to be
swapped/modified later -- it's deliberately NOT fused into one opaque
matmul: Q/K/V projections, per-head score computation, masking, softmax,
and the output projection are all separate, inspectable steps, each one
individually traceable via `trace` (see TraceRecorder in trace.py).
"""
import numpy as np
from layers import Linear, softmax, softmax_backward


def causal_mask(seq_len):
    # (seq_len, seq_len) bool, True where attention IS allowed (j <= i)
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


class MultiHeadSelfAttention:
    def __init__(self, dim, n_heads, rng):
        assert dim % n_heads == 0, "dim must divide evenly into n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = Linear(dim, dim, rng)
        self.k_proj = Linear(dim, dim, rng)
        self.v_proj = Linear(dim, dim, rng)
        self.out_proj = Linear(dim, dim, rng)

        self._cache = None

    def _split_heads(self, x):
        # (batch, seq, dim) -> (batch, n_heads, seq, head_dim)
        b, s, d = x.shape
        return x.reshape(b, s, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        # (batch, n_heads, seq, head_dim) -> (batch, seq, dim)
        b, h, s, hd = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, s, h * hd)

    def forward(self, x, trace=None):
        """
        x: (batch, seq, dim). Causal mask applied unconditionally (this is a
        decoder-only/GPT-style block).
        trace: optional TraceRecorder -- if given, records per-head attention
        weights and Q/K/V stats for this forward call.
        """
        b, s, d = x.shape
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        qh = self._split_heads(q)  # (b, h, s, hd)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = qh @ kh.transpose(0, 1, 3, 2) * scale  # (b, h, s, s)

        mask = causal_mask(s)  # (s, s)
        neg_inf = np.finfo(scores.dtype).min
        masked_scores = np.where(mask, scores, neg_inf)

        attn = softmax(masked_scores, axis=-1)  # (b, h, s, s)
        out_heads = attn @ vh  # (b, h, s, hd)
        merged = self._merge_heads(out_heads)  # (b, s, d)
        out = self.out_proj.forward(merged)

        self._cache = (qh, kh, vh, attn, mask, scale, merged)

        if trace is not None:
            trace.record("attention", {
                "attn_weights": attn.copy(),       # (b, h, s, s)
                "q_norm": float(np.linalg.norm(qh)),
                "k_norm": float(np.linalg.norm(kh)),
                "v_norm": float(np.linalg.norm(vh)),
                "out_norm": float(np.linalg.norm(out)),
            })

        return out

    def backward(self, dout):
        qh, kh, vh, attn, mask, scale, merged = self._cache
        b, h, s, hd = qh.shape

        dmerged = self.out_proj.backward(dout)  # (b, s, d)
        d_out_heads = self._split_heads(dmerged)  # (b, h, s, hd)

        d_attn = d_out_heads @ vh.transpose(0, 1, 3, 2)  # (b, h, s, s)
        dvh = attn.transpose(0, 1, 3, 2) @ d_out_heads   # (b, h, s, hd)

        d_masked_scores = softmax_backward(d_attn, attn)
        d_masked_scores = np.where(mask, d_masked_scores, 0.0)  # masked positions got no real gradient
        d_scores = d_masked_scores * scale

        dqh = d_scores @ kh               # (b, h, s, hd)
        dkh = d_scores.transpose(0, 1, 3, 2) @ qh  # (b, h, s, hd)

        dq = self._merge_heads(dqh)
        dk = self._merge_heads(dkh)
        dv = self._merge_heads(dvh)

        dx_q = self.q_proj.backward(dq)
        dx_k = self.k_proj.backward(dk)
        dx_v = self.v_proj.backward(dv)

        return dx_q + dx_k + dx_v

    def params(self):
        p = {}
        for name, layer in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("out", self.out_proj)]:
            for k, v in layer.params().items():
                p[f"{name}_{k}"] = v
        return p

    def zero_grad(self):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            layer.zero_grad()

    def sublayers(self):
        return [self.q_proj, self.k_proj, self.v_proj, self.out_proj]
