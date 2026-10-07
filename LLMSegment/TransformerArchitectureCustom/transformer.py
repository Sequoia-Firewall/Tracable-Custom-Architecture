"""
Full decoder-only (GPT-style) transformer, built entirely from layers.py +
attention.py. Pre-norm residual blocks (LN -> sublayer -> add), causal
self-attention only (no cross-attention / encoder side) -- this is the
simplest architecture that's still a real next-token language model,
chosen so "custom manipulation of architecture internals" has a small,
fully-understood baseline to diverge from.
"""
import numpy as np
from layers import Linear, LayerNorm, Embedding, GELU, cross_entropy_loss
from attention import MultiHeadSelfAttention


class FeedForward:
    def __init__(self, dim, hidden_dim, rng):
        self.fc1 = Linear(dim, hidden_dim, rng)
        self.act = GELU()
        self.fc2 = Linear(hidden_dim, dim, rng)

    def forward(self, x, trace=None):
        h = self.fc1.forward(x)
        a = self.act.forward(h)
        out = self.fc2.forward(a)
        if trace is not None:
            trace.record("feedforward", {
                "pre_act_norm": float(np.linalg.norm(h)),
                "post_act_norm": float(np.linalg.norm(a)),
            })
        return out

    def backward(self, dout):
        da = self.fc2.backward(dout)
        dh = self.act.backward(da)
        dx = self.fc1.backward(dh)
        return dx

    def sublayers(self):
        return [self.fc1, self.fc2]

    def zero_grad(self):
        for layer in self.sublayers():
            layer.zero_grad()


class TransformerBlock:
    def __init__(self, dim, n_heads, ffn_hidden_dim, rng):
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, rng)
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_hidden_dim, rng)
        self._cache = None

    def forward(self, x, trace=None):
        normed1 = self.ln1.forward(x)
        attn_out = self.attn.forward(normed1, trace=trace)
        res1 = x + attn_out

        normed2 = self.ln2.forward(res1)
        ffn_out = self.ffn.forward(normed2, trace=trace)
        res2 = res1 + ffn_out

        self._cache = True
        return res2

    def backward(self, dout):
        # res2 = res1 + ffn_out  -> gradient splits identically to both branches
        d_res1_from_ffn = dout
        d_ffn_out = dout
        d_normed2 = self.ffn.backward(d_ffn_out)
        d_res1_from_ln2 = self.ln2.backward(d_normed2)
        d_res1 = d_res1_from_ffn + d_res1_from_ln2

        d_x_from_res1 = d_res1
        d_attn_out = d_res1
        d_normed1 = self.attn.backward(d_attn_out)
        d_x_from_ln1 = self.ln1.backward(d_normed1)
        dx = d_x_from_res1 + d_x_from_ln1
        return dx

    def sublayers(self):
        return [self.ln1, self.ln2] + self.attn.sublayers() + self.ffn.sublayers()

    def zero_grad(self):
        for layer in self.sublayers():
            layer.zero_grad()


class GPTStyleTransformer:
    def __init__(self, vocab_size, dim, n_heads, n_layers, ffn_hidden_dim,
                 max_seq_len, rng_seed=0):
        rng = np.random.RandomState(rng_seed)
        self.rng = rng
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_emb = Embedding(vocab_size, dim, rng)
        self.pos_emb = Embedding(max_seq_len, dim, rng)
        self.blocks = [TransformerBlock(dim, n_heads, ffn_hidden_dim, rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, vocab_size, rng, bias=False)

        self._cache = None

    def forward(self, token_ids, trace=None):
        # token_ids: (batch, seq) int
        b, s = token_ids.shape
        assert s <= self.max_seq_len, f"sequence length {s} exceeds max_seq_len {self.max_seq_len}"

        pos_ids = np.broadcast_to(np.arange(s), (b, s))
        x = self.token_emb.forward(token_ids) + self.pos_emb.forward(pos_ids)

        for block in self.blocks:
            x = block.forward(x, trace=trace)

        normed = self.ln_f.forward(x)
        logits = self.head.forward(normed)

        self._cache = (b, s)
        return logits

    def backward(self, dlogits):
        dnormed = self.head.backward(dlogits)
        dx = self.ln_f.backward(dnormed)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        # token_emb and pos_emb both received dx (embeddings were summed)
        self.token_emb.backward(dx)
        self.pos_emb.backward(dx)

    def loss_and_backward(self, token_ids, targets, trace=None):
        logits = self.forward(token_ids, trace=trace)
        loss, dlogits = cross_entropy_loss(logits, targets)
        self.backward(dlogits)
        return loss, logits

    def all_sublayers(self):
        layers = [self.token_emb, self.pos_emb]
        for block in self.blocks:
            layers += block.sublayers()
        layers += [self.ln_f, self.head]
        return layers

    def zero_grad(self):
        for layer in self.all_sublayers():
            layer.zero_grad()

    def sgd_step(self, lr):
        for layer in self.all_sublayers():
            for name, param in layer.params().items():
                if param is None:
                    continue
                param -= lr * layer.grads[name]

    @staticmethod
    def sample_greedy(model, prompt_ids, n_new_tokens):
        """prompt_ids: 1D list/array of ids. Returns the extended sequence
        (greedy decoding, no trace -- just a sanity-check utility)."""
        ids = list(prompt_ids)
        for _ in range(n_new_tokens):
            window = np.array(ids[-model.max_seq_len:])[None, :]
            logits = model.forward(window)
            next_id = int(np.argmax(logits[0, -1]))
            ids.append(next_id)
        return ids
