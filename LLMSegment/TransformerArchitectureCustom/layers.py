"""
Core differentiable building blocks, numpy-only, no autograd framework.

Every layer follows the same manual-backprop contract used throughout:
  forward(x)     -> y        (caches whatever backward() needs on self._cache)
  backward(dy)   -> dx       (accumulates parameter gradients on self.grads)

This mirrors TCA's own ProcessingNode convention (forward_signal /
accumulate_weight_gradient / apply_weight_gradient) deliberately -- same
discipline, different math. Every layer here is gradient-checked in
grad_check.py before anything is built on top of it.
"""
import numpy as np


def xavier(shape, rng):
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=shape)


class Linear:
    def __init__(self, in_dim, out_dim, rng, bias=True):
        self.W = xavier((in_dim, out_dim), rng)
        self.b = np.zeros(out_dim) if bias else None
        self.grads = {"W": np.zeros_like(self.W)}
        if bias:
            self.grads["b"] = np.zeros_like(self.b)
        self._cache = None

    def forward(self, x):
        # x: (..., in_dim) -> (..., out_dim)
        self._cache = x
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y

    def backward(self, dy):
        x = self._cache
        flat_x = x.reshape(-1, x.shape[-1])
        flat_dy = dy.reshape(-1, dy.shape[-1])
        self.grads["W"] += flat_x.T @ flat_dy
        if self.b is not None:
            self.grads["b"] += flat_dy.sum(axis=0)
        dx = dy @ self.W.T
        return dx

    def params(self):
        return {"W": self.W, "b": self.b} if self.b is not None else {"W": self.W}

    def zero_grad(self):
        for k in self.grads:
            self.grads[k][...] = 0.0


class LayerNorm:
    """Normalizes over the last axis. gamma/beta are learned per-feature scale/shift."""

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.grads = {"gamma": np.zeros_like(self.gamma), "beta": np.zeros_like(self.beta)}
        self._cache = None

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mu) * std_inv
        self._cache = (x_hat, std_inv)
        return self.gamma * x_hat + self.beta

    def backward(self, dy):
        x_hat, std_inv = self._cache
        n = x_hat.shape[-1]

        self.grads["gamma"] += (dy * x_hat).reshape(-1, n).sum(axis=0)
        self.grads["beta"] += dy.reshape(-1, n).sum(axis=0)

        dx_hat = dy * self.gamma
        # Standard LayerNorm backward: dL/dx from dL/dx_hat, accounting for
        # the mean/variance both depending on every element of x.
        dx = (1.0 / n) * std_inv * (
            n * dx_hat
            - dx_hat.sum(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
        )
        return dx

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def zero_grad(self):
        for k in self.grads:
            self.grads[k][...] = 0.0


class Embedding:
    def __init__(self, vocab_size, dim, rng):
        self.table = rng.normal(0, 0.02, size=(vocab_size, dim))
        self.grads = {"table": np.zeros_like(self.table)}
        self._cache = None

    def forward(self, ids):
        # ids: (...,) int array -> (..., dim)
        self._cache = ids
        return self.table[ids]

    def backward(self, dy):
        ids = self._cache
        flat_ids = ids.reshape(-1)
        flat_dy = dy.reshape(-1, dy.shape[-1])
        np.add.at(self.grads["table"], flat_ids, flat_dy)
        return None  # no gradient flows further back than token ids

    def params(self):
        return {"table": self.table}

    def zero_grad(self):
        for k in self.grads:
            self.grads[k][...] = 0.0


def gelu(x):
    # tanh approximation (same one GPT-2 uses)
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_grad(x):
    c = np.sqrt(2.0 / np.pi)
    x3 = x ** 3
    inner = c * (x + 0.044715 * x3)
    t = np.tanh(inner)
    d_inner = c * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * d_inner


class GELU:
    def __init__(self):
        self._cache = None

    def forward(self, x):
        self._cache = x
        return gelu(x)

    def backward(self, dy):
        x = self._cache
        return dy * gelu_grad(x)

    def params(self):
        return {}

    def zero_grad(self):
        pass


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(dy, y):
    """dL/dx given dL/dy and y=softmax(x), along the last axis."""
    dot = (dy * y).sum(axis=-1, keepdims=True)
    return y * (dy - dot)


def cross_entropy_loss(logits, targets):
    """
    logits: (batch, seq, vocab); targets: (batch, seq) int ids.
    Returns (loss_scalar, dlogits) -- mean over all (batch*seq) positions.
    """
    b, s, v = logits.shape
    probs = softmax(logits, axis=-1)
    flat_probs = probs.reshape(-1, v)
    flat_targets = targets.reshape(-1)
    n = flat_targets.shape[0]
    correct_probs = flat_probs[np.arange(n), flat_targets]
    loss = -np.log(np.clip(correct_probs, 1e-12, None)).mean()

    dlogits = flat_probs.copy()
    dlogits[np.arange(n), flat_targets] -= 1.0
    dlogits /= n
    return loss, dlogits.reshape(b, s, v)
