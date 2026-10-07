"""
Adam optimizer, generalized to any model exposing all_sublayers() (each
layer: .params() -> {name: array or None}, .grads -> {name: array}), the
same convention sgd_step() already uses. Needed because plain SGD (the
only optimizer used everywhere else in this project) is unlikely to
converge well enough, in a tractable step budget on pure-NumPy/CPU, to
produce a baseline-test number worth comparing to anything -- Adam + an
LR schedule is what every real small-transformer baseline (including the
nanoGPT numbers this is compared against) actually uses.

State is keyed by (id(layer), param_name), not id(param) -- param arrays
are updated in-place (param -= ...) everywhere in this codebase, so
object identity is stable, but keying off the LAYER instead is more
robust regardless (correct even if some future layer ever reassigns the
attribute rather than mutating in place).
"""
import numpy as np


class Adam:
    def __init__(self, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, model, lr):
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t

        for layer in model.all_sublayers():
            for param_name, param in layer.params().items():
                if param is None:
                    continue
                grad = layer.grads[param_name]
                if self.weight_decay > 0.0:
                    grad = grad + self.weight_decay * param

                key = (id(layer), param_name)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

                m_hat = self.m[key] / bc1
                v_hat = self.v[key] / bc2
                param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def lr_schedule(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup, then cosine decay to min_lr -- the standard
    nanoGPT-style schedule."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
