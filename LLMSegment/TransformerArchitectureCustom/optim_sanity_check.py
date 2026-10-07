"""
No backward pass to gradient-check here -- Adam only consumes gradients
already verified elsewhere. What needs checking is the UPDATE RULE
itself (sign errors, bias-correction bugs). Standard way to check an
optimizer in isolation: minimize a convex quadratic with a known optimum
and confirm it actually gets there.
"""
import numpy as np
from optim import Adam, lr_schedule


class FakeLayer:
    def __init__(self, dim, rng):
        self.W = rng.normal(size=dim)
        self.grads = {}

    def params(self):
        return {"W": self.W}


class FakeModel:
    def __init__(self, layer):
        self.layer = layer

    def all_sublayers(self):
        return [self.layer]


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    target = rng.normal(size=20) * 5.0
    layer = FakeLayer(20, rng)
    model = FakeModel(layer)
    adam = Adam()

    losses = []
    for step in range(2000):
        diff = layer.W - target
        loss = 0.5 * np.sum(diff ** 2)
        layer.grads["W"] = diff  # d(0.5||W-target||^2)/dW
        lr = lr_schedule(step, warmup_steps=50, max_steps=2000, max_lr=0.5, min_lr=0.01)
        adam.step(model, lr)
        losses.append(loss)
        if step % 400 == 0:
            print(f"step {step:4d}  loss={loss:.6f}  lr={lr:.4f}")

    print(f"final loss={losses[-1]:.8f}")
    final_dist = np.linalg.norm(layer.W - target)
    print(f"final ||W - target||={final_dist:.6f}")
    assert losses[-1] < losses[0] * 1e-4, "Adam did not converge on a convex quadratic"
    assert final_dist < 0.01, f"Adam did not reach the optimum closely enough: {final_dist}"
    print("\nADAM SANITY CHECK PASSED")
