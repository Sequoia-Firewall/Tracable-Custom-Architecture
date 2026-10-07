"""
Numerical gradient checks, same discipline as TCA's DistanceValueProcessingNode
verification: perturb each input/parameter by epsilon, compare finite-difference
loss change against the analytically computed gradient. Run this after any
change to layers.py / attention.py / transformer.py before trusting the math.
"""
import numpy as np

EPS = 1e-6
TOL = 1e-4


def numeric_grad(f, x, eps=EPS):
    """f: array -> scalar. Returns a numeric gradient the same shape as x."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f(x)
        x[idx] = orig - eps
        minus = f(x)
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def check(name, analytic, numeric, tol=TOL):
    err = np.max(np.abs(analytic - numeric))
    status = "PASS" if err < tol else "FAIL"
    print(f"[{status}] {name:40s} max_abs_err={err:.8f}")
    return err < tol


def random_loss_wrapper(module, x, upstream_rng):
    """Wraps module.forward(x) into a scalar loss via a fixed random upstream
    gradient (dot with a fixed random vector) -- this is the standard trick
    for gradient-checking a layer in isolation without needing a real loss."""
    y = module.forward(x)
    weight = upstream_rng.normal(size=y.shape)
    loss = (y * weight).sum()
    return loss, weight, y


if __name__ == "__main__":
    from layers import Linear, LayerNorm, Embedding, GELU, softmax, softmax_backward, cross_entropy_loss

    rng = np.random.RandomState(0)
    all_pass = True

    # ---------------- Linear ----------------
    print("\n=== Linear ===")
    lin = Linear(4, 3, rng)
    x = rng.normal(size=(2, 5, 4))
    upstream = rng.normal(size=(2, 5, 3))

    def f_x(xv):
        lin._cache = None
        return (lin.forward(xv) * upstream).sum()

    y = lin.forward(x)
    dx = lin.backward(upstream)
    num_dx = numeric_grad(f_x, x.copy())
    all_pass &= check("Linear dx", dx, num_dx)

    lin.zero_grad()
    lin.forward(x)
    lin.backward(upstream)
    analytic_dW = lin.grads["W"].copy()

    def f_W(Wv):
        lin.W = Wv
        return (lin.forward(x) * upstream).sum()

    num_dW = numeric_grad(f_W, lin.W.copy())
    all_pass &= check("Linear dW", analytic_dW, num_dW)

    lin.zero_grad()
    lin.forward(x)
    lin.backward(upstream)
    analytic_db = lin.grads["b"].copy()

    def f_b(bv):
        lin.b = bv
        return (lin.forward(x) * upstream).sum()

    num_db = numeric_grad(f_b, lin.b.copy())
    all_pass &= check("Linear db", analytic_db, num_db)

    # ---------------- LayerNorm ----------------
    print("\n=== LayerNorm ===")
    ln = LayerNorm(4)
    x = rng.normal(size=(2, 3, 4))
    upstream = rng.normal(size=(2, 3, 4))

    def f_x(xv):
        return (ln.forward(xv) * upstream).sum()

    ln.forward(x)
    dx = ln.backward(upstream)
    num_dx = numeric_grad(f_x, x.copy())
    all_pass &= check("LayerNorm dx", dx, num_dx)

    ln.zero_grad()
    ln.forward(x)
    ln.backward(upstream)
    analytic_dgamma = ln.grads["gamma"].copy()

    def f_gamma(gv):
        ln.gamma = gv
        return (ln.forward(x) * upstream).sum()

    num_dgamma = numeric_grad(f_gamma, ln.gamma.copy())
    all_pass &= check("LayerNorm dgamma", analytic_dgamma, num_dgamma)

    ln.zero_grad()
    ln.forward(x)
    ln.backward(upstream)
    analytic_dbeta = ln.grads["beta"].copy()

    def f_beta(bv):
        ln.beta = bv
        return (ln.forward(x) * upstream).sum()

    num_dbeta = numeric_grad(f_beta, ln.beta.copy())
    all_pass &= check("LayerNorm dbeta", analytic_dbeta, num_dbeta)

    # ---------------- Embedding ----------------
    print("\n=== Embedding ===")
    emb = Embedding(10, 4, rng)
    ids = rng.randint(0, 10, size=(2, 5))
    upstream = rng.normal(size=(2, 5, 4))
    emb.forward(ids)
    emb.backward(upstream)
    analytic_dtable = emb.grads["table"].copy()

    def f_table(tv):
        emb.table = tv
        return (emb.forward(ids) * upstream).sum()

    num_dtable = numeric_grad(f_table, emb.table.copy())
    all_pass &= check("Embedding dtable", analytic_dtable, num_dtable)

    # ---------------- GELU ----------------
    print("\n=== GELU ===")
    gelu_layer = GELU()
    x = rng.normal(size=(2, 3, 4))
    upstream = rng.normal(size=(2, 3, 4))

    def f_x(xv):
        return (gelu_layer.forward(xv) * upstream).sum()

    gelu_layer.forward(x)
    dx = gelu_layer.backward(upstream)
    num_dx = numeric_grad(f_x, x.copy())
    all_pass &= check("GELU dx", dx, num_dx)

    # ---------------- softmax + cross entropy ----------------
    print("\n=== softmax backward ===")
    x = rng.normal(size=(2, 3, 5))
    upstream = rng.normal(size=(2, 3, 5))

    def f_x(xv):
        return (softmax(xv) * upstream).sum()

    y = softmax(x)
    dx = softmax_backward(upstream, y)
    num_dx = numeric_grad(f_x, x.copy())
    all_pass &= check("softmax dx", dx, num_dx)

    print("\n=== cross_entropy_loss ===")
    logits = rng.normal(size=(2, 3, 6))
    targets = rng.randint(0, 6, size=(2, 3))

    def f_logits(lv):
        loss, _ = cross_entropy_loss(lv, targets)
        return loss

    loss, dlogits = cross_entropy_loss(logits, targets)
    num_dlogits = numeric_grad(f_logits, logits.copy())
    all_pass &= check("cross_entropy dlogits", dlogits, num_dlogits)

    # ---------------- MultiHeadSelfAttention ----------------
    print("\n=== MultiHeadSelfAttention ===")
    from attention import MultiHeadSelfAttention

    attn_rng = np.random.RandomState(1)
    mha = MultiHeadSelfAttention(dim=8, n_heads=2, rng=attn_rng)
    x = attn_rng.normal(size=(2, 4, 8))
    upstream = attn_rng.normal(size=(2, 4, 8))

    def f_x(xv):
        return (mha.forward(xv) * upstream).sum()

    mha.forward(x)
    dx = mha.backward(upstream)
    num_dx = numeric_grad(f_x, x.copy())
    all_pass &= check("MHA dx", dx, num_dx)

    for name, layer in [("q_proj", mha.q_proj), ("k_proj", mha.k_proj),
                         ("v_proj", mha.v_proj), ("out_proj", mha.out_proj)]:
        mha.zero_grad()
        mha.forward(x)
        mha.backward(upstream)
        analytic_dW = layer.grads["W"].copy()

        def f_W(Wv, layer=layer):
            layer.W = Wv
            return (mha.forward(x) * upstream).sum()

        num_dW = numeric_grad(f_W, layer.W.copy())
        all_pass &= check(f"MHA {name}.dW", analytic_dW, num_dW)

    print("\n" + ("ALL LAYER GRADIENT CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
