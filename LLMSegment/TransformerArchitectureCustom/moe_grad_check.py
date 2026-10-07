"""
Gradient checks for SparseMoEFeedForward, standalone (no TransformerBlock
yet). Top-k selection is a hard decision boundary -- finite-difference
checks are only valid away from a logit tie, which a random init/eps=1e-6
essentially never lands on, so a failure here is a real bug, not a
boundary artifact (if one ever shows up intermittently, that's the first
thing to suspect).

Checks the two independent gradient paths SEPARATELY, since the ordinary
"dot output with random upstream" trick only exercises the PRIMARY path
(the selected experts + their gate logits) -- the auxiliary
load-balancing loss's gradient (the only path reaching UNSELECTED
experts' logits) needs its own isolated check, done here by zeroing the
primary upstream gradient and treating the aux loss itself as the scalar
being checked.
"""
import numpy as np
from moe import SparseMoEFeedForward
from grad_check import numeric_grad, check

if __name__ == "__main__":
    rng = np.random.RandomState(5)
    all_pass = True

    moe = SparseMoEFeedForward(dim=6, hidden_dim=10, n_experts=4, rng=rng, top_k=2, aux_loss_weight=0.1)
    x = rng.normal(size=(2, 3, 6))
    upstream = rng.normal(size=(2, 3, 6))

    # ---------------- primary + aux combined: this is what backward(dout) ACTUALLY
    # differentiates every real call -- it always bundles the aux-loss gradient in
    # (correct: in real training, aux_loss_weight*aux_loss is genuinely part of the
    # total loss every step), so the check's target scalar must include it too, or
    # the "primary-only" test is checking a different function than backward()
    # computes and will show a spurious mismatch. ----
    print("=== Primary path (output-weighted loss + aux loss, matching what backward() actually computes) ===")

    def total_scalar(xv, upstream_v):
        out = moe.forward(xv)
        return (out * upstream_v).sum() + moe.aux_loss_weight * moe.last_aux_loss

    def f_x(xv):
        return total_scalar(xv, upstream)

    moe.forward(x)
    dx = moe.backward(upstream)
    num_dx = numeric_grad(f_x, x.copy())
    all_pass &= check("MoE dx (primary+aux)", dx, num_dx, tol=1e-3)

    moe.zero_grad()
    moe.forward(x)
    moe.backward(upstream)
    analytic_dW = moe.gate.grads["W"].copy()

    def f_gateW(Wv):
        moe.gate.W = Wv
        return total_scalar(x, upstream)

    num_dW = numeric_grad(f_gateW, moe.gate.W.copy())
    all_pass &= check("MoE gate.dW (primary+aux)", analytic_dW, num_dW, tol=1e-3)

    moe.zero_grad()
    moe.forward(x)
    moe.backward(upstream)
    expert0_fc1 = moe.experts[0].fc1
    analytic_dW0 = expert0_fc1.grads["W"].copy()

    def f_expert0W(Wv):
        expert0_fc1.W = Wv
        return total_scalar(x, upstream)  # aux loss doesn't depend on expert weights at all,
        # so this is equivalent to the output-only version for this particular parameter --
        # included for consistency, not because it changes anything here.

    num_dW0 = numeric_grad(f_expert0W, expert0_fc1.W.copy())
    all_pass &= check("MoE expert0.fc1.dW (primary+aux)", analytic_dW0, num_dW0, tol=1e-3)

    # ---------------- aux-loss path, isolated ----------------
    print("\n=== Auxiliary load-balancing loss path (isolated) ===")
    zero_upstream = np.zeros_like(x)

    def f_x_aux(xv):
        moe.forward(xv)
        return moe.last_aux_loss * moe.aux_loss_weight  # backward() already applies aux_loss_weight once;
        # here we need the scalar whose gradient backward() actually computes:
        # d(aux_loss_weight * last_aux_loss)/d(.)

    moe.forward(x)
    aux_val = moe.last_aux_loss * moe.aux_loss_weight
    dx_aux = moe.backward(zero_upstream)  # primary path contributes exactly 0 when upstream is 0
    num_dx_aux = numeric_grad(f_x_aux, x.copy())
    all_pass &= check("MoE dx (aux loss only)", dx_aux, num_dx_aux, tol=1e-3)

    moe.zero_grad()
    moe.forward(x)
    moe.backward(zero_upstream)
    analytic_gateW_aux = moe.gate.grads["W"].copy()

    def f_gateW_aux(Wv):
        moe.gate.W = Wv
        moe.forward(x)
        return moe.last_aux_loss * moe.aux_loss_weight

    num_gateW_aux = numeric_grad(f_gateW_aux, moe.gate.W.copy())
    all_pass &= check("MoE gate.dW (aux loss only)", analytic_gateW_aux, num_gateW_aux, tol=1e-3)

    # ---------------- shared experts + outlier handling (Tier 2 additions) ----------------
    print("\n=== Shared experts + outlier (noise) handling ===")
    from judge import DistributionJudge

    moe2 = SparseMoEFeedForward(dim=6, hidden_dim=10, n_experts=4, rng=rng, top_k=2,
                                 aux_loss_weight=0.1, n_shared_experts=2,
                                 route_noise_to_shared_only=True)
    x2 = rng.normal(size=(2, 3, 6))
    upstream2 = rng.normal(size=(2, 3, 6))

    # Fit a tiny Judge directly on x2's own token vectors, with one token forced
    # far away so it's guaranteed to be noise (cluster -1) -- this exercises the
    # keep_mask=0 branch, not just the "no judge" default path.
    flat_x2 = x2.reshape(-1, 6).copy()
    flat_x2[0] += 1000.0  # force this one token into its own isolated region
    judge2 = DistributionJudge(eps=2.0, min_samples=2)
    judge2.fit(flat_x2)
    moe2.attach_judge(judge2, prior_strength=0.5)

    def f_x2(xv):
        return (moe2.forward(xv) * upstream2).sum() + moe2.aux_loss_weight * moe2.last_aux_loss

    moe2.forward(x2)
    dx2 = moe2.backward(upstream2)
    num_dx2 = numeric_grad(f_x2, x2.copy())
    all_pass &= check("MoE+shared+outlier dx", dx2, num_dx2, tol=1e-3)

    moe2.zero_grad()
    moe2.forward(x2)
    moe2.backward(upstream2)
    shared0_fc1 = moe2.shared_experts[0].fc1
    analytic_shared0 = shared0_fc1.grads["W"].copy()

    def f_shared0W(Wv):
        shared0_fc1.W = Wv
        return (moe2.forward(x2) * upstream2).sum() + moe2.aux_loss_weight * moe2.last_aux_loss

    num_shared0 = numeric_grad(f_shared0W, shared0_fc1.W.copy())
    all_pass &= check("MoE shared_expert0.fc1.dW", analytic_shared0, num_shared0, tol=1e-3)

    moe2.zero_grad()
    moe2.forward(x2)
    moe2.backward(upstream2)
    analytic_gate2 = moe2.gate.grads["W"].copy()

    def f_gate2W(Wv):
        moe2.gate.W = Wv
        return (moe2.forward(x2) * upstream2).sum() + moe2.aux_loss_weight * moe2.last_aux_loss

    num_gate2 = numeric_grad(f_gate2W, moe2.gate.W.copy())
    all_pass &= check("MoE gate.dW (with outlier masking active)", analytic_gate2, num_gate2, tol=1e-3)

    # Sanity: confirm the outlier branch actually engaged (otherwise the check above
    # isn't testing what it claims to).
    _, _, _, _, keep_mask2, *_ = moe2._cache
    n_outliers = int((keep_mask2 == 0.0).sum())
    print(f"[{'PASS' if n_outliers > 0 else 'FAIL'}] outlier branch engaged for {n_outliers} token(s) in this check")
    all_pass &= n_outliers > 0

    print("\n" + ("ALL MoE GRADIENT CHECKS PASSED" if all_pass else "SOME MoE CHECKS FAILED"))
    assert all_pass
