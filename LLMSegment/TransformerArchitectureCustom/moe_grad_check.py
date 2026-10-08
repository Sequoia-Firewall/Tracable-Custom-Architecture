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
from layers import softmax
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
    moe2.prior_strength = 0.5

    # No JudgeLayer in this standalone test -- query the Judge directly and treat
    # cluster_ids as FIXED across the perturbations below (same reasoning as the
    # top-k mask itself: a discrete nearest-neighbor assignment is locally constant
    # under a tiny finite-difference perturbation almost everywhere).
    _, cluster_labels2, _ = judge2.query_batch(x2.reshape(-1, 6))
    cluster_ids2 = cluster_labels2.reshape(x2.shape[0], x2.shape[1])

    def f_x2(xv):
        return (moe2.forward(xv, cluster_ids=cluster_ids2) * upstream2).sum() + \
            moe2.aux_loss_weight * moe2.last_aux_loss

    moe2.forward(x2, cluster_ids=cluster_ids2)
    dx2 = moe2.backward(upstream2)
    num_dx2 = numeric_grad(f_x2, x2.copy())
    all_pass &= check("MoE+shared+outlier dx", dx2, num_dx2, tol=1e-3)

    moe2.zero_grad()
    moe2.forward(x2, cluster_ids=cluster_ids2)
    moe2.backward(upstream2)
    shared0_fc1 = moe2.shared_experts[0].fc1
    analytic_shared0 = shared0_fc1.grads["W"].copy()

    def f_shared0W(Wv):
        shared0_fc1.W = Wv
        return (moe2.forward(x2, cluster_ids=cluster_ids2) * upstream2).sum() + \
            moe2.aux_loss_weight * moe2.last_aux_loss

    num_shared0 = numeric_grad(f_shared0W, shared0_fc1.W.copy())
    all_pass &= check("MoE shared_expert0.fc1.dW", analytic_shared0, num_shared0, tol=1e-3)

    moe2.zero_grad()
    moe2.forward(x2, cluster_ids=cluster_ids2)
    moe2.backward(upstream2)
    analytic_gate2 = moe2.gate.grads["W"].copy()

    def f_gate2W(Wv):
        moe2.gate.W = Wv
        return (moe2.forward(x2, cluster_ids=cluster_ids2) * upstream2).sum() + \
            moe2.aux_loss_weight * moe2.last_aux_loss

    num_gate2 = numeric_grad(f_gate2W, moe2.gate.W.copy())
    all_pass &= check("MoE gate.dW (with outlier masking active)", analytic_gate2, num_gate2, tol=1e-3)

    # Sanity: confirm the outlier branch actually engaged (otherwise the check above
    # isn't testing what it claims to).
    _, _, _, _, keep_mask2, *_ = moe2._cache
    n_outliers = int((keep_mask2 == 0.0).sum())
    print(f"[{'PASS' if n_outliers > 0 else 'FAIL'}] outlier branch engaged for {n_outliers} token(s) in this check")
    all_pass &= n_outliers > 0

    # ---------------- ReviewerLayer, isolated ----------------
    print("\n=== ReviewerLayer (isolated) ===")
    from moe import ReviewerLayer

    reviewer = ReviewerLayer()
    rrng = np.random.RandomState(7)
    b, s, n_e, dim = 2, 3, 4, 5
    expert_outs_r = rrng.normal(size=(b, s, n_e, dim)) * 2.0
    # Build a realistic top-2-style gate_weights: random logits, softmax
    # over the top 2 per token, zero elsewhere -- exercises a genuine
    # selection PATTERN, not an arbitrary dense weight vector.
    raw_logits_r = rrng.normal(size=(b, s, n_e))
    topk_idx_r = np.argsort(-raw_logits_r, axis=-1)[..., :2]
    sel_mask = np.zeros_like(raw_logits_r, dtype=bool)
    np.put_along_axis(sel_mask, topk_idx_r, True, axis=-1)
    masked_logits_r = np.where(sel_mask, raw_logits_r, np.finfo(raw_logits_r.dtype).min)
    gate_weights_r = softmax(masked_logits_r, axis=-1)
    keep_mask_r = np.ones((b, s))
    keep_mask_r[0, 0] = 0.0  # one forced "outlier" token, exercises that path too

    upstream_r = rrng.normal(size=(b, s, dim))

    def f_gate_r(gw):
        combined, _ = reviewer.combine(gw, expert_outs_r, keep_mask_r)
        out = (combined[..., :, None] * expert_outs_r).sum(axis=-2)
        return (out * upstream_r).sum()

    combined0, cache0 = reviewer.combine(gate_weights_r, expert_outs_r, keep_mask_r)
    d_effective = np.einsum("bsd,bsed->bse", upstream_r, expert_outs_r)
    d_gate_analytic = reviewer.backward(d_effective, cache0, keep_mask_r)

    # Only check at SELECTED (nonzero gate_weight) entries -- gate_weights==0
    # is a hard decision boundary (raw_confidence's "* (gate_weights > 0)"
    # mask flips discontinuously right at zero), same category of issue as
    # top-k ties elsewhere in this file. Unselected entries get their
    # (correct, zero) gradient from the existing softmax_backward step in
    # moe.py, not from ReviewerLayer itself.
    num_grad_full = numeric_grad(f_gate_r, gate_weights_r.copy())
    selected = gate_weights_r > 0
    err = np.max(np.abs(d_gate_analytic[selected] - num_grad_full[selected]))
    ok = err < 1e-3
    print(f"[{'PASS' if ok else 'FAIL'}] ReviewerLayer d_gate_weights (selected entries only)  max_abs_err={err:.8f}")
    all_pass &= ok

    # ---------------- ReviewerLayer wired into a real SparseMoEFeedForward ----------------
    print("\n=== SparseMoEFeedForward with use_reviewer=True (integration) ===")
    # aux_loss_weight=0 here deliberately -- that gradient path is already
    # thoroughly covered above; this section isolates the Reviewer integration.
    moe3 = SparseMoEFeedForward(dim=6, hidden_dim=10, n_experts=4, rng=rng, top_k=2,
                                 aux_loss_weight=0.0, use_reviewer=True)
    x3 = rng.normal(size=(2, 3, 6))
    upstream3 = rng.normal(size=(2, 3, 6))

    moe3.forward(x3)
    dx3 = moe3.backward(upstream3)

    moe3.zero_grad()
    moe3.forward(x3)
    moe3.backward(upstream3)
    analytic_gate3 = moe3.gate.grads["W"].copy()

    def f_gate3W(Wv):
        moe3.gate.W = Wv
        return (moe3.forward(x3) * upstream3).sum()

    num_gate3 = numeric_grad(f_gate3W, moe3.gate.W.copy())
    all_pass &= check("MoE+reviewer gate.dW", analytic_gate3, num_gate3, tol=1e-3)

    # dx/expert-weight checks need a DIFFERENT target scalar: ReviewerLayer's
    # confidence (r) is deliberately stop-gradient w.r.t. expert_outs (see
    # moe.py docstring -- TCA's own variance is bookkeeping, not trained
    # either), so backward() computes the gradient TREATING r AS CONSTANT.
    # A naive finite-difference perturbs expert_outs, which genuinely does
    # change r too -- that's the TRUE full derivative of a different
    # (non-stop-gradient) function than backward() implements, not a bug in
    # backward(). Freeze r at its original value to check the function
    # backward() actually represents (same "measure what backward() computes"
    # fix as the aux-loss lesson above, applied to a stop-gradient instead
    # of an included-but-forgotten term). S = sum(g*r) is NOT frozen --
    # S genuinely depends on g (the gate's own softmax output), and the
    # analytic backward's "dot" correction term already accounts for
    # exactly that dependence; freezing S too would test a different
    # (wrong) function and was the actual bug in this check's first draft.
    moe3.zero_grad()
    moe3.forward(x3)
    moe3.backward(upstream3)
    _, _, _, gate_weights3, keep_mask3, _, _, _, reviewer_cache3 = moe3._cache
    r_fixed = reviewer_cache3["r"]

    def frozen_r_output(xv):
        logits = moe3.gate.forward(xv)
        topk_idx = np.argsort(-logits, axis=-1)[..., :moe3.top_k]
        mask = np.zeros_like(logits, dtype=bool)
        np.put_along_axis(mask, topk_idx, True, axis=-1)
        neg_inf = np.finfo(logits.dtype).min
        gw = softmax(np.where(mask, logits, neg_inf), axis=-1)
        expert_outs_v = np.stack([e.forward(xv) for e in moe3.experts], axis=-2)
        S = (gw * r_fixed).sum(axis=-1, keepdims=True)  # recomputed fresh from (perturbed gw, frozen r)
        combined = gw * r_fixed / S
        effective = combined * keep_mask3[..., None]
        return (effective[..., :, None] * expert_outs_v).sum(axis=-2)

    expert0_fc1_3 = moe3.experts[0].fc1
    analytic_e0_3 = expert0_fc1_3.grads["W"].copy()

    def f_expert0W_3(Wv):
        expert0_fc1_3.W = Wv
        return (frozen_r_output(x3) * upstream3).sum()

    num_e0_3 = numeric_grad(f_expert0W_3, expert0_fc1_3.W.copy())
    all_pass &= check("MoE+reviewer expert0.fc1.dW (r frozen)", analytic_e0_3, num_e0_3, tol=1e-3)

    def f_x3_frozen(xv):
        return (frozen_r_output(xv) * upstream3).sum()

    num_dx3_frozen = numeric_grad(f_x3_frozen, x3.copy())
    all_pass &= check("MoE+reviewer dx (r frozen)", dx3, num_dx3_frozen, tol=1e-3)

    print("\n" + ("ALL MoE GRADIENT CHECKS PASSED" if all_pass else "SOME MoE CHECKS FAILED"))
    assert all_pass
