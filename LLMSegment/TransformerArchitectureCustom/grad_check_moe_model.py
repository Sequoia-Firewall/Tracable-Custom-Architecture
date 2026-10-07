"""
End-to-end numerical gradient check on a full GPTStyleTransformer with
EVERY block's FFN replaced by an independent SparseMoEFeedForward (Option
A: each layer has its own gate+experts). Mirrors grad_check_full_model.py
but for the MoE-wired model -- catches wiring mistakes at the
block<->MoE-layer boundary, and confirms different blocks' MoE layers
really are independent (perturbing block0's expert shouldn't be confused
with block1's in the gradient bookkeeping).
"""
import numpy as np
from transformer import GPTStyleTransformer
from grad_check import numeric_grad, check
from layers import cross_entropy_loss

TOL = 2e-3  # same loosening as grad_check_full_model.py -- accumulated float
            # error over many composed ops, now with MoE's extra gate+aux-loss ops


def loss_fn(model, token_ids, targets):
    """
    The scalar backward() ACTUALLY differentiates end-to-end: cross-entropy
    PLUS every MoE block's aux_loss_weight*aux_loss term (each block's
    backward() unconditionally includes its own -- correct for real
    training, where every one of those terms is genuinely always part of
    the total loss every step). Omitting them here would silently check a
    different function than backward() computes, as it did in
    moe_grad_check.py's first draft.
    """
    logits = model.forward(token_ids)
    loss, _ = cross_entropy_loss(logits, targets)
    for block in model.blocks:
        if hasattr(block.ffn, "last_aux_loss"):
            loss = loss + block.ffn.aux_loss_weight * block.ffn.last_aux_loss
    return loss


if __name__ == "__main__":
    rng = np.random.RandomState(11)
    model = GPTStyleTransformer(
        vocab_size=6, dim=8, n_heads=2, n_layers=2, ffn_hidden_dim=12,
        max_seq_len=5, rng_seed=9,
        moe_cfg={"n_experts": 3, "top_k": 2, "aux_loss_weight": 0.05},
    )
    token_ids = rng.randint(0, 6, size=(2, 4))
    targets = rng.randint(0, 6, size=(2, 4))

    all_pass = True
    model.zero_grad()
    loss, _ = model.loss_and_backward(token_ids, targets)
    print(f"initial loss={loss:.4f}")

    checks = [
        ("token_emb.table", model.token_emb, "table"),
        ("block0.ffn.gate.W", model.blocks[0].ffn.gate, "W"),
        ("block0.ffn.expert0.fc1.W", model.blocks[0].ffn.experts[0].fc1, "W"),
        ("block0.ffn.expert2.fc2.W", model.blocks[0].ffn.experts[2].fc2, "W"),
        ("block1.ffn.gate.W", model.blocks[1].ffn.gate, "W"),
        ("block1.ffn.expert1.fc1.W", model.blocks[1].ffn.experts[1].fc1, "W"),
        ("block0.attn.q_proj.W", model.blocks[0].attn.q_proj, "W"),
        ("block1.ln2.gamma", model.blocks[1].ln2, "gamma"),
        ("ln_f.gamma", model.ln_f, "gamma"),
        ("head.W", model.head, "W"),
    ]

    for name, layer, param_name in checks:
        model.zero_grad()
        model.loss_and_backward(token_ids, targets)
        analytic = layer.grads[param_name].copy()
        param = getattr(layer, param_name)

        def f(pv, layer=layer, param_name=param_name):
            setattr(layer, param_name, pv)
            return loss_fn(model, token_ids, targets)

        numeric = numeric_grad(f, param.copy())
        setattr(layer, param_name, param)
        all_pass &= check(name, analytic, numeric, tol=TOL)

    # Independence check: perturbing block0's expert0 weight should produce
    # ZERO gradient at block1's expert0 -- they're separate objects/params,
    # not accidentally shared/aliased.
    same_object = model.blocks[0].ffn.experts[0].fc1.W is model.blocks[1].ffn.experts[0].fc1.W
    print(f"\n[{'FAIL' if same_object else 'PASS'}] block0/block1 expert0.fc1.W are independent objects "
          f"(not aliased): {not same_object}")
    all_pass &= not same_object

    # ---------------- shared experts + judge-driven outlier handling, at full-model scale ----------------
    print("\n=== Full model with shared experts + judge outlier handling ===")
    from judge import DistributionJudge

    model2 = GPTStyleTransformer(
        vocab_size=6, dim=8, n_heads=2, n_layers=2, ffn_hidden_dim=12,
        max_seq_len=5, rng_seed=13,
        moe_cfg={"n_experts": 3, "top_k": 2, "aux_loss_weight": 0.05, "n_shared_experts": 2},
    )
    token_ids2 = rng.randint(0, 6, size=(2, 4))
    targets2 = rng.randint(0, 6, size=(2, 4))

    model2.forward(token_ids2)
    ffn_in, *_ = model2.blocks[0].ffn._cache
    flat = ffn_in.reshape(-1, ffn_in.shape[-1]).copy()
    flat[0] += 1000.0  # guarantee at least one real outlier
    judge = DistributionJudge(eps=2.0, min_samples=2)
    judge.fit(flat)
    model2.blocks[0].ffn.attach_judge(judge, prior_strength=0.3)

    def loss_fn2(m, t, y):
        return loss_fn(m, t, y)  # same aux-loss-inclusive scalar, reused for this model

    checks2 = [
        ("block0.ffn.gate.W", model2.blocks[0].ffn.gate, "W"),
        ("block0.ffn.shared0.fc1.W", model2.blocks[0].ffn.shared_experts[0].fc1, "W"),
        ("block0.ffn.expert1.fc2.W", model2.blocks[0].ffn.experts[1].fc2, "W"),
        ("block1.ffn.shared1.fc2.W", model2.blocks[1].ffn.shared_experts[1].fc2, "W"),
    ]
    for name, layer, param_name in checks2:
        model2.zero_grad()
        model2.loss_and_backward(token_ids2, targets2)
        analytic = layer.grads[param_name].copy()
        param = getattr(layer, param_name)

        def f(pv, layer=layer, param_name=param_name):
            setattr(layer, param_name, pv)
            return loss_fn2(model2, token_ids2, targets2)

        numeric = numeric_grad(f, param.copy())
        setattr(layer, param_name, param)
        all_pass &= check(name, analytic, numeric, tol=TOL)

    _, _, _, _, keep_mask, *_ = model2.blocks[0].ffn._cache
    n_outliers = int((keep_mask == 0.0).sum())
    print(f"[{'PASS' if n_outliers > 0 else 'FAIL'}] outlier branch engaged for {n_outliers} token(s)")
    all_pass &= n_outliers > 0

    print("\n" + ("ALL MoE FULL-MODEL GRADIENT CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
    assert all_pass
