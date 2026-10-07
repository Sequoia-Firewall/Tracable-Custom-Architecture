"""
End-to-end numerical gradient check on the FULL composed GPTStyleTransformer
(embeddings -> N blocks -> final LN -> output head -> cross-entropy loss).
Per-layer checks (grad_check.py) verify each piece in isolation; this
catches wiring mistakes at the composition boundaries themselves -- e.g. a
residual branch that drops or double-counts a gradient contribution --
which per-layer checks cannot see.
"""
import numpy as np
from transformer import GPTStyleTransformer
from grad_check import numeric_grad, check

EPS = 1e-6
TOL = 2e-3  # looser than the per-layer 1e-4 -- accumulated float error over
            # many composed ops (2 layers x (attn + ffn) + embeddings + head)


def loss_fn(model, token_ids, targets):
    logits = model.forward(token_ids)
    from layers import cross_entropy_loss
    loss, _ = cross_entropy_loss(logits, targets)
    return loss


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    model = GPTStyleTransformer(
        vocab_size=6, dim=8, n_heads=2, n_layers=2, ffn_hidden_dim=16,
        max_seq_len=5, rng_seed=7,
    )
    token_ids = rng.randint(0, 6, size=(2, 4))
    targets = rng.randint(0, 6, size=(2, 4))

    all_pass = True

    model.zero_grad()
    loss, logits = model.loss_and_backward(token_ids, targets)
    print(f"initial loss={loss:.4f}")

    checks = [
        ("token_emb.table", model.token_emb, "table"),
        ("pos_emb.table", model.pos_emb, "table"),
        ("block0.ln1.gamma", model.blocks[0].ln1, "gamma"),
        ("block0.attn.q_proj.W", model.blocks[0].attn.q_proj, "W"),
        ("block0.attn.out_proj.W", model.blocks[0].attn.out_proj, "W"),
        ("block0.ffn.fc1.W", model.blocks[0].ffn.fc1, "W"),
        ("block0.ffn.fc2.W", model.blocks[0].ffn.fc2, "W"),
        ("block1.ln2.beta", model.blocks[1].ln2, "beta"),
        ("block1.attn.k_proj.W", model.blocks[1].attn.k_proj, "W"),
        ("block1.ffn.fc2.W", model.blocks[1].ffn.fc2, "W"),
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
        setattr(layer, param_name, param)  # restore exact original object
        all_pass &= check(name, analytic, numeric, tol=TOL)

    print("\n" + ("ALL FULL-MODEL GRADIENT CHECKS PASSED" if all_pass else "SOME FULL-MODEL CHECKS FAILED"))
