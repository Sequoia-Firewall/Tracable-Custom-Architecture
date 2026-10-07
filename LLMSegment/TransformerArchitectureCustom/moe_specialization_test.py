"""
Does Option A ("each block gets its own independent experts") actually do
anything, or is it just plumbing with no real effect? Three honest checks,
not assumed:

1. Does the MoE-wired model still learn the task as well as the dense
   model did (train_toy.py's bar: 100% held-out accuracy)?
2. Does the auxiliary load-balancing loss actually matter -- compare
   expert-usage balance WITH vs WITHOUT it (aux_loss_weight=0 ablation).
3. Do the two layers' gates actually specialize DIFFERENTLY from each
   other, or do they end up making the same per-token choice (which would
   mean "independent" experts in name only)?
"""
import numpy as np
from transformer import GPTStyleTransformer

VOCAB_SIZE = 6
DELAY = 2
SEQ_LEN = 12
BATCH_SIZE = 16
N_STEPS = 400
LR = 0.05


def make_batch(rng, batch_size, seq_len, delay, vocab_size):
    tokens = np.zeros((batch_size, seq_len), dtype=np.int64)
    tokens[:, :delay] = rng.randint(0, vocab_size, size=(batch_size, delay))
    for i in range(delay, seq_len):
        tokens[:, i] = tokens[:, i - delay]
    return tokens[:, :-1], tokens[:, 1:]


def train(moe_cfg, seed):
    rng = np.random.RandomState(seed)
    model = GPTStyleTransformer(
        vocab_size=VOCAB_SIZE, dim=32, n_heads=4, n_layers=2, ffn_hidden_dim=64,
        max_seq_len=SEQ_LEN, rng_seed=seed, moe_cfg=moe_cfg,
    )
    for step in range(N_STEPS):
        inputs, targets = make_batch(rng, BATCH_SIZE, SEQ_LEN, DELAY, VOCAB_SIZE)
        model.zero_grad()
        loss, _ = model.loss_and_backward(inputs, targets)
        model.sgd_step(LR)
    return model, loss


def eval_accuracy(model, rng):
    test_inputs, test_targets = make_batch(rng, 200, SEQ_LEN, DELAY, VOCAB_SIZE)
    logits = model.forward(test_inputs)
    preds = logits.argmax(axis=-1)
    determined_mask = (np.arange(test_inputs.shape[1]) + 1) >= DELAY
    return (preds[:, determined_mask] == test_targets[:, determined_mask]).mean()


def expert_usage(model, inputs):
    """Returns, per block, the fraction of tokens routed to each expert
    (reads mask straight from the ffn's cache after a forward pass)."""
    model.forward(inputs)
    usage = []
    for block in model.blocks:
        _, _, mask, _, _, _, _ = block.ffn._cache
        usage.append(mask.mean(axis=(0, 1)))  # (n_experts,)
    return usage


def imbalance_ratio(usage_frac):
    """max/mean -- 1.0 would be perfectly balanced; n_experts would mean
    total collapse onto one expert."""
    return usage_frac.max() / usage_frac.mean()


if __name__ == "__main__":
    N_EXPERTS = 4
    print("=== 1. Does the MoE-wired model still learn the task? ===")
    model_aux, final_loss = train({"n_experts": N_EXPERTS, "top_k": 1, "aux_loss_weight": 0.02}, seed=1)
    acc = eval_accuracy(model_aux, np.random.RandomState(100))
    print(f"final train loss={final_loss:.4f}, held-out accuracy={acc * 100:.1f}%")
    assert acc > 0.95, f"MoE model did not learn the task well: accuracy={acc:.3f}"
    print("[PASS] MoE-wired model learns the task (matches the dense-FFN bar)")

    print("\n=== 2. Does the aux load-balancing loss actually prevent collapse? ===")
    model_no_aux, _ = train({"n_experts": N_EXPERTS, "top_k": 1, "aux_loss_weight": 0.0}, seed=1)
    eval_rng = np.random.RandomState(42)
    eval_inputs, _ = make_batch(eval_rng, 300, SEQ_LEN, DELAY, VOCAB_SIZE)

    usage_aux = expert_usage(model_aux, eval_inputs)
    usage_no_aux = expert_usage(model_no_aux, eval_inputs)

    for i in range(2):
        ir_aux = imbalance_ratio(usage_aux[i])
        ir_no_aux = imbalance_ratio(usage_no_aux[i])
        print(f"  block{i}: usage WITH aux={np.round(usage_aux[i], 3)} (imbalance={ir_aux:.2f}x)  "
              f"WITHOUT aux={np.round(usage_no_aux[i], 3)} (imbalance={ir_no_aux:.2f}x)")

    mean_ir_aux = np.mean([imbalance_ratio(u) for u in usage_aux])
    mean_ir_no_aux = np.mean([imbalance_ratio(u) for u in usage_no_aux])
    print(f"  mean imbalance: with aux={mean_ir_aux:.2f}x, without aux={mean_ir_no_aux:.2f}x "
          f"(perfectly balanced=1.0x, total collapse={N_EXPERTS}.0x)")
    if mean_ir_no_aux > mean_ir_aux:
        print("[PASS] aux loss measurably reduces expert-usage imbalance")
    else:
        print("[INFO] aux loss did NOT reduce imbalance in this run -- reporting honestly, not asserting")

    print("\n=== 3. Do the two layers' gates specialize DIFFERENTLY from each other? ===")
    model_aux.forward(eval_inputs)
    _, _, mask0, _, _, _, _ = model_aux.blocks[0].ffn._cache
    _, _, mask1, _, _, _, _ = model_aux.blocks[1].ffn._cache
    chosen0 = mask0.argmax(axis=-1)  # (batch, seq) -- top-1, so argmax of the one-hot mask is the chosen expert
    chosen1 = mask1.argmax(axis=-1)
    agreement = (chosen0 == chosen1).mean()
    print(f"  block0/block1 chosen-expert agreement rate: {agreement * 100:.1f}% "
          f"(chance level for {N_EXPERTS} experts ~= {100 / N_EXPERTS:.1f}%)")
    if agreement < 0.6:
        print("[PASS] the two layers route tokens to experts largely independently of each other "
              "(not just copying the same decision)")
    else:
        print("[INFO] layers show high agreement -- reporting honestly; could mean genuinely "
              "correlated specialization, or both collapsing toward similar simple heuristics")

    print("\nMoE SPECIALIZATION TEST COMPLETE")
