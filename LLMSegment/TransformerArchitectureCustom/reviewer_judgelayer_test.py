"""
Two things to confirm about the refactor requested to bring this
architecture closer to TCA: (1) JudgeLayer is now a real, VISIBLE,
traceable layer -- not just internal bookkeeping -- and (2) ReviewerLayer
actually changes the final combination weights, not a no-op sitting on
top of the gate's own softmax.
"""
import numpy as np
from transformer import GPTStyleTransformer
from judge import DistributionJudge
from trace import TraceRecorder

VOCAB_SIZE = 6
DELAY = 2
SEQ_LEN = 12
BATCH_SIZE = 16
LR = 0.05
N_STEPS = 300


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
    for _ in range(N_STEPS):
        inputs, targets = make_batch(rng, BATCH_SIZE, SEQ_LEN, DELAY, VOCAB_SIZE)
        model.zero_grad()
        model.loss_and_backward(inputs, targets)
        model.sgd_step(LR)
    return model, rng


def eval_accuracy(model, rng):
    test_inputs, test_targets = make_batch(rng, 200, SEQ_LEN, DELAY, VOCAB_SIZE)
    logits = model.forward(test_inputs)
    preds = logits.argmax(axis=-1)
    determined = (np.arange(test_inputs.shape[1]) + 1) >= DELAY
    return (preds[:, determined] == test_targets[:, determined]).mean()


if __name__ == "__main__":
    all_pass = True
    N_EXPERTS = 4

    print("=== 1. JudgeLayer: is it actually a visible, traceable layer now? ===")
    model, rng = train(
        {"n_experts": N_EXPERTS, "top_k": 1, "aux_loss_weight": 0.02, "n_shared_experts": 1}, seed=1,
    )
    fit_inputs, _ = make_batch(rng, 300, SEQ_LEN, DELAY, VOCAB_SIZE)
    model.forward(fit_inputs)
    ffn_vectors, *_ = model.blocks[0].ffn._cache
    flat = ffn_vectors.reshape(-1, ffn_vectors.shape[-1])
    judge = DistributionJudge(eps=np.median(
        np.linalg.norm(flat[:150, None, :] - flat[None, :150, :], axis=-1)[np.triu_indices(150, k=1)]
    ) * 0.4, min_samples=10)
    judge.fit(flat)
    model.blocks[0].judge_layer.attach(judge)

    trace = TraceRecorder()
    test_inputs, _ = make_batch(rng, 20, SEQ_LEN, DELAY, VOCAB_SIZE)
    model.forward(test_inputs, trace=trace)
    judge_records = [r for r in trace.records if r["kind"] == "judge"]
    # Only block0 has a judge attached -- JudgeLayer.forward() returns None (no
    # record at all) when nothing's attached, so exactly 1 record is expected,
    # not one per block.
    ok = len(judge_records) == 1
    print(f"  judge trace records found: {len(judge_records)} (expected 1 -- only block0 has a judge attached)")
    print(f"  block0's judge record: {judge_records[0]}")
    all_pass &= ok and "cluster_histogram" in judge_records[0] and "mean_novelty_distance" in judge_records[0]
    print(f"[{'PASS' if all_pass else 'FAIL'}] JudgeLayer produces real, inspectable trace records "
          f"(cluster histogram + novelty distance), independent of the FFN it feeds")

    print("\n=== 2. ReviewerLayer: does it actually change the combination weights? ===")
    model_r, rng_r = train(
        {"n_experts": N_EXPERTS, "top_k": 2, "aux_loss_weight": 0.02, "use_reviewer": True}, seed=2,
    )
    eval_inputs, _ = make_batch(rng_r, 100, SEQ_LEN, DELAY, VOCAB_SIZE)
    model_r.forward(eval_inputs)
    _, _, _, gate_weights_r, keep_mask_r, _, _, expert_outs_r, reviewer_cache_r = model_r.blocks[0].ffn._cache

    from moe import ReviewerLayer
    reviewer_weights, _ = ReviewerLayer().combine(gate_weights_r, expert_outs_r, keep_mask_r)
    diff = np.abs(reviewer_weights - gate_weights_r * keep_mask_r[..., None])
    mean_diff = diff[gate_weights_r > 0].mean()
    print(f"  mean |reviewer_weight - gate_weight| over selected experts: {mean_diff:.4f}")
    ok2 = mean_diff > 1e-3
    print(f"[{'PASS' if ok2 else 'FAIL'}] Reviewer combination differs meaningfully from raw gate weighting "
          f"(not a no-op)")
    all_pass &= ok2

    acc_r = eval_accuracy(model_r, np.random.RandomState(500))
    print(f"  held-out accuracy with reviewer active: {acc_r * 100:.1f}%")
    ok3 = acc_r > 0.95
    print(f"[{'PASS' if ok3 else 'FAIL'}] model with ReviewerLayer active still learns the task correctly")
    all_pass &= ok3

    print("\n" + ("ALL JUDGELAYER/REVIEWER TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    assert all_pass
