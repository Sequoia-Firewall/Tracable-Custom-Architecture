"""
Smoke test: does this transformer actually learn, not just pass gradient
checks? Task: a tiny synthetic "copy after delay" sequence task over a
6-token vocabulary -- predict token[i] = token[i - k] for a fixed delay k.
This requires real attention (looking back k positions), not just a
memorized per-position bias, so it's a reasonable minimal test of whether
attention is wired correctly end-to-end.
"""
import numpy as np
from transformer import GPTStyleTransformer
from trace import TraceRecorder

VOCAB_SIZE = 6
DELAY = 3
SEQ_LEN = 12
BATCH_SIZE = 16
N_STEPS = 400
LR = 0.05


def make_batch(rng, batch_size, seq_len, delay, vocab_size):
    # Random tokens for the first `delay` positions, then token[i]=token[i-delay].
    tokens = np.zeros((batch_size, seq_len), dtype=np.int64)
    tokens[:, :delay] = rng.randint(0, vocab_size, size=(batch_size, delay))
    for i in range(delay, seq_len):
        tokens[:, i] = tokens[:, i - delay]
    # Inputs are tokens[:, :-1], targets are tokens[:, 1:] (standard next-token
    # framing) -- the model must learn "copy from `delay` steps back" to
    # predict well on the tail of the sequence.
    return tokens[:, :-1], tokens[:, 1:]


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    model = GPTStyleTransformer(
        vocab_size=VOCAB_SIZE, dim=32, n_heads=4, n_layers=2, ffn_hidden_dim=64,
        max_seq_len=SEQ_LEN, rng_seed=1,
    )

    print(f"Task: predict token[i] = token[i-{DELAY}] over vocab={VOCAB_SIZE}, seq_len={SEQ_LEN}")
    print(f"Random-guess baseline loss ~= ln({VOCAB_SIZE}) = {np.log(VOCAB_SIZE):.4f}\n")

    losses = []
    for step in range(N_STEPS):
        inputs, targets = make_batch(rng, BATCH_SIZE, SEQ_LEN, DELAY, VOCAB_SIZE)
        model.zero_grad()
        loss, _ = model.loss_and_backward(inputs, targets)
        model.sgd_step(LR)
        losses.append(loss)
        if step % 50 == 0 or step == N_STEPS - 1:
            recent = np.mean(losses[-50:])
            print(f"step {step:4d}  loss={loss:.4f}  recent_avg={recent:.4f}")

    # ---- accuracy on a fresh held-out batch, positions where the copy
    # target is actually determined (i >= delay) ----
    test_inputs, test_targets = make_batch(rng, 200, SEQ_LEN, DELAY, VOCAB_SIZE)
    trace = TraceRecorder()
    logits = model.forward(test_inputs, trace=trace)
    preds = logits.argmax(axis=-1)
    # input index i corresponds to predicting original token[i+1]; the copy
    # rule is "determined" once i+1 >= delay.
    determined_mask = (np.arange(test_inputs.shape[1]) + 1) >= DELAY
    acc = (preds[:, determined_mask] == test_targets[:, determined_mask]).mean()
    print(f"\nHeld-out accuracy on determined positions: {acc * 100:.1f}%")

    print("\nAttention entropy summary (first test batch forward pass):")
    for layer_summary in trace.attention_summary():
        print(f"  layer {layer_summary['layer_index']}: "
              f"per-head mean entropy = {[f'{e:.3f}' for e in layer_summary['per_head_mean_entropy']]}")

    final_loss = np.mean(losses[-20:])
    assert final_loss < 0.3 * np.log(VOCAB_SIZE), (
        f"model did not learn the copy task: final loss {final_loss:.4f} "
        f"not meaningfully below random-guess baseline {np.log(VOCAB_SIZE):.4f}"
    )
    assert acc > 0.9, f"held-out accuracy too low: {acc:.3f}"
    print("\nTOY TRAINING TASK PASSED -- model learned the delayed-copy task")
