"""
Real baseline test: char-level language modeling on tiny-shakespeare,
compared against PUBLISHED reference numbers for similarly-sized
from-scratch transformers on the SAME corpus (Karpathy's nanoGPT char
example) -- not a toy synthetic task, and not a claim of competing with
real LLM quality (pure NumPy/CPU is nowhere near that; see README).

Model sized to ~826K params (dim=128, 4 layers, 4 heads, ffn_hidden=512,
block_size=128), matching nanoGPT's own reported ~0.83M-parameter
reference point (val loss 1.7236 after 3000 iterations) as closely as
practical, specifically so the comparison means something -- not an
arbitrary small config.

Honest limits on how literal this comparison can be: different optimizer
defaults/batch size/exact architecture details (nanoGPT uses dropout,
weight-tied embeddings, possibly different init) than this from-scratch
implementation; single run, not averaged over seeds; CPU-only so the step
budget is chosen for tractability, not matched to nanoGPT's compute
budget. This is "does our from-scratch transformer get into the same
ballpark as a known-good reference at the same parameter count," not a
controlled scientific comparison.
"""
import time
import numpy as np
from tokenizer import CharTokenizer
from transformer import GPTStyleTransformer
from layers import cross_entropy_loss
from optim import Adam, lr_schedule

BLOCK_SIZE = 128
BATCH_SIZE = 32
DIM = 128
N_HEADS = 4
N_LAYERS = 4
FFN_HIDDEN = 512

MAX_STEPS = 3000
WARMUP_STEPS = 100
MAX_LR = 3e-3
MIN_LR = 3e-4
EVAL_EVERY = 200
EVAL_BATCHES = 20


def get_batch(data, batch_size, block_size, rng):
    ix = rng.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def estimate_loss(model, data, rng, n_batches):
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, rng)
        logits = model.forward(x)
        loss, _ = cross_entropy_loss(logits, y)
        losses.append(loss)
    return float(np.mean(losses))


if __name__ == "__main__":
    text = open("data/tinyshakespeare.txt").read()
    tok = CharTokenizer(text)
    ids = np.array(tok.encode(text), dtype=np.int64)

    n = len(ids)
    split = int(n * 0.9)
    train_data, val_data = ids[:split], ids[split:]
    print(f"vocab_size={tok.vocab_size}  train_chars={len(train_data)}  val_chars={len(val_data)}")

    model = GPTStyleTransformer(
        vocab_size=tok.vocab_size, dim=DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        ffn_hidden_dim=FFN_HIDDEN, max_seq_len=BLOCK_SIZE, rng_seed=1337,
    )
    total_params = sum(p.size for layer in model.all_sublayers() for p in layer.params().values() if p is not None)
    print(f"model params={total_params:,} (nanoGPT reference point: ~830,000 -> val loss 1.7236 @ 3000 iters)")

    adam = Adam(beta1=0.9, beta2=0.95)
    train_rng = np.random.RandomState(0)
    eval_rng = np.random.RandomState(1)

    t0 = time.time()
    for step in range(MAX_STEPS):
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, train_rng)
        lr = lr_schedule(step, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)

        model.zero_grad()
        loss, _ = model.loss_and_backward(x, y)
        adam.step(model, lr)

        if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            val_loss = estimate_loss(model, val_data, eval_rng, EVAL_BATCHES)
            elapsed = time.time() - t0
            bpc = val_loss / np.log(2)
            print(f"step {step:5d}  train_loss={loss:.4f}  val_loss={val_loss:.4f}  "
                  f"bits/char={bpc:.4f}  lr={lr:.5f}  elapsed={elapsed:.1f}s")

    final_val_loss = estimate_loss(model, val_data, np.random.RandomState(999), EVAL_BATCHES * 3)
    print(f"\nFinal val loss: {final_val_loss:.4f}  (bits/char: {final_val_loss / np.log(2):.4f})")
    print(f"Reference (nanoGPT, ~830K params, 3000 iters): val loss 1.7236")
