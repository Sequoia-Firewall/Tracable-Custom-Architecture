"""
The actual scientific question DCST organizes its whole proposal around,
run on our own mechanism: does a REAL data cluster produce a better
outcome than an arbitrary same-sized partition (the "shuffled-cluster
control"), and does DCST's "arbitrary assignment" prior-init help beyond
starting from zero?

Four configs, all continuing from the IDENTICAL warmed-up model (same
rng_seed + same deterministic warm-up steps, so no checkpoint machinery
needed -- replaying the same seed reproduces the same model bit-for-bit):
  (a) baseline       -- no judge bias at all
  (b) real_zero      -- real Judge, bias starts at zero, credit-updated every step
  (c) real_prior     -- real Judge, DCST-style strong prior-init, credit-updated
  (d) shuffled_zero  -- SAME real Judge's geometry, labels randomly permuted
                        (preserves cluster sizes -- DCST's actual recipe),
                        zero-init, credit-updated every step

(b)/(c) vs (a) asks "does the bias mechanism help at all." (b)/(c) vs (d)
asks the sharper question: does it help BECAUSE the clusters are real, or
would any same-sized partition do just as well?
"""
import numpy as np
from transformer import GPTStyleTransformer
from judge import DistributionJudge
from layers import softmax

VOCAB_SIZE = 6
DELAY = 2
SEQ_LEN = 12
BATCH_SIZE = 16
LR = 0.05
N_WARMUP = 200
N_CONTINUE = 200
MOE_CFG = {"n_experts": 4, "top_k": 1, "aux_loss_weight": 0.02, "n_shared_experts": 1}


def make_batch(rng, batch_size, seq_len, delay, vocab_size):
    tokens = np.zeros((batch_size, seq_len), dtype=np.int64)
    tokens[:, :delay] = rng.randint(0, vocab_size, size=(batch_size, delay))
    for i in range(delay, seq_len):
        tokens[:, i] = tokens[:, i - delay]
    return tokens[:, :-1], tokens[:, 1:]


def per_token_loss(logits, targets):
    probs = softmax(logits, axis=-1)
    b, s, v = probs.shape
    flat_probs = probs.reshape(-1, v)
    flat_targets = targets.reshape(-1)
    eps = 1e-12
    loss = -np.log(np.clip(flat_probs[np.arange(len(flat_targets)), flat_targets], eps, None))
    return loss.reshape(b, s)


def fresh_warmed_up_model(seed):
    """Deterministic: same seed -> same data sequence -> bit-identical model.
    Used once per config so each starts from the exact same point."""
    rng = np.random.RandomState(seed)
    model = GPTStyleTransformer(
        vocab_size=VOCAB_SIZE, dim=32, n_heads=4, n_layers=2, ffn_hidden_dim=64,
        max_seq_len=SEQ_LEN, rng_seed=seed, moe_cfg=MOE_CFG,
    )
    for _ in range(N_WARMUP):
        inputs, targets = make_batch(rng, BATCH_SIZE, SEQ_LEN, DELAY, VOCAB_SIZE)
        model.zero_grad()
        model.loss_and_backward(inputs, targets)
        model.sgd_step(LR)
    return model, rng  # rng's state continues seamlessly into the continuation phase


def eval_loss(model, rng):
    test_inputs, test_targets = make_batch(rng, 300, SEQ_LEN, DELAY, VOCAB_SIZE)
    logits = model.forward(test_inputs)
    loss = per_token_loss(logits, test_targets).mean()
    preds = logits.argmax(axis=-1)
    determined = (np.arange(test_inputs.shape[1]) + 1) >= DELAY
    acc = (preds[:, determined] == test_targets[:, determined]).mean()
    return loss, acc


def run_config(name, judge_mode, prior_strength, seed=7):
    """judge_mode: None (no judge), 'real', or 'shuffled'."""
    model, rng = fresh_warmed_up_model(seed)

    if judge_mode is not None:
        fit_inputs, _ = make_batch(rng, 300, SEQ_LEN, DELAY, VOCAB_SIZE)
        model.forward(fit_inputs)
        ffn_vectors, *_ = model.blocks[0].ffn._cache
        flat_vectors = ffn_vectors.reshape(-1, ffn_vectors.shape[-1])

        sample = flat_vectors[:200]
        pairwise = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
        median_dist = np.median(pairwise[np.triu_indices(len(sample), k=1)])
        judge = DistributionJudge(eps=median_dist * 0.4, min_samples=10)
        judge.fit(flat_vectors)

        if judge_mode == "shuffled":
            shuffle_rng = np.random.RandomState(seed + 1000)
            judge.labels_ = shuffle_rng.permutation(judge.labels_)  # same sizes, random assignment

        model.blocks[0].ffn.prior_strength = prior_strength
        model.blocks[0].judge_layer.attach(judge)

    for _ in range(N_CONTINUE):
        inputs, targets = make_batch(rng, BATCH_SIZE, SEQ_LEN, DELAY, VOCAB_SIZE)
        model.zero_grad()
        model.loss_and_backward(inputs, targets)
        model.sgd_step(LR)

        if judge_mode is not None:
            fresh_logits = model.forward(inputs)  # post-sgd_step logits for this batch's credit signal
            token_losses = per_token_loss(fresh_logits, targets)
            x_batch, _, mask, *_ = model.blocks[0].ffn._cache
            chosen = mask.argmax(axis=-1)
            cluster_ids = model.blocks[0].judge_layer.forward(x_batch)
            model.blocks[0].ffn.update_judge_bias_credit(
                cluster_ids.flatten(), chosen.flatten(), token_losses.flatten(), lr=0.1,
            )

    eval_rng = np.random.RandomState(999)
    loss, acc = eval_loss(model, eval_rng)
    print(f"[{name:15s}] held-out loss={loss:.4f}  accuracy={acc * 100:.1f}%")
    return loss, acc


if __name__ == "__main__":
    print(f"Warmup={N_WARMUP} steps, continuation={N_CONTINUE} steps, same seed/data for all 4 configs\n")

    results = {}
    results["baseline"] = run_config("baseline", judge_mode=None, prior_strength=0.0)
    results["real_zero"] = run_config("real_zero", judge_mode="real", prior_strength=0.0)
    results["real_prior"] = run_config("real_prior", judge_mode="real", prior_strength=0.5)
    results["shuffled_zero"] = run_config("shuffled_zero", judge_mode="shuffled", prior_strength=0.0)

    print("\n=== Summary ===")
    base_loss = results["baseline"][0]
    for name, (loss, acc) in results.items():
        delta = (base_loss - loss) / base_loss * 100
        print(f"  {name:15s} loss={loss:.4f} (vs baseline: {delta:+.1f}%)  accuracy={acc*100:.1f}%")

    print("\n=== Interpretation ===")
    real_best = min(results["real_zero"][0], results["real_prior"][0])
    if real_best < results["baseline"][0] and real_best < results["shuffled_zero"][0]:
        print("Real-cluster bias beats BOTH the no-bias baseline AND the shuffled control: "
              "evidence the data structure itself is doing real work here.")
    elif real_best < results["baseline"][0]:
        print("Real-cluster bias beats the baseline, but NOT clearly better than the shuffled "
              "control: gains may come from generic structured perturbation, not the specific "
              "cluster structure (DCST's own anticipated null result).")
    else:
        print("Real-cluster bias does not beat the no-bias baseline in this run.")

    if results["real_prior"][0] < results["real_zero"][0]:
        print("DCST-style strong prior-init beat zero-init.")
    else:
        print("Zero-init (no prior) did at least as well as the strong prior-init in this run.")
