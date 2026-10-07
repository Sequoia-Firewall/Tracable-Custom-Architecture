"""
The actual question this whole thread is trying to answer: does the
DistributionJudge's novelty distance (last-token hidden state vs. the
training set) help predict how wrong the model will be, BEYOND what the
model's own softmax confidence already tells you? Not assumed -- tested,
via a clean ablation: train one ConfidenceEstimator on
[raw_softmax_surprise] alone, another on
[raw_softmax_surprise, judge_novelty_distance], compare held-out error.

Setup: train the usual delayed-copy model (token[i]=token[i-DELAY]) on
in-distribution sequences only. Build an EVALUATION set that mixes:
  - in-distribution sequences (same rule, same vocab)
  - out-of-distribution sequences (same vocab, but NO copy structure --
    uniformly random tokens) -- the model never saw anything like this,
    so this is exactly the regime the novelty signal is supposed to flag.
Both get a real, well-defined actual_error (the true next-token loss,
known because we generated the sequence ourselves) regardless of which
regime they came from.
"""
import numpy as np
from transformer import GPTStyleTransformer
from judge import DistributionJudge
from confidence import ConfidenceEstimator
from layers import softmax

VOCAB_SIZE = 6
DELAY = 2
SEQ_LEN = 10
TRAIN_STEPS = 300
BATCH_SIZE = 16
LR = 0.05


def make_copy_batch(rng, batch_size, seq_len, delay, vocab_size):
    tokens = np.zeros((batch_size, seq_len), dtype=np.int64)
    tokens[:, :delay] = rng.randint(0, vocab_size, size=(batch_size, delay))
    for i in range(delay, seq_len):
        tokens[:, i] = tokens[:, i - delay]
    return tokens[:, :-1], tokens[:, 1:]


def make_random_batch(rng, batch_size, seq_len, vocab_size):
    """No copy structure at all -- out-of-distribution relative to training."""
    tokens = rng.randint(0, vocab_size, size=(batch_size, seq_len))
    return tokens[:, :-1], tokens[:, 1:]


def per_example_last_position_loss_and_features(model, inputs, targets):
    """
    Returns, per example in the batch:
      actual_error: true next-token cross-entropy loss AT THE LAST POSITION
      surprise: -log(softmax max-prob) at the last position -- the raw
                confidence signal the model gives you for free
      hidden: last-token final hidden state (for the Judge to query)
    """
    logits, hidden = model.forward(inputs, return_hidden=True)
    last_logits = logits[:, -1, :]       # (batch, vocab)
    last_hidden = hidden[:, -1, :]        # (batch, dim)
    last_targets = targets[:, -1]         # (batch,)

    probs = softmax(last_logits, axis=-1)
    eps = 1e-12
    actual_error = -np.log(np.clip(probs[np.arange(len(probs)), last_targets], eps, None))
    surprise = -np.log(np.clip(probs.max(axis=-1), eps, None))
    return actual_error, surprise, last_hidden


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    model = GPTStyleTransformer(
        vocab_size=VOCAB_SIZE, dim=16, n_heads=2, n_layers=2, ffn_hidden_dim=32,
        max_seq_len=SEQ_LEN, rng_seed=2,
    )

    print(f"Training on delay={DELAY} copy task only ({TRAIN_STEPS} steps)...")
    for step in range(TRAIN_STEPS):
        inputs, targets = make_copy_batch(rng, BATCH_SIZE, SEQ_LEN, DELAY, VOCAB_SIZE)
        model.zero_grad()
        loss, _ = model.loss_and_backward(inputs, targets)
        model.sgd_step(LR)
        if step % 100 == 0 or step == TRAIN_STEPS - 1:
            print(f"  step {step:3d}  loss={loss:.4f}")

    # ---- fit the Judge on last-token hidden states across training data ----
    print("\nFitting DistributionJudge on training-distribution hidden states...")
    train_inputs, train_targets = make_copy_batch(rng, 300, SEQ_LEN, DELAY, VOCAB_SIZE)
    _, _, train_hidden = per_example_last_position_loss_and_features(model, train_inputs, train_targets)
    judge = DistributionJudge(eps=np.median(
        np.linalg.norm(train_hidden[:, None, :] - train_hidden[None, :, :], axis=-1)
        [np.triu_indices(len(train_hidden), k=1)]
    ) * 0.3, min_samples=5)
    judge.fit(train_hidden)
    pop = judge.cluster_population_summary()
    print(f"  cluster population: {pop}")

    # ---- build a MIXED in-distribution / OOD evaluation set ----
    rng_eval = np.random.RandomState(99)
    n_each = 150
    id_inputs, id_targets = make_copy_batch(rng_eval, n_each, SEQ_LEN, DELAY, VOCAB_SIZE)
    ood_inputs, ood_targets = make_random_batch(rng_eval, n_each, SEQ_LEN, VOCAB_SIZE)

    id_error, id_surprise, id_hidden = per_example_last_position_loss_and_features(model, id_inputs, id_targets)
    ood_error, ood_surprise, ood_hidden = per_example_last_position_loss_and_features(model, ood_inputs, ood_targets)

    all_error = np.concatenate([id_error, ood_error])
    all_surprise = np.concatenate([id_surprise, ood_surprise])
    all_hidden = np.concatenate([id_hidden, ood_hidden], axis=0)
    is_ood = np.concatenate([np.zeros(n_each, dtype=bool), np.ones(n_each, dtype=bool)])

    _, _, all_novelty = judge.query_batch(all_hidden)

    print(f"\nMean actual error: in-dist={id_error.mean():.4f}  ood={ood_error.mean():.4f}")
    print(f"Mean novelty distance: in-dist={all_novelty[~is_ood].mean():.4f}  "
          f"ood={all_novelty[is_ood].mean():.4f}")
    print(f"Mean raw surprise: in-dist={id_surprise.mean():.4f}  ood={ood_surprise.mean():.4f}")

    # ---- train/test split ----
    n = len(all_error)
    perm = rng_eval.permutation(n)
    split = int(n * 0.6)
    train_idx, test_idx = perm[:split], perm[split:]

    def run_ablation(name, feature_matrix):
        est = ConfidenceEstimator(n_features=feature_matrix.shape[1], rng=np.random.RandomState(1))
        train_feat, train_err = feature_matrix[train_idx], all_error[train_idx]
        test_feat, test_err = feature_matrix[test_idx], all_error[test_idx]
        for _ in range(500):
            est.train_step(train_feat, train_err, lr=0.05)
        pred_test = est.predict_error(test_feat)
        mae = np.mean(np.abs(pred_test - test_err))
        corr = np.corrcoef(pred_test, test_err)[0, 1]
        print(f"  [{name}] held-out MAE={mae:.4f}  corr(pred,actual)={corr:.4f}")
        return mae, corr

    print("\n--- Ablation: does novelty distance improve error prediction beyond raw softmax surprise? ---")
    baseline_features = all_surprise[:, None]
    augmented_features = np.stack([all_surprise, all_novelty], axis=1)

    baseline_mae, baseline_corr = run_ablation("surprise only (baseline)", baseline_features)
    augmented_mae, augmented_corr = run_ablation("surprise + novelty (judge-augmented)", augmented_features)

    improvement_mae = (baseline_mae - augmented_mae) / baseline_mae * 100
    print(f"\nResult: judge-augmented MAE is {improvement_mae:+.1f}% vs baseline "
          f"({'better' if improvement_mae > 0 else 'worse/no better'})")
    print(f"Result: judge-augmented corr {augmented_corr:.4f} vs baseline corr {baseline_corr:.4f}")
