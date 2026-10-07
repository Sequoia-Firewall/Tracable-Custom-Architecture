"""
ConfidenceEstimator -- same shape as TCA's Components/ConfidenceEstimator.py
(predicted_error = softplus(dot(w, features) + bias), confidence =
1/(1+predicted_error), trained by gradient descent against real observed
ABSOLUTE error), generalized from TCA's fixed 2 features to an arbitrary
feature vector -- needed here to run a clean ablation: train one estimator
on [raw_softmax_surprise] alone, another on
[raw_softmax_surprise, judge_novelty_distance], and compare held-out error
on real data rather than assuming the novelty feature helps.
"""
import numpy as np


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def softplus_grad(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class ConfidenceEstimator:
    GRAD_CLIP = 5.0
    WEIGHT_CLIP = 20.0

    def __init__(self, n_features, rng=None):
        rng = rng or np.random.RandomState(0)
        self.w = rng.normal(0, 0.1, size=n_features)
        self.b = 0.0

    def predict_error(self, features):
        features = np.asarray(features, dtype=np.float64)
        z = features @ self.w + self.b
        return softplus(z)

    def confidence(self, features):
        return 1.0 / (1.0 + self.predict_error(features))

    def train_step(self, features, actual_error, lr):
        """
        features: (batch, n_features); actual_error: (batch,) real observed
        error (e.g. per-example loss). Trains against ABSOLUTE error target
        directly (matching TCA's own fix for the squared-error-saturation
        bug found earlier this project), one plain gradient step, no batching
        machinery beyond a mean over the batch.
        """
        features = np.asarray(features, dtype=np.float64)
        actual_error = np.asarray(actual_error, dtype=np.float64)
        z = features @ self.w + self.b
        pred_error = softplus(z)

        # Loss = mean(|pred_error - actual_error|); d|u|/du = sign(u)
        diff = pred_error - actual_error
        dloss_dpred = np.sign(diff) / len(diff)
        dz = dloss_dpred * softplus_grad(z)

        dw = features.T @ dz
        db = dz.sum()

        dw = np.clip(dw, -self.GRAD_CLIP, self.GRAD_CLIP)
        db = np.clip(db, -self.GRAD_CLIP, self.GRAD_CLIP)

        self.w = np.clip(self.w - lr * dw, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)
        self.b = np.clip(self.b - lr * db, -self.WEIGHT_CLIP, self.WEIGHT_CLIP)

        return np.mean(np.abs(diff))
