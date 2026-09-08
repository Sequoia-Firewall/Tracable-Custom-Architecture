import Logger

class HandlerNode:
    def __init__(self, logger=None, classification=4):
        self.reports = {
            'segment': [],
            'segment_relevance': [],
            'predictions': [],
            'confidence': []
        }
        self.last_breakdown = {}   # populated after each process_reports() call
        self.Logger = logger if logger else Logger.Logger('HandlerNode.log', 4)
        self.classification = classification

    def display(self, message, Loud=False):
        message = f"[HandlerNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        self.Logger.log(message, self.classification, Loud)
    def receive_report(self, segment_id, segment_relevance, prediction, confidence=1.0):
        self.reports['segment'].append(segment_id)
        self.reports['segment_relevance'].append(segment_relevance)
        self.reports['predictions'].append(prediction)
        self.reports['confidence'].append(confidence)

    def process_reports(self, loud: bool, aggregation_mode: str = "bma") -> dict | None:
        if not self.reports['predictions']:
            self.display("No predictions to process.", Loud=loud)
            return None

        # Group predictions, confidence, and relevance by segment_id
        segments: dict[int, dict] = {}
        for seg_id, relevance, prediction, confidence in zip(
            self.reports['segment'],
            self.reports['segment_relevance'],
            self.reports['predictions'],
            self.reports['confidence']
        ):
            if seg_id not in segments:
                segments[seg_id] = {'relevance': relevance, 'predictions': [], 'confidences': []}
            segments[seg_id]['predictions'].append(prediction)
            segments[seg_id]['confidences'].append(confidence)

        seg_ids: list[int] = []
        means: list[float] = []
        weights: list[float] = []

        for seg_id, data in segments.items():
            preds = data['predictions']
            confs = data['confidences']
            # Confidence-weighted mean across this segment's reviewers, instead
            # of a flat unweighted mean — a reviewer whose collected signals had
            # lower accumulated variance (smoother path) is trusted more.
            conf_total = sum(confs)
            if conf_total > 0:
                mean_s = sum(p * c for p, c in zip(preds, confs)) / conf_total
            else:
                mean_s = sum(preds) / len(preds)
            seg_ids.append(seg_id)
            means.append(mean_s)

            if aggregation_mode == "bma":
                # Bayesian Model Averaging: relevance / inter-reviewer variance.
                # Raw weights are computed here and normalized below after the loop
                # so no single segment dominates due to near-zero variance.
                eps = 1e-9
                var_s = sum((p - mean_s) ** 2 for p in preds) / len(preds) if len(preds) > 1 else eps
                weight_s = data['relevance'] / max(var_s, eps)
                self.display(f"Segment {seg_id}: mean={mean_s:.4f} var={var_s:.6f} relevance={data['relevance']:.4f} weight={weight_s:.4f}", Loud=loud)
            elif aggregation_mode == "relevance_weighted":
                weight_s = data['relevance']
                self.display(f"Segment {seg_id}: mean={mean_s:.4f} relevance={data['relevance']:.4f}", Loud=loud)
            else:  # simple_mean
                weight_s = 1.0
                self.display(f"Segment {seg_id}: mean={mean_s:.4f}", Loud=loud)

            weights.append(weight_s)

        # Normalize BMA weights relative to their maximum so no single segment
        # dominates when its inter-reviewer variance happens to be near zero.
        if aggregation_mode == "bma" and weights:
            max_w = max(weights)
            if max_w > 0:
                weights = [w / max_w for w in weights]

        total_weight = sum(weights)
        if total_weight == 0.0:  # all relevances zero — fall back to equal weights
            weights = [1.0] * len(weights)
            total_weight = float(len(weights))
        norm_weights = [w / total_weight for w in weights]
        final_prediction = sum(m * w for m, w in zip(means, norm_weights))

        # Confidence: how much the contributing segments agree, not how
        # "correct" the answer is — inverse of the weighted variance of
        # segment means around the final prediction, same 1/(1+x) shape
        # already used for relevance elsewhere in this codebase. A single
        # contributing segment has nothing to disagree with, so it reads as
        # maximally confident (1.0) — that's a real limit of this measure,
        # not a bug: it reflects cross-segment agreement specifically, not
        # overall prediction quality.
        weighted_var = sum(w * (m - final_prediction) ** 2 for m, w in zip(means, norm_weights))
        confidence = 1.0 / (1.0 + weighted_var)

        # Dominant segment: whichever contributing segment ended up with the
        # highest final aggregation weight — the one that most shaped the
        # final prediction, not necessarily the one JudgeNode ranked highest
        # by relevance before any of this reviewer/variance math ran.
        dominant_idx = max(range(len(seg_ids)), key=lambda i: norm_weights[i]) if seg_ids else None
        dominant_segment_id = seg_ids[dominant_idx] if dominant_idx is not None else None

        self.display(
            f"[{aggregation_mode}] Final aggregated prediction: {final_prediction:.4f}  "
            f"confidence={confidence:.4f}  dominant_segment={dominant_segment_id}",
            Loud=loud
        )

        breakdown = {
            'score':               final_prediction,
            'confidence':          confidence,
            'dominant_segment_id': dominant_segment_id,
            'aggregation_mode':    aggregation_mode,
            'segments': {
                seg_id: {
                    'mean':        means[i],
                    'weight':      norm_weights[i],
                    'relevance':   segments[seg_id]['relevance'],
                    'n_reviewers': len(segments[seg_id]['predictions']),
                }
                for i, seg_id in enumerate(seg_ids)
            },
        }
        self.last_breakdown = breakdown

        self.reports = {
            'segment': [],
            'segment_relevance': [],
            'predictions': [],
            'confidence': []
        }
        return breakdown

        

        
        