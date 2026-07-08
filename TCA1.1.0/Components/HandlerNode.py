import Logger

class HandlerNode:
    def __init__(self, logger=None, classification=4):
        self.reports = {
            'segment': [],
            'segment_relevance': [],
            'predictions': []
        }
        self.last_breakdown = {}   # populated after each process_reports() call
        self.Logger = logger if logger else Logger.Logger('HandlerNode.log', 4)
        self.classification = classification

    def display(self, message, Loud=False):
        message = f"[HandlerNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        self.Logger.log(message, self.classification, Loud)
    def receive_report(self, segment_id, segment_relevance, prediction):
        self.reports['segment'].append(segment_id)
        self.reports['segment_relevance'].append(segment_relevance)
        self.reports['predictions'].append(prediction)

    def process_reports(self, loud: bool, aggregation_mode: str = "bma") -> float | None:
        if not self.reports['predictions']:
            self.display("No predictions to process.", Loud=loud)
            return None

        # Group predictions and relevance by segment_id
        segments: dict[int, dict] = {}
        for seg_id, relevance, prediction in zip(
            self.reports['segment'],
            self.reports['segment_relevance'],
            self.reports['predictions']
        ):
            if seg_id not in segments:
                segments[seg_id] = {'relevance': relevance, 'predictions': []}
            segments[seg_id]['predictions'].append(prediction)

        means: list[float] = []
        weights: list[float] = []

        for seg_id, data in segments.items():
            preds = data['predictions']
            mean_s = sum(preds) / len(preds)
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
        final_prediction = sum(m * w for m, w in zip(means, weights)) / total_weight

        self.display(f"[{aggregation_mode}] Final aggregated prediction: {final_prediction:.4f}", Loud=loud)
        self.reports = {
            'segment': [],
            'segment_relevance': [],
            'predictions': []
        }
        return final_prediction

        

        
        