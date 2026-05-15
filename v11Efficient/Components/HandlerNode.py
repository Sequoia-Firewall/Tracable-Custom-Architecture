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

    def process_reports(self, loud: bool) -> float | None:
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

        # Bayesian Model Averaging:
        # weight_s = relevance_s / max(variance_s, eps), clamped to avoid
        # degenerate dominance when a segment outputs a constant (var → 0).
        eps = 1e-9
        max_weight = 1e6
        weights: list[float] = []
        means: list[float] = []

        for seg_id, data in segments.items():
            preds = data['predictions']
            mean_s = sum(preds) / len(preds)
            var_s = sum((p - mean_s) ** 2 for p in preds) / len(preds) if len(preds) > 1 else eps
            weight_s = min(data['relevance'] / max(var_s, eps), max_weight)
            means.append(mean_s)
            weights.append(weight_s)
            self.display(f"Segment {seg_id}: mean={mean_s:.4f} var={var_s:.6f} relevance={data['relevance']:.4f} weight={weight_s:.4f}", Loud=loud)

        total_weight = sum(weights)
        final_prediction = sum(m * w for m, w in zip(means, weights)) / total_weight

        self.display(f"Final aggregated prediction: {final_prediction:.4f}", Loud=loud)
        self.reports = {
            'segment': [],
            'segment_relevance': [],
            'predictions': []
        }
        return final_prediction

        

        
        