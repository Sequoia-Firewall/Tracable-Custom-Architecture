class HandlerNode:
    """
    - handler: 
    -- using segment relevance calculated above, will take predictions from segments and will calculate a final prediction
    -- purely mathematical
    """
    def __init__(self, logging_enabled=False):
        self.logging_enabled = logging_enabled
        self.reviewer_reports = []
    
    def display(self, message):
        if self.logging_enabled:
            print(f"[HandlerNode] {message}")

    def receive_report(self, report):
        self.display(f"Received report: {report}")
        self.reviewer_reports.append(report)

    def process(self):
        self.display(f"Processing reports...{self.reviewer_reports}")
        if not self.reviewer_reports:
            self.display("No reports to process.")
            return None
        total = 0.0
        for report in self.reviewer_reports:
            self.display(f"Processing report: {report}")
            #report structure is: [prediction, segment_relevance_score]
            total += report[0]

        total = total / len(self.reviewer_reports)
        self.display(f"Initial average prediction: {total}")

        total_relevance = sum(report[1] for report in self.reviewer_reports)
        if total_relevance == 0:
            return total

        for prediction, relevance in self.reviewer_reports:
            weight = relevance / total_relevance
            total += weight * (prediction - total)
        
        self.display(f"Final prediction after applying relevance scores: {total}")
        return total
    



        