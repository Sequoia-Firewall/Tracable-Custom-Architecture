class HandlerNode:
    """
    - handler: 
    -- using segment relevance calculated above, will take predictions from segments and will calculate a final prediction
    -- purely mathematical
    """
    def __init__(self, logging_enabled=False, logger = None):
        self.logging_enabled = logging_enabled
        self.reviewer_reports = []
        self.logger = logger
    
    def display(self, message, Loud):
        if self.logging_enabled:
            if self.logger is None:
                raise ValueError("Logger is not set for HandlerNode.")
            message = (f"[HandlerNode] {message}")
            self.logger.log(message, Loud)

    def receive_report(self, report, Loud):
        self.display(f"Received report: {report}", Loud= Loud)
        self.reviewer_reports.append(report)

    def process(self, Loud):
        self.display(f"Processing reports...{self.reviewer_reports}", Loud= Loud)
        if not self.reviewer_reports:
            self.display("No reports to process.", Loud)
            return None
        total = 0.0
        for report in self.reviewer_reports:
            self.display(f"Processing report: {report}", Loud = Loud)
            #report structure is: [prediction, segment_relevance_score]
            total += report[0]

        total = total / len(self.reviewer_reports)
        self.display(f"Initial average prediction: {total}", Loud= Loud)

        total_relevance = sum(report[1] for report in self.reviewer_reports)
        if total_relevance == 0:
            return total

        for prediction, relevance in self.reviewer_reports:
            weight = relevance / total_relevance
            total += weight * (prediction - total)
        
        self.display(f"Final prediction after applying relevance scores: {total}", Loud= Loud)
        return total
    



        