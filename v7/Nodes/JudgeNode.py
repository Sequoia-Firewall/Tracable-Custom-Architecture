"""
- Judge Node:
 -- will compute 1 different types of predictions. 
 -- will predict segment relevance: 
 --- using values of input data, will use an algorithm to compute the relevance of different segments
 --- will activate the top 50% of relevant segments
 --- segment relevance scores to be used later in handler to compute final prediction
 -- will route data to the relevant segments and their splitter for further processing
 -- integrated ML for segment relevance calculations
"""

class JudgeNode:
    def __init__(self, logging_enabled=False):
        self.logging_enabled = logging_enabled
        self.segments = []  # connected segments (SplitterNodes)
        self.segment_weights = []  # segment_id → relevance score
        self.features = []  # list of feature names

    def display(self, message):
        if self.logging_enabled:
            print(f"[JudgeNode] {message}")
    
    def calculate_segment_relevance(self, input_data):
        self.display("Calculating segment relevance.")

        segment_relevance = {}
        for segment in self.segments:
            seg_id = segment['id']

            # Placeholder relevance (replace with ML later)
            segment_relevance[seg_id] = 1.0

        self.segment_weights = segment_relevance
        self.display(f"Segment relevance: {segment_relevance}")
        return segment_relevance
    
    def select_segments(self, segment_relevance):
        self.display("Selecting top 50% relevant segments.")
        if not segment_relevance:
            self.display("No segment weights calculated; cannot select segments.")
            return []
        
        sorted_segments = sorted(segment_relevance.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, len(sorted_segments) // 2)
        selected_segments = [self.segments[idx] for idx, _ in sorted_segments[:top_count]]
        
        self.display(f"Selected segments: {[segment['splitter'].position for segment in selected_segments]}")
        return selected_segments
    
    
    
    
