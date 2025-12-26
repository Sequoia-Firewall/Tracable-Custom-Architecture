"""
from notes:
- judge : puts weights on different categories potentially based on category variables : distributes based on top weights
  -- note useful categories: age, gender, study hours, attendance, internet access, sleep hours, sleep quality, study method, facility rating, exam difficulty
  ---exam score is goal
  -- determines relevance of different features to find splitter priority (lowest 25% splitters are dropped for current input)
  -- will connect to splitter nodes to begin NN processing
feature priority determiner
routing determination
"""

from .BaseNode import BaseNode

class JudgeNode(BaseNode):
    def __init__(self, position):
        self.position = position  # Position in the nexus (likely origin (0,0))
        self.splitters = []  # List of connected splitter nodes
        self.dataset_features = []  # List of features in the dataset for purpose of specific use case
        
    def process(self, data):
        # Implement the processing logic for the JudgeNode
        pass

    def set_splitters(self, splitters):
        self.splitters = splitters

    def set_dataset_features(self, features):
        self.dataset_features = features

    def determine_feature_priority(self, data):
        # Implement the feature priority determination logic
        pass
    
    def determine_segment_priority(self, feature_priority):
        # Implement the segment priority determination logic
        pass

    def determine_splitters(self, feature_priority):
        # Implement the logic to determine which splitters to use based on feature priority
        pass

    def load_dataset_features(self, features):
        self.dataset_features = features