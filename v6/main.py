import Nodes.BaseNode as BaseNode
import Nodes.JudgeNode as JudgeNode
import Nodes.Handler as HandlerNode
import Nodes.Splitter as SplitterNode
import Nodes.Reviewer as ReviewerNode
import Nodes.Processing as ProcessingNode


class main:
    def __init__(self):
        self.dataset = None
        self.segments = []
        self.segments_count = 0
        self.judge_node = None
        self.handler_node = None
        self.processing_node_percentage = 0.9  # 90% of standard segment position

    def load_dataset(self, dataset):
        self.dataset = dataset
        self.dataset_features = dataset.columns.tolist()
        
    def initialize_base_framework(self, dimensions=2, max_x=10):
        # Initialize the preprocess node
        self.preprocess_node = BaseNode.BaseNode(position=(0,) * dimensions)

        # Initialize Judge Node at origin
        self.judge_node = JudgeNode.JudgeNode(position=(0,) * dimensions)
        self.judge_node.set_dataset_features(self.dataset_features)
        
        # Initialize Handler Node at origin
        self.handler_node = HandlerNode.HandlerNode(position=(0,) * dimensions)

        #create default segments
        self.create_default_segments(dimensions, max_x)

        for segment in self.segments:
            # Connect Judge Node to Splitter Node
            self.judge_node.set_splitters(self.judge_node.splitters + [segment['splitter']])
            
            # Connect Handler Node to Reviewer Node
            self.handler_node.set_reviewers(self.handler_node.reviewers + [segment['reviewer_node']])

    def create_default_segments(self, dimensions, max_x):
        self.segments = []
        self.segments_count = 2 ** dimensions
        for i in range(self.segments_count):
            # Calculate position based on binary representation of segment index
            # Each bit determines positive (0) or negative (1) for that dimension
            splitter_pos = []
            reviewer_pos = []
            for dim in range(dimensions):
                # Check if bit at position 'dim' is set
                if (i >> dim) & 1:
                    splitter_pos.append(-1)
                    reviewer_pos.append(-max_x)
                else:
                    splitter_pos.append(1)
                    reviewer_pos.append(max_x)
            
            segment = {
                'index': i,
                'splitter': SplitterNode.SplitterNode(position=tuple(splitter_pos)),
                'processing_nodes': [],
                'reviewer_node': ReviewerNode.ReviewerNode(position=tuple(reviewer_pos))
            }
            self.segments.append(segment)
    
    def set_custom_segments(self, segments):
        self.segments = segments
        self.segments_count = len(segments)