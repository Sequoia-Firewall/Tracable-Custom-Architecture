import sys
LOGGING_ENABLED = '--debug' in sys.argv
import pandas as pd
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table
from tqdm import tqdm
import os
import pickle

import Nodes.BaseNode as BaseNode
import Nodes.JudgeNode as JudgeNode
import Nodes.Handler as HandlerNode
import Nodes.Splitter as SplitterNode
import Nodes.Reviewer as ReviewerNode
import Nodes.Processing as ProcessingNode
import Nodes.PreProcessNode as PreProcessNode
from collections import deque

class main:
        
    def __init__(self):
        if LOGGING_ENABLED:
            print('[DEBUG] main class initialized')
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
        self.preprocess_node = PreProcessNode.PreProcessNode(position=(0,) * dimensions)
        # Ensure feature order and stats are frozen for preprocessing
        self.preprocess_node.feature_order = self.dataset_features.copy()
        # Optionally, compute and store dataset-level stats for normalization
        if self.dataset is not None:
            for f in self.dataset_features:
                if f not in self.dataset.columns:
                    continue

                col = pd.to_numeric(self.dataset[f], errors="coerce").dropna()
                if col.empty:
                    continue

                self.preprocess_node.means[f] = col.mean()
                self.preprocess_node.stds[f] = col.std() or 1.0
                self.preprocess_node.mins[f] = col.min()
                self.preprocess_node.maxs[f] = col.max()
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
            # Ensure processing nodes are only within their segment
            segment['splitter'].nodes_in_segment = segment['processing_nodes']
        for node in segment['processing_nodes']:
                node.neighbors = sorted(
                    segment['processing_nodes'],
                    key=lambda n: np.linalg.norm(np.array(n.position) - np.array(node.position))
                )[1:6]
            # Ensure each reviewer forwards to the handler
        for segment in self.segments:
            segment['reviewer_node'].handler = self.handler_node
    def create_default_segments(self, dimensions, max_x):
        import math
        self.segments = []
        self.segments_count = 2 ** dimensions
        for i in range(self.segments_count):
            splitter_pos = []
            reviewer_pos = []
            for dim in range(dimensions):
                if (i >> dim) & 1:
                    splitter_pos.append(-1)
                    reviewer_pos.append(-max_x)
                else:
                    splitter_pos.append(1)
                    reviewer_pos.append(max_x)

            # Calculate grid size for a square/cube grid
            grid_size = int(round(max_x * self.processing_node_percentage))
            if grid_size < 2:
                grid_size = 2
            total_slots = grid_size ** dimensions
            usable_slots = max(1, total_slots - 2)
            num_nodes = max(2, int(math.floor(usable_slots * 0.9)))
            node_positions = self.generate_evenly_spaced_positions(dimensions, max_x, coverage=0.9, num_nodes=num_nodes)
            reviewer_pos_tuple = tuple(reviewer_pos)
            processing_nodes = [ProcessingNode.ProcessingNode(position=pos, reviewer_position=reviewer_pos_tuple) for pos in node_positions]
            splitter = SplitterNode.SplitterNode(position=tuple(splitter_pos), nodes_in_segment=processing_nodes, dimenstions=dimensions, segment_id=i)
            reviewer = ReviewerNode.ReviewerNode(position=tuple(reviewer_pos), splitter=splitter)
            segment = {
                'index': i,
                'splitter': splitter,
                'processing_nodes': processing_nodes,
                'reviewer_node': reviewer
            }
            
            self.segments.append(segment)

    def generate_evenly_spaced_positions(self, dimensions, max_x, coverage=0.9, num_nodes=10):
        # Generate positions in a grid covering 90% of the space, unique and evenly spaced, excluding the origin
        from itertools import product
        grid_size = int(round(num_nodes ** (1/dimensions)))
        if grid_size < 2:
            grid_size = 2
        step = (max_x * coverage * 2) / (grid_size - 1) if grid_size > 1 else 0
        start = -max_x * coverage
        grid_axes = [ [start + i * step for i in range(grid_size)] for _ in range(dimensions) ]
        positions = list(product(*grid_axes))
        # Remove the origin (0, 0, ...) if present
        positions = [pos for pos in positions if not all(abs(coord) < 1e-8 for coord in pos)]
        # Limit to num_nodes if needed
        return positions[:num_nodes]
    
    def set_custom_segments(self, segments):
        self.segments = segments
        self.segments_count = len(segments)

    def train(self, X, y):
        """
        Training pipeline:
        1. Preprocess data
        2. Fit JudgeNode (global + per-segment Bayesian models)
        3. Propagate training signals through splitters → processing → reviewers
        4. Feed observed error back to Judge for variance + relevance updates
        5. Move processing nodes if it boosts accuracy

        X: iterable of input samples (dict-like)
        y: iterable of target values
        """

        if self.judge_node is None:
            raise ValueError(
                "Framework not initialized. Call initialize_base_framework() first."
            )

        # ---------- 1. Preprocess ----------
        X_proc = [self.preprocess_node.process(x) for x in X]

        # ---------- 2. Fit Judge ----------
        self.judge_node.fit(
            X_proc,
            y,
            feature_names=self.dataset_features
        )

        # ---------- 3. Signal-based training ----------

        console = Console()
        for idx, (x_i, y_i) in enumerate(tqdm(zip(X_proc, y), total=len(y), desc="Training")):
            judge_output = self.judge_node.process(x_i)
            priority_features = judge_output["priority_features"]
            splitter_scores = judge_output["splitter_scores"]
            splitter_scores.sort(key=lambda x: x[1], reverse=True)
            n_drop = int(len(splitter_scores) * 0.25)
            active_splitters = [s for s, _ in splitter_scores[:-n_drop]] \
                if n_drop > 0 else [s for s, _ in splitter_scores]
            routed = self.judge_node.route_to_splitters(
                x_i,
                priority_features,
                active_splitters=active_splitters
            )

            for splitter, carrier in routed:
                segment = self.segments[splitter.segment_id]
                reviewer = segment['reviewer_node']

                if not splitter.closest_nodes:
                    splitter.compute_closest_nodes(
                        len(segment['processing_nodes']),
                        segment['processing_nodes']
                    )

                signals = splitter.process(carrier)
                reviewer.amount_live_signals = len(signals)

                for node, signal in zip(splitter.closest_nodes, signals):
                    node.receive_signal(signal)
                    node.forward_signal(reviewer)

            # ---------- 4. Feedback ----------
            for segment in self.segments:
                reviewer = segment['reviewer_node']
                if not reviewer.signals:
                    continue
                reviewed_signal = reviewer.signals[0]
                pred = getattr(reviewed_signal, "prediction", None)
                var = getattr(reviewed_signal, "accumulated_variance", None)
                if pred is None:
                    continue
                error = y_i - pred
                self.judge_node.update_feature_relevance_from_feedback(
                    {
                    "input": x_i,
                    "target": y_i,
                    "predicted": pred,
                    "error": error
                }
                )
                self.judge_node.update_segment_variance_from_feedback(
                    {
                        "input": x_i,
                        "segment_errors": {
                            reviewed_signal.segment_id: error**2
                        }
                    }
                )
            # ---------- 5. Move processing nodes if it boosts accuracy ----------
            # Only move processing nodes if error is defined for this sample
            if 'error' in locals():
                for segment in self.segments:
                    for node in segment['processing_nodes']:
                        if abs(error) > 1.0:
                            old_position = tuple(node.position)
                            # Convert x_i to a list of values based on its type
                            if isinstance(x_i, dict):
                                x_values = list(x_i.values())
                            elif hasattr(x_i, '__iter__') and not isinstance(x_i, str):
                                x_values = list(x_i)
                            else:
                                x_values = [x_i]
                            
                            new_position = tuple(
                                np.array(node.position) + 0.1 * (np.array(x_values[:len(node.position)]) - np.array(node.position))
                            )
                            node.position = new_position
                            if LOGGING_ENABLED:
                                print(f"[DEBUG] ProcessingNode moved from {old_position} to {new_position} due to error {error}")


    def infer(self, x):
        if self.judge_node is None or self.handler_node is None:
            raise ValueError("Framework not initialized.")

        # Reset handler state
        self.handler_node.reviewer_reports.clear()
        self.handler_node.segment_weights.clear()

        # Preprocess
        x_proc = self.preprocess_node.process(x)

        # Judge
        relevance_scores = self.judge_node.calculate_feature_relevance(x_proc)
        ranked_features = self.judge_node.rank_features(relevance_scores)
        routed = self.judge_node.route_to_splitters(x_proc, ranked_features)


        console = Console()
        live_table = Table(title="Signal Live Monitor (Inference)")
        live_table.add_column("Segment")
        live_table.add_column("Live Signals")
        live_table.add_column("Dead Signals")
        live_table.add_column("To Reviewer")
        console.print(live_table)
        self.segment_queue_count = 0
        self.segment_demonstration = []
        for segment in self.segments:
            splitter = segment['splitter']
            processing_nodes = segment['processing_nodes']
            reviewer = segment['reviewer_node']
            segment_id = segment['index']

            reviewer.reset()

            if not splitter.closest_nodes:
                splitter.compute_closest_nodes(
                    all_node_count=len(processing_nodes),
                    nodes_in_segment=processing_nodes
                )

            segment_relevance = 1.0
            for splitter_obj, signal_data in routed:
                if splitter_obj == splitter:
                    segment_relevance = signal_data.get('segment_relevance', 1.0)

            self.handler_node.segment_weights[segment_id] = segment_relevance

            carrier_data = {
                'input_data': x_proc,
                'feature_relevance': ranked_features,
                'segment_feature_relevance': {},
                'segment_relevance': segment_relevance
            }

            signals = splitter.process(carrier_data)
            reviewer.expected_signals = len(splitter.closest_nodes)
            queue = deque()
            added_nodes_count = 0
            for node, signal in zip(splitter.closest_nodes, signals):
                node.receive_signal(signal)
                queue.append(node)
                added_nodes_count += 1
            print(f"[DEBUG] Segment {segment_id}: Added {added_nodes_count} processing nodes to the queue.")
            self.segment_demonstration.append([segment_id, added_nodes_count])
            if len(queue) != 0:
                print(f"[DEBUG] Segment {segment_id}: Initial queue length is {len(queue)}.")
                self.segment_queue_count += 1
            while queue:
                node = queue.popleft()
                if node.signal is not None:
                    print(f"Input data: {node.signal.input_data}, Feature relevance: {node.signal.feature_relevance}, Segment relevance: {segment_relevance}, Prediction: {getattr(node.signal, 'prediction', None)}")
                
                node.process()


                next_node, next_signal = node.forward_signal()

                if next_signal is None:
                    continue

                if next_node is None or next_node is reviewer:
                    reviewer.receive_signal(next_signal)
                    print(f"[DEBUG] Segment {segment_id}: Signal forwarded to reviewer from node at position {node.position}. with signal prediction {getattr(next_signal, 'prediction', None)}")
                else:
                    next_node.receive_signal(next_signal)
                    queue.append(next_node)

            reviewer.process()
        self.handler_node.receive_reports()
        return self.handler_node.process()
    
    def save_state(self, filename="Demo.nexus"):
        state = {
            'segments': [
                {
                    'index': seg['index'],
                    'splitter_position': seg['splitter'].position,
                    'processing_nodes': [
                        {
                            'position': node.position,
                            'weights': getattr(node, 'weights', {})
                        } for node in seg['processing_nodes']
                    ],
                    'reviewer_position': seg['reviewer_node'].position
                } for seg in self.segments
            ],
            'judge_node': {
                'position': getattr(self.judge_node, 'position', None)
            },
            'handler_node': {
                'position': getattr(self.handler_node, 'position', None)
            },
            'dataset_features': getattr(self, 'dataset_features', [])
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)
        print(f"Nexus state saved to {filename}")

    def load_state(self, filename="Demo.nexus"):
        if not os.path.exists(filename):
            print(f"State file {filename} not found.")
            return False
        with open(filename, "rb") as f:
            state = pickle.load(f)
        # Restore dataset features
        self.dataset_features = state.get('dataset_features', [])
        # Restore judge and handler positions
        if self.judge_node:
            self.judge_node.position = state['judge_node'].get('position', self.judge_node.position)
        if self.handler_node:
            self.handler_node.position = state['handler_node'].get('position', self.handler_node.position)
        # Restore segments
        for seg, seg_state in zip(self.segments, state['segments']):
            seg['splitter'].position = seg_state['splitter_position']
            seg['reviewer_node'].position = seg_state['reviewer_position']
            for node, node_state in zip(seg['processing_nodes'], seg_state['processing_nodes']):
                node.position = node_state['position']
                node.weights = node_state['weights']
        print(f"Nexus state loaded from {filename}")
        return True

    def run_demo(self):
            # Load dataset
            df = pd.read_csv("v6/Exam_Score_Prediction.csv")
            dropped_courses = []
            for column in df.select_dtypes(include=['object']).columns:
                unique_counts = df[column].nunique()
                print(f"Column: {column}, Unique Values: {unique_counts}")
                if unique_counts > 5:
                    # If there are too many unique values, we can choose to drop them
                    df = df.drop(columns=[column])
                    dropped_courses.append(column)
            print (f"Dropped columns: {dropped_courses}")

            for row in df['gender'].index:
                if df.at[row, 'gender'] == 'male':
                    df.at[row, 'gender'] = 0
                elif df.at[row, 'gender'] == 'female':
                    df.at[row, 'gender'] = 1
                else:
                    df.at[row, 'gender'] = 2
            for row in df['internet_access'].index:
                if df.at[row, 'internet_access'] == 'no':
                    df.at[row, 'internet_access'] = 0
                elif df.at[row, 'internet_access'] == 'yes':
                    df.at[row, 'internet_access'] = 1
                else:
                    df.at[row, 'internet_access'] = 2
            for row in df['sleep_quality'].index:
                if df.at[row, 'sleep_quality'] == 'poor':
                    df.at[row, 'sleep_quality'] = 0
                elif df.at[row, 'sleep_quality'] == 'average':
                    df.at[row, 'sleep_quality'] = 1
                elif df.at[row, 'sleep_quality'] == 'good':
                    df.at[row, 'sleep_quality'] = 2
                else:
                    df.at[row, 'sleep_quality'] = 3
            for row in df['study_method'].index:
                if df.at[row, 'study_method'] == 'coaching':
                    df.at[row, 'study_method'] = 0
                elif df.at[row, 'study_method'] == 'online videos':
                    df.at[row, 'study_method'] = 1
                elif df.at[row, 'study_method'] == 'mixed':
                    df.at[row, 'study_method'] = 2
                elif df.at[row, 'study_method'] == 'self-study':
                    df.at[row, 'study_method'] = 3
                elif df.at[row, 'study_method'] == 'group study':
                    df.at[row, 'study_method'] = 4
            for row in df['facility_rating'].index:
                if df.at[row, 'facility_rating'] == 'low':
                    df.at[row, 'facility_rating'] = 0
                elif df.at[row, 'facility_rating'] == 'medium':
                    df.at[row, 'facility_rating'] = 1
                elif df.at[row, 'facility_rating'] == 'high':
                    df.at[row, 'facility_rating'] = 2
                else:
                    df.at[row, 'facility_rating'] = 3
            for row in df['exam_difficulty'].index:
                if df.at[row, 'exam_difficulty'] == 'easy':
                    df.at[row, 'exam_difficulty'] = 0
                elif df.at[row, 'exam_difficulty'] == 'moderate':
                    df.at[row, 'exam_difficulty'] = 1
                elif df.at[row, 'exam_difficulty'] == 'hard':
                    df.at[row, 'exam_difficulty'] = 2
                else:
                    df.at[row, 'exam_difficulty'] = 3
            self.dataset_features = df.columns.drop('exam_score').tolist()

            X = df.drop('exam_score', axis=1)
            y = df['exam_score']
            print("Dataset loaded. starting main nexus")
            # Initialize main nexus
            nexus = main()
            nexus.load_dataset(X)
            # Use 2D, max_x=50 for a small test nexus
            nexus.initialize_base_framework(dimensions=2, max_x=50)
            
            # Check for Demo.nexus file
            if os.path.exists("Demo.nexus"):
                print("Demo.nexus file found. Would you like to load it and skip training? (y/n): ", end="")
                choice = input().strip().lower()
                if choice == "y":
                    nexus.load_state("Demo.nexus")
                    # Optionally skip training
                    print("Loaded saved nexus state. Skipping training.")
                    node_count = 0
                    total_signals_to_reviewers = 0
                    for segment in nexus.segments:
                        print(f"Segment {segment['index']}: {len(segment['processing_nodes'])} processing nodes")
                        for node in segment['processing_nodes']:
                            if node.weights:
                                node_count += 1
                    print(f"Total activated processing nodes: {node_count}")
                else:
                    # Train
                    nexus.train(X.to_dict(orient='records'), y.values)
                    nexus.save_state("Demo.nexus")
                    
            else:
                # Train
                nexus.train(X.to_dict(orient='records'), y.values)
                 # After training, save state
                nexus.save_state("Demo.nexus")
            # Test inference on a random row
            test_x = X.iloc[0].to_dict()
            result = nexus.infer(test_x)
            # Minimal addition: count signals created by all splitters for this inference
            total_signals_created = 0
            for segment in nexus.segments:
                splitter = segment['splitter']
                # If splitter has a signals attribute or similar, count them; otherwise, count closest_nodes
                if hasattr(splitter, 'closest_nodes'):
                    total_signals_created += len(splitter.closest_nodes)
            
            """
            for segment in nexus.segments:
                for node in segment['processing_nodes']:
                    if node.weights:
                        print(f"Node at position {node.position} has weights: {node.weights}")
            """
            print("Amount of live segments with queued nodes:", nexus.segment_queue_count)
            print("Test input:", test_x)
            print("segment node amounts for demo:", nexus.segment_demonstration)
            print("Predicted score:", result)
            print("Actual score:", y.iloc[0])
            node_count = 0
            total_signals_to_reviewers = 0
            for segment in nexus.segments:
                print(f"Segment {segment['index']}: {len(segment['processing_nodes'])} processing nodes")
                for node in segment['processing_nodes']:
                    if node.weights:
                        node_count += 1
                # Add up signals that reached the reviewer for this segment
                total_signals_to_reviewers += segment['reviewer_node'].signal_arrival_count
            print(f"Total activated processing nodes: {node_count}")
            print(f"Total signals created during inference: {total_signals_created}")
            print(f"Total signals that reached reviewers: {total_signals_to_reviewers}")

           

if __name__ == "__main__":
    nexus = main()
    demo = input("Run demo? (y/n): ")
    if demo.lower() == 'y':
        print("Running demo...")
        nexus.run_demo()
    else: 
        print("Demo skipped.")
        print("main loop still in progress")