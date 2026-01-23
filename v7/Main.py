"""
Main module for DragonChild v7 testing under testing group 3.
Dataset is exam score prediction
Will include training and testing phases
"""

from Nodes.PreProcessingNode import PreProcessingNode
from Nodes.ProcessingNode import ProcessingNode
from Nodes.HandlerNode import HandlerNode
from Nodes.Signal import Signal
from Nodes.JudgeNode import JudgeNode
from Nodes.SplitterNode import SplitterNode
from Nodes.ReviewerNode import ReviewerNode

import math

from tqdm import tqdm

class Main:
    def __init__(self, logging_enabled=False, dimensions=2, max_x=None, processing_node_percentage=0.9):
        self.logging_enabled = logging_enabled
        self.max_x = max_x
        self.df = None
        self.Judge = None
        self.Handler = None
        self.segments = []
        self.dimensions = dimensions
        self.processing_node_percentage = processing_node_percentage

    def display(self, message):
        if self.logging_enabled:
            print(f"[Main] {message}")

    def base_preprocessing(self, file_path):
        self.display(f"Starting base preprocessing on dataset: {file_path}")
        preprocessor = PreProcessingNode(logging_enabled=self.logging_enabled)
        processed_data = preprocessor.process(file_path)
        self.display("Base preprocessing complete.")
        return processed_data
    def generate_evenly_spaced_positions(
        self, dimensions, max_x, splitter_pos, coverage=0.9, num_nodes=10
    ):
        from itertools import product

        grid_size = int(round(num_nodes ** (1 / dimensions)))
        grid_size = max(2, grid_size)

        axes = []
        for d in range(dimensions):
            if splitter_pos[d] > 0:
                start = splitter_pos[d]
                end = max_x * coverage
            else:
                start = -max_x * coverage
                end = splitter_pos[d]

            step = (end - start) / (grid_size - 1)
            axes.append([start + i * step for i in range(grid_size)])

        positions = list(product(*axes))
        return positions[:num_nodes]
    def initialize_nexus(self):
        self.display("Initializing Nexus structure.")
        if self.max_x is None:
            raise ValueError("max_x must be defined to initialize Nexus.")
        self.Judge = JudgeNode(logging_enabled=self.logging_enabled)
        self.Handler = HandlerNode(logging_enabled=self.logging_enabled)
        total_segments = 2 ** self.dimensions
        for i in range(total_segments):
            
            splitter_pos = []
            reviewer_pos = []
            for dim in range(self.dimensions):
                if (i >> dim) & 1:
                    splitter_pos.append(-1)
                    reviewer_pos.append(-self.max_x)
                else:
                    splitter_pos.append(1)
                    reviewer_pos.append(self.max_x)

            splitter = SplitterNode(position=splitter_pos, logging_enabled=self.logging_enabled)
            reviewer = ReviewerNode(logging_enabled=self.logging_enabled, position=reviewer_pos, max_x=self.max_x)

            full_grid = int(self.max_x) ** self.dimensions   # 10 x 10 = 100
            usable_slots = max(1, full_grid - 2)             # remove origin + buffer
            num_nodes = max(2, int(math.floor(
                usable_slots * self.processing_node_percentage
            )))

            node_positions = self.generate_evenly_spaced_positions(self.dimensions, self.max_x, splitter_pos= splitter_pos, coverage=0.9, num_nodes=num_nodes)
            processing_nodes = [ProcessingNode(position=pos, max_x=self.max_x) for pos in node_positions]
            splitter.connect_processing_nodes(processing_nodes)
            connectable_nodes = []
            connectable_nodes.extend(processing_nodes)
            connectable_nodes.append(reviewer)
            for node in processing_nodes:
                node.find_nearest_neighbors(connectable_nodes, percentage=0.03)
            self.segments.append({
                'id': i,
                'reviewer': reviewer,
                'splitter': splitter,
                'processor': processing_nodes
            })
        self.Judge.segments = self.segments

            

    def run_baseline(self, file_path, nexus_file=None):
        self.display("Running DragonChild v7 main process.")
        self.df = self.base_preprocessing(file_path)
        if self.df is None:
            raise ValueError("Preprocessing failed; DataFrame is None.")
        self.display(f"Processed DataFrame shape: {self.df.shape}")

        if nexus_file:
            self.display(f"Loading Nexus structure from file: {nexus_file}")
            # Placeholder for loading nexus structure
            pass
        else:
            self.display("No Nexus file provided; initializing new Nexus structure.")
            self.initialize_nexus()
            
        self.display("Nexus structure initialized.")

    def train(self):
        self.display("Training phase not yet implemented.")
        pass

    def test(self):
        self.display("Testing phase not yet implemented.")
        pass

    def infer(self, input_data):
        self.display("Inference phase not yet implemented.")
        if self.Judge is None or self.Handler is None:
            self.display("Nexus structure not initialized; cannot perform inference.")
            return None
        segment_relevance = self.Judge.calculate_segment_relevance(input_data)
        activated_segments = self.Judge.select_segments(segment_relevance)
        segment_predictions = []
        segment_relevances = []
        for segment in activated_segments:
            if segment['splitter'] is None:
                self.display("Segment splitter not found; skipping segment.")
                continue

            reviewer_processed = False

            segment_id = segment['id']
            segment_relevance_score = segment_relevance.get(segment_id, 0.0)
            feature_relevance = segment['splitter'].calculate_feature_relevance(input_data)
            signal = segment['splitter'].generate_signal(segment_relevance_score, feature_relevance, len(segment['processor']), input_data)
            segment['splitter'].signal = signal
            expected_signal_count, signal_clones = segment['splitter'].forward_signals()
            self.display(f"Expected signals count for reviewer at {segment['reviewer'].position}: {expected_signal_count}")
            segment['reviewer'].expected_signals = expected_signal_count
            expected_steps = expected_signal_count * signal.life
            pbar = tqdm(
                total=expected_steps,
                desc=f"Segment {segment_id} @ {segment['splitter'].position}",
                leave=False
            )   
            
            max_steps = expected_steps * 3
            steps = 0
            self.display(f"Beginning processing loop for segment {segment_id} using {len(signal_clones)} signals. Max amount of steps: {max_steps}")

            while True:
                steps += 1
                if steps > max_steps:
                    self.display("Max inference steps reached; aborting segment.")
                    break
                active_nodes = [node for node in segment['processor'] if node.signal is not None]

                while active_nodes:
                    node = active_nodes.pop(0)
                    signal = node.process()

                    # tqdm: update progress
                    pbar.update(1)

                    if signal:
                        pbar.set_postfix({
                            "node": node.position,
                            "life": signal.life,
                            "μ": round(signal.prediction, 3),
                            "σ²": round(signal.accumulated_variance, 3)
                        })

                        next_node = node.choose_next_node()
                        if next_node:
                            next_node.receive_signal(signal)
                            if next_node not in active_nodes:
                                active_nodes.append(next_node)
                        if node.queued:
                            active_nodes.append(node)
                        if node.queued != True:
                            node.signal = None  # Clear signal after forwarding

                    if segment['reviewer'].prepped and not reviewer_processed:
                        self.display(
                            f"Reviewer at {segment['reviewer'].position} is prepped to process all signals."
                        )
                        pbar.close()
                        seg_pred, seg_rel = segment['reviewer'].process()
                        segment_predictions.append(seg_pred)
                        segment_relevances.append(seg_rel)
                        reviewer_processed = True
                        break

                dead_signals = [sig for sig in signal_clones if sig is None or not sig.alive]

                if len(dead_signals) >= expected_signal_count and not reviewer_processed:
                    self.display(
                        f"All signals for reviewer at {segment['reviewer'].position} are dead; proceeding to process."
                    )
                    pbar.close()
                    segment['reviewer'].prepped = True
                    seg_pred, seg_rel = segment['reviewer'].process()
                    segment_predictions.append(seg_pred)
                    segment_relevances.append(seg_rel)
                    reviewer_processed = True
                    break
                if not active_nodes and not reviewer_processed:
                    self.display(
                        f"No active nodes remaining for segment {segment_id}; forcing reviewer process."
                    )
                    self.display("Validating no active signals remain.")
                    for sig in signal_clones:
                        if sig is not None and sig.alive:
                            self.display("Active carrier signal found; bug identified in processing.")
                        self.display(f"signal life: {sig.life}, signal position: {sig.position}, signal amount of visited nodes: {len(sig.visited_nodes)}")
                            
                    for node in segment['processor']:
                        if node.signal is not None:
                            self.display("Active node signal found; bug identified in processing.")
                            
                    self.display(f"Only {len(segment['reviewer'].signals)} signals collected.")
                    pbar.close()
                    segment['reviewer'].prepped = True
                    seg_pred, seg_rel = segment['reviewer'].process()
                    segment_predictions.append(seg_pred)
                    segment_relevances.append(seg_rel)
                    reviewer_processed = True
                    break
        self.display("All segments processed for inference.")
        self.display(f"Beginning final aggregation at Handler node with {len(segment_predictions)} segments.")
        for pred, rel in zip(segment_predictions, segment_relevances):
            self.Handler.receive_report([pred, rel])
        final_prediction = self.Handler.process()
        self.display(f"Final aggregated prediction: {final_prediction}")
        







        

    
if __name__ == "__main__":
    print("Starting DragonChild v7 Main Module Test")
    print("=========================================")
    print("Note: This is a structural test; full functionality not implemented.")
    main_module = Main(logging_enabled=True, dimensions=2, max_x=10, processing_node_percentage=0.9)
    test_file_path = "Exam_Score_Prediction.csv"  # Replace with actual path
    main_module.run_baseline(test_file_path)
    for segment in main_module.segments:
        print(f"Segment Reviewer: {segment['reviewer']}")
        print(f"Segment Splitter: {segment['splitter']}")
        print(f"Number of Processing Nodes: {len(segment['processor'])}")
        node_count = 0
        for node in segment['processor']:
            if segment['reviewer'] in node.connected_nodes:
                print(f"reviewer connected to Node: {node}")
                node_count += 1
        print(f"Number of connected nodes to reviewer: {node_count} for segment {segment['reviewer'].position}")
    print("DragonChild v7 Baseline Initialization Module Test Complete")
    print("=========================================")
    print("Testinging inference with basic input data.")
    df = Main().base_preprocessing(test_file_path)
    if df is not None:
        sample_input = df.iloc[0].to_dict()
        print(f"Sample input data: {sample_input}")
        main_module.infer(sample_input)
        print("Inference test complete.")
        print("Actual score: ", df.iloc[0]['exam_score'])
    else:
        print("Preprocessing failed; cannot perform inference test.")


        
