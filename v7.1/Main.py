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
from Nodes.Logger import Logger
import math
import random
from tqdm import tqdm
import time
import numpy as np
class Main:
    def __init__(self, logging_enabled=False, dimensions=2, max_x=None, processing_node_percentage=0.9):
        self.logging_enabled = logging_enabled
        self.max_x = max_x
        self.df = None
        self.Judge = None
        self.Handler = None
        self.PreProcessor = None
        self.segments = []
        self.logger = None
        self.dimensions = dimensions
        self.processing_node_percentage = processing_node_percentage

    def display(self, message, Loud):
        if self.logging_enabled:
            if self.logger is None:
                self.logger = Logger(enabled=True)
            message = (f"[Main] {message}")
            self.logger.log(message, Loud=Loud)

    def base_preprocessing(self, file_path, Loud):
        self.display(f"Starting base preprocessing on dataset: {file_path}", Loud= Loud)
        preprocessor = PreProcessingNode(logging_enabled=self.logging_enabled, logger=self.logger)
        processed_data = preprocessor.process(file_path, Loud= Loud)
        self.PreProcessor = preprocessor
        self.display("Base preprocessing complete.", Loud= Loud)
        return processed_data
    def generate_evenly_spaced_positions(
        self, dimensions, max_x, splitter_pos, Loud, coverage=0.9, num_nodes=10, 
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
    def initialize_nexus(self, Loud):
        self.display("Initializing Nexus structure.", Loud= Loud)
        if self.logging_enabled:
            self.logger = Logger(enabled=True)
            self.display(f"Nexus dimensions: {self.dimensions}, max_x: {self.max_x}, processing node percentage: {self.processing_node_percentage}", Loud= Loud)
        if self.max_x is None:
            raise ValueError("max_x must be defined to initialize Nexus.")
        self.Judge = JudgeNode(dimensions=self.dimensions, logging_enabled=self.logging_enabled, logger=self.logger)
        self.Handler = HandlerNode(logging_enabled=self.logging_enabled, logger=self.logger)
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

            splitter = SplitterNode(position=splitter_pos, logging_enabled=self.logging_enabled, logger=self.logger)
            reviewer = ReviewerNode(logging_enabled=self.logging_enabled, position=reviewer_pos, max_x=self.max_x, logger=self.logger)
            
            full_grid = int(self.max_x) ** self.dimensions   # 10 x 10 = 100
            usable_slots = max(1, full_grid - 2)             # remove origin + buffer
            num_nodes = max(2, int(math.floor(
                usable_slots * self.processing_node_percentage
            )))

            node_positions = self.generate_evenly_spaced_positions(self.dimensions, self.max_x, splitter_pos= splitter_pos, coverage=0.9, num_nodes=num_nodes, Loud= Loud)
            processing_nodes = [ProcessingNode(position=pos, max_x=self.max_x, logging_enabled=self.logging_enabled, logger=self.logger) for pos in node_positions]
            splitter.connect_processing_nodes(processing_nodes, Loud= Loud)
            connectable_nodes = []
            connectable_nodes.extend(processing_nodes)
            connectable_nodes.append(reviewer)
            for node in processing_nodes:
                node.find_nearest_neighbors(connectable_nodes, percentage=0.03, Loud= Loud)
            self.segments.append({
                'id': i,
                'reviewer': reviewer,
                'splitter': splitter,
                'processor': processing_nodes
            })
        self.Judge.segments = self.segments

            

    def run_baseline(self, file_path, Loud, nexus_file=None):
        self.display("Running DragonChild v7 main process.", Loud= Loud)
        self.df = self.base_preprocessing(file_path, Loud= Loud)
        if self.df is None:
            raise ValueError("Preprocessing failed; DataFrame is None.")
        self.display(f"Processed DataFrame shape: {self.df.shape}", Loud= Loud)

        if nexus_file:
            self.display(f"Loading Nexus structure from file: {nexus_file}", Loud= Loud)
            # Placeholder for loading nexus structure
            pass
        else:
            self.display("No Nexus file provided; initializing new Nexus structure.", Loud= Loud)
            self.initialize_nexus(Loud= Loud)
            
        self.display("Nexus structure initialized.", Loud= Loud)

    def train(self, Loud):
        with tqdm(total=5, desc="Training Phase", leave=True) as train_pbar:
            self.display("Training phase not yet implemented.", Loud= Loud)
            train_pbar.set_description("Preparing Training Data")
            if self.df is None:
                self.display("No preprocessed data available for training.", Loud= Loud)
                return
            training_rows = random.sample(self.df.index.tolist(), int(4 * (len(self.df) / 5))) 
            train_X = self.df.loc[training_rows].drop(columns=['student_id'])
            train_X = train_X.drop(columns=['exam_score']).to_dict(orient='records')
            train_y = self.df.loc[training_rows]['exam_score'].tolist()
            self.test_X = self.df.drop(index=training_rows).drop(columns=['student_id'])
            self.test_y = self.df.drop(index=training_rows)['exam_score'].tolist()
            self.display(f"Training data size: {len(train_X)}, Testing data size: {len(self.test_X)}", Loud= Loud)
            train_pbar.update(1)
            train_pbar.set_description("Judge Training")

            if self.Judge is None:
                self.display("Nexus structure not initialized; cannot perform training.", Loud= Loud)
                return
            training_vectors = []
            for row in train_X:
                vectorized_data = self.Judge.vectorize_input(row, Loud= Loud)
                training_vectors.append(vectorized_data)
            self.Judge.train(training_vectors, Loud = Loud, max_amount_scaleups=15)
            self.Judge.assign_segments_to_clusters(Loud= Loud)
            self.Judge.save_cluster_map(training_vectors, self.Judge.segment_cluster_map, Loud= Loud)
            segments_training_split = {}
            train_pbar.update(1)

            train_pbar.set_description("Assigning Training Data to Segments")
            self.display("Assigning training data to segments based on Judge decisions.", Loud= Loud)
            for row in train_X:
                segment_relevances = self.Judge.calculate_segment_relevance(row, Loud= Loud)
                activated_segments = self.Judge.select_segments(segment_relevances, Loud= Loud)
                for segment in activated_segments:
                    segment_id = segment['id']
                    if segment_id not in segments_training_split:
                        segments_training_split[segment_id] = {
                            'X': [],
                            'y': []
                        }
                    segments_training_split[segment_id]['X'].append(row)
                    index = train_X.index(row)
                    segments_training_split[segment_id]['y'].append(train_y[index])        
            self.display("Judge finished training and assigned training data to segments.", Loud= Loud)
            train_pbar.update(1)

            train_pbar.set_description("Reviewing Segment Training Data")
            self.display(f"Segments with training data: {list(segments_training_split.keys())}", Loud= Loud)
            self.display("Amount of training data for each segment:", Loud= Loud)
            for segment_id, data in segments_training_split.items():
                self.display(f"Segment {segment_id}: {len(data['X'])} samples", Loud= Loud)
            self.display("Sample training data for segment 0:", Loud= Loud)
            self.display(segments_training_split.get(0, {'X': [], 'y': []})['X'][:2], Loud= Loud)
            train_pbar.update(1)

            train_pbar.set_description("Segment-wise Training")
            self.display("Beginning training for each segment's Splitter node and processor nodes.", Loud= Loud)

            def supervised_step(segment, row, target, Loud=Loud):
                splitter = segment['splitter']

                # 1. Attention (feature relevance)
                feature_relevance = splitter.calculate_feature_relevance(row, Loud=Loud)

                # 2. Generate signal
                signal = splitter.generate_signal(
                    segment_relevance=1.0,
                    feature_relevance=feature_relevance,
                    node_count=len(segment['processor']),
                    inputdata=row,
                    Loud=Loud
                )

                splitter.signal = signal
                _, _ = splitter.forward_signals(Loud=Loud)

                # 3. Forward pass + local processing
                preds = []
                active_nodes = []

                for node in segment['processor']:
                    if node.signal:
                        out = node.process(Loud=Loud)
                        if out:
                            preds.append(out.prediction)
                            active_nodes.append(node)

                if not preds:
                    return

                prediction = sum(preds) / len(preds)
                residual = target - prediction

                # 4. Train processing nodes FIRST (value learning)
                for node in active_nodes:
                    node.train(target, Loud)
                    node.update_geometry(residual, Loud)

                # 5. Train splitter SECOND (attention learning)
                splitter.update_feature_relations(row, residual, Loud=Loud)
            self.display("Starting segment-wise training loop.", Loud= Loud)
            segment_iter = tqdm(self.segments, desc="Segments", leave=True)
            for segment in segment_iter:
                segment_id = segment['id']
                if segment_id in segments_training_split:
                    seg_X = segments_training_split[segment_id]['X']
                    seg_y = segments_training_split[segment_id]['y']
                    # Progress bar for supervised steps
                    with tqdm(total=len(seg_X), desc=f"Supervised Steps (Segment {segment_id})", leave=False) as sup_pbar:
                        for row, y in zip(seg_X, seg_y):
                            vec = self.Judge.vectorize_input(row, Loud= Loud)
                            vec_dict = self.Judge.vector_to_feature_dict(vec)
                            supervised_step(segment, vec_dict, y, Loud= Loud)
                            sup_pbar.update(1)
                else:
                    self.display(f"No training data assigned to segment {segment_id}; skipping training for this segment.", Loud= Loud)
                    seg_X, seg_y = [], []
                segment_iter.set_postfix({
                    "Segment ID": segment_id,
                    "Trained Samples": len(seg_X)
                })
                segment_iter.refresh()
            train_pbar.update(1)
            train_pbar.set_description("Finalizing Training")
            self.display("Training phase complete.", Loud= Loud)
                        


        

    def test(self, Loud):
        self.display("Testing phase not yet implemented.", Loud= Loud)
        if self.df is None or self.test_X is None or self.test_y is None:
            self.display("No preprocessed data available for testing.", Loud= Loud)
            return  
        glo_preds = []
        glo_acts = []
        responses = []
        accuracy = 0.0
        precision = 0.0
        
        with tqdm(total=len(self.test_X), desc="Testing Phase", leave=True) as test_pbar:
            for input_data, actual in zip(self.test_X.to_dict(orient='records'), self.test_y):
                prediction = self.infer(input_data, Loud=Loud)
                responses.append((prediction, actual))
                test_pbar.update(1)

        # Example: For regression, count predictions within 5 points of actual
        correct = 0
        for pred, act in responses:
            self.display(f"Predicted: {pred}, Actual: {act}", Loud= Loud)
            if pred is not None and abs(pred - act) < 5:  # adjust threshold as needed
                correct += 1

        preds = [p for p, a in responses if p is not None]
        acts = [a for p, a in responses if p is not None]
        if preds and acts:
            accuracy = correct / len(responses) if responses else 0
            self.display(f"Test accuracy (within threshold): {accuracy:.2f}", Loud= Loud)

            # Accuracy: mean absolute error (lower is better)
            mae = np.mean([abs(p - a) for p, a in zip(preds, acts)])
            self.display(f"Mean Absolute Error (MAE): {mae:.2f}", Loud= Loud)
            # Accuracy as 1 - normalized MAE (to max actual value)
            max_actual = max(acts) if acts else 1
            accuracy = 1 - (mae / max_actual) if max_actual else 0
            self.display(f"Accuracy (1 - normalized MAE): {accuracy:.2f}", Loud= Loud)
        else:
            self.display("No valid predictions for accuracy calculation.", Loud= Loud)

        # Precision: closeness of medians
        if preds and acts:
            median_pred = np.median(preds)
            median_act = np.median(acts)
            median_diff = abs(median_pred - median_act)
            # Precision as 1 - normalized median difference
            precision = 1 - (median_diff / max_actual) if max_actual else 0
            self.display(f"Precision (1 - normalized median difference): {precision:.2f}", Loud= True)
            self.display(f"Final Test Results - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}", Loud= True)
        else:
            self.display("No valid predictions for precision calculation.", Loud= True)

    def infer(self, input_data, Loud):
        self.display("Inference phase not yet implemented.", Loud= Loud)
        if self.Judge is None or self.Handler is None:
            self.display("Nexus structure not initialized; cannot perform inference.", Loud= Loud)
            return None
        
        segment_relevance = self.Judge.calculate_segment_relevance(input_data, Loud= Loud)
        activated_segments = self.Judge.select_segments(segment_relevance, Loud= Loud)
        segment_predictions = []
        segment_relevances = []
        for segment in activated_segments:
            if segment['splitter'] is None:
                self.display("Segment splitter not found; skipping segment.", Loud= Loud)
                continue

            reviewer_processed = False

            segment_id = segment['id']
            segment_relevance_score = segment_relevance.get(segment_id, 0.0)
            feature_relevance = segment['splitter'].calculate_feature_relevance(input_data, Loud= Loud)
            signal = segment['splitter'].generate_signal(segment_relevance_score, feature_relevance, len(segment['processor']), input_data, Loud= Loud)
            segment['splitter'].signal = signal
            expected_signal_count, signal_clones = segment['splitter'].forward_signals(Loud)
            self.display(f"Expected signals count for reviewer at {segment['reviewer'].position}: {expected_signal_count}", Loud= Loud)
            segment['reviewer'].expected_signals = expected_signal_count
            expected_steps = expected_signal_count * signal.life
            pbar = tqdm(
                total=expected_steps,
                desc=f"Segment {segment_id} @ {segment['splitter'].position}",
                leave=False
            )   
            
            max_steps = expected_steps * 3
            steps = 0
            self.display(f"Beginning processing loop for segment {segment_id} using {len(signal_clones)} signals. Max amount of steps: {max_steps}", Loud= Loud)

            while True:
                steps += 1
                if steps > max_steps:
                    self.display("Max inference steps reached; aborting segment.", Loud)
                    break
                active_nodes = [node for node in segment['processor'] if node.signal is not None]

                while active_nodes:
                    node = active_nodes.pop(0)
                    signal = node.process(Loud)

                    # tqdm: update progress
                    pbar.update(1)

                    if signal:
                        pbar.set_postfix({
                            "node": node.position,
                            "life": signal.life,
                            "μ": round(signal.prediction, 3),
                            "σ²": round(signal.accumulated_variance, 3)
                        })

                        next_node = node.choose_next_node(Loud= Loud)
                        if next_node:
                            next_node.receive_signal(signal, Loud= Loud)
                            if next_node not in active_nodes:
                                active_nodes.append(next_node)
                        if node.queued:
                            active_nodes.append(node)
                        if node.queued != True:
                            node.signal = None  # Clear signal after forwarding

                    if segment['reviewer'].prepped and not reviewer_processed:
                        self.display(
                            f"Reviewer at {segment['reviewer'].position} is prepped to process all signals.", Loud= Loud
                        )
                        pbar.close()
                        seg_pred, seg_rel = segment['reviewer'].process(Loud= Loud)
                        segment_predictions.append(seg_pred)
                        segment_relevances.append(seg_rel)
                        reviewer_processed = True
                        break

                dead_signals = [sig for sig in signal_clones if sig is None or not sig.alive]

                if len(dead_signals) >= expected_signal_count and not reviewer_processed:
                    self.display(
                        f"All signals for reviewer at {segment['reviewer'].position} are dead; proceeding to process.", Loud= Loud
                    )
                    pbar.close()
                    segment['reviewer'].prepped = True
                    seg_pred, seg_rel = segment['reviewer'].process(Loud= Loud)
                    segment_predictions.append(seg_pred)
                    segment_relevances.append(seg_rel)
                    reviewer_processed = True
                    break
                if not active_nodes and not reviewer_processed:
                    self.display(
                        f"No active nodes remaining for segment {segment_id}; forcing reviewer process.", Loud= Loud
                    )
                    self.display("Validating no active signals remain.", Loud= Loud)
                    for sig in signal_clones:
                        if sig is not None and sig.alive:
                            self.display("Active carrier signal found; bug identified in processing.", Loud= Loud)
                        self.display(f"signal life: {sig.life}, signal position: {sig.position}, signal amount of visited nodes: {len(sig.visited_nodes)}", Loud= Loud)
                            
                    for node in segment['processor']:
                        if node.signal is not None:
                            self.display("Active node signal found; bug identified in processing.", Loud= Loud)
                            
                    self.display(f"Only {len(segment['reviewer'].signals)} signals collected.", Loud= Loud)
                    pbar.close()
                    segment['reviewer'].prepped = True
                    seg_pred, seg_rel = segment['reviewer'].process(Loud= Loud)
                    segment_predictions.append(seg_pred)
                    segment_relevances.append(seg_rel)
                    reviewer_processed = True
                    break
        self.display("All segments processed for inference.", Loud= Loud)
        self.display(f"Beginning final aggregation at Handler node with {len(segment_predictions)} segments.", Loud= Loud)
        for pred, rel in zip(segment_predictions, segment_relevances):
            self.Handler.receive_report([pred, rel], Loud= Loud)
        final_prediction = self.Handler.process(Loud= Loud)
        self.display(f"Final aggregated prediction: {final_prediction}", Loud= Loud)
        return final_prediction
        







        

    
if __name__ == "__main__":
    
    print("Starting DragonChild v7 Main Module Test")
    print("=========================================")
    print("Note: This is a structural test; full functionality not implemented.")
    print("=========================================")
    print("Initializing Main module with logging enabled.")
    input("Press Enter to continue...")
    main_module = Main(logging_enabled=True, dimensions=2, max_x=100, processing_node_percentage=0.9)
    Main.display(main_module, "Running Baseline Initialization", Loud= True)
    test_file_path = "Exam_Score_Prediction.csv"  # Replace with actual path
    main_module.run_baseline(test_file_path, Loud = True)
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
    Main.display(main_module, "Baseline Initialization Complete", Loud= True)
    print("DragonChild v7 Baseline Initialization Module Test Complete")
    print("=========================================")
    print("Testinging Pre-Training inference with basic input data.")
    input("Press Enter to continue...")
    Main.display(main_module, "Running pre-train Inference Test Phase", Loud= True)
    time.sleep(3)
    df = Main().base_preprocessing(test_file_path, Loud = True)
    if df is not None:
        sample_input = df.iloc[0].to_dict()
        print(f"Sample input data: {sample_input}")
        main_module.infer(sample_input, Loud = True)
        print("Inference test complete.")
        print("Actual score: ", df.iloc[0]['exam_score'])
    else:
        print("Preprocessing failed; cannot perform inference test.")

    Main.display(main_module, "Pre-Training Inference Test Phase Complete", Loud= True)

    print("==========================================")
    print("Beginning training phase test.")
    input("Press Enter to continue...")
    Main.display(main_module, "Running Quiet Training Phase", Loud= False)
    print("Running Quiet Training Phase...")
    main_module.train(Loud = False)
    print("Training phase test complete.")
    print("==========================================")
    print("Conducting Test phase post-training.")
    input("Press Enter to continue...")
    Main.display(main_module, "Running Silent Test Phase", Loud= False)
    main_module.test(Loud = False)


