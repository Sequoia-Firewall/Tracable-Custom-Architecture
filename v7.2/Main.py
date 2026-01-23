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
import gc
from Nodes.Logger import Logger
import math
import random
from tqdm import tqdm
import time
import numpy as np

random.seed(42)
np.random.seed(42)

class Main:
    def __init__(self, ignored_features = [], logging_enabled=False, dimensions=2, max_x=None, processing_node_percentage=0.9):
        self.ignored_features = ignored_features
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
        self.mean_y = 0.0

    def display(self, message, Loud):
        if self.logging_enabled:
            if self.logger is None:
                self.logger = Logger(enabled=True)
            message = (f"[Main] {message}")
            self.logger.log(message, Loud=Loud)

    def base_preprocessing(self, file_path, demo, Loud):
        self.display(f"Starting base preprocessing on dataset: {file_path}", Loud= Loud)
        preprocessor = PreProcessingNode(logging_enabled=self.logging_enabled, logger=self.logger)
        processed_data = preprocessor.process(file_path, demo = demo,Loud= Loud)
        self.PreProcessor = preprocessor
        
        if processed_data is None:
            self.display("Preprocessing failed; no data returned.", Loud= Loud)
            return None
        
        self.mean_y = processed_data['exam_score'].mean()
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
        from tqdm import tqdm
        self.display("Initializing Nexus structure.", Loud= Loud)
        with tqdm(total=3, desc="Nexus Initialization", leave=True) as nexus_pbar:
            if self.logging_enabled:
                self.logger = Logger(enabled=True)
                self.display(f"Nexus dimensions: {self.dimensions}, max_x: {self.max_x}, processing node percentage: {self.processing_node_percentage}", Loud= Loud)
            if self.max_x is None:
                raise ValueError("max_x must be defined to initialize Nexus.")
            self.Judge = JudgeNode(dimensions=self.dimensions, logging_enabled=self.logging_enabled, logger=self.logger)
            self.Handler = HandlerNode(logging_enabled=self.logging_enabled, logger=self.logger)
            total_segments = 2 ** self.dimensions
            nexus_pbar.update(1)
            nexus_pbar.set_description("Creating Segments")


            def create_segment(i, tqdm_lock=None):
                splitter_pos = []
                reviewer_pos = []
                if self.max_x is None:
                    raise ValueError("max_x must be defined to create segments.")
                for dim in range(self.dimensions):
                    if (i >> dim) & 1:
                        splitter_pos.append(-1)
                        reviewer_pos.append(-self.max_x)
                    else:
                        splitter_pos.append(1)
                        reviewer_pos.append(self.max_x)

                splitter = SplitterNode(position=splitter_pos, logging_enabled=self.logging_enabled, logger=self.logger)
                reviewer = ReviewerNode(logging_enabled=self.logging_enabled, position=reviewer_pos, max_x=self.max_x, logger=self.logger)
                full_grid = int(self.max_x) ** self.dimensions
                usable_slots = max(1, full_grid - 2)
                num_nodes = max(2, int(math.floor(usable_slots * self.processing_node_percentage)))
                node_positions = self.generate_evenly_spaced_positions(self.dimensions, self.max_x, splitter_pos=splitter_pos, coverage=0.9, num_nodes=num_nodes, Loud=Loud)
                processing_nodes = [ProcessingNode(position=pos, max_x=self.max_x, logging_enabled=self.logging_enabled, ignored_features= self.ignored_features, logger=self.logger) for pos in node_positions]
                splitter.connect_processing_nodes(processing_nodes, Loud=Loud)
                if tqdm_lock:
                    with tqdm_lock:
                        segment_pbar.update(1)
                return {
                    'id': i,
                    'reviewer': reviewer,
                    'splitter': splitter,
                    'processor': processing_nodes
                }
            from tqdm import tqdm
            with tqdm(total=total_segments, desc="Creating Segments", leave=False) as segment_pbar:
                segment_pbar.set_description("Creating Segments")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(create_segment, i) for i in range(total_segments)]
                    for future in concurrent.futures.as_completed(futures):
                        segment_pbar.update(1)
                    self.segments = [f.result() for f in futures]
                for _ in self.segments:
                    segment_pbar.update(1)

                self.Judge.segments = self.segments
                nexus_pbar.update(1)
                nexus_pbar.set_description("Connecting processing nodes")

        
            
            import concurrent.futures
            from tqdm import tqdm
            def connect_segment(segment, bar):
                for node in segment['processor']:
                    node.find_nearest_neighbors(segment['processor'] + [segment['reviewer']], percentage=0.03, Loud=Loud)
                    bar.update(1)
                bar.close()
            bars = []
            tasks = []
            for i, segment in enumerate(self.segments):
                bar = tqdm(total=len(segment['processor']), desc=f"Segment {segment['id']} Connecting", position=i, leave=False)
                bars.append(bar)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, segment in enumerate(self.segments):
                    bar = bars[i]
                    tasks.append(executor.submit(connect_segment, segment, bar))
                concurrent.futures.wait(tasks)

            for bar in bars:
                bar.close()
            for segment in self.segments:
                nodes_connected = []
                for node in segment['processor']:
                    if segment['reviewer'] in node.connected_nodes:
                        nodes_connected.append(node)
                segment['reviewer'].check_enough_nodes(nodes_connected, segment['processor'], Loud= Loud)
                nodes_connected = []
                for node in segment['processor']:
                    if segment['reviewer'] in node.connected_nodes:
                        nodes_connected.append(node)

                self.display(f"Segment {segment['id']} - Reviewer connected to {len(nodes_connected)}/{len(segment['processor'])} processing nodes.", Loud= Loud)
            
            self.display(f"Initialized {total_segments} segments with processing nodes.", Loud= Loud)

            

    def run_baseline(self, file_path, demo, Loud, nexus_file=None):
        self.display("Running DragonChild v7 main process.", Loud= Loud)
        self.df = self.base_preprocessing(file_path, demo, Loud= Loud)
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
        gc_was_enabled = gc.isenabled()
        gc.disable() #to prevent other errors resulting from GC use and multithreading
        with tqdm(total=6, desc="Training Phase", leave=True) as train_pbar:
            self.display("Training phase not yet implemented.", Loud= Loud)
            train_pbar.set_description("Preparing Training Data")
            if self.df is None:
                self.display("No preprocessed data available for training.", Loud= Loud)
                return []
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
                return []
            training_vectors = []
            for row in train_X:
                vectorized_data = self.Judge.vectorize_input(row, Loud= Loud)
                training_vectors.append(vectorized_data)
            self.Judge.train(training_vectors, Loud = Loud)
            self.Judge.save_cluster_map(training_vectors, self.Judge.segment_cluster_map, Loud= Loud)
            segments_training_split = {}
            train_pbar.update(1)

            train_pbar.set_description("Assigning Training Data to Segments")
            self.display("Assigning training data to segments based on Judge decisions.", Loud= Loud)
            def assign_row(row):
                if self.Judge is None:
                    self.display("Nexus structure not initialized; cannot assign training data.", Loud= Loud)
                    return []
                segment_relevances = self.Judge.calculate_segment_relevance(row, Loud= Loud)
                activated_segments = self.Judge.select_segments(segment_relevances, Loud= Loud)
                result = []
                for segment in activated_segments:
                    segment_id = segment['id']
                    result.append((segment_id, row, train_X.index(row),))
                if result is None:
                    self.display("No segments activated for this row; possible issue in Judge logic.", Loud= Loud)
                    return []
                return result

            results = []
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for res in executor.map(assign_row, train_X):
                    results.extend(res)
            executor.shutdown(wait=True)
            for item in results:
                if len(item) == 3:
                    segment_id, row, idx = item # type: ignore
                    if segment_id not in segments_training_split:
                        segments_training_split[segment_id] = {
                            'X': [],
                            'y': []
                        }
                    segments_training_split[segment_id]['X'].append(row)
                    segments_training_split[segment_id]['y'].append(train_y[idx])
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
            gc.collect()
            def supervised_step(segment, row, target, Loud=Loud):
                from collections import deque
                
                splitter = segment['splitter']
                reviewer = segment['reviewer']

                # 1. Generate and distribute initial signals
                feature_relevance = splitter.calculate_feature_relevance(row, Loud=Loud)
                signal = splitter.generate_signal(
                    segment_relevance=1.0,
                    feature_relevance=feature_relevance,
                    node_count=len(segment['processor']),
                    inputdata=row,
                    Loud=Loud,
                    mean=self.mean_y
                )

                splitter.signal = signal
                expected_signals, _ = splitter.forward_signals(Loud=Loud)

                # 2. Initialize active queue with nodes that received initial signals
                active_nodes = deque(
                    node for node in segment['processor'] if node.signal is not None
                )

                preds = []
                trained_nodes = []
                node_signals = {}  # ← NEW: Store signals before clearing
                max_steps = expected_signals * signal.life * 2
                steps = 0

                # 3. PROPAGATE SIGNALS through the network
                while active_nodes and steps < max_steps:
                    steps += 1
                    node = active_nodes.popleft()
                    
                    # Process at this node
                    result = node.process(Loud=Loud)
                    if result and hasattr(result, "prediction"):
                        preds.append(result.prediction)
                        trained_nodes.append(node)
                        node_signals[node] = node.signal  # ← NEW: Save signal before clearing
                        
                        # CRITICAL: Forward to next node
                        next_node = node.choose_next_node(Loud=Loud)
                        
                        if next_node:
                            next_node.receive_signal(result, Loud=Loud)
                            
                            # Don't forward to reviewer during training
                            if next_node is not reviewer:
                                active_nodes.append(next_node)
                        
                        # Re-queue if node has queued signals
                        if node.queued:
                            active_nodes.append(node)
                        # ← REMOVED: else: node.signal = None

                # 4. Calculate prediction and train all nodes that participated
                if not preds:
                    return

                prediction = sum(preds) / len(preds)
                residual = target - prediction

                # 5. Train all nodes that processed signals
                for node in trained_nodes:
                    if node in node_signals:  # ← NEW: Check if we have saved signal
                        node.signal = node_signals[node]  # ← NEW: Restore signal
                        node.train(target, Loud)
                        # Optional: update geometry less frequently
                        if random.random() < 0.1:  # Only 10% of the time
                            node.update_geometry(residual, Loud)
                        node.signal = None  # ← NEW: Clear after training

                # 6. Train splitter
                splitter.update_feature_relations(row, residual, Loud=Loud)
            gc.collect()
            self.display("Starting segment-wise training loop.", Loud= Loud)
            import concurrent.futures
            def train_segment(segment):
                if self.Judge is None:
                    self.display("Nexus structure not initialized; cannot train segment.", Loud= Loud)
                    return (None, 0)
                segment_id = segment['id']
                if segment_id in segments_training_split:
                    seg_X = segments_training_split[segment_id]['X']
                    seg_y = segments_training_split[segment_id]['y']
                    with tqdm(total=len(seg_X), desc=f"Supervised Steps (Segment {segment_id})", leave=False) as sup_pbar:
                        for row, y in zip(seg_X, seg_y):
                            vec = self.Judge.vectorize_input(row, Loud= Loud)
                            vec_dict = self.Judge.vector_to_feature_dict(vec)
                            supervised_step(segment, vec_dict, y, Loud= Loud)
                            sup_pbar.update(1)
                    return (segment_id, len(seg_X))
                else:
                    self.display(f"No training data assigned to segment {segment_id}; skipping training for this segment.", Loud= Loud)
                    return (segment_id, 0)

            segment_iter = tqdm(self.segments, desc="Segments", leave=True)
            results: list[tuple[int | None, int]] = [(None, 0)] * len(self.segments)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_idx = {executor.submit(train_segment, segment): idx for idx, segment in enumerate(self.segments)}
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    segment_id, trained_samples = future.result()
                    results[idx] = (segment_id, trained_samples)
                    segment_iter.set_postfix({
                        "Segment ID": segment_id,
                        "Trained Samples": trained_samples
                    })
                    segment_iter.update(1)
            segment_iter.refresh()
            gc.collect()
            train_pbar.update(1)
            train_pbar.set_description("Refreshing Node Connections")
            bars = []
            for i, segment in enumerate(self.segments):
                bar = tqdm(total=len(segment['processor']), desc=f"Segment {segment['id']} Refreshing", position=i, leave=False)
                bars.append(bar)
            def refresh_segment_geometry(segment, Loud, pbar):
                nodes = segment['processor']
                reviewer = segment['reviewer']
                for node in nodes:
                    node.find_nearest_neighbors(
                        nodes + [reviewer],
                        percentage=0.03,
                        Loud=Loud
                    )
                    pbar.update(1)
                pbar.close()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(refresh_segment_geometry, segment, Loud, bars[i]) for i, segment in enumerate(self.segments)]
                concurrent.futures.wait(futures)
            for bar in bars:
                bar.close()
            total_weight_changes = {'segment': [], 'Feature': [], 'weight_change': []}
            for segment in self.segments:
                total_weight_changes['segment'].append(segment['id'])
                rand_node = random.choice(segment['processor'])
                features = getattr(rand_node, 'total_weight_changes', {}).get('Feature', [])
                weight_changes = getattr(rand_node, 'total_weight_changes', {}).get('weight_change', [])
                if not features or not weight_changes:
                    self.display(f"Warning: No weight changes found for segment {segment['id']} (node: {rand_node}).", Loud=True)
                for feature, weight_change in zip(features, weight_changes):
                    total_weight_changes['Feature'].append(feature)
                    total_weight_changes['weight_change'].append(weight_change)
            gc.collect()
            train_pbar.update(1)
            train_pbar.set_description("Finalizing Training Phase")
            untrained_nodes = []
            for segment in self.segments:
                for node in segment['processor']:
                    if node.times_trained == 0:
                        untrained_nodes.append((segment['id'], node))

            self.display(f"Untrained nodes: {len(untrained_nodes)} / {sum(len(s['processor']) for s in self.segments)}", Loud=True)
            

            if gc_was_enabled:
                gc.enable()
            self.display("Training phase complete.", Loud= True)
            self.display(f"Aggregated weight changes from randomly selected processing nodes: {total_weight_changes}", Loud= True)
                        


        

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
        from collections import deque
        with tqdm(total=4, desc="Inference Phase", leave=True) as infer_pbar:
            self.display("Inference phase started.", Loud=Loud)

            if self.Judge is None or self.Handler is None:
                self.display("Nexus structure not initialized; cannot perform inference.", Loud=Loud)
                return None

            # 1. Judge decision
            segment_relevance = self.Judge.calculate_segment_relevance(input_data, Loud=Loud)
            activated_segments = self.Judge.select_segments(segment_relevance, Loud=Loud)
            infer_pbar.update(1)
            infer_pbar.set_description("Processing Segments")

            segment_predictions = []
            segment_relevances = []

            # 2. Process each segment
            for segment in activated_segments:
                segment_id = segment['id']
                splitter = segment['splitter']
                reviewer = segment['reviewer']
                processors = segment['processor']

                reviewer_processed = False

                feature_relevance = splitter.calculate_feature_relevance(input_data, Loud=Loud)
                signal = splitter.generate_signal(
                    segment_relevance.get(segment_id, 0.0),
                    feature_relevance,
                    len(processors),
                    input_data,
                    Loud=Loud,
                    mean = self.mean_y
                )

                splitter.signal = signal
                expected_signal_count, signal_clones = splitter.forward_signals(Loud=Loud)
                reviewer.expected_signals = expected_signal_count

                # 3. Initialize active node queue (NO global rescans)
                active_nodes = deque(
                    node for node in processors if node.signal is not None
                )

                max_steps = expected_signal_count * signal.life * 2
                steps = 0

                pbar = tqdm(
                    total=max_steps,
                    desc=f"Segment {segment_id} @ {splitter.position}",
                    leave=False
                )

                # 4. Event-driven processing loop
                while active_nodes and not reviewer_processed:
                    steps += 1
                    if steps > max_steps:
                        self.display(f"Max steps reached for segment {segment_id}.", Loud=Loud)
                        break

                    node = active_nodes.popleft()
                    result = node.process(Loud=Loud)
                    pbar.update(1)

                    if result is None:
                        continue

                    # Processing node produced a signal
                    if hasattr(result, "prediction"):
                        sig = result
                        next_node = node.choose_next_node(Loud=Loud)

                        if next_node:
                            next_node.receive_signal(sig, Loud=Loud)
                            if next_node is reviewer:
                                reviewer.receive_signal(sig, Loud=Loud)
                            elif next_node.signal is not None:
                                active_nodes.append(next_node)

                        if node.queued:
                            active_nodes.append(node)
                        else:
                            node.signal = None

                    # Reviewer ready → stop
                    if reviewer.prepped:
                        pbar.close()
                        pred, rel = reviewer.process(Loud=Loud)
                        segment_predictions.append(pred)
                        segment_relevances.append(rel)
                        reviewer_processed = True
                        break

                # 5. Forced reviewer processing if signals died out
                if not reviewer_processed:
                    reviewer.prepped = True
                    pbar.close()
                    pred, rel = reviewer.process(Loud=Loud)
                    segment_predictions.append(pred)
                    segment_relevances.append(rel)

            infer_pbar.update(1)
            infer_pbar.set_description("Aggregating Results")

            # 6. Final aggregation
            for pred, rel in zip(segment_predictions, segment_relevances):
                if pred is not None:
                    self.Handler.receive_report([pred, rel], Loud=Loud)

            infer_pbar.update(1)
            infer_pbar.set_description("Finalizing Prediction")

            final_prediction = self.Handler.process(Loud=Loud)
            infer_pbar.update(1)

            self.display(f"Final aggregated prediction: {final_prediction}", Loud=True)
            self.display(f"True Answer: {input_data.get('exam_score', 'N/A')}", Loud=True)
            return final_prediction

            







        

    
if __name__ == "__main__":
    #temp value
    demo = False

    print("Starting DragonChild v7 Main Module Test")
    print("=========================================")
    print("Note: This is a structural test; full functionality not implemented.")
    print("=========================================")
    print("Initializing Main module with logging enabled.")
    #input("Press Enter to continue...")
    main_module = Main(logging_enabled=True, dimensions=2, max_x=20, ignored_features=('student_id', 'exam_score'), processing_node_percentage=0.9)
    Main.display(main_module, "Running Baseline Initialization", Loud= True)
    test_file_path = "Exam_Score_Prediction.csv"  # Replace with actual path
    main_module.run_baseline(test_file_path, demo=demo, Loud = False)
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
    #input("Press Enter to continue...")
    Main.display(main_module, "Running pre-train Inference Test Phase", Loud= True)
    time.sleep(3)
    df = Main().base_preprocessing(test_file_path, demo=demo, Loud = True)
    if df is not None:
        sample_input = df.iloc[0].to_dict()
        print(f"Sample input data: {sample_input}")
        pred = main_module.infer(sample_input, Loud = False)
        print(f"Inference test complete: {pred}")
        print("Actual score: ", df.iloc[0]['exam_score'])
    else:
        print("Preprocessing failed; cannot perform inference test.")

    Main.display(main_module, "Pre-Training Inference Test Phase Complete", Loud= True)

    print("==========================================")
    print("Beginning training phase test.")
    #input("Press Enter to continue...")
    Main.display(main_module, "Running Quiet Training Phase", Loud= False)
    print("Running Quiet Training Phase...")
    main_module.train(Loud = False)
    print("Training phase test complete.")
    print("==========================================")
    print("Conducting Test phase post-training.")
    input("Press Enter to continue...")
    Main.display(main_module, "Running Silent Test Phase", Loud= False)
    main_module.test(Loud = False)


