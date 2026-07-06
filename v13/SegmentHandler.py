#This is the code to manage just one segment which is used to allow for modularity
from Components.SplitterNode import SplitterNode
from Components.ProcessingNode import ProcessingNode
from Components.Signal import Signal
from Components.ReviewerNode import ReviewerNode
import math
import itertools

class SegmentHandler:
    def __init__(self, maxX, segmentComponents = None, target='exam_score', logger = None, connection_percentage=.08, density = .95, dimensions = 2, classification = 1, segment_id = 0):
        self.segmentComponents = segmentComponents
        self.max_x = maxX
        self.target = target
        self.logger = logger
        self.connection_percentage = connection_percentage
        self.segment_id = segment_id
        self.density = density
        self.dimensions = dimensions
        self.classification = classification
        self.best_epoch_metrics = None

        if self.logger is None:
            raise ValueError("Logger must be provided for SegmentHandler.")

    def display(self, message, classification = None, Loud = True):
        message = f"[Main]: {message}"
        if self.logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.logger.log(message, classification, Loud)

    def initializeSegment(self, loc = []): #loc used to determine if x or y is positive or negative
        if self.max_x is None:
            raise ValueError("max_x must be defined to create segments.")
        if self.logger is None:
            raise ValueError("Logger must be provided to create segments.")

        if len(loc) != self.dimensions and len(loc) != 0:
            raise ValueError(f"loc must have length {self.dimensions} to determine axis signs in {self.dimensions}D space")

        def apply_loc(pos: tuple) -> tuple:
            if not loc:
                return pos
            return tuple(int(coord * sign) for coord, sign in zip(pos, loc))

        has_progress = hasattr(self.logger, 'make_progress')

        splitter_position = apply_loc(tuple(1 for _ in range(self.dimensions)))
        reviewer_position = apply_loc(tuple(self.max_x for _ in range(self.dimensions)))

        splitter = SplitterNode(position=splitter_position, connection_percentage=self.connection_percentage, Logger=self.logger, classification=self.classification)

        # 2D: three equidistant points on the quarter-circle arc at radius max_x
        #     (x-axis, 45°, y-axis). N-D: hypercube corners with ≥1 coord at max_x.
        if self.dimensions == 2:
            r = self.max_x
            reviewer_positions = [
                apply_loc((r, 0)),
                apply_loc((int(round(r * math.cos(math.pi / 4))), int(round(r * math.sin(math.pi / 4))))),
                apply_loc((0, r)),
            ]
        else:
            reviewer_positions = [apply_loc(pos) for pos in itertools.product([0, self.max_x], repeat=self.dimensions) if any(c == self.max_x for c in pos)]
        reviewer = [ReviewerNode(position=pos, Logger=self.logger, classification=self.classification) for pos in reviewer_positions]

        self.segmentComponents = {
            'splitter': splitter,
            'reviewer': reviewer,
            'processing_nodes': []
        }

        full_grid    = int(self.max_x) ** self.dimensions
        usable_slots = max(1, full_grid - 2)
        num_nodes    = max(2, int(math.floor(usable_slots * self.density)))

        def calculate_node_positions(num_nodes, dimensions, max_x):
            reserved = set(map(tuple, reviewer_positions)) | {splitter_position}
            if dimensions == 2:
                # Polar grid across the quarter circle: n_r radial levels × n_theta angular steps
                n_r     = math.ceil(num_nodes ** 0.5)
                n_theta = math.ceil(num_nodes / n_r)
                seen, positions = set(reserved), []
                for i in range(n_r):
                    r = max_x * (i + 1) / (n_r + 1)
                    for j in range(n_theta):
                        theta = (math.pi / 2) * j / max(n_theta - 1, 1)
                        pos   = (int(round(r * math.cos(theta))), int(round(r * math.sin(theta))))
                        if pos not in seen:
                            seen.add(pos)
                            positions.append(pos)
                            if len(positions) >= num_nodes:
                                break
                    if len(positions) >= num_nodes:
                        break
                return positions, 0
            else:
                side     = math.ceil(num_nodes ** (1 / dimensions))
                axis     = [int(round(max_x * (k + 1) / (side + 1))) for k in range(side)]
                all_grid = list(itertools.product(axis, repeat=dimensions))
                positions, breaks = [], 0
                for pos in all_grid:
                    if len(positions) >= num_nodes:
                        break
                    if pos not in reserved:
                        positions.append(pos)
                    else:
                        breaks += 1
                return positions, breaks

        node_positions, breaks = calculate_node_positions(num_nodes, self.dimensions, self.max_x)
        node_positions = [apply_loc(pos) for pos in node_positions]
        self.display(f"Calculated {len(node_positions)} unique node positions with {breaks} breaks to avoid duplicates.", self.classification, True)

        def _create_nodes():
            if self.segmentComponents is None:
                raise ValueError("Segment not initialized. Call initializeSegment() first.")
            for pos in node_positions:
                processing_node = ProcessingNode(position=pos, Logger=self.logger, classification=self.classification)
                self.segmentComponents['processing_nodes'].append(processing_node)

        def _connect_nodes():
            if self.segmentComponents is None:
                raise ValueError("Segment not initialized. Call initializeSegment() first.")
            all_connectable_nodes = self.segmentComponents['processing_nodes'] + reviewer
            for node in self.segmentComponents['processing_nodes']:
                node.connect_nearest_nodes(all_connectable_nodes, self.connection_percentage)

        if has_progress:
            with self.logger.make_progress() as progress:
                node_task    = progress.add_task(f"Segment {self.segment_id} Creating Nodes",    total=len(node_positions))
                connect_task = progress.add_task(f"Segment {self.segment_id} Connecting Nodes",  total=len(node_positions))

                for pos in node_positions:
                    processing_node = ProcessingNode(position=pos, Logger=self.logger, classification=self.classification)
                    self.segmentComponents['processing_nodes'].append(processing_node)
                    progress.update(node_task, advance=1)

                all_connectable_nodes = self.segmentComponents['processing_nodes'] + reviewer
                for node in self.segmentComponents['processing_nodes']:
                    node.connect_nearest_nodes(all_connectable_nodes, self.connection_percentage)
                    progress.update(connect_task, advance=1)
        else:
            _create_nodes()
            _connect_nodes()

        # Connect splitter to processing nodes
        splitter.calculate_nearest_neighbors(self.segmentComponents['processing_nodes'])

        # Ensure every reviewer has at least one connected processing node
        for rev in self.segmentComponents['reviewer']:
            nodes_connected = [node for node in self.segmentComponents['processing_nodes'] if rev in node.connected_nodes]
            self.display(f"Segment {self.segment_id} - Reviewer {rev.position} connected to {len(nodes_connected)}/{len(self.segmentComponents['processing_nodes'])} processing nodes.", classification=4)
            if len(nodes_connected) == 0:
                self.display(f"Warning: Reviewer {rev.position} in segment {self.segment_id} has no connected processing nodes! Forcibly connecting", classification=2)
                rev.force_connect_nearest_nodes(self.segmentComponents['processing_nodes'], self.connection_percentage)
                nodes_connected = [node for node in self.segmentComponents['processing_nodes'] if rev in node.connected_nodes]
                self.display(f"After forcing, Reviewer {rev.position} connected to {len(nodes_connected)}/{len(self.segmentComponents['processing_nodes'])} processing nodes.", classification=4)

        # Save default graph immediately after initialization
        self.visualize_2d(save_path="default_segment_graph.png")
        self.display("Default segment graph saved → default_segment_graph.png", classification=4)

    def visualize_2d(self, save_path=None, epoch=None):
        """Render a matplotlib scatter plot of the segment, mirroring render_nexus() in OldMain."""
        if self.dimensions != 2:
            self.display("visualize_2d() is only supported for 2-dimensional segments.", classification=2)
            return
        if self.segmentComponents is None:
            self.display("Segment not initialized. Call initializeSegment() first.", classification=2)
            return

        self.display(f"Rendering Segment {self.segment_id} structure.", classification=4)

        splitter   = self.segmentComponents['splitter']
        reviewers  = self.segmentComponents['reviewer']
        processors = self.segmentComponents['processing_nodes']

        title = f"Segment {self.segment_id} Structure (2D)"
        if epoch is not None:
            title += f" — Epoch {epoch}"

        # When saving to a file use the non-interactive Agg backend (no Tk, no
        # thread conflicts with Rich progress bars).  When showing interactively
        # fall back to the normal pyplot path.
        if save_path:
            import os
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig = Figure(figsize=(8, 8))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))

        processor_labeled = False
        edge_labeled      = False
        reviewer_labeled  = False

        for node in processors:
            ax.scatter(
                node.position[0], node.position[1],
                color='green', s=10, alpha=0.7,
                label='Processing Node' if not processor_labeled else None,
                zorder=2
            )
            processor_labeled = True
            for target in node.connected_nodes:
                ax.plot(
                    [node.position[0], target.position[0]],
                    [node.position[1], target.position[1]],
                    color='grey', linewidth=0.5, alpha=0.5,
                    label='Connection' if not edge_labeled else None,
                    zorder=1
                )
                edge_labeled = True

        for target in splitter.connected_nodes:
            ax.plot(
                [splitter.position[0], target.position[0]],
                [splitter.position[1], target.position[1]],
                color='blue', linewidth=0.5, alpha=0.5, zorder=1
            )

        ax.scatter(splitter.position[0], splitter.position[1], color='blue', s=80, label='Splitter', zorder=3)
        for rev in reviewers:
            ax.scatter(rev.position[0], rev.position[1], color='red', s=80,
                       label='Reviewer' if not reviewer_labeled else None, zorder=3)
            reviewer_labeled = True

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend(loc='upper left', frameon=True)
        fig.tight_layout()

        if save_path:
            dirpart = os.path.dirname(save_path)
            if dirpart:
                os.makedirs(dirpart, exist_ok=True)
            fig.savefig(save_path)
        else:
            fig.savefig(f"segment_{self.segment_id}_structure.png")
            plt.show()

    def segmentInfer(self, pre_processed, loud=True):
        """
        Run inference for this segment.

        Parameters
        ----------
        pre_processed : output of PreProcessingNode.process_data()
        loud          : whether to emit display() calls

        Returns
        -------
        List of report dicts, one per reviewer that produced a valid prediction.
        Each dict is ready to unpack into HandlerNode.receive_report():
            {'id': segment_id, 'prediction': float, 'reviewer_position': tuple}
        """
        if self.segmentComponents is None:
            raise ValueError("Segment not initialized. Call initializeSegment() first.")

        has_progress = hasattr(self.logger, 'make_progress')

        splitter         = self.segmentComponents['splitter']
        reviewers        = self.segmentComponents['reviewer']
        processing_nodes = self.segmentComponents['processing_nodes']

        # Clear state from any previous inference
        for node in processing_nodes:
            node.clear_signals()
        for rev in reviewers:
            rev.signals = []

        def _run(bar=None):
            def _step(desc):
                if bar:
                    bar.set_description(desc)
                    bar.update(1)

            # --- Generate and dispatch signals ---
            _step(f"Segment {self.segment_id} Generating signals")
            signals = list(splitter.process(pre_processed, self.max_x))
            self.display(
                f"Generated {len(signals)} signals, Splitter has {len(splitter.connected_nodes)} connected nodes",
                classification=4, Loud=loud
            )
            for node, signal in zip(splitter.connected_nodes, signals):
                node.receive_signal(signal)
            self.display("Dispatched signals to processing nodes.", classification=4, Loud=loud)

            # --- Processing loop ---
            _step(f"Segment {self.segment_id} Processing signals")
            active         = True
            loop_iteration = 0
            max_iter       = 500
            num_signals    = len(signals)

            while active and loop_iteration < max_iter:
                for node in processing_nodes:
                    scaled_delta = node.process_signal()
                    if scaled_delta is not None:
                        ret = node.forward_signal()
                        if not ret:
                            self.display(f"Processing node at position {node.position} could not forward signal; it may be expired.", 1, Loud=False)
                            continue
                active       = False
                active_count = 0
                for signal in signals:
                    if signal.is_active() and not signal.collected:
                        active = True
                        active_count += 1
                if not signals:
                    active = False
                loop_iteration += 1
                if loop_iteration % 100 == 0 and bar:
                    bar.set_description(f"Segment {self.segment_id} Processing signals ({active_count}/{num_signals} active, iter {loop_iteration})")

            self.display(f"Signals processing complete after {loop_iteration} iterations.", classification=4, Loud=loud)

            # --- Collect reviewer reports ---
            _step(f"Segment {self.segment_id} Reviewing signals")
            reports = []
            for rev in reviewers:
                collected_pct = len([s for s in rev.signals if s.collected]) / len(rev.signals) if rev.signals else 0
                self.display(f"Reviewer {rev.position} signal collected %: {collected_pct:.2%}", classification=4, Loud=loud)
                prediction = rev.review_signals()
                if prediction is not None:
                    self.display(f"Reviewer {rev.position} prediction: {prediction}", classification=4, Loud=loud)
                    reports.append({'id': self.segment_id, 'prediction': prediction, 'reviewer_position': rev.position})

            return reports
        if self.logger is None:
            raise ValueError("Logger must be provided to run inference.")
        if has_progress:
            with self.logger.make_progress() as progress:
                task = progress.add_task(f"Segment {self.segment_id} Inference", total=3)
                from Components.RichConsole import ProgressBar
                return _run(ProgressBar(progress, task))
        else:
            return _run()


    # ------------------------------------------------------------------
    # Training hyperparameters  (mirror OldTrain.Trainer constants)
    # ------------------------------------------------------------------
    WEIGHT_LR                  = 0.01
    POSITION_LR                = 0.005
    SPLITTER_LR                = 0.01
    MAX_POSITION_STEP          = 2.0
    CONVERGENCE_THRESHOLD      = 1e-4
    LR_DECAY                   = 0.85   # multiply all LRs by this factor each epoch
    PLATEAU_THRESHOLD          = 3.0    # % — max test-error delta over 3 epochs to call plateau
    PLATEAU_POSITION_LR_SCALE  = 0.01   # scale lr_p by this factor once plateau detected

    # ------------------------------------------------------------------
    # Training-mode forward pass
    # ------------------------------------------------------------------

    def _forward_segment(self, pre_processed):
        """
        Training forward pass.  Mirrors segmentInfer but calls
        train_process_signal() so path_contributions are recorded.

        Returns (signals, reviewer_data) where reviewer_data is a list of
        (reviewer, signals_snapshot, prediction) — the snapshot is taken
        *before* review_signals() clears rev.signals so backprop can use it.
        """
        if self.segmentComponents is None:
            raise ValueError("Segment not initialized. Call initializeSegment() first.")
        sc               = self.segmentComponents
        splitter         = sc['splitter']
        reviewers        = sc['reviewer']
        processing_nodes = sc['processing_nodes']

        for node in processing_nodes:
            node.clear_signals()
        for rev in reviewers:
            rev.signals = []

        # Lazy weight initialisation
        for node in processing_nodes:
            if not node.weights:
                node.initialize_weights(pre_processed)

        signals = list(splitter.process(pre_processed, self.max_x))
        for node, signal in zip(splitter.connected_nodes, signals):
            node.receive_signal(signal)

        active         = True
        loop_iteration = 0
        while active and loop_iteration < 500:
            for node in processing_nodes:
                delta = node.train_process_signal()
                if delta is not None:
                    node.forward_signal()
            active = any(s.is_active() and not s.collected for s in signals)
            if not signals:
                active = False
            loop_iteration += 1

        # Snapshot each reviewer's collected signals before review_signals() clears them
        reviewer_data = []
        for rev in reviewers:
            rev_signals = list(rev.signals)
            prediction  = rev.review_signals()
            reviewer_data.append((rev, rev_signals, prediction))

        return signals, reviewer_data

    # ------------------------------------------------------------------
    # Loss & backprop
    # ------------------------------------------------------------------

    @staticmethod
    def _mse(prediction, target):
        if prediction is None:
            return None
        return (prediction - target) ** 2

    def _backprop(self, reviewer_data, target, mode):
        """Backpropagate through each reviewer's collected signals."""
        total_loss = 0.0
        count      = 0
        for _rev, rev_signals, prediction in reviewer_data:
            if prediction is None:
                continue
            dL_dseg   = 2.0 * (prediction - target)
            collected = [s for s in rev_signals if s.is_active()]
            if not collected:
                continue
            inv_vars = [1.0 / max(s.variance, 1e-9) for s in collected]
            total_w  = sum(inv_vars)
            if total_w == 0:
                continue
            rev_loss = 0.0
            for signal, w in zip(collected, inv_vars):
                dL_dpred_i = dL_dseg * (w / total_w)
                for contrib in signal.path_contributions.values():
                    node = contrib['node']
                    if mode == 'weights':
                        node.accumulate_weight_gradient(dL_dpred_i, signal)
                    else:
                        node.accumulate_position_gradient(dL_dpred_i, signal)
                loss_i = self._mse(signal.prediction, target)
                if loss_i is not None:
                    rev_loss += loss_i
            total_loss += rev_loss / len(collected)
            count      += 1
        return total_loss / count if count > 0 else None

    def _backprop_splitter(self, reviewer_data, target):
        """Backpropagate feature-relevance gradients to the splitter."""
        if self.segmentComponents is None:
            raise ValueError("Segment not initialized. Call initializeSegment() first.")
        splitter   = self.segmentComponents['splitter']
        total_loss = 0.0
        count      = 0
        for _rev, rev_signals, prediction in reviewer_data:
            if prediction is None:
                continue
            dL_dseg   = 2.0 * (prediction - target)
            collected = [s for s in rev_signals if s.is_active()]
            if not collected:
                continue
            inv_vars = [1.0 / max(s.variance, 1e-9) for s in collected]
            total_w  = sum(inv_vars)
            if total_w == 0:
                continue
            for signal, w in zip(collected, inv_vars):
                dL_dpred_i = dL_dseg * (w / total_w)
                splitter.accumulate_feature_relevance_gradient(dL_dpred_i, signal)
                loss_i = self._mse(signal.prediction, target)
                if loss_i is not None:
                    total_loss += loss_i
                    count      += 1
        return total_loss / count if count > 0 else None

    # ------------------------------------------------------------------
    # Gradient application
    # ------------------------------------------------------------------

    def _apply_and_reset(self, mode, lr):
        if self.segmentComponents is None:
            raise ValueError("Segment not initialized. Call initializeSegment() first.")
        sc               = self.segmentComponents
        processing_nodes = sc['processing_nodes']
        reviewers        = sc['reviewer']
        for node in processing_nodes:
            if mode == 'weights':
                node.apply_weight_gradient(lr)
            else:
                node.apply_position_gradient(lr, self.MAX_POSITION_STEP)
            node.reset_gradients()
        if mode == 'positions':
            node_list = processing_nodes + reviewers
            for node in processing_nodes:
                node.connected_nodes = []
                node.connect_nearest_nodes(node_list=node_list,
                                           connection_percentage=self.connection_percentage)

    # ------------------------------------------------------------------
    # Angular repulsion (post-epoch topology regulariser)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Single-epoch helpers
    # ------------------------------------------------------------------

    def _epoch(self, sample_list, lr_w, lr_p, lr_s, desc, freeze_connections=False):
        """
        One epoch: for each sample — single forward pass, then
          1. weight update  (lr_w)
          2. position update with arc-bounds enforcement (lr_p) + reconnect
             (reconnect skipped when freeze_connections=True)
          3. splitter feature-relevance update (lr_s)
        Returns avg weight-phase loss across the dataset.
        """
        if self.segmentComponents is None:
            raise ValueError("Segment not initialized. Call initializeSegment() first.")
        sc               = self.segmentComponents
        splitter         = sc['splitter']
        processing_nodes = sc['processing_nodes']
        reviewers        = sc['reviewer']
        node_list        = processing_nodes + reviewers
        total_loss       = 0.0
        count            = 0
        has_prog         = hasattr(self.logger, 'make_progress')

        def _run_samples(progress=None, task=None):
            nonlocal total_loss, count
            for sample in sample_list:
                target = sample.get(self.target)
                if target is None:
                    if progress:
                        progress.update(task, advance=1)
                    continue

                features = {k: v for k, v in sample.items() if k != self.target}
                _, reviewer_data = self._forward_segment(features)

                # 1. Weight update
                loss = self._backprop(reviewer_data, target, 'weights')
                for node in processing_nodes:
                    node.apply_weight_gradient(lr_w)
                    node.weight_gradients = {}

                # 2. Position update — bounds enforced inside apply_position_gradient
                self._backprop(reviewer_data, target, 'positions')
                for node in processing_nodes:
                    node.apply_position_gradient(lr_p, self.MAX_POSITION_STEP, self.max_x)
                    node.position_gradient = [0.0] * len(node.position)
                # Reconnect after positions shift (skipped during plateau)
                if not freeze_connections:
                    for node in processing_nodes:
                        node.connected_nodes = []
                        node.connect_nearest_nodes(node_list, self.connection_percentage)

                # 3. Splitter feature-relevance update
                splitter.reset_feature_relevance_gradients()
                self._backprop_splitter(reviewer_data, target)
                splitter.apply_feature_relevance_gradient(lr_s)

                if loss is not None:
                    total_loss += loss
                    count      += 1
                if progress:
                    progress.update(task, advance=1)

        if has_prog:
            if self.logger is None:
                raise ValueError("Logger must be provided to run epoch with progress.")
            with self.logger.make_progress(transient=True) as progress:
                task = progress.add_task(desc, total=len(sample_list))
                _run_samples(progress, task)
        else:
            _run_samples()

        # Post-epoch reconnect so topology reflects final positions (skipped during plateau).
        if not freeze_connections:
            for node in processing_nodes:
                node.connected_nodes = []
                node.connect_nearest_nodes(node_list, self.connection_percentage)

        return total_loss / count if count > 0 else float('inf')

    # ------------------------------------------------------------------
    # Test-set evaluation helpers
    # ------------------------------------------------------------------

    def _eval_test_predictions(self, test_list):
        """Return (predictions, actuals) lists for every test sample that
        produces at least one reviewer prediction.  Uses the training forward
        pass (no gradients applied) to avoid progress-bar overhead."""
        predictions, actuals = [], []
        for sample in test_list:
            target_val = sample.get(self.target)
            if target_val is None:
                continue
            features = {k: v for k, v in sample.items() if k != self.target}
            _, reviewer_data = self._forward_segment(features)
            preds = [pred for _, _, pred in reviewer_data if pred is not None]
            if not preds:
                continue
            predictions.append(sum(preds) / len(preds))
            actuals.append(float(target_val))
        return predictions, actuals

    def _run_final_test_eval(self, best_epoch, test_list, epoch_history=None,
                             best_test_err=None, n_train=0, epoch_count=0):
        """Restore best nexseg, run inference on full test split, display metrics, append to CSV."""
        import os, csv, datetime as _dt
        nexseg_file = f"segment_{self.segment_id}.nexseg"
        if best_epoch is None or not os.path.exists(nexseg_file):
            self.display("No best segment saved — skipping final test evaluation.", classification=3)
            return

        # Restore best segment components
        self.display(f"Restoring best segment (epoch {best_epoch}) from {nexseg_file}.", classification=4)
        restored = SegmentHandler.load_nexseg(
            nexseg_file, logger=self.logger,
            connection_percentage=self.connection_percentage,
            classification=self.classification
        )
        self.segmentComponents = restored.segmentComponents

        # Collect predictions on the full test split
        self.display(f"Evaluating on {len(test_list)} test samples…", classification=4)
        predictions, actuals = self._eval_test_predictions(test_list)
        n = len(predictions)
        if n == 0:
            self.display("No predictions produced on test split.", classification=3)
            return

        # Regression metrics
        mae  = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n)
        mean_a  = sum(actuals) / n
        ss_tot  = sum((a - mean_a) ** 2 for a in actuals)
        ss_res  = sum((p - a) ** 2 for p, a in zip(predictions, actuals))
        r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        mape    = sum(abs(p - a) / abs(a) * 100.0
                      for p, a in zip(predictions, actuals) if a != 0) / n

        # Within-threshold accuracy
        def _within(pct):
            return sum(1 for p, a in zip(predictions, actuals)
                       if a != 0 and abs(p - a) / abs(a) <= pct) / n * 100.0

        acc_5  = _within(0.05)
        acc_10 = _within(0.10)
        acc_20 = _within(0.20)

        # Direction-based precision / recall / F1 (median split)
        # Positive class = actual above median; correct = prediction also above median
        sorted_a = sorted(actuals)
        median_a = sorted_a[n // 2]
        actual_pos = [1 if a >= median_a else 0 for a in actuals]
        pred_pos   = [1 if p >= median_a else 0 for p in predictions]
        tp = sum(1 for ap, pp in zip(actual_pos, pred_pos) if ap == 1 and pp == 1)
        fp = sum(1 for ap, pp in zip(actual_pos, pred_pos) if ap == 0 and pp == 1)
        fn = sum(1 for ap, pp in zip(actual_pos, pred_pos) if ap == 1 and pp == 0)
        tn = sum(1 for ap, pp in zip(actual_pos, pred_pos) if ap == 0 and pp == 0)
        precision  = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall     = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1         = (2 * precision * recall / (precision + recall)
                      if not math.isnan(precision) and not math.isnan(recall)
                      and (precision + recall) > 0 else float('nan'))
        dir_acc    = (tp + tn) / n * 100.0

        # Train error at best epoch (from history)
        related_train_err = float('nan')
        if epoch_history:
            for entry in epoch_history:
                if entry['epoch'] == best_epoch:
                    valid = [v for v in entry['errors'].values() if not math.isnan(v)]
                    if valid:
                        related_train_err = sum(valid) / len(valid)
                    break

        console = getattr(self.logger, 'console', None)
        if console is not None:
            from rich.table import Table
            from rich import box
            from rich.rule import Rule
            console.print(Rule("[bold green]Final Test Evaluation (Best Segment)[/bold green]"))
            t = Table(
                title=f"Test Split Metrics  (n={n}, best epoch={best_epoch})",
                box=box.ROUNDED, border_style="green", show_lines=True
            )
            t.add_column("Metric",  style="bold white")
            t.add_column("Value",   style="bright_green", justify="right")
            t.add_row("MAE",                    f"{mae:.4f}")
            t.add_row("RMSE",                   f"{rmse:.4f}")
            t.add_row("R²",                     f"{r2:.4f}" if not math.isnan(r2) else "—")
            t.add_row("MAPE",                   f"{mape:.2f}%")
            t.add_section()
            t.add_row("Within  5% accuracy",    f"{acc_5:.1f}%")
            t.add_row("Within 10% accuracy",    f"{acc_10:.1f}%")
            t.add_row("Within 20% accuracy",    f"{acc_20:.1f}%")
            t.add_section()
            t.add_row("Direction accuracy",     f"{dir_acc:.1f}%")
            t.add_row("Precision (median)",     f"{precision:.4f}" if not math.isnan(precision) else "—")
            t.add_row("Recall    (median)",     f"{recall:.4f}"    if not math.isnan(recall)    else "—")
            t.add_row("F1        (median)",     f"{f1:.4f}"        if not math.isnan(f1)        else "—")
            console.print(t)
        else:
            self.display(
                f"Test Metrics | MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  "
                f"MAPE={mape:.2f}%  Acc@10%={acc_10:.1f}%  F1={f1:.4f}",
                classification=4
            )

        # ── Append run to error-epoch.csv ─────────────────────────────
        csv_path = "error-epoch.csv"
        fieldnames = [
            'timestamp', 'segment_id', 'target', 'max_x', 'dimensions',
            'epoch_count', 'n_train', 'n_test',
            'best_epoch', 'best_test_error_pct', 'related_train_error_pct',
            'mae', 'rmse', 'r2', 'mape_pct',
            'acc_5_pct', 'acc_10_pct', 'acc_20_pct',
            'direction_acc_pct', 'precision', 'recall', 'f1',
            'tp', 'fp', 'fn', 'tn',
            'connection_pct', 'density',
        ]
        row = {
            'timestamp':              _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'segment_id':             self.segment_id,
            'target':                 self.target,
            'max_x':                  self.max_x,
            'dimensions':             self.dimensions,
            'epoch_count':            epoch_count,
            'n_train':                n_train,
            'n_test':                 n,
            'best_epoch':             best_epoch,
            'best_test_error_pct':    f"{best_test_err:.4f}" if best_test_err is not None and not math.isnan(best_test_err) else '',
            'related_train_error_pct':f"{related_train_err:.4f}" if not math.isnan(related_train_err) else '',
            'mae':                    f"{mae:.6f}",
            'rmse':                   f"{rmse:.6f}",
            'r2':                     f"{r2:.6f}"  if not math.isnan(r2)        else '',
            'mape_pct':               f"{mape:.4f}",
            'acc_5_pct':              f"{acc_5:.2f}",
            'acc_10_pct':             f"{acc_10:.2f}",
            'acc_20_pct':             f"{acc_20:.2f}",
            'direction_acc_pct':      f"{dir_acc:.2f}",
            'precision':              f"{precision:.6f}" if not math.isnan(precision) else '',
            'recall':                 f"{recall:.6f}"    if not math.isnan(recall)    else '',
            'f1':                     f"{f1:.6f}"        if not math.isnan(f1)        else '',
            'tp':                     tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'connection_pct':         self.connection_percentage,
            'density':                self.density,
        }
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.display(f"Run metrics appended to {csv_path}.", classification=4)

        # Store for SystemHandler final report
        self.best_epoch_metrics = {
            'segment_id': self.segment_id,
            'best_epoch': best_epoch,
            'n_train': n_train,
            'n_test': n,
            'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
            'acc_5': acc_5, 'acc_10': acc_10, 'acc_20': acc_20,
            'dir_acc': dir_acc,
            'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }

    def _eval_avg_error(self, test_list):
        """Average % error across all samples in test_list using inference forward pass."""
        errors = []
        for sample in test_list:
            target_val = sample.get(self.target)
            if target_val is None or target_val == 0:
                continue
            features = {k: v for k, v in sample.items() if k != self.target}
            _, reviewer_data = self._forward_segment(features)
            preds = [pred for _, _, pred in reviewer_data if pred is not None]
            if not preds:
                continue
            avg_pred = sum(preds) / len(preds)
            errors.append(abs(avg_pred - target_val) / abs(target_val) * 100.0)
        return sum(errors) / len(errors) if errors else float('nan')

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save_nexseg(self, filename=None):
        if self.segmentComponents is None:
            raise ValueError("Segment not initialized. Call initializeSegment() first.")
        """Serialise current segment state (weights, positions, connections) to a .nexseg JSON file."""
        import json
        if filename is None:
            filename = f"segment_{self.segment_id}.nexseg"
        sc = self.segmentComponents
        state = {
            'segment_id': self.segment_id,
            'max_x': self.max_x,
            'dimensions': self.dimensions,
            'target': self.target,
            'splitter': {
                'position': list(sc['splitter'].position),
                'signal_weights': sc['splitter'].signal_weights,
            },
            'reviewers': [list(rev.position) for rev in sc['reviewer']],
            'processing_nodes': [
                {
                    'position': list(node.position),
                    'weights': node.weights,
                    'connected_positions': [list(n.position) for n in node.connected_nodes],
                }
                for node in sc['processing_nodes']
            ],
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        return filename

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune(self):
        if self.segmentComponents is None:
            raise ValueError("Segment not initialized. Call initializeSegment() first.")
        """Remove processing nodes that were never activated during training."""
        sc               = self.segmentComponents
        processing_nodes = sc['processing_nodes']
        before    = len(processing_nodes)
        surviving = [n for n in processing_nodes if n.activation_count > 0]
        pruned_set = set(processing_nodes) - set(surviving)
        sc['processing_nodes'] = surviving
        if pruned_set:
            for node in surviving:
                node.connected_nodes = [n for n in node.connected_nodes if n not in pruned_set]
        pruned = before - len(surviving)
        if pruned > 0:
            self.display(f"Pruned {pruned}/{before} unused processing nodes.", classification=4)

    # ------------------------------------------------------------------
    # Public training entry point
    # ------------------------------------------------------------------

    def train(self, dataset, epoch_count=3, preprocessor=None):
        """
        Train the segment on a dataset.

        Parameters
        ----------
        dataset      : pd.DataFrame or path to a CSV file
        epoch_count  : maximum number of training epochs
        preprocessor : optional shared PreProcesingNode (already fitted on the
                       full dataset). When provided the segment uses the same
                       vocabulary as inference, preventing encoding mismatches.
                       If None a new preprocessor is created and fitted locally.
        """
        import os
        import datetime as _dt
        import pandas as pd
        from Components.PreProcessingNode import PreProcesingNode

        if self.segmentComponents is None:
            raise ValueError("Segment not initialized. Call initializeSegment() first.")

        sc = self.segmentComponents

        # ── Run folder for per-epoch structure graphs ─────────────────
        run_tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        seg_dir = os.path.join("runs", run_tag, f"segment_{self.segment_id}")
        os.makedirs(seg_dir, exist_ok=True)

        # ── Step 1: Pre-process dataset ───────────────────────────────
        self.display("Training Step 1: Processing dataset.", classification=4)
        if isinstance(dataset, str):
            dataset = pd.read_csv(dataset)
        if preprocessor is None:
            preprocessor = PreProcesingNode(Logger=self.logger, logger_classification=self.classification)
            processed_df = preprocessor.process_dataset(dataset.copy())
        else:
            # Reuse shared vocabulary but avoid accumulating rows into the
            # global preprocessor's data list (process_dataset appends to
            # self.data and returns ALL rows ever seen, not just this batch).
            _saved_data = preprocessor.data
            preprocessor.data = []
            processed_df = preprocessor.process_dataset(dataset.copy())
            preprocessor.data = _saved_data
        all_samples  = processed_df.to_dict(orient='records')

        # 80/20 train/test split
        split       = max(1, int(len(all_samples) * 0.8))
        sample_list = all_samples[:split]
        test_list   = all_samples[split:]
        self.display(
            f"  {len(all_samples)} samples → train={len(sample_list)}, test={len(test_list)}.",
            classification=4
        )

        # ── Step 2: Initialise weights ────────────────────────────────
        self.display("Training Step 2: Initialising node weights.", classification=4)
        if sample_list:
            init_sample = sample_list[0]
            for node in sc['processing_nodes']:
                if not node.weights:
                    node.initialize_weights(init_sample)
            splitter = sc['splitter']
            if not splitter.signal_weights:
                splitter.initialize_signal_weights(init_sample)
        self.display("  Node and splitter weights initialised.", classification=4)

        # ── Step 3: Per-epoch W→P→S training with LR decay ───────────
        self.display(
            f"Training Step 3: weight→position training with LR decay "
            f"for up to {epoch_count} epoch(s).",
            classification=4
        )
        # Eval sample taken from test split so it was never seen during training
        eval_sample = test_list[0]
        eval_actual = eval_sample.get(self.target)
        eval_pre    = {k: v for k, v in eval_sample.items() if k != self.target}

        prev_loss          = float('inf')
        has_prog           = hasattr(self.logger, 'make_progress')
        epoch_history      = []   # [{epoch, loss, errors, test_avg_error, plateau}]
        best_epoch         = None
        best_test_err      = float('inf')

        def _run_epochs(progress=None, epoch_task=None):
            nonlocal prev_loss, best_epoch, best_test_err
            for epoch in range(epoch_count):
                decay = self.LR_DECAY ** epoch
                lr_w  = self.WEIGHT_LR   * decay
                lr_s  = self.SPLITTER_LR * decay

                # ── Plateau detection over last 3 test errors ─────────
                plateau = False
                if len(epoch_history) >= 3:
                    last3 = [e['test_avg_error'] for e in epoch_history[-3:]
                             if not math.isnan(e.get('test_avg_error', float('nan')))]
                    if len(last3) == 3:
                        plateau = (max(last3) - min(last3)) < self.PLATEAU_THRESHOLD

                lr_p = self.POSITION_LR * decay * (
                    self.PLATEAU_POSITION_LR_SCALE if plateau else 1.0
                )
                tag  = f"Epoch {epoch + 1}/{epoch_count}"

                self.display(
                    f"{tag} | decay={decay:.4f}  lr_w={lr_w:.5f}  "
                    f"lr_p={lr_p:.6f}  lr_s={lr_s:.5f}"
                    + ("  [PLATEAU — connections frozen]" if plateau else ""),
                    classification=4
                )

                loss = self._epoch(sample_list, lr_w, lr_p, lr_s, tag,
                                   freeze_connections=plateau)
                delta = abs(prev_loss - loss)
                self.display(
                    f"{tag} | Loss: {loss:.6f}  delta: {delta:.2e}",
                    classification=4
                )

                # ── Epoch-end accuracy check ──────────────────────────
                reports = self.segmentInfer(eval_pre.copy(), loud=False)
                epoch_errs = {
                    r['reviewer_position']:
                        abs(r['prediction'] - eval_actual) / abs(eval_actual) * 100.0
                        if eval_actual else float('nan')
                    for r in reports
                }
                # ── Test-set average error ────────────────────────────
                test_avg_err = self._eval_avg_error(test_list)

                # ── Per-epoch structure graph ─────────────────────────
                if self.dimensions == 2:
                    graph_path = os.path.join(seg_dir, f"epoch_{epoch + 1}.png")
                    self.visualize_2d(save_path=graph_path, epoch=epoch + 1)

                epoch_history.append({
                    'epoch': epoch + 1, 'loss': loss,
                    'errors': epoch_errs, 'test_avg_error': test_avg_err,
                    'plateau': plateau
                })

                # ── Track best by lowest test average error, save when improved ─
                valid_errs  = [v for v in epoch_errs.values() if not math.isnan(v)]
                avg_err     = sum(valid_errs) / len(valid_errs) if valid_errs else float('nan')
                if not math.isnan(test_avg_err) and test_avg_err < best_test_err:
                    best_test_err = test_avg_err
                    best_epoch    = epoch + 1
                    nexseg_path   = self._save_nexseg()
                    self.display(
                        f"{tag} | New best — train={avg_err:.2f}%  test={test_avg_err:.2f}%  "
                        f"saved → {nexseg_path}",
                        classification=4
                    )

                if not reports:
                    self.display(f"{tag} | Eval: no reviewer predictions.", classification=3)
                else:
                    console = getattr(self.logger, 'console', None)
                    if console is not None:
                        from rich.table import Table
                        from rich import box
                        eval_table = Table(
                            title=f"{tag} — Eval Inference",
                            box=box.ROUNDED, border_style="cyan", show_lines=True
                        )
                        eval_table.add_column("Reviewer",   style="bold white")
                        eval_table.add_column("Prediction", style="bright_green", justify="right")
                        eval_table.add_column("Actual",     style="bold white",   justify="right")
                        eval_table.add_column("Error %",    style="red",          justify="right")
                        for r in reports:
                            pred = r['prediction']
                            err  = (f"{abs(pred - eval_actual) / abs(eval_actual) * 100:.2f}%"
                                    if eval_actual else "—")
                            eval_table.add_row(
                                str(r['reviewer_position']),
                                f"{pred:.4f}",
                                f"{eval_actual:.4f}",
                                err,
                            )
                        console.print(eval_table)
                    else:
                        for r in reports:
                            pred = r['prediction']
                            err  = (abs(pred - eval_actual) / abs(eval_actual) * 100.0
                                    if eval_actual else float('nan'))
                            self.display(
                                f"{tag} | Reviewer {r['reviewer_position']}  "
                                f"pred={pred:.4f}  actual={eval_actual:.4f}  err={err:.2f}%",
                                classification=4
                            )

                if progress:
                    progress.update(epoch_task, advance=1)

                if delta < self.CONVERGENCE_THRESHOLD:
                    self.display(
                        f"Converged at epoch {epoch + 1} (delta={delta:.2e}). Stopping early.",
                        classification=4
                    )
                    break
                prev_loss = loss

        if has_prog:
            if self.logger is None:
                raise ValueError("Logger must be provided to run training with progress bars.")
            with self.logger.make_progress() as progress:
                epoch_task = progress.add_task("Training Epochs", total=epoch_count)
                _run_epochs(progress, epoch_task)
        else:
            _run_epochs()

        # ── Step 4: Prune unused nodes ────────────────────────────────
        self.display("Training Step 4: Pruning unused processing nodes.", classification=4)
        self._prune()

        # ── Final best-epoch summary ──────────────────────────────────
        if best_epoch is not None:
            self.display(
                f"Training complete. Best epoch: {best_epoch}  |  "
                f"Best test error: {best_test_err:.2f}%  "
                f"|  Saved → segment_{self.segment_id}.nexseg",
                classification=4
            )
        else:
            self.display("Training complete. No valid eval predictions were produced.", classification=3)

        # ── Step 5: Restore best segment & evaluate on full test split ─
        self._run_final_test_eval(
            best_epoch, test_list,
            epoch_history=epoch_history,
            best_test_err=best_test_err,
            n_train=len(sample_list),
            epoch_count=epoch_count,
        )

        return epoch_history

    # ------------------------------------------------------------------
    # Load a saved .nexseg file
    # ------------------------------------------------------------------

    @classmethod
    def load_nexseg(cls, filename, logger, connection_percentage=0.08, classification=1):
        """
        Restore a SegmentHandler from a .nexseg JSON file saved by _save_nexseg().

        Parameters
        ----------
        filename             : path to the .nexseg file
        logger               : logger instance to attach to all nodes
        connection_percentage: forwarded to SplitterNode (not stored in file)
        classification       : log classification level

        Returns
        -------
        A fully wired SegmentHandler whose segmentComponents are ready for inference.
        """
        import json

        with open(filename) as f:
            state = json.load(f)

        handler = cls(
            maxX=state['max_x'],
            target=state['target'],
            logger=logger,
            dimensions=state['dimensions'],
            segment_id=state['segment_id'],
            connection_percentage=connection_percentage,
            classification=classification,
        )

        # Reviewers
        reviewers = [
            ReviewerNode(position=tuple(pos), Logger=logger, classification=classification)
            for pos in state['reviewers']
        ]

        # Processing nodes — weights restored, connections wired below
        pos_to_node = {}
        processing_nodes = []
        for nd in state['processing_nodes']:
            node = ProcessingNode(
                position=tuple(nd['position']),
                Logger=logger,
                classification=classification,
            )
            node.weights = nd['weights']
            pos_to_node[tuple(nd['position'])] = node
            processing_nodes.append(node)

        # Splitter
        spl = state['splitter']
        splitter = SplitterNode(
            position=tuple(spl['position']),
            connection_percentage=connection_percentage,
            Logger=logger,
            classification=classification,
        )
        splitter.signal_weights = spl['signal_weights']

        # Restore processing-node connections by position lookup
        rev_pos_map = {tuple(rev.position): rev for rev in reviewers}
        all_by_pos  = {**pos_to_node, **rev_pos_map}
        for nd_data, node in zip(state['processing_nodes'], processing_nodes):
            for cpos in nd_data['connected_positions']:
                target_node = all_by_pos.get(tuple(cpos))
                if target_node is not None:
                    node.connected_nodes.append(target_node)

        # Splitter → nearest processing nodes
        splitter.calculate_nearest_neighbors(processing_nodes)

        handler.segmentComponents = {
            'splitter':          splitter,
            'reviewer':          reviewers,
            'processing_nodes':  processing_nodes,
        }

        handler.display(
            f"Loaded segment {handler.segment_id} from '{filename}' "
            f"({len(processing_nodes)} nodes, {len(reviewers)} reviewers).",
            classification=4
        )
        return handler


if __name__ == "__main__":
    import os
    import sys
    import time
    import pandas as pd
    sys.path.insert(0, ".")
    import Components.RichConsole as RichConsole
    from Components.PreProcessingNode import PreProcesingNode
    from rich.table import Table
    from rich.rule import Rule
    from rich import box
    from rich.prompt import Prompt

    logger = RichConsole.RichLogger(f"segment_demo_{int(time.time())}.log", log_level=4)
    logger.console.print(Rule("[bold cyan]Segment Handler Demo[/bold cyan]"))

    NEXSEG_FILE = "segment_0.nexseg"
    DATASET_CSV = "Exam_Score_Prediction.csv"
    TARGET_COL  = "exam_score"

    # ── Load dataset & shared preprocessor (needed in both paths) ────
    dataset    = pd.read_csv(DATASET_CSV)
    sample_raw = dataset.iloc[0].to_dict()
    actual     = sample_raw.get(TARGET_COL)
    logger.log(f"Evaluation sample loaded. Actual {TARGET_COL}: {actual}", 4, True)

    preprocessor = PreProcesingNode(Logger=logger, logger_classification=4)
    _ = preprocessor.process_dataset(dataset.copy())   # warm up vocabulary

    epoch_history = []
    pre_reports   = []

    # ── Decision: load saved segment or train from scratch ───────────
    if os.path.exists(NEXSEG_FILE):
        logger.console.print(f"\n[bold yellow]Found saved segment:[/bold yellow] {NEXSEG_FILE}")
        choice = Prompt.ask(
            "  Load saved segment or train a new one?",
            choices=["load", "train"],
            default="load"
        )
    else:
        logger.console.print(f"\n[dim]No saved segment found ({NEXSEG_FILE}).[/dim]")
        choice = "train"

    if choice == "load":
        # ── Load from .nexseg ─────────────────────────────────────────
        logger.console.print(Rule("[bold green]Loading Saved Segment[/bold green]"))
        handler = SegmentHandler.load_nexseg(
            NEXSEG_FILE, logger=logger,
            connection_percentage=0.1, classification=4
        )
    else:
        # ── Build & initialise segment ────────────────────────────────
        logger.console.print(Rule("[bold cyan]Initialising New Segment[/bold cyan]"))
        handler = SegmentHandler(
            maxX=10, target=TARGET_COL, logger=logger,
            connection_percentage=0.1, density=0.8,
            dimensions=2, classification=4, segment_id=0
        )
        handler.initializeSegment()

        # ── Pre-training inference ────────────────────────────────────
        logger.console.print(Rule("[bold yellow]Pre-Training Inference[/bold yellow]"))
        pre_input   = preprocessor.process_data(sample_raw.copy())
        pre_reports = handler.segmentInfer(pre_input, loud=True)
        logger.log(f"Pre-training: {len(pre_reports)} reviewer report(s).", 4, True)

        pre_table = Table(title="Pre-Training Reviewer Predictions",
                          box=box.ROUNDED, border_style="yellow", show_lines=True)
        pre_table.add_column("Reviewer Position", style="bold white")
        pre_table.add_column("Prediction",        style="yellow",     justify="right")
        pre_table.add_column("Actual",            style="bold white", justify="right")
        pre_table.add_column("Error %",           style="red",        justify="right")
        for r in pre_reports:
            pred = r['prediction']
            err  = f"{abs(pred - actual) / abs(actual) * 100:.2f}%" if actual else "—"
            pre_table.add_row(str(r['reviewer_position']), f"{pred:.4f}", f"{actual:.4f}", err)
        logger.console.print(pre_table)

        # ── Train ─────────────────────────────────────────────────────
        logger.console.print(Rule("[bold green]Training[/bold green]"))
        epoch_history = handler.train(dataset, epoch_count=20)

    import math as _math

    # ── Best epoch summary (training path only) ───────────────────────
    if epoch_history:
        best_e = min(
            epoch_history,
            key=lambda e: (
                sum(v for v in e['errors'].values() if not _math.isnan(v)) /
                max(1, sum(1 for v in e['errors'].values() if not _math.isnan(v)))
            )
        )
        best_train_vals = [v for v in best_e['errors'].values() if not _math.isnan(v)]
        best_train_avg  = sum(best_train_vals) / len(best_train_vals) if best_train_vals else float('nan')
        best_test_avg   = best_e.get('test_avg_error', float('nan'))

        logger.console.print(Rule("[bold magenta]Best Epoch Summary[/bold magenta]"))
        best_table = Table(box=box.ROUNDED, border_style="magenta", show_lines=True)
        best_table.add_column("Metric",    style="bold white")
        best_table.add_column("Value",     style="bright_magenta", justify="right")
        best_table.add_row("Best Epoch",          str(best_e['epoch']))
        best_table.add_row("Training Loss",       f"{best_e['loss']:.6f}")
        best_table.add_row("Train Avg Error %",   f"{best_train_avg:.2f}%" if not _math.isnan(best_train_avg) else "—")
        best_table.add_row("Test Avg Error %",    f"{best_test_avg:.2f}%"  if not _math.isnan(best_test_avg)  else "—")
        logger.console.print(best_table)

    # ── Post-training inference ───────────────────────────────────────
    logger.console.print(Rule("[bold yellow]Post-Training Inference[/bold yellow]"))
    post_input   = preprocessor.process_data(sample_raw.copy())
    post_reports = handler.segmentInfer(post_input, loud=True)
    logger.log(f"Post-training: {len(post_reports)} reviewer report(s).", 4, True)

    # ── Accuracy comparison table ─────────────────────────────────────
    logger.console.print(Rule("[bold cyan]Accuracy Comparison[/bold cyan]"))

    pre_map  = {r['reviewer_position']: r['prediction'] for r in pre_reports}
    post_map = {r['reviewer_position']: r['prediction'] for r in post_reports}
    all_pos  = sorted(set(pre_map) | set(post_map), key=lambda p: (p[0], p[1]))

    cmp_table = Table(
        title=f"Reviewer Predictions vs Actual ({actual})",
        box=box.ROUNDED, border_style="cyan", show_lines=True
    )
    cmp_table.add_column("Reviewer",       style="bold white",   no_wrap=True)
    cmp_table.add_column("Pre-Train Pred", style="yellow",       justify="right")
    cmp_table.add_column("Post-Train Pred",style="bright_green", justify="right")
    cmp_table.add_column("Actual",         style="bold white",   justify="right")
    cmp_table.add_column("Pre Err %",      style="red",          justify="right")
    cmp_table.add_column("Post Err %",     style="green",        justify="right")

    def _fmt(val):
        return f"{val:.4f}" if val is not None else "—"

    def _err_pct(pred):
        if pred is None or actual is None or actual == 0:
            return float('nan')
        return abs(pred - actual) / abs(actual) * 100.0

    def _err(pred):
        v = _err_pct(pred)
        return f"{v:.2f}%" if not _math.isnan(v) else "—"

    pre_errs, post_errs = [], []
    for pos in all_pos:
        pre_pred  = pre_map.get(pos)
        post_pred = post_map.get(pos)
        cmp_table.add_row(
            str(pos),
            _fmt(pre_pred), _fmt(post_pred), _fmt(actual),
            _err(pre_pred), _err(post_pred),
        )
        pv = _err_pct(pre_pred);  post_v = _err_pct(post_pred)
        if not _math.isnan(pv):   pre_errs.append(pv)
        if not _math.isnan(post_v): post_errs.append(post_v)

    # Summary row
    pre_avg_str  = f"{sum(pre_errs)  / len(pre_errs):.2f}%"  if pre_errs  else "—"
    post_avg_str = f"{sum(post_errs) / len(post_errs):.2f}%" if post_errs else "—"
    cmp_table.add_section()
    cmp_table.add_row(
        "[bold]Average[/bold]", "—", "—", _fmt(actual),
        f"[red]{pre_avg_str}[/red]", f"[green]{post_avg_str}[/green]"
    )

    logger.console.print(cmp_table)

    # ── Epoch error graph ─────────────────────────────────────────────
    if epoch_history:
        import matplotlib.pyplot as plt

        epochs       = [e['epoch'] for e in epoch_history]
        all_positions = sorted({pos for e in epoch_history for pos in e['errors']})

        fig, ax = plt.subplots(figsize=(10, 6))
        for pos in all_positions:
            errs = [e['errors'].get(pos, float('nan')) for e in epoch_history]
            ax.plot(epochs, errs, marker='o', label=f"Reviewer {pos}")

        # Train avg error (across reviewers) and test avg error
        avg_errs = []
        for e in epoch_history:
            vals = [v for v in e['errors'].values() if not _math.isnan(v)]
            avg_errs.append(sum(vals) / len(vals) if vals else float('nan'))
        ax.plot(epochs, avg_errs, marker='s', linestyle='--', linewidth=2,
                color='black', label='Train Avg')

        test_errs = [e.get('test_avg_error', float('nan')) for e in epoch_history]
        ax.plot(epochs, test_errs, marker='^', linestyle=':', linewidth=2,
                color='red', label='Test Avg')

        # Shade plateau epochs
        for e in epoch_history:
            if e.get('plateau'):
                ax.axvspan(e['epoch'] - 0.5, e['epoch'] + 0.5,
                           color='lightblue', alpha=0.3,
                           label='Plateau (frozen)' if e['epoch'] == next(
                               x['epoch'] for x in epoch_history if x.get('plateau')
                           ) else None)

        ax.set_title("Epoch vs Error %")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error %")
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("epoch_error.png")
        plt.show()

        # ── Train vs Test avg error graph ─────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(epochs, avg_errs, marker='s', linewidth=2, color='steelblue', label='Train Avg Error %')
        ax2.plot(epochs, test_errs, marker='^', linewidth=2, color='tomato',   label='Test Avg Error %')

        # Mark best epoch (lowest test avg)
        valid_test = [(i, v) for i, v in enumerate(test_errs) if not _math.isnan(v)]
        if valid_test:
            best_i, best_v = min(valid_test, key=lambda x: x[1])
            ax2.axvline(x=epochs[best_i], color='gold', linestyle='--', linewidth=1.5,
                        label=f"Best Epoch ({epochs[best_i]})")
            ax2.annotate(f"{best_v:.1f}%", xy=(epochs[best_i], best_v),
                         xytext=(8, 8), textcoords='offset points', fontsize=9, color='tomato')

        ax2.set_title("Train vs Test Average Error % per Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Avg Error %")
        ax2.set_xticks(epochs)
        ax2.legend()
        ax2.grid(True, alpha=0.4)
        fig2.tight_layout()
        fig2.savefig("epoch_train_test_error.png")
        plt.show()

    # ── Visualise segment structure ───────────────────────────────────
    if handler.dimensions == 2:
        handler.visualize_2d()