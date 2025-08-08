from brainNexus import BrainNexus
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import time
import random
import copy
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    spatial_learning_rate: float = 0.01
    connection_threshold: float = 0.1
    spatial_pressure: float = 0.05
    exploration_rate: float = 0.1
    batch_size: int = 8  # Reduced for better error tolerance
    max_epochs: int = 100
    convergence_threshold: float = 0.001
    rl_discount_factor: float = 0.95
    rl_exploration_decay: float = 0.995
    error_tolerance: float = 0.8  # Allow 80% of batch to fail and still continue

@dataclass
class SpatialGradient:
    """Gradient information for spatial optimization."""
    node_id: int
    position_gradient: np.ndarray
    magnitude: float
    confidence: float

class BrainNexusLearn(BrainNexus):
    def __init__(self, demo: bool = False, config: Optional[TrainingConfig] = None, output_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the learning-capable BrainNexus with spatial optimization.
        
        Args:
            demo: Whether to run in demo mode
            config: Training configuration parameters
            output_config: Output system configuration
        """
        super().__init__(demo=demo, output_config=output_config)
        
        # Training configuration
        self.config = config or TrainingConfig()
        
        # Learning state
        self.training_mode = False
        self.epoch = 0
        self.loss_history = []
        self.spatial_efficiency_history = []
        
        # Gradient tracking
        self.weight_gradients = defaultdict(dict)  # {node_id: {connection_id: gradient}}
        self.spatial_gradients = []  # List of SpatialGradient objects
        self.momentum_weights = defaultdict(float)  # {(from_node, to_node): momentum}
        self.momentum_positions = defaultdict(lambda: np.zeros(self.dimensions))
        
        # Reinforcement learning state
        self.rl_mode = False
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.q_values = defaultdict(float)
        self.state_actions = defaultdict(list)  # Store actual actions per state
        self.state_action_counts = defaultdict(int)
        
        # Spatial optimization tracking
        self.node_performance_history = defaultdict(list)
        self.connection_strength_history = defaultdict(list)
        self.spatial_clustering_targets = {}
        
        # Training metrics
        self.training_stats = {
            'supervised_accuracy': [],
            'spatial_efficiency': [],
            'connection_optimization': [],
            'reinforcement_rewards': [],
            'convergence_rate': []
        }
        
        if self.demo:
            print("🎓 BrainNexusLearn initialized with spatial learning capabilities")
            print(f"   Learning rate: {self.config.learning_rate}")
            print(f"   Spatial learning rate: {self.config.spatial_learning_rate}")
            print(f"   Batch size: {self.config.batch_size}")

    def set_training_mode(self, training: bool = True):
        """Enable or disable training mode."""
        self.training_mode = training
        if self.demo:
            mode = "TRAINING" if training else "INFERENCE"
            print(f"🎯 Mode set to: {mode}")

    def supervised_train(self, training_data: List[Tuple[Any, Any]], 
                        validation_data: Optional[List[Tuple[Any, Any]]] = None) -> Dict[str, Any]:
        """
        Train the BrainNexus using supervised learning with spatial optimization.
        
        Args:
            training_data: List of (input, target) pairs
            validation_data: Optional validation dataset
            
        Returns:
            Dict containing training statistics and final model state
        """
        self.set_training_mode(True)
        start_time = time.time()
        
        if self.demo:
            print(f"🎓 Starting supervised training with {len(training_data)} samples")
            print(f"   Target epochs: {self.config.max_epochs}")
            print(f"   Batch size: {self.config.batch_size}")
        
        best_loss = float('inf')
        best_spatial_efficiency = 0.0
        patience_counter = 0
        convergence_threshold = self.config.convergence_threshold
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            batch_count = 0
            
            # Shuffle training data
            shuffled_data = random.sample(training_data, len(training_data))
            
            # Process in batches
            for i in range(0, len(shuffled_data), self.config.batch_size):
                batch = shuffled_data[i:i + self.config.batch_size]
                
                # Forward pass and loss calculation
                batch_loss, batch_accuracy = self._process_supervised_batch(batch)
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                batch_count += 1
                
                # Backward pass - update weights and positions
                self._backward_pass_supervised()
                
                # Spatial optimization
                self._optimize_spatial_structure()
                
                # Connection optimization
                self._optimize_connections()
            
            # Calculate epoch averages
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            avg_accuracy = epoch_accuracy / batch_count if batch_count > 0 else 0.0
            
            # Calculate spatial efficiency
            spatial_efficiency = self._calculate_spatial_efficiency()
            
            # Update histories
            self.loss_history.append(avg_loss)
            self.spatial_efficiency_history.append(spatial_efficiency)
            self.training_stats['supervised_accuracy'].append(avg_accuracy)
            self.training_stats['spatial_efficiency'].append(spatial_efficiency)
            
            # Validation if provided
            val_loss, val_accuracy = 0.0, 0.0
            if validation_data:
                val_loss, val_accuracy = self._validate(validation_data)
            
            # Progress reporting
            if self.demo and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.config.max_epochs}:")
                print(f"    Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.3f}")
                print(f"    Spatial Efficiency: {spatial_efficiency:.3f}")
                if validation_data:
                    print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.3f}")
            
            # Early stopping check
            if avg_loss < best_loss - convergence_threshold:
                best_loss = avg_loss
                best_spatial_efficiency = spatial_efficiency
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:  # Early stopping
                    if self.demo:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
        
        training_time = time.time() - start_time
        self.set_training_mode(False)
        
        if self.demo:
            print(f"✅ Supervised training completed in {training_time:.2f}s")
            print(f"   Final loss: {best_loss:.4f}")
            print(f"   Best spatial efficiency: {best_spatial_efficiency:.3f}")
            print(f"   Epochs completed: {epoch + 1}")
        
        return {
            'training_time': training_time,
            'final_loss': best_loss,
            'final_accuracy': avg_accuracy,
            'spatial_efficiency': best_spatial_efficiency,
            'epochs_completed': epoch + 1,
            'loss_history': self.loss_history.copy(),
            'training_stats': copy.deepcopy(self.training_stats)
        }

    def _process_supervised_batch(self, batch: List[Tuple[Any, Any]]) -> Tuple[float, float]:
        """
        Process a batch of supervised training data.
        
        Args:
            batch: List of (input, target) pairs
            
        Returns:
            Tuple of (batch_loss, batch_accuracy)
        """
        total_loss = 0.0
        total_accuracy = 0.0
        
        # Clear gradients
        self.weight_gradients.clear()
        self.spatial_gradients.clear()
        
        for input_data, target in batch:
            # Forward pass
            result = self.run(input_data, trace_execution=True)
            prediction = result['result']
            
            # Calculate loss and gradients
            loss = self._calculate_loss(prediction, target)
            accuracy = self._calculate_accuracy(prediction, target)
            
            total_loss += loss
            total_accuracy += accuracy
            
            # Compute gradients
            self._compute_gradients(result, target)
        
        batch_size = len(batch)
        return total_loss / batch_size, total_accuracy / batch_size

    def _calculate_loss(self, prediction: Dict[str, Any], target: Any) -> float:
        """Calculate loss between prediction and target."""
        try:
            # Handle case where prediction might be malformed
            if not isinstance(prediction, dict):
                return 1.0  # High loss for invalid prediction
            
            pred_probs = prediction.get('probabilities', [0.5, 0.5])
            
            # Ensure pred_probs is a valid list/array
            if not isinstance(pred_probs, (list, tuple, np.ndarray)):
                pred_probs = [0.5, 0.5]
            
            # Convert to tensor safely
            pred_tensor = torch.tensor(pred_probs, dtype=torch.float32)
            
            # Convert target to one-hot if needed
            if isinstance(target, int):
                target_size = len(pred_probs) if len(pred_probs) > 0 else 2
                target_tensor = torch.zeros(target_size, dtype=torch.float32)
                if 0 <= target < target_size:
                    target_tensor[target] = 1.0
            else:
                target_tensor = torch.tensor(target, dtype=torch.float32)
            
            # Ensure tensors have same size
            if pred_tensor.size() != target_tensor.size():
                return 1.0  # High loss for size mismatch
            
            # Cross-entropy loss with error handling
            loss = F.cross_entropy(pred_tensor.unsqueeze(0), target_tensor.unsqueeze(0))
            return float(loss.item())
            
        except Exception as e:
            if hasattr(self, 'demo') and self.demo:
                print(f"  ⚠️  Loss calculation error: {e}")
            return 1.0  # Default high loss

    def _calculate_accuracy(self, prediction: Dict[str, Any], target: Any) -> float:
        """Calculate accuracy between prediction and target."""
        try:
            # Handle case where prediction might be malformed
            if not isinstance(prediction, dict):
                return 0.0  # Zero accuracy for invalid prediction
            
            pred_class = prediction.get('prediction', 0)
            
            # Ensure pred_class is a valid number
            if not isinstance(pred_class, (int, float)):
                pred_class = 0
            
            if isinstance(target, int):
                return 1.0 if int(pred_class) == target else 0.0
            else:
                try:
                    target_tensor = torch.tensor(target)
                    target_class = target_tensor.argmax().item()
                    return 1.0 if int(pred_class) == target_class else 0.0
                except Exception:
                    return 0.0  # Default to zero accuracy if target conversion fails
                    
        except Exception as e:
            if hasattr(self, 'demo') and self.demo:
                print(f"  ⚠️  Accuracy calculation error: {e}")
            return 0.0  # Default zero accuracy

    def _compute_gradients(self, result: Dict[str, Any], target: Any):
        """
        Compute gradients for weights and spatial positions.
        
        Args:
            result: Forward pass result with trace
            target: Target output
        """
        trace = result.get('trace', {})
        if not trace:
            return
        
        # Get prediction error
        prediction = result['result']
        error = self._get_prediction_error(prediction, target)
        
        # Compute weight gradients using backpropagation through the traced execution
        node_activations = trace.get('node_activations', {})
        connection_flows = trace.get('connection_flows', [])
        
        # Backpropagate through the network
        node_errors = {}
        
        # Start with handler error
        handler_ids = self.get_nodes_by_type('Handler')
        if handler_ids:
            node_errors[handler_ids[0]] = error
        
        # Propagate backwards through reviewers
        reviewer_ids = self.get_nodes_by_type('Reviewer')
        for reviewer_id in reviewer_ids:
            if reviewer_id in node_activations:
                # Gradient from handler
                handler_connections = [flow for flow in connection_flows 
                                     if flow['from'] == reviewer_id]
                reviewer_error = sum(flow['weight'] * node_errors.get(flow['to'], 0.0) 
                                   for flow in handler_connections)
                node_errors[reviewer_id] = reviewer_error
                
                # Compute weight gradients for reviewer connections
                for flow in handler_connections:
                    self._update_weight_gradient(reviewer_id, flow['to'], reviewer_error)
        
        # Continue backpropagation through retainers, computational nodes, etc.
        self._backpropagate_through_network(node_errors, connection_flows)
        
        # Compute spatial gradients
        self._compute_spatial_gradients(node_errors, node_activations)

    def _get_prediction_error(self, prediction: Dict[str, Any], target: Any) -> float:
        """Calculate prediction error for backpropagation."""
        pred_probs = prediction.get('probabilities', [0.5, 0.5])
        confidence = prediction.get('confidence', 0.5)
        
        if isinstance(target, int):
            target_prob = pred_probs[target] if target < len(pred_probs) else 0.0
            error = target_prob - 1.0  # We want target class to have probability 1.0
        else:
            # For vector targets
            target_tensor = torch.tensor(target)
            pred_tensor = torch.tensor(pred_probs)
            error = F.mse_loss(pred_tensor, target_tensor).item()
        
        return error

    def _backpropagate_through_network(self, node_errors: Dict[int, float], 
                                     connection_flows: List[Dict]):
        """Backpropagate errors through the entire network."""
        # Process each layer backwards
        layer_types = ['Retainer', 'Computational', 'Splitter', 'Judge', 'Controller']
        
        for layer_type in layer_types:
            layer_nodes = self.get_nodes_by_type(layer_type)
            
            for node_id in layer_nodes:
                if node_id not in node_errors:
                    # Calculate error from downstream connections
                    outgoing_flows = [flow for flow in connection_flows 
                                    if flow['from'] == node_id]
                    
                    total_error = 0.0
                    for flow in outgoing_flows:
                        downstream_error = node_errors.get(flow['to'], 0.0)
                        total_error += flow['weight'] * downstream_error
                        
                        # Update weight gradient
                        self._update_weight_gradient(node_id, flow['to'], downstream_error)
                    
                    node_errors[node_id] = total_error

    def _update_weight_gradient(self, from_node: int, to_node: int, error: float):
        """Update weight gradient for a connection."""
        if from_node not in self.weight_gradients:
            self.weight_gradients[from_node] = {}
        
        current_weight = self.get_connection_weight(from_node, to_node) or 0.5
        gradient = error * self.config.learning_rate
        
        self.weight_gradients[from_node][to_node] = gradient

    def _compute_spatial_gradients(self, node_errors: Dict[int, float], 
                                 node_activations: Dict[int, Any]):
        """Compute gradients for spatial node positions."""
        self.spatial_gradients.clear()
        
        for node_id, error in node_errors.items():
            if abs(error) < 1e-6:  # Skip negligible errors
                continue
                
            node = self.node_registry.get(node_id)
            if not node:
                continue
            
            # Calculate spatial gradient based on local neighborhood performance
            position_gradient = self._calculate_position_gradient(node_id, error)
            
            if np.linalg.norm(position_gradient) > 0:
                spatial_grad = SpatialGradient(
                    node_id=node_id,
                    position_gradient=position_gradient,
                    magnitude=float(np.linalg.norm(position_gradient)),
                    confidence=min(abs(error), 1.0)
                )
                self.spatial_gradients.append(spatial_grad)

    def _calculate_position_gradient(self, node_id: int, error: float) -> np.ndarray:
        """
        Calculate spatial position gradient for a node.
        
        Args:
            node_id: ID of the node
            error: Backpropagated error
            
        Returns:
            Position gradient vector
        """
        node = self.node_registry[node_id]
        current_pos = np.array(node.node_position)
        gradient = np.zeros(self.dimensions)
        
        # Get connected nodes
        connections = self.get_node_connections(node_id)
        all_connected = connections['incoming'] + connections['outgoing'] + connections['bidirectional']
        
        if not all_connected:
            return gradient
        
        # Calculate gradient based on connection strengths and spatial distances
        for connected_id in all_connected:
            connected_node = self.node_registry.get(connected_id)
            if not connected_node:
                continue
            
            connected_pos = np.array(connected_node.node_position)
            distance_vector = connected_pos - current_pos
            distance = np.linalg.norm(distance_vector)
            
            if distance > 0:
                # Get connection weight
                weight = self.get_connection_weight(node_id, connected_id) or \
                        self.get_connection_weight(connected_id, node_id) or 0.5
                
                # Gradient encourages movement toward strongly connected nodes
                # and away from weakly connected nodes
                direction = distance_vector / distance
                strength_factor = (weight - 0.5) * 2  # Range [-1, 1]
                gradient += strength_factor * direction * abs(error) * self.config.spatial_pressure
        
        return gradient

    def _backward_pass_supervised(self):
        """Apply computed gradients to update weights."""
        for from_node, connections in self.weight_gradients.items():
            for to_node, gradient in connections.items():
                current_weight = self.get_connection_weight(from_node, to_node) or 0.5
                
                # Apply momentum
                momentum_key = (from_node, to_node)
                if momentum_key not in self.momentum_weights:
                    self.momentum_weights[momentum_key] = 0.0
                
                momentum = self.config.momentum * self.momentum_weights[momentum_key]
                weight_update = gradient + momentum
                
                # Update weight with regularization
                new_weight = current_weight - weight_update
                new_weight -= self.config.weight_decay * current_weight  # L2 regularization
                new_weight = np.clip(new_weight, 0.0, 1.0)  # Keep weights in valid range
                
                # Apply update
                self.update_connection_weight(from_node, to_node, new_weight)
                self.momentum_weights[momentum_key] = weight_update

    def _optimize_spatial_structure(self):
        """Optimize spatial positions of nodes based on gradients."""
        for spatial_grad in self.spatial_gradients:
            try:
                node_id = spatial_grad.node_id
                gradient = spatial_grad.position_gradient
                
                # Apply momentum to position updates
                momentum = self.config.momentum * self.momentum_positions[node_id]
                position_update = self.config.spatial_learning_rate * gradient + momentum
                
                # Get current position
                node = self.node_registry.get(node_id)
                if not node or not hasattr(node, 'node_position'):
                    continue
                    
                current_pos = np.array(node.node_position)
                
                # Calculate new position
                new_position = current_pos + position_update
                
                # Apply spatial constraints (keep nodes within reasonable bounds)
                max_bound = 10000.0
                new_position = np.clip(new_position, -max_bound, max_bound)
                
                # Update node position
                self.move_node(node_id, new_position.tolist())
                self.momentum_positions[node_id] = position_update
                
            except Exception as e:
                if self.demo:
                    print(f"  ⚠️  Error optimizing spatial position for node {spatial_grad.node_id}: {e}")
                continue

    def _optimize_connections(self):
        """Optimize network connections by adding/removing based on performance."""
        # Get computational nodes for connection optimization
        comp_nodes = self.get_nodes_by_type('Computational')
        
        # Track connection performance
        for node_id in comp_nodes:
            try:
                node_performance = self._evaluate_node_performance(node_id)
                self.node_performance_history[node_id].append(node_performance)
                
                # Only optimize if we have enough history
                if len(self.node_performance_history[node_id]) < 5:
                    continue
                
                # Check if node is underperforming
                recent_performance = np.mean(self.node_performance_history[node_id][-5:])
                
                if recent_performance < 0.3:  # Poor performance threshold
                    self._strengthen_connections(node_id)
                elif recent_performance > 0.8:  # Good performance
                    self._optimize_weak_connections(node_id)
                    
            except Exception as e:
                if self.demo:
                    print(f"  ⚠️  Skipping connection optimization for node {node_id}: {e}")
                continue

    def _evaluate_node_performance(self, node_id: int) -> float:
        """Evaluate performance of a node based on recent usage and gradients."""
        node = self.node_registry.get(node_id)
        if not node:
            return 0.0
        
        # Base performance on usage frequency
        usage_score = min(getattr(node, 'times_called', 0) / 100.0, 1.0)
        
        # Factor in gradient magnitude (how much the node contributes to learning)
        gradient_score = 0.0
        for spatial_grad in self.spatial_gradients:
            if spatial_grad.node_id == node_id:
                gradient_score = min(spatial_grad.magnitude, 1.0)
                break
        
        # Combine scores
        performance = 0.6 * usage_score + 0.4 * gradient_score
        return performance

    def _strengthen_connections(self, node_id: int):
        """Strengthen connections for underperforming nodes."""
        try:
            connections = self.get_node_connections(node_id)
            
            # Find nearby nodes to connect to
            node = self.node_registry[node_id]
            if not hasattr(node, 'node_position'):
                return
                
            nearby_nodes = self.get_nodes_in_radius(node.node_position, 1000.0)
            
            # Add connections to nearby high-performing nodes
            for nearby_id in nearby_nodes[:5]:  # Limit to 5 new connections
                if (nearby_id != node_id and 
                    nearby_id not in connections['outgoing'] and
                    nearby_id not in connections['incoming']):
                    
                    # Check performance of nearby node
                    try:
                        nearby_performance = self._evaluate_node_performance(nearby_id)
                        if nearby_performance > 0.5:
                            weight = 0.3 + 0.4 * nearby_performance  # Weight based on performance
                            self.connect_nodes(node_id, nearby_id, weight=weight)
                    except Exception as e:
                        if self.demo:
                            print(f"  ⚠️  Error evaluating nearby node {nearby_id}: {e}")
                        continue
                        
        except Exception as e:
            if self.demo:
                print(f"  ⚠️  Error strengthening connections for node {node_id}: {e}")

    def _optimize_weak_connections(self, node_id: int):
        """Remove or weaken poor connections for well-performing nodes."""
        try:
            connections = self.get_node_connections(node_id)
            
            # Check outgoing connections
            for connected_id in connections['outgoing']:
                try:
                    current_weight = self.get_connection_weight(node_id, connected_id)
                    if current_weight and current_weight < self.config.connection_threshold:
                        # Remove weak connections
                        self.disconnect_nodes(node_id, connected_id)
                except Exception as e:
                    if self.demo:
                        print(f"  ⚠️  Error optimizing connection {node_id}->{connected_id}: {e}")
                    continue
                    
        except Exception as e:
            if self.demo:
                print(f"  ⚠️  Error optimizing weak connections for node {node_id}: {e}")

    def _calculate_spatial_efficiency(self) -> float:
        """Calculate the spatial efficiency of the current node arrangement."""
        if len(self.neural_nodes) == 0:
            return 0.0
        
        total_efficiency = 0.0
        connection_count = 0
        
        # Check efficiency of all connections
        for node in self.neural_nodes:
            try:
                # Check if node has required attributes
                if not hasattr(node, 'node_id') or not hasattr(node, 'node_position'):
                    continue
                    
                node_id = node.node_id
                connections = self.get_node_connections(node_id)
                
                for connected_id in connections['outgoing'] + connections['bidirectional']:
                    connected_node = self.node_registry.get(connected_id)
                    if not connected_node or not hasattr(connected_node, 'node_position'):
                        continue
                    
                    # Calculate spatial distance
                    try:
                        distance = np.linalg.norm(
                            np.array(node.node_position) - np.array(connected_node.node_position)
                        )
                        
                        # Get connection weight
                        weight = self.get_connection_weight(node_id, connected_id) or 0.5
                        
                        # Efficiency: stronger connections should have shorter distances
                        if distance > 0:
                            efficiency = weight / (1.0 + distance / 1000.0)  # Normalize distance
                            total_efficiency += efficiency
                            connection_count += 1
                    except (TypeError, ValueError, AttributeError) as e:
                        if self.demo:
                            print(f"  ⚠️  Skipping spatial efficiency calculation for connection {node_id}->{connected_id}: {e}")
                        continue
                        
            except Exception as e:
                if self.demo:
                    print(f"  ⚠️  Skipping node in spatial efficiency calculation: {e}")
                continue
        
        return float(total_efficiency / connection_count) if connection_count > 0 else 0.0

    def _validate(self, validation_data: List[Tuple[Any, Any]]) -> Tuple[float, float]:
        """Validate model on validation dataset."""
        self.set_training_mode(False)
        
        total_loss = 0.0
        total_accuracy = 0.0
        valid_samples = 0
        
        for input_data, target in validation_data:
            try:
                result = self.run(input_data, trace_execution=False)
                prediction = result['result']
                
                loss = self._calculate_loss(prediction, target)
                accuracy = self._calculate_accuracy(prediction, target)
                
                total_loss += loss
                total_accuracy += accuracy
                valid_samples += 1
                
            except (TypeError, AttributeError, ValueError) as e:
                if self.demo:
                    print(f"  ⚠️  Skipping validation item due to error: {e}")
                # Skip this sample but continue with validation
                continue
                
            except Exception as e:
                if self.demo:
                    print(f"  ❌ Unexpected error in validation: {e}")
                # Use default poor performance for failed samples
                total_loss += 1.0
                total_accuracy += 0.0
                valid_samples += 1
        
        self.set_training_mode(True)
        
        if valid_samples > 0:
            return total_loss / valid_samples, total_accuracy / valid_samples
        else:
            return 1.0, 0.0  # Poor default performance if all validation failed

    # ===============================================================================
    # REINFORCEMENT LEARNING METHODS
    # ===============================================================================

    def enable_reinforcement_learning(self, mode: bool = True):
        """Enable or disable reinforcement learning mode."""
        self.rl_mode = mode
        if self.demo:
            state = "ENABLED" if mode else "DISABLED"
            print(f"🎮 Reinforcement Learning: {state}")

    def reinforcement_train(self, environment_fn, episodes: int = 1000, 
                          max_steps_per_episode: int = 200) -> Dict[str, Any]:
        """
        Train using reinforcement learning with Q-learning variants.
        
        Args:
            environment_fn: Function that returns (state, reward, done, info) given action
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Dict containing RL training statistics
        """
        self.enable_reinforcement_learning(True)
        start_time = time.time()
        
        if self.demo:
            print(f"🎮 Starting reinforcement learning for {episodes} episodes")
            print(f"   Exploration rate: {self.config.exploration_rate}")
            print(f"   Discount factor: {self.config.rl_discount_factor}")
        
        episode_rewards = []
        episode_lengths = []
        failed_episodes = 0
        
        for episode in range(episodes):
            total_reward = 0.0
            steps = 0
            
            try:
                # Reset environment
                reset_result = environment_fn(action=None, reset=True)
                
                # Validate environment response
                if not isinstance(reset_result, dict):
                    if self.demo:
                        print(f"  ⚠️  Environment reset returned invalid state: {type(reset_result)}")
                    # Use default state
                    state = {'efficiency': 0.5, 'performance': 0.5}
                else:
                    state = reset_result
                
                for step in range(max_steps_per_episode):
                    # Choose action using epsilon-greedy strategy
                    action = self._choose_rl_action(state)
                    
                    # Execute action in environment and on network
                    action_success = self.execute_rl_action(action)
                    
                    # Get environment response
                    try:
                        env_response = environment_fn(action=action)
                        
                        # Validate environment response
                        if isinstance(env_response, tuple) and len(env_response) == 4:
                            next_state, reward, done, info = env_response
                        else:
                            if self.demo:
                                print(f"  ⚠️  Environment returned invalid response: {type(env_response)}")
                            # Use default values
                            next_state = state
                            reward = -0.1
                            done = True
                            info = {}
                            
                        # Validate state
                        if not isinstance(next_state, dict):
                            if self.demo:
                                print(f"  ⚠️  Environment returned invalid next_state: {type(next_state)}")
                            next_state = state  # Keep current state
                            
                    except Exception as e:
                        if self.demo:
                            print(f"  ⚠️  Environment error at episode {episode}, step {step}: {e}")
                        reward = -0.1  # Small penalty for environment errors
                        next_state = state  # Keep same state
                        done = True  # End episode early
                        info = {}
                    
                    # Adjust reward based on action success
                    if not action_success:
                        reward -= 0.05  # Small penalty for failed actions
                    
                    # Update Q-values
                    self._update_q_values(state, action, reward, next_state, done)
                    
                    # Track performance
                    total_reward += reward
                    steps += 1
                    
                    # Store experience
                    self.action_history.append((state, action, reward, next_state, done))
                    self.reward_history.append(reward)
                    
                    if done:
                        break
                    
                    state = next_state
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
            except Exception as e:
                if self.demo:
                    print(f"  ❌ Episode {episode} failed: {e}")
                failed_episodes += 1
                # Add poor performance for failed episodes
                episode_rewards.append(-1.0)
                episode_lengths.append(1)
            
            # Decay exploration rate
            self.config.exploration_rate *= self.config.rl_exploration_decay
            self.config.exploration_rate = max(self.config.exploration_rate, 0.01)
            
            # Spatial adaptation based on episode performance
            if episode % 50 == 0 and len(episode_rewards) >= 50:
                try:
                    self._adapt_spatial_structure_rl(episode_rewards[-50:])
                except Exception as e:
                    if self.demo:
                        print(f"  ⚠️  Spatial adaptation failed: {e}")
            
            # Progress reporting
            if self.demo and (episode + 1) % 100 == 0:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                print(f"  Episode {episode + 1}/{episodes}:")
                print(f"    Avg Reward (last {len(recent_rewards)}): {avg_reward:.3f}")
                print(f"    Avg Length (last {len(recent_lengths)}): {avg_length:.1f}")
                print(f"    Failed episodes: {failed_episodes}")
                print(f"    Exploration Rate: {self.config.exploration_rate:.3f}")
        
        training_time = time.time() - start_time
        self.enable_reinforcement_learning(False)
        
        # Update training stats
        self.training_stats['reinforcement_rewards'].extend(episode_rewards)
        
        if self.demo:
            final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"✅ RL training completed in {training_time:.2f}s")
            print(f"   Final average reward: {final_avg_reward:.3f}")
            print(f"   Total episodes: {episodes}")
            print(f"   Failed episodes: {failed_episodes}")
        
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        return {
            'training_time': training_time,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_exploration_rate': self.config.exploration_rate,
            'q_table_size': len(self.q_values),
            'avg_final_reward': final_avg_reward,
            'failed_episodes': failed_episodes
        }

    def _choose_rl_action(self, state: Any) -> Dict[str, Any]:
        """
        Choose action using epsilon-greedy strategy.
        
        Args:
            state: Current environment state
            
        Returns:
            Action dictionary with neural network configuration changes
        """
        try:
            state_key = self._state_to_key(state)
            
            if random.random() < self.config.exploration_rate:
                # Exploration: random action
                action = self._generate_random_action()
            else:
                # Exploitation: best known action
                action = self._get_best_action(state_key)
            
            # Validate action is a dictionary
            if not isinstance(action, dict):
                if self.demo:
                    print(f"  ⚠️  Invalid action type returned: {type(action)}, generating random action")
                action = self._generate_random_action()
            
            return action
            
        except Exception as e:
            if self.demo:
                print(f"  ⚠️  Error choosing RL action: {e}, using no_action")
            return {'type': 'no_action'}

    def _generate_random_action(self) -> Dict[str, Any]:
        """Generate a random action for exploration."""
        action_type = random.choice(['move_node', 'adjust_weight', 'add_connection', 'remove_connection'])
        
        comp_nodes = self.get_nodes_by_type('Computational')
        if not comp_nodes:
            return {'type': 'no_action'}
        
        node_id = random.choice(comp_nodes)
        
        if action_type == 'move_node':
            # Random position change
            current_pos = np.array(self.node_registry[node_id].node_position)
            offset = np.random.normal(0, 100, size=self.dimensions)
            new_pos = current_pos + offset
            return {
                'type': 'move_node',
                'node_id': node_id,
                'new_position': new_pos.tolist()
            }
        
        elif action_type == 'adjust_weight':
            connections = self.get_node_connections(node_id)
            if connections['outgoing']:
                target_id = random.choice(connections['outgoing'])
                weight_change = random.uniform(-0.1, 0.1)
                return {
                    'type': 'adjust_weight',
                    'from_node': node_id,
                    'to_node': target_id,
                    'weight_change': weight_change
                }
        
        elif action_type == 'add_connection':
            # Find nearby nodes to connect to
            nearby_nodes = self.get_nodes_in_radius(
                self.node_registry[node_id].node_position, 1000.0
            )
            if nearby_nodes:
                target_id = random.choice(nearby_nodes)
                if target_id != node_id:
                    return {
                        'type': 'add_connection',
                        'from_node': node_id,
                        'to_node': target_id,
                        'weight': random.uniform(0.1, 0.9)
                    }
        
        elif action_type == 'remove_connection':
            connections = self.get_node_connections(node_id)
            if connections['outgoing']:
                target_id = random.choice(connections['outgoing'])
                return {
                    'type': 'remove_connection',
                    'from_node': node_id,
                    'to_node': target_id
                }
        
        return {'type': 'no_action'}

    def _get_best_action(self, state_key: str) -> Dict[str, Any]:
        """Get the best known action for a given state."""
        best_action = None
        best_q_value = float('-inf')
        
        # Check stored actions for this state
        stored_actions = self.state_actions.get(state_key, [])
        
        for action in stored_actions:
            action_key = self._action_to_key(action)
            q_value = self.q_values.get((state_key, action_key), 0.0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        if best_action is None:
            # No known good action, generate random
            return self._generate_random_action()
        
        return best_action

    def _update_q_values(self, state: Any, action: Dict[str, Any], reward: float,
                        next_state: Any, done: bool):
        """Update Q-values using Q-learning."""
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)
        
        # Store the action for this state if not already stored
        if action not in self.state_actions[state_key]:
            self.state_actions[state_key].append(action.copy())
        
        # Current Q-value
        current_q = self.q_values.get((state_key, action_key), 0.0)
        
        # Best Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            # Find max Q-value for next state
            max_next_q = 0.0
            next_actions = self.state_actions.get(next_state_key, [])
            for next_action in next_actions:
                next_action_key = self._action_to_key(next_action)
                q_val = self.q_values.get((next_state_key, next_action_key), 0.0)
                max_next_q = max(max_next_q, q_val)
        
        # Q-learning update
        learning_rate = self.config.learning_rate
        discount = self.config.rl_discount_factor
        
        new_q = current_q + learning_rate * (reward + discount * max_next_q - current_q)
        self.q_values[(state_key, action_key)] = new_q
        
        # Track state-action counts for exploration
        self.state_action_counts[(state_key, action_key)] += 1

    def _state_to_key(self, state: Any) -> str:
        """Convert state to a hashable key."""
        try:
            if isinstance(state, dict):
                # Sort keys for consistent hashing
                items = sorted(state.items())
                return str(items)  # Use string representation instead of hash
            elif isinstance(state, (list, tuple)):
                return str(tuple(state))
            else:
                return str(state)
        except Exception as e:
            # Fallback for any problematic states
            return f"state_error_{id(state)}"

    def _action_to_key(self, action: Dict[str, Any]) -> str:
        """Convert action to a hashable key."""
        try:
            if not isinstance(action, dict):
                return str(action)
            # Sort items for consistent hashing, handle unhashable values
            safe_items = []
            for k, v in sorted(action.items()):
                if isinstance(v, (list, dict)):
                    safe_items.append((k, str(v)))
                else:
                    safe_items.append((k, v))
            return str(safe_items)
        except Exception as e:
            # Fallback for any problematic actions
            return f"action_error_{id(action)}"

    def _adapt_spatial_structure_rl(self, recent_rewards: List[float]):
        """Adapt spatial structure based on RL performance."""
        if len(recent_rewards) < 10:
            return
        
        avg_reward = np.mean(recent_rewards)
        reward_trend = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5])
        
        if self.demo:
            print(f"    RL Spatial Adaptation: avg_reward={avg_reward:.3f}, trend={reward_trend:.3f}")
        
        # If performance is declining, encourage more exploration
        if reward_trend < -0.1:
            self._increase_spatial_exploration()
        elif reward_trend > 0.1:
            self._consolidate_spatial_structure()

    def _increase_spatial_exploration(self):
        """Increase spatial exploration by moving underperforming nodes."""
        comp_nodes = self.get_nodes_by_type('Computational')
        
        # Move 10% of computational nodes to random positions
        nodes_to_move = random.sample(comp_nodes, max(1, len(comp_nodes) // 10))
        
        for node_id in nodes_to_move:
            current_pos = np.array(self.node_registry[node_id].node_position)
            random_offset = np.random.normal(0, 500, size=self.dimensions)
            new_pos = current_pos + random_offset
            self.move_node(node_id, new_pos.tolist())

    def _consolidate_spatial_structure(self):
        """Consolidate spatial structure by clustering high-performing nodes."""
        comp_nodes = self.get_nodes_by_type('Computational')
        
        # Find high-performing nodes (those with many recent activations)
        high_performers = []
        for node_id in comp_nodes:
            node = self.node_registry[node_id]
            if getattr(node, 'times_called', 0) > 10:
                high_performers.append(node_id)
        
        if len(high_performers) < 2:
            return
        
        # Move average performers closer to high performers
        for node_id in comp_nodes:
            if node_id not in high_performers:
                node = self.node_registry[node_id]
                current_pos = np.array(node.node_position)
                
                # Find nearest high performer
                min_distance = float('inf')
                nearest_performer = None
                
                for hp_id in high_performers:
                    hp_pos = np.array(self.node_registry[hp_id].node_position)
                    distance = np.linalg.norm(current_pos - hp_pos)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_performer = hp_id
                
                if nearest_performer:
                    # Move 20% closer to nearest high performer
                    target_pos = np.array(self.node_registry[nearest_performer].node_position)
                    move_vector = (target_pos - current_pos) * 0.2
                    new_pos = current_pos + move_vector
                    self.move_node(node_id, new_pos.tolist())

    # ===============================================================================
    # UTILITY AND ANALYSIS METHODS
    # ===============================================================================

    def move_node(self, node_id: int, new_position: List[float]) -> bool:
        """
        Override move_node to handle spatial optimization properly.
        
        Args:
            node_id: ID of the node to move
            new_position: New position as [x, y, z]
            
        Returns:
            True if move was successful
        """
        try:
            # Get the node
            node = self.node_registry.get(node_id)
            if not node:
                if self.demo:
                    print(f"  ⚠️  Node {node_id} not found for movement")
                return False
            
            # Ensure new_position is valid
            if not isinstance(new_position, (list, tuple)) or len(new_position) != self.dimensions:
                if self.demo:
                    print(f"  ⚠️  Invalid position format for node {node_id}")
                return False
            
            # Convert to float and validate
            try:
                new_pos = [float(x) for x in new_position]
            except (ValueError, TypeError):
                if self.demo:
                    print(f"  ⚠️  Invalid position values for node {node_id}")
                return False
            
            # Update node object
            node.node_position = new_pos
            
            # Calculate spatial affinity safely
            try:
                spatial_affinity = self._calculate_spatial_affinity(new_pos)
                node.spatial_affinity = spatial_affinity
            except Exception as e:
                if self.demo:
                    print(f"  ⚠️  Error calculating spatial affinity: {e}")
                spatial_affinity = 0.5  # Default value
                node.spatial_affinity = spatial_affinity
            
            # Update DataFrame record safely
            try:
                mask = self.node_records['Node_ID'] == node_id
                if mask.any():
                    # Update position - convert to string representation for DataFrame storage
                    self.node_records.loc[mask, 'Node_Position'] = str(new_pos)
                    self.node_records.loc[mask, 'Spatial_Affinity'] = spatial_affinity
            except Exception as e:
                if self.demo:
                    print(f"  ⚠️  Error updating node records: {e}")
                # Continue anyway - the node object is updated
            
            # Rebuild spatial index safely
            try:
                self._rebuild_spatial_index()
            except Exception as e:
                if self.demo:
                    print(f"  ⚠️  Error rebuilding spatial index: {e}")
                # Continue anyway
            
            return True
            
        except Exception as e:
            if self.demo:
                print(f"  ❌ Failed to move node {node_id}: {e}")
            return False

    def execute_rl_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a reinforcement learning action on the network.
        
        Args:
            action: Action dictionary
            
        Returns:
            True if action was successfully executed
        """
        try:
            # Validate action is a dictionary
            if not isinstance(action, dict):
                if self.demo:
                    print(f"Action execution failed: action is not a dict: {type(action)}")
                return False
                
            action_type = action.get('type', 'no_action')
            
            if action_type == 'move_node':
                node_id = action.get('node_id')
                new_position = action.get('new_position')
                if node_id is None or new_position is None:
                    return False
                return self.move_node(node_id, new_position)
            
            elif action_type == 'adjust_weight':
                from_node = action.get('from_node')
                to_node = action.get('to_node')
                weight_change = action.get('weight_change')
                if from_node is None or to_node is None or weight_change is None:
                    return False
                
                current_weight = self.get_connection_weight(from_node, to_node) or 0.5
                new_weight = np.clip(current_weight + weight_change, 0.0, 1.0)
                return self.update_connection_weight(from_node, to_node, new_weight)
            
            elif action_type == 'add_connection':
                from_node = action.get('from_node')
                to_node = action.get('to_node')
                weight = action.get('weight')
                if from_node is None or to_node is None or weight is None:
                    return False
                return self.connect_nodes(from_node, to_node, weight)
            
            elif action_type == 'remove_connection':
                from_node = action.get('from_node')
                to_node = action.get('to_node')
                if from_node is None or to_node is None:
                    return False
                return self.disconnect_nodes(from_node, to_node)
            
            elif action_type == 'no_action':
                return True
            
            return False
            
        except Exception as e:
            if self.demo:
                print(f"Action execution failed: {e}")
            return False

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary and statistics."""
        summary = {
            'training_mode': self.training_mode,
            'rl_mode': self.rl_mode,
            'current_epoch': self.epoch,
            'loss_history': self.loss_history.copy(),
            'spatial_efficiency_history': self.spatial_efficiency_history.copy(),
            'training_stats': copy.deepcopy(self.training_stats),
            'network_stats': {
                'total_nodes': len(self.neural_nodes),
                'total_connections': sum(len(self.get_node_connections(node.node_id)['outgoing']) 
                                       for node in self.neural_nodes),
                'reuse_candidates': len(self.reuse_candidates),
                'spatial_efficiency': self._calculate_spatial_efficiency()
            },
            'rl_stats': {
                'q_table_size': len(self.q_values),
                'total_experiences': len(self.action_history),
                'exploration_rate': self.config.exploration_rate
            } if self.rl_mode else {}
        }
        
        return summary

    def save_model_state(self, filepath: str):
        """Save the current model state for later loading."""
        state = {
            'config': self.config.__dict__,
            'loss_history': self.loss_history,
            'spatial_efficiency_history': self.spatial_efficiency_history,
            'training_stats': self.training_stats,
            'q_values': dict(self.q_values),
            'momentum_weights': dict(self.momentum_weights),
            'node_performance_history': dict(self.node_performance_history),
            'epoch': self.epoch
        }
        
        # Save node positions and connections
        node_data = []
        for node in self.neural_nodes:
            connections = self.get_node_connections(node.node_id)
            node_data.append({
                'id': node.node_id,
                'type': node.node_type,
                'position': node.node_position,
                'connections': connections
            })
        
        state['nodes'] = node_data
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        if self.demo:
            print(f"💾 Model state saved to {filepath}")

    def load_model_state(self, filepath: str):
        """Load a previously saved model state."""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore training state
            self.loss_history = state.get('loss_history', [])
            self.spatial_efficiency_history = state.get('spatial_efficiency_history', [])
            self.training_stats = state.get('training_stats', {})
            self.q_values = defaultdict(float, state.get('q_values', {}))
            self.momentum_weights = defaultdict(float, state.get('momentum_weights', {}))
            self.node_performance_history = defaultdict(list, state.get('node_performance_history', {}))
            self.epoch = state.get('epoch', 0)
            
            # Restore configuration
            config_data = state.get('config', {})
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            if self.demo:
                print(f"📂 Model state loaded from {filepath}")
                print(f"   Restored epoch: {self.epoch}")
                print(f"   Loss history length: {len(self.loss_history)}")
                print(f"   Q-table size: {len(self.q_values)}")
                
        except Exception as e:
            if self.demo:
                print(f"❌ Failed to load model state: {e}")
            raise
        