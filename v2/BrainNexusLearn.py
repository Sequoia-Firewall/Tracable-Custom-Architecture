from brainNexus import BrainNexus
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import time
import random
import copy
import re
import json
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
    
    # Enhanced training modes
    data_type: str = 'classification'  # 'classification', 'conversation', 'sequence', 'text_generation'
    unsupervised_weight: float = 0.3  # Weight for unsupervised loss component
    context_window: int = 5  # For conversation/sequence training
    autoencoder_dims: Optional[List[int]] = None  # For unsupervised representation learning
    clustering_k: int = 8  # For unsupervised clustering
    
    # Conversation-specific parameters
    conversation_max_length: int = 512  # Maximum conversation length
    response_generation: bool = True  # Whether to generate responses
    conversation_context_weight: float = 0.7  # How much to weight conversation context

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
            'convergence_rate': [],
            'unsupervised_loss': [],
            'conversation_coherence': [],
            'sequence_perplexity': []
        }
        
        # Enhanced learning state for different data types
        self.conversation_memory = deque(maxlen=1000)  # Store conversation context
        self.sequence_patterns = defaultdict(list)  # Learn sequence patterns
        self.unsupervised_embeddings = {}  # Store learned representations
        self.clustering_centroids = None  # For unsupervised clustering
        
        if self.demo:
            print("🎓 BrainNexusLearn initialized with spatial learning capabilities")
            print(f"   Learning rate: {self.config.learning_rate}")
            print(f"   Data type: {self.config.data_type}")
            print(f"   Unsupervised weight: {self.config.unsupervised_weight}")
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

    def conversation_train(self, conversations: List[List[Dict[str, str]]], 
                          validation_conversations: Optional[List[List[Dict[str, str]]]] = None) -> Dict[str, Any]:
        """
        Train the BrainNexus on conversation data.
        
        Args:
            conversations: List of conversations, where each conversation is a list of 
                         {"role": "user/assistant", "content": "text"} dictionaries
            validation_conversations: Optional validation conversations
            
        Returns:
            Dict containing training statistics
        """
        self.set_training_mode(True)
        start_time = time.time()
        
        if self.demo:
            print(f"💬 Starting conversation training with {len(conversations)} conversations")
            print(f"   Context window: {self.config.context_window}")
            print(f"   Max conversation length: {self.config.conversation_max_length}")
        
        # Preprocess conversations into training pairs
        training_data = self._preprocess_conversations(conversations)
        
        # Preprocess validation data if provided
        validation_data = None
        if validation_conversations:
            validation_data = self._preprocess_conversations(validation_conversations)
        
        # Use enhanced supervised training with conversation-specific loss
        original_data_type = self.config.data_type
        self.config.data_type = 'conversation'
        
        results = self.supervised_train(training_data, validation_data)
        
        # Restore original data type
        self.config.data_type = original_data_type
        
        if self.demo:
            avg_coherence = np.mean(self.training_stats['conversation_coherence']) if self.training_stats['conversation_coherence'] else 0.0
            print(f"✅ Conversation training completed")
            print(f"   Average conversation coherence: {avg_coherence:.3f}")
        
        return results

    def sequence_train(self, sequences: List[List[Any]], 
                      validation_sequences: Optional[List[List[Any]]] = None) -> Dict[str, Any]:
        """
        Train the BrainNexus on sequence data (time series, text, etc.).
        
        Args:
            sequences: List of sequences to learn from
            validation_sequences: Optional validation sequences
            
        Returns:
            Dict containing training statistics
        """
        self.set_training_mode(True)
        start_time = time.time()
        
        if self.demo:
            print(f"📈 Starting sequence training with {len(sequences)} sequences")
            print(f"   Context window: {self.config.context_window}")
        
        # Preprocess sequences into training pairs
        training_data = self._preprocess_sequences(sequences)
        
        # Preprocess validation data if provided
        validation_data = None
        if validation_sequences:
            validation_data = self._preprocess_sequences(validation_sequences)
        
        # Use enhanced supervised training with sequence-specific loss
        original_data_type = self.config.data_type
        self.config.data_type = 'sequence'
        
        results = self.supervised_train(training_data, validation_data)
        
        # Restore original data type
        self.config.data_type = original_data_type
        
        if self.demo:
            avg_perplexity = np.mean(self.training_stats['sequence_perplexity']) if self.training_stats['sequence_perplexity'] else 0.0
            print(f"✅ Sequence training completed")
            print(f"   Average sequence perplexity: {avg_perplexity:.3f}")
        
        return results

    def unsupervised_train(self, data: List[Any], method: str = 'autoencoder') -> Dict[str, Any]:
        """
        Train the BrainNexus using unsupervised learning.
        
        Args:
            data: List of input data (no targets)
            method: Unsupervised method ('autoencoder', 'clustering', 'contrastive')
            
        Returns:
            Dict containing training statistics
        """
        self.set_training_mode(True)
        start_time = time.time()
        
        if self.demo:
            print(f"🔬 Starting unsupervised training with {len(data)} samples")
            print(f"   Method: {method}")
            print(f"   Clustering K: {self.config.clustering_k}")
        
        if method == 'autoencoder':
            results = self._train_autoencoder(data)
        elif method == 'clustering':
            results = self._train_clustering(data)
        elif method == 'contrastive':
            results = self._train_contrastive(data)
        else:
            raise ValueError(f"Unknown unsupervised method: {method}")
        
        training_time = time.time() - start_time
        self.set_training_mode(False)
        
        if self.demo:
            print(f"✅ Unsupervised training completed in {training_time:.2f}s")
        
        results['training_time'] = training_time
        return results

    def hybrid_train(self, supervised_data: List[Tuple[Any, Any]], 
                    unsupervised_data: List[Any],
                    validation_data: Optional[List[Tuple[Any, Any]]] = None) -> Dict[str, Any]:
        """
        Train using both supervised and unsupervised learning simultaneously.
        
        Args:
            supervised_data: List of (input, target) pairs
            unsupervised_data: List of input data without targets
            validation_data: Optional validation data
            
        Returns:
            Dict containing training statistics
        """
        self.set_training_mode(True)
        start_time = time.time()
        
        if self.demo:
            print(f"🔄 Starting hybrid training")
            print(f"   Supervised samples: {len(supervised_data)}")
            print(f"   Unsupervised samples: {len(unsupervised_data)}")
            print(f"   Unsupervised weight: {self.config.unsupervised_weight}")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            # Supervised training phase
            supervised_loss, supervised_acc = self._process_supervised_batch(
                random.sample(supervised_data, min(len(supervised_data), self.config.batch_size))
            )
            
            # Unsupervised training phase
            unsupervised_loss = self._process_unsupervised_batch(
                random.sample(unsupervised_data, min(len(unsupervised_data), self.config.batch_size))
            )
            
            # Combined loss
            total_loss = supervised_loss + self.config.unsupervised_weight * unsupervised_loss
            
            # Update networks
            self._backward_pass_supervised()
            self._optimize_spatial_structure()
            
            # Track metrics
            self.loss_history.append(total_loss)
            self.training_stats['unsupervised_loss'].append(unsupervised_loss)
            
            # Progress reporting
            if self.demo and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.config.max_epochs}:")
                print(f"    Supervised Loss: {supervised_loss:.4f}, Accuracy: {supervised_acc:.3f}")
                print(f"    Unsupervised Loss: {unsupervised_loss:.4f}")
                print(f"    Total Loss: {total_loss:.4f}")
            
            # Early stopping
            if total_loss < best_loss - self.config.convergence_threshold:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    if self.demo:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
        
        training_time = time.time() - start_time
        self.set_training_mode(False)
        
        if self.demo:
            print(f"✅ Hybrid training completed in {training_time:.2f}s")
            print(f"   Final combined loss: {best_loss:.4f}")
        
        return {
            'training_time': training_time,
            'final_loss': best_loss,
            'epochs_completed': epoch + 1,
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
        """Calculate loss between prediction and target based on data type."""
        try:
            # Handle case where prediction might be malformed
            if not isinstance(prediction, dict):
                return 1.0  # High loss for invalid prediction
            
            # Get data type for specialized loss calculation
            data_type = getattr(self.config, 'data_type', 'classification')
            
            if data_type == 'conversation':
                return self._calculate_conversation_loss(prediction, target)
            elif data_type == 'sequence':
                return self._calculate_sequence_loss(prediction, target)
            else:
                # Default classification loss
                return self._calculate_classification_loss(prediction, target)
            
        except Exception as e:
            if hasattr(self, 'demo') and self.demo:
                print(f"  ⚠️  Loss calculation error: {e}")
            return 1.0  # Default high loss

    def _calculate_classification_loss(self, prediction: Dict[str, Any], target: Any) -> float:
        """Calculate classification loss."""
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

    def _calculate_conversation_loss(self, prediction: Dict[str, Any], target: Any) -> float:
        """Calculate conversation-specific loss."""
        # For conversations, we can calculate coherence-based loss
        pred_text = prediction.get('prediction', '')
        target_text = target if isinstance(target, str) else str(target)
        
        # Simple coherence measure (can be enhanced with embeddings)
        coherence_score = self._calculate_text_coherence(pred_text, target_text)
        conversation_loss = 1.0 - coherence_score
        
        # Track conversation coherence
        self.training_stats['conversation_coherence'].append(coherence_score)
        
        return conversation_loss

    def _calculate_sequence_loss(self, prediction: Dict[str, Any], target: Any) -> float:
        """Calculate sequence-specific loss."""
        # For sequences, we're predicting the next element
        pred_probs = prediction.get('probabilities', [0.5, 0.5])
        confidence = prediction.get('confidence', 0.5)
        
        # Convert target to numerical format for loss calculation
        if isinstance(target, str):
            # For string targets, use simple categorical error
            predicted_label = prediction.get('prediction', '')
            sequence_error = 0.0 if str(predicted_label) == str(target) else 1.0
        elif isinstance(target, (int, float)):
            # For numerical targets, try to extract numerical prediction
            try:
                predicted_value = float(prediction.get('prediction', 0))
                sequence_error = (predicted_value - float(target)) ** 2
            except (ValueError, TypeError):
                # Fallback to categorical error
                sequence_error = 0.0 if str(prediction.get('prediction', '')) == str(target) else 1.0
        else:
            # For other types, use confidence-based error
            sequence_error = 1.0 - confidence
        
        # Normalize error and add uncertainty penalty
        sequence_loss = sequence_error + (1.0 - confidence) * 0.1
        
        # Calculate and track perplexity
        perplexity = np.exp(min(sequence_loss, 10)) if sequence_loss > 0 else 1.0
        self.training_stats['sequence_perplexity'].append(perplexity)
        
        return min(sequence_loss, 10.0)  # Cap maximum loss

    def _calculate_text_coherence(self, pred_text: str, target_text: str) -> float:
        """Calculate simple text coherence score."""
        if not pred_text or not target_text:
            return 0.0
        
        # Simple word overlap-based coherence
        pred_words = set(pred_text.lower().split())
        target_words = set(target_text.lower().split())
        
        if len(target_words) == 0:
            return 0.0
        
        overlap = len(pred_words.intersection(target_words))
        coherence = overlap / len(target_words)
        
        return min(coherence, 1.0)

    def _calculate_accuracy(self, prediction: Dict[str, Any], target: Any) -> float:
        """Calculate accuracy between prediction and target."""
        try:
            # Handle case where prediction might be malformed
            if not isinstance(prediction, dict):
                return 0.0  # Zero accuracy for invalid prediction
            
            pred_class = prediction.get('prediction', 0)
            
            if isinstance(target, int):
                # Integer classification target
                if isinstance(pred_class, (int, float)):
                    return 1.0 if int(pred_class) == target else 0.0
                else:
                    return 0.0
            elif isinstance(target, str):
                # String sequence target
                pred_str = str(pred_class)
                return 1.0 if pred_str == target else 0.0
            elif isinstance(target, float):
                # Float regression target
                if isinstance(pred_class, (int, float)):
                    # Use tolerance-based accuracy for float targets
                    tolerance = 0.1
                    return 1.0 if abs(float(pred_class) - target) <= tolerance else 0.0
                else:
                    return 0.0
            else:
                # For other target types, try tensor conversion
                try:
                    target_tensor = torch.tensor(target, dtype=torch.float32)
                    target_class = target_tensor.argmax().item()
                    if isinstance(pred_class, (int, float)):
                        return 1.0 if int(pred_class) == target_class else 0.0
                    else:
                        return 0.0
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
        elif isinstance(target, float):
            # For float targets, convert to tensor safely
            try:
                target_tensor = torch.tensor([float(target)], dtype=torch.float32)
                pred_tensor = torch.tensor([pred_probs[0]] if pred_probs else [0.5], dtype=torch.float32)
                error = F.mse_loss(pred_tensor, target_tensor).item()
            except (ValueError, TypeError):
                error = 1.0 - confidence  # Fallback to confidence-based error
        elif isinstance(target, str):
            # For string targets, use categorical accuracy
            predicted_label = str(prediction.get('prediction', ''))
            error = 0.0 if predicted_label == target else 1.0
        else:
            # For other types, use confidence-based error
            error = 1.0 - confidence
        
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

    # Data Preprocessing Methods
    def _preprocess_conversations(self, conversations: List[List[Dict[str, str]]]) -> List[Tuple[Any, Any]]:
        """Convert conversations into training pairs."""
        training_pairs = []
        
        for conversation in conversations:
            if len(conversation) < 2:
                continue
            
            # Create context-response pairs
            for i in range(1, len(conversation)):
                # Context: previous messages (up to context window)
                start_idx = max(0, i - self.config.context_window)
                context_messages = conversation[start_idx:i]
                
                # Format context
                context = {
                    'conversation_history': context_messages,
                    'features': self._extract_conversation_features(context_messages),
                    'type': 'conversation'
                }
                
                # Target: current response
                target = conversation[i]['content'] if 'content' in conversation[i] else str(conversation[i])
                
                training_pairs.append((context, target))
        
        return training_pairs

    def _preprocess_sequences(self, sequences: List[List[Any]]) -> List[Tuple[Any, Any]]:
        """Convert sequences into training pairs."""
        training_pairs = []
        
        for sequence in sequences:
            if len(sequence) <= self.config.context_window:
                continue
            
            # Create input-target pairs with sliding window
            for i in range(self.config.context_window, len(sequence)):
                # Input: previous elements
                input_sequence = sequence[i - self.config.context_window:i]
                
                # Format input
                input_data = {
                    'sequence': input_sequence,
                    'features': self._extract_sequence_features(input_sequence),
                    'type': 'sequence'
                }
                
                # Target: next element
                target = sequence[i]
                
                training_pairs.append((input_data, target))
        
        return training_pairs

    def _extract_conversation_features(self, messages: List[Dict[str, str]]) -> List[float]:
        """Extract numerical features from conversation messages."""
        if not messages:
            return [0.0] * 10  # Default feature vector
        
        features = []
        
        # Basic features
        features.append(len(messages))  # Number of messages
        features.append(sum(len(msg.get('content', '').split()) for msg in messages))  # Total words
        features.append(len(set(msg.get('role', 'unknown') for msg in messages)))  # Unique speakers
        
        # Simple sentiment/tone features (placeholder)
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        features.append(total_chars)  # Total characters
        
        # Question/exclamation counts
        all_content = ' '.join(msg.get('content', '') for msg in messages)
        features.append(all_content.count('?'))  # Questions
        features.append(all_content.count('!'))  # Exclamations
        
        # Average message length
        avg_length = total_chars / len(messages) if messages else 0
        features.append(avg_length)
        
        # Pad or truncate to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]

    def _extract_sequence_features(self, sequence: List[Any]) -> List[float]:
        """Extract numerical features from sequence data."""
        if not sequence:
            return [0.0] * 10
        
        features = []
        
        # Basic sequence statistics
        features.append(len(sequence))  # Length
        
        # Handle numerical sequences
        numerical_values = []
        for item in sequence:
            if isinstance(item, (int, float)):
                numerical_values.append(float(item))
            elif isinstance(item, str) and item.replace('.', '').replace('-', '').isdigit():
                numerical_values.append(float(item))
        
        if numerical_values:
            features.append(np.mean(numerical_values))  # Mean
            features.append(np.std(numerical_values))   # Std dev
            features.append(np.min(numerical_values))   # Min
            features.append(np.max(numerical_values))   # Max
            features.append(len(set(numerical_values))) # Unique values
        else:
            features.extend([0.0] * 5)
        
        # Handle categorical sequences
        unique_items = len(set(str(item) for item in sequence))
        features.append(unique_items)  # Unique items
        
        # Sequence complexity (simple measure)
        transitions = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i-1])
        features.append(transitions)  # State transitions
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]

    # Unsupervised Learning Methods
    def _process_unsupervised_batch(self, batch: List[Any]) -> float:
        """Process a batch of unsupervised data."""
        total_loss = 0.0
        
        for data in batch:
            # Run forward pass
            result = self.run(data, trace_execution=True)
            
            # Calculate unsupervised loss (reconstruction or clustering)
            unsupervised_loss = self._calculate_unsupervised_loss(result, data)
            total_loss += unsupervised_loss
        
        return total_loss / len(batch) if batch else 0.0

    def _calculate_unsupervised_loss(self, result: Dict[str, Any], original_data: Any) -> float:
        """Calculate unsupervised learning loss."""
        # Simple reconstruction loss
        prediction = result.get('result', {})
        
        # Try to reconstruct input from internal representation
        if isinstance(original_data, dict) and 'features' in original_data:
            original_features = original_data['features']
            predicted_features = prediction.get('probabilities', [])
            
            if len(predicted_features) >= len(original_features):
                # Reconstruction error
                reconstruction_error = 0.0
                for i, orig_val in enumerate(original_features):
                    pred_val = predicted_features[i] if i < len(predicted_features) else 0.0
                    reconstruction_error += (orig_val - pred_val) ** 2
                
                return reconstruction_error / len(original_features)
        
        # Default clustering-based loss
        return 0.5  # Placeholder

    def _train_autoencoder(self, data: List[Any]) -> Dict[str, Any]:
        """Train using autoencoder approach."""
        if self.demo:
            print("  🔄 Training autoencoder...")
        
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        # Process data in batches
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            batch_loss = self._process_unsupervised_batch(batch)
            total_reconstruction_loss += batch_loss
            num_batches += 1
        
        avg_loss = total_reconstruction_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'method': 'autoencoder',
            'reconstruction_loss': avg_loss,
            'samples_processed': len(data)
        }

    def _train_clustering(self, data: List[Any]) -> Dict[str, Any]:
        """Train using clustering approach."""
        if self.demo:
            print("  🔄 Training clustering...")
        
        # Extract features from data
        features = []
        for item in data:
            if isinstance(item, dict) and 'features' in item:
                features.append(item['features'])
            else:
                # Convert to feature vector
                if isinstance(item, (list, tuple)):
                    features.append([float(x) if isinstance(x, (int, float)) else 0.0 for x in item[:10]])
                else:
                    features.append([float(hash(str(item)) % 1000) / 1000.0] + [0.0] * 9)
        
        if not features:
            return {'method': 'clustering', 'error': 'No features extracted'}
        
        # Simple k-means clustering
        features_array = np.array(features)
        k = min(self.config.clustering_k, len(features))
        
        # Initialize centroids randomly
        centroids = features_array[np.random.choice(len(features_array), k, replace=False)]
        
        # Run k-means iterations
        for iteration in range(10):  # Simple fixed iterations
            # Assign points to clusters
            distances = np.linalg.norm(features_array[:, np.newaxis] - centroids, axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([features_array[assignments == i].mean(axis=0) 
                                    for i in range(k)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Store centroids for future use
        self.clustering_centroids = centroids
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = sum(np.linalg.norm(features_array[assignments == i] - centroids[i])**2 
                     for i in range(k) if np.any(assignments == i))
        
        return {
            'method': 'clustering',
            'inertia': inertia,
            'num_clusters': k,
            'samples_processed': len(data)
        }

    def _train_contrastive(self, data: List[Any]) -> Dict[str, Any]:
        """Train using contrastive learning approach."""
        if self.demo:
            print("  🔄 Training contrastive...")
        
        # Simple contrastive learning: compare similar and dissimilar pairs
        contrastive_loss = 0.0
        num_pairs = 0
        
        for i in range(0, len(data) - 1, 2):
            if i + 1 >= len(data):
                break
            
            # Get two data points
            data1, data2 = data[i], data[i + 1]
            
            # Run forward pass for both
            result1 = self.run(data1, trace_execution=False)
            result2 = self.run(data2, trace_execution=False)
            
            # Extract representations
            repr1 = result1.get('result', {}).get('probabilities', [])
            repr2 = result2.get('result', {}).get('probabilities', [])
            
            if repr1 and repr2:
                # Calculate similarity
                min_len = min(len(repr1), len(repr2))
                if min_len > 0:
                    similarity = sum(repr1[j] * repr2[j] for j in range(min_len)) / min_len
                    
                    # Contrastive loss (encourage diversity)
                    contrastive_loss += max(0, 0.5 - abs(similarity))
                    num_pairs += 1
        
        avg_loss = contrastive_loss / num_pairs if num_pairs > 0 else 0.0
        
        return {
            'method': 'contrastive',
            'contrastive_loss': avg_loss,
            'pairs_processed': num_pairs
        }

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
    
    # ===== FLAT TEXT TRAINING INTEGRATION =====
    
    def train_on_flat_text(self, text: str, text_title: str = "", text_type: str = "document", 
                          context_window: int = 10, overlap: int = 5, 
                          training_approach: str = "all") -> Dict[str, Any]:
        """
        Train BrainNexus directly on flat text (Wikipedia pages, books, articles, etc.)
        
        Args:
            text: Raw text content to train on
            text_title: Title/name of the text source
            text_type: Type of text ("wikipedia", "book", "article", "webpage", etc.)
            context_window: Number of tokens to use as input context
            overlap: Number of tokens to overlap between sequences
            training_approach: "all", "token_prediction", "sentence_completion", "paragraph_continuation"
            
        Returns:
            Training results dictionary
        """
        if self.demo:
            print(f"📖 Training on flat text: {text_title or 'Untitled'}")
            print(f"   Text type: {text_type}")
            print(f"   Text length: {len(text)} characters")
            print(f"   Context window: {context_window}")
            print(f"   Training approach: {training_approach}")
        
        # Preprocess text into training data
        training_data = self._preprocess_flat_text(
            text, text_title, text_type, context_window, overlap, training_approach
        )
        
        # Configure for text generation
        old_data_type = self.config.data_type
        self.config.data_type = 'text_generation'
        self.config.context_window = context_window
        
        try:
            # Split into training and validation
            random.shuffle(training_data)
            split_point = int(0.8 * len(training_data))
            train_data = training_data[:split_point]
            val_data = training_data[split_point:] if split_point < len(training_data) else []
            
            if self.demo:
                print(f"   Training samples: {len(train_data)}")
                print(f"   Validation samples: {len(val_data)}")
            
            # Train the model
            results = self.supervised_train(train_data, val_data)
            
            # Add text-specific metadata
            results['text_metadata'] = {
                'title': text_title,
                'type': text_type,
                'text_length': len(text),
                'context_window': context_window,
                'training_approach': training_approach,
                'vocabulary_size': len(set(text.split())),
                'total_samples': len(training_data)
            }
            
            if self.demo:
                print(f"✅ Flat text training completed!")
                vocab_size = results['text_metadata']['vocabulary_size']
                print(f"   Vocabulary size: {vocab_size}")
                print(f"   Final accuracy: {results.get('final_accuracy', 0):.3f}")
            
            return results
            
        finally:
            # Restore original configuration
            self.config.data_type = old_data_type
    
    def _preprocess_flat_text(self, text: str, title: str, text_type: str, 
                             context_window: int, overlap: int, 
                             approach: str) -> List[Tuple[Dict, str]]:
        """Preprocess flat text into training data."""
        
        # Clean the text
        if text_type.lower() == "wikipedia":
            cleaned_text = self._clean_wikipedia_text(text)
        else:
            cleaned_text = self._clean_general_text(text)
        
        training_pairs = []
        
        # Apply training approaches
        if approach in ["all", "token_prediction"]:
            token_data = self._create_token_prediction_data(
                cleaned_text, text_type, title, context_window
            )
            training_pairs.extend(token_data)
        
        if approach in ["all", "sentence_completion"]:
            sentence_data = self._create_sentence_completion_data(
                cleaned_text, text_type, title
            )
            training_pairs.extend(sentence_data)
        
        if approach in ["all", "paragraph_continuation"]:
            paragraph_data = self._create_paragraph_data(
                cleaned_text, text_type, title
            )
            training_pairs.extend(paragraph_data)
        
        return training_pairs
    
    def _clean_wikipedia_text(self, text: str) -> str:
        """Clean Wikipedia-specific markup and formatting."""
        # Remove Wikipedia markup
        text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Remove templates
        text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', text)  # Remove links, keep text
        text = re.sub(r'\[http[^\]]*\]', '', text)  # Remove external links
        text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        text = re.sub(r'==+.*?==+', '', text)  # Remove section headers
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines
        text = re.sub(r'^\s*\*.*$', '', text, flags=re.MULTILINE)  # Remove bullet points
        
        return self._clean_general_text(text)
    
    def _clean_general_text(self, text: str) -> str:
        """General text cleaning for any document."""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)  # Keep basic punctuation
        text = text.strip()
        return text
    
    def _create_token_prediction_data(self, text: str, source: str, title: str, 
                                    context_window: int) -> List[Tuple[Dict, str]]:
        """Create token-by-token prediction training data."""
        training_pairs = []
        
        # Tokenize (simple word-based)
        tokens = text.split()
        
        # Create sliding window pairs
        for i in range(context_window, len(tokens)):
            # Input: context window of previous tokens
            context_tokens = tokens[i - context_window:i]
            context_text = " ".join(context_tokens)
            
            # Extract features from context
            features = self._extract_text_features(context_text)
            
            input_data = {
                'text': context_text,
                'features': features,
                'context_length': len(context_tokens),
                'task_type': 'token_prediction',
                'metadata': {
                    'source': source,
                    'title': title,
                    'position': i / len(tokens)  # Position in document
                }
            }
            
            # Target: next token
            target = tokens[i]
            
            training_pairs.append((input_data, target))
        
        return training_pairs
    
    def _create_sentence_completion_data(self, text: str, source: str, title: str) -> List[Tuple[Dict, str]]:
        """Create sentence completion training data."""
        training_pairs = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 5:  # Skip very short sentences
                continue
            
            # Create multiple completion points in each sentence
            for split_point in range(2, min(len(words) - 1, 8)):
                context_words = words[:split_point]
                context_text = " ".join(context_words)
                
                # Extract features
                features = self._extract_text_features(context_text)
                
                input_data = {
                    'text': context_text,
                    'features': features,
                    'context_length': len(context_words),
                    'task_type': 'sentence_completion',
                    'metadata': {
                        'source': source,
                        'title': title,
                        'completion_point': split_point / len(words)
                    }
                }
                
                # Target: next word in sentence
                target = words[split_point]
                
                training_pairs.append((input_data, target))
        
        return training_pairs
    
    def _create_paragraph_data(self, text: str, source: str, title: str) -> List[Tuple[Dict, str]]:
        """Create paragraph-level understanding data."""
        training_pairs = []
        
        # Split into paragraphs
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        for paragraph in paragraphs:
            sentences = re.split(r'[.!?]+', paragraph)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                continue
            
            # Use first part of paragraph to predict continuation
            split_point = len(sentences) // 2
            context_sentences = sentences[:split_point]
            context_text = ". ".join(context_sentences) + "."
            
            # Extract features
            features = self._extract_text_features(context_text)
            
            input_data = {
                'text': context_text,
                'features': features,
                'context_length': len(context_text.split()),
                'task_type': 'paragraph_continuation',
                'metadata': {
                    'source': source,
                    'title': title,
                    'paragraph_length': len(paragraph)
                }
            }
            
            # Target: first word of continuation
            if split_point < len(sentences):
                target_sentence = sentences[split_point].strip()
                if target_sentence:
                    target = target_sentence.split()[0]
                    training_pairs.append((input_data, target))
        
        return training_pairs
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract numerical features from text."""
        words = text.split()
        
        features = []
        
        # Basic text statistics
        features.append(len(words))  # Word count
        features.append(len(text))   # Character count
        features.append(len(text) / len(words) if words else 0)  # Avg word length
        
        # Punctuation features
        features.append(text.count('.'))   # Periods
        features.append(text.count(','))   # Commas
        features.append(text.count('?'))   # Questions
        features.append(text.count('!'))   # Exclamations
        
        # Complexity features
        unique_words = len(set(word.lower() for word in words))
        features.append(unique_words)  # Vocabulary richness
        features.append(unique_words / len(words) if words else 0)  # Diversity ratio
        
        # Sentence structure
        sentences = len(re.split(r'[.!?]+', text))
        features.append(sentences)  # Sentence count
        
        # Ensure exactly 10 features
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]
    
    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """
        Generate text continuation from a prompt using the trained model.
        
        Args:
            prompt: Input text to continue
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text continuation
        """
        if self.demo:
            print(f"🎯 Generating text from prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        generated_tokens = []
        current_text = prompt
        
        for i in range(max_length):
            # Extract features from current context
            features = self._extract_text_features(current_text)
            
            # Create input data
            input_data = {
                'text': current_text,
                'features': features,
                'context_length': len(current_text.split()),
                'task_type': 'text_generation',
                'metadata': {
                    'generation_step': i,
                    'temperature': temperature
                }
            }
            
            # Run inference
            try:
                # Use the BrainNexus run method for inference
                brain_result = self.run(input_data, trace_execution=False)
                
                # Extract prediction from brain result
                prediction = brain_result.get('prediction', '')
                confidence = brain_result.get('confidence', 0.5)
                
                # If no direct prediction, try to get from output
                if not prediction and 'output' in brain_result:
                    prediction = str(brain_result['output'])
                elif not prediction and 'result' in brain_result:
                    prediction = str(brain_result['result'])
                
                result = {'prediction': prediction, 'confidence': confidence}
            except Exception as e:
                if self.demo:
                    print(f"⚠️ Inference error at step {i}: {e}")
                break
            
            if result and 'prediction' in result:
                next_token = result['prediction']
                
                # Apply temperature sampling if confidence is available
                if 'confidence' in result and temperature != 1.0:
                    confidence = result['confidence']
                    # Adjust token selection based on temperature
                    if random.random() > confidence * temperature:
                        # Add some randomness for lower temperature
                        vocab = current_text.split()
                        if vocab:
                            next_token = random.choice(vocab[-10:])  # Choose from recent vocab
                
                generated_tokens.append(str(next_token))
                current_text = current_text + " " + str(next_token)
                
                # Stop at sentence endings
                if str(next_token).endswith(('.', '!', '?')):
                    break
            else:
                break
        
        generated_text = " ".join(generated_tokens)
        
        if self.demo:
            print(f"✅ Generated {len(generated_tokens)} tokens")
        
        return generated_text
        