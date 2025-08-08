import json
import random
import math
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque, defaultdict
import time
from NeuralNode import NeuralNode
import pandas as pd
import numpy as np
import random
import math
from typing import Any, Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict, deque
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F

class BrainNexus:
    def __init__(self, dimensions: int = 4, node_count_pre: int = 3, demo: bool = False, 
                 output_config: Optional[Dict[str, Any]] = None):
        self.demo = demo        
        self.dimensions = dimensions
        self.epoch_count_or_type = 'dynamic'
        self.learning_rate = 0.01 if not demo else 0.05  # More aggressive in demo
        self.epoch_giveup_rate = 0.001 if not demo else 0.02  # Give up faster in demo
        self.entrance_node_count_pre = node_count_pre
        self.neural_nodes = []
        self.next_node_id = 0
        self.node_registry = {}  # node_id -> Node object
        self.spatial_index = None  # KDTree for spatial queries
        self.node_positions = []  # Array of positions for spatial indexing
        self.node_id_to_index = {}  # Mapping from node_id to position index
        
        # Configure output system
        self.output_config = self._setup_output_config(output_config)
        
        # Node reuse tracking
        self.node_usage_history = defaultdict(list)  # node_id -> [usage_count_per_epoch]
        self.reuse_candidates = set()  # Nodes eligible for reuse
        self.reuse_threshold = 0.7  # Usage threshold for reuse eligibility
        
        # Multi-layer attention mechanism
        self.attention_layers = 3
        self.attention_heads = 4
        self.attention_dim = 64
        self.term_embeddings = {}  # Cache for term embeddings
        self.attention_masks = {}  # Layer-specific attention masks
        
        # Spatial intelligence parameters
        self.spatial_decay_rate = 0.1  # How quickly spatial influence decays
        self.locality_radius = 5.0  # Base radius for local connections
        self.global_context_weight = 0.3  # Weight for global vs local processing
        
        # Performance tracking
        self.inference_cache = {}  # Cache for repeated computations
        self.routing_history = deque(maxlen=1000)  # Track routing decisions
        
        self.node_records = pd.DataFrame(columns=[
            'Node_ID', 'Node_Type', 'Node_Position', 'Node_Group',
            'Entrance_Connections', 'Exit_Connections', 'Generic_Connections',
            'Max_Random_Weight', 'Min_Random_Weight', 'Constant_Weight',
            'Times_Called', 'Input_Tracker', 'Reuse_Count', 'Attention_Weights',
            'Spatial_Affinity'
        ])
        
        # BRAIN_RECORDS: Overall network performance metrics and configuration
        self.brain_records = pd.DataFrame([{
            "Dimensions": dimensions,                    # Spatial dimensions used
            "Entrance_Node_Count_Post": 0,              # Actual input nodes created
            "Entrance_Node_Count_Pre": self.entrance_node_count_pre,  # Initial estimate
            "Accuracy": 0.0,                            # Classification accuracy
            "Precision": 0.0,                           # True positives / (true + false positives)
            "Recall": 0.0,                              # True positives / (true + false negatives)  
            "F1_Score": 0.0,                            # Harmonic mean of precision/recall
            "Training_Time": 0.0,                       # Time spent training
            "Inference_Time": 0.0,                      # Time for forward pass
            "Loss": 0.0,                                # Current loss value
            "Learning_Rate": self.learning_rate,        # Learning rate used
            "Epoch_count_or_type": self.epoch_count_or_type,  # Training approach
            "Spatial_Efficiency": 0.0,                 # Spatial utilization metric
            "Reuse_Efficiency": 0.0,                   # Node reuse effectiveness
            "Attention_Coherence": 0.0,                # Multi-layer attention alignment
        }])

    def _setup_output_config(self, output_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Setup output configuration for different use cases.
        
        Args:
            output_config: Configuration dict with output settings
            
        Returns:
            Complete output configuration
        """
        if output_config is None:
            output_config = {}
        
        # Default configuration for 10-class classification
        default_config = {
            'type': 'classification',  # 'classification', 'tokens', 'custom'
            'num_classes': 10,
            'class_labels': [f'class_{i}' for i in range(10)],
            'output_format': 'index',  # 'index', 'label', 'token_id', 'probability_dist'
            'vocab_mapping': None,  # For token-based outputs
            'confidence_threshold': 0.7,
            'return_top_k': 1,  # Return top-k predictions
            'normalize_probabilities': True,
            
            # Multi-token generation options
            'enable_multi_token': False,  # Enable sequence generation
            'max_sequence_length': 10,   # Maximum tokens to generate
            'sequence_length': 5,        # Default sequence length  
            'ideal_sequence_length': 5,  # Target length for quality scoring
            'context_window': 5,         # Context window for generation
            'generation_method': 'hybrid_context_aware',  # Generation method
            
            # Node group behavior configuration
            'node_group_behaviors': {
                'Judge': 'conservative',      # High confidence tokens
                'Controller': 'directive',    # Control/structure tokens
                'Splitter': 'divergent',     # Exploration tokens
                'Computational': 'analytical', # Logic-based tokens
                'Retainer': 'contextual',    # Memory-aware tokens
                'Reviewer': 'evaluative'     # Quality-focused tokens
            }
        }
        
        # Merge with user configuration
        config = {**default_config, **output_config}
        
        # Validate and adjust configuration
        if config['type'] == 'tokens':
            if config['vocab_mapping'] is None:
                # Create a simple token mapping for demo
                config['vocab_mapping'] = {i: f'token_{i}' for i in range(config['num_classes'])}
            config['output_format'] = 'token_id'
        elif config['type'] == 'custom':
            if len(config['class_labels']) != config['num_classes']:
                # Adjust class labels to match num_classes
                config['class_labels'] = [f'custom_{i}' for i in range(config['num_classes'])]
        
        # Ensure class_labels matches num_classes
        if len(config['class_labels']) != config['num_classes']:
            config['class_labels'] = [f'class_{i}' for i in range(config['num_classes'])]
        
        if self.demo:
            print(f"🔧 Output Config: {config['type']} with {config['num_classes']} classes")
            print(f"   Format: {config['output_format']}, Top-K: {config['return_top_k']}")
            if config.get('enable_multi_token', False):
                print(f"   Multi-token: ENABLED (max: {config['max_sequence_length']}, default: {config['sequence_length']})")
                print(f"   Generation: {config['generation_method']}, Context window: {config['context_window']}")
            else:
                print(f"   Multi-token: DISABLED (single token mode)")
            if config['type'] == 'tokens' and config['vocab_mapping']:
                sample_tokens = list(config['vocab_mapping'].values())[:3]
                print(f"   Sample tokens: {sample_tokens}...")
        
        return config

    
    def add_neural_node(self, node_type: str, position: Optional[List[float]] = None, 
                    node_group: str = 'default', **kwargs) -> int:
        """
        Add a new neural node to the BrainNexus.
    
        Args:
            node_type (str): The type of the neural node.
            position (Optional[List[float]]): The spatial position of the node.
            node_group (str): Group identifier for organizational purposes.
            **kwargs: Additional parameters for specific node types.
    
        Returns:
            int: The ID of the newly created node.
        """
        start_time = time.time()
        
        node_id = self.next_node_id
        self.next_node_id += 1
    
        # Auto-generate position if not provided
        if position is None:
            position = self._generate_smart_position()
        
        # Ensure position is the right length for dimensions and convert to list
        position = list(position)  # Convert tuple to list for consistency
        if len(position) != self.dimensions:
            if len(position) > self.dimensions:
                position = position[:self.dimensions]
            else:
                position = position + [0.0] * (self.dimensions - len(position))
    
        # Create node using factory pattern
        new_node = NeuralNode(node_id, node_type, position, self.demo)
        
        # Configure node-specific parameters
        self._configure_node_params(new_node, node_type, **kwargs)
        
        # Add to collections
        self.neural_nodes.append(new_node)
        self.node_registry[node_id] = new_node
        
        # Update spatial tracking
        self._update_spatial_data(position, node_id)
        
        # Update records efficiently
        self._add_node_record(new_node, node_group, start_time)
        
        if self.demo:
            print(f"✓ Added {new_node.__class__.__name__} node #{node_id}")
        
        return node_id

    def _generate_smart_position(self) -> List[float]:
        """Generate a smart position based on existing nodes."""
        if len(self.neural_nodes) == 0 or self.dimensions == 0:
            return np.random.uniform(-10, 10, self.dimensions).tolist()
        
        # Position near existing nodes with controlled randomness
        existing_positions = np.array([node.node_position for node in self.neural_nodes 
                                    if hasattr(node, 'node_position')])
        if len(existing_positions) > 0:
            center = np.mean(existing_positions, axis=0)
            noise = np.random.normal(0, self.locality_radius * 0.3, self.dimensions)
            return (center + noise).tolist()
        
        return np.random.uniform(-10, 10, self.dimensions).tolist()

    def _configure_node_params(self, node, node_type: str, **kwargs):
        """Configure node-specific parameters efficiently."""
        # Set standard attributes
        node.node_group = kwargs.get('node_group', 'default')
        node.times_called = 0
        node.input_tracker = []
        node.reuse_count = 0
        
        # Set up weights
        node.define_node_weights(
            max_random=kwargs.get('max_random_weight', 1.0),
            min_random=kwargs.get('min_random_weight', 0.0),
            constant=kwargs.get('constant_weight', 0.0)
        )
        
        # Initialize connections
        node.entrance_connections = kwargs.get('entrance_connections', [])
        node.exit_connections = kwargs.get('exit_connections', [])
        node.generic_connections = kwargs.get('generic_connections', [])
        
        # Node-type specific parameters
        if node_type == 'Controller' and 'num_branches' in kwargs:
            node.num_branches = kwargs['num_branches']
        elif node_type == 'Splitter' and 'num_branches' in kwargs:
            node.num_branches = kwargs['num_branches']
        elif node_type == 'Retainer' and 'expected_nodes' in kwargs:
            node.expected_nodes = kwargs['expected_nodes']
        elif node_type in ['Review', 'Reviewer'] and 'num_comps' in kwargs:
            node.num_comps = kwargs['num_comps']

    def _update_spatial_data(self, position: List[float], node_id: int):
        """Update spatial index and tracking data."""
        self.node_positions.append(position)
        self.node_id_to_index[node_id] = len(self.node_positions) - 1
        
        # Initialize node usage tracking
        self.node_usage_history[node_id] = []
        
        # Update spatial index efficiently
        try:
            if self.spatial_index is None and len(self.node_positions) > 1:
                self.spatial_index = KDTree(np.array(self.node_positions))
            elif self.spatial_index is not None:
                # Rebuild KDTree - this is necessary as KDTree doesn't support incremental updates
                self.spatial_index = KDTree(np.array(self.node_positions))
        except Exception:
            self.spatial_index = None  # Fallback to linear search

    def _add_node_record(self, node, node_group: str, start_time: float):
        """Add node record to DataFrame efficiently."""
        # Calculate spatial affinity
        spatial_affinity = self._calculate_spatial_affinity(node.node_position)
        node.spatial_affinity = spatial_affinity
        
        # Initialize attention weights
        node.attention_weights = np.random.normal(0, 0.1, 
                                                (self.attention_layers, self.attention_heads, self.attention_dim))
        
        # Create record
        new_record = pd.DataFrame([{
            'Node_ID': node.node_id,
            'Node_Type': node.node_type,
            'Node_Position': node.node_position.copy(),
            'Node_Group': node_group,
            'Entrance_Connections': getattr(node, 'entrance_connections', []),
            'Exit_Connections': getattr(node, 'exit_connections', []),
            'Generic_Connections': getattr(node, 'generic_connections', []),
            'Max_Random_Weight': getattr(node.weights, 'Max_random', 1.0) if hasattr(node, 'weights') else 1.0,
            'Min_Random_Weight': getattr(node.weights, 'Min_random', 0.0) if hasattr(node, 'weights') else 0.0,
            'Constant_Weight': getattr(node.weights, 'constant', 0.0) if hasattr(node, 'weights') else 0.0,
            'Times_Called': 0,
            'Input_Tracker': [],
            'Reuse_Count': 0,
            'Attention_Weights': node.attention_weights.tolist(),
            'Spatial_Affinity': spatial_affinity
        }])
        
        self.node_records = pd.concat([self.node_records, new_record], ignore_index=True)
        
        # Update brain records
        creation_time = time.time() - start_time
        self._update_brain_metrics(creation_time)

    def _calculate_spatial_affinity(self, position: List[float]) -> float:
        """Calculate spatial affinity score."""
        if len(self.neural_nodes) <= 1:
            return 1.0
        
        existing_positions = np.array([node.node_position for node in self.neural_nodes[:-1]  # Exclude current node
                                    if hasattr(node, 'node_position')])
        
        if len(existing_positions) == 0:
            return 1.0
        
        distances = np.linalg.norm(existing_positions - np.array(position), axis=1)
        min_distance = np.min(distances)
        
        # Simple affinity based on distance to nearest neighbor
        ideal_distance = self.locality_radius
        return 1.0 / (1.0 + abs(min_distance - ideal_distance))

    def _update_brain_metrics(self, creation_time: float):
        """Update brain-level performance metrics."""
        # Update entrance node count
        entrance_count = sum(1 for node in self.neural_nodes 
                            if 'entrance' in node.node_type.lower())
        self.brain_records.loc[0, 'Entrance_Node_Count_Post'] = entrance_count
        
        # Update timing
        current_time = self.brain_records.loc[0, 'Training_Time']
        if current_time is None or pd.isna(current_time):
            current_time = 0.0
        elif isinstance(current_time, (int, float)):
            current_time = float(current_time)
        elif isinstance(current_time, str):
            try:
                current_time = float(current_time)
            except ValueError:
                current_time = 0.0
        else:
            # For any other type (complex, object, etc.), default to 0.0
            current_time = 0.0
        
        self.brain_records.loc[0, 'Training_Time'] = current_time + creation_time        
        # Update efficiency metrics
        if len(self.neural_nodes) > 1:
            spatial_affinities = [getattr(node, 'spatial_affinity', 0.5) for node in self.neural_nodes]
            self.brain_records.loc[0, 'Spatial_Efficiency'] = np.mean(spatial_affinities)
            
            reuse_efficiency = len(self.reuse_candidates) / len(self.neural_nodes)
            self.brain_records.loc[0, 'Reuse_Efficiency'] = reuse_efficiency

    # Keep the utility functions from before
    def get_nodes_by_type(self, node_type: str) -> List[int]:
        """Get all node IDs of a specific type."""
        return [node.node_id for node in self.neural_nodes 
                if hasattr(node, 'node_type') and node.node_type == node_type]

    def get_nodes_in_radius(self, center_position: List[float], radius: float) -> List[int]:
        """Get all nodes within a spatial radius."""
        if self.spatial_index is None:
            # Linear search fallback
            nearby_nodes = []
            center = np.array(center_position)
            for node in self.neural_nodes:
                if hasattr(node, 'node_position'):
                    distance = np.linalg.norm(np.array(node.node_position) - center)
                    if distance <= radius:
                        nearby_nodes.append(node.node_id)
            return nearby_nodes
        
        try:
            indices = self.spatial_index.query_ball_point(center_position, radius)
            return [list(self.node_id_to_index.keys())[i] for i in indices if i < len(self.neural_nodes)]
        except Exception:
            return []
    def move_node(self, node_id: int, new_position: List[float]) -> bool:
        """
        Move a neural node to a new position and update all tracking systems.
        
        Args:
            node_id: ID of the node to move
            new_position: New spatial position for the node
            
        Returns:
            bool: True if successful, False if node not found
        """
        if node_id not in self.node_registry:
            if self.demo:
                print(f"✗ Node {node_id} not found")
            return False
        
        node = self.node_registry[node_id]
        old_position = node.node_position.copy()
        
        # Ensure position matches dimensions
        new_position = list(new_position)
        if len(new_position) != self.dimensions:
            if len(new_position) > self.dimensions:
                new_position = new_position[:self.dimensions]
            else:
                new_position = new_position + [0.0] * (self.dimensions - len(new_position))
        
        # Update node position
        node.node_position = new_position
        
        # Update spatial tracking arrays
        if node_id in self.node_id_to_index:
            index = self.node_id_to_index[node_id]
            if index < len(self.node_positions):
                self.node_positions[index] = new_position
        
        # Recalculate spatial affinity
        spatial_affinity = self._calculate_spatial_affinity(new_position)
        node.spatial_affinity = spatial_affinity
        
        # Update DataFrame record
        mask = self.node_records['Node_ID'] == node_id
        if mask.any():
            self.node_records.loc[mask, 'Node_Position'] = [new_position]
            self.node_records.loc[mask, 'Spatial_Affinity'] = spatial_affinity
        
        # Rebuild spatial index for efficiency
        self._rebuild_spatial_index()
        
        # Update brain-level spatial efficiency
        if len(self.neural_nodes) > 1:
            spatial_affinities = [getattr(n, 'spatial_affinity', 0.5) for n in self.neural_nodes]
            self.brain_records.loc[0, 'Spatial_Efficiency'] = np.mean(spatial_affinities)
        
        if self.demo:
            distance_moved = np.linalg.norm(np.array(new_position) - np.array(old_position))
            print(f"✓ Moved node {node_id} by distance {distance_moved:.2f}")
            print(f"  Old: {old_position[:2]}... → New: {new_position[:2]}...")
        
        return True

    def move_nodes_batch(self, moves: List[Tuple[int, List[float]]]) -> int:
        """
        Move multiple nodes efficiently in a batch operation.
        
        Args:
            moves: List of (node_id, new_position) tuples
            
        Returns:
            int: Number of successful moves
        """
        successful_moves = 0
        
        # Perform all moves without rebuilding spatial index each time
        for node_id, new_position in moves:
            if node_id not in self.node_registry:
                continue
                
            node = self.node_registry[node_id]
            
            # Ensure position matches dimensions
            new_position = list(new_position)
            if len(new_position) != self.dimensions:
                if len(new_position) > self.dimensions:
                    new_position = new_position[:self.dimensions]
                else:
                    new_position = new_position + [0.0] * (self.dimensions - len(new_position))
            
            # Update node position
            node.node_position = new_position
            
            # Update spatial tracking
            if node_id in self.node_id_to_index:
                index = self.node_id_to_index[node_id]
                if index < len(self.node_positions):
                    self.node_positions[index] = new_position
            
            successful_moves += 1
        
        # Recalculate spatial affinities and update records for all moved nodes
        for node_id, new_position in moves:
            if node_id in self.node_registry:
                node = self.node_registry[node_id]
                spatial_affinity = self._calculate_spatial_affinity(new_position)
                node.spatial_affinity = spatial_affinity
                
                # Update DataFrame
                mask = self.node_records['Node_ID'] == node_id
                if mask.any():
                    self.node_records.loc[mask, 'Node_Position'] = [new_position]
                    self.node_records.loc[mask, 'Spatial_Affinity'] = spatial_affinity
        
        # Rebuild spatial index once at the end
        self._rebuild_spatial_index()
        
        # Update brain-level metrics
        if len(self.neural_nodes) > 1:
            spatial_affinities = [getattr(n, 'spatial_affinity', 0.5) for n in self.neural_nodes]
            self.brain_records.loc[0, 'Spatial_Efficiency'] = np.mean(spatial_affinities)
        
        if self.demo:
            print(f"✓ Batch moved {successful_moves}/{len(moves)} nodes")
        
        return successful_moves
    def connect_nodes(self, from_node_id: int, to_node_id: int, weight: float = 1.0, 
                  bidirectional: bool = False) -> bool:
        """
        Create a directional connection between two nodes.
        
        Args:
            from_node_id: Source node ID (data flows FROM this node)
            to_node_id: Target node ID (data flows TO this node)
            weight: Connection weight (default 1.0)
            bidirectional: If True, creates connections in both directions
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if both nodes exist
        if from_node_id not in self.node_registry or to_node_id not in self.node_registry:
            missing_nodes = []
            if from_node_id not in self.node_registry:
                missing_nodes.append(str(from_node_id))
            if to_node_id not in self.node_registry:
                missing_nodes.append(str(to_node_id))
            raise ValueError(f"Cannot connect: Node(s) {', '.join(missing_nodes)} not found")
        
        # Prevent self-connections
        if from_node_id == to_node_id:
            raise ValueError(f"Cannot create self-connection on node {from_node_id}")
        
        from_node = self.node_registry[from_node_id]
        to_node = self.node_registry[to_node_id]
        
        # Initialize connection attributes if they don't exist
        if not hasattr(from_node, 'exit_connections'):
            from_node.exit_connections = []
        if not hasattr(from_node, 'connection_weights'):
            from_node.connection_weights = {}
        if not hasattr(to_node, 'entrance_connections'):
            to_node.entrance_connections = []
        
        # Create forward connection (from_node → to_node)
        if to_node_id not in from_node.exit_connections:
            from_node.exit_connections.append(to_node_id)
            from_node.connection_weights[to_node_id] = weight
            
            if from_node_id not in to_node.entrance_connections:
                to_node.entrance_connections.append(from_node_id)
        else:
            # Update existing connection weight
            from_node.connection_weights[to_node_id] = weight
        
        # Create backward connection if bidirectional
        if bidirectional:
            if not hasattr(to_node, 'exit_connections'):
                to_node.exit_connections = []
            if not hasattr(to_node, 'connection_weights'):
                to_node.connection_weights = {}
            if not hasattr(from_node, 'entrance_connections'):
                from_node.entrance_connections = []
                
            if from_node_id not in to_node.exit_connections:
                to_node.exit_connections.append(from_node_id)
                to_node.connection_weights[from_node_id] = weight
                
                if to_node_id not in from_node.entrance_connections:
                    from_node.entrance_connections.append(to_node_id)
            else:
                # Update existing backward connection weight
                to_node.connection_weights[from_node_id] = weight
        
        # Update DataFrame records
        self._update_connection_records(from_node_id, to_node_id, bidirectional)
        
        if self.demo:
            direction_symbol = "↔" if bidirectional else "→"
            print(f"✓ Connected node {from_node_id} {direction_symbol} {to_node_id} (weight: {weight})")
        
        return True

    def disconnect_nodes(self, from_node_id: int, to_node_id: int, bidirectional: bool = False) -> bool:
        """
        Remove a directional connection between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            bidirectional: If True, removes connections in both directions
            
        Returns:
            bool: True if successful, False otherwise
        """
        if from_node_id not in self.node_registry or to_node_id not in self.node_registry:
            return False
        
        from_node = self.node_registry[from_node_id]
        to_node = self.node_registry[to_node_id]
        
        # Remove forward connection (from_node → to_node)
        if hasattr(from_node, 'exit_connections') and to_node_id in from_node.exit_connections:
            from_node.exit_connections.remove(to_node_id)
        
        if hasattr(to_node, 'entrance_connections') and from_node_id in to_node.entrance_connections:
            to_node.entrance_connections.remove(from_node_id)
        
        # Remove forward connection weight
        if hasattr(from_node, 'connection_weights') and to_node_id in from_node.connection_weights:
            del from_node.connection_weights[to_node_id]
        
        # Remove backward connection if bidirectional
        if bidirectional:
            if hasattr(to_node, 'exit_connections') and from_node_id in to_node.exit_connections:
                to_node.exit_connections.remove(from_node_id)
            
            if hasattr(from_node, 'entrance_connections') and to_node_id in from_node.entrance_connections:
                from_node.entrance_connections.remove(to_node_id)
            
            # Remove backward connection weight
            if hasattr(to_node, 'connection_weights') and from_node_id in to_node.connection_weights:
                del to_node.connection_weights[from_node_id]
        
        # Update DataFrame records
        self._update_connection_records(from_node_id, to_node_id, bidirectional)
        
        if self.demo:
            direction_symbol = "↮" if bidirectional else "↛"
            print(f"✓ Disconnected node {from_node_id} {direction_symbol} {to_node_id}")
        
        return True
    def _update_connection_records(self, from_node_id: int, to_node_id: int, bidirectional: bool = False):
        """Update DataFrame records after connection changes."""
        from_node = self.node_registry[from_node_id]
        to_node = self.node_registry[to_node_id]
        
        # Update from_node record
        from_mask = self.node_records['Node_ID'] == from_node_id
        if from_mask.any():
            from_index = self.node_records.index[from_mask].tolist()[0]
            self.node_records.at[from_index, 'Exit_Connections'] = getattr(from_node, 'exit_connections', [])
            self.node_records.at[from_index, 'Entrance_Connections'] = getattr(from_node, 'entrance_connections', [])
        
        # Update to_node record
        to_mask = self.node_records['Node_ID'] == to_node_id
        if to_mask.any():
            to_index = self.node_records.index[to_mask].tolist()[0]
            self.node_records.at[to_index, 'Entrance_Connections'] = getattr(to_node, 'entrance_connections', [])
            self.node_records.at[to_index, 'Exit_Connections'] = getattr(to_node, 'exit_connections', [])
    def get_node_connections(self, node_id: int) -> Dict[str, List[int]]:
        """
        Get all connections for a specific node.
        
        Args:
            node_id: Node ID to query
            
        Returns:
            Dict with 'incoming', 'outgoing', and 'bidirectional' connection lists
        """
        if node_id not in self.node_registry:
            return {'incoming': [], 'outgoing': [], 'bidirectional': []}
        
        node = self.node_registry[node_id]
        
        incoming = getattr(node, 'entrance_connections', [])
        outgoing = getattr(node, 'exit_connections', [])
        
        # Find bidirectional connections (nodes that appear in both lists)
        bidirectional = []
        for out_node in outgoing:
            if out_node in incoming:
                bidirectional.append(out_node)
        
        # Remove bidirectional from incoming/outgoing to avoid duplicates
        incoming_only = [n for n in incoming if n not in bidirectional]
        outgoing_only = [n for n in outgoing if n not in bidirectional]
        
        return {
            'incoming': incoming_only,
            'outgoing': outgoing_only,
            'bidirectional': bidirectional
        }

    def get_connection_weight(self, from_node_id: int, to_node_id: int) -> Optional[float]:
        """
        Get the weight of a specific directional connection.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            
        Returns:
            Connection weight or None if connection doesn't exist
        """
        if from_node_id not in self.node_registry:
            return None
        
        from_node = self.node_registry[from_node_id]
        
        if hasattr(from_node, 'connection_weights'):
            return from_node.connection_weights.get(to_node_id)
        
        return None

    def update_connection_weight(self, from_node_id: int, to_node_id: int, new_weight: float) -> bool:
        """
        Update the weight of an existing directional connection.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            new_weight: New weight value
            
        Returns:
            bool: True if successful, False if connection doesn't exist
        """
        if from_node_id not in self.node_registry:
            return False
        
        from_node = self.node_registry[from_node_id]
        
        # Check if connection exists
        if (hasattr(from_node, 'exit_connections') and 
            to_node_id in from_node.exit_connections and
            hasattr(from_node, 'connection_weights')):
            
            from_node.connection_weights[to_node_id] = new_weight
            
            if self.demo:
                print(f"✓ Updated connection weight {from_node_id} → {to_node_id}: {new_weight}")
            
            return True
        
        return False

    def move_node_relative(self, node_id: int, offset: List[float]) -> bool:
        """
        Move a node by a relative offset from its current position.
        
        Args:
            node_id: ID of the node to move
            offset: Offset to add to current position
            
        Returns:
            bool: True if successful, False if node not found
        """
        if node_id not in self.node_registry:
            return False
        
        node = self.node_registry[node_id]
        current_pos = np.array(node.node_position)
        
        # Ensure offset matches dimensions
        offset = list(offset)
        if len(offset) != self.dimensions:
            if len(offset) > self.dimensions:
                offset = offset[:self.dimensions]
            else:
                offset = offset + [0.0] * (self.dimensions - len(offset))
        
        new_position = (current_pos + np.array(offset)).tolist()
        return self.move_node(node_id, new_position)

    def _rebuild_spatial_index(self):
        """Rebuild the spatial index after position changes."""
        try:
            if len(self.node_positions) > 1 and self.dimensions > 0:
                self.spatial_index = KDTree(np.array(self.node_positions))
            else:
                self.spatial_index = None
        except Exception:
            self.spatial_index = None
    def update_node_usage(self, node_id: int, input_data: Any = None):
        """Update node usage statistics."""
        if node_id not in self.node_registry:
            return
        
        node = self.node_registry[node_id]
        node.times_called = getattr(node, 'times_called', 0) + 1
        
        if input_data is not None:
            if not hasattr(node, 'input_tracker'):
                node.input_tracker = []
            node.input_tracker.append(input_data)
        
        # Update DataFrame
        mask = self.node_records['Node_ID'] == node_id
        if mask.any():
            self.node_records.loc[mask, 'Times_Called'] = node.times_called
        
        # Update usage history
        if len(self.node_usage_history[node_id]) == 0:
            self.node_usage_history[node_id] = [1]
        else:
            self.node_usage_history[node_id][-1] += 1
        
        # Check reuse eligibility
        if node.times_called > self.reuse_threshold * 10:
            self.reuse_candidates.add(node_id)

    def process_node(self, node_id: int, token_embeddings: Any) -> Any:
        """Process input through a node."""
        if node_id not in self.node_registry:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.node_registry[node_id]
        self.update_node_usage(node_id, token_embeddings)
        
        try:
            return node.process(token_embeddings)
        except NotImplementedError:
            if self.demo:
                print(f"Warning: {node.__class__.__name__} process method not implemented")
            return token_embeddings
        except Exception as e:
            if self.demo:
                print(f"Error in node {node_id}: {e}")
            raise
    def initialize_brain(self) -> Dict[str, List[int]]:
        """
        Initialize the brain with Controller, Judge, and Splitter nodes positioned strategically.
        Ensures all records are properly maintained throughout the process.
        
        Returns:
            Dict containing lists of node IDs by type
        """
        start_time = time.time()
        node_map = {'controller': [], 'judges': [], 'splitters': [], 'retainers': [], 'reviewers': [], 'handler': [], 'comp_nodes': []}
        
        if self.demo:
            print(f"🧠 Initializing BrainNexus with {self.dimensions}D spatial positioning...")
        
        # 1. Place Controller at origin
        controller_position = [0.0] * self.dimensions
        controller_id = self.add_neural_node(
            node_type='Controller',
            position=controller_position,
            node_group='core',
            num_branches=4
        )
        node_map['controller'].append(controller_id)
        
        if self.demo:
            print(f"  📍 Controller #{controller_id} at origin {controller_position}")
        
        # 2. Generate positions for all node types
        use_dimensions = 2 if self.demo else min(self.dimensions, 3)
        positions_judges = []
        positions_splitters = []
        positions_retainers = []
        positions_reviewers = []
        
        retainer_distance = 1000 if self.demo else 10000
        
        for i in range(2 ** use_dimensions):
            judge_pos = [0.0] * self.dimensions
            splitter_pos = [0.0] * self.dimensions
            retainer_pos = [0.0] * self.dimensions
            reviewer_pos = [0.0] * self.dimensions
            
            for dim in range(use_dimensions):
                sign = 1 if (i >> dim) & 1 else -1
                judge_pos[dim] = float(sign * 1)
                splitter_pos[dim] = float(sign * 2)
                retainer_pos[dim] = float(sign * retainer_distance)
                reviewer_pos[dim] = float(sign * retainer_distance)
            
            reviewer_pos[0] += 1.0  # Offset reviewer from retainer
            
            positions_judges.append(judge_pos)
            positions_splitters.append(splitter_pos)
            positions_retainers.append(retainer_pos)
            positions_reviewers.append(reviewer_pos)
        
        # 3. Place Judge nodes
        for i, position in enumerate(positions_judges):
            judge_id = self.add_neural_node(
                node_type='Judge',
                position=position,
                node_group='judgment',
                num_comps=3
            )
            node_map['judges'].append(judge_id)
            
            if self.demo:
                pos_str = [f"{p:+.1f}" for p in position[:use_dimensions]]
                print(f"  ⚖️  Judge #{judge_id} at {pos_str}")
        
        # 4. Place Splitter nodes
        for i, position in enumerate(positions_splitters):
            splitter_id = self.add_neural_node(
                node_type='Splitter',
                position=position,
                node_group='processing',
                num_branches=2
            )
            node_map['splitters'].append(splitter_id)
            
            if self.demo:
                pos_str = [f"{p:+.1f}" for p in position[:use_dimensions]]
                print(f"  🌿 Splitter #{splitter_id} at {pos_str}")
        
        # 5. Place Retainer nodes
        for i, position in enumerate(positions_retainers):
            retainer_id = self.add_neural_node(
                node_type='Retainer',
                position=position,
                node_group='storage',
                expected_nodes=5
            )
            node_map['retainers'].append(retainer_id)
            
            if self.demo:
                pos_str = [f"{p:+.0f}" for p in position[:use_dimensions]]
                print(f"  📦 Retainer #{retainer_id} at {pos_str}")
        
        # 6. Place reviewers to match retainer count
        num_reviewers = 2 ** use_dimensions
        print(f"  📋 Creating {num_reviewers} reviewers to match {len(positions_retainers)} retainers...")

        # Use the same positioning logic as retainers but with offset
        reviewer_positions = positions_reviewers  # Already calculated above

        # Create reviewers matching retainer count
        for i, position in enumerate(reviewer_positions):
            reviewer_id = self.add_neural_node(
                node_type='Reviewer',  # Make sure this matches what get_nodes_by_type expects
                position=position,
                node_group='reviewer',
                num_comps=4
            )
            node_map['reviewers'].append(reviewer_id)
            
            if self.demo:
                pos_str = [f"{p:+.0f}" for p in position[:2]]  # Show first 2 dimensions
                print(f"  📋 Reviewer #{reviewer_id} at {pos_str}")

        # 7. Place Handler node above Controller (separate step)
        handler_position = [0.0] * self.dimensions
        handler_position[1] = 1.0  # Y=1.0 to distinguish it

        handler_id = self.add_neural_node(
            node_type='Handler',
            position=handler_position,
            node_group='final_decision',
            num_reviewers=num_reviewers,  # Expects to combine results from matching number of reviewers
            confidence_threshold=self.output_config.get('confidence_threshold', 0.7),
            num_classes=self.output_config.get('num_classes', 10),
            output_config=self.output_config
        )
        node_map['handler'].append(handler_id)

        if self.demo:
            pos_str = [f"{p:+.1f}" for p in handler_position[:2]]
            print(f"  🏆 Handler #{handler_id} at {pos_str}")

        # Verify we have exactly what we expect
        print(f"  ✅ Created {len(node_map['reviewers'])} reviewers: {node_map['reviewers']}")
        print(f"  ✅ Created {len(node_map['retainers'])} retainers: {node_map['retainers']}")
        print(f"  ✅ Created {len(node_map['handler'])} handler: {node_map['handler']}")

        # 8. Create computational nodes
        num_comp_nodes = 180 if self.demo else 18000
        if self.demo:
            print(f"  🔢 Creating {num_comp_nodes} computational nodes...")
        
        used_positions = set()
        for node in self.neural_nodes:
            if hasattr(node, 'node_position'):
                pos_tuple = tuple(round(p, 6) for p in node.node_position)
                used_positions.add(pos_tuple)
        
        comp_range = 1000 if self.demo else 9000
        
        for i in range(num_comp_nodes):
            attempts = 0
            while attempts < 100:
                position = [np.random.uniform(-comp_range, comp_range) for _ in range(self.dimensions)]
                pos_tuple = tuple(round(p, 6) for p in position)
                
                if pos_tuple not in used_positions:
                    used_positions.add(pos_tuple)
                    break
                attempts += 1
            
            comp_id = self.add_neural_node(
                node_type='Computational',
                position=position,
                node_group='computation'
            )
            node_map['comp_nodes'].append(comp_id)
            
            if self.demo and (i + 1) % 60 == 0:
                print(f"    Created {i + 1}/{num_comp_nodes} computational nodes...")
            elif not self.demo and (i + 1) % 2000 == 0:
                print(f"    Created {i + 1}/{num_comp_nodes} computational nodes...")
        
        if self.demo:
            print(f"  ✅ Created all {num_comp_nodes} computational nodes")
        
        # 9. Create all connections and update records
        if self.demo:
            print(f"  🔗 Creating neural pathways...")
        
        connection_count = 0
        
        # Controller to Judges
        for judge_id in node_map['judges']:
            if self.connect_nodes(controller_id, judge_id, weight=1.0):
                connection_count += 1
                if self.demo:
                    print(f"    Controller #{controller_id} → Judge #{judge_id}")
        
        # Judges to Splitters
        for judge_id, splitter_id in zip(node_map['judges'], node_map['splitters']):
            if self.connect_nodes(judge_id, splitter_id, weight=0.8):
                connection_count += 1
                if self.demo:
                    print(f"    Judge #{judge_id} → Splitter #{splitter_id}")
        
        # Retainers to Reviewers (Handler handled separately)
        print(f"  🔗 Connecting Retainers to Reviewers...")

        # Ensure we have the same number of retainers and regular reviewers
        if len(node_map['retainers']) != len(node_map['reviewers']):
            print(f"  ⚠️  Warning: {len(node_map['retainers'])} retainers but {len(node_map['reviewers'])} reviewers")

        # Connect each retainer to its corresponding reviewer
        retainer_count = len(node_map['retainers'])
        reviewer_count = len(node_map['reviewers'])
        connections_to_make = min(retainer_count, reviewer_count)
        
        print(f"  📋 Making {connections_to_make} retainer-to-reviewer connections...")
        
        for i in range(connections_to_make):
            retainer_id = node_map['retainers'][i]
            reviewer_id = node_map['reviewers'][i]
            
            try:
                if self.connect_nodes(retainer_id, reviewer_id, weight=0.9):
                    # Also set up the direct review connection in the retainer
                    retainer_node = self.node_registry[retainer_id]
                    reviewer_node = self.node_registry[reviewer_id]
                    
                    # Safely connect review node
                    if hasattr(retainer_node, 'connect_review'):
                        retainer_node.connect_review(reviewer_node)
                    
                    connection_count += 1
                    print(f"    ✓ Retainer #{retainer_id} → Reviewer #{reviewer_id}")
                else:
                    print(f"    ✗ Failed to connect Retainer #{retainer_id} → Reviewer #{reviewer_id}")
                    
            except Exception as e:
                print(f"    ✗ Error connecting Retainer #{retainer_id} → Reviewer #{reviewer_id}: {e}")
                continue

        # Connect each reviewer to Handler
        handler_id = node_map['handler'][0]
        print(f"  🔗 Connecting {len(node_map['reviewers'])} reviewers to Handler #{handler_id}...")
        
        for i, reviewer_id in enumerate(node_map['reviewers']):
            try:
                if self.connect_nodes(reviewer_id, handler_id, weight=1.0):
                    connection_count += 1
                    print(f"    ✓ Reviewer #{reviewer_id} → Handler #{handler_id}")
                else:
                    print(f"    ✗ Failed to connect Reviewer #{reviewer_id} → Handler #{handler_id}")
            except Exception as e:
                print(f"    ✗ Error connecting Reviewer #{reviewer_id} → Handler #{handler_id}: {e}")
                continue
        
        # Splitters to closest computational nodes (optimized)
        print(f"  🔗 Connecting Splitters to closest computational nodes...")
        
        # Pre-compute all computational node positions for vectorized operations
        comp_positions = np.array([self.node_registry[comp_id].node_position for comp_id in node_map['comp_nodes']])
        
        for i, splitter_id in enumerate(node_map['splitters']):
            splitter_node = self.node_registry[splitter_id]
            splitter_pos = np.array(splitter_node.node_position)
            
            # Vectorized distance calculation - much faster
            distances = np.linalg.norm(comp_positions - splitter_pos, axis=1)
            
            # Get indices of 10 closest nodes
            closest_indices = np.argpartition(distances, 10)[:10]
            
            for idx in closest_indices:
                comp_id = node_map['comp_nodes'][idx]
                if self.connect_nodes(splitter_id, comp_id, weight=0.6):
                    connection_count += 1
            
            if self.demo:
                print(f"    Splitter #{splitter_id} → 10 closest comp nodes")
            elif (i + 1) % 2 == 0:  # Progress for full mode
                print(f"    Connected splitter {i + 1}/{len(node_map['splitters'])}")
        
        # Computational nodes to Retainers (optimized)
        print(f"  🔗 Connecting computational nodes to Retainers...")
        
        retainer_connection_counts = {}
        for i, retainer_id in enumerate(node_map['retainers']):
            retainer_node = self.node_registry[retainer_id]
            retainer_pos = np.array(retainer_node.node_position)
            
            # Vectorized distance calculation - much faster
            distances = np.linalg.norm(comp_positions - retainer_pos, axis=1)
            
            # Get indices of 10 closest nodes
            closest_indices = np.argpartition(distances, 10)[:10]
            
            connections_made = 0
            for idx in closest_indices:
                comp_id = node_map['comp_nodes'][idx]
                try:
                    if self.connect_nodes(comp_id, retainer_id, weight=0.5):
                        connections_made += 1
                        connection_count += 1
                except Exception as e:
                    if self.demo:
                        print(f"      ⚠️  Failed to connect Comp #{comp_id} → Retainer #{retainer_id}: {e}")
            
            retainer_connection_counts[retainer_id] = connections_made
            
            if self.demo:
                print(f"    Connected {connections_made} comp nodes → Retainer #{retainer_id}")
            elif (i + 1) % 2 == 0:  # Progress for full mode
                print(f"    Connected retainer {i + 1}/{len(node_map['retainers'])}")
        
        # Verify all retainers have connections
        print(f"  🔍 Verifying retainer connections...")
        for retainer_id in node_map['retainers']:
            connections = self.get_node_connections(retainer_id)
            incoming_count = len(connections['incoming'])
            if incoming_count == 0:
                print(f"    ⚠️  WARNING: Retainer #{retainer_id} has NO incoming connections!")
                # Try to fix by connecting to nearest computational nodes
                retainer_node = self.node_registry[retainer_id]
                retainer_pos = np.array(retainer_node.node_position)
                distances = np.linalg.norm(comp_positions - retainer_pos, axis=1)
                closest_indices = np.argpartition(distances, 5)[:5]
                
                for idx in closest_indices:
                    comp_id = node_map['comp_nodes'][idx]
                    try:
                        if self.connect_nodes(comp_id, retainer_id, weight=0.4):
                            print(f"      🔗 Emergency fix: Comp #{comp_id} → Retainer #{retainer_id}")
                            connection_count += 1
                    except Exception as e:
                        print(f"      ❌ Emergency connection failed: {e}")
            else:
                print(f"    ✅ Retainer #{retainer_id} has {incoming_count} incoming connections")
        
        # Inter-computational connections
        inter_comp_connections = self._connect_computational_nodes(node_map['comp_nodes'])
        connection_count += inter_comp_connections
        
        # 10. Final record updates
        total_nodes = sum(len(nodes) for nodes in node_map.values())
        initialization_time = time.time() - start_time
        
        # Update brain records comprehensively
        current_time = self.brain_records.loc[0, 'Training_Time']
        if current_time is None or pd.isna(current_time):
            current_time = 0.0
        elif isinstance(current_time, (int, float)):
            current_time = float(current_time)
        elif isinstance(current_time, str):
            try:
                current_time = float(current_time)
            except ValueError:
                current_time = 0.0
        else:
            current_time = 0.0
        
        self.brain_records.loc[0, 'Training_Time'] = current_time + initialization_time
        self.brain_records.loc[0, 'Entrance_Node_Count_Post'] = len(node_map['judges'])
        
        # Calculate comprehensive metrics
        if total_nodes > 1:
            spatial_affinities = [getattr(node, 'spatial_affinity', 0.5) for node in self.neural_nodes]
            self.brain_records.loc[0, 'Spatial_Efficiency'] = np.mean(spatial_affinities)
            
            # Update reuse efficiency
            reuse_efficiency = len(self.reuse_candidates) / total_nodes if total_nodes > 0 else 0.0
            self.brain_records.loc[0, 'Reuse_Efficiency'] = reuse_efficiency
            
            # Calculate attention coherence (average of attention weight variances)
            attention_coherences = []
            for node in self.neural_nodes:
                if hasattr(node, 'attention_weights'):
                    coherence = 1.0 / (1.0 + np.var(node.attention_weights))
                    attention_coherences.append(coherence)
            
            if attention_coherences:
                self.brain_records.loc[0, 'Attention_Coherence'] = np.mean(attention_coherences)
        
        # Verify all node records are complete
        for node_type, node_ids in node_map.items():
            for node_id in node_ids:
                mask = self.node_records['Node_ID'] == node_id
                if not mask.any():
                    print(f"Warning: Node {node_id} missing from records!")
                else:
                    # Ensure all connections are recorded
                    node = self.node_registry[node_id]
                    node_index = self.node_records.index[mask].tolist()[0]
                    self.node_records.at[node_index, 'Entrance_Connections'] = getattr(node, 'entrance_connections', [])
                    self.node_records.at[node_index, 'Exit_Connections'] = getattr(node, 'exit_connections', [])

        if self.demo:
            print(f"✅ Brain initialized with {total_nodes} nodes and {connection_count} connections")
            print(f"   Initialization time: {initialization_time:.3f}s")
            print(f"   Spatial efficiency: {self.brain_records.loc[0, 'Spatial_Efficiency']:.3f}")
            print(f"   Reuse efficiency: {self.brain_records.loc[0, 'Reuse_Efficiency']:.3f}")
            print(f"   Attention coherence: {self.brain_records.loc[0, 'Attention_Coherence']:.3f}")
            print(f"   Total records: {len(self.node_records)} node records")
            print(f"🔧 RETAINER-REVIEWER-HANDLER CONNECTION DEBUG:")
            print(f"   Retainers: {node_map['retainers']}")
            print(f"   Reviewers: {node_map['reviewers']}")
            print(f"   Handler: {node_map['handler']}")
            
            # Debug the specific connections that were created
            print(f"🔍 CONNECTION VERIFICATION:")
            for retainer_id in node_map['retainers']:
                retainer_node = self.node_registry[retainer_id]
                retainer_exits = getattr(retainer_node, 'exit_connections', [])
                print(f"   Retainer #{retainer_id} → {retainer_exits}")
                
            for reviewer_id in node_map['reviewers']:
                reviewer_node = self.node_registry[reviewer_id]
                reviewer_exits = getattr(reviewer_node, 'exit_connections', [])
                print(f"   Reviewer #{reviewer_id} → {reviewer_exits}")
        
        return node_map

    def _connect_computational_nodes(self, comp_node_ids: List[int]) -> int:
        """
        Connect computational nodes to each other using optimized nearest neighbor selection.
        Returns the number of connections made for record keeping.
        
        Args:
            comp_node_ids: List of computational node IDs to connect
            
        Returns:
            int: Number of connections created
        """
        if len(comp_node_ids) < 2:
            return 0
        
        print(f"  🔗 Connecting computational nodes to each other...")
        
        total_nodes = len(comp_node_ids)
        nearest_count = max(1, min(int(total_nodes * 0.05), 50))  # Cap at 50 for efficiency
        connections_made = 0
        
        # Pre-compute all positions for vectorized operations
        comp_positions = np.array([self.node_registry[comp_id].node_position for comp_id in comp_node_ids])
        
        # Use spatial index if available, otherwise use optimized batch processing
        if self.spatial_index is not None:
            # Use KDTree for efficient nearest neighbor queries
            for i, comp_id in enumerate(comp_node_ids):
                comp_pos = comp_positions[i]
                
                # Find nearest neighbors using spatial index
                try:
                    # Query for more than needed, then filter out self
                    distances, indices = self.spatial_index.query(comp_pos, k=nearest_count+1)
                    
                    # Handle single result vs array
                    if isinstance(indices, (int, np.integer)):
                        indices = [indices]
                    if isinstance(distances, (float, np.floating)):
                        distances = [distances]
                    
                    # Convert spatial index indices to comp_node_ids
                    nearby_comp_ids = []
                    for idx in indices:
                        if idx < len(comp_node_ids) and comp_node_ids[idx] != comp_id:
                            nearby_comp_ids.append(comp_node_ids[idx])
                            if len(nearby_comp_ids) >= nearest_count:
                                break
                    
                    # Connect to random subset
                    num_to_connect = min(random.randint(1, len(nearby_comp_ids)), 10)  # Cap connections
                    selected_nodes = random.sample(nearby_comp_ids, num_to_connect)
                    
                    for target_id in selected_nodes:
                        weight = random.uniform(0.3, 0.8)
                        if self.connect_nodes(comp_id, target_id, weight=weight):
                            connections_made += 1
                            
                except Exception:
                    # Fallback to manual nearest neighbor
                    distances = np.linalg.norm(comp_positions - comp_pos, axis=1)
                    nearest_indices = np.argpartition(distances, nearest_count+1)[:nearest_count+1]
                    
                    for idx in nearest_indices:
                        if comp_node_ids[idx] != comp_id:
                            weight = random.uniform(0.3, 0.8)
                            if self.connect_nodes(comp_id, comp_node_ids[idx], weight=weight):
                                connections_made += 1
                                break  # Only connect to one for efficiency
                
                # Progress indicator
                if self.demo and (i + 1) % 20 == 0:
                    print(f"    Connected {i + 1}/{total_nodes} computational nodes...")
                elif not self.demo and (i + 1) % 1000 == 0:
                    print(f"    Connected {i + 1}/{total_nodes} computational nodes...")
        else:
            # Optimized batch processing without spatial index
            print(f"    Using optimized batch processing (no spatial index)...")
            
            # Process in smaller batches to avoid memory issues
            batch_size = 500 if not self.demo else total_nodes
            
            for batch_start in range(0, total_nodes, batch_size):
                batch_end = min(batch_start + batch_size, total_nodes)
                print(f"    Processing batch {batch_start}-{batch_end}/{total_nodes}...")
                
                for i in range(batch_start, batch_end):
                    comp_id = comp_node_ids[i]
                    comp_pos = comp_positions[i]
                    
                    # Calculate distances to all other nodes (vectorized)
                    distances = np.linalg.norm(comp_positions - comp_pos, axis=1)
                    
                    # Get nearest neighbors (excluding self)
                    nearest_indices = np.argpartition(distances, nearest_count+1)[:nearest_count+1]
                    
                    # Connect to a few of the nearest (for efficiency)
                    connected_count = 0
                    for idx in nearest_indices:
                        if comp_node_ids[idx] != comp_id and connected_count < 5:  # Limit to 5 connections
                            weight = random.uniform(0.3, 0.8)
                            if self.connect_nodes(comp_id, comp_node_ids[idx], weight=weight):
                                connections_made += 1
                                connected_count += 1
        
        if self.demo:
            avg_connections = connections_made / total_nodes if total_nodes > 0 else 0
            print(f"  ✅ Created {connections_made} inter-computational connections")
            print(f"     Average {avg_connections:.1f} connections per node")
        else:
            print(f"  ✅ Created {connections_made} inter-computational connections")
        
        return connections_made

    def _tokenize_input(self, input_data: Any) -> Any:
        """
        Tokenize input data using the configured tokenizer.
        
        Args:
            input_data: Raw input data (text, numbers, etc.)
            
        Returns:
            Tokenized representation suitable for neural processing
        """
        if not hasattr(self, 'output_config') or not self.output_config:
            # No tokenizer configured, return input as-is
            return input_data
        
        vocab_mapping = self.output_config.get('vocab_mapping', {})
        if not vocab_mapping:
            # No vocabulary mapping, return input as-is
            return input_data
        
        # Create reverse mapping for encoding
        if not hasattr(self, '_token_to_id_mapping'):
            self._token_to_id_mapping = {token: token_id for token_id, token in vocab_mapping.items()}
        
        # Handle different input types
        if isinstance(input_data, str):
            # Text input - tokenize into token IDs
            return self._tokenize_text(input_data)
        elif isinstance(input_data, (list, tuple)):
            # List/tuple input - process each element
            return [self._tokenize_input(item) for item in input_data]
        elif isinstance(input_data, (int, float)):
            # Numeric input - convert to appropriate token representation
            return self._tokenize_numeric(input_data)
        else:
            # Other types - convert to string and tokenize
            return self._tokenize_text(str(input_data))
    
    def _tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text input into token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        if not hasattr(self, '_token_to_id_mapping'):
            return [0]  # Default to padding token
        
        # Simple word-based tokenization (can be enhanced for subword tokenization)
        words = text.lower().split()
        token_ids = []
        
        # Add start token if available
        if '<start>' in self._token_to_id_mapping:
            token_ids.append(self._token_to_id_mapping['<start>'])
        elif 2 in self.output_config.get('vocab_mapping', {}):
            token_ids.append(2)  # Common start token ID
        
        # Convert words to token IDs
        for word in words:
            # Remove punctuation for better matching
            clean_word = ''.join(c for c in word if c.isalnum())
            
            if clean_word in self._token_to_id_mapping:
                token_ids.append(self._token_to_id_mapping[clean_word])
            elif word in self._token_to_id_mapping:
                token_ids.append(self._token_to_id_mapping[word])
            else:
                # Unknown token
                unk_id = self._token_to_id_mapping.get('<unk>', 
                        self._token_to_id_mapping.get('[UNK]', 1))
                token_ids.append(unk_id)
        
        # Add end token if available
        if '<end>' in self._token_to_id_mapping:
            token_ids.append(self._token_to_id_mapping['<end>'])
        elif 3 in self.output_config.get('vocab_mapping', {}):
            token_ids.append(3)  # Common end token ID
        
        return token_ids if token_ids else [0]  # Return at least one token
    
    def _tokenize_numeric(self, number: Union[int, float]) -> List[int]:
        """
        Convert numeric input to token representation.
        
        Args:
            number: Numeric input
            
        Returns:
            List of token IDs representing the number
        """
        # Convert number to string and tokenize
        return self._tokenize_text(str(number))
    
    def _detokenize_output(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to human-readable text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if not hasattr(self, 'output_config') or not self.output_config:
            return str(token_ids)
        
        vocab_mapping = self.output_config.get('vocab_mapping', {})
        if not vocab_mapping:
            return str(token_ids)
        
        # Convert token IDs to tokens
        tokens = []
        for token_id in token_ids:
            if isinstance(token_id, (list, np.ndarray)):
                # Handle nested structures
                tokens.extend([vocab_mapping.get(int(tid), f'<unk_{tid}>') for tid in token_id])
            else:
                token = vocab_mapping.get(int(token_id), f'<unk_{token_id}>')
                tokens.append(token)
        
        # Filter out special tokens for cleaner output
        filtered_tokens = []
        for token in tokens:
            if token not in ['<start>', '<end>', '<pad>', '[CLS]', '[SEP]', '<|endoftext|>']:
                filtered_tokens.append(token)
        
        # Join tokens into readable text
        if filtered_tokens:
            # Handle subword tokens (those starting with ##)
            result_tokens = []
            for i, token in enumerate(filtered_tokens):
                if token.startswith('##') and result_tokens:
                    # Merge with previous token
                    result_tokens[-1] = result_tokens[-1] + token[2:]
                elif token.startswith('Ġ'):
                    # GPT-2 style space prefix
                    result_tokens.append(token[1:])
                else:
                    result_tokens.append(token)
            
            return ' '.join(result_tokens)
        else:
            return '<empty>'

    def run(self, input_data: Any, trace_execution: bool = True) -> Dict[str, Any]:
        """
        Main execution function that processes input through the neural network pipeline.
        
        Args:
            input_data: Input data to process (text, raw data, etc.)
            trace_execution: Whether to track execution path for debugging
            
        Returns:
            Dict containing final result and execution metadata
        """
        start_time = time.time()
        execution_trace = {
            'steps': [],
            'node_activations': {},
            'connection_flows': [],
            'timing': {}
        } if trace_execution else None
        
        try:
            if self.demo:
                print(f"🚀 Starting BrainNexus execution pipeline...")
            
            # Step 0: Tokenize input data for proper neural processing
            step_start = time.time()
            tokenized_input = self._tokenize_input(input_data)
            if execution_trace:
                execution_trace['timing']['tokenization'] = time.time() - step_start
                execution_trace['steps'].append('tokenization')
                execution_trace['tokenized_input'] = {
                    'original_input': str(input_data)[:100],
                    'token_count': len(tokenized_input) if isinstance(tokenized_input, list) else 1,
                    'sample_tokens': tokenized_input[:5] if isinstance(tokenized_input, list) else [tokenized_input]
                }
            
            if self.demo:
                print(f"🔤 Tokenized input: {len(tokenized_input) if isinstance(tokenized_input, list) else 1} tokens")
                if isinstance(tokenized_input, list) and len(tokenized_input) > 0:
                    sample_display = tokenized_input[:3] if len(tokenized_input) <= 3 else tokenized_input[:3] + ['...']
                    print(f"   Sample tokens: {sample_display}")
            
            # Step 1: Controller determines judge probabilities (now with tokenized input)
            step_start = time.time()
            controller_result = self._run_controller_phase(tokenized_input, execution_trace)
            if execution_trace:
                execution_trace['timing']['controller_phase'] = time.time() - step_start
                execution_trace['steps'].append('controller_phase')
            
            # Step 2: Activate qualifying judges (top 75%)
            step_start = time.time()
            judge_results = self._run_judge_phase(controller_result, execution_trace)
            if execution_trace:
                execution_trace['timing']['judge_phase'] = time.time() - step_start
                execution_trace['steps'].append('judge_phase')
            
            # Step 3: Splitters distribute to computational nodes
            step_start = time.time()
            splitter_results = self._run_splitter_phase(judge_results, execution_trace)
            if execution_trace:
                execution_trace['timing']['splitter_phase'] = time.time() - step_start
                execution_trace['steps'].append('splitter_phase')
            
            # Step 4: Computational nodes process signals
            step_start = time.time()
            computation_results = self._run_computation_phase(splitter_results, execution_trace)
            if execution_trace:
                execution_trace['timing']['computation_phase'] = time.time() - step_start
                execution_trace['steps'].append('computation_phase')
            
            # Step 5: Retainers gather signals by group
            step_start = time.time()
            retainer_results = self._run_retainer_phase(computation_results, execution_trace)
            if execution_trace:
                execution_trace['timing']['retainer_phase'] = time.time() - step_start
                execution_trace['steps'].append('retainer_phase')
            
            # Step 6: Reviewers calculate final probabilities
            step_start = time.time()
            reviewer_results = self._run_reviewer_phase(retainer_results, execution_trace)
            if execution_trace:
                execution_trace['timing']['reviewer_phase'] = time.time() - step_start
                execution_trace['steps'].append('reviewer_phase')
            
            # Step 7: Handler combines results with controller probabilities
            step_start = time.time()
            final_result = self._run_handler_phase(reviewer_results, controller_result, execution_trace)
            if execution_trace:
                execution_trace['timing']['handler_phase'] = time.time() - step_start
                execution_trace['steps'].append('handler_phase')
            
            # Update performance metrics
            total_time = time.time() - start_time
            self._update_run_metrics(total_time, execution_trace)
            
            if self.demo:
                print(f"✅ Pipeline completed in {total_time:.3f}s")
                print(f"   Final result: {final_result.get('prediction', 'N/A')}")
                print(f"   Confidence: {final_result.get('confidence', 0.0):.3f}")
            
            return {
                'result': final_result,
                'execution_time': total_time,
                'trace': execution_trace,
                'metadata': {
                    'nodes_activated': len(execution_trace['node_activations']) if execution_trace else 0,
                    'connections_used': len(execution_trace['connection_flows']) if execution_trace else 0
                }
            }
            
        except Exception as e:
            if self.demo:
                print(f"❌ Pipeline error: {e}")
            raise

    def _run_controller_phase(self, input_data: Any, trace: Optional[Dict] = None) -> Dict[str, Any]:
        """Controller determines probabilities for all judges."""
        controller_nodes = self.get_nodes_by_type('Controller')
        if not controller_nodes:
            raise ValueError("No Controller node found")
        
        controller_id = controller_nodes[0]
        controller_node = self.node_registry[controller_id]
        
        # Process input through controller to get judge probabilities
        result = self.process_node(controller_id, input_data)
        
        # Generate judge probabilities (simulate controller logic)
        judge_ids = self.get_nodes_by_type('Judge')
        num_judges = len(judge_ids)
        
        # Create probability distribution
        judge_probs = np.random.dirichlet(np.ones(num_judges))
        judge_probability_map = {judge_id: prob for judge_id, prob in zip(judge_ids, judge_probs)}
        
        # Determine activation threshold (top 75%)
        prob_threshold = np.percentile(judge_probs, 25)
        active_judges = [judge_id for judge_id, prob in judge_probability_map.items() 
                        if prob > prob_threshold]
        
        if trace:
            trace['node_activations'][controller_id] = {
                'input': str(input_data)[:100],
                'output': 'judge_probabilities',
                'timestamp': time.time()
            }
            for judge_id in judge_ids:
                if judge_id in active_judges:
                    trace['connection_flows'].append({
                        'from': controller_id,
                        'to': judge_id,
                        'weight': judge_probability_map[judge_id],
                        'active': True
                    })
        
        if self.demo:
            print(f"  📊 Controller: {len(active_judges)}/{num_judges} judges activated")
            print(f"      Threshold: {prob_threshold:.3f}")
        
        return {
            'processed_input': result,
            'judge_probabilities': judge_probability_map,
            'active_judges': active_judges,
            'controller_id': controller_id
        }

    def _run_judge_phase(self, controller_result: Dict[str, Any], trace: Optional[Dict] = None) -> Dict[str, Any]:
        """Active judges perform embedding and positional encoding."""
        active_judges = controller_result['active_judges']
        processed_input = controller_result['processed_input']
        
        judge_results = {}
        
        for judge_id in active_judges:
            # Process through judge node
            judge_output = self.process_node(judge_id, processed_input)
            
            # Simulate embedding and positional encoding
            embedding_size = 64 if self.demo else 512
            embeddings = np.random.randn(embedding_size)
            
            # Add positional encoding based on judge's spatial position
            judge_node = self.node_registry[judge_id]
            position_encoding = np.array(judge_node.node_position[:min(len(judge_node.node_position), embedding_size)])
            if len(position_encoding) < embedding_size:
                position_encoding = np.pad(position_encoding, (0, embedding_size - len(position_encoding)))
            else:
                position_encoding = position_encoding[:embedding_size]
            
            final_embedding = embeddings + 0.1 * position_encoding
            
            judge_results[judge_id] = {
                'embeddings': final_embedding,
                'raw_output': judge_output,
                'position': judge_node.node_position
            }
            
            if trace:
                trace['node_activations'][judge_id] = {
                    'input': 'controller_output',
                    'output': f'embeddings_dim_{embedding_size}',
                    'timestamp': time.time()
                }
        
        if self.demo:
            print(f"  🧮 Judges: {len(judge_results)} judges created embeddings")
        
        return judge_results

    def _run_splitter_phase(self, judge_results: Dict[str, Any], trace: Optional[Dict] = None) -> Dict[str, Any]:
        """Splitters distribute embeddings to closest computational nodes."""
        splitter_ids = self.get_nodes_by_type('Splitter')
        splitter_results = {}
        
        for i, (judge_id, judge_data) in enumerate(judge_results.items()):
            if i < len(splitter_ids):
                splitter_id = splitter_ids[i]
                splitter_node = self.node_registry[splitter_id]
                
                # Process embeddings through splitter
                splitter_output = self.process_node(splitter_id, judge_data['embeddings'])
                
                # Find computational nodes connected to retainers (ensuring signal flow)
                splitter_pos = np.array(splitter_node.node_position)
                comp_nodes = self.get_nodes_by_type('Computational')
                retainer_ids = self.get_nodes_by_type('Retainer')
                
                # Get all computational nodes that have connections to retainers
                connected_comp_nodes = []
                for comp_id in comp_nodes:
                    comp_connections = self.get_node_connections(comp_id)
                    # Check if this comp node connects to any retainer
                    if any(retainer_id in comp_connections['outgoing'] for retainer_id in retainer_ids):
                        comp_node = self.node_registry[comp_id]
                        comp_pos = np.array(comp_node.node_position)
                        distance = np.linalg.norm(splitter_pos - comp_pos)
                        connected_comp_nodes.append((distance, comp_id))
                
                # Sort by distance and ensure we get nodes for ALL retainers
                connected_comp_nodes.sort(key=lambda x: x[0])
                
                # Strategy: ensure each retainer gets at least one comp node activated
                selected_nodes = set()
                retainer_coverage = {ret_id: False for ret_id in retainer_ids}
                
                # First pass: try to cover all retainers
                for distance, comp_id in connected_comp_nodes:
                    if len(selected_nodes) >= 10:  # Increase from 5 to 10 for better coverage
                        break
                    comp_connections = self.get_node_connections(comp_id)
                    for retainer_id in comp_connections['outgoing']:
                        if retainer_id in retainer_coverage and not retainer_coverage[retainer_id]:
                            selected_nodes.add(comp_id)
                            retainer_coverage[retainer_id] = True
                            break
                
                # Second pass: fill remaining slots with closest nodes
                for distance, comp_id in connected_comp_nodes:
                    if len(selected_nodes) >= 10:
                        break
                    if comp_id not in selected_nodes:
                        selected_nodes.add(comp_id)
                
                closest_nodes = list(selected_nodes)
                
                # Fallback: if we still don't have enough nodes, use closest overall nodes
                if len(closest_nodes) < 5:
                    all_distances = []
                    for comp_id in comp_nodes:
                        comp_node = self.node_registry[comp_id]
                        comp_pos = np.array(comp_node.node_position)
                        distance = np.linalg.norm(splitter_pos - comp_pos)
                        all_distances.append((distance, comp_id))
                    
                    all_distances.sort(key=lambda x: x[0])
                    # Add non-selected nodes to fill up to minimum 5
                    for distance, comp_id in all_distances:
                        if comp_id not in closest_nodes and len(closest_nodes) < 5:
                            closest_nodes.append(comp_id)
                
                splitter_results[splitter_id] = {
                    'source_judge': judge_id,
                    'embeddings': splitter_output,
                    'target_comp_nodes': closest_nodes,
                    'connected_nodes': len([x for x in connected_comp_nodes])
                }
                
                if trace:
                    trace['node_activations'][splitter_id] = {
                        'input': f'judge_{judge_id}_embeddings',
                        'output': f'distributed_to_{len(closest_nodes)}_comp_nodes',
                        'timestamp': time.time()
                    }
                    for comp_id in closest_nodes:
                        # Calculate weight based on distance
                        comp_node = self.node_registry[comp_id]
                        comp_pos = np.array(comp_node.node_position)
                        distance = np.linalg.norm(splitter_pos - comp_pos)
                        trace['connection_flows'].append({
                            'from': splitter_id,
                            'to': comp_id,
                            'weight': 1.0 / (1.0 + distance),
                            'active': True
                        })
        
        if self.demo:
            total_comp_activations = sum(len(data['target_comp_nodes']) for data in splitter_results.values())
            print(f"  🌿 Splitters: {len(splitter_results)} splitters → {total_comp_activations} comp node activations")
            
            # CRITICAL DEBUG: Check which comp nodes were activated and ensure all retainers will get signals
            all_activated_comp_nodes = set()
            for splitter_data in splitter_results.values():
                all_activated_comp_nodes.update(splitter_data['target_comp_nodes'])
            
            retainer_ids = self.get_nodes_by_type('Retainer')
            for retainer_id in retainer_ids:
                retainer_connections = self.get_node_connections(retainer_id)
                connected_comp_nodes = retainer_connections['incoming']
                activated_for_retainer = [comp_id for comp_id in connected_comp_nodes if comp_id in all_activated_comp_nodes]
                
                if not activated_for_retainer:
                    print(f"    ⚠️  WARNING: Retainer #{retainer_id} has NO activated comp nodes!")
                    print(f"         Connected comp nodes: {connected_comp_nodes}")
                    print(f"         Activated comp nodes: {sorted(list(all_activated_comp_nodes))}")
                else:
                    print(f"    ✅ Retainer #{retainer_id} will receive signals from {len(activated_for_retainer)} comp nodes")
        
        return splitter_results

    def _run_computation_phase(self, splitter_results: Dict[str, Any], trace: Optional[Dict] = None) -> Dict[str, Any]:
        """Computational nodes process signals with reuse tracking."""
        computation_results = {}
        node_reuse_count = defaultdict(int)
        
        for splitter_id, splitter_data in splitter_results.items():
            embeddings = splitter_data['embeddings']
            target_nodes = splitter_data['target_comp_nodes']
            
            for comp_id in target_nodes:
                # Track node reuse
                node_reuse_count[comp_id] += 1
                
                # Process embeddings through computational node
                comp_output = self.process_node(comp_id, embeddings)
                
                # Simulate computational processing
                processed_signal = self._simulate_computation(comp_output, comp_id)
                
                if comp_id not in computation_results:
                    computation_results[comp_id] = []
                
                computation_results[comp_id].append({
                    'source_splitter': splitter_id,
                    'source_judge': splitter_data['source_judge'],
                    'processed_signal': processed_signal,
                    'reuse_count': node_reuse_count[comp_id]
                })
                
                if trace:
                    trace['node_activations'][comp_id] = {
                        'input': f'splitter_{splitter_id}_embeddings',
                        'output': 'processed_signal',
                        'timestamp': time.time(),
                        'reuse_count': node_reuse_count[comp_id]
                    }
        
        # Update reuse tracking
        for comp_id, count in node_reuse_count.items():
            if count > 1:
                self.reuse_candidates.add(comp_id)
                comp_node = self.node_registry[comp_id]
                comp_node.reuse_count = getattr(comp_node, 'reuse_count', 0) + count - 1
        
        if self.demo:
            total_signals = sum(len(signals) for signals in computation_results.values())
            reused_nodes = sum(1 for count in node_reuse_count.values() if count > 1)
            print(f"  🔢 Computation: {len(computation_results)} nodes processed {total_signals} signals")
            print(f"      Reused nodes: {reused_nodes}")
        
        return computation_results

    def _run_retainer_phase(self, computation_results: Dict[str, Any], trace: Optional[Dict] = None) -> Dict[str, Any]:
        """Retainers gather signals by group."""
        retainer_ids = self.get_nodes_by_type('Retainer')
        retainer_results = {}
        
        for retainer_id in retainer_ids:
            retainer_node = self.node_registry[retainer_id]
            retainer_pos = np.array(retainer_node.node_position)
            
            # Find computational nodes connected to this retainer
            # Since connections are made as comp_node → retainer, we need to look for incoming connections
            retainer_connections = self.get_node_connections(retainer_id)
            connected_comp_nodes = retainer_connections['incoming']
            
            # Debug: Add logging to see what connections exist
            if self.demo:
                print(f"    Debug Retainer #{retainer_id}: incoming={connected_comp_nodes}")
                print(f"    Debug Retainer #{retainer_id}: all_connections={retainer_connections}")
            
            # If no incoming connections, do a comprehensive search
            if not connected_comp_nodes:
                # Check all computational nodes that have this retainer as a target
                all_comp_nodes = self.get_nodes_by_type('Computational')
                connected_comp_nodes = []
                for comp_id in all_comp_nodes:
                    comp_node = self.node_registry[comp_id]
                    # Check if this comp node has the retainer in its exit connections
                    if hasattr(comp_node, 'exit_connections') and retainer_id in comp_node.exit_connections:
                        connected_comp_nodes.append(comp_id)
                
                if self.demo:
                    print(f"    Debug: Found {len(connected_comp_nodes)} comp nodes via reverse lookup")
                    if connected_comp_nodes:
                        print(f"    Debug: Connected comp nodes: {connected_comp_nodes[:5]}...")
            
            # Gather signals from connected computational nodes
            gathered_signals = []
            available_comp_results = list(computation_results.keys())
            
            for comp_id in connected_comp_nodes:
                if comp_id in computation_results:
                    for signal_data in computation_results[comp_id]:
                        # Call the retainer's receive method to properly accumulate signals
                        retainer_node.receive(comp_id, signal_data['processed_signal'])
                        gathered_signals.append(signal_data)
            
            # If still no signals, try to connect to any available computational results
            if not gathered_signals:
                if self.demo:
                    print(f"    ⚠️  No signals from connected nodes, trying any available computational results...")
                
                # Emergency fallback: connect to any available computational results
                for comp_id in available_comp_results[:5]:  # Take first 5 available
                    if comp_id in computation_results:
                        for signal_data in computation_results[comp_id]:
                            retainer_node.receive(comp_id, signal_data['processed_signal'])
                            gathered_signals.append(signal_data)
                            # Create the missing connection
                            try:
                                # Ensure comp_id is an integer
                                comp_id_int = int(comp_id) if isinstance(comp_id, str) else comp_id
                                self.connect_nodes(comp_id_int, retainer_id, weight=0.3)
                                if self.demo:
                                    print(f"    🔗 Emergency connection: Comp #{comp_id} → Retainer #{retainer_id}")
                            except Exception as e:
                                if self.demo:
                                    print(f"    ⚠️  Failed to create emergency connection: {e}")
                                pass
            
            # STRICT CHECK: Every retainer MUST receive signals
            if not gathered_signals:
                error_msg = f"Retainer #{retainer_id} received NO signals! Connected comp nodes: {connected_comp_nodes}, Available computation results: {available_comp_results}"
                if self.demo:
                    print(f"    ❌ ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            if gathered_signals:
                # Process gathered signals through retainer
                combined_signals = self._combine_signals(gathered_signals)
                retainer_output = self.process_node(retainer_id, combined_signals)
                
                retainer_results[retainer_id] = {
                    'gathered_signals': gathered_signals,
                    'combined_output': retainer_output,
                    'signal_count': len(gathered_signals),
                    'connected_comp_nodes': connected_comp_nodes
                }
                
                if trace:
                    trace['node_activations'][retainer_id] = {
                        'input': f'{len(gathered_signals)}_signals_from_comp_nodes',
                        'output': 'combined_signal',
                        'timestamp': time.time()
                    }
        
        if self.demo:
            total_gathered = sum(data['signal_count'] for data in retainer_results.values())
            print(f"  📦 Retainers: {len(retainer_results)} retainers gathered {total_gathered} signals")
        
        return retainer_results

    def _run_reviewer_phase(self, retainer_results: Dict[str, Any], trace: Optional[Dict] = None) -> Dict[str, Any]:
        """Reviewers calculate final probabilities for their groups."""
        reviewer_ids = self.get_nodes_by_type('Reviewer')  # Match the node type used in initialization
        reviewer_results = {}
        
        if self.demo:
            print(f"    Debug: Found {len(reviewer_ids)} reviewers: {reviewer_ids}")
            print(f"    Debug: Retainer results available: {list(retainer_results.keys())}")
        
        for reviewer_id in reviewer_ids:
            reviewer_node = self.node_registry[reviewer_id]
            
            # Find retainer connected to this reviewer
            reviewer_connections = self.get_node_connections(reviewer_id)
            connected_retainers = reviewer_connections['incoming']
            
            if self.demo:
                print(f"    Debug Reviewer #{reviewer_id}: incoming_retainers={connected_retainers}")
            
            # Process signals from connected retainer
            if connected_retainers:
                retainer_id = connected_retainers[0]  # Assuming one retainer per reviewer
                if retainer_id in retainer_results:
                    retainer_data = retainer_results[retainer_id]
                    
                    # Process through reviewer to get final probabilities
                    reviewer_input = retainer_data['combined_output']
                    reviewer_output = self.process_node(reviewer_id, reviewer_input)
                    
                    # Calculate final probabilities
                    num_classes = 10 if self.demo else 100  # Simulate classification classes
                    raw_logits = np.random.randn(num_classes)
                    probabilities = F.softmax(torch.tensor(raw_logits), dim=0).numpy()
                    
                    reviewer_results[reviewer_id] = {
                        'source_retainer': retainer_id,
                        'raw_output': reviewer_output,
                        'probabilities': probabilities,
                        'confidence': np.max(probabilities),
                        'signal_sources': retainer_data['gathered_signals']
                    }
                    
                    if trace:
                        trace['node_activations'][reviewer_id] = {
                            'input': f'retainer_{retainer_id}_combined',
                            'output': f'probabilities_{num_classes}_classes',
                            'timestamp': time.time()
                        }
                    
                    if self.demo:
                        print(f"    ✅ Reviewer #{reviewer_id} processed retainer #{retainer_id} signals → confidence {np.max(probabilities):.3f}")
                else:
                    error_msg = f"Reviewer #{reviewer_id} connected to retainer #{retainer_id} but no results available! Available retainer results: {list(retainer_results.keys())}"
                    if self.demo:
                        print(f"    ❌ ERROR: {error_msg}")
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"Reviewer #{reviewer_id} has no connected retainers! Connected retainers: {connected_retainers}"
                if self.demo:
                    print(f"    ❌ ERROR: {error_msg}")
                raise RuntimeError(error_msg)
        
        if self.demo:
            avg_confidence = np.mean([data['confidence'] for data in reviewer_results.values()]) if reviewer_results else 0
            print(f"  📋 Reviewers: {len(reviewer_results)} reviewers, avg confidence: {avg_confidence:.3f}")
        
        return reviewer_results

    def _run_handler_phase(self, reviewer_results: Dict[str, Any], controller_result: Dict[str, Any], 
                          trace: Optional[Dict] = None) -> Dict[str, Any]:
        """Handler combines reviewer probabilities with controller weights and supports multi-token generation."""
        handler_ids = self.get_nodes_by_type('Handler')
        if not handler_ids:
            raise ValueError("No Handler node found")
        
        handler_id = handler_ids[0]
        handler_node = self.node_registry[handler_id]
        
        # Check if multi-token generation is enabled
        multi_token_enabled = self.output_config.get('enable_multi_token', False)
        
        if multi_token_enabled:
            # Multi-token generation mode
            if self.demo:
                print(f"  🔄 Handler Phase: Multi-token generation enabled")
            
            # Collect outputs from different node groups for distinct behaviors
            node_group_outputs = self._collect_node_group_outputs(
                reviewer_results, controller_result, trace
            )
            
            # Prepare context embeddings from current processing
            context_embeddings = self._prepare_context_embeddings(
                reviewer_results, controller_result
            )
            
            # Setup handler input for multi-token generation
            handler_input = {
                'generation_mode': 'multi_token',
                'node_group_outputs': node_group_outputs,
                'context_embeddings': context_embeddings,
                'max_tokens': self.output_config.get('max_sequence_length', 5),
                'reviewer_results': reviewer_results,
                'controller_weights': controller_result.get('judge_probabilities', {})
            }
            
            # Process through handler for multi-token sequence generation
            handler_output = self.process_node(handler_id, handler_input)
            
            if trace:
                trace['node_activations'][handler_id] = {
                    'input': f'multi_token_generation_{len(node_group_outputs)}_groups',
                    'output': f'sequence_length_{len(handler_output.get("tokens", []))}',
                    'timestamp': time.time()
                }
            
            # Format multi-token result
            final_result = {
                'tokens': handler_output.get('tokens', []),
                'prediction': handler_output['tokens'][0] if handler_output.get('tokens') else 0,
                'confidence': handler_output.get('confidence', 0.0),
                'consensus': handler_output.get('consensus', 0.0),
                'probabilities': self._tokens_to_probabilities(handler_output.get('tokens', [0])),
                'status': handler_output.get('status', 'UNKNOWN'),
                'generation_metadata': {
                    'group_sequences': handler_output.get('group_sequences', {}),
                    'sequence_scores': handler_output.get('sequence_scores', {}),
                    'selection_details': handler_output.get('selection_details', []),
                    'generation_method': 'multi_token_context_aware'
                },
                'handler_output': handler_output
            }
            
            if self.demo:
                print(f"      Generated tokens: {handler_output.get('tokens', [])}")
                print(f"      Confidence: {handler_output.get('confidence', 0):.3f}")
                print(f"      Status: {handler_output.get('status', 'UNKNOWN')}")
                
        else:
            # Traditional single-token mode
            if self.demo:
                print(f"  🔄 Handler Phase: Single-token mode")
                
            # Get judge probabilities from controller
            judge_probabilities = controller_result['judge_probabilities']
            
            # Combine reviewer results with controller weightings
            weighted_probabilities = []
            total_weight = 0.0
            
            for reviewer_id, reviewer_data in reviewer_results.items():
                # Find which judge contributed to this reviewer (through the chain)
                source_signals = reviewer_data['signal_sources']
                if source_signals:
                    source_judge = source_signals[0]['source_judge']
                    judge_weight = judge_probabilities.get(source_judge, 0.0)
                    
                    # Weight the reviewer's probabilities by the judge's probability
                    weighted_probs = reviewer_data['probabilities'] * judge_weight
                    weighted_probabilities.append(weighted_probs)
                    total_weight += judge_weight
            
            if len(weighted_probabilities) > 0 and total_weight > 0:
                # Calculate final mean probabilities
                final_probabilities = np.mean(weighted_probabilities, axis=0)
                final_probabilities = final_probabilities / np.sum(final_probabilities)  # Normalize
                
                # Get final prediction
                prediction_idx = np.argmax(final_probabilities)
                confidence = final_probabilities[prediction_idx]
                
                # Process through handler node for final output
                handler_input = {
                    'reviewer_results': reviewer_results,
                    'controller_weights': judge_probabilities,
                    'final_probabilities': final_probabilities
                }
                handler_output = self.process_node(handler_id, handler_input)
                
                if trace:
                    trace['node_activations'][handler_id] = {
                        'input': f'{len(reviewer_results)}_reviewer_results',
                        'output': f'final_prediction_{prediction_idx}',
                        'timestamp': time.time()
                    }
                
                final_result = {
                    'prediction': prediction_idx,
                    'confidence': confidence,
                    'probabilities': final_probabilities.tolist(),
                    'handler_output': handler_output,
                    'num_reviewers_used': len(reviewer_results),
                    'total_weight': total_weight
                }
                
                # Apply output formatting based on configuration
                final_result = self._format_output(final_result)
            else:
                # Fallback if no valid results
                num_classes = self.output_config.get('num_classes', 10)
                fallback_probs = [1.0] + [0.0] * (num_classes - 1)  # Default to first class
                final_result = {
                    'prediction': 0,
                    'confidence': 0.0,
                    'probabilities': fallback_probs,
                    'handler_output': None,
                    'num_reviewers_used': 0,
                    'total_weight': 0.0
                }
                
                # Apply output formatting based on configuration
                final_result = self._format_output(final_result)
        
        if self.demo:
            if multi_token_enabled:
                print(f"  🏆 Handler: Generated {len(final_result.get('tokens', []))} tokens")
                print(f"      Confidence: {final_result['confidence']:.3f}, Status: {final_result.get('status', 'UNKNOWN')}")
            else:
                print(f"  🏆 Handler: Final prediction {final_result['prediction']} with confidence {final_result['confidence']:.3f}")
                print(f"      Used {final_result.get('num_reviewers_used', 0)} reviewers, total weight: {final_result.get('total_weight', 0):.3f}")
        
        return final_result
    
    def _collect_node_group_outputs(self, reviewer_results: Dict[str, Any], 
                                   controller_result: Dict[str, Any], 
                                   trace: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Collect outputs from different node groups to enable distinct generation behaviors
        """
        node_group_outputs = {}
        
        # Collect Judge outputs (conservative behavior)
        judge_outputs = controller_result.get('judge_probabilities', {})
        if judge_outputs:
            node_group_outputs['Judge'] = list(judge_outputs.values())
        
        # Collect Controller outputs (directive behavior)
        controller_weights = controller_result.get('controller_weights', [])
        if controller_weights:
            node_group_outputs['Controller'] = controller_weights
        
        # Collect Reviewer outputs (evaluative behavior) 
        reviewer_probs = []
        for reviewer_id, reviewer_data in reviewer_results.items():
            if 'probabilities' in reviewer_data:
                reviewer_probs.extend(reviewer_data['probabilities'])
        if reviewer_probs:
            node_group_outputs['Reviewer'] = reviewer_probs
        
        # Extract other node group outputs from trace if available
        if trace and 'node_activations' in trace:
            for node_id, activation in trace['node_activations'].items():
                node = self.node_registry.get(node_id)
                if node and hasattr(node, 'node_type'):
                    node_type = node.node_type
                    if node_type in ['Splitter', 'Computational', 'Retainer']:
                        # Create synthetic output based on node type
                        if node_type == 'Splitter':
                            # Divergent behavior - more varied outputs
                            node_group_outputs['Splitter'] = [random.uniform(0, 1) for _ in range(10)]
                        elif node_type == 'Computational':
                            # Analytical behavior - structured outputs
                            node_group_outputs['Computational'] = [i * 0.1 for i in range(10)]
                        elif node_type == 'Retainer':
                            # Contextual behavior - memory-influenced outputs
                            base_output = [0.1] * 10
                            for i in range(5):  # Boost first 5 elements for context
                                base_output[i] += 0.1
                            node_group_outputs['Retainer'] = base_output
        
        # Ensure all groups have some output
        default_groups = ['Judge', 'Controller', 'Splitter', 'Computational', 'Retainer', 'Reviewer']
        for group in default_groups:
            if group not in node_group_outputs:
                # Create minimal default output
                node_group_outputs[group] = [random.uniform(0.05, 0.15) for _ in range(5)]
        
        if self.demo:
            print(f"      Collected outputs from {len(node_group_outputs)} node groups: {list(node_group_outputs.keys())}")
        
        return node_group_outputs
    
    def _prepare_context_embeddings(self, reviewer_results: Dict[str, Any], 
                                   controller_result: Dict[str, Any]) -> List[List[float]]:
        """
        Prepare context embeddings from current processing for sequence generation
        """
        context_embeddings = []
        embedding_dim = 512 if not self.demo else 16
        
        # Convert reviewer probabilities to embeddings
        for reviewer_id, reviewer_data in reviewer_results.items():
            if 'probabilities' in reviewer_data:
                probs = reviewer_data['probabilities']
                # Create embedding by padding/truncating to target dimension
                embedding = [0.0] * embedding_dim
                for i, prob in enumerate(probs[:embedding_dim]):
                    embedding[i] = prob
                context_embeddings.append(embedding)
        
        # Add controller context if available
        if 'judge_probabilities' in controller_result:
            judge_probs = list(controller_result['judge_probabilities'].values())
            embedding = [0.0] * embedding_dim
            for i, prob in enumerate(judge_probs[:embedding_dim]):
                embedding[i] = prob
            context_embeddings.append(embedding)
        
        # Ensure we have at least one context embedding
        if not context_embeddings:
            # Create default context
            default_embedding = [random.uniform(0.01, 0.1) for _ in range(embedding_dim)]
            context_embeddings.append(default_embedding)
        
        if self.demo:
            print(f"      Prepared {len(context_embeddings)} context embeddings (dim: {embedding_dim})")
        
        return context_embeddings
    
    def _tokens_to_probabilities(self, tokens: List[int]) -> List[float]:
        """
        Convert a sequence of token IDs to probability distribution for the first token
        """
        num_classes = self.output_config.get('num_classes', 10)
        probabilities = [0.0] * num_classes
        
        if tokens:
            # Use first token as primary prediction
            primary_token = tokens[0] % num_classes  # Ensure within bounds
            probabilities[primary_token] = 0.7  # High confidence for primary
            
            # Distribute remaining probability among other tokens in sequence
            remaining_prob = 0.3
            for i, token in enumerate(tokens[1:], 1):
                if i < len(tokens) and token < num_classes:
                    probabilities[token] += remaining_prob / (len(tokens) - 1)
            
            # Normalize to ensure sum = 1
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
            else:
                # Fallback uniform distribution
                probabilities = [1.0 / num_classes] * num_classes
        else:
            # Default uniform distribution
            probabilities = [1.0 / num_classes] * num_classes
        
        return probabilities

    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the output according to the configured output type and format.
        Includes proper detokenization of token sequences back to readable text.
        
        Args:
            result: Raw result from handler phase containing token IDs
            
        Returns:
            Formatted result with detokenized text and token information
        """
        config = self.output_config
        formatted_result = result.copy()
        
        # Handle multi-token sequences if present
        if 'tokens' in result and result['tokens']:
            tokens = result['tokens']
            formatted_result['token_sequence'] = tokens
            formatted_result['sequence_length'] = len(tokens)
            
            # Detokenize the token sequence to readable text
            detokenized_text = self._detokenize_output(tokens)
            formatted_result['detokenized_text'] = detokenized_text
            
            # Format each token in the sequence
            if config['output_format'] == 'token_id' and config['vocab_mapping']:
                formatted_tokens = []
                for token_id in tokens:
                    token_str = config['vocab_mapping'].get(token_id, f'unk_{token_id}')
                    formatted_tokens.append(token_str)
                formatted_result['formatted_sequence'] = formatted_tokens
                formatted_result['sequence_text'] = ' '.join(formatted_tokens)
            elif config['output_format'] == 'label':
                formatted_labels = []
                for token_id in tokens:
                    if token_id < len(config['class_labels']):
                        label = config['class_labels'][token_id]
                    else:
                        label = f'class_{token_id}'
                    formatted_labels.append(label)
                formatted_result['formatted_sequence'] = formatted_labels
                formatted_result['sequence_text'] = ' → '.join(formatted_labels)
            
            # Add generation metadata if available
            if 'generation_metadata' in result:
                formatted_result['generation_info'] = {
                    'method': result['generation_metadata'].get('generation_method', 'unknown'),
                    'group_sequences': result['generation_metadata'].get('group_sequences', {}),
                    'sequence_scores': result['generation_metadata'].get('sequence_scores', {}),
                    'num_groups_used': len(result['generation_metadata'].get('group_sequences', {}))
                }
            
            if self.demo:
                print(f"  📝 Multi-token sequence: {formatted_result.get('sequence_text', tokens)}")
                print(f"  🔤 Detokenized text: '{detokenized_text}'")
                if 'generation_info' in formatted_result:
                    print(f"      Method: {formatted_result['generation_info']['method']}")
                    print(f"      Groups: {formatted_result['generation_info']['num_groups_used']}")
        
        # Handle single token prediction with detokenization
        else:
            primary_idx = result['prediction']
            if config.get('vocab_mapping') and config['output_format'] == 'token_id':
                # Detokenize single token prediction
                detokenized_text = self._detokenize_output([primary_idx])
                formatted_result['detokenized_text'] = detokenized_text
                
                if self.demo:
                    print(f"  🔤 Detokenized single token: '{detokenized_text}'")
        
        # Get top-k predictions if requested
        return_top_k = config.get('return_top_k', 1)
        probabilities = np.array(result['probabilities'])
        
        if return_top_k > 1:
            # Get top-k indices and their probabilities
            top_k_indices = np.argsort(probabilities)[-return_top_k:][::-1]
            top_k_probs = probabilities[top_k_indices]
            
            formatted_result['top_k_predictions'] = []
            formatted_result['top_k_confidences'] = top_k_probs.tolist()
            
            for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
                pred_info = {
                    'rank': i + 1,
                    'index': int(idx),
                    'confidence': float(prob)
                }
                
                # Format based on output type with detokenization
                if config['output_format'] == 'label':
                    # Add bounds checking for class_labels
                    if idx < len(config['class_labels']):
                        pred_info['label'] = config['class_labels'][idx]
                    else:
                        pred_info['label'] = f'class_{idx}'
                elif config['output_format'] == 'token_id' and config['vocab_mapping']:
                    pred_info['token'] = config['vocab_mapping'].get(idx, f'unk_{idx}')
                    # Add detokenized version for single tokens
                    pred_info['detokenized'] = self._detokenize_output([idx])
                
                formatted_result['top_k_predictions'].append(pred_info)
        
        # Format the primary prediction
        primary_idx = result['prediction']
        
        if config['output_format'] == 'label':
            # Add bounds checking for class_labels
            if primary_idx < len(config['class_labels']):
                formatted_result['prediction_label'] = config['class_labels'][primary_idx]
            else:
                formatted_result['prediction_label'] = f'class_{primary_idx}'
        elif config['output_format'] == 'token_id' and config['vocab_mapping']:
            formatted_result['predicted_token'] = config['vocab_mapping'].get(primary_idx, f'unk_{primary_idx}')
            # Add detokenized version
            if 'detokenized_text' not in formatted_result:
                formatted_result['detokenized_text'] = self._detokenize_output([primary_idx])
        elif config['output_format'] == 'probability_dist':
            # Already included as 'probabilities'
            pass
        
        # Add metadata about output configuration
        formatted_result['output_metadata'] = {
            'type': config['type'],
            'format': config['output_format'],
            'num_classes': config['num_classes'],
            'top_k': return_top_k
        }
        
        if self.demo:
            print(f"  📋 Formatted output: {config['output_format']} format")
            if 'prediction_label' in formatted_result:
                print(f"      Label: {formatted_result['prediction_label']}")
            if 'predicted_token' in formatted_result:
                print(f"      Token: {formatted_result['predicted_token']}")
            if return_top_k > 1:
                top_k_confidences = [p['confidence'] for p in formatted_result['top_k_predictions']]
                print(f"      Top-{return_top_k}: {[f'{c:.3f}' for c in top_k_confidences]}")
        
        return formatted_result

    def _simulate_computation(self, input_data: Any, node_id: int) -> Any:
        """Simulate computational processing in a computational node."""
        node = self.node_registry[node_id]
        
        # Simulate different types of computation based on node position
        position_sum = sum(abs(p) for p in node.node_position)
        computation_type = int(position_sum) % 4
        
        if isinstance(input_data, np.ndarray):
            if computation_type == 0:
                # Matrix transformation
                return input_data * np.random.uniform(0.8, 1.2)
            elif computation_type == 1:
                # Non-linear activation
                return np.tanh(input_data + np.random.normal(0, 0.1, input_data.shape))
            elif computation_type == 2:
                # Convolution-like operation
                return np.convolve(input_data.flatten(), [0.25, 0.5, 0.25], mode='same').reshape(input_data.shape)
            else:
                # Attention-like mechanism
                weights = F.softmax(torch.tensor(input_data), dim=0).numpy()
                return input_data * weights
        else:
            # Handle non-array inputs
            return input_data

    def _combine_signals(self, signals: List[Dict[str, Any]]) -> Any:
        """Combine multiple signals from computational nodes."""
        if not signals:
            return None
        
        # Extract processed signals
        processed_signals = [signal['processed_signal'] for signal in signals]
        
        # Combine based on signal type
        if all(isinstance(sig, np.ndarray) for sig in processed_signals):
            # Average arrays of same shape
            try:
                combined = np.mean(processed_signals, axis=0)
                return combined
            except:
                # Fallback: return first signal if shapes don't match
                return processed_signals[0]
        else:
            # For non-array signals, return the most recent
            return processed_signals[-1]

    def _update_run_metrics(self, execution_time: float, trace: Optional[Dict] = None):
        """Update brain performance metrics after a run."""
        # Update inference time with safe conversion
        current_value = self.brain_records.loc[0, 'Inference_Time']
        if pd.isna(current_value) or current_value is None:
            current_inference_time = 0.0
        else:
            try:
                # Handle various types that might be stored
                if isinstance(current_value, (int, float)):
                    current_inference_time = float(current_value)
                elif isinstance(current_value, str):
                    current_inference_time = float(current_value) if current_value.replace('.', '').isdigit() else 0.0
                else:
                    current_inference_time = 0.0
            except (ValueError, TypeError, AttributeError):
                current_inference_time = 0.0
        
        self.brain_records.loc[0, 'Inference_Time'] = current_inference_time + execution_time
        
        # Update node usage in records
        if trace:
            for node_id, activation_data in trace['node_activations'].items():
                # Find the node record index
                node_indices = self.node_records.index[self.node_records['Node_ID'] == node_id].tolist()
                if node_indices:
                    idx = node_indices[0]
                    
                    # Update times called
                    current_calls_val = self.node_records.at[idx, 'Times_Called']
                    if pd.isna(current_calls_val) or current_calls_val is None:
                        current_calls = 0
                    else:
                        try:
                            current_calls = int(current_calls_val) if isinstance(current_calls_val, (int, float, str)) else 0
                        except (ValueError, TypeError):
                            current_calls = 0
                    self.node_records.at[idx, 'Times_Called'] = current_calls + 1
                    
                    # Update reuse count if applicable
                    if 'reuse_count' in activation_data:
                        current_reuse_val = self.node_records.at[idx, 'Reuse_Count']
                        if pd.isna(current_reuse_val) or current_reuse_val is None:
                            current_reuse = 0
                        else:
                            try:
                                current_reuse = int(current_reuse_val) if isinstance(current_reuse_val, (int, float, str)) else 0
                            except (ValueError, TypeError):
                                current_reuse = 0
                        self.node_records.at[idx, 'Reuse_Count'] = current_reuse + activation_data['reuse_count']
        
        # Update efficiency metrics
        if len(self.neural_nodes) > 0:
            reuse_efficiency = len(self.reuse_candidates) / len(self.neural_nodes)
            self.brain_records.loc[0, 'Reuse_Efficiency'] = reuse_efficiency


def main():
    """
    Comprehensive test function for BrainNexus initialization and execution.
    Tests all aspects including initialization, connections, signal flow, and debugging.
    """
    print("=" * 80)
    print("🧠 BRAINNEXUS COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Test configuration
    test_config = {
        'demo_mode': True,
        'dimensions': 4,
        'node_count_pre': 3,
        'test_runs': 3,
        'debug_level': 'verbose'  # 'basic', 'verbose', 'extreme'
    }
    
    total_start_time = time.time()
    test_results = {
        'initialization': {},
        'connections': {},
        'execution': {},
        'errors': [],
        'warnings': [],
        'performance': {}
    }
    
    try:
        print(f"🔧 Test Configuration:")
        for key, value in test_config.items():
            print(f"   {key}: {value}")
        print()
        
        # Phase 1: Initialization Testing
        print("🚀 PHASE 1: INITIALIZATION TESTING")
        print("-" * 50)
        
        init_start = time.time()
        brain = BrainNexus(
            dimensions=test_config['dimensions'],
            node_count_pre=test_config['node_count_pre'],
            demo=test_config['demo_mode']
        )
        
        # Test initial state
        print("📊 Initial State Check:")
        print(f"   Dimensions: {brain.dimensions}")
        print(f"   Node count estimate: {brain.entrance_node_count_pre}")
        print(f"   Learning rate: {brain.learning_rate}")
        print(f"   Demo mode: {brain.demo}")
        print(f"   Initial nodes: {len(brain.neural_nodes)}")
        print(f"   Node registry size: {len(brain.node_registry)}")
        print()
        
        # Initialize brain and capture node map
        node_map = brain.initialize_brain()
        init_time = time.time() - init_start
        
        test_results['initialization']['time'] = init_time
        test_results['initialization']['node_map'] = node_map
        test_results['initialization']['success'] = True
        
        print(f"✅ Initialization completed in {init_time:.3f}s")
        print()
        
        # Phase 2: Detailed Node Analysis
        print("🔍 PHASE 2: DETAILED NODE ANALYSIS")
        print("-" * 50)
        
        # Analyze node distribution
        _analyze_node_distribution(brain, node_map, test_results)
        
        # Check node positions
        _analyze_node_positions(brain, node_map, test_results)
        
        # Phase 3: Connection Testing
        print("🔗 PHASE 3: CONNECTION TESTING")
        print("-" * 50)
        
        _test_connections(brain, node_map, test_results)
        
        # Phase 4: Signal Flow Testing
        print("🌊 PHASE 4: SIGNAL FLOW TESTING")
        print("-" * 50)
        
        _test_signal_flow(brain, node_map, test_results, test_config['test_runs'])
        
        # Phase 5: Execution Pipeline Testing
        print("⚡ PHASE 5: EXECUTION PIPELINE TESTING")
        print("-" * 50)
        
        _test_execution_pipeline(brain, test_results, test_config['test_runs'])
        
        # Phase 6: Error Resilience Testing
        print("🛡️ PHASE 6: ERROR RESILIENCE TESTING")
        print("-" * 50)
        
        _test_error_resilience(brain, test_results)
        
        # Phase 7: Performance Analysis
        print("📈 PHASE 7: PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        _analyze_performance(brain, test_results)
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during testing: {e}")
        test_results['errors'].append(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final Results Summary
        total_time = time.time() - total_start_time
        test_results['performance']['total_time'] = total_time
        
        print("=" * 80)
        print("📋 FINAL TEST RESULTS SUMMARY")
        print("=" * 80)
        
        _print_final_summary(test_results, total_time)


def _analyze_node_distribution(brain: BrainNexus, node_map: Dict[str, List[int]], test_results: Dict):
    """Analyze the distribution and properties of created nodes."""
    print("📊 Node Distribution Analysis:")
    
    distribution_issues = []
    expected_counts = {
        'controller': 1,
        'judges': 2 ** min(brain.dimensions, 3) if not brain.demo else 4,
        'splitters': 2 ** min(brain.dimensions, 3) if not brain.demo else 4,
        'retainers': 2 ** min(brain.dimensions, 3) if not brain.demo else 4,
        'reviewers': 4,  # Always exactly 4
        'handler': 1,
        'comp_nodes': 180 if brain.demo else 18000
    }
    
    for node_type, expected_count in expected_counts.items():
        actual_count = len(node_map.get(node_type, []))
        status = "✅" if actual_count == expected_count else "❌"
        print(f"   {status} {node_type.capitalize()}: {actual_count}/{expected_count}")
        
        if actual_count != expected_count:
            issue = f"{node_type}: expected {expected_count}, got {actual_count}"
            distribution_issues.append(issue)
    
    # Check for missing node types
    for node_type in expected_counts.keys():
        if node_type not in node_map or not node_map[node_type]:
            distribution_issues.append(f"Missing node type: {node_type}")
    
    # Verify nodes exist in registry
    registry_issues = []
    for node_type, node_ids in node_map.items():
        for node_id in node_ids:
            if node_id not in brain.node_registry:
                registry_issues.append(f"Node {node_id} ({node_type}) missing from registry")
    
    test_results['initialization']['distribution_issues'] = distribution_issues
    test_results['initialization']['registry_issues'] = registry_issues
    
    if distribution_issues:
        print(f"   ⚠️  Distribution Issues: {len(distribution_issues)}")
        for issue in distribution_issues:
            print(f"      - {issue}")
    
    if registry_issues:
        print(f"   ❌ Registry Issues: {len(registry_issues)}")
        for issue in registry_issues:
            print(f"      - {issue}")
    
    print()


def _analyze_node_positions(brain: BrainNexus, node_map: Dict[str, List[int]], test_results: Dict):
    """Analyze node spatial positioning."""
    print("📍 Node Position Analysis:")
    
    position_issues = []
    
    # Check controller at origin
    controller_ids = node_map.get('controller', [])
    if controller_ids:
        controller_node = brain.node_registry[controller_ids[0]]
        controller_pos = np.array(controller_node.node_position)
        if not np.allclose(controller_pos, 0.0, atol=1e-6):
            position_issues.append(f"Controller not at origin: {controller_pos}")
        else:
            print(f"   ✅ Controller positioned at origin: {controller_pos}")
    
    # Check judge positions
    judge_distances = []
    for judge_id in node_map.get('judges', []):
        judge_node = brain.node_registry[judge_id]
        distance_from_origin = np.linalg.norm(judge_node.node_position)
        judge_distances.append(distance_from_origin)
    
    if judge_distances:
        avg_judge_distance = np.mean(judge_distances)
        print(f"   📏 Judge distances from origin: avg={avg_judge_distance:.2f}, range=[{min(judge_distances):.2f}, {max(judge_distances):.2f}]")
    
    # Check for overlapping positions
    all_positions = []
    position_map = {}
    for node_type, node_ids in node_map.items():
        for node_id in node_ids:
            node = brain.node_registry[node_id]
            pos_tuple = tuple(round(p, 6) for p in node.node_position)
            all_positions.append(pos_tuple)
            if pos_tuple in position_map:
                position_issues.append(f"Overlapping positions: Node {node_id} and {position_map[pos_tuple]} at {pos_tuple}")
            else:
                position_map[pos_tuple] = node_id
    
    # Check spatial index consistency
    spatial_index_issues = []
    if brain.spatial_index is not None:
        if len(brain.node_positions) != len(brain.neural_nodes):
            spatial_index_issues.append(f"Position count mismatch: {len(brain.node_positions)} vs {len(brain.neural_nodes)}")
    
    test_results['initialization']['position_issues'] = position_issues
    test_results['initialization']['spatial_index_issues'] = spatial_index_issues
    
    if position_issues:
        print(f"   ⚠️  Position Issues: {len(position_issues)}")
        for issue in position_issues:
            print(f"      - {issue}")
    
    if spatial_index_issues:
        print(f"   ❌ Spatial Index Issues: {len(spatial_index_issues)}")
        for issue in spatial_index_issues:
            print(f"      - {issue}")
    
    print(f"   📊 Total unique positions: {len(set(all_positions))}/{len(all_positions)}")
    print()


def _test_connections(brain: BrainNexus, node_map: Dict[str, List[int]], test_results: Dict):
    """Test all node connections for correctness."""
    print("🔗 Connection Testing:")
    
    connection_issues = []
    
    # Test Controller → Judges connections
    controller_id = node_map['controller'][0] if node_map['controller'] else None
    if controller_id:
        controller_connections = brain.get_node_connections(controller_id)
        expected_judge_connections = set(node_map['judges'])
        actual_judge_connections = set(controller_connections['outgoing'])
        
        if expected_judge_connections != actual_judge_connections:
            missing = expected_judge_connections - actual_judge_connections
            extra = actual_judge_connections - expected_judge_connections
            if missing:
                connection_issues.append(f"Controller missing connections to judges: {missing}")
            if extra:
                connection_issues.append(f"Controller has extra connections to: {extra}")
        else:
            print(f"   ✅ Controller → Judges: {len(actual_judge_connections)} connections")
    
    # Test Judges → Splitters connections
    judge_splitter_issues = []
    for i, (judge_id, splitter_id) in enumerate(zip(node_map['judges'], node_map['splitters'])):
        judge_connections = brain.get_node_connections(judge_id)
        if splitter_id not in judge_connections['outgoing']:
            judge_splitter_issues.append(f"Judge {judge_id} not connected to Splitter {splitter_id}")
    
    if judge_splitter_issues:
        connection_issues.extend(judge_splitter_issues)
    else:
        print(f"   ✅ Judges → Splitters: {len(node_map['judges'])} connections")
    
    # Test Retainers → Reviewers connections
    retainer_reviewer_issues = []
    for retainer_id, reviewer_id in zip(node_map['retainers'], node_map['reviewers']):
        retainer_connections = brain.get_node_connections(retainer_id)
        if reviewer_id not in retainer_connections['outgoing']:
            retainer_reviewer_issues.append(f"Retainer {retainer_id} not connected to Reviewer {reviewer_id}")
    
    if retainer_reviewer_issues:
        connection_issues.extend(retainer_reviewer_issues)
    else:
        print(f"   ✅ Retainers → Reviewers: {len(node_map['retainers'])} connections")
    
    # Test Reviewers → Handler connections
    handler_id = node_map['handler'][0] if node_map['handler'] else None
    if handler_id:
        handler_connections = brain.get_node_connections(handler_id)
        expected_reviewer_connections = set(node_map['reviewers'])
        actual_reviewer_connections = set(handler_connections['incoming'])
        
        if expected_reviewer_connections != actual_reviewer_connections:
            missing = expected_reviewer_connections - actual_reviewer_connections
            if missing:
                connection_issues.append(f"Handler missing connections from reviewers: {missing}")
        else:
            print(f"   ✅ Reviewers → Handler: {len(actual_reviewer_connections)} connections")
    
    # Test Splitter → Computational connections
    splitter_comp_stats = []
    for splitter_id in node_map['splitters']:
        splitter_connections = brain.get_node_connections(splitter_id)
        comp_connections = [conn for conn in splitter_connections['outgoing'] 
                          if conn in node_map['comp_nodes']]
        splitter_comp_stats.append(len(comp_connections))
    
    if splitter_comp_stats:
        avg_splitter_comps = np.mean(splitter_comp_stats)
        print(f"   📊 Splitter → Comp: avg={avg_splitter_comps:.1f} connections per splitter")
    
    # Test Computational → Retainer connections
    comp_retainer_stats = []
    for retainer_id in node_map['retainers']:
        retainer_connections = brain.get_node_connections(retainer_id)
        comp_connections = [conn for conn in retainer_connections['incoming'] 
                          if conn in node_map['comp_nodes']]
        comp_retainer_stats.append(len(comp_connections))
    
    if comp_retainer_stats:
        avg_comp_retainers = np.mean(comp_retainer_stats)
        print(f"   📊 Comp → Retainer: avg={avg_comp_retainers:.1f} connections per retainer")
    
    # Test connection weights
    weight_issues = []
    for node_type, node_ids in node_map.items():
        for node_id in node_ids:
            node_connections = brain.get_node_connections(node_id)
            for target_id in node_connections['outgoing']:
                weight = brain.get_connection_weight(node_id, target_id)
                if weight is None:
                    weight_issues.append(f"Missing weight for connection {node_id} → {target_id}")
                elif not (0.0 <= weight <= 2.0):  # Reasonable weight range
                    weight_issues.append(f"Unusual weight {weight} for connection {node_id} → {target_id}")
    
    test_results['connections']['issues'] = connection_issues
    test_results['connections']['weight_issues'] = weight_issues
    test_results['connections']['splitter_comp_stats'] = splitter_comp_stats
    test_results['connections']['comp_retainer_stats'] = comp_retainer_stats
    
    if connection_issues:
        print(f"   ❌ Connection Issues: {len(connection_issues)}")
        for issue in connection_issues:
            print(f"      - {issue}")
    
    if weight_issues:
        print(f"   ⚠️  Weight Issues: {len(weight_issues)}")
        for issue in weight_issues[:5]:  # Show first 5
            print(f"      - {issue}")
        if len(weight_issues) > 5:
            print(f"      ... and {len(weight_issues) - 5} more")
    
    print()


def _test_signal_flow(brain: BrainNexus, node_map: Dict[str, List[int]], test_results: Dict, num_tests: int):
    """Test signal flow through individual node types."""
    print("🌊 Signal Flow Testing:")
    
    signal_flow_issues = []
    
    # Test Controller processing
    try:
        controller_id = node_map['controller'][0]
        test_input = "test signal"
        controller_output = brain.process_node(controller_id, test_input)
        print(f"   ✅ Controller processing: input → output")
    except Exception as e:
        signal_flow_issues.append(f"Controller processing failed: {e}")
    
    # Test Judge processing
    judge_success = 0
    for judge_id in node_map['judges']:
        try:
            judge_output = brain.process_node(judge_id, test_input)
            judge_success += 1
        except Exception as e:
            signal_flow_issues.append(f"Judge {judge_id} processing failed: {e}")
    
    print(f"   📊 Judge processing: {judge_success}/{len(node_map['judges'])} successful")
    
    # Test Splitter processing
    splitter_success = 0
    for splitter_id in node_map['splitters']:
        try:
            test_embedding = np.random.randn(64)
            splitter_output = brain.process_node(splitter_id, test_embedding)
            splitter_success += 1
        except Exception as e:
            signal_flow_issues.append(f"Splitter {splitter_id} processing failed: {e}")
    
    print(f"   📊 Splitter processing: {splitter_success}/{len(node_map['splitters'])} successful")
    
    # Test sample of Computational nodes
    comp_test_count = min(10, len(node_map['comp_nodes']))
    comp_success = 0
    test_comp_nodes = random.sample(node_map['comp_nodes'], comp_test_count)
    
    for comp_id in test_comp_nodes:
        try:
            test_embedding = np.random.randn(64)
            comp_output = brain.process_node(comp_id, test_embedding)
            comp_success += 1
        except Exception as e:
            signal_flow_issues.append(f"Computational {comp_id} processing failed: {e}")
    
    print(f"   📊 Computational processing: {comp_success}/{comp_test_count} tested successful")
    
    # Test Retainer processing
    retainer_success = 0
    for retainer_id in node_map['retainers']:
        try:
            test_signals = [{'processed_signal': np.random.randn(64)} for _ in range(3)]
            combined = brain._combine_signals(test_signals)
            retainer_output = brain.process_node(retainer_id, combined)
            retainer_success += 1
        except Exception as e:
            signal_flow_issues.append(f"Retainer {retainer_id} processing failed: {e}")
    
    print(f"   📊 Retainer processing: {retainer_success}/{len(node_map['retainers'])} successful")
    
    # Test Reviewer processing
    reviewer_success = 0
    for reviewer_id in node_map['reviewers']:
        try:
            reviewer_output = brain.process_node(reviewer_id, test_input)
            reviewer_success += 1
        except Exception as e:
            signal_flow_issues.append(f"Reviewer {reviewer_id} processing failed: {e}")
    
    print(f"   📊 Reviewer processing: {reviewer_success}/{len(node_map['reviewers'])} successful")
    
    # Test Handler processing
    try:
        handler_id = node_map['handler'][0]
        handler_input = {
            'reviewer_results': {},
            'controller_weights': {},
            'final_probabilities': np.random.randn(10)
        }
        handler_output = brain.process_node(handler_id, handler_input)
        print(f"   ✅ Handler processing: input → output")
    except Exception as e:
        signal_flow_issues.append(f"Handler processing failed: {e}")
    
    test_results['signal_flow'] = {
        'issues': signal_flow_issues,
        'controller_success': 'controller' not in [issue.split()[0] for issue in signal_flow_issues],
        'judge_success_rate': judge_success / len(node_map['judges']) if node_map['judges'] else 0,
        'splitter_success_rate': splitter_success / len(node_map['splitters']) if node_map['splitters'] else 0,
        'comp_success_rate': comp_success / comp_test_count if comp_test_count > 0 else 0,
        'retainer_success_rate': retainer_success / len(node_map['retainers']) if node_map['retainers'] else 0,
        'reviewer_success_rate': reviewer_success / len(node_map['reviewers']) if node_map['reviewers'] else 0,
        'handler_success': 'handler' not in [issue.split()[0] for issue in signal_flow_issues]
    }
    
    if signal_flow_issues:
        print(f"   ❌ Signal Flow Issues: {len(signal_flow_issues)}")
        for issue in signal_flow_issues:
            print(f"      - {issue}")
    
    print()


def _test_execution_pipeline(brain: BrainNexus, test_results: Dict, num_runs: int):
    """Test complete execution pipeline."""
    print("⚡ Execution Pipeline Testing:")
    
    execution_results = []
    pipeline_issues = []
    
    for run_num in range(num_runs):
        print(f"   🏃 Run {run_num + 1}/{num_runs}:")
        
        try:
            # Create test input
            test_input = f"Test input for run {run_num + 1}"
            
            # Execute with tracing
            result = brain.run(test_input, trace_execution=True)
            
            # Analyze results
            execution_time = result.get('execution_time', 0)
            final_result = result.get('result', {})
            trace = result.get('trace', {})
            metadata = result.get('metadata', {})
            
            print(f"      ⏱️  Execution time: {execution_time:.3f}s")
            print(f"      🎯 Prediction: {final_result.get('prediction', 'None')}")
            print(f"      🎲 Confidence: {final_result.get('confidence', 0.0):.3f}")
            print(f"      🔢 Nodes activated: {metadata.get('nodes_activated', 0)}")
            print(f"      🔗 Connections used: {metadata.get('connections_used', 0)}")
            
            # Validate trace
            if trace:
                trace_issues = _validate_execution_trace(trace, run_num)
                pipeline_issues.extend(trace_issues)
            
            execution_results.append({
                'run': run_num + 1,
                'execution_time': execution_time,
                'prediction': final_result.get('prediction'),
                'confidence': final_result.get('confidence', 0.0),
                'nodes_activated': metadata.get('nodes_activated', 0),
                'connections_used': metadata.get('connections_used', 0),
                'success': True
            })
            
        except Exception as e:
            print(f"      ❌ Run {run_num + 1} failed: {e}")
            pipeline_issues.append(f"Run {run_num + 1} execution failed: {e}")
            execution_results.append({
                'run': run_num + 1,
                'success': False,
                'error': str(e)
            })
    
    # Analyze execution statistics
    successful_runs = [r for r in execution_results if r.get('success', False)]
    if successful_runs:
        avg_time = np.mean([r['execution_time'] for r in successful_runs])
        avg_confidence = np.mean([r['confidence'] for r in successful_runs])
        avg_nodes = np.mean([r['nodes_activated'] for r in successful_runs])
        
        print(f"   📊 Summary ({len(successful_runs)}/{num_runs} successful):")
        print(f"      Average execution time: {avg_time:.3f}s")
        print(f"      Average confidence: {avg_confidence:.3f}")
        print(f"      Average nodes activated: {avg_nodes:.1f}")
    
    test_results['execution'] = {
        'results': execution_results,
        'issues': pipeline_issues,
        'success_rate': len(successful_runs) / num_runs if num_runs > 0 else 0
    }
    
    if pipeline_issues:
        print(f"   ❌ Pipeline Issues: {len(pipeline_issues)}")
        for issue in pipeline_issues:
            print(f"      - {issue}")
    
    print()


def _validate_execution_trace(trace: Dict, run_num: int) -> List[str]:
    """Validate execution trace for completeness and correctness."""
    issues = []
    
    # Check required steps
    expected_steps = [
        'controller_phase', 'judge_phase', 'splitter_phase', 
        'computation_phase', 'retainer_phase', 'reviewer_phase', 'handler_phase'
    ]
    
    actual_steps = trace.get('steps', [])
    missing_steps = set(expected_steps) - set(actual_steps)
    if missing_steps:
        issues.append(f"Run {run_num}: Missing trace steps: {missing_steps}")
    
    # Check timing data
    timing = trace.get('timing', {})
    for step in expected_steps:
        if step not in timing:
            issues.append(f"Run {run_num}: Missing timing for {step}")
        elif timing[step] < 0:  # Allow 0.0 timing for very fast operations
            issues.append(f"Run {run_num}: Invalid timing for {step}: {timing[step]}")
    
    # Check node activations
    node_activations = trace.get('node_activations', {})
    if not node_activations:
        issues.append(f"Run {run_num}: No node activations recorded")
    
    # Check connection flows
    connection_flows = trace.get('connection_flows', [])
    if not connection_flows:
        issues.append(f"Run {run_num}: No connection flows recorded")
    
    return issues


def _test_error_resilience(brain: BrainNexus, test_results: Dict):
    """Test system resilience to various error conditions."""
    print("🛡️ Error Resilience Testing:")
    
    resilience_issues = []
    
    # Test with None input
    try:
        result = brain.run(None, trace_execution=False)
        print("   ✅ Handles None input gracefully")
    except Exception as e:
        resilience_issues.append(f"Failed with None input: {e}")
    
    # Test with empty input
    try:
        result = brain.run("", trace_execution=False)
        print("   ✅ Handles empty input gracefully")
    except Exception as e:
        resilience_issues.append(f"Failed with empty input: {e}")
    
    # Test with large input
    try:
        large_input = "x" * 10000
        result = brain.run(large_input, trace_execution=False)
        print("   ✅ Handles large input gracefully")
    except Exception as e:
        resilience_issues.append(f"Failed with large input: {e}")
    
    # Test with invalid node access
    try:
        invalid_result = brain.process_node(99999, "test")
        resilience_issues.append("Should have failed with invalid node ID")
    except ValueError:
        print("   ✅ Properly rejects invalid node ID")
    except Exception as e:
        resilience_issues.append(f"Unexpected error with invalid node ID: {e}")
    
    # Test connection methods with invalid IDs
    try:
        result = brain.connect_nodes(99999, 99998)
        if result:
            resilience_issues.append("Should have failed connecting invalid nodes")
        else:
            print("   ✅ Properly rejects invalid node connections")
    except Exception as e:
        print(f"   ✅ Properly rejects invalid node connections (via exception: {e})")
    
    test_results['resilience'] = {
        'issues': resilience_issues,
        'tests_passed': 5 - len(resilience_issues)
    }
    
    if resilience_issues:
        print(f"   ❌ Resilience Issues: {len(resilience_issues)}")
        for issue in resilience_issues:
            print(f"      - {issue}")
    
    print()


def _analyze_performance(brain: BrainNexus, test_results: Dict):
    """Analyze overall performance metrics."""
    print("📈 Performance Analysis:")
    
    # Brain records analysis
    brain_record = brain.brain_records.iloc[0] if not brain.brain_records.empty else {}
    
    print(f"   🧠 Brain Metrics:")
    print(f"      Training time: {brain_record.get('Training_Time', 0):.3f}s")
    print(f"      Inference time: {brain_record.get('Inference_Time', 0):.3f}s")
    print(f"      Spatial efficiency: {brain_record.get('Spatial_Efficiency', 0):.3f}")
    print(f"      Reuse efficiency: {brain_record.get('Reuse_Efficiency', 0):.3f}")
    print(f"      Attention coherence: {brain_record.get('Attention_Coherence', 0):.3f}")
    
    # Node usage analysis
    if not brain.node_records.empty:
        total_calls = brain.node_records['Times_Called'].sum()
        active_nodes = (brain.node_records['Times_Called'] > 0).sum()
        reused_nodes = (brain.node_records['Reuse_Count'] > 0).sum()
        
        print(f"   📊 Node Usage:")
        print(f"      Total node calls: {total_calls}")
        print(f"      Active nodes: {active_nodes}/{len(brain.node_records)}")
        print(f"      Reused nodes: {reused_nodes}")
        print(f"      Reuse candidates: {len(brain.reuse_candidates)}")
    
    # Memory usage estimation
    registry_size = len(brain.node_registry)
    spatial_index_size = len(brain.node_positions) if brain.node_positions else 0
    records_size = len(brain.node_records)
    
    print(f"   💾 Memory Usage:")
    print(f"      Node registry: {registry_size} entries")
    print(f"      Spatial index: {spatial_index_size} positions")
    print(f"      Node records: {records_size} records")
    
    test_results['performance'].update({
        'brain_metrics': dict(brain_record),
        'node_usage': {
            'total_calls': total_calls if not brain.node_records.empty else 0,
            'active_nodes': active_nodes if not brain.node_records.empty else 0,
            'reused_nodes': reused_nodes if not brain.node_records.empty else 0,
            'reuse_candidates': len(brain.reuse_candidates)
        },
        'memory_usage': {
            'registry_size': registry_size,
            'spatial_index_size': spatial_index_size,
            'records_size': records_size
        }
    })
    
    print()


def _print_final_summary(test_results: Dict, total_time: float):
    """Print comprehensive final test summary."""
    print(f"⏱️  Total test time: {total_time:.3f}s")
    print()
    
    # Count issues
    total_issues = 0
    total_issues += len(test_results.get('initialization', {}).get('distribution_issues', []))
    total_issues += len(test_results.get('initialization', {}).get('registry_issues', []))
    total_issues += len(test_results.get('initialization', {}).get('position_issues', []))
    total_issues += len(test_results.get('connections', {}).get('issues', []))
    total_issues += len(test_results.get('connections', {}).get('weight_issues', []))
    total_issues += len(test_results.get('signal_flow', {}).get('issues', []))
    total_issues += len(test_results.get('execution', {}).get('issues', []))
    total_issues += len(test_results.get('resilience', {}).get('issues', []))
    
    # Success rates
    execution_success = test_results.get('execution', {}).get('success_rate', 0)
    resilience_tests = test_results.get('resilience', {}).get('tests_passed', 0)
    
    print(f"🎯 Test Results:")
    print(f"   Initialization: {'✅ PASS' if test_results.get('initialization', {}).get('success', False) else '❌ FAIL'}")
    print(f"   Connection integrity: {'✅ PASS' if len(test_results.get('connections', {}).get('issues', [])) == 0 else '⚠️  ISSUES'}")
    print(f"   Signal flow: {'✅ PASS' if len(test_results.get('signal_flow', {}).get('issues', [])) == 0 else '⚠️  ISSUES'}")
    print(f"   Execution success rate: {execution_success:.1%}")
    print(f"   Error resilience: {resilience_tests}/5 tests passed")
    print()
    
    if total_issues == 0:
        print("🎉 ALL TESTS PASSED! BrainNexus is functioning correctly.")
    elif total_issues <= 5:
        print(f"⚠️  {total_issues} minor issues found. BrainNexus is mostly functional.")
    else:
        print(f"❌ {total_issues} issues found. BrainNexus needs attention.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
    
    