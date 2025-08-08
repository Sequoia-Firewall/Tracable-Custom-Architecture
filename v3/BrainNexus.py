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
from BrainSegment import NexusSegment
# Mistral v3 Tekken tokenizer imports
try:
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.tool_calls import Function, Tool
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    MISTRAL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Mistral tokenizer not available: {e}")
    print("   Falling back to simple tokenization. Install with: pip install mistral-common")
    MISTRAL_AVAILABLE = False
import pandas as pd
import numpy as np
import random
import math
from typing import Any, Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict, deque
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
import os
import sys
import traceback
import pickle
import hashlib
import pickle
from datetime import datetime
import hashlib
import numpy as np

class BrainNexus:
    def __init__(self, dimensions: int = 4, node_count_pre: Optional[int] = None, demo: bool = False, 
                 output_config: Optional[Dict[str, Any]] = None, mode: str = 'default'):
        self.demo = demo        
        self.dimensions = dimensions
        self.epoch_count_or_type = 'dynamic'
        self.learning_rate = 0.01 if not demo else 0.05  # More aggressive in demo
        self.epoch_giveup_rate = 0.001 if not demo else 0.02  # Give up faster in demo
        
        # Calculate dynamic pre-node count based on dimensional hypercube vertices
        # Each dimension creates 2 polarities (+/-), so total vertices = 2^dimensions
        # This represents the optimal number of entrance nodes or segments for complete coverage
        if node_count_pre is None:
            self.entrance_node_count_pre = 2 ** dimensions
            self._dynamic_pre_nodes = True
        else:
            self.entrance_node_count_pre = node_count_pre
            self._dynamic_pre_nodes = False
        
        # Store dimensional partitioning info
        self.dimensional_vertices = 2 ** dimensions
        self.dimensional_coverage = {
            'vertices': self.dimensional_vertices,
            'quadrants_2d': 4 if dimensions >= 2 else 1,
            'octants_3d': 8 if dimensions >= 3 else (4 if dimensions == 2 else 1),
            'hypercubes_4d': 16 if dimensions >= 4 else (8 if dimensions == 3 else (4 if dimensions == 2 else 1))
        }
        
        self.neural_nodes = []
        self.mode = mode
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
            "Dimensional_Vertices": self.dimensional_vertices,  # 2^dimensions hypercube vertices
            "Dynamic_Pre_Nodes": self._dynamic_pre_nodes,      # Whether pre-node count was auto-calculated
            "Entrance_Node_Count_Post": 0,              # Actual input nodes created
            "Entrance_Node_Count_Pre": self.entrance_node_count_pre,  # Initial estimate (now dynamic)
            "Quadrants_2D": self.dimensional_coverage['quadrants_2d'],     # 2D quadrant coverage
            "Octants_3D": self.dimensional_coverage['octants_3d'],         # 3D octant coverage  
            "Hypercubes_4D": self.dimensional_coverage['hypercubes_4d'],   # 4D+ hypercube coverage
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
            "Dimensional_Coverage_Ratio": 0.0,         # Actual nodes / optimal nodes ratio
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
            
            # Display dimensional information
            print(f"🎯 Dimensional Configuration:")
            print(f"   Dimensions: {self.dimensions}")
            print(f"   Hypercube vertices (2^{self.dimensions}): {self.dimensional_vertices}")
            print(f"   Entrance nodes (pre): {self.entrance_node_count_pre}")
            print(f"   Dynamic pre-nodes: {'YES' if self._dynamic_pre_nodes else 'NO'}")
            if self.dimensions == 2:
                print(f"   2D Quadrants: NE, NW, SE, SW (4 total)")
            elif self.dimensions == 3:
                print(f"   3D Octants: 8 total (+/- combinations)")
            elif self.dimensions >= 4:
                print(f"   4D+ Hypercubes: {self.dimensional_vertices} total vertices")
        
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
        assert position is not None, "Position should not be None at this point"
        self._update_spatial_data(position, node_id)
        
        # Update records efficiently
        self._add_node_record(new_node, node_group, start_time)
        
        if self.demo:
            print(f"✓ Added {new_node.__class__.__name__} node #{node_id}")
        
        # Periodic cleanup based on node count
        self.schedule_periodic_cleanup()
        
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

    def generate_hypercube_vertices(self, radius: float = 10.0) -> List[List[float]]:
        """
        Generate all vertices of a hypercube for optimal entrance node placement.
        Each vertex represents a unique combination of dimensional polarities.
        
        Args:
            radius: Distance from origin to each vertex
            
        Returns:
            List of positions representing all hypercube vertices
        """
        vertices = []
        
        # Generate all combinations of +/- for each dimension
        # For n dimensions, there are 2^n vertices
        for i in range(2 ** self.dimensions):
            vertex = []
            for dim in range(self.dimensions):
                # Check if bit 'dim' is set in number 'i'
                polarity = 1 if (i >> dim) & 1 else -1
                vertex.append(polarity * radius)
            vertices.append(vertex)
        
        return vertices

    def generate_optimal_entrance_positions(self, radius: float = 10.0) -> List[List[float]]:
        """
        Generate optimal positions for entrance nodes based on hypercube vertices.
        This ensures even coverage of the dimensional space.
        
        Args:
            radius: Distance from origin for entrance nodes
            
        Returns:
            List of optimal entrance node positions
        """
        if self._dynamic_pre_nodes:
            # Use hypercube vertices for complete dimensional coverage
            return self.generate_hypercube_vertices(radius)
        else:
            # Use smart positioning for manually specified count
            positions = []
            for i in range(self.entrance_node_count_pre):
                # Distribute evenly around the hypersphere
                angle_step = (2 * np.pi) / self.entrance_node_count_pre
                position = []
                
                for dim in range(self.dimensions):
                    if dim == 0:
                        # First dimension uses circular distribution
                        coord = radius * np.cos(i * angle_step)
                    elif dim == 1:
                        # Second dimension uses circular distribution  
                        coord = radius * np.sin(i * angle_step)
                    else:
                        # Higher dimensions use spherical-like distribution
                        coord = radius * np.cos(i * angle_step + dim * np.pi / 4)
                    position.append(coord)
                
                positions.append(position)
            
            return positions

    def calculate_dimensional_coverage(self) -> float:
        """
        Calculate how well the current nodes cover the dimensional space.
        
        Returns:
            Coverage ratio (0.0 to 1.0+) where 1.0 means optimal coverage
        """
        if len(self.neural_nodes) == 0:
            return 0.0
        
        # Calculate actual coverage ratio
        # Count all nodes that could serve as entrance nodes
        entrance_nodes = [node for node in self.neural_nodes 
                         if hasattr(node, 'node_type') and 
                         ('entrance' in str(node.node_type).lower() or
                          'controller' in str(node.node_type).lower())]
        
        actual_entrance_count = len(entrance_nodes)
        optimal_count = self.dimensional_vertices
        
        coverage_ratio = actual_entrance_count / optimal_count if optimal_count > 0 else 0.0
        
        # Update brain records
        if len(self.brain_records) > 0:
            self.brain_records.loc[0, 'Dimensional_Coverage_Ratio'] = coverage_ratio
        
        return coverage_ratio

    def get_dimensional_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dimensional configuration information.
        
        Returns:
            Dictionary with dimensional setup details
        """
        coverage_ratio = self.calculate_dimensional_coverage()
        vertices = self.generate_hypercube_vertices()
        
        return {
            'dimensions': self.dimensions,
            'hypercube_vertices': self.dimensional_vertices,
            'vertices_coordinates': vertices[:8] if len(vertices) > 8 else vertices,  # Show first 8
            'entrance_nodes_pre': self.entrance_node_count_pre,
            'entrance_nodes_post': len([n for n in self.neural_nodes 
                                      if 'entrance' in getattr(n, 'node_type', '').lower()]),
            'dynamic_pre_nodes': self._dynamic_pre_nodes,
            'coverage_ratio': coverage_ratio,
            'coverage_status': 'OPTIMAL' if 0.9 <= coverage_ratio <= 1.1 
                              else 'OVER' if coverage_ratio > 1.1 
                              else 'UNDER',
            'dimensional_partitions': self.dimensional_coverage,
            'optimal_positions_available': len(vertices)
        }

    def create_optimal_entrance_nodes(self, node_type: str = 'Controller', 
                                    radius: float = 10.0) -> List[int]:
        """
        Create entrance nodes at optimal hypercube vertex positions.
        
        Args:
            node_type: Type of nodes to create at entrance positions
            radius: Distance from origin for entrance nodes
            
        Returns:
            List of created node IDs
        """
        optimal_positions = self.generate_optimal_entrance_positions(radius)
        created_node_ids = []
        
        if self.demo:
            print(f"🎯 Creating {len(optimal_positions)} optimal entrance nodes:")
            print(f"   Node type: {node_type}")
            print(f"   Positions based on: {'hypercube vertices' if self._dynamic_pre_nodes else 'even distribution'}")
        
        for i, position in enumerate(optimal_positions):
            node_id = self.add_neural_node(
                node_type=node_type,
                position=position,
                node_group=f'entrance_optimal_{i}'
            )
            created_node_ids.append(node_id)
            
            if self.demo:
                pos_str = [f"{p:.1f}" for p in position[:3]]  # Show first 3 dimensions
                print(f"   ✓ Node {node_id} at [{', '.join(pos_str)}{'...' if len(position) > 3 else ''}]")
        
        # Update coverage ratio
        coverage_ratio = self.calculate_dimensional_coverage()
        
        if self.demo:
            print(f"   Dimensional coverage: {coverage_ratio:.2%}")
            if coverage_ratio >= 0.9:
                print(f"   ✅ Excellent dimensional coverage achieved!")
            elif coverage_ratio >= 0.7:
                print(f"   ⚠️  Good coverage, consider adding more entrance nodes")
            else:
                print(f"   ❌ Poor coverage, more entrance nodes needed")
        
        return created_node_ids

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
        # Get node position safely 
        node_position = getattr(node, 'node_position', getattr(node, 'position', [0.0] * self.dimensions))
        if isinstance(node_position, tuple):
            node_position = list(node_position)
        elif not isinstance(node_position, list):
            node_position = [0.0] * self.dimensions
            
        # Calculate spatial affinity
        spatial_affinity = self._calculate_spatial_affinity(node_position)
        node.spatial_affinity = spatial_affinity
        
        # Initialize attention weights
        attention_weights = getattr(node, 'attention_weights', None)
        if attention_weights is None:
            attention_weights = np.random.normal(0, 0.1, 
                                                (self.attention_layers, self.attention_heads, self.attention_dim))
            node.attention_weights = attention_weights
        
        # Create record - handle different node types with different attribute names
        node_id_value = getattr(node, 'node_id', getattr(node, 'judge_id', getattr(node, 'splitter_id', self.next_node_id - 1)))
        node_type_value = getattr(node, 'node_type', node.__class__.__name__)
        
        # Get attention weights safely for storage
        attention_weights_for_storage = getattr(node, 'attention_weights', [])
        try:
            if isinstance(attention_weights_for_storage, np.ndarray):
                attention_weights_for_storage = attention_weights_for_storage.tolist()
            elif not isinstance(attention_weights_for_storage, list):
                attention_weights_for_storage = []
        except:
            attention_weights_for_storage = []
        
        new_record = pd.DataFrame([{
            'Node_ID': node_id_value,
            'Node_Type': node_type_value,
            'Node_Position': node_position,
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
            'Attention_Weights': attention_weights_for_storage,
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
        
        # Get existing positions and normalize dimensions
        existing_nodes = self.neural_nodes[:-1]  # Exclude current node
        if len(existing_nodes) == 0:
            return 1.0
        
        # Normalize all positions to the same dimension space
        target_dim = self.dimensions
        position_array = np.array(position)
        
        # Ensure position has correct dimensions
        if len(position_array) < target_dim:
            position_array = np.pad(position_array, (0, target_dim - len(position_array)), 'constant')
        elif len(position_array) > target_dim:
            position_array = position_array[:target_dim]
        
        # Collect and normalize existing positions
        normalized_positions = []
        for node in existing_nodes:
            if hasattr(node, 'node_position'):
                node_pos = np.array(node.node_position)
                # Normalize to target dimensions
                if len(node_pos) < target_dim:
                    node_pos = np.pad(node_pos, (0, target_dim - len(node_pos)), 'constant')
                elif len(node_pos) > target_dim:
                    node_pos = node_pos[:target_dim]
                normalized_positions.append(node_pos)
        
        if len(normalized_positions) == 0:
            return 1.0
        
        existing_positions = np.array(normalized_positions)
        distances = np.linalg.norm(existing_positions - position_array, axis=1)
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
        
        # Cleanup after batch operations if significant changes were made
        if successful_moves > 10:
            cleanup_stats = self.schedule_periodic_cleanup(cleanup_interval=50, cleanup_tier='light')
            if cleanup_stats and self.demo:
                print(f"   Post-batch cleanup: {cleanup_stats['cleaned_items']}")
        
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

    # =====================================
    # MEMORY CLEANUP SYSTEM
    # =====================================
    
    def cleanup_memory(self, cleanup_tier: str = 'partial', force_cleanup: bool = False, 
                      cleanup_segments: bool = True) -> Dict[str, Any]:
        """
        Multi-tier memory cleanup system for BrainNexus architecture.
        
        Args:
            cleanup_tier: Level of cleanup - 'light', 'partial', 'aggressive', 'nuclear'
            force_cleanup: Skip safety checks and force cleanup
            cleanup_segments: Whether to also clean segment memory
            
        Returns:
            Dict with cleanup statistics and results
        """
        start_time = time.time()
        cleanup_stats = {
            'tier': cleanup_tier,
            'start_time': datetime.now().isoformat(),
            'pre_cleanup_memory': self._get_memory_usage_stats(),
            'cleaned_items': defaultdict(int),
            'errors': [],
            'segments_cleaned': 0
        }
        
        if self.demo:
            print(f"🧹 Starting {cleanup_tier.upper()} memory cleanup...")
            print(f"   Pre-cleanup: {cleanup_stats['pre_cleanup_memory']}")
        
        try:
            # Execute cleanup based on tier
            if cleanup_tier == 'light':
                self._light_cleanup(cleanup_stats, force_cleanup)
            elif cleanup_tier == 'partial':
                self._partial_cleanup(cleanup_stats, force_cleanup)
            elif cleanup_tier == 'aggressive':
                self._aggressive_cleanup(cleanup_stats, force_cleanup)
            elif cleanup_tier == 'nuclear':
                self._nuclear_cleanup(cleanup_stats, force_cleanup)
            else:
                raise ValueError(f"Unknown cleanup tier: {cleanup_tier}")
            
            # Clean segments if requested
            if cleanup_segments and hasattr(self, 'segments'):
                cleanup_stats['segments_cleaned'] = self._cleanup_segments(cleanup_tier, force_cleanup)
            
            # Update brain records
            self._update_cleanup_metrics(cleanup_stats)
            
        except Exception as e:
            cleanup_stats['errors'].append(f"Cleanup error: {str(e)}")
            if self.demo:
                print(f"❌ Cleanup error: {e}")
        
        # Calculate final statistics
        cleanup_time = time.time() - start_time
        cleanup_stats['cleanup_time'] = cleanup_time
        cleanup_stats['post_cleanup_memory'] = self._get_memory_usage_stats()
        cleanup_stats['memory_freed'] = self._calculate_memory_freed(
            cleanup_stats['pre_cleanup_memory'], 
            cleanup_stats['post_cleanup_memory']
        )
        
        if self.demo:
            print(f"✅ {cleanup_tier.upper()} cleanup completed in {cleanup_time:.3f}s")
            print(f"   Items cleaned: {dict(cleanup_stats['cleaned_items'])}")
            print(f"   Memory freed: {cleanup_stats['memory_freed']}")
            if cleanup_stats['errors']:
                print(f"   Errors: {len(cleanup_stats['errors'])}")
        
        return cleanup_stats

    def _light_cleanup(self, stats: Dict[str, Any], force: bool = False):
        """Light cleanup - only remove expired and obviously stale data."""
        
        # Clean inference cache - remove old entries
        if hasattr(self, 'inference_cache'):
            original_size = len(self.inference_cache)
            if original_size > 1000:  # Only clean if cache is large
                # Keep only the most recent 500 entries
                cache_items = list(self.inference_cache.items())
                self.inference_cache = dict(cache_items[-500:])
                stats['cleaned_items']['inference_cache'] = original_size - len(self.inference_cache)
        
        # Trim routing history if too long
        if hasattr(self, 'routing_history'):
            original_size = len(self.routing_history)
            if original_size > 750:
                # Keep only recent 500 entries
                trimmed = list(self.routing_history)[-500:]
                self.routing_history.clear()
                self.routing_history.extend(trimmed)
                stats['cleaned_items']['routing_history'] = original_size - len(self.routing_history)
        
        # Clean node input trackers - limit to recent entries
        cleaned_trackers = 0
        for node_id, node in self.node_registry.items():
            if hasattr(node, 'input_tracker') and len(node.input_tracker) > 100:
                node.input_tracker = node.input_tracker[-50:]  # Keep only recent 50
                cleaned_trackers += 1
        stats['cleaned_items']['node_input_trackers'] = cleaned_trackers
        
        # Trim node usage history
        usage_cleaned = 0
        for node_id in list(self.node_usage_history.keys()):
            if len(self.node_usage_history[node_id]) > 100:
                self.node_usage_history[node_id] = self.node_usage_history[node_id][-50:]
                usage_cleaned += 1
        stats['cleaned_items']['node_usage_history'] = usage_cleaned

    def _partial_cleanup(self, stats: Dict[str, Any], force: bool = False):
        """Partial cleanup - moderate memory cleanup with some data loss."""
        
        # Start with light cleanup
        self._light_cleanup(stats, force)
        
        # Clear term embeddings cache if large
        if hasattr(self, 'term_embeddings'):
            original_size = len(self.term_embeddings)
            if original_size > 500 or force:
                self.term_embeddings.clear()
                stats['cleaned_items']['term_embeddings'] = original_size
        
        # Clear attention masks cache
        if hasattr(self, 'attention_masks'):
            original_size = len(self.attention_masks)
            if original_size > 0:
                self.attention_masks.clear()
                stats['cleaned_items']['attention_masks'] = original_size
        
        # Reset reuse candidates if too many
        if len(self.reuse_candidates) > 100:
            original_size = len(self.reuse_candidates)
            # Keep only the most recently used candidates
            recent_candidates = set()
            for node_id in list(self.reuse_candidates):
                if node_id in self.node_registry:
                    node = self.node_registry[node_id]
                    if getattr(node, 'times_called', 0) > self.reuse_threshold * 5:
                        recent_candidates.add(node_id)
                if len(recent_candidates) >= 50:
                    break
            self.reuse_candidates = recent_candidates
            stats['cleaned_items']['reuse_candidates'] = original_size - len(self.reuse_candidates)
        
        # Clean DataFrame records - remove unused columns or optimize data types
        original_rows = len(self.node_records)
        if original_rows > 1000:
            # Keep only records for existing nodes
            existing_node_ids = set(self.node_registry.keys())
            self.node_records = self.node_records[
                self.node_records['Node_ID'].isin(existing_node_ids)
            ].copy()
            stats['cleaned_items']['node_records'] = original_rows - len(self.node_records)

    def _aggressive_cleanup(self, stats: Dict[str, Any], force: bool = False):
        """Aggressive cleanup - significant memory cleanup with data loss."""
        
        # Start with partial cleanup
        self._partial_cleanup(stats, force)
        
        # Clear inference cache completely
        if hasattr(self, 'inference_cache') and len(self.inference_cache) > 0:
            original_size = len(self.inference_cache)
            self.inference_cache.clear()
            stats['cleaned_items']['inference_cache'] += original_size
        
        # Clear routing history completely
        if hasattr(self, 'routing_history') and len(self.routing_history) > 0:
            original_size = len(self.routing_history)
            self.routing_history.clear()
            stats['cleaned_items']['routing_history'] += original_size
        
        # Reset all node input trackers
        cleaned_trackers = 0
        for node_id, node in self.node_registry.items():
            if hasattr(node, 'input_tracker') and len(node.input_tracker) > 0:
                node.input_tracker.clear()
                cleaned_trackers += 1
        stats['cleaned_items']['node_input_trackers'] += cleaned_trackers
        
        # Clear node usage history for inactive nodes
        inactive_nodes = []
        for node_id in list(self.node_usage_history.keys()):
            if node_id in self.node_registry:
                node = self.node_registry[node_id]
                if getattr(node, 'times_called', 0) < self.reuse_threshold:
                    inactive_nodes.append(node_id)
        
        for node_id in inactive_nodes:
            del self.node_usage_history[node_id]
        stats['cleaned_items']['inactive_node_histories'] = len(inactive_nodes)
        
        # Rebuild spatial index to optimize memory
        if self.spatial_index is not None:
            self._rebuild_spatial_index()
            stats['cleaned_items']['spatial_index_rebuilt'] = 1

    def _nuclear_cleanup(self, stats: Dict[str, Any], force: bool = False):
        """Nuclear cleanup - complete memory reset with major data loss."""
        
        if not force:
            # Safety check - confirm nuclear cleanup
            total_nodes = len(self.node_registry)
            if total_nodes > 10:
                stats['errors'].append("Nuclear cleanup requires force=True for safety with >10 nodes")
                return
        
        # Clear all caches and histories
        cache_attrs = ['inference_cache', 'term_embeddings', 'attention_masks']
        for attr in cache_attrs:
            if hasattr(self, attr):
                original_size = len(getattr(self, attr, {}))
                if original_size > 0:
                    setattr(self, attr, {})
                    stats['cleaned_items'][attr] = original_size
        
        # Clear all tracking data
        if hasattr(self, 'routing_history'):
            original_size = len(self.routing_history)
            self.routing_history.clear()
            stats['cleaned_items']['routing_history'] = original_size
        
        # Clear all node histories and tracking
        self.node_usage_history.clear()
        self.reuse_candidates.clear()
        stats['cleaned_items']['all_node_tracking'] = 1
        
        # Reset all node input trackers
        for node_id, node in self.node_registry.items():
            if hasattr(node, 'input_tracker'):
                node.input_tracker.clear()
            if hasattr(node, 'times_called'):
                node.times_called = 0
        
        # Reset spatial tracking (keep positions but clear index)
        self.spatial_index = None
        self.node_id_to_index.clear()
        
        # Reset node_positions and rebuild mapping
        if len(self.neural_nodes) > 0:
            self.node_positions.clear()
            for i, node in enumerate(self.neural_nodes):
                if hasattr(node, 'node_position'):
                    self.node_positions.append(node.node_position)
                    self.node_id_to_index[getattr(node, 'node_id', i)] = i
        
        stats['cleaned_items']['complete_reset'] = 1

    def _cleanup_segments(self, cleanup_tier: str, force: bool = False) -> int:
        """Clean memory in all segments."""
        segments_cleaned = 0
        
        # Check if we have segment references
        segments_to_clean = []
        
        # Look for segments in various possible locations
        if hasattr(self, 'segments') and isinstance(self.segments, dict):
            segments_to_clean.extend(self.segments.values())
        elif hasattr(self, 'loaded_segments') and isinstance(self.loaded_segments, list):
            segments_to_clean.extend(self.loaded_segments)
        
        for segment in segments_to_clean:
            try:
                if hasattr(segment, 'cleanup_memory'):
                    segment.cleanup_memory(cleanup_tier, force)
                else:
                    # Manual segment cleanup
                    self._manual_segment_cleanup(segment, cleanup_tier, force)
                segments_cleaned += 1
            except Exception as e:
                if self.demo:
                    print(f"⚠️  Segment cleanup error: {e}")
        
        return segments_cleaned

    def _manual_segment_cleanup(self, segment: Any, cleanup_tier: str, force: bool = False):
        """Manually clean segment memory when segment doesn't have cleanup_memory method."""
        
        # Clean attention cache
        if hasattr(segment, 'attention_cache') and isinstance(segment.attention_cache, dict):
            if cleanup_tier in ['aggressive', 'nuclear']:
                segment.attention_cache.clear()
            elif cleanup_tier == 'partial' and len(segment.attention_cache) > 100:
                # Keep only recent entries
                cache_items = list(segment.attention_cache.items())[-50:]
                segment.attention_cache.clear()
                segment.attention_cache.update(cache_items)
        
        # Clean processing results
        if hasattr(segment, 'processing_results') and isinstance(segment.processing_results, dict):
            if cleanup_tier in ['aggressive', 'nuclear']:
                segment.processing_results.clear()
            elif cleanup_tier == 'partial' and len(segment.processing_results) > 200:
                # Keep only recent results
                results_items = list(segment.processing_results.items())[-100:]
                segment.processing_results.clear()
                segment.processing_results.update(results_items)
        
        # Clean result cache
        if hasattr(segment, 'result_cache') and isinstance(segment.result_cache, dict):
            if cleanup_tier in ['partial', 'aggressive', 'nuclear']:
                segment.result_cache.clear()
        
        # Clean pattern memory
        if hasattr(segment, 'pattern_memory') and hasattr(segment.pattern_memory, 'clear'):
            if cleanup_tier == 'nuclear':
                segment.pattern_memory.clear()
            elif cleanup_tier == 'aggressive' and len(segment.pattern_memory) > 200:
                # Keep only recent 100 patterns
                recent_patterns = list(segment.pattern_memory)[-100:]
                segment.pattern_memory.clear()
                segment.pattern_memory.extend(recent_patterns)

    def _get_memory_usage_stats(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        stats = {
            'total_nodes': len(self.neural_nodes),
            'registry_size': len(self.node_registry),
            'inference_cache': len(getattr(self, 'inference_cache', {})),
            'routing_history': len(getattr(self, 'routing_history', [])),
            'term_embeddings': len(getattr(self, 'term_embeddings', {})),
            'attention_masks': len(getattr(self, 'attention_masks', {})),
            'node_usage_history': len(self.node_usage_history),
            'reuse_candidates': len(self.reuse_candidates),
            'node_records_rows': len(self.node_records),
            'spatial_positions': len(self.node_positions)
        }
        
        # Add node input tracker sizes
        total_input_tracker_size = 0
        for node in self.node_registry.values():
            if hasattr(node, 'input_tracker'):
                total_input_tracker_size += len(node.input_tracker)
        stats['total_input_trackers'] = total_input_tracker_size
        
        return stats

    def _calculate_memory_freed(self, pre_stats: Dict[str, int], post_stats: Dict[str, int]) -> Dict[str, int]:
        """Calculate memory freed by cleanup."""
        freed = {}
        for key in pre_stats:
            if key in post_stats:
                freed[key] = pre_stats[key] - post_stats[key]
        
        # Calculate total items freed
        freed['total_items'] = sum(max(0, val) for val in freed.values())
        return freed

    def _update_cleanup_metrics(self, cleanup_stats: Dict[str, Any]):
        """Update brain records with cleanup metrics."""
        if len(self.brain_records) > 0:
            # Add cleanup time to training time (as maintenance time)
            current_time = self.brain_records.loc[0, 'Training_Time']
            if pd.isna(current_time):
                current_time = 0.0
            else:
                current_time = float(current_time)
            
            self.brain_records.loc[0, 'Training_Time'] = current_time + cleanup_stats.get('cleanup_time', 0.0)

    def schedule_periodic_cleanup(self, cleanup_interval: int = 1000, cleanup_tier: str = 'light'):
        """
        Schedule periodic cleanup based on node creation count.
        Call this method periodically during training or processing.
        """
        if not hasattr(self, '_cleanup_counter'):
            self._cleanup_counter = 0
            self._cleanup_config = {
                'interval': cleanup_interval,
                'tier': cleanup_tier,
                'last_cleanup': 0
            }
        
        self._cleanup_counter += 1
        
        # Trigger cleanup if interval reached
        if self._cleanup_counter - self._cleanup_config['last_cleanup'] >= cleanup_interval:
            if self.demo:
                print(f"🕐 Triggering scheduled {cleanup_tier} cleanup (counter: {self._cleanup_counter})")
            
            cleanup_stats = self.cleanup_memory(cleanup_tier)
            self._cleanup_config['last_cleanup'] = self._cleanup_counter
            
            return cleanup_stats
        
        return None

    def emergency_cleanup(self) -> Dict[str, Any]:
        """
        Emergency cleanup when memory usage is critical.
        Automatically determines cleanup tier based on current state.
        """
        memory_stats = self._get_memory_usage_stats()
        total_items = sum(memory_stats.values())
        
        # Determine cleanup tier based on memory usage
        if total_items > 10000:
            cleanup_tier = 'nuclear'
            force = True
        elif total_items > 5000:
            cleanup_tier = 'aggressive'
            force = False
        elif total_items > 2000:
            cleanup_tier = 'partial'
            force = False
        else:
            cleanup_tier = 'light'
            force = False
        
        if self.demo:
            print(f"🚨 EMERGENCY CLEANUP: {total_items} items detected, using {cleanup_tier.upper()} tier")
        
        return self.cleanup_memory(cleanup_tier, force_cleanup=force)

    def get_memory_status(self) -> Dict[str, Any]:
        """Get detailed memory status and recommendations."""
        stats = self._get_memory_usage_stats()
        total_items = sum(stats.values())
        
        # Determine memory pressure level
        if total_items > 10000:
            pressure = 'CRITICAL'
            recommendation = 'emergency_cleanup() immediately'
        elif total_items > 5000:
            pressure = 'HIGH'
            recommendation = "cleanup_memory('aggressive')"
        elif total_items > 2000:
            pressure = 'MODERATE'
            recommendation = "cleanup_memory('partial')"
        elif total_items > 1000:
            pressure = 'LOW'
            recommendation = "cleanup_memory('light')"
        else:
            pressure = 'OPTIMAL'
            recommendation = 'No cleanup needed'
        
        return {
            'memory_stats': stats,
            'total_items': total_items,
            'pressure_level': pressure,
            'recommendation': recommendation,
            'cleanup_available': True
        }
    @classmethod
    def load_segment(cls, 
                    filepath: str,
                    brain_nexus_ref: Any,
                    validate_integrity: bool = True,
                    restore_connections: bool = True,
                    demo: bool = False) -> 'NexusSegment':
        """
        Load a complete NexusSegment from a saved .pkl file.
        
        Args:
            filepath (str): Path to the saved segment file
            brain_nexus_ref (Any): Reference to the parent BrainNexus instance
            validate_integrity (bool): Whether to validate the loaded data integrity
            restore_connections (bool): Whether to restore node connections in brain_nexus
            demo (bool): Enable debug output
            
        Returns:
            NexusSegment: Loaded and restored segment instance
            
        Raises:
            FileNotFoundError: If segment file doesn't exist
            ValueError: If data validation fails
            PicklingError: If file cannot be unpickled
        """
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Segment file not found: {filepath}")
        
        try:
            # Load the save package
            with open(filepath, 'rb') as f:
                save_package = pickle.load(f)
            
            if demo:
                print(f"📁 Loading segment from: {filepath}")
            
            # Extract components
            metadata = save_package['metadata']
            segment_data = save_package['segment_data']
            validation_hash = save_package.get('validation_hash', '')
            
            if demo:
                print(f"   Segment ID: {metadata['segment_id']}")
                print(f"   Dimensional Signature: {metadata['dimensional_signature']}")
                print(f"   Total Nodes: {metadata['total_nodes']}")
                print(f"   Save Timestamp: {metadata['save_timestamp']}")
            
            # Validate integrity if requested
            if validate_integrity:
                cls._validate_loaded_data(segment_data, validation_hash, demo)
            
            # Create new segment instance
            segment = cls._create_segment_from_data(segment_data, brain_nexus_ref, demo)
            
            # Restore nodes in brain_nexus
            cls._restore_nodes_to_brain_nexus(segment, segment_data, brain_nexus_ref, demo)
            
            # Restore connections if requested
            if restore_connections:
                cls._restore_node_connections(segment, segment_data, brain_nexus_ref, demo)
            
            # Update load metadata
            segment.last_load_time = datetime.now()
            segment.last_load_path = filepath
            segment.load_metadata = metadata
            
            if demo:
                print(f"✅ Segment {segment.segment_id} loaded successfully!")
                cls._print_load_summary(segment, metadata)
            
            return segment
            
        except Exception as e:
            error_msg = f"Failed to load segment from {filepath}: {str(e)}"
            if demo:
                print(f"❌ {error_msg}")
            raise ValueError(error_msg) from e

    @classmethod
    def _validate_loaded_data(cls, segment_data: Dict[str, Any], validation_hash: str, demo: bool = False):
        """Validate the integrity of loaded segment data."""
        if not validation_hash:
            if demo:
                print("⚠️  No validation hash found, skipping integrity check")
            return
        
        # Recreate validation hash
        hash_data = {
            'segment_id': segment_data['segment_id'],
            'dimensional_signature': segment_data['dimensional_signature'],
            'node_count': len(segment_data['segment_nodes_data']),
            'config_hash': str(hash(str(sorted(segment_data['config'].items()))))
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        computed_hash = hashlib.md5(hash_string.encode()).hexdigest()
        
        if computed_hash != validation_hash:
            raise ValueError(f"Data integrity check failed: {computed_hash} != {validation_hash}")
        
        if demo:
            print("✓ Data integrity validated")

    @classmethod
    def _create_segment_from_data(cls, segment_data: Dict[str, Any], brain_nexus_ref: Any, demo: bool = False) -> 'NexusSegment':
        """Create a new NexusSegment instance from loaded data."""
        # Create segment with core parameters
        segment = cls(
            segment_id=segment_data['segment_id'],
            dimensional_assignment=segment_data['dimensional_assignment'],
            brain_nexus_ref=brain_nexus_ref,
            hypercube_bounds=segment_data['hypercube_bounds'],
            segment_config=segment_data['config'],
            demo=demo
        )
        
        # Restore dimensional properties
        segment.dimensions = segment_data['dimensions']
        segment.max_dimension_index = segment_data['max_dimension_index']
        segment.effective_brain_dimensions = segment_data['effective_brain_dimensions']
        segment.dimensional_signature = segment_data['dimensional_signature']
        segment.segment_center = segment_data['segment_center']
        segment.segment_radius = segment_data['segment_radius']
        segment.dimensional_volume = segment_data['dimensional_volume']
        segment.dimensional_density = segment_data['dimensional_density']
        
        # Restore spatial zones
        segment.spatial_zones = cls._deserialize_spatial_zones(segment_data['spatial_zones'])
        
        # Restore state and activation data
        segment.activation_state = segment_data['activation_state']
        segment.relevance_score = segment_data['relevance_score']
        segment.activation_threshold = segment_data['activation_threshold']
        segment.last_activation_time = segment_data['last_activation_time']
        segment.activation_history = deque(segment_data['activation_history'], maxlen=100)
        
        # Restore judge management
        segment.active_judges = set(segment_data['active_judges'])
        segment.judge_relevance_scores = segment_data['judge_relevance_scores']
        segment.max_active_judges = segment_data['max_active_judges']
        segment.judge_activation_ratio = segment_data['judge_activation_ratio']
        
        # Restore processing caches
        segment.attention_cache = cls._deserialize_attention_cache(segment_data['attention_cache'])
        segment.embedding_transformations = cls._deserialize_embedding_transformations(segment_data['embedding_transformations'])
        segment.positional_encodings = cls._deserialize_positional_encodings(segment_data['positional_encodings'])
        
        # Restore pipeline state
        segment.pipeline_state = segment_data['pipeline_state']
        segment.processing_results = cls._deserialize_processing_results(segment_data['processing_results'])
        
        # Restore performance metrics
        segment.computation_budget = segment_data['computation_budget']
        segment.remaining_budget = segment_data['remaining_budget']
        segment.efficiency_metrics = segment_data['efficiency_metrics']
        
        # Restore communication and memory
        segment.communication_channels = cls._deserialize_communication_channels(segment_data['communication_channels'])
        segment.shared_embeddings = cls._deserialize_shared_embeddings(segment_data['shared_embeddings'])
        segment.synchronization_points = segment_data['synchronization_points']
        segment.result_cache = cls._deserialize_result_cache(segment_data['result_cache'])
        segment.pattern_memory = deque(segment_data['pattern_memory'], maxlen=1000)
        segment.failure_patterns = deque(segment_data['failure_patterns'], maxlen=100)
        
        # Restore learning and adaptation
        segment.learning_rate = segment_data['learning_rate']
        segment.adaptation_history = segment_data['adaptation_history']
        segment.success_patterns = defaultdict(int, segment_data['success_patterns'])
        segment.dimensional_preferences = segment_data['dimensional_preferences']
        
        # Restore resource management
        segment.resource_limits = segment_data['resource_limits']
        segment.current_resources = segment_data['current_resources']
        
        # Restore quality metrics
        segment.quality_metrics = cls._deserialize_quality_metrics(segment_data['quality_metrics'])
        
        # Restore lifecycle information
        segment.creation_time = segment_data['creation_time']
        segment.last_access_time = segment_data['last_access_time']
        segment.lifecycle_state = segment_data['lifecycle_state']
        segment.cleanup_scheduled = segment_data['cleanup_scheduled']
        
        # Restore compatibility info
        if 'dimensional_compatibility' in segment_data:
            segment.dimensional_compatibility = segment_data['dimensional_compatibility']
        
        return segment

    @classmethod
    def _restore_nodes_to_brain_nexus(cls, segment: 'NexusSegment', segment_data: Dict[str, Any], 
                                    brain_nexus_ref: Any, demo: bool = False):
        """Restore all segment nodes to the BrainNexus instance."""
        segment_nodes_data = segment_data['segment_nodes_data']
        node_type_registry = segment_data['node_type_registry']
        
        if demo:
            print(f"🔄 Restoring {len(segment_nodes_data)} nodes to BrainNexus...")
        
        # Clear existing segment nodes
        segment.segment_nodes = {}
        segment.node_type_registry = {node_type: [] for node_type in node_type_registry.keys()}
        
        # Restore each node
        for node_id_str, node_data in segment_nodes_data.items():
            node_id = int(node_id_str)
            
            # Skip if node already exists in brain_nexus
            if node_id in brain_nexus_ref.node_registry:
                if demo:
                    print(f"   ⚠️  Node {node_id} already exists, skipping creation")
                segment.segment_nodes[node_id] = brain_nexus_ref.node_registry[node_id]
                continue
            
            # Create node in brain_nexus
            try:
                created_node_id = brain_nexus_ref.add_neural_node(
                    node_type=node_data['node_type'],
                    position=node_data['node_position'],
                    node_group=node_data.get('node_group', f'segment_{segment.segment_id}'),
                    segment_id=segment.segment_id,
                    **node_data.get('properties', {})
                )
                
                # Update node ID mapping if different
                if created_node_id != node_id:
                    if demo:
                        print(f"   ⚠️  Node ID mismatch: expected {node_id}, got {created_node_id}")
                    # Update segment tracking
                    node_id = created_node_id
                
                # Store in segment
                segment.segment_nodes[node_id] = brain_nexus_ref.node_registry[node_id]
                
                # Update node type registry
                node_type = node_data['node_type'].lower()
                if node_type.endswith('s'):
                    node_type = node_type[:-1]  # Remove plural
                if f"{node_type}s" in segment.node_type_registry:
                    segment.node_type_registry[f"{node_type}s"].append(node_id)
                
                # Restore node state
                node_obj = brain_nexus_ref.node_registry[node_id]
                if 'state' in node_data:
                    for attr, value in node_data['state'].items():
                        if hasattr(node_obj, attr):
                            setattr(node_obj, attr, value)
                
                if demo and len(segment_nodes_data) <= 10:  # Limit output for large segments
                    print(f"   ✓ Restored {node_data['node_type']} node {node_id}")
                    
            except Exception as e:
                if demo:
                    print(f"   ❌ Failed to restore node {node_id}: {str(e)}")
                continue
        
        if demo:
            total_restored = len(segment.segment_nodes)
            print(f"✅ Restored {total_restored}/{len(segment_nodes_data)} nodes")

    @classmethod
    def _restore_node_connections(cls, segment: 'NexusSegment', segment_data: Dict[str, Any], 
                                brain_nexus_ref: Any, demo: bool = False):
        """Restore all node connections in the BrainNexus."""
        segment_nodes_data = segment_data['segment_nodes_data']
        connection_count = 0
        
        if demo:
            print(f"🔗 Restoring node connections...")
        
        # Restore connections from node data
        for node_id_str, node_data in segment_nodes_data.items():
            node_id = int(node_id_str)
            
            if node_id not in brain_nexus_ref.node_registry:
                continue
            
            connections_data = node_data.get('connections', {})
            
            # Restore outgoing connections
            outgoing = connections_data.get('outgoing', {})
            for target_id_str, conn_info in outgoing.items():
                target_id = int(target_id_str)
                weight = conn_info.get('weight', 1.0)
                
                if target_id in brain_nexus_ref.node_registry:
                    try:
                        brain_nexus_ref.connect_nodes(node_id, target_id, weight=weight)
                        connection_count += 1
                    except Exception as e:
                        if demo:
                            print(f"   ⚠️  Failed to connect {node_id} → {target_id}: {str(e)}")
        
        # Restore external connections
        external_connections = segment_data.get('external_connections', {})
        for segment_id_str, connections in external_connections.items():
            # These will be restored when other segments are loaded
            pass
        
        if demo:
            print(f"✅ Restored {connection_count} connections")

    @classmethod
    def _deserialize_spatial_zones(cls, zones_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize spatial zones data."""
        return {
            zone_name: {
                'center': zone_data['center'],
                'radius': zone_data['radius'],
                'node_capacity': zone_data['node_capacity'],
                'current_occupancy': zone_data['current_occupancy'],
                'dimensional_bounds': zone_data['dimensional_bounds'],
                'zone_type': zone_data['zone_type'],
                'priority_dimensions': zone_data['priority_dimensions']
            }
            for zone_name, zone_data in zones_data.items()
        }

    @classmethod
    def _deserialize_attention_cache(cls, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize attention cache, reconstructing tensor data."""
        deserialized_cache = {}
        
        for key, attention_info in cache_data.items():
            data_type = attention_info.get('data_type', 'native')
            
            if data_type == 'pytorch_tensor':
                # Reconstruct PyTorch tensor
                try:
                    import torch
                    data_array = np.array(attention_info['data'])
                    tensor = torch.from_numpy(data_array).float()
                    deserialized_cache[key] = tensor
                except ImportError:
                    # Fallback to numpy if PyTorch not available
                    deserialized_cache[key] = np.array(attention_info['data'])
            elif data_type == 'numpy_array':
                deserialized_cache[key] = np.array(attention_info['data'])
            elif data_type == 'native':
                deserialized_cache[key] = attention_info['data']
            else:
                # For error or string_repr types
                deserialized_cache[key] = attention_info.get('data', '')
        
        return deserialized_cache

    @classmethod
    def _deserialize_embedding_transformations(cls, transformations_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize embedding transformations."""
        return cls._deserialize_attention_cache(transformations_data)

    @classmethod
    def _deserialize_positional_encodings(cls, encodings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize positional encodings."""
        return cls._deserialize_attention_cache(encodings_data)

    @classmethod
    def _deserialize_processing_results(cls, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize processing results."""
        return results_data

    @classmethod
    def _deserialize_communication_channels(cls, channels_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize communication channels."""
        return {int(segment_id): channel_state for segment_id, channel_state in channels_data.items()}

    @classmethod
    def _deserialize_shared_embeddings(cls, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize shared embeddings."""
        return cls._deserialize_attention_cache(embeddings_data)

    @classmethod
    def _deserialize_result_cache(cls, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize result cache."""
        return cache_data

    @classmethod
    def _deserialize_quality_metrics(cls, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize quality metrics with deque reconstruction."""
        metrics = {}
        
        for metric_name, metric_value in metrics_data.items():
            if isinstance(metric_value, list) and 'history' in metric_name:
                # Reconstruct deques for history metrics
                maxlen = 100 if 'history' in metric_name else None
                metrics[metric_name] = deque(metric_value, maxlen=maxlen)
            else:
                metrics[metric_name] = metric_value
        
        return metrics

    @classmethod
    def _print_load_summary(cls, segment: 'NexusSegment', metadata: Dict[str, Any]):
        """Print a summary of the load operation."""
        print(f"\n📊 Load Summary for Segment {segment.segment_id}:")
        print(f"   Dimensional Signature: {segment.dimensional_signature}")
        print(f"   Nodes Loaded: {len(segment.segment_nodes)}")
        print(f"   Spatial Zones: {len(segment.spatial_zones)}")
        print(f"   Active Judges: {len(segment.active_judges)}")
        print(f"   Activation State: {segment.activation_state}")
        print(f"   Pipeline State: {sum(1 for v in segment.pipeline_state.values() if v)}/{len(segment.pipeline_state)} complete")
        print(f"   Current Resources: {segment.current_resources['nodes_count']} nodes, {segment.current_resources['connections_count']} connections")
        print(f"   Quality Metrics Available: {len(segment.quality_metrics)}")
        print(f"   Cache Entries: {len(segment.attention_cache) + len(segment.result_cache)}")

    def initialize_brain(self, segments_dir: str = "segments"):
        """
        Initialize the brain by loading all segments from the segments directory.
        
        Args:
            segments_dir (str): Directory path containing segment files (default: "segments")
        """
        if not os.path.exists(segments_dir):
            if self.demo:
                print(f"⚠️  Segments directory '{segments_dir}' not found, creating empty directory")
            os.makedirs(segments_dir, exist_ok=True)
            return
        
        if self.demo:
            print(f"🧠 Initializing brain from segments directory: {segments_dir}")
        
        # Find all .pkl files in the segments directory
        segment_files = []
        for filename in os.listdir(segments_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(segments_dir, filename)
                segment_files.append(filepath)
        
        if not segment_files:
            if self.demo:
                print(f"   No segment files found in '{segments_dir}'")
            return
        
        if self.demo:
            print(f"   Found {len(segment_files)} segment file(s)")
        
        # Load each segment file
        loaded_segments = []
        for filepath in sorted(segment_files):  # Sort for consistent loading order
            try:
                if self.demo:
                    print(f"   Loading: {os.path.basename(filepath)}")
                
                # Load the segment using the class method
                segment = self.load_segment(
                    filepath=filepath,
                    brain_nexus_ref=self,
                    validate_integrity=True,
                    restore_connections=True,
                    demo=self.demo
                )
                
                loaded_segments.append(segment)
                
                # Initialize controller node at origin for this segment
                origin_position = [0.0] * self.dimensions
                controller_id = self.add_neural_node(
                    node_type='Controller',
                    position=origin_position,
                    node_group=f'segment_{segment.segment_id}_controller'
                )
                
                # Connect controller to all judge nodes in this segment
                judge_nodes = []
                for node_id, node in segment.segment_nodes.items():
                    if hasattr(node, 'node_type') and node.node_type.lower() == 'judge':
                        judge_nodes.append(node_id)
                
                # Create bidirectional connections between controller and judges
                for judge_id in judge_nodes:
                    try:
                        self.connect_nodes(controller_id, judge_id, weight=1.0, bidirectional=True)
                        if self.demo:
                            print(f"   ✓ Connected Controller {controller_id} ↔ Judge {judge_id}")
                    except Exception as e:
                        if self.demo:
                            print(f"   ⚠️  Failed to connect Controller {controller_id} to Judge {judge_id}: {str(e)}")
                
                if self.demo and judge_nodes:
                    print(f"   ✓ Created Controller node {controller_id} with {len(judge_nodes)} judge connections")
                elif self.demo and not judge_nodes:
                    print(f"   ⚠️  No judge nodes found in segment {segment.segment_id}")
                
                # Initialize handler node at origin for this segment
                handler_id = self.add_neural_node(
                    node_type='Handler',
                    position=origin_position,
                    node_group=f'segment_{segment.segment_id}_handler'
                )
                
                # Connect handler to all reviewer nodes in this segment
                reviewer_nodes = []
                for node_id, node in segment.segment_nodes.items():
                    if hasattr(node, 'node_type') and node.node_type.lower() == 'reviewer':
                        reviewer_nodes.append(node_id)
                
                # Create bidirectional connections between handler and reviewers
                for reviewer_id in reviewer_nodes:
                    try:
                        self.connect_nodes(handler_id, reviewer_id, weight=1.0, bidirectional=True)
                        if self.demo:
                            print(f"   ✓ Connected Handler {handler_id} ↔ Reviewer {reviewer_id}")
                    except Exception as e:
                        if self.demo:
                            print(f"   ⚠️  Failed to connect Handler {handler_id} to Reviewer {reviewer_id}: {str(e)}")
                
                if self.demo and reviewer_nodes:
                    print(f"   ✓ Created Handler node {handler_id} with {len(reviewer_nodes)} reviewer connections")
                elif self.demo and not reviewer_nodes:
                    print(f"   ⚠️  No reviewer nodes found in segment {segment.segment_id}")
                
            except Exception as e:
                if self.demo:
                    print(f"   ❌ Failed to load {os.path.basename(filepath)}: {str(e)}")
                continue
        
        if self.demo:
            print(f"✅ Brain initialization complete!")
            print(f"   Loaded {len(loaded_segments)} segment(s)")
            print(f"   Total nodes in brain: {len(self.neural_nodes)}")
            print(f"   Node registry size: {len(self.node_registry)}")
            
            if loaded_segments:
                total_segment_nodes = sum(len(seg.segment_nodes) for seg in loaded_segments)
                print(f"   Nodes from segments: {total_segment_nodes}")
        
        # Cleanup after initialization to optimize memory
        if loaded_segments:
            cleanup_stats = self.cleanup_memory('partial', cleanup_segments=True)
            if self.demo:
                print(f"   Post-initialization cleanup: {cleanup_stats['cleaned_items']}")
                print(f"   Memory freed: {cleanup_stats['memory_freed']['total_items']} items")
        
        return loaded_segments

    def process_multidimensional_pipeline(self, input_data: Any, task_type: str = 'general', 
                                        judge_activation_ratio: float = 0.5,
                                        computational_selection_ratio: float = 0.01) -> Dict[str, Any]:
        """
        Process input through the multidimensional brain pipeline using dynamic judge activation
        and hierarchical neural processing across hypercube segments.
        
        Args:
            input_data: Input data to process (embeddings, images, text, etc.)
            task_type: Type of AI task ('llm', 'vision', 'classification', 'general')
            judge_activation_ratio: Percentage of judges to activate (default 0.5 for top 50%)
            computational_selection_ratio: Percentage of computational nodes to use (default 0.01 for 1%)
            
        Returns:
            Dict containing final probabilities, processing traces, and metadata
        """
        if self.demo:
            print(f"🧠 Starting multidimensional pipeline for task: {task_type}")
            print(f"   Judge activation ratio: {judge_activation_ratio * 100}%")
            print(f"   Computational selection ratio: {computational_selection_ratio * 100}%")
        
        pipeline_results = {
            'final_probabilities': None,
            'processing_traces': {},
            'judge_activations': {},
            'segment_results': {},
            'metadata': {
                'task_type': task_type,
                'input_shape': getattr(input_data, 'shape', len(input_data) if hasattr(input_data, '__len__') else 'scalar'),
                'processing_time': 0,
                'active_judges_count': 0,
                'active_computational_nodes': 0
            }
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Controller determines relevant judges for the task
            if self.demo:
                print("📊 Step 1: Controller selecting relevant judges...")
            
            controller_nodes = self.get_nodes_by_type('Controller')
            if not controller_nodes:
                raise ValueError("No Controller nodes found. Initialize brain segments first.")
            
            controller_id = controller_nodes[0]  # Use first controller
            judge_relevance_scores = self._controller_evaluate_judges(controller_id, input_data, task_type)
            
            # Step 2: Dynamic judge activation (top 50% most relevant)
            active_judges = self._activate_top_judges(judge_relevance_scores, judge_activation_ratio)
            pipeline_results['judge_activations'] = active_judges
            pipeline_results['metadata']['active_judges_count'] = len(active_judges)
            
            if self.demo:
                print(f"   Activated {len(active_judges)} judges from {len(judge_relevance_scores)} total")
            
            # Step 3: Dynamic dimensional expansion based on active judges
            dimensional_layout = self._calculate_dynamic_dimensions(active_judges)
            
            if self.demo:
                print(f"📐 Step 2: Dynamic dimensional layout: {dimensional_layout['total_dimensions']} dimensions")
            
            # Step 4: Process through each active judge and their brain segments
            segment_results = {}
            for judge_id, relevance_score in active_judges.items():
                if self.demo:
                    print(f"⚖️  Step 3: Processing through Judge {judge_id} (relevance: {relevance_score:.3f})")
                
                # Generate attention masks, embeddings, and positional encodings
                judge_attention_data = self._judge_generate_attention_data(judge_id, input_data, task_type)
                
                # Find associated splitters for this judge
                splitter_nodes = self._find_judge_splitters(judge_id)
                
                segment_result = {
                    'judge_id': judge_id,
                    'attention_data': judge_attention_data,
                    'splitter_results': {},
                    'computational_traces': {},
                    'retainer_aggregations': {},
                    'reviewer_probabilities': None
                }
                
                # Step 5: Pass attention data to splitters
                for splitter_id in splitter_nodes:
                    if self.demo:
                        print(f"🔀     Processing through Splitter {splitter_id}")
                    
                    # Find closest 1% computational nodes
                    computational_nodes = self._find_closest_computational_nodes(
                        splitter_id, judge_attention_data, computational_selection_ratio
                    )
                    
                    splitter_result = {
                        'splitter_id': splitter_id,
                        'computational_nodes': computational_nodes,
                        'processing_results': {}
                    }
                    
                    # Step 6: Process through computational nodes
                    computational_results = {}
                    for comp_node_id in computational_nodes:
                        comp_result = self._process_computational_node(
                            comp_node_id, judge_attention_data, input_data
                        )
                        computational_results[comp_node_id] = comp_result
                        
                        if self.demo and len(computational_nodes) <= 5:
                            print(f"💻       Processed Computational node {comp_node_id}")
                    
                    splitter_result['processing_results'] = computational_results
                    segment_result['splitter_results'][splitter_id] = splitter_result
                    
                    # Step 7: Gather signals at retainers
                    retainer_nodes = self._find_splitter_retainers(splitter_id)
                    for retainer_id in retainer_nodes:
                        if self.demo:
                            print(f"🗃️      Aggregating at Retainer {retainer_id}")
                        
                        retainer_aggregation = self._aggregate_at_retainer(
                            retainer_id, computational_results
                        )
                        segment_result['retainer_aggregations'][retainer_id] = retainer_aggregation
                        
                        # Step 8: Pass to reviewer for final probabilities
                        reviewer_nodes = self._find_retainer_reviewers(retainer_id)
                        for reviewer_id in reviewer_nodes:
                            if self.demo:
                                print(f"📋        Reviewing at Reviewer {reviewer_id}")
                            
                            reviewer_probs = self._reviewer_final_probabilities(
                                reviewer_id, retainer_aggregation, task_type
                            )
                            segment_result['reviewer_probabilities'] = reviewer_probs
                
                segment_results[judge_id] = segment_result
                pipeline_results['segment_results'][judge_id] = segment_result
            
            # Step 9: Handler combines all reviewer probabilities using controller weights
            if self.demo:
                print("🤝 Step 4: Handler combining all segment results...")
            
            handler_nodes = self.get_nodes_by_type('Handler')
            if not handler_nodes:
                raise ValueError("No Handler nodes found. Initialize brain segments first.")
            
            handler_id = handler_nodes[0]  # Use first handler
            final_result = self._handler_combine_probabilities(
                handler_id, segment_results, active_judges, task_type
            )
            
            pipeline_results['final_probabilities'] = final_result
            pipeline_results['metadata']['processing_time'] = time.time() - start_time
            pipeline_results['metadata']['active_computational_nodes'] = sum(
                len(seg['splitter_results']) for seg in segment_results.values()
            )
            
            # Perform light cleanup after processing
            cleanup_stats = self.schedule_periodic_cleanup(cleanup_interval=100, cleanup_tier='light')
            if cleanup_stats and self.demo:
                print(f"   Post-processing cleanup: {cleanup_stats['cleaned_items']}")
            
            if self.demo:
                print(f"✅ Pipeline complete! Processing time: {pipeline_results['metadata']['processing_time']:.3f}s")
                print(f"   Final result shape: {getattr(final_result, 'shape', len(final_result) if hasattr(final_result, '__len__') else 'scalar')}")
            
            return pipeline_results
            
        except Exception as e:
            if self.demo:
                print(f"❌ Pipeline failed: {str(e)}")
            pipeline_results['error'] = str(e)
            pipeline_results['metadata']['processing_time'] = time.time() - start_time
            return pipeline_results

    def _controller_evaluate_judges(self, controller_id: int, input_data: Any, task_type: str) -> Dict[int, float]:
        """Controller evaluates all judges for task relevance."""
        judge_nodes = self.get_nodes_by_type('Judge')
        judge_scores = {}
        
        # Get controller's learned preferences for this task type
        controller_node = self.node_registry[controller_id]
        task_preferences = getattr(controller_node, 'task_preferences', {}).get(task_type, {})
        
        # Calculate input characteristics for scoring
        if hasattr(input_data, 'shape'):
            input_magnitude = float(np.linalg.norm(input_data))
            # Safe entropy calculation
            abs_data = np.abs(input_data)
            nonzero_mask = abs_data > 1e-8
            if np.any(nonzero_mask):
                log_data = np.zeros_like(abs_data)
                log_data[nonzero_mask] = np.log(abs_data[nonzero_mask])
                entropy_sum = np.sum(abs_data[nonzero_mask] * log_data[nonzero_mask])
                input_entropy = -float(entropy_sum)
            else:
                input_entropy = 0.0
            input_variance = float(np.var(input_data))
        elif hasattr(input_data, '__len__'):
            input_array = np.array(input_data, dtype=np.float64)
            input_magnitude = float(np.linalg.norm(input_array))
            # Safe entropy calculation
            abs_array = np.abs(input_array)
            nonzero_mask = abs_array > 1e-8
            if np.any(nonzero_mask):
                log_array = np.zeros_like(abs_array)
                log_array[nonzero_mask] = np.log(abs_array[nonzero_mask])
                entropy_sum = np.sum(abs_array[nonzero_mask] * log_array[nonzero_mask])
                input_entropy = -float(entropy_sum)
            else:
                input_entropy = 0.0
            input_variance = float(np.var(input_array))
        else:
            input_magnitude = abs(float(input_data))
            input_entropy = 0.5
            input_variance = 0.1
        
        for i, judge_id in enumerate(judge_nodes):
            # Calculate relevance based on:
            # 1. Task-specific base score
            if task_type == 'classification':
                base_score = 0.8 + 0.3 * (i % 3) / 2.0  # Vary by judge index
            elif task_type == 'llm':
                base_score = 0.7 + 0.4 * ((i + 1) % 4) / 3.0
            elif task_type == 'vision':
                base_score = 0.6 + 0.5 * ((i + 2) % 5) / 4.0
            else:
                base_score = 0.5 + 0.3 * (i % 2)
            
            # 2. Spatial relevance with input characteristics
            judge_node = self.node_registry[judge_id]
            spatial_score = self._calculate_spatial_relevance(judge_node.node_position, input_data)
            
            # 3. Judge specialization alignment
            position_characteristics = np.array(judge_node.node_position)
            position_norm = np.linalg.norm(position_characteristics)
            
            if task_type == 'classification':
                specialization_score = 1.0 - abs(position_norm - input_magnitude * 0.1) / 2.0
            elif task_type == 'llm':
                specialization_score = 1.0 - abs(position_characteristics[0] - input_entropy * 0.1) / 2.0
            elif task_type == 'vision':
                specialization_score = 1.0 - abs(position_characteristics[1] - input_variance) / 2.0
            else:
                specialization_score = spatial_score
            
            specialization_score = max(0.2, min(1.0, specialization_score))
            
            # 4. Connection strength with controller
            connection_weight = self.get_connection_weight(controller_id, judge_id) or 0.5
            connection_factor = 0.8 + 0.4 * connection_weight
            
            # 5. Historical performance (simulated)
            historical_preference = task_preferences.get(judge_id, 0.5)
            
            # Combine all factors with different weights
            final_score = (
                base_score * 0.3 +
                spatial_score * 0.25 +
                specialization_score * 0.25 +
                connection_factor * 0.1 +
                historical_preference * 0.1
            )
            
            judge_scores[judge_id] = max(0.1, min(1.0, final_score))
        
        return judge_scores

    def _activate_top_judges(self, judge_scores: Dict[int, float], activation_ratio: float) -> Dict[int, float]:
        """Activate top percentage of judges based on relevance scores."""
        sorted_judges = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
        num_to_activate = max(1, int(len(sorted_judges) * activation_ratio))
        return dict(sorted_judges[:num_to_activate])

    def _calculate_dynamic_dimensions(self, active_judges: Dict[int, float]) -> Dict[str, Any]:
        """Calculate dimensional layout based on active judges."""
        num_judges = len(active_judges)
        dimensions_needed = max(2, (num_judges + 1) // 2)  # 2 judges per dimension
        
        layout = {
            'total_dimensions': dimensions_needed,
            'judges_per_dimension': 2,
            'judge_positions': {},
            'hypercube_bounds': [-1000000, 1000000]  # Full hypercube range
        }
        
        # Assign judges to dimensional positions
        judge_ids = list(active_judges.keys())
        for i, judge_id in enumerate(judge_ids):
            dim_index = i // 2
            polarity = 1 if (i % 2) == 0 else -1
            
            layout['judge_positions'][judge_id] = {
                'dimension': dim_index,
                'polarity': polarity,
                'coordinate': polarity  # ±1 for judges
            }
        
        return layout

    def _judge_generate_attention_data(self, judge_id: int, input_data: Any, task_type: str) -> Dict[str, Any]:
        """Judge generates attention masks, embeddings, and positional encodings."""
        judge_node = self.node_registry[judge_id]
        
        # Generate attention masks based on judge specialization
        attention_masks = self._generate_attention_masks(judge_node, input_data, task_type)
        
        # Generate embeddings
        embeddings = self._generate_embeddings(judge_node, input_data)
        
        # Generate positional encodings
        positional_encodings = self._generate_positional_encodings(judge_node, input_data)
        
        return {
            'attention_masks': attention_masks,
            'embeddings': embeddings,
            'positional_encodings': positional_encodings,
            'judge_specialization': getattr(judge_node, 'specialization', task_type)
        }

    def _find_judge_splitters(self, judge_id: int) -> List[int]:
        """Find splitter nodes connected to a specific judge."""
        connections = self.get_node_connections(judge_id)
        splitter_nodes = []
        
        # Check all connected nodes for splitters
        all_connected = connections['outgoing'] + connections['bidirectional']
        for node_id in all_connected:
            if node_id in self.node_registry:
                node = self.node_registry[node_id]
                if hasattr(node, 'node_type') and node.node_type.lower() == 'splitter':
                    splitter_nodes.append(node_id)
        
        return splitter_nodes

    def _find_closest_computational_nodes(self, splitter_id: int, attention_data: Dict, selection_ratio: float) -> List[int]:
        """Find closest computational nodes based on spatial proximity and connections."""
        splitter_node = self.node_registry[splitter_id]
        splitter_position = splitter_node.node_position
        
        # Get all computational nodes
        computational_nodes = []
        for node_id, node in self.node_registry.items():
            if hasattr(node, 'node_type') and 'computational' in node.node_type.lower():
                distance = np.linalg.norm(np.array(node.node_position) - np.array(splitter_position))
                computational_nodes.append((node_id, distance))
        
        # Sort by distance and select top percentage
        computational_nodes.sort(key=lambda x: x[1])
        num_to_select = max(1, int(len(computational_nodes) * selection_ratio))
        
        return [node_id for node_id, _ in computational_nodes[:num_to_select]]

    def _process_computational_node(self, node_id: int, attention_data: Dict, input_data: Any) -> Dict[str, Any]:
        """Process data through a computational node."""
        comp_node = self.node_registry[node_id]
        
        # Apply attention masks to input
        attended_input = self._apply_attention_masks(input_data, attention_data['attention_masks'])
        
        # Process through node (this would be the actual neural computation)
        try:
            processed_output = comp_node.process(attended_input)
        except (NotImplementedError, AttributeError):
            # Fallback processing
            processed_output = self._default_computational_processing(attended_input, attention_data)
        
        return {
            'node_id': node_id,
            'input': attended_input,
            'output': processed_output,
            'processing_time': time.time()
        }

    def _find_splitter_retainers(self, splitter_id: int) -> List[int]:
        """Find retainer nodes connected to computational nodes that this splitter uses."""
        # Get all computational nodes connected to this splitter
        splitter_connections = self.get_node_connections(splitter_id)
        computational_nodes = []
        
        all_connected = splitter_connections['outgoing'] + splitter_connections['bidirectional']
        for node_id in all_connected:
            if node_id in self.node_registry:
                node = self.node_registry[node_id]
                if hasattr(node, 'node_type') and 'computational' in node.node_type.lower():
                    computational_nodes.append(node_id)
        
        # Now find retainer nodes connected to these computational nodes
        retainer_nodes = []
        for comp_node_id in computational_nodes:
            comp_connections = self.get_node_connections(comp_node_id)
            comp_all_connected = comp_connections['outgoing'] + comp_connections['bidirectional']
            
            for node_id in comp_all_connected:
                if node_id in self.node_registry:
                    node = self.node_registry[node_id]
                    if hasattr(node, 'node_type') and node.node_type.lower() == 'retainer':
                        if node_id not in retainer_nodes:
                            retainer_nodes.append(node_id)
        
        return retainer_nodes

    def _aggregate_at_retainer(self, retainer_id: int, computational_results: Dict) -> Dict[str, Any]:
        """Aggregate all computational results at a retainer node."""
        retainer_node = self.node_registry[retainer_id]
        
        # Collect all outputs
        outputs = [result['output'] for result in computational_results.values()]
        
        # Aggregate (mean, weighted sum, etc.)
        if outputs:
            if hasattr(outputs[0], 'shape'):  # Tensor-like data
                aggregated = np.mean(outputs, axis=0)
            else:  # Scalar or list data
                aggregated = np.mean([np.array(out) for out in outputs], axis=0)
        else:
            aggregated = np.array([0.0])
        
        return {
            'retainer_id': retainer_id,
            'aggregated_output': aggregated,
            'source_count': len(computational_results),
            'aggregation_method': 'mean'
        }

    def _find_retainer_reviewers(self, retainer_id: int) -> List[int]:
        """Find reviewer nodes connected to a specific retainer."""
        connections = self.get_node_connections(retainer_id)
        reviewer_nodes = []
        
        all_connected = connections['outgoing'] + connections['bidirectional']
        for node_id in all_connected:
            if node_id in self.node_registry:
                node = self.node_registry[node_id]
                if hasattr(node, 'node_type') and node.node_type.lower() == 'reviewer':
                    reviewer_nodes.append(node_id)
        
        return reviewer_nodes

    def _reviewer_final_probabilities(self, reviewer_id: int, retainer_data: Dict, task_type: str) -> np.ndarray:
        """Reviewer generates final probabilities from aggregated data."""
        reviewer_node = self.node_registry[reviewer_id]
        aggregated_output = retainer_data['aggregated_output']
        
        if self.demo:
            print(f"📋        Reviewer {reviewer_id} processing aggregated output shape: {aggregated_output.shape}")
            print(f"          Output range: [{np.min(aggregated_output):.3f}, {np.max(aggregated_output):.3f}]")
        
        # Ensure we have enough values for the output classes
        num_classes = self.output_config.get('num_classes', 10)
        
        if len(aggregated_output) >= num_classes:
            # Use first num_classes values
            logits = aggregated_output[:num_classes]
        else:
            # Pad or extend the output
            logits = np.zeros(num_classes)
            logits[:len(aggregated_output)] = aggregated_output
            # Fill remaining with slightly different values to avoid uniform distribution
            for i in range(len(aggregated_output), num_classes):
                logits[i] = np.mean(aggregated_output) * (0.8 + 0.4 * (i / num_classes))
        
        # Apply task-specific processing to create meaningful differentiation
        if task_type == 'classification':
            # Add task-specific bias based on reviewer position
            position_bias = np.array(reviewer_node.node_position[:3])  # Use first 3 dimensions
            position_magnitude = np.linalg.norm(position_bias)
            
            # Create position-dependent biases
            biases = np.sin(np.arange(num_classes) * position_magnitude * 0.1) * 0.5
            logits = logits + biases
            
        elif task_type == 'llm':
            # Language model specific - emphasize different vocabulary ranges
            reviewer_bias = hash(str(reviewer_id)) % 100 / 100.0
            vocab_bias = np.sin(np.arange(num_classes) * reviewer_bias * 2 * np.pi) * 0.3
            logits = logits + vocab_bias
            
        elif task_type == 'vision':
            # Vision specific - spatial biases
            spatial_pattern = np.cos(np.arange(num_classes) * 0.2) * 0.4
            logits = logits + spatial_pattern
        
        # Apply reviewer-specific transformation
        reviewer_factor = (hash(str(reviewer_id)) % 1000) / 1000.0
        transformation = 1.0 + reviewer_factor * 0.5  # Scale between 1.0 and 1.5
        logits = logits * transformation
        
        # Apply softmax to get valid probabilities
        exp_values = np.exp(logits - np.max(logits))
        probabilities = exp_values / np.sum(exp_values)
        
        if self.demo:
            max_idx = np.argmax(probabilities)
            print(f"          Final probabilities - Top class {max_idx}: {probabilities[max_idx]:.3f}")
            print(f"          Probability entropy: {-np.sum(probabilities * np.log(probabilities + 1e-8)):.3f}")
        
        return probabilities

    def _handler_combine_probabilities(self, handler_id: int, segment_results: Dict, 
                                     active_judges: Dict[int, float], task_type: str) -> np.ndarray:
        """Handler combines all reviewer probabilities using judge relevance weights."""
        handler_node = self.node_registry[handler_id]
        
        all_probabilities = []
        weights = []
        
        if self.demo:
            print(f"🤝 Handler {handler_id} combining results from {len(active_judges)} judges")
        
        for judge_id, judge_relevance in active_judges.items():
            if judge_id in segment_results:
                segment = segment_results[judge_id]
                if segment['reviewer_probabilities'] is not None:
                    all_probabilities.append(segment['reviewer_probabilities'])
                    weights.append(judge_relevance)
                    
                    if self.demo:
                        top_class = np.argmax(segment['reviewer_probabilities'])
                        top_prob = segment['reviewer_probabilities'][top_class]
                        print(f"    Judge {judge_id} (weight: {judge_relevance:.3f}): top class {top_class} = {top_prob:.3f}")
        
        if not all_probabilities:
            # Fallback: return uniform distribution
            num_classes = self.output_config.get('num_classes', 10)
            if self.demo:
                print(f"    ⚠️  No probabilities found, returning uniform distribution")
            return np.ones(num_classes) / num_classes
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        if self.demo:
            print(f"    Normalized judge weights: {[f'{w:.3f}' for w in weights]}")
        
        # Ensure all probability arrays have the same shape
        target_length = len(all_probabilities[0])
        aligned_probs = []
        
        for i, probs in enumerate(all_probabilities):
            if len(probs) != target_length:
                # Resize or pad as needed
                if len(probs) < target_length:
                    padded = np.zeros(target_length)
                    padded[:len(probs)] = probs
                    aligned_probs.append(padded)
                else:
                    aligned_probs.append(probs[:target_length])
            else:
                aligned_probs.append(probs)
        
        # Weighted combination with task-specific processing
        final_probabilities = np.zeros(target_length)
        for i, (probs, weight) in enumerate(zip(aligned_probs, weights)):
            final_probabilities += probs * weight
        
        # Apply handler-specific adjustment based on task type
        if task_type == 'classification':
            # Enhance discrimination for classification
            final_probabilities = final_probabilities ** 1.2
        elif task_type == 'llm':
            # Smooth probabilities for language modeling
            final_probabilities = final_probabilities ** 0.9
        elif task_type == 'vision':
            # Sharpen probabilities for vision tasks
            final_probabilities = final_probabilities ** 1.1
        
        # Ensure probabilities sum to 1
        if np.sum(final_probabilities) > 0:
            final_probabilities = final_probabilities / np.sum(final_probabilities)
        else:
            # Fallback to uniform if all zeros
            final_probabilities = np.ones(target_length) / target_length
        
        if self.demo:
            max_idx = np.argmax(final_probabilities)
            max_prob = final_probabilities[max_idx]
            entropy = -np.sum(final_probabilities * np.log(final_probabilities + 1e-8))
            print(f"    Final result: top class {max_idx} = {max_prob:.3f}, entropy = {entropy:.3f}")
        
        return final_probabilities

    # Helper methods for attention and processing
    def _calculate_spatial_relevance(self, node_position: List[float], input_data: Any) -> float:
        """Calculate spatial relevance of a node to input data."""
        # Calculate relevance based on input data characteristics
        if hasattr(input_data, 'shape'):
            input_magnitude = np.linalg.norm(input_data)
        elif hasattr(input_data, '__len__'):
            input_magnitude = np.linalg.norm(np.array(input_data))
        else:
            input_magnitude = abs(float(input_data))
        
        position_norm = np.linalg.norm(node_position)
        
        # Create meaningful relationship between input characteristics and spatial position
        relevance = np.exp(-abs(input_magnitude * 0.1 - position_norm) / 2.0)
        return float(max(0.1, min(1.0, relevance)))

    def _generate_attention_masks(self, judge_node: Any, input_data: Any, task_type: str) -> np.ndarray:
        """Generate attention masks based on judge specialization and task type."""
        
        # First, determine the input format and get appropriate dimensions
        if isinstance(input_data, str):
            # For text input, create attention based on tokenized length
            tokens = self._tokenize_input(input_data)
            input_dim = len(tokens)
        elif isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            # Multiple text inputs - use average length
            all_tokens = [self._tokenize_input(text) for text in input_data]
            input_dim = int(np.mean([len(tokens) for tokens in all_tokens]))
        elif isinstance(input_data, list) and all(isinstance(x, (int, float)) for x in input_data):
            # Already tokenized or numerical
            input_dim = len(input_data)
        else:
            # Get input dimensions for other data types
            try:
                # Try numpy array/tensor-like objects first
                shape = getattr(input_data, 'shape', None)
                if shape is not None and hasattr(shape, '__len__') and len(shape) > 0:
                    input_dim = shape[-1]
                elif hasattr(input_data, '__len__') and not isinstance(input_data, str):
                    input_dim = len(input_data)
                else:
                    input_dim = 1
            except (AttributeError, TypeError):
                input_dim = 1
        
        # Limit input_dim to reasonable size
        input_dim = min(input_dim, 512)  # Max sequence length
        
        # Base attention pattern depends on task type and judge position
        position_hash = hash(tuple(judge_node.node_position)) % 1000
        np.random.seed(position_hash)  # Deterministic but different per judge
        
        if task_type == 'classification':
            # Focus on discriminative features
            mask = np.random.beta(2, 1, input_dim)  # Skewed towards higher attention
        elif task_type == 'llm':
            # Sequential attention pattern with recency bias
            sequence_weights = np.linspace(0.5, 1.5, input_dim)
            noise = np.random.uniform(0.8, 1.2, input_dim)
            mask = sequence_weights * noise
            # Add causal mask for language modeling
            causal_decay = np.exp(-0.1 * np.arange(input_dim))
            mask = mask * causal_decay
        elif task_type == 'vision':
            # Spatial attention patterns
            if input_dim >= 64:
                # Create spatial-like attention for image data
                spatial_dim = int(np.sqrt(input_dim))
                if spatial_dim * spatial_dim == input_dim:
                    spatial_mask = np.random.uniform(0.3, 1.0, (spatial_dim, spatial_dim))
                    # Add center bias for vision tasks
                    center = spatial_dim // 2
                    y, x = np.ogrid[:spatial_dim, :spatial_dim]
                    center_dist = np.sqrt((x - center)**2 + (y - center)**2)
                    center_bias = np.exp(-center_dist / (spatial_dim * 0.3))
                    spatial_mask = spatial_mask * (0.5 + 0.5 * center_bias)
                    mask = spatial_mask.flatten()
                else:
                    # Non-square dimensions - use gamma distribution
                    mask = np.random.gamma(2, 0.5, input_dim)
            else:
                # Small input - use gamma distribution
                mask = np.random.gamma(2, 0.5, input_dim)
        else:
            # General task - uniform with slight variation
            mask = np.random.uniform(0.5, 1.0, input_dim)
        
        # Apply task-specific attention patterns
        if task_type == 'llm':
            # Language models need to attend to recent tokens more
            recent_boost = np.exp(-0.05 * np.arange(input_dim)[::-1])
            mask = mask * (0.7 + 0.3 * recent_boost)
        elif task_type == 'classification':
            # Classification needs to focus on discriminative features
            # Boost attention on certain positions based on judge specialization
            judge_specialty = hash(str(judge_node.node_id)) % input_dim
            specialty_boost = np.exp(-0.1 * np.abs(np.arange(input_dim) - judge_specialty))
            mask = mask * (0.8 + 0.2 * specialty_boost)
        
        # Normalize and add judge-specific bias
        judge_id_hash = hash(str(judge_node.node_id)) % 100
        judge_bias = 0.8 + 0.4 * (judge_id_hash / 100.0)
        mask = mask * judge_bias
        
        # Reset random seed
        np.random.seed(None)
        
        # Ensure mask sums to 1 (probability distribution)
        return mask / np.sum(mask)

    def _generate_embeddings(self, judge_node: Any, input_data: Any) -> np.ndarray:
        """Generate embeddings from input data with proper tokenization and embedding lookup."""
        
        # First, handle tokenization if input is text
        if isinstance(input_data, str):
            # Tokenize text input
            tokens = self._tokenize_input(input_data)
            # Convert tokens to embeddings
            embeddings = self._tokens_to_embeddings(tokens, judge_node)
        elif isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            # List of strings - tokenize each and aggregate
            all_embeddings = []
            for text in input_data:
                tokens = self._tokenize_input(text)
                text_embeddings = self._tokens_to_embeddings(tokens, judge_node)
                all_embeddings.append(text_embeddings)
            embeddings = np.mean(all_embeddings, axis=0)
        elif isinstance(input_data, list) and all(isinstance(x, (int, float)) for x in input_data):
            # List of numbers - already tokenized, convert to embeddings
            embeddings = self._token_ids_to_embeddings(input_data, judge_node)
        else:
            # Handle other data types (arrays, tensors, etc.)
            try:
                # Try to get flatten method safely
                flatten_method = getattr(input_data, 'flatten', None)
                if flatten_method is not None:
                    raw_data = flatten_method()
                elif hasattr(input_data, '__iter__') and not isinstance(input_data, str):
                    raw_data = np.array(list(input_data)).flatten()
                else:
                    # For scalar or non-iterable data
                    if isinstance(input_data, (int, float)):
                        raw_data = np.array([float(input_data)])
                    else:
                        raw_data = np.array([0.0])
            except (ValueError, TypeError, AttributeError):
                # Fallback for any conversion errors
                raw_data = np.array([0.0])
            
            # Convert raw data to embedding space
            embeddings = self._raw_data_to_embeddings(raw_data, judge_node)
        
        # Add judge-specific positional transformation
        judge_transform = self._get_judge_embedding_transform(judge_node, len(embeddings))
        embeddings = embeddings + judge_transform
        
        return embeddings
    
    def _tokenize_input(self, text: str) -> List[int]:
        """Tokenize text input into token IDs using Mistral v3 Tekken tokenizer."""
        # Initialize Mistral tokenizer if not already done
        if not hasattr(self, 'mistral_tokenizer'):
            self._initialize_mistral_tokenizer()
        
        if MISTRAL_AVAILABLE and hasattr(self, 'mistral_tokenizer') and self.mistral_tokenizer is not None:
            try:
                # Use Mistral v3 Tekken tokenizer
                tokens = self.mistral_tokenizer.encode_chat_completion(
                    ChatCompletionRequest(
                        messages=[UserMessage(content=text)],
                        model="nemostral"
                    )
                ).tokens
                
                # Convert to list of integers
                token_ids = [int(token) for token in tokens]
                
                if self.demo:
                    print(f"🔤 Mistral tokenized '{text[:50]}...' -> {len(token_ids)} tokens")
                
                return token_ids
                
            except Exception as e:
                if self.demo:
                    print(f"⚠️  Mistral tokenization failed: {e}, falling back to simple tokenizer")
                return self._simple_tokenize_input(text)
        else:
            # Fallback to simple tokenization
            return self._simple_tokenize_input(text)
    
    def _initialize_mistral_tokenizer(self):
        """Initialize the Mistral v3 Tekken tokenizer."""
        if MISTRAL_AVAILABLE:
            try:
                # Primary: Load Mistral v3 with Tekken enabled
                self.mistral_tokenizer = MistralTokenizer.v3(is_tekken=True)
                
                # Alternative: Load from specific model if needed
                # self.mistral_tokenizer = MistralTokenizer.from_model("nemostral")
                
                # Get vocabulary size from Mistral tokenizer
                if hasattr(self.mistral_tokenizer, 'n_words'):
                    self.vocab_size = self.mistral_tokenizer.n_words
                elif hasattr(self.mistral_tokenizer, 'vocab_size'):
                    self.vocab_size = self.mistral_tokenizer.vocab_size
                else:
                    # Default Mistral v3 vocabulary size
                    self.vocab_size = 131072  # Mistral v3 typical vocab size
                
                self.embedding_dim = 4096  # Mistral v3 embedding dimension
                
                # Initialize embedding matrix for Mistral vocabulary
                if not hasattr(self, 'embedding_matrix'):
                    self._initialize_embedding_matrix()
                
                if self.demo:
                    print(f"✅ Mistral v3 Tekken tokenizer initialized")
                    print(f"   Vocabulary size: {self.vocab_size}")
                    print(f"   Embedding dimension: {self.embedding_dim}")
                
                return True
                
            except Exception as e:
                if self.demo:
                    print(f"❌ Failed to initialize Mistral tokenizer: {e}")
                self.mistral_tokenizer = None
                self._initialize_simple_vocabulary()
                return False
        else:
            if self.demo:
                print(f"⚠️  Mistral not available, using simple tokenization")
            self.mistral_tokenizer = None
            self._initialize_simple_vocabulary()
            return False
    
    def _simple_tokenize_input(self, text: str) -> List[int]:
        """Simple word-based tokenization (fallback when Mistral is not available)."""
        # Simple word-based tokenization (fallback)
        words = text.lower().split()
        
        # Create or use vocabulary
        if not hasattr(self, 'vocabulary'):
            self._initialize_simple_vocabulary()
        
        token_ids = []
        for word in words:
            if word in self.vocabulary:
                token_ids.append(self.vocabulary[word])
            else:
                # Handle unknown words
                if '[UNK]' in self.vocabulary:
                    token_ids.append(self.vocabulary['[UNK]'])
                else:
                    # Add new word to vocabulary (for training)
                    new_id = len(self.vocabulary)
                    self.vocabulary[word] = new_id
                    token_ids.append(new_id)
        
        # Add special tokens
        if '[BOS]' in self.vocabulary:
            token_ids.insert(0, self.vocabulary['[BOS]'])
        if '[EOS]' in self.vocabulary:
            token_ids.append(self.vocabulary['[EOS]'])
            
        return token_ids
    
    def _initialize_vocabulary(self):
        """Initialize vocabulary - delegates to Mistral or simple tokenization."""
        self._initialize_mistral_tokenizer()
    
    def _initialize_simple_vocabulary(self):
        """Initialize basic vocabulary with special tokens (fallback method)."""
        self.vocabulary = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[MASK]': 4,
            # Add common words
            'the': 5, 'a': 6, 'an': 7, 'and': 8, 'or': 9, 'but': 10,
            'is': 11, 'are': 12, 'was': 13, 'were': 14, 'be': 15,
            'have': 16, 'has': 17, 'had': 18, 'do': 19, 'does': 20, 'did': 21,
            'will': 22, 'would': 23, 'could': 24, 'should': 25, 'may': 26, 'might': 27,
            'this': 28, 'that': 29, 'these': 30, 'those': 31,
            'i': 32, 'you': 33, 'he': 34, 'she': 35, 'it': 36, 'we': 37, 'they': 38,
            'me': 39, 'him': 40, 'her': 41, 'us': 42, 'them': 43,
            'my': 44, 'your': 45, 'his': 46, 'her': 47, 'its': 48, 'our': 49, 'their': 50,
            'in': 51, 'on': 52, 'at': 53, 'by': 54, 'for': 55, 'with': 56, 'to': 57, 'from': 58,
            'up': 59, 'down': 60, 'out': 61, 'off': 62, 'over': 63, 'under': 64,
            'what': 65, 'who': 66, 'when': 67, 'where': 68, 'why': 69, 'how': 70,
            'yes': 71, 'no': 72, 'not': 73, 'very': 74, 'much': 75, 'many': 76, 'more': 77, 'most': 78,
            'good': 79, 'bad': 80, 'big': 81, 'small': 82, 'long': 83, 'short': 84,
            'new': 85, 'old': 86, 'first': 87, 'last': 88, 'next': 89, 'other': 90,
            'know': 91, 'think': 92, 'see': 93, 'get': 94, 'give': 95, 'take': 96, 'come': 97, 'go': 98, 'make': 99
        }
        self.vocab_size = len(self.vocabulary)
        self.embedding_dim = 768  # Standard embedding dimension (fallback)
        
        # Initialize embedding matrix
        if not hasattr(self, 'embedding_matrix'):
            self._initialize_embedding_matrix()
        
        if self.demo:
            print(f"📝 Simple vocabulary initialized with {self.vocab_size} tokens")
    
    def _initialize_embedding_matrix(self):
        """Initialize the embedding matrix with random weights."""
        # In production, use pre-trained embeddings (Word2Vec, GloVe, etc.)
        np.random.seed(42)  # For reproducibility
        self.embedding_matrix = np.random.normal(0, 0.1, (self.vocab_size, self.embedding_dim))
        
        # Special token embeddings
        self.embedding_matrix[0] = np.zeros(self.embedding_dim)  # [PAD]
        self.embedding_matrix[1] = np.random.normal(0, 0.05, self.embedding_dim)  # [UNK]
        
        np.random.seed(None)  # Reset seed
    
    def _tokens_to_embeddings(self, token_ids: List[int], judge_node: Any) -> np.ndarray:
        """Convert token IDs to embeddings using embedding lookup."""
        if not hasattr(self, 'embedding_matrix'):
            self._initialize_embedding_matrix()
        
        # Pad or truncate sequence to standard length
        max_seq_len = 512  # Standard sequence length
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        else:
            # Pad with [PAD] tokens
            token_ids = token_ids + [0] * (max_seq_len - len(token_ids))
        
        # Look up embeddings
        embeddings = []
        for token_id in token_ids:
            if token_id < len(self.embedding_matrix):
                embeddings.append(self.embedding_matrix[token_id])
            else:
                # Use [UNK] embedding for out-of-vocabulary tokens
                embeddings.append(self.embedding_matrix[1])
        
        embeddings = np.array(embeddings)  # Shape: (seq_len, embedding_dim)
        
        # Apply positional encodings
        embeddings = self._add_positional_encodings(embeddings)
        
        # Pool to single representation (for this judge)
        # Different judges can use different pooling strategies
        judge_id_hash = hash(str(judge_node.node_id)) % 4
        if judge_id_hash == 0:
            # Mean pooling
            pooled = np.mean(embeddings, axis=0)
        elif judge_id_hash == 1:
            # Max pooling
            pooled = np.max(embeddings, axis=0)
        elif judge_id_hash == 2:
            # First token (CLS-style)
            pooled = embeddings[0]
        else:
            # Last non-padded token
            # Find last non-zero embedding
            non_zero_mask = np.any(embeddings != 0, axis=1)
            if np.any(non_zero_mask):
                last_idx = np.where(non_zero_mask)[0][-1]
                pooled = embeddings[last_idx]
            else:
                pooled = embeddings[0]
        
        return pooled
    
    def _token_ids_to_embeddings(self, token_ids: List[int], judge_node: Any) -> np.ndarray:
        """Convert pre-tokenized IDs to embeddings."""
        return self._tokens_to_embeddings(token_ids, judge_node)
    
    def _raw_data_to_embeddings(self, raw_data: np.ndarray, judge_node: Any) -> np.ndarray:
        """Convert raw numerical data to embedding space."""
        target_dim = self.embedding_dim if hasattr(self, 'embedding_dim') else 768
        
        if len(raw_data) == target_dim:
            embeddings = raw_data.copy()
        elif len(raw_data) > target_dim:
            # Dimensionality reduction via chunking and averaging
            chunk_size = len(raw_data) // target_dim
            embeddings = np.array([
                np.mean(raw_data[i*chunk_size:(i+1)*chunk_size]) 
                for i in range(target_dim)
            ])
        else:
            # Dimensionality expansion via interpolation
            embeddings = np.interp(
                np.linspace(0, len(raw_data)-1, target_dim),
                np.arange(len(raw_data)),
                raw_data
            )
        
        # Apply nonlinear transformation to make it more embedding-like
        embeddings = np.tanh(embeddings * 0.5)
        
        return embeddings
    
    def _add_positional_encodings(self, embeddings: np.ndarray) -> np.ndarray:
        """Add sinusoidal positional encodings to embeddings."""
        seq_len, d_model = embeddings.shape
        
        # Create positional encoding matrix
        pe = np.zeros((seq_len, d_model))
        position = np.arange(seq_len).reshape(-1, 1)
        
        # Calculate div_term for sinusoidal encoding
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cosine to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        
        return embeddings + pe
    
    def _get_judge_embedding_transform(self, judge_node: Any, embedding_dim: int) -> np.ndarray:
        """Get judge-specific embedding transformation."""
        # Create deterministic but unique transformation per judge
        judge_seed = hash(str(judge_node.node_id)) % 10000
        np.random.seed(judge_seed)
        
        # Small positional transformation based on judge's spatial position
        position = np.array(judge_node.node_position)
        position_norm = np.linalg.norm(position)
        
        # Scale transformation by judge position
        transform_scale = 0.01 * (1 + position_norm * 0.1)
        transform = np.random.normal(0, transform_scale, embedding_dim)
        
        np.random.seed(None)  # Reset seed
        return transform
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text using Mistral tokenizer or fallback."""
        if MISTRAL_AVAILABLE and hasattr(self, 'mistral_tokenizer') and self.mistral_tokenizer is not None:
            try:
                # Use Mistral decoder
                decoded_text = self.mistral_tokenizer.decode(token_ids)
                return decoded_text
            except Exception as e:
                if self.demo:
                    print(f"⚠️  Mistral decoding failed: {e}, using fallback")
                return self._simple_decode_tokens(token_ids)
        else:
            return self._simple_decode_tokens(token_ids)
    
    def _simple_decode_tokens(self, token_ids: List[int]) -> str:
        """Simple decoding for fallback tokenization."""
        if not hasattr(self, 'vocabulary'):
            return "[UNKNOWN]"
        
        # Create reverse vocabulary
        if not hasattr(self, 'reverse_vocabulary'):
            self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}
        
        words = []
        for token_id in token_ids:
            if token_id in self.reverse_vocabulary:
                word = self.reverse_vocabulary[token_id]
                # Skip special tokens in output
                if word not in ['[BOS]', '[EOS]', '[PAD]']:
                    words.append(word)
            else:
                words.append('[UNK]')
        
        return ' '.join(words)
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get information about the current tokenizer."""
        # Check if Mistral tokenizer is active
        using_mistral = (MISTRAL_AVAILABLE and hasattr(self, 'mistral_tokenizer') and self.mistral_tokenizer is not None)
        
        info = {
            'tokenizer_type': 'mistral_v3_tekken' if using_mistral else 'simple',
            'vocab_size': getattr(self, 'vocab_size', 99 if not using_mistral else 131072),
            'embedding_dim': getattr(self, 'embedding_dim', 768 if not using_mistral else 4096),
            'mistral_available': MISTRAL_AVAILABLE
        }
        
        if using_mistral:
            # Add Mistral-specific information
            info['mistral_version'] = 'v3'
            info['tekken_enabled'] = True
            info['model'] = 'nemostral'
        
        return info

    def _generate_positional_encodings(self, judge_node: Any, input_data: Any) -> np.ndarray:
        """Generate positional encodings."""
        position = judge_node.node_position
        # Create sinusoidal positional encodings
        encodings = []
        for i, pos in enumerate(position):
            encodings.extend([np.sin(pos), np.cos(pos)])
        return np.array(encodings)

    def _apply_attention_masks(self, input_data: Any, attention_masks: np.ndarray) -> Any:
        """Apply attention masks to input data."""
        if hasattr(input_data, 'shape'):
            # Tensor-like data
            if input_data.shape == attention_masks.shape:
                return input_data * attention_masks
            else:
                # Broadcast if possible
                try:
                    return input_data * attention_masks
                except:
                    return input_data
        else:
            return input_data

    def _default_computational_processing(self, input_data: Any, attention_data: Dict) -> np.ndarray:
        """Default computational processing when node doesn't have custom processing."""
        if hasattr(input_data, '__iter__') and not isinstance(input_data, str):
            input_array = np.array(input_data)
            
            # Apply different transformations based on input characteristics
            input_mean = np.mean(input_array)
            input_std = np.std(input_array)
            
            # Create more meaningful transformation
            # 1. Nonlinear transformation
            transformed = np.tanh(input_array * 0.5 + input_mean)
            
            # 2. Add attention-weighted noise
            if 'attention_masks' in attention_data:
                attention_weights = attention_data['attention_masks']
                if hasattr(attention_weights, '__len__') and len(attention_weights) == len(transformed):
                    noise_scale = attention_weights * input_std * 0.1
                    transformed += np.random.normal(0, 1, len(transformed)) * noise_scale
            
            # 3. Apply output scaling based on task context
            output_scale = 1.0 + input_std * 0.2
            return transformed * output_scale
        else:
            # Scalar input
            scalar_val = float(input_data)
            # Apply nonlinear transformation with some randomness
            transformed = np.tanh(scalar_val * 0.5) * (1.0 + np.random.normal(0, 0.1))
            return np.array([transformed])

    def _generate_token_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Generate token probabilities for LLM tasks."""
        vocab_size = self.output_config.get('num_classes', 10000)
        if len(logits) != vocab_size:
            # Resize logits to vocabulary size
            if len(logits) < vocab_size:
                padded_logits = np.zeros(vocab_size)
                padded_logits[:len(logits)] = logits
                logits = padded_logits
            else:
                logits = logits[:vocab_size]
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def _generate_vision_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Generate probabilities for computer vision tasks."""
        num_classes = self.output_config.get('num_classes', 1000)  # ImageNet default
        
        # Simple mapping from features to class probabilities
        if len(features) != num_classes:
            # Resize features
            if len(features) < num_classes:
                padded = np.zeros(num_classes)
                padded[:len(features)] = features
                features = padded
            else:
                features = features[:num_classes]
        
        # Apply softmax
        exp_features = np.exp(features - np.max(features))
        return exp_features / np.sum(exp_features)


def main():
    """
    Main function to test the multidimensional brain pipeline with comprehensive debugging.
    """
    print("🚀 Starting BrainNexus Multidimensional Pipeline Test")
    print("=" * 60)
    
    try:
        # Initialize brain with demo mode for detailed output
        print("📋 Step 1: Initializing BrainNexus...")
        brain = BrainNexus(dimensions=4, demo=True, mode='default')
        print(f"   ✓ BrainNexus initialized with {brain.dimensions} dimensions")
        print(f"   ✓ Output config: {brain.output_config['type']} with {brain.output_config['num_classes']} classes")
        
        # Create test nodes since segment loading has issues
        print("\n� Step 2: Creating test neural network...")
        
        # Create judges at different positions
        judge_positions = [
            [1.0, 0.0, 0.0, 0.0],    # Judge 1 - dim 0 positive
            [-1.0, 0.0, 0.0, 0.0],   # Judge 2 - dim 0 negative  
            [0.0, 1.0, 0.0, 0.0],    # Judge 3 - dim 1 positive
            [0.0, -1.0, 0.0, 0.0],   # Judge 4 - dim 1 negative
            [0.0, 0.0, 1.0, 0.0],    # Judge 5 - dim 2 positive
            [0.0, 0.0, -1.0, 0.0],   # Judge 6 - dim 2 negative
        ]
        
        judge_ids = []
        for i, pos in enumerate(judge_positions):
            judge_id = brain.add_neural_node(
                node_type='Judge',
                position=pos,
                node_group=f'test_judges',
                specialization=np.random.choice(['general', 'llm', 'vision', 'classification'])
            )
            judge_ids.append(judge_id)
            print(f"   ✓ Created Judge {judge_id} at position {pos}")
        
        # Create splitters connected to judges
        splitter_ids = []
        for i in range(3):
            splitter_id = brain.add_neural_node(
                node_type='Splitter',
                position=[0.0, 0.0, 0.0, float(i)],
                node_group='test_splitters'
            )
            splitter_ids.append(splitter_id)
            print(f"   ✓ Created Splitter {splitter_id}")
        
        # Create computational nodes
        computational_ids = []
        for i in range(20):  # Create 20 computational nodes
            pos = [np.random.uniform(-2, 2) for _ in range(4)]
            comp_id = brain.add_neural_node(
                node_type='Computational',
                position=pos,
                node_group='test_computational'
            )
            computational_ids.append(comp_id)
        
        print(f"   ✓ Created {len(computational_ids)} Computational nodes")
        
        # Create retainers
        retainer_ids = []
        for i in range(2):
            retainer_id = brain.add_neural_node(
                node_type='Retainer',
                position=[0.0, 0.0, float(i), 0.0],
                node_group='test_retainers'
            )
            retainer_ids.append(retainer_id)
            print(f"   ✓ Created Retainer {retainer_id}")
        
        # Create reviewers
        reviewer_ids = []
        for i in range(2):
            reviewer_id = brain.add_neural_node(
                node_type='Reviewer',
                position=[float(i), 0.0, 0.0, 0.0],
                node_group='test_reviewers'
            )
            reviewer_ids.append(reviewer_id)
            print(f"   ✓ Created Reviewer {reviewer_id}")
        
        # Create controller at origin
        controller_id = brain.add_neural_node(
            node_type='Controller',
            position=[0.0, 0.0, 0.0, 0.0],
            node_group='test_controller'
        )
        print(f"   ✓ Created Controller {controller_id} at origin")
        
        # Create handler at origin
        handler_id = brain.add_neural_node(
            node_type='Handler',
            position=[0.0, 0.0, 0.0, 0.0],
            node_group='test_handler'
        )
        print(f"   ✓ Created Handler {handler_id} at origin")
        
        # Connect controller to judges
        for judge_id in judge_ids:
            brain.connect_nodes(controller_id, judge_id, weight=1.0, bidirectional=True)
        print(f"   ✓ Connected Controller to {len(judge_ids)} judges")
        
        # Connect handler to reviewers
        for reviewer_id in reviewer_ids:
            brain.connect_nodes(handler_id, reviewer_id, weight=1.0, bidirectional=True)
        print(f"   ✓ Connected Handler to {len(reviewer_ids)} reviewers")
        
        # Connect judges to splitters
        for judge_id in judge_ids[:3]:  # First 3 judges
            for splitter_id in splitter_ids:
                brain.connect_nodes(judge_id, splitter_id, weight=0.8)
        
        # Connect splitters to computational nodes
        for splitter_id in splitter_ids:
            for comp_id in computational_ids[:10]:  # Connect to first 10 computational nodes
                brain.connect_nodes(splitter_id, comp_id, weight=0.6)
        
        # Connect computational nodes to retainers
        for comp_id in computational_ids:
            for retainer_id in retainer_ids:
                brain.connect_nodes(comp_id, retainer_id, weight=0.4)
        
        # Connect retainers to reviewers
        for retainer_id in retainer_ids:
            for reviewer_id in reviewer_ids:
                brain.connect_nodes(retainer_id, reviewer_id, weight=0.7)
        
        print(f"   ✓ Created full processing pipeline with connections")
        print(f"   ✓ Total nodes created: {len(brain.neural_nodes)}")
        
        # Test different input types and tasks
        print("\n🔬 Step 3: Testing pipeline with different scenarios...")
        
        test_scenarios = [
            {
                'name': 'Classification Task - Random Vector',
                'input_data': np.random.randn(64),
                'task_type': 'classification',
                'judge_ratio': 0.5,
                'comp_ratio': 0.1
            },
            {
                'name': 'LLM Task - Text Embedding',
                'input_data': np.random.randn(128),
                'task_type': 'llm',
                'judge_ratio': 0.6,
                'comp_ratio': 0.05
            },
            {
                'name': 'Vision Task - Image Features',
                'input_data': np.random.randn(256),
                'task_type': 'vision',
                'judge_ratio': 0.4,
                'comp_ratio': 0.2
            }
        ]
        
        successful_tests = 0
        total_tests = len(test_scenarios)
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🧪 Test {i}/{total_tests}: {scenario['name']}")
            print(f"   Input shape: {scenario['input_data'].shape}")
            print(f"   Task type: {scenario['task_type']}")
            print(f"   Judge activation ratio: {scenario['judge_ratio'] * 100}%")
            print(f"   Computational selection ratio: {scenario['comp_ratio'] * 100}%")
            
            try:
                # Run the pipeline
                results = brain.process_multidimensional_pipeline(
                    input_data=scenario['input_data'],
                    task_type=scenario['task_type'],
                    judge_activation_ratio=scenario['judge_ratio'],
                    computational_selection_ratio=scenario['comp_ratio']
                )
                
                # Analyze results
                if 'error' in results:
                    print(f"   ❌ Pipeline error: {results['error']}")
                    print(f"   Processing time: {results['metadata']['processing_time']:.3f}s")
                    traceback.print_exc()
                else:
                    print(f"   ✅ Pipeline completed successfully!")
                    print(f"   Processing time: {results['metadata']['processing_time']:.3f}s")
                    print(f"   Active judges: {results['metadata']['active_judges_count']}")
                    print(f"   Active computational nodes: {results['metadata']['active_computational_nodes']}")
                    
                    # Analyze final probabilities
                    final_probs = results['final_probabilities']
                    if final_probs is not None:
                        prob_shape = final_probs.shape if hasattr(final_probs, 'shape') else len(final_probs) if hasattr(final_probs, '__len__') else 'scalar'
                        print(f"   Final probabilities shape: {prob_shape}")
                        if hasattr(final_probs, '__len__') and len(final_probs) > 0:
                            print(f"   Top prediction: {np.argmax(final_probs)} (confidence: {np.max(final_probs):.3f})")
                            print(f"   Probability sum: {np.sum(final_probs):.3f} (should be ~1.0)")
                    else:
                        print(f"   ⚠️  No final probabilities generated")
                    
                    successful_tests += 1
                    
            except Exception as e:
                print(f"   ❌ Test failed with exception: {str(e)}")
                print(f"   Exception type: {type(e).__name__}")
                traceback.print_exc()
                continue
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Tests passed: {successful_tests}/{total_tests}")
        print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! The multidimensional brain pipeline is working correctly.")
        elif successful_tests > 0:
            print("⚠️  Some tests passed. Check the errors above for issues to resolve.")
        else:
            print("❌ All tests failed. The pipeline needs debugging.")
        
        # Brain state summary
        print(f"\n🧠 Final Brain State:")
        print(f"   Total nodes: {len(brain.neural_nodes)}")
        print(f"   Node registry size: {len(brain.node_registry)}")
        print(f"   Spatial index: {'Active' if brain.spatial_index is not None else 'Inactive'}")
        print(f"   Reuse candidates: {len(brain.reuse_candidates)}")
        
        # Node type breakdown
        node_type_counts = {}
        for node in brain.neural_nodes:
            node_type = getattr(node, 'node_type', 'Unknown')
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        print(f"   Node types:")
        for node_type, count in sorted(node_type_counts.items()):
            print(f"     - {node_type}: {count} nodes")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR in main function:")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        print(f"   Location: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
