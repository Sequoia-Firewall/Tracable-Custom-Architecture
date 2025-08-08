import json
import random
import math
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from collections import deque, defaultdict
import time
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
import pickle
import os
from datetime import datetime
import sys
class NexusSegment:
    """
    A multidimensional brain segment that operates within specific dimensional polarities.
    Each segment contains judges, splitters, computational nodes, retainers, and reviewers
    working together to process specific aspects of the input space.
    """
    
    def __init__(self, 
                 segment_id: int,
                 dimensional_assignment: Dict[int, int],  # {dimension: polarity (-1 or +1)}
                 brain_nexus_ref: Any,  # Reference to parent BrainNexus
                 hypercube_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (-1000000.0, 1000000.0),
                 segment_config: Optional[Dict[str, Any]] = None,
                 demo: bool = False):
        """
        Initialize a NexusSegment for multidimensional brain processing.
        
        Args:
            segment_id (int): Unique identifier for this segment
            dimensional_assignment (Dict[int, int]): Maps dimension index to polarity (-1 or +1)
                                                   e.g. {0: 1, 1: -1} = +x, -y quadrant
            brain_nexus_ref (Any): Reference to the parent BrainNexus instance
            hypercube_bounds (Tuple[float, float]): Min/max bounds for hypercube positioning
            segment_config (Optional[Dict]): Configuration overrides for this segment
            demo (bool): Enable debug output and demonstrations
        """
        
        # Core identity and hierarchy
        self.segment_id = segment_id
        self.dimensional_assignment = dimensional_assignment.copy()
        self.brain_nexus = brain_nexus_ref
        self.demo = demo
        self.hypercube_bounds = hypercube_bounds
        
        # Normalize hypercube bounds format for internal use
        self._normalized_bounds = self._normalize_hypercube_bounds()
        
        # Configuration management - MUST be done first
        self.config = self._initialize_segment_config(segment_config)
        
        # Dimensional properties - dynamically calculated
        self.dimensions = len(dimensional_assignment)
        self.max_dimension_index = max(dimensional_assignment.keys()) if dimensional_assignment else 0
        self.effective_brain_dimensions = self._calculate_effective_dimensions()
        self.dimensional_signature = self._calculate_dimensional_signature()
        self.segment_center = self._calculate_segment_center()
        self.segment_radius = self._calculate_segment_radius()
        self.dimensional_volume = self._calculate_dimensional_volume()
        self.dimensional_density = self._calculate_dimensional_density()
        
        # Node management and tracking
        self.segment_nodes = {}  # node_id -> node_object mapping
        self.node_type_registry = {
            'judges': [],
            'splitters': [], 
            'computational': [],
            'retainers': [],
            'reviewers': []
        }
        
        # Spatial and connectivity management
        self.spatial_zones = self._initialize_spatial_zones()
        self.connection_matrix = {}  # Tracks internal connections
        self.external_connections = defaultdict(list)  # Connections to other segments
        
        # Dynamic loading and activation
        self.activation_state = 'dormant'  # 'dormant', 'loading', 'active', 'processing'
        self.relevance_score = 0.0
        self.activation_threshold = 0.5
        self.last_activation_time = 0.0
        self.activation_history = deque(maxlen=100)
        
        # Judge-specific management
        self.active_judges = set()  # Currently active judge node_ids
        self.judge_relevance_scores = {}  # judge_id -> relevance_score
        self.max_active_judges = self.config.get('max_active_judges', 10)
        self.judge_activation_ratio = 0.5  # Top 50% most relevant judges
        
        # Attention and embedding management
        self.attention_cache = {}  # Cached attention masks from judges
        self.embedding_transformations = {}  # Judge-specific embedding transforms
        self.positional_encodings = {}  # Position-aware encodings for this segment
        
        # Processing pipeline state
        self.pipeline_state = {
            'judges_processed': False,
            'splitters_activated': False,
            'computational_complete': False,
            'retainers_gathered': False,
            'reviewers_finalized': False
        }
        self.processing_results = {}  # Intermediate results storage
        
        # Performance and optimization
        self.computation_budget = self.config.get('computation_budget', 1000)
        self.remaining_budget = self.computation_budget
        self.efficiency_metrics = {
            'nodes_utilized': 0,
            'connections_active': 0,
            'reuse_efficiency': 0.0,
            'spatial_efficiency': 0.0,
            'processing_time': 0.0
        }
        
        # Inter-segment communication
        self.communication_channels = {}  # segment_id -> communication_state
        self.shared_embeddings = {}  # Cross-segment embedding sharing
        self.synchronization_points = []  # Points requiring cross-segment sync
        
        # Memory and caching
        self.result_cache = {}  # Cache for repeated computations
        self.pattern_memory = deque(maxlen=1000)  # Remember successful patterns
        self.failure_patterns = deque(maxlen=100)  # Learn from failures
        
        # Adaptive learning
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.adaptation_history = []
        self.success_patterns = defaultdict(int)
        self.dimensional_preferences = {}  # Learn which dimensions work best
        
        # Resource management
        # Extract resource limits properly - check both nested and direct formats
        resource_limits = self.config.get('resource_limits', {})
        self.resource_limits = {
            'max_nodes': resource_limits.get('max_nodes', self.config.get('max_nodes', 1000)),
            'max_connections': resource_limits.get('max_connections', self.config.get('max_connections', 10000)),
            'memory_limit': resource_limits.get('memory_limit', self.config.get('memory_limit', 1024 * 1024 * 100)),  # 100MB
            'processing_timeout': resource_limits.get('processing_timeout', self.config.get('processing_timeout', 30.0))  # 30 seconds
        }
        self.current_resources = {
            'nodes_count': 0,
            'connections_count': 0,
            'memory_usage': 0,
            'processing_start_time': None
        }
        
        # Quality assurance and monitoring
        self.quality_metrics = {
            'accuracy_history': deque(maxlen=100),
            'precision_history': deque(maxlen=100),
            'recall_history': deque(maxlen=100),
            'f1_history': deque(maxlen=100),
            'error_rate': 0.0,
            'anomaly_detection': False
        }
        
        # Thread safety and concurrency (for future multi-threading)
        self.processing_lock = None  # Will be initialized if threading is needed
        self.state_lock = None
        self.concurrent_processing = False
        
        # Segment lifecycle management
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
        self.lifecycle_state = 'initialized'
        self.cleanup_scheduled = False
        
        # Integration with BrainNexus
        self._register_with_brain_nexus()
        self._sync_dimensional_compatibility()
        self._initialize_base_nodes()
        
        if self.demo:
            self._print_initialization_summary()
    
    def _calculate_effective_dimensions(self) -> int:
        """
        Calculate the effective dimensionality this segment operates in.
        Considers both brain dimensions and segment requirements.
        
        Returns:
            int: Effective number of dimensions for this segment
        """
        brain_dims = getattr(self.brain_nexus, 'dimensions', 4)
        segment_max_dim = self.max_dimension_index + 1 if self.dimensional_assignment else 0
        
        # Use the maximum needed to accommodate both brain and segment needs
        return max(brain_dims, segment_max_dim)
    
    def _calculate_dimensional_volume(self) -> float:
        """
        Calculate the hypervolume of this segment in N-dimensional space.
        
        Returns:
            float: Estimated volume occupied by this segment
        """
        if self.dimensions == 0:
            return 1.0  # Point volume
        
        # Volume of hypersphere in N dimensions: V = (π^(n/2) * r^n) / Γ(n/2 + 1)
        # Approximation for computational efficiency
        radius = self.segment_radius
        n = self.dimensions
        
        if n <= 3:
            # Exact formulas for common cases
            if n == 1:
                return 2 * radius  # Line segment
            elif n == 2:
                return math.pi * radius * radius  # Circle
            elif n == 3:
                return (4/3) * math.pi * radius * radius * radius  # Sphere
        else:
            # Stirling's approximation for higher dimensions
            # V ≈ (2πe/n)^(n/2) * r^n
            volume = math.pow(2 * math.pi * math.e / n, n/2) * math.pow(radius, n)
            return volume
    
    def _calculate_dimensional_density(self) -> float:
        """
        Calculate expected node density in this dimensional space.
        
        Returns:
            float: Nodes per unit hypervolume
        """
        total_expected_nodes = (
            self.config['judges_per_segment'] +
            self.config['judges_per_segment'] * self.config['splitters_per_judge'] +
            self.config['computational_nodes_base'] +
            self.config['retainers_per_group'] * 10 +  # Estimate
            self.config['reviewers_per_retainer'] * 10   # Estimate
        )
        
        volume = self.dimensional_volume
        if volume <= 0:
            return 1.0
        
        return total_expected_nodes / volume
    
    def _sync_dimensional_compatibility(self):
        """
        Ensure dimensional compatibility with the parent BrainNexus.
        Updates brain dimensions if segment requires higher dimensionality.
        """
        current_brain_dims = getattr(self.brain_nexus, 'dimensions', 4)
        required_dims = self.effective_brain_dimensions
        
        if required_dims > current_brain_dims:
            if self.demo:
                print(f"🔄 Expanding BrainNexus dimensions: {current_brain_dims}D → {required_dims}D")
            
            # Update brain dimensions
            self.brain_nexus.dimensions = required_dims
            
            # Extend existing node positions to new dimensionality
            if hasattr(self.brain_nexus, 'node_positions') and self.brain_nexus.node_positions:
                extended_positions = []
                for pos in self.brain_nexus.node_positions:
                    if len(pos) < required_dims:
                        # Pad with zeros for new dimensions
                        extended_pos = list(pos) + [0.0] * (required_dims - len(pos))
                        extended_positions.append(extended_pos)
                    else:
                        extended_positions.append(pos)
                self.brain_nexus.node_positions = extended_positions
                
                # Rebuild spatial index with new dimensionality
                if hasattr(self.brain_nexus, '_rebuild_spatial_index'):
                    self.brain_nexus._rebuild_spatial_index()
            
            # Update existing neural nodes positions
            if hasattr(self.brain_nexus, 'neural_nodes'):
                for node in self.brain_nexus.neural_nodes:
                    if hasattr(node, 'node_position'):
                        current_pos = node.node_position
                        if len(current_pos) < required_dims:
                            node.node_position = list(current_pos) + [0.0] * (required_dims - len(current_pos))
        
        # Store compatibility info
        self.dimensional_compatibility = {
            'brain_original_dims': current_brain_dims,
            'segment_required_dims': required_dims,
            'compatibility_achieved': True,
            'dimension_extension_applied': required_dims > current_brain_dims
        }
    
    def _calculate_dimensional_signature(self) -> str:
        """
        Create a unique signature for this segment's dimensional assignment.
        Supports unlimited dimensions dynamically.
        
        Returns:
            str: Signature like "2D_+x-y" or "10D_+d0-d1+d2-d3+d4-d5+d6-d7+d8-d9"
        """
        if not self.dimensional_assignment:
            return "0D_origin"
        
        signature_parts = [f"{len(self.dimensional_assignment)}D"]
        
        # Sort by dimension index for consistency
        sorted_dims = sorted(self.dimensional_assignment.items())
        
        # Extended dimension labels for readability up to 26D, then numeric
        dim_labels = ['x', 'y', 'z', 'w', 'v', 'u', 't', 's', 'r', 'q', 'p', 'o', 'n', 'm', 
                     'l', 'k', 'j', 'i', 'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
        
        dim_signature = ""
        for dim_idx, polarity in sorted_dims:
            if dim_idx < len(dim_labels):
                label = dim_labels[dim_idx]
            else:
                label = f"d{dim_idx}"
            
            polarity_symbol = "+" if polarity >= 0 else "-"
            dim_signature += f"{polarity_symbol}{label}"
        
        signature_parts.append(dim_signature)
        return "_".join(signature_parts)
    
    def _calculate_segment_center(self) -> List[float]:
        """
        Calculate the center position of this segment in the hypercube.
        Dynamically supports any number of dimensions (2D, 3D, 10D, etc.).
        
        Returns:
            List[float]: Center coordinates in the brain's dimensional space
        """
        # Get brain dimensions dynamically, with fallback
        brain_dims = getattr(self.brain_nexus, 'dimensions', 
                           max(4, max(self.dimensional_assignment.keys()) + 1 if self.dimensional_assignment else 4))
        
        # Initialize center at origin for all dimensions
        center = [0.0] * brain_dims
        
        if not self.dimensional_assignment:
            return center
        
        # Position based on dimensional assignment and hypercube bounds
        bound_range = self._normalized_bounds[1] - self._normalized_bounds[0]
        midpoint = (self._normalized_bounds[0] + self._normalized_bounds[1]) / 2
        
        # Calculate positioning for each assigned dimension
        for dim_idx, polarity in self.dimensional_assignment.items():
            if dim_idx < brain_dims:
                # Place at 75% of the way to the bound in the assigned polarity
                if polarity >= 0:
                    center[dim_idx] = midpoint + (bound_range * 0.25)
                else:
                    center[dim_idx] = midpoint - (bound_range * 0.25)
            elif dim_idx >= brain_dims:
                # Extend brain dimensions if segment requires higher dimensionality
                additional_dims = dim_idx + 1 - brain_dims
                center.extend([0.0] * additional_dims)
                brain_dims = len(center)
                
                # Now apply the positioning
                if polarity >= 0:
                    center[dim_idx] = midpoint + (bound_range * 0.25)
                else:
                    center[dim_idx] = midpoint - (bound_range * 0.25)
        
        return center
    
    def _calculate_segment_radius(self) -> float:
        """
        Calculate the effective radius of this segment for spatial operations.
        Dynamically scales with dimensionality for optimal coverage.
        
        Returns:
            float: Radius for spatial queries and node placement
        """
        if not self.dimensional_assignment:
            return 1000.0  # Small default radius
        
        # Base radius on hypercube bounds and number of dimensions
        bound_range = self._normalized_bounds[1] - self._normalized_bounds[0]
        base_radius = bound_range * 0.15  # 15% of total range as base
        
        # Dynamic scaling for dimensionality - more dimensions need adjusted spacing
        num_dims = len(self.dimensional_assignment)
        
        if num_dims <= 3:
            # 2D/3D: Standard scaling
            dimensional_factor = math.sqrt(num_dims)
        elif num_dims <= 6:
            # 4D-6D: Moderate scaling adjustment
            dimensional_factor = math.pow(num_dims, 0.6)
        else:
            # 7D+: Logarithmic scaling to prevent too-small radii in high dimensions
            dimensional_factor = math.log(num_dims) + math.sqrt(num_dims)
        
        # Apply curse of dimensionality compensation
        if num_dims > 10:
            # For very high dimensions, use more conservative scaling
            curse_compensation = 1.0 + (0.1 * math.log10(num_dims))
            dimensional_factor /= curse_compensation
        
        adjusted_radius = base_radius / max(1.0, dimensional_factor)
        
        # Ensure minimum viable radius regardless of dimensionality
        min_radius = bound_range * 0.001  # 0.1% of total range minimum
        max_radius = bound_range * 0.4    # 40% of total range maximum
        
        return max(min_radius, min(max_radius, adjusted_radius))
    
    def _normalize_hypercube_bounds(self) -> Tuple[float, float]:
        """
        Normalize hypercube bounds to a single tuple format for internal calculations.
        
        Returns:
            Tuple[float, float]: (min_bound, max_bound) for all dimensions
        """
        if isinstance(self.hypercube_bounds, list) and len(self.hypercube_bounds) > 0:
            # Use the first dimension's bounds as the base (assuming all dimensions have same bounds)
            if isinstance(self.hypercube_bounds[0], tuple) and len(self.hypercube_bounds[0]) == 2:
                return self.hypercube_bounds[0]
        elif isinstance(self.hypercube_bounds, tuple) and len(self.hypercube_bounds) == 2:
            # Already in the correct format
            return self.hypercube_bounds
        
        # Fallback to default bounds (from origin to max value)
        return (0.0, 1000000.0)
    
    def _get_memory_optimized_embed_dim(self) -> int:
        """
        Get memory-optimized embedding dimensions based on segment configuration and demo mode.
        
        Returns:
            int: Optimized embedding dimension
        """
        base_embed_dim = getattr(self.brain_nexus, 'embedding_dim', 768)
        
        # Reduce dimensions significantly for demo mode
        if self.demo:
            return min(128, base_embed_dim // 6)  # Use 128 or 1/6th of base, whichever is smaller
        
        # For production, use smaller dimensions for certain segment types
        if hasattr(self, 'config') and self.config.get('resource_limits', {}).get('max_nodes', 1000) <= 200:
            return min(256, base_embed_dim // 3)  # Small segments get smaller embeddings
        
        return base_embed_dim
    
    def _initialize_segment_config(self, segment_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Initialize configuration with defaults and user overrides.
        
        Args:
            segment_config: User-provided configuration overrides
            
        Returns:
            Dict[str, Any]: Complete configuration dictionary
        """
        # Default configuration
        default_config = {
            # Node distribution
            'judges_per_segment': 20,
            'splitters_per_judge': 2,
            'computational_nodes_base': 100,
            'retainers_per_group': 5,
            'reviewers_per_retainer': 1,
            
            # Activation and selection
            'max_active_judges': 10,
            'judge_selection_ratio': 0.5,  # Top 50%
            'computational_selection_ratio': 0.01,  # Top 1%
            
            # Performance parameters
            'computation_budget': 1000,
            'learning_rate': 0.01,
            'activation_threshold': 0.5,
            
            # Spatial parameters
            'spatial_clustering': True,
            'cluster_radius_factor': 0.1,
            'inter_cluster_distance': 2.0,
            
            # Resource limits
            'max_nodes': 1000,
            'max_connections': 10000,
            'memory_limit': 1024 * 1024 * 100,  # 100MB
            'processing_timeout': 30.0,
            
            # Quality and reliability
            'error_tolerance': 0.05,
            'quality_threshold': 0.8,
            'anomaly_detection_enabled': True,
            
            # Integration settings
            'brain_nexus_integration': True,
            'cross_segment_communication': True,
            'shared_embedding_space': True
        }
        
        # Merge with user configuration
        if segment_config:
            config = {**default_config, **segment_config}
        else:
            config = default_config
        
        # Validate and adjust configuration
        config = self._validate_segment_config(config)
        
        return config
    
    def _validate_segment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and adjust configuration values to ensure consistency.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Dict[str, Any]: Validated and adjusted configuration
        """
        # Ensure ratios are between 0 and 1
        ratio_keys = ['judge_selection_ratio', 'computational_selection_ratio']
        for key in ratio_keys:
            if key in config:
                config[key] = max(0.01, min(1.0, config[key]))
        
        # Ensure positive values for counts
        count_keys = ['judges_per_segment', 'computational_nodes_base', 'max_nodes']
        for key in count_keys:
            if key in config and config[key] <= 0:
                config[key] = 1
        
        # Adjust max_active_judges based on judges_per_segment
        if config['max_active_judges'] > config['judges_per_segment']:
            config['max_active_judges'] = config['judges_per_segment']
        
        # Ensure computational budget is reasonable
        if config['computation_budget'] <= 0:
            config['computation_budget'] = 100
        
        return config
    
    def _initialize_spatial_zones(self) -> Dict[str, Any]:
        """
        Initialize spatial zones within this segment for organized node placement.
        Dynamically adapts to any number of dimensions.
        
        Returns:
            Dict[str, Any]: Spatial zone definitions and boundaries
        """
        # Calculate zone positions relative to segment center
        center = self.segment_center.copy()
        radius = self.segment_radius
        
        # Dynamic zone offset calculation based on dimensionality
        if self.dimensions <= 2:
            # 2D: Simple quadrant-based offsets
            zone_offsets = {
                'judge_zone': [0.0] * len(center),
                'splitter_zone': [radius * 0.1] + [0.0] * (len(center) - 1),
                'computational_zone': [0.0] * len(center),
                'retainer_zone': [-radius * 0.1] + [0.0] * (len(center) - 1),
                'reviewer_zone': [-radius * 0.2] + [0.0] * (len(center) - 1)
            }
        elif self.dimensions <= 4:
            # 3D-4D: Layered positioning
            zone_offsets = {
                'judge_zone': [0.0] * len(center),
                'splitter_zone': [radius * 0.1, radius * 0.05] + [0.0] * (len(center) - 2),
                'computational_zone': [0.0] * len(center),
                'retainer_zone': [-radius * 0.1, -radius * 0.05] + [0.0] * (len(center) - 2),
                'reviewer_zone': [-radius * 0.2, -radius * 0.1] + [0.0] * (len(center) - 2)
            }
        else:
            # 5D+: Distributed positioning across multiple dimensions
            zone_offsets = {}
            zone_names = ['judge_zone', 'splitter_zone', 'computational_zone', 'retainer_zone', 'reviewer_zone']
            
            for i, zone_name in enumerate(zone_names):
                offset = [0.0] * len(center)
                
                # Distribute zones across available dimensions
                for dim_idx in range(min(self.dimensions, len(center))):
                    # Create spiral-like distribution in high-dimensional space
                    angle_factor = (2 * math.pi * i) / len(zone_names)
                    dimension_factor = (dim_idx + 1) / self.dimensions
                    
                    offset_magnitude = radius * (0.1 + 0.1 * i) * dimension_factor
                    offset[dim_idx] = offset_magnitude * math.cos(angle_factor + dim_idx)
                
                zone_offsets[zone_name] = offset
        
        # Create zone definitions
        zones = {}
        zone_configs = {
            'judge_zone': {
                'radius_factor': 0.8,
                'capacity_base': self.config['judges_per_segment']
            },
            'splitter_zone': {
                'radius_factor': 0.6,
                'capacity_base': self.config['judges_per_segment'] * self.config['splitters_per_judge']
            },
            'computational_zone': {
                'radius_factor': 1.2,
                'capacity_base': self.config['computational_nodes_base']
            },
            'retainer_zone': {
                'radius_factor': 0.4,
                'capacity_base': self.config['retainers_per_group'] * 10
            },
            'reviewer_zone': {
                'radius_factor': 0.2,
                'capacity_base': self.config['reviewers_per_retainer'] * 10
            }
        }
        
        for zone_name, zone_config in zone_configs.items():
            zone_center = [c + o for c, o in zip(center, zone_offsets[zone_name])]
            zone_radius = radius * zone_config['radius_factor']
            
            # Adjust capacity for higher dimensions (more space available)
            capacity_multiplier = 1.0
            if self.dimensions > 4:
                # Gradually increase capacity for higher dimensions
                capacity_multiplier = 1.0 + (0.2 * math.log(self.dimensions - 3))
            
            zones[zone_name] = {
                'center': zone_center,
                'radius': zone_radius,
                'node_capacity': int(zone_config['capacity_base'] * capacity_multiplier),
                'current_occupancy': 0,
                'dimensional_bounds': self._calculate_zone_bounds(zone_center, zone_radius),
                'zone_type': zone_name.replace('_zone', ''),
                'priority_dimensions': list(range(min(3, self.dimensions)))  # Primary dimensions for this zone
            }
        
        return zones
    
    def _calculate_zone_bounds(self, center: List[float], radius: float) -> Dict[str, List[float]]:
        """
        Calculate N-dimensional bounding box for a zone.
        
        Args:
            center: Zone center coordinates
            radius: Zone radius
            
        Returns:
            Dict with 'min_bounds' and 'max_bounds' lists
        """
        min_bounds = [c - radius for c in center]
        max_bounds = [c + radius for c in center]
        
        return {
            'min_bounds': min_bounds,
            'max_bounds': max_bounds
        }
    
    def _register_with_brain_nexus(self):
        """Register this segment with the parent BrainNexus."""
        if hasattr(self.brain_nexus, 'segments'):
            if not hasattr(self.brain_nexus, 'segments'):
                self.brain_nexus.segments = {}
            self.brain_nexus.segments[self.segment_id] = self
        else:
            # Create segments registry if it doesn't exist
            self.brain_nexus.segments = {self.segment_id: self}
        
        # Register dimensional assignment
        if hasattr(self.brain_nexus, 'dimensional_segments'):
            if not hasattr(self.brain_nexus, 'dimensional_segments'):
                self.brain_nexus.dimensional_segments = {}
            self.brain_nexus.dimensional_segments[self.dimensional_signature] = self.segment_id
        else:
            self.brain_nexus.dimensional_segments = {self.dimensional_signature: self.segment_id}
    
    def _initialize_base_nodes(self):
        """
        Initialize the basic node structure for this segment.
        Creates judges, splitters, reviewers, and retainers at specific dimensional positions.
        """
        # This will be implemented in future methods
        # For now, just set up the framework
        self.pipeline_state = {key: False for key in self.pipeline_state.keys()}
        self.processing_results = {}
        self.efficiency_metrics = {key: 0.0 if isinstance(val, (int, float)) else val 
                                 for key, val in self.efficiency_metrics.items()}
        
        # Create the core nodes for this segment
        self._create_segment_nodes()
    
    def _create_segment_nodes(self):
        """
        Create and position the core nodes for this segment based on dimensional assignment.
        Places nodes at specific dimensional coordinates according to the architecture.
        Dynamically allocates 85% of available space to computational nodes.
        """
        if not self.dimensional_assignment:
            if self.demo:
                print(f"⚠️  No dimensional assignment for segment {self.segment_id}, skipping node creation")
            return
        
        # Determine scale based on demo/default/full mode
        scale_mode = self._determine_scale_mode()
        furthest_distance = self._get_furthest_distance(scale_mode)
        
        # Calculate dynamic node allocation (85% computational nodes)
        max_nodes = self.resource_limits['max_nodes']
        node_allocation = self._calculate_dynamic_node_allocation(max_nodes)
        
        if self.demo:
            print(f"\n🏗️  Creating nodes for segment {self.segment_id} ({scale_mode} mode)")
            print(f"   Furthest distance: {furthest_distance}")
            print(f"   Dimensional assignment: {self.dimensional_assignment}")
            print(f"   Node allocation: {node_allocation}")
        
        # Create judge nodes at ±1 positions
        judge_nodes = self._create_judge_nodes(node_allocation['judges'])
        
        # Create splitter nodes at ±2 positions, connected to judges
        splitter_nodes = self._create_splitter_nodes(judge_nodes, node_allocation['splitters'])
        
        # Create computational nodes at intermediate positions (85% of total space)
        computational_nodes = self._create_computational_nodes_dynamic(
            judge_nodes, splitter_nodes, node_allocation['computational']
        )
        
        # Create reviewer and retainer nodes at furthest positions
        reviewer_nodes, retainer_nodes = self._create_reviewer_retainer_nodes_dynamic(
            furthest_distance, node_allocation['reviewers'], node_allocation['retainers']
        )
        
        # Update node registries
        self._update_node_registries(judge_nodes, splitter_nodes, computational_nodes, reviewer_nodes, retainer_nodes)
        
        if self.demo:
            self._print_node_creation_summary(judge_nodes, splitter_nodes, computational_nodes, reviewer_nodes, retainer_nodes)
    
    def _determine_scale_mode(self) -> str:
        """
        Determine the scale mode based on demo flag and configuration.
        
        Returns:
            str: 'demo', 'default', or 'full'
        """
        if self.demo:
            return 'demo'
        elif hasattr(self.brain_nexus, 'mode') and self.brain_nexus.mode == 'full':
            return 'full'
        else:
            return 'default'
    
    def _get_furthest_distance(self, scale_mode: str) -> float:
        """
        Get the furthest distance value based on scale mode.
        
        Args:
            scale_mode: 'demo', 'default', or 'full'
            
        Returns:
            float: Distance value for furthest nodes
        """
        distance_map = {
            'demo': 100.0,
            'default': 1000.0,
            'full': 1000000.0
        }
        return distance_map.get(scale_mode, 1000.0)
    
    def _calculate_dynamic_node_allocation(self, max_nodes: int) -> Dict[str, int]:
        """
        Calculate dynamic node allocation with 85% computational nodes.
        
        Args:
            max_nodes: Maximum number of nodes allowed in this segment
            
        Returns:
            Dict[str, int]: Allocation per node type
        """
        # Reserve minimum required nodes for core functionality
        dimensions = len(self.dimensional_assignment)
        min_judges = max(1, dimensions)  # At least 1 judge per dimension
        min_splitters = max(1, dimensions)  # At least 1 splitter per dimension
        min_reviewers = max(1, int(dimensions * 0.5))  # Fewer reviewers
        min_retainers = max(1, int(dimensions * 0.3))  # Even fewer retainers
        
        # Calculate minimum required nodes
        min_required = min_judges + min_splitters + min_reviewers + min_retainers
        
        # Ensure we don't exceed limits
        if min_required >= max_nodes:
            # Fallback to minimal allocation
            computational_nodes = max(1, max_nodes - min_required)
            return {
                'judges': max(1, min_judges // 2),
                'splitters': max(1, min_splitters // 2),
                'computational': computational_nodes,
                'reviewers': max(1, min_reviewers // 2),
                'retainers': max(1, min_retainers // 2)
            }
        
        # Allocate 85% of available space to computational nodes
        available_for_computational = max_nodes - min_required
        computational_nodes = int(available_for_computational * 0.85)
        
        # Remaining 15% distributed among other node types proportionally
        remaining_nodes = max_nodes - computational_nodes - min_required
        
        # Distribute remaining nodes proportionally
        extra_judges = int(remaining_nodes * 0.4)  # 40% of remaining
        extra_splitters = int(remaining_nodes * 0.3)  # 30% of remaining
        extra_reviewers = int(remaining_nodes * 0.2)  # 20% of remaining
        extra_retainers = remaining_nodes - extra_judges - extra_splitters - extra_reviewers  # Rest
        
        allocation = {
            'judges': min_judges + extra_judges,
            'splitters': min_splitters + extra_splitters,
            'computational': computational_nodes,
            'reviewers': min_reviewers + extra_reviewers,
            'retainers': min_retainers + extra_retainers
        }
        
        # Verify total doesn't exceed max_nodes
        total_allocated = sum(allocation.values())
        if total_allocated > max_nodes:
            # Reduce computational nodes to fit
            excess = total_allocated - max_nodes
            allocation['computational'] = max(1, allocation['computational'] - excess)
        
        return allocation
    
    def _create_judge_nodes(self, target_count: Optional[int] = None) -> List[int]:
        """
        Create judge nodes at ±1 positions for each dimensional polarity.
        
        Args:
            target_count: Target number of judge nodes to create
        
        Returns:
            List[int]: List of created judge node IDs
        """
        judge_nodes = []
        dimensions = list(self.dimensional_assignment.keys())
        
        if target_count is None:
            target_count = len(dimensions)  # Default: one per dimension
        
        # Create judge nodes up to target count
        nodes_created = 0
        
        # First, create one judge per dimensional assignment at ±1 position
        for dim_idx, polarity in self.dimensional_assignment.items():
            if nodes_created >= target_count:
                break
                
            # Calculate position: ±1 in assigned dimension, 0 in others
            position = [0.0] * self.effective_brain_dimensions
            position[dim_idx] = 1.0 if polarity > 0 else -1.0
            
            # Create judge node using brain nexus
            judge_id = self.brain_nexus.add_neural_node(
                node_type='Judge',
                position=position,
                node_group=f'segment_{self.segment_id}_judges',
                segment_id=self.segment_id,
                dimensional_assignment={dim_idx: polarity}
            )
            
            judge_nodes.append(judge_id)
            self.segment_nodes[judge_id] = self.brain_nexus.node_registry[judge_id]
            self.node_type_registry['judges'].append(judge_id)
            
            # Track judge relevance (initially neutral)
            self.judge_relevance_scores[judge_id] = 0.5
            nodes_created += 1
            
            if self.demo:
                polarity_str = '+' if polarity is not None and polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][dim_idx] if dim_idx < 10 else f'd{dim_idx}'
                print(f"   ✓ Judge {judge_id}: {polarity_str}1 {dim_name} at {position[:3]}...")
        
        # Create additional judge nodes if needed
        while nodes_created < target_count:
            # Create judges at varied positions
            dim_idx = nodes_created % len(dimensions)
            variation = (nodes_created // len(dimensions)) * 0.2  # Slight position variation
            polarity = self.dimensional_assignment[dimensions[dim_idx]]
            
            position = [0.0] * self.effective_brain_dimensions
            position[dim_idx] = (1.0 + variation) * (1.0 if polarity > 0 else -1.0)
            
            judge_id = self.brain_nexus.add_neural_node(
                node_type='Judge',
                position=position,
                node_group=f'segment_{self.segment_id}_judges',
                segment_id=self.segment_id,
                dimensional_assignment={dim_idx: polarity}
            )
            
            judge_nodes.append(judge_id)
            self.segment_nodes[judge_id] = self.brain_nexus.node_registry[judge_id]
            self.node_type_registry['judges'].append(judge_id)
            self.judge_relevance_scores[judge_id] = 0.5
            nodes_created += 1
            
            if self.demo:
                polarity_str = '+' if polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][dim_idx] if dim_idx < 10 else f'd{dim_idx}'
                print(f"   ✓ Judge {judge_id}: {polarity_str}{1.0+variation:.1f} {dim_name} at {position[:3]}...")
        
        return judge_nodes
    
    def _create_splitter_nodes(self, judge_nodes: List[int], target_count: Optional[int] = None) -> List[int]:
        """
        Create splitter nodes at ±2 positions, connected to corresponding judges.
        
        Args:
            judge_nodes: List of judge node IDs to connect to
            target_count: Target number of splitter nodes to create
            
        Returns:
            List[int]: List of created splitter node IDs
        """
        splitter_nodes = []
        
        if target_count is None:
            target_count = len(judge_nodes)  # Default: one per judge
        
        nodes_created = 0
        
        # Create splitter nodes corresponding to judge nodes
        for judge_id in judge_nodes:
            if nodes_created >= target_count:
                break
                
            judge_node = self.brain_nexus.node_registry[judge_id]
            judge_position = judge_node.node_position
            
            # Find the non-zero dimension (the one that was ±1)
            active_dim_idx = None
            polarity = None
            for i, coord in enumerate(judge_position):
                if abs(coord) > 0.5:  # Should be ±1
                    active_dim_idx = i
                    polarity = 1 if coord > 0 else -1
                    break
            
            if active_dim_idx is None:
                continue  # Skip if we can't find the active dimension
            
            # Create splitter position at ±2 in the same dimension
            splitter_position = [0.0] * self.effective_brain_dimensions
            # Ensure polarity is not None before comparison
            if polarity is None:
                polarity = 1  # Default to positive if undetermined
            splitter_position[active_dim_idx] = 2.0 if polarity > 0 else -2.0
            
            # Create splitter node
            splitter_id = self.brain_nexus.add_neural_node(
                node_type='Splitter',
                position=splitter_position,
                node_group=f'segment_{self.segment_id}_splitters',
                segment_id=self.segment_id,
                num_branches=self.config.get('splitters_per_judge', 2)
            )
            
            splitter_nodes.append(splitter_id)
            self.segment_nodes[splitter_id] = self.brain_nexus.node_registry[splitter_id]
            self.node_type_registry['splitters'].append(splitter_id)
            
            # Connect judge to splitter
            self.brain_nexus.connect_nodes(judge_id, splitter_id, weight=1.0)
            nodes_created += 1
            
            if self.demo:
                polarity_str = '+' if polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][active_dim_idx] if active_dim_idx < 10 else f'd{active_dim_idx}'
                print(f"   ✓ Splitter {splitter_id}: {polarity_str}2 {dim_name} at {splitter_position[:3]}... → Judge {judge_id}")
        
        # Create additional splitter nodes if needed
        while nodes_created < target_count:
            # Create splitters at varied positions
            judge_idx = nodes_created % len(judge_nodes)
            judge_id = judge_nodes[judge_idx]
            judge_node = self.brain_nexus.node_registry[judge_id]
            judge_position = judge_node.node_position
            
            # Find active dimension
            active_dim_idx = None
            polarity = None
            for i, coord in enumerate(judge_position):
                if abs(coord) > 0.5:
                    active_dim_idx = i
                    polarity = 1 if coord > 0 else -1
                    break
            
            if active_dim_idx is None:
                active_dim_idx = 0
                polarity = 1
            
            if polarity is None:
                polarity = 1
            
            # Create varied splitter position
            variation = (nodes_created // len(judge_nodes)) * 0.3
            splitter_position = [0.0] * self.effective_brain_dimensions
            splitter_position[active_dim_idx] = (2.0 + variation) * (1.0 if polarity > 0 else -1.0)
            
            splitter_id = self.brain_nexus.add_neural_node(
                node_type='Splitter',
                position=splitter_position,
                node_group=f'segment_{self.segment_id}_splitters',
                segment_id=self.segment_id,
                num_branches=self.config.get('splitters_per_judge', 2)
            )
            
            splitter_nodes.append(splitter_id)
            self.segment_nodes[splitter_id] = self.brain_nexus.node_registry[splitter_id]
            self.node_type_registry['splitters'].append(splitter_id)
            self.brain_nexus.connect_nodes(judge_id, splitter_id, weight=0.8)
            nodes_created += 1
            
            if self.demo:
                polarity_str = '+' if polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][active_dim_idx] if active_dim_idx < 10 else f'd{active_dim_idx}'
                print(f"   ✓ Splitter {splitter_id}: {polarity_str}{2.0+variation:.1f} {dim_name} at {splitter_position[:3]}... → Judge {judge_id}")
        
        return splitter_nodes
    
    def _create_computational_nodes(self, judge_nodes: List[int], splitter_nodes: List[int]) -> List[int]:
        """
        Create computational nodes at intermediate positions between judges and splitters.
        These nodes can potentially evolve into judge nodes with high performance.
        
        Args:
            judge_nodes: List of judge node IDs for connection
            splitter_nodes: List of splitter node IDs for connection
            
        Returns:
            List[int]: IDs of created computational nodes
        """
        computational_nodes = []
        
        # Create computational nodes at intermediate positions (±1.5 distance)
        dimensions = list(self.dimensional_assignment.keys())
        comp_distance = 1.5  # Between judges (±1) and splitters (±2)
        
        # Create one computational node per active dimension
        for dim_idx, polarity in self.dimensional_assignment.items():
            if dim_idx >= len(dimensions):
                continue
                
            # Create position at intermediate distance
            comp_position = np.zeros(len(dimensions))
            comp_position[dim_idx] = polarity * comp_distance
            
            # Create the computational node using brain nexus ID system
            comp_id = self.brain_nexus.next_node_id
            self.brain_nexus.next_node_id += 1
            
            # Import here to avoid circular imports
            from computations import Computational
            
            comp_node = Computational(
                node_id=comp_id,
                position=comp_position.tolist(),
                embed_dim=self._get_memory_optimized_embed_dim(),
                demo=self.demo
            )
            
            # Add to brain nexus registries
            self.brain_nexus.node_registry[comp_id] = comp_node
            if hasattr(self.brain_nexus, 'all_nodes'):
                self.brain_nexus.all_nodes[comp_id] = comp_node
            self.segment_nodes[comp_id] = comp_node
            computational_nodes.append(comp_id)
            
            # Connect to judges and splitters
            for judge_id in judge_nodes:
                self.brain_nexus.connect_nodes(comp_id, judge_id, weight=0.8)
            for splitter_id in splitter_nodes:
                self.brain_nexus.connect_nodes(comp_id, splitter_id, weight=0.6)
            
            # Register in node type registry
            self.node_type_registry['computational'].append(comp_id)
            
            if self.demo:
                polarity_str = '+' if polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][dim_idx] if dim_idx < 10 else f'd{dim_idx}'
                print(f"   ✓ Computational {comp_id}: {polarity_str}1.5 {dim_name} at {comp_position[:3]}... (evolution candidate)")
        
        return computational_nodes
    
    def _create_computational_nodes_dynamic(self, judge_nodes: List[int], splitter_nodes: List[int], target_count: int) -> List[int]:
        """
        Create computational nodes dynamically to fill 85% of segment space.
        These nodes are distributed throughout the dimensional space and can evolve into judge nodes.
        
        Args:
            judge_nodes: List of judge node IDs for connection
            splitter_nodes: List of splitter node IDs for connection
            target_count: Target number of computational nodes to create (85% of total space)
            
        Returns:
            List[int]: IDs of created computational nodes
        """
        computational_nodes = []
        dimensions = list(self.dimensional_assignment.keys())
        
        if self.demo:
            print(f"   Creating {target_count} computational nodes (85% of space allocation)")
        
        # Create computational nodes distributed throughout the space
        for i in range(target_count):
            # Distribute nodes across different dimensional positions
            dim_idx = i % len(dimensions) if dimensions else 0
            polarity = self.dimensional_assignment.get(dim_idx, 1)
            
            # Create varied positions throughout the space
            # Use different distance patterns for spatial distribution
            distance_patterns = [1.2, 1.5, 1.8, 2.2, 2.5, 2.8]  # Various distances from origin
            pattern_idx = i % len(distance_patterns)
            comp_distance = distance_patterns[pattern_idx]
            
            # Add spatial variation for better distribution
            spatial_variation = (i // len(dimensions)) * 0.1
            actual_distance = comp_distance + spatial_variation
            
            # Create position
            comp_position = np.zeros(max(len(dimensions), self.effective_brain_dimensions))
            
            # Primary dimension placement
            comp_position[dim_idx] = polarity * actual_distance
            
            # Add secondary dimensional components for better space utilization
            if len(dimensions) > 1:
                secondary_dim = (dim_idx + 1) % len(dimensions)
                secondary_polarity = self.dimensional_assignment.get(secondary_dim, 1)
                comp_position[secondary_dim] = secondary_polarity * (actual_distance * 0.3)
            
            # Create the computational node using brain nexus ID system
            comp_id = self.brain_nexus.next_node_id
            self.brain_nexus.next_node_id += 1
            
            # Import here to avoid circular imports
            from computations import Computational
            
            comp_node = Computational(
                node_id=comp_id,
                position=comp_position.tolist()[:self.effective_brain_dimensions],
                embed_dim=self._get_memory_optimized_embed_dim(),
                demo=self.demo
            )
            
            # Add to brain nexus registries
            self.brain_nexus.node_registry[comp_id] = comp_node
            if hasattr(self.brain_nexus, 'all_nodes'):
                self.brain_nexus.all_nodes[comp_id] = comp_node
            self.segment_nodes[comp_id] = comp_node
            computational_nodes.append(comp_id)
            
            # Connect to judges and splitters with varied weights
            connection_weight_judge = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, or 0.9
            connection_weight_splitter = 0.5 + (i % 4) * 0.1  # 0.5, 0.6, 0.7, or 0.8
            
            for judge_id in judge_nodes:
                self.brain_nexus.connect_nodes(comp_id, judge_id, weight=connection_weight_judge)
            for splitter_id in splitter_nodes:
                self.brain_nexus.connect_nodes(comp_id, splitter_id, weight=connection_weight_splitter)
            
            # Register in node type registry
            self.node_type_registry['computational'].append(comp_id)
            
            if self.demo and (i < 5 or i % (target_count // 10 + 1) == 0):  # Show first 5 and every 10th
                polarity_str = '+' if polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][dim_idx] if dim_idx < 10 else f'd{dim_idx}'
                print(f"   ✓ Computational {comp_id}: {polarity_str}{actual_distance:.1f} {dim_name} at {comp_position[:3]}... (evolution candidate {i+1}/{target_count})")
        
        return computational_nodes
    
    def _create_reviewer_retainer_nodes(self, furthest_distance: float) -> Tuple[List[int], List[int]]:
        """
        Create reviewer and retainer nodes at the furthest positions from origin.
        
        Args:
            furthest_distance: Distance for the furthest nodes
            
        Returns:
            Tuple[List[int], List[int]]: (reviewer_nodes, retainer_nodes)
        """
        reviewer_nodes = []
        retainer_nodes = []
        
        # Create reviewer/retainer pairs for each dimensional assignment
        for dim_idx, polarity in self.dimensional_assignment.items():
            # Reviewer at furthest point: ±furthest_distance
            reviewer_position = [0.0] * self.effective_brain_dimensions
            reviewer_position[dim_idx] = furthest_distance if polarity > 0 else -furthest_distance
            
            reviewer_id = self.brain_nexus.add_neural_node(
                node_type='Reviewer',
                position=reviewer_position,
                node_group=f'segment_{self.segment_id}_reviewers',
                segment_id=self.segment_id,
                num_comps=self.config.get('reviewers_per_retainer', 1)
            )
            
            reviewer_nodes.append(reviewer_id)
            self.segment_nodes[reviewer_id] = self.brain_nexus.node_registry[reviewer_id]
            self.node_type_registry['reviewers'].append(reviewer_id)
            
            # Retainer at ±(furthest_distance - 1) - closer to origin
            retainer_position = [0.0] * self.effective_brain_dimensions
            retainer_distance = furthest_distance - 1
            retainer_position[dim_idx] = retainer_distance if polarity > 0 else -retainer_distance
            
            retainer_id = self.brain_nexus.add_neural_node(
                node_type='Retainer',
                position=retainer_position,
                node_group=f'segment_{self.segment_id}_retainers',
                segment_id=self.segment_id,
                expected_nodes=self.config.get('retainers_per_group', 5)
            )
            
            retainer_nodes.append(retainer_id)
            self.segment_nodes[retainer_id] = self.brain_nexus.node_registry[retainer_id]
            self.node_type_registry['retainers'].append(retainer_id)
            
            # Connect retainer to reviewer
            self.brain_nexus.connect_nodes(retainer_id, reviewer_id, weight=1.0)
            
            if self.demo:
                polarity_str = '+' if polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][dim_idx] if dim_idx < 10 else f'd{dim_idx}'
                print(f"   ✓ Reviewer {reviewer_id}: {polarity_str}{furthest_distance:.0f} {dim_name}")
                print(f"   ✓ Retainer {retainer_id}: {polarity_str}{retainer_distance:.0f} {dim_name} → Reviewer {reviewer_id}")
        
        return reviewer_nodes, retainer_nodes
    
    def _create_reviewer_retainer_nodes_dynamic(self, furthest_distance: float, reviewer_count: int, retainer_count: int) -> Tuple[List[int], List[int]]:
        """
        Create reviewer and retainer nodes dynamically based on target counts.
        
        Args:
            furthest_distance: Distance for the furthest nodes
            reviewer_count: Target number of reviewer nodes
            retainer_count: Target number of retainer nodes
            
        Returns:
            Tuple[List[int], List[int]]: (reviewer_nodes, retainer_nodes)
        """
        reviewer_nodes = []
        retainer_nodes = []
        dimensions = list(self.dimensional_assignment.keys())
        
        # Create reviewer nodes
        for i in range(reviewer_count):
            dim_idx = i % len(dimensions) if dimensions else 0
            polarity = self.dimensional_assignment.get(dim_idx, 1)
            
            # Vary the distance slightly for different reviewers
            distance_variation = (i // len(dimensions)) * 0.1
            actual_distance = furthest_distance - distance_variation
            
            reviewer_position = [0.0] * self.effective_brain_dimensions
            reviewer_position[dim_idx] = actual_distance if polarity > 0 else -actual_distance
            
            reviewer_id = self.brain_nexus.add_neural_node(
                node_type='Reviewer',
                position=reviewer_position,
                node_group=f'segment_{self.segment_id}_reviewers',
                segment_id=self.segment_id,
                num_comps=self.config.get('reviewers_per_retainer', 1)
            )
            
            reviewer_nodes.append(reviewer_id)
            self.segment_nodes[reviewer_id] = self.brain_nexus.node_registry[reviewer_id]
            self.node_type_registry['reviewers'].append(reviewer_id)
            
            if self.demo and (i < 3 or i == reviewer_count - 1):  # Show first 3 and last
                polarity_str = '+' if polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][dim_idx] if dim_idx < 10 else f'd{dim_idx}'
                print(f"   ✓ Reviewer {reviewer_id}: {polarity_str}{actual_distance:.1f} {dim_name} ({i+1}/{reviewer_count})")
        
        # Create retainer nodes
        for i in range(retainer_count):
            dim_idx = i % len(dimensions) if dimensions else 0
            polarity = self.dimensional_assignment.get(dim_idx, 1)
            
            # Retainers are closer to origin than reviewers
            distance_variation = (i // len(dimensions)) * 0.1
            retainer_distance = furthest_distance - 1 - distance_variation
            
            retainer_position = [0.0] * self.effective_brain_dimensions
            retainer_position[dim_idx] = retainer_distance if polarity > 0 else -retainer_distance
            
            retainer_id = self.brain_nexus.add_neural_node(
                node_type='Retainer',
                position=retainer_position,
                node_group=f'segment_{self.segment_id}_retainers',
                segment_id=self.segment_id,
                expected_nodes=self.config.get('retainers_per_group', 5)
            )
            
            retainer_nodes.append(retainer_id)
            self.segment_nodes[retainer_id] = self.brain_nexus.node_registry[retainer_id]
            self.node_type_registry['retainers'].append(retainer_id)
            
            # Connect retainer to corresponding reviewer (round-robin)
            if reviewer_nodes:
                reviewer_id = reviewer_nodes[i % len(reviewer_nodes)]
                self.brain_nexus.connect_nodes(retainer_id, reviewer_id, weight=1.0)
            
            if self.demo and (i < 3 or i == retainer_count - 1):  # Show first 3 and last
                polarity_str = '+' if polarity > 0 else '-'
                dim_name = ['x','y','z','w','v','u','t','s','r','q'][dim_idx] if dim_idx < 10 else f'd{dim_idx}'
                reviewer_connection = f" → Reviewer {reviewer_nodes[i % len(reviewer_nodes)]}" if reviewer_nodes else ""
                print(f"   ✓ Retainer {retainer_id}: {polarity_str}{retainer_distance:.1f} {dim_name} ({i+1}/{retainer_count}){reviewer_connection}")
        
        return reviewer_nodes, retainer_nodes
    
    def _update_node_registries(self, judge_nodes: List[int], splitter_nodes: List[int], 
                               computational_nodes: List[int], reviewer_nodes: List[int], retainer_nodes: List[int]):
        """
        Update internal registries and tracking for created nodes.
        
        Args:
            judge_nodes: List of judge node IDs
            splitter_nodes: List of splitter node IDs  
            computational_nodes: List of computational node IDs
            reviewer_nodes: List of reviewer node IDs
            retainer_nodes: List of retainer node IDs
        """
        all_nodes = judge_nodes + splitter_nodes + computational_nodes + reviewer_nodes + retainer_nodes
        
        # Update resource tracking
        self.current_resources['nodes_count'] = len(all_nodes)
        
        # Update connection count (judges→splitters + retainers→reviewers)
        connection_count = len(judge_nodes) + len(retainer_nodes)
        self.current_resources['connections_count'] = connection_count
        
        # Update spatial zone occupancy
        for zone_name, zone_info in self.spatial_zones.items():
            if 'judge' in zone_name:
                zone_info['current_occupancy'] = len(judge_nodes)
            elif 'splitter' in zone_name:
                zone_info['current_occupancy'] = len(splitter_nodes)
            elif 'reviewer' in zone_name:
                zone_info['current_occupancy'] = len(reviewer_nodes)
            elif 'retainer' in zone_name:
                zone_info['current_occupancy'] = len(retainer_nodes)
        
        # Initialize all judges as potentially active (will be filtered during processing)
        self.active_judges = set(judge_nodes)
        
        # Update efficiency metrics
        self.efficiency_metrics['nodes_utilized'] = len(all_nodes)
        self.efficiency_metrics['connections_active'] = connection_count
    
    def _print_node_creation_summary(self, judge_nodes: List[int], splitter_nodes: List[int],
                                    computational_nodes: List[int], reviewer_nodes: List[int], retainer_nodes: List[int]):
        """Print a summary of node creation for debugging."""
        total_nodes = len(judge_nodes) + len(splitter_nodes) + len(computational_nodes) + len(reviewer_nodes) + len(retainer_nodes)
        total_connections = len(judge_nodes) + len(computational_nodes) + len(retainer_nodes)  # Judge→Splitter + Computational connections + Retainer→Reviewer
        
        print(f"\n📊 Node Creation Summary for Segment {self.segment_id}:")
        print(f"   Judge Nodes: {len(judge_nodes)} (at ±1 positions)")
        print(f"   Splitter Nodes: {len(splitter_nodes)} (at ±2 positions)")  
        print(f"   Computational Nodes: {len(computational_nodes)} (at ±1.5 positions) 🧬")
        print(f"   Reviewer Nodes: {len(reviewer_nodes)} (at furthest positions)")
        print(f"   Retainer Nodes: {len(retainer_nodes)} (at furthest-1 positions)")
        print(f"   Total Nodes: {total_nodes}")
        print(f"   Total Connections: {total_connections}")
        
        print(f"\n🔗 Connection Pattern:")
        print(f"   Judges → Splitters: {len(judge_nodes)} connections")
        print(f"   Retainers → Reviewers: {len(retainer_nodes)} connections")
        
        # Show resource utilization
        node_capacity = sum(zone['node_capacity'] for zone in self.spatial_zones.values())
        utilization = (total_nodes / node_capacity) * 100 if node_capacity > 0 else 0
        print(f"   Resource Utilization: {utilization:.1f}% ({total_nodes}/{node_capacity})")

    
    def _print_initialization_summary(self):
        """Print a comprehensive summary of the segment initialization for debugging."""
        print(f"\n🧠 NexusSegment {self.segment_id} Initialized")
        print(f"   Dimensional Signature: {self.dimensional_signature}")
        print(f"   Assignment: {self.dimensional_assignment}")
        print(f"   Dimensions: Segment={self.dimensions}D, Effective={self.effective_brain_dimensions}D, Max_Index={self.max_dimension_index}")
        
        # Show center position (truncated for readability)
        center_display = [f'{x:.1f}' for x in self.segment_center[:min(6, len(self.segment_center))]]
        if len(self.segment_center) > 6:
            center_display.append('...')
        print(f"   Center Position: [{', '.join(center_display)}]")
        print(f"   Segment Radius: {self.segment_radius:.1f}")
        print(f"   Dimensional Volume: {self.dimensional_volume:.2e}")
        print(f"   Dimensional Density: {self.dimensional_density:.2f} nodes/unit")
        # Display hypercube bounds in a readable format
        if isinstance(self.hypercube_bounds, list) and len(self.hypercube_bounds) > 0:
            dim_labels = ['x', 'y', 'z', 'w', 'v', 'u', 't', 's', 'r', 'q']
            bounds_display = []
            for i, bounds in enumerate(self.hypercube_bounds):
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    label = dim_labels[i] if i < len(dim_labels) else f'd{i}'
                    min_bound, max_bound = bounds
                    bounds_display.append(f"{label}:[{min_bound:.0f}, {max_bound:.0f}]")
            print(f"   Hypercube Bounds: {' × '.join(bounds_display)}")
        else:
            print(f"   Hypercube Bounds: {self.hypercube_bounds}")
        
        # Dimensional compatibility info
        if hasattr(self, 'dimensional_compatibility'):
            compat = self.dimensional_compatibility
            if compat['dimension_extension_applied']:
                print(f"   🔄 Dimensional Extension: {compat['brain_original_dims']}D → {compat['segment_required_dims']}D")
            else:
                print(f"   ✓ Dimensional Compatibility: {compat['segment_required_dims']}D")
        
        print(f"\n📊 Configuration:")
        key_configs = ['judges_per_segment', 'max_active_judges', 'computational_nodes_base', 
                      'computation_budget', 'judge_selection_ratio', 'computational_selection_ratio']
        for key in key_configs:
            if key in self.config:
                value = self.config[key]
                if isinstance(value, float) and 0 < value < 1:
                    print(f"   {key}: {value:.1%}")
                else:
                    print(f"   {key}: {value}")
        
        print(f"\n🗺️  Spatial Zones ({len(self.spatial_zones)} zones):")
        for zone_name, zone_info in self.spatial_zones.items():
            zone_center = zone_info['center']
            center_truncated = [f'{x:.1f}' for x in zone_center[:3]]
            if len(zone_center) > 3:
                center_truncated.append('...')
            
            print(f"   {zone_name}: capacity={zone_info['node_capacity']}, "
                  f"radius={zone_info['radius']:.1f}, center=[{', '.join(center_truncated)}]")
        
        print(f"\n🔗 Integration Status:")
        print(f"   Brain Nexus: {'✓' if self.config['brain_nexus_integration'] else '✗'}")
        print(f"   Cross-Segment Comm: {'✓' if self.config['cross_segment_communication'] else '✗'}")
        print(f"   Shared Embeddings: {'✓' if self.config['shared_embedding_space'] else '✗'}")
        print(f"   Lifecycle: {self.lifecycle_state} → {self.activation_state}")
        
        # Performance expectations
        expected_nodes = sum(zone['node_capacity'] for zone in self.spatial_zones.values())
        print(f"\n⚡ Performance Expectations:")
        print(f"   Expected Total Nodes: {expected_nodes}")
        print(f"   Computation Budget: {self.config['computation_budget']}")
        print(f"   Memory Limit: {self.resource_limits['memory_limit'] / (1024*1024):.0f}MB")
        print(f"   Processing Timeout: {self.resource_limits['processing_timeout']:.1f}s")
    def save_segment(self, 
                 filename: Optional[str] = None, 
                 segments_dir: str = "segments",
                 include_brain_nexus_ref: bool = False,
                 compression_level: int = 2) -> str:
        """
        Save the complete NexusSegment state including all nodes, positions, connections, and weights.
        
        Args:
            filename (Optional[str]): Custom filename (without .pkl extension). 
                                    If None, uses datetime format: "segment_{segment_id}_{timestamp}.pkl"
            segments_dir (str): Directory to save segments in. Created if doesn't exist.
            include_brain_nexus_ref (bool): Whether to include brain_nexus reference (use carefully - can cause circular refs)
            compression_level (int): Pickle protocol level (0-5, higher = more compressed but requires newer Python)
            
        Returns:
            str: Full path to saved segment file
            
        Raises:
            OSError: If directory creation or file writing fails
            PicklingError: If segment data cannot be serialized
        """
        
        # Create segments directory if it doesn't exist
        os.makedirs(segments_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"segment_{self.segment_id}_{timestamp}"
        
        # Ensure .pkl extension
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        # Full file path
        filepath = os.path.join(segments_dir, filename)
        
        # Prepare segment data for serialization
        segment_data = self._prepare_segment_data_for_saving(include_brain_nexus_ref)
        
        # Add metadata
        save_metadata = {
            'save_timestamp': datetime.now().isoformat(),
            'segment_id': self.segment_id,
            'dimensional_signature': self.dimensional_signature,
            'total_nodes': len(self.segment_nodes),
            'total_connections': self._count_total_connections(),
            'brain_nexus_included': include_brain_nexus_ref,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'save_format_version': "1.0"
        }
        
        # Complete save package
        save_package = {
            'metadata': save_metadata,
            'segment_data': segment_data,
            'validation_hash': self._generate_validation_hash(segment_data)
        }
        
        try:
            # Save with specified compression level
            with open(filepath, 'wb') as f:
                pickle.dump(save_package, f, protocol=compression_level)
            
            if self.demo:
                self._print_save_summary(filepath, save_metadata, len(segment_data))
            
            # Update segment state
            self.last_save_time = datetime.now()
            self.last_save_path = filepath
            
            return filepath
            
        except Exception as e:
            error_msg = f"Failed to save segment {self.segment_id} to {filepath}: {str(e)}"
            if self.demo:
                print(f"❌ {error_msg}")
            raise OSError(error_msg) from e

    def _prepare_segment_data_for_saving(self, include_brain_nexus_ref: bool) -> Dict[str, Any]:
        """
        Prepare all segment data for serialization, handling complex objects and references.
        
        Args:
            include_brain_nexus_ref (bool): Whether to include brain_nexus reference
            
        Returns:
            Dict[str, Any]: Serializable segment data
        """
        
        # Core segment properties
        segment_data = {
            # Identity and configuration
            'segment_id': self.segment_id,
            'dimensional_assignment': self.dimensional_assignment.copy(),
            'hypercube_bounds': self.hypercube_bounds,
            'config': self.config.copy(),
            'demo': self.demo,
            
            # Dimensional properties
            'dimensions': self.dimensions,
            'max_dimension_index': self.max_dimension_index,
            'effective_brain_dimensions': self.effective_brain_dimensions,
            'dimensional_signature': self.dimensional_signature,
            'segment_center': self.segment_center.copy(),
            'segment_radius': self.segment_radius,
            'dimensional_volume': self.dimensional_volume,
            'dimensional_density': self.dimensional_density,
            
            # Node data - the core of the segment
            'segment_nodes_data': self._serialize_segment_nodes(),
            'node_type_registry': {
                node_type: node_list.copy() for node_type, node_list in self.node_type_registry.items()
            },
            
            # Spatial organization
            'spatial_zones': self._serialize_spatial_zones(),
            'connection_matrix': self._serialize_connection_matrix(),
            'external_connections': self._serialize_external_connections(),
            
            # State and activation
            'activation_state': self.activation_state,
            'relevance_score': self.relevance_score,
            'activation_threshold': self.activation_threshold,
            'last_activation_time': self.last_activation_time,
            'activation_history': list(self.activation_history),
            
            # Judge management
            'active_judges': list(self.active_judges),
            'judge_relevance_scores': self.judge_relevance_scores.copy(),
            'max_active_judges': self.max_active_judges,
            'judge_activation_ratio': self.judge_activation_ratio,
            
            # Processing caches and transformations
            'attention_cache': self._serialize_attention_cache(),
            'embedding_transformations': self._serialize_embedding_transformations(),
            'positional_encodings': self._serialize_positional_encodings(),
            
            # Pipeline state
            'pipeline_state': self.pipeline_state.copy(),
            'processing_results': self._serialize_processing_results(),
            
            # Performance and metrics
            'computation_budget': self.computation_budget,
            'remaining_budget': self.remaining_budget,
            'efficiency_metrics': self.efficiency_metrics.copy(),
            
            # Communication and sharing
            'communication_channels': self._serialize_communication_channels(),
            'shared_embeddings': self._serialize_shared_embeddings(),
            'synchronization_points': self.synchronization_points.copy(),
            
            # Memory and caching
            'result_cache': self._serialize_result_cache(),
            'pattern_memory': list(self.pattern_memory),
            'failure_patterns': list(self.failure_patterns),
            
            # Learning and adaptation
            'learning_rate': self.learning_rate,
            'adaptation_history': self.adaptation_history.copy(),
            'success_patterns': dict(self.success_patterns),
            'dimensional_preferences': self.dimensional_preferences.copy(),
            
            # Resource management
            'resource_limits': self.resource_limits.copy(),
            'current_resources': self.current_resources.copy(),
            
            # Quality metrics
            'quality_metrics': self._serialize_quality_metrics(),
            
            # Lifecycle information
            'creation_time': self.creation_time,
            'last_access_time': self.last_access_time,
            'lifecycle_state': self.lifecycle_state,
            'cleanup_scheduled': self.cleanup_scheduled,
            
            # Compatibility information
            'dimensional_compatibility': getattr(self, 'dimensional_compatibility', {}),
        }
        
        # Conditionally include brain_nexus reference
        if include_brain_nexus_ref:
            segment_data['brain_nexus_data'] = self._serialize_brain_nexus_reference()
        else:
            segment_data['brain_nexus_data'] = None
            segment_data['brain_nexus_metadata'] = self._extract_brain_nexus_metadata()
        
        return segment_data

    def _serialize_segment_nodes(self) -> Dict[int, Dict[str, Any]]:
        """
        Serialize all segment nodes with their positions, properties, and connections.
        
        Returns:
            Dict[int, Dict[str, Any]]: Serialized node data keyed by node_id
        """
        serialized_nodes = {}
        
        for node_id, node_obj in self.segment_nodes.items():
            try:
                # Extract node properties
                node_data = {
                    'node_id': node_id,
                    'node_type': getattr(node_obj, 'node_type', 'Unknown'),
                    'node_position': getattr(node_obj, 'node_position', []).copy() if hasattr(node_obj, 'node_position') else [],
                    'node_group': getattr(node_obj, 'node_group', ''),
                    'segment_id': getattr(node_obj, 'segment_id', self.segment_id),
                    
                    # Node-specific properties
                    'properties': {},
                    'state': {},
                    'connections': {}
                }
                
                # Serialize node-type specific properties
                if hasattr(node_obj, 'dimensional_assignment'):
                    node_data['properties']['dimensional_assignment'] = getattr(node_obj, 'dimensional_assignment', {})
                
                if hasattr(node_obj, 'num_branches'):  # Splitter nodes
                    node_data['properties']['num_branches'] = getattr(node_obj, 'num_branches', 2)
                
                if hasattr(node_obj, 'num_comps'):  # Reviewer nodes
                    node_data['properties']['num_comps'] = getattr(node_obj, 'num_comps', 1)
                
                if hasattr(node_obj, 'expected_nodes'):  # Retainer nodes
                    node_data['properties']['expected_nodes'] = getattr(node_obj, 'expected_nodes', 5)
                
                # Serialize node connections from brain_nexus
                if hasattr(self.brain_nexus, 'connection_matrix') and node_id in self.brain_nexus.connection_matrix:
                    connections = self.brain_nexus.connection_matrix[node_id]
                    node_data['connections'] = {
                        'outgoing': {},
                        'incoming': {}
                    }
                    
                    # Outgoing connections
                    for target_id, weight in connections.items():
                        node_data['connections']['outgoing'][target_id] = {
                            'weight': weight,
                            'target_id': target_id
                        }
                    
                    # Find incoming connections
                    for source_id, source_connections in self.brain_nexus.connection_matrix.items():
                        if node_id in source_connections:
                            if 'incoming' not in node_data['connections']:
                                node_data['connections']['incoming'] = {}
                            node_data['connections']['incoming'][source_id] = {
                                'weight': source_connections[node_id],
                                'source_id': source_id
                            }
                
                # Additional node state if available
                if hasattr(node_obj, 'activation_level'):
                    node_data['state']['activation_level'] = node_obj.activation_level
                
                if hasattr(node_obj, 'last_output'):
                    node_data['state']['last_output'] = node_obj.last_output
                
                if hasattr(node_obj, 'processing_history'):
                    node_data['state']['processing_history'] = list(node_obj.processing_history) if hasattr(node_obj.processing_history, '__iter__') else []
                
                serialized_nodes[node_id] = node_data
                
            except Exception as e:
                if self.demo:
                    print(f"⚠️  Warning: Could not fully serialize node {node_id}: {str(e)}")
                # Store minimal node data
                serialized_nodes[node_id] = {
                    'node_id': node_id,
                    'node_type': 'Unknown',
                    'node_position': [],
                    'serialization_error': str(e)
                }
        
        return serialized_nodes

    def _serialize_spatial_zones(self) -> Dict[str, Any]:
        """Serialize spatial zones data."""
        return {
            zone_name: {
                'center': zone_data['center'].copy() if isinstance(zone_data['center'], list) else zone_data['center'],
                'radius': zone_data['radius'],
                'node_capacity': zone_data['node_capacity'],
                'current_occupancy': zone_data['current_occupancy'],
                'dimensional_bounds': zone_data['dimensional_bounds'],
                'zone_type': zone_data['zone_type'],
                'priority_dimensions': zone_data['priority_dimensions'].copy()
            }
            for zone_name, zone_data in self.spatial_zones.items()
        }

    def _serialize_connection_matrix(self) -> Dict[str, Any]:
        """Serialize internal connection matrix."""
        return {
            str(source_id): {str(target_id): weight for target_id, weight in connections.items()}
            for source_id, connections in self.connection_matrix.items()
        }

    def _serialize_external_connections(self) -> Dict[str, List[Dict[str, Any]]]:
        """Serialize external connections to other segments."""
        serialized_external = {}
        
        for segment_id, connections in self.external_connections.items():
            serialized_connections = []
            for connection in connections:
                if isinstance(connection, dict):
                    serialized_connections.append(connection.copy())
                else:
                    # Handle other connection types
                    serialized_connections.append({'connection_data': str(connection)})
            serialized_external[str(segment_id)] = serialized_connections
        
        return serialized_external

    def _serialize_attention_cache(self) -> Dict[str, Any]:
        """Serialize attention cache, handling tensor data."""
        serialized_cache = {}
        
        for key, attention_data in self.attention_cache.items():
            try:
                if hasattr(attention_data, 'numpy'):  # PyTorch tensor
                    serialized_cache[str(key)] = {
                        'data_type': 'pytorch_tensor',
                        'data': attention_data.detach().cpu().numpy().tolist(),
                        'shape': list(attention_data.shape),
                        'dtype': str(attention_data.dtype)
                    }
                elif hasattr(attention_data, 'tolist'):  # NumPy array
                    serialized_cache[str(key)] = {
                        'data_type': 'numpy_array',
                        'data': attention_data.tolist(),
                        'shape': list(attention_data.shape),
                        'dtype': str(attention_data.dtype)
                    }
                elif isinstance(attention_data, (list, dict, str, int, float, bool)):
                    serialized_cache[str(key)] = {
                        'data_type': 'native',
                        'data': attention_data
                    }
                else:
                    serialized_cache[str(key)] = {
                        'data_type': 'string_repr',
                        'data': str(attention_data)
                    }
            except Exception as e:
                serialized_cache[str(key)] = {
                    'data_type': 'error',
                    'error': str(e)
                }
        
        return serialized_cache

    def _serialize_embedding_transformations(self) -> Dict[str, Any]:
        """Serialize embedding transformations."""
        return self._serialize_attention_cache.__func__(self)  # Reuse attention cache logic

    def _serialize_positional_encodings(self) -> Dict[str, Any]:
        """Serialize positional encodings."""
        return self._serialize_attention_cache.__func__(self)  # Reuse attention cache logic

    def _serialize_processing_results(self) -> Dict[str, Any]:
        """Serialize processing results."""
        serialized_results = {}
        
        for key, result in self.processing_results.items():
            try:
                if isinstance(result, (dict, list, str, int, float, bool)):
                    serialized_results[str(key)] = result
                else:
                    serialized_results[str(key)] = str(result)
            except:
                serialized_results[str(key)] = f"<Unserializable: {type(result).__name__}>"
        
        return serialized_results

    def _serialize_communication_channels(self) -> Dict[str, Any]:
        """Serialize communication channels."""
        return {
            str(segment_id): channel_state
            for segment_id, channel_state in self.communication_channels.items()
        }

    def _serialize_shared_embeddings(self) -> Dict[str, Any]:
        """Serialize shared embeddings."""
        return self._serialize_attention_cache.__func__(self)  # Reuse attention cache logic

    def _serialize_result_cache(self) -> Dict[str, Any]:
        """Serialize result cache."""
        return self._serialize_processing_results.__func__(self)  # Reuse processing results logic

    def _serialize_quality_metrics(self) -> Dict[str, Any]:
        """Serialize quality metrics with deque handling."""
        metrics = {}
        
        for metric_name, metric_value in self.quality_metrics.items():
            if hasattr(metric_value, 'copy') and callable(metric_value.copy):
                # Handle deques and lists
                metrics[metric_name] = list(metric_value)
            else:
                metrics[metric_name] = metric_value
        
        return metrics

    def _serialize_brain_nexus_reference(self) -> Dict[str, Any]:
        """
        Serialize minimal brain_nexus reference data (use with caution).
        Only captures essential connectivity information.
        """
        if not hasattr(self, 'brain_nexus') or self.brain_nexus is None:
            return {}
        
        brain_data = {
            'dimensions': getattr(self.brain_nexus, 'dimensions', 4),
            'connection_matrix_subset': {},
            'node_positions_subset': {},
            'metadata': {
                'total_nodes': len(getattr(self.brain_nexus, 'node_registry', {})),
                'segments_count': len(getattr(self.brain_nexus, 'segments', {}))
            }
        }
        
        # Only include connections involving this segment's nodes
        if hasattr(self.brain_nexus, 'connection_matrix'):
            for node_id in self.segment_nodes.keys():
                if node_id in self.brain_nexus.connection_matrix:
                    brain_data['connection_matrix_subset'][node_id] = self.brain_nexus.connection_matrix[node_id]
        
        # Only include positions for this segment's nodes
        if hasattr(self.brain_nexus, 'node_registry'):
            for node_id in self.segment_nodes.keys():
                if node_id in self.brain_nexus.node_registry:
                    node = self.brain_nexus.node_registry[node_id]
                    if hasattr(node, 'node_position'):
                        brain_data['node_positions_subset'][node_id] = node.node_position.copy()
        
        return brain_data

    def _extract_brain_nexus_metadata(self) -> Dict[str, Any]:
        """Extract essential brain_nexus metadata without full reference."""
        if not hasattr(self, 'brain_nexus') or self.brain_nexus is None:
            return {}
        
        return {
            'brain_dimensions': getattr(self.brain_nexus, 'dimensions', 4),
            'brain_total_nodes': len(getattr(self.brain_nexus, 'node_registry', {})),
            'brain_total_segments': len(getattr(self.brain_nexus, 'segments', {})),
            'brain_type': type(self.brain_nexus).__name__
        }

    def _count_total_connections(self) -> int:
        """Count total connections involving this segment's nodes."""
        total_connections = 0
        
        # Count internal connections
        total_connections += len(self.connection_matrix)
        
        # Count external connections
        for connections in self.external_connections.values():
            total_connections += len(connections)
        
        # Count connections in brain_nexus involving our nodes
        if hasattr(self.brain_nexus, 'connection_matrix'):
            for node_id in self.segment_nodes.keys():
                if node_id in self.brain_nexus.connection_matrix:
                    total_connections += len(self.brain_nexus.connection_matrix[node_id])
        
        return total_connections

    def _generate_validation_hash(self, segment_data: Dict[str, Any]) -> str:
        """Generate a validation hash for integrity checking."""
        import hashlib
        
        # Create a simplified representation for hashing
        hash_data = {
            'segment_id': segment_data['segment_id'],
            'dimensional_signature': segment_data['dimensional_signature'],
            'node_count': len(segment_data['segment_nodes_data']),
            'config_hash': str(hash(str(sorted(segment_data['config'].items()))))
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _print_save_summary(self, filepath: str, metadata: Dict[str, Any], data_size: int):
        """Print a summary of the save operation."""
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        
        print(f"\n💾 Segment {self.segment_id} Saved Successfully!")
        print(f"   File: {filepath}")
        print(f"   File Size: {file_size / 1024:.1f} KB")
        print(f"   Nodes Saved: {metadata['total_nodes']}")
        print(f"   Connections Saved: {metadata['total_connections']}")
        print(f"   Dimensional Signature: {metadata['dimensional_signature']}")
        print(f"   Save Timestamp: {metadata['save_timestamp']}")
        print(f"   Brain Nexus Included: {'Yes' if metadata['brain_nexus_included'] else 'No'}")
        print(f"   Format Version: {metadata['save_format_version']}")

    # =====================================
    # SEGMENT MEMORY CLEANUP SYSTEM
    # =====================================
    
    def cleanup_memory(self, cleanup_tier: str = 'partial', force_cleanup: bool = False) -> Dict[str, Any]:
        """
        Multi-tier memory cleanup system for NexusSegment.
        
        Args:
            cleanup_tier: Level of cleanup - 'light', 'partial', 'aggressive', 'nuclear'
            force_cleanup: Skip safety checks and force cleanup
            
        Returns:
            Dict with cleanup statistics and results
        """
        start_time = time.time()
        cleanup_stats = {
            'tier': cleanup_tier,
            'start_time': datetime.now().isoformat(),
            'pre_cleanup_memory': self._get_segment_memory_stats(),
            'cleaned_items': defaultdict(int),
            'errors': []
        }
        
        if self.demo:
            print(f"🧹 Segment {self.segment_id}: {cleanup_tier.upper()} memory cleanup...")
        
        try:
            # Execute cleanup based on tier
            if cleanup_tier == 'light':
                self._segment_light_cleanup(cleanup_stats, force_cleanup)
            elif cleanup_tier == 'partial':
                self._segment_partial_cleanup(cleanup_stats, force_cleanup)
            elif cleanup_tier == 'aggressive':
                self._segment_aggressive_cleanup(cleanup_stats, force_cleanup)
            elif cleanup_tier == 'nuclear':
                self._segment_nuclear_cleanup(cleanup_stats, force_cleanup)
            else:
                raise ValueError(f"Unknown cleanup tier: {cleanup_tier}")
                
        except Exception as e:
            cleanup_stats['errors'].append(f"Segment cleanup error: {str(e)}")
            if self.demo:
                print(f"❌ Segment cleanup error: {e}")
        
        # Calculate final statistics
        cleanup_time = time.time() - start_time
        cleanup_stats['cleanup_time'] = cleanup_time
        cleanup_stats['post_cleanup_memory'] = self._get_segment_memory_stats()
        cleanup_stats['memory_freed'] = self._calculate_segment_memory_freed(
            cleanup_stats['pre_cleanup_memory'], 
            cleanup_stats['post_cleanup_memory']
        )
        
        if self.demo:
            print(f"✅ Segment cleanup completed in {cleanup_time:.3f}s")
            print(f"   Items cleaned: {dict(cleanup_stats['cleaned_items'])}")
        
        return cleanup_stats

    def _segment_light_cleanup(self, stats: Dict[str, Any], force: bool = False):
        """Light cleanup for segment - remove only old/expired data."""
        
        # Trim attention cache if too large
        if len(self.attention_cache) > 200:
            # Keep only recent attention entries
            cache_items = list(self.attention_cache.items())[-100:]
            original_size = len(self.attention_cache)
            self.attention_cache.clear()
            self.attention_cache.update(cache_items)
            stats['cleaned_items']['attention_cache'] = original_size - len(self.attention_cache)
        
        # Trim result cache
        if len(self.result_cache) > 300:
            cache_items = list(self.result_cache.items())[-150:]
            original_size = len(self.result_cache)
            self.result_cache.clear()
            self.result_cache.update(cache_items)
            stats['cleaned_items']['result_cache'] = original_size - len(self.result_cache)
        
        # Trim pattern memory if full
        if len(self.pattern_memory) > 800:
            # Keep only recent patterns
            recent_patterns = list(self.pattern_memory)[-400:]
            original_size = len(self.pattern_memory)
            self.pattern_memory.clear()
            self.pattern_memory.extend(recent_patterns)
            stats['cleaned_items']['pattern_memory'] = original_size - len(self.pattern_memory)
        
        # Clean processing results - keep only recent
        if len(self.processing_results) > 100:
            results_items = list(self.processing_results.items())[-50:]
            original_size = len(self.processing_results)
            self.processing_results.clear()
            self.processing_results.update(results_items)
            stats['cleaned_items']['processing_results'] = original_size - len(self.processing_results)

    def _segment_partial_cleanup(self, stats: Dict[str, Any], force: bool = False):
        """Partial cleanup - moderate memory cleanup."""
        
        # Start with light cleanup
        self._segment_light_cleanup(stats, force)
        
        # Clear embedding transformations if large
        if len(self.embedding_transformations) > 100:
            original_size = len(self.embedding_transformations)
            self.embedding_transformations.clear()
            stats['cleaned_items']['embedding_transformations'] = original_size
        
        # Clear positional encodings
        if len(self.positional_encodings) > 0:
            original_size = len(self.positional_encodings)
            self.positional_encodings.clear()
            stats['cleaned_items']['positional_encodings'] = original_size
        
        # Clear shared embeddings if not actively used
        if len(self.shared_embeddings) > 50:
            original_size = len(self.shared_embeddings)
            self.shared_embeddings.clear()
            stats['cleaned_items']['shared_embeddings'] = original_size
        
        # Clean communication channels for inactive segments
        inactive_channels = []
        for seg_id, channel_state in self.communication_channels.items():
            if isinstance(channel_state, dict) and not channel_state.get('active', False):
                inactive_channels.append(seg_id)
        
        for seg_id in inactive_channels:
            del self.communication_channels[seg_id]
        stats['cleaned_items']['inactive_channels'] = len(inactive_channels)
        
        # Trim adaptation history
        if len(self.adaptation_history) > 200:
            original_size = len(self.adaptation_history)
            self.adaptation_history = self.adaptation_history[-100:]
            stats['cleaned_items']['adaptation_history'] = original_size - len(self.adaptation_history)

    def _segment_aggressive_cleanup(self, stats: Dict[str, Any], force: bool = False):
        """Aggressive cleanup - significant memory cleanup."""
        
        # Start with partial cleanup
        self._segment_partial_cleanup(stats, force)
        
        # Clear all caches completely
        cache_attrs = ['attention_cache', 'result_cache', 'embedding_transformations', 
                      'positional_encodings', 'shared_embeddings']
        
        for attr in cache_attrs:
            if hasattr(self, attr):
                original_size = len(getattr(self, attr, {}))
                if original_size > 0:
                    getattr(self, attr).clear()
                    stats['cleaned_items'][attr] += original_size
        
        # Clear processing results completely
        original_size = len(self.processing_results)
        self.processing_results.clear()
        stats['cleaned_items']['processing_results'] += original_size
        
        # Reset pipeline state
        for key in self.pipeline_state:
            self.pipeline_state[key] = False
        stats['cleaned_items']['pipeline_state_reset'] = 1
        
        # Clear failure patterns
        original_size = len(self.failure_patterns)
        self.failure_patterns.clear()
        stats['cleaned_items']['failure_patterns'] = original_size
        
        # Clear success patterns for low-performing patterns
        original_size = len(self.success_patterns)
        # Keep only high-performing patterns
        high_performers = {k: v for k, v in self.success_patterns.items() if v > 5}
        self.success_patterns.clear()
        self.success_patterns.update(high_performers)
        stats['cleaned_items']['success_patterns'] = original_size - len(self.success_patterns)

    def _segment_nuclear_cleanup(self, stats: Dict[str, Any], force: bool = False):
        """Nuclear cleanup - complete memory reset."""
        
        if not force:
            # Safety check
            if len(self.segment_nodes) > 20:
                stats['errors'].append("Nuclear cleanup requires force=True for segments with >20 nodes")
                return
        
        # Clear all memory structures
        memory_attrs = [
            'attention_cache', 'embedding_transformations', 'positional_encodings',
            'processing_results', 'shared_embeddings', 'result_cache'
        ]
        
        for attr in memory_attrs:
            if hasattr(self, attr):
                original_size = len(getattr(self, attr, {}))
                if original_size > 0:
                    setattr(self, attr, {})
                    stats['cleaned_items'][attr] = original_size
        
        # Clear all tracking and history
        self.pattern_memory.clear()
        self.failure_patterns.clear()
        self.adaptation_history.clear()
        self.success_patterns.clear()
        self.dimensional_preferences.clear()
        self.communication_channels.clear()
        self.synchronization_points.clear()
        
        # Reset all state
        for key in self.pipeline_state:
            self.pipeline_state[key] = False
        
        # Reset efficiency metrics
        for key in self.efficiency_metrics:
            if isinstance(self.efficiency_metrics[key], (int, float)):
                self.efficiency_metrics[key] = 0.0
        
        # Reset computation budget
        self.remaining_budget = self.computation_budget
        
        # Clear judge management
        self.active_judges.clear()
        self.judge_relevance_scores.clear()
        
        stats['cleaned_items']['nuclear_reset'] = 1

    def _get_segment_memory_stats(self) -> Dict[str, int]:
        """Get current segment memory usage statistics."""
        stats = {
            'total_nodes': len(self.segment_nodes),
            'attention_cache': len(self.attention_cache),
            'result_cache': len(self.result_cache),
            'processing_results': len(self.processing_results),
            'embedding_transformations': len(self.embedding_transformations),
            'positional_encodings': len(self.positional_encodings),
            'shared_embeddings': len(self.shared_embeddings),
            'pattern_memory': len(self.pattern_memory),
            'failure_patterns': len(self.failure_patterns),
            'adaptation_history': len(self.adaptation_history),
            'success_patterns': len(self.success_patterns),
            'communication_channels': len(self.communication_channels),
            'active_judges': len(self.active_judges),
            'judge_relevance_scores': len(self.judge_relevance_scores),
            'synchronization_points': len(self.synchronization_points)
        }
        return stats

    def _calculate_segment_memory_freed(self, pre_stats: Dict[str, int], post_stats: Dict[str, int]) -> Dict[str, int]:
        """Calculate memory freed by segment cleanup."""
        freed = {}
        for key in pre_stats:
            if key in post_stats:
                freed[key] = pre_stats[key] - post_stats[key]
        
        freed['total_items'] = sum(max(0, val) for val in freed.values())
        return freed

    def get_segment_memory_status(self) -> Dict[str, Any]:
        """Get detailed segment memory status and recommendations."""
        stats = self._get_segment_memory_stats()
        total_items = sum(stats.values())
        
        # Determine memory pressure level for segment
        if total_items > 5000:
            pressure = 'CRITICAL'
            recommendation = "cleanup_memory('nuclear', force=True)"
        elif total_items > 2000:
            pressure = 'HIGH'
            recommendation = "cleanup_memory('aggressive')"
        elif total_items > 1000:
            pressure = 'MODERATE'
            recommendation = "cleanup_memory('partial')"
        elif total_items > 500:
            pressure = 'LOW'
            recommendation = "cleanup_memory('light')"
        else:
            pressure = 'OPTIMAL'
            recommendation = 'No cleanup needed'
        
        return {
            'segment_id': self.segment_id,
            'memory_stats': stats,
            'total_items': total_items,
            'pressure_level': pressure,
            'recommendation': recommendation,
            'nodes_count': len(self.segment_nodes),
            'dimensional_signature': getattr(self, 'dimensional_signature', 'unknown')
        }