import random
import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any, Optional, Union
import uuid
import threading
import concurrent.futures
import time
from threading import Lock, RLock

class Controller:
    """Controller node that manages dynamic judge activation and dimensional space"""
    
    def __init__(self, node_id: str, node_position: Tuple[float, float, float] = (0.0, 0.0, 0.0), 
                 num_branches: int = 4, demo: bool = False):
        self.node_id = node_id
        self.node_position = node_position
        self.num_branches = num_branches
        self.demo = demo
        self.node_type = "Controller"
        
        # Dynamic dimensional management
        self.dimensions = {}  # dim_id -> {name, axis_info, scale, importance}
        self.judge_positions = {}  # judge_id -> position_vector
        self.judge_registry = {}  # judge_id -> Judge instance
        self.active_judges = set()
        self.dimension_counter = 0
        
        # Dynamic embedding configuration
        self.base_embed_dim = 128
        self.current_embed_dim = 128
        self.max_embed_dim = 512
        
        # Token processing capabilities
        self.token_params = self._init_token_params()
        
        # Judge management
        self.judge_activation_threshold = 0.5
        self.max_active_judges = None  # Will be set based on top 50%
        
        # Learning parameters
        self.learning_rate = 0.01
        self.judge_relevance_scores = {}  # judge_id -> relevance_score
        
        # Gradient flow parameters
        self.gradient_accumulation = {}
        self.gradient_momentum = 0.9
        self.gradient_clip_value = 1.0
        
        # Processing history
        self.processing_history = deque(maxlen=20)
        
        if demo:
            print(f"Controller {node_id} initialized at {node_position}")
    
    def _init_token_params(self) -> Dict[str, Any]:
        """Initialize token processing parameters with multi-modal support"""
        vocab_size = 50000
        
        return {
            'vocab_size': vocab_size,
            'embed_dim': self.current_embed_dim,
            'token_embedding_matrix': np.random.randn(vocab_size, self.current_embed_dim) * 0.1,
            'special_tokens': {
                'pad_token': 0,
                'unk_token': 1,
                'cls_token': 2,
                'sep_token': 3,
                'mask_token': 4
            },
            'max_sequence_length': 1024,
            'position_embeddings': np.random.randn(1024, self.current_embed_dim) * 0.1,
            
            # Multi-modal Token Parameters
            'vision_projection': np.random.randn(2048, self.current_embed_dim) * 0.1,  # Vision features to embedding
            'audio_projection': np.random.randn(512, self.current_embed_dim) * 0.1,   # Audio features to embedding
            'tabular_projection': np.random.randn(256, self.current_embed_dim) * 0.1, # Tabular features to embedding
            'modal_fusion_weights': {
                'text': 1.0,
                'vision': 1.0,
                'audio': 0.8,
                'tabular': 0.6
            },
            'modal_scales': {
                'text': np.ones(self.current_embed_dim) * 1.0,
                'vision': np.ones(self.current_embed_dim) * 0.9,
                'audio': np.ones(self.current_embed_dim) * 0.7,
                'tabular': np.ones(self.current_embed_dim) * 0.5
            }
        }
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        """Define weights for the controller node"""
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
        return self.weights
    def register_judge(self, judge: 'Judge'):
        """Register a judge with the controller"""
        self.judge_registry[judge.judge_id] = judge
        self.judge_relevance_scores[judge.judge_id] = random.uniform(0.1, 1.0)
        
        # Assign dimensional position
        self._assign_judge_position(judge.judge_id)
        
        if self.demo:
            print(f"Registered judge {judge.judge_id}")
    
    def _assign_judge_position(self, judge_id: str):
        """Assign a judge to a dimensional position"""
        num_judges = len(self.judge_registry)
        required_dims = math.ceil(num_judges / 2)
        
        # Create dimensions as needed
        current_dims = len(self.dimensions)
        for i in range(required_dims - current_dims):
            dim_id = f"dim_{self.dimension_counter}"
            self.dimensions[dim_id] = {
                'name': dim_id,
                'axis_index': self.dimension_counter,
                'positive_judge': None,
                'negative_judge': None,
                'scale': random.uniform(0.5, 2.0),
                'importance': random.uniform(0.1, 1.0)
            }
            self.dimension_counter += 1
        
        # Find available position
        judge_index = num_judges - 1  # Current judge index
        dim_index = judge_index // 2
        position_sign = 1 if judge_index % 2 == 0 else -1
        
        dim_list = list(self.dimensions.keys())
        target_dim_id = dim_list[dim_index]
        
        # Create position vector
        position_vector = np.zeros(required_dims)
        position_vector[dim_index] = position_sign * self.dimensions[target_dim_id]['scale']
        
        self.judge_positions[judge_id] = position_vector
        
        # Update dimension assignment
        if position_sign > 0:
            self.dimensions[target_dim_id]['positive_judge'] = judge_id
        else:
            self.dimensions[target_dim_id]['negative_judge'] = judge_id
    
    def tokens_to_embeddings(self, tokens: Union[List[int], List[str], np.ndarray], 
                            token_type: str = 'ids') -> np.ndarray:
        """Convert tokens to embeddings"""
        if tokens is None or len(tokens) == 0:
            return np.array([]).reshape(0, self.current_embed_dim)
        
        # Convert tokens to IDs if they're strings
        if token_type == 'strings':
            if isinstance(tokens, list) and all(isinstance(t, str) for t in tokens):
                token_ids = self._strings_to_ids(tokens)  # type: ignore
            else:
                raise TypeError("Expected a list of strings for token_type='strings'")
        else:
            if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
                token_ids = np.array(tokens)
            elif isinstance(tokens, np.ndarray):
                token_ids = tokens
            else:
                raise TypeError("Expected a list of ints or a numpy array for token_type='ids'")
        
        # Handle out-of-vocabulary tokens
        vocab_size = self.token_params['vocab_size']
        unk_token = self.token_params['special_tokens']['unk_token']
        token_ids = np.where(token_ids >= vocab_size, unk_token, token_ids)
        token_ids = np.where(token_ids < 0, unk_token, token_ids)
        
        # Get base embeddings
        embeddings = self.token_params['token_embedding_matrix'][token_ids]
        
        # Add positional embeddings
        seq_len = len(token_ids)
        max_len = min(seq_len, self.token_params['max_sequence_length'])
        pos_embeddings = self.token_params['position_embeddings'][:max_len]
        
        if seq_len <= max_len:
            embeddings[:seq_len] += pos_embeddings[:seq_len]
        else:
            for i in range(seq_len):
                pos_idx = i % self.token_params['max_sequence_length']
                embeddings[i] += self.token_params['position_embeddings'][pos_idx]
        
        return embeddings
    
    def _strings_to_ids(self, string_tokens: List[str]) -> np.ndarray:
        """Convert string tokens to IDs using hash-based mapping"""
        token_ids = []
        for token in string_tokens:
            token_hash = hash(token) % (self.token_params['vocab_size'] - 10)
            token_id = abs(token_hash) + 10
            token_ids.append(token_id)
        return np.array(token_ids)
    
    def process_multimodal_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Process multi-modal input (vision, audio, tabular, text)"""
        modal_embeddings = []
        
        for modality, data in input_data.items():
            if modality == 'text':
                # Process text tokens
                embeddings = self.tokens_to_embeddings(data, token_type='ids' if isinstance(data[0], int) else 'strings')
            elif modality == 'vision' and data is not None:
                # Project vision features
                vision_features = np.array(data).reshape(-1, 2048)  # Assume 2048-dim vision features
                embeddings = np.dot(vision_features, self.token_params['vision_projection'])
            elif modality == 'audio' and data is not None:
                # Project audio features
                audio_features = np.array(data).reshape(-1, 512)  # Assume 512-dim audio features
                embeddings = np.dot(audio_features, self.token_params['audio_projection'])
            elif modality == 'tabular' and data is not None:
                # Project tabular features
                tabular_features = np.array(data).reshape(-1, 256)  # Assume 256-dim tabular features
                embeddings = np.dot(tabular_features, self.token_params['tabular_projection'])
            else:
                continue
            
            # Apply modality-specific scaling
            scaled_embeddings = self._apply_modal_fusion(embeddings, modality)
            modal_embeddings.append(scaled_embeddings)
        
        # Concatenate all modal embeddings
        if modal_embeddings:
            combined_embeddings = np.concatenate(modal_embeddings, axis=0)
        else:
            combined_embeddings = np.random.randn(1, self.current_embed_dim) * 0.1
        
        return combined_embeddings
    
    def _apply_modal_fusion(self, embeddings: np.ndarray, modality: str) -> np.ndarray:
        """Apply modality-specific scaling and fusion"""
        if modality not in self.token_params['modal_fusion_weights']:
            return embeddings
        
        # Get modality weight and scale
        fusion_weight = self.token_params['modal_fusion_weights'][modality]
        modal_scale = self.token_params['modal_scales'][modality]
        
        # Apply scaling and weighting
        scaled_embeddings = embeddings * modal_scale * fusion_weight
        
        return scaled_embeddings
    
    def calculate_judge_probabilities(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Calculate activation probabilities for all judges"""
        if embeddings.size == 0:
            return {}
        
        probabilities = {}
        embedding_features = np.mean(embeddings, axis=0)  # Global features
        
        for judge_id, judge in self.judge_registry.items():
            # Base relevance score
            base_score = self.judge_relevance_scores[judge_id]
            
            # Dimensional influence
            position = self.judge_positions.get(judge_id, np.zeros(1))
            dimensional_influence = self._calculate_dimensional_influence(embedding_features, position)
            
            # Historical performance
            historical_factor = self._get_historical_performance(judge_id)
            
            # Combine factors
            probability = base_score * dimensional_influence * historical_factor
            probability = max(0.0, min(1.0, probability))  # Clamp to [0,1]
            
            probabilities[judge_id] = probability
        
        return probabilities
    
    def _calculate_dimensional_influence(self, features: np.ndarray, position: np.ndarray) -> float:
        """Calculate how well features align with judge's dimensional position using advanced metrics"""
        if position.size == 0 or features.size == 0:
            return 1.0
        
        # Prepare feature and position vectors
        feature_projection = features[:len(position)] if len(features) >= len(position) else features
        position_projection = position[:len(feature_projection)]
        
        # Cosine similarity
        feat_norm = np.linalg.norm(feature_projection)
        pos_norm = np.linalg.norm(position_projection)
        
        if feat_norm == 0 or pos_norm == 0:
            cosine_sim = 0.0
        else:
            cosine_sim = np.dot(feature_projection, position_projection) / (feat_norm * pos_norm)
        
        # Magnitude ratio influence
        magnitude_ratio = min(feat_norm, pos_norm) / (max(feat_norm, pos_norm) + 1e-8)
        
        # Non-linear activation using tanh
        combined_influence = cosine_sim * magnitude_ratio
        activated_influence = np.tanh(combined_influence * 2.0)  # Scale and apply tanh
        
        # Return scaled influence (centered around 1.0)
        return 1.0 + 0.3 * activated_influence
    
    def _get_historical_performance(self, judge_id: str) -> float:
        """Get historical performance factor for a judge"""
        # Simplified - in practice would track actual performance metrics
        return random.uniform(0.8, 1.2)
    
    def select_active_judges(self, probabilities: Dict[str, float]) -> List[str]:
        """Select top 50% of judges based on probabilities"""
        if not probabilities:
            return []
        
        # Sort judges by probability
        sorted_judges = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 50%
        num_active = max(1, len(sorted_judges) // 2)
        active_judge_ids = [judge_id for judge_id, _ in sorted_judges[:num_active]]
        
        # Update active set
        self.active_judges = set(active_judge_ids)
        
        return active_judge_ids
    
    def expand_embedding_dimension(self, target_dim: int):
        """Dynamically expand embedding dimensions including multi-modal projections"""
        if target_dim <= self.current_embed_dim or target_dim > self.max_embed_dim:
            return
        
        old_dim = self.current_embed_dim
        self.current_embed_dim = target_dim
        
        # Expand token embedding matrix
        old_matrix = self.token_params['token_embedding_matrix']
        new_matrix = np.random.randn(self.token_params['vocab_size'], target_dim) * 0.05
        new_matrix[:, :old_dim] = old_matrix
        self.token_params['token_embedding_matrix'] = new_matrix
        
        # Expand positional embeddings
        old_pos = self.token_params['position_embeddings']
        new_pos = np.random.randn(self.token_params['max_sequence_length'], target_dim) * 0.05
        new_pos[:, :old_dim] = old_pos
        self.token_params['position_embeddings'] = new_pos
        
        # Expand multi-modal projection matrices
        for modal_type in ['vision_projection', 'audio_projection', 'tabular_projection']:
            if modal_type in self.token_params:
                old_proj = self.token_params[modal_type]
                input_dim = old_proj.shape[0]
                new_proj = np.random.randn(input_dim, target_dim) * 0.05
                new_proj[:, :old_dim] = old_proj
                self.token_params[modal_type] = new_proj
        
        # Expand modal scales
        for modal_type in self.token_params['modal_scales']:
            old_scale = self.token_params['modal_scales'][modal_type]
            new_scale = np.ones(target_dim)
            new_scale[:old_dim] = old_scale
            self.token_params['modal_scales'][modal_type] = new_scale
        
        # Update embedding dimension
        self.token_params['embed_dim'] = target_dim
        
        if self.demo:
            print(f"Controller expanded embeddings from {old_dim}D to {target_dim}D")
    
    def process(self, input_data: Any, input_type: str = 'auto', modality: str = 'text') -> Tuple[np.ndarray, Dict[str, float], List[str]]:
        """
        Main processing method with multi-modal support
        Returns: (embeddings, judge_probabilities, active_judge_ids)
        """
        # Handle multi-modal input
        if isinstance(input_data, dict) and any(key in ['text', 'vision', 'audio', 'tabular'] for key in input_data.keys()):
            embeddings = self.process_multimodal_input(input_data)
        else:
            # Convert input to embeddings
            if input_type == 'auto':
                input_type = self._detect_input_type(input_data)
            
            if input_type in ['tokens', 'token_ids'] and not isinstance(input_data, dict):
                embeddings = self.tokens_to_embeddings(input_data, token_type='ids')
            elif input_type == 'strings' and not isinstance(input_data, dict):
                embeddings = self.tokens_to_embeddings(input_data, token_type='strings')
            elif input_type == 'embeddings':
                embeddings = np.array(input_data) if not isinstance(input_data, np.ndarray) else input_data
            else:
                embeddings = np.random.randn(1, self.current_embed_dim) * 0.1
            
            # Apply modality-specific fusion if specified
            if modality != 'text':
                embeddings = self._apply_modal_fusion(embeddings, modality)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Calculate judge probabilities
        probabilities = self.calculate_judge_probabilities(embeddings)
        
        # Select active judges
        active_judge_ids = self.select_active_judges(probabilities)
        
        # Store processing history
        self.processing_history.append({
            'embeddings': embeddings,
            'probabilities': probabilities,
            'active_judges': active_judge_ids.copy(),
            'modality': modality,
            'timestamp': len(self.processing_history)
        })
        
        if self.demo:
            print(f"Controller processed {input_type} ({modality}) -> {len(embeddings)} embeddings")
            print(f"Activated {len(active_judge_ids)} judges: {active_judge_ids}")
        
        return embeddings, probabilities, active_judge_ids
    
    def _detect_input_type(self, input_data: Any) -> str:
        """Auto-detect input type"""
        if isinstance(input_data, (list, np.ndarray)):
            if len(input_data) == 0:
                return 'embeddings'
            first_elem = input_data[0]
            if isinstance(first_elem, str):
                return 'strings'
            elif isinstance(first_elem, (int, np.integer)) and first_elem >= 0:
                return 'token_ids'
            else:
                return 'embeddings'
        elif isinstance(input_data, str):
            return 'strings'
        elif isinstance(input_data, (int, np.integer)):
            return 'token_ids'
        else:
            return 'embeddings'
    
    def update_judge_relevance(self, judge_id: str, performance_score: float):
        """Update judge relevance based on performance"""
        if judge_id in self.judge_relevance_scores:
            current_score = self.judge_relevance_scores[judge_id]
            # Exponential moving average
            self.judge_relevance_scores[judge_id] = (
                0.9 * current_score + 0.1 * performance_score
            )
    
    def compute_gradients(self, loss: float, judge_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute gradients for judge relevance scores"""
        gradients = {}
        
        for judge_id, output in judge_outputs.items():
            if judge_id in self.judge_relevance_scores:
                # Simple gradient computation (in practice would use backprop)
                grad = loss * np.mean(output) * self.learning_rate
                gradients[judge_id] = np.array([grad])
                
                # Apply gradient clipping
                if np.abs(grad) > self.gradient_clip_value:
                    gradients[judge_id] = np.sign(grad) * self.gradient_clip_value
        
        return gradients
    
    def apply_gradients(self, gradients: Dict[str, np.ndarray]):
        """Apply gradients with momentum to judge relevance scores"""
        for judge_id, grad in gradients.items():
            if judge_id in self.judge_relevance_scores:
                # Initialize momentum if not exists
                if judge_id not in self.gradient_accumulation:
                    self.gradient_accumulation[judge_id] = np.zeros_like(grad)
                
                # Apply momentum
                self.gradient_accumulation[judge_id] = (
                    self.gradient_momentum * self.gradient_accumulation[judge_id] + 
                    (1 - self.gradient_momentum) * grad
                )
                
                # Update relevance score
                self.judge_relevance_scores[judge_id] -= float(self.gradient_accumulation[judge_id][0])
                
                # Clamp to valid range
                self.judge_relevance_scores[judge_id] = max(0.01, min(1.0, self.judge_relevance_scores[judge_id]))
    
    def get_dimensional_info(self) -> Dict[str, Any]:
        """Get information about dimensional space"""
        return {
            'num_dimensions': len(self.dimensions),
            'num_judges': len(self.judge_registry),
            'active_judges': len(self.active_judges),
            'embed_dim': self.current_embed_dim,
            'dimensions': self.dimensions.copy()
        }
    

class Judge:
    """Dynamic Judge node with dimensional awareness"""
    
    def __init__(self, node_id: str, node_position: Tuple[float, float, float], demo: bool = False):
        self.judge_id = node_id
        self.node_id = node_id  # Add for consistency with other node types
        self.node_position = node_position
        self.demo = demo
        self.node_type = "Judge"
        
        # Dynamic configuration
        self.base_embed_dim = 128
        self.current_embed_dim = 128
        self.max_embed_dim = 512
        
        # Controller reference (set when registered)
        self.controller = None
        self.dimensional_position = None
        
        # Initialize attention components
        self.attention_config = {
            'num_heads': 2,
            'head_dim': self.base_embed_dim // 2,
            'max_heads': 8,
            'embed_dim': self.current_embed_dim
        }
        
        self._init_attention_weights()
        
        # Specialization parameters
        self.specialization_strength = random.uniform(0.1, 1.0)
        self.adaptation_rate = random.uniform(0.01, 0.1)
        self.dimension_sensitivity = random.uniform(0.5, 2.0)
        
        # Processing state
        self.processing_history = deque(maxlen=10)
        self.activation_level = 0.0
        self.is_active = False
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.gradient_history = {}
        
        # Transformation matrices for dimensional influence
        self._init_transformation_matrices()
        
        if demo:
            print(f"Judge {self.judge_id} initialized at {node_position}")
    
    def _init_attention_weights(self):
        """Initialize attention weight matrices"""
        config = self.attention_config
        embed_dim = config['embed_dim']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        self.attention_weights = {
            'query_weights': np.random.randn(num_heads, embed_dim, head_dim) * 0.1,
            'key_weights': np.random.randn(num_heads, embed_dim, head_dim) * 0.1,
            'value_weights': np.random.randn(num_heads, embed_dim, head_dim) * 0.1,
            'output_projection': np.random.randn(embed_dim, embed_dim) * 0.1,
            'layer_norm_gamma': np.ones(embed_dim),
            'layer_norm_beta': np.zeros(embed_dim)
        }
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        """Define node weights for the judge"""
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
    def _init_transformation_matrices(self):
        """Initialize transformation matrices for dimensional influence"""
        embed_dim = self.current_embed_dim
        
        # Rotation-like transformation matrices for different modalities
        self.transformation_matrices = {
            'text': self._create_rotation_matrix(embed_dim, angle=0.0),
            'vision': self._create_rotation_matrix(embed_dim, angle=np.pi/4),
            'audio': self._create_rotation_matrix(embed_dim, angle=np.pi/2),
            'tabular': self._create_rotation_matrix(embed_dim, angle=3*np.pi/4),
            'multimodal': self._create_rotation_matrix(embed_dim, angle=np.pi/6)
        }
        
        # Amplitude modulation matrices for specialization
        self.amplitude_matrices = {
            'text': np.diag(np.random.uniform(0.8, 1.2, embed_dim)),
            'vision': np.diag(np.random.uniform(0.7, 1.3, embed_dim)),
            'audio': np.diag(np.random.uniform(0.6, 1.4, embed_dim)),
            'tabular': np.diag(np.random.uniform(0.5, 1.5, embed_dim)),
            'multimodal': np.diag(np.random.uniform(0.9, 1.1, embed_dim))
        }
        
        # Universal specialization vectors
        self.specialization_vectors = {
            'text': np.random.randn(embed_dim) * 0.1,
            'vision': np.random.randn(embed_dim) * 0.15,
            'audio': np.random.randn(embed_dim) * 0.12,
            'tabular': np.random.randn(embed_dim) * 0.08,
            'multimodal': np.random.randn(embed_dim) * 0.05
        }
    
    def _create_rotation_matrix(self, dim: int, angle: float) -> np.ndarray:
        """Create a rotation-like transformation matrix"""
        # For higher dimensions, create block rotation matrices
        matrix = np.eye(dim)
        
        # Apply 2D rotations to pairs of dimensions
        for i in range(0, dim - 1, 2):
            cos_a = math.cos(angle + i * 0.1)  # Vary angle slightly for each pair
            sin_a = math.sin(angle + i * 0.1)
            
            # 2D rotation submatrix
            matrix[i, i] = cos_a
            matrix[i, i+1] = -sin_a
            matrix[i+1, i] = sin_a
            matrix[i+1, i+1] = cos_a
        
        return matrix
    
    def set_controller(self, controller: Controller):
        """Set the controller reference and get dimensional position"""
        self.controller = controller
        if self.judge_id in controller.judge_positions:
            self.dimensional_position = controller.judge_positions[self.judge_id]
    
    def expand_dimensions(self, target_dim: int):
        """Dynamically expand to match controller's embedding dimension"""
        if target_dim <= self.current_embed_dim or target_dim > self.max_embed_dim:
            return
        
        old_dim = self.current_embed_dim
        self.current_embed_dim = target_dim
        
        # Expand attention matrices
        self._expand_attention_matrices(old_dim, target_dim)
        
        # Update configuration
        self.attention_config['embed_dim'] = target_dim
        self.attention_config['head_dim'] = target_dim // self.attention_config['num_heads']
        
        # Expand transformation matrices
        self._expand_transformation_matrices(old_dim, target_dim)
        
        if self.demo:
            print(f"Judge {self.judge_id} expanded from {old_dim}D to {target_dim}D")
    
    def _expand_attention_matrices(self, old_dim: int, new_dim: int):
        """Expand attention matrices to new dimensions"""
        for key, matrix in self.attention_weights.items():
            if key in ['query_weights', 'key_weights', 'value_weights']:
                old_shape = matrix.shape
                new_head_dim = new_dim // self.attention_config['num_heads']
                new_matrix = np.random.randn(old_shape[0], new_dim, new_head_dim) * 0.05
                
                # Copy existing weights
                copy_embed_dim = min(old_shape[1], new_dim)
                copy_head_dim = min(old_shape[2], new_head_dim)
                new_matrix[:, :copy_embed_dim, :copy_head_dim] = matrix[:, :copy_embed_dim, :copy_head_dim]
                
                self.attention_weights[key] = new_matrix
                
            elif key == 'output_projection':
                new_matrix = np.random.randn(new_dim, new_dim) * 0.05
                copy_dim = min(old_dim, new_dim)
                new_matrix[:copy_dim, :copy_dim] = matrix[:copy_dim, :copy_dim]
                self.attention_weights[key] = new_matrix
                
            elif key in ['layer_norm_gamma', 'layer_norm_beta']:
                new_vector = np.ones(new_dim) if 'gamma' in key else np.zeros(new_dim)
                copy_dim = min(old_dim, new_dim)
                new_vector[:copy_dim] = matrix[:copy_dim]
                self.attention_weights[key] = new_vector
    
    def _expand_transformation_matrices(self, old_dim: int, new_dim: int):
        """Expand transformation matrices to new dimensions"""
        for modality in self.transformation_matrices:
            # Expand rotation matrices
            old_rot = self.transformation_matrices[modality]
            new_rot = np.eye(new_dim)
            copy_dim = min(old_dim, new_dim)
            new_rot[:copy_dim, :copy_dim] = old_rot[:copy_dim, :copy_dim]
            self.transformation_matrices[modality] = new_rot
            
            # Expand amplitude matrices
            old_amp = self.amplitude_matrices[modality]
            new_amp = np.eye(new_dim)
            new_amp[:copy_dim, :copy_dim] = old_amp[:copy_dim, :copy_dim]
            self.amplitude_matrices[modality] = new_amp
            
            # Expand specialization vectors
            old_spec = self.specialization_vectors[modality]
            new_spec = np.zeros(new_dim)
            new_spec[:copy_dim] = old_spec[:copy_dim]
            self.specialization_vectors[modality] = new_spec
    
    def calculate_dimensional_influence(self, embeddings: np.ndarray, modality: str = 'text') -> np.ndarray:
        """Apply dimensional positioning influence using transformation matrices and amplitude modulation"""
        if embeddings.size == 0 or self.dimensional_position is None:
            return embeddings
        
        # Ensure modality exists in transformation matrices
        if modality not in self.transformation_matrices:
            modality = 'multimodal'
        
        # Get transformation components
        rotation_matrix = self.transformation_matrices[modality]
        amplitude_matrix = self.amplitude_matrices[modality]
        specialization_vector = self.specialization_vectors[modality]
        
        # Apply dimensional position as rotation angle modifier
        position_strength = np.linalg.norm(self.dimensional_position)
        position_angle_modifier = np.tanh(position_strength) * 0.5  # Bounded influence
        
        # Create position-influenced rotation matrix
        dynamic_rotation = self._create_position_influenced_rotation(
            rotation_matrix, self.dimensional_position, position_angle_modifier
        )
        
        # Process embeddings
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        transformed_embeddings = []
        for i, embedding in enumerate(embeddings):
            # Adjust embedding dimension if needed
            if len(embedding) != self.current_embed_dim:
                embedding = self._adjust_embedding_vector(embedding)
            
            # Step 1: Apply rotation-like transformation
            rotated = np.dot(dynamic_rotation, embedding)
            
            # Step 2: Apply amplitude modulation based on dimensional position
            amplitude_weights = self._calculate_amplitude_weights(embedding, modality)
            modulated = np.dot(amplitude_matrix * amplitude_weights, rotated)
            
            # Step 3: Add specialization bias
            specialized = modulated + specialization_vector * self.specialization_strength
            
            # Step 4: Apply dimension sensitivity scaling
            dimension_scaling = self._calculate_dimension_scaling(embedding, modality)
            final_embedding = specialized * dimension_scaling
            
            transformed_embeddings.append(final_embedding)
        
        result = np.array(transformed_embeddings)
        
        # Apply global dimensional influence
        global_influence = self._calculate_global_dimensional_influence(result, modality)
        
        return result * global_influence
    
    def _create_position_influenced_rotation(self, base_rotation: np.ndarray, 
                                           position: np.ndarray, angle_modifier: float) -> np.ndarray:
        """Create rotation matrix influenced by dimensional position"""
        dim = base_rotation.shape[0]
        influenced_rotation = base_rotation.copy()
        
        # Apply position influence to rotation angles
        for i, pos_val in enumerate(position[:min(len(position), dim//2)]):
            if abs(pos_val) > 0.1:
                # Modify rotation for this dimension pair
                angle_shift = pos_val * angle_modifier * 0.1
                cos_shift = math.cos(angle_shift)
                sin_shift = math.sin(angle_shift)
                
                # Apply rotation modification
                idx1, idx2 = i*2, i*2+1
                if idx2 < dim:
                    influenced_rotation[idx1, idx1] *= cos_shift
                    influenced_rotation[idx1, idx2] *= -sin_shift
                    influenced_rotation[idx2, idx1] *= sin_shift
                    influenced_rotation[idx2, idx2] *= cos_shift
        
        return influenced_rotation
    
    def _calculate_amplitude_weights(self, embedding: np.ndarray, modality: str) -> np.ndarray:
        """Calculate amplitude weights based on embedding characteristics"""
        # Calculate statistical properties of embedding
        mean_val = np.mean(embedding)
        std_val = np.std(embedding) + 1e-8
        energy = np.sum(embedding ** 2)
        
        # Modality-specific amplitude scaling
        modality_scales = {
            'text': 1.0,
            'vision': 1.2,
            'audio': 0.9,
            'tabular': 0.8,
            'multimodal': 1.1
        }
        
        base_scale = modality_scales.get(modality, 1.0)
        
        # Create adaptive amplitude weights
        amplitude_weights = np.ones(len(embedding))
        
        # Energy-based modulation
        energy_factor = np.tanh(energy * 0.1) * 0.3 + 0.7
        
        # Statistical diversity modulation
        diversity_factor = np.tanh(std_val) * 0.2 + 0.8
        
        # Position-based amplitude variation
        if self.dimensional_position is not None:
            for i, pos_val in enumerate(self.dimensional_position[:len(embedding)]):
                amplitude_weights[i] *= (1.0 + pos_val * 0.1)
        
        return amplitude_weights * base_scale * energy_factor * diversity_factor
    
    def _calculate_dimension_scaling(self, embedding: np.ndarray, modality: str) -> float:
        """Calculate scaling factor based on dimensional sensitivity"""
        if self.dimensional_position is None:
            return 1.0
        
        # Calculate alignment between embedding and dimensional position
        embed_norm = np.linalg.norm(embedding) + 1e-8
        pos_norm = np.linalg.norm(self.dimensional_position) + 1e-8
        
        # Cosine similarity for alignment
        min_len = min(len(embedding), len(self.dimensional_position))
        alignment = np.dot(embedding[:min_len], self.dimensional_position[:min_len]) / (embed_norm * pos_norm)
        
        # Modality-specific sensitivity
        modality_sensitivity = {
            'text': 1.0,
            'vision': 1.3,
            'audio': 0.8,
            'tabular': 0.6,
            'multimodal': 1.1
        }
        
        sensitivity = modality_sensitivity.get(modality, 1.0) * self.dimension_sensitivity
        
        # Calculate final scaling
        scaling = 1.0 + alignment * sensitivity * 0.2
        
        return max(0.1, min(2.0, scaling))  # Clamp to reasonable range
    
    def _calculate_global_dimensional_influence(self, embeddings: np.ndarray, modality: str) -> float:
        """Calculate global influence factor for the entire embedding set"""
        if self.dimensional_position is None or embeddings.size == 0:
            return 1.0
        
        # Calculate global statistics
        global_mean = np.mean(embeddings)
        global_energy = np.mean(np.sum(embeddings ** 2, axis=1))
        
        # Position influence on global scaling
        position_strength = np.linalg.norm(self.dimensional_position)
        position_influence = np.tanh(position_strength * 0.5)
        
        # Modality-specific global influence
        modality_global_influence = {
            'text': 1.0,
            'vision': 1.1,
            'audio': 0.95,
            'tabular': 0.85,
            'multimodal': 1.05
        }
        
        base_influence = modality_global_influence.get(modality, 1.0)
        
        # Combine factors
        global_influence = base_influence * (1.0 + position_influence * 0.15)
        
        return max(0.5, min(1.5, global_influence))  # Clamp to reasonable range
    
    def _adjust_embedding_vector(self, embedding: np.ndarray) -> np.ndarray:
        """Adjust single embedding vector to match current dimension"""
        current_dim = len(embedding)
        target_dim = self.current_embed_dim
        
        if current_dim == target_dim:
            return embedding
        elif current_dim < target_dim:
            # Pad with zeros
            padding = np.zeros(target_dim - current_dim)
            return np.concatenate([embedding, padding])
        else:
            # Truncate
            return embedding[:target_dim]
    
    def multi_head_attention(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform multi-head attention with current configuration"""
        if embeddings.size == 0:
            return embeddings
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Adjust embedding dimension if needed
        if embeddings.shape[-1] != self.current_embed_dim:
            embeddings = self._adjust_embedding_dimension(embeddings)
        
        seq_len = embeddings.shape[0]
        num_heads = self.attention_config['num_heads']
        head_dim = self.attention_config['head_dim']
        
        attention_outputs = []
        
        for head in range(num_heads):
            # Generate Q, K, V
            Q = np.dot(embeddings, self.attention_weights['query_weights'][head])
            K = np.dot(embeddings, self.attention_weights['key_weights'][head])
            V = np.dot(embeddings, self.attention_weights['value_weights'][head])
            
            # Attention scores
            scores = np.dot(Q, K.T) / math.sqrt(head_dim)
            attention_probs = self._softmax(scores)
            
            # Apply attention
            head_output = np.dot(attention_probs, V)
            attention_outputs.append(head_output)
        
        # Concatenate and project
        concatenated = np.concatenate(attention_outputs, axis=-1)
        output = np.dot(concatenated, self.attention_weights['output_projection'])
        
        # Layer normalization
        return self._layer_norm(output)
    
    def _adjust_embedding_dimension(self, embeddings: np.ndarray) -> np.ndarray:
        """Adjust embedding dimension to match current configuration"""
        current_dim = embeddings.shape[-1]
        target_dim = self.current_embed_dim
        
        if current_dim == target_dim:
            return embeddings
        elif current_dim < target_dim:
            # Pad with zeros
            padding = np.zeros(embeddings.shape[:-1] + (target_dim - current_dim,))
            return np.concatenate([embeddings, padding], axis=-1)
        else:
            # Truncate
            return embeddings[..., :target_dim]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        gamma = self.attention_weights['layer_norm_gamma']
        beta = self.attention_weights['layer_norm_beta']
        
        normalized = (x - mean) / np.sqrt(var + 1e-6)
        return gamma * normalized + beta
    
    def process(self, embeddings: np.ndarray, activation_probability: float = 1.0, modality: str = 'text') -> Tuple[np.ndarray, np.ndarray]:
        """
        Process embeddings with dimensional awareness and modality-specific transformations
        Returns: (processed_embeddings, attention_scores)
        """
        if embeddings.size == 0:
            return np.array([]), np.array([])
        
        # Set activation state
        self.is_active = activation_probability > 0.5
        self.activation_level = activation_probability
        
        # Apply dimensional influence with modality-specific transformations
        dimensional_embeddings = self.calculate_dimensional_influence(embeddings, modality)
        
        # Multi-head attention
        attention_output = self.multi_head_attention(dimensional_embeddings)
        
        # Calculate attention scores
        attention_scores = np.linalg.norm(attention_output, axis=-1)
        attention_scores = attention_scores / (np.max(attention_scores) + 1e-8)
        
        # Apply activation level
        final_embeddings = attention_output * activation_probability
        final_scores = attention_scores * activation_probability
        
        # Store processing history
        self.processing_history.append({
            'embeddings': final_embeddings,
            'scores': final_scores,
            'activation': activation_probability,
            'modality': modality,
            'timestamp': len(self.processing_history)
        })
        
        if self.demo and self.is_active:
            print(f"Judge {self.judge_id} processed {len(final_embeddings)} embeddings (activation: {activation_probability:.3f}, modality: {modality})")
        
        return final_embeddings, final_scores
    
    def get_specialization_info(self) -> Dict[str, Any]:
        """Get information about judge's specialization"""
        return {
            'judge_id': self.judge_id,
            'position': self.node_position,
            'dimensional_position': self.dimensional_position.tolist() if self.dimensional_position is not None else None,
            'embed_dim': self.current_embed_dim,
            'num_heads': self.attention_config['num_heads'],
            'is_active': self.is_active,
            'activation_level': self.activation_level,
            'specialization_strength': self.specialization_strength
        }
    
class Splitter:
    """Splitter node that receives judge outputs and distributes to computational nodes with multithreading"""
    
    def __init__(self, node_id: str, node_position: Tuple[float, float, float], num_branches: int = 1, demo: bool = False,
                 max_concurrent_signals: int = 8, thread_pool_size: int = 16):
        self.node_id = node_id
        self.node_position = node_position
        self.num_branches = num_branches
        self.demo = demo
        self.node_type = "Splitter"
        self.branches = [f"Branch_{i}" for i in range(num_branches)]
        self.processing_history = deque(maxlen=10)
        
        # Multithreading configuration
        self.max_concurrent_signals = max_concurrent_signals
        self.thread_pool_size = thread_pool_size
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_pool_size, 
            thread_name_prefix=f"Splitter_{node_id}"
        )
        
        # Thread-safe locks for shared resources
        self.signal_registry_lock = RLock()
        self.computational_nodes_lock = RLock()
        self.performance_tracking_lock = RLock()
        self.spatial_index_lock = RLock()
        
        # Signal processing queues and futures
        self.active_signal_futures = {}  # signal_id -> Future
        self.signal_processing_queue = deque()  # Queue for pending signals
        self.concurrent_signal_count = 0
        self.concurrent_signal_lock = Lock()
        
        # Computational node management
        self.computational_nodes = {}  # node_id -> ComputationalNode
        self.node_positions = {}  # node_id -> 3D position
        self.node_connections = {}  # node_id -> list of connected node_ids
        self.node_reuse_count = {}  # node_id -> usage count
        self.max_computational_nodes = 1000
        self.active_nodes_percentage = 0.01  # Top 1% of nodes
        
        # Signal tracking system
        self.signal_registry = {}  # signal_id -> SignalTrace
        self.active_signals = set()
        self.completed_signals = deque(maxlen=100)
        self.signal_counter = 0
        
        # Thread-safe signal counter
        self.signal_counter_lock = Lock()
        
        # Embedding and attention processing
        self.current_embed_dim = 128
        self.max_embed_dim = 512
        self.attention_aggregation_weights = np.random.uniform(0.8, 1.2, size=self.num_branches)
        
        # Multi-modal processing capabilities (must be defined before distribution matrices)
        self.modality_weights = {
            'text': 1.0,
            'vision': 1.2,
            'audio': 0.9,
            'tabular': 0.8,
            'multimodal': 1.1
        }
        
        # Distribution matrices for computational nodes
        self.distribution_matrices = self._init_distribution_matrices()
        
        # Spatial indexing for efficient node selection
        self.spatial_index = {}  # spatial_hash -> list of node_ids
        self.spatial_resolution = 0.1  # Resolution for spatial hashing
        
        # Performance tracking
        self.node_performance_scores = {}  # node_id -> performance_score
        self.branch_performance = {}  # branch_id -> performance_metrics
        
        # Performance monitoring for threading
        self.thread_performance_metrics = {
            'total_signals_processed': 0,
            'avg_processing_time': 0.0,
            'concurrent_peak': 0,
            'thread_efficiency': 1.0,
            'queue_wait_times': deque(maxlen=100)
        }
        
        if demo:
            print(f"Splitter {node_id} initialized with {thread_pool_size} threads - ready for node connections")
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
        return self.weights
        
    def connect_computational_node(self, comp_node: 'Computational'):
        """Connect a computational node to this splitter"""
        node_id = comp_node.node_id
        position = comp_node.position
        
        self.computational_nodes[node_id] = comp_node
        self.node_positions[node_id] = position
        self.node_performance_scores[node_id] = random.uniform(0.5, 1.0)
        self.node_reuse_count[node_id] = 0
        
        # Add to spatial index
        spatial_hash = self._get_spatial_hash(position)
        if spatial_hash not in self.spatial_index:
            self.spatial_index[spatial_hash] = []
        self.spatial_index[spatial_hash].append(node_id)
        
        # Set splitter reference in the computational node
        comp_node.splitter_id = self.node_id
        
        if self.demo:
            print(f"Connected computational node {node_id} to splitter {self.node_id}")
    
    def connect_multiple_nodes(self, comp_nodes: List['Computational']):
        """Connect multiple computational nodes at once"""
        for comp_node in comp_nodes:
            self.connect_computational_node(comp_node)
        
        # Create connections between nodes after all are added
        self._create_node_connections()
        
        if self.demo:
            print(f"Connected {len(comp_nodes)} computational nodes to splitter {self.node_id}")
    
    def disconnect_computational_node(self, node_id: str):
        """Disconnect a computational node from this splitter"""
        if node_id in self.computational_nodes:
            position = self.node_positions[node_id]
            spatial_hash = self._get_spatial_hash(position)
            
            # Remove from spatial index
            if spatial_hash in self.spatial_index:
                if node_id in self.spatial_index[spatial_hash]:
                    self.spatial_index[spatial_hash].remove(node_id)
                if not self.spatial_index[spatial_hash]:
                    del self.spatial_index[spatial_hash]
            
            # Remove from all tracking structures
            del self.computational_nodes[node_id]
            del self.node_positions[node_id]
            del self.node_performance_scores[node_id]
            del self.node_reuse_count[node_id]
            
            # Remove from connections
            if node_id in self.node_connections:
                del self.node_connections[node_id]
            
            # Remove references to this node from other nodes' connections
            for other_node_id, connections in self.node_connections.items():
                if node_id in connections:
                    connections.remove(node_id)
            
            if self.demo:
                print(f"Disconnected computational node {node_id} from splitter {self.node_id}")
    
    def _init_distribution_matrices(self) -> Dict[str, np.ndarray]:
        """Initialize distribution matrices for different modalities"""
        matrices = {}
        for modality in ['text', 'vision', 'audio', 'tabular', 'multimodal']:
            # Create distribution matrix for this modality
            matrix = np.random.randn(self.num_branches, self.current_embed_dim) * 0.1
            # Apply modality-specific scaling
            matrix *= self.modality_weights.get(modality, 1.0)
            matrices[modality] = matrix
        return matrices
    
    def _get_spatial_hash(self, position: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Get spatial hash for position-based indexing"""
        x, y, z = position
        hash_x = int(x / self.spatial_resolution)
        hash_y = int(y / self.spatial_resolution)
        hash_z = int(z / self.spatial_resolution)
        return (hash_x, hash_y, hash_z)
    
    def _create_node_connections(self):
        """Create connections between computational nodes based on proximity"""
        connection_radius = 1.0  # Radius for node connections
        
        for node_id, position in self.node_positions.items():
            connections = []
            x, y, z = position
            
            # Check nearby spatial regions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        check_hash = (
                            int(x / self.spatial_resolution) + dx,
                            int(y / self.spatial_resolution) + dy,
                            int(z / self.spatial_resolution) + dz
                        )
                        
                        if check_hash in self.spatial_index:
                            for other_node_id in self.spatial_index[check_hash]:
                                if other_node_id != node_id:
                                    other_pos = self.node_positions[other_node_id]
                                    distance = self._calculate_distance(position, other_pos)
                                    
                                    if distance <= connection_radius:
                                        connections.append(other_node_id)
            
            self.node_connections[node_id] = connections
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def create_signal_trace(self, judge_id: str, embeddings: np.ndarray, attention_scores: np.ndarray, 
                           modality: str = 'text') -> str:
        """Create a new signal trace for tracking through the network (thread-safe)"""
        with self.signal_counter_lock:
            signal_id = f"{self.node_id}_signal_{self.signal_counter}"
            self.signal_counter += 1
        
        signal_trace = SignalTrace(
            signal_id=signal_id,
            source_judge_id=judge_id,
            splitter_id=self.node_id,
            embeddings=embeddings.copy(),
            attention_scores=attention_scores.copy(),
            modality=modality,
            timestamp=int(time.time() * 1000)  # Convert to milliseconds integer
        )
        
        with self.signal_registry_lock:
            self.signal_registry[signal_id] = signal_trace
            self.active_signals.add(signal_id)
        
        if self.demo:
            print(f"Created signal trace {signal_id} from judge {judge_id}")
        
        return signal_id
    
    def process_signal_async(self, judge_id: str, embeddings: np.ndarray, attention_scores: np.ndarray,
                           modality: str = 'text') -> str:
        """Asynchronously process a signal in a separate thread"""
        # Check if we can process immediately or need to queue
        with self.concurrent_signal_lock:
            if self.concurrent_signal_count < self.max_concurrent_signals:
                self.concurrent_signal_count += 1
                can_process_immediately = True
            else:
                can_process_immediately = False
        
        # Create signal trace
        signal_id = self.create_signal_trace(judge_id, embeddings, attention_scores, modality)
        
        if can_process_immediately:
            # Submit to thread pool for immediate processing
            future = self.thread_pool.submit(
                self._process_single_signal_threaded, 
                signal_id, judge_id, embeddings, attention_scores, modality
            )
            self.active_signal_futures[signal_id] = future
            
            if self.demo:
                print(f"Signal {signal_id} submitted for immediate processing")
        else:
            # Add to queue for later processing
            queue_entry = {
                'signal_id': signal_id,
                'judge_id': judge_id,
                'embeddings': embeddings.copy(),
                'attention_scores': attention_scores.copy(),
                'modality': modality,
                'queued_time': time.time()
            }
            self.signal_processing_queue.append(queue_entry)
            
            if self.demo:
                print(f"Signal {signal_id} queued (concurrent limit reached)")
        
        return signal_id
    
    def _process_single_signal_threaded(self, signal_id: str, judge_id: str, 
                                      embeddings: np.ndarray, attention_scores: np.ndarray, 
                                      modality: str) -> Dict[str, Any]:
        """Process a single signal in a dedicated thread"""
        start_time = time.time()
        
        try:
            if self.demo:
                print(f"Thread {threading.current_thread().name} processing signal {signal_id}")
            
            # Select computational nodes (thread-safe)
            selected_nodes = self._select_computational_nodes_threadsafe(
                embeddings, attention_scores, modality
            )
            
            # Distribute to nodes
            distribution_results = self._distribute_to_nodes_threadsafe(signal_id, selected_nodes)
            
            # Aggregate results
            branch_aggregations = self._aggregate_node_results_threadsafe(signal_id)
            
            # Complete processing
            final_output = self._complete_signal_processing_threadsafe(signal_id)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_thread_performance_metrics(processing_time, len(selected_nodes))
            
            # Store in processing history (thread-safe)
            with self.signal_registry_lock:
                self.processing_history.append({
                    'signal_id': signal_id,
                    'judge_id': judge_id,
                    'nodes_used': len(selected_nodes),
                    'processing_time': processing_time,
                    'thread_name': threading.current_thread().name,
                    'timestamp': time.time()
                })
            
            if self.demo:
                print(f"Signal {signal_id} completed in {processing_time:.3f}s")
            
            return final_output
            
        except Exception as e:
            print(f"Error processing signal {signal_id} in thread: {str(e)}")
            return {}
        finally:
            # Decrement concurrent count and process queued signals
            with self.concurrent_signal_lock:
                self.concurrent_signal_count -= 1
            
            # Remove from active futures
            if signal_id in self.active_signal_futures:
                del self.active_signal_futures[signal_id]
            
            # Process next queued signal if any
            self._process_next_queued_signal()
    
    def _process_next_queued_signal(self):
        """Process the next signal in the queue if capacity allows"""
        if not self.signal_processing_queue:
            return
        
        with self.concurrent_signal_lock:
            if self.concurrent_signal_count < self.max_concurrent_signals:
                self.concurrent_signal_count += 1
                can_process = True
            else:
                can_process = False
        
        if can_process and self.signal_processing_queue:
            queue_entry = self.signal_processing_queue.popleft()
            
            # Track queue wait time
            wait_time = time.time() - queue_entry['queued_time']
            self.thread_performance_metrics['queue_wait_times'].append(wait_time)
            
            # Submit to thread pool
            future = self.thread_pool.submit(
                self._process_single_signal_threaded,
                queue_entry['signal_id'], queue_entry['judge_id'],
                queue_entry['embeddings'], queue_entry['attention_scores'],
                queue_entry['modality']
            )
            self.active_signal_futures[queue_entry['signal_id']] = future
            
            if self.demo:
                print(f"Queued signal {queue_entry['signal_id']} now processing (waited {wait_time:.3f}s)")
    
    def _select_computational_nodes_threadsafe(self, embeddings: np.ndarray, attention_scores: np.ndarray, 
                                             modality: str = 'text') -> List[str]:
        """Thread-safe version of select_computational_nodes"""
        if embeddings.size == 0:
            return []
        
        with self.computational_nodes_lock:
            # Calculate number of nodes to activate (1%)
            num_active_nodes = max(1, int(len(self.computational_nodes) * self.active_nodes_percentage))
            
            # Calculate relevance scores for each computational node
            node_relevances = {}
            
            # Get embedding centroid for comparison
            embedding_centroid = np.mean(embeddings, axis=0) if embeddings.ndim > 1 else embeddings
            attention_mean = np.mean(attention_scores) if attention_scores.size > 0 else 0.5
            
            for node_id, comp_node in self.computational_nodes.items():
                # Spatial relevance (distance to splitter)
                node_pos = self.node_positions[node_id]
                spatial_distance = self._calculate_distance(self.node_position, node_pos)
                spatial_relevance = 1.0 / (1.0 + spatial_distance)
                
                # Embedding relevance (cosine similarity)
                node_embedding = comp_node.get_representative_embedding()
                if node_embedding is not None and embedding_centroid.size > 0:
                    # Adjust dimensions if needed
                    min_dim = min(len(embedding_centroid), len(node_embedding))
                    embed_sim = np.dot(embedding_centroid[:min_dim], node_embedding[:min_dim])
                    embed_norm = (np.linalg.norm(embedding_centroid[:min_dim]) * 
                                 np.linalg.norm(node_embedding[:min_dim]))
                    embedding_relevance = embed_sim / (embed_norm + 1e-8)
                else:
                    embedding_relevance = 0.5
                
                # Attention relevance
                attention_relevance = attention_mean
                
                # Performance relevance
                performance_relevance = self.node_performance_scores[node_id]
                
                # Reuse penalty (prefer less used nodes for diversity)
                reuse_penalty = 1.0 / (1.0 + self.node_reuse_count[node_id] * 0.1)
                
                # Modality-specific weighting
                modality_weight = self.modality_weights.get(modality, 1.0)
                
                # Combine relevance factors
                total_relevance = (
                    spatial_relevance * 0.3 +
                    embedding_relevance * 0.4 +
                    attention_relevance * 0.2 +
                    performance_relevance * 0.1
                ) * reuse_penalty * modality_weight
                
                node_relevances[node_id] = total_relevance
            
            # Select top nodes
            selected_nodes = sorted(node_relevances.items(), key=lambda x: x[1], reverse=True)
            selected_node_ids = [node_id for node_id, _ in selected_nodes[:num_active_nodes]]
            
            # Update reuse counts
            for node_id in selected_node_ids:
                self.node_reuse_count[node_id] += 1
            
            return selected_node_ids
    
    def select_computational_nodes(self, embeddings: np.ndarray, attention_scores: np.ndarray, 
                                  modality: str = 'text') -> List[str]:
        """Select the closest 1% of computational nodes based on embeddings and attention"""
        if embeddings.size == 0:
            return []
        
        # Calculate number of nodes to activate (1%)
        num_active_nodes = max(1, int(len(self.computational_nodes) * self.active_nodes_percentage))
        
        # Calculate relevance scores for each computational node
        node_relevances = {}
        
        # Get embedding centroid for comparison
        embedding_centroid = np.mean(embeddings, axis=0) if embeddings.ndim > 1 else embeddings
        attention_mean = np.mean(attention_scores) if attention_scores.size > 0 else 0.5
        
        for node_id, comp_node in self.computational_nodes.items():
            # Spatial relevance (distance to splitter)
            node_pos = self.node_positions[node_id]
            spatial_distance = self._calculate_distance(self.node_position, node_pos)
            spatial_relevance = 1.0 / (1.0 + spatial_distance)
            
            # Embedding relevance (cosine similarity)
            node_embedding = comp_node.get_representative_embedding()
            if node_embedding is not None and embedding_centroid.size > 0:
                # Adjust dimensions if needed
                min_dim = min(len(embedding_centroid), len(node_embedding))
                embed_sim = np.dot(embedding_centroid[:min_dim], node_embedding[:min_dim])
                embed_norm = (np.linalg.norm(embedding_centroid[:min_dim]) * 
                             np.linalg.norm(node_embedding[:min_dim]))
                embedding_relevance = embed_sim / (embed_norm + 1e-8)
            else:
                embedding_relevance = 0.5
            
            # Attention relevance
            attention_relevance = attention_mean
            
            # Performance relevance
            performance_relevance = self.node_performance_scores[node_id]
            
            # Reuse penalty (prefer less used nodes for diversity)
            reuse_penalty = 1.0 / (1.0 + self.node_reuse_count[node_id] * 0.1)
            
            # Modality-specific weighting
            modality_weight = self.modality_weights.get(modality, 1.0)
            
            # Combine relevance factors
            total_relevance = (
                spatial_relevance * 0.3 +
                embedding_relevance * 0.4 +
                attention_relevance * 0.2 +
                performance_relevance * 0.1
            ) * reuse_penalty * modality_weight
            
            node_relevances[node_id] = total_relevance
        
        # Select top nodes
        selected_nodes = sorted(node_relevances.items(), key=lambda x: x[1], reverse=True)
        selected_node_ids = [node_id for node_id, _ in selected_nodes[:num_active_nodes]]
        
        # Update reuse counts
        for node_id in selected_node_ids:
            self.node_reuse_count[node_id] += 1
        
        return selected_node_ids
    
    def distribute_to_nodes(self, signal_id: str, selected_node_ids: List[str]) -> Dict[str, Any]:
        """Distribute signal to selected computational nodes"""
        if signal_id not in self.signal_registry:
            return {}
        
        signal_trace = self.signal_registry[signal_id]
        distribution_results = {}
        
        # Get distribution matrix for this modality
        dist_matrix = self.distribution_matrices.get(signal_trace.modality, 
                                                    self.distribution_matrices['multimodal'])
        
        for node_id in selected_node_ids:
            if node_id in self.computational_nodes:
                comp_node = self.computational_nodes[node_id]
                
                # Process signal through computational node
                node_output = comp_node.process_signal(
                    embeddings=signal_trace.embeddings,
                    attention_scores=signal_trace.attention_scores,
                    modality=signal_trace.modality,
                    distribution_matrix=dist_matrix
                )
                
                distribution_results[node_id] = node_output
                
                # Update signal trace
                signal_trace.add_node_result(node_id, node_output)
        
        # Update signal status
        signal_trace.processing_status = 'distributed'
        
        if self.demo:
            print(f"Distributed signal {signal_id} to {len(selected_node_ids)} computational nodes")
        
        return distribution_results
    
    def aggregate_node_results(self, signal_id: str) -> Dict[str, np.ndarray]:
        """Aggregate results from computational nodes for a signal"""
        if signal_id not in self.signal_registry:
            return {}
        
        signal_trace = self.signal_registry[signal_id]
        node_results = signal_trace.node_results
        
        if not node_results:
            return {}
        
        # Aggregate embeddings and scores across branches
        branch_aggregations = {}
        
        for branch_idx in range(self.num_branches):
            branch_id = f"Branch_{branch_idx}"
            branch_embeddings = []
            branch_scores = []
            
            for node_id, result in node_results.items():
                if 'branch_outputs' in result and branch_idx < len(result['branch_outputs']):
                    branch_output = result['branch_outputs'][branch_idx]
                    if 'embeddings' in branch_output:
                        branch_embeddings.append(branch_output['embeddings'])
                    if 'scores' in branch_output:
                        branch_scores.append(branch_output['scores'])
            
            if branch_embeddings:
                # Weighted aggregation
                aggregated_embeddings = np.mean(branch_embeddings, axis=0)
                aggregated_scores = np.mean(branch_scores, axis=0) if branch_scores else np.ones(len(aggregated_embeddings))
                
                # Apply branch-specific weighting
                branch_weight = self.attention_aggregation_weights[branch_idx]
                aggregated_embeddings *= branch_weight
                aggregated_scores *= branch_weight
                
                branch_aggregations[branch_id] = {
                    'embeddings': aggregated_embeddings,
                    'scores': aggregated_scores,
                    'node_count': len(branch_embeddings),
                    'modality': signal_trace.modality
                }
        
        # Update signal trace
        signal_trace.branch_aggregations = branch_aggregations
        signal_trace.processing_status = 'aggregated'
        
        return branch_aggregations
    
    def complete_signal_processing(self, signal_id: str) -> Dict[str, Any]:
        """Complete signal processing and prepare for retainer"""
        if signal_id not in self.signal_registry:
            return {}
        
        signal_trace = self.signal_registry[signal_id]
        
        # Ensure aggregation is complete
        if signal_trace.processing_status != 'aggregated':
            self.aggregate_node_results(signal_id)
        
        # Prepare final output for retainer
        final_output = {
            'signal_id': signal_id,
            'source_judge_id': signal_trace.source_judge_id,
            'splitter_id': self.node_id,
            'modality': signal_trace.modality,
            'branch_results': signal_trace.branch_aggregations,
            'processing_metadata': {
                'nodes_used': len(signal_trace.node_results),
                'branches_active': len(signal_trace.branch_aggregations),
                'processing_time': signal_trace.get_processing_time(),
                'signal_strength': signal_trace.calculate_signal_strength()
            }
        }
        
        # Mark signal as completed
        signal_trace.processing_status = 'completed'
        self.active_signals.discard(signal_id)
        self.completed_signals.append(signal_id)
        
        # Update performance metrics
        self._update_performance_metrics(signal_trace)
        
        if self.demo:
            print(f"Completed signal processing for {signal_id}")
        
        return final_output
    
    def process_judge_output(self, judge_id: str, embeddings: np.ndarray, attention_scores: np.ndarray,
                           modality: str = 'text', async_processing: bool = True) -> Dict[str, Any]:
        """Main processing method for judge outputs with optional async processing"""
        if embeddings.size == 0:
            return {}
        
        if async_processing:
            # Use asynchronous processing for better performance
            signal_id = self.process_signal_async(judge_id, embeddings, attention_scores, modality)
            
            # Return immediately with signal ID for later retrieval
            return {
                'signal_id': signal_id,
                'status': 'processing_async',
                'message': 'Signal submitted for asynchronous processing',
                'retrieval_method': 'use get_signal_result() with signal_id'
            }
        else:
            # Original synchronous processing
            # Create signal trace
            signal_id = self.create_signal_trace(judge_id, embeddings, attention_scores, modality)
            
            # Select computational nodes
            selected_nodes = self.select_computational_nodes(embeddings, attention_scores, modality)
            
            # Distribute to nodes
            distribution_results = self.distribute_to_nodes(signal_id, selected_nodes)
            
            # Aggregate results
            branch_aggregations = self.aggregate_node_results(signal_id)
            
            # Complete processing
            final_output = self.complete_signal_processing(signal_id)
            
            # Store in processing history
            self.processing_history.append({
                'signal_id': signal_id,
                'judge_id': judge_id,
                'nodes_used': len(selected_nodes),
                'final_output': final_output,
                'timestamp': len(self.processing_history)
            })
            
            return final_output
    
    def _update_performance_metrics(self, signal_trace: 'SignalTrace'):
        """Update performance metrics based on signal processing"""
        # Update node performance scores
        for node_id in signal_trace.node_results:
            if node_id in self.node_performance_scores:
                # Simple performance update (in practice would use actual metrics)
                current_score = self.node_performance_scores[node_id]
                processing_quality = signal_trace.calculate_signal_strength()
                new_score = 0.9 * current_score + 0.1 * processing_quality
                self.node_performance_scores[node_id] = max(0.1, min(1.0, new_score))
        
        # Update branch performance
        for branch_id, aggregation in signal_trace.branch_aggregations.items():
            if branch_id not in self.branch_performance:
                self.branch_performance[branch_id] = {'total_signals': 0, 'avg_quality': 0.5}
            
            branch_perf = self.branch_performance[branch_id]
            branch_perf['total_signals'] += 1
            signal_quality = np.mean(aggregation['scores']) if aggregation['scores'].size > 0 else 0.5
            branch_perf['avg_quality'] = (0.9 * branch_perf['avg_quality'] + 0.1 * signal_quality)
    
    # Thread-safe versions of signal processing methods
    def _distribute_to_nodes_threadsafe(self, signal_id: str, selected_node_ids: List[str]) -> Dict[str, Any]:
        """Thread-safe version of distribute_to_nodes"""
        with self.signal_registry_lock:
            if signal_id not in self.signal_registry:
                return {}
            signal_trace = self.signal_registry[signal_id]
        
        distribution_results = {}
        
        # Get distribution matrix for this modality
        dist_matrix = self.distribution_matrices.get(signal_trace.modality, 
                                                    self.distribution_matrices['multimodal'])
        
        with self.computational_nodes_lock:
            for node_id in selected_node_ids:
                if node_id in self.computational_nodes:
                    comp_node = self.computational_nodes[node_id]
                    
                    # Process signal through computational node
                    node_output = comp_node.process_signal(
                        embeddings=signal_trace.embeddings,
                        attention_scores=signal_trace.attention_scores,
                        modality=signal_trace.modality,
                        distribution_matrix=dist_matrix
                    )
                    
                    distribution_results[node_id] = node_output
                    
                    # Update signal trace
                    signal_trace.add_node_result(node_id, node_output)
        
        # Update signal status
        with self.signal_registry_lock:
            signal_trace.processing_status = 'distributed'
        
        if self.demo:
            print(f"Distributed signal {signal_id} to {len(selected_node_ids)} computational nodes")
        
        return distribution_results
    
    def _aggregate_node_results_threadsafe(self, signal_id: str) -> Dict[str, np.ndarray]:
        """Thread-safe version of aggregate_node_results"""
        with self.signal_registry_lock:
            if signal_id not in self.signal_registry:
                return {}
            signal_trace = self.signal_registry[signal_id]
            node_results = signal_trace.node_results
        
        if not node_results:
            return {}
        
        # Aggregate embeddings and scores across branches
        branch_aggregations = {}
        
        for branch_idx in range(self.num_branches):
            branch_id = f"Branch_{branch_idx}"
            branch_embeddings = []
            branch_scores = []
            
            for node_id, result in node_results.items():
                if 'branch_outputs' in result and branch_idx < len(result['branch_outputs']):
                    branch_output = result['branch_outputs'][branch_idx]
                    if 'embeddings' in branch_output:
                        branch_embeddings.append(branch_output['embeddings'])
                    if 'scores' in branch_output:
                        branch_scores.append(branch_output['scores'])
            
            if branch_embeddings:
                # Weighted aggregation
                aggregated_embeddings = np.mean(branch_embeddings, axis=0)
                aggregated_scores = np.mean(branch_scores, axis=0) if branch_scores else np.ones(len(aggregated_embeddings))
                
                # Apply branch-specific weighting
                branch_weight = self.attention_aggregation_weights[branch_idx]
                aggregated_embeddings *= branch_weight
                aggregated_scores *= branch_weight
                
                branch_aggregations[branch_id] = {
                    'embeddings': aggregated_embeddings,
                    'scores': aggregated_scores,
                    'node_count': len(branch_embeddings),
                    'modality': signal_trace.modality
                }
        
        # Update signal trace
        with self.signal_registry_lock:
            signal_trace.branch_aggregations = branch_aggregations
            signal_trace.processing_status = 'aggregated'
        
        return branch_aggregations
    
    def _complete_signal_processing_threadsafe(self, signal_id: str) -> Dict[str, Any]:
        """Thread-safe version of complete_signal_processing"""
        with self.signal_registry_lock:
            if signal_id not in self.signal_registry:
                return {}
            signal_trace = self.signal_registry[signal_id]
        
        # Ensure aggregation is complete
        if signal_trace.processing_status != 'aggregated':
            self._aggregate_node_results_threadsafe(signal_id)
        
        # Prepare final output for retainer
        final_output = {
            'signal_id': signal_id,
            'source_judge_id': signal_trace.source_judge_id,
            'splitter_id': self.node_id,
            'modality': signal_trace.modality,
            'branch_results': signal_trace.branch_aggregations,
            'processing_metadata': {
                'nodes_used': len(signal_trace.node_results),
                'branches_active': len(signal_trace.branch_aggregations),
                'processing_time': signal_trace.get_processing_time(),
                'signal_strength': signal_trace.calculate_signal_strength(),
                'thread_name': threading.current_thread().name
            }
        }
        
        # Mark signal as completed
        with self.signal_registry_lock:
            signal_trace.processing_status = 'completed'
            self.active_signals.discard(signal_id)
            self.completed_signals.append(signal_id)
        
        # Update performance metrics (thread-safe)
        self._update_performance_metrics_threadsafe(signal_trace)
        
        if self.demo:
            print(f"Completed signal processing for {signal_id}")
        
        return final_output
    
    def _update_performance_metrics_threadsafe(self, signal_trace: 'SignalTrace'):
        """Thread-safe version of performance metrics update"""
        with self.performance_tracking_lock:
            # Update node performance scores
            for node_id in signal_trace.node_results:
                if node_id in self.node_performance_scores:
                    # Simple performance update (in practice would use actual metrics)
                    current_score = self.node_performance_scores[node_id]
                    processing_quality = signal_trace.calculate_signal_strength()
                    new_score = 0.9 * current_score + 0.1 * processing_quality
                    self.node_performance_scores[node_id] = max(0.1, min(1.0, new_score))
            
            # Update branch performance
            for branch_id, aggregation in signal_trace.branch_aggregations.items():
                if branch_id not in self.branch_performance:
                    self.branch_performance[branch_id] = {'total_signals': 0, 'avg_quality': 0.5}
                
                branch_perf = self.branch_performance[branch_id]
                branch_perf['total_signals'] += 1
                signal_quality = np.mean(aggregation['scores']) if aggregation['scores'].size > 0 else 0.5
                branch_perf['avg_quality'] = (0.9 * branch_perf['avg_quality'] + 0.1 * signal_quality)
    
    def _update_thread_performance_metrics(self, processing_time: float, nodes_used: int):
        """Update threading-specific performance metrics"""
        with self.performance_tracking_lock:
            self.thread_performance_metrics['total_signals_processed'] += 1
            
            # Update average processing time
            current_avg = self.thread_performance_metrics['avg_processing_time']
            count = self.thread_performance_metrics['total_signals_processed']
            self.thread_performance_metrics['avg_processing_time'] = (
                (current_avg * (count - 1) + processing_time) / count
            )
            
            # Update concurrent peak
            current_concurrent = self.concurrent_signal_count
            if current_concurrent > self.thread_performance_metrics['concurrent_peak']:
                self.thread_performance_metrics['concurrent_peak'] = current_concurrent
            
            # Calculate thread efficiency (signals processed per second)
            if processing_time > 0:
                efficiency = 1.0 / processing_time
                current_efficiency = self.thread_performance_metrics['thread_efficiency']
                self.thread_performance_metrics['thread_efficiency'] = (
                    0.9 * current_efficiency + 0.1 * efficiency
                )
    
    def get_threading_status(self) -> Dict[str, Any]:
        """Get comprehensive threading status information"""
        with self.concurrent_signal_lock:
            concurrent_count = self.concurrent_signal_count
        
        return {
            'active_threads': len(self.active_signal_futures),
            'concurrent_signals': concurrent_count,
            'max_concurrent': self.max_concurrent_signals,
            'queued_signals': len(self.signal_processing_queue),
            'thread_pool_size': self.thread_pool_size,
            'performance_metrics': self.thread_performance_metrics.copy(),
            'avg_queue_wait_time': (
                np.mean(list(self.thread_performance_metrics['queue_wait_times'])) 
                if self.thread_performance_metrics['queue_wait_times'] else 0.0
            )
        }
    
    def shutdown_threading(self):
        """Gracefully shutdown the thread pool"""
        if self.demo:
            print(f"Shutting down thread pool for Splitter {self.node_id}")
        
        # Wait for all active signals to complete
        for signal_id, future in list(self.active_signal_futures.items()):
            try:
                future.result(timeout=10.0)  # 10 second timeout per signal
            except concurrent.futures.TimeoutError:
                print(f"Warning: Signal {signal_id} timed out during shutdown")
            except Exception as e:
                print(f"Error completing signal {signal_id} during shutdown: {e}")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
    
    def process_judge_output_async(self, judge_id: str, embeddings: np.ndarray, attention_scores: np.ndarray,
                                 modality: str = 'text') -> str:
        """Asynchronous version of process_judge_output that returns signal_id immediately"""
        if embeddings.size == 0:
            return ""
        
        # Use the async processing method
        signal_id = self.process_signal_async(judge_id, embeddings, attention_scores, modality)
        return signal_id
    
    def get_signal_result(self, signal_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get the result of an asynchronously processed signal"""
        if signal_id in self.active_signal_futures:
            try:
                future = self.active_signal_futures[signal_id]
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                return {'error': 'timeout', 'signal_id': signal_id}
            except Exception as e:
                return {'error': str(e), 'signal_id': signal_id}
        
        # Check if signal is completed
        with self.signal_registry_lock:
            if signal_id in self.signal_registry:
                signal_trace = self.signal_registry[signal_id]
                if signal_trace.processing_status == 'completed':
                    return {
                        'signal_id': signal_id,
                        'status': 'completed',
                        'branch_results': signal_trace.branch_aggregations,
                        'metadata': {
                            'nodes_used': len(signal_trace.node_results),
                            'processing_time': signal_trace.get_processing_time()
                        }
                    }
        
        return {'error': 'signal_not_found', 'signal_id': signal_id}
    
    def expand_dimensions(self, target_dim: int):
        """Expand embedding dimensions for all computational nodes"""
        if target_dim <= self.current_embed_dim or target_dim > self.max_embed_dim:
            return
        
        old_dim = self.current_embed_dim
        self.current_embed_dim = target_dim
        
        # Expand distribution matrices
        for modality in self.distribution_matrices:
            old_matrix = self.distribution_matrices[modality]
            new_matrix = np.random.randn(self.num_branches, target_dim) * 0.05
            new_matrix[:, :old_dim] = old_matrix[:, :old_dim]
            self.distribution_matrices[modality] = new_matrix
        
        # Expand all computational nodes
        for comp_node in self.computational_nodes.values():
            comp_node.expand_dimensions(target_dim)
        
        if self.demo:
            print(f"Splitter {self.node_id} expanded from {old_dim}D to {target_dim}D")
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive status information including threading metrics"""
        threading_status = self.get_threading_status()
        
        return {
            'node_id': self.node_id,
            'position': self.node_position,
            'num_branches': self.num_branches,
            'computational_nodes': len(self.computational_nodes),
            'active_signals': len(self.active_signals),
            'completed_signals': len(self.completed_signals),
            'embed_dim': self.current_embed_dim,
            'branch_performance': self.branch_performance.copy(),
            'avg_node_performance': np.mean(list(self.node_performance_scores.values())) if self.node_performance_scores else 0.0,
            'total_node_usage': sum(self.node_reuse_count.values()),
            'threading_info': threading_status,
            'node_type': self.node_type
        }
    
    def __del__(self):
        """Destructor to ensure proper cleanup of thread pool"""
        try:
            if hasattr(self, 'thread_pool') and self.thread_pool:
                self.shutdown_threading()
        except Exception as e:
            # Silently handle cleanup errors to avoid issues during garbage collection
            pass


class SignalTrace:
    """Tracks signal processing through the splitter network"""
    
    def __init__(self, signal_id: str, source_judge_id: str, splitter_id: str,
                 embeddings: np.ndarray, attention_scores: np.ndarray, 
                 modality: str, timestamp: int):
        self.signal_id = signal_id
        self.source_judge_id = source_judge_id
        self.splitter_id = splitter_id
        self.embeddings = embeddings
        self.attention_scores = attention_scores
        self.modality = modality
        self.creation_timestamp = timestamp
        
        # Processing tracking
        self.processing_status = 'created'  # created -> distributed -> aggregated -> completed
        self.node_results = {}  # node_id -> processing_result
        self.branch_aggregations = {}  # branch_id -> aggregated_result
        
        # Timing
        self.distribution_time = None
        self.aggregation_time = None
        self.completion_time = None
    
    def add_node_result(self, node_id: str, result: Dict[str, Any]):
        """Add processing result from a computational node"""
        self.node_results[node_id] = result
    
    def calculate_signal_strength(self) -> float:
        """Calculate overall signal strength"""
        if self.attention_scores.size == 0:
            return 0.5
        
        base_strength = np.mean(self.attention_scores)
        
        # Factor in number of nodes used
        node_factor = min(1.0, len(self.node_results) / 10.0)  # Normalize by expected ~10 nodes
        
        # Factor in processing completeness
        completeness_factor = 1.0 if self.processing_status == 'completed' else 0.7
        
        return float(base_strength * node_factor * completeness_factor)
    
    def get_processing_time(self) -> float:
        """Get total processing time"""
        if self.completion_time is not None:
            return self.completion_time - self.creation_timestamp
        return 0.0


class Computational:
    """Enhanced computational node for multidimensional AI brain architecture"""
    
    def __init__(self, node_id: str, position: Tuple[float, float, float], 
                 embed_dim: int = 128, splitter_id: str = "", max_judges: int = 128, demo: bool = False):
        self.node_id = node_id
        self.position = position
        self.node_position = position  # Add compatibility attribute for brain interface
        self.embed_dim = embed_dim
        self.splitter_id = splitter_id
        self.max_judges = max_judges
        self.demo = demo  # Add demo flag for debugging
        self.node_type = "Computational"  # Add node_type for consistency
        
        # Memory-optimized processing components based on demo mode and embed_dim
        if demo or embed_dim <= 128:
            # Lightweight configuration for demo/small segments
            self.weight_matrices = {
                'primary': np.random.randn(embed_dim, embed_dim) * 0.1,
                # Skip additional matrices for memory efficiency
            }
            
            self.bias_vectors = {
                'primary': np.random.randn(embed_dim) * 0.05,
            }
        else:
            # Full configuration for production segments
            self.weight_matrices = {
                'primary': np.random.randn(embed_dim, embed_dim) * 0.1,
                'judge_attention': np.random.randn(embed_dim, embed_dim) * 0.08,
                'cross_modal': np.random.randn(embed_dim, embed_dim) * 0.06,
                'positional': np.random.randn(embed_dim, embed_dim) * 0.05
            }
            
            self.bias_vectors = {
                'primary': np.random.randn(embed_dim) * 0.05,
                'judge_attention': np.random.randn(embed_dim) * 0.03,
                'positional': np.random.randn(embed_dim) * 0.02
            }
        
        # Dynamic dimensional system for judge positioning
        self.dimensional_axes = {}  # Will store judge positioning in N-dimensional space
        self.active_judges = set()  # Currently active judge IDs
        self.judge_affinities = {}  # Judge-specific processing preferences
        
        # Memory-optimized representative state
        if demo or embed_dim <= 128:
            # Single embedding for lightweight mode
            self.representative_embeddings = {
                'unified': np.random.randn(embed_dim) * 0.1,
            }
        else:
            # Multi-layered representative state for different abstraction levels
            self.representative_embeddings = {
                'local': np.random.randn(embed_dim) * 0.1,
                'regional': np.random.randn(embed_dim) * 0.08,
                'global': np.random.randn(embed_dim) * 0.06
            }
        
        # Enhanced counters and metrics
        self.processing_count = 0
        self.last_activation = 0.0
        self.judge_activation_history = {}  # Track which judges activated this node
        self.signal_trace_ids = []  # For signal tracing across the network
        
        # Memory-optimized modality handling
        if demo or embed_dim <= 128:
            # Simplified modality preferences for lightweight mode
            self.modality_preferences = {
                'text': {'base_affinity': random.uniform(0.8, 1.2)},
                'unified': {'base_affinity': random.uniform(0.7, 1.3)}
            }
        else:
            # Sophisticated modality handling with hierarchical preferences
            self.modality_preferences = {
                'text': {
                    'base_affinity': random.uniform(0.8, 1.2),
                    'semantic_weight': random.uniform(0.7, 1.3),
                    'syntactic_weight': random.uniform(0.6, 1.1),
                    'contextual_weight': random.uniform(0.8, 1.2)
                },
                'vision': {
                    'base_affinity': random.uniform(0.7, 1.3),
                    'spatial_weight': random.uniform(0.8, 1.4),
                    'feature_weight': random.uniform(0.6, 1.2),
                    'temporal_weight': random.uniform(0.5, 1.0)
                },
                'audio': {
                    'base_affinity': random.uniform(0.6, 1.4),
                    'spectral_weight': random.uniform(0.7, 1.3),
                    'temporal_weight': random.uniform(0.8, 1.4),
                    'harmonic_weight': random.uniform(0.5, 1.1)
                },
                'tabular': {
                    'base_affinity': random.uniform(0.5, 1.5),
                    'numerical_weight': random.uniform(0.8, 1.2),
                    'categorical_weight': random.uniform(0.6, 1.4),
                    'relational_weight': random.uniform(0.7, 1.3)
                },
                'multimodal': {
                    'base_affinity': random.uniform(0.9, 1.1),
                    'fusion_weight': random.uniform(0.8, 1.4),
                    'alignment_weight': random.uniform(0.7, 1.2),
                    'coherence_weight': random.uniform(0.9, 1.3)
                }
            }
        
        # Connection strength tracking for reuse optimization
        self.connection_strengths = {}  # Maps to neighboring nodes
        self.reuse_efficiency = 1.0
        
        # Basic connection attributes for consistency with other node types
        self.entrance_connections = []
        self.exit_connections = []
        self.generic_connections = []
        
        if demo:
            print(f"Computational node {node_id} initialized at {position} with {embed_dim}D embeddings")
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
        return self.weights

    def process_signal(self, embeddings: np.ndarray, attention_scores: np.ndarray,
                      modality: str, distribution_matrix: np.ndarray,
                      judge_masks: Optional[Dict[str, np.ndarray]] = None,
                      positional_encodings: Optional[np.ndarray] = None,
                      signal_trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced signal processing with judge guidance and multi-modal support"""
        if embeddings.size == 0:
            return {'node_id': self.node_id, 'branch_outputs': [], 'activation': 0.0}
        
        # Track signal for tracing
        if signal_trace_id:
            self.signal_trace_ids.append(signal_trace_id)
        
        # Ensure embeddings match our dimension
        if embeddings.shape[-1] != self.embed_dim:
            embeddings = self._resize_embeddings(embeddings)
        
        # Enhanced multi-stage transformation
        # Stage 1: Primary transformation with modality-specific weights
        modality_weights = self._get_modality_weights(modality)
        primary_transform = np.dot(embeddings, self.weight_matrices['primary']) + \
                          self.bias_vectors['primary'] * modality_weights['base_affinity']
        
        # Stage 2: Judge-guided attention application
        if judge_masks:
            judge_transform = self._apply_judge_guidance(primary_transform, judge_masks)
        else:
            judge_transform = primary_transform
        
        # Stage 3: Positional encoding integration
        if positional_encodings is not None:
            pos_transform = np.dot(positional_encodings, self.weight_matrices['positional'])
            final_transform = judge_transform + pos_transform * 0.3
        else:
            final_transform = judge_transform
        
        # Enhanced activation with multi-factor consideration
        base_activation = np.tanh(np.mean(final_transform * attention_scores.reshape(-1, 1)))
        judge_boost = self._calculate_judge_boost(judge_masks) if judge_masks else 1.0
        modality_factor = self._calculate_modality_factor(modality, final_transform)
        
        activation = float(base_activation * judge_boost * modality_factor)
        self.last_activation = activation
        
        # Process branches with enhanced distribution
        branch_outputs = self._process_branches_enhanced(
            final_transform, attention_scores, distribution_matrix, 
            activation, modality, judge_masks
        )
        
        # Update sophisticated state tracking
        self.processing_count += 1
        self._update_representative_embeddings(embeddings, modality)
        if judge_masks:
            self._update_judge_activation_history(judge_masks, activation)
        
        return {
            'node_id': self.node_id,
            'branch_outputs': branch_outputs,
            'activation': activation,
            'modality': modality,
            'processing_count': self.processing_count,
            'signal_trace_id': signal_trace_id,
            'judge_contributions': self._get_judge_contributions(judge_masks) if judge_masks else {},
            'dimensional_position': self._get_current_dimensional_position()
        }
    
    def _apply_judge_guidance(self, embeddings: np.ndarray, judge_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply judge-specific attention masks and transformations"""
        judge_weighted_sum = np.zeros_like(embeddings)
        total_weight = 0.0
        
        for judge_id, mask in judge_masks.items():
            if judge_id not in self.judge_affinities:
                # Initialize judge affinity for this node
                self.judge_affinities[judge_id] = random.uniform(0.5, 1.5)
            
            judge_weight = self.judge_affinities[judge_id]
            masked_embeddings = embeddings * mask.reshape(-1, 1)
            
            # Apply judge-specific transformation
            judge_transform = np.dot(masked_embeddings, self.weight_matrices['judge_attention'])
            judge_weighted_sum += judge_transform * judge_weight
            total_weight += judge_weight
            
            self.active_judges.add(judge_id)
        
        if total_weight > 0:
            return judge_weighted_sum / total_weight
        return embeddings
    
    def _process_branches_enhanced(self, transformed: np.ndarray, attention_scores: np.ndarray,
                                 distribution_matrix: np.ndarray, activation: float,
                                 modality: str, judge_masks: Optional[Dict[str, np.ndarray]] = None) -> List[Dict]:
        """Enhanced branch processing with judge-aware distribution"""
        num_branches = distribution_matrix.shape[0]
        branch_outputs = []
        
        for branch_idx in range(num_branches):
            # Enhanced branch transformation with cross-modal considerations
            base_result = np.dot(transformed.T, distribution_matrix[branch_idx].reshape(-1, 1)).flatten()
            
            # Apply cross-modal transformation if dealing with multimodal input
            if modality == 'multimodal':
                cross_modal_result = np.dot(base_result.reshape(1, -1), self.weight_matrices['cross_modal']).flatten()
                branch_result = (base_result + cross_modal_result) / 2.0
            else:
                branch_result = base_result
            
            # Enhanced scoring with judge influence
            base_scores = attention_scores * activation
            if judge_masks:
                judge_influence = np.mean([np.mean(mask) for mask in judge_masks.values()])
                branch_scores = base_scores * (1.0 + 0.3 * judge_influence)
            else:
                branch_scores = base_scores
            
            # Apply modality-specific branch modulation
            modality_boost = self._get_branch_modality_boost(modality, branch_idx)
            branch_scores *= modality_boost
            
            branch_outputs.append({
                'embeddings': branch_result,
                'scores': branch_scores,
                'activation': activation,
                'branch_index': branch_idx,
                'modality_boost': modality_boost,
                'connection_strength': self._update_connection_strength(branch_idx, activation)
            })
        
        return branch_outputs
    
    def update_dimensional_position(self, active_judge_ids: List[str]):
        """Update node's position in dynamic N-dimensional judge space"""
        num_judges = len(active_judge_ids)
        if num_judges == 0:
            return
        
        # Calculate required dimensions (num_judges / 2)
        required_dims = max(1, num_judges // 2)
        
        # Clear and rebuild dimensional axes
        self.dimensional_axes = {}
        
        for dim in range(required_dims):
            axis_name = f"dim_{dim}"
            # Assign judges to positive and negative sides of this dimension
            pos_judges = active_judge_ids[dim * 2:dim * 2 + 1]
            neg_judges = active_judge_ids[dim * 2 + 1:dim * 2 + 2] if dim * 2 + 1 < num_judges else []
            
            # Calculate position based on judge affinities
            pos_strength = sum(self.judge_affinities.get(jid, 1.0) for jid in pos_judges)
            neg_strength = sum(self.judge_affinities.get(jid, 1.0) for jid in neg_judges)
            
            # Position ranges from -1 to +1
            if pos_strength + neg_strength > 0:
                position = (pos_strength - neg_strength) / (pos_strength + neg_strength)
            else:
                position = 0.0
            
            self.dimensional_axes[axis_name] = {
                'position': position,
                'positive_judges': pos_judges,
                'negative_judges': neg_judges
            }
    
    def calculate_judge_relevance(self, judge_id: str, task_embeddings: np.ndarray) -> float:
        """Calculate how relevant this node is for a specific judge's task"""
        if judge_id not in self.judge_affinities:
            return 0.5  # Neutral relevance for unknown judges
        
        # Base affinity
        base_relevance = self.judge_affinities[judge_id]
        
        # Similarity with representative embeddings
        similarity_scores = []
        for level, repr_emb in self.representative_embeddings.items():
            similarity = np.dot(repr_emb, task_embeddings.flatten()) / (
                np.linalg.norm(repr_emb) * np.linalg.norm(task_embeddings.flatten()) + 1e-8
            )
            similarity_scores.append(abs(similarity))
        
        # Weighted combination
        embedding_relevance = np.mean(similarity_scores)
        final_relevance = 0.6 * base_relevance + 0.4 * embedding_relevance
        
        return min(1.0, max(0.0, final_relevance))
    
    def adapt_to_judge_feedback(self, judge_id: str, feedback_score: float, learning_rate: float = 0.01):
        """Adapt node parameters based on judge feedback"""
        if judge_id not in self.judge_affinities:
            self.judge_affinities[judge_id] = 1.0
        
        # Update judge affinity based on feedback
        self.judge_affinities[judge_id] += learning_rate * (feedback_score - 0.5) * 2
        self.judge_affinities[judge_id] = max(0.1, min(2.0, self.judge_affinities[judge_id]))
        
        # Update reuse efficiency based on performance
        if feedback_score > 0.7:
            self.reuse_efficiency = min(2.0, self.reuse_efficiency * 1.01)
        elif feedback_score < 0.3:
            self.reuse_efficiency = max(0.5, self.reuse_efficiency * 0.99)
    
    # ... [The rest of the helper methods remain largely the same but with enhanced functionality]
    
    def _get_modality_weights(self, modality: str) -> Dict[str, float]:
        """Get comprehensive modality weights"""
        return self.modality_preferences.get(modality, self.modality_preferences['multimodal'])
    
    def _calculate_judge_boost(self, judge_masks: Dict[str, np.ndarray]) -> float:
        """Calculate activation boost from active judges"""
        if not judge_masks:
            return 1.0
        
        active_affinities = [self.judge_affinities.get(jid, 1.0) for jid in judge_masks.keys()]
        return 1.0 + 0.2 * (float(np.mean(active_affinities)) - 1.0)
    
    def _calculate_modality_factor(self, modality: str, embeddings: np.ndarray) -> float:
        """Enhanced modality factor calculation"""
        weights = self._get_modality_weights(modality)
        base_factor = weights['base_affinity']
        
        # Add complexity based on embedding characteristics
        embedding_complexity = np.std(embeddings)
        complexity_boost = 1.0 + 0.1 * embedding_complexity
        
        return float(base_factor * complexity_boost)
    
    def _get_branch_modality_boost(self, modality: str, branch_idx: int) -> float:
        """Get modality-specific boost for branch processing"""
        weights = self._get_modality_weights(modality)
        
        # Different branches get different modality-specific boosts
        if modality == 'vision' and branch_idx % 3 == 0:
            return weights.get('spatial_weight', 1.0)
        elif modality == 'text' and branch_idx % 3 == 1:
            return weights.get('semantic_weight', 1.0)
        elif modality == 'audio' and branch_idx % 3 == 2:
            return weights.get('temporal_weight', 1.0)
        
        return weights.get('base_affinity', 1.0)
    
    def _update_connection_strength(self, branch_idx: int, activation: float) -> float:
        """Update and return connection strength for this branch"""
        if branch_idx not in self.connection_strengths:
            self.connection_strengths[branch_idx] = 0.5
        
        # Exponential moving average of connection strength
        self.connection_strengths[branch_idx] = (
            0.9 * self.connection_strengths[branch_idx] + 0.1 * activation
        )
        
        return self.connection_strengths[branch_idx]
    
    def _update_representative_embeddings(self, embeddings: np.ndarray, modality: str):
        """Update multi-level representative embeddings"""
        embedding_mean = np.mean(embeddings, axis=0)
        
        # Different update rates for different abstraction levels
        self.representative_embeddings['local'] = (
            0.8 * self.representative_embeddings['local'] + 0.2 * embedding_mean
        )
        self.representative_embeddings['regional'] = (
            0.9 * self.representative_embeddings['regional'] + 0.1 * embedding_mean
        )
        self.representative_embeddings['global'] = (
            0.95 * self.representative_embeddings['global'] + 0.05 * embedding_mean
        )
    
    def _update_judge_activation_history(self, judge_masks: Dict[str, np.ndarray], activation: float):
        """Update history of judge activations for this node"""
        for judge_id in judge_masks.keys():
            if judge_id not in self.judge_activation_history:
                self.judge_activation_history[judge_id] = []
            
            self.judge_activation_history[judge_id].append(activation)
            
            # Keep only recent history (last 100 activations)
            if len(self.judge_activation_history[judge_id]) > 100:
                self.judge_activation_history[judge_id] = self.judge_activation_history[judge_id][-100:]
    
    def _get_judge_contributions(self, judge_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate individual judge contributions to processing"""
        contributions = {}
        for judge_id, mask in judge_masks.items():
            mask_strength = np.mean(mask)
            affinity = self.judge_affinities.get(judge_id, 1.0)
            contributions[judge_id] = mask_strength * affinity
        
        return contributions
    
    def _get_current_dimensional_position(self) -> Dict[str, float]:
        """Get current position in N-dimensional judge space"""
        return {dim: data['position'] for dim, data in self.dimensional_axes.items()}
    
    
    def _resize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Efficiently resize embeddings to match node dimension"""
        current_dim = embeddings.shape[-1]
        
        if current_dim == self.embed_dim:
            return embeddings
        elif current_dim < self.embed_dim:
            # Zero-pad efficiently
            pad_width = [(0, 0)] * (embeddings.ndim - 1) + [(0, self.embed_dim - current_dim)]
            return np.pad(embeddings, pad_width, mode='constant', constant_values=0)
        else:
            # Truncate
            return embeddings[..., :self.embed_dim]
    
    def _update_representative_embedding(self, embeddings: np.ndarray):
        """Update representative embedding with exponential moving average"""
        embedding_mean = np.mean(embeddings, axis=0)
        self.representative_embedding = (0.9 * self.representative_embedding + 0.1 * embedding_mean)
    
    def get_representative_embedding(self) -> np.ndarray:
        """Get representative embedding for similarity calculations"""
        return self.representative_embedding
    
    def expand_dimensions(self, target_dim: int):
        """Efficiently expand node dimensions"""
        if target_dim <= self.embed_dim:
            return
        
        old_dim = self.embed_dim
        self.embed_dim = target_dim
        
        # Expand weight matrix efficiently
        new_weight = np.zeros((target_dim, target_dim))
        new_weight[:old_dim, :old_dim] = self.weight_matrix
        # Add small random values to new dimensions
        new_weight[old_dim:, old_dim:] = np.random.randn(target_dim - old_dim, target_dim - old_dim) * 0.05
        self.weight_matrix = new_weight
        
        # Expand bias vector
        new_bias = np.zeros(target_dim)
        new_bias[:old_dim] = self.bias_vector
        new_bias[old_dim:] = np.random.randn(target_dim - old_dim) * 0.05
        self.bias_vector = new_bias
        
        # Expand representative embedding
        new_repr = np.zeros(target_dim)
        new_repr[:old_dim] = self.representative_embedding
        new_repr[old_dim:] = np.random.randn(target_dim - old_dim) * 0.05
        self.representative_embedding = new_repr
    
    def evolve_to_judge(self):
        """Transform this computational node into a judge node."""
        # Store current state for preservation
        preserved_state = {
            'embed_dim': self.embed_dim,
            'position': self.position,
            'node_position': self.node_position,
            'processing_count': self.processing_count,
            'last_activation': self.last_activation,
            'representative_embeddings': self.representative_embeddings.copy(),
            'judge_affinities': self.judge_affinities.copy(),
            'active_judges': self.active_judges.copy(),
            'dimensional_axes': self.dimensional_axes.copy()
        }
        
        # Change node type
        self.node_type = "Judge"
        
        # Initialize judge-specific attributes
        self.judge_id = self.node_id
        self.base_embed_dim = self.embed_dim
        self.current_embed_dim = self.embed_dim
        self.max_embed_dim = 512
        
        # Initialize attention configuration
        self.attention_config = {
            'num_heads': 2,
            'head_dim': self.embed_dim // 2,
            'max_heads': 8,
            'embed_dim': self.current_embed_dim
        }
        
        # Initialize judge transformation matrices
        self.transformation_matrices = {}
        self.amplitude_matrices = {}
        self.specialization_vectors = {}
        self._init_judge_transformation_matrices()
        
        # Initialize attention weights
        self.attention_weights = {}
        self._init_judge_attention_weights()
        
        # Initialize judge-specific parameters
        self.specialization_strength = random.uniform(0.8, 1.2)  # Higher for evolved judges
        self.adaptation_rate = random.uniform(0.05, 0.15)  # Higher learning rate
        self.dimension_sensitivity = random.uniform(1.0, 2.5)  # Enhanced sensitivity
        
        # Processing state
        self.processing_history = deque(maxlen=10)
        self.activation_level = 0.0
        self.is_active = False
        
        # Learning parameters
        self.learning_rate = 0.015  # Slightly higher for evolved nodes
        self.momentum = 0.9
        self.gradient_history = {}
        
        # Controller reference
        self.controller = None
        self.dimensional_position = None
        
        if self.demo:
            print(f"🧬 Computational node {self.node_id} evolved to Judge")
            print(f"   Preserved: {self.processing_count} processing experiences")
            print(f"   Enhanced specialization strength: {self.specialization_strength:.3f}")
        
        return preserved_state
    
    def _init_judge_transformation_matrices(self):
        """Initialize transformation matrices for judge functionality."""
        embed_dim = self.embed_dim
        
        # Rotation-like transformation matrices for different modalities
        self.transformation_matrices = {
            'text': self._create_rotation_matrix(embed_dim, angle=0.0),
            'vision': self._create_rotation_matrix(embed_dim, angle=np.pi/4),
            'audio': self._create_rotation_matrix(embed_dim, angle=np.pi/2),
            'tabular': self._create_rotation_matrix(embed_dim, angle=3*np.pi/4),
            'multimodal': self._create_rotation_matrix(embed_dim, angle=np.pi/6)
        }
        
        # Amplitude modulation matrices
        self.amplitude_matrices = {
            'text': np.diag(np.random.uniform(0.8, 1.2, embed_dim)),
            'vision': np.diag(np.random.uniform(0.7, 1.3, embed_dim)),
            'audio': np.diag(np.random.uniform(0.6, 1.4, embed_dim)),
            'tabular': np.diag(np.random.uniform(0.5, 1.5, embed_dim)),
            'multimodal': np.diag(np.random.uniform(0.9, 1.1, embed_dim))
        }
        
        # Specialization vectors
        self.specialization_vectors = {
            'text': np.random.randn(embed_dim) * 0.1,
            'vision': np.random.randn(embed_dim) * 0.15,
            'audio': np.random.randn(embed_dim) * 0.12,
            'tabular': np.random.randn(embed_dim) * 0.08,
            'multimodal': np.random.randn(embed_dim) * 0.05
        }
    
    def _init_judge_attention_weights(self):
        """Initialize attention weight matrices for judge functionality."""
        config = self.attention_config
        embed_dim = config['embed_dim']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        
        self.attention_weights = {
            'query_weights': np.random.randn(num_heads, embed_dim, head_dim) * 0.1,
            'key_weights': np.random.randn(num_heads, embed_dim, head_dim) * 0.1,
            'value_weights': np.random.randn(num_heads, embed_dim, head_dim) * 0.1,
            'output_projection': np.random.randn(embed_dim, embed_dim) * 0.1,
            'layer_norm_gamma': np.ones(embed_dim),
            'layer_norm_beta': np.zeros(embed_dim)
        }
    
    def _create_rotation_matrix(self, dim: int, angle: float) -> np.ndarray:
        """Create a rotation-like transformation matrix."""
        matrix = np.eye(dim)
        
        # Apply 2D rotations to pairs of dimensions
        for i in range(0, dim - 1, 2):
            cos_a = math.cos(angle + i * 0.1)
            sin_a = math.sin(angle + i * 0.1)
            
            # 2D rotation submatrix
            matrix[i, i] = cos_a
            matrix[i, i+1] = -sin_a
            matrix[i+1, i] = sin_a
            matrix[i+1, i+1] = cos_a
        
        return matrix
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get minimal node information"""
        return {
            'node_id': self.node_id,
            'position': self.position,
            'embed_dim': self.embed_dim,
            'processing_count': self.processing_count,
            'last_activation': self.last_activation,
            'splitter_id': self.splitter_id
        }

class Repeater:
    """Simple repeater node that forwards all inputs from computational nodes"""
    
    def __init__(self, node_id: str, node_position: Tuple[float, float, float], demo: bool = False):
        self.node_id = node_id
        self.node_position = node_position
        self.demo = demo
        self.node_type = "Repeater"
        
        # Simple state tracking
        self.processing_count = 0
        self.last_input_count = 0
        
        # Input/output tracking
        self.input_buffer = []  # Temporary storage for inputs
        self.processing_history = deque(maxlen=5)  # Keep minimal history
        
        if demo:
            print(f"Repeater {node_id} initialized at {node_position}")
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
        return self.weights
    def process_inputs(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simply repeat all inputs with minimal processing overhead"""
        if not inputs:
            return []
        
        # Store input count for tracking
        self.last_input_count = len(inputs)
        self.processing_count += 1
        
        # Simply copy inputs with minimal metadata addition
        outputs = []
        for i, input_data in enumerate(inputs):
            output = input_data.copy()  # Shallow copy for efficiency
            
            # Add minimal repeater metadata
            if 'processing_metadata' not in output:
                output['processing_metadata'] = {}
            
            output['processing_metadata']['repeater_id'] = self.node_id
            output['processing_metadata']['repeat_index'] = i
            output['processing_metadata']['processing_count'] = self.processing_count
            
            outputs.append(output)
        
        # Store minimal history
        self.processing_history.append({
            'input_count': len(inputs),
            'output_count': len(outputs),
            'timestamp': self.processing_count
        })
        
        if self.demo:
            print(f"Repeater {self.node_id} repeated {len(inputs)} inputs")
        
        return outputs
    
    def process_single_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single input for convenience"""
        result = self.process_inputs([input_data])
        return result[0] if result else {}
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get minimal status information"""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'position': self.node_position,
            'processing_count': self.processing_count,
            'last_input_count': self.last_input_count,
            'history_length': len(self.processing_history)
        }
    
    def clear_history(self):
        """Clear processing history to free memory"""
        self.processing_history.clear()
        self.input_buffer.clear()
        
class Retainer:
    """Retainer node that collects outputs from computational nodes and forwards to a review node when ALL signals are received"""
    def __init__(self, node_id, node_position, expected_nodes, demo=False):
        self.node_id = node_id
        self.node_position = node_position
        # Ensure expected_nodes is always a list
        if isinstance(expected_nodes, (int, float)):
            self.expected_nodes = [int(expected_nodes)]
        elif isinstance(expected_nodes, list):
            self.expected_nodes = expected_nodes
        else:
            self.expected_nodes = []
        self.demo = demo
        self.collected_outputs = {}
        self.review_node = None
        self.node_type = "Retainer"

    def connect_review(self, review_node):
        self.review_node = review_node
        if self.demo:
            print(f"[Retainer {self.node_id}] Connected to review node {review_node.node_id}")
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
    def receive(self, sender_id, output_probs):
        """Called by each computational node once it has finished processing."""
        # Ensure the output_probs is in the correct format
        if output_probs is None:
            output_probs = []
        
        # Handle non-list outputs by converting them
        if not isinstance(output_probs, list):
            output_probs = [output_probs]
            
        self.collected_outputs[sender_id] = output_probs
        print(f"[Retainer {self.node_id}] Received from {sender_id}: {output_probs}")

        if self.ready_to_forward():
            self.forward_to_review()

    def ready_to_forward(self):
        # Ensure expected_nodes is always iterable
        if not hasattr(self, 'expected_nodes') or self.expected_nodes is None:
            self.expected_nodes = []
        elif isinstance(self.expected_nodes, (int, float)):
            self.expected_nodes = [int(self.expected_nodes)]
        elif not isinstance(self.expected_nodes, (list, tuple)):
            self.expected_nodes = []
            
        return all(node_id in self.collected_outputs for node_id in self.expected_nodes)

    def forward_to_review(self):
        # Ensure expected_nodes is always iterable
        if not hasattr(self, 'expected_nodes') or self.expected_nodes is None:
            self.expected_nodes = []
        elif isinstance(self.expected_nodes, (int, float)):
            self.expected_nodes = [int(self.expected_nodes)]
        elif not isinstance(self.expected_nodes, (list, tuple)):
            self.expected_nodes = []
            
        if not self.review_node:
            print(f"[Retainer {self.node_id}] No review node connected.")
            # Still return the collected outputs for potential downstream processing
            ordered_outputs = []
            for node_id in self.expected_nodes:
                if node_id in self.collected_outputs:
                    output = self.collected_outputs[node_id]
                    # Ensure each output is a list of probabilities
                    if not isinstance(output, list):
                        output = [float(output)]  # Convert single value to list
                    elif not all(isinstance(x, (int, float)) for x in output):
                        # If it's a list but not of numbers, create a default output
                        output = [0.25, 0.25, 0.25, 0.25]  # Default uniform distribution
                    ordered_outputs.append(output)
            # Clear for next round even when no review node is connected
            self.collected_outputs.clear()
            return ordered_outputs

        # Ensure we're passing the output in the expected format for the reviewer
        ordered_outputs = []
        for node_id in self.expected_nodes:
            if node_id in self.collected_outputs:
                output = self.collected_outputs[node_id]
                # Ensure each output is a list of probabilities
                if not isinstance(output, list):
                    output = [float(output)]  # Convert single value to list
                elif not all(isinstance(x, (int, float)) for x in output):
                    # If it's a list but not of numbers, create a default output
                    output = [0.25, 0.25, 0.25, 0.25]  # Default uniform distribution
                ordered_outputs.append(output)
                
        result = self.review_node.aggregate(ordered_outputs)

        print(f"[Retainer {self.node_id}] → Review {self.review_node.node_id}: Final result {result}")

        # Clear for next round
        self.collected_outputs.clear()
        return result
        
    def process(self, input_data):
        # For Retainer, process should receive data and forward it to the review node
        try:
            # Ensure expected_nodes is iterable
            if not hasattr(self, 'expected_nodes') or self.expected_nodes is None:
                self.expected_nodes = []
            elif isinstance(self.expected_nodes, (int, float)):
                self.expected_nodes = [int(self.expected_nodes)]
            elif not isinstance(self.expected_nodes, (list, tuple)):
                self.expected_nodes = []
            
            # Handle different input types
            if isinstance(input_data, dict):
                # If it's already a dictionary of node_id -> output, merge with existing
                if any(node_id in input_data for node_id in self.expected_nodes):
                    for k, v in input_data.items():
                        if k in self.expected_nodes:
                            self.collected_outputs[k] = v if isinstance(v, list) else [float(v)]
                # Extract from a structured dict with 'sender_id' and 'output'
                elif 'sender_id' in input_data and 'output' in input_data:
                    sender_id = input_data['sender_id']
                    output = input_data['output']
                    # Convert output to list if it's not already
                    if not isinstance(output, list):
                        output = [float(output)]
                    self.collected_outputs[sender_id] = output
            
            # Handle tuple input (sender_id, output)
            elif isinstance(input_data, tuple) and len(input_data) == 2:
                sender_id, output = input_data
                # Convert output to list if it's not already
                if not isinstance(output, list):
                    output = [float(output)]
                self.collected_outputs[sender_id] = output
            
            # Handle list input - this should be treated as a single result from one node
            elif isinstance(input_data, list):
                # If we don't have any collected outputs yet, treat this as the first input
                if not self.collected_outputs and self.expected_nodes:
                    # Assign to the first expected node if we don't know the sender
                    first_node = self.expected_nodes[0]
                    self.collected_outputs[first_node] = input_data
                elif len(self.collected_outputs) < len(self.expected_nodes):
                    # Find the next expected node that doesn't have data yet
                    for node_id in self.expected_nodes:
                        if node_id not in self.collected_outputs:
                            self.collected_outputs[node_id] = input_data
                            break
            
            # Handle simple scalar input
            elif isinstance(input_data, (int, float)):
                # If we don't have any collected outputs yet, treat this as the first input
                if not self.collected_outputs and self.expected_nodes:
                    first_node = self.expected_nodes[0]
                    self.collected_outputs[first_node] = [float(input_data)]
                elif len(self.collected_outputs) < len(self.expected_nodes):
                    # Find the next expected node that doesn't have data yet
                    for node_id in self.expected_nodes:
                        if node_id not in self.collected_outputs:
                            self.collected_outputs[node_id] = [float(input_data)]
                            break
            
            # If we have all expected outputs, forward to the review node
            result = None
            if self.ready_to_forward():
                result = self.forward_to_review()
                
            # Return the result if we have one, otherwise return current collection state
            if result is not None:
                return result
            else:
                return dict(self.collected_outputs)  # Return a copy to avoid mutation issues
            
        except Exception as e:
            print(f"Error in Retainer {self.node_id} process: {str(e)}")
            # Ensure expected_nodes is iterable before using it
            if not hasattr(self, 'expected_nodes') or self.expected_nodes is None:
                self.expected_nodes = []
            elif isinstance(self.expected_nodes, (int, float)):
                self.expected_nodes = [int(self.expected_nodes)]
            elif not isinstance(self.expected_nodes, (list, tuple)):
                self.expected_nodes = []
            # Return a safe fallback
            return {node_id: [0.25, 0.25, 0.25, 0.25] for node_id in self.expected_nodes}

class Reviewer:
    """Reviewer node that aggregates outputs from retainer nodes and calculates final probabilities"""
    
    def __init__(self, node_id: str, node_position: Tuple[float, float, float], 
                 num_comps: int = 1, demo: bool = False):
        self.node_id = node_id
        self.node_position = node_position
        self.num_comps = num_comps  # Number of components to expect
        self.demo = demo
        self.node_type = "Reviewer"
        
        # Aggregation strategies
        self.aggregation_strategies = {
            'weighted_average': self._weighted_average,
            'max_confidence': self._max_confidence,
            'consensus': self._consensus_voting,
            'dynamic': self._dynamic_aggregation
        }
        self.current_strategy = 'dynamic'
        
        # Performance tracking
        self.review_history = deque(maxlen=20)
        self.confidence_scores = deque(maxlen=20)
        self.strategy_performance = {strategy: 0.5 for strategy in self.aggregation_strategies}
        
        # Adaptive weights for different input types
        self.input_weights = {
            'high_confidence': 1.2,
            'medium_confidence': 1.0,
            'low_confidence': 0.7,
            'uncertain': 0.5
        }
        
        # Final probability calibration
        self.calibration_factor = 1.0
        self.temperature = 1.0  # For softmax temperature scaling
        
        # Statistical tracking
        self.total_reviews = 0
        self.avg_confidence = 0.5
        self.last_output_probs = None
        
        if demo:
            print(f"Reviewer {node_id} initialized with {self.current_strategy} aggregation strategy")
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
        return self.weights
    def aggregate(self, input_outputs: List[List[float]]) -> List[float]:
        """Main aggregation method called by retainer nodes"""
        if not input_outputs:
            return [0.25, 0.25, 0.25, 0.25]  # Default uniform distribution
        
        # Ensure all inputs are proper probability lists
        cleaned_outputs = self._clean_input_outputs(input_outputs)
        
        # Calculate confidence scores for each input
        input_confidences = [self._calculate_confidence(output) for output in cleaned_outputs]
        
        # Apply current aggregation strategy
        aggregated_probs = self.aggregation_strategies[self.current_strategy](
            cleaned_outputs, input_confidences
        )
        
        # Apply calibration and temperature scaling
        final_probs = self._calibrate_probabilities(aggregated_probs)
        
        # Update tracking and potentially adapt strategy
        self._update_performance_tracking(cleaned_outputs, input_confidences, final_probs)
        
        # Store for analysis
        self.last_output_probs = final_probs
        self.total_reviews += 1
        
        if self.demo:
            print(f"Reviewer {self.node_id} aggregated {len(input_outputs)} inputs → {[f'{p:.3f}' for p in final_probs]}")
        
        return final_probs
    
    def _clean_input_outputs(self, input_outputs: List[List[float]]) -> List[List[float]]:
        """Clean and normalize input probability lists"""
        cleaned = []
        
        for output in input_outputs:
            if not isinstance(output, list):
                # Convert single values to uniform distribution
                cleaned.append([0.25, 0.25, 0.25, 0.25])
                continue
            
            # Ensure we have exactly 4 probabilities
            if len(output) < 4:
                # Pad with equal probability for missing values
                remaining_prob = (1.0 - sum(output)) / (4 - len(output))
                output.extend([remaining_prob] * (4 - len(output)))
            elif len(output) > 4:
                # Truncate and renormalize
                output = output[:4]
            
            # Normalize to sum to 1.0
            total = sum(output)
            if total > 0:
                output = [p / total for p in output]
            else:
                output = [0.25, 0.25, 0.25, 0.25]
            
            cleaned.append(output)
        
        return cleaned
    
    def _calculate_confidence(self, probabilities: List[float]) -> float:
        """Calculate confidence score for a probability distribution"""
        # Entropy-based confidence (lower entropy = higher confidence)
        entropy = -sum(p * math.log(p + 1e-8) for p in probabilities)
        max_entropy = math.log(len(probabilities))
        confidence = 1.0 - (entropy / max_entropy)
        
        # Max probability confidence
        max_prob_confidence = max(probabilities)
        
        # Variance-based confidence (lower variance = higher confidence for non-uniform)
        mean_prob = sum(probabilities) / len(probabilities)
        variance = sum((p - mean_prob) ** 2 for p in probabilities) / len(probabilities)
        variance_confidence = min(1.0, variance * 4)  # Scale variance
        
        # Combine confidence measures
        combined_confidence = (confidence * 0.4 + max_prob_confidence * 0.4 + variance_confidence * 0.2)
        
        return max(0.1, min(1.0, combined_confidence))
    
    def _weighted_average(self, outputs: List[List[float]], confidences: List[float]) -> List[float]:
        """Weighted average aggregation based on confidence scores"""
        if not outputs:
            return [0.25, 0.25, 0.25, 0.25]
        
        # Normalize confidence weights
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weights = [c / total_confidence for c in confidences]
        else:
            weights = [1.0 / len(outputs)] * len(outputs)
        
        # Calculate weighted average
        num_probs = len(outputs[0])
        aggregated = [0.0] * num_probs
        
        for i, output in enumerate(outputs):
            weight = weights[i]
            for j in range(num_probs):
                aggregated[j] += output[j] * weight
        
        return aggregated
    
    def _max_confidence(self, outputs: List[List[float]], confidences: List[float]) -> List[float]:
        """Select the output with highest confidence"""
        if not outputs:
            return [0.25, 0.25, 0.25, 0.25]
        
        max_confidence_idx = confidences.index(max(confidences))
        return outputs[max_confidence_idx].copy()
    
    def _consensus_voting(self, outputs: List[List[float]], confidences: List[float]) -> List[float]:
        """Consensus voting: give more weight to outputs that agree"""
        if not outputs:
            return [0.25, 0.25, 0.25, 0.25]
        
        if len(outputs) == 1:
            return outputs[0].copy()
        
        # Calculate pairwise agreements
        agreements = []
        for i, output1 in enumerate(outputs):
            agreement_score = 0.0
            for j, output2 in enumerate(outputs):
                if i != j:
                    # Cosine similarity between probability distributions
                    dot_product = sum(a * b for a, b in zip(output1, output2))
                    norm1 = math.sqrt(sum(a * a for a in output1))
                    norm2 = math.sqrt(sum(b * b for b in output2))
                    similarity = dot_product / (norm1 * norm2 + 1e-8)
                    agreement_score += similarity
            agreements.append(agreement_score / (len(outputs) - 1))
        
        # Combine confidence and agreement scores
        consensus_weights = [conf * (1.0 + agree) for conf, agree in zip(confidences, agreements)]
        
        # Normalize weights
        total_weight = sum(consensus_weights)
        if total_weight > 0:
            consensus_weights = [w / total_weight for w in consensus_weights]
        else:
            consensus_weights = [1.0 / len(outputs)] * len(outputs)
        
        # Calculate consensus-weighted average
        num_probs = len(outputs[0])
        aggregated = [0.0] * num_probs
        
        for i, output in enumerate(outputs):
            weight = consensus_weights[i]
            for j in range(num_probs):
                aggregated[j] += output[j] * weight
        
        return aggregated
    
    def _dynamic_aggregation(self, outputs: List[List[float]], confidences: List[float]) -> List[float]:
        """Dynamic strategy selection based on input characteristics"""
        if not outputs:
            return [0.25, 0.25, 0.25, 0.25]
        
        # Analyze input characteristics
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        num_inputs = len(outputs)
        
        # Select strategy based on characteristics
        if num_inputs == 1:
            return outputs[0].copy()
        elif avg_confidence > 0.8 and confidence_variance < 0.1:
            # High confidence, low variance: use weighted average
            return self._weighted_average(outputs, confidences)
        elif confidence_variance > 0.3:
            # High variance in confidence: use max confidence
            return self._max_confidence(outputs, confidences)
        else:
            # Moderate confidence: use consensus voting
            return self._consensus_voting(outputs, confidences)
    
    def _calibrate_probabilities(self, probabilities: List[float]) -> List[float]:
        """Apply calibration and temperature scaling to final probabilities"""
        # Apply calibration factor
        calibrated = [p * self.calibration_factor for p in probabilities]
        
        # Ensure we still have valid probabilities
        total = sum(calibrated)
        if total > 0:
            calibrated = [p / total for p in calibrated]
        else:
            calibrated = [0.25, 0.25, 0.25, 0.25]
        
        # Apply temperature scaling (softmax with temperature)
        if self.temperature != 1.0:
            # Convert to logits, apply temperature, convert back to probabilities
            logits = [math.log(p + 1e-8) for p in calibrated]
            temp_logits = [l / self.temperature for l in logits]
            
            # Softmax
            max_logit = max(temp_logits)
            exp_logits = [math.exp(l - max_logit) for l in temp_logits]
            sum_exp = sum(exp_logits)
            calibrated = [e / sum_exp for e in exp_logits]
        
        return calibrated
    
    def _update_performance_tracking(self, outputs: List[List[float]], confidences: List[float], 
                                   final_probs: List[float]):
        """Update performance tracking and adapt strategy if needed"""
        # Calculate overall confidence for this review
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        self.confidence_scores.append(overall_confidence)
        
        # Update average confidence
        self.avg_confidence = 0.9 * self.avg_confidence + 0.1 * overall_confidence
        
        # Store review history
        review_record = {
            'inputs': len(outputs),
            'avg_confidence': overall_confidence,
            'strategy_used': self.current_strategy,
            'final_probs': final_probs.copy(),
            'max_prob': max(final_probs),
            'entropy': -sum(p * math.log(p + 1e-8) for p in final_probs)
        }
        self.review_history.append(review_record)
        
        # Periodically evaluate and potentially change strategy
        if len(self.review_history) >= 10 and self.total_reviews % 10 == 0:
            self._evaluate_strategy_performance()
    
    def _evaluate_strategy_performance(self):
        """Evaluate strategy performance and potentially switch"""
        if len(self.review_history) < 5:
            return
        
        # Analyze recent performance
        recent_reviews = list(self.review_history)[-5:]
        
        # Metrics: average confidence, entropy (decision clarity), max probability
        avg_confidence = sum(r['avg_confidence'] for r in recent_reviews) / len(recent_reviews)
        avg_entropy = sum(r['entropy'] for r in recent_reviews) / len(recent_reviews)
        avg_max_prob = sum(r['max_prob'] for r in recent_reviews) / len(recent_reviews)
        
        # Performance score (higher is better)
        performance_score = (avg_confidence * 0.4 + (1.0 - avg_entropy/math.log(4)) * 0.3 + avg_max_prob * 0.3)
        
        # Update strategy performance
        current_performance = self.strategy_performance[self.current_strategy]
        self.strategy_performance[self.current_strategy] = (0.7 * current_performance + 0.3 * performance_score)
        
        # Consider switching to best performing strategy
        best_strategy = max(self.strategy_performance.items(), key=lambda x: x[1])
        if best_strategy[0] != self.current_strategy and best_strategy[1] > current_performance + 0.1:
            old_strategy = self.current_strategy
            self.current_strategy = best_strategy[0]
            if self.demo:
                print(f"Reviewer {self.node_id} switched from {old_strategy} to {self.current_strategy}")
    
    def set_aggregation_strategy(self, strategy: str):
        """Manually set aggregation strategy"""
        if strategy in self.aggregation_strategies:
            self.current_strategy = strategy
            if self.demo:
                print(f"Reviewer {self.node_id} strategy set to {strategy}")
        else:
            print(f"Unknown strategy: {strategy}. Available: {list(self.aggregation_strategies.keys())}")
    
    def set_temperature(self, temperature: float):
        """Set temperature for probability calibration"""
        self.temperature = max(0.1, min(10.0, temperature))  # Clamp to reasonable range
        if self.demo:
            print(f"Reviewer {self.node_id} temperature set to {self.temperature}")
    
    def get_review_stats(self) -> Dict[str, Any]:
        """Get comprehensive review statistics"""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'total_reviews': self.total_reviews,
            'current_strategy': self.current_strategy,
            'avg_confidence': self.avg_confidence,
            'temperature': self.temperature,
            'calibration_factor': self.calibration_factor,
            'strategy_performance': self.strategy_performance.copy(),
            'last_output_probs': self.last_output_probs,
            'recent_confidence': list(self.confidence_scores)[-5:] if self.confidence_scores else []
        }
    
    def reset_performance_tracking(self):
        """Reset performance tracking for fresh evaluation"""
        self.review_history.clear()
        self.confidence_scores.clear()
        self.strategy_performance = {strategy: 0.5 for strategy in self.aggregation_strategies}
        self.avg_confidence = 0.5
        if self.demo:
            print(f"Reviewer {self.node_id} performance tracking reset")


class Handler:
    """Handler node that takes responses from multiple reviewers and calculates final probabilities"""
    
    def __init__(self, node_id: str, node_position: Tuple[float, float, float], 
                 num_reviewers: int = 4, num_output_classes: int = 4, demo: bool = False):
        self.node_id = node_id
        self.node_position = node_position
        self.num_reviewers = num_reviewers  # Number of reviewers to expect
        self.num_output_classes = num_output_classes
        self.demo = demo
        self.node_type = "Handler"
        
        # Processing strategies for combining reviewer outputs
        self.combination_strategies = {
            'weighted_ensemble': self._weighted_ensemble,
            'confidence_weighted': self._confidence_weighted_combination,
            'majority_vote': self._majority_vote,
            'bayesian_fusion': self._bayesian_fusion,
            'adaptive': self._adaptive_combination
        }
        self.current_strategy = 'adaptive'
        
        # Reviewer management
        self.connected_reviewers = {}  # reviewer_id -> reviewer_reference
        self.reviewer_weights = {}  # reviewer_id -> weight
        self.reviewer_performance = {}  # reviewer_id -> performance_metrics
        self.reviewer_specializations = {}  # reviewer_id -> specialization_info
        
        # Final probability calibration
        self.calibration_parameters = {
            'temperature': 1.0,
            'confidence_threshold': 0.6,
            'consensus_threshold': 0.5,
            'smoothing_factor': 0.1
        }
        
        # Decision tracking and learning
        self.decision_history = deque(maxlen=50)
        self.performance_metrics = {
            'total_decisions': 0,
            'avg_confidence': 0.5,
            'avg_consensus': 0.5,
            'strategy_performance': {strategy: 0.5 for strategy in self.combination_strategies}
        }
        
        # Output formatting and post-processing
        self.output_processors = {
            'softmax': self._apply_softmax,
            'normalize': self._normalize_probabilities,
            'clip': self._clip_probabilities,
            'smooth': self._apply_smoothing
        }
        
        # Meta-learning for strategy adaptation
        self.strategy_adaptation_threshold = 10  # Decisions before considering strategy change
        self.adaptation_sensitivity = 0.1  # How sensitive to performance differences
        
        if demo:
            print(f"Handler {node_id} initialized with {self.current_strategy} combination strategy")
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
        return self.weights
    def connect_reviewer(self, reviewer: 'Reviewer', specialization: str = 'general', weight: float = 1.0):
        """Connect a reviewer node to this handler"""
        reviewer_id = reviewer.node_id
        self.connected_reviewers[reviewer_id] = reviewer
        self.reviewer_weights[reviewer_id] = weight
        self.reviewer_specializations[reviewer_id] = specialization
        self.reviewer_performance[reviewer_id] = {
            'decisions_made': 0,
            'avg_confidence': 0.5,
            'consistency_score': 0.5,
            'specialization_score': 0.5
        }
        
        if self.demo:
            print(f"Connected reviewer {reviewer_id} with specialization '{specialization}' and weight {weight}")
    
    def disconnect_reviewer(self, reviewer_id: str):
        """Disconnect a reviewer from this handler"""
        if reviewer_id in self.connected_reviewers:
            del self.connected_reviewers[reviewer_id]
            del self.reviewer_weights[reviewer_id]
            del self.reviewer_specializations[reviewer_id] 
            del self.reviewer_performance[reviewer_id]
            
            if self.demo:
                print(f"Disconnected reviewer {reviewer_id}")
    
    def process_reviewer_outputs(self, reviewer_outputs: Dict[str, List[float]]) -> Dict[str, Any]:
        """Main method to process outputs from multiple reviewers and generate final probabilities"""
        if not reviewer_outputs:
            return self._create_default_response("NO_REVIEWER_OUTPUTS")
        
        # Validate and clean reviewer outputs
        cleaned_outputs = self._validate_reviewer_outputs(reviewer_outputs)
        
        if not cleaned_outputs:
            return self._create_default_response("ALL_INVALID_OUTPUTS")
        
        # Calculate confidence scores for each reviewer output
        reviewer_confidences = self._calculate_reviewer_confidences(cleaned_outputs)
        
        # Apply current combination strategy
        combined_probabilities = self.combination_strategies[self.current_strategy](
            cleaned_outputs, reviewer_confidences
        )
        
        # Post-process and calibrate final probabilities
        final_probabilities = self._post_process_probabilities(combined_probabilities)
        
        # Calculate decision metrics
        decision_metrics = self._calculate_decision_metrics(
            cleaned_outputs, reviewer_confidences, final_probabilities
        )
        
        # Update performance tracking and adapt strategy if needed
        self._update_performance_tracking(
            cleaned_outputs, reviewer_confidences, final_probabilities, decision_metrics
        )
        
        # Create comprehensive response
        response = {
            'final_probabilities': final_probabilities,
            'decision_metrics': decision_metrics,
            'reviewer_contributions': self._analyze_reviewer_contributions(cleaned_outputs, reviewer_confidences),
            'processing_metadata': {
                'strategy_used': self.current_strategy,
                'num_reviewers': len(cleaned_outputs),
                'calibration_applied': self.calibration_parameters.copy(),
                'handler_id': self.node_id
            }
        }
        
        if self.demo:
            print(f"Handler {self.node_id} processed {len(cleaned_outputs)} reviewer outputs")
            print(f"Final probabilities: {[f'{p:.3f}' for p in final_probabilities]}")
            print(f"Confidence: {decision_metrics['confidence']:.3f}, Consensus: {decision_metrics['consensus']:.3f}")
        
        return response
    
    def _validate_reviewer_outputs(self, reviewer_outputs: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Validate and clean reviewer outputs"""
        cleaned = {}
        
        for reviewer_id, output in reviewer_outputs.items():
            # Skip invalid outputs
            if not isinstance(output, list) or len(output) == 0:
                if self.demo:
                    print(f"Skipping invalid output from reviewer {reviewer_id}")
                continue
            
            # Handle different output formats
            if len(output) != self.num_output_classes:
                if len(output) < self.num_output_classes:
                    # Pad with uniform probability for missing classes
                    remaining_prob = (1.0 - sum(output)) / (self.num_output_classes - len(output))
                    output.extend([max(0.0, remaining_prob)] * (self.num_output_classes - len(output)))
                else:
                    # Truncate and renormalize
                    output = output[:self.num_output_classes]
            
            # Ensure non-negative probabilities
            output = [max(0.0, p) for p in output]
            
            # Normalize to sum to 1.0
            total = sum(output)
            if total > 0:
                output = [p / total for p in output]
            else:
                output = [1.0 / self.num_output_classes] * self.num_output_classes
            
            cleaned[reviewer_id] = output
        
        return cleaned
    
    def _calculate_reviewer_confidences(self, reviewer_outputs: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate confidence scores for each reviewer output"""
        confidences = {}
        
        for reviewer_id, probabilities in reviewer_outputs.items():
            # Entropy-based confidence (lower entropy = higher confidence)
            entropy = -sum(p * math.log(p + 1e-8) for p in probabilities)
            max_entropy = math.log(len(probabilities))
            entropy_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
            
            # Max probability confidence
            max_prob_confidence = max(probabilities)
            
            # Variance-based confidence (higher variance can indicate lower confidence for uniform distributions)
            mean_prob = sum(probabilities) / len(probabilities)
            variance = sum((p - mean_prob) ** 2 for p in probabilities) / len(probabilities)
            variance_confidence = min(1.0, variance * 4)  # Scale and clamp
            
            # Historical performance weight
            historical_weight = 1.0
            if reviewer_id in self.reviewer_performance:
                hist_confidence = self.reviewer_performance[reviewer_id]['avg_confidence']
                historical_weight = 0.7 + 0.3 * hist_confidence
            
            # Combine confidence measures
            combined_confidence = (
                entropy_confidence * 0.4 + 
                max_prob_confidence * 0.3 + 
                variance_confidence * 0.2 +
                historical_weight * 0.1
            )
            
            confidences[reviewer_id] = max(0.1, min(1.0, combined_confidence))
        
        return confidences
    
    def _weighted_ensemble(self, outputs: Dict[str, List[float]], confidences: Dict[str, float]) -> List[float]:
        """Weighted ensemble based on reviewer weights and confidences"""
        if not outputs:
            return [1.0 / self.num_output_classes] * self.num_output_classes
        
        weighted_sum = [0.0] * self.num_output_classes
        total_weight = 0.0
        
        for reviewer_id, probabilities in outputs.items():
            # Combine static weight and dynamic confidence
            reviewer_weight = self.reviewer_weights.get(reviewer_id, 1.0)
            confidence = confidences.get(reviewer_id, 0.5)
            combined_weight = reviewer_weight * confidence
            
            for i in range(self.num_output_classes):
                weighted_sum[i] += probabilities[i] * combined_weight
            
            total_weight += combined_weight
        
        # Normalize by total weight
        if total_weight > 0:
            return [prob / total_weight for prob in weighted_sum]
        else:
            return [1.0 / self.num_output_classes] * self.num_output_classes
    
    def _confidence_weighted_combination(self, outputs: Dict[str, List[float]], 
                                       confidences: Dict[str, float]) -> List[float]:
        """Combination weighted purely by confidence scores"""
        if not outputs:
            return [1.0 / self.num_output_classes] * self.num_output_classes
        
        # Normalize confidence weights
        total_confidence = sum(confidences.values())
        if total_confidence <= 0:
            # Equal weighting if no confidence information
            weight = 1.0 / len(outputs)
            weights = {reviewer_id: weight for reviewer_id in outputs.keys()}
        else:
            weights = {reviewer_id: conf / total_confidence for reviewer_id, conf in confidences.items()}
        
        # Weighted combination
        result = [0.0] * self.num_output_classes
        for reviewer_id, probabilities in outputs.items():
            weight = weights[reviewer_id]
            for i in range(self.num_output_classes):
                result[i] += probabilities[i] * weight
        
        return result
    
    def _majority_vote(self, outputs: Dict[str, List[float]], confidences: Dict[str, float]) -> List[float]:
        """Majority voting based on highest probability class from each reviewer"""
        if not outputs:
            return [1.0 / self.num_output_classes] * self.num_output_classes
        
        # Get the predicted class from each reviewer
        class_votes = [0.0] * self.num_output_classes  # Use float for votes
        
        for reviewer_id, probabilities in outputs.items():
            predicted_class = probabilities.index(max(probabilities))
            confidence = confidences.get(reviewer_id, 1.0)
            
            # Weight vote by confidence
            class_votes[predicted_class] += confidence
        
        # Convert votes to probabilities
        total_votes = sum(class_votes)
        if total_votes > 0:
            return [votes / total_votes for votes in class_votes]
        else:
            return [1.0 / self.num_output_classes] * self.num_output_classes
    
    def _bayesian_fusion(self, outputs: Dict[str, List[float]], confidences: Dict[str, float]) -> List[float]:
        """Bayesian fusion of reviewer outputs"""
        if not outputs:
            return [1.0 / self.num_output_classes] * self.num_output_classes
        
        # Start with uniform prior
        result = [1.0 / self.num_output_classes] * self.num_output_classes
        
        # Sequentially update with each reviewer's output
        for reviewer_id, probabilities in outputs.items():
            confidence = confidences.get(reviewer_id, 0.5)
            
            # Bayesian update: posterior ∝ prior × likelihood
            for i in range(self.num_output_classes):
                # Use confidence to weight the likelihood
                likelihood = confidence * probabilities[i] + (1 - confidence) * (1.0 / self.num_output_classes)
                result[i] *= likelihood
            
            # Normalize
            total = sum(result)
            if total > 0:
                result = [p / total for p in result]
        
        return result
    
    def _adaptive_combination(self, outputs: Dict[str, List[float]], confidences: Dict[str, float]) -> List[float]:
        """Adaptive combination that selects the best strategy based on current conditions"""
        if not outputs:
            return [1.0 / self.num_output_classes] * self.num_output_classes
        
        # Analyze current conditions
        num_reviewers = len(outputs)
        avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.5
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences.values()) / len(confidences)
        
        # Agreement analysis
        consensus_score = self._calculate_consensus_score(list(outputs.values()))
        
        # Select strategy based on conditions
        if num_reviewers == 1:
            # Single reviewer: just return the output
            return list(outputs.values())[0]
        elif avg_confidence > 0.8 and confidence_variance < 0.1:
            # High confidence, low variance: use weighted ensemble
            return self._weighted_ensemble(outputs, confidences)
        elif consensus_score < 0.3:
            # Low consensus: use confidence weighting to favor more confident reviewers
            return self._confidence_weighted_combination(outputs, confidences)
        elif avg_confidence < 0.5:
            # Low overall confidence: use majority voting for stability
            return self._majority_vote(outputs, confidences)
        else:
            # Moderate conditions: use Bayesian fusion
            return self._bayesian_fusion(outputs, confidences)
    
    def _calculate_consensus_score(self, probability_lists: List[List[float]]) -> float:
        """Calculate how much the reviewers agree with each other"""
        if len(probability_lists) < 2:
            return 1.0
        
        consensus_score = 0.0
        comparisons = 0
        
        for i in range(len(probability_lists)):
            for j in range(i + 1, len(probability_lists)):
                # Calculate similarity (1 - distance)
                distance = sum(abs(probability_lists[i][k] - probability_lists[j][k]) 
                             for k in range(len(probability_lists[i])))
                similarity = 1.0 - (distance / 2.0)
                consensus_score += similarity
                comparisons += 1
        
        return consensus_score / comparisons if comparisons > 0 else 0.0
    
    def _post_process_probabilities(self, probabilities: List[float]) -> List[float]:
        """Apply post-processing and calibration to final probabilities"""
        result = probabilities.copy()
        
        # Apply temperature scaling
        if self.calibration_parameters['temperature'] != 1.0:
            result = self._apply_temperature_scaling(result, self.calibration_parameters['temperature'])
        
        # Apply smoothing
        if self.calibration_parameters['smoothing_factor'] > 0:
            result = self._apply_smoothing(result)
        
        # Ensure valid probabilities
        result = self._normalize_probabilities(result)
        result = self._clip_probabilities(result)
        
        return result
    
    def _apply_temperature_scaling(self, probabilities: List[float], temperature: float) -> List[float]:
        """Apply temperature scaling for calibration"""
        if temperature <= 0:
            # Degenerate case: return one-hot encoding of max probability
            max_idx = probabilities.index(max(probabilities))
            result = [0.0] * len(probabilities)
            result[max_idx] = 1.0
            return result
        
        # Convert to logits, apply temperature, convert back
        logits = [math.log(max(p, 1e-8)) for p in probabilities]
        scaled_logits = [l / temperature for l in logits]
        
        # Apply softmax
        return self._apply_softmax(scaled_logits)
    
    def _apply_softmax(self, values: List[float]) -> List[float]:
        """Apply softmax to convert values to probabilities"""
        max_val = max(values)
        exp_values = [math.exp(v - max_val) for v in values]
        total = sum(exp_values)
        return [e / total for e in exp_values] if total > 0 else [1.0 / len(values)] * len(values)
    
    def _normalize_probabilities(self, probabilities: List[float]) -> List[float]:
        """Normalize probabilities to sum to 1.0"""
        total = sum(probabilities)
        if total > 0:
            return [p / total for p in probabilities]
        else:
            return [1.0 / len(probabilities)] * len(probabilities)
    
    def _clip_probabilities(self, probabilities: List[float], 
                           min_prob: float = 1e-6, max_prob: float = 1.0 - 1e-6) -> List[float]:
        """Clip probabilities to valid range"""
        clipped = [max(min_prob, min(max_prob, p)) for p in probabilities]
        return self._normalize_probabilities(clipped)
    
    def _apply_smoothing(self, probabilities: List[float]) -> List[float]:
        """Apply label smoothing to probabilities"""
        smoothing = self.calibration_parameters['smoothing_factor']
        uniform = 1.0 / len(probabilities)
        
        smoothed = [(1 - smoothing) * p + smoothing * uniform for p in probabilities]
        return self._normalize_probabilities(smoothed)
    
    def _calculate_decision_metrics(self, outputs: Dict[str, List[float]], 
                                  confidences: Dict[str, float], 
                                  final_probs: List[float]) -> Dict[str, float]:
        """Calculate comprehensive decision metrics"""
        # Overall confidence (entropy-based)
        entropy = -sum(p * math.log(p + 1e-8) for p in final_probs)
        max_entropy = math.log(len(final_probs))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Consensus among reviewers
        consensus = self._calculate_consensus_score(list(outputs.values()))
        
        # Decision strength (max probability)
        decision_strength = max(final_probs)
        
        # Reviewer agreement with final decision
        avg_agreement = 0.0
        if outputs:
            agreements = []
            for output in outputs.values():
                agreement = sum(output[i] * final_probs[i] for i in range(len(final_probs)))
                agreements.append(agreement)
            avg_agreement = sum(agreements) / len(agreements)
        
        # Uncertainty (normalized entropy)
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return {
            'confidence': confidence,
            'consensus': consensus,
            'decision_strength': decision_strength,
            'reviewer_agreement': avg_agreement,
            'uncertainty': uncertainty,
            'entropy': entropy
        }
    
    def _analyze_reviewer_contributions(self, outputs: Dict[str, List[float]], 
                                      confidences: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Analyze individual reviewer contributions to the final decision"""
        contributions = {}
        
        for reviewer_id, output in outputs.items():
            confidence = confidences.get(reviewer_id, 0.5)
            weight = self.reviewer_weights.get(reviewer_id, 1.0)
            specialization = self.reviewer_specializations.get(reviewer_id, 'general')
            
            # Calculate this reviewer's decision characteristics
            predicted_class = output.index(max(output))
            decision_certainty = max(output)
            output_entropy = -sum(p * math.log(p + 1e-8) for p in output)
            
            contributions[reviewer_id] = {
                'predicted_class': predicted_class,
                'decision_certainty': decision_certainty,
                'confidence': confidence,
                'weight': weight,
                'specialization': specialization,
                'entropy': output_entropy,
                'contribution_score': confidence * weight
            }
        
        return contributions
    
    def _update_performance_tracking(self, outputs: Dict[str, List[float]], 
                                   confidences: Dict[str, float], 
                                   final_probs: List[float],
                                   metrics: Dict[str, float]):
        """Update performance tracking and adapt strategy if needed"""
        # Update overall performance metrics
        self.performance_metrics['total_decisions'] += 1
        
        # Running averages
        alpha = 0.1  # Learning rate for running averages
        self.performance_metrics['avg_confidence'] = (
            (1 - alpha) * self.performance_metrics['avg_confidence'] + 
            alpha * metrics['confidence']
        )
        self.performance_metrics['avg_consensus'] = (
            (1 - alpha) * self.performance_metrics['avg_consensus'] + 
            alpha * metrics['consensus']
        )
        
        # Update individual reviewer performance
        for reviewer_id, output in outputs.items():
            if reviewer_id in self.reviewer_performance:
                perf = self.reviewer_performance[reviewer_id]
                perf['decisions_made'] += 1
                
                # Update reviewer-specific metrics
                reviewer_confidence = confidences.get(reviewer_id, 0.5)
                perf['avg_confidence'] = (
                    (1 - alpha) * perf['avg_confidence'] + alpha * reviewer_confidence
                )
                
                # Calculate consistency with final decision
                consistency = sum(output[i] * final_probs[i] for i in range(len(final_probs)))
                perf['consistency_score'] = (
                    (1 - alpha) * perf['consistency_score'] + alpha * consistency
                )
        
        # Store decision in history
        decision_record = {
            'final_probs': final_probs.copy(),
            'metrics': metrics.copy(),
            'strategy_used': self.current_strategy,
            'num_reviewers': len(outputs)
        }
        self.decision_history.append(decision_record)
        
        # Consider strategy adaptation
        if (self.performance_metrics['total_decisions'] % self.strategy_adaptation_threshold == 0 and
            len(self.decision_history) >= self.strategy_adaptation_threshold):
            self._consider_strategy_adaptation()
    
    def _consider_strategy_adaptation(self):
        """Consider adapting the combination strategy based on recent performance"""
        if len(self.decision_history) < self.strategy_adaptation_threshold:
            return
        
        # Analyze recent performance by strategy
        recent_decisions = list(self.decision_history)[-self.strategy_adaptation_threshold:]
        strategy_performance = {}
        
        for strategy in self.combination_strategies.keys():
            decisions_with_strategy = [d for d in recent_decisions if d['strategy_used'] == strategy]
            if decisions_with_strategy:
                avg_confidence = sum(d['metrics']['confidence'] for d in decisions_with_strategy) / len(decisions_with_strategy)
                avg_consensus = sum(d['metrics']['consensus'] for d in decisions_with_strategy) / len(decisions_with_strategy)
                strategy_performance[strategy] = 0.6 * avg_confidence + 0.4 * avg_consensus
        
        # Find best performing strategy
        if strategy_performance:
            best_strategy = max(strategy_performance.items(), key=lambda x: x[1])
            current_performance = strategy_performance.get(self.current_strategy, 0.5)
            
            # Switch if new strategy is significantly better
            if (best_strategy[0] != self.current_strategy and 
                best_strategy[1] > current_performance + self.adaptation_sensitivity):
                
                old_strategy = self.current_strategy
                self.current_strategy = best_strategy[0]
                
                if self.demo:
                    print(f"Handler {self.node_id} adapted strategy: {old_strategy} → {self.current_strategy}")
                    print(f"Performance improvement: {current_performance:.3f} → {best_strategy[1]:.3f}")
    
    def _create_default_response(self, reason: str) -> Dict[str, Any]:
        """Create a default response when normal processing fails"""
        default_probs = [1.0 / self.num_output_classes] * self.num_output_classes
        
        return {
            'final_probabilities': default_probs,
            'decision_metrics': {
                'confidence': 0.0,
                'consensus': 0.0,
                'decision_strength': 1.0 / self.num_output_classes,
                'reviewer_agreement': 0.0,
                'uncertainty': 1.0,
                'entropy': math.log(self.num_output_classes)
            },
            'reviewer_contributions': {},
            'processing_metadata': {
                'strategy_used': 'default',
                'num_reviewers': 0,
                'failure_reason': reason,
                'handler_id': self.node_id
            }
        }
    
    def set_combination_strategy(self, strategy: str):
        """Manually set the combination strategy"""
        if strategy in self.combination_strategies:
            self.current_strategy = strategy
            if self.demo:
                print(f"Handler {self.node_id} strategy set to {strategy}")
        else:
            print(f"Unknown strategy: {strategy}. Available: {list(self.combination_strategies.keys())}")
    
    def set_calibration_parameters(self, **kwargs):
        """Set calibration parameters"""
        for key, value in kwargs.items():
            if key in self.calibration_parameters:
                self.calibration_parameters[key] = value
                if self.demo:
                    print(f"Handler {self.node_id} calibration parameter {key} set to {value}")
    
    def get_handler_status(self) -> Dict[str, Any]:
        """Get comprehensive handler status information"""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'position': self.node_position,
            'current_strategy': self.current_strategy,
            'connected_reviewers': len(self.connected_reviewers),
            'performance_metrics': self.performance_metrics.copy(),
            'calibration_parameters': self.calibration_parameters.copy(),
            'reviewer_weights': self.reviewer_weights.copy(),
            'num_output_classes': self.num_output_classes,
            'total_decisions': self.performance_metrics['total_decisions']
        }
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Main processing method for the Handler"""
        if isinstance(input_data, dict):
            # Check if it's reviewer outputs format
            if 'reviewer_outputs' in input_data:
                return self.process_reviewer_outputs(input_data['reviewer_outputs'])
            elif all(isinstance(v, list) for v in input_data.values()):
                # Direct dictionary of reviewer_id -> probabilities
                return self.process_reviewer_outputs(input_data)
            else:
                return self._create_default_response("INVALID_INPUT_FORMAT")
        elif isinstance(input_data, list):
            # List of reviewer outputs - create temporary IDs
            reviewer_outputs = {f"reviewer_{i}": output for i, output in enumerate(input_data)}
            return self.process_reviewer_outputs(reviewer_outputs)
        else:
            return self._create_default_response("UNSUPPORTED_INPUT_TYPE")


class Placeholder:
    """
    Placeholder node that reserves hypercube vertex positions without full processing capabilities.
    Used to maintain dimensional space coverage when insufficient segments are created.
    """
    
    def __init__(self, node_id: str, node_position: Tuple[float, float, float], demo: bool = False):
        self.node_id = node_id
        self.node_position = node_position
        self.demo = demo
        self.node_type = "Placeholder"
        
        # Minimal processing capabilities
        self.times_called = 0
        self.placeholder_active = True
        self.reserved_space = True
        
        # Hypercube vertex information
        self.vertex_assignment = None  # Will store dimensional assignment
        self.hypercube_bounds = None   # Will store spatial bounds
        
        # Minimal weights for compatibility
        self.weights = {
            'Max_random': 0.0,
            'Min_random': 0.0,
            'constant': 0.0
        }
        
        # Connection placeholders (minimal connectivity)
        self.entrance_connections = []
        self.exit_connections = []
        self.generic_connections = []
        
        if demo:
            print(f"Placeholder {node_id} created at {node_position} - reserving hypercube vertex")
    
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0.0):
        """Define weights for the placeholder node (minimal values)"""
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
        return self.weights
    
    def set_vertex_assignment(self, dimensional_assignment: Dict[int, int], hypercube_bounds: List[Tuple[float, float]]):
        """Set the hypercube vertex assignment for this placeholder"""
        self.vertex_assignment = dimensional_assignment
        self.hypercube_bounds = hypercube_bounds
        
        if self.demo:
            print(f"Placeholder {self.node_id} assigned to vertex: {dimensional_assignment}")
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Minimal processing - placeholder nodes don't process data, they just reserve space.
        Returns a default response indicating this is a placeholder.
        """
        self.times_called += 1
        
        if self.demo and self.times_called % 10 == 1:  # Log every 10th call to avoid spam
            print(f"📍 Placeholder {self.node_id} called ({self.times_called} times) - no processing performed")
        
        return {
            'type': 'placeholder_response',
            'placeholder_id': self.node_id,
            'vertex_assignment': self.vertex_assignment,
            'hypercube_bounds': self.hypercube_bounds,
            'times_called': self.times_called,
            'message': 'This is a placeholder node reserving hypercube vertex space',
            'processing_performed': False,
            'reserved_space': self.reserved_space
        }
    
    def can_be_upgraded(self) -> bool:
        """Check if this placeholder can be upgraded to a full segment"""
        # Placeholder nodes can potentially be upgraded to full segments later
        return self.placeholder_active and self.reserved_space
    
    def get_status(self) -> Dict[str, Any]:
        """Get placeholder status information"""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'position': self.node_position,
            'vertex_assignment': self.vertex_assignment,
            'hypercube_bounds': self.hypercube_bounds,
            'times_called': self.times_called,
            'placeholder_active': self.placeholder_active,
            'reserved_space': self.reserved_space,
            'can_upgrade': self.can_be_upgraded()
        }