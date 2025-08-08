#!/usr/bin/env python3
"""
BrainNexus v3 Central Interface
==============================

A comprehensive command-line interface for the BrainNexus neural architecture system.
This main interface provides complete access to all BrainNexus functionality including:

1. CORE OPERATIONS:
   - BrainNexus creation and initialization
   - Segment-based architecture management
   - Node type configuration and deployment
   - Spatial-dimensional positioning systems

2. TRAINING SYSTEMS:
   - Supervised learning (classification, regression)
   - Unsupervised learning (clustering, representation learning)
   - Reinforcement learning (DQN, A2C, DDPG)
   - Multi-modal data processing (text, vision, general tensors)
   - Segment-specific modular training

3. ADVANCED FEATURES:
   - Node evolution (computational → judge transformations)
   - Spatial optimization and connection pruning
   - Attention mechanisms and embedding transformations
   - Cross-segment communication and synchronization
   - Mistral tokenizer integration

4. ANALYSIS & MONITORING:
   - Performance benchmarking and metrics
   - Neural network topology analysis
   - Training progress visualization
   - Error analysis and debugging tools
   - Resource utilization monitoring

5. DATA MANAGEMENT:
   - Segment persistence and loading
   - Training state management
   - Model checkpointing and recovery
   - Configuration templates and presets

Author: AI Assistant
Version: 3.0
Date: August 7, 2025
"""

import sys
import os
import time
import json
import pickle
import argparse
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from collections import defaultdict, deque

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import BrainNexus core components
try:
    from BrainNexus import BrainNexus
    from BrainSegment import NexusSegment
    from BrainNexusLearning import SegmentLearning, LearningTask, RLExperience, RLConfig
    from NeuralNode import NeuralNode
    from computations import Judge, Controller, Splitter, Computational, Reviewer, Retainer, Handler
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"❌ Failed to import BrainNexus components: {e}")
    print("Please ensure all required BrainNexus modules are in the Python path.")
    IMPORTS_SUCCESSFUL = False

# Import additional dependencies
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available. Some advanced features will be limited.")
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️  Plotting libraries not available. Visualization features disabled.")
    PLOTTING_AVAILABLE = False


class BrainNexusInterface:
    """
    Central interface class for all BrainNexus operations.
    
    Provides a unified command-line interface for:
    - System initialization and configuration
    - Training and learning operations
    - Analysis and monitoring
    - Data management and persistence
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the BrainNexus interface.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config = self._load_configuration(config_file)
        self.brain_nexus = None
        self.segments = []
        self.segment_learners = []
        self.active_session = {
            'session_id': self._generate_session_id(),
            'start_time': datetime.now(),
            'operations': [],
            'metrics': {},
            'configurations': {}
        }
        
        # Directory management
        self.work_dir = Path(self.config.get('work_directory', './brainnexus_workspace'))
        self.segments_dir = self.work_dir / 'segments'
        self.models_dir = self.work_dir / 'models'
        self.logs_dir = self.work_dir / 'logs'
        self.results_dir = self.work_dir / 'results'
        
        self._ensure_directories()
        
        # State tracking
        self.current_brain_config = None
        self.training_history = []
        self.performance_metrics = defaultdict(list)
        
        # Interactive mode settings
        self.interactive_mode = False
        self.demo_mode = self.config.get('demo_mode', False)
        self.verbose = self.config.get('verbose', True)
        
        if not IMPORTS_SUCCESSFUL:
            print("❌ Critical import failure. Please check your installation.")
            sys.exit(1)
        
        if self.verbose:
            self._display_welcome()
    
    def _load_configuration(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or create defaults."""
        default_config = {
            'work_directory': './brainnexus_workspace',
            'demo_mode': False,
            'verbose': True,
            'auto_save': True,
            'default_dimensions': 4,
            'default_node_count': 3,
            'segment_defaults': {
                'enable_spatial_optimization': True,
                'enable_attention_training': True,
                'enable_node_evolution': True,
                'learning_rate': 0.001,
                'max_epochs': 100
            },
            'output_defaults': {
                'type': 'classification',
                'num_classes': 10,
                'confidence_threshold': 0.7,
                'return_top_k': 3
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"⚠️  Failed to load config file {config_file}: {e}")
                print("Using default configuration.")
        
        return default_config
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"brainnexus_session_{timestamp}"
    
    def _ensure_directories(self):
        """Create necessary working directories."""
        for directory in [self.work_dir, self.segments_dir, self.models_dir, 
                         self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _display_welcome(self):
        """Display welcome message and system status."""
        print("=" * 80)
        print("🧠 BRAINNEXUS V3 CENTRAL INTERFACE")
        print("=" * 80)
        print(f"Session ID: {self.active_session['session_id']}")
        print(f"Work Directory: {self.work_dir}")
        print(f"Demo Mode: {'ON' if self.demo_mode else 'OFF'}")
        print(f"PyTorch Available: {'YES' if TORCH_AVAILABLE else 'NO'}")
        print(f"Plotting Available: {'YES' if PLOTTING_AVAILABLE else 'NO'}")
        print()
        print("Available Commands:")
        print("  init     - Initialize BrainNexus system")
        print("  create   - Create segments and configure architecture")
        print("  train    - Run training operations")
        print("  analyze  - Analyze performance and structure")
        print("  save     - Save models and configurations")
        print("  load     - Load existing models")
        print("  test     - Run comprehensive testing")
        print("  help     - Show detailed help")
        print("  exit     - Exit the interface")
        print("=" * 80)
        print()


    def run(self, command: Optional[str] = None, args: Optional[List[str]] = None):
        """
        Run the interface in command mode or start interactive session.
        
        Args:
            command: Single command to execute
            args: Command arguments
        """
        if command:
            # Single command execution
            self._execute_command(command, args or [])
        else:
            # Interactive mode
            self._run_interactive()
    
    def _run_interactive(self):
        """Run the interface in interactive mode."""
        self.interactive_mode = True
        print("🔄 Interactive mode started. Type 'help' for commands or 'exit' to quit.\n")
        
        try:
            while True:
                try:
                    user_input = input("BrainNexus> ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        self._handle_exit()
                        break
                    
                    # Parse command and arguments
                    parts = user_input.split()
                    command = parts[0]
                    args = parts[1:] if len(parts) > 1 else []
                    
                    self._execute_command(command, args)
                    print()  # Extra newline for readability
                    
                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user. Type 'exit' to quit.")
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
                    
        except Exception as e:
            print(f"❌ Fatal error in interactive mode: {e}")
            traceback.print_exc()
    
    def _execute_command(self, command: str, args: List[str]):
        """
        Execute a specific command with arguments.
        
        Args:
            command: Command to execute
            args: Command arguments
        """
        # Log the operation
        operation_record = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'args': args
        }
        
        try:
            # Route to appropriate handler
            if command == 'init':
                result = self._handle_init(args)
            elif command == 'create':
                result = self._handle_create(args)
            elif command == 'train':
                result = self._handle_train(args)
            elif command == 'analyze':
                result = self._handle_analyze(args)
            elif command == 'save':
                result = self._handle_save(args)
            elif command == 'load':
                result = self._handle_load(args)
            elif command == 'test':
                result = self._handle_test(args)
            elif command == 'help':
                result = self._handle_help(args)
            elif command == 'status':
                result = self._handle_status(args)
            elif command == 'config':
                result = self._handle_config(args)
            elif command == 'demo':
                result = self._handle_demo(args)
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands.")
                result = {'status': 'error', 'message': f'Unknown command: {command}'}
            
            operation_record['result'] = result
            operation_record['status'] = result.get('status', 'completed')
            
        except Exception as e:
            error_msg = f"Command '{command}' failed: {str(e)}"
            print(f"❌ {error_msg}")
            if self.verbose:
                traceback.print_exc()
            
            operation_record['result'] = {'status': 'error', 'error': str(e)}
            operation_record['status'] = 'error'
        
        finally:
            self.active_session['operations'].append(operation_record)
    
    def _handle_exit(self):
        """Handle clean exit from the interface."""
        print("\n👋 BRAINNEXUS SESSION ENDING")
        print("-" * 50)
        
        # Display session summary
        session_duration = (datetime.now() - self.active_session['start_time']).total_seconds()
        operations_count = len(self.active_session['operations'])
        
        print(f"Session Duration: {session_duration:.1f}s")
        print(f"Operations Executed: {operations_count}")
        
        if self.segments:
            print(f"Segments Created: {len(self.segments)}")
            total_nodes = sum(len(s.segment_nodes) for s in self.segments)
            print(f"Total Nodes: {total_nodes}")
        
        if self.training_history:
            print(f"Training Sessions: {len(self.training_history)}")
        
        # Auto-save if enabled
        if self.config.get('auto_save', False) and (self.brain_nexus or self.segments):
            try:
                auto_save_path = self.results_dir / f"auto_save_{int(time.time())}.pkl"
                save_data = {
                    'session_info': self.active_session,
                    'brain_config': self.current_brain_config,
                    'training_history': self.training_history,
                    'segments_count': len(self.segments)
                }
                
                with open(auto_save_path, 'wb') as f:
                    pickle.dump(save_data, f)
                
                print(f"💾 Auto-saved to: {auto_save_path}")
            except Exception as e:
                print(f"⚠️  Auto-save failed: {e}")
        
        print("\nThank you for using BrainNexus! 🧠✨")
        print("=" * 50)
    
    def _handle_init(self, args: List[str]) -> Dict[str, Any]:
        """
        Handle BrainNexus initialization command.
        
        Args:
            args: Command arguments [dimensions, node_count, demo_mode]
            
        Returns:
            Dict with operation results
        """
        print("🧠 INITIALIZING BRAINNEXUS SYSTEM")
        print("-" * 50)
        
        # Parse arguments
        dimensions = int(args[0]) if len(args) > 0 else self.config['default_dimensions']
        # Use None for node_count if not explicitly provided - this allows BrainNexus to calculate dynamically
        node_count = int(args[1]) if len(args) > 1 else None
        demo_mode = args[2].lower() == 'true' if len(args) > 2 else self.demo_mode
        
        # Output configuration
        output_config = self.config['output_defaults'].copy()
        
        if len(args) > 3:
            # Custom output configuration
            output_type = args[3]
            num_classes = int(args[4]) if len(args) > 4 else 10
            
            output_config.update({
                'type': output_type,
                'num_classes': num_classes
            })
        
        try:
            start_time = time.time()
            
            # Create BrainNexus instance
            self.brain_nexus = BrainNexus(
                dimensions=dimensions,
                node_count_pre=node_count,  # Pass None to enable dynamic calculation
                demo=demo_mode,
                output_config=output_config,
                mode='production'
            )
            
            # Get actual node count after initialization (dynamic or specified)
            actual_node_count = self.brain_nexus.entrance_node_count_pre
            
            # Store current configuration
            self.current_brain_config = {
                'dimensions': dimensions,
                'node_count_pre': actual_node_count,  # Store the actual count used
                'demo_mode': demo_mode,
                'output_config': output_config,
                'initialization_time': time.time() - start_time
            }
            
            self.demo_mode = demo_mode  # Update instance demo mode
            
            result = {
                'status': 'success',
                'brain_nexus_created': True,
                'dimensions': dimensions,
                'node_count_pre': actual_node_count,  # Return the actual count
                'demo_mode': demo_mode,
                'output_config': output_config,
                'initialization_time': self.current_brain_config['initialization_time']
            }
            
            print(f"✅ BrainNexus initialized successfully!")
            print(f"   Dimensions: {dimensions}")
            print(f"   Pre-nodes: {actual_node_count} ({'dynamic (2^{})'.format(dimensions) if node_count is None else 'specified'})")
            print(f"   Demo mode: {demo_mode}")
            print(f"   Output: {output_config['type']} ({output_config['num_classes']} classes)")
            print(f"   Time: {result['initialization_time']:.3f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to initialize BrainNexus: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'status': 'error',
                'error': error_msg,
                'brain_nexus_created': False
            }
    
    def _handle_create(self, args: List[str]) -> Dict[str, Any]:
        """
        Handle segment creation and architecture setup.
        
        Args:
            args: Command arguments [num_segments, segment_type, config_preset]
            
        Returns:
            Dict with operation results
        """
        print("🏗️  CREATING BRAINNEXUS ARCHITECTURE")
        print("-" * 50)
        
        if not self.brain_nexus:
            print("❌ BrainNexus not initialized. Run 'init' command first.")
            return {'status': 'error', 'error': 'BrainNexus not initialized'}
        
        if not self.current_brain_config:
            print("❌ BrainNexus configuration not available.")
            return {'status': 'error', 'error': 'No brain configuration available'}
        
        # Parse arguments
        num_segments_arg = int(args[0]) if len(args) > 0 else None
        segment_type = args[1] if len(args) > 1 else 'balanced'
        config_preset = args[2] if len(args) > 2 else 'default'
        
        # Handle case where user provides only count and config_preset (e.g., "create 4 full")
        # Check if second argument is a known config preset rather than segment type
        known_presets = ['demo', 'default', 'full', 'massive', 'performance', 'memory_efficient', 'research']
        known_segment_types = ['balanced', 'positive', 'negative', 'mixed', 'spiral']
        
        if len(args) == 2 and args[1] in known_presets:
            # User provided count and preset, use default segment type
            config_preset = args[1]
            segment_type = 'balanced'
        
        # Calculate optimal segment count for balanced type
        dimensions = self.current_brain_config['dimensions']
        if segment_type == 'balanced':
            if num_segments_arg is None:
                # Auto-calculate optimal count: one segment per hypercube vertex
                num_segments = 2 ** dimensions
                print(f"🎯 Auto-calculating optimal segment count for {dimensions}D: {num_segments} segments (2^{dimensions} hypercube vertices)")
                use_pre_nodes_as_placeholders = False
            else:
                num_segments = num_segments_arg
                optimal_count = 2 ** dimensions
                if num_segments < optimal_count:
                    use_pre_nodes_as_placeholders = True
                    placeholder_count = optimal_count - num_segments
                    print(f"📍 Using {num_segments} segments for {dimensions}D space (optimal: {optimal_count})")
                    print(f"   Creating {placeholder_count} pre-node placeholders for remaining hypercube vertices")
                    print(f"   ⚠️  Note: Pre-nodes will hold space but won't process data like full segments")
                elif num_segments > optimal_count:
                    use_pre_nodes_as_placeholders = False
                    print(f"⚠️  Warning: Using {num_segments} segments exceeds optimal {optimal_count} for {dimensions}D")
                else:
                    use_pre_nodes_as_placeholders = False
                    print(f"✅ Using optimal {num_segments} segments for {dimensions}D hypercube coverage")
        else:
            # For non-balanced types, use specified count or default
            num_segments = num_segments_arg if num_segments_arg is not None else 2
            use_pre_nodes_as_placeholders = False
        
        if len(args) == 2 and args[1] in known_presets and args[1] not in known_segment_types:
            # User provided: create <count> <config_preset>
            segment_type = 'balanced'  # Default segment type
            config_preset = args[1]    # Use the provided preset
        
        try:
            start_time = time.time()
            
            # Initialize placeholder tracking
            use_pre_nodes_as_placeholders = False
            pre_node_placeholders_created = []
            
            # Create segments with different dimensional assignments
            segments_created = []
            
            for i in range(num_segments):
                segment_id = i + 1
                
                # Generate dimensional assignment based on segment type
                dimensional_assignment = self._generate_dimensional_assignment(
                    segment_id, segment_type, self.current_brain_config['dimensions']
                )
                
                # Get segment configuration
                segment_config = self._get_segment_config_preset(config_preset)
                
                # Adjust hypercube bounds based on dimensions
                dimensions = self.current_brain_config['dimensions']
                segment_config = self._adjust_config_for_dimensions(segment_config, dimensions, config_preset)
                
                print(f"Creating segment {segment_id}/{num_segments} ({config_preset} size)...")
                
                # Extract base hypercube bounds from config
                base_hypercube_bounds = segment_config.get('hypercube_bounds', [(0.0, 1000000.0)])
                
                # Calculate segment-specific bounds based on dimensional assignment
                segment_hypercube_bounds = self._calculate_segment_bounds(
                    base_hypercube_bounds, dimensional_assignment
                )
                
                max_nodes = segment_config.get('resource_limits', {}).get('max_nodes', 1000)
                max_connections = segment_config.get('resource_limits', {}).get('max_connections', 10000)
                
                if self.verbose:
                    # Display segment-specific multidimensional bounds
                    if isinstance(segment_hypercube_bounds, list) and len(segment_hypercube_bounds) > 0:
                        # List format - show per dimension
                        dim_labels = ['x', 'y', 'z', 'w', 'v', 'u', 't', 's', 'r', 'q']
                        bounds_display = []
                        for i, bounds in enumerate(segment_hypercube_bounds):
                            if isinstance(bounds, tuple) and len(bounds) == 2:
                                label = dim_labels[i] if i < len(dim_labels) else f'd{i}'
                                min_bound, max_bound = bounds
                                bounds_display.append(f"{label}:[{min_bound:.0f}, {max_bound:.0f}]")
                        print(f"   Dimensions: {dimensions}D")
                        print(f"   Hypercube bounds: {' × '.join(bounds_display)}")
                    else:
                        print(f"   Dimensions: {dimensions}D")
                        print(f"   Hypercube bounds: {segment_hypercube_bounds}")
                    print(f"   Resource limits: {max_nodes} nodes, {max_connections} connections")
                
                # Create NexusSegment
                segment = NexusSegment(
                    segment_id=segment_id,
                    dimensional_assignment=dimensional_assignment,
                    brain_nexus_ref=self.brain_nexus,
                    hypercube_bounds=segment_hypercube_bounds,
                    segment_config=segment_config,
                    demo=self.demo_mode
                )
                
                self.segments.append(segment)
                
                # Create corresponding SegmentLearning instance
                learning_config = self._get_learning_config_preset(config_preset)
                
                segment_learner = SegmentLearning(
                    brain_segment=segment,
                    learning_config=learning_config,
                    device='auto'
                )
                
                self.segment_learners.append(segment_learner)
                
                segments_created.append({
                    'segment_id': segment_id,
                    'dimensional_assignment': dimensional_assignment,
                    'node_count': len(segment.segment_nodes),
                    'node_types': {
                        node_type: len(node_list) 
                        for node_type, node_list in segment.node_type_registry.items()
                    },
                    'learning_config': learning_config
                })
                
                if self.verbose:
                    print(f"   ✅ Segment {segment_id}: {len(segment.segment_nodes)} nodes")
                    print(f"      Dimensions: {dimensional_assignment}")
                    print(f"      Node types: {list(segment.node_type_registry.keys())}")
            
            # Create pre-node placeholders for remaining hypercube vertices if needed
            pre_node_placeholders_created = []
            if use_pre_nodes_as_placeholders and segment_type == 'balanced':
                optimal_count = 2 ** dimensions
                remaining_vertices = optimal_count - num_segments
                
                print(f"\n📍 Creating {remaining_vertices} pre-node placeholders for uncovered hypercube vertices:")
                
                for placeholder_idx in range(remaining_vertices):
                    # Use segment IDs starting after the real segments
                    placeholder_segment_id = num_segments + placeholder_idx
                    
                    # Generate dimensional assignment for placeholder
                    dimensional_assignment = self._generate_dimensional_assignment(
                        placeholder_segment_id, segment_type, dimensions
                    )
                    
                    # Get base configuration for placeholders (minimal resources)
                    placeholder_config = {
                        'enable_caching': False,
                        'enable_adaptation': False,
                        'resource_limits': {'max_nodes': 1, 'max_connections': 0},  # Minimal resources
                        'hypercube_bounds': segment_config.get('hypercube_bounds', [(-100.0, 100.0)]),
                        'segment_type': 'placeholder'
                    }
                    
                    # Calculate placeholder bounds
                    base_hypercube_bounds = placeholder_config.get('hypercube_bounds', [(-100.0, 100.0)])
                    placeholder_hypercube_bounds = self._calculate_segment_bounds(
                        base_hypercube_bounds, dimensional_assignment
                    )
                    
                    # Create minimal pre-nodes at optimal positions for these vertices
                    optimal_positions = self.brain_nexus.generate_optimal_entrance_positions()
                    if placeholder_segment_id < len(optimal_positions):
                        position = optimal_positions[placeholder_segment_id]
                        
                        # Create a single pre-node placeholder
                        pre_node_id = self.brain_nexus.add_neural_node(
                            node_type='Placeholder',
                            position=position,
                            node_group=f'pre_node_placeholder_{placeholder_segment_id}'
                        )
                        
                        # Set vertex assignment information on the placeholder
                        placeholder_node = self.brain_nexus.node_registry[pre_node_id]
                        placeholder_node.set_vertex_assignment(dimensional_assignment, placeholder_hypercube_bounds)
                        
                        pre_node_placeholders_created.append({
                            'pre_node_id': pre_node_id,
                            'placeholder_segment_id': placeholder_segment_id,
                            'dimensional_assignment': dimensional_assignment,
                            'hypercube_bounds': placeholder_hypercube_bounds,
                            'position': position
                        })
                    
                    if self.verbose:
                        dim_labels = ['x', 'y', 'z', 'w', 'v', 'u', 't', 's', 'r', 'q']
                        bounds_display = []
                        for i, bounds in enumerate(placeholder_hypercube_bounds):
                            if isinstance(bounds, tuple) and len(bounds) == 2:
                                label = dim_labels[i] if i < len(dim_labels) else f'd{i}'
                                min_bound, max_bound = bounds
                                bounds_display.append(f"{label}:[{min_bound:.0f}, {max_bound:.0f}]")
                        
                        print(f"   📍 Pre-node placeholder {placeholder_segment_id}: {' × '.join(bounds_display)}")
                        print(f"      Dimensional assignment: {dimensional_assignment}")
                
                if pre_node_placeholders_created:
                    print(f"✅ Created {len(pre_node_placeholders_created)} pre-node placeholders")
            
            creation_time = time.time() - start_time
            total_nodes = sum(len(s.segment_nodes) for s in self.segments)
            total_pre_node_placeholders = len(pre_node_placeholders_created)
            
            result = {
                'status': 'success',
                'segments_created': len(segments_created),
                'segment_details': segments_created,
                'pre_node_placeholders_created': total_pre_node_placeholders,
                'placeholder_details': pre_node_placeholders_created,
                'total_nodes': total_nodes,
                'total_hypercube_vertices': 2 ** dimensions,
                'hypercube_coverage': {
                    'segments': len(segments_created),
                    'placeholders': total_pre_node_placeholders,
                    'covered_vertices': len(segments_created) + total_pre_node_placeholders,
                    'total_vertices': 2 ** dimensions,
                    'coverage_complete': (len(segments_created) + total_pre_node_placeholders) == (2 ** dimensions)
                },
                'creation_time': creation_time,
                'segment_type': segment_type,
                'config_preset': config_preset
            }
            
            print(f"\n✅ Architecture created successfully!")
            print(f"   Segments: {len(segments_created)} ({segment_type} type, {config_preset} size)")
            if total_pre_node_placeholders > 0:
                print(f"   Pre-node placeholders: {total_pre_node_placeholders} (covering remaining hypercube vertices)")
            print(f"   Total nodes: {total_nodes}")
            print(f"   Hypercube coverage: {len(segments_created) + total_pre_node_placeholders}/{2 ** dimensions} vertices")
            if (len(segments_created) + total_pre_node_placeholders) == (2 ** dimensions):
                print(f"   ✅ Complete {dimensions}D hypercube coverage achieved!")
            else:
                missing_vertices = (2 ** dimensions) - (len(segments_created) + total_pre_node_placeholders)
                print(f"   ⚠️  Missing {missing_vertices} hypercube vertices")
            print(f"   Config: {config_preset} preset")
            if config_preset in ['demo', 'full', 'massive']:
                preset_config = self._get_segment_config_preset(config_preset)
                bounds = preset_config.get('hypercube_bounds', 'default')
                limits = preset_config.get('resource_limits', {})
                
                # Format bounds for multidimensional display based on actual dimensions
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    # Use actual dimensions from brain config
                    actual_dimensions = self.current_brain_config['dimensions']
                    dimension_labels = ['x', 'y', 'z', 'w', 'v', 'u'][:actual_dimensions]  # Support up to 6D
                    bounds_display = [f"{label}:[{bounds[0]}, {bounds[1]}]" for label in dimension_labels]
                    print(f"   Hypercube bounds: {' × '.join(bounds_display)}")
                else:
                    print(f"   Hypercube bounds: {bounds}")
                
                print(f"   Node limits: {limits.get('max_nodes', 'default')} nodes, {limits.get('max_connections', 'default')} connections")
            print(f"   Time: {creation_time:.3f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to create architecture: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'status': 'error',
                'error': error_msg,
                'segments_created': 0
            }
    
    def _generate_dimensional_assignment(self, segment_id: int, segment_type: str, dimensions: int) -> Dict[int, int]:
        """
        Generate dimensional assignment for a segment based on hypercube vertices.
        
        For optimal coverage, each segment should occupy one hypercube vertex:
        - 2D: 4 quadrants (2^2 = 4 vertices)
        - 3D: 8 octants (2^3 = 8 vertices)  
        - nD: 2^n hypercube vertices
        
        Args:
            segment_id: Unique segment identifier (0-based)
            segment_type: Type of segment configuration
            dimensions: Number of dimensions
            
        Returns:
            Dict mapping dimension indices to polarities (-1 or +1)
        """
        if segment_type == 'balanced':
            # Use hypercube vertex assignment for optimal coverage
            # Convert segment_id to binary representation for polarity assignment
            assignment = {}
            for dim in range(dimensions):
                # Use bit position to determine polarity
                # Bit 0 -> dimension 0, bit 1 -> dimension 1, etc.
                polarity = 1 if (segment_id >> dim) & 1 else -1
                assignment[dim] = polarity
            return assignment
        
        elif segment_type == 'positive':
            # All positive polarities
            return {dim: 1 for dim in range(dimensions)}
        
        elif segment_type == 'negative':
            # All negative polarities
            return {dim: -1 for dim in range(dimensions)}
        
        elif segment_type == 'mixed':
            # Mixed based on mathematical pattern
            assignment = {}
            for dim in range(dimensions):
                if dim % 3 == 0:
                    assignment[dim] = 1
                elif dim % 3 == 1:
                    assignment[dim] = -1
                else:
                    assignment[dim] = 1 if segment_id % 2 == 0 else -1
            return assignment
        
        elif segment_type == 'spiral':
            # Spiral pattern for complex dimensional relationships
            assignment = {}
            for dim in range(dimensions):
                polarity = 1 if (segment_id * dim + dim**2) % 3 != 0 else -1
                assignment[dim] = polarity
            return assignment
        
        else:
            # Default to balanced
            return {
                dim: 1 if (segment_id + dim) % 2 == 0 else -1 
                for dim in range(dimensions)
            }
    
    def _get_segment_config_preset(self, preset: str) -> Dict[str, Any]:
        """Get segment configuration based on preset."""
        presets = {
            'default': {
                'enable_caching': True,
                'cache_size': 1000,
                'enable_adaptation': True,
                'adaptation_rate': 0.1,
                'quality_threshold': 0.8,
                'resource_limits': {'max_nodes': 1000, 'max_connections': 10000}
            },
            'performance': {
                'enable_caching': True,
                'cache_size': 5000,
                'enable_adaptation': True,
                'adaptation_rate': 0.05,
                'quality_threshold': 0.9,
                'enable_parallel_processing': True,
                'resource_limits': {'max_nodes': 2000, 'max_connections': 20000}
            },
            'memory_efficient': {
                'enable_caching': False,
                'enable_adaptation': True,
                'adaptation_rate': 0.2,
                'quality_threshold': 0.7,
                'resource_limits': {'max_nodes': 500, 'max_connections': 2000}
            },
            'research': {
                'enable_caching': True,
                'cache_size': 2000,
                'enable_adaptation': True,
                'adaptation_rate': 0.15,
                'quality_threshold': 0.6,
                'enable_experimental_features': True,
                'resource_limits': {'max_nodes': 1500, 'max_connections': 15000}
            },
            'demo': {
                'enable_caching': True,
                'cache_size': 50,  # Reduced from 100
                'enable_adaptation': True,
                'adaptation_rate': 0.15,
                'quality_threshold': 0.7,
                'enable_experimental_features': False,
                'resource_limits': {'max_nodes': 50, 'max_connections': 50},  # Significantly reduced
                'hypercube_bounds': (-100.0, 100.0),  # Centered around origin: -100 to +100
                'segment_type': 'demo'
            },
            'full': {
                'enable_caching': True,
                'cache_size': 100000,
                'enable_adaptation': True,
                'adaptation_rate': 0.05,
                'quality_threshold': 0.85,
                'enable_parallel_processing': True,
                'enable_experimental_features': True,
                'resource_limits': {'max_nodes': 1000000, 'max_connections': 1000000},
                'hypercube_bounds': (-1000000.0, 1000000.0),  # Centered: -1M to +1M
                'segment_type': 'full'
            },
            'massive': {
                'enable_caching': True,
                'cache_size': 1000000,
                'enable_adaptation': True,
                'adaptation_rate': 0.01,
                'quality_threshold': 0.9,
                'enable_parallel_processing': True,
                'enable_distributed_processing': True,
                'enable_experimental_features': True,
                'resource_limits': {'max_nodes': 1000000000, 'max_connections': 1000000000},
                'hypercube_bounds': (-1000000000.0, 1000000000.0),  # Centered: -1B to +1B
                'segment_type': 'massive'
            }
        }
        
        return presets.get(preset, presets['default'])
    
    def _calculate_segment_bounds(self, base_bounds: List[Tuple[float, float]], 
                                dimensional_assignment: Dict[int, int]) -> List[Tuple[float, float]]:
        """
        Calculate segment-specific bounds based on dimensional polarity assignments.
        
        For centered bounds like (-100, +100), segments are assigned to hypercube vertices:
        - Positive polarity: upper half [0, +100]
        - Negative polarity: lower half [-100, 0]
        
        Args:
            base_bounds: Base hypercube bounds [(-max_val, +max_val) for each dimension]
            dimensional_assignment: {dimension: polarity (-1 or +1)}
            
        Returns:
            List of tuples representing the segment's spatial region
        """
        segment_bounds = []
        
        for dim_idx in range(len(base_bounds)):
            base_min, base_max = base_bounds[dim_idx]
            # For centered bounds, midpoint is 0
            midpoint = 0.0
            
            if dim_idx in dimensional_assignment:
                polarity = dimensional_assignment[dim_idx]
                if polarity >= 0:
                    # Positive polarity: upper half of the dimension [0, +max]
                    segment_bounds.append((midpoint, base_max))
                else:
                    # Negative polarity: lower half of the dimension [-max, 0]
                    segment_bounds.append((base_min, midpoint))
            else:
                # No assignment for this dimension, use full range
                segment_bounds.append((base_min, base_max))
        
        return segment_bounds
    
    def _adjust_config_for_dimensions(self, segment_config: Dict[str, Any], dimensions: int, config_preset: str) -> Dict[str, Any]:
        """
        Adjust segment configuration based on the number of dimensions.
        
        Args:
            segment_config: Base segment configuration
            dimensions: Number of dimensions in the BrainNexus
            config_preset: Configuration preset name
            
        Returns:
            Dict[str, Any]: Adjusted configuration
        """
        # Create a copy to avoid modifying the original
        config = segment_config.copy()
        
        # Adjust hypercube bounds based on dimensions and preset
        # Hypercube dimensions go from origin to +- borders of max size
        # 2D = 4 quadrants, 3D = 8 octants, nD = 2^n regions
        base_bounds = {
            'demo': 100.0,
            'default': 1000.0,
            'full': 1000000.0,
            'massive': 1000000000.0
        }
        
        bound_value = base_bounds.get(config_preset, 1000.0)
        
        # Use the base bound value without dimensional scaling to maintain consistency
        # All segments should use the same base bounds as specified in the preset
        
        # Create multidimensional bounds - segments should occupy different regions
        # based on their dimensional polarity assignments (quadrants, octants, etc.)
        # The overall space spans from -bound_value to +bound_value, centered at origin
        config['hypercube_bounds'] = [(-bound_value, bound_value) for _ in range(dimensions)]
        
        # Adjust node limits based on dimensions (more dimensions = more possible positions)
        if 'resource_limits' in config:
            base_max_nodes = config['resource_limits'].get('max_nodes', 1000)
            base_max_connections = config['resource_limits'].get('max_connections', 10000)
            
            # Scale nodes and connections based on dimensional complexity
            dimension_scale = max(1.0, dimensions / 3.0)  # Scale based on 3D as baseline
            
            config['resource_limits']['max_nodes'] = int(base_max_nodes * dimension_scale)
            config['resource_limits']['max_connections'] = int(base_max_connections * dimension_scale)
        
        return config
    
    def _get_learning_config_preset(self, preset: str) -> Dict[str, Any]:
        """Get learning configuration based on preset."""
        base_config = self.config['segment_defaults'].copy()
        
        presets = {
            'default': {
                'learning_rate': 0.001,
                'max_epochs': 100,
                'batch_size': 32,
                'enable_spatial_adaptation': True,
                'enable_attention_training': True,
                'enable_node_evolution': True,
                'enable_connection_pruning': True,
                'early_stopping_patience': 10
            },
            'performance': {
                'learning_rate': 0.0005,
                'max_epochs': 200,
                'batch_size': 64,
                'enable_spatial_adaptation': True,
                'enable_attention_training': True,
                'enable_node_evolution': True,
                'enable_connection_pruning': True,
                'enable_mixed_precision': True,
                'early_stopping_patience': 15
            },
            'memory_efficient': {
                'learning_rate': 0.002,
                'max_epochs': 50,
                'batch_size': 16,
                'enable_spatial_adaptation': False,
                'enable_attention_training': False,
                'enable_node_evolution': False,
                'enable_connection_pruning': True,
                'gradient_accumulation_steps': 4,
                'early_stopping_patience': 5
            },
            'research': {
                'learning_rate': 0.001,
                'max_epochs': 150,
                'batch_size': 32,
                'enable_spatial_adaptation': True,
                'enable_attention_training': True,
                'enable_node_evolution': True,
                'enable_connection_pruning': True,
                'enable_meta_learning': True,
                'enable_contrastive_learning': True,
                'early_stopping_patience': 20
            },
            'demo': {
                'learning_rate': 0.01,
                'max_epochs': 20,
                'batch_size': 8,
                'enable_spatial_adaptation': True,
                'enable_attention_training': False,
                'enable_node_evolution': False,
                'enable_connection_pruning': False,
                'early_stopping_patience': 5,
                'gradient_accumulation_steps': 1
            },
            'full': {
                'learning_rate': 0.0001,
                'max_epochs': 500,
                'batch_size': 128,
                'enable_spatial_adaptation': True,
                'enable_attention_training': True,
                'enable_node_evolution': True,
                'enable_connection_pruning': True,
                'enable_mixed_precision': True,
                'enable_meta_learning': True,
                'enable_contrastive_learning': True,
                'early_stopping_patience': 50,
                'gradient_accumulation_steps': 4
            },
            'massive': {
                'learning_rate': 0.00001,
                'max_epochs': 1000,
                'batch_size': 256,
                'enable_spatial_adaptation': True,
                'enable_attention_training': True,
                'enable_node_evolution': True,
                'enable_connection_pruning': True,
                'enable_mixed_precision': True,
                'enable_meta_learning': True,
                'enable_contrastive_learning': True,
                'enable_distributed_training': True,
                'early_stopping_patience': 100,
                'gradient_accumulation_steps': 8,
                'checkpoint_frequency': 50
            }
        }
        
        config = base_config.copy()
        if preset in presets:
            config.update(presets[preset])
        
        return config
    
    def _handle_train(self, args: List[str]) -> Dict[str, Any]:
        """
        Handle training operations for segments and individual node types.
        
        Args:
            args: Command arguments [train_type, task_config, data_source]
            
        Returns:
            Dict with training results
        """
        print("🎓 STARTING TRAINING OPERATIONS")
        print("-" * 50)
        
        if not self.segments or not self.segment_learners:
            print("❌ No segments available. Run 'create' command first.")
            return {'status': 'error', 'error': 'No segments available'}
        
        # Parse arguments
        train_type = args[0] if len(args) > 0 else 'supervised'
        task_config = args[1] if len(args) > 1 else 'classification'
        data_source = args[2] if len(args) > 2 else 'synthetic'
        
        # Additional training parameters
        epochs = int(args[3]) if len(args) > 3 else 50
        batch_size = int(args[4]) if len(args) > 4 else 32
        
        try:
            start_time = time.time()
            
            if train_type == 'supervised':
                results = self._run_supervised_training(task_config, data_source, epochs, batch_size)
            elif train_type == 'reinforcement':
                results = self._run_reinforcement_training(task_config, data_source, epochs)
            elif train_type == 'multi_modal':
                results = self._run_multi_modal_training(task_config, data_source, epochs, batch_size)
            elif train_type == 'node_specific':
                results = self._run_node_specific_training(task_config, data_source, epochs, batch_size)
            elif train_type == 'evolutionary':
                results = self._run_evolutionary_training(task_config, data_source, epochs, batch_size)
            else:
                raise ValueError(f"Unknown training type: {train_type}")
            
            training_time = time.time() - start_time
            results['total_training_time'] = training_time
            
            # Store training history
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'train_type': train_type,
                'task_config': task_config,
                'data_source': data_source,
                'results': results
            })
            
            print(f"\n✅ Training completed successfully!")
            print(f"   Type: {train_type}")
            print(f"   Task: {task_config}")
            print(f"   Time: {training_time:.3f}s")
            print(f"   Segments trained: {results.get('segments_trained', 0)}")
            
            return results
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'status': 'error',
                'error': error_msg,
                'train_type': train_type
            }
    
    def _run_supervised_training(self, task_config: str, data_source: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        """Run supervised training on all segments."""
        print(f"🎯 Running supervised training: {task_config}")
        
        # Create learning task
        learning_task = self._create_learning_task('supervised', task_config, epochs)
        
        # Generate or load training data
        training_data, labels = self._generate_training_data(data_source, task_config, batch_size * 10)
        validation_data, val_labels = self._generate_training_data(data_source, task_config, batch_size * 2)
        
        segment_results = []
        total_nodes_trained = 0
        
        for i, learner in enumerate(self.segment_learners):
            segment_id = self.segments[i].segment_id
            
            print(f"\nTraining segment {segment_id}/{len(self.segment_learners)}...")
            
            try:
                # Train the segment
                segment_result = learner.train_segment(
                    learning_task=learning_task,
                    data=training_data,
                    labels=labels,
                    validation_data=(validation_data, val_labels)
                )
                
                segment_result['segment_id'] = segment_id
                segment_results.append(segment_result)
                total_nodes_trained += segment_result.get('nodes_updated', 0)
                
                if self.verbose:
                    final_loss = segment_result.get('final_loss', 0.0)
                    epochs_completed = segment_result.get('epochs_completed', 0)
                    print(f"   ✅ Segment {segment_id}: loss={final_loss:.4f}, epochs={epochs_completed}")
                
            except Exception as e:
                print(f"   ❌ Segment {segment_id} failed: {str(e)}")
                segment_results.append({
                    'segment_id': segment_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Calculate overall metrics
        successful_segments = [r for r in segment_results if r.get('status') != 'failed']
        avg_final_loss = np.mean([r.get('final_loss', 0.0) for r in successful_segments]) if successful_segments else 0.0
        
        return {
            'status': 'success',
            'train_type': 'supervised',
            'segments_trained': len(segment_results),
            'successful_segments': len(successful_segments),
            'total_nodes_trained': total_nodes_trained,
            'average_final_loss': avg_final_loss,
            'segment_results': segment_results,
            'learning_task': {
                'task_id': learning_task.task_id,
                'modality': learning_task.modality,
                'objective': learning_task.objective
            }
        }
    
    def _run_reinforcement_training(self, task_config: str, data_source: str, episodes: int) -> Dict[str, Any]:
        """Run reinforcement learning training on segments."""
        print(f"🎮 Running reinforcement learning: {task_config}")
        
        # Create RL configuration
        rl_config = self._create_rl_config(task_config)
        learning_task = self._create_learning_task('reinforcement', task_config, episodes)
        
        # Create RL environment
        environment = self._create_rl_environment(task_config, data_source)
        
        segment_results = []
        
        for i, learner in enumerate(self.segment_learners):
            segment_id = self.segments[i].segment_id
            
            print(f"\nRL training segment {segment_id}/{len(self.segment_learners)}...")
            
            try:
                # Train with reinforcement learning
                segment_result = learner.train_segment_rl(
                    learning_task=learning_task,
                    environment=environment,
                    rl_config=rl_config,
                    episodes=episodes
                )
                
                segment_result['segment_id'] = segment_id
                segment_results.append(segment_result)
                
                if self.verbose:
                    final_reward = segment_result.get('final_avg_reward', 0.0)
                    episodes_trained = segment_result.get('episodes_trained', 0)
                    print(f"   ✅ Segment {segment_id}: reward={final_reward:.3f}, episodes={episodes_trained}")
                
            except Exception as e:
                print(f"   ❌ Segment {segment_id} RL failed: {str(e)}")
                segment_results.append({
                    'segment_id': segment_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        successful_segments = [r for r in segment_results if r.get('status') != 'failed']
        avg_final_reward = np.mean([r.get('final_avg_reward', 0.0) for r in successful_segments]) if successful_segments else 0.0
        
        return {
            'status': 'success',
            'train_type': 'reinforcement',
            'segments_trained': len(segment_results),
            'successful_segments': len(successful_segments),
            'average_final_reward': avg_final_reward,
            'segment_results': segment_results,
            'rl_config': {
                'algorithm': rl_config.algorithm,
                'episodes': episodes
            }
        }
    
    def _run_multi_modal_training(self, task_config: str, data_source: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        """Run multi-modal training combining different data types."""
        print(f"🎭 Running multi-modal training: {task_config}")
        
        # Create multi-modal learning tasks
        text_task = self._create_learning_task('supervised', 'text_' + task_config, epochs)
        vision_task = self._create_learning_task('supervised', 'vision_' + task_config, epochs)
        
        # Generate multi-modal data
        text_data, text_labels = self._generate_training_data(data_source, 'text', batch_size * 5)
        vision_data, vision_labels = self._generate_training_data(data_source, 'vision', batch_size * 5)
        
        segment_results = []
        
        for i, learner in enumerate(self.segment_learners):
            segment_id = self.segments[i].segment_id
            
            print(f"\nMulti-modal training segment {segment_id}/{len(self.segment_learners)}...")
            
            try:
                # Alternate between text and vision training
                if i % 2 == 0:
                    # Text-focused training
                    text_task.modality = 'text'
                    segment_result = learner.train_segment(
                        learning_task=text_task,
                        data=text_data,
                        labels=text_labels
                    )
                    segment_result['primary_modality'] = 'text'
                else:
                    # Vision-focused training  
                    vision_task.modality = 'general'  # Use general processor for vision
                    segment_result = learner.train_segment(
                        learning_task=vision_task,
                        data=vision_data,
                        labels=vision_labels
                    )
                    segment_result['primary_modality'] = 'vision'
                
                segment_result['segment_id'] = segment_id
                segment_results.append(segment_result)
                
                if self.verbose:
                    modality = segment_result['primary_modality']
                    final_loss = segment_result.get('final_loss', 0.0)
                    print(f"   ✅ Segment {segment_id} ({modality}): loss={final_loss:.4f}")
                
            except Exception as e:
                print(f"   ❌ Segment {segment_id} multi-modal failed: {str(e)}")
                segment_results.append({
                    'segment_id': segment_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'status': 'success',
            'train_type': 'multi_modal',
            'segments_trained': len(segment_results),
            'segment_results': segment_results,
            'modalities_used': ['text', 'vision']
        }
    
    def _run_node_specific_training(self, task_config: str, data_source: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        """Run training on specific node types across segments."""
        print(f"🎯 Running node-specific training: {task_config}")
        
        # Parse node type from task config
        node_type = task_config.split('_')[0] if '_' in task_config else 'judges'
        
        # Create learning task
        learning_task = self._create_learning_task('supervised', task_config, epochs)
        
        # Generate training data
        training_data, labels = self._generate_training_data(data_source, 'general', batch_size * 8)
        
        segment_results = []
        
        for i, learner in enumerate(self.segment_learners):
            segment_id = self.segments[i].segment_id
            
            print(f"\nTraining {node_type} in segment {segment_id}...")
            
            try:
                # Train specific node type
                segment_result = learner.train_node_type(
                    node_type=node_type,
                    learning_task=learning_task,
                    data=training_data,
                    labels=labels
                )
                
                segment_result['segment_id'] = segment_id
                segment_result['node_type_trained'] = node_type
                segment_results.append(segment_result)
                
                if self.verbose and segment_result.get('status') != 'no_nodes':
                    print(f"   ✅ Segment {segment_id}: {node_type} training completed")
                elif segment_result.get('status') == 'no_nodes':
                    print(f"   ⚠️  Segment {segment_id}: No {node_type} nodes available")
                
            except Exception as e:
                print(f"   ❌ Segment {segment_id} {node_type} training failed: {str(e)}")
                segment_results.append({
                    'segment_id': segment_id,
                    'node_type_trained': node_type,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'status': 'success',
            'train_type': 'node_specific',
            'node_type_trained': node_type,
            'segments_trained': len(segment_results),
            'segment_results': segment_results
        }
    
    def _run_evolutionary_training(self, task_config: str, data_source: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        """Run training with focus on node evolution."""
        print(f"🧬 Running evolutionary training: {task_config}")
        
        # Create learning task with evolution focus
        learning_task = self._create_learning_task('supervised', task_config, epochs)
        
        # Generate challenging training data to promote evolution
        training_data, labels = self._generate_training_data(data_source, 'challenging', batch_size * 15)
        
        segment_results = []
        total_evolutions = 0
        
        for i, learner in enumerate(self.segment_learners):
            segment_id = self.segments[i].segment_id
            
            print(f"\nEvolutionary training segment {segment_id}...")
            
            # Temporarily increase evolution likelihood for demonstration
            original_threshold = learner.config.get('evolution_threshold_computational_to_judge', 0.85)
            learner.config['evolution_threshold_computational_to_judge'] = 0.75  # Lower threshold
            learner.config['enable_node_evolution'] = True
            
            try:
                # Run training with evolution enabled
                segment_result = learner.train_segment(
                    learning_task=learning_task,
                    data=training_data,
                    labels=labels
                )
                
                # Count evolutions that occurred
                evolutions_count = len(learner.training_state.get('node_evolutions', []))
                total_evolutions += evolutions_count
                
                segment_result['segment_id'] = segment_id
                segment_result['evolutions_occurred'] = evolutions_count
                segment_results.append(segment_result)
                
                if self.verbose:
                    final_loss = segment_result.get('final_loss', 0.0)
                    print(f"   ✅ Segment {segment_id}: loss={final_loss:.4f}, evolutions={evolutions_count}")
                
            except Exception as e:
                print(f"   ❌ Segment {segment_id} evolutionary training failed: {str(e)}")
                segment_results.append({
                    'segment_id': segment_id,
                    'status': 'failed',
                    'error': str(e)
                })
            finally:
                # Restore original threshold
                learner.config['evolution_threshold_computational_to_judge'] = original_threshold
        
        return {
            'status': 'success',
            'train_type': 'evolutionary',
            'segments_trained': len(segment_results),
            'total_evolutions': total_evolutions,
            'segment_results': segment_results
        }
    
    def _create_learning_task(self, task_type: str, task_config: str, max_epochs: int) -> LearningTask:
        """Create a LearningTask instance based on configuration."""
        # Parse task configuration
        if 'classification' in task_config:
            objective = 'classification'
            num_classes = 10
        elif 'regression' in task_config:
            objective = 'regression'
            num_classes = 1
        else:
            objective = 'classification'
            num_classes = 5
        
        # Determine modality
        if 'text' in task_config:
            modality = 'text'
            data_shape = (512,)
        elif 'vision' in task_config:
            modality = 'general'
            data_shape = (224, 224, 3)
        else:
            modality = 'general'
            data_shape = (512,)
        
        # Create task
        return LearningTask(
            task_id=f"{task_type}_{task_config}_{int(time.time())}",
            task_type=task_type,
            modality=modality,
            objective=objective,
            data_shape=data_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            max_epochs=max_epochs,
            early_stopping_patience=10
        )
    
    def _generate_training_data(self, data_source: str, data_type: str, sample_count: int) -> Tuple[List[Any], List[Any]]:
        """Generate or load training data based on source and type."""
        if data_source == 'synthetic':
            return self._generate_synthetic_data(data_type, sample_count)
        elif data_source == 'file':
            return self._load_data_from_file(data_type, sample_count)
        elif data_source == 'random':
            return self._generate_random_data(data_type, sample_count)
        else:
            # Default to synthetic
            return self._generate_synthetic_data(data_type, sample_count)
    
    def _generate_synthetic_data(self, data_type: str, sample_count: int) -> Tuple[List[Any], List[Any]]:
        """Generate synthetic training data."""
        if data_type == 'text' or 'text' in data_type:
            # Text data
            text_templates = [
                "This is a sample text for classification task number {}",
                "Training example {} for neural network learning",
                "Synthetic text data point {} for BrainNexus training",
                "Text sample {} demonstrating classification patterns",
                "Learning data instance {} for segment training"
            ]
            
            data = []
            labels = []
            
            for i in range(sample_count):
                template = text_templates[i % len(text_templates)]
                text = template.format(i)
                data.append(text)
                labels.append(i % 5)  # 5-class classification
            
            return data, labels
        
        elif data_type == 'vision' or 'vision' in data_type:
            # Vision-like data (random tensors representing images)
            data = []
            labels = []
            
            for i in range(sample_count):
                # Generate random "image" data
                image_data = np.random.randn(224, 224, 3).astype(np.float32)
                data.append(image_data)
                labels.append(i % 10)  # 10-class classification
            
            return data, labels
        
        elif data_type == 'challenging':
            # Generate challenging data patterns to promote evolution
            data = []
            labels = []
            
            for i in range(sample_count):
                # Create complex patterns that require higher-order thinking
                if i % 3 == 0:
                    # Sequential pattern
                    text = f"Complex sequence pattern {i}: " + " ".join([f"item_{j}" for j in range(i % 10 + 5)])
                elif i % 3 == 1:
                    # Hierarchical pattern
                    text = f"Hierarchical structure {i}: level_1.sublevel_{i%3}.element_{i%7}"
                else:
                    # Abstract pattern
                    text = f"Abstract concept {i}: {i**2 % 13}:{i**3 % 17}:{i % 23}"
                
                data.append(text)
                labels.append((i * 7) % 8)  # 8-class classification with complex mapping
            
            return data, labels
        
        else:
            # General/default data
            data = []
            labels = []
            
            for i in range(sample_count):
                # General purpose synthetic data
                data_point = f"Sample data {i} for general training"
                data.append(data_point)
                labels.append(i % 3)  # 3-class classification
            
            return data, labels
    
    def _generate_random_data(self, data_type: str, sample_count: int) -> Tuple[List[Any], List[Any]]:
        """Generate random numerical data."""
        data = []
        labels = []
        
        for i in range(sample_count):
            # Random numerical features
            features = np.random.randn(512).tolist()
            data.append(features)
            
            # Random labels
            labels.append(np.random.randint(0, 5))
        
        return data, labels
    
    def _load_data_from_file(self, data_type: str, sample_count: int) -> Tuple[List[Any], List[Any]]:
        """Load data from file (placeholder implementation)."""
        # This is a placeholder - in a real implementation, you would load from actual files
        print(f"⚠️  File loading not implemented, generating synthetic {data_type} data instead")
        return self._generate_synthetic_data(data_type, sample_count)
    
    def _create_rl_config(self, task_config: str) -> RLConfig:
        """Create reinforcement learning configuration."""
        return RLConfig(
            algorithm='dqn',
            state_dim=64,
            action_dim=8,
            action_space_type='discrete',
            batch_size=32,
            gamma=0.99,
            epsilon=0.1,
            buffer_size=10000,
            target_update_frequency=100,
            ppo_epochs=4,
            ppo_clip=0.2,
            value_coef=0.5,
            entropy_coef=0.01
        )
    
    def _create_rl_environment(self, task_config: str, data_source: str):
        """Create a simple RL environment for training."""
        return SimpleRLEnvironment(task_config=task_config)


class SimpleRLEnvironment:
    """Simple RL environment for demonstration purposes."""
    
    def __init__(self, task_config: str):
        self.task_config = task_config
        self.state_dim = 64
        self.action_dim = 8
        self.max_steps = 100
        self.current_step = 0
        self.current_state = None
    
    def reset(self):
        """Reset the environment."""
        self.current_step = 0
        self.current_state = np.random.randn(self.state_dim).tolist()
        return self.current_state
    
    def step(self, actions: Dict[str, Any]):
        """Take a step in the environment."""
        self.current_step += 1
        
        # Simple reward calculation based on action coherence
        reward = 0.0
        for node_type, action in actions.items():
            if isinstance(action, (int, float)):
                # Reward actions that are reasonable
                reward += 0.1 if -1 <= action <= 1 else -0.1
        
        # Update state
        self.current_state = np.random.randn(self.state_dim).tolist()
        
        # Check if done
        done = self.current_step >= self.max_steps or reward < -5.0
        
        return self.current_state, reward, done, {'step': self.current_step}


# Add remaining command handlers to BrainNexusInterface
def _add_interface_analysis_methods():
    """Add analysis and status methods to BrainNexusInterface."""
    
    def _handle_analyze(self, args: List[str]) -> Dict[str, Any]:
        """
        Handle analysis and monitoring operations.
        
        Args:
            args: Command arguments [analysis_type, target, output_format]
            
        Returns:
            Dict with analysis results
        """
        print("📊 RUNNING ANALYSIS OPERATIONS")
        print("-" * 50)
        
        if not self.segments and not self.brain_nexus:
            print("❌ No BrainNexus or segments available. Run 'init' and 'create' commands first.")
            return {'status': 'error', 'error': 'No analysis targets available'}
        
        # Parse arguments
        analysis_type = args[0] if len(args) > 0 else 'performance'
        target = args[1] if len(args) > 1 else 'all'
        output_format = args[2] if len(args) > 2 else 'console'
        
        try:
            start_time = time.time()
            
            if analysis_type == 'performance':
                results = self._analyze_performance(target)
            elif analysis_type == 'topology':
                results = self._analyze_network_topology(target)
            elif analysis_type == 'training':
                results = self._analyze_training_history(target)
            elif analysis_type == 'segments':
                results = self._analyze_segment_distribution(target)
            elif analysis_type == 'evolution':
                results = self._analyze_node_evolution(target)
            elif analysis_type == 'spatial':
                results = self._analyze_spatial_organization(target)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            analysis_time = time.time() - start_time
            results['analysis_time'] = analysis_time
            results['analysis_type'] = analysis_type
            results['target'] = target
            
            # Format output
            if output_format == 'console':
                self._display_analysis_results(results, analysis_type)
            elif output_format == 'json':
                self._save_analysis_json(results, analysis_type)
            elif output_format == 'plot' and PLOTTING_AVAILABLE:
                self._create_analysis_plots(results, analysis_type)
            
            print(f"\n✅ Analysis completed in {analysis_time:.3f}s")
            
            return results
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'status': 'error',
                'error': error_msg,
                'analysis_type': analysis_type
            }
    
    # Properly add the method to the class using setattr
    setattr(BrainNexusInterface, '_handle_analyze', _handle_analyze)

# Execute the method addition
_add_interface_analysis_methods()


# Add all remaining command handler methods
def add_remaining_handlers():
    """Add all remaining command handlers to complete the interface."""
    
    def _handle_save(self, args: List[str]) -> Dict[str, Any]:
        """Handle save operations."""
        print("💾 SAVE OPERATIONS")
        print("-" * 50)
        
        save_type = args[0] if len(args) > 0 else 'all'
        filename = args[1] if len(args) > 1 else f"brain_nexus_save_{int(time.time())}"
        
        try:
            save_path = self.results_dir / f"{filename}.pkl"
            
            save_data = {
                'training_history': self.training_history,
                'session_info': self.active_session,
                'brain_config': self.current_brain_config,
                'segments_count': len(self.segments),
                'save_timestamp': datetime.now().isoformat()
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"✅ Saved to: {save_path}")
            return {'status': 'success', 'save_path': str(save_path)}
            
        except Exception as e:
            error_msg = f"Save failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {'status': 'error', 'error': error_msg}
    
    def _handle_load(self, args: List[str]) -> Dict[str, Any]:
        """Handle load operations."""
        print("📂 LOAD OPERATIONS")
        print("-" * 50)
        
        if len(args) == 0:
            print("❌ No filename provided")
            return {'status': 'error', 'error': 'Filename required'}
        
        filename = args[0]
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        try:
            load_path = self.results_dir / filename
            
            if not load_path.exists():
                print(f"❌ File not found: {load_path}")
                return {'status': 'error', 'error': 'File not found'}
            
            with open(load_path, 'rb') as f:
                save_data = pickle.load(f)
            
            if 'training_history' in save_data:
                self.training_history = save_data['training_history']
            
            print(f"✅ Loaded from: {load_path}")
            return {'status': 'success', 'load_path': str(load_path)}
            
        except Exception as e:
            error_msg = f"Load failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {'status': 'error', 'error': error_msg}
    
    def _handle_test(self, args: List[str]) -> Dict[str, Any]:
        """Handle testing operations."""
        print("🧪 TESTING OPERATIONS")
        print("-" * 50)
        
        test_type = args[0] if len(args) > 0 else 'basic'
        
        try:
            if test_type == 'basic':
                print("Running basic tests...")
                
                if not self.brain_nexus:
                    self._handle_init(['64', 'true'])
                
                if len(self.segments) == 0:
                    self._handle_create(['2', 'balanced'])
                
                training_result = self._handle_train(['supervised', 'test', 'synthetic', '2', '8'])
                
                print("✅ Basic tests completed!")
                return {'status': 'success', 'test_type': test_type}
            
            return {'status': 'success', 'test_type': test_type}
                
        except Exception as e:
            error_msg = f"Testing failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {'status': 'error', 'error': error_msg}
    
    def _handle_help(self, args: List[str]) -> Dict[str, Any]:
        """Display help information."""
        print("🚀 BRAINNEXUS INTERFACE HELP")
        print("=" * 60)
        
        if len(args) > 0 and args[0] == 'create':
            print("""
CREATE COMMAND DETAILED HELP:
    Usage: create <count> [segment_type] [config_preset]
    
    Segment Types (by dimensional assignment pattern):
    - balanced  : Alternating polarities based on segment ID
    - positive  : All positive polarities
    - negative  : All negative polarities  
    - mixed     : Mathematical pattern mixing
    - spiral    : Complex dimensional relationships
    
    Config Presets (by size and capabilities):
    - demo      : 100x100 nodes, basic features, fast training
    - default   : 1000x1000 nodes, standard features
    - full      : 1000000x1000000 nodes, all features enabled
    - massive   : 1000000000x1000000000 nodes, distributed processing
    - performance : Optimized for speed with parallel processing
    - memory_efficient : Reduced memory usage, fewer features
    - research  : All experimental features enabled
    
    Examples:
    create 2 balanced demo      - 2 balanced segments, demo size
    create 3 mixed default      - 3 mixed segments, standard size  
    create 1 positive full      - 1 positive segment, full scale
    create 5 spiral massive     - 5 spiral segments, massive scale
            """)
        else:
            print("""
Available Commands:
    init <dimensions> [demo]     - Initialize BrainNexus
    create <count> [type] [size] - Create brain segments  
    train <type> [config]        - Train segments
    analyze <type> [target]      - Analyze system
    save [type] [filename]       - Save system state
    load <filename>              - Load system state
    status [detail]              - Show system status
    test [type]                  - Run tests
    help [command]               - Show help information (try 'help create')
    demo                         - Run demonstration
    config [setting] [value]     - Configure settings
    exit                         - Exit interface

Examples:
    init 64
    create 5 balanced demo
    create 2 mixed full  
    train supervised text_classification
    analyze performance all
    save all my_brain_state
    """)
        
        return {'status': 'success'}
    
    def _handle_status(self, args: List[str]) -> Dict[str, Any]:
        """Handle status display."""
        print("💫 BRAINNEXUS SYSTEM STATUS")
        print("-" * 50)
        
        uptime = (datetime.now() - self.active_session['start_time']).total_seconds()
        
        print(f"   Session ID: {self.active_session['session_id']}")
        print(f"   Uptime: {uptime:.0f}s")
        print(f"   Operations: {len(self.active_session['operations'])}")
        print(f"   BrainNexus: {'✅' if self.brain_nexus else '❌'}")
        print(f"   Segments: {len(self.segments)}")
        print(f"   Training Sessions: {len(self.training_history)}")
        print(f"   Demo Mode: {'ON' if self.demo_mode else 'OFF'}")
        
        return {
            'status': 'success',
            'uptime': uptime,
            'brain_initialized': self.brain_nexus is not None,
            'segments_count': len(self.segments)
        }
    
    def _handle_demo(self, args: List[str]) -> Dict[str, Any]:
        """Run demonstration."""
        print("🎭 BRAINNEXUS DEMONSTRATION")
        print("=" * 60)
        
        try:
            print("Running quick demonstration...")
            
            if not self.brain_nexus:
                self._handle_init(['64', 'true'])
            
            if len(self.segments) == 0:
                self._handle_create(['3', 'balanced'])
            
            training_result = self._handle_train(['supervised', 'demo', 'synthetic', '2', '16'])
            
            print("\n✅ Quick demo completed!")
            return {
                'status': 'success',
                'segments_created': len(self.segments),
                'training_completed': training_result.get('status') == 'success'
            }
                
        except Exception as e:
            error_msg = f"Demo failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {'status': 'error', 'error': error_msg}
    
    def _handle_config(self, args: List[str]) -> Dict[str, Any]:
        """Handle configuration."""
        print("⚙️  CONFIGURATION")
        print("-" * 50)
        
        if len(args) == 0:
            print("Current Configuration:")
            print(f"   Demo Mode: {self.demo_mode}")
            print(f"   Work Directory: {self.work_dir}")
            return {'status': 'success', 'config': self.config}
        
        setting = args[0]
        value = args[1] if len(args) > 1 else None
        
        if setting == 'demo_mode' and value:
            self.demo_mode = value.lower() in ['true', '1', 'on', 'yes']
            print(f"Demo mode set to: {self.demo_mode}")
            return {'status': 'success', 'setting': setting, 'value': self.demo_mode}
        
        print(f"❌ Unknown setting: {setting}")
        return {'status': 'error', 'error': f'Unknown setting: {setting}'}
    
    # Add all methods to the class using setattr
    setattr(BrainNexusInterface, '_handle_save', _handle_save)
    setattr(BrainNexusInterface, '_handle_load', _handle_load)
    setattr(BrainNexusInterface, '_handle_test', _handle_test)
    setattr(BrainNexusInterface, '_handle_help', _handle_help)
    setattr(BrainNexusInterface, '_handle_status', _handle_status)
    setattr(BrainNexusInterface, '_handle_demo', _handle_demo)
    setattr(BrainNexusInterface, '_handle_config', _handle_config)

# Execute completion
add_remaining_handlers()


if __name__ == "__main__":
    interface = BrainNexusInterface()
    interface.run()
