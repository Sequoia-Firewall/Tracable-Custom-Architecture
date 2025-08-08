#!/usr/bin/env python3
"""
BrainNexus Main Interface
========================
Interactive interface for the BrainNexus spatially-organized neural network.
Supports demo mode, training, inference, and state management.
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Core imports
import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict

# Import our custom classes
try:
    from brainNexus import BrainNexus
    from BrainNexusLearn import BrainNexusLearn, TrainingConfig
    from NeuralNode import NeuralNode
    print("✓ Successfully imported BrainNexus components")
except ImportError as e:
    print(f"❌ Failed to import BrainNexus components: {e}")
    print("Make sure brainNexus.py, BrainNexusLearn.py and NeuralNode.py are in the same directory")
    exit(1)

class BrainTraceRecorder:
    """Records detailed execution traces of BrainNexus computations"""
    
    def __init__(self):
        self.trace_events = []
        self.recording = False
        self.start_time = None
        
    def start_recording(self):
        """Start recording trace events"""
        self.trace_events = []
        self.recording = True
        self.start_time = time.time()
        
    def stop_recording(self):
        """Stop recording trace events"""
        self.recording = False
        
    def log_event(self, event_type: str, node_id: str, data: Dict[str, Any]):
        """Log a computation event"""
        if not self.recording:
            return
            
        timestamp = time.time() - self.start_time if self.start_time else 0
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'node_id': node_id,
            'data': data.copy()
        }
        self.trace_events.append(event)
class BrainNexusManager:
    """Manages BrainNexus instances with save/load functionality"""
    
    def __init__(self):
        self.brain: Optional[BrainNexus] = None
        self.learner: Optional[BrainNexusLearn] = None
        self.training_data: List = []
        self.validation_data: List = []
        self.save_directory = "brain_states"
        self.ensure_save_directory()
    def list_all_nodes_detailed(self):
        """List all nodes with their positions, connections, and all record fields.
        Also saves the details to a JSON file."""
        if self.brain is None or not hasattr(self.brain, 'neural_nodes') or len(self.brain.neural_nodes) == 0:
            print("No brain or neural nodes available.")
            return

        print("\nAll Nodes in BrainNexus:")
        print("=" * 60)
        node_details = []
        
        for node_num, node in enumerate(self.brain.neural_nodes, 1):
            print(f"Node {node_num}:")
            node_info = {
                'Node_ID': getattr(node, 'node_id', 'Unknown'),
                'Node_Type': getattr(node, 'node_type', 'Unknown'),
                'Node_Position': getattr(node, 'node_position', [0, 0, 0]),
                'Spatial_Affinity': getattr(node, 'spatial_affinity', 0.0),
                'Times_Called': getattr(node, 'times_called', 0)
            }
            
            # Get connections if available
            if hasattr(self.brain, 'get_node_connections'):
                try:
                    connections = self.brain.get_node_connections(node.node_id)
                    node_info['Connections'] = connections
                except:
                    node_info['Connections'] = {'incoming': [], 'outgoing': [], 'bidirectional': []}
            
            for key, value in node_info.items():
                print(f"  {key}: {value}")
            node_details.append(node_info)
            print("-" * 40)

        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"brain_nodes_{timestamp}.json"
        json_path = os.path.join(self.save_directory, json_filename)
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(node_details, f, indent=2, default=str)
            print(f"\n✓ Node details saved to JSON: {json_path}")
        except Exception as e:
            print(f"\n❌ Failed to save node details to JSON: {e}")
    def visualize_nodes_and_connections_3d(self):
        """Visualize all node positions and their connections (3D)"""
        if self.brain is None or not hasattr(self.brain, 'neural_nodes') or len(self.brain.neural_nodes) == 0:
            print("No brain or neural nodes available.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Color mapping for node types
        type_colors = {
            'Controller': 'red',
            'Judge': 'orange', 
            'Splitter': 'yellow',
            'Computational': 'blue',
            'Retainer': 'green',
            'Reviewer': 'purple',
            'Handler': 'brown',
            'Unknown': 'gray'
        }

        for node in self.brain.neural_nodes:
            node_id = getattr(node, 'node_id', 0)
            node_type = getattr(node, 'node_type', 'Unknown')
            position = getattr(node, 'node_position', [0, 0, 0])
            
            x, y, z = position[:3]
            color = type_colors.get(node_type, 'gray')
            
            ax.scatter(x, y, z, c=color, s=100)
            ax.text(x, y, z, str(node_id))

            # Draw connections if available
            if hasattr(self.brain, 'get_node_connections'):
                try:
                    connections = self.brain.get_node_connections(node_id)
                    for target_id in connections.get('outgoing', []):
                        target_node = self.brain.node_registry.get(target_id)
                        if target_node and hasattr(target_node, 'node_position'):
                            tx, ty, tz = target_node.node_position[:3]
                            ax.plot([x, tx], [y, ty], [z, tz], color='gray', alpha=0.5)
                except:
                    pass  # Skip connections if not available

        ax.set_title("Node Positions and Connections (3D)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if hasattr(ax, 'set_zlabel'):
            ax.set_zlabel('Z')
        
        # Add legend
        legend_elements = []
        for node_type, color in type_colors.items():
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, label=node_type))
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.show()
    def visualize_brain_smart(self):
        """Automatically choose 2D or 3D visualization based on brain mode"""
        if self.brain is None or self.brain.node_records is None or len(self.brain.node_records) == 0:
            print("No brain or node records available.")
            return

        if self.brain.demo:
            print("🎨 Rendering 2D visualization (Demo mode brain)")
            self.visualize_nodes_and_connections()
        else:
            print("🎨 Rendering 3D visualization (Full mode brain)")
            self.visualize_nodes_and_connections_3d()

    def visualize_brain_both(self):
        """Show both 2D and 3D visualizations regardless of brain mode"""
        if self.brain is None or self.brain.node_records is None or len(self.brain.node_records) == 0:
            print("No brain or node records available.")
            return

        print("🎨 Rendering both 2D and 3D visualizations...")
        
        # Show 2D first
        print("Displaying 2D visualization...")
        self.visualize_nodes_and_connections()
        
        # Small delay to let first plot render
        import time
        time.sleep(0.5)
        
        # Then show 3D
        print("Displaying 3D visualization...")
        self.visualize_nodes_and_connections_3d()
    def visualize_nodes_and_connections(self):
        """Visualize all node positions and their connections (2D)"""
        if self.brain is None or self.brain.node_records is None or len(self.brain.node_records) == 0:
            print("No brain or node records available.")
            return

        G = nx.DiGraph()
        pos = {}
        for _, row in self.brain.node_records.iterrows():
            node_id = row['Node_ID']
            node_pos = row['Node_Position']
            pos[node_id] = node_pos[:2]  # Use first two dimensions for 2D
            G.add_node(node_id)

            # Add connections
            for target_id in row['Exit_Connections']:
                G.add_edge(node_id, target_id)

        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', arrows=True)
        plt.title("Node Positions and Connections (2D)")
        plt.show()
    
    def ensure_save_directory(self):
        """Create save directory if it doesn't exist"""
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"Created save directory: {self.save_directory}")
    
    def save_brain_state(self, filename: Optional[str] = None) -> str:
        """Save current brain state to file"""
        if self.brain is None:
            raise ValueError("No brain instance to save")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "demo" if self.brain.demo else "full"
            filename = f"brain_{mode}_{timestamp}.pkl"
        else:
            # Ensure custom filename has .pkl extension
            if not filename.endswith('.pkl'):
                filename = filename + '.pkl'
        
        filepath = os.path.join(self.save_directory, filename)
        
        # Package brain data including output configuration
        brain_data = {
            'node_records': self.brain.node_records,
            'brain_records': self.brain.brain_records,
            'demo': self.brain.demo,
            'dimensions': self.brain.dimensions,
            'learning_rate': self.brain.learning_rate,
            'next_node_id': self.brain.next_node_id,
            'neural_nodes_count': len(self.brain.neural_nodes),
            'save_timestamp': datetime.now().isoformat(),
            # Save output configuration for proper restoration
            'output_config': getattr(self.brain, 'output_config', None),
            'has_tokenizer': getattr(self.brain, 'output_config', {}).get('vocab_mapping') is not None if hasattr(self.brain, 'output_config') else False,
            'tokenizer_type': self._detect_tokenizer_type(),
            'vocab_size': getattr(self.brain, 'output_config', {}).get('num_classes', 10) if hasattr(self.brain, 'output_config') else 10
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(brain_data, f)
        
        print(f"✓ Brain state saved to: {filepath}")
        return filepath
    
    def load_brain_state(self, filename: str) -> bool:
        """Load brain state from file"""
        filepath = os.path.join(self.save_directory, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                brain_data = pickle.load(f)
            
            # Create new brain instance with output configuration
            output_config = brain_data.get('output_config', {})
            
            # If the brain has a tokenizer, we need to recreate it
            if brain_data.get('has_tokenizer', False):
                tokenizer_type = brain_data.get('tokenizer_type', 'custom')
                vocab_size = brain_data.get('vocab_size', 1000)
                
                print(f"  🔄 Restoring {tokenizer_type} tokenizer...")
                
                # Recreate the tokenizer based on saved type
                if tokenizer_type == 'mistral_v3_tekken':
                    try:
                        vocab_mapping = self._load_mistral_tokenizer()
                        output_config['vocab_mapping'] = vocab_mapping
                        print(f"    ✓ Restored Mistral v3 Tekken tokenizer with {len(vocab_mapping)} tokens")
                    except Exception as e:
                        print(f"    ⚠️  Failed to restore Mistral tokenizer: {e}")
                        print(f"    🔄 Using saved vocabulary mapping instead")
                
                elif tokenizer_type == 'gpt2':
                    try:
                        vocab_mapping = self._load_gpt2_tokenizer(vocab_size)
                        output_config['vocab_mapping'] = vocab_mapping
                        print(f"    ✓ Restored GPT-2 tokenizer with {len(vocab_mapping)} tokens")
                    except Exception as e:
                        print(f"    ⚠️  Failed to restore GPT-2 tokenizer: {e}")
                        print(f"    🔄 Using saved vocabulary mapping instead")
                
                elif tokenizer_type == 'bert':
                    try:
                        vocab_mapping = self._load_bert_tokenizer(vocab_size)
                        output_config['vocab_mapping'] = vocab_mapping
                        print(f"    ✓ Restored BERT tokenizer with {len(vocab_mapping)} tokens")
                    except Exception as e:
                        print(f"    ⚠️  Failed to restore BERT tokenizer: {e}")
                        print(f"    🔄 Using saved vocabulary mapping instead")
                
                elif tokenizer_type == 'basic_english':
                    vocab_mapping = self._load_basic_english_tokenizer(vocab_size)
                    output_config['vocab_mapping'] = vocab_mapping
                    print(f"    ✓ Restored basic English tokenizer with {len(vocab_mapping)} tokens")
                
                else:
                    print(f"    ℹ️  Using saved custom tokenizer vocabulary")
            
            self.brain = BrainNexus(
                dimensions=brain_data['dimensions'],
                demo=brain_data['demo'],
                output_config=output_config
            )
            
            # Restore state only if brain was created successfully
            if self.brain is not None:
                self.brain.node_records = brain_data['node_records']
                self.brain.brain_records = brain_data['brain_records']
                self.brain.next_node_id = brain_data['next_node_id']
                
                # Recreate neural_nodes list with full connection restoration
                self.brain.neural_nodes = []
                self.brain.node_registry = {}
                
                print(f"🔄 Recreating {len(self.brain.node_records)} nodes with connections...")
                
                # Step 1: Create all nodes first
                for _, row in self.brain.node_records.iterrows():
                    node_id = row['Node_ID']
                    node_type = row['Node_Type']
                    node_position = row['Node_Position']
                    
                    # Handle position data (might be stored as string or list)
                    if isinstance(node_position, str):
                        try:
                            # Try to evaluate string representation of list
                            node_position = eval(node_position)
                        except:
                            # Fallback to default position
                            node_position = [0.0, 0.0, 0.0, 0.0]
                    elif not isinstance(node_position, list):
                        node_position = [0.0, 0.0, 0.0, 0.0]
                    
                    node = NeuralNode(node_id, node_type, node_position)
                    
                    # Restore additional node attributes from records
                    if 'Times_Called' in row and not pd.isna(row['Times_Called']):
                        setattr(node, 'times_called', int(row['Times_Called']))
                    if 'Spatial_Affinity' in row and not pd.isna(row['Spatial_Affinity']):
                        setattr(node, 'spatial_affinity', float(row['Spatial_Affinity']))
                    
                    self.brain.neural_nodes.append(node)
                    self.brain.node_registry[node_id] = node
                
                # Step 2: Restore all connections between nodes
                connection_count = 0
                for _, row in self.brain.node_records.iterrows():
                    node_id = row['Node_ID']
                    node = self.brain.node_registry[node_id]
                    
                    # Initialize connection attributes only if they don't exist
                    if not hasattr(node, 'entrance_connections'):
                        setattr(node, 'entrance_connections', [])
                    if not hasattr(node, 'exit_connections'):
                        setattr(node, 'exit_connections', [])
                    if not hasattr(node, 'connection_weights'):
                        setattr(node, 'connection_weights', {})
                    
                    # Restore exit connections
                    exit_connections = row['Exit_Connections']
                    
                    # Handle different data types that pandas might store
                    if exit_connections is None:
                        exit_connections = []
                    elif isinstance(exit_connections, str):
                        try:
                            exit_connections = eval(exit_connections)
                        except:
                            exit_connections = []
                    elif hasattr(exit_connections, '__iter__') and not isinstance(exit_connections, str):
                        # Handle numpy arrays, pandas Series, or other iterables
                        try:
                            # Convert to list, handling potential numpy arrays or pandas Series
                            if hasattr(exit_connections, 'tolist'):
                                exit_connections = exit_connections.tolist()
                            else:
                                exit_connections = list(exit_connections)
                            # Filter out NaN values that might be in the list
                            exit_connections = [x for x in exit_connections if not (pd.isna(x) if hasattr(pd, 'isna') else x != x)]
                        except:
                            exit_connections = []
                    else:
                        # Scalar value or unknown type - check if it's NaN
                        try:
                            if pd.isna(exit_connections):
                                exit_connections = []
                            else:
                                exit_connections = [exit_connections]  # Wrap scalar in list
                        except:
                            exit_connections = []
                    
                    # Ensure we have a proper list of integers (node IDs)
                    if isinstance(exit_connections, list):
                        for target_id in exit_connections:
                            if target_id in self.brain.node_registry:
                                node.exit_connections.append(target_id)
                                # Set default weight
                                node.connection_weights[target_id] = 0.5
                                
                                # Add to target's entrance connections
                                target_node = self.brain.node_registry[target_id]
                                if not hasattr(target_node, 'entrance_connections'):
                                    setattr(target_node, 'entrance_connections', [])
                                if node_id not in target_node.entrance_connections:
                                    target_node.entrance_connections.append(node_id)
                                
                                connection_count += 1
                
                print(f"✓ Restored {len(self.brain.neural_nodes)} neural nodes")
                print(f"✓ Restored {connection_count} connections")
                print(f"  Sample Nodes: {', '.join(str(node.node_id) for node in self.brain.neural_nodes[:5])}...")
                
                # Verify critical connections are restored
                retainer_nodes = [n for n in self.brain.neural_nodes if n.node_type == 'Retainer']
                reviewer_nodes = [n for n in self.brain.neural_nodes if n.node_type == 'Reviewer']
                
                if retainer_nodes and reviewer_nodes:
                    print(f"🔍 Verifying critical connections:")
                    for i, (retainer, reviewer) in enumerate(zip(retainer_nodes, reviewer_nodes)):
                        retainer_exits = getattr(retainer, 'exit_connections', [])
                        reviewer_entrances = getattr(reviewer, 'entrance_connections', [])
                        
                        print(f"    Retainer #{retainer.node_id}: exit_connections={retainer_exits}")
                        print(f"    Reviewer #{reviewer.node_id}: entrance_connections={reviewer_entrances}")
                        
                        if reviewer.node_id in retainer_exits:
                            print(f"  ✓ Retainer #{retainer.node_id} → Reviewer #{reviewer.node_id}")
                        else:
                            print(f"  ❌ Missing: Retainer #{retainer.node_id} ↛ Reviewer #{reviewer.node_id}")
                            # Emergency fix: restore the connection
                            try:
                                retainer.exit_connections.append(reviewer.node_id)
                                retainer.connection_weights[reviewer.node_id] = 0.9
                                if not hasattr(reviewer, 'entrance_connections'):
                                    setattr(reviewer, 'entrance_connections', [])
                                reviewer.entrance_connections.append(retainer.node_id)
                                print(f"    🔧 Emergency fix applied")
                                print(f"    After fix - Retainer #{retainer.node_id}: exit_connections={retainer.exit_connections}")
                                print(f"    After fix - Reviewer #{reviewer.node_id}: entrance_connections={reviewer.entrance_connections}")
                            except Exception as e:
                                print(f"    ⚠️  Emergency fix failed: {e}")
                    
                    # Final verification using get_node_connections method
                    print(f"🔍 Final verification using get_node_connections:")
                    for reviewer in reviewer_nodes:
                        reviewer_connections = self.brain.get_node_connections(reviewer.node_id)
                        print(f"    Reviewer #{reviewer.node_id}: connections={reviewer_connections}")
                
                # Rebuild spatial index
                try:
                    self.brain._rebuild_spatial_index()
                    print(f"✓ Rebuilt spatial index")
                except Exception as e:
                    print(f"⚠️  Warning: Could not rebuild spatial index: {e}")
            else:
                print("❌ Failed to initialize BrainNexus instance during load.")
                return False
            
            print(f"✓ Brain state loaded from: {filepath}")
            print(f"  Mode: {'Demo' if brain_data['demo'] else 'Full'}")
            print(f"  Nodes: {len(brain_data['node_records'])}")
            print(f"  Saved: {brain_data['save_timestamp']}")
            
            # Show output configuration details
            if output_config:
                output_type = output_config.get('type', 'unknown')
                print(f"  Output Type: {output_type}")
                
                if brain_data.get('has_tokenizer', False):
                    tokenizer_type = brain_data.get('tokenizer_type', 'unknown')
                    vocab_size = brain_data.get('vocab_size', 0)
                    print(f"  Tokenizer: {tokenizer_type} ({vocab_size} tokens)")
                    
                    # Show sample tokens if available
                    if 'vocab_mapping' in output_config and output_config['vocab_mapping']:
                        sample_tokens = list(output_config['vocab_mapping'].values())[:5]
                        print(f"  Sample Tokens: {sample_tokens}...")
                else:
                    num_classes = output_config.get('num_classes', 0)
                    print(f"  Classes: {num_classes}")
                    if output_config.get('class_labels') and len(output_config['class_labels']) <= 10:
                        print(f"  Labels: {output_config['class_labels']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load brain state: {e}")
            return False
    
    def list_saved_states(self):
        """List all saved brain states with tokenizer information"""
        files = [f for f in os.listdir(self.save_directory) if f.endswith('.pkl')]
        if not files:
            print("No saved brain states found.")
            return []
        
        print("\nSaved Brain States:")
        print("-" * 70)
        for i, filename in enumerate(files, 1):
            filepath = os.path.join(self.save_directory, filename)
            size = os.path.getsize(filepath) / 1024  # KB
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            # Try to load basic info to show tokenizer type
            tokenizer_info = "Unknown"
            brain_mode = "Unknown"
            try:
                with open(filepath, 'rb') as f:
                    brain_data = pickle.load(f)
                    tokenizer_type = brain_data.get('tokenizer_type', 'none')
                    has_tokenizer = brain_data.get('has_tokenizer', False)
                    brain_mode = "Demo" if brain_data.get('demo', False) else "Full"
                    vocab_size = brain_data.get('vocab_size', 0)
                    
                    if has_tokenizer and tokenizer_type != 'none':
                        tokenizer_info = f"{tokenizer_type} ({vocab_size} tokens)"
                    else:
                        output_type = brain_data.get('output_config', {}).get('type', 'classification')
                        num_classes = brain_data.get('output_config', {}).get('num_classes', 10)
                        tokenizer_info = f"{output_type} ({num_classes} classes)"
            except Exception:
                tokenizer_info = "Legacy format"
            
            print(f"{i:2d}. {filename}")
            print(f"    Size: {size:.1f} KB | Mode: {brain_mode} | Output: {tokenizer_info}")
            print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")
            print()
        
        return files
    
    def configure_output_system(self) -> Dict[str, Any]:
        """
        Interactive configuration of the BrainNexus output system
        
        Returns:
            Dict containing output configuration
        """
        print("\n🔧 CONFIGURING OUTPUT SYSTEM")
        print("=" * 40)
        
        # Step 1: Choose output type
        print("\n1. Choose output type:")
        print("   1. Classification (e.g., image classification, sentiment analysis)")
        print("   2. Token Prediction (e.g., language models, next-word prediction)")
        print("   3. Custom Labels (e.g., days of week, custom categories)")
        
        type_choice = input("Enter choice (1-3, default 1): ").strip()
        output_types = {'1': 'classification', '2': 'tokens', '3': 'custom'}
        output_type = output_types.get(type_choice, 'classification')
        
        # Step 2: Configure number of classes
        if output_type == 'tokens':
            print(f"\n2. Vocabulary size for {output_type}:")
            print("   Common sizes: 1000 (small), 10000 (medium), 50000 (large)")
            num_classes_input = input("Enter vocabulary size (default 1000): ").strip()
            num_classes = int(num_classes_input) if num_classes_input.isdigit() else 1000
        else:
            print(f"\n2. Number of classes for {output_type}:")
            print("   Common sizes: 2 (binary), 3 (sentiment), 10 (digits), 1000 (ImageNet)")
            num_classes_input = input("Enter number of classes (default 10): ").strip()
            num_classes = int(num_classes_input) if num_classes_input.isdigit() else 10
        
        # Step 3: Configure labels/vocabulary
        class_labels = []
        vocab_mapping = None
        
        if output_type == 'tokens':
            print(f"\n3. Token vocabulary configuration:")
            print("   Available tokenizers:")
            print("   1. GPT-2 tokenizer (50,257 tokens)")
            print("   2. BERT tokenizer (30,522 tokens)")
            print("   3. Mistral v3 Tekken tokenizer (~32,000 tokens)")
            print("   4. Basic English tokenizer (10,000 tokens)")
            print("   5. Custom/Manual tokenizer")
            
            tokenizer_choice = input("Select tokenizer (1-5, default 1): ").strip()
            
            if tokenizer_choice == '2':
                # BERT tokenizer
                vocab_mapping = self._load_bert_tokenizer(num_classes)
                print(f"   ✓ Loaded BERT tokenizer with {len(vocab_mapping)} tokens")
            elif tokenizer_choice == '3':
                # Mistral v3 Tekken tokenizer - use full vocabulary
                vocab_mapping = self._load_mistral_tokenizer()
                print(f"   ✓ Loaded Mistral v3 Tekken tokenizer with {len(vocab_mapping)} tokens")
            elif tokenizer_choice == '4':
                # Basic English tokenizer
                vocab_mapping = self._load_basic_english_tokenizer(num_classes)
                print(f"   ✓ Loaded basic English tokenizer with {len(vocab_mapping)} tokens")
            elif tokenizer_choice == '5':
                # Custom/Manual - original behavior
                use_preset = input("Use preset vocabulary? (y/n, default y): ").strip().lower()
                
                if use_preset != 'n':
                    # Create a basic vocabulary with common tokens
                    vocab_mapping = {
                        0: '<pad>', 1: '<unk>', 2: '<start>', 3: '<end>',
                        4: 'the', 5: 'and', 6: 'is', 7: 'to', 8: 'a', 9: 'in'
                    }
                    # Fill the rest with generated tokens
                    for i in range(10, num_classes):
                        vocab_mapping[i] = f'token_{i:04d}'
                    print(f"   ✓ Created vocabulary with {num_classes} tokens")
                    print(f"   Sample tokens: {list(vocab_mapping.values())[:10]}...")
                else:
                    print("   ⚠️  Using auto-generated token mapping")
                    vocab_mapping = {i: f'token_{i:04d}' for i in range(num_classes)}
            else:
                # Default: GPT-2 tokenizer
                vocab_mapping = self._load_gpt2_tokenizer(num_classes)
                print(f"   ✓ Loaded GPT-2 tokenizer with {len(vocab_mapping)} tokens")
            
            # Update num_classes to match actual vocabulary size
            if vocab_mapping:
                num_classes = len(vocab_mapping)
                print(f"   ✓ Updated vocabulary size to {num_classes} tokens")
                sample_tokens = list(vocab_mapping.values())[:10]
                print(f"   Sample tokens: {sample_tokens}...")
                
                # Auto-suggest multi-token generation for tokenizer models
                print(f"\n   🚀 TOKENIZER DETECTED: Would you like multi-token sequence generation?")
                print(f"      This enables the brain to generate sequences of tokens instead of single predictions.")
                enable_multi = input("   Enable multi-token generation? (Y/n, default Y): ").strip().lower()
                
                if enable_multi != 'n':
                    print(f"   ✅ Multi-token generation will be configured automatically!")
                else:
                    print(f"   📌 Single-token mode selected")
        
        elif output_type == 'custom' or (output_type == 'classification' and num_classes <= 20):
            print(f"\n3. Class labels for {num_classes} classes:")
            custom_labels = input("Enter custom labels (comma-separated) or press Enter for auto: ").strip()
            
            if custom_labels:
                class_labels = [label.strip() for label in custom_labels.split(',')]
                if len(class_labels) != num_classes:
                    print(f"   ⚠️  Provided {len(class_labels)} labels but need {num_classes}. Using auto-generation.")
                    class_labels = [f'class_{i}' for i in range(num_classes)]
                else:
                    print(f"   ✓ Using custom labels: {class_labels}")
            else:
                class_labels = [f'class_{i}' for i in range(num_classes)]
                print(f"   ✓ Using auto-generated labels: {class_labels}")
        else:
            # For large classification tasks, use auto-generated labels
            class_labels = [f'class_{i}' for i in range(num_classes)]
            print(f"   ✓ Using auto-generated labels for {num_classes} classes")
        
        # Step 4: Output format
        print(f"\n4. Output format:")
        if output_type == 'tokens':
            print("   1. token_id (return token strings)")
            print("   2. index (return token indices)")
            format_choice = input("Enter choice (1-2, default 1): ").strip()
            output_format = 'token_id' if format_choice != '2' else 'index'
        else:
            print("   1. label (return human-readable labels)")
            print("   2. index (return class indices)")
            print("   3. probability_dist (focus on full probability distribution)")
            format_choice = input("Enter choice (1-3, default 1): ").strip()
            format_map = {'1': 'label', '2': 'index', '3': 'probability_dist'}
            output_format = format_map.get(format_choice, 'label')
        
        # Step 5: Advanced settings
        print(f"\n5. Advanced settings:")
        
        # Confidence threshold
        conf_input = input("Confidence threshold (0.0-1.0, default 0.7): ").strip()
        try:
            confidence_threshold = float(conf_input) if conf_input else 0.7
            confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        except ValueError:
            confidence_threshold = 0.7
        
        # Top-k predictions
        topk_input = input(f"Return top-k predictions (1-{min(num_classes, 10)}, default 1): ").strip()
        try:
            return_top_k = int(topk_input) if topk_input else 1
            return_top_k = max(1, min(return_top_k, num_classes))
        except ValueError:
            return_top_k = 1
        
        # Build configuration
        config = {
            'type': output_type,
            'num_classes': num_classes,
            'class_labels': class_labels,
            'output_format': output_format,
            'confidence_threshold': confidence_threshold,
            'return_top_k': return_top_k,
            'normalize_probabilities': True
        }
        
        if vocab_mapping:
            config['vocab_mapping'] = vocab_mapping
        
        # Display summary
        print(f"\n✅ OUTPUT CONFIGURATION SUMMARY:")
        print(f"   Type: {output_type}")
        print(f"   Classes/Vocabulary: {num_classes}")
        print(f"   Output Format: {output_format}")
        print(f"   Confidence Threshold: {confidence_threshold}")
        print(f"   Top-K Predictions: {return_top_k}")
        
        if output_type == 'tokens':
            sample_tokens = list(vocab_mapping.values())[:5] if vocab_mapping else ['auto-generated']
            print(f"   Sample Tokens: {sample_tokens}...")
        elif len(class_labels) <= 10:
            print(f"   Class Labels: {class_labels}")
        else:
            print(f"   Class Labels: {class_labels[:5]}... (and {len(class_labels)-5} more)")
        
        return config
    
    def _configure_multi_token_generation(self, output_config: Dict[str, Any]) -> None:
        """
        Auto-configure multi-token generation for tokenizer-based models
        
        Args:
            output_config: Output configuration to modify
        """
        # Check if this is a tokenizer-based model
        is_tokenizer_model = (
            output_config.get('type') == 'tokens' or
            'vocab_mapping' in output_config or
            output_config.get('num_classes', 0) >= 1000  # Large vocabulary suggests tokenizer
        )
        
        if is_tokenizer_model and not output_config.get('enable_multi_token', False):
            print(f"\n🚀 TOKENIZER DETECTED - Enabling Multi-Token Generation")
            print("=" * 60)
            
            # Ask user if they want multi-token generation
            enable_multi = input(
                "Enable multi-token sequence generation? (Y/n, default Y): "
            ).strip().lower()
            
            if enable_multi != 'n':
                # Determine optimal sequence parameters based on vocabulary size
                vocab_size = output_config.get('num_classes', 1000)
                
                if vocab_size >= 30000:
                    # Large model (GPT-2, BERT, Mistral)
                    default_max_length = 15
                    default_sequence_length = 8
                    default_context_window = 6
                    print("  📊 Large vocabulary detected - optimizing for language modeling")
                
                elif vocab_size >= 10000:
                    # Medium model
                    default_max_length = 12
                    default_sequence_length = 6
                    default_context_window = 5
                    print("  📊 Medium vocabulary detected - optimizing for sequence generation")
                
                else:
                    # Small model
                    default_max_length = 10
                    default_sequence_length = 5
                    default_context_window = 4
                    print("  📊 Small vocabulary detected - conservative sequence parameters")
                
                # Get user preferences for sequence length
                sequence_input = input(
                    f"Sequence length (1-{default_max_length}, default {default_sequence_length}): "
                ).strip()
                
                try:
                    sequence_length = int(sequence_input) if sequence_input else default_sequence_length
                    sequence_length = max(1, min(sequence_length, default_max_length))
                except ValueError:
                    sequence_length = default_sequence_length
                
                # Auto-configure multi-token settings
                output_config.update({
                    'enable_multi_token': True,
                    'max_sequence_length': sequence_length + 3,  # Allow some flexibility
                    'sequence_length': sequence_length,
                    'ideal_sequence_length': sequence_length,
                    'context_window': min(default_context_window, sequence_length),
                    'generation_method': 'hybrid_context_aware',
                    
                    # Node group behavior configuration - the heart of our system
                    'node_group_behaviors': {
                        'Judge': 'conservative',      # High confidence, established tokens
                        'Controller': 'directive',    # Structure and control tokens  
                        'Splitter': 'divergent',     # Multiple path exploration
                        'Computational': 'analytical', # Logic-based token selection
                        'Retainer': 'contextual',    # Memory-aware token generation
                        'Reviewer': 'evaluative'     # Quality-focused token selection
                    }
                })
                
                print(f"  ✅ Multi-token generation configured!")
                print(f"     • Sequence Length: {sequence_length}")
                print(f"     • Context Window: {output_config['context_window']}")
                print(f"     • Generation Method: {output_config['generation_method']}")
                print(f"     • Node Group Behaviors: {len(output_config['node_group_behaviors'])}")
                
                # Show brief explanation of node behaviors
                print(f"\n  🎭 Node Group Behaviors:")
                for group, behavior in output_config['node_group_behaviors'].items():
                    descriptions = {
                        'conservative': 'High-confidence, established tokens',
                        'directive': 'Structure and control tokens',
                        'divergent': 'Multiple path exploration', 
                        'analytical': 'Logic-based token selection',
                        'contextual': 'Memory-aware token generation',
                        'evaluative': 'Quality-focused token selection'
                    }
                    print(f"     • {group}: {descriptions.get(behavior, behavior)}")
                
            else:
                print("  📌 Multi-token generation disabled - using single-token mode")
        
        elif output_config.get('enable_multi_token', False):
            print(f"  🚀 Multi-token generation already configured")
        
        else:
            print(f"  📌 Classification model - single-token mode optimal")
    
    def _load_gpt2_tokenizer(self, max_vocab_size: int) -> Dict[int, str]:
        """
        Load GPT-2 tokenizer vocabulary (simulated)
        In practice, this would use the transformers library
        """
        try:
            # Try to import transformers for real tokenizer
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            # Get the vocabulary and limit to max_vocab_size
            vocab = tokenizer.get_vocab()
            # Sort by token IDs and take first max_vocab_size
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:max_vocab_size]
            
            # Create mapping from index to token
            vocab_mapping = {idx: token for token, idx in sorted_vocab}
            print(f"   ✓ Loaded real GPT-2 tokenizer")
            
        except ImportError:
            print(f"   ⚠️  transformers library not available, using simulated GPT-2 vocab")
            # Simulated GPT-2 vocabulary with common patterns
            vocab_mapping = {
                0: '<|endoftext|>', 1: '!', 2: '"', 3: '#', 4: '$', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')',
                10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3',
                20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '<', 29: '=',
                30: '>', 31: '?', 32: '@', 33: 'A', 34: 'B', 35: 'C', 36: 'D', 37: 'E', 38: 'F', 39: 'G',
                40: 'H', 41: 'I', 42: 'J', 43: 'K', 44: 'L', 45: 'M', 46: 'N', 47: 'O', 48: 'P', 49: 'Q',
                50: 'R', 51: 'S', 52: 'T', 53: 'U', 54: 'V', 55: 'W', 56: 'X', 57: 'Y', 58: 'Z', 59: '[',
                60: '\\', 61: ']', 62: '^', 63: '_', 64: '`', 65: 'a', 66: 'b', 67: 'c', 68: 'd', 69: 'e',
                70: 'f', 71: 'g', 72: 'h', 73: 'i', 74: 'j', 75: 'k', 76: 'l', 77: 'm', 78: 'n', 79: 'o',
                80: 'p', 81: 'q', 82: 'r', 83: 's', 84: 't', 85: 'u', 86: 'v', 87: 'w', 88: 'x', 89: 'y',
                90: 'z', 91: '{', 92: '|', 93: '}', 94: '~', 95: '¡', 96: '¢', 97: '£', 98: '¤', 99: '¥'
            }
            
            # Add common words and subwords
            common_tokens = [
                'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are',
                'as', 'with', 'his', 'they', 'I', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by',
                'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each',
                'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then',
                'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
                'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find',
                'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part', 'over', 'new', 'sound', 'take',
                'only', 'little', 'work', 'know', 'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very',
                'after', 'thing', 'our', 'just', 'name', 'good', 'sentence', 'man', 'think', 'say', 'great', 'where',
                'help', 'through', 'much', 'before', 'line', 'right', 'too', 'mean', 'old', 'any', 'same', 'tell'
            ]
            
            idx = 100
            for token in common_tokens:
                if idx < max_vocab_size:
                    vocab_mapping[idx] = token
                    idx += 1
            
            # Fill remaining with subword patterns
            subword_patterns = ['ing', 'ed', 'er', 's', 'ly', 'tion', 'al', 'ness', 'ment', 'ful', 'less', 'able']
            for pattern in subword_patterns:
                if idx < max_vocab_size:
                    vocab_mapping[idx] = pattern
                    idx += 1
            
            # Fill rest with generated tokens
            while idx < max_vocab_size:
                vocab_mapping[idx] = f'Ġtoken_{idx}'  # Ġ prefix simulates GPT-2 space encoding
                idx += 1
        
        return vocab_mapping
    
    def _load_bert_tokenizer(self, max_vocab_size: int) -> Dict[int, str]:
        """
        Load BERT tokenizer vocabulary (simulated)
        """
        try:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Get the vocabulary and limit to max_vocab_size
            vocab = tokenizer.get_vocab()
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:max_vocab_size]
            vocab_mapping = {idx: token for token, idx in sorted_vocab}
            print(f"   ✓ Loaded real BERT tokenizer")
            
        except ImportError:
            print(f"   ⚠️  transformers library not available, using simulated BERT vocab")
            # Simulated BERT vocabulary
            vocab_mapping = {
                0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]', 4: '[MASK]',
                5: '!', 6: '"', 7: '#', 8: '$', 9: '%', 10: '&', 11: "'", 12: '(', 13: ')', 14: '*',
                15: '+', 16: ',', 17: '-', 18: '.', 19: '/', 20: '0', 21: '1', 22: '2', 23: '3', 24: '4'
            }
            
            # Add alphabet
            for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
                vocab_mapping[25 + i] = char
            
            # Add common words
            common_words = [
                'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are',
                'as', 'with', 'his', 'they', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word',
                'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which',
                'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them'
            ]
            
            idx = 51
            for word in common_words:
                if idx < max_vocab_size:
                    vocab_mapping[idx] = word
                    idx += 1
            
            # Add WordPiece subwords (BERT style)
            subwords = ['##ing', '##ed', '##er', '##s', '##ly', '##tion', '##al', '##ness', '##ment', '##ful']
            for subword in subwords:
                if idx < max_vocab_size:
                    vocab_mapping[idx] = subword
                    idx += 1
            
            # Fill rest
            while idx < max_vocab_size:
                vocab_mapping[idx] = f'word_{idx}'
                idx += 1
        
        return vocab_mapping
    
    def _load_mistral_tokenizer(self, max_vocab_size: Optional[int] = None) -> Dict[int, str]:
        """
        Load Mistral v3 Tekken tokenizer vocabulary
        """
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy
            
            # Load the Mistral v3 Tekken tokenizer with explicit special token policy
            tokenizer = MistralTokenizer.v3(is_tekken=True)
            
            # Get the actual vocabulary size from the tokenizer
            vocab_size = getattr(tokenizer, 'vocab_size', 32000)
            
            # Use actual vocab size unless a smaller max is specifically requested
            if max_vocab_size is None:
                actual_vocab_size = vocab_size
                print(f"   📊 Using full Mistral v3 Tekken vocabulary size: {actual_vocab_size}")
            else:
                actual_vocab_size = min(vocab_size, max_vocab_size)
                if actual_vocab_size < vocab_size:
                    print(f"   ⚠️  Limiting vocabulary to {actual_vocab_size} tokens (full size: {vocab_size})")
                else:
                    print(f"   📊 Using full Mistral v3 Tekken vocabulary size: {actual_vocab_size}")
            
            vocab_mapping = {}
            
            # Method 1: Try to decode individual token IDs (Default - more reliable)
            try:
                print(f"   📝 Attempting to decode {actual_vocab_size} tokens...")
                for i in range(actual_vocab_size):
                    try:
                        # Use the tokenizer's decode method with special token policy
                        token = tokenizer.decode([i], special_token_policy=SpecialTokenPolicy.KEEP)
                        if token and token.strip():  # Only add non-empty tokens
                            vocab_mapping[i] = token
                    except Exception as decode_err:
                        # If individual decode fails, create placeholder
                        vocab_mapping[i] = f'<token_{i}>'
                
                if len(vocab_mapping) > 0:
                    print(f"   ✓ Successfully decoded {len(vocab_mapping)} tokens")
                else:
                    raise ValueError("Failed to decode any tokens")
                    
            except Exception as e1:
                print(f"   ⚠️  Method 1 (decode) failed: {e1}")
                
                # Method 2: Try to get vocabulary directly from the tokenizer model (Fallback)
                try:
                    if hasattr(tokenizer, '_model') and hasattr(tokenizer._model, 'get_vocab'):
                        vocab = tokenizer._model.get_vocab()
                        if vocab and len(vocab) > 0:
                            # Sort by token IDs and take first max_vocab_size
                            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:actual_vocab_size]
                            vocab_mapping = {idx: token for token, idx in sorted_vocab}
                            print(f"   ✓ Extracted vocabulary via get_vocab() method")
                        else:
                            raise ValueError("Empty vocabulary from get_vocab()")
                    else:
                        raise ValueError("No get_vocab() method available")
                except Exception as e2:
                    print(f"   ⚠️  Method 2 (get_vocab) failed: {e2}")
                    raise ImportError("All vocabulary extraction methods failed")
            
            print(f"   ✓ Loaded real Mistral v3 Tekken tokenizer")
            
        except ImportError as e:
            raise ImportError(f"   ⚠️  mistral_common library not available: {e}")
        
        except Exception as e:
            raise ImportError(f"   ❌ Failed to load Mistral tokenizer: {e}")
            
        return vocab_mapping
    
    def _load_basic_english_tokenizer(self, max_vocab_size: int) -> Dict[int, str]:
        """
        Load a basic English tokenizer with most common words
        """
        vocab_mapping = {
            0: '<pad>', 1: '<unk>', 2: '<start>', 3: '<end>', 4: '<mask>',
        }
        
        # Most common English words
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
            'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
            'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
            'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use',
            'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us', 'is', 'water', 'long', 'very', 'when', 'much', 'before', 'here', 'through',
            'when', 'much', 'before', 'move', 'right', 'boy', 'old', 'too', 'same', 'tell', 'does', 'set', 'three',
            'want', 'air', 'well', 'also', 'play', 'small', 'end', 'put', 'home', 'read', 'hand', 'port', 'large',
            'spell', 'add', 'even', 'land', 'here', 'must', 'big', 'high', 'such', 'follow', 'act', 'why', 'ask',
            'men', 'change', 'went', 'light', 'kind', 'off', 'need', 'house', 'picture', 'try', 'us', 'again',
            'animal', 'point', 'mother', 'world', 'near', 'build', 'self', 'earth', 'father', 'head', 'stand',
            'own', 'page', 'should', 'country', 'found', 'answer', 'school', 'grow', 'study', 'still', 'learn',
            'plant', 'cover', 'food', 'sun', 'four', 'between', 'state', 'keep', 'eye', 'never', 'last', 'let',
            'thought', 'city', 'tree', 'cross', 'farm', 'hard', 'start', 'might', 'story', 'saw', 'far', 'sea',
            'draw', 'left', 'late', 'run', 'while', 'press', 'close', 'night', 'real', 'life', 'few', 'north'
        ]
        
        idx = 5
        for word in common_words:
            if idx < max_vocab_size:
                vocab_mapping[idx] = word
                idx += 1
        
        # Add punctuation
        punctuation = ['.', ',', '!', '?', ';', ':', "'", '"', '(', ')', '[', ']', '{', '}', '-', '_']
        for punct in punctuation:
            if idx < max_vocab_size:
                vocab_mapping[idx] = punct
                idx += 1
        
        # Add numbers
        for i in range(100):
            if idx < max_vocab_size:
                vocab_mapping[idx] = str(i)
                idx += 1
        
        # Fill rest with generated tokens
        while idx < max_vocab_size:
            vocab_mapping[idx] = f'word_{idx}'
            idx += 1
        
        return vocab_mapping
    
    def _detect_tokenizer_type(self) -> str:
        """Detect the type of tokenizer currently in use"""
        if not hasattr(self.brain, 'output_config') or not self.brain.output_config:
            return 'none'
        
        output_config = self.brain.output_config
        vocab_mapping = output_config.get('vocab_mapping', {})
        
        if not vocab_mapping:
            return 'none'
        
        # Check for Mistral tokenizer patterns
        sample_tokens = list(vocab_mapping.values())[:10]
        if any('<unk>' in str(token) or '[INST]' in str(token) or '[/INST]' in str(token) for token in sample_tokens):
            return 'mistral_v3_tekken'
        
        # Check for GPT-2 tokenizer patterns
        if any('<|endoftext|>' in str(token) or 'Ġ' in str(token) for token in sample_tokens):
            return 'gpt2'
        
        # Check for BERT tokenizer patterns
        if any('[PAD]' in str(token) or '[CLS]' in str(token) or '##' in str(token) for token in sample_tokens):
            return 'bert'
        
        # Check for basic English tokenizer patterns
        if '<pad>' in vocab_mapping.get(0, '') and '<unk>' in vocab_mapping.get(1, ''):
            return 'basic_english'
        
        # Default to custom if none of the above match
        return 'custom'
    
    def initialize_new_brain(self, demo_mode: bool = False) -> bool:
        """Initialize a new brain instance with configurable output system"""
        print(f"\n🧠 Initializing {'Demo' if demo_mode else 'Full'} BrainNexus...")
        
        # Step 1: Configure output system
        print("\n📋 First, let's configure the output system for your use case:")
        configure_output = input("Configure custom output system? (y/n, default n): ").strip().lower()
        
        output_config = None
        if configure_output == 'y':
            output_config = self.configure_output_system()
        else:
            # Use default configuration
            print("✓ Using default 10-class classification system")
            output_config = {
                'type': 'classification',
                'num_classes': 10,
                'class_labels': [f'class_{i}' for i in range(10)],
                'output_format': 'index',
                'confidence_threshold': 0.7,
                'return_top_k': 1,
                'normalize_probabilities': True
            }
        
        # Step 1.5: Auto-enable multi-token generation for tokenizer-based models
        self._configure_multi_token_generation(output_config)
        
        # Step 2: Create learning-capable brain with output configuration
        training_config = TrainingConfig(
            learning_rate=0.01,
            spatial_learning_rate=0.005,
            batch_size=8,
            max_epochs=50
        )
        
        # Create BrainNexusLearn with output configuration
        self.learner = BrainNexusLearn(demo=demo_mode, config=training_config, output_config=output_config)
        self.brain = self.learner  # BrainNexusLearn extends BrainNexus
        
        # Step 3: Initialize the brain structure
        try:
            print(f"\n🏗️  Building neural architecture...")
            node_map = self.brain.initialize_brain()
            if node_map and len(self.brain.neural_nodes) > 0:
                node_count = len(self.brain.neural_nodes)
                comp_count = len(self.brain.get_nodes_by_type('Computational'))
                print(f"✓ Brain initialized successfully!")
                print(f"  Total nodes: {node_count}")
                print(f"  Computational nodes: {comp_count}")
                print(f"  Architecture: {'4 Quadrants (2D)' if demo_mode else '8 Octants (3D)'}")
                
                # Display output configuration summary
                print(f"\n📊 Output System Configuration:")
                print(f"  Type: {output_config['type']}")
                print(f"  Classes/Vocabulary: {output_config['num_classes']}")
                print(f"  Output Format: {output_config['output_format']}")
                print(f"  Confidence Threshold: {output_config['confidence_threshold']}")
                print(f"  Top-K Predictions: {output_config['return_top_k']}")
                
                # Show multi-token status
                if output_config.get('enable_multi_token', False):
                    print(f"\n🚀 Multi-Token Generation: ENABLED")
                    print(f"  Generation Method: {output_config.get('generation_method', 'hybrid_context_aware')}")
                    print(f"  Max Sequence Length: {output_config.get('max_sequence_length', 10)}")
                    print(f"  Context Window: {output_config.get('context_window', 5)}")
                    print(f"  Node Group Behaviors: {len(output_config.get('node_group_behaviors', {}))}")
                else:
                    print(f"\n📌 Single-Token Mode: Standard prediction")
                
                if output_config['type'] == 'tokens' and 'vocab_mapping' in output_config:
                    sample_tokens = list(output_config['vocab_mapping'].values())[:5]
                    print(f"  Sample Tokens: {sample_tokens}...")
                elif len(output_config['class_labels']) <= 10:
                    print(f"  Class Labels: {output_config['class_labels']}")
                else:
                    print(f"  Class Labels: {output_config['class_labels'][:5]}... (and {len(output_config['class_labels'])-5} more)")
                
                return True
            else:
                print("❌ Failed to initialize brain structure")
                return False
        except Exception as e:
            print(f"❌ Failed to initialize brain: {e}")
            return False
    
    def setup_training_data(self, num_samples: int = 100) -> bool:
        """Setup training with generated sample data"""
        if self.learner is None:
            print("❌ No brain instance available. Initialize a brain first.")
            return False
        
        try:
            # Generate sample training data
            training_data = []
            for _ in range(num_samples):
                # Create random input (simulating feature vector)
                input_data = {
                    'features': np.random.randn(10).tolist(),
                    'sequence_length': random.randint(5, 20)
                }
                # Create target (random classification for demo)
                target = random.randint(0, 9)  # 10-class classification
                training_data.append((input_data, target))
            
            # Store training data in the manager
            self.training_data = training_data
            
            # Generate validation data
            validation_data = []
            for _ in range(num_samples // 5):  # 20% for validation
                input_data = {
                    'features': np.random.randn(10).tolist(),
                    'sequence_length': random.randint(5, 20)
                }
                target = random.randint(0, 9)
                validation_data.append((input_data, target))
            
            self.validation_data = validation_data
            
            print(f"✓ Training data generated: {len(training_data)} training, {len(validation_data)} validation examples")
            return True
        except Exception as e:
            print(f"❌ Failed to setup training data: {e}")
            return False
    
    def load_training_data(self, filename: str) -> bool:
        """Load training data from file"""
        if not os.path.exists(filename):
            print(f"❌ Training file not found: {filename}")
            return False
        
        try:
            # Try to load as JSON first, then pickle
            if filename.endswith('.json'):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.training_data = data.get('training_data', [])
                    self.validation_data = data.get('validation_data', [])
            else:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.training_data = data.get('training_data', [])
                    self.validation_data = data.get('validation_data', [])
            
            print(f"✓ Loaded {len(self.training_data)} training examples from {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to load training data: {e}")
            return False
    
    def load_conversation_data(self, filename: str) -> bool:
        """Load conversation training data from file"""
        if not os.path.exists(filename):
            print(f"❌ Conversation file not found: {filename}")
            return False
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations = data.get('conversations', [])
                    validation_conversations = data.get('validation_conversations', [])
            else:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    conversations = data.get('conversations', [])
                    validation_conversations = data.get('validation_conversations', [])
            
            # Store conversation data
            self.training_data = conversations
            self.validation_data = validation_conversations
            
            print(f"✓ Loaded {len(conversations)} conversations from {filename}")
            if validation_conversations:
                print(f"  + {len(validation_conversations)} validation conversations")
            return True
        except Exception as e:
            print(f"❌ Failed to load conversation data: {e}")
            return False
    
    def load_sequence_data(self, filename: str) -> bool:
        """Load sequence training data from file"""
        if not os.path.exists(filename):
            print(f"❌ Sequence file not found: {filename}")
            return False
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sequences = data.get('sequences', [])
                    validation_sequences = data.get('validation_sequences', [])
            else:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    sequences = data.get('sequences', [])
                    validation_sequences = data.get('validation_sequences', [])
            
            # Store sequence data
            self.training_data = sequences
            self.validation_data = validation_sequences
            
            print(f"✓ Loaded {len(sequences)} sequences from {filename}")
            if validation_sequences:
                print(f"  + {len(validation_sequences)} validation sequences")
            return True
        except Exception as e:
            print(f"❌ Failed to load sequence data: {e}")
            return False
    
    def train_on_conversations(self, conversations: Optional[List] = None, 
                              validation_conversations: Optional[List] = None) -> bool:
        """Train the brain on conversation data"""
        if self.learner is None:
            print("❌ No learning brain instance available. Initialize a brain first.")
            return False
        
        # Use provided data or stored data
        conv_data = conversations if conversations is not None else self.training_data
        val_data = validation_conversations if validation_conversations is not None else self.validation_data
        
        if not conv_data:
            print("❌ No conversation data available. Load conversation data first.")
            return False
        
        try:
            print(f"🗣️  Starting conversation training...")
            results = self.learner.conversation_train(conv_data, val_data)
            
            print(f"✅ Conversation training completed!")
            print(f"   Training time: {results['training_time']:.2f}s")
            print(f"   Final loss: {results['final_loss']:.4f}")
            print(f"   Epochs: {results['epochs_completed']}")
            
            return True
        except Exception as e:
            print(f"❌ Conversation training failed: {e}")
            return False
    
    def train_on_sequences(self, sequences: Optional[List] = None, 
                          validation_sequences: Optional[List] = None) -> bool:
        """Train the brain on sequence data"""
        if self.learner is None:
            print("❌ No learning brain instance available. Initialize a brain first.")
            return False
        
        # Use provided data or stored data
        seq_data = sequences if sequences is not None else self.training_data
        val_data = validation_sequences if validation_sequences is not None else self.validation_data
        
        if not seq_data:
            print("❌ No sequence data available. Load sequence data first.")
            return False
        
        try:
            print(f"📊 Starting sequence training...")
            results = self.learner.sequence_train(seq_data, val_data)
            
            print(f"✅ Sequence training completed!")
            print(f"   Training time: {results['training_time']:.2f}s")
            print(f"   Final loss: {results['final_loss']:.4f}")
            print(f"   Epochs: {results['epochs_completed']}")
            
            return True
        except Exception as e:
            print(f"❌ Sequence training failed: {e}")
            return False
    
    def train_unsupervised(self, data: Optional[List] = None, method: str = 'autoencoder') -> bool:
        """Train the brain using unsupervised learning"""
        if self.learner is None:
            print("❌ No learning brain instance available. Initialize a brain first.")
            return False
        
        # Use provided data or stored data (extract inputs only for unsupervised)
        unsup_data = data
        if unsup_data is None:
            if self.training_data:
                # Extract inputs from supervised data
                unsup_data = [item[0] if isinstance(item, (tuple, list)) and len(item) >= 2 else item 
                             for item in self.training_data]
            else:
                print("❌ No data available for unsupervised training.")
                return False
        
        try:
            print(f"🔬 Starting unsupervised training ({method})...")
            results = self.learner.unsupervised_train(unsup_data, method)
            
            print(f"✅ Unsupervised training completed!")
            print(f"   Training time: {results['training_time']:.2f}s")
            print(f"   Method: {results.get('method', method)}")
            
            if 'reconstruction_loss' in results:
                print(f"   Reconstruction loss: {results['reconstruction_loss']:.4f}")
            if 'inertia' in results:
                print(f"   Clustering inertia: {results['inertia']:.4f}")
                print(f"   Number of clusters: {results.get('num_clusters', 'unknown')}")
            
            return True
        except Exception as e:
            print(f"❌ Unsupervised training failed: {e}")
            return False
    
    def train_hybrid(self, supervised_data: Optional[List] = None, 
                    unsupervised_data: Optional[List] = None) -> bool:
        """Train the brain using both supervised and unsupervised learning"""
        if self.learner is None:
            print("❌ No learning brain instance available. Initialize a brain first.")
            return False
        
        # Use provided data or stored data
        sup_data = supervised_data if supervised_data is not None else self.training_data
        
        # Generate unsupervised data from supervised data if not provided
        if unsupervised_data is None and sup_data:
            unsupervised_data = [item[0] if isinstance(item, (tuple, list)) and len(item) >= 2 else item 
                               for item in sup_data]
        
        if not sup_data or not unsupervised_data:
            print("❌ Need both supervised and unsupervised data for hybrid training.")
            return False
        
        try:
            print(f"🔄 Starting hybrid training...")
            results = self.learner.hybrid_train(sup_data, unsupervised_data, self.validation_data)
            
            print(f"✅ Hybrid training completed!")
            print(f"   Training time: {results['training_time']:.2f}s")
            print(f"   Final combined loss: {results['final_loss']:.4f}")
            print(f"   Epochs: {results['epochs_completed']}")
            
            return True
        except Exception as e:
            print(f"❌ Hybrid training failed: {e}")
            return False
    
    def train_on_flat_text(self, text: str, text_title: str = "", text_type: str = "document",
                          context_window: int = 10, training_approach: str = "all") -> bool:
        """
        Train BrainNexus directly on flat text (Wikipedia pages, books, articles, etc.)
        
        Args:
            text: Raw text content to train on
            text_title: Title/name of the text source
            text_type: Type of text ("wikipedia", "book", "article", "webpage", etc.)
            context_window: Number of tokens to use as input context
            training_approach: "all", "token_prediction", "sentence_completion", "paragraph_continuation"
            
        Returns:
            True if training succeeded, False otherwise
        """
        if self.learner is None:
            print("❌ No learning brain instance available. Initialize a brain first.")
            return False
        
        if not text or len(text.strip()) < 100:
            print("❌ Text is too short for training. Provide at least 100 characters.")
            return False
        
        try:
            print(f"📖 Starting flat text training...")
            print(f"   Title: {text_title or 'Untitled'}")
            print(f"   Type: {text_type}")
            print(f"   Text length: {len(text)} characters")
            print(f"   Approach: {training_approach}")
            
            results = self.learner.train_on_flat_text(
                text=text,
                text_title=text_title,
                text_type=text_type,
                context_window=context_window,
                training_approach=training_approach
            )
            
            print(f"✅ Flat text training completed!")
            print(f"   Training time: {results['training_time']:.2f}s")
            print(f"   Final accuracy: {results.get('final_accuracy', 0):.3f}")
            print(f"   Vocabulary size: {results['text_metadata']['vocabulary_size']}")
            print(f"   Total samples: {results['text_metadata']['total_samples']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Flat text training failed: {e}")
            return False
    
    def train_on_file(self, file_path: str, text_type: str = "document", 
                     context_window: int = 10, training_approach: str = "all") -> bool:
        """
        Train BrainNexus on a text file.
        
        Args:
            file_path: Path to the text file
            text_type: Type of text ("wikipedia", "book", "article", etc.)
            context_window: Number of tokens to use as input context  
            training_approach: Training approach to use
            
        Returns:
            True if training succeeded, False otherwise
        """
        try:
            # Load text from file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract title from filename
            import os
            text_title = os.path.splitext(os.path.basename(file_path))[0]
            
            print(f"📄 Loading text from: {file_path}")
            
            return self.train_on_flat_text(
                text=text,
                text_title=text_title,
                text_type=text_type,
                context_window=context_window,
                training_approach=training_approach
            )
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return False
        except Exception as e:
            print(f"❌ Failed to load file: {e}")
            return False
    
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
        if self.learner is None:
            print("❌ No learning brain instance available. Initialize a brain first.")
            return ""
        
        try:
            generated_text = self.learner.generate_text(prompt, max_length, temperature)
            return generated_text
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            return ""
    
    def run_inference(self, input_data: str) -> Dict[str, Any]:
        """Run inference on input data"""
        if self.brain is None:
            raise ValueError("No brain instance available")
        
        start_time = time.time()
        brain_result = self.brain.run(input_data)
        inference_time = time.time() - start_time
        
        # Extract the actual result from nested structure
        actual_result = brain_result.get('result', {})
        
        # Create flattened result for backward compatibility
        flattened_result = {
            **actual_result,  # Include all the formatted output fields
            'pipeline_time': brain_result.get('execution_time', inference_time),
            'inference_time': inference_time,
            'execution_trace': brain_result.get('trace', {}),
            'metadata': brain_result.get('metadata', {})
        }
        
        return flattened_result
    def traceroute(self, input_data: str, output_filename: Optional[str] = None, create_video: bool = True):
        """
        Trace a computation through the brain network and optionally create visualization
        
        Args:
            input_data: Input to process
            output_filename: Optional filename prefix for saved files
            create_video: Whether to create animated visualization
        """
        if self.brain is None:
            print("❌ No brain instance available")
            return None
            
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"traceroute_{timestamp}"
        
        print(f"🔍 Starting traceroute for input: '{input_data}'")
        
        # Create recorder and patch brain methods to record events
        recorder = BrainTraceRecorder()
        original_methods = self._patch_brain_for_tracing(recorder)
        
        try:
            # Record the computation
            recorder.start_recording()
            start_time = time.time()
            
            result = self.brain.run(input_data)
            total_time = time.time() - start_time
            recorder.stop_recording()
            
            print(f"✓ Computation completed in {total_time:.4f}s")
            print(f"✓ Recorded {len(recorder.trace_events)} events")
            
            # Save trace data
            trace_data = {
                'input_data': input_data,
                'result': result,
                'total_time': total_time,
                'events': recorder.trace_events,
                'brain_stats': {
                    'mode': 'demo' if self.brain.demo else 'full',
                    'total_nodes': len(self.brain.node_records),
                    'dimensions': self.brain.dimensions
                }
            }
            
            json_path = os.path.join(self.save_directory, f"{output_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)
            print(f"✓ Trace data saved to: {json_path}")
            
            # Create visualization if requested
            if create_video:
                print("🎬 Creating visualization...")
                video_path = self._create_traceroute_video(recorder.trace_events, output_filename)
                if video_path:
                    print(f"✓ Video saved to: {video_path}")
            
            # Print summary
            self._print_trace_summary(recorder.trace_events, total_time)
            
            return trace_data
            
        finally:
            # Restore original methods
            self._restore_brain_methods(original_methods)

    def _patch_brain_for_tracing(self, recorder: BrainTraceRecorder) -> Dict[str, Any]:
        """Patch brain methods to record trace events"""
        original_methods = {}
        
        # Patch the main run method
    
        original_methods['run'] = self.brain.run
        
        def traced_run(input_data, *args, **kwargs):
            recorder.log_event('computation_start', 'BRAIN', {
                'input_data': str(input_data)[:100],
                'method': 'run'
            })
            
            result = original_methods['run'](input_data, *args, **kwargs)
            
            recorder.log_event('computation_complete', 'BRAIN', {
                'output_data': str(result.get('final_output'))[:100] if result else None,
                'method': 'run',
                'inference_cost': result.get('inference_cost', 0) if result else 0
            })
            
            return result

        self.brain.run = traced_run

        # Patch intelligent_routing method
        if hasattr(self.brain, 'intelligent_routing'):
            original_methods['intelligent_routing'] = self.brain.intelligent_routing
            
            def traced_intelligent_routing(tokens, *args, **kwargs):
                recorder.log_event('routing_start', 'ROUTER', {
                    'input_tokens': len(tokens) if hasattr(tokens, '__len__') else str(tokens)[:50],
                    'method': 'intelligent_routing'
                })
                
                result = original_methods['intelligent_routing'](tokens, *args, **kwargs)
                
                recorder.log_event('routing_complete', 'ROUTER', {
                    'routing_decisions': len(result) if result else 0,
                    'method': 'intelligent_routing'
                })
                
                return result
                
            self.brain.intelligent_routing = traced_intelligent_routing
        
        # Patch individual node run methods
        if hasattr(self.brain, 'node_registry'):
            for node_id, node in self.brain.node_registry.items():
                if hasattr(node, 'run'):
                    # Store original method
                    original_methods[f'node_{node_id}_run'] = node.run
                    
                    def create_traced_node_run(original_node_run, traced_node_id):
                        def traced_node_run(data, *args, **kwargs):
                            recorder.log_event('node_start', str(traced_node_id), {
                                'input_data': str(data)[:100],
                                'node_type': getattr(node, 'node_type', 'Unknown'),
                                'method': 'run'
                            })
                            
                            result = original_node_run(data, *args, **kwargs)
                            
                            recorder.log_event('node_complete', str(traced_node_id), {
                                'output_data': str(result)[:100] if result is not None else None,
                                'node_type': getattr(node, 'node_type', 'Unknown'),
                                'method': 'run'
                            })
                            
                            return result
                        return traced_node_run
                    
                    # Apply the traced version
                    node.run = create_traced_node_run(node.run, node_id)
        
        # Patch attention-related methods if they exist
        if hasattr(self.brain, 'multi_layer_attention'):
            original_methods['multi_layer_attention'] = self.brain.multi_layer_attention
            
            def traced_multi_layer_attention(*args, **kwargs):
                recorder.log_event('attention_start', 'ATTENTION', {
                    'method': 'multi_layer_attention',
                    'layer': kwargs.get('layer_idx', 'unknown')
                })
                
                result = original_methods['multi_layer_attention'](*args, **kwargs)
                
                recorder.log_event('attention_complete', 'ATTENTION', {
                    'method': 'multi_layer_attention',
                    'layer': kwargs.get('layer_idx', 'unknown')
                })
                
                return result
                
            self.brain.multi_layer_attention = traced_multi_layer_attention
        
        return original_methods

    def _restore_brain_methods(self, original_methods: Dict[str, Any]):
        """Restore original brain methods"""
        for method_name, original_method in original_methods.items():
            if method_name.startswith('node_') and method_name.endswith('_run'):
                # Restore individual node methods
                node_id = method_name.split('_')[1]  # Extract node_id from 'node_X_run'
                if hasattr(self.brain, 'node_registry') and int(node_id) in self.brain.node_registry:
                    self.brain.node_registry[int(node_id)].run = original_method
            elif hasattr(self.brain, method_name):
                # Restore brain-level methods
                setattr(self.brain, method_name, original_method)

    def _get_node_type(self, node_id: str) -> str:
        """Get node type for a given node ID"""
        if self.brain.node_records is not None:
            node_row = self.brain.node_records[self.brain.node_records['Node_ID'] == node_id]
            if not node_row.empty:
                return node_row.iloc[0].get('Node_Type', 'Unknown')
        return 'Unknown'

    def _get_node_position(self, node_id: str) -> Tuple[float, float, float]:
        """Get 3D position for a given node ID"""
        if self.brain.node_records is not None:
            node_row = self.brain.node_records[self.brain.node_records['Node_ID'] == node_id]
            if not node_row.empty:
                pos = node_row.iloc[0]['Node_Position']
                return tuple(pos[:3])
        return (0, 0, 0)

    def _create_traceroute_video(self, events: List[Dict], filename_prefix: str) -> Optional[str]:
        """Create animated visualization of the computation trace"""
        if not events:
            print("❌ No events to visualize")
            return None
        
        # Prepare data
        max_time = max(event['timestamp'] for event in events)
        fps = 30
        duration = max(5.0, max_time)  # At least 5 seconds
        total_frames = int(duration * fps)
        
        # Get node positions and types
        node_positions = {}
        node_types = {}
        for _, row in self.brain.node_records.iterrows():
            node_id = row['Node_ID']
            node_positions[node_id] = row['Node_Position'][:3]
            node_types[node_id] = row['Node_Type']
        
        # Set up the figure
        is_3d = not self.brain.demo
        fig = plt.figure(figsize=(12, 9))
        
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            ax = fig.add_subplot(111)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
        
        # Color mapping for node types
        type_colors = {
            'Controller': 'red',
            'Judge': 'orange', 
            'Splitter': 'yellow',
            'Computational': 'blue',
            'Retainer': 'green',
            'Reviewer': 'purple',
            'Unknown': 'gray'
        }
        
        # Animation data
        active_nodes = set()
        node_activity = defaultdict(float)  # Activity level for each node
        
        def animate(frame):
            ax.clear()
            current_time = (frame / fps) * (max_time / duration)
            
            # Update active nodes based on current time
            for event in events:
                if event['timestamp'] <= current_time:
                    node_id = event['node_id']
                    if event['event_type'] == 'node_start':
                        active_nodes.add(node_id)
                        node_activity[node_id] = 1.0
                    elif event['event_type'] == 'node_complete':
                        if node_id in active_nodes:
                            active_nodes.discard(node_id)
                        node_activity[node_id] = max(0, node_activity[node_id] - 0.1)
            
            # Decay activity levels
            for node_id in list(node_activity.keys()):
                if node_id not in active_nodes:
                    node_activity[node_id] = max(0, node_activity[node_id] - 0.05)
                    if node_activity[node_id] == 0:
                        del node_activity[node_id]
            
            # Draw nodes
            for node_id, pos in node_positions.items():
                node_type = node_types.get(node_id, 'Unknown')
                base_color = type_colors.get(node_type, 'gray')
                activity = node_activity.get(node_id, 0)
                
                # Calculate visual properties based on activity
                alpha = 0.3 + 0.7 * activity
                size = 50 + 200 * activity
                
                if is_3d:
                    ax.scatter(pos[0], pos[1], pos[2], 
                            c=base_color, s=size, alpha=alpha,
                            edgecolors='black', linewidth=1)
                else:
                    ax.scatter(pos[0], pos[1], 
                            c=base_color, s=size, alpha=alpha,
                            edgecolors='black', linewidth=1)
            
            # Draw connections for active nodes
            for node_id in active_nodes:
                if node_id in node_positions:
                    node_row = self.brain.node_records[self.brain.node_records['Node_ID'] == node_id]
                    if not node_row.empty:
                        connections = node_row.iloc[0]['Exit_Connections']
                        x1, y1, z1 = node_positions[node_id]
                        
                        for target_id in connections:
                            if target_id in node_positions:
                                x2, y2, z2 = node_positions[target_id]
                                
                                if is_3d:
                                    ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                        color='red', alpha=0.6, linewidth=2)
                                else:
                                    ax.plot([x1, x2], [y1, y2], 
                                        color='red', alpha=0.6, linewidth=2)
            
            # Set title with current time
            ax.set_title(f'BrainNexus Computation Trace\nTime: {current_time:.3f}s / {max_time:.3f}s\nActive Nodes: {len(active_nodes)}')
            
            # Set axis limits
            if node_positions:
                all_pos = np.array(list(node_positions.values()))
                margin = 1.0
                
                if is_3d:
                    ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
                    ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
                    ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)
                else:
                    ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
                    ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
            
            # Add legend
            legend_elements = []
            for node_type, color in type_colors.items():
                if any(node_types.get(nid) == node_type for nid in node_positions):
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=8, label=node_type))
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Create animation
        print(f"Creating animation with {total_frames} frames at {fps} FPS...")
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, blit=False)
        
        # Save animation
        video_path = os.path.join(self.save_directory, f"{filename_prefix}_trace.mp4")
        try:
            anim.save(video_path, writer='ffmpeg', fps=fps, dpi=100)
            return video_path
        except Exception as e:
            print(f"❌ Failed to save video: {e}")
            print("Note: Requires ffmpeg to be installed for video creation")
            
            # Try saving as GIF instead
            gif_path = os.path.join(self.save_directory, f"{filename_prefix}_trace.gif")
            try:
                anim.save(gif_path, writer='pillow', fps=fps//2)  # Lower FPS for GIF
                print(f"✓ Saved as GIF instead: {gif_path}")
                return gif_path
            except Exception as e2:
                print(f"❌ Also failed to save GIF: {e2}")
                return None

    def _print_trace_summary(self, events: List[Dict], total_time: float):
        """Print a summary of the trace events"""
        print(f"\n📊 Trace Summary:")
        print(f"  Total Events: {len(events)}")
        print(f"  Total Time: {total_time:.4f}s")
        
        # Count events by type
        event_counts = defaultdict(int)
        node_activity = defaultdict(int)
        
        for event in events:
            event_counts[event['event_type']] += 1
            node_activity[event['node_id']] += 1
        
        print(f"\n  Event Breakdown:")
        for event_type, count in event_counts.items():
            print(f"    {event_type}: {count}")
        
        print(f"\n  Most Active Nodes:")
        sorted_nodes = sorted(node_activity.items(), key=lambda x: x[1], reverse=True)
        for node_id, activity in sorted_nodes[:5]:
            node_type = self._get_node_type(node_id)
            print(f"    {node_id} ({node_type}): {activity} events")
        
        if events:
            avg_time_between = total_time / len(events)
            print(f"\n  Average time between events: {avg_time_between:.6f}s")

def print_header():
    """Print application header"""
    print("=" * 60)
    print("🧠 BrainNexus - Spatially-Organized Neural Network")
    print("=" * 60)
    print("A novel neural architecture using 3D spatial organization")
    print("for distributed computation and adaptive learning.")
    print()

def print_architecture_analysis():
    """Print pros/cons analysis of the spatial architecture"""
    print("\n" + "="*60)
    print("🏗️  SPATIAL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    print("\n✅ POTENTIAL ADVANTAGES:")
    print("1. 🌐 SPATIAL LOCALITY:")
    print("   - Nodes process related information based on physical proximity")
    print("   - Natural clustering of similar computations")
    print("   - Reduced 'connection distance' for related operations")
    
    print("\n2. 🔄 DYNAMIC ADAPTATION:")
    print("   - Nodes can physically move toward optimal positions")
    print("   - Connections rewire based on spatial relationships")
    print("   - Self-organizing network topology")
    
    print("\n3. 🎯 SPECIALIZED REGIONS:")
    print("   - Different quadrants/octants can specialize for different tasks")
    print("   - Natural load balancing across spatial regions")
    print("   - Hierarchical processing from center to periphery")
    
    print("\n4. 🧬 BIOLOGICAL INSPIRATION:")
    print("   - Mimics brain cortical organization")
    print("   - Spatial memory and processing patterns")
    print("   - Natural fault tolerance through redundancy")
    
    print("\n❌ POTENTIAL DISADVANTAGES:")
    print("1. ⚡ COMPUTATIONAL OVERHEAD:")
    print("   - Spatial distance calculations for every operation")
    print("   - Complex routing logic through multiple node types")
    print("   - Higher memory usage for position tracking")
    
    print("\n2. 🎲 LEARNING INEFFICIENCY:")
    print("   - Random perturbation learning vs gradient-based optimization")
    print("   - No backpropagation - purely evolutionary approach")
    print("   - Potentially slow convergence on complex problems")
    
    print("\n3. 🌀 COMPLEXITY WITHOUT PROVEN BENEFIT:")
    print("   - Multi-layer routing (Controller→Judge→Splitter→Comp→Retainer→Reviewer)")
    print("   - Unclear why spatial organization improves performance")
    print("   - May not scale well to very large networks")
    
    print("\n4. 🔧 IMPLEMENTATION CHALLENGES:")
    print("   - Difficult to debug and interpret")
    print("   - No established best practices")
    print("   - Limited theoretical foundation")
    
    print("\n🤔 SPATIAL SIGNIFICANCE:")
    print("• The 3D positioning creates implicit feature clustering")
    print("• Distance-based connections may discover natural data relationships")
    print("• Could be powerful for problems with inherent spatial structure")
    print("• May struggle with abstract/non-spatial learning tasks")
    
    print("\n💡 BEST USE CASES:")
    print("• Image/video processing (natural 2D/3D structure)")
    print("• Robotics and control systems (spatial awareness)")
    print("• Pattern recognition with geometric features")
    print("• Problems requiring adaptive network topology")

def main():
    """Main application loop"""
    print_header()
    
    manager = BrainNexusManager()
    
    while True:
        print("\n" + "="*50)
        print("🎮 BRAINNEXUS CONTROL PANEL")
        print("="*50)
        print("1. Initialize New Brain (Demo Mode) - 🔧 WITH OUTPUT CONFIGURATION")
        print("2. Initialize New Brain (Full Mode) - 🔧 WITH OUTPUT CONFIGURATION")
        print("3. Save Current Brain State")
        print("4. Load Brain State")
        print("5. List Saved States")
        print("6. Generate Training Data")
        print("7. Load Training Data from File")
        print("8. Run Supervised Training")
        print("9. Run Reinforcement Learning")
        print("10. Run Inference")
        print("11. View Brain Statistics")
        print("12. Architecture Analysis")
        print("13. Test Spatial Optimization")
        print("14. List All Node Details")
        print("15. Visualize Brain (Smart - 2D/3D based on mode)")
        print("16. Visualize Brain (Both 2D and 3D)")
        print("17. Run Traceroute (with video)")
        print("18. Run Traceroute (no video)")
        print("19. Train on Flat Text (Direct Input)")
        print("20. Train on Text File")
        print("21. Generate Text from Prompt")
        print("22. Exit")
        print("\n🎯 NEW FEATURES:")
        print("   • Configurable output systems (Classification, Tokens, Custom)")
        print("   • Multiple output formats (Labels, Indices, Token IDs)")
        print("   • Top-K predictions with confidence scoring")
        print("   • Custom vocabularies for language models")
        print("   🚀 MULTI-TOKEN GENERATION: Sequence generation with context-aware selection")
        print("   🎭 NODE GROUP BEHAVIORS: Different generation strategies per node type")
        print("   📖 FLAT TEXT TRAINING: Train directly on Wikipedia pages, books, articles!")
        print("   🎨 TEXT GENERATION: Generate text continuations from prompts!")

        try:
            choice = input("\nEnter your choice (1-22): ").strip()

            if choice == '1':
                manager.initialize_new_brain(demo_mode=True)
            
            elif choice == '2':
                manager.initialize_new_brain(demo_mode=False)
            
            elif choice == '3':
                if manager.brain is None:
                    print("❌ No brain instance to save. Initialize a brain first.")
                else:
                    filename = input("Enter filename (or press Enter for auto): ").strip()
                    if not filename:
                        filename = None
                    else:
                        print(f"💾 Saving as: {filename if filename.endswith('.pkl') else filename + '.pkl'}")
                    manager.save_brain_state(filename)
            
            elif choice == '4':
                files = manager.list_saved_states()
                if files:
                    try:
                        file_num = int(input("\nEnter file number to load: ")) - 1
                        if 0 <= file_num < len(files):
                            manager.load_brain_state(files[file_num])
                        else:
                            print("❌ Invalid file number")
                    except ValueError:
                        print("❌ Please enter a valid number")
            
            elif choice == '5':
                manager.list_saved_states()
            
            elif choice == '6':
                num_samples = input("Enter number of training samples (default 100): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 100
                manager.setup_training_data(num_samples)
            
            elif choice == '7':
                training_file = input("Enter training data file path: ").strip()
                manager.load_training_data(training_file)
            
            elif choice == '8':
                if manager.learner is None:
                    print("❌ No learner available. Initialize a brain first.")
                elif not manager.training_data:
                    print("❌ No training data. Use option 6 or 7 first.")
                else:
                    max_epochs = input("Enter max epochs (default 20): ").strip()
                    max_epochs = int(max_epochs) if max_epochs.isdigit() else 20
                    
                    # Update config
                    manager.learner.config.max_epochs = max_epochs
                    
                    print(f"\n🎓 Starting supervised training...")
                    start_time = time.time()
                    
                    validation_data = manager.validation_data if manager.validation_data else None
                    results = manager.learner.supervised_train(manager.training_data, validation_data)
                    
                    training_time = time.time() - start_time
                    
                    print(f"\n📊 Training Results:")
                    print(f"  Final Loss: {results['final_loss']:.4f}")
                    print(f"  Final Accuracy: {results['final_accuracy']:.3f}")
                    print(f"  Spatial Efficiency: {results['spatial_efficiency']:.3f}")
                    print(f"  Epochs: {results['epochs_completed']}")
                    print(f"  Time: {training_time:.2f} seconds")
            
            elif choice == '9':
                if manager.learner is None:
                    print("❌ No learner available. Initialize a brain first.")
                else:
                    episodes = input("Enter number of episodes (default 200): ").strip()
                    episodes = int(episodes) if episodes.isdigit() else 200
                    
                    print(f"\n� Starting reinforcement learning...")
                    
                    # Simple environment function for demo
                    def simple_environment(action=None, reset=False):
                        if reset:
                            return {'efficiency': 0.5, 'performance': 0.5}
                        
                        if action is None:
                            return None, 0.0, False, {}
                        
                        action_type = action.get('type', 'no_action')
                        if action_type == 'move_node':
                            reward = random.uniform(-0.2, 0.5)
                        elif action_type == 'adjust_weight':
                            reward = random.uniform(-0.1, 0.3)
                        elif action_type == 'add_connection':
                            reward = random.uniform(0.0, 0.4)
                        elif action_type == 'remove_connection':
                            reward = random.uniform(-0.3, 0.2)
                        else:
                            reward = 0.0
                        
                        new_state = {
                            'efficiency': max(0.0, min(1.0, 0.5 + random.uniform(-0.1, 0.1))),
                            'performance': max(0.0, min(1.0, 0.5 + random.uniform(-0.1, 0.1)))
                        }
                        done = random.random() < 0.1
                        return new_state, reward, done, {'action_type': action_type}
                    
                    start_time = time.time()
                    results = manager.learner.reinforcement_train(simple_environment, episodes)
                    rl_time = time.time() - start_time
                    
                    print(f"\n📊 RL Training Results:")
                    print(f"  Final Average Reward: {results['avg_final_reward']:.4f}")
                    print(f"  Q-table Size: {results['q_table_size']}")
                    print(f"  Failed Episodes: {results['failed_episodes']}")
                    print(f"  Time: {rl_time:.2f} seconds")
            
            elif choice == '10':
                if manager.brain is None:
                    print("❌ No brain instance. Initialize or load a brain first.")
                else:
                    input_data = input("Enter input data: ").strip()
                    if input_data:
                        print("\n🧠 Running inference...")
                        
                        # Check if multi-token is enabled
                        multi_token_enabled = (
                            hasattr(manager.brain, 'output_config') and 
                            manager.brain.output_config and
                            manager.brain.output_config.get('enable_multi_token', False)
                        )
                        
                        if multi_token_enabled:
                            print("🚀 Multi-token generation enabled!")
                        
                        result = manager.run_inference(input_data)
                        
                        print(f"\n📊 Inference Results:")
                        
                        # Enhanced display for multi-token results
                        if multi_token_enabled and 'tokens' in result:
                            print(f"  🎯 Generated Sequence: {result.get('tokens', [])}")
                            if 'sequence_text' in result:
                                print(f"  📝 Token Text: {result['sequence_text']}")
                            if 'group_sequences' in result and result['group_sequences']:
                                print(f"  🎭 Node Group Contributions:")
                                for group, sequence in result['group_sequences'].items():
                                    print(f"     • {group}: {sequence}")
                            print(f"  🎯 Consensus: {result.get('consensus', 0):.3f}")
                            print(f"  🔍 Status: {result.get('status', 'Unknown')}")
                        else:
                            print(f"  📊 Output: {result.get('result', result)}")  # Show the actual result data
                        
                        print(f"  ⏱️  Pipeline Time: {result.get('pipeline_time', 0):.4f}s")
                        print(f"  💪 Confidence: {result.get('confidence', 0):.3f}")
                        
                        if 'error' in result:
                            print(f"  ❌ Error: {result['error']}")
                        
                        # Show multi-token metadata if available
                        if multi_token_enabled and 'metadata' in result:
                            metadata = result['metadata']
                            if 'generation_method' in metadata:
                                print(f"  🧠 Generation Method: {metadata['generation_method']}")
                            if 'num_groups' in metadata:
                                print(f"  🏷️  Node Groups Used: {metadata['num_groups']}")
                        
                    else:
                        print("❌ Please enter some input data")
            
            elif choice == '11':
                if manager.brain is None:
                    print("❌ No brain instance available")
                else:
                    print(f"\n📊 Brain Statistics:")
                    print(f"  Mode: {'Demo' if manager.brain.demo else 'Full'}")
                    print(f"  Total Nodes: {len(manager.brain.neural_nodes)}")
                    
                    # Count by type
                    type_counts = {}
                    for node in manager.brain.neural_nodes:
                        node_type = getattr(node, 'node_type', 'Unknown')
                        type_counts[node_type] = type_counts.get(node_type, 0) + 1
                    
                    for node_type, count in type_counts.items():
                        print(f"  {node_type}: {count}")
                    
                    print(f"  Dimensions: {manager.brain.dimensions}")
                    
                    if manager.learner:
                        summary = manager.learner.get_training_summary()
                        print(f"  Training Mode: {summary['training_mode']}")
                        print(f"  RL Mode: {summary['rl_mode']}")
                        print(f"  Current Epoch: {summary['current_epoch']}")
            
            elif choice == '12':
                print_architecture_analysis()
            
            elif choice == '13':
                if manager.learner is None:
                    print("❌ No learner available. Initialize a brain first.")
                else:
                    print("\n�️ Testing spatial optimization...")
                    
                    # Test spatial efficiency calculation
                    initial_efficiency = manager.learner._calculate_spatial_efficiency()
                    print(f"Initial spatial efficiency: {initial_efficiency:.4f}")
                    
                    # Test node movement
                    comp_nodes = manager.learner.get_nodes_by_type('Computational')
                    if comp_nodes:
                        test_node = comp_nodes[0]
                        initial_pos = manager.learner.node_registry[test_node].node_position.copy()
                        print(f"Testing node movement (Node #{test_node})...")
                        print(f"  Initial position: {[f'{p:.1f}' for p in initial_pos[:3]]}")
                        
                        # Move node
                        new_pos = [p + random.uniform(-100, 100) for p in initial_pos]
                        success = manager.learner.move_node(test_node, new_pos)
                        
                        if success:
                            final_pos = manager.learner.node_registry[test_node].node_position
                            print(f"  New position: {[f'{p:.1f}' for p in final_pos[:3]]}")
                            
                            new_efficiency = manager.learner._calculate_spatial_efficiency()
                            print(f"  New spatial efficiency: {new_efficiency:.4f}")
                            print(f"  Efficiency change: {new_efficiency - initial_efficiency:+.4f}")
                        else:
                            print("  ❌ Node movement failed")
                    
                    # Test connection optimization
                    print("\nTesting connection optimization...")
                    try:
                        manager.learner._optimize_connections()
                        print("✓ Connection optimization completed")
                    except Exception as e:
                        print(f"❌ Connection optimization failed: {e}")
            
            elif choice == '14':
                if manager.brain is None:
                    print("❌ No brain instance available")
                else:
                    manager.list_all_nodes_detailed()

            elif choice == '15':
                if manager.brain is None:
                    print("❌ No brain instance available")
                else:
                    manager.visualize_brain_smart()

            elif choice == '16':
                if manager.brain is None:
                    print("❌ No brain instance available")
                else:
                    manager.visualize_brain_both()
                    
            elif choice == '17':
                if manager.brain is None:
                    print("❌ No brain instance available")
                else:
                    input_data = input("Enter input data to trace: ").strip()
                    if input_data:
                        manager.traceroute(input_data, create_video=True)

            elif choice == '18':
                if manager.brain is None:
                    print("❌ No brain instance available")
                else:
                    input_data = input("Enter input data to trace: ").strip()
                    if input_data:
                        manager.traceroute(input_data, create_video=False)
                        
            elif choice == '19':
                # Train on flat text (direct input)
                if manager.learner is None:
                    print("❌ No learning brain instance available. Initialize a brain first.")
                else:
                    print("\n📖 FLAT TEXT TRAINING - DIRECT INPUT")
                    print("You can paste Wikipedia content, book text, articles, etc.")
                    print("Enter 'quit' on a new line to finish input.\n")
                    
                    text_lines = []
                    while True:
                        line = input("> ")
                        if line.strip().lower() == 'quit':
                            break
                        text_lines.append(line)
                    
                    if text_lines:
                        text = '\n'.join(text_lines)
                        text_title = input("Enter title for this text (optional): ").strip()
                        
                        print("\nSelect text type:")
                        print("1. Wikipedia page")
                        print("2. Book/Novel")
                        print("3. Article/Essay")
                        print("4. Webpage")
                        print("5. Other document")
                        
                        type_choice = input("Enter choice (1-5, default 5): ").strip()
                        text_types = {
                            '1': 'wikipedia',
                            '2': 'book', 
                            '3': 'article',
                            '4': 'webpage',
                            '5': 'document'
                        }
                        text_type = text_types.get(type_choice, 'document')
                        
                        print("\nSelect training approach:")
                        print("1. All approaches (token prediction + sentence completion + paragraph continuation)")
                        print("2. Token prediction only (next word prediction)")
                        print("3. Sentence completion only")
                        print("4. Paragraph continuation only")
                        
                        approach_choice = input("Enter choice (1-4, default 1): ").strip()
                        approaches = {
                            '1': 'all',
                            '2': 'token_prediction',
                            '3': 'sentence_completion',
                            '4': 'paragraph_continuation'
                        }
                        training_approach = approaches.get(approach_choice, 'all')
                        
                        context_window = input("Enter context window size (default 10): ").strip()
                        context_window = int(context_window) if context_window.isdigit() else 10
                        
                        manager.train_on_flat_text(
                            text=text,
                            text_title=text_title,
                            text_type=text_type,
                            context_window=context_window,
                            training_approach=training_approach
                        )
                    else:
                        print("❌ No text entered.")
            
            elif choice == '20':
                # Train on text file
                if manager.learner is None:
                    print("❌ No learning brain instance available. Initialize a brain first.")
                else:
                    file_path = input("Enter path to text file: ").strip()
                    if file_path:
                        print("\nSelect text type:")
                        print("1. Wikipedia page")
                        print("2. Book/Novel")
                        print("3. Article/Essay")
                        print("4. Webpage")
                        print("5. Other document")
                        
                        type_choice = input("Enter choice (1-5, default 5): ").strip()
                        text_types = {
                            '1': 'wikipedia',
                            '2': 'book', 
                            '3': 'article',
                            '4': 'webpage',
                            '5': 'document'
                        }
                        text_type = text_types.get(type_choice, 'document')
                        
                        print("\nSelect training approach:")
                        print("1. All approaches")
                        print("2. Token prediction only")
                        print("3. Sentence completion only")
                        print("4. Paragraph continuation only")
                        
                        approach_choice = input("Enter choice (1-4, default 1): ").strip()
                        approaches = {
                            '1': 'all',
                            '2': 'token_prediction',
                            '3': 'sentence_completion',
                            '4': 'paragraph_continuation'
                        }
                        training_approach = approaches.get(approach_choice, 'all')
                        
                        context_window = input("Enter context window size (default 10): ").strip()
                        context_window = int(context_window) if context_window.isdigit() else 10
                        
                        manager.train_on_file(
                            file_path=file_path,
                            text_type=text_type,
                            context_window=context_window,
                            training_approach=training_approach
                        )
                    else:
                        print("❌ No file path entered.")
            
            elif choice == '21':
                # Generate text from prompt
                if manager.learner is None:
                    print("❌ No learning brain instance available. Initialize a brain first.")
                else:
                    prompt = input("Enter text prompt to continue: ").strip()
                    if prompt:
                        max_length = input("Enter max length (default 50): ").strip()
                        max_length = int(max_length) if max_length.isdigit() else 50
                        
                        temperature = input("Enter temperature 0-2 (default 0.8): ").strip()
                        try:
                            temperature = float(temperature) if temperature else 0.8
                            temperature = max(0.1, min(2.0, temperature))  # Clamp to valid range
                        except ValueError:
                            temperature = 0.8
                        
                        print(f"\n🎨 Generating text...")
                        print(f"Prompt: '{prompt}'")
                        print(f"Max length: {max_length} tokens")
                        print(f"Temperature: {temperature}")
                        
                        generated = manager.generate_text(prompt, max_length, temperature)
                        
                        if generated:
                            print(f"\n✨ Generated text:")
                            print(f"'{prompt} {generated}'")
                        else:
                            print("❌ Failed to generate text.")
                    else:
                        print("❌ No prompt entered.")
                        
            elif choice == '22':
                print("\n👋 Goodbye! Thanks for using BrainNexus!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-22.")
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or report this issue.")

if __name__ == "__main__":
    main()