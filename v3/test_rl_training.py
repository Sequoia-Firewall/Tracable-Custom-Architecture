#!/usr/bin/env python3
"""
Test script for reinforcement learning training capabilities.
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BrainNexusLearning import (
    SegmentLearning, LearningTask, RLConfig,
    DiscretePolicyNetwork, ContinuousPolicyNetwork, ValueNetwork,
    ReplayBuffer, RLEnvironment
)
from BrainNexus import BrainNexus
from BrainSegment import NexusSegment

class RLTrainingDemo:
    """Demonstration of RL training capabilities."""
    
    def __init__(self):
        # Create a BrainNexus instance
        self.brain = BrainNexus(demo=True)
        
        # Create a test segment manually
        dimensional_assignment = {0: 1, 1: 1}  # +x, +y quadrant
        self.brain_segment = NexusSegment(
            segment_id=0,
            dimensional_assignment=dimensional_assignment,
            brain_nexus_ref=self.brain,
            demo=True
        )
        
        # Initialize the segment
        print("Initializing test brain segment...")
        self.brain_segment._create_segment_nodes()
        
        # Create the learning system
        self.learning = SegmentLearning(self.brain_segment)
        
        # Create RL environment with proper state dimension
        self.rl_environment = RLEnvironment(self.brain_segment, state_dim=256)
        
    def test_rl_config_creation(self):
        """Test RL configuration creation."""
        print("Testing RL Configuration Creation...")
        
        rl_config = RLConfig(
            algorithm='dqn',
            epsilon=0.1,
            gamma=0.95,
            buffer_size=10000,
            target_update_frequency=100,
            ppo_epochs=4,
            ppo_clip=0.2,
            value_coef=0.5,
            entropy_coef=0.01
        )
        
        print(f"✓ Created RL config with algorithm: {rl_config.algorithm}")
        print(f"  Epsilon: {rl_config.epsilon}, Gamma: {rl_config.gamma}")
        print(f"  Buffer size: {rl_config.buffer_size}")
        
        return rl_config
    
    def test_network_creation(self):
        """Test neural network creation for different node types."""
        print("\nTesting RL Network Creation...")
        
        # Test discrete policy network (for categorical actions)
        discrete_net = DiscretePolicyNetwork(
            state_dim=256,
            action_dim=4,
            device=self.learning.device
        )
        
        # Test continuous policy network (for continuous actions)
        continuous_net = ContinuousPolicyNetwork(
            state_dim=256,
            action_dim=2,
            device=self.learning.device
        )
        
        # Test value network
        value_net = ValueNetwork(
            state_dim=256,
            device=self.learning.device
        )
        
        print("✓ Created discrete policy network")
        print("✓ Created continuous policy network") 
        print("✓ Created value network")
        
        # Test action selection
        test_state = torch.randn(1, 256)
        discrete_action, discrete_log_prob = discrete_net.get_action(test_state, epsilon=0.1)
        continuous_action, continuous_log_prob = continuous_net.get_action(test_state)
        value = value_net(test_state)
        
        print(f"  Discrete action: {discrete_action.item()}")
        print(f"  Continuous action: {continuous_action.numpy()}")
        print(f"  State value: {value.item():.4f}")
        
        return discrete_net, continuous_net, value_net
    
    def test_rl_environment(self):
        """Test RL environment integration."""
        print("\nTesting RL Environment Integration...")
        
        # Use the RL environment we created
        env = self.rl_environment
        
        # Test state creation
        state = env.get_state(self.brain_segment)
        print(f"✓ Created state tensor with shape: {state.shape}")
        
        # Test action application
        test_action = np.array([0.1, -0.2, 0.3])  # Example action
        env.apply_action(self.brain_segment, test_action)
        print("✓ Applied action to segment")
        
        # Test reward calculation
        reward = env.calculate_reward(self.brain_segment, test_action, {
            'accuracy': 0.85,
            'efficiency': 0.7,
            'complexity': 0.6
        })
        print(f"✓ Calculated reward: {reward:.4f}")
        
        return env
    
    def test_segment_rl_training(self, rl_config):
        """Test reinforcement learning training on a segment."""
        print("\nTesting Segment RL Training...")
        
        # Create training task
        learning_task = LearningTask(
            task_id='test_rl_task',
            task_type='reinforcement',
            modality='general',
            objective='classification',
            data_shape=(256,),
            num_classes=4,
            learning_rate=0.01,
            batch_size=8,
            max_epochs=50
        )
        
        # Get initial segment state
        initial_weights = {}
        for i, node in enumerate(self.brain_segment.segment_nodes.values()):
            if hasattr(node, 'weights'):
                initial_weights[f'node_{i}'] = dict(node.weights)
        
        if initial_weights:
            print(f"Initial node weights sample: {list(list(initial_weights.values())[0].keys())[:3]}")
        
        # Train with RL
        print(f"Training segment with RL algorithm: {rl_config.algorithm}")
        
        try:
            training_result = self.learning.train_segment_rl(
                learning_task=learning_task,
                environment=self.rl_environment,
                rl_config=rl_config,
                episodes=10  # Small number for testing
            )
            
            print("✓ RL training completed successfully")
            print(f"  Training episodes: {training_result.get('episodes_trained', 'N/A')}")
            print(f"  Final reward: {training_result.get('final_avg_reward', 0):.4f}")
            
            # Check if weights changed
            final_weights = {}
            for i, node in enumerate(self.brain_segment.segment_nodes.values()):
                if hasattr(node, 'weights'):
                    final_weights[f'node_{i}'] = dict(node.weights)
            
            # Compare weights
            weight_changes = 0
            if initial_weights and final_weights:
                for node_key in initial_weights:
                    if node_key in final_weights:
                        for weight_key in initial_weights[node_key]:
                            if weight_key in final_weights[node_key]:
                                if abs(initial_weights[node_key][weight_key] - final_weights[node_key][weight_key]) > 1e-6:
                                    weight_changes += 1
            
            print(f"  Weight parameters changed: {weight_changes}")
            
            if weight_changes > 0:
                print("✓ RL training successfully modified neural components")
            else:
                print("⚠ RL training did not modify weights (may need more episodes)")
            
            return training_result
            
        except Exception as e:
            print(f"✗ RL training failed: {str(e)}")
            return None
    
    def test_multiple_rl_algorithms(self):
        """Test different RL algorithms."""
        print("\nTesting Multiple RL Algorithms...")
        
        algorithms = ['dqn', 'ppo', 'a2c', 'ddpg']
        results = {}
        
        for algorithm in algorithms:
            print(f"\nTesting {algorithm.upper()} algorithm...")
            
            rl_config = RLConfig(
                algorithm=algorithm,
                epsilon=0.1,
                gamma=0.95,
                buffer_size=1000,
                ppo_epochs=2 if algorithm == 'ppo' else 1
            )
            
            # Quick training test
            try:
                learning_task = LearningTask(
                    task_id=f'test_{algorithm}',
                    task_type='reinforcement',
                    modality='general',
                    objective='optimization',
                    data_shape=(256,),
                    learning_rate=0.01,
                    batch_size=4,
                    max_epochs=10
                )
                
                result = self.learning.train_segment_rl(
                    learning_task=learning_task,
                    environment=self.rl_environment,
                    rl_config=rl_config,
                    episodes=5  # Very small for testing
                )
                
                results[algorithm] = {
                    'success': True,
                    'final_reward': result.get('final_avg_reward', 0),
                    'episodes': result.get('episodes_trained', 0)
                }
                
                print(f"✓ {algorithm.upper()} completed successfully")
                
            except Exception as e:
                results[algorithm] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"✗ {algorithm.upper()} failed: {str(e)}")
        
        return results
    
    def run_comprehensive_test(self):
        """Run all RL tests."""
        print("=== Reinforcement Learning Training Test ===\n")
        
        # Test 1: RL Configuration
        rl_config = self.test_rl_config_creation()
        
        # Test 2: Network Creation
        networks = self.test_network_creation()
        
        # Test 3: Environment Integration
        environment = self.test_rl_environment()
        
        # Test 4: Single Algorithm Training
        training_result = self.test_segment_rl_training(rl_config)
        
        # Test 5: Multiple Algorithms
        algorithm_results = self.test_multiple_rl_algorithms()
        
        # Summary
        print("\n=== Test Summary ===")
        print("✓ RL configuration creation: PASSED")
        print("✓ Neural network creation: PASSED")
        print("✓ Environment integration: PASSED")
        
        if training_result:
            print("✓ RL training execution: PASSED")
        else:
            print("✗ RL training execution: FAILED")
        
        successful_algorithms = sum(1 for result in algorithm_results.values() if result['success'])
        print(f"✓ Algorithm compatibility: {successful_algorithms}/{len(algorithm_results)} PASSED")
        
        print(f"\n🧠 Reinforcement learning system is {'OPERATIONAL' if training_result and successful_algorithms > 0 else 'NEEDS DEBUGGING'}")
        
        return {
            'rl_config': rl_config,
            'networks': networks,
            'environment': environment,
            'training_result': training_result,
            'algorithm_results': algorithm_results
        }

def main():
    """Run the RL training demonstration."""
    demo = RLTrainingDemo()
    results = demo.run_comprehensive_test()
    
    if results['training_result']:
        print("\n🎯 Ready for advanced RL-based neural training!")
    else:
        print("\n⚠ System needs debugging before production use.")

if __name__ == "__main__":
    main()
