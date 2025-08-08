#!/usr/bin/env python3
"""
Comprehensive example of using reinforcement learning with the BrainNexus system.
This demonstrates how to train brain segments using different RL algorithms.
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BrainNexusLearning import SegmentLearning, LearningTask, RLConfig
from BrainNexus import BrainNexus

class RLTrainingExample:
    """
    Example class showing how to use RL training with BrainNexus.
    """
    
    def __init__(self):
        self.brain = BrainNexus()
        self.learning = SegmentLearning(self.brain)
        
        # Initialize the brain
        self.brain.initialize_brain()
        print(f"Initialized brain with {len(self.brain.segments)} segments")
    
    def example_1_basic_dqn_training(self):
        """
        Example 1: Basic DQN training for decision making.
        """
        print("\n=== Example 1: Basic DQN Training ===")
        
        # Configure DQN
        rl_config = RLConfig(
            algorithm='dqn',
            epsilon=0.1,          # 10% exploration
            gamma=0.95,           # Discount factor
            buffer_size=10000,    # Experience replay buffer size
            batch_size=32,        # Training batch size
            learning_rate=0.001,  # Learning rate
            target_update_frequency=100  # Update target network every 100 steps
        )
        
        # Create a classification task
        learning_task = LearningTask(
            task_type='classification',
            input_data=[
                "classify this positive example",
                "classify this negative example", 
                "ambiguous case to classify",
                "clear positive case",
                "clear negative case"
            ],
            expected_output=[
                "positive",
                "negative",
                "neutral", 
                "positive",
                "negative"
            ],
            learning_rate=0.01,
            max_iterations=100,
            batch_size=16,
            success_threshold=0.85
        )
        
        # Train segment 0 with DQN
        print("Training with DQN algorithm...")
        result = self.learning.train_segment_rl(
            segment_id=0,
            learning_task=learning_task,
            rl_config=rl_config,
            episodes=50
        )
        
        print(f"DQN Training Results:")
        print(f"  Episodes: {result.get('episodes', 'N/A')}")
        print(f"  Final Reward: {result.get('final_reward', 0):.4f}")
        print(f"  Average Loss: {result.get('average_loss', 0):.6f}")
        
        return result
    
    def example_2_ppo_continuous_control(self):
        """
        Example 2: PPO training for continuous control tasks.
        """
        print("\n=== Example 2: PPO Continuous Control ===")
        
        # Configure PPO
        rl_config = RLConfig(
            algorithm='ppo',
            gamma=0.99,           # High discount for long-term planning
            buffer_size=2048,     # PPO typically uses larger buffers
            batch_size=64,        # Larger batches for stable updates
            learning_rate=0.0003, # Conservative learning rate
            ppo_epochs=4,         # Multiple epochs per update
            ppo_clip=0.2,         # Clipping parameter
            value_coef=0.5,       # Value function coefficient
            entropy_coef=0.01     # Entropy bonus for exploration
        )
        
        # Create an optimization task
        learning_task = LearningTask(
            task_type='optimization',
            input_data=[
                "optimize neural pathway efficiency",
                "balance exploration vs exploitation", 
                "minimize computational overhead",
                "maximize information retention",
                "adapt to changing patterns"
            ],
            expected_output=[
                "efficient_pathway",
                "balanced_strategy",
                "optimized_computation",
                "retained_information", 
                "adaptive_response"
            ],
            learning_rate=0.005,
            max_iterations=200,
            batch_size=8,
            success_threshold=0.9
        )
        
        # Train segment 1 with PPO
        print("Training with PPO algorithm...")
        result = self.learning.train_segment_rl(
            segment_id=1 if len(self.brain.segments) > 1 else 0,
            learning_task=learning_task,
            rl_config=rl_config,
            episodes=30
        )
        
        print(f"PPO Training Results:")
        print(f"  Episodes: {result.get('episodes', 'N/A')}")
        print(f"  Final Reward: {result.get('final_reward', 0):.4f}")
        print(f"  Average Loss: {result.get('average_loss', 0):.6f}")
        
        return result
    
    def example_3_a2c_fast_learning(self):
        """
        Example 3: A2C training for fast learning scenarios.
        """
        print("\n=== Example 3: A2C Fast Learning ===")
        
        # Configure A2C
        rl_config = RLConfig(
            algorithm='a2c',
            gamma=0.9,            # Lower discount for immediate rewards
            batch_size=16,        # Smaller batches for faster updates
            learning_rate=0.01,   # Higher learning rate for fast adaptation
            value_coef=0.25,      # Lower value coefficient
            entropy_coef=0.05     # Higher entropy for more exploration
        )
        
        # Create a pattern recognition task
        learning_task = LearningTask(
            task_type='pattern_recognition',
            input_data=[
                "pattern ABC ABC ABC",
                "sequence 123 123 123",
                "rhythm dum dum da dum dum da", 
                "structure X-Y-Z X-Y-Z X-Y-Z",
                "cycle start-middle-end start-middle-end"
            ],
            expected_output=[
                "repeating_triplet",
                "numeric_sequence",
                "rhythmic_pattern",
                "structural_cycle",
                "temporal_loop"
            ],
            learning_rate=0.02,
            max_iterations=75,
            batch_size=4,
            success_threshold=0.8
        )
        
        # Train with A2C
        print("Training with A2C algorithm...")
        result = self.learning.train_segment_rl(
            segment_id=0,
            learning_task=learning_task,
            rl_config=rl_config,
            episodes=25
        )
        
        print(f"A2C Training Results:")
        print(f"  Episodes: {result.get('episodes', 'N/A')}")
        print(f"  Final Reward: {result.get('final_reward', 0):.4f}")
        print(f"  Average Loss: {result.get('average_loss', 0):.6f}")
        
        return result
    
    def example_4_ddpg_precision_control(self):
        """
        Example 4: DDPG training for precision control tasks.
        """
        print("\n=== Example 4: DDPG Precision Control ===")
        
        # Configure DDPG
        rl_config = RLConfig(
            algorithm='ddpg',
            gamma=0.98,           # High discount for precise control
            buffer_size=100000,   # Large buffer for stability
            batch_size=128,       # Large batches for stable gradients
            learning_rate=0.0001, # Very conservative learning rate
            tau=0.001,            # Soft target updates
            update_frequency=1    # Update every step
        )
        
        # Create a precision task
        learning_task = LearningTask(
            task_type='precision_control',
            input_data=[
                "fine-tune parameter alpha=0.847",
                "adjust weight beta=1.234 precisely", 
                "calibrate threshold gamma=0.156",
                "set boundary delta=2.789 exactly",
                "configure limit epsilon=0.001"
            ],
            expected_output=[
                "alpha_tuned_0.847",
                "beta_adjusted_1.234",
                "gamma_calibrated_0.156",
                "delta_set_2.789",
                "epsilon_configured_0.001"
            ],
            learning_rate=0.001,
            max_iterations=150,
            batch_size=8,
            success_threshold=0.95
        )
        
        # Train with DDPG
        print("Training with DDPG algorithm...")
        result = self.learning.train_segment_rl(
            segment_id=0,
            learning_task=learning_task,
            rl_config=rl_config,
            episodes=20
        )
        
        print(f"DDPG Training Results:")
        print(f"  Episodes: {result.get('episodes', 'N/A')}")
        print(f"  Final Reward: {result.get('final_reward', 0):.4f}")
        print(f"  Average Loss: {result.get('average_loss', 0):.6f}")
        
        return result
    
    def example_5_multi_algorithm_comparison(self):
        """
        Example 5: Compare different algorithms on the same task.
        """
        print("\n=== Example 5: Multi-Algorithm Comparison ===")
        
        # Common task for all algorithms
        common_task = LearningTask(
            task_type='decision_making',
            input_data=[
                "choose optimal path A or B",
                "select best strategy X or Y",
                "pick efficient method 1 or 2"
            ],
            expected_output=[
                "path_A_optimal",
                "strategy_X_best", 
                "method_1_efficient"
            ],
            learning_rate=0.01,
            max_iterations=50,
            batch_size=4,
            success_threshold=0.8
        )
        
        algorithms = ['dqn', 'ppo', 'a2c', 'ddpg']
        results = {}
        
        for algorithm in algorithms:
            print(f"\nTesting {algorithm.upper()}...")
            
            # Configure each algorithm appropriately
            if algorithm == 'dqn':
                rl_config = RLConfig(algorithm='dqn', epsilon=0.1, gamma=0.95)
            elif algorithm == 'ppo':
                rl_config = RLConfig(algorithm='ppo', gamma=0.99, ppo_epochs=2)
            elif algorithm == 'a2c':
                rl_config = RLConfig(algorithm='a2c', gamma=0.9, entropy_coef=0.01)
            else:  # ddpg
                rl_config = RLConfig(algorithm='ddpg', gamma=0.98, tau=0.01)
            
            try:
                result = self.learning.train_segment_rl(
                    segment_id=0,
                    learning_task=common_task,
                    rl_config=rl_config,
                    episodes=10  # Short comparison
                )
                
                results[algorithm] = {
                    'success': True,
                    'final_reward': result.get('final_reward', 0),
                    'episodes': result.get('episodes', 0),
                    'loss': result.get('average_loss', 0)
                }
                
                print(f"  {algorithm.upper()}: Reward={result.get('final_reward', 0):.4f}")
                
            except Exception as e:
                results[algorithm] = {'success': False, 'error': str(e)}
                print(f"  {algorithm.upper()}: FAILED - {str(e)}")
        
        # Summary
        successful = [alg for alg, res in results.items() if res['success']]
        print(f"\nSuccessful algorithms: {', '.join(successful)}")
        
        if successful:
            best_algorithm = max(successful, 
                               key=lambda alg: results[alg]['final_reward'])
            print(f"Best performing: {best_algorithm.upper()} "
                  f"(reward: {results[best_algorithm]['final_reward']:.4f})")
        
        return results
    
    def run_all_examples(self):
        """
        Run all RL training examples.
        """
        print("🧠 BrainNexus Reinforcement Learning Training Examples")
        print("=" * 60)
        
        results = {}
        
        # Run examples
        try:
            results['dqn'] = self.example_1_basic_dqn_training()
        except Exception as e:
            print(f"Example 1 failed: {e}")
            results['dqn'] = None
        
        try:
            results['ppo'] = self.example_2_ppo_continuous_control()
        except Exception as e:
            print(f"Example 2 failed: {e}")
            results['ppo'] = None
        
        try:
            results['a2c'] = self.example_3_a2c_fast_learning()
        except Exception as e:
            print(f"Example 3 failed: {e}")
            results['a2c'] = None
        
        try:
            results['ddpg'] = self.example_4_ddpg_precision_control()
        except Exception as e:
            print(f"Example 4 failed: {e}")
            results['ddpg'] = None
        
        try:
            results['comparison'] = self.example_5_multi_algorithm_comparison()
        except Exception as e:
            print(f"Example 5 failed: {e}")
            results['comparison'] = None
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎯 TRAINING SUMMARY")
        print("=" * 60)
        
        successful_examples = sum(1 for result in results.values() if result is not None)
        print(f"Successful examples: {successful_examples}/5")
        
        if successful_examples > 0:
            print("\n✅ Reinforcement learning system is operational!")
            print("   Ready for production neural training.")
        else:
            print("\n⚠️ System needs debugging before use.")
        
        print("\nNext steps:")
        print("- Use these examples as templates for your specific tasks")
        print("- Adjust hyperparameters based on your data")
        print("- Monitor training progress and rewards")
        print("- Scale up episode counts for real applications")
        
        return results

def main():
    """
    Main function to run the RL training examples.
    """
    try:
        example = RLTrainingExample()
        results = example.run_all_examples()
        
        print(f"\n🚀 RL training demonstration complete!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
