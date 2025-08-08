#!/usr/bin/env python3
"""
Test script for BrainNexusLearn - demonstrates supervised and reinforcement learning.
"""

from BrainNexusLearn import BrainNexusLearn, TrainingConfig
import numpy as np
import random
import time

def generate_sample_data(num_samples: int = 100):
    """Generate sample training data for supervised learning."""
    training_data = []
    
    for _ in range(num_samples):
        # Create random input (simulating some feature vector)
        input_data = {
            'features': np.random.randn(10).tolist(),
            'sequence_length': random.randint(5, 20)
        }
        
        # Create target (random classification for demo)
        target = random.randint(0, 9)  # 10-class classification
        
        training_data.append((input_data, target))
    
    return training_data

def simple_environment(action=None, reset=False):
    """
    Simple environment for reinforcement learning demonstration.
    
    Args:
        action: Action dictionary from the RL agent
        reset: Whether to reset the environment
        
    Returns:
        (state, reward, done, info) tuple
    """
    if reset:
        # Return initial state
        return {
            'network_efficiency': 0.5,
            'recent_accuracy': 0.6,
            'spatial_coherence': 0.4
        }
    
    if action is None:
        return None, 0.0, False, {}
    
    # Simulate environment response to action
    action_type = action.get('type', 'no_action')
    
    # Calculate reward based on action type and randomness
    if action_type == 'move_node':
        reward = random.uniform(-0.2, 0.5)  # Moving nodes can be good or bad
    elif action_type == 'adjust_weight':
        reward = random.uniform(-0.1, 0.3)  # Weight adjustments usually small impact
    elif action_type == 'add_connection':
        reward = random.uniform(0.0, 0.4)   # Adding connections usually beneficial
    elif action_type == 'remove_connection':
        reward = random.uniform(-0.3, 0.2)  # Removing can be good for efficiency
    else:
        reward = 0.0  # No action = no reward
    
    # Simulate new state
    new_state = {
        'network_efficiency': max(0.0, min(1.0, 0.5 + random.uniform(-0.1, 0.1))),
        'recent_accuracy': max(0.0, min(1.0, 0.6 + random.uniform(-0.05, 0.05))),
        'spatial_coherence': max(0.0, min(1.0, 0.4 + random.uniform(-0.1, 0.1)))
    }
    
    # Episode ends randomly (for demo purposes)
    done = random.random() < 0.1
    
    return new_state, reward, done, {'action_executed': action_type}

def test_supervised_learning():
    """Test supervised learning capabilities."""
    print("=" * 60)
    print("🎓 TESTING SUPERVISED LEARNING")
    print("=" * 60)
    
    # Create learning configuration
    config = TrainingConfig(
        learning_rate=0.01,
        spatial_learning_rate=0.005,
        batch_size=10,
        max_epochs=20,
        convergence_threshold=0.01
    )
    
    # Initialize learner
    learner = BrainNexusLearn(demo=True, config=config)
    
    # Initialize the brain
    print("\n🧠 Initializing brain structure...")
    node_map = learner.initialize_brain()
    print(f"✅ Brain initialized with {len(learner.neural_nodes)} nodes")
    
    # Generate training data
    print("\n📚 Generating training data...")
    training_data = generate_sample_data(100)
    validation_data = generate_sample_data(20)
    print(f"✅ Generated {len(training_data)} training samples, {len(validation_data)} validation samples")
    
    # Train the model
    print("\n🎯 Starting supervised training...")
    training_results = learner.supervised_train(training_data, validation_data)
    
    print(f"\n✅ Training completed!")
    print(f"   Final loss: {training_results['final_loss']:.4f}")
    print(f"   Final accuracy: {training_results['final_accuracy']:.3f}")
    print(f"   Spatial efficiency: {training_results['spatial_efficiency']:.3f}")
    print(f"   Epochs completed: {training_results['epochs_completed']}")
    
    return learner

def test_reinforcement_learning(learner):
    """Test reinforcement learning capabilities."""
    print("\n" + "=" * 60)
    print("🎮 TESTING REINFORCEMENT LEARNING")
    print("=" * 60)
    
    # Configure for RL
    learner.config.exploration_rate = 0.3
    learner.config.rl_discount_factor = 0.95
    
    print("\n🎮 Starting reinforcement learning...")
    rl_results = learner.reinforcement_train(
        environment_fn=simple_environment,
        episodes=200,
        max_steps_per_episode=50
    )
    
    print(f"\n✅ RL training completed!")
    print(f"   Final average reward: {rl_results['avg_final_reward']:.4f}")
    print(f"   Q-table size: {rl_results['q_table_size']}")
    print(f"   Final exploration rate: {rl_results['final_exploration_rate']:.3f}")
    
    return rl_results

def test_spatial_optimization(learner):
    """Test spatial optimization capabilities."""
    print("\n" + "=" * 60)
    print("🗺️  TESTING SPATIAL OPTIMIZATION")
    print("=" * 60)
    
    # Get initial spatial metrics
    initial_efficiency = learner._calculate_spatial_efficiency()
    print(f"📊 Initial spatial efficiency: {initial_efficiency:.4f}")
    
    # Test moving nodes
    comp_nodes = learner.get_nodes_by_type('Computational')
    if comp_nodes:
        test_node = comp_nodes[0]
        initial_pos = learner.node_registry[test_node].node_position.copy()
        
        print(f"🔄 Testing node movement (Node #{test_node})...")
        print(f"   Initial position: {[f'{p:.1f}' for p in initial_pos[:3]]}")
        
        # Move node to a new position
        new_pos = [p + random.uniform(-100, 100) for p in initial_pos]
        success = learner.move_node(test_node, new_pos)
        
        if success:
            final_pos = learner.node_registry[test_node].node_position
            print(f"   New position: {[f'{p:.1f}' for p in final_pos[:3]]}")
            
            # Recalculate efficiency
            new_efficiency = learner._calculate_spatial_efficiency()
            print(f"   New spatial efficiency: {new_efficiency:.4f}")
            efficiency_change = new_efficiency - initial_efficiency
            print(f"   Efficiency change: {efficiency_change:+.4f}")
        else:
            print("   ❌ Node movement failed")
    
    # Test connection optimization
    print(f"\n🔗 Testing connection optimization...")
    total_connections_before = sum(len(learner.get_node_connections(node.node_id)['outgoing']) 
                                  for node in learner.neural_nodes)
    print(f"   Connections before optimization: {total_connections_before}")
    
    # Run optimization
    learner._optimize_connections()
    
    total_connections_after = sum(len(learner.get_node_connections(node.node_id)['outgoing']) 
                                 for node in learner.neural_nodes)
    print(f"   Connections after optimization: {total_connections_after}")
    connection_change = total_connections_after - total_connections_before
    print(f"   Connection change: {connection_change:+d}")

def main():
    """Main test function."""
    print("🚀 BrainNexusLearn Comprehensive Testing")
    print("========================================")
    
    start_time = time.time()
    
    try:
        # Test supervised learning
        learner = test_supervised_learning()
        
        # Test spatial optimization
        test_spatial_optimization(learner)
        
        # Test reinforcement learning
        test_reinforcement_learning(learner)
        
        # Get final summary
        print("\n" + "=" * 60)
        print("📊 FINAL SUMMARY")
        print("=" * 60)
        
        summary = learner.get_training_summary()
        print(f"🧠 Network Statistics:")
        print(f"   Total nodes: {summary['network_stats']['total_nodes']}")
        print(f"   Total connections: {summary['network_stats']['total_connections']}")
        print(f"   Spatial efficiency: {summary['network_stats']['spatial_efficiency']:.4f}")
        print(f"   Reuse candidates: {summary['network_stats']['reuse_candidates']}")
        
        if summary['rl_stats']:
            print(f"\n🎮 RL Statistics:")
            print(f"   Q-table size: {summary['rl_stats']['q_table_size']}")
            print(f"   Total experiences: {summary['rl_stats']['total_experiences']}")
            print(f"   Exploration rate: {summary['rl_stats']['exploration_rate']:.3f}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total test time: {total_time:.2f}s")
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
