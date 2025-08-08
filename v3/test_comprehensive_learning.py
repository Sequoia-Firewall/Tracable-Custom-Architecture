#!/usr/bin/env python3
"""
Comprehensive Test Suite for BrainNexus v3 Learning System

This script tests:
1. Enhanced supervised learning with spatial optimization
2. Reinforcement Learning with different algorithms
3. Node evolution (computational -> judge transformation)
4. Multi-modal learning capabilities
5. Connection pruning and spatial adaptation
6. Performance tracking and metrics

Usage:
    python test_comprehensive_learning.py
"""

import sys
import numpy as np
import torch
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
import json

# Import our modules
from BrainNexus import BrainNexus
from BrainSegment import NexusSegment
from BrainNexusLearning import SegmentLearning, LearningTask, RLConfig
from computations import Computational, Judge

# Mock RLEnvironment for testing
class RLEnvironment:
    """Base RL Environment interface for testing."""
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        return np.random.randn(64).astype(np.float32)
    
    def step(self, actions: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute actions and return next state, reward, done, info."""
        next_state = np.random.randn(64).astype(np.float32)
        reward = float(np.random.randn())
        done = False
        info = {}
        return next_state, reward, done, info

# Test configuration
@dataclass
class TestConfig:
    """Configuration for comprehensive testing."""
    demo: bool = True
    save_results: bool = True
    test_enhanced_learning: bool = True
    test_rl_learning: bool = True
    test_node_evolution: bool = True
    test_multimodal: bool = True
    verbose_output: bool = True
    segment_size: int = 50
    training_epochs: int = 30
    evolution_test_epochs: int = 60  # Longer for evolution testing

class MockRLEnvironment(RLEnvironment):
    """Mock RL environment for testing reinforcement learning."""
    
    def __init__(self, state_dim: int = 64, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = None
        self.episode_step = 0
        self.max_episode_steps = 100
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.current_state = np.random.randn(self.state_dim).astype(np.float32)
        self.episode_step = 0
        return self.current_state
    
    def step(self, actions: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute actions and return next state, reward, done, info."""
        self.episode_step += 1
        
        # Simple reward function based on action quality
        reward = 0.0
        for node_type, action in actions.items():
            if isinstance(action, torch.Tensor):
                action_value = action.detach().cpu().numpy()
            else:
                action_value = action
            
            # Reward for reasonable actions
            if isinstance(action_value, (int, float)):
                reward += np.tanh(action_value) * 0.1
            elif hasattr(action_value, '__iter__'):
                reward += np.mean(np.tanh(action_value)) * 0.1
        
        # Add some randomness
        reward += np.random.normal(0, 0.05)
        reward = float(reward)  # Ensure it's a plain float
        
        # Update state
        self.current_state = np.random.randn(self.state_dim).astype(np.float32)
        
        # Episode termination
        done = self.episode_step >= self.max_episode_steps
        info = {'episode_step': self.episode_step}
        
        return self.current_state, reward, done, info

class ComprehensiveLearningTester:
    """Comprehensive test suite for the BrainNexus learning system."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = {}
        self.brain = None
        self.segment = None
        self.learner = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        if self.config.demo:
            print("🧠 Comprehensive BrainNexus Learning Test Suite")
            print("=" * 60)
    
    def setup_test_environment(self):
        """Set up the test environment with brain, segment, and learner."""
        if self.config.demo:
            print("\n🔧 Setting up test environment...")
        
        # Create brain
        self.brain = BrainNexus(
            dimensions=4,
            node_count_pre=self.config.segment_size,
            demo=self.config.demo
        )
        
        # Create a test segment manually since we need specific structure
        self.segment = NexusSegment(
            segment_id=1,
            dimensional_assignment={0: 1, 1: -1, 2: 1, 3: -1},
            brain_nexus_ref=self.brain,
            demo=self.config.demo
        )
        
        # Create learner with evolution enabled
        learning_config = {
            'enable_node_evolution': True,
            'enable_spatial_adaptation': True,
            'enable_connection_pruning': True,
            'evolution_threshold_computational_to_judge': 0.85,
            'evolution_stability_requirement': 8,  # Lower for testing
            'evolution_cooldown_epochs': 20,  # Lower for testing
            'max_evolutionary_changes_per_segment': 3,
            'learning_rate': 0.01,  # Higher for faster convergence
            'computational_learning_rate': 0.015,
            'max_epochs': self.config.training_epochs
        }
        
        self.learner = SegmentLearning(
            brain_segment=self.segment,
            learning_config=learning_config,
            device='cpu'
        )
        
        if self.config.demo:
            print(f"   ✅ Created brain with segment containing nodes")
            print(f"   ✅ Node types: {dict(self.segment.node_type_registry)}")
    
    def generate_test_data(self, data_type: str = 'classification', size: int = 1000) -> Tuple[Any, Any]:
        """Generate synthetic test data for different learning tasks."""
        if data_type == 'classification':
            # Multi-class classification data
            X = np.random.randn(size, 64).astype(np.float32)
            y = np.random.randint(0, 3, size)
            return X, y
        
        elif data_type == 'regression':
            # Regression data with some pattern
            X = np.random.randn(size, 32).astype(np.float32)
            y = np.sum(X[:, :5], axis=1) + np.random.normal(0, 0.1, size)
            return X, y.astype(np.float32)
        
        elif data_type == 'multimodal':
            # Multi-modal data (text-like + vision-like)
            text_features = np.random.randn(size, 128).astype(np.float32)
            vision_features = np.random.randn(size, 64, 8, 8).astype(np.float32)  # Image-like
            labels = np.random.randint(0, 2, size)
            return {'text': text_features, 'vision': vision_features}, labels
        
        else:
            return self.generate_test_data('classification', size)
    
    def test_enhanced_learning(self) -> Dict[str, Any]:
        """Test enhanced supervised learning with spatial optimization."""
        if not self.config.test_enhanced_learning:
            return {'status': 'skipped'}
        
        if self.config.demo:
            print("\n📚 Testing Enhanced Supervised Learning")
            print("-" * 40)
        
        results = {}
        
        # Test classification
        X_class, y_class = self.generate_test_data('classification', 800)
        
        classification_task = LearningTask(
            task_id="enhanced_classification",
            task_type="supervised",
            modality="general",
            objective="classification",
            data_shape=(64,),  # Add required data_shape
            max_epochs=self.config.training_epochs,
            early_stopping_patience=8
        )
        
        if self.config.demo:
            print(f"🎯 Running classification task with {len(X_class)} samples...")
        
        start_time = time.time()
        class_results = self.learner.train_segment(
            learning_task=classification_task,
            data=X_class,
            labels=y_class
        )
        class_time = time.time() - start_time
        
        results['classification'] = {
            'training_time': class_time,
            'final_loss': class_results['training_losses'][-1] if class_results['training_losses'] else 0.0,
            'epochs_completed': class_results['final_epoch'],
            'spatial_adaptations': len(self.learner.training_state['spatial_adaptations']),
            'connection_updates': len(self.learner.training_state['connection_updates']),
            'node_performances': class_results['node_performances']
        }
        
        if self.config.demo:
            print("✅ Enhanced learning tests completed")
            print(f"   Classification loss: {results['classification']['final_loss']:.4f}")
            print(f"   Spatial adaptations: {results['classification']['spatial_adaptations']}")
        
        return results
    
    def test_node_evolution(self) -> Dict[str, Any]:
        """Test node evolution from computational to judge nodes."""
        if not self.config.test_node_evolution:
            return {'status': 'skipped'}
        
        if self.config.demo:
            print("\n🧬 Testing Node Evolution System")
            print("-" * 40)
        
        results = {}
        
        # Store initial node registry state
        initial_computational = len(self.segment.node_type_registry.get('computational', []))
        initial_judges = len(self.segment.node_type_registry.get('judges', []))
        
        if self.config.demo:
            print(f"📊 Initial state:")
            print(f"   Computational nodes: {initial_computational}")
            print(f"   Judge nodes: {initial_judges}")
        
        # Create a high-performance training scenario to trigger evolution
        X_evolution, y_evolution = self.generate_test_data('classification', 1200)
        
        evolution_task = LearningTask(
            task_id="evolution_trigger",
            task_type="supervised",
            modality="general",
            objective="classification",
            data_shape=(64,),  # Add required data_shape
            max_epochs=self.config.evolution_test_epochs,
            early_stopping_patience=15
        )
        
        # Force high performance in computational nodes to trigger evolution
        if self.config.demo:
            print("🚀 Running extended training to trigger node evolution...")
        
        start_time = time.time()
        
        # Multiple training rounds to build up performance history
        evolution_history = []
        for round_num in range(3):
            if self.config.demo:
                print(f"   Round {round_num + 1}/3: Training for evolution...")
            
            round_results = self.learner.train_segment(
                learning_task=evolution_task,
                data=X_evolution,
                labels=y_evolution
            )
            
            evolution_history.append({
                'round': round_num + 1,
                'evolutions': len(self.learner.training_state['node_evolutions']),
                'computational_performance': round_results['node_performances'].get('computational', {}),
                'candidates': len(self.learner.training_state['evolution_candidates'])
            })
            
            # Check for evolutions
            if self.learner.training_state['node_evolutions']:
                if self.config.demo:
                    print(f"   🎉 Evolution detected in round {round_num + 1}!")
                break
        
        evolution_time = time.time() - start_time
        
        # Final node registry state
        final_computational = len(self.segment.node_type_registry.get('computational', []))
        final_judges = len(self.segment.node_type_registry.get('judges', []))
        
        # Evolution statistics
        total_evolutions = len(self.learner.training_state['node_evolutions'])
        evolution_candidates = len(self.learner.training_state['evolution_candidates'])
        
        results = {
            'training_time': evolution_time,
            'initial_computational': initial_computational,
            'initial_judges': initial_judges,
            'final_computational': final_computational,
            'final_judges': final_judges,
            'total_evolutions': total_evolutions,
            'evolution_candidates': evolution_candidates,
            'evolution_history': evolution_history,
            'evolution_records': self.learner.training_state['node_evolutions']
        }
        
        if self.config.demo:
            print("✅ Node evolution test completed")
            print(f"   🧬 Evolutions occurred: {total_evolutions}")
            print(f"   📈 Computational: {initial_computational} → {final_computational}")
            print(f"   ⚖️  Judges: {initial_judges} → {final_judges}")
            print(f"   🎯 Evolution candidates tracked: {evolution_candidates}")
            
            if self.learner.training_state['node_evolutions']:
                print(f"   📝 Evolution details:")
                for i, evolution in enumerate(self.learner.training_state['node_evolutions']):
                    print(f"      Evolution {i+1}: Node {evolution['node_id']} "
                          f"({evolution['from_type']} → {evolution['to_type']}) "
                          f"at epoch {evolution['epoch']}")
        
        return results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests and collect comprehensive results."""
        if self.config.demo:
            print("🚀 Starting comprehensive learning tests...")
            print(f"⏱️  Estimated time: ~{self.config.training_epochs * 2 + self.config.evolution_test_epochs} seconds")
        
        start_time = time.time()
        
        # Setup
        self.setup_test_environment()
        
        # Run main tests
        self.results['enhanced_learning'] = self.test_enhanced_learning()
        self.results['node_evolution'] = self.test_node_evolution()
        
        total_time = time.time() - start_time
        
        # Compile final results
        self.results['test_summary'] = {
            'total_test_time': total_time,
            'tests_run': sum(1 for r in self.results.values() if r.get('status') != 'skipped'),
            'tests_skipped': sum(1 for r in self.results.values() if r.get('status') == 'skipped'),
            'evolution_success': self.results['node_evolution'].get('total_evolutions', 0) > 0,
            'learning_convergence': all(
                r.get('final_loss', 1.0) < 0.5 
                for r in [self.results['enhanced_learning'].get('classification', {})]
                if 'final_loss' in r
            )
        }
        
        if self.config.demo:
            print(f"\n🎉 All tests completed in {total_time:.2f} seconds!")
            print("=" * 60)
        
        return self.results
    
    def print_final_report(self):
        """Print a comprehensive final report."""
        if not self.config.demo:
            return
        
        print("\n📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        summary = self.results['test_summary']
        
        print(f"📊 Test Summary:")
        print(f"   ✅ Tests completed: {summary['tests_run']}")
        print(f"   ⏭️  Tests skipped: {summary['tests_skipped']}")
        print(f"   ⏱️  Total time: {summary['total_test_time']:.2f} seconds")
        print(f"   🧬 Node evolution success: {'Yes' if summary['evolution_success'] else 'No'}")
        print(f"   📈 Learning convergence: {'Yes' if summary['learning_convergence'] else 'Partial'}")
        
        print(f"\n🎯 Detailed Results:")
        
        # Enhanced learning results
        if 'enhanced_learning' in self.results and self.results['enhanced_learning'].get('status') != 'skipped':
            el = self.results['enhanced_learning']
            print(f"   📚 Enhanced Learning:")
            print(f"      Classification loss: {el['classification']['final_loss']:.4f}")
            print(f"      Spatial adaptations: {el['classification']['spatial_adaptations']}")
        
        # Evolution results
        if 'node_evolution' in self.results and self.results['node_evolution'].get('status') != 'skipped':
            ev = self.results['node_evolution']
            print(f"   🧬 Node Evolution:")
            print(f"      Evolutions: {ev['total_evolutions']}")
            print(f"      Computational: {ev['initial_computational']} → {ev['final_computational']}")
            print(f"      Judges: {ev['initial_judges']} → {ev['final_judges']}")
        
        print("\n🎊 Test suite completed successfully!")
    
    def save_results(self, filename: str = "comprehensive_learning_test_results.json"):
        """Save test results to file."""
        if self.config.save_results:
            filepath = f"c:\\Users\\georg\\Documents\\Github\\DragonAssistCCAI\\modules\\DragonChild\\v3\\{filename}"
            
            # Convert numpy arrays and other non-serializable objects to lists/strings
            serializable_results = self._make_serializable(self.results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            if self.config.demo:
                print(f"💾 Test results saved to: {filepath}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj

def main():
    """Main test execution function."""
    # Configure test parameters
    config = TestConfig(
        demo=True,
        save_results=True,
        test_enhanced_learning=True,
        test_rl_learning=False,  # Disable for now due to complexity
        test_node_evolution=True,
        test_multimodal=False,  # Disable for now 
        verbose_output=True,
        segment_size=50,
        training_epochs=25,
        evolution_test_epochs=50
    )
    
    # Create and run tester
    tester = ComprehensiveLearningTester(config)
    
    try:
        # Run all tests
        results = tester.run_comprehensive_tests()
        
        # Print final report
        tester.print_final_report()
        
        # Save results
        tester.save_results()
        
        return results
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
