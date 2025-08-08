#!/usr/bin/env python3
"""
Quick Test of Enhanced Training System

This script demonstrates that the modular training system is actually
affecting neural components (weights, connections, positions, attention).
"""

import sys
import os
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BrainNexus import BrainNexus
from BrainSegment import NexusSegment
from BrainNexusLearning import SegmentLearning, LearningTask


def quick_training_test():
    print("🧪 Quick Enhanced Training Test")
    print("="*50)
    
    # Create a simple 2D brain
    brain = BrainNexus(dimensions=2, demo=True)
    
    # Create one segment for focused testing
    segment = NexusSegment(
        segment_id=1,
        dimensional_assignment={0: 1, 1: 1},  # +x, +y quadrant
        brain_nexus_ref=brain,
        demo=True
    )
    
    print(f"\n📊 Created segment with {len(segment.segment_nodes)} nodes")
    print(f"   Node types: {list(segment.node_type_registry.keys())}")
    
    # Capture initial state
    initial_weights = {}
    for node_id, node in segment.segment_nodes.items():
        if hasattr(node, 'weights'):
            initial_weights[node_id] = dict(node.weights)
    
    print(f"\n🔍 Initial State Captured:")
    print(f"   Nodes with weights: {len(initial_weights)}")
    print(f"   Sample weight values: {list(list(initial_weights.values())[0].values()) if initial_weights else 'None'}")
    
    # Create learner with aggressive settings
    learner = SegmentLearning(
        brain_segment=segment,
        learning_config={
            'learning_rate': 0.01,  # High learning rate for visible changes
            'max_epochs': 3,
            'batch_size': 8,
            'enable_spatial_adaptation': True,
            'enable_attention_training': True,
            'enable_connection_pruning': True,
            'spatial_update_frequency': 1
        }
    )
    
    # Simple training task
    task = LearningTask(
        task_id="quick_test",
        task_type="supervised",
        modality="text",
        objective="classification",
        data_shape=(50,),
        num_classes=3,
        learning_rate=0.01,
        max_epochs=3
    )
    
    # Simple training data
    data = ["test input " + str(i) for i in range(20)]
    labels = [i % 3 for i in range(20)]
    
    print(f"\n🏋️ Training for {task.max_epochs} epochs...")
    
    # Train the segment
    results = learner.train_segment(
        learning_task=task,
        data=data,
        labels=labels
    )
    
    # Capture final state
    final_weights = {}
    for node_id, node in segment.segment_nodes.items():
        if hasattr(node, 'weights'):
            final_weights[node_id] = dict(node.weights)
    
    print(f"\n🔍 Final State Captured:")
    print(f"   Training results: {list(results.keys())}")
    
    # Compare states
    changes_detected = 0
    total_weight_changes = 0
    
    for node_id in initial_weights:
        if node_id in final_weights:
            initial = initial_weights[node_id]
            final = final_weights[node_id]
            
            node_changes = 0
            for key in initial:
                if key in final:
                    change = abs(final[key] - initial[key])
                    if change > 1e-6:
                        node_changes += 1
                        total_weight_changes += 1
            
            if node_changes > 0:
                changes_detected += 1
                print(f"   Node {node_id}: {node_changes} weight parameters changed")
                print(f"      Before: {initial}")
                print(f"      After:  {final}")
    
    print(f"\n📈 Training Effectiveness Summary:")
    print(f"   • Nodes with weight changes: {changes_detected}/{len(initial_weights)}")
    print(f"   • Total weight parameters modified: {total_weight_changes}")
    print(f"   • Training loss progression: {results.get('training_losses', 'N/A')}")
    print(f"   • Neural changes detected: {bool(results.get('neural_changes', {}))}")
    
    if 'neural_changes' in results:
        for node_type, changes in results['neural_changes'].items():
            if isinstance(changes, dict) and changes.get('changes_detected'):
                print(f"   • {node_type}: {changes}")
    
    # Verdict
    if changes_detected > 0 or total_weight_changes > 0:
        print(f"\n✅ SUCCESS: Real neural training detected!")
        print(f"   The system is actually modifying neural components.")
    else:
        print(f"\n❌ WARNING: No neural changes detected.")
        print(f"   Training may not be affecting the underlying architecture.")
    
    return {
        'changes_detected': changes_detected,
        'weight_changes': total_weight_changes,
        'training_results': results
    }


if __name__ == "__main__":
    try:
        test_results = quick_training_test()
        print(f"\n🏁 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
