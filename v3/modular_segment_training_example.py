#!/usr/bin/env python3
"""
Modular Segment Training Example

Demonstrat    # Create segment learning systems - enable real neural updates with better visibility
    segment_learners = []
    for i, segment in enumerate(segments):
        learner = SegmentLearning(
            brain_segment=segment,
            learning_config={
                'learning_rate': 0.002,  # Slightly higher for more visible changes
                'max_epochs': 5,  # Even shorter for demonstration
                'batch_size': 16,
                'enable_spatial_adaptation': True,
                'enable_attention_training': True,
                'enable_connection_pruning': True,
                'spatial_update_frequency': 1,  # Update every epoch to show changes
                'early_stopping_patience': 10  # Allow more epochs before stopping
            }
        )
        segment_learners.append(learner)n individual segments of the BrainNexus architecture
using the SegmentLearning system. This example shows:

1. Creating segments with different dimensional assignments
2. Training segments individually on different tasks
3. Training specific node types within segments
4. Combining trained segments for full brain performance

This modular approach allows for:
- Specialized training per dimensional region
- Independent optimization of segment components
- Efficient resource utilization
- Parallel training of multiple segments
"""

import sys
import os
import numpy as np
import torch

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BrainNexus import BrainNexus
from BrainSegment import NexusSegment
from BrainNexusLearning import SegmentLearning, LearningTask


def demonstrate_segment_modular_training():
    """Demonstrate the modular segment training capabilities."""
    
    print("🧠 Modular Segment Training Demonstration")
    print("="*60)
    
    # Create a 4D brain
    brain = BrainNexus(dimensions=4, demo=True)
    
    print(f"\n📊 Created 4D BrainNexus with {brain.dimensions} dimensions")
    
    # Create multiple segments with different dimensional assignments
    segments = []
    
    # Segment 1: Focuses on +x, +y quadrant (text processing)
    segment_1 = NexusSegment(
        segment_id=1,
        dimensional_assignment={0: 1, 1: 1},  # +x, +y
        brain_nexus_ref=brain,
        demo=True
    )
    segments.append(segment_1)
    
    # Segment 2: Focuses on -x, +y quadrant (vision processing) 
    segment_2 = NexusSegment(
        segment_id=2,
        dimensional_assignment={0: -1, 1: 1},  # -x, +y
        brain_nexus_ref=brain,
        demo=True
    )
    segments.append(segment_2)
    
    # Segment 3: Focuses on z, w dimensions (multimodal)
    segment_3 = NexusSegment(
        segment_id=3,
        dimensional_assignment={2: 1, 3: -1},  # +z, -w
        brain_nexus_ref=brain,
        demo=True
    )
    segments.append(segment_3)
    
    print(f"\\n🏗️  Created {len(segments)} specialized segments")
    
    # Create segment learners
    segment_learners = []
    for segment in segments:
        learner = SegmentLearning(
            brain_segment=segment,
            learning_config={
                'learning_rate': 0.001,
                'max_epochs': 20,
                'batch_size': 16,
                'enable_spatial_adaptation': True,
                'enable_attention_training': True
            }
        )
        segment_learners.append(learner)
    
    print(f"\\n🎓 Created {len(segment_learners)} segment learning systems")
    
    # Demonstrate individual segment training
    print(f"\\n" + "="*60)
    print("INDIVIDUAL SEGMENT TRAINING")
    print("="*60)
    
    # Train Segment 1 on text classification
    print(f"\\n📝 Training Segment 1 (text processing)")
    text_task = LearningTask(
        task_id="text_classification",
        task_type="supervised", 
        modality="text",
        objective="classification",
        data_shape=(100,),
        num_classes=5,
        learning_rate=0.001,
        max_epochs=15
    )
    
    text_data = [
        "This is a positive sentiment example",
        "This is a negative sentiment example", 
        "Neutral text goes here",
        "Another positive example",
        "Another negative example"
    ] * 20  # Repeat to get more samples
    
    text_labels = [1, 0, 2, 1, 0] * 20  # Corresponding labels
    
    text_results = segment_learners[0].train_segment(
        learning_task=text_task,
        data=text_data,
        labels=text_labels
    )
    
    print(f"   ✅ Segment 1 training complete")
    print(f"      Final loss: {text_results.get('training_losses', [0])[-1]:.4f}")
    print(f"      Node performances: {list(text_results['node_performances'].keys())}")
    
    # Show comprehensive neural changes from training
    if 'neural_changes' in text_results:
        print(f"\n🧠 Comprehensive Neural Architecture Changes:")
        total_weight_changes = 0
        total_position_changes = 0
        total_connection_changes = 0
        total_attention_changes = 0
        
        for node_type, changes in text_results['neural_changes'].items():
            if isinstance(changes, dict) and changes.get('changes_detected'):
                print(f"      {node_type.upper()}: {changes}")
                total_weight_changes += changes.get('weight_changes', 0)
                total_position_changes += changes.get('position_changes', 0)
                total_connection_changes += changes.get('connection_changes', 0)
                total_attention_changes += changes.get('attention_changes', 0)
        
        print(f"\n📊 Total Neural Updates Summary:")
        print(f"      • Total weight parameters modified: {total_weight_changes}")
        print(f"      • Total node positions adjusted: {total_position_changes}")
        print(f"      • Total connection changes: {total_connection_changes}")
        print(f"      • Total attention cache updates: {total_attention_changes}")
        print(f"      • Total nodes affected: {text_results.get('nodes_updated', 0)}")
        
        # Show learning progression
        if 'training_losses' in text_results and len(text_results['training_losses']) > 1:
            initial_loss = text_results['training_losses'][0]
            final_loss = text_results['training_losses'][-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
            print(f"      • Learning improvement: {improvement:.2f}% loss reduction")
            print(f"      • Training epochs completed: {len(text_results['training_losses'])}")
    
    # Train Segment 2 on vision-like data
    print(f"\\n👁️  Training Segment 2 (vision processing)")
    vision_task = LearningTask(
        task_id="image_classification",
        task_type="supervised",
        modality="vision", 
        objective="classification",
        data_shape=(224, 224, 3),
        num_classes=10
    )
    
    # Simulate image data with random arrays
    vision_data = [np.random.randn(224, 224, 3) for _ in range(100)]
    vision_labels = np.random.randint(0, 10, 100)
    
    vision_results = segment_learners[1].train_segment(
        learning_task=vision_task,
        data=vision_data,
        labels=vision_labels
    )
    
    print(f"   ✅ Segment 2 training complete")
    print(f"      Final loss: {vision_results.get('training_losses', [0])[-1]:.4f}")
    
    # Train Segment 3 on multimodal data
    print(f"\\n🔀 Training Segment 3 (multimodal processing)")
    multimodal_task = LearningTask(
        task_id="multimodal_fusion",
        task_type="supervised",
        modality="multimodal",
        objective="regression",
        data_shape=(512,),
        learning_rate=0.0005
    )
    
    # Simulate multimodal data
    multimodal_data = [np.random.randn(512) for _ in range(80)]
    multimodal_labels = np.random.randn(80)
    
    multimodal_results = segment_learners[2].train_segment(
        learning_task=multimodal_task,
        data=multimodal_data,
        labels=multimodal_labels
    )
    
    print(f"   ✅ Segment 3 training complete")
    print(f"      Final loss: {multimodal_results.get('training_losses', [0])[-1]:.4f}")
    
    # Demonstrate node-type specific training
    print(f"\\n" + "="*60)
    print("NODE-TYPE SPECIFIC TRAINING")
    print("="*60)
    
    # Train only judge nodes in Segment 1
    print(f"\\n⚖️  Training only Judge nodes in Segment 1")
    judge_results = segment_learners[0].train_node_type(
        node_type='judges',
        learning_task=text_task,
        data=text_data[:10],  # Smaller subset for demo
        labels=text_labels[:10]
    )
    
    print(f"   ✅ Judge training: {judge_results['status']}")
    print(f"      Judges trained: {judge_results.get('nodes', 0)}")
    
    # Train only splitter nodes in Segment 2  
    print(f"\\n🔄 Training only Splitter nodes in Segment 2")
    splitter_results = segment_learners[1].train_node_type(
        node_type='splitters',
        learning_task=vision_task,
        data=vision_data[:10],
        labels=vision_labels[:10]
    )
    
    print(f"   ✅ Splitter training: {splitter_results['status']}")
    print(f"      Splitters trained: {splitter_results.get('nodes', 0)}")
    
    # Get comprehensive performance metrics
    print(f"\\n" + "="*60) 
    print("SEGMENT PERFORMANCE ANALYSIS")
    print("="*60)
    
    for i, learner in enumerate(segment_learners):
        performance = learner.get_segment_performance()
        print(f"\\n📊 Segment {i+1} Performance:")
        print(f"   Segment ID: {performance['segment_id']}")
        print(f"   Dimensional assignment: {performance['dimensional_assignment']}")
        print(f"   Node types with performance data: {list(performance['node_performances'].keys())}")
        print(f"   Spatial metrics: {performance['spatial_metrics']}")
        print(f"   Resource utilization: {performance['resource_utilization']}")
    
    # Save segment states
    print(f"\\n💾 Saving segment states...")
    for i, learner in enumerate(segment_learners):
        filepath = f"segment_{learner.segment_id}_trained_state.pkl"
        learner.save_segment_state(filepath)
        print(f"   Segment {learner.segment_id} saved to {filepath}")
    
    # Demonstrate loading and resuming training
    print(f"\\n📂 Demonstrating state loading...")
    new_learner = SegmentLearning(segments[0])
    new_learner.load_segment_state("segment_1_trained_state.pkl")
    print(f"   ✅ Segment 1 state loaded successfully")
    
    print(f"\\n" + "="*60)
    print("🎯 MODULAR TRAINING BENEFITS DEMONSTRATED")
    print("="*60)
    
    print("✅ Individual segment training on specialized tasks")
    print("✅ Node-type specific training within segments") 
    print("✅ Spatial and connection optimization per segment")
    print("✅ Independent performance tracking and metrics")
    print("✅ State saving and loading for trained segments")
    print("✅ Modular architecture allows parallel training")
    print("✅ Specialized processing per dimensional region")
    
    print(f"\\n🚀 The modular approach enables:")
    print("   • Training segments on different hardware")
    print("   • Specialized optimization per task type")
    print("   • Efficient resource utilization") 
    print("   • Independent deployment of trained segments")
    print("   • Easy experimentation with segment configurations")
    
    return {
        'segments_trained': len(segment_learners),
        'text_performance': text_results,
        'vision_performance': vision_results,
        'multimodal_performance': multimodal_results,
        'segment_performances': [learner.get_segment_performance() for learner in segment_learners]
    }


if __name__ == "__main__":
    try:
        results = demonstrate_segment_modular_training()
        print(f"\\n🎉 Modular training demonstration completed successfully!")
        print(f"   Total segments trained: {results['segments_trained']}")
        
    except Exception as e:
        print(f"❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
