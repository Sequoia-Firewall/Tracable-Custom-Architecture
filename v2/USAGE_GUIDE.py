#!/usr/bin/env python3
"""
Enhanced BrainNexus Usage Guide
==============================
This guide shows how to use the enhanced BrainNexus training capabilities.

First, run this script to generate sample data:
> python create_sample_training_data.py

Then follow the examples below to train on different types of data.
"""

import json
import os

def usage_guide():
    """Complete usage guide for enhanced BrainNexus training."""
    
    print("🧠 Enhanced BrainNexus Training Guide")
    print("="*60)
    
    print("\n📋 Step 1: Generate Sample Data")
    print("-"*40)
    print("Run this command to create sample training data:")
    print("  > python create_sample_training_data.py")
    print("\nThis will create 4 files:")
    print("  • sample_classification_data.json - Classification training data")
    print("  • sample_conversation_data.json - Conversation training data")
    print("  • sample_sequence_data.json - Sequence prediction data")
    print("  • sample_text_generation_data.json - Text generation data")
    
    print("\n🚀 Step 2: Basic Usage Examples")
    print("-"*40)
    
    # Classification Example
    print("\n1. Classification Training:")
    print("""
from main import BrainNexusManager

# Initialize manager and brain
manager = BrainNexusManager()
manager.initialize_new_brain(demo_mode=True)

# Load classification data
success = manager.load_training_data('sample_classification_data.json')
if success:
    print(f"Loaded {len(manager.training_data)} training samples")
    
    # Train the brain
    results = manager.train_brain()
    print(f"Training completed with {results['final_accuracy']:.3f} accuracy")
    
    # Save trained brain
    brain_file = manager.save_brain_state("classification_brain")
    print(f"Brain saved to: {brain_file}")
""")
    
    # Conversation Example
    print("\n2. Conversation Training:")
    print("""
from main import BrainNexusManager

# Initialize manager and brain
manager = BrainNexusManager()
manager.initialize_new_brain(demo_mode=True)

# Load conversation data
success = manager.load_conversation_data('sample_conversation_data.json')
if success:
    print(f"Loaded {len(manager.training_data)} conversation samples")
    
    # Train on conversations
    success = manager.train_on_conversations()
    if success:
        print("Conversation training completed!")
        
        # Save trained brain
        brain_file = manager.save_brain_state("conversation_brain")
        print(f"Brain saved to: {brain_file}")
""")
    
    # Sequence Example
    print("\n3. Sequence Training:")
    print("""
from main import BrainNexusManager

# Initialize manager and brain
manager = BrainNexusManager()
manager.initialize_new_brain(demo_mode=True)

# Load sequence data
success = manager.load_sequence_data('sample_sequence_data.json')
if success:
    print(f"Loaded {len(manager.training_data)} sequence samples")
    
    # Train on sequences
    success = manager.train_on_sequences()
    if success:
        print("Sequence training completed!")
        
        # Save trained brain
        brain_file = manager.save_brain_state("sequence_brain")
        print(f"Brain saved to: {brain_file}")
""")
    
    # Unsupervised Example
    print("\n4. Unsupervised Training:")
    print("""
from main import BrainNexusManager

# Initialize manager and brain
manager = BrainNexusManager()
manager.initialize_new_brain(demo_mode=True)

# Create or load unlabeled data
unlabeled_data = [
    {'features': [1.5, -0.5, 0.8, -1.2, 0.3, 1.1, -0.7, 0.4, -0.2, 0.9]},
    {'features': [2.1, 0.3, -0.6, 1.8, -0.9, 0.2, 1.4, -0.8, 0.7, -0.3]},
    # ... more data
]

# Try different unsupervised methods
methods = ['autoencoder', 'clustering', 'contrastive']
for method in methods:
    print(f"Training with {method}...")
    success = manager.train_unsupervised(unlabeled_data, method=method)
    if success:
        print(f"{method} training completed!")

# Save trained brain
brain_file = manager.save_brain_state("unsupervised_brain")
print(f"Brain saved to: {brain_file}")
""")
    
    # Hybrid Example
    print("\n5. Hybrid Training (Supervised + Unsupervised):")
    print("""
from main import BrainNexusManager

# Initialize manager and brain
manager = BrainNexusManager()
manager.initialize_new_brain(demo_mode=True)

# Load labeled data for supervised training
manager.load_training_data('sample_classification_data.json')

# Create unlabeled data for unsupervised training
unlabeled_data = [
    {'features': [1.5, -0.5, 0.8, -1.2, 0.3, 1.1, -0.7, 0.4, -0.2, 0.9]},
    {'features': [2.1, 0.3, -0.6, 1.8, -0.9, 0.2, 1.4, -0.8, 0.7, -0.3]},
    # ... more unlabeled data
]

# Hybrid training combines both approaches
success = manager.train_hybrid(
    supervised_data=manager.training_data,
    unsupervised_data=unlabeled_data
)

if success:
    print("Hybrid training completed!")
    
    # Save trained brain
    brain_file = manager.save_brain_state("hybrid_brain")
    print(f"Brain saved to: {brain_file}")
""")
    
    print("\n🔧 Step 3: Advanced Configuration")
    print("-"*40)
    print("""
# Custom training configuration
from BrainNexusLearn import TrainingConfig

# Create custom config
config = TrainingConfig(
    learning_rate=0.001,          # How fast the brain learns
    batch_size=8,                 # Training batch size
    max_epochs=50,                # Maximum training iterations
    data_type='conversation',     # Type of data: 'classification', 'conversation', 'sequence'
    context_window=5,             # Context size for conversations/sequences
    conversation_max_length=512,  # Max length for conversation processing
    clustering_k=6,               # Number of clusters for unsupervised learning
    unsupervised_weight=0.3       # Weight of unsupervised loss (0.0-1.0)
)

# Initialize with custom config
from BrainNexusLearn import BrainNexusLearn
learner = BrainNexusLearn(demo=True, config=config)

# Use in manager
manager = BrainNexusManager()
manager.learner = learner
manager.brain = learner
""")
    
    print("\n📊 Step 4: Testing and Inference")
    print("-"*40)
    print("""
# Load a trained brain
manager = BrainNexusManager()
success = manager.load_brain_state("classification_brain.pkl")

if success:
    # Test with new data
    test_input = "Hello, how are you?"
    result = manager.run_inference(test_input)
    
    print(f"Input: {test_input}")
    print(f"Confidence: {result.get('confidence', 0):.3f}")
    print(f"Prediction: {result.get('prediction', 'unknown')}")
""")
    
    print("\n💾 Step 5: Data Formats")
    print("-"*40)
    print("Classification data format (JSON):")
    print("""
{
  "training_data": [
    {
      "input": {
        "features": [1.5, -0.5, 0.8, -1.2, 0.3, 1.1, -0.7, 0.4, -0.2, 0.9],
        "sequence_length": 12,
        "metadata": {"category": "A"}
      },
      "output": 0
    }
  ],
  "validation_data": [...]
}""")
    
    print("\nConversation data format (JSON):")
    print("""
{
  "conversations": [
    [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there! How can I help you?"},
      {"role": "user", "content": "What's the weather like?"},
      {"role": "assistant", "content": "I'd be happy to help with weather info!"}
    ]
  ]
}""")
    
    print("\nSequence data format (JSON):")
    print("""
{
  "sequences": [
    [1, 3, 5, 7, 9, 11, 13, 15],
    [2, 4, 6, 8, 10, 12, 14, 16],
    ["A", "B", "A", "B", "A", "B"]
  ]
}""")
    
    print("\n✅ Step 6: Quick Start Checklist")
    print("-"*40)
    print("1. ✓ Run: python create_sample_training_data.py")
    print("2. ✓ Choose your training type (classification/conversation/sequence/unsupervised)")
    print("3. ✓ Initialize manager and brain")
    print("4. ✓ Load your data")
    print("5. ✓ Train the brain")
    print("6. ✓ Save the trained brain")
    print("7. ✓ Test with new data")
    
    print("\n🎯 Troubleshooting Tips")
    print("-"*40)
    print("• If training fails: Reduce batch_size or learning_rate")
    print("• If accuracy is low: Increase max_epochs or add more data")
    print("• If memory issues: Reduce context_window or batch_size")
    print("• If slow training: Increase batch_size or reduce max_epochs")
    
    print("\n🔗 File Dependencies")
    print("-"*40)
    print("Core files needed:")
    print("  • main.py - Main interface")
    print("  • BrainNexusLearn.py - Enhanced learning engine")
    print("  • brainNexus.py - Brain network")
    print("  • NeuralNode.py - Neural nodes")
    print("  • computations.py - Mathematical operations")
    
    print("\n🚀 You're ready to start training! Run the sample data script first.")

def create_simple_test():
    """Create a simple test script that definitely works."""
    
    test_script = '''#!/usr/bin/env python3
"""
Simple BrainNexus Test
=====================
A basic test to verify the enhanced training system works.
"""

def simple_test():
    """Simple test that should work."""
    print("🧠 Simple BrainNexus Test")
    print("="*40)
    
    try:
        # Import the main components
        from main import BrainNexusManager
        print("✓ Successfully imported BrainNexusManager")
        
        from BrainNexusLearn import BrainNexusLearn, TrainingConfig
        print("✓ Successfully imported BrainNexusLearn and TrainingConfig")
        
        # Create a basic configuration
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            max_epochs=10
        )
        print("✓ Created TrainingConfig")
        
        # Create manager
        manager = BrainNexusManager()
        print("✓ Created BrainNexusManager")
        
        # Initialize brain
        success = manager.initialize_new_brain(demo_mode=True)
        if success:
            print("✓ Brain initialized successfully")
            print(f"  Brain has {len(manager.brain.neural_nodes)} neural nodes")
        else:
            print("❌ Failed to initialize brain")
            return False
        
        # Create some simple test data
        test_data = []
        for i in range(20):
            features = [i % 3, (i * 2) % 5, (i + 1) % 4, i % 2, (i * 3) % 7, 
                       (i + 2) % 3, i % 6, (i * 4) % 3, (i + 3) % 2, i % 8]
            target = i % 3  # Simple 3-class classification
            
            input_data = {
                'features': features,
                'sequence_length': len(features),
                'metadata': {'sample_id': i}
            }
            test_data.append((input_data, target))
        
        print(f"✓ Created {len(test_data)} test samples")
        
        # Set the training data
        manager.training_data = test_data[:15]
        manager.validation_data = test_data[15:]
        
        print("✓ Assigned training and validation data")
        
        # Try basic training
        print("🚀 Starting training...")
        results = manager.train_brain()
        
        if results and 'final_accuracy' in results:
            print(f"✅ Training completed!")
            print(f"   Final accuracy: {results['final_accuracy']:.3f}")
            print(f"   Training epochs: {results.get('epochs_completed', 'unknown')}")
        else:
            print("⚠️ Training completed but no results returned")
        
        # Test inference
        print("🔍 Testing inference...")
        test_input = "test input"
        result = manager.run_inference(test_input)
        print(f"   Test result: {result}")
        
        print("✅ Simple test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\\n🎉 All tests passed! Your BrainNexus system is working correctly.")
    else:
        print("\\n🔧 Some issues detected. Check the error messages above.")
'''
    
    with open('simple_test.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created simple_test.py - run this to verify everything works!")

if __name__ == "__main__":
    usage_guide()
    print("\n" + "="*60)
    create_simple_test()
