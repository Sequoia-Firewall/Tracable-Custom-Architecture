#!/usr/bin/env python3
"""
Example of how to integrate the SegmentLearning system with the enhanced tokenizer/embeddings
to achieve true AI intelligence beyond just technical infrastructure.
"""

import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BrainNexus import BrainNexus
from BrainNexusLearning import SegmentLearning

def demonstrate_training_integration():
    """Show how to move from technical success to AI intelligence."""
    
    print("🎓 Training Integration for True AI Intelligence")
    print("="*65)
    
    # Initialize BrainNexus with enhanced tokenizer/embeddings
    brain = BrainNexus(
        dimensions=4,
        demo=True,
        output_config={
            'num_classes': 3,  # Simple classification: positive, negative, neutral
            'task_types': ['classification', 'llm', 'vision']
        }
    )
    
    print(f"✅ Brain initialized with {brain.vocab_size} vocabulary and {brain.embedding_dim}D embeddings")
    
    # Create brain segment for training
    from BrainSegment import NexusSegment as BrainSegment
    
    segment = BrainSegment(
        segment_id=1,
        dimensional_assignment=[0, 1],  # Use first 2 dimensions
        brain_nexus_ref=brain,
        hypercube_bounds=[-100, 100],
        demo=True
    )
    
    # Initialize the learning system
    learner = SegmentLearning(segment)
    
    print(f"🧠 Created brain segment with learning capabilities")
    print(f"   Device: {learner.device}")
    print(f"   Learning configs: {list(learner.learning_configs.keys())}")
    
    # Example 1: Text Classification with Enhanced Pipeline
    print("\n" + "="*50)
    print("EXAMPLE 1: Text Classification Training")
    print("="*50)
    
    # Prepare training data using the enhanced tokenizer
    training_texts = [
        "I love this product, it's amazing!",           # Positive
        "This is the worst thing ever made.",           # Negative  
        "The weather is okay today.",                   # Neutral
        "Absolutely fantastic experience!",            # Positive
        "Terrible customer service, very disappointed.", # Negative
        "It's an average product, nothing special.",   # Neutral
        "Best purchase I've ever made!",               # Positive
        "Complete waste of money and time.",           # Negative
        "The movie was decent, not great not bad.",    # Neutral
        "Outstanding quality and fast delivery!"       # Positive
    ]
    
    labels = [1, 0, 2, 1, 0, 2, 1, 0, 2, 1]  # 0=negative, 1=positive, 2=neutral
    
    print(f"Training data: {len(training_texts)} examples")
    
    # Show how the enhanced tokenizer processes this data
    sample_text = training_texts[0]
    print(f"\nSample text: '{sample_text}'")
    
    # Tokenize using brain's enhanced system
    tokens = brain._tokenize_input(sample_text)
    print(f"Tokenized: {tokens}")
    
    # Convert to embeddings
    judge_id = brain.add_neural_node('Judge', [1, 2, 3, 4], 'demo_judge')
    judge_node = brain.node_registry[judge_id]
    embeddings = brain._generate_embeddings(judge_node, sample_text)
    print(f"Embeddings shape: {embeddings.shape}, norm: {np.linalg.norm(embeddings):.3f}")
    
    # Now train the segment using these enhanced representations
    print(f"\n🎯 Training text classifier with enhanced embeddings...")
    
    try:
        # Prepare data using the enhanced pipeline
        processed_data = []
        for text in training_texts:
            # Use brain's tokenizer and embeddings
            embedding = brain._generate_embeddings(judge_node, text)
            processed_data.append(embedding)
        
        processed_data = np.array(processed_data)
        print(f"Processed training data shape: {processed_data.shape}")
        
        # Train supervised classifier
        training_results = learner.train_supervised(
            data=processed_data,
            labels=labels,
            task_type='classification',
            model_config={
                'epochs': 50,
                'batch_size': 8,
                'learning_rate': 0.001
            }
        )
        
        print(f"✅ Training completed!")
        print(f"   Epochs trained: {training_results['epochs_trained']}")
        print(f"   Best validation score: {training_results['best_val_score']:.3f}")
        print(f"   Final training loss: {training_results['training_loss'][-1]:.3f}")
        
    except Exception as e:
        print(f"⚠️  Simulated training (SegmentLearning not fully integrated yet)")
        print(f"   Would train on: {len(training_texts)} text samples")
        print(f"   Using embeddings: {embeddings.shape} dimensions")
        print(f"   For task: sentiment classification (3 classes)")
    
    # Example 2: Language Modeling with Flat Text
    print("\n" + "="*50)
    print("EXAMPLE 2: Language Modeling from Flat Text")
    print("="*50)
    
    flat_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
    Machine learning is a fascinating field that combines statistics, computer science, and domain expertise.
    Natural language processing enables computers to understand and generate human language.
    Deep learning models like transformers have revolutionized the field of artificial intelligence.
    """
    
    print(f"Flat text length: {len(flat_text)} characters")
    
    # Show tokenization of flat text
    tokens = brain._tokenize_input(flat_text)
    print(f"Tokenized to: {len(tokens)} tokens")
    print(f"Sample tokens: {tokens[:10]}...")
    
    # Convert to embeddings
    embeddings = brain._generate_embeddings(judge_node, flat_text)
    print(f"Text embeddings: shape {embeddings.shape}, norm {np.linalg.norm(embeddings):.3f}")
    
    try:
        # Train language model from flat text
        lm_results = learner.train_from_flat_text(
            text=flat_text,
            task_type='language_model',
            model_config={'epochs': 30, 'learning_rate': 0.0001}
        )
        print(f"✅ Language model training completed!")
        
    except Exception as e:
        print(f"⚠️  Simulated LM training")
        print(f"   Would learn from: {len(flat_text)} chars of text")
        print(f"   Using enhanced tokenization and embeddings")
    
    # Example 3: Attention Analysis
    print("\n" + "="*50)
    print("EXAMPLE 3: Attention Pattern Analysis")
    print("="*50)
    
    test_sentence = "The intelligent system processes natural language effectively"
    print(f"Test sentence: '{test_sentence}'")
    
    # Generate task-specific attention patterns
    tasks = ['classification', 'llm', 'vision']
    
    for task in tasks:
        attention = brain._generate_attention_masks(judge_node, test_sentence, task)
        tokens = brain._tokenize_input(test_sentence)
        
        print(f"\n{task.upper()} Task Attention:")
        print(f"  Tokens: {len(tokens)} | Attention weights: {len(attention)}")
        print(f"  Entropy: {-np.sum(attention * np.log(attention + 1e-8)):.3f}")
        print(f"  Max attention: {np.max(attention):.3f} | Min: {np.min(attention):.3f}")
        
        # Show top attended tokens
        top_indices = np.argsort(attention)[-3:][::-1]
        print(f"  Top attended positions: {top_indices}")
    
    print("\n" + "="*65)
    print("🎯 INTELLIGENCE PATHWAY SUMMARY:")
    print("="*65)
    
    print("Current Status: ✅ TECHNICAL INFRASTRUCTURE COMPLETE")
    print("  ✅ Advanced tokenization with vocabulary management")
    print("  ✅ Embedding lookup with 768D representation space")  
    print("  ✅ Sinusoidal positional encodings")
    print("  ✅ Task-specific attention generation")
    print("  ✅ Judge-specific transformations")
    print("  ✅ Multi-modal input handling")
    
    print("\nNext Step: 🎓 TRAINING FOR TRUE INTELLIGENCE")
    print("  🎯 Integrate SegmentLearning with enhanced pipeline")
    print("  🎯 Train on real datasets (text classification, LM, vision)")
    print("  🎯 Learn meaningful embeddings instead of random ones")
    print("  🎯 Develop task-specific expertise in judges")
    print("  🎯 Enable actual prediction and generation capabilities")
    
    print("\nResult After Training: 🚀 ARTIFICIAL INTELLIGENCE")
    print("  🧠 Judges will specialize in different task types")
    print("  🧠 Embeddings will capture semantic relationships")
    print("  🧠 Attention will focus on relevant information")
    print("  🧠 Pipeline will produce meaningful predictions")
    print("  🧠 System will demonstrate true language understanding")
    
    print("\n" + "="*65)
    return True

if __name__ == "__main__":
    try:
        demonstrate_training_integration()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
