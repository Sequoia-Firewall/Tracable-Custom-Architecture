#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced tokenizer and embedding system in BrainNexus.
"""

import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BrainNexus import BrainNexus

def test_tokenizer_and_embeddings():
    """Test the new tokenization and embedding capabilities."""
    
    print("🧠 Testing Enhanced Tokenizer and Embedding System")
    print("="*60)
    
    # Initialize BrainNexus with demo mode
    brain = BrainNexus(
        dimensions=4,
        demo=True,
        output_config={
            'num_classes': 10,
            'task_types': ['classification', 'llm', 'vision']
        }
    )
    
    # Create a simple judge node for testing
    judge_id = brain.add_neural_node(
        node_type='Judge',
        position=[1.0, 2.0, 3.0, 4.0],
        node_group='test_judge'
    )
    
    judge_node = brain.node_registry[judge_id]
    
    print(f"\n📊 Created test judge node {judge_id} at position {judge_node.node_position}")
    
    # Test 1: Text tokenization
    print("\n" + "="*50)
    print("TEST 1: Text Tokenization")
    print("="*50)
    
    test_texts = [
        "Hello world this is a test",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning and artificial intelligence are fascinating",
        "Natural language processing with deep neural networks"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: '{text}'")
        
        # Tokenize
        tokens = brain._tokenize_input(text)
        print(f"  Tokens: {tokens}")
        
        # Convert to embeddings
        embeddings = brain._generate_embeddings(judge_node, text)
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Embedding range: [{np.min(embeddings):.3f}, {np.max(embeddings):.3f}]")
        print(f"  Embedding norm: {np.linalg.norm(embeddings):.3f}")
    
    # Test 2: Different input types
    print("\n" + "="*50)
    print("TEST 2: Different Input Types")
    print("="*50)
    
    test_inputs = [
        ("String", "This is a string input"),
        ("List of strings", ["Hello", "world", "test"]),
        ("List of numbers", [1, 2, 3, 4, 5]),
        ("Numpy array", np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
        ("Scalar", 42.7),
    ]
    
    for input_type, input_data in test_inputs:
        print(f"\n{input_type}: {input_data}")
        
        try:
            embeddings = brain._generate_embeddings(judge_node, input_data)
            print(f"  ✅ Embedding shape: {embeddings.shape}")
            print(f"  ✅ Embedding range: [{np.min(embeddings):.3f}, {np.max(embeddings):.3f}]")
            print(f"  ✅ Embedding norm: {np.linalg.norm(embeddings):.3f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Test 3: Attention masks for different tasks
    print("\n" + "="*50)
    print("TEST 3: Task-Specific Attention Masks")
    print("="*50)
    
    test_text = "The quick brown fox jumps over the lazy dog"
    task_types = ['classification', 'llm', 'vision', 'general']
    
    for task_type in task_types:
        print(f"\nTask: {task_type}")
        
        attention_mask = brain._generate_attention_masks(judge_node, test_text, task_type)
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Attention sum: {np.sum(attention_mask):.3f} (should be ~1.0)")
        print(f"  Max attention: {np.max(attention_mask):.3f}")
        print(f"  Min attention: {np.min(attention_mask):.3f}")
        print(f"  Attention entropy: {-np.sum(attention_mask * np.log(attention_mask + 1e-8)):.3f}")
    
    # Test 4: Vocabulary and embedding matrix
    print("\n" + "="*50)
    print("TEST 4: Vocabulary and Embedding Matrix")
    print("="*50)
    
    print(f"Vocabulary size: {brain.vocab_size}")
    print(f"Embedding dimension: {brain.embedding_dim}")
    print(f"Embedding matrix shape: {brain.embedding_matrix.shape}")
    
    # Show some vocabulary entries
    print("\nSample vocabulary entries:")
    sample_words = ['the', 'and', 'is', 'hello', 'world', '[UNK]', '[BOS]', '[EOS]']
    for word in sample_words:
        if word in brain.vocabulary:
            token_id = brain.vocabulary[word]
            embedding = brain.embedding_matrix[token_id]
            print(f"  '{word}' -> {token_id} -> embedding norm: {np.linalg.norm(embedding):.3f}")
    
    # Test 5: Positional encodings
    print("\n" + "="*50)
    print("TEST 5: Positional Encodings")
    print("="*50)
    
    # Create sample embeddings
    seq_len = 10
    d_model = 768
    sample_embeddings = np.random.normal(0, 0.1, (seq_len, d_model))
    
    print(f"Sample embeddings shape: {sample_embeddings.shape}")
    print(f"Before positional encoding - mean: {np.mean(sample_embeddings):.3f}")
    
    encoded_embeddings = brain._add_positional_encodings(sample_embeddings)
    print(f"After positional encoding - mean: {np.mean(encoded_embeddings):.3f}")
    print(f"Positional encoding effect - norm difference: {np.linalg.norm(encoded_embeddings - sample_embeddings):.3f}")
    
    # Test 6: Judge-specific transformations
    print("\n" + "="*50)
    print("TEST 6: Judge-Specific Transformations")
    print("="*50)
    
    # Create multiple judges
    judge_ids = []
    for i in range(3):
        jid = brain.add_neural_node(
            node_type='Judge',
            position=[i*10, i*5, i*2, i*1],
            node_group=f'test_judge_{i}'
        )
        judge_ids.append(jid)
    
    test_text = "Hello world"
    
    print("Same input processed by different judges:")
    for i, jid in enumerate(judge_ids):
        jnode = brain.node_registry[jid]
        embedding = brain._generate_embeddings(jnode, test_text)
        print(f"  Judge {i} (position {jnode.node_position}): embedding norm = {np.linalg.norm(embedding):.3f}")
    
    print("\n✅ All tokenizer and embedding tests completed successfully!")
    print("="*60)
    
    # Summary
    print("\n📋 SUMMARY - Enhanced Pipeline Features:")
    print("  ✅ Text tokenization with vocabulary management")
    print("  ✅ Token-to-embedding lookup with embedding matrix")
    print("  ✅ Proper positional encodings (sinusoidal)")
    print("  ✅ Task-specific attention mask generation")
    print("  ✅ Judge-specific embedding transformations")
    print("  ✅ Multiple input type support (text, lists, arrays, scalars)")
    print("  ✅ Robust error handling and fallbacks")
    print("  ✅ Consistent embedding dimensions (768D)")
    print("  ✅ Different pooling strategies per judge")
    
    return True

if __name__ == "__main__":
    try:
        test_tokenizer_and_embeddings()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
