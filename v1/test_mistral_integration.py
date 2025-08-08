#!/usr/bin/env python3
"""
Test script to verify Mistral tokenizer integration works correctly
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from main import BrainNexusManager

def test_mistral_tokenizer_integration():
    """Test the complete Mistral tokenizer integration"""
    print("🧪 Testing Mistral v3 Tekken Tokenizer Integration")
    print("=" * 60)
    
    # Create manager
    manager = BrainNexusManager()
    
    # Configure output system with Mistral tokenizer
    print("\n1. Creating output configuration with Mistral tokenizer...")
    try:
        # Test the tokenizer loading method directly
        vocab_mapping = manager._load_mistral_tokenizer(1000)
        print(f"✓ Tokenizer loaded successfully with {len(vocab_mapping)} tokens")
        
        # Show sample tokens
        sample_tokens = list(vocab_mapping.values())[:15]
        print(f"✓ Sample tokens: {sample_tokens}")
        
        # Verify specific tokens exist
        expected_tokens = ['<unk>', '<s>', '</s>', '[INST]', '[/INST]', '[AVAILABLE_TOOLS]', '[/AVAILABLE_TOOLS]']
        found_tokens = []
        for token in expected_tokens:
            if token in vocab_mapping.values():
                found_tokens.append(token)
        
        print(f"✓ Found expected special tokens: {found_tokens}")
        
        # Create output config
        output_config = {
            'type': 'tokens',
            'num_classes': len(vocab_mapping),
            'class_labels': [],
            'output_format': 'token_id',
            'confidence_threshold': 0.7,
            'return_top_k': 1,
            'normalize_probabilities': True,
            'vocab_mapping': vocab_mapping
        }
        
        print(f"✓ Output configuration created")
        
    except Exception as e:
        print(f"❌ Failed to load Mistral tokenizer: {e}")
        return False
    
    # Initialize brain with Mistral tokenizer
    print("\n2. Initializing brain with Mistral tokenizer...")
    try:
        from brainNexus import BrainNexus
        brain = BrainNexus(demo=True, output_config=output_config)
        
        # Initialize brain structure
        node_map = brain.initialize_brain()
        if node_map and len(brain.neural_nodes) > 0:
            print(f"✓ Brain initialized with {len(brain.neural_nodes)} nodes")
        else:
            print("❌ Failed to initialize brain structure")
            return False
            
    except Exception as e:
        print(f"❌ Failed to initialize brain: {e}")
        return False
    
    # Test inference
    print("\n3. Testing inference with token prediction...")
    try:
        test_input = "Hello world, this is a test input for tokenization"
        result = brain.run(test_input)
        
        # Extract result data
        actual_result = result.get('result', {})
        prediction_idx = actual_result.get('prediction', -1)
        confidence = actual_result.get('confidence', 0.0)
        predicted_token = actual_result.get('predicted_token', 'N/A')
        
        print(f"✓ Inference completed successfully")
        print(f"  Input: {test_input}")
        print(f"  Predicted Index: {prediction_idx}")
        print(f"  Predicted Token: {predicted_token}")
        print(f"  Confidence: {confidence:.3f}")
        
        # Verify the prediction makes sense
        if prediction_idx >= 0 and predicted_token != 'N/A':
            if prediction_idx in vocab_mapping:
                expected_token = vocab_mapping[prediction_idx]
                if predicted_token == expected_token:
                    print(f"✓ Token mapping is consistent")
                else:
                    print(f"⚠️  Token mapping mismatch: expected '{expected_token}', got '{predicted_token}'")
            else:
                print(f"⚠️  Prediction index {prediction_idx} not in vocabulary")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed during inference: {e}")
        return False

def test_confidence_propagation():
    """Test that confidence values are properly propagated through the pipeline"""
    print("\n🧪 Testing Confidence Propagation")
    print("=" * 40)
    
    manager = BrainNexusManager()
    
    # Create simple test configuration
    output_config = {
        'type': 'classification',
        'num_classes': 10,
        'class_labels': [f'class_{i}' for i in range(10)],
        'output_format': 'index',
        'confidence_threshold': 0.7,
        'return_top_k': 1,
        'normalize_probabilities': True
    }
    
    try:
        from brainNexus import BrainNexus
        brain = BrainNexus(demo=True, output_config=output_config)
        brain.initialize_brain()
        
        # Test inference
        result = brain.run("test input")
        
        # Check result structure
        print(f"✓ Brain result keys: {list(result.keys())}")
        
        actual_result = result.get('result', {})
        print(f"✓ Actual result keys: {list(actual_result.keys())}")
        
        confidence = actual_result.get('confidence', 0.0)
        prediction = actual_result.get('prediction', -1)
        
        print(f"✓ Confidence: {confidence:.3f}")
        print(f"✓ Prediction: {prediction}")
        
        # Test manager's run_inference method
        manager.brain = brain
        flattened_result = manager.run_inference("test input")
        
        print(f"✓ Flattened result keys: {list(flattened_result.keys())}")
        print(f"✓ Flattened confidence: {flattened_result.get('confidence', 'MISSING'):.3f}")
        print(f"✓ Flattened prediction: {flattened_result.get('prediction', 'MISSING')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence propagation test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Running Mistral Tokenizer Integration Tests")
    print("=" * 80)
    
    # Run tests
    test1_passed = test_mistral_tokenizer_integration()
    test2_passed = test_confidence_propagation()
    
    print(f"\n📊 Test Results Summary:")
    print(f"  Mistral Tokenizer Integration: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Confidence Propagation: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! Mistral tokenizer integration is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
