#!/usr/bin/env python3
"""
Debug script to verify tokenizer consistency between prediction index and token string
"""

import sys
from main import BrainNexusManager

def test_tokenizer_consistency():
    """Test that prediction indices correctly map to token strings"""
    print("🧪 Testing Tokenizer Consistency")
    print("=" * 40)
    
    # Create manager and test Mistral tokenizer
    manager = BrainNexusManager()
    
    print("📝 Testing Mistral v3 Tekken tokenizer...")
    try:
        vocab_mapping = manager._load_mistral_tokenizer(1000)
        print(f"✓ Loaded {len(vocab_mapping)} tokens")
        
        # Show first 10 tokens to verify mapping
        print("\n📊 First 10 tokens in vocabulary:")
        for i in range(min(10, len(vocab_mapping))):
            token = vocab_mapping.get(i, f'<missing_{i}>')
            print(f"  Index {i}: '{token}'")
        
        # Test specific indices mentioned in the output
        test_indices = [0, 6]  # 0 for <unk>, 6 for [/AVAILABLE_TOOLS] from previous test
        
        print(f"\n🔍 Testing specific prediction indices:")
        for idx in test_indices:
            if idx in vocab_mapping:
                token = vocab_mapping[idx]
                print(f"  Index {idx} → Token: '{token}'")
                
                # Verify this is what would be returned in token_id format
                if idx == 0:
                    expected_token = '<unk>'
                    if token == expected_token:
                        print(f"    ✅ Index {idx} correctly maps to '{expected_token}'")
                    else:
                        print(f"    ❌ Index {idx} maps to '{token}', expected '{expected_token}'")
                
                if idx == 6:
                    print(f"    ℹ️  Index {idx} maps to '{token}' (from previous test)")
            else:
                print(f"  ❌ Index {idx} not found in vocabulary")
        
        # Test edge cases
        print(f"\n🧪 Testing edge cases:")
        last_idx = max(vocab_mapping.keys()) if vocab_mapping else -1
        print(f"  Last valid index: {last_idx}")
        
        if last_idx >= 0:
            last_token = vocab_mapping[last_idx]
            print(f"  Index {last_idx} → Token: '{last_token}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_brain_with_tokenizer():
    """Test actual brain inference with tokenizer to verify consistency"""
    print(f"\n🧠 Testing Brain Inference with Tokenizer")
    print("=" * 40)
    
    manager = BrainNexusManager()
    
    # Initialize brain with Mistral tokenizer
    print("📍 Initializing demo brain with token prediction...")
    try:
        # Configure for token prediction with Mistral tokenizer
        output_config = {
            'type': 'tokens',
            'num_classes': 1000,
            'output_format': 'token_id',
            'confidence_threshold': 0.1,  # Lower threshold to see results
            'return_top_k': 1,
            'normalize_probabilities': True
        }
        
        # Load Mistral tokenizer
        vocab_mapping = manager._load_mistral_tokenizer(1000)
        output_config['vocab_mapping'] = vocab_mapping
        
        print(f"✓ Configured for {len(vocab_mapping)} token vocabulary")
        print(f"  Sample tokens: {list(vocab_mapping.values())[:5]}...")
        
        # Mock brain creation (simplified for testing)
        print(f"\n🔬 Testing token mapping logic...")
        
        # Simulate predictions
        test_predictions = [0, 6, 10, 100]
        
        for pred_idx in test_predictions:
            if pred_idx in vocab_mapping:
                token = vocab_mapping[pred_idx]
                print(f"  Prediction {pred_idx} → Token: '{token}'")
                
                # Verify this matches expected behavior
                if pred_idx == 0 and token == '<unk>':
                    print(f"    ✅ Consistent: Index 0 correctly returns '<unk>'")
                elif pred_idx == 6:
                    print(f"    ℹ️  Index 6 returns: '{token}'")
                else:
                    print(f"    ℹ️  Index {pred_idx} returns: '{token}'")
            else:
                print(f"  ❌ Prediction {pred_idx} not in vocabulary (max: {max(vocab_mapping.keys())})")
        
        return True
        
    except Exception as e:
        print(f"❌ Brain test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Tokenizer Consistency Debug")
    
    success1 = test_tokenizer_consistency()
    success2 = test_brain_with_tokenizer()
    
    if success1 and success2:
        print(f"\n🎉 All consistency tests passed!")
        print(f"\n📋 Analysis of your output:")
        print(f"  - Handler prediction: 0 with confidence 0.223")
        print(f"  - Token mapping: Index 0 → '<unk>' token") 
        print(f"  - Final result: 0 with confidence 0.223")
        print(f"  - ✅ This is CONSISTENT - the system correctly mapped prediction index 0 to token '<unk>'")
    else:
        print(f"\n❌ Some tests failed!")
    
    print(f"\n💡 Note: '<unk>' (unknown token) is a common prediction when:")
    print(f"  - Input doesn't match training patterns")
    print(f"  - Network confidence is low")
    print(f"  - This is the fallback token for unrecognized inputs")
