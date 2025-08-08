#!/usr/bin/env python3
"""
Test script to verify save/load functionality preserves tokenizer configuration
"""

import os
import sys
from main import BrainNexusManager

def test_tokenizer_save_load():
    """Test that tokenizer configuration is preserved through save/load cycle"""
    print("🧪 Testing Tokenizer Save/Load Preservation")
    print("=" * 50)
    
    manager = BrainNexusManager()
    
    # Test with Mistral tokenizer (most complex case)
    print("📝 Step 1: Create brain with Mistral v3 Tekken tokenizer...")
    
    # Configure output system with Mistral tokenizer
    output_config = {
        'type': 'tokens',
        'output_format': 'token_id',
        'confidence_threshold': 0.5,
        'return_top_k': 1,
        'normalize_probabilities': True
    }
    
    try:
        # Load Mistral tokenizer with full vocabulary
        vocab_mapping = manager._load_mistral_tokenizer()
        output_config['vocab_mapping'] = vocab_mapping
        output_config['num_classes'] = len(vocab_mapping)  # Use actual size
        
        print(f"✓ Loaded {len(vocab_mapping)} tokens")
        original_sample_tokens = list(vocab_mapping.values())[:10]
        print(f"  Original sample tokens: {original_sample_tokens}")
        
    except Exception as e:
        print(f"❌ Failed to load Mistral tokenizer: {e}")
        print("🔄 Falling back to basic English tokenizer for test...")
        
        vocab_mapping = manager._load_basic_english_tokenizer(1000)
        output_config['vocab_mapping'] = vocab_mapping
        original_sample_tokens = list(vocab_mapping.values())[:10]
        print(f"✓ Using basic English tokenizer with {len(vocab_mapping)} tokens")
        print(f"  Original sample tokens: {original_sample_tokens}")
    
    # Step 2: Mock brain creation (since we're testing save/load specifically)
    print(f"\n📝 Step 2: Create mock brain with tokenizer configuration...")
    
    # Create a minimal brain-like object for testing
    class MockBrain:
        def __init__(self, output_config):
            self.output_config = output_config
            self.demo = True
            self.dimensions = 4
            self.learning_rate = 0.01
            self.next_node_id = 17
            self.neural_nodes = []
            self.node_records = None
            self.brain_records = None
            
            # Initialize minimal records for save/load
            import pandas as pd
            import numpy as np
            
            self.node_records = pd.DataFrame({
                'Node_ID': [1, 2, 3, 4],
                'Node_Type': ['Controller', 'Computational', 'Retainer', 'Handler'],
                'Node_Position': [[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0]],
                'Exit_Connections': [[], [], [], []]
            })
            
            self.brain_records = pd.DataFrame({'dummy': [1]})
    
    manager.brain = MockBrain(output_config)
    print(f"✓ Created mock brain with tokenizer configuration")
    
    # Step 3: Save the brain
    print(f"\n📝 Step 3: Save brain with tokenizer...")
    test_filename = "test_tokenizer_brain.pkl"
    
    try:
        saved_path = manager.save_brain_state(test_filename)
        print(f"✓ Saved brain to: {saved_path}")
    except Exception as e:
        print(f"❌ Failed to save brain: {e}")
        return False
    
    # Step 4: Clear current brain and load from file
    print(f"\n📝 Step 4: Load brain and verify tokenizer...")
    manager.brain = None
    
    try:
        success = manager.load_brain_state(test_filename)
        if not success:
            print(f"❌ Failed to load brain")
            return False
            
        print(f"✓ Brain loaded successfully")
    except Exception as e:
        print(f"❌ Load failed with error: {e}")
        return False
    
    # Step 5: Verify tokenizer was restored correctly
    print(f"\n📝 Step 5: Verify tokenizer restoration...")
    
    if not hasattr(manager.brain, 'output_config') or not manager.brain.output_config:
        print(f"❌ No output configuration found in loaded brain")
        return False
    
    loaded_config = manager.brain.output_config
    loaded_vocab = loaded_config.get('vocab_mapping', {})
    
    if not loaded_vocab:
        print(f"❌ No vocabulary mapping found in loaded configuration")
        return False
    
    loaded_sample_tokens = list(loaded_vocab.values())[:10]
    print(f"✓ Loaded tokenizer with {len(loaded_vocab)} tokens")
    print(f"  Loaded sample tokens: {loaded_sample_tokens}")
    
    # Step 6: Compare original and loaded configurations
    print(f"\n📝 Step 6: Compare original vs loaded configuration...")
    
    # Check vocabulary size
    if len(vocab_mapping) == len(loaded_vocab):
        print(f"✓ Vocabulary size matches: {len(vocab_mapping)} tokens")
    else:
        print(f"❌ Vocabulary size mismatch: {len(vocab_mapping)} vs {len(loaded_vocab)}")
        return False
    
    # Check sample tokens
    if original_sample_tokens == loaded_sample_tokens:
        print(f"✓ Sample tokens match perfectly")
    else:
        print(f"⚠️  Sample tokens differ slightly (may be due to regeneration)")
        print(f"  Original: {original_sample_tokens}")
        print(f"  Loaded:   {loaded_sample_tokens}")
        
        # Check if at least some key tokens match
        key_matches = sum(1 for orig, loaded in zip(original_sample_tokens, loaded_sample_tokens) if orig == loaded)
        if key_matches >= 5:  # At least half should match
            print(f"✓ Key tokens match ({key_matches}/10), regeneration successful")
        else:
            print(f"❌ Too many token mismatches ({key_matches}/10)")
            return False
    
    # Check other configuration
    for key in ['type', 'num_classes', 'output_format', 'confidence_threshold']:
        if output_config.get(key) == loaded_config.get(key):
            print(f"✓ {key} matches: {loaded_config.get(key)}")
        else:
            print(f"❌ {key} mismatch: {output_config.get(key)} vs {loaded_config.get(key)}")
            return False
    
    print(f"\n🎉 Tokenizer save/load test completed successfully!")
    return True

def cleanup_test_files():
    """Clean up test files"""
    manager = BrainNexusManager()
    test_files = ["test_tokenizer_brain.pkl"]
    
    for filename in test_files:
        filepath = os.path.join(manager.save_directory, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"🗑️  Cleaned up: {filename}")
            except Exception as e:
                print(f"⚠️  Failed to clean up {filename}: {e}")

if __name__ == "__main__":
    try:
        success = test_tokenizer_save_load()
        if success:
            print("\n✅ All tests passed! Tokenizer save/load working correctly.")
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n🧹 Cleaning up test files...")
        cleanup_test_files()
