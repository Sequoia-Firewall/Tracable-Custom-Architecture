#!/usr/bin/env python3
"""
Quick test to check actual Mistral v3 Tekken vocabulary size
"""

from main import BrainNexusManager

def check_mistral_vocab_size():
    """Check the actual vocabulary size of Mistral v3 Tekken tokenizer"""
    print("🔍 Checking Mistral v3 Tekken Vocabulary Size")
    print("=" * 45)
    
    manager = BrainNexusManager()
    
    try:
        # Load full Mistral tokenizer
        vocab_mapping = manager._load_mistral_tokenizer()
        
        print(f"📊 Actual Mistral v3 Tekken vocabulary size: {len(vocab_mapping)} tokens")
        print(f"📝 First 10 tokens: {list(vocab_mapping.values())[:10]}")
        print(f"📝 Last 5 tokens: {list(vocab_mapping.values())[-5:]}")
        
        # Check some specific token indices
        important_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print(f"\n🎯 Important token mappings:")
        for idx in important_indices:
            if idx in vocab_mapping:
                print(f"   Index {idx}: '{vocab_mapping[idx]}'")
        
        return len(vocab_mapping)
        
    except Exception as e:
        print(f"❌ Failed to load Mistral tokenizer: {e}")
        return None

if __name__ == "__main__":
    vocab_size = check_mistral_vocab_size()
    if vocab_size:
        print(f"\n✅ Mistral v3 Tekken has {vocab_size} tokens in its vocabulary")
    else:
        print(f"\n❌ Could not determine Mistral vocabulary size")
