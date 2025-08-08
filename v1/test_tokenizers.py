"""
Test script for the enhanced tokenizer functionality in main.py
"""

import sys
import os

# Add the current directory to the path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BrainNexusManager

def test_tokenizer_loading():
    """Test the tokenizer loading functionality"""
    print("🧪 Testing Tokenizer Loading Functionality")
    print("=" * 50)
    
    manager = BrainNexusManager()
    
    # Test GPT-2 tokenizer
    print("\n1. Testing GPT-2 Tokenizer:")
    gpt2_vocab = manager._load_gpt2_tokenizer(1000)
    print(f"   Loaded {len(gpt2_vocab)} tokens")
    print(f"   Sample tokens: {list(gpt2_vocab.values())[:10]}")
    
    # Test BERT tokenizer
    print("\n2. Testing BERT Tokenizer:")
    bert_vocab = manager._load_bert_tokenizer(1000)
    print(f"   Loaded {len(bert_vocab)} tokens")
    print(f"   Sample tokens: {list(bert_vocab.values())[:10]}")
    
    # Test Mistral v3 Tekken tokenizer
    print("\n3. Testing Mistral v3 Tekken Tokenizer:")
    mistral_vocab = manager._load_mistral_tokenizer(1000)
    print(f"   Loaded {len(mistral_vocab)} tokens")
    print(f"   Sample tokens: {list(mistral_vocab.values())[:10]}")
    
    # Test Basic English tokenizer
    print("\n4. Testing Basic English Tokenizer:")
    basic_vocab = manager._load_basic_english_tokenizer(1000)
    print(f"   Loaded {len(basic_vocab)} tokens")
    print(f"   Sample tokens: {list(basic_vocab.values())[:10]}")
    
    print("\n✅ All tokenizer tests completed successfully!")
    
    # Check for special tokens
    print("\n🔍 Special Token Analysis:")
    
    print(f"\nGPT-2 special tokens:")
    special_gpt2 = [token for token in gpt2_vocab.values() if token.startswith('<') or token.startswith('Ġ')][:5]
    print(f"   {special_gpt2}")
    
    print(f"\nBERT special tokens:")
    special_bert = [token for token in bert_vocab.values() if '[' in token or '##' in token][:5]
    print(f"   {special_bert}")
    
    print(f"\nMistral special tokens:")
    special_mistral = [token for token in mistral_vocab.values() if '<' in token or '▁' in token][:5]
    print(f"   {special_mistral}")
    
    print(f"\nBasic English special tokens:")
    special_basic = [token for token in basic_vocab.values() if '<' in token][:5]
    print(f"   {special_basic}")

if __name__ == "__main__":
    test_tokenizer_loading()
