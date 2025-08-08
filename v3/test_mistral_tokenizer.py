#!/usr/bin/env python3
"""
Test script to demonstrate Mistral v3 Tekken tokenizer integration in BrainNexus.
"""

import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BrainNexus import BrainNexus

def test_mistral_tokenizer():
    """Test the Mistral v3 Tekken tokenizer integration."""
    
    print("🚀 Testing Mistral v3 Tekken Tokenizer Integration")
    print("="*65)
    
    # Initialize BrainNexus with Mistral tokenizer
    brain = BrainNexus(
        dimensions=4,
        demo=True,
        output_config={
            'num_classes': 10,
            'task_types': ['classification', 'llm', 'vision']
        }
    )
    
    # Get tokenizer information
    tokenizer_info = brain.get_tokenizer_info()
    print(f"\n📊 Tokenizer Information:")
    print(f"   Type: {tokenizer_info['tokenizer_type']}")
    print(f"   Vocabulary size: {tokenizer_info['vocab_size']:,}")
    print(f"   Embedding dimension: {tokenizer_info['embedding_dim']}")
    print(f"   Mistral available: {tokenizer_info['mistral_available']}")
    
    # Debug tokenizer state
    print(f"\n🔍 Debug Info:")
    print(f"   Has mistral_tokenizer attr: {hasattr(brain, 'mistral_tokenizer')}")
    if hasattr(brain, 'mistral_tokenizer'):
        print(f"   mistral_tokenizer is not None: {brain.mistral_tokenizer is not None}")
        print(f"   mistral_tokenizer type: {type(brain.mistral_tokenizer)}")
    print(f"   Has vocab_size attr: {hasattr(brain, 'vocab_size')}")
    if hasattr(brain, 'vocab_size'):
        print(f"   vocab_size value: {brain.vocab_size}")
    print(f"   Has embedding_dim attr: {hasattr(brain, 'embedding_dim')}")
    if hasattr(brain, 'embedding_dim'):
        print(f"   embedding_dim value: {brain.embedding_dim}")
    
    if tokenizer_info['tokenizer_type'] == 'mistral_v3_tekken':
        print(f"   ✅ Mistral version: {tokenizer_info['mistral_version']}")
        print(f"   ✅ Tekken enabled: {tokenizer_info['tekken_enabled']}")
        print(f"   ✅ Model: {tokenizer_info['model']}")
    else:
        print(f"   ⚠️  Using fallback tokenization")
    
    # Create a judge node for testing
    judge_id = brain.add_neural_node(
        node_type='Judge',
        position=[1.0, 2.0, 3.0, 4.0],
        node_group='mistral_test_judge'
    )
    
    judge_node = brain.node_registry[judge_id]
    print(f"\n🧠 Created test judge node {judge_id}")
    
    # Test various text samples
    test_texts = [
        "Hello world, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are transforming our world.",
        "Natural language processing enables computers to understand human language.",
        "Mistral v3 with Tekken tokenization provides advanced text processing capabilities.",
        "This is a test of the tokenization system with various punctuation marks: !@#$%^&*()",
        "Can the tokenizer handle special characters like émojis? 🤖🚀✨",
        "What about numbers: 123, 456.789, and mathematical symbols like π ≈ 3.14159?",
        "Code snippets: def tokenize(text): return model.encode(text)",
        "Multiple languages: Bonjour, Hola, こんにちは, مرحبا, Здравствуйте"
    ]
    
    print(f"\n" + "="*60)
    print("TOKENIZATION TESTS")
    print("="*60)
    
    total_tokens = 0
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text}'")
        
        # Tokenize using Mistral/fallback
        tokens = brain._tokenize_input(text)
        total_tokens += len(tokens)
        
        print(f"  Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        
        # Try to decode back
        try:
            decoded = brain.decode_tokens(tokens)
            print(f"  Decoded: '{decoded}'")
            
            # Check if round-trip is successful
            if decoded.strip().replace(' ', '').lower() in text.replace(' ', '').lower():
                print(f"  ✅ Round-trip successful")
            else:
                print(f"  ⚠️  Round-trip differs (normal for some tokenizers)")
        except Exception as e:
            print(f"  ❌ Decoding error: {e}")
        
        # Generate embeddings
        try:
            embeddings = brain._generate_embeddings(judge_node, text)
            print(f"  Embeddings: shape {embeddings.shape}, norm {np.linalg.norm(embeddings):.3f}")
        except Exception as e:
            print(f"  ❌ Embedding error: {e}")
    
    print(f"\n📈 Total tokens processed: {total_tokens:,}")
    print(f"📈 Average tokens per text: {total_tokens / len(test_texts):.1f}")
    
    # Test attention generation with Mistral tokens
    print(f"\n" + "="*60)
    print("ATTENTION MASK GENERATION")
    print("="*60)
    
    test_text = "The Mistral v3 Tekken tokenizer provides advanced natural language processing capabilities."
    
    task_types = ['classification', 'llm', 'vision', 'general']
    
    for task_type in task_types:
        print(f"\nTask: {task_type.upper()}")
        
        tokens = brain._tokenize_input(test_text)
        attention_mask = brain._generate_attention_masks(judge_node, test_text, task_type)
        
        print(f"  Text length: {len(test_text)} chars")
        print(f"  Token count: {len(tokens)}")
        print(f"  Attention shape: {attention_mask.shape}")
        print(f"  Attention sum: {np.sum(attention_mask):.3f} (should be ~1.0)")
        print(f"  Max/Min attention: {np.max(attention_mask):.3f} / {np.min(attention_mask):.3f}")
    
    # Test embedding consistency
    print(f"\n" + "="*60)
    print("EMBEDDING CONSISTENCY TEST")
    print("="*60)
    
    consistency_text = "This is a consistency test for embeddings."
    
    print(f"Text: '{consistency_text}'")
    
    # Generate embeddings multiple times
    embeddings_list = []
    for i in range(3):
        emb = brain._generate_embeddings(judge_node, consistency_text)
        embeddings_list.append(emb)
        print(f"  Run {i+1}: norm = {np.linalg.norm(emb):.3f}, mean = {np.mean(emb):.3f}")
    
    # Check consistency
    emb1, emb2, emb3 = embeddings_list
    diff_12 = np.linalg.norm(emb1 - emb2)
    diff_23 = np.linalg.norm(emb2 - emb3)
    diff_13 = np.linalg.norm(emb1 - emb3)
    
    print(f"  Embedding differences:")
    print(f"    Run 1 vs Run 2: {diff_12:.6f}")
    print(f"    Run 2 vs Run 3: {diff_23:.6f}")
    print(f"    Run 1 vs Run 3: {diff_13:.6f}")
    
    if max(diff_12, diff_23, diff_13) < 1e-10:
        print(f"  ✅ Embeddings are perfectly consistent")
    elif max(diff_12, diff_23, diff_13) < 0.01:
        print(f"  ✅ Embeddings are highly consistent")
    else:
        print(f"  ⚠️  Embeddings have some variation (may be due to positional transforms)")
    
    # Performance comparison
    print(f"\n" + "="*60)
    print("TOKENIZATION PERFORMANCE")
    print("="*60)
    
    import time
    
    perf_text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeat for longer text
    
    print(f"Performance test text: {len(perf_text)} characters")
    
    # Time tokenization
    start_time = time.time()
    for _ in range(100):  # 100 iterations
        tokens = brain._tokenize_input(perf_text)
    tokenization_time = time.time() - start_time
    
    # Time embedding generation
    start_time = time.time()
    for _ in range(100):  # 100 iterations
        embeddings = brain._generate_embeddings(judge_node, perf_text)
    embedding_time = time.time() - start_time
    
    print(f"  Tokenization (100x): {tokenization_time:.3f}s ({tokenization_time*10:.1f}ms per call)")
    print(f"  Embedding (100x): {embedding_time:.3f}s ({embedding_time*10:.1f}ms per call)")
    print(f"  Total pipeline: {(tokenization_time + embedding_time)*10:.1f}ms per text")
    
    print(f"\n" + "="*65)
    print("🎯 MISTRAL V3 TEKKEN INTEGRATION SUMMARY")
    print("="*65)
    
    # Re-check tokenizer info to ensure we have current state
    final_tokenizer_info = brain.get_tokenizer_info()
    print(f"🔍 Final tokenizer check: {final_tokenizer_info}")

    if 'tekken' in final_tokenizer_info['tokenizer_type']:
        print("✅ SUCCESS: Mistral v3 Tekken tokenizer is active!")
        print(f"  📊 Vocabulary: {final_tokenizer_info['vocab_size']:,} tokens")
        print(f"  📊 Embeddings: {final_tokenizer_info['embedding_dim']} dimensions")
        print("  🔤 Advanced subword tokenization")
        print("  🌍 Multi-language support")
        print("  🚀 State-of-the-art NLP processing")
        print("  🧠 Fully integrated with BrainNexus pipeline")
    else:
        print("⚠️  FALLBACK: Using simple tokenization")
        print("  📦 Install mistral-common for full functionality:")
        print("     pip install mistral-common")
        print(f"  📊 Current vocabulary: {final_tokenizer_info['vocab_size']} tokens")
        print(f"  📊 Current embeddings: {final_tokenizer_info['embedding_dim']} dimensions")
    
    print(f"\n🔗 Ready for integration with:")
    print("  🎓 SegmentLearning for training")
    print("  🧠 Multidimensional pipeline processing")
    print("  🤖 Advanced AI task execution")
    
    return True

if __name__ == "__main__":
    try:
        test_mistral_tokenizer()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
