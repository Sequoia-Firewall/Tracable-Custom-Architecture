"""
BrainNexus Output Configuration Examples

This file demonstrates how to configure BrainNexus for different use cases:
1. Classification tasks
2. LLM token prediction
3. Custom label mapping
4. Top-k predictions
"""

from brainNexus import BrainNexus
import numpy as np

def example_classification():
    """Standard multi-class classification example"""
    print("=== Multi-Class Classification Example ===")
    
    output_config = {
        'type': 'classification',
        'num_classes': 5,
        'class_labels': ['cat', 'dog', 'bird', 'fish', 'hamster'],
        'output_format': 'label',
        'confidence_threshold': 0.6,
        'return_top_k': 3
    }
    
    brain = BrainNexus(dimensions=4, demo=True, output_config=output_config)
    node_map = brain.initialize_brain()
    
    # Run prediction
    result = brain.run("classify this animal", trace_execution=True)
    
    print(f"Predicted class: {result.get('prediction_label', 'N/A')}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'top_k_predictions' in result:
        print("\nTop 3 predictions:")
        for pred in result['top_k_predictions']:
            print(f"  {pred['rank']}. {pred.get('label', f'class_{pred['index']}')} ({pred['confidence']:.3f})")
    
    return brain, result

def example_llm_tokens():
    """LLM token prediction example"""
    print("\n=== LLM Token Prediction Example ===")
    
    # Simple vocabulary mapping for demonstration
    vocab_mapping = {
        0: '<pad>', 1: '<unk>', 2: 'the', 3: 'and', 4: 'is',
        5: 'to', 6: 'a', 7: 'in', 8: 'it', 9: 'you',
        10: 'that', 11: 'he', 12: 'was', 13: 'for', 14: 'on'
    }
    
    output_config = {
        'type': 'tokens',
        'num_classes': 15,
        'vocab_mapping': vocab_mapping,
        'output_format': 'token_id',
        'confidence_threshold': 0.5,
        'return_top_k': 5
    }
    
    brain = BrainNexus(dimensions=4, demo=True, output_config=output_config)
    node_map = brain.initialize_brain()
    
    # Run next token prediction
    result = brain.run("predict next token", trace_execution=True)
    
    print(f"Predicted token: {result.get('predicted_token', 'N/A')}")
    print(f"Token ID: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'top_k_predictions' in result:
        print("\nTop 5 token predictions:")
        for pred in result['top_k_predictions']:
            token = pred.get('token', f'id_{pred['index']}')
            print(f"  {pred['rank']}. {token} ({pred['confidence']:.3f})")
    
    return brain, result

def example_custom_labels():
    """Custom label mapping example"""
    print("\n=== Custom Label Mapping Example ===")
    
    output_config = {
        'type': 'custom',
        'num_classes': 7,
        'class_labels': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'output_format': 'label',
        'confidence_threshold': 0.4,
        'return_top_k': 3
    }
    
    brain = BrainNexus(dimensions=4, demo=True, output_config=output_config)
    node_map = brain.initialize_brain()
    
    # Run day prediction
    result = brain.run("what day is it", trace_execution=True)
    
    print(f"Predicted day: {result.get('prediction_label', 'N/A')}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'top_k_predictions' in result:
        print("\nTop 3 day predictions:")
        for pred in result['top_k_predictions']:
            print(f"  {pred['rank']}. {pred.get('label', f'day_{pred['index']}')} ({pred['confidence']:.3f})")
    
    return brain, result

def example_probability_distribution():
    """Raw probability distribution example"""
    print("\n=== Probability Distribution Example ===")
    
    output_config = {
        'type': 'classification',
        'num_classes': 4,
        'class_labels': ['North', 'South', 'East', 'West'],
        'output_format': 'probability_dist',
        'confidence_threshold': 0.3,
        'return_top_k': 4,
        'normalize_probabilities': True
    }
    
    brain = BrainNexus(dimensions=4, demo=True, output_config=output_config)
    node_map = brain.initialize_brain()
    
    # Run direction prediction
    result = brain.run("which direction", trace_execution=True)
    
    print(f"Primary prediction: {result['prediction']} ({result.get('prediction_label', 'N/A')})")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Full probability distribution: {[f'{p:.3f}' for p in result['probabilities']]}")
    
    if 'top_k_predictions' in result:
        print("\nAll direction probabilities:")
        for pred in result['top_k_predictions']:
            label = pred.get('label', f'direction_{pred['index']}')
            print(f"  {pred['rank']}. {label}: {pred['confidence']:.3f}")
    
    return brain, result

def example_large_vocabulary():
    """Large vocabulary example (e.g., for language models)"""
    print("\n=== Large Vocabulary Example ===")
    
    # Create a larger vocabulary (e.g., common English words)
    vocab_size = 1000
    vocab_mapping = {i: f"word_{i:04d}" for i in range(vocab_size)}
    
    # Add some common words
    common_words = ['the', 'and', 'is', 'to', 'a', 'in', 'it', 'you', 'that', 'he', 
                   'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at']
    for i, word in enumerate(common_words):
        vocab_mapping[i] = word
    
    output_config = {
        'type': 'tokens',
        'num_classes': vocab_size,
        'vocab_mapping': vocab_mapping,
        'output_format': 'token_id',
        'confidence_threshold': 0.1,  # Lower threshold for large vocab
        'return_top_k': 10
    }
    
    brain = BrainNexus(dimensions=4, demo=False, output_config=output_config)  # demo=False for large vocab
    print(f"Initializing brain with {vocab_size} vocabulary...")
    node_map = brain.initialize_brain()
    
    # Run prediction
    result = brain.run("large vocabulary prediction", trace_execution=False)
    
    print(f"Predicted token: {result.get('predicted_token', 'N/A')}")
    print(f"Token ID: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'top_k_predictions' in result:
        print("\nTop 10 predictions:")
        for pred in result['top_k_predictions'][:5]:  # Show top 5
            token = pred.get('token', f'id_{pred['index']}')
            print(f"  {pred['rank']}. {token} ({pred['confidence']:.3f})")
        print("  ...")
    
    return brain, result

def main():
    """Run all examples to demonstrate different configurations"""
    
    print("BrainNexus Configurable Output System Examples")
    print("=" * 50)
    
    # Run examples
    examples = [
        example_classification,
        example_llm_tokens,
        example_custom_labels,
        example_probability_distribution,
        # example_large_vocabulary  # Uncomment to test large vocabulary
    ]
    
    results = []
    for example_func in examples:
        try:
            brain, result = example_func()
            results.append((example_func.__name__, result))
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
            print("-" * 50)
    
    # Summary
    print("\n=== CONFIGURATION SUMMARY ===")
    print("The BrainNexus output system supports:")
    print("• Multiple output types: 'classification', 'tokens', 'custom'")
    print("• Flexible output formats: 'index', 'label', 'token_id', 'probability_dist'")
    print("• Configurable number of classes/vocabulary size")
    print("• Top-k predictions with confidence scores")
    print("• Custom label and token mappings")
    print("• Adjustable confidence thresholds")
    
    return results

if __name__ == "__main__":
    results = main()
