#!/usr/bin/env python3
"""
BrainNexus Training Data Structure Examples
==========================================
This file shows the exact data structures required for training different types of data
with the enhanced BrainNexus system.

Each example includes:
1. The expected data format
2. Sample data
3. How to save/load the data
4. How to train with the data
"""

import json
import pickle
import random
from typing import List, Dict, Any, Tuple

def classification_data_structure():
    """
    CLASSIFICATION DATA STRUCTURE
    ============================
    For standard classification tasks (image classification, sentiment analysis, etc.)
    """
    
    print("📊 CLASSIFICATION DATA STRUCTURE")
    print("="*50)
    
    # Basic structure for classification
    sample_input = {
        'features': [1.5, -0.5, 0.8, -1.2, 0.3, 1.1, -0.7, 0.4, -0.2, 0.9],  # Feature vector (required)
        'sequence_length': 12,                                                   # Length info (optional)
        'metadata': {                                                           # Additional info (optional)
            'sample_id': 0,
            'category': 'A',
            'source': 'synthetic'
        }
    }
    
    sample_target = 2  # Class index (0, 1, 2, etc.)
    
    # Complete training data structure
    classification_data = {
        'training_data': [
            # Each item is a tuple: (input_dict, target_class)
            (sample_input, sample_target),
            
            # More examples...
            ({
                'features': [2.1, 0.3, -0.6, 1.8, -0.9, 0.2, 1.4, -0.8, 0.7, -0.3],
                'sequence_length': 15,
                'metadata': {'sample_id': 1, 'category': 'B'}
            }, 0),
            
            ({
                'features': [-0.8, 1.2, 0.5, -1.5, 0.9, -0.4, 0.7, 1.1, -0.6, 0.2],
                'sequence_length': 8,
                'metadata': {'sample_id': 2, 'category': 'C'}
            }, 1),
        ],
        
        'validation_data': [
            # Same structure as training_data but for validation
            ({
                'features': [0.9, -1.1, 1.3, 0.4, -0.7, 1.8, -0.2, 0.6, 0.8, -1.4],
                'sequence_length': 10,
                'metadata': {'sample_id': 100, 'category': 'A'}
            }, 2),
        ]
    }
    
    print("Structure:")
    print("  Root keys: 'training_data', 'validation_data'")
    print("  Each data point: (input_dict, target_class)")
    print("  Input dict required keys: 'features' (list of floats)")
    print("  Input dict optional keys: 'sequence_length', 'metadata'")
    print("  Target: integer class index (0, 1, 2, ...)")
    
    print(f"\nSample input: {sample_input}")
    print(f"Sample target: {sample_target}")
    
    return classification_data

def conversation_data_structure():
    """
    CONVERSATION DATA STRUCTURE
    ==========================
    For training on conversation/dialogue data
    """
    
    print("\n💬 CONVERSATION DATA STRUCTURE")
    print("="*50)
    
    # Single conversation structure
    sample_conversation = [
        {"role": "user", "content": "Hello, how are you today?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you?"},
        {"role": "user", "content": "Can you tell me about the weather?"},
        {"role": "assistant", "content": "I'd be happy to discuss weather! What specifically would you like to know?"}
    ]
    
    # Complete conversation data structure
    conversation_data = {
        'conversations': [
            # Each conversation is a list of message dictionaries
            sample_conversation,
            
            # Another conversation
            [
                {"role": "user", "content": "I need help with programming."},
                {"role": "assistant", "content": "Great! I'd love to help with programming. What language are you working with?"},
                {"role": "user", "content": "I'm learning Python."},
                {"role": "assistant", "content": "Python is excellent for beginners! What aspect would you like to explore?"}
            ],
            
            # Short conversation
            [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        ],
        
        'validation_conversations': [
            # Same structure for validation
            [
                {"role": "user", "content": "Hi there!"},
                {"role": "assistant", "content": "Hello! How can I assist you today?"}
            ]
        ]
    }
    
    print("Structure:")
    print("  Root keys: 'conversations', 'validation_conversations' (optional)")
    print("  Each conversation: list of message dictionaries")
    print("  Each message: {'role': 'user'|'assistant', 'content': 'text'}")
    print("  Minimum 2 messages per conversation")
    
    print(f"\nSample conversation: {sample_conversation}")
    
    return conversation_data

def sequence_data_structure():
    """
    SEQUENCE DATA STRUCTURE
    =======================
    For training on sequence prediction (time series, patterns, etc.)
    """
    
    print("\n📈 SEQUENCE DATA STRUCTURE")
    print("="*50)
    
    # Various types of sequences
    numerical_sequence = [1, 3, 5, 7, 9, 11, 13, 15]  # Odd numbers
    categorical_sequence = ["A", "B", "A", "B", "A", "B"]  # Pattern
    mixed_sequence = [1, 2.5, 4, 5.5, 7, 8.5]  # Mixed numbers
    
    # Complete sequence data structure
    sequence_data = {
        'sequences': [
            # Each sequence is a list of elements
            numerical_sequence,
            categorical_sequence,
            mixed_sequence,
            
            # More examples
            [2, 4, 6, 8, 10, 12, 14, 16],  # Even numbers
            [1, 1, 2, 3, 5, 8, 13, 21],    # Fibonacci
            ["X", "Y", "Z", "X", "Y", "Z"], # ABC pattern
            [5, 10, 15, 20, 25, 30],       # Multiples of 5
            [1, 4, 9, 16, 25, 36, 49]      # Squares
        ],
        
        'validation_sequences': [
            # Same structure for validation
            [3, 6, 9, 12, 15],           # Multiples of 3
            ["P", "Q", "P", "Q", "P"]    # Simple pattern
        ]
    }
    
    print("Structure:")
    print("  Root keys: 'sequences', 'validation_sequences' (optional)")
    print("  Each sequence: list of elements (numbers, strings, or mixed)")
    print("  Minimum 3 elements per sequence for meaningful prediction")
    print("  Elements can be: int, float, str, or mixed types")
    
    print(f"\nSample sequences:")
    print(f"  Numerical: {numerical_sequence}")
    print(f"  Categorical: {categorical_sequence}")
    print(f"  Mixed: {mixed_sequence}")
    
    return sequence_data

def text_generation_data_structure():
    """
    TEXT GENERATION DATA STRUCTURE
    ==============================
    For training on text generation tasks
    """
    
    print("\n📝 TEXT GENERATION DATA STRUCTURE")
    print("="*50)
    
    # Text generation uses similar structure to classification
    # but with text-specific preprocessing
    
    sample_text_input = {
        'text': "The quick brown fox",           # Raw text input
        'features': [4, 15, 3.75, 0, 0, 0, 0, 0, 0, 0],  # Extracted features
        'context_length': 4,                    # Number of words/tokens
        'metadata': {
            'genre': 'prose',
            'language': 'english'
        }
    }
    
    sample_text_target = "jumps"  # Next word/token
    
    text_generation_data = {
        'training_data': [
            (sample_text_input, sample_text_target),
            
            # More examples
            ({
                'text': "I love programming in",
                'features': [4, 18, 4.5, 0, 0, 0, 0, 0, 0, 0],
                'context_length': 4,
                'metadata': {'genre': 'technical'}
            }, "Python"),
            
            ({
                'text': "The weather is very",
                'features': [4, 17, 4.25, 0, 0, 0, 0, 0, 0, 0],
                'context_length': 4,
                'metadata': {'genre': 'conversation'}
            }, "nice"),
        ],
        
        'validation_data': [
            ({
                'text': "Good morning, how are",
                'features': [4, 19, 4.75, 0, 0, 0, 0, 0, 0, 0],
                'context_length': 4,
                'metadata': {'genre': 'greeting'}
            }, "you"),
        ]
    }
    
    print("Structure:")
    print("  Root keys: 'training_data', 'validation_data'")
    print("  Each data point: (input_dict, target_text)")
    print("  Input dict required keys: 'text', 'features'")
    print("  Input dict optional keys: 'context_length', 'metadata'")
    print("  Target: string (next word/token)")
    
    print(f"\nSample input: {sample_text_input}")
    print(f"Sample target: '{sample_text_target}'")
    
    return text_generation_data

def unsupervised_data_structure():
    """
    UNSUPERVISED DATA STRUCTURE
    ===========================
    For unsupervised learning (clustering, autoencoders, etc.)
    """
    
    print("\n🔬 UNSUPERVISED DATA STRUCTURE")
    print("="*50)
    
    # Unsupervised data - no targets needed
    sample_unsupervised_input = {
        'features': [1.5, -0.5, 0.8, -1.2, 0.3, 1.1, -0.7, 0.4, -0.2, 0.9],
        'sample_id': 0,
        'metadata': {
            'source': 'sensor_reading',
            'timestamp': '2025-01-01T12:00:00'
        }
    }
    
    # Unsupervised data is just a list of input dictionaries
    unsupervised_data = [
        sample_unsupervised_input,
        
        # More examples
        {
            'features': [2.1, 0.3, -0.6, 1.8, -0.9, 0.2, 1.4, -0.8, 0.7, -0.3],
            'sample_id': 1,
            'metadata': {'source': 'sensor_reading', 'timestamp': '2025-01-01T12:01:00'}
        },
        
        {
            'features': [-0.8, 1.2, 0.5, -1.5, 0.9, -0.4, 0.7, 1.1, -0.6, 0.2],
            'sample_id': 2,
            'metadata': {'source': 'sensor_reading', 'timestamp': '2025-01-01T12:02:00'}
        }
    ]
    
    print("Structure:")
    print("  Root: list of input dictionaries (no targets)")
    print("  Each input: dictionary with 'features' key (required)")
    print("  Optional keys: 'sample_id', 'metadata', or any other info")
    print("  No target/label information needed")
    
    print(f"\nSample input: {sample_unsupervised_input}")
    
    return unsupervised_data

def feature_extraction_examples():
    """
    FEATURE EXTRACTION EXAMPLES
    ===========================
    How features are extracted for different data types
    """
    
    print("\n🔧 FEATURE EXTRACTION EXAMPLES")
    print("="*50)
    
    print("1. CONVERSATION FEATURES (automatically extracted):")
    conversation = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help?"}
    ]
    
    # Example features that would be extracted:
    conv_features = [
        2,      # Number of messages
        10,     # Total words
        2,      # Unique speakers
        31,     # Total characters
        1,      # Question count
        1,      # Exclamation count
        15.5,   # Average message length
        0, 0, 0 # Padding to 10 features
    ]
    
    print(f"  Input: {conversation}")
    print(f"  Extracted features: {conv_features}")
    print("  Features: [msg_count, word_count, speakers, chars, questions, exclamations, avg_length, ...]")
    
    print("\n2. SEQUENCE FEATURES (automatically extracted):")
    sequence = [2, 4, 6, 8, 10]
    
    # Example features that would be extracted:
    seq_features = [
        5,    # Length
        6.0,  # Mean
        3.16, # Std dev
        2,    # Min
        10,   # Max
        5,    # Unique values
        5,    # Unique items
        4,    # State transitions
        0, 0  # Padding to 10 features
    ]
    
    print(f"  Input: {sequence}")
    print(f"  Extracted features: {seq_features}")
    print("  Features: [length, mean, std, min, max, unique_vals, unique_items, transitions, ...]")
    
    print("\n3. MANUAL FEATURES (for classification/text generation):")
    print("  You provide the features directly in the 'features' field")
    print("  Should be a list of 10 floats for consistency")
    print("  Example: [1.5, -0.5, 0.8, -1.2, 0.3, 1.1, -0.7, 0.4, -0.2, 0.9]")

def file_formats_and_usage():
    """
    FILE FORMATS AND USAGE EXAMPLES
    ===============================
    How to save, load, and use the data
    """
    
    print("\n💾 FILE FORMATS AND USAGE")
    print("="*50)
    
    # Example usage code
    usage_code = '''
# SAVING DATA TO FILES
import json
import pickle

# 1. Save as JSON (human-readable)
with open('classification_data.json', 'w') as f:
    json.dump(classification_data, f, indent=2)

# 2. Save as Pickle (binary, faster)
with open('conversation_data.pkl', 'wb') as f:
    pickle.dump(conversation_data, f)

# LOADING AND TRAINING
from main import BrainNexusManager

# Initialize manager and brain
manager = BrainNexusManager()
manager.initialize_new_brain(demo_mode=True)

# 1. CLASSIFICATION TRAINING
success = manager.load_training_data('classification_data.json')
if success:
    results = manager.learner.supervised_train(manager.training_data, manager.validation_data)
    print(f"Training completed with {results['final_accuracy']:.3f} accuracy")

# 2. CONVERSATION TRAINING
success = manager.load_conversation_data('conversation_data.json')
if success:
    success = manager.train_on_conversations()
    print(f"Conversation training: {'✓' if success else '✗'}")

# 3. SEQUENCE TRAINING
success = manager.load_sequence_data('sequence_data.json')
if success:
    success = manager.train_on_sequences()
    print(f"Sequence training: {'✓' if success else '✗'}")

# 4. UNSUPERVISED TRAINING (no file loading needed)
unsupervised_data = [...]  # Your unlabeled data
success = manager.train_unsupervised(unsupervised_data, method='clustering')
print(f"Unsupervised training: {'✓' if success else '✗'}")

# 5. HYBRID TRAINING
success = manager.train_hybrid(
    supervised_data=manager.training_data,
    unsupervised_data=unsupervised_data
)
print(f"Hybrid training: {'✓' if success else '✗'}")
'''
    
    print("File formats:")
    print("  • JSON (.json) - Human-readable, good for debugging")
    print("  • Pickle (.pkl) - Binary, faster loading, smaller files")
    print("  • Both formats are supported for all data types")
    
    print("\nUsage example:")
    print(usage_code)

def create_sample_files():
    """
    CREATE SAMPLE FILES
    ===================
    Generate actual sample files for testing
    """
    
    print("\n📁 CREATING SAMPLE FILES")
    print("="*50)
    
    # Generate all data types
    class_data = classification_data_structure()
    conv_data = conversation_data_structure()
    seq_data = sequence_data_structure()
    text_data = text_generation_data_structure()
    unsup_data = unsupervised_data_structure()
    
    # Expand data for more realistic examples
    print("Generating larger datasets...")
    
    # Classification data - 100 samples
    for i in range(97):  # Already have 3
        features = [random.uniform(-2, 2) for _ in range(10)]
        target = random.randint(0, 2)
        class_data['training_data'].append(({
            'features': features,
            'sequence_length': random.randint(5, 20),
            'metadata': {'sample_id': i+3, 'category': ['A', 'B', 'C'][target]}
        }, target))
    
    # Add more validation data
    for i in range(19):  # Already have 1
        features = [random.uniform(-2, 2) for _ in range(10)]
        target = random.randint(0, 2)
        class_data['validation_data'].append(({
            'features': features,
            'sequence_length': random.randint(5, 20),
            'metadata': {'sample_id': i+101, 'category': ['A', 'B', 'C'][target]}
        }, target))
    
    # Save files
    files_created = []
    
    # Save classification data
    with open('sample_classification_data.json', 'w') as f:
        json.dump(class_data, f, indent=2)
    files_created.append('sample_classification_data.json')
    
    # Save conversation data
    with open('sample_conversation_data.json', 'w') as f:
        json.dump(conv_data, f, indent=2)
    files_created.append('sample_conversation_data.json')
    
    # Save sequence data
    with open('sample_sequence_data.json', 'w') as f:
        json.dump(seq_data, f, indent=2)
    files_created.append('sample_sequence_data.json')
    
    # Save text generation data
    with open('sample_text_generation_data.json', 'w') as f:
        json.dump(text_data, f, indent=2)
    files_created.append('sample_text_generation_data.json')
    
    # Save unsupervised data
    with open('sample_unsupervised_data.json', 'w') as f:
        json.dump(unsup_data, f, indent=2)
    files_created.append('sample_unsupervised_data.json')
    
    print(f"✅ Created {len(files_created)} sample files:")
    for filename in files_created:
        print(f"  • {filename}")
    
    return files_created

def validation_checklist():
    """
    VALIDATION CHECKLIST
    ====================
    Common issues and how to avoid them
    """
    
    print("\n✅ VALIDATION CHECKLIST")
    print("="*50)
    
    checklist = [
        "✓ Classification data: Each input has 'features' list",
        "✓ Classification targets: Integer class indices (0, 1, 2, ...)",
        "✓ Conversation data: Each message has 'role' and 'content'",
        "✓ Conversation roles: Only 'user' and 'assistant' supported",
        "✓ Sequence data: Each sequence has at least 3 elements",
        "✓ Sequence elements: Can be numbers, strings, or mixed",
        "✓ Features: List of exactly 10 float values for consistency",
        "✓ File format: Valid JSON or pickle format",
        "✓ Data structure: Correct root keys ('training_data', 'conversations', etc.)",
        "✓ No missing data: All required fields present",
        "✓ Consistent types: Features are floats, targets match expected type"
    ]
    
    print("Before training, verify:")
    for item in checklist:
        print(f"  {item}")
    
    print("\nCommon errors to avoid:")
    print("  ❌ Features as strings instead of floats")
    print("  ❌ Wrong number of features (not 10)")
    print("  ❌ Missing 'features' key in input dictionaries")
    print("  ❌ Conversation messages without 'role' or 'content'")
    print("  ❌ Sequences with fewer than 3 elements")
    print("  ❌ Inconsistent data types within sequences")
    print("  ❌ File format errors (malformed JSON)")

def main():
    """
    Main function to demonstrate all data structures
    """
    
    print("🧠 BrainNexus Training Data Structure Guide")
    print("="*80)
    
    # Show all data structures
    classification_data_structure()
    conversation_data_structure()
    sequence_data_structure()
    text_generation_data_structure()
    unsupervised_data_structure()
    
    # Show feature extraction
    feature_extraction_examples()
    
    # Show file usage
    file_formats_and_usage()
    
    # Create sample files
    files = create_sample_files()
    
    # Show validation checklist
    validation_checklist()
    
    print("\n" + "="*80)
    print("🎉 GUIDE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the data structures above")
    print("2. Use the created sample files to test training")
    print("3. Create your own data following these patterns")
    print("4. Run training with: python simple_sequence_test.py")
    print("\nGenerated sample files:")
    for filename in files:
        print(f"  • {filename}")

if __name__ == "__main__":
    main()
