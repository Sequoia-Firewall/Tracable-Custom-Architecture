"""
BrainNexus Output Configuration Guide

This guide explains how to configure BrainNexus for different use cases
by customizing the output system.
"""

# ===== CONFIGURATION OPTIONS =====

OUTPUT_CONFIG_SCHEMA = {
    # === Core Configuration ===
    'type': {
        'description': 'Type of output system',
        'options': ['classification', 'tokens', 'custom'],
        'default': 'classification',
        'examples': {
            'classification': 'Standard multi-class classification (e.g., image classification)',
            'tokens': 'Token prediction for language models (e.g., next word prediction)',
            'custom': 'Custom labeling system (e.g., day of week, sentiment analysis)'
        }
    },
    
    'num_classes': {
        'description': 'Number of output classes/tokens in vocabulary',
        'type': 'int',
        'default': 10,
        'range': '1 to unlimited',
        'examples': {
            'small': 10,      # Simple classification
            'medium': 100,    # More complex classification
            'large': 50000,   # Full language model vocabulary
            'xlarge': 100000  # Large language model vocabulary
        }
    },
    
    'output_format': {
        'description': 'Format of the returned prediction',
        'options': ['index', 'label', 'token_id', 'probability_dist'],
        'default': 'index',
        'examples': {
            'index': 'Return class index (0, 1, 2, ...)',
            'label': 'Return human-readable label ("cat", "dog", "bird")',
            'token_id': 'Return token with ID mapping ("the", "and", "is")',
            'probability_dist': 'Focus on full probability distribution'
        }
    },
    
    # === Label/Token Mapping ===
    'class_labels': {
        'description': 'List of human-readable labels for each class',
        'type': 'list[str]',
        'default': 'Auto-generated as ["class_0", "class_1", ...]',
        'requirement': 'Length must match num_classes',
        'examples': {
            'animals': ['cat', 'dog', 'bird', 'fish', 'hamster'],
            'sentiment': ['negative', 'neutral', 'positive'],
            'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        }
    },
    
    'vocab_mapping': {
        'description': 'Dictionary mapping class indices to tokens (for type="tokens")',
        'type': 'dict[int, str]',
        'default': 'None (auto-generated for tokens type)',
        'requirement': 'Keys should cover range 0 to num_classes-1',
        'examples': {
            'simple': {0: '<pad>', 1: '<unk>', 2: 'the', 3: 'and', 4: 'is'},
            'common_words': {0: 'the', 1: 'and', 2: 'is', 3: 'to', 4: 'a', 5: 'in'},
            'full_vocab': 'Load from tokenizer vocabulary file'
        }
    },
    
    # === Prediction Behavior ===
    'confidence_threshold': {
        'description': 'Minimum confidence required for accepting prediction',
        'type': 'float',
        'default': 0.7,
        'range': '0.0 to 1.0',
        'examples': {
            'strict': 0.9,      # Very confident predictions only
            'balanced': 0.7,    # Moderately confident predictions
            'permissive': 0.3,  # Accept lower confidence predictions
            'exploratory': 0.1  # Accept almost any prediction
        }
    },
    
    'return_top_k': {
        'description': 'Number of top predictions to return (ranked by confidence)',
        'type': 'int',
        'default': 1,
        'range': '1 to num_classes',
        'examples': {
            'single': 1,     # Just the best prediction
            'few': 3,        # Top 3 predictions
            'many': 10,      # Top 10 predictions
            'all': 'num_classes'  # All predictions ranked
        }
    },
    
    'normalize_probabilities': {
        'description': 'Whether to normalize probability distributions to sum to 1.0',
        'type': 'bool',
        'default': True,
        'options': [True, False]
    }
}

# ===== USE CASE TEMPLATES =====

USE_CASE_TEMPLATES = {
    
    # === Image Classification ===
    'image_classification': {
        'description': 'Standard image classification (e.g., CIFAR-10, ImageNet)',
        'config': {
            'type': 'classification',
            'num_classes': 10,  # Adjust for your dataset
            'class_labels': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck'],
            'output_format': 'label',
            'confidence_threshold': 0.7,
            'return_top_k': 3
        }
    },
    
    # === Sentiment Analysis ===
    'sentiment_analysis': {
        'description': 'Text sentiment classification',
        'config': {
            'type': 'classification',
            'num_classes': 3,
            'class_labels': ['negative', 'neutral', 'positive'],
            'output_format': 'label',
            'confidence_threshold': 0.6,
            'return_top_k': 3
        }
    },
    
    # === Language Model (Small) ===
    'language_model_small': {
        'description': 'Small language model (e.g., GPT-2 small)',
        'config': {
            'type': 'tokens',
            'num_classes': 50257,  # GPT-2 vocabulary size
            'vocab_mapping': 'Load from tokenizer',  # Would need actual tokenizer
            'output_format': 'token_id',
            'confidence_threshold': 0.1,  # Lower threshold for large vocab
            'return_top_k': 10
        }
    },
    
    # === Question Answering ===
    'question_answering': {
        'description': 'Multiple choice question answering',
        'config': {
            'type': 'classification',
            'num_classes': 4,
            'class_labels': ['A', 'B', 'C', 'D'],
            'output_format': 'label',
            'confidence_threshold': 0.5,
            'return_top_k': 4
        }
    },
    
    # === Named Entity Recognition ===
    'named_entity_recognition': {
        'description': 'Token-level NER classification',
        'config': {
            'type': 'classification',
            'num_classes': 9,
            'class_labels': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 
                           'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
            'output_format': 'label',
            'confidence_threshold': 0.6,
            'return_top_k': 3
        }
    },
    
    # === Time Series Classification ===
    'time_series_classification': {
        'description': 'Classify time series patterns',
        'config': {
            'type': 'classification',
            'num_classes': 5,
            'class_labels': ['trend_up', 'trend_down', 'stable', 'volatile', 'seasonal'],
            'output_format': 'label',
            'confidence_threshold': 0.65,
            'return_top_k': 3
        }
    },
    
    # === Custom Domain Example ===
    'medical_diagnosis': {
        'description': 'Medical condition classification',
        'config': {
            'type': 'classification',
            'num_classes': 6,
            'class_labels': ['healthy', 'diabetes', 'hypertension', 'heart_disease', 
                           'respiratory', 'other'],
            'output_format': 'label',
            'confidence_threshold': 0.8,  # High threshold for medical
            'return_top_k': 3
        }
    }
}

# ===== IMPLEMENTATION EXAMPLES =====

from typing import Optional, Dict, Any, List, Tuple

def create_brain_for_use_case(use_case_name: str, custom_config: Optional[Dict[str, Any]] = None):
    """
    Create a BrainNexus instance configured for a specific use case.
    
    Args:
        use_case_name: Name of the use case template to use
        custom_config: Additional configuration to override template
        
    Returns:
        Configured BrainNexus instance
    """
    if use_case_name not in USE_CASE_TEMPLATES:
        available = list(USE_CASE_TEMPLATES.keys())
        raise ValueError(f"Unknown use case '{use_case_name}'. Available: {available}")
    
    # Get template configuration
    template = USE_CASE_TEMPLATES[use_case_name]
    output_config = template['config'].copy()
    
    # Apply custom overrides
    if custom_config:
        output_config.update(custom_config)
    
    # Import and create BrainNexus
    from brainNexus import BrainNexus
    
    brain = BrainNexus(
        dimensions=4,
        demo=True,  # Set to False for production
        output_config=output_config
    )
    
    print(f"Created BrainNexus for use case: {use_case_name}")
    print(f"Description: {template['description']}")
    print(f"Configuration: {output_config}")
    
    return brain

# ===== VALIDATION HELPERS =====

def validate_output_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate an output configuration dictionary.
    
    Args:
        config: Output configuration to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields and types
    if 'num_classes' in config:
        if not isinstance(config['num_classes'], int) or config['num_classes'] < 1:
            errors.append("num_classes must be a positive integer")
    
    if 'confidence_threshold' in config:
        if not (0.0 <= config['confidence_threshold'] <= 1.0):
            errors.append("confidence_threshold must be between 0.0 and 1.0")
    
    if 'return_top_k' in config:
        num_classes = config.get('num_classes', 10)
        if config['return_top_k'] > num_classes:
            errors.append("return_top_k cannot exceed num_classes")
    
    # Check class_labels consistency
    if 'class_labels' in config and 'num_classes' in config:
        if len(config['class_labels']) != config['num_classes']:
            errors.append("Length of class_labels must match num_classes")
    
    # Check vocab_mapping for tokens type
    if config.get('type') == 'tokens':
        if 'vocab_mapping' in config and config['vocab_mapping']:
            num_classes = config.get('num_classes', 10)
            vocab_keys = set(config['vocab_mapping'].keys())
            expected_keys = set(range(num_classes))
            if not expected_keys.issubset(vocab_keys):
                missing = expected_keys - vocab_keys
                errors.append(f"vocab_mapping missing keys for indices: {missing}")
    
    # Check valid enum values
    valid_types = ['classification', 'tokens', 'custom']
    if 'type' in config and config['type'] not in valid_types:
        errors.append(f"type must be one of {valid_types}")
    
    valid_formats = ['index', 'label', 'token_id', 'probability_dist']
    if 'output_format' in config and config['output_format'] not in valid_formats:
        errors.append(f"output_format must be one of {valid_formats}")
    
    return len(errors) == 0, errors

# ===== MIGRATION HELPERS =====

def migrate_from_fixed_classification(num_classes: int = 10) -> Dict[str, Any]:
    """
    Create configuration for migrating from the old fixed 10-class system.
    
    Args:
        num_classes: Number of classes in your classification task
        
    Returns:
        Output configuration dictionary
    """
    return {
        'type': 'classification',
        'num_classes': num_classes,
        'class_labels': [f'class_{i}' for i in range(num_classes)],
        'output_format': 'index',  # Keep same as before
        'confidence_threshold': 0.7,
        'return_top_k': 1
    }

def create_token_config_from_tokenizer(tokenizer, max_vocab_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Create token configuration from a tokenizer object.
    
    Args:
        tokenizer: Tokenizer object with vocab attribute
        max_vocab_size: Limit vocabulary size (optional)
        
    Returns:
        Output configuration dictionary
    """
    # Extract vocabulary
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab
    else:
        raise ValueError("Tokenizer must have 'vocab' or 'get_vocab()' attribute")
    
    # Reverse mapping (token -> id becomes id -> token)
    vocab_mapping = {v: k for k, v in vocab.items()}
    
    # Limit vocabulary size if requested
    if max_vocab_size and len(vocab_mapping) > max_vocab_size:
        vocab_mapping = {k: v for k, v in vocab_mapping.items() if k < max_vocab_size}
    
    return {
        'type': 'tokens',
        'num_classes': len(vocab_mapping),
        'vocab_mapping': vocab_mapping,
        'output_format': 'token_id',
        'confidence_threshold': 0.1,  # Lower for large vocab
        'return_top_k': 10
    }
