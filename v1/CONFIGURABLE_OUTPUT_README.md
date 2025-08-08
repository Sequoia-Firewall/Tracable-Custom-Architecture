# BrainNexus Configurable Output System

## Overview

The BrainNexus main interface has been enhanced with a **configurable output system** that allows you to customize the neural network for different use cases including classification, language modeling, and custom prediction tasks.

## 🚀 Quick Start

1. **Run the main interface:**
   ```bash
   python main.py
   ```

2. **Choose initialization option:**
   - Option 1: Demo Mode (faster, smaller network)
   - Option 2: Full Mode (larger, production-ready network)

3. **Configure output system:**
   - When prompted "Configure custom output system? (y/n)", enter `y`
   - Follow the interactive prompts to set up your configuration

## 🔧 Configuration Options

### Output Types

#### 1. **Classification** (`type: 'classification'`)
Perfect for image classification, sentiment analysis, etc.
- **Use cases:** Image recognition, text classification, medical diagnosis
- **Returns:** Class indices or human-readable labels
- **Example:** Classifying images as 'cat', 'dog', 'bird'

#### 2. **Token Prediction** (`type: 'tokens'`)
Designed for language models and sequence prediction.
- **Use cases:** Next-word prediction, language generation, translation
- **Returns:** Token IDs or token strings from vocabulary
- **Example:** Predicting next word as 'the', 'and', 'is'

#### 3. **Custom Labels** (`type: 'custom'`)
For domain-specific applications with custom categories.
- **Use cases:** Days of week, custom categories, specialized domains
- **Returns:** Custom labels or indices
- **Example:** Predicting day as 'Monday', 'Tuesday', 'Wednesday'

### Configuration Parameters

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `num_classes` | Number of output classes/vocabulary size | 1 to unlimited | 10 |
| `class_labels` | Human-readable labels for classes | List of strings | Auto-generated |
| `vocab_mapping` | Token ID to string mapping (for tokens) | Dict {id: token} | Auto-generated |
| `output_format` | Format of returned predictions | 'index', 'label', 'token_id', 'probability_dist' | 'index' |
| `confidence_threshold` | Minimum confidence for predictions | 0.0 to 1.0 | 0.7 |
| `return_top_k` | Number of top predictions to return | 1 to num_classes | 1 |

## 📋 Interactive Configuration Examples

### Example 1: Image Classification
```
Configure custom output system? (y/n): y

1. Choose output type: 1 (Classification)
2. Number of classes: 5
3. Class labels: cat,dog,bird,fish,hamster
4. Output format: 1 (label)
5. Confidence threshold: 0.7
6. Top-k predictions: 3
```

**Result:** Returns predictions like:
```python
{
    'prediction': 2,
    'prediction_label': 'bird',
    'confidence': 0.85,
    'top_k_predictions': [
        {'rank': 1, 'label': 'bird', 'confidence': 0.85},
        {'rank': 2, 'label': 'cat', 'confidence': 0.10},
        {'rank': 3, 'label': 'dog', 'confidence': 0.05}
    ]
}
```

### Example 2: Language Model
```
Configure custom output system? (y/n): y

1. Choose output type: 2 (Token Prediction)
2. Vocabulary size: 1000
3. Token vocabulary: y (use preset)
4. Output format: 1 (token_id)
5. Confidence threshold: 0.5
6. Top-k predictions: 5
```

**Result:** Returns predictions like:
```python
{
    'prediction': 4,
    'predicted_token': 'the',
    'confidence': 0.65,
    'top_k_predictions': [
        {'rank': 1, 'token': 'the', 'confidence': 0.65},
        {'rank': 2, 'token': 'and', 'confidence': 0.15},
        {'rank': 3, 'token': 'is', 'confidence': 0.10}
    ]
}
```

### Example 3: Sentiment Analysis
```
Configure custom output system? (y/n): y

1. Choose output type: 1 (Classification)
2. Number of classes: 3
3. Class labels: negative,neutral,positive
4. Output format: 1 (label)
5. Confidence threshold: 0.6
6. Top-k predictions: 3
```

## 🎯 Pre-built Configurations

For quick setup, you can also use pre-built configurations in your code:

```python
# Image Classification
config = {
    'type': 'classification',
    'num_classes': 10,
    'class_labels': ['class_0', 'class_1', ..., 'class_9'],
    'output_format': 'label',
    'confidence_threshold': 0.7,
    'return_top_k': 3
}

# Language Model
config = {
    'type': 'tokens',
    'num_classes': 50000,
    'vocab_mapping': load_vocabulary(),  # Your vocabulary
    'output_format': 'token_id',
    'confidence_threshold': 0.1,
    'return_top_k': 10
}

# Custom Domain
config = {
    'type': 'custom',
    'num_classes': 7,
    'class_labels': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'output_format': 'label',
    'confidence_threshold': 0.5,
    'return_top_k': 3
}
```

## 🛠️ Advanced Features

### 1. **Top-K Predictions**
Get multiple ranked predictions with confidence scores:
```python
result = brain.run("input")
for pred in result['top_k_predictions']:
    print(f"{pred['rank']}. {pred['label']} ({pred['confidence']:.3f})")
```

### 2. **Confidence Thresholding**
Set minimum confidence levels for accepting predictions:
- High threshold (0.8-0.9): Very confident predictions only
- Medium threshold (0.6-0.7): Balanced confidence
- Low threshold (0.1-0.3): Accept uncertain predictions

### 3. **Multiple Output Formats**
- **Index:** Returns class/token indices (0, 1, 2, ...)
- **Label:** Returns human-readable labels ('cat', 'dog', 'bird')
- **Token ID:** Returns token strings ('the', 'and', 'is')
- **Probability Distribution:** Focus on full probability arrays

### 4. **Custom Vocabularies**
For language models, provide your own vocabulary mapping:
```python
vocab = {
    0: '<pad>', 1: '<unk>', 2: '<start>', 3: '<end>',
    4: 'hello', 5: 'world', 6: 'python', 7: 'brain'
}
```

## 🧪 Testing

Run the test script to see all configurations in action:
```bash
python test_main_config.py
```

This demonstrates:
- Image classification with custom labels
- Token prediction with vocabulary
- Sentiment analysis configuration
- Automatic brain initialization
- Sample predictions with each configuration

## 🔄 Migration from Old System

If you have existing code using the old fixed 10-class system:

**Old way:**
```python
brain = BrainNexus(dimensions=4, demo=True)
```

**New way (backward compatible):**
```python
# Same behavior as before
brain = BrainNexus(dimensions=4, demo=True)

# Or explicitly configure
config = {
    'type': 'classification',
    'num_classes': 10,
    'output_format': 'index'  # Same as old system
}
brain = BrainNexus(dimensions=4, demo=True, output_config=config)
```

## 📁 Files Added/Modified

### New Files:
- `output_config_examples.py` - Comprehensive usage examples
- `output_config_guide.py` - Complete documentation and helpers
- `demo_configurable_output.py` - Simple demonstration script
- `test_main_config.py` - Test script for main.py integration

### Modified Files:
- `main.py` - Added interactive configuration system
- `brainNexus.py` - Added output configuration support
- `computations.py` - Updated Handler class for configurable outputs
- `BrainNexusLearn.py` - Added output_config parameter

## 🎓 Learning Resources

1. **Start Simple:** Use default configuration first
2. **Experiment:** Try different output types with small datasets
3. **Scale Up:** Increase vocabulary/classes for production use
4. **Monitor:** Watch confidence scores to tune thresholds
5. **Iterate:** Adjust configuration based on results

## 🚨 Tips & Best Practices

- **Classification:** Use labels for interpretability, indices for speed
- **Language Models:** Start with small vocabularies, expand gradually
- **Confidence:** Higher thresholds for critical applications
- **Top-K:** Use for uncertainty estimation and debugging
- **Custom Labels:** Keep names short and descriptive

The configurable output system makes BrainNexus adaptable to virtually any prediction task while maintaining the power of spatial neural organization!
