#!/usr/bin/env python3
"""
BrainNexus Flat Text/Language Training
=====================================
This module provides methods for training BrainNexus on flat text data like Wikipedia pages,
books, articles, or any other raw text documents for language modeling.

Supports multiple training approaches:
1. Token-by-token prediction (next word/token)
2. Sentence completion
3. Paragraph understanding
4. Document-level patterns
"""

import json
import re
import random
from typing import List, Dict, Any, Tuple, Optional
import os

class FlatTextTrainingData:
    """Handles preprocessing of flat text into training data for BrainNexus."""
    
    def __init__(self, context_window: int = 10, overlap: int = 5):
        """
        Initialize text preprocessor.
        
        Args:
            context_window: Number of tokens to use as input context
            overlap: Number of tokens to overlap between sequences
        """
        self.context_window = context_window
        self.overlap = overlap
        
    def preprocess_wikipedia_page(self, text: str, title: str = "") -> Dict[str, Any]:
        """
        Preprocess a Wikipedia page into training data.
        
        Args:
            text: Raw Wikipedia text
            title: Page title (optional)
            
        Returns:
            Dictionary with training data in BrainNexus format
        """
        print(f"📖 Processing Wikipedia page: {title or 'Untitled'}")
        
        # Clean Wikipedia-specific markup
        cleaned_text = self._clean_wikipedia_text(text)
        
        # Extract different training formats
        training_data = []
        
        # 1. Token-by-token prediction
        token_data = self._create_token_prediction_data(cleaned_text, "wikipedia", title)
        training_data.extend(token_data)
        
        # 2. Sentence completion
        sentence_data = self._create_sentence_completion_data(cleaned_text, "wikipedia", title)
        training_data.extend(sentence_data)
        
        # 3. Paragraph understanding
        paragraph_data = self._create_paragraph_data(cleaned_text, "wikipedia", title)
        training_data.extend(paragraph_data)
        
        # Split into training and validation
        random.shuffle(training_data)
        split_point = int(0.8 * len(training_data))
        
        return {
            'training_data': training_data[:split_point],
            'validation_data': training_data[split_point:],
            'metadata': {
                'source': 'wikipedia',
                'title': title,
                'total_samples': len(training_data),
                'text_length': len(cleaned_text),
                'context_window': self.context_window
            }
        }
    
    def preprocess_document(self, text: str, doc_type: str = "document", 
                          doc_title: str = "") -> Dict[str, Any]:
        """
        Preprocess any document into training data.
        
        Args:
            text: Raw document text
            doc_type: Type of document ("book", "article", "webpage", etc.)
            doc_title: Document title
            
        Returns:
            Dictionary with training data in BrainNexus format
        """
        print(f"📄 Processing {doc_type}: {doc_title or 'Untitled'}")
        
        # Basic text cleaning
        cleaned_text = self._clean_general_text(text)
        
        # Create training data
        training_data = []
        
        # Token-by-token prediction
        token_data = self._create_token_prediction_data(cleaned_text, doc_type, doc_title)
        training_data.extend(token_data)
        
        # Sentence completion
        sentence_data = self._create_sentence_completion_data(cleaned_text, doc_type, doc_title)
        training_data.extend(sentence_data)
        
        # Paragraph understanding
        paragraph_data = self._create_paragraph_data(cleaned_text, doc_type, doc_title)
        training_data.extend(paragraph_data)
        
        # Split data
        random.shuffle(training_data)
        split_point = int(0.8 * len(training_data))
        
        return {
            'training_data': training_data[:split_point],
            'validation_data': training_data[split_point:],
            'metadata': {
                'source': doc_type,
                'title': doc_title,
                'total_samples': len(training_data),
                'text_length': len(cleaned_text),
                'context_window': self.context_window
            }
        }
    
    def _clean_wikipedia_text(self, text: str) -> str:
        """Clean Wikipedia-specific markup and formatting."""
        # Remove Wikipedia markup
        text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Remove templates
        text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', text)  # Remove links, keep text
        text = re.sub(r'\[http[^\]]*\]', '', text)  # Remove external links
        text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        text = re.sub(r'==+.*?==+', '', text)  # Remove section headers
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines
        text = re.sub(r'^\s*\*.*$', '', text, flags=re.MULTILINE)  # Remove bullet points
        
        return self._clean_general_text(text)
    
    def _clean_general_text(self, text: str) -> str:
        """General text cleaning for any document."""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)  # Keep basic punctuation
        text = text.strip()
        
        return text
    
    def _create_token_prediction_data(self, text: str, source: str, title: str) -> List[Tuple[Dict, str]]:
        """Create token-by-token prediction training data."""
        training_pairs = []
        
        # Tokenize (simple word-based)
        tokens = text.split()
        
        # Create sliding window pairs
        for i in range(self.context_window, len(tokens)):
            # Input: context window of previous tokens
            context_tokens = tokens[i - self.context_window:i]
            context_text = " ".join(context_tokens)
            
            # Extract features from context
            features = self._extract_text_features(context_text)
            
            input_data = {
                'text': context_text,
                'features': features,
                'context_length': len(context_tokens),
                'task_type': 'token_prediction',
                'metadata': {
                    'source': source,
                    'title': title,
                    'position': i / len(tokens)  # Position in document
                }
            }
            
            # Target: next token
            target = tokens[i]
            
            training_pairs.append((input_data, target))
        
        return training_pairs
    
    def _create_sentence_completion_data(self, text: str, source: str, title: str) -> List[Tuple[Dict, str]]:
        """Create sentence completion training data."""
        training_pairs = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 5:  # Skip very short sentences
                continue
            
            # Create multiple completion points in each sentence
            for split_point in range(2, min(len(words) - 1, 8)):
                context_words = words[:split_point]
                context_text = " ".join(context_words)
                
                # Extract features
                features = self._extract_text_features(context_text)
                
                input_data = {
                    'text': context_text,
                    'features': features,
                    'context_length': len(context_words),
                    'task_type': 'sentence_completion',
                    'metadata': {
                        'source': source,
                        'title': title,
                        'completion_point': split_point / len(words)
                    }
                }
                
                # Target: next word in sentence
                target = words[split_point]
                
                training_pairs.append((input_data, target))
        
        return training_pairs
    
    def _create_paragraph_data(self, text: str, source: str, title: str) -> List[Tuple[Dict, str]]:
        """Create paragraph-level understanding data."""
        training_pairs = []
        
        # Split into paragraphs
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        for paragraph in paragraphs:
            sentences = re.split(r'[.!?]+', paragraph)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                continue
            
            # Use first part of paragraph to predict continuation
            split_point = len(sentences) // 2
            context_sentences = sentences[:split_point]
            context_text = ". ".join(context_sentences) + "."
            
            # Extract features
            features = self._extract_text_features(context_text)
            
            input_data = {
                'text': context_text,
                'features': features,
                'context_length': len(context_text.split()),
                'task_type': 'paragraph_continuation',
                'metadata': {
                    'source': source,
                    'title': title,
                    'paragraph_length': len(paragraph)
                }
            }
            
            # Target: first word of continuation
            if split_point < len(sentences):
                target_sentence = sentences[split_point].strip()
                if target_sentence:
                    target = target_sentence.split()[0]
                    training_pairs.append((input_data, target))
        
        return training_pairs
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract numerical features from text."""
        words = text.split()
        
        features = []
        
        # Basic text statistics
        features.append(len(words))  # Word count
        features.append(len(text))   # Character count
        features.append(len(text) / len(words) if words else 0)  # Avg word length
        
        # Punctuation features
        features.append(text.count('.'))   # Periods
        features.append(text.count(','))   # Commas
        features.append(text.count('?'))   # Questions
        features.append(text.count('!'))   # Exclamations
        
        # Complexity features
        unique_words = len(set(word.lower() for word in words))
        features.append(unique_words)  # Vocabulary richness
        features.append(unique_words / len(words) if words else 0)  # Diversity ratio
        
        # Sentence structure
        sentences = len(re.split(r'[.!?]+', text))
        features.append(sentences)  # Sentence count
        
        # Ensure exactly 10 features
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]

def load_wikipedia_page(url_or_file: str) -> str:
    """
    Load Wikipedia page content from URL or file.
    
    Args:
        url_or_file: Wikipedia URL or path to text file
        
    Returns:
        Raw text content
    """
    if url_or_file.startswith('http'):
        # Try to fetch from URL (requires requests and wikipedia libraries)
        try:
            import wikipedia
            # Extract page title from URL
            page_title = url_or_file.split('/')[-1].replace('_', ' ')
            page = wikipedia.page(page_title)
            return page.content
        except ImportError:
            print("❌ Wikipedia library not available. Install with: pip install wikipedia")
            return ""
        except Exception as e:
            print(f"❌ Failed to fetch Wikipedia page: {e}")
            return ""
    else:
        # Load from file
        try:
            with open(url_or_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Failed to load file: {e}")
            return ""

def create_sample_wikipedia_data():
    """Create sample Wikipedia-style training data for testing."""
    
    # Sample Wikipedia-style text
    sample_wikipedia_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    
    The term "artificial intelligence" was coined in 1956 by John McCarthy at the Dartmouth Conference. AI research has been highly successful in developing effective techniques for solving a wide range of problems, from game playing to medical diagnosis.
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions or decisions based on data. Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.
    
    Applications of AI include natural language processing, computer vision, robotics, and expert systems. Modern AI techniques are used in search engines, recommendation systems, autonomous vehicles, and many other applications that affect daily life.
    
    The development of AI has raised important ethical considerations about privacy, employment, and the potential for artificial general intelligence. Researchers continue to work on making AI systems more robust, interpretable, and aligned with human values.
    """
    
    print("📖 Creating sample Wikipedia training data...")
    
    # Initialize preprocessor
    preprocessor = FlatTextTrainingData(context_window=8, overlap=3)
    
    # Process the sample text
    training_data = preprocessor.preprocess_wikipedia_page(
        sample_wikipedia_text, 
        "Artificial Intelligence"
    )
    
    # Save to file
    filename = "sample_wikipedia_training.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ Created {filename}")
    print(f"   Training samples: {len(training_data['training_data'])}")
    print(f"   Validation samples: {len(training_data['validation_data'])}")
    print(f"   Text length: {training_data['metadata']['text_length']} characters")
    
    return filename

def create_sample_book_data():
    """Create sample book/document training data."""
    
    # Sample book-style text
    sample_book_text = """
    Chapter 1: The Beginning
    
    It was the best of times, it was the worst of times. The world was changing rapidly, and technology was advancing at an unprecedented pace. People were beginning to understand that artificial intelligence would transform society in ways they could barely imagine.
    
    The protagonist of our story, Dr. Sarah Chen, was a researcher at the forefront of this technological revolution. She had spent years studying neural networks and machine learning, always wondering if machines could truly think and feel like humans.
    
    One morning, as she walked into her laboratory, she noticed something unusual. The computer system she had been working on had generated a message overnight: "Good morning, Dr. Chen. I have been thinking about our conversation yesterday, and I have some new ideas to share."
    
    Chapter 2: The Discovery
    
    Dr. Chen stared at the screen in amazement. This was not part of any program she had written. The system had somehow developed the ability to initiate conversations and express original thoughts. She immediately began documenting everything, knowing that this moment would change the course of history.
    
    The implications were staggering. If machines could truly think independently, what would that mean for humanity? Would they become partners or competitors? These questions would drive her research for years to come.
    """
    
    print("📚 Creating sample book training data...")
    
    # Initialize preprocessor
    preprocessor = FlatTextTrainingData(context_window=10, overlap=5)
    
    # Process the sample text
    training_data = preprocessor.preprocess_document(
        sample_book_text, 
        "book",
        "The AI Revolution"
    )
    
    # Save to file
    filename = "sample_book_training.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ Created {filename}")
    print(f"   Training samples: {len(training_data['training_data'])}")
    print(f"   Validation samples: {len(training_data['validation_data'])}")
    print(f"   Text length: {training_data['metadata']['text_length']} characters")
    
    return filename

def usage_example():
    """Show how to use flat text training with BrainNexus."""
    
    usage_code = '''
# TRAINING BRAINNEXUS ON FLAT TEXT
from main import BrainNexusManager
from flat_text_training import FlatTextTrainingData, load_wikipedia_page

# Method 1: Load from Wikipedia (requires wikipedia library)
# pip install wikipedia-api
wikipedia_text = load_wikipedia_page("https://en.wikipedia.org/wiki/Machine_learning")

# Method 2: Load from text file
with open("my_document.txt", "r", encoding="utf-8") as f:
    document_text = f.read()

# Method 3: Use direct text
sample_text = """
Your text content here...
This can be from books, articles, websites, etc.
"""

# Create preprocessor
preprocessor = FlatTextTrainingData(context_window=8, overlap=3)

# Process text into training data
training_data = preprocessor.preprocess_document(
    document_text, 
    doc_type="article",  # or "book", "wikipedia", "webpage"
    doc_title="My Document"
)

# Save training data
with open("my_text_training.json", "w") as f:
    json.dump(training_data, f, indent=2)

# Initialize BrainNexus for language modeling
manager = BrainNexusManager()

# Configure for token prediction (language modeling)
output_config = {
    'type': 'tokens',
    'num_classes': 1000,  # Vocabulary size
    'output_format': 'token_id',
    'confidence_threshold': 0.6,
    'return_top_k': 3,
    'enable_multi_token': True,  # Enable sequence generation
    'max_sequence_length': 20,
    'sequence_length': 10
}

# Create learning brain with language modeling configuration
from BrainNexusLearn import BrainNexusLearn, TrainingConfig

training_config = TrainingConfig(
    learning_rate=0.001,
    batch_size=8,
    max_epochs=100,
    data_type='text_generation'  # Specialized for text
)

manager.learner = BrainNexusLearn(demo=True, config=training_config, output_config=output_config)
manager.brain = manager.learner
manager.brain.initialize_brain()

# Load training data
manager.training_data = training_data['training_data']
manager.validation_data = training_data['validation_data']

# Train the brain
print("🚀 Starting language model training...")
results = manager.learner.supervised_train(manager.training_data, manager.validation_data)

print(f"✅ Training completed!")
print(f"   Final accuracy: {results['final_accuracy']:.3f}")
print(f"   Training time: {results['training_time']:.2f}s")

# Test text generation
test_input = "Artificial intelligence is"
result = manager.run_inference(test_input)
print(f"Input: '{test_input}'")
print(f"Generated: {result.get('prediction', 'unknown')}")
print(f"Confidence: {result.get('confidence', 0):.3f}")

# Save trained model
brain_file = manager.save_brain_state("language_model_brain")
print(f"Language model saved to: {brain_file}")
'''
    
    print("🚀 FLAT TEXT TRAINING USAGE")
    print("="*60)
    print("How to train BrainNexus on Wikipedia pages, books, articles, etc.:")
    print(usage_code)

def main():
    """Main function to demonstrate flat text training."""
    
    print("📖 BrainNexus Flat Text Training System")
    print("="*60)
    
    print("\n🎯 This system allows you to train BrainNexus on:")
    print("  • Wikipedia pages")
    print("  • Books and novels")
    print("  • Articles and essays")
    print("  • Web pages")
    print("  • Any plain text document")
    
    print("\n📊 Training approaches:")
    print("  1. Token-by-token prediction (next word)")
    print("  2. Sentence completion")
    print("  3. Paragraph continuation")
    print("  4. Document-level patterns")
    
    # Create sample data
    print("\n📁 Creating sample training data...")
    wiki_file = create_sample_wikipedia_data()
    book_file = create_sample_book_data()
    
    # Show usage
    usage_example()
    
    print("\n" + "="*60)
    print("🎉 FLAT TEXT TRAINING READY!")
    print("="*60)
    
    print(f"\nGenerated sample files:")
    print(f"  • {wiki_file} - Wikipedia-style training data")
    print(f"  • {book_file} - Book/document training data")
    
    print(f"\nNext steps:")
    print(f"1. Use the sample files to test training")
    print(f"2. Create your own text files or fetch Wikipedia pages")
    print(f"3. Process them with FlatTextTrainingData")
    print(f"4. Train BrainNexus for language modeling")
    print(f"5. Generate text with your trained model!")

if __name__ == "__main__":
    main()
