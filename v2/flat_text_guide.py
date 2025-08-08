#!/usr/bin/env python3
"""
BrainNexus Flat Text Training - Usage Guide
==========================================
Complete guide for training BrainNexus on flat text like Wikipedia pages, books, articles, etc.

The flat text training capability is now fully integrated into BrainNexus!
"""

def print_usage_guide():
    """Print comprehensive usage guide for flat text training."""
    
    print("📖 BRAINNEXUS FLAT TEXT TRAINING - COMPLETE GUIDE")
    print("="*70)
    
    print("\n🎯 WHAT IS FLAT TEXT TRAINING?")
    print("-"*40)
    print("Train BrainNexus directly on raw text documents to learn:")
    print("• Language patterns and word relationships")
    print("• Context understanding and text generation")
    print("• Domain-specific vocabulary and concepts")
    print("• Writing styles and document structures")
    
    print("\n📚 SUPPORTED TEXT TYPES:")
    print("-"*40)
    print("• Wikipedia pages and articles")
    print("• Books, novels, and literature")
    print("• Academic papers and research")
    print("• News articles and blog posts")
    print("• Web pages and documentation")
    print("• Technical manuals and guides")
    print("• Any plain text document!")
    
    print("\n🚀 THREE WAYS TO USE IT:")
    print("-"*40)
    
    print("\n1. MAIN MENU INTERFACE:")
    print("   Run: python main.py")
    print("   Choose option 19: Train on Flat Text (Direct Input)")
    print("   Choose option 20: Train on Text File")
    print("   Choose option 21: Generate Text from Prompt")
    
    print("\n2. PROGRAMMATIC USAGE:")
    code_example = '''
from main import BrainNexusManager

# Initialize
manager = BrainNexusManager()
manager.initialize_new_brain(demo_mode=True)

# Method A: Train on direct text
text = """Your text content here..."""
manager.train_on_flat_text(
    text=text,
    text_title="My Document",
    text_type="article",
    context_window=10,
    training_approach="all"
)

# Method B: Train on file
manager.train_on_file(
    file_path="my_document.txt",
    text_type="book",
    context_window=8,
    training_approach="token_prediction"
)

# Generate text
generated = manager.generate_text(
    prompt="The future of AI is",
    max_length=30,
    temperature=0.8
)
print(f"Generated: {generated}")
'''
    print(code_example)
    
    print("\n3. ADVANCED INTEGRATION:")
    advanced_code = '''
from BrainNexusLearn import BrainNexusLearn, TrainingConfig

# Configure for text generation
config = TrainingConfig(
    learning_rate=0.001,
    batch_size=8,
    max_epochs=50,
    data_type='text_generation'
)

learner = BrainNexusLearn(demo=True, config=config)
learner.initialize_brain()

# Train directly
results = learner.train_on_flat_text(
    text=your_text,
    text_title="Custom Training",
    text_type="wikipedia",
    context_window=12,
    training_approach="all"
)

# Generate with advanced options
generated = learner.generate_text(
    prompt="Starting prompt",
    max_length=100,
    temperature=1.2  # More creative
)
'''
    print(advanced_code)
    
    print("\n⚙️ TRAINING PARAMETERS:")
    print("-"*40)
    
    print("\ntext_type options:")
    print("• 'wikipedia' - Optimized for Wikipedia content")
    print("• 'book' - Best for novels and literature")
    print("• 'article' - Good for news and blog posts")
    print("• 'webpage' - For web content and documentation")
    print("• 'document' - General purpose (default)")
    
    print("\ntraining_approach options:")
    print("• 'all' - Complete training (recommended)")
    print("• 'token_prediction' - Next word prediction only")
    print("• 'sentence_completion' - Sentence-level understanding")
    print("• 'paragraph_continuation' - Paragraph-level patterns")
    
    print("\ncontext_window (default 10):")
    print("• Smaller (5-8): Faster training, local patterns")
    print("• Medium (8-12): Balanced performance")
    print("• Larger (12-20): Better context, slower training")
    
    print("\ntemperature for generation:")
    print("• 0.1-0.5: Conservative, coherent text")
    print("• 0.6-1.0: Balanced creativity")
    print("• 1.1-2.0: More creative and diverse")
    
    print("\n📊 TRAINING APPROACHES EXPLAINED:")
    print("-"*40)
    
    print("\n1. TOKEN PREDICTION:")
    print("   • Learns to predict the next word in sequence")
    print("   • Input: 'Artificial intelligence is'")
    print("   • Target: 'revolutionary'")
    print("   • Best for: Language modeling, word completion")
    
    print("\n2. SENTENCE COMPLETION:")
    print("   • Learns to complete partial sentences")
    print("   • Input: 'Machine learning algorithms can'")
    print("   • Target: 'process'")
    print("   • Best for: Natural sentence generation")
    
    print("\n3. PARAGRAPH CONTINUATION:")
    print("   • Learns to continue multi-sentence text")
    print("   • Input: First part of paragraph")
    print("   • Target: First word of continuation")
    print("   • Best for: Long-form text generation")
    
    print("\n📈 TRAINING WORKFLOW:")
    print("-"*40)
    print("1. Text preprocessing (automatic)")
    print("   • Clean markup and formatting")
    print("   • Extract meaningful sentences")
    print("   • Create training samples")
    
    print("\n2. Feature extraction (automatic)")
    print("   • Word count and character statistics")
    print("   • Punctuation and structure analysis")
    print("   • Vocabulary richness metrics")
    
    print("\n3. Model training")
    print("   • 80% training data, 20% validation")
    print("   • Supervised learning with text targets")
    print("   • Spatial optimization for efficiency")
    
    print("\n4. Text generation")
    print("   • Context-aware next word prediction")
    print("   • Temperature-based sampling")
    print("   • Coherent multi-token sequences")
    
    print("\n💡 TIPS FOR BEST RESULTS:")
    print("-"*40)
    print("• Use high-quality, well-written text sources")
    print("• Longer texts (1000+ words) train better")
    print("• Match text_type to your content for optimal preprocessing")
    print("• Start with 'all' training approach, then specialize")
    print("• Adjust context_window based on text complexity")
    print("• Use lower temperature for formal text, higher for creative")
    print("• Save trained models to avoid retraining")
    
    print("\n🔥 EXAMPLE WORKFLOWS:")
    print("-"*40)
    
    print("\nWIKIPEDIA TRAINING:")
    print("1. Copy Wikipedia article content")
    print("2. Use text_type='wikipedia'")
    print("3. Use training_approach='all'")
    print("4. Set context_window=10-15")
    print("5. Generate encyclopedia-style text")
    
    print("\nBOOK/NOVEL TRAINING:")
    print("1. Load book text from file")
    print("2. Use text_type='book'")
    print("3. Use training_approach='all'")
    print("4. Set context_window=8-12")
    print("5. Generate narrative-style text")
    
    print("\nTECHNICAL DOCUMENTATION:")
    print("1. Load documentation files")
    print("2. Use text_type='document'")
    print("3. Use training_approach='sentence_completion'")
    print("4. Set context_window=6-10")
    print("5. Generate technical explanations")
    
    print("\n⚠️ TROUBLESHOOTING:")
    print("-"*40)
    print("• Text too short: Use at least 500-1000 words")
    print("• Poor generation: Try different temperature values")
    print("• Training slow: Reduce context_window or batch_size")
    print("• Low accuracy: Use 'all' approach and more epochs")
    print("• Memory issues: Use smaller batch_size in config")
    
    print("\n🎉 SUCCESS INDICATORS:")
    print("-"*40)
    print("• Training accuracy > 0.3 (30%)")
    print("• Generated text is grammatically correct")
    print("• Generated text stays on topic")
    print("• Vocabulary matches source material")
    print("• Generated sentences make logical sense")
    
    print("\n" + "="*70)
    print("🚀 READY TO START TRAINING!")
    print("Run: python main.py")
    print("Choose options 19-21 for flat text training!")
    print("="*70)

def print_quick_start():
    """Print a quick start guide."""
    
    print("\n⚡ QUICK START - 3 STEPS:")
    print("-"*30)
    print("1. Run: python main.py")
    print("2. Choose option 1 or 2 to initialize brain")
    print("3. Choose option 19 to paste text and train")
    print("4. Choose option 21 to generate text!")
    print("\nThat's it! 🎉")

def main():
    """Main function."""
    print_usage_guide()
    print_quick_start()

if __name__ == "__main__":
    main()
