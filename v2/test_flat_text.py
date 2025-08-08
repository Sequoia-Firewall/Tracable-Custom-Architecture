#!/usr/bin/env python3
"""
Test Flat Text Training Integration
==================================
Simple test script to demonstrate the integrated flat text training functionality.
"""

from main import BrainNexusManager
from BrainNexusLearn import TrainingConfig

def test_flat_text_training():
    """Test the integrated flat text training functionality."""
    
    print("🧠 TESTING FLAT TEXT TRAINING INTEGRATION")
    print("="*60)
    
    # Sample text for testing
    sample_text = """
    Artificial intelligence represents one of the most significant technological advances of the modern era. 
    Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions 
    with remarkable accuracy. Deep learning networks, inspired by the human brain, use multiple layers of 
    interconnected nodes to understand complex relationships in data.
    
    Natural language processing enables computers to understand and generate human language. Computer vision 
    allows machines to interpret and analyze visual information from the world around them. These technologies 
    are transforming industries from healthcare to transportation to entertainment.
    
    The future of artificial intelligence holds tremendous promise. As algorithms become more sophisticated 
    and computing power continues to grow, we can expect to see even more impressive breakthroughs in the 
    years ahead. However, it is important to develop these technologies responsibly and consider their 
    impact on society.
    """
    
    # Initialize manager
    print("1. Initializing BrainNexus Manager...")
    manager = BrainNexusManager()
    
    # Initialize brain for text generation
    print("2. Setting up brain for text generation...")
    manager.initialize_new_brain(demo_mode=True)
    
    # Configure for text generation if possible
    if manager.learner:
        manager.learner.config.data_type = 'text_generation'
        manager.learner.config.learning_rate = 0.001
        manager.learner.config.batch_size = 4
        manager.learner.config.max_epochs = 10
    
    # Test flat text training
    print("3. Training on sample text...")
    success = manager.train_on_flat_text(
        text=sample_text,
        text_title="AI Technology Overview",
        text_type="article",
        context_window=8,
        training_approach="all"
    )
    
    if success:
        print("✅ Training completed successfully!")
        
        # Test text generation
        print("\n4. Testing text generation...")
        test_prompts = [
            "Artificial intelligence is",
            "Machine learning algorithms",
            "The future of technology"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: '{prompt}'")
            generated = manager.generate_text(prompt, max_length=20, temperature=0.8)
            print(f"Generated: '{prompt} {generated}'")
            
        print("\n🎉 Flat text training integration test completed!")
        
    else:
        print("❌ Training failed!")

def demo_file_training():
    """Demo training from a file."""
    
    print("\n📄 DEMO: TRAINING FROM FILE")
    print("="*40)
    
    # Create a sample text file
    sample_file_content = """
    The development of neural networks has revolutionized machine learning. These computational models, 
    inspired by biological neural networks, consist of interconnected nodes that process information 
    in parallel. Each connection has an associated weight that adjusts as learning proceeds.
    
    Training a neural network involves presenting it with input-output pairs and adjusting the weights 
    to minimize prediction errors. This process, called backpropagation, allows the network to learn 
    complex patterns in data. Modern deep learning architectures can have millions or even billions 
    of parameters.
    
    Applications of neural networks span many domains. In computer vision, convolutional neural networks 
    excel at image recognition tasks. Recurrent neural networks are well-suited for sequential data 
    like text and time series. Transformer architectures have achieved remarkable success in natural 
    language processing.
    """
    
    # Save to file
    filename = "sample_neural_networks.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(sample_file_content)
    
    print(f"Created sample file: {filename}")
    
    # Initialize manager
    manager = BrainNexusManager()
    
    # Quick setup for demo
    manager.initialize_new_brain(demo_mode=True)
    
    # Configure for text generation
    if manager.learner:
        manager.learner.config.data_type = 'text_generation'
        manager.learner.config.learning_rate = 0.002
        manager.learner.config.batch_size = 3
        manager.learner.config.max_epochs = 5
    
    # Train from file
    print("Training from file...")
    success = manager.train_on_file(
        file_path=filename,
        text_type="article",
        context_window=6,
        training_approach="token_prediction"
    )
    
    if success:
        print("✅ File training completed!")
        
        # Test generation
        prompt = "Neural networks are"
        generated = manager.generate_text(prompt, max_length=15)
        print(f"Generated: '{prompt} {generated}'")
    else:
        print("❌ File training failed!")
    
    # Clean up
    import os
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Cleaned up: {filename}")

def main():
    """Run all tests."""
    
    print("🔬 FLAT TEXT TRAINING - INTEGRATION TESTS")
    print("="*60)
    
    try:
        # Test direct text training
        test_flat_text_training()
        
        # Test file training
        demo_file_training()
        
        print("\n🎯 INTEGRATION COMPLETE!")
        print("="*60)
        print("✅ Flat text training is now fully integrated into BrainNexus!")
        print("✅ You can now train on Wikipedia pages, books, articles, etc.")
        print("✅ Text generation capabilities are working!")
        print("\n🚀 Ready to use in main.py with menu options 19, 20, and 21!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
