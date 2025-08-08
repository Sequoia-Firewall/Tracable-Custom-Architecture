"""
Test script to demonstrate the enhanced main.py with configurable output system
"""

from main import BrainNexusManager

def test_configuration_system():
    """Test the enhanced main.py with different configurations"""
    print("🧪 Testing Enhanced BrainNexus Main Interface")
    print("=" * 50)
    
    # Create manager
    manager = BrainNexusManager()
    
    # Test different configurations programmatically
    test_configs = [
        {
            'name': 'Image Classification',
            'config': {
                'type': 'classification',
                'num_classes': 5,
                'class_labels': ['cat', 'dog', 'bird', 'fish', 'hamster'],
                'output_format': 'label',
                'confidence_threshold': 0.7,
                'return_top_k': 3
            }
        },
        {
            'name': 'Token Prediction',
            'config': {
                'type': 'tokens',
                'num_classes': 100,
                'vocab_mapping': {i: f'token_{i:03d}' for i in range(100)},
                'output_format': 'token_id',
                'confidence_threshold': 0.5,
                'return_top_k': 5
            }
        },
        {
            'name': 'Sentiment Analysis',
            'config': {
                'type': 'classification',
                'num_classes': 3,
                'class_labels': ['negative', 'neutral', 'positive'],
                'output_format': 'label',
                'confidence_threshold': 0.6,
                'return_top_k': 3
            }
        }
    ]
    
    for i, test in enumerate(test_configs):
        print(f"\n🔬 Test {i+1}: {test['name']}")
        print("-" * 30)
        
        try:
            # Simulate the configuration by directly creating brain with config
            from BrainNexusLearn import BrainNexusLearn, TrainingConfig
            
            training_config = TrainingConfig()
            learner = BrainNexusLearn(demo=True, config=training_config, output_config=test['config'])
            
            print(f"✓ Created brain with {test['name']} configuration")
            print(f"  Type: {test['config']['type']}")
            print(f"  Classes: {test['config']['num_classes']}")
            print(f"  Format: {test['config']['output_format']}")
            
            # Initialize the brain
            node_map = learner.initialize_brain()
            if node_map:
                print(f"✓ Brain initialized successfully")
                print(f"  Total nodes: {len(learner.neural_nodes)}")
                
                # Test a prediction
                result = learner.run("test input", trace_execution=False)
                print(f"✓ Test prediction completed")
                print(f"  Raw prediction: {result.get('prediction', 'N/A')}")
                
                if 'prediction_label' in result:
                    print(f"  Predicted label: {result['prediction_label']}")
                if 'predicted_token' in result:
                    print(f"  Predicted token: {result['predicted_token']}")
                
                print(f"  Confidence: {result.get('confidence', 0.0):.3f}")
                
                if 'top_k_predictions' in result:
                    print(f"  Top-{len(result['top_k_predictions'])} predictions available")
            else:
                print("❌ Failed to initialize brain")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n✅ Configuration system testing complete!")
    print(f"\n📝 Instructions for Interactive Use:")
    print(f"   1. Run: python main.py")
    print(f"   2. Choose option 1 or 2 (Demo/Full mode)")
    print(f"   3. When prompted 'Configure custom output system? (y/n)', enter 'y'")
    print(f"   4. Follow the interactive prompts to configure your output system")
    print(f"   5. Your brain will be initialized with the custom configuration")

if __name__ == "__main__":
    test_configuration_system()
