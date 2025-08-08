#!/usr/bin/env python3
"""
Quick test to isolate the connection issue between retainers and reviewers
"""

import sys
import time
from brainNexus import BrainNexus
from BrainNexusLearn import BrainNexusLearn, TrainingConfig

def test_brainnexus_connections():
    """Test base BrainNexus connections"""
    print("🧪 Testing base BrainNexus connections...")
    
    try:
        # Create basic brain
        brain = BrainNexus(dimensions=4, demo=True)
        
        # Initialize
        print("📍 Initializing brain...")
        node_map = brain.initialize_brain()
        
        # Check retainer-reviewer connections immediately after initialization
        print(f"\n🔍 Checking retainer-reviewer connections:")
        for i, (retainer_id, reviewer_id) in enumerate(zip(node_map['retainers'], node_map['reviewers'])):
            retainer_connections = brain.get_node_connections(retainer_id)
            reviewer_connections = brain.get_node_connections(reviewer_id)
            
            print(f"  Retainer #{retainer_id} outgoing: {retainer_connections['outgoing']}")
            print(f"  Reviewer #{reviewer_id} incoming: {reviewer_connections['incoming']}")
            
            if reviewer_id in retainer_connections['outgoing']:
                print(f"    ✅ Connection verified: Retainer {retainer_id} → Reviewer {reviewer_id}")
            else:
                print(f"    ❌ Missing connection: Retainer {retainer_id} ↛ Reviewer {reviewer_id}")
        
        # Test basic run
        print(f"\n🚀 Testing basic run...")
        result = brain.run("test input")
        
        print(f"✅ Base BrainNexus test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Base BrainNexus test FAILED: {e}")
        return False

def test_brainnexuslearn_connections():
    """Test BrainNexusLearn connections"""
    print("\n🧪 Testing BrainNexusLearn connections...")
    
    try:
        # Create learning brain
        training_config = TrainingConfig(
            learning_rate=0.01,
            spatial_learning_rate=0.005,
            batch_size=8,
            max_epochs=50
        )
        
        brain = BrainNexusLearn(demo=True, config=training_config)
        
        # Initialize
        print("📍 Initializing learning brain...")
        node_map = brain.initialize_brain()
        
        # Check retainer-reviewer connections immediately after initialization
        print(f"\n🔍 Checking retainer-reviewer connections BEFORE any training:")
        for i, (retainer_id, reviewer_id) in enumerate(zip(node_map['retainers'], node_map['reviewers'])):
            retainer_connections = brain.get_node_connections(retainer_id)
            reviewer_connections = brain.get_node_connections(reviewer_id)
            
            print(f"  Retainer #{retainer_id} outgoing: {retainer_connections['outgoing']}")
            print(f"  Reviewer #{reviewer_id} incoming: {reviewer_connections['incoming']}")
            
            if reviewer_id in retainer_connections['outgoing']:
                print(f"    ✅ Connection verified: Retainer {retainer_id} → Reviewer {reviewer_id}")
            else:
                print(f"    ❌ Missing connection: Retainer {retainer_id} ↛ Reviewer {reviewer_id}")
        
        # Test run immediately after initialization
        print(f"\n🚀 Testing run immediately after initialization...")
        result = brain.run("test input")
        print(f"✅ Immediate run PASSED")
        
        # Check connections again after run
        print(f"\n🔍 Checking retainer-reviewer connections AFTER first run:")
        for i, (retainer_id, reviewer_id) in enumerate(zip(node_map['retainers'], node_map['reviewers'])):
            retainer_connections = brain.get_node_connections(retainer_id)
            reviewer_connections = brain.get_node_connections(reviewer_id)
            
            if reviewer_id in retainer_connections['outgoing']:
                print(f"    ✅ Connection still intact: Retainer {retainer_id} → Reviewer {reviewer_id}")
            else:
                print(f"    ❌ Connection BROKEN after run: Retainer {retainer_id} ↛ Reviewer {reviewer_id}")
        
        print(f"✅ BrainNexusLearn test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ BrainNexusLearn test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting connection isolation tests...")
    
    # Test base BrainNexus first
    base_success = test_brainnexus_connections()
    
    # Test BrainNexusLearn
    learn_success = test_brainnexuslearn_connections()
    
    if base_success and learn_success:
        print("\n🎉 All tests passed! Connection issue not reproduced.")
    elif base_success and not learn_success:
        print("\n⚠️  Base BrainNexus works but BrainNexusLearn has connection issues!")
    else:
        print("\n❌ Connection issues detected in base implementation.")
