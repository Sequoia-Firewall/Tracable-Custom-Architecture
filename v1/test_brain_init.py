#!/usr/bin/env python3
"""
Quick test to check if brain initialization works without hanging
"""

import sys
import time
from brainNexus import BrainNexus

def test_full_brain_init():
    """Test full brain initialization"""
    print("🧪 Testing FULL brain initialization...")
    start_time = time.time()
    
    try:
        # Create full mode brain (not demo)
        brain = BrainNexus(dimensions=4, demo=False)
        
        print("📍 Starting brain initialization...")
        node_map = brain.initialize_brain()
        
        initialization_time = time.time() - start_time
        print(f"✅ FULL brain initialization completed in {initialization_time:.2f}s")
        
        # Verify counts
        print(f"📊 Node counts:")
        for node_type, node_list in node_map.items():
            print(f"   {node_type}: {len(node_list)}")
            
        return True
        
    except Exception as e:
        print(f"❌ FULL brain initialization failed: {e}")
        return False

def test_demo_brain_init():
    """Test demo brain initialization"""
    print("\n🧪 Testing DEMO brain initialization...")
    start_time = time.time()
    
    try:
        # Create demo mode brain
        brain = BrainNexus(dimensions=4, demo=True)
        
        print("📍 Starting brain initialization...")
        node_map = brain.initialize_brain()
        
        initialization_time = time.time() - start_time
        print(f"✅ DEMO brain initialization completed in {initialization_time:.2f}s")
        
        # Verify counts
        print(f"📊 Node counts:")
        for node_type, node_list in node_map.items():
            print(f"   {node_type}: {len(node_list)}")
            
        return True
        
    except Exception as e:
        print(f"❌ DEMO brain initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting brain initialization tests...")
    
    # Test demo first (should be faster)
    demo_success = test_demo_brain_init()
    
    # Only test full if demo worked
    if demo_success:
        full_success = test_full_brain_init()
        
        if full_success:
            print("\n🎉 All tests passed! Brain initialization working correctly.")
        else:
            print("\n⚠️  Demo passed but full brain failed.")
    else:
        print("\n❌ Demo brain failed, skipping full brain test.")
