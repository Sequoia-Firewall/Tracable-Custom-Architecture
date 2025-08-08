#!/usr/bin/env python3
"""
Test script for dynamic node allocation with 85% computational nodes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dynamic_allocation():
    """Test the dynamic node allocation system."""
    try:
        from main import BrainNexusInterface
        
        print("🧪 Testing Dynamic Node Allocation")
        print("=" * 50)
        
        interface = BrainNexusInterface()
        interface.demo_mode = True
        interface.verbose = True
        
        # Initialize BrainNexus
        print("\n1. Initializing BrainNexus...")
        init_result = interface._handle_init(['4'])  # 4 dimensions
        if init_result['status'] != 'success':
            print(f"❌ Initialization failed: {init_result}")
            return
        
        print("✅ BrainNexus initialized successfully")
        
        # Test different segment sizes with memory-optimized configurations
        test_cases = [
            ('demo', 1, 'balanced'),  # Reduced to 1 segment for memory
            # Skip other tests for now to focus on memory optimization
        ]
        
        for config_preset, num_segments, segment_type in test_cases:
            print(f"\n2. Testing {config_preset} configuration...")
            
            # Create segments
            create_result = interface._handle_create([str(num_segments), segment_type, config_preset])
            
            if create_result['status'] == 'success':
                print(f"✅ Created {create_result['segments_created']} segments with {config_preset} size")
                print(f"   Total nodes: {create_result['total_nodes']}")
                
                # Analyze node distribution
                for segment in interface.segments:
                    total_nodes = len(segment.segment_nodes)
                    computational_count = len(segment.node_type_registry['computational'])
                    computational_percentage = (computational_count / total_nodes * 100) if total_nodes > 0 else 0
                    
                    print(f"\n   Segment {segment.segment_id} Analysis:")
                    print(f"      Total nodes: {total_nodes}")
                    print(f"      Judges: {len(segment.node_type_registry['judges'])}")
                    print(f"      Splitters: {len(segment.node_type_registry['splitters'])}")
                    print(f"      Computational: {computational_count} ({computational_percentage:.1f}%)")
                    print(f"      Reviewers: {len(segment.node_type_registry['reviewers'])}")
                    print(f"      Retainers: {len(segment.node_type_registry['retainers'])}")
                    print(f"      Resource limit: {segment.resource_limits['max_nodes']} nodes")
                    
                    if computational_percentage >= 80:  # Allow some tolerance
                        print(f"      ✅ Computational nodes are {computational_percentage:.1f}% (target: 85%)")
                    else:
                        print(f"      ⚠️  Computational nodes are {computational_percentage:.1f}% (expected: ~85%)")
            else:
                print(f"❌ Failed to create segments: {create_result}")
            
            # Clear for next test
            interface.segments.clear()
            interface.segment_learners.clear()
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dynamic_allocation()
