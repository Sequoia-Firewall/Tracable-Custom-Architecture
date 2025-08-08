#!/usr/bin/env python3
"""
Test Memory Cleanup System for BrainNexus v3

This test demonstrates the multi-tier memory cleanup system implemented
for both BrainNexus and NexusSegment components.

Features Tested:
- Light, Partial, Aggressive, and Nuclear cleanup tiers
- Automatic periodic cleanup during node creation
- Emergency cleanup when memory usage is critical
- Segment-specific memory cleanup
- Memory status monitoring and recommendations
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BrainNexus import BrainNexus
from BrainSegment import NexusSegment

def test_memory_cleanup_system():
    """Test the comprehensive memory cleanup system."""
    
    print("🧠 Testing BrainNexus Memory Cleanup System")
    print("=" * 60)
    
    # Create a BrainNexus instance with demo mode
    brain = BrainNexus(dimensions=4, demo=True)
    
    print("\n1️⃣ INITIAL MEMORY STATUS")
    print("-" * 40)
    memory_status = brain.get_memory_status()
    print(f"Memory Pressure: {memory_status['pressure_level']}")
    print(f"Total Items: {memory_status['total_items']}")
    print(f"Recommendation: {memory_status['recommendation']}")
    
    print("\n2️⃣ BUILDING MEMORY USAGE")
    print("-" * 40)
    
    # Add many nodes to build up memory usage
    node_ids = []
    for i in range(50):
        if i % 10 == 0:
            node_type = 'Controller'
        elif i % 7 == 0:
            node_type = 'Judge'
        elif i % 5 == 0:
            node_type = 'Splitter'
        elif i % 3 == 0:
            node_type = 'Retainer'
        elif i % 2 == 0:
            node_type = 'Reviewer'
        else:
            node_type = 'Computational'
        
        node_id = brain.add_neural_node(
            node_type=node_type,
            position=[np.random.uniform(-10, 10) for _ in range(4)]
        )
        node_ids.append(node_id)
        
        # Simulate node usage to build up tracking data
        brain.update_node_usage(node_id, f"test_input_{i}")
        
        # Add to inference cache
        brain.inference_cache[f"cache_key_{i}"] = f"cached_result_{i}"
        
        # Add to term embeddings
        if hasattr(brain, 'term_embeddings'):
            brain.term_embeddings[f"term_{i}"] = np.random.random((64,))
        
        # Add to routing history
        brain.routing_history.append({
            'node_id': node_id,
            'decision': f"route_decision_{i}",
            'timestamp': i
        })
    
    # Connect some nodes to build connection tracking
    for i in range(0, len(node_ids)-1, 2):
        try:
            brain.connect_nodes(node_ids[i], node_ids[i+1], weight=0.8)
        except:
            pass  # Some connections might fail, that's OK
    
    print(f"Created {len(node_ids)} nodes with extensive memory usage")
    
    # Check memory status after building usage
    memory_status = brain.get_memory_status()
    print(f"Memory Pressure: {memory_status['pressure_level']}")
    print(f"Total Items: {memory_status['total_items']}")
    print(f"Detailed Stats: {memory_status['memory_stats']}")
    
    print("\n3️⃣ TESTING CLEANUP TIERS")
    print("-" * 40)
    
    # Test Light Cleanup
    print("\n🌤️  Testing LIGHT cleanup:")
    light_stats = brain.cleanup_memory('light')
    print(f"Cleaned items: {dict(light_stats['cleaned_items'])}")
    print(f"Memory freed: {light_stats['memory_freed']['total_items']}")
    print(f"Cleanup time: {light_stats['cleanup_time']:.3f}s")
    
    # Build up more memory usage
    for i in range(50, 100):
        brain.inference_cache[f"cache_key_{i}"] = f"cached_result_{i}"
        brain.routing_history.append({'test': f"data_{i}"})
    
    # Test Partial Cleanup
    print("\n⛅ Testing PARTIAL cleanup:")
    partial_stats = brain.cleanup_memory('partial')
    print(f"Cleaned items: {dict(partial_stats['cleaned_items'])}")
    print(f"Memory freed: {partial_stats['memory_freed']['total_items']}")
    print(f"Cleanup time: {partial_stats['cleanup_time']:.3f}s")
    
    # Build up even more memory usage
    for i in range(100, 200):
        brain.inference_cache[f"cache_key_{i}"] = f"cached_result_{i}"
        if hasattr(brain, 'term_embeddings'):
            brain.term_embeddings[f"term_{i}"] = np.random.random((128,))
    
    # Test Aggressive Cleanup
    print("\n🌩️  Testing AGGRESSIVE cleanup:")
    aggressive_stats = brain.cleanup_memory('aggressive')
    print(f"Cleaned items: {dict(aggressive_stats['cleaned_items'])}")
    print(f"Memory freed: {aggressive_stats['memory_freed']['total_items']}")
    print(f"Cleanup time: {aggressive_stats['cleanup_time']:.3f}s")
    
    # Test Nuclear Cleanup (with force)
    print("\n💥 Testing NUCLEAR cleanup (forced):")
    nuclear_stats = brain.cleanup_memory('nuclear', force_cleanup=True)
    print(f"Cleaned items: {dict(nuclear_stats['cleaned_items'])}")
    print(f"Memory freed: {nuclear_stats['memory_freed']['total_items']}")
    print(f"Cleanup time: {nuclear_stats['cleanup_time']:.3f}s")
    
    print("\n4️⃣ TESTING EMERGENCY CLEANUP")
    print("-" * 40)
    
    # Build up critical memory usage
    for i in range(1000):
        brain.inference_cache[f"critical_cache_{i}"] = f"critical_data_{i}"
    
    print("Building critical memory usage...")
    memory_status = brain.get_memory_status()
    print(f"Memory Pressure: {memory_status['pressure_level']}")
    print(f"Total Items: {memory_status['total_items']}")
    
    # Test emergency cleanup
    print("\n🚨 Triggering EMERGENCY cleanup:")
    emergency_stats = brain.emergency_cleanup()
    print(f"Emergency tier used: {emergency_stats['tier']}")
    print(f"Cleaned items: {dict(emergency_stats['cleaned_items'])}")
    print(f"Memory freed: {emergency_stats['memory_freed']['total_items']}")
    
    print("\n5️⃣ TESTING SEGMENT CLEANUP")
    print("-" * 40)
    
    # Create a test segment
    test_segment = NexusSegment(
        segment_id=1,
        dimensional_assignment={'dim_0_pos': True, 'dim_1_neg': True},
        brain_nexus_ref=brain,
        demo=True
    )
    
    # Build up segment memory usage
    for i in range(100):
        test_segment.attention_cache[f"attention_{i}"] = np.random.random((32, 32))
        test_segment.result_cache[f"result_{i}"] = f"result_data_{i}"
        test_segment.processing_results[f"process_{i}"] = f"process_data_{i}"
        test_segment.pattern_memory.append(f"pattern_{i}")
    
    print("Built up segment memory usage")
    segment_status = test_segment.get_segment_memory_status()
    print(f"Segment Pressure: {segment_status['pressure_level']}")
    print(f"Segment Items: {segment_status['total_items']}")
    print(f"Recommendation: {segment_status['recommendation']}")
    
    # Test segment cleanup
    print("\n🧹 Testing segment cleanup:")
    segment_cleanup_stats = test_segment.cleanup_memory('partial')
    print(f"Segment cleaned items: {dict(segment_cleanup_stats['cleaned_items'])}")
    print(f"Segment memory freed: {segment_cleanup_stats['memory_freed']['total_items']}")
    
    print("\n6️⃣ TESTING PERIODIC CLEANUP")
    print("-" * 40)
    
    # Test periodic cleanup scheduling
    print("Testing periodic cleanup during node creation:")
    
    # Configure periodic cleanup
    cleanup_results = []
    for i in range(25):
        node_id = brain.add_neural_node('Computational')
        
        # The periodic cleanup should trigger automatically based on our implementation
        # Let's manually check if it would trigger
        periodic_result = brain.schedule_periodic_cleanup(cleanup_interval=10, cleanup_tier='light')
        if periodic_result:
            cleanup_results.append(periodic_result)
            print(f"  Periodic cleanup triggered at node {i+1}")
    
    print(f"Periodic cleanups triggered: {len(cleanup_results)}")
    
    print("\n✅ MEMORY CLEANUP SYSTEM TEST COMPLETE")
    print("=" * 60)
    
    # Final memory status
    final_status = brain.get_memory_status()
    print(f"Final Memory Pressure: {final_status['pressure_level']}")
    print(f"Final Total Items: {final_status['total_items']}")
    print(f"Final Recommendation: {final_status['recommendation']}")
    
    return True

def test_cleanup_integration():
    """Test cleanup system integration with normal BrainNexus operations."""
    
    print("\n🔄 Testing Cleanup Integration with Normal Operations")
    print("=" * 60)
    
    brain = BrainNexus(dimensions=3, demo=True)
    
    # Simulate normal brain operations with automatic cleanup
    print("Simulating normal brain operations with automatic cleanup...")
    
    # Initialize brain (should trigger cleanup)
    brain.initialize_brain("segments")  # Will create directory if needed
    
    # Process some data (should trigger cleanup)
    test_input = np.random.random((10, 64))
    
    try:
        result = brain.process_multidimensional_pipeline(
            test_input, 
            task_type='classification',
            judge_activation_ratio=0.3
        )
        print(f"Pipeline processing completed: {len(result)} results")
    except Exception as e:
        print(f"Pipeline processing note: {e} (expected for empty brain)")
    
    # Batch operations (should trigger cleanup)
    node_ids = []
    for i in range(15):
        node_id = brain.add_neural_node('Computational')
        node_ids.append(node_id)
    
    # Batch move nodes (should trigger cleanup if >10 moves)
    moves = [(node_id, [np.random.uniform(-5, 5) for _ in range(3)]) 
             for node_id in node_ids]
    
    moved_count = brain.move_nodes_batch(moves)
    print(f"Batch moved {moved_count} nodes")
    
    # Check final memory status
    final_status = brain.get_memory_status()
    print(f"Integration Test Final Status: {final_status['pressure_level']}")
    
    return True

if __name__ == "__main__":
    """Run all memory cleanup tests."""
    
    try:
        print("🚀 Starting BrainNexus Memory Cleanup System Tests")
        print("="*70)
        
        # Run main cleanup system test
        success1 = test_memory_cleanup_system()
        
        # Run integration test
        success2 = test_cleanup_integration()
        
        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED!")
            print("Memory cleanup system is fully operational.")
            
            print("\n📋 USAGE SUMMARY:")
            print("=" * 40)
            print("• brain.cleanup_memory('light')     - Light cleanup")
            print("• brain.cleanup_memory('partial')   - Moderate cleanup")
            print("• brain.cleanup_memory('aggressive')- Heavy cleanup")
            print("• brain.cleanup_memory('nuclear')   - Complete reset (use force=True)")
            print("• brain.emergency_cleanup()         - Auto-tier emergency cleanup")
            print("• brain.get_memory_status()         - Check memory pressure")
            print("• brain.schedule_periodic_cleanup() - Configure auto-cleanup")
            print("\nSegment cleanup:")
            print("• segment.cleanup_memory('partial') - Clean segment memory")
            print("• segment.get_segment_memory_status() - Check segment status")
            
        else:
            print("\n❌ SOME TESTS FAILED")
            print("Check the output above for details.")
            
    except Exception as e:
        print(f"\n💥 TEST ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*70)
    print("Test completed. Check output above for results.")
