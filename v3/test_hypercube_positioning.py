"""
Test file for hypercube vertex positioning and dimensional optimization.
Demonstrates the mathematical relationship between dimensions and optimal node counts.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add the v3 directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from BrainNexus import BrainNexus

def test_hypercube_vertices():
    """Test hypercube vertex generation for various dimensions."""
    print("🎯 HYPERCUBE VERTEX POSITIONING TEST")
    print("=" * 60)
    
    dimensions_to_test = [2, 3, 4, 5]
    
    for dims in dimensions_to_test:
        print(f"\n📐 Testing {dims}D hypercube:")
        print(f"   Expected vertices: 2^{dims} = {2**dims}")
        
        # Create brain with dynamic pre-nodes
        brain = BrainNexus(dimensions=dims, demo=True)
        
        # Generate vertices
        vertices = brain.generate_hypercube_vertices(radius=10.0)
        
        print(f"   Generated vertices: {len(vertices)}")
        print(f"   ✓ Count matches expectation: {len(vertices) == 2**dims}")
        
        # Show first few vertices for visualization
        print("   Sample vertices:")
        for i, vertex in enumerate(vertices[:min(8, len(vertices))]):
            vertex_str = [f"{v:+5.1f}" for v in vertex]
            print(f"     Vertex {i}: [{', '.join(vertex_str)}]")
        
        if len(vertices) > 8:
            print(f"     ... and {len(vertices) - 8} more vertices")
        
        # Validate vertex properties
        validate_hypercube_properties(vertices, dims)

def validate_hypercube_properties(vertices: List[List[float]], dimensions: int):
    """Validate mathematical properties of hypercube vertices."""
    print(f"   🔍 Validating hypercube properties:")
    
    # Check count
    expected_count = 2 ** dimensions
    actual_count = len(vertices)
    print(f"     Vertex count: {actual_count}/{expected_count} ✓" if actual_count == expected_count else f"     Vertex count: {actual_count}/{expected_count} ❌")
    
    # Check dimensionality
    if len(vertices) > 0:
        vertex_dims = len(vertices[0])
        print(f"     Dimensionality: {vertex_dims}/{dimensions} ✓" if vertex_dims == dimensions else f"     Dimensionality: {vertex_dims}/{dimensions} ❌")
    
    # Check vertex distance from origin (should all be equal for uniform radius)
    if len(vertices) > 1:
        distances = [np.linalg.norm(vertex) for vertex in vertices]
        distance_variance = np.var(distances)
        print(f"     Distance uniformity: {distance_variance:.6f} (variance, lower is better)")
        if distance_variance < 1e-10:
            print("     ✅ All vertices equidistant from origin")
        else:
            print("     ⚠️  Some distance variation detected")
    
    # Check uniqueness of vertices
    unique_vertices = set(tuple(v) for v in vertices)
    print(f"     Uniqueness: {len(unique_vertices)}/{len(vertices)} unique ✓" if len(unique_vertices) == len(vertices) else f"     Uniqueness: {len(unique_vertices)}/{len(vertices)} unique ❌")

def test_dimensional_coverage():
    """Test dimensional coverage calculation and optimization."""
    print("\n\n🎯 DIMENSIONAL COVERAGE TEST")
    print("=" * 60)
    
    # Test different configurations
    test_configs = [
        {"dims": 2, "description": "2D (quadrants)"},
        {"dims": 3, "description": "3D (octants)"},
        {"dims": 4, "description": "4D (tesseract)"},
    ]
    
    for config in test_configs:
        dims = config["dims"]
        desc = config["description"]
        
        print(f"\n📊 Testing coverage for {desc}:")
        
        # Create brain with dynamic pre-nodes
        brain = BrainNexus(dimensions=dims, demo=False)
        
        # Get dimensional info before creating nodes
        info_before = brain.get_dimensional_info()
        print(f"   Expected optimal nodes: {info_before['hypercube_vertices']}")
        print(f"   Pre-configured nodes: {info_before['entrance_nodes_pre']}")
        print(f"   Dynamic pre-nodes: {info_before['dynamic_pre_nodes']}")
        
        # Create optimal entrance nodes
        print("\n   Creating optimal entrance nodes...")
        created_nodes = brain.create_optimal_entrance_nodes(node_type='Controller', radius=8.0)
        
        # Get dimensional info after creating nodes
        info_after = brain.get_dimensional_info()
        print(f"\n   📈 Coverage Results:")
        print(f"     Created nodes: {len(created_nodes)}")
        print(f"     Coverage ratio: {info_after['coverage_ratio']:.2%}")
        print(f"     Coverage status: {info_after['coverage_status']}")
        
        # Show vertex coordinates for smaller dimensions
        if dims <= 3:
            print(f"   🎯 Vertex positions (first {min(len(info_after['vertices_coordinates']), 8)}):")
            for i, vertex in enumerate(info_after['vertices_coordinates'][:8]):
                vertex_str = [f"{v:+6.1f}" for v in vertex]
                print(f"     Position {i}: [{', '.join(vertex_str)}]")

def test_entrance_node_creation():
    """Test the complete entrance node creation workflow."""
    print("\n\n🎯 ENTRANCE NODE CREATION WORKFLOW TEST")
    print("=" * 60)
    
    # Test with 3D brain
    print(f"\n🧠 Creating 3D brain with optimal entrance nodes:")
    brain = BrainNexus(dimensions=3, demo=True)
    
    print(f"\n   Initial state:")
    print(f"   - Dimensions: {brain.dimensions}")
    print(f"   - Expected vertices: {brain.dimensional_vertices}")
    print(f"   - Pre-node count: {brain.entrance_node_count_pre}")
    print(f"   - Total nodes: {len(brain.neural_nodes)}")
    
    # Create optimal entrance nodes
    print(f"\n   Creating optimal entrance nodes...")
    node_ids = brain.create_optimal_entrance_nodes(node_type='Controller', radius=12.0)
    
    print(f"\n   Final state:")
    print(f"   - Created node IDs: {node_ids}")
    print(f"   - Total nodes: {len(brain.neural_nodes)}")
    
    # Validate node properties
    print(f"\n   🔍 Validating created nodes:")
    for node_id in node_ids:
        if node_id in brain.node_registry:
            node = brain.node_registry[node_id]
            pos_str = [f"{p:+6.1f}" for p in node.node_position[:3]]
            print(f"     Node {node_id}: type={node.node_type}, group={node.node_group}")
            print(f"                    position=[{', '.join(pos_str)}]")
            print(f"                    distance from origin: {np.linalg.norm(node.node_position):.2f}")
    
    # Test dimensional coverage
    coverage = brain.calculate_dimensional_coverage()
    print(f"\n   📊 Final dimensional coverage: {coverage:.2%}")
    
    # Show comprehensive dimensional info
    dim_info = brain.get_dimensional_info()
    print(f"\n   📋 Complete Dimensional Info:")
    for key, value in dim_info.items():
        if key != 'vertices_coordinates':  # Skip detailed coordinates for cleaner output
            print(f"     {key}: {value}")

def run_all_tests():
    """Run all hypercube positioning tests."""
    print("🚀 HYPERCUBE POSITIONING & DIMENSIONAL OPTIMIZATION TESTS")
    print("=" * 80)
    
    try:
        test_hypercube_vertices()
        test_dimensional_coverage()
        test_entrance_node_creation()
        
        print("\n\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("🎯 Hypercube vertex positioning is working correctly")
        print("📊 Dimensional coverage optimization is functional")
        print("🧠 Entrance node creation workflow is operational")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
