#!/usr/bin/env python3
"""
Test script to verify that segment resource limits are being applied correctly.
"""

from BrainNexus import BrainNexus
from BrainSegment import NexusSegment

# Test configuration mimicking the demo preset
demo_config = {
    'enable_caching': True,
    'cache_size': 50,
    'enable_adaptation': True,
    'adaptation_rate': 0.15,
    'quality_threshold': 0.7,
    'enable_experimental_features': False,
    'resource_limits': {'max_nodes': 50, 'max_connections': 50},
    'hypercube_bounds': (-100.0, 100.0),
    'segment_type': 'demo'
}

print("🧪 Testing segment resource limit enforcement...")
print(f"Expected max nodes: {demo_config['resource_limits']['max_nodes']}")

# Initialize BrainNexus
brain_nexus = BrainNexus(dimensions=2, demo=True)

# Create test segment with demo config
dimensional_assignment = {0: 1, 1: -1}  # +x, -y quadrant
hypercube_bounds = [(-100.0, 100.0), (-100.0, 100.0)]

segment = NexusSegment(
    segment_id=1,
    dimensional_assignment=dimensional_assignment,
    brain_nexus_ref=brain_nexus,
    hypercube_bounds=hypercube_bounds,
    segment_config=demo_config,
    demo=True
)

print(f"\n📊 Segment Results:")
print(f"   Segment ID: {segment.segment_id}")
print(f"   Expected max nodes: {demo_config['resource_limits']['max_nodes']}")
print(f"   Configured max nodes: {segment.resource_limits['max_nodes']}")
print(f"   Actual nodes created: {len(segment.segment_nodes)}")
print(f"   Node type breakdown:")
for node_type, nodes in segment.node_type_registry.items():
    print(f"     {node_type}: {len(nodes)}")

# Validate
if len(segment.segment_nodes) <= demo_config['resource_limits']['max_nodes']:
    print("✅ Resource limits are being respected!")
else:
    print(f"❌ Resource limits violated! Created {len(segment.segment_nodes)} nodes, limit was {demo_config['resource_limits']['max_nodes']}")
