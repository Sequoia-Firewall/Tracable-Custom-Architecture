# Memory Cleanup System for BrainNexus v3

## Overview

The BrainNexus v3 architecture now includes a comprehensive multi-tier memory cleanup system designed to optimize memory usage and prevent memory bloat during long-running operations. This system operates at both the BrainNexus level and the NexusSegment level.

## Features

### 🧹 Multi-Tier Cleanup System
- **Light**: Conservative cleanup, removes only expired/stale data
- **Partial**: Moderate cleanup with some data loss
- **Aggressive**: Heavy cleanup with significant data loss
- **Nuclear**: Complete memory reset (requires force flag for safety)

### 🔄 Automatic Integration
- Periodic cleanup during node creation
- Post-processing cleanup after pipeline operations
- Batch operation cleanup after large operations
- Emergency cleanup when memory pressure is critical

### 📊 Memory Monitoring
- Real-time memory usage statistics
- Memory pressure level assessment
- Automatic recommendations for cleanup actions
- Detailed cleanup reports with items freed

## Usage Guide

### Basic Cleanup Operations

```python
from BrainNexus import BrainNexus

brain = BrainNexus(dimensions=4, demo=True)

# Check current memory status
status = brain.get_memory_status()
print(f"Memory pressure: {status['pressure_level']}")
print(f"Recommendation: {status['recommendation']}")

# Perform cleanup based on need
cleanup_stats = brain.cleanup_memory('partial')
print(f"Items cleaned: {cleanup_stats['cleaned_items']}")
print(f"Memory freed: {cleanup_stats['memory_freed']}")
```

### Cleanup Tiers

#### Light Cleanup (`cleanup_tier='light'`)
- Trims inference cache to 500 most recent entries
- Reduces routing history to 500 recent entries  
- Limits node input trackers to 50 recent entries
- Trims node usage history to 50 recent entries per node
- **Use case**: Regular maintenance, minimal data loss

#### Partial Cleanup (`cleanup_tier='partial'`)
- Includes all light cleanup actions
- Clears term embeddings cache if >500 entries
- Clears attention masks cache completely
- Reduces reuse candidates to top 50 most active
- Optimizes DataFrame records for existing nodes only
- **Use case**: Moderate memory pressure, acceptable data loss

#### Aggressive Cleanup (`cleanup_tier='aggressive'`)
- Includes all partial cleanup actions
- Completely clears inference cache
- Completely clears routing history
- Resets all node input trackers
- Removes usage history for inactive nodes
- Rebuilds spatial index for optimization
- **Use case**: High memory pressure, significant data loss acceptable

#### Nuclear Cleanup (`cleanup_tier='nuclear'`)
- Clears ALL caches and histories completely
- Resets all node tracking and statistics
- Clears spatial tracking (maintains positions)
- Resets node call counters to zero
- Requires `force_cleanup=True` for safety with >10 nodes
- **Use case**: Critical memory situations, complete reset acceptable

### Emergency Cleanup

```python
# Automatic cleanup tier selection based on memory usage
emergency_stats = brain.emergency_cleanup()
print(f"Emergency cleanup used tier: {emergency_stats['tier']}")
```

The emergency cleanup automatically selects the appropriate tier:
- **Nuclear**: >10,000 total items (with force=True)
- **Aggressive**: >5,000 total items
- **Partial**: >2,000 total items  
- **Light**: >1,000 total items

### Periodic Cleanup

```python
# Configure automatic cleanup every 100 node creations
brain.schedule_periodic_cleanup(cleanup_interval=100, cleanup_tier='light')

# The cleanup will trigger automatically during:
# - Node creation (add_neural_node)
# - Pipeline processing
# - Batch operations
```

### Segment Cleanup

```python
from BrainSegment import NexusSegment

# Create or load a segment
segment = NexusSegment(segment_id=1, ...)

# Check segment memory status
segment_status = segment.get_segment_memory_status()
print(f"Segment pressure: {segment_status['pressure_level']}")

# Clean segment memory
segment_cleanup_stats = segment.cleanup_memory('partial')
print(f"Segment items cleaned: {segment_cleanup_stats['cleaned_items']}")
```

## Memory Structures Managed

### BrainNexus Level
- **inference_cache**: Cached computation results
- **routing_history**: Historical routing decisions (deque, max 1000)
- **node_usage_history**: Per-node usage statistics
- **term_embeddings**: Cached term embeddings
- **attention_masks**: Layer-specific attention masks
- **reuse_candidates**: Nodes eligible for reuse
- **node_records**: DataFrame with node metadata
- **spatial tracking**: Node positions and spatial index

### NexusSegment Level
- **attention_cache**: Cached attention masks from judges
- **result_cache**: Cached computation results
- **processing_results**: Intermediate processing results
- **embedding_transformations**: Judge-specific transforms
- **positional_encodings**: Position-aware encodings
- **shared_embeddings**: Cross-segment embedding sharing
- **pattern_memory**: Successful pattern memory (deque, max 1000)
- **failure_patterns**: Failure pattern memory (deque, max 100)
- **adaptation_history**: Learning adaptation history
- **success_patterns**: Success pattern statistics
- **communication_channels**: Inter-segment communication state

## Integration Points

The cleanup system is automatically integrated at these strategic points:

1. **Node Creation**: Light cleanup every 1000 nodes (configurable)
2. **Pipeline Processing**: Light cleanup every 100 operations (configurable) 
3. **Batch Operations**: Light cleanup after >10 operations
4. **Brain Initialization**: Partial cleanup after loading segments
5. **Emergency Triggers**: Automatic cleanup when memory pressure is critical

## Safety Features

- **Force Protection**: Nuclear cleanup requires explicit force flag
- **Node Count Checks**: Additional safety for large node networks
- **Error Handling**: Graceful degradation if cleanup fails
- **Statistics Tracking**: Detailed reporting of all cleanup operations
- **Demo Mode Output**: Verbose logging when demo=True

## Performance Impact

- **Light Cleanup**: <1ms typically, minimal performance impact
- **Partial Cleanup**: 1-5ms typically, low performance impact
- **Aggressive Cleanup**: 5-20ms typically, moderate performance impact
- **Nuclear Cleanup**: 10-50ms typically, higher performance impact but complete optimization

## Best Practices

1. **Regular Maintenance**: Use light cleanup regularly during training
2. **Batch Processing**: Enable periodic cleanup for long-running batch operations
3. **Memory Monitoring**: Check memory status before intensive operations
4. **Emergency Preparation**: Configure emergency cleanup thresholds appropriately
5. **Segment Management**: Clean segments periodically during multi-segment operations
6. **Testing**: Use demo mode to monitor cleanup effectiveness

## Example: Complete Memory Management Workflow

```python
from BrainNexus import BrainNexus

# Initialize with cleanup monitoring
brain = BrainNexus(dimensions=4, demo=True)

# Configure periodic cleanup
brain.schedule_periodic_cleanup(cleanup_interval=50, cleanup_tier='light')

# During training/processing
for epoch in range(1000):
    # ... training operations ...
    
    # Check memory every 100 epochs
    if epoch % 100 == 0:
        status = brain.get_memory_status()
        if status['pressure_level'] in ['HIGH', 'CRITICAL']:
            print(f"Memory pressure {status['pressure_level']}: {status['recommendation']}")
            brain.emergency_cleanup()
    
    # ... more operations ...

# Final cleanup before saving
brain.cleanup_memory('aggressive')
```

This memory cleanup system ensures optimal performance and prevents memory bloat in long-running BrainNexus applications while providing fine-grained control over memory management strategies.
