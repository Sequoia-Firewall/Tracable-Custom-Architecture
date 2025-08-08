#!/usr/bin/env python3
"""
Test hypercube segment assignment for 2D case
"""

def test_hypercube_assignment():
    """Test that 2D generates 4 proper quadrants"""
    dimensions = 2
    
    print(f"Testing {dimensions}D hypercube vertex assignment:")
    print(f"Expected segments: 2^{dimensions} = {2**dimensions}")
    
    # Test the binary assignment logic
    for segment_id in range(2**dimensions):
        assignment = {}
        for dim in range(dimensions):
            # Use bit position to determine polarity
            polarity = 1 if (segment_id >> dim) & 1 else -1
            assignment[dim] = polarity
        
        print(f"Segment {segment_id}: {assignment}")
        
        # Calculate bounds for demo size (-100 to +100)
        segment_bounds = []
        for dim_idx in range(dimensions):
            base_min, base_max = -100.0, 100.0
            midpoint = 0.0
            
            if dim_idx in assignment:
                polarity = assignment[dim_idx]
                if polarity >= 0:
                    segment_bounds.append((midpoint, base_max))
                else:
                    segment_bounds.append((base_min, midpoint))
            else:
                segment_bounds.append((base_min, base_max))
        
        # Format bounds display
        dimension_labels = ['x', 'y']
        bounds_display = []
        for i, bounds in enumerate(segment_bounds):
            bounds_display.append(f"{dimension_labels[i]}:[{bounds[0]}, {bounds[1]}]")
        
        print(f"  Bounds: {' × '.join(bounds_display)}")

if __name__ == "__main__":
    test_hypercube_assignment()
