#!/usr/bin/env python3

import sys
import os
import numpy as np

# Import the enhanced solver
from arc_prize_pipeline import NeuroSymbolicARCSolver, TENSORFLOW_AVAILABLE

def test_simple_transformation():
    print("Testing Enhanced NeuroSymbolic Solver...")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, skipping test")
        return
    
    # Create solver
    solver = NeuroSymbolicARCSolver()
    
    # Create a simple test case
    inp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    out = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int)  # Horizontal flip
    
    # Create a mock challenge
    challenge = {
        'train': [
            {
                'input': inp.tolist(),
                'output': out.tolist()
            }
        ]
    }
    
    # Create mock solution
    solution = {
        'train': [
            {
                'output': out.tolist()
            }
        ]
    }
    
    print(f"Input:\n{inp}")
    print(f"Expected output:\n{out}")
    
    # Test the solver
    try:
        result = solver.solve_puzzle(challenge, solution, debug=True)
        print(f"Solver result:\n{result}")
        
        if np.array_equal(result, out):
            print("✅ SUCCESS: Solver correctly identified and applied transformation!")
        else:
            print("❌ FAILURE: Solver result doesn't match expected output")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_object_detection():
    print("\nTesting Object Detection...")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, skipping test")
        return
        
    solver = NeuroSymbolicARCSolver()
    
    # Create a grid with distinct objects
    grid = np.array([
        [0, 0, 0, 1, 1],
        [0, 2, 0, 1, 1], 
        [0, 2, 0, 0, 0],
        [3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)
    
    print(f"Test grid:\n{grid}")
    
    try:
        objects = solver.extract_objects_advanced(grid)
        print(f"Detected {len(objects)} objects:")
        
        for i, obj in enumerate(objects):
            print(f"  Object {i+1}: color={obj['color']}, pos=({obj['x']},{obj['y']}), size={obj['w']}x{obj['h']}, shape={obj['shape']}")
            
        if len(objects) >= 3:  # Should detect at least 3 objects (colors 1, 2, 3)
            print("✅ SUCCESS: Object detection working correctly!")
        else:
            print("❌ FAILURE: Not enough objects detected")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_transformation()
    test_object_detection()
    print("\nTest completed.")
