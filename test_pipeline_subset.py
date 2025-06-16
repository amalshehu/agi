#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
from collections import Counter

# Add arc directory to path
sys.path.append('arc')

from arc_prize_pipeline import NeuroSymbolicARCSolver, TENSORFLOW_AVAILABLE

def load_json(fn):
    with open(os.path.join('arc/arc-prize-2025', fn), 'r') as f:
        return json.load(f)

def normalize_map(obj):
    return obj if isinstance(obj, dict) else {item['id']: item for item in obj}

def get_train_output(sol):
    if isinstance(sol, dict) and sol.get('train'):
        out = sol['train'][0].get('output')
        if out is not None:
            return out
    if isinstance(sol, list) and sol:
        first = sol[0]
        if isinstance(first, dict) and 'output' in first:
            return first['output']
        if isinstance(first, list):
            return first
    return None

def test_non_uniform_subset():
    """Test enhanced solver on a subset of non-uniform puzzles"""
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, skipping test")
        return
    
    print("Loading data subset...")
    
    # Load data
    train_ch = load_json('arc-agi_training_challenges.json')
    train_sol = load_json('arc-agi_training_solutions.json')
    ch_map_tr = normalize_map(train_ch)
    sol_map_tr = normalize_map(train_sol)
    
    # Create neuro-symbolic solver
    neuro_solver = NeuroSymbolicARCSolver()
    
    # Test on first 20 puzzles to identify non-uniform ones
    test_count = 0
    correct_count = 0
    total_tested = 0
    
    for i, (pid, ch) in enumerate(ch_map_tr.items()):
        if test_count >= 10:  # Limit to 10 tests
            break
            
        try:
            inp = np.array(ch['train'][0]['input'])
            out_list = get_train_output(sol_map_tr.get(pid))
            if out_list is None:
                continue
            out = np.array(out_list)
            
            # Only test non-uniform puzzles (different shapes or complex transformations)
            if inp.shape != out.shape or not np.array_equal(inp, out):
                test_count += 1
                total_tested += 1
                
                print(f"\nTesting puzzle {pid} ({test_count}/10)")
                print(f"  Input shape: {inp.shape}, Output shape: {out.shape}")
                
                # Debug the solution data
                solution_data = sol_map_tr[pid]
                print(f"  Solution type: {type(solution_data)}")
                if isinstance(solution_data, dict):
                    print(f"  Solution keys: {solution_data.keys()}")
                elif isinstance(solution_data, list):
                    print(f"  Solution length: {len(solution_data)}")
                    if solution_data:
                        print(f"  First element type: {type(solution_data[0])}")
                
                # Test solver
                pred = neuro_solver.solve_puzzle(ch, solution_data, debug=True)
                
                # Check if correct
                if np.array_equal(pred, out):
                    correct_count += 1
                    print(f"  ✅ CORRECT!")
                else:
                    print(f"  ❌ INCORRECT")
                    
        except Exception as e:
            print(f"  Error processing {pid}: {e}")
            continue
    
    print(f"\nResults:")
    print(f"Total non-uniform puzzles tested: {total_tested}")
    print(f"Correct solutions: {correct_count}")
    print(f"Accuracy: {correct_count/total_tested*100:.1f}%" if total_tested > 0 else "No puzzles tested")

if __name__ == "__main__":
    test_non_uniform_subset()
