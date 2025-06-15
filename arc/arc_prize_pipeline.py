# arc_prize_pipeline.py

import os
import sys
import json
import zipfile
from collections import Counter
import numpy as np

# Make sure we can import from the root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arc_prize_solvers import solve  # advanced‐only solver

DATA_DIR = 'arc-prize-2025'

def load_json(filename):
    with open(os.path.join(DATA_DIR, filename), 'r') as f:
        return json.load(f)

def normalize_map(obj):
    return obj if isinstance(obj, dict) else {item['id']: item for item in obj}

def get_train_output(sol):
    # handle both dict and list formats
    if isinstance(sol, dict) and sol.get('train'):
        return sol['train'][0].get('output')
    if isinstance(sol, list) and sol:
        first = sol[0]
        if isinstance(first, dict) and 'output' in first:
            return first['output']
        if isinstance(first, list):
            return first
    return None

def validate_puzzle(ch):
    errs = []
    if 'train' not in ch or not ch['train']:
        errs.append('missing train example')
    else:
        inp = ch['train'][0]['input']
        h = len(inp)
        w = len(inp[0]) if h>0 else 0
        if any(len(row)!=w for row in inp):
            errs.append('non-rectangular input')
        colors = {c for row in inp for c in row}
        if any((c<0 or c>9) for c in colors):
            errs.append('out-of-range color')
    return errs

def validate_set(ch_map, name):
    all_errs = {}
    for pid, ch in ch_map.items():
        errs = validate_puzzle(ch)
        if errs:
            all_errs[pid] = errs
    print(f"Loaded {len(ch_map)} {name} puzzles.")
    if all_errs:
        print(f"  Found {len(all_errs)} puzzles with issues:")
        by_err = Counter(err for errs in all_errs.values() for err in errs)
        for err, cnt in by_err.items():
            print(f"    {err}: {cnt}")
    else:
        print(f"  All {name} puzzles passed validation.")

def main():
    # Unzip data if needed
    if not os.path.isdir(DATA_DIR):
        with zipfile.ZipFile('arc-prize-2025.zip','r') as z:
            z.extractall()

    # Load
    train_ch   = load_json('arc-agi_training_challenges.json')
    train_sol  = load_json('arc-agi_training_solutions.json')
    eval_ch    = load_json('arc-agi_evaluation_challenges.json')

    ch_map_tr  = normalize_map(train_ch)
    sol_map_tr = normalize_map(train_sol)
    ch_map_ev  = normalize_map(eval_ch)

    # Validate
    validate_set(ch_map_tr, 'training')
    validate_set(ch_map_ev,   'evaluation')

    # --- Advanced Ensemble Baseline on Training ---
    print("\nAdvanced Ensemble Baseline on TRAINING:")
    total, correct = 0, 0
    for pid, ch in ch_map_tr.items():
        sol = sol_map_tr.get(pid)
        out_list = get_train_output(sol)
        if out_list is None:
            continue
        tgt = np.array(out_list, dtype=int)
        pred = solve(ch, debug=(total<3))  # debug first 3
        ok = int(np.array_equal(pred, tgt))
        total  += 1
        correct += ok
    acc = correct/total if total else 0.0
    print(f"  {correct}/{total} = {acc:.2%}")

    # --- Advanced Ensemble Baseline on Evaluation ---
    print("\nAdvanced Ensemble Baseline on EVALUATION:")
    total_ev, correct_ev = 0, 0
    for pid, ch in ch_map_ev.items():
        # evaluation has no solutions; we skip accuracy but demonstrate output shapes
        pred = solve(ch, debug=(total_ev<3))
        print(f"  Puzzle {pid}: predicted shape {pred.shape}")
        total_ev += 1
    print(f"  Solved {total_ev} evaluation puzzles (accuracy N/A)")

if __name__ == '__main__':
    main()
