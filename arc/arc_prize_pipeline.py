# arc_prize_pipeline.py

import os
import json
import zipfile
from collections import Counter

# ———————————————————————————————————————————————————————————————
# Unzip dataset if needed
# ———————————————————————————————————————————————————————————————
ZIP_PATH = 'arc-prize-2025.zip'
DATA_DIR = 'arc-prize-2025'
if not os.path.isdir(DATA_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall()

# ———————————————————————————————————————————————————————————————
# JSON Loading & Normalization
# ———————————————————————————————————————————————————————————————
def load_json(fname):
    with open(os.path.join(DATA_DIR, fname), 'r') as f:
        return json.load(f)

def normalize_map(obj):
    """Turn list‐of‐objects (with 'id') or dict into {id: obj}"""
    if isinstance(obj, dict):
        return obj
    return {item['id']: item for item in obj}

# ———————————————————————————————————————————————————————————————
# Validation
# ———————————————————————————————————————————————————————————————
def validate_puzzle(ch):
    errs = []
    if not ch.get('train'):
        errs.append('missing train example')
    else:
        inp = ch['train'][0]['input']
        # rectangular?
        w0 = len(inp[0])
        if any(len(row) != w0 for row in inp):
            errs.append('non-rectangular input')
        # colors in [0,9]?
        colors = {c for row in inp for c in row}
        if any(c < 0 or c > 9 for c in colors):
            errs.append('out-of-range color')
    return errs

def validate_set(ch_map, name):
    """Validate every puzzle in ch_map; print summary for set name."""
    all_errs = {}
    for pid, ch in ch_map.items():
        errs = validate_puzzle(ch)
        if errs:
            all_errs[pid] = errs

    print(f"Loaded {len(ch_map)} {name} puzzles.")
    if all_errs:
        print(f"  Found {len(all_errs)} puzzles with issues in {name}:")
        by_err = Counter(err for errs in all_errs.values() for err in errs)
        for err, cnt in by_err.items():
            print(f"    {err}: {cnt}")
    else:
        print(f"  All {name} puzzles passed validation.")

# ———————————————————————————————————————————————————————————————
# Utilities for solution formatting
# ———————————————————————————————————————————————————————————————
def get_train_output(sol_entry):
    """
    Given a solutions‐map entry (could be dict or list), return the first
    training‐output grid (list of lists) or None if unavailable.
    """
    # dict w/ 'train'
    if isinstance(sol_entry, dict) and 'train' in sol_entry and sol_entry['train']:
        first = sol_entry['train'][0]
        if isinstance(first, dict) and 'output' in first:
            return first['output']
    # list of dicts w/ 'output'
    if isinstance(sol_entry, list) and sol_entry:
        first = sol_entry[0]
        if isinstance(first, dict) and 'output' in first:
            return first['output']
        # or list-of-lists directly
        if isinstance(first, list):
            return first
    return None

# ———————————————————————————————————————————————————————————————
# Shape‐Transformation Analysis (training only)
# ———————————————————————————————————————————————————————————————
def analyze_shape_transformations(ch_map, sol_map):
    cats = Counter()
    samples = {}

    for pid, ch in ch_map.items():
        sol_raw = sol_map.get(pid)
        out = get_train_output(sol_raw)
        if out is None:
            continue

        inp = ch['train'][0]['input']
        h_in, w_in = len(inp), len(inp[0])
        h_out, w_out = len(out), len(out[0])

        scale_h = h_out / h_in
        scale_w = w_out / w_in

        if scale_h == scale_w == 1:
            cat = 'identity'
        elif scale_h == scale_w and float(scale_h).is_integer():
            cat = f'uniform {int(scale_h)}×'
        else:
            cat = 'non-uniform'

        cats[cat] += 1
        if len(samples.get(cat, [])) < 5:
            samples.setdefault(cat, []).append(pid)

    print('\nShape‐Transformation Categories (training):')
    for cat, cnt in cats.items():
        smp_ids = ', '.join(samples[cat])
        print(f'  {cat}: {cnt} puzzles (e.g. {smp_ids})')

# ———————————————————————————————————————————————————————————————
# Main
# ———————————————————————————————————————————————————————————————
def main():
    # Load & normalize both sets
    train_ch  = load_json('arc-agi_training_challenges.json')
    train_sol = load_json('arc-agi_training_solutions.json')
    eval_ch   = load_json('arc-agi_evaluation_challenges.json')

    ch_map_train = normalize_map(train_ch)
    sol_map_train= normalize_map(train_sol)
    ch_map_eval  = normalize_map(eval_ch)

    # Validate both
    validate_set(ch_map_train, 'training')
    validate_set(ch_map_eval,   'evaluation')

    # Shape‐transformation analysis on training only
    analyze_shape_transformations(ch_map_train, sol_map_train)

if __name__ == '__main__':
    main()
