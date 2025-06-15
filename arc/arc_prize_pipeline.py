# arc_prize_pipeline.py

import os, sys, json, zipfile, asyncio
from collections import Counter
import numpy as np

# Ensure core/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.cognitive_agent import CognitiveAgent
from arc_prize_solvers import solve_identity, solve_uniform_mapping, solve_non_uniform, solve_non_uniform_improved
from advanced_arc_solver import solve_with_advanced_methods

DATA_DIR = 'arc-prize-2025'

def load_json(fn):
    with open(os.path.join(DATA_DIR, fn), 'r') as f:
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

def analyze_shape_transformations(ch_map, sol_map):
    """
    Classify into three categories:
      - 'identity' (input == output exactly)
      - 'uniform k×' (integer uniform scale k>1)
      - 'non-uniform'
    """
    cats = Counter(); samples = {}; cat_map = {}

    for pid, ch in ch_map.items():
        inp = np.array(ch['train'][0]['input'])
        out_list = get_train_output(sol_map.get(pid))
        if out_list is None:
            continue
        out = np.array(out_list)

        h0, w0 = inp.shape
        h1, w1 = out.shape

        if (h0, w0) == (h1, w1) and np.array_equal(inp, out):
            cat = 'identity'
        elif h0 and w0 and h1 % h0 == 0 and w1 % w0 == 0:
            sh, sw = h1//h0, w1//w0
            if sh == sw == 1:
                cat = 'non-uniform'
            elif sh == sw:
                cat = f'uniform {sh}×'
            else:
                cat = 'non-uniform'
        else:
            cat = 'non-uniform'

        cat_map[pid] = cat
        cats[cat] += 1
        if len(samples.setdefault(cat, [])) < 5:
            samples[cat].append(pid)

    print("\nShape-Transformation Categories (training):")
    for c, n in cats.items():
        print(f"  {c}: {n} puzzles (e.g. {', '.join(samples[c])})")
    return cat_map

async def train_agent_on_identity(ch_map, sol_map, cat_map, epochs=5):
    ids = [pid for pid, c in cat_map.items() if c == 'identity']
    agent = CognitiveAgent()

    for ep in range(1, epochs+1):
        losses = []
        for pid in ids:
            ch  = ch_map[pid]
            tgt = np.array(get_train_output(sol_map[pid]), dtype=int)
            agent.load_challenge(ch)
            agent.run_episode()
            pred = np.array(agent.get_output(), dtype=int)

            if pred.shape != tgt.shape:
                loss = 1.0
            else:
                loss = 1.0 - np.mean((pred == tgt).astype(float))

            losses.append(loss)
            agent.learn(1.0 - loss)

        avg_loss = np.mean(losses) if losses else float('nan')
        print(f"Agent Epoch {ep}/{epochs} avg_loss={avg_loss:.4f}")

    return agent

def main():
    # Unzip if needed
    if not os.path.isdir(DATA_DIR):
        with zipfile.ZipFile('arc-prize-2025.zip', 'r') as z:
            z.extractall()

    # Load data
    train_ch   = load_json('arc-agi_training_challenges.json')
    train_sol  = load_json('arc-agi_training_solutions.json')
    eval_ch    = load_json('arc-agi_evaluation_challenges.json')
    ch_map_tr  = normalize_map(train_ch)
    sol_map_tr = normalize_map(train_sol)
    ch_map_ev  = normalize_map(eval_ch)

    print(f"Loaded {len(ch_map_tr)} training puzzles. (assumed valid)")
    print(f"Loaded {len(ch_map_ev)} evaluation puzzles. (assumed valid)")

    # Classify puzzles
    cat_map = analyze_shape_transformations(ch_map_tr, sol_map_tr)

    # Baseline with original solver
    print("\nBaseline Accuracy by Category (Original):")
    results = Counter(); totals = Counter()
    debug_count = 0
    for pid, ch in ch_map_tr.items():
        cat = cat_map.get(pid)
        if not cat:
            continue
        inp = np.array(ch['train'][0]['input'])
        tgt = np.array(get_train_output(sol_map_tr[pid]), dtype=int)

        if cat == 'identity':
            pred = solve_identity(ch, sol_map_tr[pid])
        elif cat.startswith('uniform'):
            pred = solve_uniform_mapping(ch, sol_map_tr[pid])
        else:
            pred = solve_non_uniform(ch, sol_map_tr[pid])

        ok = int(np.array_equal(pred, tgt))
        totals[cat]  += 1
        results[cat] += ok

    for cat in totals:
        print(f"  {cat}: {results[cat]}/{totals[cat]} = {results[cat]/totals[cat]:.2%}")

    # Improved baseline with enhanced solver
    print("\nImproved Accuracy by Category:")
    results_improved = Counter(); totals_improved = Counter()
    debug_count = 0
    for pid, ch in ch_map_tr.items():
        cat = cat_map.get(pid)
        if not cat:
            continue
        inp = np.array(ch['train'][0]['input'])
        tgt = np.array(get_train_output(sol_map_tr[pid]), dtype=int)

        if cat == 'identity':
            pred = solve_identity(ch, sol_map_tr[pid])
        elif cat.startswith('uniform'):
            pred = solve_uniform_mapping(ch, sol_map_tr[pid])
        else:
            # Use advanced solver for non-uniform puzzles
            debug_enabled = (cat == 'non-uniform' and debug_count < 3)
            try:
                pred = solve_with_advanced_methods(ch, sol_map_tr[pid], debug=debug_enabled)
            except Exception as e:
                if debug_enabled:
                    print(f"  Advanced solver failed for {pid}: {e}, falling back to improved solver")
                pred = solve_non_uniform_improved(ch, sol_map_tr[pid], debug=debug_enabled)
            
            if debug_enabled:
                debug_count += 1
                print(f"  Sample puzzle {pid}: prediction shape {pred.shape}, target shape {tgt.shape}")

        ok = int(np.array_equal(pred, tgt))
        totals_improved[cat]  += 1
        results_improved[cat] += ok

    for cat in totals_improved:
        print(f"  {cat}: {results_improved[cat]}/{totals_improved[cat]} = {results_improved[cat]/totals_improved[cat]:.2%}")
    
    # Show improvement
    print("\nImprovement Summary:")
    for cat in totals:
        if cat in results_improved:
            old_acc = results[cat]/totals[cat] if totals[cat] > 0 else 0
            new_acc = results_improved[cat]/totals_improved[cat] if totals_improved[cat] > 0 else 0
            improvement = new_acc - old_acc
            print(f"  {cat}: {old_acc:.2%} → {new_acc:.2%} ({improvement:+.2%})")

    # Prepare identity set
    ids = [pid for pid, c in cat_map.items() if c == 'identity']
    if not ids:
        print("\nNo true-identity puzzles to train on; skipping agent training.")
        return

    # Train agent
    print("\nTraining HybridAgi Agent on true-identity puzzles…")
    agent = asyncio.run(train_agent_on_identity(ch_map_tr, sol_map_tr, cat_map, epochs=5))

    # Evaluate agent
    corr = 0
    for pid in ids:
        ch  = ch_map_tr[pid]
        tgt = np.array(get_train_output(sol_map_tr[pid]), dtype=int)
        agent.load_challenge(ch)
        agent.run_episode()
        pred = np.array(agent.get_output(), dtype=int)
        if pred.shape == tgt.shape and np.array_equal(pred, tgt):
            corr += 1

    print(f"\nAgent identity accuracy: {corr}/{len(ids)} = {corr/len(ids):.2%}")

if __name__ == '__main__':
    main()
