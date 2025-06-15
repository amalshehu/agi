# kaggle_competition_notebook.py

"""
Kaggle ARC-Prize 2025 Notebook

Features:
- Load and inspect ARC-Prize JSON data
- Compute summary statistics
- Extract features (symmetry, connected components, color histogram)
- Cluster puzzles based on extracted features
- Baseline rule-based solver modules
- Baseline evaluation metrics
- HybridAgi training pipeline
"""

import json
import os
import sys
import pickle
import numpy as np
from collections import Counter, deque
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# Ensure project root is in PYTHONPATH so core/ can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---------------------------------------------------------------------

# Toggle flags
ENABLE_TRAINING = True

# Dataset directory
DATA_DIR = 'arc-prize-2025'

# ---- Utility Functions ----

def load_json(filename):
    """Load a JSON file from DATA_DIR"""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        return json.load(f)

# ---- Feature Extraction ----

def extract_color_histogram(grid):
    """Return a dict mapping color code to frequency"""
    flat = grid.flatten().tolist()
    return dict(Counter(flat))


def is_symmetric(mask, axis):
    """Check symmetry of a binary mask (1 for filled, 0 for empty)"""
    if axis == 'horizontal':
        return np.array_equal(mask, mask[::-1, :])
    if axis == 'vertical':
        return np.array_equal(mask, mask[:, ::-1])
    if axis == 'main_diag':
        return mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T)
    return False


def count_connected_components(grid):
    """Count connected components of non-zero cells using flood fill"""
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    comps = 0
    def neighbors(r, c):
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0 and not visited[i, j]:
                comps += 1
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    r, c = queue.popleft()
                    for nr, nc in neighbors(r, c):
                        if grid[nr, nc] == grid[i, j] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
    return comps


def extract_features(ch):
    """Generate numeric features for clustering from one challenge's first train input"""
    if not ch.get('train'):
        return None
    grid = np.array(ch['train'][0]['input'])
    h, w = grid.shape
    color_hist = extract_color_histogram(grid)
    mask = (grid != 0).astype(int)
    return np.array([
        h, w,
        len(color_hist),
        max(color_hist.values()),
        int(is_symmetric(mask, 'horizontal')),
        int(is_symmetric(mask, 'vertical')),
        int(is_symmetric(mask, 'main_diag')),
        count_connected_components(grid)
    ])

# ---- Clustering ----

def build_feature_matrix(ch_map):
    """Build feature matrix and return (features, ids)"""
    feats, ids = [], []
    for key, ch in ch_map.items():
        vec = extract_features(ch)
        if vec is not None:
            feats.append(vec)
            ids.append(key)
    return np.vstack(feats), ids


def cluster_puzzles(features, n_clusters=5):
    """Cluster with KMeans and return model, labels"""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(features)
    return model, labels

# ---- Baseline Solvers ----

def solve_symmetry(grid):
    """Fill zeros by mirroring along detected symmetry"""
    out = grid.copy()
    mask = (out != 0).astype(int)
    # Horizontal
    if is_symmetric(mask, 'horizontal'):
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i, j] == 0:
                    out[i, j] = out[-i - 1, j]
        return out
    # Vertical
    if is_symmetric(mask, 'vertical'):
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i, j] == 0:
                    out[i, j] = out[i, -j - 1]
        return out
    # Main diagonal
    if out.shape[0] == out.shape[1] and is_symmetric(mask, 'main_diag'):
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i, j] == 0:
                    out[i, j] = out[j, i]
        return out
    return None


def solve_histogram_fill(grid):
    """Fill zeros with most frequent non-zero color"""
    out = grid.copy()
    hist = Counter(out.flatten().tolist())
    hist.pop(0, None)
    if not hist:
        return None
    top = hist.most_common(1)[0][0]
    out[out == 0] = top
    return out


def solve_connected_complement(grid):
    """Iteratively fill zeros from adjacent non-zero neighbors"""
    out = grid.copy()
    h, w = out.shape
    changed = True
    while changed:
        changed = False
        for i in range(h):
            for j in range(w):
                if out[i, j] == 0:
                    for nr, nc in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                        if 0 <= nr < h and 0 <= nc < w and out[nr, nc] != 0:
                            out[i, j] = out[nr, nc]
                            changed = True
                            break
    return out


def solve_challenge(ch, label):
    """Route to solver based on cluster label"""
    grid = np.array(ch['train'][0]['input'])
    if label == 0:
        return solve_symmetry(grid)
    if label == 1:
        return solve_histogram_fill(grid)
    if label == 2:
        return solve_connected_complement(grid)
    # Fallback: return input unchanged
    return grid.copy()

# ---- Baseline Evaluation ----

def evaluate_baseline(ch_map, sol_map, labels, ids):
    total = solved = 0
    cluster_stats = {lbl: {'total':0,'solved':0} for lbl in set(labels)}
    for key, lbl in zip(ids, labels):
        sol = sol_map.get(key)
        if sol is None:
            continue
        if isinstance(sol, dict) and 'train' in sol:
            target = np.array(sol['train'][0]['output'])
        elif isinstance(sol, list) and sol and isinstance(sol[0], dict):
            target = np.array(sol[0]['output'])
        else:
            continue
        ch = ch_map.get(key)
        pred = solve_challenge(ch, lbl)
        if pred is None:
            continue
        correct = np.array_equal(pred, target)
        total += 1; solved += int(correct)
        cluster_stats[lbl]['total'] += 1
        cluster_stats[lbl]['solved'] += int(correct)
    overall = solved / total if total else 0
    print(f"Baseline overall accuracy: {solved}/{total} = {overall:.2%}")
    for lbl, stats in sorted(cluster_stats.items()):
        if stats['total']:
            acc = stats['solved']/stats['total']
            print(f" Cluster {lbl}: {stats['solved']}/{stats['total']} = {acc:.2%}")

# ---- HybridAgi Training Pipeline ----
from core.cognitive_agent import CognitiveAgent

def compute_reward(pred, target):
    """Reward = element-wise match fraction"""
    return np.mean((pred == target).astype(float))


def train_hybridagi_agent(ch_map, sol_map, epochs=3):
    agent = CognitiveAgent()
    for ep in range(1, epochs+1):
        losses = []
        for key, sol in sol_map.items():
            if isinstance(sol, dict) and 'train' in sol:
                target = sol['train'][0]['output']
            elif isinstance(sol, list) and sol and isinstance(sol[0], dict):
                target = sol[0]['output']
            else:
                continue
            agent.reset()
            agent.load_challenge(ch_map[key])
            agent.run_episode()
            pred = np.array(agent.get_output())
            reward = compute_reward(pred, np.array(target))
            agent.learn(reward)
            losses.append(1 - reward)
        avg_loss = np.mean(losses) if losses else float('nan')
        print(f"Epoch {ep}/{epochs} avg_loss={avg_loss:.4f}")
    return agent

# ---- Main Pipeline ----

def main():
    # Load data
    train_ch = load_json('arc-agi_training_challenges.json')
    train_sol = load_json('arc-agi_training_solutions.json')

    # Build consistent mappings based on 'id' fields when available
    if isinstance(train_ch, dict):
        ch_map = train_ch
    elif isinstance(train_ch, list) and all(isinstance(c, dict) and 'id' in c for c in train_ch):
        ch_map = {c['id']: c for c in train_ch}
    else:
        ch_map = {i: c for i, c in enumerate(train_ch)}

    if isinstance(train_sol, dict):
        sol_map = train_sol
    elif isinstance(train_sol, list) and all(isinstance(s, dict) and 'id' in s for s in train_sol):
        sol_map = {s['id']: s for s in train_sol}
    else:
        sol_map = {i: s for i, s in enumerate(train_sol)}

    # Cluster puzzles
    feats, ids = build_feature_matrix(ch_map)
    _, labels = cluster_puzzles(feats)
    print('Cluster distribution:', Counter(labels))

    # Evaluate baseline solvers
    evaluate_baseline(ch_map, sol_map, labels, ids)

    # Train HybridAgi agent
    if ENABLE_TRAINING:
        agent = train_hybridagi_agent(ch_map, sol_map)
        try:
            with open('hybridagi_agent.pkl', 'wb') as f:
                pickle.dump(agent, f)
            print('Agent trained and saved to hybridagi_agent.pkl')
        except Exception as e:
            print(f'Warning: failed to serialize agent ({e})')

if __name__ == '__main__':
    main()