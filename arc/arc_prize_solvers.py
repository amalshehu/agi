# arc_prize_solvers.py

import numpy as np
from collections import Counter, deque

# ———————————————————————————————————————————————————————————————
# Baseline helper solvers for identity puzzles
# ———————————————————————————————————————————————————————————————
def solve_symmetry(grid):
    mask = (grid != 0).astype(int)
    # horizontal
    if np.array_equal(mask, mask[::-1, :]):
        out = grid.copy()
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i,j] == 0:
                    out[i,j] = out[-i-1, j]
        return out
    # vertical
    if np.array_equal(mask, mask[:, ::-1]):
        out = grid.copy()
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i,j] == 0:
                    out[i,j] = out[i, -j-1]
        return out
    # main diagonal
    if grid.shape[0] == grid.shape[1] and np.array_equal(mask, mask.T):
        out = grid.copy()
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i,j] == 0:
                    out[i,j] = out[j, i]
        return out
    return None

def solve_histogram_fill(grid):
    out = grid.copy()
    freq = Counter(out.flatten().tolist())
    freq.pop(0, None)
    if not freq:
        return None
    fill = freq.most_common(1)[0][0]
    out[out == 0] = fill
    return out

def solve_connected_complement(grid):
    out = grid.copy()
    h, w = out.shape
    changed = True
    while changed:
        changed = False
        for i in range(h):
            for j in range(w):
                if out[i,j] == 0:
                    for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                        ni, nj = i+di, j+dj
                        if 0<=ni<h and 0<=nj<w and out[ni,nj] != 0:
                            out[i,j] = out[ni,nj]
                            changed = True
                            break
    return out

def solve_identity(grid, _target=None):
    """For identity puzzles (shape unchanged), try symmetry, connectivity, then histogram."""
    for fn in (solve_symmetry, solve_connected_complement, solve_histogram_fill):
        result = fn(grid)
        if result is not None:
            return result
    return grid.copy()

# ———————————————————————————————————————————————————————————————
# Uniform‐scale tiling solver
# ———————————————————————————————————————————————————————————————
def solve_uniform_scale(grid, scale):
    """
    Tile the input grid by the integer scale factor.
    E.g., a 2×2 block → scale=3 → repeated 3×3 to give 6×6.
    """
    return np.tile(grid, (scale, scale))

# ———————————————————————————————————————————————————————————————
# Non‐uniform solver (pad/crop)
# ———————————————————————————————————————————————————————————————
def solve_non_uniform(grid, target):
    """
    Center‐align the input within the output shape, 
    padding with the most common color from target.
    """
    grid = np.array(grid)
    tgt = np.array(target)
    h_in, w_in = grid.shape
    h_out, w_out = tgt.shape

    pad_h = (h_out - h_in) // 2
    pad_w = (w_out - w_in) // 2

    bg = Counter(tgt.flatten().tolist()).most_common(1)[0][0]
    out = np.full((h_out, w_out), bg, dtype=int)
    out[pad_h:pad_h+h_in, pad_w:pad_w+w_in] = grid
    return out
