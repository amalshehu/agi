# arc_prize_solvers.py

import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Callable, Optional
import itertools

def extract_train_output(sol_entry):
    """Return the first train‐output grid or None."""
    if isinstance(sol_entry, dict) and sol_entry.get('train'):
        first = sol_entry['train'][0]
        if isinstance(first, dict) and 'output' in first:
            return first['output']
    if isinstance(sol_entry, list) and sol_entry:
        first = sol_entry[0]
        if isinstance(first, dict) and 'output' in first:
            return first['output']
        if isinstance(first, list):
            return first
    return None

def solve_identity(challenge, solution):
    """
    Identity puzzles: build input→output color map from the single train example,
    then apply it to the train input.
    """
    train_out = extract_train_output(solution)
    inp = np.array(challenge['train'][0]['input'], dtype=int)
    if train_out is None:
        return inp.copy()
    out = np.array(train_out, dtype=int)

    mapping = {}
    for (i,j), c in np.ndenumerate(inp):
        mapping[int(c)] = int(out[i,j])

    vals = list(mapping.values())
    if len(vals) != len(set(vals)):
        # fallback: fill zeros with most common nonzero color
        freq = Counter(out.flatten().tolist())
        freq.pop(0, None)
        fill = freq.most_common(1)[0][0] if freq else 0
        res = inp.copy()
        res[res == 0] = fill
        return res

    vec = np.vectorize(lambda c: mapping.get(int(c), c))
    return vec(inp)

def solve_uniform_mapping(challenge, solution):
    inp = np.array(challenge['train'][0]['input'], dtype=int)
    train_out = extract_train_output(solution)
    if train_out is None:
        return inp.copy()
    out = np.array(train_out, dtype=int)

    h0, w0 = inp.shape; H, W = out.shape
    sh, sw = H // h0, W // w0

    blocks = {}
    for i in range(h0):
        for j in range(w0):
            bi, bj = i*sh, j*sw
            blocks[(i,j)] = out[bi:bi+sh, bj:bj+sw]

    rows = [np.hstack([blocks[(i,j)] for j in range(w0)]) for i in range(h0)]
    return np.vstack(rows)

def solve_non_uniform(challenge, solution):
    inp = np.array(challenge['train'][0]['input'], dtype=int)
    train_out = extract_train_output(solution)
    if train_out is None:
        return inp.copy()
    out0 = np.array(train_out, dtype=int)

    h0, w0 = inp.shape; H0, W0 = out0.shape

    for i in range(H0-h0+1):
        for j in range(W0-w0+1):
            if np.array_equal(out0[i:i+h0, j:j+w0], inp):
                bg = Counter(out0.flatten().tolist()).most_common(1)[0][0]
                out = np.full((H0,W0), bg, dtype=int)
                out[i:i+h0, j:j+w0] = inp
                return out

    # Fallback: tile
    out = np.zeros((H0,W0), dtype=int)
    for i in range(H0):
        for j in range(W0):
            out[i,j] = inp[i % h0, j % w0]
    return out

# Feature extraction and transformation functions
def extract_features(grid: np.ndarray) -> Dict:
    """Extract features from a grid for pattern matching."""
    features = {}
    
    # Basic shape
    features['shape'] = grid.shape
    
    # Color histogram
    colors, counts = np.unique(grid.flatten(), return_counts=True)
    features['color_hist'] = dict(zip(colors.tolist(), counts.tolist()))
    features['num_colors'] = len(colors)
    features['most_common_color'] = colors[np.argmax(counts)]
    
    # Connected components for each color
    features['components'] = {}
    for color in colors:
        mask = (grid == color)
        # Simple connected component count (rough approximation)
        features['components'][int(color)] = np.sum(mask)
    
    # Symmetries
    features['is_symmetric_h'] = np.array_equal(grid, np.fliplr(grid))
    features['is_symmetric_v'] = np.array_equal(grid, np.flipud(grid))
    
    return features

def apply_color_mapping(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """Apply color mapping to grid."""
    result = grid.copy()
    for old_color, new_color in mapping.items():
        result[grid == old_color] = new_color
    return result

def learn_color_mapping(inp: np.ndarray, out: np.ndarray) -> Optional[Dict[int, int]]:
    """Learn color mapping from input to output if shapes match."""
    if inp.shape != out.shape:
        return None
    
    mapping = {}
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            old_c, new_c = int(inp[i,j]), int(out[i,j])
            if old_c in mapping and mapping[old_c] != new_c:
                return None  # Inconsistent mapping
            mapping[old_c] = new_c
    
    return mapping

def score_transformation(pred: np.ndarray, target: np.ndarray) -> float:
    """Score how well prediction matches target (1.0 = perfect, 0.0 = completely wrong)."""
    if pred.shape != target.shape:
        return 0.0
    return np.mean((pred == target).astype(float))

def generate_candidate_transforms() -> List[Tuple[str, Callable]]:
    """Generate list of candidate transformation functions."""
    transforms = [
        ("identity", lambda g: g),
        ("flip_horizontal", lambda g: np.fliplr(g)),
        ("flip_vertical", lambda g: np.flipud(g)),
        ("rotate_90", lambda g: np.rot90(g, k=1)),
        ("rotate_180", lambda g: np.rot90(g, k=2)),
        ("rotate_270", lambda g: np.rot90(g, k=3)),
        ("transpose", lambda g: g.T if g.shape[0] == g.shape[1] else g),
    ]
    return transforms

def detect_pattern_and_apply(inp: np.ndarray, out: np.ndarray) -> Optional[Callable]:
    """Detect patterns between input and output and return transformation function."""
    if inp.shape != out.shape:
        return None
    
    # Check for color remapping + transformations
    transforms = generate_candidate_transforms()
    
    for name, transform_func in transforms:
        try:
            transformed = transform_func(inp)
            if transformed.shape == out.shape:
                # Try to learn color mapping from transformed input to output
                color_map = learn_color_mapping(transformed, out)
                if color_map:
                    score = score_transformation(apply_color_mapping(transformed, color_map), out)
                    if score > 0.9:  # High confidence threshold
                        return lambda g: apply_color_mapping(transform_func(g), color_map)
        except:
            continue
    
    return None

def analyze_shapes_and_objects(grid: np.ndarray) -> Dict:
    """Analyze shapes and objects in the grid."""
    h, w = grid.shape
    objects = {}
    
    # Find rectangular regions for each color
    for color in np.unique(grid):
        if color == 0:  # Skip background
            continue
        
        mask = (grid == color)
        if not np.any(mask):
            continue
            
        # Find bounding box
        rows, cols = np.where(mask)
        if len(rows) > 0:
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            objects[int(color)] = {
                'bbox': (min_row, min_col, max_row + 1, max_col + 1),
                'area': len(rows),
                'shape': (max_row - min_row + 1, max_col - min_col + 1)
            }
    
    return {'objects': objects, 'num_objects': len(objects)}

def try_object_transformations(inp: np.ndarray, out: np.ndarray) -> Optional[Callable]:
    """Try object-level transformations."""
    if inp.shape != out.shape:
        return None
    
    inp_analysis = analyze_shapes_and_objects(inp)
    out_analysis = analyze_shapes_and_objects(out)
    
    # If same number of objects, try object mapping
    if inp_analysis['num_objects'] == out_analysis['num_objects']:
        # Simple object color mapping
        inp_colors = set(inp_analysis['objects'].keys())
        out_colors = set(out_analysis['objects'].keys())
        
        if len(inp_colors) == len(out_colors):
            # Try to find a consistent mapping
            inp_colors_sorted = sorted(inp_colors, key=lambda c: inp_analysis['objects'][c]['area'])
            out_colors_sorted = sorted(out_colors, key=lambda c: out_analysis['objects'][c]['area'])
            
            color_map = dict(zip(inp_colors_sorted, out_colors_sorted))
            
            # Add background mapping
            bg_inp = Counter(inp.flatten()).most_common(1)[0][0]
            bg_out = Counter(out.flatten()).most_common(1)[0][0]
            color_map[bg_inp] = bg_out
            
            mapped = apply_color_mapping(inp, color_map)
            if score_transformation(mapped, out) > 0.8:
                return lambda g: apply_color_mapping(g, color_map)
    
    return None

def try_shape_transforms(inp: np.ndarray, target_shape: Tuple[int, int]) -> List[Tuple[str, np.ndarray]]:
    """Try different ways to reshape input to target shape."""
    results = []
    h0, w0 = inp.shape
    h1, w1 = target_shape
    
    # Tiling
    if h1 >= h0 and w1 >= w0:
        tiled = np.zeros((h1, w1), dtype=inp.dtype)
        for i in range(h1):
            for j in range(w1):
                tiled[i,j] = inp[i % h0, j % w0]
        results.append(("tile", tiled))
    
    # Center placement with background
    if h1 >= h0 and w1 >= w0:
        bg_color = Counter(inp.flatten()).most_common(1)[0][0]
        centered = np.full((h1, w1), bg_color, dtype=inp.dtype)
        start_i, start_j = (h1 - h0) // 2, (w1 - w0) // 2
        centered[start_i:start_i+h0, start_j:start_j+w0] = inp
        results.append(("center", centered))
    
    # Top-left placement with background
    if h1 >= h0 and w1 >= w0:
        bg_color = Counter(inp.flatten()).most_common(1)[0][0]
        topleft = np.full((h1, w1), bg_color, dtype=inp.dtype)
        topleft[:h0, :w0] = inp
        results.append(("topleft", topleft))
    
    # Scaling (simple nearest neighbor)
    if h1 % h0 == 0 and w1 % w0 == 0:
        scale_h, scale_w = h1 // h0, w1 // w0
        if scale_h == scale_w:  # Uniform scaling
            scaled = np.repeat(np.repeat(inp, scale_h, axis=0), scale_w, axis=1)
            results.append((f"scale_{scale_h}x", scaled))
    
    # Cropping (if target is smaller)
    if h1 <= h0 and w1 <= w0:
        # Center crop
        start_i, start_j = (h0 - h1) // 2, (w0 - w1) // 2
        cropped = inp[start_i:start_i+h1, start_j:start_j+w1]
        results.append(("center_crop", cropped))
        
        # Top-left crop
        cropped = inp[:h1, :w1]
        results.append(("topleft_crop", cropped))
    
    return results

def try_advanced_transformations(inp: np.ndarray, out: np.ndarray) -> Optional[Callable]:
    """Try more advanced transformation patterns."""
    if inp.shape == out.shape:
        # Try combinations of transformations
        transforms = generate_candidate_transforms()
        
        # Try transform + color mapping combinations
        for t1_name, t1_func in transforms:
            try:
                transformed1 = t1_func(inp)
                for t2_name, t2_func in transforms:
                    if t1_name == t2_name and t1_name != "identity":
                        continue  # Skip same transform twice (except identity)
                    try:
                        transformed2 = t2_func(transformed1)
                        if transformed2.shape == out.shape:
                            # Try direct match
                            score = score_transformation(transformed2, out)
                            if score > 0.9:
                                def combined_transform(g):
                                    return t2_func(t1_func(g))
                                return combined_transform
                            
                            # Try with color mapping
                            color_map = learn_color_mapping(transformed2, out)
                            if color_map:
                                mapped = apply_color_mapping(transformed2, color_map)
                                score = score_transformation(mapped, out)
                                if score > 0.9:
                                    def combined_transform_with_color(g):
                                        return apply_color_mapping(t2_func(t1_func(g)), color_map)
                                    return combined_transform_with_color
                    except:
                        continue
            except:
                continue
    
    return None

def solve_non_uniform_improved(challenge, solution, debug=False):
    """Improved non-uniform solver with rule search and multiple examples."""
    train_examples = challenge.get('train', [])
    if not train_examples:
        return np.array([[0]], dtype=int)
    
    # Get all training input-output pairs
    train_pairs = []
    for i, example in enumerate(train_examples):
        inp = np.array(example['input'], dtype=int)
        
        # Get corresponding output
        if isinstance(solution, dict) and solution.get('train'):
            if i < len(solution['train']):
                out_data = solution['train'][i].get('output')
            else:
                out_data = solution['train'][0].get('output')  # Fallback to first
        elif isinstance(solution, list):
            if i < len(solution):
                out_data = solution[i] if isinstance(solution[i], list) else solution[i].get('output')
            else:
                out_data = solution[0] if isinstance(solution[0], list) else solution[0].get('output')
        else:
            out_data = None
            
        if out_data is not None:
            out = np.array(out_data, dtype=int)
            train_pairs.append((inp, out))
    
    if not train_pairs:
        return np.array([[0]], dtype=int)
    
    if debug and len(train_pairs) > 0:
        inp, out = train_pairs[0]
        print(f"\nDEBUG Non-uniform example:")
        print(f"Input shape: {inp.shape}, Output shape: {out.shape}")
        print(f"Input:\n{inp}")
        print(f"Output:\n{out}")
    
    # Try different transformation strategies
    best_transform = None
    best_score = 0.0
    best_name = ""
    
    # Strategy 1: Direct transformations for same-shape cases
    for inp, out in train_pairs:
        if inp.shape == out.shape:
            # Try pattern detection first (most sophisticated)
            pattern_transform = detect_pattern_and_apply(inp, out)
            if pattern_transform:
                try:
                    result = pattern_transform(inp)
                    score = score_transformation(result, out)
                    if score > best_score:
                        best_score = score
                        best_transform = pattern_transform
                        best_name = "pattern_detection"
                except:
                    pass
            
            # Try object-level transformations
            object_transform = try_object_transformations(inp, out)
            if object_transform:
                try:
                    result = object_transform(inp)
                    score = score_transformation(result, out)
                    if score > best_score:
                        best_score = score
                        best_transform = object_transform
                        best_name = "object_mapping"
                except:
                    pass
            
            # Try advanced transformations (combinations)
            advanced_transform = try_advanced_transformations(inp, out)
            if advanced_transform:
                try:
                    result = advanced_transform(inp)
                    score = score_transformation(result, out)
                    if score > best_score:
                        best_score = score
                        best_transform = advanced_transform
                        best_name = "advanced_combination"
                except:
                    pass
            
            # Try basic transformations
            transforms = generate_candidate_transforms()
            for name, transform_func in transforms:
                try:
                    transformed = transform_func(inp)
                    if transformed.shape == out.shape:
                        score = score_transformation(transformed, out)
                        if score > best_score:
                            best_score = score
                            best_transform = transform_func
                            best_name = name
                except:
                    continue
            
            # Try color mapping
            color_map = learn_color_mapping(inp, out)
            if color_map:
                mapped = apply_color_mapping(inp, color_map)
                score = score_transformation(mapped, out)
                if score > best_score:
                    best_score = score
                    best_transform = lambda g: apply_color_mapping(g, color_map)
                    best_name = "color_mapping"
    
    # Strategy 2: Shape transformations for different-shape cases
    if best_score < 1.0:
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                shape_transforms = try_shape_transforms(inp, out.shape)
                for name, transformed in shape_transforms:
                    score = score_transformation(transformed, out)
                    if score > best_score:
                        best_score = score
                        best_transform = lambda g: try_shape_transforms(g, out.shape)[0][1]  # Use first method
                        best_name = name
    
    if debug:
        print(f"Best transformation: {best_name} (score: {best_score:.3f})")
    
    # Apply best transformation to first input
    first_input = train_pairs[0][0]
    if best_transform and best_score > 0.0:
        try:
            result = best_transform(first_input)
            return result
        except:
            pass
    
    # Fallback to original logic
    return solve_non_uniform(challenge, solution)
