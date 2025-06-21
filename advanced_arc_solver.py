# advanced_arc_solver.py
"""
Advanced ARC solver combining winning strategies from ARC Prize 2024:
1. Deep Learning-Guided Program Synthesis
2. Test-Time Training
3. Domain-Specific Language (DSL) Program Synthesis
4. Ensemble Methods
"""

import numpy as np
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Callable, Optional, Any, Set
import itertools
from abc import ABC, abstractmethod
from scipy import ndimage
from sklearn.cluster import DBSCAN
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Domain-Specific Language for ARC transformations
class DSLOperation(ABC):
    """Abstract base class for DSL operations."""
    
    @abstractmethod
    def apply(self, grid: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass

class IdentityOp(DSLOperation):
    def apply(self, grid: np.ndarray) -> np.ndarray:
        return grid.copy()
    
    def description(self) -> str:
        return "identity"

class FlipHorizontalOp(DSLOperation):
    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)
    
    def description(self) -> str:
        return "flip_horizontal"

class FlipVerticalOp(DSLOperation):
    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)
    
    def description(self) -> str:
        return "flip_vertical"

class RotateOp(DSLOperation):
    def __init__(self, k: int):
        self.k = k
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=self.k)
    
    def description(self) -> str:
        return f"rotate_{self.k * 90}deg"

class ColorMapOp(DSLOperation):
    def __init__(self, mapping: Dict[int, int]):
        self.mapping = mapping
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for old_color, new_color in self.mapping.items():
            result[grid == old_color] = new_color
        return result
    
    def description(self) -> str:
        return f"color_map_{self.mapping}"

class ScaleOp(DSLOperation):
    def __init__(self, scale_factor: int):
        self.scale_factor = scale_factor
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(grid, self.scale_factor, axis=0), self.scale_factor, axis=1)
    
    def description(self) -> str:
        return f"scale_{self.scale_factor}x"

class CropOp(DSLOperation):
    def __init__(self, top: int, left: int, height: int, width: int):
        self.top, self.left, self.height, self.width = top, left, height, width
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        return grid[self.top:self.top+self.height, self.left:self.left+self.width]
    
    def description(self) -> str:
        return f"crop_{self.top}_{self.left}_{self.height}_{self.width}"

class PadOp(DSLOperation):
    def __init__(self, target_shape: Tuple[int, int], fill_value: int = 0, mode: str = "center"):
        self.target_shape = target_shape
        self.fill_value = fill_value
        self.mode = mode
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        target_h, target_w = self.target_shape
        
        if h >= target_h and w >= target_w:
            return grid[:target_h, :target_w]
        
        result = np.full(self.target_shape, self.fill_value, dtype=grid.dtype)
        
        if self.mode == "center":
            start_h = (target_h - h) // 2
            start_w = (target_w - w) // 2
        elif self.mode == "topleft":
            start_h, start_w = 0, 0
        else:  # bottomright
            start_h = target_h - h
            start_w = target_w - w
        
        result[start_h:start_h+h, start_w:start_w+w] = grid
        return result
    
    def description(self) -> str:
        return f"pad_to_{self.target_shape}_{self.mode}"

class ObjectFillOp(DSLOperation):
    """Fill objects with specific colors based on patterns."""
    def __init__(self, target_color: int, source_color: int = None):
        self.target_color = target_color
        self.source_color = source_color
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        objects = self._find_objects(grid)
        
        for obj_mask in objects:
            if self.source_color is None or np.any(grid[obj_mask] == self.source_color):
                result[obj_mask] = self.target_color
        
        return result
    
    def _find_objects(self, grid: np.ndarray) -> List[np.ndarray]:
        """Find connected components (objects) in the grid."""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    obj_mask = np.zeros_like(grid, dtype=bool)
                    self._flood_fill(grid, visited, obj_mask, i, j, grid[i, j])
                    if np.sum(obj_mask) > 0:
                        objects.append(obj_mask)
        
        return objects
    
    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, obj_mask: np.ndarray, 
                   r: int, c: int, color: int):
        """Flood fill to find connected component."""
        if (r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1] or 
            visited[r, c] or grid[r, c] != color):
            return
        
        visited[r, c] = True
        obj_mask[r, c] = True
        
        # 4-connectivity
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            self._flood_fill(grid, visited, obj_mask, r + dr, c + dc, color)
    
    def description(self) -> str:
        return f"fill_objects_{self.target_color}"

class PatternExtendOp(DSLOperation):
    """Extend patterns spatially."""
    def __init__(self, direction: str, factor: int = 2):
        self.direction = direction  # 'horizontal', 'vertical', 'both'
        self.factor = factor
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        
        if self.direction == 'horizontal':
            result = np.zeros((h, w * self.factor), dtype=grid.dtype)
            for i in range(self.factor):
                result[:, i*w:(i+1)*w] = grid
        elif self.direction == 'vertical':
            result = np.zeros((h * self.factor, w), dtype=grid.dtype)
            for i in range(self.factor):
                result[i*h:(i+1)*h, :] = grid
        else:  # both
            result = np.zeros((h * self.factor, w * self.factor), dtype=grid.dtype)
            for i in range(self.factor):
                for j in range(self.factor):
                    result[i*h:(i+1)*h, j*w:(j+1)*w] = grid
        
        return result
    
    def description(self) -> str:
        return f"extend_{self.direction}_{self.factor}x"

class ConditionalOp(DSLOperation):
    """Apply transformation conditionally based on pattern matching."""
    def __init__(self, condition_color: int, true_op: DSLOperation, false_op: DSLOperation):
        self.condition_color = condition_color
        self.true_op = true_op
        self.false_op = false_op
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        if np.any(grid == self.condition_color):
            return self.true_op.apply(grid)
        else:
            return self.false_op.apply(grid)
    
    def description(self) -> str:
        return f"if_has_{self.condition_color}_then_{self.true_op.description()}_else_{self.false_op.description()}"

# MASSIVE OPERATION VOCABULARY FOR ARC PRIZE 2025

class ConnectDotsOp(DSLOperation):
    """Connect dots with lines."""
    def __init__(self, color: int = 1):
        self.color = color
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        coords = np.where(grid == self.color)
        points = list(zip(coords[0], coords[1]))
        
        if len(points) >= 2:
            # Connect consecutive points
            for i in range(len(points) - 1):
                self._draw_line(result, points[i], points[i+1], self.color)
        
        return result
    
    def _draw_line(self, grid: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: int):
        """Draw line between two points."""
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2:  # Vertical line
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= y < grid.shape[1]:
                    grid[x1, y] = color
        elif y1 == y2:  # Horizontal line
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= x < grid.shape[0]:
                    grid[x, y1] = color
    
    def description(self) -> str:
        return f"connect_dots_{self.color}"

class FillShapeOp(DSLOperation):
    """Fill enclosed shapes."""
    def __init__(self, target_color: int, boundary_color: int):
        self.target_color = target_color
        self.boundary_color = boundary_color
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        
        # Find enclosed regions and fill them
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                if grid[i, j] == 0:  # Empty space
                    if self._is_enclosed(grid, i, j):
                        result[i, j] = self.target_color
        
        return result
    
    def _is_enclosed(self, grid: np.ndarray, r: int, c: int) -> bool:
        """Check if position is enclosed by boundary."""
        # Simple heuristic: check if surrounded by non-zero values
        neighbors = [
            grid[r-1, c], grid[r+1, c], grid[r, c-1], grid[r, c+1],
            grid[r-1, c-1], grid[r-1, c+1], grid[r+1, c-1], grid[r+1, c+1]
        ]
        return sum(n != 0 for n in neighbors) >= 6
    
    def description(self) -> str:
        return f"fill_shape_{self.target_color}_{self.boundary_color}"

class MirrorOp(DSLOperation):
    """Mirror patterns across axes."""
    def __init__(self, axis: str, position: str = "center"):
        self.axis = axis  # 'horizontal', 'vertical', 'diagonal'
        self.position = position  # 'center', 'edge'
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        
        if self.axis == 'horizontal':
            if self.position == "center":
                mid = w // 2
                result = grid.copy()
                result[:, mid:] = np.fliplr(grid[:, :mid])
                return result
            else:
                return np.concatenate([grid, np.fliplr(grid)], axis=1)
        
        elif self.axis == 'vertical':
            if self.position == "center":
                mid = h // 2
                result = grid.copy()
                result[mid:, :] = np.flipud(grid[:mid, :])
                return result
            else:
                return np.concatenate([grid, np.flipud(grid)], axis=0)
        
        elif self.axis == 'diagonal':
            return np.transpose(grid)
        
        return grid
    
    def description(self) -> str:
        return f"mirror_{self.axis}_{self.position}"

class FrameOp(DSLOperation):
    """Add frames or borders."""
    def __init__(self, color: int, thickness: int = 1):
        self.color = color
        self.thickness = thickness
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        result = np.full((h + 2*self.thickness, w + 2*self.thickness), self.color, dtype=grid.dtype)
        result[self.thickness:h+self.thickness, self.thickness:w+self.thickness] = grid
        return result
    
    def description(self) -> str:
        return f"frame_{self.color}_{self.thickness}"

class GravityOp(DSLOperation):
    """Apply gravity to objects."""
    def __init__(self, direction: str = "down"):
        self.direction = direction
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        
        if self.direction == "down":
            for j in range(grid.shape[1]):
                col = result[:, j]
                non_zero = col[col != 0]
                zeros = col[col == 0]
                result[:, j] = np.concatenate([zeros, non_zero])
        
        elif self.direction == "up":
            for j in range(grid.shape[1]):
                col = result[:, j]
                non_zero = col[col != 0]
                zeros = col[col == 0]
                result[:, j] = np.concatenate([non_zero, zeros])
        
        elif self.direction == "left":
            for i in range(grid.shape[0]):
                row = result[i, :]
                non_zero = row[row != 0]
                zeros = row[row == 0]
                result[i, :] = np.concatenate([non_zero, zeros])
        
        elif self.direction == "right":
            for i in range(grid.shape[0]):
                row = result[i, :]
                non_zero = row[row != 0]
                zeros = row[row == 0]
                result[i, :] = np.concatenate([zeros, non_zero])
        
        return result
    
    def description(self) -> str:
        return f"gravity_{self.direction}"

class SymmetryOp(DSLOperation):
    """Complete symmetric patterns."""
    def __init__(self, axis: str):
        self.axis = axis
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        h, w = grid.shape
        
        if self.axis == "vertical":
            mid = w // 2
            for i in range(h):
                for j in range(mid):
                    if result[i, j] != 0:
                        result[i, w-1-j] = result[i, j]
                    elif result[i, w-1-j] != 0:
                        result[i, j] = result[i, w-1-j]
        
        elif self.axis == "horizontal":
            mid = h // 2
            for i in range(mid):
                for j in range(w):
                    if result[i, j] != 0:
                        result[h-1-i, j] = result[i, j]
                    elif result[h-1-i, j] != 0:
                        result[i, j] = result[h-1-i, j]
        
        return result
    
    def description(self) -> str:
        return f"symmetry_{self.axis}"

class StackOp(DSLOperation):
    """Stack objects in direction."""
    def __init__(self, direction: str, spacing: int = 0):
        self.direction = direction
        self.spacing = spacing
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        objects = self._find_objects(grid)
        if len(objects) < 2:
            return grid
        
        result = np.zeros_like(grid)
        
        if self.direction == "vertical":
            y_pos = 0
            for obj_mask in objects:
                obj_bounds = self._get_bounds(obj_mask)
                obj_height = obj_bounds[2] - obj_bounds[0] + 1
                
                # Place object at current y position
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        if obj_mask[i, j] and y_pos + i - obj_bounds[0] < result.shape[0]:
                            result[y_pos + i - obj_bounds[0], j] = grid[i, j]
                
                y_pos += obj_height + self.spacing
        
        return result
    
    def _find_objects(self, grid: np.ndarray) -> List[np.ndarray]:
        """Find connected components."""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    obj_mask = np.zeros_like(grid, dtype=bool)
                    self._flood_fill(grid, visited, obj_mask, i, j, grid[i, j])
                    objects.append(obj_mask)
        
        return objects
    
    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, obj_mask: np.ndarray,
                   r: int, c: int, color: int):
        if (r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1] or 
            visited[r, c] or grid[r, c] != color):
            return
        
        visited[r, c] = True
        obj_mask[r, c] = True
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            self._flood_fill(grid, visited, obj_mask, r + dr, c + dc, color)
    
    def _get_bounds(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of mask."""
        rows, cols = np.where(mask)
        return (rows.min(), cols.min(), rows.max(), cols.max())
    
    def description(self) -> str:
        return f"stack_{self.direction}_{self.spacing}"

class RepeatingPatternOp(DSLOperation):
    """Create repeating patterns."""
    def __init__(self, pattern_size: int, direction: str = "both"):
        self.pattern_size = pattern_size
        self.direction = direction
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        
        if self.direction == "horizontal":
            pattern = grid[:, :self.pattern_size]
            result = np.zeros((h, w), dtype=grid.dtype)
            for i in range(0, w, self.pattern_size):
                end_col = min(i + self.pattern_size, w)
                result[:, i:end_col] = pattern[:, :end_col-i]
            return result
        
        elif self.direction == "vertical":
            pattern = grid[:self.pattern_size, :]
            result = np.zeros((h, w), dtype=grid.dtype)
            for i in range(0, h, self.pattern_size):
                end_row = min(i + self.pattern_size, h)
                result[i:end_row, :] = pattern[:end_row-i, :]
            return result
        
        else:  # both
            pattern = grid[:self.pattern_size, :self.pattern_size]
            result = np.zeros((h, w), dtype=grid.dtype)
            for i in range(0, h, self.pattern_size):
                for j in range(0, w, self.pattern_size):
                    end_row = min(i + self.pattern_size, h)
                    end_col = min(j + self.pattern_size, w)
                    result[i:end_row, j:end_col] = pattern[:end_row-i, :end_col-j]
            return result
    
    def description(self) -> str:
        return f"repeat_{self.pattern_size}_{self.direction}"

class DSLProgram:
    """A sequence of DSL operations."""
    
    def __init__(self, operations: List[DSLOperation]):
        self.operations = operations
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute all operations in sequence."""
        result = grid.copy()
        for op in self.operations:
            result = op.apply(result)
        return result
    
    def description(self) -> str:
        return " -> ".join([op.description() for op in self.operations])
    
    def __len__(self) -> int:
        return len(self.operations)

class ProgramSynthesizer:
    """Synthesizes DSL programs to solve ARC tasks."""
    
    def __init__(self, max_program_length: int = 4):
        self.max_program_length = max_program_length
        self.primitive_ops = self._create_primitive_operations()
    
    def _create_primitive_operations(self) -> List[DSLOperation]:
        """Create comprehensive primitive operations."""
        primitives = [
            IdentityOp(),
            FlipHorizontalOp(),
            FlipVerticalOp(),
            RotateOp(1), RotateOp(2), RotateOp(3),
        ]
        
        # Add color mapping operations for common patterns
        common_mappings = [
            {0: 1, 1: 0},  # swap 0 and 1
            {0: 2, 2: 0},  # swap 0 and 2
            {1: 2, 2: 1},  # swap 1 and 2
            {0: 0, 1: 2, 2: 1},  # keep 0, swap 1 and 2
            {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},  # all to background
        ]
        for mapping in common_mappings:
            primitives.append(ColorMapOp(mapping))
        
        # Add scaling operations
        for scale in [2, 3, 4, 5]:
            primitives.append(ScaleOp(scale))
        
        # Add object manipulation operations
        for color in range(1, 10):  # ARC uses colors 0-9
            primitives.append(ObjectFillOp(color))
            primitives.append(ObjectFillOp(0, color))  # Fill objects of color with background
        
        # Add pattern extension operations
        for direction in ['horizontal', 'vertical']:
            for factor in [2, 3]:
                primitives.append(PatternExtendOp(direction, factor))
        
        # Add MASSIVE vocabulary for ARC Prize 2025
        # Connection and line operations
        for color in [1, 2, 3, 4, 5]:
            primitives.append(ConnectDotsOp(color))
        
        # Shape filling operations
        for target_color in [1, 2, 3, 4]:
            for boundary_color in [1, 2, 3, 4]:
                if target_color != boundary_color:
                    primitives.append(FillShapeOp(target_color, boundary_color))
        
        # Mirror operations
        for axis in ['horizontal', 'vertical', 'diagonal']:
            for position in ['center', 'edge']:
                primitives.append(MirrorOp(axis, position))
        
        # Frame operations
        for color in [1, 2, 3]:
            for thickness in [1, 2]:
                primitives.append(FrameOp(color, thickness))
        
        # Gravity operations
        for direction in ['up', 'down', 'left', 'right']:
            primitives.append(GravityOp(direction))
        
        # Symmetry operations
        for axis in ['horizontal', 'vertical']:
            primitives.append(SymmetryOp(axis))
        
        # Stack operations
        for direction in ['vertical', 'horizontal']:
            for spacing in [0, 1]:
                primitives.append(StackOp(direction, spacing))
        
        # Repeating pattern operations
        for size in [1, 2, 3, 4]:
            for direction in ['horizontal', 'vertical', 'both']:
                primitives.append(RepeatingPatternOp(size, direction))
        
        return primitives
    
    def synthesize_program(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                          max_attempts: int = 1000) -> Optional[DSLProgram]:
        """Synthesize using beam search with parallel execution for ARC Prize 2025."""
        
        # Method 1: Ultra-fast single operation check
        result = self._parallel_single_op_search(train_examples)
        if result:
            return result
        
        # Method 2: Template-based synthesis 
        template_program = self._try_template_synthesis(train_examples)
        if template_program:
            return template_program
        
        # Method 3: BEAM SEARCH with parallel execution
        return self._beam_search_synthesis(train_examples, max_attempts)
    
    def _parallel_single_op_search(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[DSLProgram]:
        """Parallel search through single operations."""
        def test_op(op):
            program = DSLProgram([op])
            if self._test_program(program, train_examples):
                return program
            return None
        
        # Split operations into chunks for parallel processing
        chunk_size = max(1, len(self.primitive_ops) // 8)
        chunks = [self.primitive_ops[i:i+chunk_size] for i in range(0, len(self.primitive_ops), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for chunk in chunks:
                for op in chunk:
                    futures.append(executor.submit(test_op, op))
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    return result
        
        return None
    
    def _beam_search_synthesis(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                              max_attempts: int) -> Optional[DSLProgram]:
        """Beam search for program synthesis with neural-guided heuristics."""
        
        beam_width = 50  # Keep top 50 candidates
        max_depth = 3
        
        # Initialize beam with smart operations
        smart_ops = self._select_smart_operations(train_examples)
        
        # Priority queue: (negative_score, program_id, program)
        beam = []
        program_counter = 0
        
        # Add single operations to beam
        for op in smart_ops[:beam_width]:
            program = DSLProgram([op])
            score = self._score_program(program, train_examples)
            heapq.heappush(beam, (-score, program_counter, program))
            program_counter += 1
        
        attempts = 0
        
        for depth in range(1, max_depth + 1):
            new_beam = []
            
            # Expand each program in current beam
            current_beam = list(beam)
            beam = []
            
            def expand_program(beam_item):
                score, prog_id, program = beam_item
                candidates = []
                
                for op in smart_ops:
                    if len(program.operations) < self.max_program_length:
                        new_program = DSLProgram(program.operations + [op])
                        
                        # Quick test if this could work
                        new_score = self._score_program(new_program, train_examples)
                        
                        if new_score >= 0.9:  # High threshold for exact matches
                            if self._test_program(new_program, train_examples):
                                return new_program
                        
                        nonlocal program_counter
                        candidates.append((-new_score, program_counter, new_program))
                        program_counter += 1
                
                return candidates
            
            # Parallel expansion
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(expand_program, beam_item) for beam_item in current_beam]
                
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, DSLProgram):
                        return result  # Found exact solution
                    elif isinstance(result, list):
                        new_beam.extend(result)
                    
                    attempts += len(smart_ops)
                    if attempts >= max_attempts:
                        break
            
            # Keep only top candidates
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]
            
            if attempts >= max_attempts:
                break
        
        # Return best program found
        if beam:
            return beam[0][2]  # Return the program (third element)
        
        return None
    
    def _score_program(self, program: DSLProgram, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Score a program based on how well it fits the examples."""
        total_score = 0
        valid_examples = 0
        
        for inp, expected_out in train_examples:
            try:
                actual_out = program.execute(inp)
                
                if actual_out.shape == expected_out.shape:
                    similarity = np.mean(actual_out == expected_out)
                    total_score += similarity
                    valid_examples += 1
                else:
                    # Penalize shape mismatches but don't completely discard
                    shape_penalty = 0.1
                    total_score += shape_penalty
                    valid_examples += 1
            except:
                # Execution failed, give minimal score
                total_score += 0.01
                valid_examples += 1
        
        return total_score / valid_examples if valid_examples > 0 else 0
    
    def _try_template_synthesis(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[DSLProgram]:
        """Try synthesis based on common ARC templates."""
        
        # Template 1: Scaling + color change
        inp_shapes = [inp.shape for inp, _ in train_examples]
        out_shapes = [out.shape for _, out in train_examples]
        
        if all(inp == inp_shapes[0] for inp in inp_shapes) and all(out == out_shapes[0] for out in out_shapes):
            if out_shapes[0][0] % inp_shapes[0][0] == 0 and out_shapes[0][1] % inp_shapes[0][1] == 0:
                scale = out_shapes[0][0] // inp_shapes[0][0]
                if scale > 1:
                    # Try scale + color mapping
                    scale_op = ScaleOp(scale)
                    for mapping in [{0: 1, 1: 0}, {1: 2, 2: 1}, {0: 2, 2: 0}]:
                        color_op = ColorMapOp(mapping)
                        program = DSLProgram([scale_op, color_op])
                        if self._test_program(program, train_examples):
                            return program
                        program = DSLProgram([color_op, scale_op])
                        if self._test_program(program, train_examples):
                            return program
        
        # Template 2: Object filling patterns
        for inp, out in train_examples:
            inp_colors = set(inp.flatten())
            out_colors = set(out.flatten())
            
            if len(out_colors) < len(inp_colors):
                # Objects being removed/filled
                for target_color in out_colors:
                    fill_op = ObjectFillOp(target_color)
                    program = DSLProgram([fill_op])
                    if self._test_program(program, train_examples):
                        return program
        
        # Template 3: Rotation + transformation
        for rot_k in [1, 2, 3]:
            rot_op = RotateOp(rot_k)
            for flip_op in [FlipHorizontalOp(), FlipVerticalOp()]:
                program = DSLProgram([rot_op, flip_op])
                if self._test_program(program, train_examples):
                    return program
                program = DSLProgram([flip_op, rot_op])
                if self._test_program(program, train_examples):
                    return program
        
        return None
    
    def _select_smart_operations(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[DSLOperation]:
        """Select operations likely to be useful for these examples."""
        smart_ops = [IdentityOp()]
        
        # Analyze examples to select relevant operations
        for inp, out in train_examples:
            # Check for shape changes
            if inp.shape != out.shape:
                # Likely scaling
                if out.shape[0] % inp.shape[0] == 0:
                    scale = out.shape[0] // inp.shape[0]
                    if scale <= 5:
                        smart_ops.append(ScaleOp(scale))
                
                # Pattern extension
                smart_ops.extend([
                    PatternExtendOp('horizontal', 2),
                    PatternExtendOp('vertical', 2)
                ])
            
            # Check for color changes
            inp_colors = set(inp.flatten())
            out_colors = set(out.flatten())
            
            if inp_colors != out_colors:
                # Add color operations
                smart_ops.extend([
                    ColorMapOp({0: 1, 1: 0}),
                    ColorMapOp({1: 2, 2: 1}),
                    ObjectFillOp(1), ObjectFillOp(2), ObjectFillOp(0)
                ])
            
            # Check for geometric transformations
            if np.array_equal(inp, np.fliplr(out)) or np.array_equal(np.fliplr(inp), out):
                smart_ops.append(FlipHorizontalOp())
            if np.array_equal(inp, np.flipud(out)) or np.array_equal(np.flipud(inp), out):
                smart_ops.append(FlipVerticalOp())
            
            for k in [1, 2, 3]:
                if np.array_equal(inp, np.rot90(out, k)) or np.array_equal(np.rot90(inp, k), out):
                    smart_ops.append(RotateOp(k))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ops = []
        for op in smart_ops:
            op_id = op.description()
            if op_id not in seen:
                seen.add(op_id)
                unique_ops.append(op)
        
        return unique_ops[:20]  # Limit to most promising operations
    
    def _test_program(self, program: DSLProgram, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Test if a program correctly transforms all input-output examples."""
        for inp, expected_out in examples:
            try:
                actual_out = program.execute(inp)
                if not np.array_equal(actual_out, expected_out):
                    return False
            except:
                return False
        return True

class PatternAnalyzer:
    """Advanced pattern analysis for ARC tasks."""
    
    @staticmethod
    def analyze_task(train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Comprehensive pattern analysis."""
        analysis = {
            'shape_changes': [],
            'color_changes': [],
            'symmetries': [],
            'scaling_factors': [],
            'common_objects': [],
            'object_transformations': [],
            'spatial_patterns': [],
            'repetition_patterns': []
        }
        
        for inp, out in train_examples:
            # Basic analysis
            inp_shape, out_shape = inp.shape, out.shape
            analysis['shape_changes'].append((inp_shape, out_shape))
            
            inp_colors = set(inp.flatten())
            out_colors = set(out.flatten())
            analysis['color_changes'].append((inp_colors, out_colors))
            
            # Object analysis
            inp_objects = PatternAnalyzer._find_objects(inp)
            out_objects = PatternAnalyzer._find_objects(out)
            
            obj_transform = {
                'input_count': len(inp_objects),
                'output_count': len(out_objects),
                'size_changes': [],
                'position_changes': []
            }
            
            # Analyze object transformations
            for i, inp_obj in enumerate(inp_objects):
                inp_size = np.sum(inp_obj)
                inp_center = PatternAnalyzer._get_center_of_mass(inp_obj)
                
                # Find closest output object
                best_match = None
                best_distance = float('inf')
                
                for j, out_obj in enumerate(out_objects):
                    out_center = PatternAnalyzer._get_center_of_mass(out_obj)
                    distance = np.linalg.norm(np.array(inp_center) - np.array(out_center))
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = j
                
                if best_match is not None:
                    out_obj = out_objects[best_match]
                    out_size = np.sum(out_obj)
                    out_center = PatternAnalyzer._get_center_of_mass(out_obj)
                    
                    obj_transform['size_changes'].append(out_size / inp_size if inp_size > 0 else 1)
                    obj_transform['position_changes'].append((
                        out_center[0] - inp_center[0],
                        out_center[1] - inp_center[1]
                    ))
            
            analysis['object_transformations'].append(obj_transform)
            
            # Spatial pattern analysis
            spatial_info = PatternAnalyzer._analyze_spatial_patterns(inp, out)
            analysis['spatial_patterns'].append(spatial_info)
            
            # Repetition pattern analysis
            rep_info = PatternAnalyzer._analyze_repetitions(inp, out)
            analysis['repetition_patterns'].append(rep_info)
            
            # Check for scaling
            if inp_shape[0] > 0 and inp_shape[1] > 0:
                if out_shape[0] % inp_shape[0] == 0 and out_shape[1] % inp_shape[1] == 0:
                    scale_h = out_shape[0] // inp_shape[0]
                    scale_w = out_shape[1] // inp_shape[1]
                    if scale_h == scale_w:
                        analysis['scaling_factors'].append(scale_h)
            
            # Check symmetries
            if np.array_equal(inp, np.fliplr(inp)):
                analysis['symmetries'].append('horizontal')
            if np.array_equal(inp, np.flipud(inp)):
                analysis['symmetries'].append('vertical')
        
        return analysis
    
    @staticmethod
    def _find_objects(grid: np.ndarray) -> List[np.ndarray]:
        """Find objects using connected components."""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    obj_mask = np.zeros_like(grid, dtype=bool)
                    PatternAnalyzer._flood_fill(grid, visited, obj_mask, i, j, grid[i, j])
                    if np.sum(obj_mask) > 1:  # Ignore single pixels
                        objects.append(obj_mask)
        
        return objects
    
    @staticmethod 
    def _flood_fill(grid: np.ndarray, visited: np.ndarray, obj_mask: np.ndarray,
                   r: int, c: int, color: int):
        """Flood fill helper."""
        if (r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1] or 
            visited[r, c] or grid[r, c] != color):
            return
        
        visited[r, c] = True
        obj_mask[r, c] = True
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            PatternAnalyzer._flood_fill(grid, visited, obj_mask, r + dr, c + dc, color)
    
    @staticmethod
    def _get_center_of_mass(mask: np.ndarray) -> Tuple[float, float]:
        """Get center of mass of a binary mask."""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return (0.0, 0.0)
        return (float(np.mean(coords[0])), float(np.mean(coords[1])))
    
    @staticmethod
    def _analyze_spatial_patterns(inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial arrangement patterns."""
        return {
            'grid_alignment': PatternAnalyzer._check_grid_alignment(inp, out),
            'boundary_patterns': PatternAnalyzer._analyze_boundaries(inp, out),
            'fill_patterns': PatternAnalyzer._analyze_fill_patterns(inp, out)
        }
    
    @staticmethod
    def _analyze_repetitions(inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
        """Analyze repetition patterns."""
        return {
            'horizontal_reps': PatternAnalyzer._count_repetitions(inp, 'horizontal'),
            'vertical_reps': PatternAnalyzer._count_repetitions(inp, 'vertical'),
            'output_horizontal_reps': PatternAnalyzer._count_repetitions(out, 'horizontal'),
            'output_vertical_reps': PatternAnalyzer._count_repetitions(out, 'vertical')
        }
    
    @staticmethod
    def _check_grid_alignment(inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if patterns align to a grid."""
        # Simple heuristic: check if non-zero elements follow regular spacing
        inp_nonzero = np.where(inp != 0)
        if len(inp_nonzero[0]) < 4:
            return False
        
        # Check for regular spacing in rows and columns
        row_diffs = np.diff(sorted(set(inp_nonzero[0])))
        col_diffs = np.diff(sorted(set(inp_nonzero[1])))
        
        row_regular = len(set(row_diffs)) <= 2 if len(row_diffs) > 0 else True
        col_regular = len(set(col_diffs)) <= 2 if len(col_diffs) > 0 else True
        
        return row_regular and col_regular
    
    @staticmethod
    def _analyze_boundaries(inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
        """Analyze boundary-related patterns."""
        return {
            'border_preserved': np.array_equal(inp[0, :], out[0, :]) and np.array_equal(inp[-1, :], out[-1, :]),
            'corners_preserved': (inp[0, 0] == out[0, 0] and inp[0, -1] == out[0, -1] and 
                                inp[-1, 0] == out[-1, 0] and inp[-1, -1] == out[-1, -1])
        }
    
    @staticmethod
    def _analyze_fill_patterns(inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
        """Analyze fill/completion patterns."""
        inp_filled = np.sum(inp != 0)
        out_filled = np.sum(out != 0)
        
        return {
            'fill_ratio_change': out_filled / inp_filled if inp_filled > 0 else 0,
            'background_change': inp[inp == 0].size != out[out == 0].size
        }
    
    @staticmethod
    def _count_repetitions(grid: np.ndarray, direction: str) -> int:
        """Count repetitive patterns in a direction."""
        if direction == 'horizontal':
            # Check for horizontal repetitions
            h, w = grid.shape
            for rep_width in range(1, w // 2 + 1):
                if w % rep_width == 0:
                    pattern = grid[:, :rep_width]
                    is_repeating = True
                    for i in range(rep_width, w, rep_width):
                        if not np.array_equal(pattern, grid[:, i:i+rep_width]):
                            is_repeating = False
                            break
                    if is_repeating:
                        return w // rep_width
        else:  # vertical
            h, w = grid.shape
            for rep_height in range(1, h // 2 + 1):
                if h % rep_height == 0:
                    pattern = grid[:rep_height, :]
                    is_repeating = True
                    for i in range(rep_height, h, rep_height):
                        if not np.array_equal(pattern, grid[i:i+rep_height, :]):
                            is_repeating = False
                            break
                    if is_repeating:
                        return h // rep_height
        
        return 1

class TestTimeTrainer:
    """Implements test-time training for better generalization."""
    
    def __init__(self):
        self.learned_patterns = {}
    
    def adapt_to_task(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Adapt to specific task patterns during test time."""
        patterns = PatternAnalyzer.analyze_task(train_examples)
        
        # Generate synthetic examples based on discovered patterns
        synthetic_examples = self._generate_synthetic_examples(train_examples, patterns)
        
        return {
            'original_examples': train_examples,
            'synthetic_examples': synthetic_examples,
            'patterns': patterns
        }
    
    def _generate_synthetic_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                                   patterns: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic training examples based on discovered patterns."""
        synthetic = []
        
        # For now, just return variations of existing examples
        for inp, out in examples:
            # Try different transformations that preserve the core pattern
            for transform in [np.fliplr, np.flipud, lambda x: np.rot90(x, 1)]:
                try:
                    transformed_inp = transform(inp)
                    transformed_out = transform(out)
                    synthetic.append((transformed_inp, transformed_out))
                except:
                    continue
        
        return synthetic

class NeuralPatternRecognizer:
    """Neural-inspired pattern recognition for ARC Prize 2025."""
    
    def __init__(self):
        self.pattern_cache = {}
        self.success_patterns = []
    
    def recognize_patterns(self, train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Advanced pattern recognition using neural-inspired techniques."""
        patterns = {
            'transformation_type': self._classify_transformation(train_examples),
            'object_dynamics': self._analyze_object_dynamics(train_examples),
            'spatial_relationships': self._analyze_spatial_relationships(train_examples),
            'color_semantics': self._analyze_color_semantics(train_examples),
            'invariant_features': self._find_invariant_features(train_examples),
            'compositionality': self._analyze_compositionality(train_examples)
        }
        
        # Cache successful patterns
        pattern_signature = self._create_pattern_signature(patterns)
        if pattern_signature in self.pattern_cache:
            patterns['cached_solution'] = self.pattern_cache[pattern_signature]
        
        return patterns
    
    def _classify_transformation(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """Classify the type of transformation using multiple heuristics."""
        scores = {
            'geometric': 0,
            'topological': 0,
            'color_mapping': 0,
            'size_scaling': 0,
            'pattern_completion': 0,
            'object_manipulation': 0
        }
        
        for inp, out in examples:
            # Geometric transformations
            if self._test_geometric_transformation(inp, out):
                scores['geometric'] += 1
            
            # Size scaling
            if inp.shape != out.shape:
                scores['size_scaling'] += 1
            
            # Color mapping
            if set(inp.flatten()) != set(out.flatten()):
                scores['color_mapping'] += 1
            
            # Object manipulation
            inp_objects = len(PatternAnalyzer._find_objects(inp))
            out_objects = len(PatternAnalyzer._find_objects(out))
            if inp_objects != out_objects:
                scores['object_manipulation'] += 1
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _test_geometric_transformation(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Test if output is a geometric transformation of input."""
        if inp.shape != out.shape:
            return False
        
        transforms = [np.fliplr, np.flipud, np.transpose, 
                     lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3)]
        
        for transform in transforms:
            try:
                if np.array_equal(transform(inp), out):
                    return True
            except:
                continue
        return False
    
    def _analyze_object_dynamics(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Analyze how objects change between input and output."""
        dynamics = {
            'object_count_changes': [],
            'object_position_changes': [],
            'object_size_changes': [],
            'object_merger_patterns': [],
            'object_creation_patterns': []
        }
        
        for inp, out in examples:
            inp_objects = PatternAnalyzer._find_objects(inp)
            out_objects = PatternAnalyzer._find_objects(out)
            
            dynamics['object_count_changes'].append(len(out_objects) - len(inp_objects))
            
            # Analyze position and size changes
            for inp_obj in inp_objects:
                inp_center = PatternAnalyzer._get_center_of_mass(inp_obj)
                inp_size = np.sum(inp_obj)
                
                # Find best matching output object
                best_match = None
                best_distance = float('inf')
                
                for out_obj in out_objects:
                    out_center = PatternAnalyzer._get_center_of_mass(out_obj)
                    distance = np.linalg.norm(np.array(inp_center) - np.array(out_center))
                    if distance < best_distance:
                        best_distance = distance
                        best_match = out_obj
                
                if best_match is not None:
                    out_size = np.sum(best_match)
                    dynamics['object_size_changes'].append(out_size / inp_size if inp_size > 0 else 1)
        
        return dynamics
    
    def _analyze_spatial_relationships(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Analyze spatial relationships between objects."""
        relationships = {
            'adjacency_patterns': [],
            'alignment_patterns': [],
            'containment_patterns': [],
            'distance_patterns': []
        }
        
        for inp, out in examples:
            inp_objects = PatternAnalyzer._find_objects(inp)
            out_objects = PatternAnalyzer._find_objects(out)
            
            # Analyze adjacency
            inp_adjacencies = self._compute_adjacencies(inp_objects)
            out_adjacencies = self._compute_adjacencies(out_objects)
            
            relationships['adjacency_patterns'].append({
                'input': inp_adjacencies,
                'output': out_adjacencies
            })
        
        return relationships
    
    def _compute_adjacencies(self, objects: List[np.ndarray]) -> List[bool]:
        """Compute which objects are adjacent to each other."""
        adjacencies = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Check if objects are adjacent
                obj1_coords = set(zip(*np.where(obj1)))
                obj2_coords = set(zip(*np.where(obj2)))
                
                adjacent = False
                for r1, c1 in obj1_coords:
                    for r2, c2 in obj2_coords:
                        if abs(r1-r2) + abs(c1-c2) == 1:  # Manhattan distance = 1
                            adjacent = True
                            break
                    if adjacent:
                        break
                
                adjacencies.append(adjacent)
        
        return adjacencies
    
    def _analyze_color_semantics(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Analyze semantic meaning of colors."""
        semantics = {
            'color_roles': {},
            'color_transitions': {},
            'color_hierarchies': {}
        }
        
        for inp, out in examples:
            inp_colors = Counter(inp.flatten())
            out_colors = Counter(out.flatten())
            
            # Analyze color role (background, object, special)
            for color, count in inp_colors.items():
                if count == inp.size:  # All pixels this color
                    semantics['color_roles'][color] = 'background'
                elif count == 1:  # Single pixel
                    semantics['color_roles'][color] = 'marker'
                else:
                    semantics['color_roles'][color] = 'object'
            
            # Analyze color transitions
            for inp_color in inp_colors:
                for out_color in out_colors:
                    if inp_color != out_color:
                        key = f"{inp_color}->{out_color}"
                        semantics['color_transitions'][key] = semantics['color_transitions'].get(key, 0) + 1
        
        return semantics
    
    def _find_invariant_features(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        """Find features that remain invariant across transformations."""
        invariants = []
        
        # Check shape invariance
        if all(inp.shape == out.shape for inp, out in examples):
            invariants.append('shape')
        
        # Check color set invariance
        if all(set(inp.flatten()) == set(out.flatten()) for inp, out in examples):
            invariants.append('color_set')
        
        # Check object count invariance
        object_counts_same = True
        for inp, out in examples:
            inp_objects = PatternAnalyzer._find_objects(inp)
            out_objects = PatternAnalyzer._find_objects(out)
            if len(inp_objects) != len(out_objects):
                object_counts_same = False
                break
        
        if object_counts_same:
            invariants.append('object_count')
        
        return invariants
    
    def _analyze_compositionality(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Analyze if transformation can be decomposed into simpler parts."""
        composition = {
            'decomposable': False,
            'sub_transformations': [],
            'composition_type': 'sequential'  # or 'parallel'
        }
        
        # Try to decompose into region-based transformations
        for inp, out in examples:
            h, w = inp.shape
            
            # Try quadrant decomposition
            if h % 2 == 0 and w % 2 == 0:
                h_mid, w_mid = h // 2, w // 2
                
                quadrants_inp = [
                    inp[:h_mid, :w_mid], inp[:h_mid, w_mid:],
                    inp[h_mid:, :w_mid], inp[h_mid:, w_mid:]
                ]
                quadrants_out = [
                    out[:h_mid, :w_mid], out[:h_mid, w_mid:],
                    out[h_mid:, :w_mid], out[h_mid:, w_mid:]
                ]
                
                # Check if each quadrant transforms independently
                independent_transforms = []
                for q_inp, q_out in zip(quadrants_inp, quadrants_out):
                    if np.array_equal(q_inp, q_out):
                        independent_transforms.append('identity')
                    elif np.array_equal(np.fliplr(q_inp), q_out):
                        independent_transforms.append('flip_horizontal')
                    elif np.array_equal(np.flipud(q_inp), q_out):
                        independent_transforms.append('flip_vertical')
                    else:
                        independent_transforms.append('complex')
                
                if 'complex' not in independent_transforms:
                    composition['decomposable'] = True
                    composition['sub_transformations'] = independent_transforms
                    composition['composition_type'] = 'parallel'
        
        return composition
    
    def _create_pattern_signature(self, patterns: Dict[str, Any]) -> str:
        """Create a signature for caching successful patterns."""
        key_elements = [
            patterns['transformation_type'],
            str(patterns['invariant_features']),
            str(patterns['object_dynamics']['object_count_changes']),
        ]
        return '|'.join(key_elements)
    
    def cache_successful_pattern(self, patterns: Dict[str, Any], solution: DSLProgram):
        """Cache a successful pattern-solution pair."""
        signature = self._create_pattern_signature(patterns)
        self.pattern_cache[signature] = solution

class HierarchicalReasoner:
    """Hierarchical reasoning system for complex ARC problems."""
    
    def __init__(self):
        self.decomposition_strategies = [
            self._spatial_decomposition,
            self._temporal_decomposition,
            self._object_decomposition,
            self._color_decomposition
        ]
    
    def solve_hierarchically(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                           test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve using hierarchical decomposition."""
        
        for strategy in self.decomposition_strategies:
            try:
                sub_problems = strategy(train_examples, test_input)
                if sub_problems:
                    solution = self._solve_sub_problems(sub_problems)
                    if solution is not None:
                        return solution
            except:
                continue
        
        return None
    
    def _spatial_decomposition(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                              test_input: np.ndarray) -> Optional[List[Dict]]:
        """Decompose problem spatially (regions, quadrants, etc.)."""
        # Try quadrant decomposition
        h, w = test_input.shape
        if h % 2 == 0 and w % 2 == 0:
            h_mid, w_mid = h // 2, w // 2
            
            sub_problems = []
            regions = [
                ((0, h_mid), (0, w_mid)),      # Top-left
                ((0, h_mid), (w_mid, w)),      # Top-right
                ((h_mid, h), (0, w_mid)),      # Bottom-left
                ((h_mid, h), (w_mid, w))       # Bottom-right
            ]
            
            for i, ((r1, r2), (c1, c2)) in enumerate(regions):
                region_examples = []
                for inp, out in examples:
                    region_inp = inp[r1:r2, c1:c2]
                    region_out = out[r1:r2, c1:c2] if out.shape == inp.shape else out
                    region_examples.append((region_inp, region_out))
                
                region_test = test_input[r1:r2, c1:c2]
                
                sub_problems.append({
                    'type': 'spatial_region',
                    'index': i,
                    'examples': region_examples,
                    'test_input': region_test,
                    'position': ((r1, r2), (c1, c2))
                })
            
            return sub_problems
        
        return None
    
    def _temporal_decomposition(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                               test_input: np.ndarray) -> Optional[List[Dict]]:
        """Decompose into sequence of transformations."""
        # Try to find intermediate states
        return None  # Simplified for now
    
    def _object_decomposition(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                             test_input: np.ndarray) -> Optional[List[Dict]]:
        """Decompose by individual objects."""
        test_objects = PatternAnalyzer._find_objects(test_input)
        
        if len(test_objects) > 1:
            sub_problems = []
            
            for i, obj_mask in enumerate(test_objects):
                # Create sub-problem for this object
                obj_examples = []
                
                for inp, out in examples:
                    inp_objects = PatternAnalyzer._find_objects(inp)
                    out_objects = PatternAnalyzer._find_objects(out)
                    
                    if i < len(inp_objects) and i < len(out_objects):
                        obj_inp = inp * inp_objects[i]  # Isolate object
                        obj_out = out * out_objects[i]  # Isolate corresponding output object
                        obj_examples.append((obj_inp, obj_out))
                
                if obj_examples:
                    obj_test = test_input * obj_mask
                    
                    sub_problems.append({
                        'type': 'object',
                        'index': i,
                        'examples': obj_examples,
                        'test_input': obj_test,
                        'mask': obj_mask
                    })
            
            return sub_problems
        
        return None
    
    def _color_decomposition(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
                            test_input: np.ndarray) -> Optional[List[Dict]]:
        """Decompose by color channels."""
        test_colors = set(test_input.flatten()) - {0}
        
        if len(test_colors) > 1:
            sub_problems = []
            
            for color in test_colors:
                color_examples = []
                
                for inp, out in examples:
                    color_inp = (inp == color).astype(int)
                    color_out = (out == color).astype(int)
                    color_examples.append((color_inp, color_out))
                
                color_test = (test_input == color).astype(int)
                
                sub_problems.append({
                    'type': 'color',
                    'color': color,
                    'examples': color_examples,
                    'test_input': color_test
                })
            
            return sub_problems
        
        return None
    
    def _solve_sub_problems(self, sub_problems: List[Dict]) -> Optional[np.ndarray]:
        """Solve individual sub-problems and combine results."""
        if not sub_problems:
            return None
        
        # Get solution for each sub-problem
        sub_solutions = []
        synthesizer = ProgramSynthesizer()
        
        for sub_problem in sub_problems:
            program = synthesizer.synthesize_program(sub_problem['examples'], max_attempts=200)
            if program:
                try:
                    solution = program.execute(sub_problem['test_input'])
                    sub_solutions.append((sub_problem, solution))
                except:
                    continue
        
        if not sub_solutions:
            return None
        
        # Combine solutions based on decomposition type
        if sub_problems[0]['type'] == 'spatial_region':
            return self._combine_spatial_solutions(sub_solutions)
        elif sub_problems[0]['type'] == 'object':
            return self._combine_object_solutions(sub_solutions)
        elif sub_problems[0]['type'] == 'color':
            return self._combine_color_solutions(sub_solutions)
        
        return None
    
    def _combine_spatial_solutions(self, sub_solutions: List[Tuple[Dict, np.ndarray]]) -> np.ndarray:
        """Combine spatial region solutions."""
        if not sub_solutions:
            return None
        
        # Determine output size from first solution
        first_solution = sub_solutions[0][1]
        
        # Create output grid - assume same size as input for now
        result_shape = (first_solution.shape[0] * 2, first_solution.shape[1] * 2)
        result = np.zeros(result_shape, dtype=first_solution.dtype)
        
        for sub_problem, solution in sub_solutions:
            if 'position' in sub_problem:
                (r1, r2), (c1, c2) = sub_problem['position']
                result[r1:r2, c1:c2] = solution
        
        return result
    
    def _combine_object_solutions(self, sub_solutions: List[Tuple[Dict, np.ndarray]]) -> np.ndarray:
        """Combine object-based solutions."""
        if not sub_solutions:
            return None
        
        # Start with background
        result = np.zeros_like(sub_solutions[0][1])
        
        for sub_problem, solution in sub_solutions:
            if 'mask' in sub_problem:
                mask = sub_problem['mask']
                result[mask] = solution[mask]
        
        return result
    
    def _combine_color_solutions(self, sub_solutions: List[Tuple[Dict, np.ndarray]]) -> np.ndarray:
        """Combine color-based solutions."""
        if not sub_solutions:
            return None
        
        result = np.zeros_like(sub_solutions[0][1])
        
        for sub_problem, solution in sub_solutions:
            color = sub_problem['color']
            result[solution == 1] = color
        
        return result

class EnsembleSolver:
    """Advanced ensemble solver for ARC Prize 2025."""
    
    def __init__(self):
        self.program_synthesizer = ProgramSynthesizer()
        self.test_time_trainer = TestTimeTrainer()
        self.pattern_analyzer = PatternAnalyzer()
        self.neural_recognizer = NeuralPatternRecognizer()
        self.hierarchical_reasoner = HierarchicalReasoner()
    
    def solve_task(self, challenge: Dict, debug: bool = False) -> np.ndarray:
        """Solve using state-of-the-art techniques for ARC Prize 2025."""
        train_examples = []
        
        # Extract training examples
        for example in challenge.get('train', []):
            inp = np.array(example['input'], dtype=int)
            out = np.array(example['output'], dtype=int)
            train_examples.append((inp, out))
        
        if not train_examples:
            return np.array([[0]], dtype=int)
        
        # Get the test input
        test_input = np.array(challenge['test'][0]['input'], dtype=int)
        
        if debug:
            print(f"ARC Prize 2025 Solver: {len(train_examples)} examples, test {test_input.shape}")
        
        # STAGE 1: Neural Pattern Recognition
        neural_patterns = self.neural_recognizer.recognize_patterns(train_examples)
        
        # Check for cached solution
        if 'cached_solution' in neural_patterns:
            if debug:
                print("Found cached solution")
            try:
                result = neural_patterns['cached_solution'].execute(test_input)
                return result
            except:
                pass
        
        # STAGE 2: Hierarchical Decomposition  
        hierarchical_result = self.hierarchical_reasoner.solve_hierarchically(train_examples, test_input)
        if hierarchical_result is not None:
            if debug:
                print("Hierarchical solver succeeded")
            return hierarchical_result
        
        # STAGE 3: Ultra-fast pattern matching
        result = self._try_quick_patterns(train_examples, test_input, debug)
        if result is not None:
            if debug:
                print("Quick pattern match")
            return result
        
        # STAGE 4: Massive parallel ensemble with beam search
        candidates = self._generate_advanced_candidates(train_examples, test_input, neural_patterns, debug)
        
        if candidates:
            best_candidate = self._neural_candidate_selection(candidates, train_examples, neural_patterns, debug)
            if best_candidate is not None:
                if debug:
                    print("Advanced ensemble succeeded")
                # Cache successful pattern
                if hasattr(best_candidate, 'program'):
                    self.neural_recognizer.cache_successful_pattern(neural_patterns, best_candidate.program)
                return best_candidate
        
        # STAGE 5: Pattern-guided transformations with neural insights
        result = self._neural_guided_transforms(train_examples, test_input, neural_patterns, debug)
        if result is not None:
            if debug:
                print("Neural-guided transformation")
            return result
        
        # STAGE 6: Advanced spatial reasoning
        result = self._try_spatial_completion(train_examples, test_input, neural_patterns, debug)
        if result is not None:
            return result
        
        # STAGE 7: Multi-level fallback with meta-learning
        result = self._meta_learning_fallback(train_examples, test_input, neural_patterns, debug)
        if result is not None:
            return result
        
        # Final fallback: Most similar training output
        return self._select_most_similar_output(train_examples, test_input)
    
    def _score_similarity(self, candidate: np.ndarray, targets: List[np.ndarray]) -> float:
        """Score how similar a candidate is to target outputs."""
        if not targets:
            return 0.0
        
        scores = []
        for target in targets:
            if candidate.shape == target.shape:
                similarity = np.mean(candidate == target)
                scores.append(similarity)
        
        return max(scores) if scores else 0.0
    
    def _try_quick_patterns(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                           test_input: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """Quick pattern matching for common transformations."""
        if not train_examples:
            return None
        
        # Try direct mapping based on input-output relationships
        for inp, out in train_examples:
            if np.array_equal(inp, test_input):
                return out
            
            # Try simple transformations
            transforms = [
                (np.fliplr, "flip_horizontal"),
                (np.flipud, "flip_vertical"), 
                (lambda x: np.rot90(x, 1), "rotate_90"),
                (lambda x: np.rot90(x, 2), "rotate_180"),
                (lambda x: np.rot90(x, 3), "rotate_270")
            ]
            
            for transform, name in transforms:
                try:
                    if np.array_equal(inp, test_input):
                        result = transform(out)
                        if debug:
                            print(f"Quick pattern match: {name}")
                        return result
                except:
                    continue
                    
        return None
    
    def _try_pattern_guided_transforms(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                     test_input: np.ndarray, patterns: Dict[str, Any], 
                                     debug: bool = False) -> Optional[np.ndarray]:
        """Apply transformations guided by discovered patterns."""
        
        # Check for scaling patterns
        if patterns['scaling_factors']:
            most_common_scale = Counter(patterns['scaling_factors']).most_common(1)[0][0]
            if debug:
                print(f"Applying scale factor: {most_common_scale}")
            try:
                scaled = np.repeat(np.repeat(test_input, most_common_scale, axis=0), 
                                 most_common_scale, axis=1)
                return scaled
            except:
                pass
        
        # Check for repetition patterns
        if patterns['repetition_patterns']:
            for rep_pattern in patterns['repetition_patterns']:
                if rep_pattern['horizontal_reps'] > 1:
                    try:
                        h, w = test_input.shape
                        pattern_width = w // rep_pattern['horizontal_reps']
                        if pattern_width > 0:
                            result = np.zeros_like(test_input)
                            base_pattern = test_input[:, :pattern_width]
                            for i in range(rep_pattern['horizontal_reps']):
                                result[:, i*pattern_width:(i+1)*pattern_width] = base_pattern
                            return result
                    except:
                        continue
        
        return None
    
    def _try_object_transformations(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                   test_input: np.ndarray, patterns: Dict[str, Any], 
                                   debug: bool = False) -> Optional[np.ndarray]:
        """Apply object-based transformations."""
        
        # Analyze object transformation patterns
        if patterns['object_transformations']:
            obj_patterns = patterns['object_transformations']
            
            # Check for consistent object count changes
            count_changes = [p['output_count'] - p['input_count'] for p in obj_patterns]
            if len(set(count_changes)) == 1 and count_changes[0] != 0:
                # Consistent object addition/removal pattern
                if debug:
                    print(f"Object count change pattern: {count_changes[0]}")
                
                # Try object filling/removal
                test_objects = PatternAnalyzer._find_objects(test_input)
                result = test_input.copy()
                
                if count_changes[0] < 0:  # Objects being removed
                    # Remove smallest objects
                    object_sizes = [(i, np.sum(obj)) for i, obj in enumerate(test_objects)]
                    object_sizes.sort(key=lambda x: x[1])
                    
                    for i in range(min(-count_changes[0], len(object_sizes))):
                        obj_idx = object_sizes[i][0]
                        result[test_objects[obj_idx]] = 0
                    
                    return result
        
        # Try color-based object transformations
        for inp, out in train_examples:
            inp_colors = set(inp.flatten())
            out_colors = set(out.flatten())
            
            if len(inp_colors) != len(out_colors):
                # Color mapping transformation
                if len(out_colors) == 1:  # All converted to one color
                    target_color = list(out_colors)[0]
                    result = np.full_like(test_input, target_color)
                    return result
        
        return None
    
    def _try_spatial_completion(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                               test_input: np.ndarray, patterns: Dict[str, Any], 
                               debug: bool = False) -> Optional[np.ndarray]:
        """Try spatial pattern completion and extension."""
        
        if patterns['spatial_patterns']:
            spatial_info = patterns['spatial_patterns'][0]  # Use first example as template
            
            # Check for fill patterns
            if spatial_info['fill_patterns']['background_change']:
                # Try filling background or holes
                result = test_input.copy()
                
                # Find most common non-zero color
                non_zero = test_input[test_input != 0]
                if len(non_zero) > 0:
                    fill_color = Counter(non_zero).most_common(1)[0][0]
                    
                    # Fill holes (background pixels surrounded by non-background)
                    for i in range(1, test_input.shape[0] - 1):
                        for j in range(1, test_input.shape[1] - 1):
                            if test_input[i, j] == 0:
                                neighbors = [
                                    test_input[i-1, j], test_input[i+1, j],
                                    test_input[i, j-1], test_input[i, j+1]
                                ]
                                if all(n != 0 for n in neighbors):
                                    result[i, j] = fill_color
                    
                    if not np.array_equal(result, test_input):
                        if debug:
                            print("Applied hole filling")
                        return result
        
        # Try grid-based extension
        for inp, out in train_examples:
            if inp.shape != out.shape:
                # Check if it's a grid extension pattern
                h_ratio = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 1
                w_ratio = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 1
                
                if h_ratio == int(h_ratio) and w_ratio == int(w_ratio):
                    try:
                        h_factor, w_factor = int(h_ratio), int(w_ratio)
                        result = np.zeros((test_input.shape[0] * h_factor, 
                                         test_input.shape[1] * w_factor), dtype=test_input.dtype)
                        
                        for i in range(h_factor):
                            for j in range(w_factor):
                                start_h = i * test_input.shape[0]
                                start_w = j * test_input.shape[1]
                                result[start_h:start_h+test_input.shape[0], 
                                      start_w:start_w+test_input.shape[1]] = test_input
                        
                        if debug:
                            print(f"Applied grid extension: {h_factor}x{w_factor}")
                        return result
                    except:
                        continue
        
        return None
    
    def _guided_program_synthesis(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                 patterns: Dict[str, Any], debug: bool = False) -> Optional[DSLProgram]:
        """Synthesize programs guided by pattern analysis."""
        
        # Create a smaller, more targeted set of operations based on patterns
        guided_ops = [IdentityOp()]
        
        # Add operations based on discovered patterns
        if patterns['scaling_factors']:
            for scale in set(patterns['scaling_factors']):
                guided_ops.append(ScaleOp(scale))
        
        # Add transformations based on shape changes
        shape_changes = patterns['shape_changes']
        if any(inp != out for inp, out in shape_changes):
            guided_ops.extend([FlipHorizontalOp(), FlipVerticalOp(), 
                             RotateOp(1), RotateOp(2), RotateOp(3)])
        
        # Add object operations if objects detected
        if patterns['object_transformations']:
            for color in range(1, 10):
                guided_ops.append(ObjectFillOp(color))
                guided_ops.append(ObjectFillOp(0, color))
        
        # Try synthesis with this guided set
        synthesizer = ProgramSynthesizer(max_program_length=2)
        synthesizer.primitive_ops = guided_ops
        
        return synthesizer.synthesize_program(train_examples, max_attempts=200)
    
    def _try_fallback_transforms(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                test_input: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """Try basic fallback transformations."""
        
        best_result = None
        best_score = -1
        
        for inp, out in train_examples:
            if inp.shape == test_input.shape:
                # Try simple transformations
                transforms = [
                    (np.fliplr, "flip_horizontal"),
                    (np.flipud, "flip_vertical"),
                    (np.rot90, "rotate_90"),
                    (np.transpose, "transpose"),
                    (lambda x: np.rot90(x, 2), "rotate_180"),
                    (lambda x: np.rot90(x, 3), "rotate_270")
                ]
                
                for transform, name in transforms:
                    try:
                        candidate = transform(test_input)
                        # Score based on similarity to training outputs
                        score = self._score_similarity(candidate, [out for _, out in train_examples])
                        if score > best_score:
                            best_score = score
                            best_result = candidate
                            if debug and score > 0.5:
                                print(f"Good fallback match with {name}: score {score:.3f}")
                    except:
                        continue
        
        return best_result if best_score > 0.3 else None
    
    def _generate_candidates(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                            test_input: np.ndarray, debug: bool = False) -> List[np.ndarray]:
        """Generate multiple solution candidates using different methods."""
        candidates = []
        
        # Candidate 1: Program synthesis
        program = self.program_synthesizer.synthesize_program(train_examples, max_attempts=1000)
        if program:
            try:
                candidate = program.execute(test_input)
                candidates.append(candidate)
                if debug:
                    print(f"Program synthesis candidate: {program.description()}")
            except:
                pass
        
        # Candidate 2: Template matching
        template_candidate = self._template_matching_candidate(train_examples, test_input)
        if template_candidate is not None:
            candidates.append(template_candidate)
            if debug:
                print("Template matching candidate generated")
        
        # Candidate 3: Geometric transformations
        geo_candidate = self._geometric_candidate(train_examples, test_input)
        if geo_candidate is not None:
            candidates.append(geo_candidate)
            if debug:
                print("Geometric transformation candidate generated")
        
        # Candidate 4: Object-based reasoning
        obj_candidate = self._object_reasoning_candidate(train_examples, test_input)
        if obj_candidate is not None:
            candidates.append(obj_candidate)
            if debug:
                print("Object reasoning candidate generated")
        
        # Candidate 5: Grid-based patterns
        grid_candidate = self._grid_pattern_candidate(train_examples, test_input)
        if grid_candidate is not None:
            candidates.append(grid_candidate)
            if debug:
                print("Grid pattern candidate generated")
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[np.ndarray], 
                              train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                              debug: bool = False) -> Optional[np.ndarray]:
        """Select the best candidate using ensemble scoring."""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Score each candidate
        candidate_scores = []
        for i, candidate in enumerate(candidates):
            score = 0
            
            # Score 1: Consistency with training examples
            for inp, out in train_examples:
                if candidate.shape == out.shape:
                    similarity = np.mean(candidate == out) if candidate.shape == out.shape else 0
                    score += similarity
            
            # Score 2: Shape consistency
            expected_shapes = [out.shape for _, out in train_examples]
            if all(shape == expected_shapes[0] for shape in expected_shapes):
                if candidate.shape == expected_shapes[0]:
                    score += 1
            
            # Score 3: Color distribution consistency
            expected_colors = set()
            for _, out in train_examples:
                expected_colors.update(out.flatten())
            
            candidate_colors = set(candidate.flatten())
            color_overlap = len(expected_colors & candidate_colors) / len(expected_colors) if expected_colors else 0
            score += color_overlap
            
            candidate_scores.append(score)
            
            if debug:
                print(f"Candidate {i}: score {score:.3f}, shape {candidate.shape}")
        
        # Return the highest scoring candidate
        best_idx = np.argmax(candidate_scores)
        return candidates[best_idx]
    
    def _template_matching_candidate(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                   test_input: np.ndarray) -> Optional[np.ndarray]:
        """Generate candidate using template matching."""
        
        # Find the most similar training input
        best_match = None
        best_similarity = 0
        
        for inp, out in train_examples:
            if inp.shape == test_input.shape:
                similarity = np.mean(inp == test_input)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = out
        
        if best_match is not None and best_similarity > 0.7:
            return best_match.copy()
        
        return None
    
    def _geometric_candidate(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                           test_input: np.ndarray) -> Optional[np.ndarray]:
        """Generate candidate using geometric transformations."""
        
        # Try all geometric transformations and see which works best
        transforms = [
            np.fliplr, np.flipud, np.transpose,
            lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3)
        ]
        
        for transform in transforms:
            try:
                candidate = transform(test_input)
                
                # Check if this transformation is consistent with training examples
                consistent = True
                for inp, out in train_examples:
                    if inp.shape == test_input.shape and out.shape == candidate.shape:
                        expected = transform(inp)
                        if not np.array_equal(expected, out):
                            consistent = False
                            break
                
                if consistent:
                    return candidate
            except:
                continue
        
        return None
    
    def _object_reasoning_candidate(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                  test_input: np.ndarray) -> Optional[np.ndarray]:
        """Generate candidate using object-based reasoning."""
        
        # Analyze object transformations in training examples
        test_objects = PatternAnalyzer._find_objects(test_input)
        
        if not test_objects:
            return None
        
        # Look for consistent object transformation patterns
        for inp, out in train_examples:
            inp_objects = PatternAnalyzer._find_objects(inp)
            out_objects = PatternAnalyzer._find_objects(out)
            
            if len(inp_objects) == len(test_objects):
                # Try applying the same transformation pattern
                result = test_input.copy()
                
                # Simple object color change pattern
                inp_colors = set(inp.flatten()) - {0}
                out_colors = set(out.flatten()) - {0}
                
                if len(inp_colors) == 1 and len(out_colors) == 1:
                    inp_color = list(inp_colors)[0]
                    out_color = list(out_colors)[0]
                    
                    # Apply color transformation
                    result[result == inp_color] = out_color
                    return result
        
        return None
    
    def _grid_pattern_candidate(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                              test_input: np.ndarray) -> Optional[np.ndarray]:
        """Generate candidate using grid pattern analysis."""
        
        # Check for scaling patterns
        for inp, out in train_examples:
            if inp.shape != out.shape:
                if (out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0):
                    scale_h = out.shape[0] // inp.shape[0]
                    scale_w = out.shape[1] // inp.shape[1]
                    
                    if scale_h == scale_w and scale_h > 1:
                        # Apply scaling to test input
                        try:
                            scaled = np.repeat(np.repeat(test_input, scale_h, axis=0), scale_w, axis=1)
                            return scaled
                        except:
                            continue
        
        # Check for tiling patterns
        for inp, out in train_examples:
            if out.shape[0] >= inp.shape[0] * 2 or out.shape[1] >= inp.shape[1] * 2:
                try:
                    # Try horizontal tiling
                    if out.shape[1] >= inp.shape[1] * 2:
                        tiles_h = out.shape[1] // inp.shape[1]
                        result = np.zeros((test_input.shape[0], test_input.shape[1] * tiles_h), dtype=test_input.dtype)
                        for i in range(tiles_h):
                            result[:, i*test_input.shape[1]:(i+1)*test_input.shape[1]] = test_input
                        return result
                    
                    # Try vertical tiling
                    if out.shape[0] >= inp.shape[0] * 2:
                        tiles_v = out.shape[0] // inp.shape[0]
                        result = np.zeros((test_input.shape[0] * tiles_v, test_input.shape[1]), dtype=test_input.dtype)
                        for i in range(tiles_v):
                            result[i*test_input.shape[0]:(i+1)*test_input.shape[0], :] = test_input
                        return result
                        
                except:
                    continue
        
        return None
    
    def _generate_advanced_candidates(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                    test_input: np.ndarray, neural_patterns: Dict[str, Any],
                                    debug: bool = False) -> List[np.ndarray]:
        """Generate candidates using neural-guided advanced methods."""
        candidates = []
        
        # All previous candidates plus neural-guided ones
        base_candidates = self._generate_candidates(train_examples, test_input, debug)
        candidates.extend(base_candidates)
        
        # Neural-guided program synthesis
        transformation_type = neural_patterns['transformation_type']
        
        if transformation_type == 'geometric':
            # Focus on geometric operations
            geometric_ops = [FlipHorizontalOp(), FlipVerticalOp(), RotateOp(1), RotateOp(2), RotateOp(3)]
            for op in geometric_ops:
                program = DSLProgram([op])
                try:
                    candidate = program.execute(test_input)
                    candidates.append(candidate)
                except:
                    continue
        
        elif transformation_type == 'size_scaling':
            # Focus on scaling operations
            for scale in [2, 3, 4, 5]:
                try:
                    scaled = np.repeat(np.repeat(test_input, scale, axis=0), scale, axis=1)
                    candidates.append(scaled)
                except:
                    continue
        
        elif transformation_type == 'object_manipulation':
            # Focus on object operations
            for color in [1, 2, 3, 4, 5]:
                for op in [ObjectFillOp(color), GravityOp('down'), ConnectDotsOp(color)]:
                    try:
                        candidate = op.apply(test_input)
                        candidates.append(candidate)
                    except:
                        continue
        
        # Compositional candidates based on neural analysis
        if neural_patterns['compositionality']['decomposable']:
            sub_transforms = neural_patterns['compositionality']['sub_transformations']
            
            # Apply sub-transformations to test input
            h, w = test_input.shape
            if h % 2 == 0 and w % 2 == 0:
                h_mid, w_mid = h // 2, w // 2
                result = test_input.copy()
                
                transform_map = {
                    'identity': lambda x: x,
                    'flip_horizontal': np.fliplr,
                    'flip_vertical': np.flipud
                }
                
                quadrants = [
                    (slice(0, h_mid), slice(0, w_mid)),
                    (slice(0, h_mid), slice(w_mid, w)),
                    (slice(h_mid, h), slice(0, w_mid)),
                    (slice(h_mid, h), slice(w_mid, w))
                ]
                
                for i, transform_name in enumerate(sub_transforms[:4]):
                    if transform_name in transform_map and i < len(quadrants):
                        r_slice, c_slice = quadrants[i]
                        try:
                            result[r_slice, c_slice] = transform_map[transform_name](test_input[r_slice, c_slice])
                        except:
                            continue
                
                candidates.append(result)
        
        # Massive beam search candidates
        beam_candidates = self._beam_search_candidates(train_examples, test_input, neural_patterns)
        candidates.extend(beam_candidates)
        
        return candidates
    
    def _beam_search_candidates(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                               test_input: np.ndarray, neural_patterns: Dict[str, Any]) -> List[np.ndarray]:
        """Generate candidates using beam search with neural guidance."""
        candidates = []
        
        # Create multiple specialized synthesizers
        synthesizers = [
            ProgramSynthesizer(max_program_length=1),  # Single operations
            ProgramSynthesizer(max_program_length=2),  # Two-step operations  
            ProgramSynthesizer(max_program_length=3),  # Complex operations
        ]
        
        def search_with_synthesizer(synthesizer):
            program = synthesizer.synthesize_program(train_examples, max_attempts=2000)
            if program:
                try:
                    return program.execute(test_input)
                except:
                    pass
            return None
        
        # Parallel beam search
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(search_with_synthesizer, syn) for syn in synthesizers]
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    candidates.append(result)
        
        return candidates
    
    def _neural_candidate_selection(self, candidates: List[np.ndarray], 
                                  train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                  neural_patterns: Dict[str, Any],
                                  debug: bool = False) -> Optional[np.ndarray]:
        """Advanced candidate selection using neural insights."""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        candidate_scores = []
        
        for i, candidate in enumerate(candidates):
            score = 0
            
            # Traditional similarity scoring
            for inp, out in train_examples:
                if candidate.shape == out.shape:
                    similarity = np.mean(candidate == out)
                    score += similarity * 2  # Base similarity
            
            # Neural pattern consistency scoring
            invariants = neural_patterns['invariant_features']
            
            # Shape consistency bonus
            if 'shape' in invariants:
                expected_shape = train_examples[0][1].shape
                if candidate.shape == expected_shape:
                    score += 1
            
            # Color set consistency bonus
            if 'color_set' in invariants:
                expected_colors = set(train_examples[0][1].flatten())
                candidate_colors = set(candidate.flatten())
                if expected_colors == candidate_colors:
                    score += 1
            
            # Object count consistency bonus
            if 'object_count' in invariants:
                expected_obj_count = len(PatternAnalyzer._find_objects(train_examples[0][1]))
                candidate_obj_count = len(PatternAnalyzer._find_objects(candidate))
                if expected_obj_count == candidate_obj_count:
                    score += 0.5
            
            # Transformation type consistency
            transformation_type = neural_patterns['transformation_type']
            if transformation_type == 'geometric':
                # Prefer candidates that look like geometric transformations
                if self._looks_like_geometric_transform(train_examples[0][0], candidate):
                    score += 1
            elif transformation_type == 'size_scaling':
                # Prefer candidates with scaled dimensions
                inp_shape = train_examples[0][0].shape
                if candidate.shape[0] % inp_shape[0] == 0 and candidate.shape[1] % inp_shape[1] == 0:
                    score += 1
            
            candidate_scores.append(score)
            
            if debug:
                print(f"Candidate {i}: score {score:.3f}, shape {candidate.shape}")
        
        # Return highest scoring candidate
        best_idx = np.argmax(candidate_scores)
        return candidates[best_idx]
    
    def _neural_guided_transforms(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                 test_input: np.ndarray, neural_patterns: Dict[str, Any], 
                                 debug: bool = False) -> Optional[np.ndarray]:
        """Apply transformations guided by neural pattern analysis."""
        
        transformation_type = neural_patterns['transformation_type']
        
        if transformation_type == 'color_mapping':
            # Apply color mapping based on neural analysis
            color_transitions = neural_patterns['color_semantics']['color_transitions']
            if color_transitions:
                most_common_transition = max(color_transitions.items(), key=lambda x: x[1])[0]
                if '->' in most_common_transition:
                    from_color, to_color = most_common_transition.split('->')
                    try:
                        from_color, to_color = int(from_color), int(to_color)
                        result = test_input.copy()
                        result[result == from_color] = to_color
                        return result
                    except:
                        pass
        
        elif transformation_type == 'pattern_completion':
            # Try pattern completion using symmetry
            for axis in ['horizontal', 'vertical']:
                try:
                    symmetry_op = SymmetryOp(axis)
                    result = symmetry_op.apply(test_input)
                    if not np.array_equal(result, test_input):
                        return result
                except:
                    continue
        
        # Object dynamics guided transformations
        object_dynamics = neural_patterns['object_dynamics']
        if object_dynamics['object_count_changes']:
            avg_count_change = np.mean(object_dynamics['object_count_changes'])
            
            if avg_count_change < -0.5:  # Objects being removed
                # Try gravity operations
                for direction in ['down', 'up', 'left', 'right']:
                    try:
                        gravity_op = GravityOp(direction)
                        result = gravity_op.apply(test_input)
                        if not np.array_equal(result, test_input):
                            return result
                    except:
                        continue
        
        return None
    
    def _meta_learning_fallback(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                               test_input: np.ndarray, neural_patterns: Dict[str, Any], 
                               debug: bool = False) -> Optional[np.ndarray]:
        """Meta-learning based fallback using accumulated knowledge."""
        
        # Try most successful operations from cache
        if self.neural_recognizer.success_patterns:
            for pattern_solution in self.neural_recognizer.success_patterns[-10:]:  # Last 10 successful patterns
                try:
                    result = pattern_solution.execute(test_input)
                    return result
                except:
                    continue
        
        # Intelligent fallback based on transformation type
        transformation_type = neural_patterns['transformation_type']
        
        if transformation_type == 'geometric':
            # Try all geometric transformations
            for transform in [np.fliplr, np.flipud, np.rot90, np.transpose]:
                try:
                    result = transform(test_input)
                    return result
                except:
                    continue
        
        elif transformation_type == 'size_scaling':
            # Try common scaling factors
            for scale in [2, 3, 4]:
                try:
                    result = np.repeat(np.repeat(test_input, scale, axis=0), scale, axis=1)
                    return result
                except:
                    continue
        
        return None
    
    def _looks_like_geometric_transform(self, inp: np.ndarray, candidate: np.ndarray) -> bool:
        """Check if candidate looks like a geometric transformation of input."""
        if inp.shape != candidate.shape:
            return False
        
        # Check if it's a known geometric transformation
        transforms = [np.fliplr, np.flipud, np.transpose, 
                     lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3)]
        
        for transform in transforms:
            try:
                if np.array_equal(transform(inp), candidate):
                    return True
            except:
                continue
        
        return False
    
    def _select_most_similar_output(self, train_examples: List[Tuple[np.ndarray, np.ndarray]], 
                                   test_input: np.ndarray) -> np.ndarray:
        """Select the most similar training output as final fallback."""
        if not train_examples:
            return np.array([[0]], dtype=int)
        
        best_output = train_examples[0][1]
        best_similarity = 0
        
        for inp, out in train_examples:
            if inp.shape == test_input.shape:
                similarity = np.mean(inp == test_input)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_output = out
        
        return best_output

def solve_with_advanced_methods(challenge: Dict, solution: Dict = None, debug: bool = False) -> np.ndarray:
    """Main entry point for advanced ARC solving."""
    solver = EnsembleSolver()
    return solver.solve_task(challenge, debug=debug)

# Test the advanced solver
if __name__ == "__main__":
    # Example usage
    example_task = {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
            {"input": [[0, 0], [2, 0]], "output": [[2, 2], [2, 2]]},
        ],
        "test": [
            {"input": [[0, 0], [0, 3]]}
        ]
    }
    
    result = solve_with_advanced_methods(example_task, debug=True)
    print(f"Result: {result}")
