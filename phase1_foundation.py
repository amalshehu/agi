# phase1_foundation.py
"""
🚀 ARC Prize 2025 - Phase 1: Foundation Setup
Human-like abstraction engine with dual-pathway architecture
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import cv2
from scipy import ndimage
from collections import defaultdict
import sympy as sp
from sympy import symbols, Eq, solve
import random

class ObjectType(Enum):
    """Semantic object types for ARC puzzles"""
    SINGLE_PIXEL = "single_pixel"
    LINE_HORIZONTAL = "line_horizontal" 
    LINE_VERTICAL = "line_vertical"
    LINE_DIAGONAL = "line_diagonal"
    RECTANGLE_FILLED = "rectangle_filled"
    RECTANGLE_HOLLOW = "rectangle_hollow"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    CROSS = "cross"
    SCATTERED = "scattered"
    FRAME = "frame"
    CORNER = "corner"
    UNKNOWN = "unknown"

class TransformationType(Enum):
    """Primitive transformation types"""
    IDENTITY = "identity"
    TRANSLATION = "translation"
    ROTATION_90 = "rotation_90"
    ROTATION_180 = "rotation_180" 
    ROTATION_270 = "rotation_270"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    SCALE_UNIFORM = "scale_uniform"
    COLOR_MAP = "color_map"
    FILL_PATTERN = "fill_pattern"
    CONNECT_OBJECTS = "connect_objects"
    GRAVITY = "gravity"
    SYMMETRY_COMPLETE = "symmetry_complete"

@dataclass
class SemanticObject:
    """Enhanced semantic representation of ARC objects"""
    id: int
    object_type: ObjectType
    pixels: List[Tuple[int, int]]  # List of (row, col) coordinates
    color: int
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    center: Tuple[float, float]  # (center_row, center_col)
    area: int
    perimeter: int
    compactness: float  # 4π*area/perimeter²
    orientation: float  # Principal axis angle
    symmetries: List[str]  # ["horizontal", "vertical", "diagonal", "rotational"]
    topology: Dict[str, any]  # Holes, connectivity, etc.
    
    def __post_init__(self):
        """Compute derived properties"""
        if self.pixels:
            self.area = len(self.pixels)
            self.perimeter = self._compute_perimeter()
            self.compactness = 4 * np.pi * self.area / (self.perimeter ** 2) if self.perimeter > 0 else 0
            self.orientation = self._compute_orientation()
            self.symmetries = self._detect_symmetries()
            self.topology = self._analyze_topology()
    
    def _compute_perimeter(self) -> int:
        """Compute object perimeter using 4-connectivity"""
        pixel_set = set(self.pixels)
        perimeter = 0
        
        for r, c in self.pixels:
            # Check 4-connected neighbors
            neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
            boundary_edges = sum(1 for nr, nc in neighbors if (nr, nc) not in pixel_set)
            perimeter += boundary_edges
            
        return perimeter
    
    def _compute_orientation(self) -> float:
        """Compute principal axis orientation with performance optimization"""
        if len(self.pixels) < 2:
            return 0.0
        
        # Performance optimization: limit large objects to avoid covariance bottleneck
        coords = np.array(self.pixels)
        if len(coords) > 1000:  # Sample large objects
            indices = np.random.choice(len(coords), 1000, replace=False)
            coords = coords[indices]
            
        try:
            cov_matrix = np.cov(coords.T)
            if cov_matrix.size == 1:  # Handle 1D case
                return 0.0
            eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
            
            # Get angle of principal eigenvector
            principal_vec = eigenvecs[:, np.argmax(eigenvals)]
            return np.arctan2(principal_vec[1], principal_vec[0])
        except Exception:
            # Fallback: return 0 if computation fails
            return 0.0
    
    def _detect_symmetries(self) -> List[str]:
        """Detect object symmetries"""
        symmetries = []
        
        if len(self.pixels) < 2:
            return ["all"]
            
        coords = np.array(self.pixels)
        center_r, center_c = self.center
        
        # Check horizontal symmetry
        reflected_h = [(int(2*center_r - r), c) for r, c in self.pixels]
        if set(reflected_h) == set(self.pixels):
            symmetries.append("horizontal")
            
        # Check vertical symmetry  
        reflected_v = [(r, int(2*center_c - c)) for r, c in self.pixels]
        if set(reflected_v) == set(self.pixels):
            symmetries.append("vertical")
            
        # Check rotational symmetry (180°)
        reflected_rot = [(int(2*center_r - r), int(2*center_c - c)) for r, c in self.pixels]
        if set(reflected_rot) == set(self.pixels):
            symmetries.append("rotational_180")
            
        return symmetries
    
    def _analyze_topology(self) -> Dict[str, any]:
        """Analyze topological features"""
        return {
            "is_connected": True,  # Flood-fill guarantees this
            "has_holes": self._has_holes(),
            "connectivity": self._compute_connectivity(),
            "convex_hull_ratio": self._convex_hull_ratio()
        }
    
    def _has_holes(self) -> bool:
        """Check if object has holes using flood-fill from boundary"""
        min_r, min_c, max_r, max_c = self.bounding_box
        
        # Create binary mask
        mask = np.zeros((max_r - min_r + 3, max_c - min_c + 3), dtype=bool)
        for r, c in self.pixels:
            mask[r - min_r + 1, c - min_c + 1] = True
            
        # Flood fill from boundary - if any interior pixels remain unfilled, we have holes
        filled = np.zeros_like(mask)
        # Start flood fill from all boundary pixels
        for i in range(mask.shape[0]):
            for j in [0, mask.shape[1]-1]:
                if not mask[i, j] and not filled[i, j]:
                    self._flood_fill(mask, filled, i, j)
        for i in [0, mask.shape[0]-1]:
            for j in range(mask.shape[1]):
                if not mask[i, j] and not filled[i, j]:
                    self._flood_fill(mask, filled, i, j)
                    
        # Check if any non-object pixels remain unfilled (indicating holes)
        interior_unfilled = (~mask) & (~filled)
        return np.any(interior_unfilled)
    
    def _flood_fill(self, mask: np.ndarray, filled: np.ndarray, start_r: int, start_c: int):
        """Flood fill helper function"""
        if (start_r < 0 or start_r >= mask.shape[0] or 
            start_c < 0 or start_c >= mask.shape[1] or
            mask[start_r, start_c] or filled[start_r, start_c]):
            return
            
        filled[start_r, start_c] = True
        
        # Recursively fill neighbors
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            self._flood_fill(mask, filled, start_r + dr, start_c + dc)
    
    def _compute_connectivity(self) -> int:
        """Compute 4-connectivity number"""
        # For now, return number of 4-connected components (should be 1)
        return 1
    
    def _convex_hull_ratio(self) -> float:
        """Ratio of object area to convex hull area"""
        if len(self.pixels) < 3:
            return 1.0
            
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.pixels)
            return self.area / hull.volume
        except:
            return 1.0

class AdvancedObjectExtractor:
    """Phase 1: Enhanced object extraction with semantic understanding"""
    
    def __init__(self):
        self.object_type_classifier = ObjectTypeClassifier()
        
    def extract_semantic_objects(self, grid: np.ndarray) -> List[SemanticObject]:
        """Extract objects with full semantic analysis"""
        objects = []
        object_id = 0
        
        # Get all unique colors except background (0)
        colors = [c for c in np.unique(grid) if c != 0]
        
        for color in colors:
            # Extract all connected components of this color
            component_objects = self._extract_color_components(grid, color, object_id)
            objects.extend(component_objects)
            object_id += len(component_objects)
            
        return objects
    
    def _extract_color_components(self, grid: np.ndarray, color: int, start_id: int) -> List[SemanticObject]:
        """Extract connected components for a specific color"""
        # Create binary mask for this color
        mask = (grid == color).astype(np.uint8)
        
        # Find connected components using OpenCV
        num_components, labels = cv2.connectedComponents(mask, connectivity=4)
        
        objects = []
        for component_id in range(1, num_components):  # Skip background (0)
            # Get pixels for this component
            component_mask = (labels == component_id)
            pixels = list(zip(*np.where(component_mask)))
            
            if not pixels:
                continue
                
            # Compute bounding box
            rows, cols = zip(*pixels)
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)
            bounding_box = (min_row, min_col, max_row, max_col)
            
            # Compute center
            center = (np.mean(rows), np.mean(cols))
            
            # Classify object type
            object_type = self.object_type_classifier.classify(pixels, bounding_box)
            
            # Create semantic object
            semantic_obj = SemanticObject(
                id=start_id + len(objects),
                object_type=object_type,
                pixels=pixels,
                color=color,
                bounding_box=bounding_box,
                center=center,
                area=0,  # Will be computed in __post_init__
                perimeter=0,  # Will be computed in __post_init__
                compactness=0.0,  # Will be computed in __post_init__
                orientation=0.0,  # Will be computed in __post_init__
                symmetries=[],  # Will be computed in __post_init__
                topology={}  # Will be computed in __post_init__
            )
            
            objects.append(semantic_obj)
            
        return objects

class ObjectTypeClassifier:
    """Classify semantic object types based on geometric properties"""
    
    def classify(self, pixels: List[Tuple[int, int]], bounding_box: Tuple[int, int, int, int]) -> ObjectType:
        """Classify object type based on geometric analysis"""
        if len(pixels) == 1:
            return ObjectType.SINGLE_PIXEL
            
        min_row, min_col, max_row, max_col = bounding_box
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        area = len(pixels)
        bbox_area = width * height
        
        # Check for lines
        if width == 1 and height > 1:
            return ObjectType.LINE_VERTICAL
        elif height == 1 and width > 1:
            return ObjectType.LINE_HORIZONTAL
        elif width == height and self._is_diagonal_line(pixels):
            return ObjectType.LINE_DIAGONAL
            
        # Check for rectangles
        if area == bbox_area:
            return ObjectType.RECTANGLE_FILLED
        elif self._is_hollow_rectangle(pixels, bounding_box):
            return ObjectType.RECTANGLE_HOLLOW
            
        # Check for specific shapes
        if self._is_l_shape(pixels, bounding_box):
            return ObjectType.L_SHAPE
        elif self._is_t_shape(pixels, bounding_box):
            return ObjectType.T_SHAPE
        elif self._is_cross(pixels, bounding_box):
            return ObjectType.CROSS
        elif self._is_frame(pixels, bounding_box):
            return ObjectType.FRAME
        elif self._is_corner(pixels, bounding_box):
            return ObjectType.CORNER
            
        # Check if scattered (low compactness)
        perimeter = self._compute_simple_perimeter(pixels)
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        if compactness < 0.3:
            return ObjectType.SCATTERED
            
        return ObjectType.UNKNOWN
    
    def _is_diagonal_line(self, pixels: List[Tuple[int, int]]) -> bool:
        """Check if pixels form a diagonal line"""
        if len(pixels) < 2:
            return False
            
        # Check if all pixels lie on the same diagonal
        sorted_pixels = sorted(pixels)
        first_r, first_c = sorted_pixels[0]
        
        # Check positive diagonal (slope = 1)
        pos_diag = all(r - first_r == c - first_c for r, c in pixels)
        # Check negative diagonal (slope = -1)  
        neg_diag = all(r - first_r == -(c - first_c) for r, c in pixels)
        
        return pos_diag or neg_diag
    
    def _is_hollow_rectangle(self, pixels: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if object is a hollow rectangle (frame)"""
        min_r, min_c, max_r, max_c = bbox
        pixel_set = set(pixels)
        
        # Check if all boundary pixels are present and interior is empty
        boundary_pixels = set()
        
        # Top and bottom edges
        for c in range(min_c, max_c + 1):
            boundary_pixels.add((min_r, c))
            boundary_pixels.add((max_r, c))
            
        # Left and right edges
        for r in range(min_r, max_r + 1):
            boundary_pixels.add((r, min_c))
            boundary_pixels.add((r, max_c))
            
        return pixel_set == boundary_pixels
    
    def _is_l_shape(self, pixels: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if object forms an L shape"""
        # Simple heuristic: L-shape has two dominant orientations
        # and occupies about half the bounding box
        min_r, min_c, max_r, max_c = bbox
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        
        return (len(pixels) / bbox_area > 0.4 and 
                len(pixels) / bbox_area < 0.7 and
                bbox_area > 4)
    
    def _is_t_shape(self, pixels: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if object forms a T shape"""
        # T-shape typically has a horizontal bar and vertical stem
        min_r, min_c, max_r, max_c = bbox
        
        # Check for horizontal concentration at top/bottom
        pixel_set = set(pixels)
        
        # Count pixels in top and bottom rows
        top_pixels = sum(1 for r, c in pixels if r == min_r)
        bottom_pixels = sum(1 for r, c in pixels if r == max_r)
        
        # Check if one edge is dominant and there's a central stem
        width = max_c - min_c + 1
        return (top_pixels >= width * 0.8 or bottom_pixels >= width * 0.8)
    
    def _is_cross(self, pixels: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if object forms a cross/plus shape"""
        min_r, min_c, max_r, max_c = bbox
        center_r = (min_r + max_r) / 2
        center_c = (min_c + max_c) / 2
        
        # Cross should have pixels concentrated around center lines
        horizontal_pixels = sum(1 for r, c in pixels if abs(r - center_r) <= 1)
        vertical_pixels = sum(1 for r, c in pixels if abs(c - center_c) <= 1)
        
        return (horizontal_pixels >= len(pixels) * 0.6 and 
                vertical_pixels >= len(pixels) * 0.6)
    
    def _is_frame(self, pixels: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if object is a frame (hollow rectangle with thick borders)"""
        return self._is_hollow_rectangle(pixels, bbox)
    
    def _is_corner(self, pixels: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if object forms a corner shape"""
        min_r, min_c, max_r, max_c = bbox
        pixel_set = set(pixels)
        
        # Check for L-shaped corner in any of the 4 orientations
        corners = [
            (min_r, min_c),  # Top-left
            (min_r, max_c),  # Top-right  
            (max_r, min_c),  # Bottom-left
            (max_r, max_c)   # Bottom-right
        ]
        
        for corner_r, corner_c in corners:
            if self._forms_corner_at(pixel_set, corner_r, corner_c, bbox):
                return True
                
        return False
    
    def _forms_corner_at(self, pixel_set: Set[Tuple[int, int]], 
                        corner_r: int, corner_c: int, 
                        bbox: Tuple[int, int, int, int]) -> bool:
        """Check if pixels form a corner starting at given position"""
        min_r, min_c, max_r, max_c = bbox
        
        # Check if corner pixel is present
        if (corner_r, corner_c) not in pixel_set:
            return False
            
        # Count adjacent pixels in the two perpendicular directions
        if corner_r == min_r and corner_c == min_c:  # Top-left corner
            right_pixels = sum(1 for r, c in pixel_set if r == min_r and c > min_c)
            down_pixels = sum(1 for r, c in pixel_set if c == min_c and r > min_r)
            return right_pixels >= 1 and down_pixels >= 1
            
        # Similar logic for other corners...
        return False
    
    def _compute_simple_perimeter(self, pixels: List[Tuple[int, int]]) -> int:
        """Simple perimeter computation"""
        pixel_set = set(pixels)
        perimeter = 0
        
        for r, c in pixels:
            neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
            boundary_edges = sum(1 for nr, nc in neighbors if (nr, nc) not in pixel_set)
            perimeter += boundary_edges
            
        return perimeter

def test_phase1_foundation():
    """Test the Phase 1 foundation components"""
    print("🚀 Testing Phase 1 Foundation Components...")
    
    # Test with simple examples
    test_grids = [
        # Horizontal line
        np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]]),
        
        # L-shape
        np.array([[1, 0, 0],
                  [1, 0, 0],
                  [1, 1, 1]]),
        
        # Filled rectangle
        np.array([[2, 2, 2],
                  [2, 2, 2]]),
        
        # Multiple objects
        np.array([[1, 0, 2],
                  [0, 0, 2],
                  [3, 3, 0]])
    ]
    
    extractor = AdvancedObjectExtractor()
    
    for i, grid in enumerate(test_grids):
        print(f"\nTest Grid {i+1}:")
        print(grid)
        
        objects = extractor.extract_semantic_objects(grid)
        print(f"Found {len(objects)} objects:")
        
        for obj in objects:
            print(f"  Object {obj.id}: {obj.object_type.value}")
            print(f"    Color: {obj.color}, Area: {obj.area}")
            print(f"    Symmetries: {obj.symmetries}")
            print(f"    Compactness: {obj.compactness:.3f}")
            print(f"    Center: ({obj.center[0]:.1f}, {obj.center[1]:.1f})")
    
    print("\n✅ Phase 1 Foundation Test Complete!")

if __name__ == "__main__":
    test_phase1_foundation()
