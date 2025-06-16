# symbolic_reasoning.py
"""
🧠 ARC Prize 2025 - Symbolic Reasoning & Rule Extraction
Phase 1: DSL for transformation patterns and rule discovery
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import sympy as sp
from sympy import symbols, Eq, solve, Matrix
from phase1_foundation import SemanticObject, ObjectType, TransformationType

class RuleOperator(Enum):
    """Symbolic operators for transformation rules"""
    TRANSFORM = "transform"
    IF_THEN = "if_then"
    FOR_EACH = "for_each"
    WHERE = "where"
    APPLY = "apply"
    COMBINE = "combine"
    SEQUENCE = "sequence"

@dataclass
class SymbolicRule:
    """Symbolic representation of transformation rules"""
    id: str
    operator: RuleOperator
    conditions: List[str]  # Symbolic conditions
    actions: List[str]     # Symbolic actions
    parameters: Dict[str, Any]
    confidence: float
    examples_supported: List[int]  # Which training examples this rule explains
    
    def __str__(self) -> str:
        return f"Rule {self.id}: {self.operator.value}({', '.join(self.conditions)}) -> {', '.join(self.actions)}"

class SymbolicTransformation:
    """Symbolic representation of individual transformations"""
    
    def __init__(self, name: str, source_pattern: str, target_pattern: str, 
                 constraints: List[str] = None):
        self.name = name
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern
        self.constraints = constraints or []
        self.parameters = {}
        
    def apply(self, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Apply transformation to objects"""
        # This will be implemented with specific transformation logic
        pass
    
    def can_apply(self, objects: List[SemanticObject]) -> bool:
        """Check if transformation can be applied to given objects"""
        # Check constraints and patterns
        return True

class RuleExtractor:
    """Extract symbolic rules from training examples"""
    
    def __init__(self):
        self.known_transformations = self._initialize_transformations()
        self.symbolic_vars = self._setup_symbolic_variables()
        
    def _setup_symbolic_variables(self) -> Dict[str, sp.Symbol]:
        """Setup symbolic variables for rule extraction"""
        return {
            'x': symbols('x'),  # x-coordinate
            'y': symbols('y'),  # y-coordinate
            'color': symbols('color'),  # object color
            'size': symbols('size'),  # object size
            'angle': symbols('angle'),  # rotation angle
            'scale': symbols('scale'),  # scaling factor
            'dx': symbols('dx'),  # x displacement
            'dy': symbols('dy'),  # y displacement
        }
    
    def _initialize_transformations(self) -> Dict[str, SymbolicTransformation]:
        """Initialize library of known transformations"""
        transformations = {}
        
        # Identity transformation
        transformations['identity'] = SymbolicTransformation(
            name='identity',
            source_pattern='object(x, y, color, properties)',
            target_pattern='object(x, y, color, properties)',
            constraints=[]
        )
        
        # Translation
        transformations['translate'] = SymbolicTransformation(
            name='translate',
            source_pattern='object(x, y, color, properties)',
            target_pattern='object(x + dx, y + dy, color, properties)',
            constraints=['dx != 0 OR dy != 0']
        )
        
        # Rotation (90 degrees)
        transformations['rotate_90'] = SymbolicTransformation(
            name='rotate_90',
            source_pattern='object(x, y, color, properties)',
            target_pattern='object(-y + center_x + center_y, x - center_x + center_y, color, rotate_properties(properties, 90))',
            constraints=['center_x >= 0', 'center_y >= 0']
        )
        
        # Horizontal flip
        transformations['flip_h'] = SymbolicTransformation(
            name='flip_h',
            source_pattern='object(x, y, color, properties)',
            target_pattern='object(2*center_x - x, y, color, flip_properties(properties, "horizontal"))',
            constraints=['center_x >= 0']
        )
        
        # Vertical flip  
        transformations['flip_v'] = SymbolicTransformation(
            name='flip_v',
            source_pattern='object(x, y, color, properties)',
            target_pattern='object(x, 2*center_y - y, color, flip_properties(properties, "vertical"))',
            constraints=['center_y >= 0']
        )
        
        # Uniform scaling
        transformations['scale_uniform'] = SymbolicTransformation(
            name='scale_uniform',
            source_pattern='object(x, y, color, properties)',
            target_pattern='object(x*scale, y*scale, color, scale_properties(properties, scale))',
            constraints=['scale > 0', 'scale != 1']
        )
        
        # Color mapping
        transformations['recolor'] = SymbolicTransformation(
            name='recolor',
            source_pattern='object(x, y, old_color, properties)',
            target_pattern='object(x, y, new_color, properties)',
            constraints=['old_color != new_color', 'color_map(old_color) = new_color']
        )
        
        return transformations
    
    def extract_rules(self, training_examples: List[Tuple[List[SemanticObject], List[SemanticObject]]]) -> List[SymbolicRule]:
        """Extract symbolic rules from training examples"""
        rules = []
        
        # Analyze each training example
        for i, (input_objects, output_objects) in enumerate(training_examples):
            example_rules = self._analyze_example(input_objects, output_objects, i)
            rules.extend(example_rules)
        
        # Consolidate and generalize rules
        consolidated_rules = self._consolidate_rules(rules)
        
        return consolidated_rules
    
    def _analyze_example(self, input_objects: List[SemanticObject], 
                        output_objects: List[SemanticObject], 
                        example_id: int) -> List[SymbolicRule]:
        """Analyze single training example for transformation patterns"""
        rules = []
        
        # Try to match objects between input and output
        object_mappings = self._match_objects(input_objects, output_objects)
        
        for mapping in object_mappings:
            # Extract transformation for this mapping
            transformation = self._identify_transformation(mapping['input'], mapping['output'])
            
            if transformation:
                # Create symbolic rule
                rule = self._create_symbolic_rule(transformation, mapping, example_id)
                rules.append(rule)
        
        return rules
    
    def _match_objects(self, input_objects: List[SemanticObject], 
                      output_objects: List[SemanticObject]) -> List[Dict[str, SemanticObject]]:
        """Match input objects to output objects based on properties"""
        mappings = []
        
        # Simple greedy matching based on color and size similarity
        unmatched_outputs = output_objects.copy()
        
        for inp_obj in input_objects:
            best_match = None
            best_score = -1
            
            for out_obj in unmatched_outputs:
                score = self._compute_similarity(inp_obj, out_obj)
                if score > best_score and score > 0.3:  # Minimum similarity threshold
                    best_score = score
                    best_match = out_obj
            
            if best_match:
                mappings.append({'input': inp_obj, 'output': best_match})
                unmatched_outputs.remove(best_match)
        
        return mappings
    
    def _compute_similarity(self, obj1: SemanticObject, obj2: SemanticObject) -> float:
        """Compute similarity score between two objects"""
        score = 0.0
        
        # Color similarity (most important)
        if obj1.color == obj2.color:
            score += 0.4
        
        # Size similarity
        size_ratio = min(obj1.area, obj2.area) / max(obj1.area, obj2.area)
        score += 0.3 * size_ratio
        
        # Shape similarity
        if obj1.object_type == obj2.object_type:
            score += 0.3
        
        return score
    
    def _identify_transformation(self, input_obj: SemanticObject, 
                               output_obj: SemanticObject) -> Optional[str]:
        """Identify the transformation between input and output objects"""
        
        # Check for identity
        if self._is_identical(input_obj, output_obj):
            return 'identity'
        
        # Check for translation
        if (input_obj.color == output_obj.color and 
            input_obj.area == output_obj.area and
            input_obj.object_type == output_obj.object_type):
            
            dx = output_obj.center[1] - input_obj.center[1]
            dy = output_obj.center[0] - input_obj.center[0]
            
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                return 'translate'
        
        # Check for horizontal flip
        if self._is_horizontal_flip(input_obj, output_obj):
            return 'flip_h'
        
        # Check for vertical flip
        if self._is_vertical_flip(input_obj, output_obj):
            return 'flip_v'
        
        # Check for rotation
        rotation = self._detect_rotation(input_obj, output_obj)
        if rotation:
            return f'rotate_{rotation}'
        
        # Check for scaling
        scale_factor = self._detect_scaling(input_obj, output_obj)
        if scale_factor and scale_factor != 1.0:
            return 'scale_uniform'
        
        # Check for recoloring
        if (input_obj.area == output_obj.area and
            input_obj.object_type == output_obj.object_type and
            input_obj.color != output_obj.color):
            return 'recolor'
        
        return None
    
    def _is_identical(self, obj1: SemanticObject, obj2: SemanticObject) -> bool:
        """Check if two objects are identical"""
        return (obj1.color == obj2.color and
                obj1.area == obj2.area and
                obj1.object_type == obj2.object_type and
                abs(obj1.center[0] - obj2.center[0]) < 0.1 and
                abs(obj1.center[1] - obj2.center[1]) < 0.1)
    
    def _is_horizontal_flip(self, input_obj: SemanticObject, output_obj: SemanticObject) -> bool:
        """Check if output is horizontal flip of input"""
        if (input_obj.color != output_obj.color or 
            input_obj.area != output_obj.area):
            return False
        
        # Check if pixel positions are horizontally mirrored
        input_pixels = set(input_obj.pixels)
        output_pixels = set(output_obj.pixels)
        
        # Find potential flip axis
        min_col = min(min(c for r, c in input_pixels), min(c for r, c in output_pixels))
        max_col = max(max(c for r, c in input_pixels), max(c for r, c in output_pixels))
        flip_axis = (min_col + max_col) / 2
        
        # Check if all pixels are mirrored across this axis
        for r, c in input_pixels:
            mirrored_c = 2 * flip_axis - c
            if (r, int(round(mirrored_c))) not in output_pixels:
                return False
        
        return True
    
    def _is_vertical_flip(self, input_obj: SemanticObject, output_obj: SemanticObject) -> bool:
        """Check if output is vertical flip of input"""
        if (input_obj.color != output_obj.color or 
            input_obj.area != output_obj.area):
            return False
        
        # Similar logic to horizontal flip but for rows
        input_pixels = set(input_obj.pixels)
        output_pixels = set(output_obj.pixels)
        
        min_row = min(min(r for r, c in input_pixels), min(r for r, c in output_pixels))
        max_row = max(max(r for r, c in input_pixels), max(r for r, c in output_pixels))
        flip_axis = (min_row + max_row) / 2
        
        for r, c in input_pixels:
            mirrored_r = 2 * flip_axis - r
            if (int(round(mirrored_r)), c) not in output_pixels:
                return False
        
        return True
    
    def _detect_rotation(self, input_obj: SemanticObject, output_obj: SemanticObject) -> Optional[int]:
        """Detect rotation angle between objects"""
        if (input_obj.color != output_obj.color or 
            input_obj.area != output_obj.area):
            return None
        
        # For simple shapes, check if dimensions are swapped (90° rotation)
        input_bbox = input_obj.bounding_box
        output_bbox = output_obj.bounding_box
        
        input_width = input_bbox[3] - input_bbox[1] + 1
        input_height = input_bbox[2] - input_bbox[0] + 1
        output_width = output_bbox[3] - output_bbox[1] + 1
        output_height = output_bbox[2] - output_bbox[0] + 1
        
        if input_width == output_height and input_height == output_width:
            return 90
        
        # Check for 180° rotation
        if input_width == output_width and input_height == output_height:
            # Could be 180° rotation - need more sophisticated check
            return 180
        
        return None
    
    def _detect_scaling(self, input_obj: SemanticObject, output_obj: SemanticObject) -> Optional[float]:
        """Detect scaling factor between objects"""
        if input_obj.color != output_obj.color:
            return None
        
        # Check area ratio
        if input_obj.area == 0:
            return None
        
        area_ratio = output_obj.area / input_obj.area
        
        # Check if it's a perfect square (uniform scaling)
        scale_factor = np.sqrt(area_ratio)
        
        if abs(scale_factor - round(scale_factor)) < 0.1:
            return round(scale_factor)
        
        return None
    
    def _create_symbolic_rule(self, transformation: str, 
                            mapping: Dict[str, SemanticObject], 
                            example_id: int) -> SymbolicRule:
        """Create symbolic rule from identified transformation"""
        
        input_obj = mapping['input']
        output_obj = mapping['output']
        
        # Create conditions based on input object properties
        conditions = [
            f"object_type = {input_obj.object_type.value}",
            f"color = {input_obj.color}"
        ]
        
        # Create actions based on transformation
        if transformation == 'identity':
            actions = ["maintain_all_properties"]
        elif transformation == 'translate':
            dx = output_obj.center[1] - input_obj.center[1]
            dy = output_obj.center[0] - input_obj.center[0]
            actions = [f"translate(dx={dx:.1f}, dy={dy:.1f})"]
        elif transformation == 'flip_h':
            actions = ["flip_horizontal"]
        elif transformation == 'flip_v':
            actions = ["flip_vertical"]
        elif transformation.startswith('rotate_'):
            angle = transformation.split('_')[1]
            actions = [f"rotate({angle}°)"]
        elif transformation == 'scale_uniform':
            scale = self._detect_scaling(input_obj, output_obj)
            actions = [f"scale_uniform({scale})"]
        elif transformation == 'recolor':
            actions = [f"recolor({input_obj.color} -> {output_obj.color})"]
        else:
            actions = [f"unknown_transformation({transformation})"]
        
        return SymbolicRule(
            id=f"rule_{example_id}_{transformation}",
            operator=RuleOperator.TRANSFORM,
            conditions=conditions,
            actions=actions,
            parameters={
                'transformation': transformation,
                'input_properties': {
                    'type': input_obj.object_type.value,
                    'color': input_obj.color,
                    'area': input_obj.area
                },
                'output_properties': {
                    'type': output_obj.object_type.value,
                    'color': output_obj.color,
                    'area': output_obj.area
                }
            },
            confidence=0.8,  # Initial confidence
            examples_supported=[example_id]
        )
    
    def _consolidate_rules(self, rules: List[SymbolicRule]) -> List[SymbolicRule]:
        """Consolidate similar rules and increase confidence"""
        consolidated = []
        
        # Group rules by transformation type
        rule_groups = {}
        for rule in rules:
            transform_type = rule.parameters.get('transformation', 'unknown')
            if transform_type not in rule_groups:
                rule_groups[transform_type] = []
            rule_groups[transform_type].append(rule)
        
        # Consolidate each group
        for transform_type, group_rules in rule_groups.items():
            if len(group_rules) == 1:
                consolidated.extend(group_rules)
            else:
                # Merge similar rules
                merged_rule = self._merge_rules(group_rules, transform_type)
                consolidated.append(merged_rule)
        
        return consolidated
    
    def _merge_rules(self, rules: List[SymbolicRule], transform_type: str) -> SymbolicRule:
        """Merge similar rules into a single generalized rule"""
        
        # Combine conditions (intersection)
        common_conditions = set(rules[0].conditions)
        for rule in rules[1:]:
            common_conditions &= set(rule.conditions)
        
        # Combine actions (most common)
        all_actions = []
        for rule in rules:
            all_actions.extend(rule.actions)
        
        action_counts = {}
        for action in all_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        most_common_actions = [action for action, count in action_counts.items() 
                              if count >= len(rules) / 2]
        
        # Combine examples supported
        all_examples = []
        for rule in rules:
            all_examples.extend(rule.examples_supported)
        
        # Increase confidence based on number of supporting examples
        confidence = min(0.95, 0.5 + 0.1 * len(set(all_examples)))
        
        return SymbolicRule(
            id=f"merged_{transform_type}_{len(rules)}_examples",
            operator=RuleOperator.TRANSFORM,
            conditions=list(common_conditions),
            actions=most_common_actions,
            parameters={
                'transformation': transform_type,
                'supporting_rules': len(rules),
                'generalized': True
            },
            confidence=confidence,
            examples_supported=list(set(all_examples))
        )

def test_symbolic_reasoning():
    """Test symbolic reasoning components"""
    print("🧠 Testing Symbolic Reasoning & Rule Extraction...")
    
    # Create mock semantic objects for testing
    from phase1_foundation import SemanticObject, ObjectType
    
    # Example 1: Horizontal flip
    input_obj1 = SemanticObject(
        id=0, object_type=ObjectType.LINE_HORIZONTAL,
        pixels=[(1, 1), (1, 2), (1, 3)], color=1,
        bounding_box=(1, 1, 1, 3), center=(1, 2),
        area=3, perimeter=8, compactness=0.5, orientation=0.0,
        symmetries=[], topology={}
    )
    
    output_obj1 = SemanticObject(
        id=0, object_type=ObjectType.LINE_HORIZONTAL,
        pixels=[(1, 5), (1, 6), (1, 7)], color=1,
        bounding_box=(1, 5, 1, 7), center=(1, 6),
        area=3, perimeter=8, compactness=0.5, orientation=0.0,
        symmetries=[], topology={}
    )
    
    # Example 2: Scaling
    input_obj2 = SemanticObject(
        id=1, object_type=ObjectType.RECTANGLE_FILLED,
        pixels=[(0, 0), (0, 1), (1, 0), (1, 1)], color=2,
        bounding_box=(0, 0, 1, 1), center=(0.5, 0.5),
        area=4, perimeter=8, compactness=0.5, orientation=0.0,
        symmetries=[], topology={}
    )
    
    output_obj2 = SemanticObject(
        id=1, object_type=ObjectType.RECTANGLE_FILLED,
        pixels=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), 
                (2, 0), (2, 1), (2, 2)], color=2,
        bounding_box=(0, 0, 2, 2), center=(1, 1),
        area=9, perimeter=12, compactness=0.5, orientation=0.0,
        symmetries=[], topology={}
    )
    
    training_examples = [
        ([input_obj1], [output_obj1]),  # Translation/flip example
        ([input_obj2], [output_obj2])   # Scaling example
    ]
    
    extractor = RuleExtractor()
    rules = extractor.extract_rules(training_examples)
    
    print(f"\nExtracted {len(rules)} symbolic rules:")
    for rule in rules:
        print(f"  {rule}")
        print(f"    Confidence: {rule.confidence:.2f}")
        print(f"    Supports examples: {rule.examples_supported}")
        print()
    
    print("✅ Symbolic Reasoning Test Complete!")

if __name__ == "__main__":
    test_symbolic_reasoning()
