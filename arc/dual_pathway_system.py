# dual_pathway_system.py
"""
🧠⚙️ ARC Prize 2025 - Dual-Pathway Neural-Symbolic System
Phase 1: Integration of neural pattern detection with symbolic reasoning
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from phase1_foundation import AdvancedObjectExtractor, SemanticObject, ObjectType
from symbolic_reasoning import RuleExtractor, SymbolicRule, SymbolicTransformation

class PathwayMode(Enum):
    """Processing pathway selection"""
    NEURAL_ONLY = "neural_only"
    SYMBOLIC_ONLY = "symbolic_only"
    DUAL_PATHWAY = "dual_pathway"
    ADAPTIVE = "adaptive"

@dataclass
class ProcessingDecision:
    """Decision on which pathway(s) to use"""
    primary_pathway: str
    confidence: float
    reasoning: str
    fallback_pathway: Optional[str] = None

if TORCH_AVAILABLE:
    class LightweightCNN(nn.Module):
        """Lightweight CNN for pattern detection (based on MobileNetV3)"""
        
        def __init__(self, input_size: int = 30, num_classes: int = 20):
            super().__init__()
            
            # Depthwise separable convolutions for efficiency
            self.features = nn.Sequential(
                # Initial conv
                nn.Conv2d(10, 16, 3, padding=1),  # 10 colors max in ARC
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                
                # Depthwise separable block 1
                self._make_depthwise_sep(16, 32, 3, 1),
                nn.MaxPool2d(2),
                
                # Depthwise separable block 2
                self._make_depthwise_sep(32, 64, 3, 1),
                nn.MaxPool2d(2),
                
                # Depthwise separable block 3
                self._make_depthwise_sep(64, 128, 3, 1),
                nn.AdaptiveAvgPool2d(1)
            )
            
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        
        def _make_depthwise_sep(self, in_channels: int, out_channels: int, 
                               kernel_size: int, stride: int) -> nn.Sequential:
            """Create depthwise separable convolution block"""
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                         padding=kernel_size//2, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                
                # Pointwise
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.features(x)
            features = features.view(features.size(0), -1)
            return self.classifier(features)
else:
    LightweightCNN = None

class NeuralPatternDetector:
    """Neural pathway for low-level pattern detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = LightweightCNN()
            
            if model_path:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None
        
        # Pattern categories the neural network can detect
        self.pattern_categories = [
            'uniform_scaling', 'rotation_90', 'rotation_180', 'rotation_270',
            'flip_horizontal', 'flip_vertical', 'translation', 'color_mapping',
            'object_addition', 'object_removal', 'pattern_completion',
            'symmetry_horizontal', 'symmetry_vertical', 'gravity_effect',
            'flood_fill', 'line_drawing', 'shape_transformation',
            'spatial_relationship', 'complex_composition', 'unknown'
        ]
    
    def detect_patterns(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, float]:
        """Detect transformation patterns using neural network"""
        
        if not TORCH_AVAILABLE or self.model is None:
            # Return uniform low confidence scores when torch not available
            pattern_scores = {}
            for category in self.pattern_categories:
                pattern_scores[category] = 0.1  # Low uniform confidence
            return pattern_scores
        
        # Prepare input for neural network
        neural_input = self._prepare_neural_input(input_grid, output_grid)
        
        with torch.no_grad():
            neural_input = neural_input.to(self.device)
            logits = self.model(neural_input.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Map to pattern categories
        pattern_scores = {}
        for i, category in enumerate(self.pattern_categories):
            pattern_scores[category] = float(probabilities[i])
        
        return pattern_scores
    
    def _prepare_neural_input(self, input_grid: np.ndarray, output_grid: np.ndarray) -> torch.Tensor:
        """Prepare grids for neural network input"""
        
        # Normalize grid sizes to 30x30 (standard ARC size)
        target_size = 30
        
        input_resized = self._resize_grid(input_grid, target_size)
        output_resized = self._resize_grid(output_grid, target_size)
        
        # Create 10-channel input (one-hot encoding for colors 0-9)
        channels = []
        
        # Input grid channels (5 channels)
        for color in range(5):
            channel = (input_resized == color).astype(np.float32)
            channels.append(channel)
        
        # Output grid channels (5 channels) 
        for color in range(5):
            channel = (output_resized == color).astype(np.float32)
            channels.append(channel)
        
        # Stack channels and convert to tensor
        neural_input = np.stack(channels, axis=0)
        if TORCH_AVAILABLE:
            return torch.FloatTensor(neural_input)
        else:
            return neural_input  # Return numpy array when torch not available
    
    def _resize_grid(self, grid: np.ndarray, target_size: int) -> np.ndarray:
        """Resize grid to target size using nearest neighbor"""
        if grid.shape[0] <= target_size and grid.shape[1] <= target_size:
            # Pad to target size
            padded = np.zeros((target_size, target_size), dtype=grid.dtype)
            padded[:grid.shape[0], :grid.shape[1]] = grid
            return padded
        else:
            # Downsample using simple decimation
            step_r = grid.shape[0] // target_size
            step_c = grid.shape[1] // target_size
            return grid[::max(1, step_r), ::max(1, step_c)][:target_size, :target_size]

class PathwayCoordinator:
    """Coordinates between neural and symbolic pathways"""
    
    def __init__(self):
        self.neural_detector = NeuralPatternDetector()
        self.object_extractor = AdvancedObjectExtractor()
        self.rule_extractor = RuleExtractor()
        
        # Decision thresholds
        self.neural_confidence_threshold = 0.7
        self.symbolic_confidence_threshold = 0.6
        self.complexity_threshold = 0.5
    
    def decide_pathway(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                      training_examples: List[Tuple[np.ndarray, np.ndarray]]) -> ProcessingDecision:
        """Decide which pathway(s) to use for processing"""
        
        # Analyze puzzle complexity
        complexity_score = self._assess_complexity(input_grid, output_grid)
        
        # Get neural pattern confidence
        neural_patterns = self.neural_detector.detect_patterns(input_grid, output_grid)
        max_neural_confidence = max(neural_patterns.values())
        
        # Get symbolic analysis capability
        symbolic_feasibility = self._assess_symbolic_feasibility(input_grid, output_grid)
        
        # Decision logic
        if max_neural_confidence > self.neural_confidence_threshold and complexity_score < self.complexity_threshold:
            return ProcessingDecision(
                primary_pathway="neural",
                confidence=max_neural_confidence,
                reasoning=f"High neural confidence ({max_neural_confidence:.3f}) for simple pattern",
                fallback_pathway="symbolic"
            )
        
        elif symbolic_feasibility > self.symbolic_confidence_threshold:
            return ProcessingDecision(
                primary_pathway="symbolic", 
                confidence=symbolic_feasibility,
                reasoning=f"Good symbolic feasibility ({symbolic_feasibility:.3f})",
                fallback_pathway="neural"
            )
        
        else:
            return ProcessingDecision(
                primary_pathway="dual",
                confidence=(max_neural_confidence + symbolic_feasibility) / 2,
                reasoning="Moderate confidence in both pathways - using dual approach",
                fallback_pathway=None
            )
    
    def _assess_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> float:
        """Assess puzzle complexity (0 = simple, 1 = complex)"""
        complexity_factors = []
        
        # Grid size complexity
        total_pixels = input_grid.size + output_grid.size
        size_complexity = min(1.0, total_pixels / 1800)  # Normalize by max ARC size
        complexity_factors.append(size_complexity)
        
        # Color diversity
        unique_colors = len(set(input_grid.flatten()) | set(output_grid.flatten()))
        color_complexity = min(1.0, unique_colors / 10)  # Max 10 colors in ARC
        complexity_factors.append(color_complexity)
        
        # Object count complexity
        input_objects = self.object_extractor.extract_semantic_objects(input_grid)
        output_objects = self.object_extractor.extract_semantic_objects(output_grid)
        object_complexity = min(1.0, (len(input_objects) + len(output_objects)) / 20)
        complexity_factors.append(object_complexity)
        
        # Shape diversity complexity
        input_shapes = set(obj.object_type for obj in input_objects)
        output_shapes = set(obj.object_type for obj in output_objects)
        shape_complexity = min(1.0, len(input_shapes | output_shapes) / 10)
        complexity_factors.append(shape_complexity)
        
        return np.mean(complexity_factors)
    
    def _assess_symbolic_feasibility(self, input_grid: np.ndarray, output_grid: np.ndarray) -> float:
        """Assess how feasible symbolic reasoning is for this puzzle"""
        feasibility_score = 0.0
        
        # Extract objects
        input_objects = self.object_extractor.extract_semantic_objects(input_grid)
        output_objects = self.object_extractor.extract_semantic_objects(output_grid)
        
        # Object count similarity (good for 1-to-1 mappings)
        if len(input_objects) > 0 and len(output_objects) > 0:
            count_ratio = min(len(input_objects), len(output_objects)) / max(len(input_objects), len(output_objects))
            feasibility_score += 0.3 * count_ratio
        
        # Recognizable object types
        known_types = [obj for obj in input_objects + output_objects 
                      if obj.object_type != ObjectType.UNKNOWN]
        if len(input_objects) + len(output_objects) > 0:
            type_recognition_rate = len(known_types) / (len(input_objects) + len(output_objects))
            feasibility_score += 0.3 * type_recognition_rate
        
        # Grid structure preservation
        if input_grid.shape == output_grid.shape:
            feasibility_score += 0.2
        
        # Color consistency
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        color_overlap = len(input_colors & output_colors) / len(input_colors | output_colors) if input_colors | output_colors else 0
        feasibility_score += 0.2 * color_overlap
        
        return min(1.0, feasibility_score)

class DualPathwaySystem:
    """Main dual-pathway neural-symbolic system"""
    
    def __init__(self):
        self.coordinator = PathwayCoordinator()
        self.neural_detector = self.coordinator.neural_detector
        self.object_extractor = self.coordinator.object_extractor
        self.rule_extractor = self.coordinator.rule_extractor
        
    def analyze_puzzle(self, training_examples: List[Tuple[np.ndarray, np.ndarray]], 
                      test_input: np.ndarray) -> Dict[str, Any]:
        """Analyze puzzle using dual-pathway approach"""
        
        if not training_examples:
            return {"error": "No training examples provided"}
        
        # Analyze first training example to decide pathway
        input_grid, output_grid = training_examples[0]
        decision = self.coordinator.decide_pathway(input_grid, output_grid, training_examples)
        
        results = {
            "pathway_decision": decision,
            "neural_analysis": {},
            "symbolic_analysis": {},
            "integrated_solution": {}
        }
        
        # Execute primary pathway
        if decision.primary_pathway in ["neural", "dual"]:
            results["neural_analysis"] = self._neural_analysis(training_examples, test_input)
        
        if decision.primary_pathway in ["symbolic", "dual"]:
            results["symbolic_analysis"] = self._symbolic_analysis(training_examples, test_input)
        
        # Integrate results
        results["integrated_solution"] = self._integrate_analyses(results, decision)
        
        return results
    
    def _neural_analysis(self, training_examples: List[Tuple[np.ndarray, np.ndarray]], 
                        test_input: np.ndarray) -> Dict[str, Any]:
        """Perform neural pathway analysis"""
        
        pattern_detections = []
        
        for i, (inp, out) in enumerate(training_examples):
            patterns = self.neural_detector.detect_patterns(inp, out)
            pattern_detections.append({
                "example_id": i,
                "patterns": patterns,
                "top_pattern": max(patterns.items(), key=lambda x: x[1])
            })
        
        # Find consistent patterns across examples
        consistent_patterns = self._find_consistent_neural_patterns(pattern_detections)
        
        return {
            "individual_detections": pattern_detections,
            "consistent_patterns": consistent_patterns,
            "recommended_transformation": consistent_patterns[0] if consistent_patterns else None
        }
    
    def _symbolic_analysis(self, training_examples: List[Tuple[np.ndarray, np.ndarray]], 
                          test_input: np.ndarray) -> Dict[str, Any]:
        """Perform symbolic pathway analysis"""
        
        # Extract semantic objects for all examples
        semantic_examples = []
        for inp, out in training_examples:
            inp_objects = self.object_extractor.extract_semantic_objects(inp)
            out_objects = self.object_extractor.extract_semantic_objects(out)
            semantic_examples.append((inp_objects, out_objects))
        
        # Extract symbolic rules
        rules = self.rule_extractor.extract_rules(semantic_examples)
        
        # Analyze test input
        test_objects = self.object_extractor.extract_semantic_objects(test_input)
        
        return {
            "semantic_examples": len(semantic_examples),
            "extracted_rules": [{"id": rule.id, "confidence": rule.confidence, 
                               "conditions": rule.conditions, "actions": rule.actions} 
                               for rule in rules],
            "test_objects": [{"id": obj.id, "type": obj.object_type.value, 
                            "color": obj.color, "area": obj.area} 
                           for obj in test_objects],
            "applicable_rules": [rule for rule in rules if self._rule_applies(rule, test_objects)]
        }
    
    def _find_consistent_neural_patterns(self, detections: List[Dict]) -> List[Tuple[str, float]]:
        """Find patterns consistent across multiple examples"""
        
        if not detections:
            return []
        
        # Average pattern scores across examples
        pattern_averages = {}
        pattern_names = detections[0]["patterns"].keys()
        
        for pattern_name in pattern_names:
            scores = [detection["patterns"][pattern_name] for detection in detections]
            pattern_averages[pattern_name] = np.mean(scores)
        
        # Sort by average confidence
        sorted_patterns = sorted(pattern_averages.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by minimum threshold
        consistent_patterns = [(name, score) for name, score in sorted_patterns 
                             if score > 0.3]
        
        return consistent_patterns[:3]  # Top 3 patterns
    
    def _rule_applies(self, rule: SymbolicRule, test_objects: List[SemanticObject]) -> bool:
        """Check if a symbolic rule applies to test objects"""
        
        # Simple check based on conditions
        for condition in rule.conditions:
            if "object_type" in condition and test_objects:
                # Check if any test object matches the condition
                condition_satisfied = False
                for obj in test_objects:
                    if f"object_type = {obj.object_type.value}" == condition:
                        condition_satisfied = True
                        break
                if not condition_satisfied:
                    return False
        
        return True
    
    def _integrate_analyses(self, results: Dict[str, Any], decision: ProcessingDecision) -> Dict[str, Any]:
        """Integrate neural and symbolic analyses"""
        
        integration = {
            "primary_pathway": decision.primary_pathway,
            "confidence": decision.confidence,
            "recommendations": []
        }
        
        # Neural recommendations
        if "neural_analysis" in results and results["neural_analysis"]:
            neural_rec = results["neural_analysis"].get("recommended_transformation")
            if neural_rec:
                integration["recommendations"].append({
                    "source": "neural",
                    "transformation": neural_rec[0],
                    "confidence": neural_rec[1],
                    "type": "pattern_based"
                })
        
        # Symbolic recommendations  
        if "symbolic_analysis" in results and results["symbolic_analysis"]:
            applicable_rules = results["symbolic_analysis"].get("applicable_rules", [])
            for rule in applicable_rules:
                integration["recommendations"].append({
                    "source": "symbolic", 
                    "transformation": rule.actions,
                    "confidence": rule.confidence,
                    "type": "rule_based"
                })
        
        # Rank recommendations by confidence
        integration["recommendations"].sort(key=lambda x: x["confidence"], reverse=True)
        
        return integration

def test_dual_pathway_system():
    """Test the dual-pathway system"""
    print("🧠⚙️ Testing Dual-Pathway Neural-Symbolic System...")
    
    # Create test examples
    # Example 1: Simple horizontal flip
    input1 = np.array([[1, 0, 0],
                       [1, 0, 0],
                       [1, 1, 1]])
    
    output1 = np.array([[0, 0, 1],
                        [0, 0, 1], 
                        [1, 1, 1]])
    
    # Example 2: Similar pattern
    input2 = np.array([[2, 0], 
                       [2, 2]])
    
    output2 = np.array([[0, 2],
                        [2, 2]])
    
    training_examples = [(input1, output1), (input2, output2)]
    test_input = np.array([[3, 0, 0, 0],
                          [3, 3, 0, 0]])
    
    # Initialize system
    system = DualPathwaySystem()
    
    # Analyze puzzle
    analysis = system.analyze_puzzle(training_examples, test_input)
    
    print(f"\nPathway Decision: {analysis['pathway_decision'].primary_pathway}")
    print(f"Confidence: {analysis['pathway_decision'].confidence:.3f}")
    print(f"Reasoning: {analysis['pathway_decision'].reasoning}")
    
    if analysis.get("neural_analysis"):
        neural = analysis["neural_analysis"]
        if neural.get("recommended_transformation"):
            pattern, conf = neural["recommended_transformation"]
            print(f"\nNeural Recommendation: {pattern} (confidence: {conf:.3f})")
    
    if analysis.get("symbolic_analysis"):
        symbolic = analysis["symbolic_analysis"]
        print(f"\nSymbolic Analysis:")
        print(f"  Extracted {len(symbolic['extracted_rules'])} rules")
        print(f"  Found {len(symbolic['applicable_rules'])} applicable rules")
    
    print(f"\nIntegrated Recommendations:")
    for i, rec in enumerate(analysis["integrated_solution"]["recommendations"][:3]):
        print(f"  {i+1}. {rec['transformation']} ({rec['source']}, conf: {rec['confidence']:.3f})")
    
    print("\n✅ Dual-Pathway System Test Complete!")

if __name__ == "__main__":
    test_dual_pathway_system()
