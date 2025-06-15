"""
ARC Prize 2025 Competition Module
Clean, standardized imports for all ARC components
"""

# Add the parent directory to Python path for core imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Core AGI imports
from core.hybrid_agi import HybridAGI
from core.cognitive_agent import CognitiveAgent

# ARC-specific imports  
from .data_loader import load_official_arc_data, ARCTask
from .model_loader import TrainedModelLoader, load_best_trained_agi

__version__ = "2.0.0"
__all__ = [
    'HybridAGI', 
    'CognitiveAgent',
    'load_official_arc_data', 
    'ARCTask',
    'TrainedModelLoader',
    'load_best_trained_agi'
]
