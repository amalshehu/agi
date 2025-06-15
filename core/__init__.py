"""
Breakthrough AGI Core Components
==================================

This module contains the core implementation of our breakthrough hybrid AGI architecture.
"""

from .memory_systems import *
from .consciousness import *
from .motor_systems import *
from .learning_pathways import *
from .cognitive_agent import CognitiveAgent
from .hybrid_agi import HybridAGI

__version__ = "1.0.0"
__author__ = "AGI Research Team"

__all__ = [
    "CognitiveAgent",
    "HybridAGI",
    "MemoryContent",
    "SensoryMemory",
    "PerceptualAssociativeMemory",
    "SpatialMemory",
    "TransientEpisodicMemory",
    "DeclarativeMemory",
    "ProceduralMemory",
    "SensoryMotorMemory",
    "GlobalWorkspace",
    "CurrentSituationalModel",
    "StructureBuildingCodelet",
    "AttentionCodelet",
    "Coalition",
    "ConsciousContent",
    "ActionSelection",
    "MotorPlanExecution",
    "LearningCoordinator"
]
