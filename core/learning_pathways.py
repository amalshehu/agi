"""
Learning Pathways - Implements the learning connections between memory systems
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
from .memory_systems import (
    MemoryContent, SensoryMemory, PerceptualAssociativeMemory, 
    SpatialMemory, TransientEpisodicMemory, DeclarativeMemory, 
    ProceduralMemory, SensoryMotorMemory
)
from .consciousness import ConsciousContent, GlobalWorkspace
from .motor_systems import MotorPlan


@dataclass
class LearningEvent:
    """Represents a learning event between memory systems"""
    id: str
    source_system: str
    target_system: str
    content: Any
    learning_type: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


class PerceptualLearning:
    """Learning pathway from Perceptual Associative Memory to Sensory Motor Memory"""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.learning_history: List[LearningEvent] = []
        self.pattern_templates: Dict[str, Any] = {}
    
    def learn_from_perceptual(self, perceptual_memory: PerceptualAssociativeMemory, 
                            sensory_motor_memory: SensoryMotorMemory) -> List[str]:
        """Learn motor patterns from perceptual associations"""
        learned_patterns = []
        
        # Get recent perceptual contents with high activation
        recent_contents = [c for c in perceptual_memory.contents[-20:] 
                          if c.activation_level > 0.5]
        
        for content in recent_contents:
            # Check if this perceptual content suggests a motor pattern
            motor_pattern = self._extract_motor_pattern(content)
            
            if motor_pattern:
                # Store in sensory motor memory
                pattern_id = sensory_motor_memory.update_from_learning(
                    "perceptual_learning", motor_pattern
                )
                learned_patterns.append(pattern_id)
                
                # Record learning event
                learning_event = LearningEvent(
                    id=str(uuid.uuid4()),
                    source_system="perceptual_associative",
                    target_system="sensory_motor",
                    content=motor_pattern,
                    learning_type="perceptual",
                    confidence=content.activation_level,
                    timestamp=datetime.now(),
                    metadata={"source_content_id": content.id}
                )
                self.learning_history.append(learning_event)
        
        return learned_patterns
    
    def _extract_motor_pattern(self, perceptual_content: MemoryContent) -> Optional[Dict[str, Any]]:
        """Extract motor pattern from perceptual content"""
        # Simple pattern extraction based on content type
        if perceptual_content.metadata.get("modality") == "visual":
            # Visual patterns might suggest reaching or grasping
            return {
                "type": "reach_toward",
                "target": perceptual_content.content,
                "confidence": perceptual_content.activation_level
            }
        elif perceptual_content.metadata.get("modality") == "auditory":
            # Auditory patterns might suggest orientation
            return {
                "type": "orient_toward",
                "source": perceptual_content.content,
                "confidence": perceptual_content.activation_level
            }
        
        return None


class SpatialLearning:
    """Learning pathway from Spatial Memory to Sensory Motor Memory"""
    
    def __init__(self):
        self.learning_rate = 0.15
        self.learning_history: List[LearningEvent] = []
        self.spatial_motor_mappings: Dict[str, Any] = {}
    
    def learn_from_spatial(self, spatial_memory: SpatialMemory, 
                         sensory_motor_memory: SensoryMotorMemory) -> List[str]:
        """Learn motor patterns from spatial relationships"""
        learned_patterns = []
        
        # Get spatial relations with high activation
        spatial_contents = [c for c in spatial_memory.contents 
                           if c.activation_level > 0.4]
        
        for content in spatial_contents:
            # Extract navigation or manipulation patterns
            motor_pattern = self._extract_spatial_motor_pattern(content)
            
            if motor_pattern:
                pattern_id = sensory_motor_memory.update_from_learning(
                    "spatial_learning", motor_pattern
                )
                learned_patterns.append(pattern_id)
                
                # Record learning event
                learning_event = LearningEvent(
                    id=str(uuid.uuid4()),
                    source_system="spatial",
                    target_system="sensory_motor",
                    content=motor_pattern,
                    learning_type="spatial",
                    confidence=content.activation_level,
                    timestamp=datetime.now(),
                    metadata={"spatial_relation": content.content}
                )
                self.learning_history.append(learning_event)
        
        return learned_patterns
    
    def _extract_spatial_motor_pattern(self, spatial_content: MemoryContent) -> Optional[Dict[str, Any]]:
        """Extract motor pattern from spatial relationship"""
        relation_data = spatial_content.content
        
        if isinstance(relation_data, dict):
            relation = relation_data.get("relation", "")
            
            if "near" in relation.lower():
                return {
                    "type": "approach",
                    "target": relation_data.get("object2"),
                    "confidence": spatial_content.activation_level
                }
            elif "far" in relation.lower():
                return {
                    "type": "navigate_to",
                    "target": relation_data.get("object2"),
                    "confidence": spatial_content.activation_level
                }
            elif "above" in relation.lower() or "below" in relation.lower():
                return {
                    "type": "vertical_movement",
                    "direction": "up" if "above" in relation.lower() else "down",
                    "confidence": spatial_content.activation_level
                }
        
        return None


class EpisodicLearning:
    """Learning pathway from Transient Episodic Memory to Declarative Memory"""
    
    def __init__(self):
        self.consolidation_threshold = 0.6
        self.learning_history: List[LearningEvent] = []
        self.consolidation_patterns: Dict[str, Any] = {}
    
    def consolidate_episodes(self, transient_memory: TransientEpisodicMemory, 
                           declarative_memory: DeclarativeMemory) -> List[str]:
        """Consolidate episodes from transient to long-term memory"""
        consolidated_ids = []
        
        # Get episodes above consolidation threshold
        episodes_to_consolidate = [ep for ep in transient_memory.episodes 
                                 if ep.activation_level > self.consolidation_threshold]
        
        for episode in episodes_to_consolidate:
            # Consolidate episode
            consolidated_id = declarative_memory.consolidate_episode(episode)
            consolidated_ids.append(consolidated_id)
            
            # Record learning event
            learning_event = LearningEvent(
                id=str(uuid.uuid4()),
                source_system="transient_episodic",
                target_system="declarative",
                content=episode.content,
                learning_type="episodic",
                confidence=episode.activation_level,
                timestamp=datetime.now(),
                metadata={"episode_id": episode.id}
            )
            self.learning_history.append(learning_event)
        
        return consolidated_ids
    
    def extract_facts_from_episodes(self, episodes: List[MemoryContent], 
                                  declarative_memory: DeclarativeMemory) -> List[str]:
        """Extract factual knowledge from episodes"""
        extracted_facts = []
        
        for episode in episodes:
            facts = self._extract_facts(episode)
            
            for fact in facts:
                fact_id = declarative_memory.store_fact(fact, episode.activation_level)
                extracted_facts.append(fact_id)
        
        return extracted_facts
    
    def _extract_facts(self, episode: MemoryContent) -> List[Any]:
        """Extract factual information from episode"""
        facts = []
        episode_data = episode.content
        
        if isinstance(episode_data, dict):
            events = episode_data.get("events", [])
            context = episode_data.get("context", {})
            
            # Extract facts from events
            for event in events:
                if isinstance(event, dict):
                    # Extract object properties
                    for key, value in event.items():
                        if key != "timestamp":
                            facts.append({
                                "subject": key,
                                "predicate": "has_property",
                                "object": value,
                                "context": context
                            })
        
        return facts


class ProceduralLearning:
    """Learning pathway from Global Workspace to Procedural Memory"""
    
    def __init__(self):
        self.learning_rate = 0.2
        self.learning_history: List[LearningEvent] = []
        self.successful_patterns: Dict[str, Any] = {}
    
    def learn_from_consciousness(self, conscious_content: ConsciousContent, 
                               procedural_memory: ProceduralMemory, 
                               success_feedback: bool = True) -> Optional[str]:
        """Learn procedures from conscious processing"""
        
        # Extract procedural pattern from conscious content
        procedure = self._extract_procedure(conscious_content)
        
        if procedure and success_feedback:
            # Store successful procedure
            procedure_id = procedural_memory.store_scheme(
                procedure, f"learned_procedure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Record learning event
            learning_event = LearningEvent(
                id=str(uuid.uuid4()),
                source_system="global_workspace",
                target_system="procedural",
                content=procedure,
                learning_type="procedural",
                confidence=conscious_content.global_availability,
                timestamp=datetime.now(),
                metadata={"coalition_type": conscious_content.coalition.type.value}
            )
            self.learning_history.append(learning_event)
            
            return procedure_id
        
        return None
    
    def _extract_procedure(self, conscious_content: ConsciousContent) -> Optional[Dict[str, Any]]:
        """Extract procedural knowledge from conscious content"""
        coalition = conscious_content.coalition
        
        # Create procedure based on coalition type and contents
        procedure = {
            "type": coalition.type.value,
            "strength_threshold": coalition.strength,
            "coherence_requirement": coalition.coherence,
            "content_patterns": []
        }
        
        # Extract patterns from coalition contents
        for content in coalition.contents:
            pattern = {
                "content_type": content.metadata.get("type", "unknown"),
                "activation_threshold": content.activation_level,
                "associations": content.associations
            }
            procedure["content_patterns"].append(pattern)
        
        return procedure


class AttentionalLearning:
    """Learning pathway from Attention Codelets to Procedural Memory"""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.learning_history: List[LearningEvent] = []
        self.attention_patterns: Dict[str, Any] = {}
    
    def learn_attention_strategies(self, attention_history: List[Dict[str, Any]], 
                                 procedural_memory: ProceduralMemory) -> List[str]:
        """Learn attention strategies from successful attention patterns"""
        learned_strategies = []
        
        # Analyze attention patterns for successful outcomes
        successful_patterns = [h for h in attention_history 
                             if h.get("success", False)]
        
        for pattern in successful_patterns:
            # Create attention strategy
            strategy = self._create_attention_strategy(pattern)
            
            if strategy:
                strategy_id = procedural_memory.store_scheme(
                    strategy, f"attention_strategy_{pattern.get('coalition_type', 'unknown')}"
                )
                learned_strategies.append(strategy_id)
                
                # Record learning event
                learning_event = LearningEvent(
                    id=str(uuid.uuid4()),
                    source_system="attention_codelets",
                    target_system="procedural",
                    content=strategy,
                    learning_type="attentional",
                    confidence=pattern.get("strength", 0.5),
                    timestamp=datetime.now(),
                    metadata={"attention_pattern": pattern}
                )
                self.learning_history.append(learning_event)
        
        return learned_strategies
    
    def _create_attention_strategy(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create attention strategy from successful pattern"""
        return {
            "type": "attention_strategy",
            "coalition_type": pattern.get("coalition_type"),
            "strength_threshold": pattern.get("strength", 0.5),
            "context_requirements": pattern.get("context", {}),
            "success_indicators": pattern.get("success_indicators", [])
        }


class LearningCoordinator:
    """Coordinates all learning pathways"""
    
    def __init__(self):
        self.perceptual_learning = PerceptualLearning()
        self.spatial_learning = SpatialLearning()
        self.episodic_learning = EpisodicLearning()
        self.procedural_learning = ProceduralLearning()
        self.attentional_learning = AttentionalLearning()
        
        self.learning_schedule = {
            "perceptual": 0.1,  # Learning frequency (0-1)
            "spatial": 0.15,
            "episodic": 0.2,
            "procedural": 0.1,
            "attentional": 0.05
        }
    
    def coordinate_learning(self, memory_systems: Dict[str, Any], 
                          conscious_content: Optional[ConsciousContent] = None,
                          motor_feedback: Optional[MotorPlan] = None) -> Dict[str, List[str]]:
        """Coordinate learning across all pathways"""
        learning_results = {}
        
        # Perceptual learning
        if "perceptual" in memory_systems and "sensory_motor" in memory_systems:
            learned = self.perceptual_learning.learn_from_perceptual(
                memory_systems["perceptual"], memory_systems["sensory_motor"]
            )
            learning_results["perceptual"] = learned
        
        # Spatial learning
        if "spatial" in memory_systems and "sensory_motor" in memory_systems:
            learned = self.spatial_learning.learn_from_spatial(
                memory_systems["spatial"], memory_systems["sensory_motor"]
            )
            learning_results["spatial"] = learned
        
        # Episodic learning
        if "transient_episodic" in memory_systems and "declarative" in memory_systems:
            learned = self.episodic_learning.consolidate_episodes(
                memory_systems["transient_episodic"], memory_systems["declarative"]
            )
            learning_results["episodic"] = learned
        
        # Procedural learning from consciousness
        if conscious_content and "procedural" in memory_systems:
            learned = self.procedural_learning.learn_from_consciousness(
                conscious_content, memory_systems["procedural"]
            )
            if learned:
                learning_results["procedural"] = [learned]
        
        return learning_results
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about learning activities"""
        return {
            "perceptual_events": len(self.perceptual_learning.learning_history),
            "spatial_events": len(self.spatial_learning.learning_history),
            "episodic_events": len(self.episodic_learning.learning_history),
            "procedural_events": len(self.procedural_learning.learning_history),
            "attentional_events": len(self.attentional_learning.learning_history)
        }
