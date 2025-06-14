"""
Memory Systems for Cognitive Architecture
Implements the various memory components based on Global Workspace Theory
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum


class MemoryType(Enum):
    SENSORY = "sensory"
    PERCEPTUAL = "perceptual"
    SPATIAL = "spatial"
    EPISODIC = "episodic"
    DECLARATIVE = "declarative"
    PROCEDURAL = "procedural"
    SENSORY_MOTOR = "sensory_motor"


@dataclass
class MemoryContent:
    """Base class for memory content"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    activation_level: float = 1.0
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensoryStimulus:
    """Represents input from internal/external environment"""
    modality: str  # visual, auditory, tactile, etc.
    data: Any
    intensity: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class SensoryMemory:
    """Temporary storage of sensory inputs"""
    
    def __init__(self, capacity: int = 100, decay_rate: float = 0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.contents: List[MemoryContent] = []
    
    def store(self, stimulus: SensoryStimulus) -> MemoryContent:
        """Store sensory stimulus"""
        content = MemoryContent(
            content=stimulus,
            activation_level=stimulus.intensity,
            metadata={"modality": stimulus.modality}
        )
        
        self.contents.append(content)
        self._maintain_capacity()
        return content
    
    def get_cues(self) -> List[MemoryContent]:
        """Extract cues for ventral and dorsal streams"""
        self._decay_activations()
        return [c for c in self.contents if c.activation_level > 0.1]
    
    def _decay_activations(self):
        """Decay activation levels over time"""
        for content in self.contents:
            content.activation_level *= (1 - self.decay_rate)
    
    def _maintain_capacity(self):
        """Remove oldest items if capacity exceeded"""
        if len(self.contents) > self.capacity:
            self.contents = self.contents[-self.capacity:]


class PerceptualAssociativeMemory:
    """Short-term perceptual processing and associative learning"""
    
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.contents: List[MemoryContent] = []
        self.associations: Dict[str, Set[str]] = {}
    
    def process_cues(self, cues: List[MemoryContent]) -> List[MemoryContent]:
        """Process perceptual cues and create associations"""
        processed = []
        
        for cue in cues:
            # Create perceptual representation
            perceptual_content = MemoryContent(
                content=cue.content,
                activation_level=cue.activation_level,
                metadata={**cue.metadata, "processed": True}
            )
            
            # Build associations
            self._build_associations(perceptual_content)
            processed.append(perceptual_content)
        
        self.contents.extend(processed)
        self._maintain_capacity()
        return processed
    
    def _build_associations(self, content: MemoryContent):
        """Build associations between contents"""
        content_id = content.id
        
        # Associate with recently active contents
        for existing in self.contents[-10:]:  # Last 10 items
            if existing.activation_level > 0.3:
                if content_id not in self.associations:
                    self.associations[content_id] = set()
                self.associations[content_id].add(existing.id)
    
    def _maintain_capacity(self):
        if len(self.contents) > self.capacity:
            self.contents = self.contents[-self.capacity:]


class SpatialMemory:
    """Short-term spatial relationships storage"""
    
    def __init__(self, capacity: int = 30):
        self.capacity = capacity
        self.spatial_map: Dict[str, Dict[str, Any]] = {}
        self.contents: List[MemoryContent] = []
    
    def store_spatial_relation(self, object1: str, object2: str, relation: str, confidence: float = 1.0):
        """Store spatial relationship between objects"""
        spatial_content = MemoryContent(
            content={"object1": object1, "object2": object2, "relation": relation},
            activation_level=confidence,
            metadata={"type": "spatial_relation"}
        )
        
        self.contents.append(spatial_content)
        self._maintain_capacity()
        return spatial_content
    
    def get_spatial_context(self, query_object: str) -> List[MemoryContent]:
        """Get spatial context for an object"""
        return [c for c in self.contents 
                if query_object in str(c.content) and c.activation_level > 0.2]
    
    def _maintain_capacity(self):
        if len(self.contents) > self.capacity:
            self.contents = self.contents[-self.capacity:]


class TransientEpisodicMemory:
    """Temporary episodic memory for integration"""
    
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.episodes: List[MemoryContent] = []
    
    def create_episode(self, events: List[Any], context: Dict[str, Any]) -> MemoryContent:
        """Create an episode from events and context"""
        episode = MemoryContent(
            content={"events": events, "context": context},
            metadata={"type": "episode", "event_count": len(events)}
        )
        
        self.episodes.append(episode)
        self._maintain_capacity()
        return episode
    
    def get_recent_episodes(self, count: int = 5) -> List[MemoryContent]:
        """Get most recent episodes"""
        return self.episodes[-count:]
    
    def _maintain_capacity(self):
        if len(self.episodes) > self.capacity:
            self.episodes = self.episodes[-self.capacity:]


class DeclarativeMemory:
    """Long-term storage of facts and events"""
    
    def __init__(self):
        self.facts: Dict[str, MemoryContent] = {}
        self.events: Dict[str, MemoryContent] = {}
        self.local_associations: Dict[str, Set[str]] = {}
    
    def consolidate_episode(self, episode: MemoryContent) -> str:
        """Consolidate episode from transient memory"""
        event_id = episode.id
        self.events[event_id] = episode
        
        # Create local associations
        self._create_local_associations(episode)
        return event_id
    
    def store_fact(self, fact: Any, confidence: float = 1.0) -> str:
        """Store a declarative fact"""
        fact_content = MemoryContent(
            content=fact,
            activation_level=confidence,
            metadata={"type": "fact"}
        )
        
        self.facts[fact_content.id] = fact_content
        return fact_content.id
    
    def retrieve_associations(self, content_id: str) -> List[MemoryContent]:
        """Get local associations for content"""
        associated_ids = self.local_associations.get(content_id, set())
        return [self.facts.get(aid) or self.events.get(aid) 
                for aid in associated_ids if aid in self.facts or aid in self.events]
    
    def _create_local_associations(self, content: MemoryContent):
        """Create associations between related content"""
        content_id = content.id
        
        # Simple association based on temporal proximity and content similarity
        if content_id not in self.local_associations:
            self.local_associations[content_id] = set()
        
        # Associate with recent events
        recent_events = list(self.events.keys())[-5:]
        for event_id in recent_events:
            if event_id != content_id:
                self.local_associations[content_id].add(event_id)


class ProceduralMemory:
    """Long-term storage of learned behaviors (schemes)"""
    
    def __init__(self):
        self.schemes: Dict[str, MemoryContent] = {}
        self.usage_count: Dict[str, int] = {}
    
    def store_scheme(self, scheme: Any, name: str) -> str:
        """Store a behavioral scheme"""
        scheme_content = MemoryContent(
            content=scheme,
            metadata={"type": "scheme", "name": name}
        )
        
        self.schemes[scheme_content.id] = scheme_content
        self.usage_count[scheme_content.id] = 0
        return scheme_content.id
    
    def retrieve_scheme(self, scheme_id: str) -> Optional[MemoryContent]:
        """Retrieve and mark scheme as used"""
        if scheme_id in self.schemes:
            self.usage_count[scheme_id] += 1
            return self.schemes[scheme_id]
        return None
    
    def get_active_schemes(self, threshold: float = 0.3) -> List[MemoryContent]:
        """Get schemes above activation threshold"""
        return [scheme for scheme in self.schemes.values() 
                if scheme.activation_level > threshold]


class SensoryMotorMemory:
    """Encodes motor patterns and supports motor learning"""
    
    def __init__(self):
        self.motor_patterns: Dict[str, MemoryContent] = {}
        self.learning_history: List[Dict[str, Any]] = []
    
    def store_motor_pattern(self, pattern: Any, pattern_type: str) -> str:
        """Store a motor pattern"""
        motor_content = MemoryContent(
            content=pattern,
            metadata={"type": "motor_pattern", "pattern_type": pattern_type}
        )
        
        self.motor_patterns[motor_content.id] = motor_content
        return motor_content.id
    
    def update_from_learning(self, learning_source: str, pattern_data: Any):
        """Update motor patterns from learning systems"""
        self.learning_history.append({
            "source": learning_source,
            "data": pattern_data,
            "timestamp": datetime.now()
        })
        
        # Create or update motor pattern
        pattern_id = self.store_motor_pattern(pattern_data, learning_source)
        return pattern_id
    
    def get_motor_patterns(self, pattern_type: Optional[str] = None) -> List[MemoryContent]:
        """Get motor patterns, optionally filtered by type"""
        patterns = list(self.motor_patterns.values())
        if pattern_type:
            patterns = [p for p in patterns 
                       if p.metadata.get("pattern_type") == pattern_type]
        return patterns
