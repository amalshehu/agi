"""
Consciousness Layer - Global Workspace Theory Implementation
Includes Global Workspace, Current Situational Model, and Codelets
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum
import heapq
from memory_systems import MemoryContent


class CoalitionType(Enum):
    PERCEPTUAL = "perceptual"
    SPATIAL = "spatial"
    EPISODIC = "episodic"
    MOTOR = "motor"
    ATTENTION = "attention"


@dataclass
class Coalition:
    """Represents a coalition of contents competing for consciousness"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: CoalitionType = CoalitionType.PERCEPTUAL
    contents: List[MemoryContent] = field(default_factory=list)
    strength: float = 0.0
    coherence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_strength(self) -> float:
        """Calculate coalition strength based on content activations"""
        if not self.contents:
            return 0.0
        
        # Base strength from content activations
        activation_sum = sum(c.activation_level for c in self.contents)
        
        # Coherence bonus
        coherence_bonus = self.coherence * 0.5
        
        # Recency bonus
        recency_bonus = max(0, 1.0 - (datetime.now() - self.timestamp).total_seconds() / 3600)
        
        self.strength = activation_sum + coherence_bonus + recency_bonus
        return self.strength


@dataclass
class ConsciousContent:
    """Content that has won the competition and entered consciousness"""
    coalition: Coalition
    global_availability: float = 1.0
    broadcast_timestamp: datetime = field(default_factory=datetime.now)
    influence_duration: float = 5.0  # seconds


class CurrentSituationalModel:
    """Integrates percepts, cues, and associations for coherent state representation"""
    
    def __init__(self, max_elements: int = 20):
        self.max_elements = max_elements
        self.current_state: Dict[str, Any] = {}
        self.active_percepts: List[MemoryContent] = []
        self.spatial_context: Dict[str, Any] = {}
        self.temporal_context: Dict[str, Any] = {}
        self.coherence_score: float = 0.0
        self.last_update: datetime = datetime.now()
    
    def integrate_percepts(self, percepts: List[MemoryContent]):
        """Integrate new perceptual information"""
        self.active_percepts = percepts[-self.max_elements:]
        self._update_coherence()
    
    def integrate_spatial_context(self, spatial_info: Dict[str, Any]):
        """Integrate spatial information"""
        self.spatial_context.update(spatial_info)
        self._update_coherence()
    
    def integrate_associations(self, associations: List[MemoryContent]):
        """Integrate associative information"""
        for assoc in associations:
            key = f"association_{assoc.id}"
            self.current_state[key] = assoc
        self._update_coherence()
    
    def get_coherent_representation(self) -> Dict[str, Any]:
        """Get current coherent representation of the situation"""
        return {
            "percepts": self.active_percepts,
            "spatial_context": self.spatial_context,
            "temporal_context": self.temporal_context,
            "coherence": self.coherence_score,
            "state": self.current_state
        }
    
    def _update_coherence(self):
        """Update coherence score based on consistency"""
        # Simple coherence measure based on content consistency
        if not self.active_percepts:
            self.coherence_score = 0.0
            return
        
        # Check temporal consistency
        time_span = 0
        if len(self.active_percepts) > 1:
            timestamps = [p.timestamp for p in self.active_percepts]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
        
        # Check activation consistency
        activations = [p.activation_level for p in self.active_percepts]
        activation_variance = sum((a - sum(activations)/len(activations))**2 for a in activations) / len(activations)
        
        # Coherence decreases with time span and activation variance
        temporal_coherence = max(0, 1.0 - time_span / 10.0)
        activation_coherence = max(0, 1.0 - activation_variance)
        
        self.coherence_score = (temporal_coherence + activation_coherence) / 2.0
        self.last_update = datetime.now()


class StructureBuildingCodelet:
    """Constructs structured coalitions for the situational model"""
    
    def __init__(self, name: str):
        self.name = name
        self.active = True
        self.priority = 1.0
    
    def build_coalition(self, contents: List[MemoryContent], context: Dict[str, Any]) -> Optional[Coalition]:
        """Build a coalition from available contents"""
        if not contents:
            return None
        
        # Group contents by similarity/relevance
        relevant_contents = self._filter_relevant_contents(contents, context)
        
        if not relevant_contents:
            return None
        
        # Create coalition
        coalition = Coalition(
            type=self._determine_coalition_type(relevant_contents),
            contents=relevant_contents,
            metadata={"builder": self.name, "context": context}
        )
        
        # Calculate coherence
        coalition.coherence = self._calculate_coherence(relevant_contents)
        coalition.calculate_strength()
        
        return coalition
    
    def _filter_relevant_contents(self, contents: List[MemoryContent], context: Dict[str, Any]) -> List[MemoryContent]:
        """Filter contents based on relevance to current context"""
        relevant = []
        threshold = 0.3
        
        for content in contents:
            relevance = self._calculate_relevance(content, context)
            if relevance > threshold:
                relevant.append(content)
        
        return relevant
    
    def _calculate_relevance(self, content: MemoryContent, context: Dict[str, Any]) -> float:
        """Calculate content relevance to context"""
        # Simple relevance based on activation and recency
        base_relevance = content.activation_level
        
        # Recency bonus
        time_diff = (datetime.now() - content.timestamp).total_seconds()
        recency_bonus = max(0, 1.0 - time_diff / 3600)  # Decay over 1 hour
        
        return (base_relevance + recency_bonus) / 2.0
    
    def _determine_coalition_type(self, contents: List[MemoryContent]) -> CoalitionType:
        """Determine the type of coalition based on contents"""
        # Simple heuristic based on content metadata
        spatial_count = sum(1 for c in contents if c.metadata.get("type") == "spatial_relation")
        motor_count = sum(1 for c in contents if c.metadata.get("type") == "motor_pattern")
        episode_count = sum(1 for c in contents if c.metadata.get("type") == "episode")
        
        if spatial_count > len(contents) / 2:
            return CoalitionType.SPATIAL
        elif motor_count > len(contents) / 2:
            return CoalitionType.MOTOR
        elif episode_count > len(contents) / 2:
            return CoalitionType.EPISODIC
        else:
            return CoalitionType.PERCEPTUAL
    
    def _calculate_coherence(self, contents: List[MemoryContent]) -> float:
        """Calculate coherence of contents within coalition"""
        if len(contents) <= 1:
            return 1.0
        
        # Check association overlap
        association_overlap = 0
        for i, content1 in enumerate(contents):
            for content2 in contents[i+1:]:
                if set(content1.associations) & set(content2.associations):
                    association_overlap += 1
        
        max_pairs = len(contents) * (len(contents) - 1) / 2
        return association_overlap / max_pairs if max_pairs > 0 else 0.0


class AttentionCodelet:
    """Directs attention and facilitates attentional learning"""
    
    def __init__(self, name: str, attention_type: str):
        self.name = name
        self.attention_type = attention_type  # "focus", "switch", "sustain"
        self.active = True
        self.priority = 1.0
        self.focus_history: List[Dict[str, Any]] = []
    
    def direct_attention(self, available_coalitions: List[Coalition], 
                        current_focus: Optional[Coalition]) -> Optional[Coalition]:
        """Direct attention to most relevant coalition"""
        if not available_coalitions:
            return current_focus
        
        if self.attention_type == "focus":
            return self._focus_attention(available_coalitions, current_focus)
        elif self.attention_type == "switch":
            return self._switch_attention(available_coalitions, current_focus)
        elif self.attention_type == "sustain":
            return self._sustain_attention(available_coalitions, current_focus)
        
        return current_focus
    
    def _focus_attention(self, coalitions: List[Coalition], current: Optional[Coalition]) -> Coalition:
        """Focus on strongest coalition"""
        return max(coalitions, key=lambda c: c.strength)
    
    def _switch_attention(self, coalitions: List[Coalition], current: Optional[Coalition]) -> Coalition:
        """Switch to different coalition if beneficial"""
        if not current:
            return max(coalitions, key=lambda c: c.strength)
        
        # Switch if another coalition is significantly stronger
        strongest = max(coalitions, key=lambda c: c.strength)
        if strongest.strength > current.strength * 1.5:
            return strongest
        
        return current
    
    def _sustain_attention(self, coalitions: List[Coalition], current: Optional[Coalition]) -> Coalition:
        """Sustain attention on current coalition if still relevant"""
        if not current:
            return max(coalitions, key=lambda c: c.strength)
        
        # Sustain if current coalition is still reasonably strong
        if current in coalitions and current.strength > 0.3:
            return current
        
        return max(coalitions, key=lambda c: c.strength)
    
    def learn_attention_pattern(self, successful_focus: Coalition, context: Dict[str, Any]):
        """Learn from successful attention patterns"""
        self.focus_history.append({
            "coalition_type": successful_focus.type,
            "strength": successful_focus.strength,
            "context": context,
            "timestamp": datetime.now()
        })
        
        # Keep recent history
        if len(self.focus_history) > 100:
            self.focus_history = self.focus_history[-100:]


class GlobalWorkspace:
    """Central arena for conscious processing"""
    
    def __init__(self, max_conscious_duration: float = 5.0):
        self.max_conscious_duration = max_conscious_duration
        self.conscious_contents: List[ConsciousContent] = []
        self.coalition_queue: List[Coalition] = []
        self.competition_threshold: float = 0.5
        self.broadcast_subscribers: List[Callable[[ConsciousContent], None]] = []
    
    def add_coalition(self, coalition: Coalition):
        """Add coalition to competition queue"""
        self.coalition_queue.append(coalition)
    
    def run_competition(self) -> Optional[ConsciousContent]:
        """Run competition among coalitions"""
        if not self.coalition_queue:
            return None
        
        # Remove expired conscious contents
        self._clean_conscious_contents()
        
        # Find winning coalition
        winning_coalition = max(self.coalition_queue, key=lambda c: c.strength)
        
        if winning_coalition.strength > self.competition_threshold:
            # Create conscious content
            conscious_content = ConsciousContent(
                coalition=winning_coalition,
                global_availability=min(1.0, winning_coalition.strength)
            )
            
            self.conscious_contents.append(conscious_content)
            
            # Broadcast to subscribers
            self._broadcast_conscious_content(conscious_content)
            
            # Clear coalition queue
            self.coalition_queue.clear()
            
            return conscious_content
        
        return None
    
    def subscribe_to_broadcasts(self, callback: Callable[[ConsciousContent], None]):
        """Subscribe to conscious content broadcasts"""
        self.broadcast_subscribers.append(callback)
    
    def get_current_conscious_content(self) -> Optional[ConsciousContent]:
        """Get currently active conscious content"""
        active_contents = [c for c in self.conscious_contents 
                          if self._is_content_active(c)]
        
        if active_contents:
            return max(active_contents, key=lambda c: c.global_availability)
        
        return None
    
    def _clean_conscious_contents(self):
        """Remove expired conscious contents"""
        current_time = datetime.now()
        self.conscious_contents = [
            content for content in self.conscious_contents
            if (current_time - content.broadcast_timestamp).total_seconds() < content.influence_duration
        ]
    
    def _is_content_active(self, content: ConsciousContent) -> bool:
        """Check if conscious content is still active"""
        elapsed = (datetime.now() - content.broadcast_timestamp).total_seconds()
        return elapsed < content.influence_duration
    
    def _broadcast_conscious_content(self, content: ConsciousContent):
        """Broadcast conscious content to all subscribers"""
        for callback in self.broadcast_subscribers:
            try:
                callback(content)
            except Exception as e:
                print(f"Error in broadcast callback: {e}")
