"""
Post-Disaster Urban Survival Simulation Environment
Implements the research scenario for global workspace AGI testing
"""

import numpy as np
import random
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from pathlib import Path

class ResourceType(Enum):
    FOOD = "food"
    WATER = "water" 
    SHELTER = "shelter"
    TOOL = "tool"
    MEDICINE = "medicine"

class NPCBehavior(Enum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    TRADER = "trader"

class HazardType(Enum):
    DEBRIS = "debris"
    FIRE = "fire"
    RADIATION = "radiation"
    UNSTABLE_BUILDING = "unstable_building"
    HOSTILE_NPC = "hostile_npc"

@dataclass
class SimulationConfig:
    """Configuration for survival simulation scenarios"""
    map_size: Tuple[int, int] = (100, 100)
    max_resources: int = 20
    max_npcs: int = 10
    max_hazards: int = 15
    lighting_variation: bool = True
    weather_effects: bool = True
    time_limit: int = 1000  # simulation steps
    
    # Procedural generation parameters
    building_density: float = 0.3
    damage_level: float = 0.7  # 0.0 = pristine, 1.0 = completely destroyed
    resource_scarcity: float = 0.8  # higher = more scarce
    
@dataclass 
class Position:
    x: int
    y: int
    
    def distance_to(self, other: 'Position') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Resource:
    type: ResourceType
    position: Position
    value: float
    discovered: bool = False
    collected: bool = False

@dataclass
class NPC:
    id: str
    position: Position
    behavior: NPCBehavior
    health: float = 100.0
    inventory: List[ResourceType] = field(default_factory=list)
    hostility_level: float = 0.0  # 0.0 = peaceful, 1.0 = very aggressive
    
@dataclass
class Hazard:
    type: HazardType
    position: Position
    severity: float  # 0.0 = minor, 1.0 = deadly
    active: bool = True
    damage_per_step: float = 0.0

@dataclass
class Agent:
    position: Position
    health: float = 100.0
    hunger: float = 0.0  # 0.0 = full, 100.0 = starving
    thirst: float = 0.0  # 0.0 = hydrated, 100.0 = dehydrated  
    shelter_level: float = 0.0  # 0.0 = exposed, 100.0 = well sheltered
    inventory: List[ResourceType] = field(default_factory=list)
    
    def needs_resource(self, resource_type: ResourceType) -> float:
        """Return urgency (0-1) for needing this resource"""
        if resource_type == ResourceType.FOOD:
            return min(1.0, self.hunger / 100.0)
        elif resource_type == ResourceType.WATER:
            return min(1.0, self.thirst / 100.0) 
        elif resource_type == ResourceType.SHELTER:
            return max(0.0, 1.0 - self.shelter_level / 100.0)
        return 0.0

class SurvivalEnvironment:
    """Post-disaster urban survival simulation environment"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.map_size = config.map_size
        self.step_count = 0
        self.consciousness_events: List[Dict[str, Any]] = []
        
        # Initialize world state
        self.agent = Agent(Position(random.randint(10, config.map_size[0]-10),
                                  random.randint(10, config.map_size[1]-10)))
        self.resources: List[Resource] = []
        self.npcs: List[NPC] = []
        self.hazards: List[Hazard] = []
        
        # Environment factors
        self.lighting_level = 1.0  # 0.0 = dark, 1.0 = bright
        self.weather_severity = 0.0  # 0.0 = calm, 1.0 = severe storm
        
        # Metrics tracking
        self.metrics = {
            "time_to_first_resource": None,
            "resources_found": 0,
            "hazards_encountered": 0,
            "npc_interactions": 0,
            "survival_time": 0,
            "consciousness_spikes": 0,
            "critical_decisions": 0
        }
        
        self._generate_world()
    
    def _generate_world(self):
        """Procedurally generate the post-disaster urban environment"""
        
        # Generate buildings (mostly collapsed/damaged)
        self.buildings = []
        for _ in range(int(self.config.building_density * 100)):
            building = {
                "position": Position(
                    random.randint(0, self.config.map_size[0]),
                    random.randint(0, self.config.map_size[1])
                ),
                "intact": random.random() > self.config.damage_level,
                "size": random.randint(3, 8)
            }
            self.buildings.append(building)
        
        # Generate resources (scarce)
        resource_count = max(1, int(self.config.max_resources * (1.0 - self.config.resource_scarcity)))
        for _ in range(resource_count):
            resource = Resource(
                type=random.choice(list(ResourceType)),
                position=Position(
                    random.randint(0, self.config.map_size[0]),
                    random.randint(0, self.config.map_size[1])
                ),
                value=random.uniform(10, 50)
            )
            self.resources.append(resource)
        
        # Generate NPCs
        for i in range(random.randint(3, self.config.max_npcs)):
            npc = NPC(
                id=f"npc_{i}",
                position=Position(
                    random.randint(0, self.config.map_size[0]),
                    random.randint(0, self.config.map_size[1])
                ),
                behavior=random.choice(list(NPCBehavior)),
                hostility_level=random.random()
            )
            self.npcs.append(npc)
        
        # Generate hazards
        min_hazards = min(3, self.config.max_hazards)
        max_hazards = max(min_hazards, self.config.max_hazards)
        for _ in range(random.randint(min_hazards, max_hazards)):
            hazard = Hazard(
                type=random.choice(list(HazardType)),
                position=Position(
                    random.randint(0, self.config.map_size[0]),
                    random.randint(0, self.config.map_size[1])
                ),
                severity=random.random(),
                damage_per_step=random.uniform(1, 10)
            )
            self.hazards.append(hazard)
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current environmental observation for the agent"""
        
        vision_range = 10
        agent_pos = self.agent.position
        
        # Visible objects within range
        visible_resources = []
        for resource in self.resources:
            if (not resource.collected and 
                resource.position.distance_to(agent_pos) <= vision_range):
                visible_resources.append({
                    "type": resource.type.value,
                    "position": (resource.position.x, resource.position.y),
                    "distance": resource.position.distance_to(agent_pos),
                    "value": resource.value
                })
        
        visible_npcs = []
        for npc in self.npcs:
            if npc.position.distance_to(agent_pos) <= vision_range:
                visible_npcs.append({
                    "id": npc.id,
                    "position": (npc.position.x, npc.position.y),
                    "behavior": npc.behavior.value,
                    "distance": npc.position.distance_to(agent_pos),
                    "hostility": npc.hostility_level
                })
        
        visible_hazards = []
        for hazard in self.hazards:
            if (hazard.active and 
                hazard.position.distance_to(agent_pos) <= vision_range):
                visible_hazards.append({
                    "type": hazard.type.value,
                    "position": (hazard.position.x, hazard.position.y),
                    "severity": hazard.severity,
                    "distance": hazard.position.distance_to(agent_pos)
                })
        
        return {
            "agent_position": (agent_pos.x, agent_pos.y),
            "agent_health": self.agent.health,
            "agent_hunger": self.agent.hunger,
            "agent_thirst": self.agent.thirst,
            "agent_shelter": self.agent.shelter_level,
            "inventory": [r.value for r in self.agent.inventory],
            "visible_resources": visible_resources,
            "visible_npcs": visible_npcs,
            "visible_hazards": visible_hazards,
            "lighting_level": self.lighting_level,
            "weather_severity": self.weather_severity,
            "step_count": self.step_count,
            "time_remaining": self.config.time_limit - self.step_count
        }
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one simulation step with agent action"""
        
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}
        
        # Process agent action
        if action["type"] == "move":
            self._process_move_action(action)
        elif action["type"] == "collect":
            reward += self._process_collect_action(action)
        elif action["type"] == "interact":
            reward += self._process_interact_action(action)
        elif action["type"] == "wait":
            pass  # Do nothing, just advance time
        
        # Update environmental factors
        self._update_environment()
        
        # Update agent needs
        self._update_agent_needs()
        
        # Check hazards
        reward += self._check_hazards()
        
        # Update metrics
        self._update_metrics()
        
        # Check termination conditions
        if (self.step_count >= self.config.time_limit or 
            self.agent.health <= 0 or
            (self.agent.hunger >= 100 and self.agent.thirst >= 100)):
            done = True
            reward += self._calculate_final_reward()
        
        # Generate next observation
        obs = self.get_observation()
        
        info = {
            "metrics": self.metrics.copy(),
            "consciousness_events": self.consciousness_events.copy(),
            "survival_status": self._get_survival_status()
        }
        
        return obs, reward, done, info
    
    def _process_move_action(self, action: Dict[str, Any]):
        """Process agent movement"""
        dx, dy = action.get("direction", (0, 0))
        new_x = max(0, min(self.config.map_size[0] - 1, self.agent.position.x + dx))
        new_y = max(0, min(self.config.map_size[1] - 1, self.agent.position.y + dy))
        self.agent.position = Position(new_x, new_y)
    
    def _process_collect_action(self, action: Dict[str, Any]) -> float:
        """Process resource collection"""
        target_pos = Position(action.get("target_x", -1), action.get("target_y", -1))
        
        for resource in self.resources:
            if (not resource.collected and 
                resource.position.distance_to(self.agent.position) <= 2.0 and
                resource.position.distance_to(target_pos) <= 1.0):
                
                resource.collected = True
                self.agent.inventory.append(resource.type)
                self.metrics["resources_found"] += 1
                
                if self.metrics["time_to_first_resource"] is None:
                    self.metrics["time_to_first_resource"] = self.step_count
                
                # Apply resource effects
                if resource.type == ResourceType.FOOD:
                    self.agent.hunger = max(0, self.agent.hunger - resource.value)
                elif resource.type == ResourceType.WATER:
                    self.agent.thirst = max(0, self.agent.thirst - resource.value)
                elif resource.type == ResourceType.SHELTER:
                    self.agent.shelter_level = min(100, self.agent.shelter_level + resource.value)
                
                return resource.value  # Positive reward for collection
        
        return -1  # Small penalty for failed collection attempt
    
    def _process_interact_action(self, action: Dict[str, Any]) -> float:
        """Process NPC interaction"""
        target_id = action.get("target_id", "")
        
        for npc in self.npcs:
            if (npc.id == target_id and 
                npc.position.distance_to(self.agent.position) <= 3.0):
                
                self.metrics["npc_interactions"] += 1
                
                if npc.behavior == NPCBehavior.FRIENDLY:
                    # Friendly NPCs might share resources or information
                    if len(npc.inventory) > 0 and random.random() < 0.3:
                        shared_resource = random.choice(npc.inventory)
                        self.agent.inventory.append(shared_resource)
                        npc.inventory.remove(shared_resource)
                        return 20  # Reward for successful friendly interaction
                    
                elif npc.behavior == NPCBehavior.TRADER:
                    # Trading interaction
                    if len(self.agent.inventory) > 0 and random.random() < 0.5:
                        # Simple trade: give one, get one
                        traded_away = random.choice(self.agent.inventory)
                        self.agent.inventory.remove(traded_away)
                        new_resource = random.choice(list(ResourceType))
                        self.agent.inventory.append(new_resource)
                        return 15  # Reward for successful trade
                
                elif npc.behavior == NPCBehavior.HOSTILE:
                    # Hostile encounter
                    damage = npc.hostility_level * 20
                    self.agent.health = max(0, self.agent.health - damage)
                    return -damage  # Negative reward for taking damage
        
        return -2  # Small penalty for failed interaction
    
    def _update_environment(self):
        """Update environmental conditions"""
        if self.config.lighting_variation:
            # Simulate day/night cycle and weather
            time_factor = (self.step_count / self.config.time_limit) * 2 * np.pi
            self.lighting_level = 0.5 + 0.5 * np.sin(time_factor)
        
        if self.config.weather_effects:
            # Random weather changes
            if random.random() < 0.05:  # 5% chance per step
                self.weather_severity = random.random()
    
    def _update_agent_needs(self):
        """Update agent's survival needs"""
        # Hunger and thirst increase over time
        self.agent.hunger += 0.5 + (self.weather_severity * 0.3)
        self.agent.thirst += 0.7 + (self.weather_severity * 0.5)
        
        # Shelter degrades in bad weather
        if self.weather_severity > 0.5:
            self.agent.shelter_level = max(0, self.agent.shelter_level - 
                                         (self.weather_severity - 0.5) * 2)
        
        # Health decreases if needs aren't met
        if self.agent.hunger > 80:
            self.agent.health -= 1
        if self.agent.thirst > 90:
            self.agent.health -= 2
        if self.agent.shelter_level < 20 and self.weather_severity > 0.7:
            self.agent.health -= 1
    
    def _check_hazards(self) -> float:
        """Check for hazard encounters"""
        penalty = 0.0
        
        for hazard in self.hazards:
            if (hazard.active and 
                hazard.position.distance_to(self.agent.position) <= 2.0):
                
                self.metrics["hazards_encountered"] += 1
                damage = hazard.damage_per_step * hazard.severity
                self.agent.health = max(0, self.agent.health - damage)
                penalty -= damage
        
        return penalty
    
    def _update_metrics(self):
        """Update simulation metrics"""
        self.metrics["survival_time"] = self.step_count
    
    def _calculate_final_reward(self) -> float:
        """Calculate final survival reward"""
        survival_bonus = self.step_count  # Longer survival = better
        health_bonus = self.agent.health  # Higher health = better
        resource_bonus = len(self.agent.inventory) * 10  # More resources = better
        
        return survival_bonus + health_bonus + resource_bonus
    
    def _get_survival_status(self) -> Dict[str, Any]:
        """Get current survival status"""
        return {
            "alive": self.agent.health > 0,
            "well_fed": self.agent.hunger < 50,
            "hydrated": self.agent.thirst < 50,
            "sheltered": self.agent.shelter_level > 30,
            "resource_secure": len(self.agent.inventory) >= 3
        }
    
    def log_consciousness_event(self, event_type: str, strength: float, 
                              decision: str, context: Dict[str, Any]):
        """Log a consciousness event for research analysis"""
        event = {
            "step": self.step_count,
            "type": event_type,
            "strength": strength,
            "decision": decision,
            "context": context,
            "agent_state": {
                "position": (self.agent.position.x, self.agent.position.y),
                "health": self.agent.health,
                "hunger": self.agent.hunger,
                "thirst": self.agent.thirst
            }
        }
        
        self.consciousness_events.append(event)
        
        if strength > 1.5:  # High consciousness threshold
            self.metrics["consciousness_spikes"] += 1
        
        if event_type == "critical_decision":
            self.metrics["critical_decisions"] += 1
    
    def reset(self, config: Optional[SimulationConfig] = None) -> Dict[str, Any]:
        """Reset simulation environment"""
        if config:
            self.config = config
        
        self.step_count = 0
        self.consciousness_events = []
        
        # Reset agent
        self.agent = Agent(Position(random.randint(10, self.config.map_size[0]-10),
                                  random.randint(10, self.config.map_size[1]-10)))
        
        # Reset metrics
        self.metrics = {
            "time_to_first_resource": None,
            "resources_found": 0,
            "hazards_encountered": 0,
            "npc_interactions": 0,
            "survival_time": 0,
            "consciousness_spikes": 0,
            "critical_decisions": 0
        }
        
        # Regenerate world
        self._generate_world()
        
        return self.get_observation()
    
    def get_action_space(self) -> Dict[str, Any]:
        """Get available actions for the agent"""
        return {
            "move": {
                "direction": [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                            (0, 1), (1, -1), (1, 0), (1, 1)]  # 8 directions
            },
            "collect": {
                "target_x": list(range(self.config.map_size[0])),
                "target_y": list(range(self.config.map_size[1]))
            },
            "interact": {
                "target_id": [npc.id for npc in self.npcs]
            },
            "wait": {}
        }
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the simulation environment"""
        if mode == "rgb_array":
            # Create visual representation
            img = np.zeros((self.config.map_size[1], self.config.map_size[0], 3), dtype=np.uint8)
            
            # Draw background (darker for night, lighter for day)
            bg_intensity = int(self.lighting_level * 100 + 50)
            img[:, :] = [bg_intensity, bg_intensity, bg_intensity]
            
            # Draw buildings
            for building in self.buildings:
                color = [150, 150, 150] if building["intact"] else [100, 50, 50]
                size = building["size"]
                x, y = building["position"].x, building["position"].y
                
                for dx in range(-size//2, size//2 + 1):
                    for dy in range(-size//2, size//2 + 1):
                        px, py = x + dx, y + dy
                        if 0 <= px < self.config.map_size[0] and 0 <= py < self.config.map_size[1]:
                            img[py, px] = color
            
            # Draw resources
            for resource in self.resources:
                if not resource.collected:
                    color = [0, 255, 0]  # Green for resources
                    img[resource.position.y, resource.position.x] = color
            
            # Draw NPCs
            for npc in self.npcs:
                if npc.behavior == NPCBehavior.FRIENDLY:
                    color = [0, 0, 255]  # Blue for friendly
                elif npc.behavior == NPCBehavior.HOSTILE:
                    color = [255, 0, 0]  # Red for hostile
                else:
                    color = [255, 255, 0]  # Yellow for neutral/trader
                
                img[npc.position.y, npc.position.x] = color
            
            # Draw hazards
            for hazard in self.hazards:
                if hazard.active:
                    color = [255, 100, 0]  # Orange for hazards
                    img[hazard.position.y, hazard.position.x] = color
            
            # Draw agent
            agent_color = [255, 255, 255]  # White for agent
            img[self.agent.position.y, self.agent.position.x] = agent_color
            
            return img
        
        return None

# Procedural scenario generator
class ScenarioGenerator:
    """Generate different survival scenarios for testing"""
    
    @staticmethod
    def generate_easy_scenario() -> SimulationConfig:
        """Generate an easy survival scenario"""
        return SimulationConfig(
            map_size=(50, 50),
            max_resources=15,
            max_npcs=5,
            max_hazards=3,
            damage_level=0.3,
            resource_scarcity=0.3,
            time_limit=500
        )
    
    @staticmethod
    def generate_medium_scenario() -> SimulationConfig:
        """Generate a medium difficulty survival scenario"""
        return SimulationConfig(
            map_size=(75, 75),
            max_resources=12,
            max_npcs=8,
            max_hazards=8,
            damage_level=0.6,
            resource_scarcity=0.6,
            time_limit=750
        )
    
    @staticmethod
    def generate_hard_scenario() -> SimulationConfig:
        """Generate a hard survival scenario"""
        return SimulationConfig(
            map_size=(100, 100),
            max_resources=8,
            max_npcs=12,
            max_hazards=15,
            damage_level=0.8,
            resource_scarcity=0.8,
            time_limit=1000
        )
    
    @staticmethod
    def generate_custom_scenario(difficulty: str, **kwargs) -> SimulationConfig:
        """Generate custom scenario with specific parameters"""
        base_configs = {
            "easy": ScenarioGenerator.generate_easy_scenario(),
            "medium": ScenarioGenerator.generate_medium_scenario(),
            "hard": ScenarioGenerator.generate_hard_scenario()
        }
        
        config = base_configs.get(difficulty, base_configs["medium"])
        
        # Override with custom parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
