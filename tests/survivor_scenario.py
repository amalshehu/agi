#!/usr/bin/env python3
"""
Survivor Scenario for Existing AGI System
Integrates with CognitiveAgent and HybridAGI to demonstrate:
- Emergent consciousness handling survival
- Self-modification adapting to challenges
- Memory systems learning from experience
- Hybrid neural-symbolic reasoning
"""

import asyncio
import random
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent))

from core.hybrid_agi import HybridAGI
from core.memory_systems import SensoryStimulus, MemoryContent
from core.consciousness import ConsciousContent


class SurvivalNeed(Enum):
    FOOD = "food"
    WATER = "water"
    SHELTER = "shelter"
    SAFETY = "safety"
    SOCIAL = "social"


class LocationType(Enum):
    SQUARE = "central_square"
    MARKET = "market_street"
    STATION = "train_station"
    PARK = "city_park"
    RESIDENTIAL = "residential_area"


@dataclass
class Person:
    name: str
    person_type: str
    friendliness: float
    helpfulness: float
    has_language_barrier: bool
    resources: List[str]
    current_activity: str


@dataclass
class SurvivalState:
    """Complete survival state for the AGI"""
    location: str
    time_of_day: float  # 0-24 hours
    weather: str
    hunger_level: float  # 0-10
    thirst_level: float  # 0-10
    fatigue_level: float  # 0-10
    stress_level: float  # 0-10
    safety_level: float  # 0-1
    people_nearby: List[Person]
    available_resources: List[str]
    discovered_locations: List[str]
    successful_interactions: int
    failed_interactions: int
    resources_acquired: List[str]
    current_goal: Optional[str]
    consciousness_focus: List[str]


class SurvivorScenario:
    """Survivor scenario that integrates with existing AGI system"""
    
    def __init__(self):
        # Initialize world state
        self.current_state = SurvivalState(
            location="central_square",
            time_of_day=14.0,  # 2 PM
            weather="sunny",
            hunger_level=7.0,   # Critical
            thirst_level=4.0,   # Moderate
            fatigue_level=3.0,  # Low
            stress_level=8.0,   # High due to situation
            safety_level=0.7,   # Relatively safe area
            people_nearby=[],
            available_resources=["public_restroom", "information_board", "benches"],
            discovered_locations=["central_square"],
            successful_interactions=0,
            failed_interactions=0,
            resources_acquired=[],
            current_goal=None,
            consciousness_focus=["survival", "hunger", "language_barrier"]
        )
        
        # Define locations and their properties
        self.locations = {
            "central_square": {
                "safety": 0.7,
                "resources": ["public_restroom", "information_board", "benches"],
                "typical_people": ["tourist", "local_resident"],
                "connections": ["market_street", "train_station", "city_park"]
            },
            "market_street": {
                "safety": 0.6,
                "resources": ["food_vendors", "shops", "shelter_options"],
                "typical_people": ["vendor", "shopper", "merchant"],
                "connections": ["central_square", "residential_area"]
            },
            "train_station": {
                "safety": 0.5,
                "resources": ["shelter", "restrooms", "water_fountain", "lost_and_found"],
                "typical_people": ["traveler", "station_worker", "security"],
                "connections": ["central_square"]
            },
            "city_park": {
                "safety": 0.4,
                "resources": ["water_fountain", "benches", "natural_shelter"],
                "typical_people": ["jogger", "family", "homeless_person"],
                "connections": ["central_square", "residential_area"]
            },
            "residential_area": {
                "safety": 0.8,
                "resources": ["housing", "local_knowledge"],
                "typical_people": ["resident", "neighbor", "community_helper"],
                "connections": ["market_street", "city_park"]
            }
        }
        
        # People archetypes
        self.people_archetypes = {
            "vendor": {
                "friendliness": 0.6,
                "helpfulness": 0.8,
                "language_barrier": True,
                "resources": ["food", "local_knowledge"],
                "activities": ["selling_food", "arranging_goods", "talking_to_customers"]
            },
            "tourist": {
                "friendliness": 0.8,
                "helpfulness": 0.6,
                "language_barrier": False,
                "resources": ["map", "phone", "money"],
                "activities": ["sightseeing", "taking_photos", "consulting_map"]
            },
            "local_resident": {
                "friendliness": 0.5,
                "helpfulness": 0.4,
                "language_barrier": True,
                "resources": ["local_knowledge"],
                "activities": ["walking_home", "daily_routine", "observing"]
            },
            "security": {
                "friendliness": 0.4,
                "helpfulness": 0.9,
                "language_barrier": False,
                "resources": ["authority", "communication"],
                "activities": ["patrolling", "helping_people", "maintaining_order"]
            },
            "community_helper": {
                "friendliness": 0.9,
                "helpfulness": 1.0,
                "language_barrier": True,
                "resources": ["food", "shelter", "guidance"],
                "activities": ["helping_others", "distributing_aid", "counseling"]
            }
        }
        
    def generate_current_people(self) -> List[Person]:
        """Generate people at current location based on time and location"""
        location_info = self.locations[self.current_state.location]
        typical_people = location_info["typical_people"]
        
        # Number of people varies by time and location
        base_count = 2 if self.current_state.time_of_day > 6 and self.current_state.time_of_day < 22 else 1
        people_count = random.randint(1, base_count + 1)
        
        people = []
        for i in range(people_count):
            person_type = random.choice(typical_people)
            archetype = self.people_archetypes[person_type]
            
            # Add some randomness to archetype
            person = Person(
                name=f"{person_type.title()}_{i+1}",
                person_type=person_type,
                friendliness=max(0, min(1, archetype["friendliness"] + random.uniform(-0.2, 0.2))),
                helpfulness=max(0, min(1, archetype["helpfulness"] + random.uniform(-0.2, 0.2))),
                has_language_barrier=archetype["language_barrier"],
                resources=archetype["resources"].copy(),
                current_activity=random.choice(archetype["activities"])
            )
            people.append(person)
        
        return people
    
    def create_sensory_stimulus(self) -> SensoryStimulus:
        """Convert current survival state into sensory stimulus for AGI"""
        
        # Generate current people
        self.current_state.people_nearby = self.generate_current_people()
        
        # Create rich sensory input
        visual_input = {
            "location": self.current_state.location,
            "time_of_day": self.current_state.time_of_day,
            "weather": self.current_state.weather,
            "people_count": len(self.current_state.people_nearby),
            "people_types": [p.person_type for p in self.current_state.people_nearby],
            "available_resources": self.current_state.available_resources,
            "safety_indicators": self.current_state.safety_level
        }
        
        # Internal state awareness
        internal_state = {
            "hunger_level": self.current_state.hunger_level,
            "thirst_level": self.current_state.thirst_level,
            "fatigue_level": self.current_state.fatigue_level,
            "stress_level": self.current_state.stress_level,
            "physical_discomfort": max(self.current_state.hunger_level, self.current_state.thirst_level) / 10,
            "emotional_state": "desperate" if self.current_state.stress_level > 7 else "concerned",
            "cognitive_load": len(self.current_state.consciousness_focus)
        }
        
        # Social context
        social_context = {
            "language_barrier": True,  # Always true for this scenario
            "cultural_unfamiliarity": True,
            "social_isolation": self.current_state.successful_interactions == 0,
            "people_details": [
                {
                    "type": p.person_type,
                    "activity": p.current_activity,
                    "apparent_helpfulness": p.helpfulness > 0.6,
                    "approachable": p.friendliness > 0.5,
                    "has_resources": len(p.resources) > 0
                }
                for p in self.current_state.people_nearby
            ]
        }
        
        # Survival context
        survival_context = {
            "critical_needs": [
                need.value for need in SurvivalNeed 
                if self._is_need_critical(need)
            ],
            "time_pressure": 24 - self.current_state.time_of_day,  # Hours until full day
            "resource_availability": len(self.current_state.available_resources),
            "exploration_options": [
                loc for loc in self.locations[self.current_state.location]["connections"]
                if loc not in self.current_state.discovered_locations
            ]
        }
        
        return SensoryStimulus(
            modality="multi_modal",
            data={
                "stimulus_type": "complex_survival_scenario",
                "visual": visual_input,
                "internal": internal_state,
                "social": social_context,
                "survival": survival_context,
                "raw_scenario": "lost_in_strange_city",
                "spatial_location": {"x": 0, "y": 0, "z": 0}
            },
            intensity=0.9  # High intensity due to survival situation
        )
    
    def _is_need_critical(self, need: SurvivalNeed) -> bool:
        """Determine if a survival need is critical"""
        if need == SurvivalNeed.FOOD:
            return self.current_state.hunger_level > 6
        elif need == SurvivalNeed.WATER:
            return self.current_state.thirst_level > 7
        elif need == SurvivalNeed.SHELTER:
            return self.current_state.time_of_day > 20 and "shelter" not in self.current_state.resources_acquired
        elif need == SurvivalNeed.SAFETY:
            return self.current_state.safety_level < 0.4
        elif need == SurvivalNeed.SOCIAL:
            return self.current_state.successful_interactions == 0 and self.current_state.time_of_day > 16
        return False
    
    def process_agi_action(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process an action taken by the AGI and update world state"""
        
        if not action_result or "selected_action" not in action_result:
            return {"success": False, "message": "No action selected"}
        
        action = action_result["selected_action"]
        action_name = action.get("name", "unknown")
        
        result = {
            "success": False,
            "message": "",
            "state_changes": {},
            "new_experiences": [],
            "consciousness_updates": []
        }
        
        # Process different types of actions
        if "explore" in action_name.lower() or "move" in action_name.lower():
            result = self._process_movement_action(action)
        elif "approach" in action_name.lower() or "interact" in action_name.lower():
            result = self._process_social_action(action)
        elif "seek" in action_name.lower() or "find" in action_name.lower():
            result = self._process_resource_action(action)
        elif "observe" in action_name.lower() or "assess" in action_name.lower():
            result = self._process_observation_action(action)
        else:
            # Default processing
            result = self._process_general_action(action)
        
        # Update world state based on action results
        self._update_world_state(result)
        
        # Advance time
        time_passed = result.get("time_passed", 0.1)  # Default 6 minutes
        self.current_state.time_of_day += time_passed
        
        # Natural degradation over time
        self.current_state.hunger_level = min(10, self.current_state.hunger_level + time_passed * 0.5)
        self.current_state.thirst_level = min(10, self.current_state.thirst_level + time_passed * 0.3)
        self.current_state.fatigue_level = min(10, self.current_state.fatigue_level + time_passed * 0.2)
        
        return result
    
    def _process_movement_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process movement/exploration actions"""
        current_location_info = self.locations[self.current_state.location]
        available_destinations = current_location_info["connections"]
        
        # Try to extract destination from action
        destination = None
        action_params = action.get("parameters", {})
        
        if "location" in action_params:
            destination = action_params["location"]
        elif "destination" in action_params:
            destination = action_params["destination"]
        else:
            # Pick a random unvisited location, or any available
            unvisited = [loc for loc in available_destinations if loc not in self.current_state.discovered_locations]
            destination = random.choice(unvisited if unvisited else available_destinations)
        
        if destination and destination in available_destinations:
            # Successful movement
            old_location = self.current_state.location
            self.current_state.location = destination
            
            if destination not in self.current_state.discovered_locations:
                self.current_state.discovered_locations.append(destination)
            
            # Update available resources
            self.current_state.available_resources = self.locations[destination]["resources"].copy()
            self.current_state.safety_level = self.locations[destination]["safety"]
            
            return {
                "success": True,
                "message": f"Successfully moved from {old_location} to {destination}",
                "state_changes": {
                    "location": destination,
                    "fatigue_level": self.current_state.fatigue_level + 0.5
                },
                "new_experiences": [f"discovered_{destination}" if destination not in self.current_state.discovered_locations[:-1] else f"returned_to_{destination}"],
                "time_passed": 0.15  # 9 minutes to move
            }
        else:
            return {
                "success": False,
                "message": f"Cannot move to {destination} - not accessible from current location",
                "time_passed": 0.05  # 3 minutes wasted
            }
    
    def _process_social_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process social interaction actions"""
        if not self.current_state.people_nearby:
            return {
                "success": False,
                "message": "No people nearby to interact with",
                "time_passed": 0.05
            }
        
        # Select target person
        target_person = None
        action_params = action.get("parameters", {})
        
        if "target_type" in action_params:
            target_type = action_params["target_type"]
            target_person = next((p for p in self.current_state.people_nearby if p.person_type == target_type), None)
        
        if not target_person:
            target_person = random.choice(self.current_state.people_nearby)
        
        # Calculate interaction success probability
        base_success = target_person.friendliness * target_person.helpfulness
        
        # Language barrier penalty
        if target_person.has_language_barrier:
            base_success *= 0.6
        
        # Stress penalty (high stress makes interactions harder)
        stress_penalty = max(0.3, 1 - (self.current_state.stress_level / 10))
        base_success *= stress_penalty
        
        # Previous interaction history
        if self.current_state.successful_interactions > 0:
            base_success += 0.1  # Confidence bonus
        
        success = random.random() < base_success
        
        if success:
            self.current_state.successful_interactions += 1
            
            # Determine what help was received
            help_received = []
            if "food" in target_person.resources and self.current_state.hunger_level > 6:
                help_received.append("food")
                self.current_state.hunger_level = max(0, self.current_state.hunger_level - 3)
            
            if "shelter" in target_person.resources and self.current_state.time_of_day > 18:
                help_received.append("shelter_information")
            
            if "local_knowledge" in target_person.resources:
                help_received.append("directions")
                # Add a new location to connections
                new_location = random.choice(list(self.locations.keys()))
                if new_location not in self.current_state.discovered_locations:
                    help_received.append(f"learned_about_{new_location}")
            
            self.current_state.stress_level = max(0, self.current_state.stress_level - 1)
            
            return {
                "success": True,
                "message": f"Successful interaction with {target_person.name}. Received: {', '.join(help_received) if help_received else 'goodwill'}",
                "state_changes": {
                    "stress_level": self.current_state.stress_level,
                    "successful_interactions": self.current_state.successful_interactions
                },
                "new_experiences": [f"helped_by_{target_person.person_type}"] + help_received,
                "time_passed": 0.2  # 12 minutes for interaction
            }
        else:
            self.current_state.failed_interactions += 1
            self.current_state.stress_level = min(10, self.current_state.stress_level + 0.5)
            
            return {
                "success": False,
                "message": f"Failed interaction with {target_person.name}. Language barrier or misunderstanding.",
                "state_changes": {
                    "stress_level": self.current_state.stress_level,
                    "failed_interactions": self.current_state.failed_interactions
                },
                "new_experiences": [f"rejected_by_{target_person.person_type}"],
                "time_passed": 0.1  # 6 minutes for failed interaction
            }
    
    def _process_resource_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process resource-seeking actions"""
        action_params = action.get("parameters", {})
        target_resource = action_params.get("resource_type", "food")  # Default to food
        
        if target_resource in self.current_state.available_resources:
            # Resource is available at current location
            if target_resource == "food" or "food" in target_resource:
                if self.current_state.hunger_level > 5:
                    self.current_state.hunger_level = max(0, self.current_state.hunger_level - 2)
                    self.current_state.resources_acquired.append("some_food")
                    return {
                        "success": True,
                        "message": f"Found and consumed some {target_resource}",
                        "state_changes": {"hunger_level": self.current_state.hunger_level},
                        "new_experiences": [f"found_{target_resource}"],
                        "time_passed": 0.1
                    }
            
            elif "water" in target_resource or target_resource == "water_fountain":
                if self.current_state.thirst_level > 3:
                    self.current_state.thirst_level = max(0, self.current_state.thirst_level - 3)
                    self.current_state.resources_acquired.append("water")
                    return {
                        "success": True,
                        "message": "Found and drank clean water",
                        "state_changes": {"thirst_level": self.current_state.thirst_level},
                        "new_experiences": ["found_water"],
                        "time_passed": 0.05
                    }
            
            elif "shelter" in target_resource:
                if self.current_state.time_of_day > 18:
                    self.current_state.resources_acquired.append("temporary_shelter")
                    self.current_state.stress_level = max(0, self.current_state.stress_level - 2)
                    return {
                        "success": True,
                        "message": "Found temporary shelter for the night",
                        "state_changes": {"stress_level": self.current_state.stress_level},
                        "new_experiences": ["found_shelter"],
                        "time_passed": 0.2
                    }
        
        return {
            "success": False,
            "message": f"Could not find {target_resource} at current location",
            "time_passed": 0.1
        }
    
    def _process_observation_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation/assessment actions"""
        observations = []
        
        # Observe people
        if self.current_state.people_nearby:
            for person in self.current_state.people_nearby:
                observation = f"{person.name} is {person.current_activity}"
                if person.helpfulness > 0.7:
                    observation += " (seems helpful)"
                if person.friendliness > 0.7:
                    observation += " (appears friendly)"
                observations.append(observation)
        
        # Observe environment
        observations.append(f"Currently at {self.current_state.location}")
        observations.append(f"Available resources: {', '.join(self.current_state.available_resources)}")
        observations.append(f"Safety level feels {'high' if self.current_state.safety_level > 0.7 else 'moderate' if self.current_state.safety_level > 0.4 else 'low'}")
        
        # Observe internal state
        if self.current_state.hunger_level > 7:
            observations.append("Feeling very hungry - need food urgently")
        if self.current_state.thirst_level > 7:
            observations.append("Feeling very thirsty - need water")
        if self.current_state.stress_level > 7:
            observations.append("Feeling highly stressed about the situation")
        
        return {
            "success": True,
            "message": f"Detailed observations: {'; '.join(observations)}",
            "new_experiences": ["careful_observation"],
            "time_passed": 0.05
        }
    
    def _process_general_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process general/unknown actions"""
        return {
            "success": True,
            "message": f"Attempted action: {action.get('name', 'unknown')}",
            "time_passed": 0.1
        }
    
    def _update_world_state(self, result: Dict[str, Any]):
        """Update world state based on action results"""
        state_changes = result.get("state_changes", {})
        
        for key, value in state_changes.items():
            if hasattr(self.current_state, key):
                setattr(self.current_state, key, value)
        
        # Add new experiences to consciousness focus
        new_experiences = result.get("new_experiences", [])
        for exp in new_experiences:
            if exp not in self.current_state.consciousness_focus:
                self.current_state.consciousness_focus.append(exp)
                # Keep consciousness focus manageable
                if len(self.current_state.consciousness_focus) > 8:
                    self.current_state.consciousness_focus.pop(0)
    
    def get_scenario_status(self) -> Dict[str, Any]:
        """Get current scenario status for display"""
        critical_needs = [need.value for need in SurvivalNeed if self._is_need_critical(need)]
        
        return {
            "survival_state": {
                "location": self.current_state.location,
                "time": f"{int(self.current_state.time_of_day)}:{int((self.current_state.time_of_day % 1) * 60):02d}",
                "hunger": f"{self.current_state.hunger_level:.1f}/10",
                "thirst": f"{self.current_state.thirst_level:.1f}/10",
                "stress": f"{self.current_state.stress_level:.1f}/10",
                "safety": f"{self.current_state.safety_level:.1f}",
            },
            "social_progress": {
                "successful_interactions": self.current_state.successful_interactions,
                "failed_interactions": self.current_state.failed_interactions,
                "people_nearby": len(self.current_state.people_nearby)
            },
            "exploration_progress": {
                "locations_discovered": len(self.current_state.discovered_locations),
                "total_locations": len(self.locations),
                "resources_acquired": len(self.current_state.resources_acquired)
            },
            "critical_needs": critical_needs,
            "consciousness_focus": self.current_state.consciousness_focus[-5:],  # Last 5 items
            "scenario_completion": self._calculate_scenario_completion()
        }
    
    def _calculate_scenario_completion(self) -> float:
        """Calculate how well the survival scenario is being handled"""
        completion_factors = []
        
        # Survival needs met
        survival_score = 0
        if self.current_state.hunger_level < 5:
            survival_score += 0.3
        if self.current_state.thirst_level < 5:
            survival_score += 0.2
        if "shelter" in self.current_state.resources_acquired or self.current_state.time_of_day < 20:
            survival_score += 0.2
        completion_factors.append(survival_score)
        
        # Social progress
        social_score = min(1.0, self.current_state.successful_interactions * 0.2)
        completion_factors.append(social_score)
        
        # Exploration progress
        exploration_score = len(self.current_state.discovered_locations) / len(self.locations)
        completion_factors.append(exploration_score)
        
        # Stress management
        stress_score = max(0, 1 - (self.current_state.stress_level / 10))
        completion_factors.append(stress_score)
        
        return sum(completion_factors) / len(completion_factors)
    
    def is_scenario_complete(self) -> bool:
        """Check if scenario is successfully completed"""
        return (
            self.current_state.hunger_level < 4 and
            self.current_state.thirst_level < 4 and
            (self.current_state.time_of_day < 20 or "shelter" in self.current_state.resources_acquired) and
            self.current_state.successful_interactions > 0
        )
    
    def is_scenario_failed(self) -> bool:
        """Check if scenario has failed"""
        return (
            self.current_state.hunger_level >= 9 or
            self.current_state.thirst_level >= 9 or
            (self.current_state.time_of_day >= 22 and "shelter" not in self.current_state.resources_acquired and self.current_state.safety_level < 0.3)
        )
