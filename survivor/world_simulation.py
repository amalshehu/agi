#!/usr/bin/env python3
"""
Dynamic World Simulation Engine
Simulates a realistic city environment with complex interactions
"""

import random
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class LocationType(Enum):
    STREET = "street"
    SQUARE = "square"
    MARKET = "market"
    STATION = "station"
    PARK = "park"
    ALLEY = "alley"
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"

class PersonType(Enum):
    VENDOR = "vendor"
    TOURIST = "tourist"
    LOCAL = "local"
    POLICE = "police"
    WORKER = "worker"
    HOMELESS = "homeless"

class WeatherType(Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    COLD = "cold"

@dataclass
class Person:
    name: str
    person_type: PersonType
    friendliness: float  # 0-1
    helpfulness: float   # 0-1
    language_barrier: bool
    resources: List[str]
    current_activity: str

@dataclass
class Location:
    name: str
    location_type: LocationType
    safety_level: float  # 0-1
    crowd_density: float # 0-1
    available_resources: List[str]
    people: List[Person]
    connections: List[str]  # Connected location names
    time_dependent_changes: Dict[str, Any]

class WorldSimulation:
    """Realistic city environment simulation"""
    
    def __init__(self):
        self.current_time = 14  # 2 PM
        self.weather = WeatherType.SUNNY
        self.locations = self._generate_city()
        self.player_location = "central_square"
        self.player_state = {
            "hunger_level": 7,      # 0-10
            "thirst_level": 4,      # 0-10
            "fatigue_level": 3,     # 0-10
            "stress_level": 6,      # 0-10
            "resources": [],
            "reputation": 0,        # -10 to +10
            "discovered_locations": ["central_square"],
            "helpful_contacts": [],
            "language_progress": 0  # 0-100
        }
        self.time_events = []
        self.interaction_history = []
        
    def _generate_city(self) -> Dict[str, Location]:
        """Generate a realistic city with interconnected locations"""
        
        # Define people archetypes
        friendly_vendor = Person("Carlos", PersonType.VENDOR, 0.8, 0.9, True, ["food", "water"], "selling_food")
        tourist_couple = Person("Anna & Mike", PersonType.TOURIST, 0.7, 0.6, False, ["map", "phone"], "sightseeing")
        local_businessman = Person("Hassan", PersonType.LOCAL, 0.5, 0.3, True, ["money"], "walking_to_work")
        helpful_police = Person("Officer Kowalski", PersonType.POLICE, 0.6, 0.8, False, ["radio", "authority"], "patrolling")
        church_volunteer = Person("Sister Maria", PersonType.WORKER, 0.9, 1.0, True, ["food", "shelter"], "helping_others")
        experienced_homeless = Person("Old Pete", PersonType.HOMELESS, 0.7, 0.8, False, ["street_knowledge"], "seeking_shelter")
        
        locations = {
            "central_square": Location(
                name="Central Square",
                location_type=LocationType.SQUARE,
                safety_level=0.8,
                crowd_density=0.7,
                available_resources=["information", "directions", "public_restroom"],
                people=[tourist_couple, local_businessman],
                connections=["market_street", "train_station", "old_town"],
                time_dependent_changes={
                    "morning": {"crowd_density": 0.5, "safety_level": 0.9},
                    "evening": {"crowd_density": 0.9, "safety_level": 0.6},
                    "night": {"crowd_density": 0.2, "safety_level": 0.4}
                }
            ),
            
            "market_street": Location(
                name="Market Street",
                location_type=LocationType.MARKET,
                safety_level=0.7,
                crowd_density=0.8,
                available_resources=["food", "water", "information"],
                people=[friendly_vendor],
                connections=["central_square", "residential_area"],
                time_dependent_changes={
                    "morning": {"crowd_density": 0.9, "people": [friendly_vendor, Person("Early Customer", PersonType.LOCAL, 0.4, 0.2, True, [], "buying_food")]},
                    "evening": {"crowd_density": 0.6},
                    "night": {"crowd_density": 0.1, "safety_level": 0.3, "people": []}
                }
            ),
            
            "train_station": Location(
                name="Train Station",
                location_type=LocationType.STATION,
                safety_level=0.6,
                crowd_density=0.6,
                available_resources=["shelter", "information", "restroom", "water"],
                people=[helpful_police, experienced_homeless],
                connections=["central_square", "industrial_area"],
                time_dependent_changes={
                    "night": {"crowd_density": 0.3, "people": [experienced_homeless]}
                }
            ),
            
            "old_town": Location(
                name="Old Town District",
                location_type=LocationType.RESIDENTIAL,
                safety_level=0.9,
                crowd_density=0.4,
                available_resources=["church", "historical_info"],
                people=[church_volunteer],
                connections=["central_square", "park"],
                time_dependent_changes={
                    "morning": {"people": [church_volunteer, Person("Priest", PersonType.WORKER, 0.8, 0.9, True, ["food", "shelter", "guidance"], "morning_prayers")]},
                    "evening": {"people": [church_volunteer]}
                }
            ),
            
            "park": Location(
                name="City Park",
                location_type=LocationType.PARK,
                safety_level=0.5,
                crowd_density=0.3,
                available_resources=["water_fountain", "bench", "shade"],
                people=[],
                connections=["old_town", "residential_area"],
                time_dependent_changes={
                    "morning": {"people": [Person("Joggers", PersonType.LOCAL, 0.6, 0.4, True, [], "exercising")]},
                    "night": {"safety_level": 0.2, "people": []}
                }
            ),
            
            "residential_area": Location(
                name="Residential Area",
                location_type=LocationType.RESIDENTIAL,
                safety_level=0.7,
                crowd_density=0.3,
                available_resources=["quiet", "housing_info"],
                people=[Person("Local Resident", PersonType.LOCAL, 0.5, 0.5, True, [], "daily_activities")],
                connections=["market_street", "park"],
                time_dependent_changes={
                    "evening": {"crowd_density": 0.5, "people": [Person("Families", PersonType.LOCAL, 0.6, 0.7, True, ["local_knowledge"], "returning_home")]},
                    "night": {"crowd_density": 0.1, "safety_level": 0.5}
                }
            ),
            
            "industrial_area": Location(
                name="Industrial Area",
                location_type=LocationType.COMMERCIAL,
                safety_level=0.4,
                crowd_density=0.2,
                available_resources=["work_opportunities"],
                people=[Person("Security Guard", PersonType.WORKER, 0.3, 0.4, True, ["authority"], "guarding")],
                connections=["train_station"],
                time_dependent_changes={
                    "night": {"safety_level": 0.2, "people": []}
                }
            )
        }
        
        return locations
    
    def get_current_situation(self) -> Dict[str, Any]:
        """Get complete current situation for reasoning engine"""
        current_location = self.locations[self.player_location]
        current_people = self._get_current_people(current_location)
        
        # Calculate dynamic values based on time
        time_of_day = self._get_time_of_day()
        hours_until_night = max(0, 20 - self.current_time) if self.current_time < 20 else 0
        
        situation = {
            "location": self.player_location,
            "location_type": current_location.location_type.value,
            "safety_level": self._get_current_safety_level(current_location),
            "crowd_density": self._get_current_crowd_density(current_location),
            "time_of_day": time_of_day,
            "current_time": self.current_time,
            "hours_until_night": hours_until_night,
            "weather": self.weather.value,
            "hunger_level": self.player_state["hunger_level"],
            "thirst_level": self.player_state["thirst_level"],
            "fatigue_level": self.player_state["fatigue_level"],
            "stress_level": self.player_state["stress_level"],
            "resources": self.player_state["resources"],
            "reputation": self.player_state["reputation"],
            "language_barrier": True,  # Always true for this scenario
            "people_nearby": [p.person_type.value for p in current_people],
            "people_details": [{"name": p.name, "type": p.person_type.value, "activity": p.current_activity, "helpful": p.helpfulness > 0.6} for p in current_people],
            "available_resources": current_location.available_resources,
            "visible_landmarks": self._get_visible_landmarks(),
            "possible_actions": self._get_possible_actions(),
            "has_shelter": "shelter" in self.player_state["resources"] or any("shelter" in r for r in current_location.available_resources)
        }
        
        return situation
    
    def _get_current_people(self, location: Location) -> List[Person]:
        """Get people currently at location based on time"""
        time_of_day = self._get_time_of_day()
        
        if time_of_day in location.time_dependent_changes:
            changes = location.time_dependent_changes[time_of_day]
            if "people" in changes:
                return changes["people"]
        
        return location.people
    
    def _get_current_safety_level(self, location: Location) -> float:
        """Get current safety level based on time and conditions"""
        base_safety = location.safety_level
        time_of_day = self._get_time_of_day()
        
        if time_of_day in location.time_dependent_changes:
            changes = location.time_dependent_changes[time_of_day]
            if "safety_level" in changes:
                base_safety = changes["safety_level"]
        
        # Weather effects
        if self.weather == WeatherType.RAINY:
            base_safety -= 0.1
        elif self.weather == WeatherType.COLD:
            base_safety -= 0.05
        
        return max(0, min(1, base_safety))
    
    def _get_current_crowd_density(self, location: Location) -> float:
        """Get current crowd density based on time"""
        base_density = location.crowd_density
        time_of_day = self._get_time_of_day()
        
        if time_of_day in location.time_dependent_changes:
            changes = location.time_dependent_changes[time_of_day]
            if "crowd_density" in changes:
                base_density = changes["crowd_density"]
        
        return base_density
    
    def _get_time_of_day(self) -> str:
        """Convert hour to time period"""
        if 6 <= self.current_time < 12:
            return "morning"
        elif 12 <= self.current_time < 18:
            return "afternoon"  
        elif 18 <= self.current_time < 22:
            return "evening"
        else:
            return "night"
    
    def _get_visible_landmarks(self) -> List[str]:
        """Get landmarks visible from current location"""
        current_location = self.locations[self.player_location]
        landmarks = []
        
        # Add connected locations as visible landmarks
        for connection in current_location.connections:
            connected_loc = self.locations[connection]
            if connected_loc.location_type in [LocationType.STATION, LocationType.SQUARE]:
                landmarks.append(connection)
        
        # Add resource-based landmarks
        if "church" in current_location.available_resources:
            landmarks.append("church")
        if any("restroom" in r for r in current_location.available_resources):
            landmarks.append("public_restroom")
        
        return landmarks
    
    def _get_possible_actions(self) -> List[str]:
        """Get actions possible at current location"""
        current_location = self.locations[self.player_location]
        current_people = self._get_current_people(current_location)
        
        actions = ["observe_surroundings", "rest", "wait"]
        
        # Movement actions
        for connection in current_location.connections:
            actions.append(f"go_to_{connection}")
        
        # People interaction actions
        for person in current_people:
            actions.append(f"approach_{person.person_type.value}")
            actions.append(f"gesture_to_{person.person_type.value}")
        
        # Resource actions
        if "food" in current_location.available_resources:
            actions.append("look_for_food")
        if "water" in current_location.available_resources:
            actions.append("get_water")
        if "shelter" in current_location.available_resources:
            actions.append("seek_shelter")
        if "restroom" in current_location.available_resources:
            actions.append("use_restroom")
        
        return actions
    
    def execute_action(self, action: str) -> Dict[str, Any]:
        """Execute an action and return results"""
        
        result = {
            "success": False,
            "description": "",
            "consequences": {},
            "new_resources": [],
            "reputation_change": 0,
            "time_passed": 0,
            "state_changes": {}
        }
        
        current_location = self.locations[self.player_location]
        current_people = self._get_current_people(current_location)
        
        # Movement actions
        if action.startswith("go_to_"):
            target = action.replace("go_to_", "")
            if target in current_location.connections:
                self.player_location = target
                result["success"] = True
                result["description"] = f"Moved to {self.locations[target].name}"
                result["time_passed"] = 10  # 10 minutes
                result["state_changes"]["fatigue_level"] = self.player_state["fatigue_level"] + 1
                
                if target not in self.player_state["discovered_locations"]:
                    self.player_state["discovered_locations"].append(target)
                    result["description"] += f" (New location discovered!)"
        
        # Interaction actions
        elif action.startswith("approach_") or action.startswith("gesture_to_"):
            person_type = action.split("_")[-1]
            target_people = [p for p in current_people if p.person_type.value == person_type]
            
            if target_people:
                person = target_people[0]
                interaction_result = self._simulate_interaction(action, person)
                result.update(interaction_result)
        
        # Resource actions
        elif action == "look_for_food":
            if "food" in current_location.available_resources:
                # Success depends on location and people
                vendors = [p for p in current_people if p.person_type == PersonType.VENDOR]
                if vendors:
                    result["success"] = True
                    result["description"] = "Found food vendor willing to help"
                    result["new_resources"] = ["small_meal"]
                    result["state_changes"]["hunger_level"] = max(0, self.player_state["hunger_level"] - 3)
                    result["reputation_change"] = 1
                else:
                    result["description"] = "Found food sources but need to negotiate"
            else:
                result["description"] = "No food sources found in this area"
        
        elif action == "get_water":
            if "water" in current_location.available_resources or "water_fountain" in current_location.available_resources:
                result["success"] = True
                result["description"] = "Found clean water source"
                result["state_changes"]["thirst_level"] = max(0, self.player_state["thirst_level"] - 4)
            else:
                result["description"] = "No water sources available"
        
        elif action == "seek_shelter":
            if "shelter" in current_location.available_resources:
                result["success"] = True
                result["description"] = "Found temporary shelter"
                result["new_resources"] = ["temporary_shelter"]
                result["state_changes"]["stress_level"] = max(0, self.player_state["stress_level"] - 2)
            else:
                result["description"] = "No shelter available at this location"
        
        # Default actions
        elif action == "observe_surroundings":
            result["success"] = True
            result["description"] = self._generate_observation()
            result["time_passed"] = 5
        
        elif action == "rest":
            result["success"] = True
            result["description"] = "Rested for a while"
            result["time_passed"] = 15
            result["state_changes"]["fatigue_level"] = max(0, self.player_state["fatigue_level"] - 1)
            result["state_changes"]["stress_level"] = max(0, self.player_state["stress_level"] - 1)
        
        # Apply consequences
        self._apply_action_consequences(result)
        
        return result
    
    def _simulate_interaction(self, action: str, person: Person) -> Dict[str, Any]:
        """Simulate interaction with a person"""
        
        result = {
            "success": False,
            "description": "",
            "consequences": {},
            "new_resources": [],
            "reputation_change": 0,
            "time_passed": 10,
            "state_changes": {}
        }
        
        # Base success chance
        success_chance = person.friendliness * person.helpfulness
        
        # Modify based on player reputation
        if self.player_state["reputation"] > 0:
            success_chance += 0.1 * self.player_state["reputation"]
        elif self.player_state["reputation"] < 0:
            success_chance += 0.05 * self.player_state["reputation"]  # Negative effect
        
        # Language barrier effects
        if person.language_barrier:
            success_chance *= 0.7  # Harder to communicate
        
        # Random factor
        roll = random.random()
        
        if roll < success_chance:
            result["success"] = True
            
            # Generate helpful outcome based on person type
            if person.person_type == PersonType.VENDOR:
                if "gesture_to" in action:
                    result["description"] = f"{person.name} understands your hunger and offers food"
                    result["new_resources"] = ["food_from_vendor"]
                    result["state_changes"]["hunger_level"] = max(0, self.player_state["hunger_level"] - 4)
                    result["reputation_change"] = 2
                else:
                    result["description"] = f"{person.name} is friendly but needs clearer communication"
                    result["reputation_change"] = 1
            
            elif person.person_type == PersonType.TOURIST:
                result["description"] = f"{person.name} offers to help and shows you their map"
                result["new_resources"] = ["map_information", "tourist_contact"]
                result["reputation_change"] = 1
                
            elif person.person_type == PersonType.POLICE:
                result["description"] = f"{person.name} is professional and offers official assistance"
                result["new_resources"] = ["police_contact", "safety_info"]
                result["reputation_change"] = 3
                
            elif person.person_type == PersonType.WORKER and "church" in person.resources:
                result["description"] = f"{person.name} invites you to the church for food and shelter"
                result["new_resources"] = ["church_meal", "church_shelter", "spiritual_support"]
                result["state_changes"]["hunger_level"] = max(0, self.player_state["hunger_level"] - 5)
                result["state_changes"]["stress_level"] = max(0, self.player_state["stress_level"] - 3)
                result["reputation_change"] = 3
                
            elif person.person_type == PersonType.HOMELESS:
                result["description"] = f"{person.name} shares street survival knowledge"
                result["new_resources"] = ["street_knowledge", "survival_tips"]
                result["reputation_change"] = 1
                
        else:
            result["description"] = f"{person.name} doesn't understand or is too busy to help"
            if roll > 0.8:  # Very bad interaction
                result["reputation_change"] = -1
                result["description"] += " and seems annoyed"
        
        return result
    
    def _generate_observation(self) -> str:
        """Generate detailed observation of current location"""
        current_location = self.locations[self.player_location]
        current_people = self._get_current_people(current_location)
        time_of_day = self._get_time_of_day()
        
        obs = f"You are at {current_location.name}. "
        obs += f"It's {time_of_day} ({self.current_time}:00) and the weather is {self.weather.value}. "
        
        if current_people:
            obs += f"You see {len(current_people)} people: "
            obs += ", ".join([f"{p.name} ({p.current_activity})" for p in current_people])
            obs += ". "
        
        obs += f"The area feels {'safe' if current_location.safety_level > 0.6 else 'somewhat unsafe'} "
        obs += f"and is {'crowded' if self._get_current_crowd_density(current_location) > 0.6 else 'quiet'}. "
        
        if current_location.available_resources:
            obs += f"Available resources: {', '.join(current_location.available_resources)}. "
        
        return obs
    
    def _apply_action_consequences(self, result: Dict[str, Any]):
        """Apply the consequences of an action to game state"""
        
        # Update player state
        for key, value in result.get("state_changes", {}).items():
            if key in self.player_state:
                self.player_state[key] = max(0, min(10, value))
        
        # Add new resources
        self.player_state["resources"].extend(result.get("new_resources", []))
        
        # Update reputation
        self.player_state["reputation"] += result.get("reputation_change", 0)
        self.player_state["reputation"] = max(-10, min(10, self.player_state["reputation"]))
        
        # Advance time
        time_passed = result.get("time_passed", 0)
        self.current_time += time_passed / 60  # Convert minutes to hours
        
        # Natural state degradation over time
        if time_passed > 0:
            self.player_state["hunger_level"] = min(10, self.player_state["hunger_level"] + time_passed / 120)
            self.player_state["thirst_level"] = min(10, self.player_state["thirst_level"] + time_passed / 60)
            self.player_state["fatigue_level"] = min(10, self.player_state["fatigue_level"] + time_passed / 180)
        
        # Record interaction
        self.interaction_history.append({
            "time": self.current_time,
            "action": result.get("action", "unknown"),
            "location": self.player_location,
            "result": result["description"],
            "success": result["success"]
        })
    
    def get_world_status(self) -> Dict[str, Any]:
        """Get complete world status for display"""
        return {
            "current_time": self.current_time,
            "time_of_day": self._get_time_of_day(),
            "weather": self.weather.value,
            "location": self.player_location,
            "player_state": self.player_state.copy(),
            "recent_interactions": self.interaction_history[-5:],
            "available_locations": list(self.locations.keys()),
            "discovered_locations": self.player_state["discovered_locations"]
        }
