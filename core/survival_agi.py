"""
Survival AGI Agent
Integrates global workspace consciousness with survival simulation environment
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time
from collections import defaultdict

from .hybrid_agi import HybridAGI
from .survival_simulation import SurvivalEnvironment, SimulationConfig, ResourceType, NPCBehavior
from .dreamer_world_model import DreamerWorldModel, DreamerConfig, WorldModelState
from .consciousness import ConsciousContent, Coalition, CoalitionType
from .memory_systems import MemoryContent

@dataclass
class SurvivalMemory:
    """Specialized memory for survival tasks"""
    location_memory: Dict[str, Any] = None
    resource_memory: Dict[str, Any] = None
    npc_memory: Dict[str, Any] = None
    hazard_memory: Dict[str, Any] = None
    strategy_memory: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.location_memory is None:
            self.location_memory = {}
        if self.resource_memory is None:
            self.resource_memory = {}
        if self.npc_memory is None:
            self.npc_memory = {}
        if self.hazard_memory is None:
            self.hazard_memory = {}
        if self.strategy_memory is None:
            self.strategy_memory = {"successful_actions": [], "failed_actions": []}

class SurvivalCognitiveModules:
    """Specialized cognitive modules for survival tasks"""
    
    def __init__(self, agi_core: HybridAGI):
        self.agi_core = agi_core
        self.survival_memory = SurvivalMemory()
        
        # Survival-specific attention codelets
        self.attention_codelets = {
            "resource_detection": self._resource_attention_codelet,
            "threat_assessment": self._threat_attention_codelet,
            "social_opportunity": self._social_attention_codelet,
            "navigation_planning": self._navigation_attention_codelet,
            "survival_monitoring": self._survival_monitoring_codelet
        }
    
    def _resource_attention_codelet(self, observation: Dict[str, Any]) -> Optional[Coalition]:
        """Detect and prioritize resources"""
        resources = observation.get("visible_resources", [])
        agent_hunger = observation.get("agent_hunger", 0)
        agent_thirst = observation.get("agent_thirst", 0)
        agent_shelter = observation.get("agent_shelter", 0)
        
        if not resources:
            return None
        
        # Calculate resource urgency
        resource_priorities = []
        for resource in resources:
            urgency = 0.0
            
            if resource["type"] == "food" and agent_hunger > 50:
                urgency = (agent_hunger / 100.0) * (1.0 / (resource["distance"] + 1))
            elif resource["type"] == "water" and agent_thirst > 60:
                urgency = (agent_thirst / 100.0) * (1.0 / (resource["distance"] + 1))
            elif resource["type"] == "shelter" and agent_shelter < 30:
                urgency = ((100 - agent_shelter) / 100.0) * (1.0 / (resource["distance"] + 1))
            
            if urgency > 0:
                resource_priorities.append((resource, urgency))
        
        if resource_priorities:
            # Find highest priority resource
            best_resource, max_urgency = max(resource_priorities, key=lambda x: x[1])
            
            if max_urgency > 0.3:  # Threshold for coalition formation
                # Create memory content for the resource action
                memory_content = MemoryContent(
                    id=f"resource_{best_resource['type']}",
                    content={
                        "type": "resource_action",
                        "target_resource": best_resource,
                        "urgency": max_urgency,
                        "action": "collect",
                        "rationale": f"Critical need for {best_resource['type']}"
                    },
                    metadata={"source": "resource_detection_codelet"},
                    activation_level=max_urgency
                )
                
                coalition = Coalition(
                    id=f"resource_acquisition_{best_resource['type']}",
                    type=CoalitionType.ATTENTION,
                    contents=[memory_content],
                    strength=max_urgency * 2.0,  # Amplify resource coalitions
                    metadata={"supporting_modules": ["perceptual_memory", "procedural_memory"]}
                )
                return coalition
        
        return None
    
    def _threat_attention_codelet(self, observation: Dict[str, Any]) -> Optional[Coalition]:
        """Assess and respond to threats"""
        hazards = observation.get("visible_hazards", [])
        npcs = observation.get("visible_npcs", [])
        agent_health = observation.get("agent_health", 100)
        
        threats = []
        
        # Check hazards
        for hazard in hazards:
            if hazard["distance"] < 10:  # Close hazards are threats
                threat_level = hazard["severity"] * (1.0 / (hazard["distance"] + 1))
                threats.append(("hazard", hazard, threat_level))
        
        # Check hostile NPCs
        for npc in npcs:
            if npc["behavior"] == "hostile" and npc["distance"] < 15:
                threat_level = npc["hostility"] * (1.0 / (npc["distance"] + 1))
                threats.append(("npc", npc, threat_level))
        
        if threats:
            # Find highest threat
            threat_type, threat_obj, max_threat = max(threats, key=lambda x: x[2])
            
            # Higher urgency if agent is injured
            if agent_health < 50:
                max_threat *= 1.5
            
            if max_threat > 0.4:  # Threat threshold
                memory_content = MemoryContent(
                    id=f"threat_{threat_type}",
                    content={
                        "type": "threat_response",
                        "threat": threat_obj,
                        "threat_level": max_threat,
                        "action": "avoid" if max_threat < 0.8 else "flee",
                        "rationale": f"Dangerous {threat_type} detected"
                    },
                    metadata={"source": "threat_assessment_codelet"},
                    activation_level=max_threat
                )
                
                coalition = Coalition(
                    id=f"threat_response_{threat_type}",
                    type=CoalitionType.MOTOR,
                    contents=[memory_content],
                    strength=max_threat * 3.0,  # Threats get high priority
                    metadata={"supporting_modules": ["spatial_memory", "motor_systems"]}
                )
                return coalition
        
        return None
    
    def _social_attention_codelet(self, observation: Dict[str, Any]) -> Optional[Coalition]:
        """Identify social opportunities"""
        npcs = observation.get("visible_npcs", [])
        inventory = observation.get("inventory", [])
        
        social_opportunities = []
        
        for npc in npcs:
            if npc["distance"] < 8:  # Close enough to interact
                opportunity_score = 0.0
                
                if npc["behavior"] == "friendly":
                    opportunity_score = 0.6 / (npc["distance"] + 1)
                elif npc["behavior"] == "trader" and len(inventory) > 0:
                    opportunity_score = 0.8 / (npc["distance"] + 1)
                
                if opportunity_score > 0:
                    social_opportunities.append((npc, opportunity_score))
        
        if social_opportunities:
            best_npc, max_opportunity = max(social_opportunities, key=lambda x: x[1])
            
            if max_opportunity > 0.2:
                memory_content = MemoryContent(
                    id=f"social_{best_npc['id']}",
                    content={
                        "type": "social_action",
                        "target_npc": best_npc,
                        "opportunity": max_opportunity,
                        "action": "interact",
                        "rationale": f"Social opportunity with {best_npc['behavior']} NPC"
                    },
                    metadata={"source": "social_opportunity_codelet"},
                    activation_level=max_opportunity * 0.7  # Lower urgency than survival needs
                )
                
                coalition = Coalition(
                    id=f"social_interaction_{best_npc['id']}",
                    type=CoalitionType.ATTENTION,
                    contents=[memory_content],
                    strength=max_opportunity,
                    metadata={"supporting_modules": ["declarative_memory", "social_cognition"]}
                )
                return coalition
        
        return None
    
    def _navigation_attention_codelet(self, observation: Dict[str, Any]) -> Optional[Coalition]:
        """Plan navigation and exploration"""
        agent_pos = observation.get("agent_position", (0, 0))
        visible_resources = observation.get("visible_resources", [])
        visible_npcs = observation.get("visible_npcs", [])
        
        # Simple exploration logic - move toward unexplored areas
        # In a real implementation, this would use spatial memory
        
        if not visible_resources and not visible_npcs:
            # No immediate targets, explore
            memory_content = MemoryContent(
                id="exploration_nav",
                content={
                    "type": "navigation_action",
                    "action": "explore",
                    "rationale": "No immediate targets, exploring for resources"
                },
                metadata={"source": "navigation_planning_codelet"},
                activation_level=0.2
            )
            
            coalition = Coalition(
                id="exploration_movement",
                type=CoalitionType.MOTOR,
                contents=[memory_content],
                strength=0.3,
                metadata={"supporting_modules": ["spatial_memory", "motor_systems"]}
            )
            return coalition
        
        return None
    
    def _survival_monitoring_codelet(self, observation: Dict[str, Any]) -> Optional[Coalition]:
        """Monitor overall survival status"""
        health = observation.get("agent_health", 100)
        hunger = observation.get("agent_hunger", 0)
        thirst = observation.get("agent_thirst", 0)
        shelter = observation.get("agent_shelter", 0)
        
        # Calculate overall survival stress
        survival_stress = (
            max(0, 100 - health) / 100.0 * 0.4 +
            hunger / 100.0 * 0.3 +
            thirst / 100.0 * 0.25 +
            max(0, 30 - shelter) / 30.0 * 0.05
        )
        
        if survival_stress > 0.7:  # High stress threshold
            memory_content = MemoryContent(
                id="survival_crisis",
                content={
                    "type": "survival_status",
                    "stress_level": survival_stress,
                    "critical_needs": {
                        "health": health < 30,
                        "hunger": hunger > 80,
                        "thirst": thirst > 90,
                        "shelter": shelter < 10
                    },
                    "action": "emergency_mode",
                    "rationale": "Critical survival situation detected"
                },
                metadata={"source": "survival_monitoring_codelet"},
                activation_level=survival_stress
            )
            
            coalition = Coalition(
                id="survival_crisis",
                type=CoalitionType.ATTENTION,
                contents=[memory_content],
                strength=survival_stress * 4.0,  # Crisis gets highest priority
                metadata={"supporting_modules": ["all_systems"]}
            )
            return coalition
        
        return None

class SurvivalAGI:
    """AGI agent specialized for survival tasks with global workspace consciousness"""
    
    def __init__(self, model_name: str = "survival_agi"):
        self.model_name = model_name
        
        # Initialize core hybrid AGI
        self.agi_core = HybridAGI(f"{model_name}_core")
        
        # Initialize survival-specific modules
        self.survival_modules = SurvivalCognitiveModules(self.agi_core)
        
        # Initialize Dreamer world model for planning
        dreamer_config = DreamerConfig(
            embed_dim=512,
            deter_dim=512,
            stoch_dim=32,
            horizon=10
        )
        
        # Observation dimension for survival environment
        obs_dim = 64  # Calculated based on encoded observation
        action_dim = 8  # 8 possible actions in survival environment
        
        self.world_model = DreamerWorldModel(obs_dim, action_dim, dreamer_config)
        self.world_model_state = self.world_model.init_state()
        
        # Action mapping for survival environment
        self.action_mapping = {
            0: {"type": "move", "direction": (-1, -1)},  # Northwest
            1: {"type": "move", "direction": (-1, 0)},   # North
            2: {"type": "move", "direction": (-1, 1)},   # Northeast
            3: {"type": "move", "direction": (0, -1)},   # West
            4: {"type": "move", "direction": (0, 1)},    # East
            5: {"type": "move", "direction": (1, -1)},   # Southwest
            6: {"type": "move", "direction": (1, 0)},    # South
            7: {"type": "move", "direction": (1, 1)},    # Southeast
        }
        
        # Performance tracking
        self.episode_metrics = []
        self.consciousness_history = []
        
    async def process_survival_situation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process survival situation through global workspace"""
        
        # 1. Run attention codelets to generate coalitions
        coalitions = []
        for codelet_name, codelet_func in self.survival_modules.attention_codelets.items():
            coalition = codelet_func(observation)
            if coalition:
                coalitions.append(coalition)
        
        # 2. Global workspace competition and selection
        if coalitions:
            # Select dominant coalition based on strength
            dominant_coalition = max(coalitions, key=lambda c: c.strength)
            
            # Get the content from the coalition
            if dominant_coalition.contents:
                conscious_content = dominant_coalition.contents[0].content
                conscious_content["coalition_id"] = dominant_coalition.id
            else:
                conscious_content = {"type": "default_action", "action": "wait"}
            
        else:
            # No strong coalitions, default processing
            conscious_content = {
                "type": "default_action", 
                "action": "wait",
                "rationale": "No clear action priorities detected"
            }
        
        # 3. Compute consciousness strength
        consciousness_strength = self._compute_consciousness_strength(
            coalitions, conscious_content, observation
        )
        
        # 4. Use world model for planning if consciousness is high
        action_plan = None
        if consciousness_strength > 1.0:
            # Update world model state with observation
            self.world_model_state, _ = self.world_model.observe(
                observation, self.world_model_state
            )
            
            # Plan action using world model
            planned_action, planning_info = self.world_model.plan_action(
                self.world_model_state, observation
            )
            action_plan = {
                "planned_action": planned_action,
                "planning_info": planning_info
            }
        
        # 5. Convert conscious content to action
        survival_action = self._conscious_content_to_action(
            conscious_content, observation, action_plan
        )
        
        # 6. Log consciousness event
        self._log_consciousness_event(
            consciousness_strength, conscious_content, survival_action, observation
        )
        
        return {
            "action": survival_action,
            "consciousness_strength": consciousness_strength,
            "dominant_coalition": conscious_content if coalitions else None,
            "all_coalitions": [c.contents[0].content if c.contents else {} for c in coalitions],
            "world_model_planning": action_plan is not None,
            "rationale": conscious_content.get("rationale", "No rationale")
        }
    
    def _compute_consciousness_strength(self, coalitions: List[Coalition], 
                                      conscious_content: Dict[str, Any],
                                      observation: Dict[str, Any]) -> float:
        """Compute emergent consciousness strength for survival situation"""
        
        base_strength = 0.5
        
        # Coalition competition increases consciousness
        if len(coalitions) > 1:
            competition_factor = len(coalitions) * 0.3
            strength_variance = np.var([c.strength for c in coalitions])
            base_strength += competition_factor + strength_variance
        
        # High urgency situations increase consciousness
        max_urgency = 0.0
        if coalitions:
            for c in coalitions:
                if c.contents:
                    urgency = c.contents[0].activation_level
                    max_urgency = max(max_urgency, urgency)
        base_strength += max_urgency * 1.5
        
        # Novel situations increase consciousness
        agent_health = observation.get("agent_health", 100)
        if agent_health < 30:  # Critical health is novel/urgent
            base_strength += 2.0
        
        visible_threats = len([h for h in observation.get("visible_hazards", []) 
                             if h["distance"] < 5])
        if visible_threats > 0:
            base_strength += visible_threats * 0.8
        
        # Survival stress increases consciousness
        hunger = observation.get("agent_hunger", 0)
        thirst = observation.get("agent_thirst", 0)
        survival_stress = (hunger + thirst) / 200.0
        base_strength += survival_stress * 1.2
        
        return min(10.0, base_strength)  # Cap at 10.0
    
    def _conscious_content_to_action(self, content: Dict[str, Any], 
                                   observation: Dict[str, Any],
                                   action_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert conscious content to survival environment action"""
        
        action_type = content.get("type", "default_action")
        
        if action_type == "resource_action":
            # Move toward and collect resource
            target = content["target_resource"]
            return {
                "type": "collect",
                "target_x": target["position"][0],
                "target_y": target["position"][1]
            }
        
        elif action_type == "threat_response":
            # Move away from threat
            threat = content["threat"]
            agent_pos = observation.get("agent_position", (0, 0))
            
            # Calculate direction away from threat
            threat_pos = threat["position"]
            dx = agent_pos[0] - threat_pos[0]
            dy = agent_pos[1] - threat_pos[1]
            
            # Normalize direction
            if dx != 0 or dy != 0:
                length = np.sqrt(dx*dx + dy*dy)
                dx = int(np.sign(dx / length))
                dy = int(np.sign(dy / length))
            
            return {
                "type": "move",
                "direction": (dx, dy)
            }
        
        elif action_type == "social_action":
            # Interact with NPC
            target_npc = content["target_npc"]
            return {
                "type": "interact",
                "target_id": target_npc["id"]
            }
        
        elif action_type == "navigation_action":
            # Exploration movement
            if action_plan and "planned_action" in action_plan:
                # Use world model planning
                planned = action_plan["planned_action"]
                action_idx = torch.argmax(planned).item()
                return self.action_mapping.get(action_idx, {"type": "wait"})
            else:
                # Random exploration
                direction = np.random.choice([
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)
                ])
                return {"type": "move", "direction": direction}
        
        elif action_type == "survival_status":
            # Emergency mode - prioritize most critical need
            critical_needs = content.get("critical_needs", {})
            
            if critical_needs.get("thirst", False):
                # Find water urgently
                water_resources = [r for r in observation.get("visible_resources", [])
                                 if r["type"] == "water"]
                if water_resources:
                    closest_water = min(water_resources, key=lambda r: r["distance"])
                    return {
                        "type": "collect",
                        "target_x": closest_water["position"][0],
                        "target_y": closest_water["position"][1]
                    }
            
            elif critical_needs.get("hunger", False):
                # Find food urgently
                food_resources = [r for r in observation.get("visible_resources", [])
                                if r["type"] == "food"]
                if food_resources:
                    closest_food = min(food_resources, key=lambda r: r["distance"])
                    return {
                        "type": "collect",
                        "target_x": closest_food["position"][0],
                        "target_y": closest_food["position"][1]
                    }
            
            # Default emergency action - find shelter or explore
            return {"type": "move", "direction": (0, 0)}
        
        else:
            # Default action
            return {"type": "wait"}
    
    def _log_consciousness_event(self, strength: float, content: Dict[str, Any],
                               action: Dict[str, Any], observation: Dict[str, Any]):
        """Log consciousness event for research analysis"""
        
        event = {
            "timestamp": time.time(),
            "consciousness_strength": strength,
            "content_type": content.get("type", "unknown"),
            "action_taken": action,
            "rationale": content.get("rationale", ""),
            "agent_state": {
                "health": observation.get("agent_health", 100),
                "hunger": observation.get("agent_hunger", 0),
                "thirst": observation.get("agent_thirst", 0),
                "position": observation.get("agent_position", (0, 0))
            },
            "environmental_context": {
                "visible_resources": len(observation.get("visible_resources", [])),
                "visible_npcs": len(observation.get("visible_npcs", [])),
                "visible_hazards": len(observation.get("visible_hazards", [])),
                "lighting": observation.get("lighting_level", 1.0),
                "weather": observation.get("weather_severity", 0.0)
            }
        }
        
        self.consciousness_history.append(event)
    
    async def run_survival_episode(self, env: SurvivalEnvironment, 
                                 max_steps: int = 1000) -> Dict[str, Any]:
        """Run a complete survival episode"""
        
        observation = env.reset()
        episode_rewards = []
        episode_actions = []
        episode_consciousness = []
        
        step = 0
        total_reward = 0.0
        done = False
        
        print(f"🏃 Starting survival episode with {self.model_name}")
        
        while not done and step < max_steps:
            # Process situation through global workspace
            decision_result = await self.process_survival_situation(observation)
            
            # Extract action
            action = decision_result["action"]
            consciousness_strength = decision_result["consciousness_strength"]
            
            # Execute action in environment
            next_observation, reward, done, info = env.step(action)
            
            # Store experience in world model
            if hasattr(self, 'world_model'):
                action_tensor = torch.zeros(len(self.action_mapping))
                if action["type"] == "move":
                    # Find matching action index
                    for idx, mapped_action in self.action_mapping.items():
                        if (mapped_action["type"] == "move" and 
                            mapped_action.get("direction") == action.get("direction")):
                            action_tensor[idx] = 1.0
                            break
                
                self.world_model.add_experience(
                    observation, action_tensor, reward, next_observation, done
                )
            
            # Log consciousness event in environment
            if consciousness_strength > 1.5:  # High consciousness threshold
                env.log_consciousness_event(
                    "high_consciousness_decision",
                    consciousness_strength,
                    str(action),
                    decision_result
                )
            
            # Track episode data
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_consciousness.append(consciousness_strength)
            
            total_reward += reward
            observation = next_observation
            step += 1
            
            # Periodic output
            if step % 100 == 0:
                print(f"Step {step}: Reward={reward:.2f}, "
                      f"Consciousness={consciousness_strength:.2f}, "
                      f"Health={observation.get('agent_health', 0):.1f}")
        
        # Episode summary
        episode_metrics = {
            "episode_length": step,
            "total_reward": total_reward,
            "average_reward": np.mean(episode_rewards),
            "survival_time": info["metrics"]["survival_time"],
            "resources_found": info["metrics"]["resources_found"],
            "hazards_encountered": info["metrics"]["hazards_encountered"],
            "consciousness_spikes": sum(1 for c in episode_consciousness if c > 1.5),
            "average_consciousness": np.mean(episode_consciousness),
            "max_consciousness": max(episode_consciousness),
            "final_agent_state": info["survival_status"]
        }
        
        self.episode_metrics.append(episode_metrics)
        
        print(f"✅ Episode completed: {episode_metrics}")
        
        return {
            "metrics": episode_metrics,
            "consciousness_events": env.consciousness_events,
            "rewards": episode_rewards,
            "actions": episode_actions,
            "consciousness_strength": episode_consciousness
        }
    
    def train_world_model(self, num_steps: int = 1000):
        """Train the world model on collected experience"""
        if not hasattr(self.world_model, 'experience_buffer'):
            print("No experience buffer found")
            return
        
        if len(self.world_model.experience_buffer) < 100:
            print("Not enough experience for training")
            return
        
        print(f"🎯 Training world model on {len(self.world_model.experience_buffer)} experiences")
        
        optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.0003)
        
        for step in range(num_steps):
            batch = self.world_model.sample_batch()
            if batch is None:
                continue
            
            losses = self.world_model.compute_losses(batch)
            total_loss = losses["world_model"]
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Training step {step}: Loss={total_loss.item():.4f}")
        
        print("✅ World model training completed")
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Generate comprehensive research summary"""
        
        if not self.episode_metrics:
            return {"error": "No episodes completed"}
        
        # Aggregate episode metrics
        total_episodes = len(self.episode_metrics)
        avg_survival_time = np.mean([m["survival_time"] for m in self.episode_metrics])
        avg_resources_found = np.mean([m["resources_found"] for m in self.episode_metrics])
        avg_consciousness = np.mean([m["average_consciousness"] for m in self.episode_metrics])
        total_consciousness_spikes = sum([m["consciousness_spikes"] for m in self.episode_metrics])
        
        # Consciousness-performance correlation
        consciousness_scores = [m["average_consciousness"] for m in self.episode_metrics]
        performance_scores = [m["total_reward"] for m in self.episode_metrics]
        
        if len(consciousness_scores) > 1:
            correlation = np.corrcoef(consciousness_scores, performance_scores)[0, 1]
        else:
            correlation = 0.0
        
        return {
            "experiment_summary": {
                "total_episodes": total_episodes,
                "average_survival_time": avg_survival_time,
                "average_resources_found": avg_resources_found,
                "total_consciousness_spikes": total_consciousness_spikes,
                "consciousness_performance_correlation": correlation
            },
            "consciousness_analysis": {
                "average_consciousness_strength": avg_consciousness,
                "max_consciousness_observed": max([m["max_consciousness"] for m in self.episode_metrics]),
                "high_consciousness_episodes": sum(1 for m in self.episode_metrics 
                                                 if m["max_consciousness"] > 2.0),
                "consciousness_events_logged": len(self.consciousness_history)
            },
            "survival_performance": {
                "average_episode_reward": np.mean([m["total_reward"] for m in self.episode_metrics]),
                "survival_success_rate": sum(1 for m in self.episode_metrics
                                           if m["final_agent_state"]["alive"]) / total_episodes,
                "resource_acquisition_efficiency": avg_resources_found / avg_survival_time if avg_survival_time > 0 else 0
            },
            "world_model_stats": {
                "experiences_collected": len(self.world_model.experience_buffer) if hasattr(self.world_model, 'experience_buffer') else 0,
                "model_parameters": sum(p.numel() for p in self.world_model.parameters())
            }
        }
