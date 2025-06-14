"""
Motor and Action Systems
Implements Action Selection and Motor Plan Execution
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from memory_systems import MemoryContent
from consciousness import ConsciousContent


class ActionType(Enum):
    MOTOR = "motor"
    COGNITIVE = "cognitive"
    COMMUNICATIVE = "communicative"
    EXPLORATORY = "exploratory"


@dataclass
class Action:
    """Represents an action that can be executed"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    action_type: ActionType = ActionType.COGNITIVE
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    cost: float = 1.0
    success_probability: float = 1.0
    execution_time: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scheme:
    """Behavioral scheme that can be instantiated"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger_conditions: List[str] = field(default_factory=list)
    action_sequence: List[Action] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 1.0
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class MotorPlan:
    """Represents a motor execution plan"""
    scheme: Scheme
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instantiated_actions: List[Action] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    status: str = "planned"  # planned, executing, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)


class ActionSelection:
    """Chooses behaviors based on situational context"""
    
    def __init__(self):
        self.available_schemes: Dict[str, Scheme] = {}
        self.action_history: List[Dict[str, Any]] = []
        self.context_evaluators: List[Callable[[Dict[str, Any]], float]] = []
        self.selection_strategy = "utility_based"  # utility_based, random, greedy
    
    def register_scheme(self, scheme: Scheme) -> str:
        """Register a behavioral scheme"""
        self.available_schemes[scheme.id] = scheme
        return scheme.id
    
    def select_action(self, conscious_content: ConsciousContent, 
                     context: Dict[str, Any]) -> Optional[Scheme]:
        """Select appropriate action based on conscious content and context"""
        
        # Extract relevant information from conscious content
        situation_info = self._extract_situation_info(conscious_content)
        
        # Evaluate available schemes
        scheme_scores = {}
        for scheme_id, scheme in self.available_schemes.items():
            score = self._evaluate_scheme(scheme, situation_info, context)
            if score > 0:
                scheme_scores[scheme_id] = score
        
        if not scheme_scores:
            return None
        
        # Select best scheme based on strategy
        if self.selection_strategy == "utility_based":
            best_scheme_id = max(scheme_scores, key=scheme_scores.get)
        elif self.selection_strategy == "greedy":
            best_scheme_id = max(scheme_scores, key=scheme_scores.get)
        else:  # random selection weighted by scores
            import random
            total_score = sum(scheme_scores.values())
            if total_score > 0:
                weights = [score/total_score for score in scheme_scores.values()]
                best_scheme_id = random.choices(list(scheme_scores.keys()), weights=weights)[0]
            else:
                return None
        
        selected_scheme = self.available_schemes[best_scheme_id]
        
        # Update usage statistics
        selected_scheme.usage_count += 1
        selected_scheme.last_used = datetime.now()
        
        # Record selection
        self.action_history.append({
            "scheme_id": best_scheme_id,
            "score": scheme_scores[best_scheme_id],
            "context": context,
            "timestamp": datetime.now()
        })
        
        return selected_scheme
    
    def _extract_situation_info(self, conscious_content: ConsciousContent) -> Dict[str, Any]:
        """Extract relevant situation information from conscious content"""
        coalition = conscious_content.coalition
        
        situation_info = {
            "coalition_type": coalition.type.value,
            "coalition_strength": coalition.strength,
            "coalition_coherence": coalition.coherence,
            "content_count": len(coalition.contents),
            "activation_levels": [c.activation_level for c in coalition.contents],
            "content_types": [c.metadata.get("type", "unknown") for c in coalition.contents]
        }
        
        return situation_info
    
    def _evaluate_scheme(self, scheme: Scheme, situation_info: Dict[str, Any], 
                        context: Dict[str, Any]) -> float:
        """Evaluate how well a scheme fits the current situation"""
        
        # Check trigger conditions
        trigger_score = self._evaluate_triggers(scheme.trigger_conditions, situation_info, context)
        if trigger_score <= 0:
            return 0.0
        
        # Check context requirements
        context_score = self._evaluate_context_requirements(scheme.context_requirements, context)
        
        # Consider scheme success rate and usage
        success_bonus = scheme.success_rate
        experience_bonus = min(1.0, scheme.usage_count / 10.0)  # Bonus for experienced schemes
        
        # Calculate overall utility
        utility = (trigger_score * 0.4 + 
                  context_score * 0.3 + 
                  success_bonus * 0.2 + 
                  experience_bonus * 0.1)
        
        return utility
    
    def _evaluate_triggers(self, triggers: List[str], situation_info: Dict[str, Any], 
                          context: Dict[str, Any]) -> float:
        """Evaluate if trigger conditions are met"""
        if not triggers:
            return 0.5  # Neutral if no specific triggers
        
        met_triggers = 0
        for trigger in triggers:
            if self._check_trigger_condition(trigger, situation_info, context):
                met_triggers += 1
        
        return met_triggers / len(triggers)
    
    def _check_trigger_condition(self, trigger: str, situation_info: Dict[str, Any], 
                                context: Dict[str, Any]) -> bool:
        """Check if a specific trigger condition is met"""
        # Simple pattern matching for trigger conditions
        trigger_lower = trigger.lower()
        
        # Check coalition type triggers
        if "perceptual" in trigger_lower:
            return situation_info.get("coalition_type") == "perceptual"
        elif "spatial" in trigger_lower:
            return situation_info.get("coalition_type") == "spatial"
        elif "motor" in trigger_lower:
            return situation_info.get("coalition_type") == "motor"
        
        # Check strength triggers
        elif "strong" in trigger_lower:
            return situation_info.get("coalition_strength", 0) > 0.7
        elif "weak" in trigger_lower:
            return situation_info.get("coalition_strength", 0) < 0.3
        
        # Check context triggers
        elif "urgent" in trigger_lower:
            return context.get("urgency", 0) > 0.5
        elif "exploratory" in trigger_lower:
            return context.get("exploration_mode", False)
        
        return False
    
    def _evaluate_context_requirements(self, requirements: Dict[str, Any], 
                                     context: Dict[str, Any]) -> float:
        """Evaluate how well context meets scheme requirements"""
        if not requirements:
            return 1.0  # No requirements means always applicable
        
        met_requirements = 0
        for req_key, req_value in requirements.items():
            context_value = context.get(req_key)
            if context_value is not None:
                if isinstance(req_value, (int, float)):
                    # Numeric requirement
                    if abs(context_value - req_value) < 0.1:
                        met_requirements += 1
                elif context_value == req_value:
                    # Exact match requirement
                    met_requirements += 1
        
        return met_requirements / len(requirements)


class MotorPlanExecution:
    """Executes selected behaviors in the environment"""
    
    def __init__(self):
        self.active_plans: Dict[str, MotorPlan] = {}
        self.execution_history: List[MotorPlan] = []
        self.environment_interface: Optional[Callable[[Action, Dict[str, Any]], Dict[str, Any]]] = None
        self.feedback_callbacks: List[Callable[[MotorPlan], None]] = []
    
    def set_environment_interface(self, interface: Callable[[Action, Dict[str, Any]], Dict[str, Any]]):
        """Set the interface for interacting with the environment"""
        self.environment_interface = interface
    
    def register_feedback_callback(self, callback: Callable[[MotorPlan], None]):
        """Register callback for execution feedback"""
        self.feedback_callbacks.append(callback)
    
    def execute_scheme(self, scheme: Scheme, context: Dict[str, Any]) -> MotorPlan:
        """Execute a behavioral scheme"""
        
        # Create motor plan
        motor_plan = MotorPlan(
            scheme=scheme,
            execution_context=context,
            status="planned"
        )
        
        # Instantiate actions with current context
        motor_plan.instantiated_actions = self._instantiate_actions(
            scheme.action_sequence, context
        )
        
        # Add to active plans
        self.active_plans[motor_plan.id] = motor_plan
        
        # Execute the plan
        self._execute_plan(motor_plan)
        
        return motor_plan
    
    def _instantiate_actions(self, actions: List[Action], context: Dict[str, Any]) -> List[Action]:
        """Instantiate actions with current context parameters"""
        instantiated = []
        
        for action in actions:
            # Create a copy of the action with context-specific parameters
            instantiated_action = Action(
                name=action.name,
                action_type=action.action_type,
                preconditions=action.preconditions.copy(),
                effects=action.effects.copy(),
                parameters={**action.parameters, **context},
                priority=action.priority,
                cost=action.cost,
                success_probability=action.success_probability,
                execution_time=action.execution_time,
                metadata=action.metadata.copy()
            )
            
            instantiated.append(instantiated_action)
        
        return instantiated
    
    def _execute_plan(self, motor_plan: MotorPlan):
        """Execute a motor plan"""
        motor_plan.status = "executing"
        motor_plan.start_time = datetime.now()
        
        try:
            for action in motor_plan.instantiated_actions:
                # Check preconditions
                if not self._check_preconditions(action, motor_plan.execution_context):
                    motor_plan.status = "failed"
                    motor_plan.results["failure_reason"] = f"Preconditions not met for action {action.name}"
                    break
                
                # Execute action
                action_result = self._execute_action(action, motor_plan.execution_context)
                
                # Store result
                motor_plan.results[action.name] = action_result
                
                # Update context with action effects
                self._apply_action_effects(action, motor_plan.execution_context, action_result)
                
                # Check if action was successful
                if not action_result.get("success", True):
                    motor_plan.status = "failed"
                    motor_plan.results["failure_reason"] = f"Action {action.name} failed"
                    break
            
            if motor_plan.status == "executing":
                motor_plan.status = "completed"
        
        except Exception as e:
            motor_plan.status = "failed"
            motor_plan.results["error"] = str(e)
        
        finally:
            motor_plan.end_time = datetime.now()
            
            # Move to history
            self.execution_history.append(motor_plan)
            if motor_plan.id in self.active_plans:
                del self.active_plans[motor_plan.id]
            
            # Send feedback
            self._send_feedback(motor_plan)
    
    def _check_preconditions(self, action: Action, context: Dict[str, Any]) -> bool:
        """Check if action preconditions are met"""
        for precondition in action.preconditions:
            if not self._evaluate_condition(precondition, context):
                return False
        return True
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string against context"""
        # Simple condition evaluation
        condition_lower = condition.lower()
        
        if "available" in condition_lower:
            resource = condition_lower.replace("available", "").strip()
            return context.get(f"{resource}_available", True)
        elif "ready" in condition_lower:
            return context.get("ready", True)
        elif "safe" in condition_lower:
            return context.get("safe", True)
        
        return True  # Default to true for unknown conditions
    
    def _execute_action(self, action: Action, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action"""
        if self.environment_interface:
            try:
                return self.environment_interface(action, context)
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            # Simulate action execution
            return {
                "success": True,
                "action": action.name,
                "timestamp": datetime.now(),
                "simulated": True
            }
    
    def _apply_action_effects(self, action: Action, context: Dict[str, Any], 
                            action_result: Dict[str, Any]):
        """Apply action effects to the context"""
        for effect in action.effects:
            effect_lower = effect.lower()
            
            if "increase" in effect_lower:
                param = effect_lower.replace("increase", "").strip()
                context[param] = context.get(param, 0) + 1
            elif "decrease" in effect_lower:
                param = effect_lower.replace("decrease", "").strip()
                context[param] = context.get(param, 0) - 1
            elif "set" in effect_lower:
                parts = effect_lower.split("set")
                if len(parts) == 2:
                    param = parts[1].strip()
                    context[param] = True
    
    def _send_feedback(self, motor_plan: MotorPlan):
        """Send feedback to registered callbacks"""
        for callback in self.feedback_callbacks:
            try:
                callback(motor_plan)
            except Exception as e:
                print(f"Error in feedback callback: {e}")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        return {
            "active_plans": len(self.active_plans),
            "completed_plans": len([p for p in self.execution_history if p.status == "completed"]),
            "failed_plans": len([p for p in self.execution_history if p.status == "failed"]),
            "total_executions": len(self.execution_history)
        }
