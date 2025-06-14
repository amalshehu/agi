"""
Cognitive Agent - Main orchestrator using LangGraph functional API
Integrates all components of the cognitive architecture
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import our cognitive architecture components
from memory_systems import (
    SensoryMemory, PerceptualAssociativeMemory, SpatialMemory,
    TransientEpisodicMemory, DeclarativeMemory, ProceduralMemory,
    SensoryMotorMemory, SensoryStimulus, MemoryContent
)
from consciousness import (
    GlobalWorkspace, CurrentSituationalModel, StructureBuildingCodelet,
    AttentionCodelet, Coalition, ConsciousContent
)
from motor_systems import ActionSelection, MotorPlanExecution, Scheme, Action, ActionType
from learning_pathways import LearningCoordinator


class AgentState(TypedDict):
    """State maintained by the cognitive agent"""
    messages: Annotated[list, add_messages]
    current_stimulus: Optional[Dict[str, Any]]
    situational_model: Dict[str, Any]
    conscious_content: Optional[Dict[str, Any]]
    active_coalitions: List[Dict[str, Any]]
    selected_action: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    learning_events: List[Dict[str, Any]]
    cycle_count: int
    timestamp: str


@dataclass
class CognitiveAgent:
    """Main cognitive agent implementing the full architecture"""
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Initialize memory systems
        self.sensory_memory = SensoryMemory()
        self.perceptual_memory = PerceptualAssociativeMemory()
        self.spatial_memory = SpatialMemory()
        self.transient_episodic = TransientEpisodicMemory()
        self.declarative_memory = DeclarativeMemory()
        self.procedural_memory = ProceduralMemory()
        self.sensory_motor_memory = SensoryMotorMemory()
        
        # Initialize consciousness layer
        self.global_workspace = GlobalWorkspace()
        self.situational_model = CurrentSituationalModel()
        self.structure_codelet = StructureBuildingCodelet("structure_builder")
        self.attention_codelet = AttentionCodelet("attention_director", "focus")
        
        # Initialize motor systems
        self.action_selection = ActionSelection()
        self.motor_execution = MotorPlanExecution()
        
        # Initialize learning coordinator
        self.learning_coordinator = LearningCoordinator()
        
        # Initialize default schemes
        self._initialize_default_schemes()
        
        # Build the LangGraph
        self.graph = self._build_graph()
        
        # Agent state
        self.cycle_count = 0
        self.last_conscious_content = None
        
    def _initialize_default_schemes(self):
        """Initialize some default behavioral schemes"""
        
        # Exploration scheme
        explore_action = Action(
            name="explore_environment",
            action_type=ActionType.EXPLORATORY,
            preconditions=["safe", "ready"],
            effects=["increase_knowledge", "update_spatial_map"],
            parameters={"exploration_radius": 5.0}
        )
        
        explore_scheme = Scheme(
            name="exploration",
            trigger_conditions=["exploratory", "weak_activation"],
            action_sequence=[explore_action],
            context_requirements={"exploration_mode": True}
        )
        
        self.action_selection.register_scheme(explore_scheme)
        
        # Response scheme
        respond_action = Action(
            name="generate_response",
            action_type=ActionType.COMMUNICATIVE,
            preconditions=["conscious_content_available"],
            effects=["communicate_understanding"],
            parameters={"response_type": "verbal"}
        )
        
        respond_scheme = Scheme(
            name="communicate",
            trigger_conditions=["perceptual", "strong"],
            action_sequence=[respond_action],
            context_requirements={"communication_required": True}
        )
        
        self.action_selection.register_scheme(respond_scheme)
        
        # Attention focusing scheme
        focus_action = Action(
            name="focus_attention",
            action_type=ActionType.COGNITIVE,
            preconditions=["multiple_stimuli"],
            effects=["increased_focus", "reduced_distraction"],
            parameters={"focus_duration": 3.0}
        )
        
        focus_scheme = Scheme(
            name="focus_attention",
            trigger_conditions=["attention", "multiple_inputs"],
            action_sequence=[focus_action],
            context_requirements={"attention_required": True}
        )
        
        self.action_selection.register_scheme(focus_scheme)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the cognitive cycle"""
        
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes for each stage of the cognitive cycle
        graph.add_node("sensory_processing", self._sensory_processing)
        graph.add_node("perceptual_integration", self._perceptual_integration)
        graph.add_node("situational_modeling", self._situational_modeling)
        graph.add_node("consciousness_competition", self._consciousness_competition)
        graph.add_node("action_selection", self._action_selection_node)
        graph.add_node("motor_execution", self._motor_execution_node)
        graph.add_node("learning_update", self._learning_update)
        
        # Define the flow
        graph.add_edge(START, "sensory_processing")
        graph.add_edge("sensory_processing", "perceptual_integration")
        graph.add_edge("perceptual_integration", "situational_modeling")
        graph.add_edge("situational_modeling", "consciousness_competition")
        graph.add_edge("consciousness_competition", "action_selection")
        graph.add_edge("action_selection", "motor_execution")
        graph.add_edge("motor_execution", "learning_update")
        graph.add_edge("learning_update", END)
        
        return graph.compile()
    
    def _sensory_processing(self, state: AgentState) -> AgentState:
        """Process sensory input"""
        
        # Extract stimulus from messages or state
        stimulus_data = state.get("current_stimulus")
        if not stimulus_data and state.get("messages"):
            # Create stimulus from latest message
            latest_message = state["messages"][-1]
            stimulus_data = {
                "modality": "linguistic",
                "data": latest_message.content if hasattr(latest_message, 'content') else str(latest_message),
                "intensity": 1.0
            }
        
        if stimulus_data:
            # Create sensory stimulus
            stimulus = SensoryStimulus(
                modality=stimulus_data.get("modality", "unknown"),
                data=stimulus_data.get("data"),
                intensity=stimulus_data.get("intensity", 1.0)
            )
            
            # Store in sensory memory
            self.sensory_memory.store(stimulus)
        
        # Update state
        state["current_stimulus"] = stimulus_data
        state["timestamp"] = datetime.now().isoformat()
        
        return state
    
    def _perceptual_integration(self, state: AgentState) -> AgentState:
        """Integrate perceptual information"""
        
        # Get cues from sensory memory
        sensory_cues = self.sensory_memory.get_cues()
        
        # Process cues in perceptual memory
        perceptual_contents = self.perceptual_memory.process_cues(sensory_cues)
        
        # Update spatial context if relevant
        for content in perceptual_contents:
            if content.metadata.get("modality") in ["spatial", "visual"]:
                # Simple spatial relation extraction
                if "near" in str(content.content).lower():
                    self.spatial_memory.store_spatial_relation(
                        "agent", "object", "near", content.activation_level
                    )
        
        return state
    
    def _situational_modeling(self, state: AgentState) -> AgentState:
        """Build current situational model"""
        
        # Get recent perceptual contents
        recent_percepts = self.perceptual_memory.contents[-10:]
        
        # Integrate into situational model
        self.situational_model.integrate_percepts(recent_percepts)
        
        # Get spatial context
        spatial_context = {}
        for content in self.spatial_memory.contents[-5:]:
            if content.activation_level > 0.3:
                spatial_context[content.id] = content.content
        
        self.situational_model.integrate_spatial_context(spatial_context)
        
        # Get associations from declarative memory
        associations = []
        for percept in recent_percepts:
            assoc = self.declarative_memory.retrieve_associations(percept.id)
            associations.extend(assoc)
        
        self.situational_model.integrate_associations(associations)
        
        # Update state with situational model
        state["situational_model"] = self.situational_model.get_coherent_representation()
        
        return state
    
    def _consciousness_competition(self, state: AgentState) -> AgentState:
        """Run consciousness competition"""
        
        # Get all available contents
        all_contents = (
            self.perceptual_memory.contents[-5:] +
            self.spatial_memory.contents[-3:] +
            self.transient_episodic.episodes[-2:]
        )
        
        # Build coalitions using structure building codelet
        coalitions = []
        context = state.get("situational_model", {})
        
        # Group contents into coalitions
        for i in range(0, len(all_contents), 3):  # Group by 3s
            content_group = all_contents[i:i+3]
            coalition = self.structure_codelet.build_coalition(content_group, context)
            if coalition:
                coalitions.append(coalition)
        
        # Add coalitions to global workspace
        for coalition in coalitions:
            self.global_workspace.add_coalition(coalition)
        
        # Run competition
        conscious_content = self.global_workspace.run_competition()
        
        # Update state
        state["active_coalitions"] = [
            {
                "id": c.id,
                "type": c.type.value,
                "strength": c.strength,
                "coherence": c.coherence
            } for c in coalitions
        ]
        
        if conscious_content:
            state["conscious_content"] = {
                "id": conscious_content.coalition.id,
                "type": conscious_content.coalition.type.value,
                "strength": conscious_content.coalition.strength,
                "availability": conscious_content.global_availability
            }
            self.last_conscious_content = conscious_content
        
        return state
    
    def _action_selection_node(self, state: AgentState) -> AgentState:
        """Select appropriate action"""
        
        if not self.last_conscious_content:
            return state
        
        # Create action context
        context = {
            "communication_required": True,
            "exploration_mode": False,
            "attention_required": len(state.get("active_coalitions", [])) > 1,
            "urgency": 0.5,
            "safe": True,
            "ready": True
        }
        
        # Select action using conscious content
        selected_scheme = self.action_selection.select_action(
            self.last_conscious_content, context
        )
        
        if selected_scheme:
            state["selected_action"] = {
                "scheme_id": selected_scheme.id,
                "scheme_name": selected_scheme.name,
                "action_count": len(selected_scheme.action_sequence),
                "context": context
            }
        
        return state
    
    def _motor_execution_node(self, state: AgentState) -> AgentState:
        """Execute selected action"""
        
        selected_action = state.get("selected_action")
        if not selected_action:
            return state
        
        # Get the scheme
        scheme_id = selected_action["scheme_id"]
        scheme = self.action_selection.available_schemes.get(scheme_id)
        
        if scheme:
            # Set up simple environment interface
            def simple_environment(action: Action, context: Dict[str, Any]) -> Dict[str, Any]:
                """Simple environment simulation"""
                if action.name == "generate_response":
                    # Generate a response based on conscious content
                    response = self._generate_response(context)
                    return {"success": True, "response": response}
                elif action.name == "explore_environment":
                    return {"success": True, "discovered": "new_information"}
                elif action.name == "focus_attention":
                    return {"success": True, "focused": True}
                else:
                    return {"success": True, "simulated": True}
            
            self.motor_execution.set_environment_interface(simple_environment)
            
            # Execute the scheme
            motor_plan = self.motor_execution.execute_scheme(
                scheme, selected_action["context"]
            )
            
            state["execution_result"] = {
                "plan_id": motor_plan.id,
                "status": motor_plan.status,
                "results": motor_plan.results
            }
        
        return state
    
    def _learning_update(self, state: AgentState) -> AgentState:
        """Update learning systems"""
        
        # Prepare memory systems for learning
        memory_systems = {
            "perceptual": self.perceptual_memory,
            "spatial": self.spatial_memory,
            "transient_episodic": self.transient_episodic,
            "declarative": self.declarative_memory,
            "procedural": self.procedural_memory,
            "sensory_motor": self.sensory_motor_memory
        }
        
        # Run coordinated learning
        learning_results = self.learning_coordinator.coordinate_learning(
            memory_systems, self.last_conscious_content
        )
        
        # Create episode for this cognitive cycle
        cycle_events = [
            state.get("current_stimulus"),
            state.get("conscious_content"),
            state.get("selected_action"),
            state.get("execution_result")
        ]
        
        cycle_context = {
            "cycle_count": self.cycle_count,
            "timestamp": state.get("timestamp"),
            "coherence": state.get("situational_model", {}).get("coherence", 0.0)
        }
        
        episode = self.transient_episodic.create_episode(cycle_events, cycle_context)
        
        # Update state
        state["learning_events"] = [
            {"pathway": pathway, "count": len(events)}
            for pathway, events in learning_results.items()
        ]
        
        state["cycle_count"] = self.cycle_count
        self.cycle_count += 1
        
        return state
    
    def _generate_response(self, context: Dict[str, Any]) -> str:
        """Generate response based on current conscious content"""
        
        if not self.last_conscious_content:
            return "I'm processing the information..."
        
        coalition = self.last_conscious_content.coalition
        
        # Generate response based on coalition type and content
        if coalition.type.value == "perceptual":
            return f"I perceive something with strength {coalition.strength:.2f} and coherence {coalition.coherence:.2f}."
        elif coalition.type.value == "spatial":
            return f"I'm aware of spatial relationships in my environment with coherence {coalition.coherence:.2f}."
        elif coalition.type.value == "episodic":
            return f"I'm recalling relevant experiences from my memory."
        else:
            return f"I'm processing {coalition.type.value} information with {len(coalition.contents)} elements."
    
    async def process_input(self, input_text: str) -> str:
        """Process input through the cognitive architecture"""
        
        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=input_text)],
            current_stimulus=None,
            situational_model={},
            conscious_content=None,
            active_coalitions=[],
            selected_action=None,
            execution_result=None,
            learning_events=[],
            cycle_count=self.cycle_count,
            timestamp=datetime.now().isoformat()
        )
        
        # Run the cognitive cycle
        result = await self.graph.ainvoke(initial_state)
        
        # Extract response from execution result
        execution_result = result.get("execution_result", {})
        response = execution_result.get("results", {}).get("generate_response", {}).get("response")
        
        if not response:
            response = "I'm still processing your input..."
        
        return response
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "cycle_count": self.cycle_count,
            "memory_stats": {
                "sensory_contents": len(self.sensory_memory.contents),
                "perceptual_contents": len(self.perceptual_memory.contents),
                "spatial_contents": len(self.spatial_memory.contents),
                "episodes": len(self.transient_episodic.episodes),
                "declarative_facts": len(self.declarative_memory.facts),
                "procedural_schemes": len(self.procedural_memory.schemes)
            },
            "consciousness_stats": {
                "current_conscious": self.last_conscious_content is not None,
                "situational_coherence": self.situational_model.coherence_score
            },
            "action_stats": self.motor_execution.get_execution_status(),
            "learning_stats": self.learning_coordinator.get_learning_statistics()
        }


# Convenience function to create and run the agent
async def run_cognitive_agent(input_text: str, agent_id: str = None) -> Dict[str, Any]:
    """Convenience function to run the cognitive agent"""
    
    agent = CognitiveAgent(agent_id)
    response = await agent.process_input(input_text)
    status = agent.get_agent_status()
    
    return {
        "response": response,
        "status": status,
        "agent_id": agent.agent_id
    }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create agent
        agent = CognitiveAgent("demo_agent")
        
        # Process some inputs
        inputs = [
            "Hello, how are you?",
            "What can you see around you?",
            "Tell me about your memory systems.",
            "Can you learn from our conversation?"
        ]
        
        for input_text in inputs:
            print(f"\nInput: {input_text}")
            response = await agent.process_input(input_text)
            print(f"Response: {response}")
            
            # Show status every few cycles
            if agent.cycle_count % 2 == 0:
                status = agent.get_agent_status()
                print(f"Status: Cycle {status['cycle_count']}, "
                      f"Coherence: {status['consciousness_stats']['situational_coherence']:.2f}")
    
    # Run the demo
    asyncio.run(main())
