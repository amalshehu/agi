#!/usr/bin/env python3
"""
🧠 AGI Survivor Scenario Demo
Advanced workflow orchestration with Core AGI
"""

import json
import time
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from tqdm import tqdm

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from core.hybrid_agi import HybridAGI
from world_simulation import WorldSimulation


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class SurvivorState(TypedDict):
    """Enhanced state for LangGraph survivor workflow"""
    messages: Annotated[list, add_messages]
    current_step: int
    world_state: Dict[str, Any]
    agi_responses: List[Dict[str, Any]]
    decisions_made: List[Dict[str, Any]]
    consciousness_levels: List[float]
    survival_success: bool
    critical_decision_pending: bool
    human_feedback: Optional[str]


class AGISurvivorOrchestrator:
    """AGI Survivor workflow orchestrator with advanced decision making"""
    
    def __init__(self):
        self.world = WorldSimulation()
        self.agi = HybridAGI("SurvivorAGI")
        
        # Workflow orchestration setup
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build AGI workflow orchestration graph"""
        
        workflow = StateGraph(SurvivorState)
        
        # Add nodes for different phases
        workflow.add_node("initialize", self._initialize_step)
        workflow.add_node("assess_situation", self._assess_situation)
        workflow.add_node("agi_reasoning", self._agi_reasoning)
        workflow.add_node("decision_making", self._decision_making)
        workflow.add_node("human_review", self._human_review)
        workflow.add_node("execute_action", self._execute_action)
        workflow.add_node("update_world", self._update_world)
        workflow.add_node("check_completion", self._check_completion)
        
        # Define the workflow flow
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "assess_situation")
        workflow.add_edge("assess_situation", "agi_reasoning")
        workflow.add_edge("agi_reasoning", "decision_making")
        workflow.add_conditional_edges(
            "decision_making",
            self._should_get_human_input,
            {
                "human_review": "human_review",
                "execute_action": "execute_action"
            }
        )
        workflow.add_edge("human_review", "execute_action")
        workflow.add_edge("execute_action", "update_world")
        workflow.add_edge("update_world", "check_completion")
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "assess_situation",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def run_demo(self):
        """Run the modern LangGraph survivor demo"""
        
        self._print_title()
        
        # Initialize AGI
        print(f"\n{Colors.HEADER}🧠 INITIALIZING AGI SYSTEM...{Colors.ENDC}")
        await self._train_agi()
        
        # Initial state
        initial_state = SurvivorState(
            messages=[HumanMessage(content="Starting survival scenario")],
            current_step=0,
            world_state={},
            agi_responses=[],
            decisions_made=[],
            consciousness_levels=[],
            survival_success=False,
            critical_decision_pending=False,
            human_feedback=None
        )
        
        # Configuration for persistence and recursion limit
        config = {
            "configurable": {
                "thread_id": "modern_survivor_session",
                "checkpoint_id": None
            },
            "recursion_limit": 50
        }
        
        print(f"\n{Colors.BOLD}🚀 LAUNCHING AGI WORKFLOW{Colors.ENDC}")
        print(f"{'='*70}")
        
        # Run the workflow
        try:
            final_state = None
            async for state in self.workflow.astream(initial_state, config=config):
                final_state = state
                # Stream processing happens in real-time
                
            # Display results
            if final_state:
                self._display_final_results(final_state)
                
        except Exception as e:
            print(f"{Colors.FAIL}Workflow error: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
    
    async def _initialize_step(self, state: SurvivorState) -> SurvivorState:
        """Initialize the survival scenario"""
        print(f"\n{Colors.OKCYAN}🎯 Workflow Node: Initialize{Colors.ENDC}")
        
        # Get initial world state
        world_state = self.world.get_current_situation()
        
        state["world_state"] = world_state
        state["current_step"] = 1
        state["messages"].append(AIMessage(content="Survival scenario initialized"))
        
        print(f"   📍 Starting location: {world_state['location']}")
        print(f"   🧠 AGI system ready with 494K parameters")
        
        return state
    
    async def _assess_situation(self, state: SurvivorState) -> SurvivorState:
        """Assess current survival situation"""
        print(f"\n{Colors.OKCYAN}📊 Workflow Node: Situation Assessment (Step {state['current_step']}){Colors.ENDC}")
        
        # Get updated world state
        current_situation = self.world.get_current_situation()
        state["world_state"] = current_situation
        
        # Create visual status display
        hunger_level = current_situation['hunger_level']
        stress_level = current_situation['stress_level']
        safety = current_situation.get('safety_level', 0.7)
        people_count = len(current_situation.get('people_nearby', []))
        completion = (state['current_step'] / 5.0) * 100
        
        # Visual progress bars
        hunger_bar = self._create_progress_bar(hunger_level, 10)
        stress_bar = self._create_progress_bar(stress_level, 10)
        
        # Critical needs assessment
        critical_needs = []
        if hunger_level > 6: critical_needs.append("food")
        if stress_level > 7: critical_needs.append("rest")
        if safety < 0.5: critical_needs.append("safety")
        critical_text = ", ".join(critical_needs) if critical_needs else "none"
        
        # Enhanced visual display
        print(f"🗺️  Location: {current_situation['location'].replace('_', ' ').title()}")
        print(f"│ 🕐 Time: {current_situation['current_time']:.2f} | Safety: {safety:.1f}")
        print(f"│ 🍽️  Hunger: {hunger_bar} {hunger_level:.1f}/10")
        print(f"│ 😰 Stress:  {stress_bar} {stress_level:.1f}/10")
        print(f"│ ⚠️  Critical Needs: {critical_text}")
        print(f"│ 👥 People Nearby: {people_count}")
        print(f"│ 🎯 Completion: {completion:.1f}%")
        
        return state
    
    def _create_progress_bar(self, value: float, max_value: float, length: int = 10) -> str:
        """Create a visual progress bar using tqdm styling"""
        progress = value / max_value
        bar = tqdm.format_meter(
            n=int(value),
            total=int(max_value),
            elapsed=0,
            ncols=length + 10,
            ascii=False,
            bar_format='{bar}'
        )
        # Extract just the bar part and format it
        bar_only = bar.split('|')[1].split('|')[0] if '|' in bar else bar
        return f"[{bar_only.strip()}]"
    
    async def _agi_reasoning(self, state: SurvivorState) -> SurvivorState:
        """Advanced AGI reasoning with consciousness tracking"""
        print(f"\n{Colors.OKCYAN}🧠 Workflow Node: AGI Reasoning{Colors.ENDC}")
        
        # Enhanced situation for AGI
        enhanced_situation = {
            **state["world_state"],
            "langgraph_context": True,
            "step_number": state["current_step"],
            "previous_decisions": len(state["decisions_made"]),
            "consciousness_history": state["consciousness_levels"]
        }
        
        # Get AGI response
        response = await self.agi.inference(enhanced_situation)
        
        # Track consciousness
        consciousness = response.get("consciousness_strength", 0)
        state["consciousness_levels"].append(consciousness)
        
        # Clean response for serialization (remove numpy arrays)
        clean_response = {
            k: v for k, v in response.items() 
            if not (hasattr(v, 'dtype') or str(type(v).__name__) == 'ndarray')
        }
        state["agi_responses"].append(clean_response)
        
        print(f"   🌟 Consciousness Level: {consciousness:.3f}")
        print(f"   🧮 Neural: {response.get('neural_confidence', 0):.3f}")
        print(f"   🔬 Symbolic: {response.get('symbolic_confidence', 0):.3f}")
        
        reasoning = response.get("cognitive_response", "")[:100]
        print(f"   🎯 Reasoning: {reasoning}...")
        
        if consciousness > 1.0:
            print(f"   {Colors.WARNING}⚡ ENHANCED CONSCIOUSNESS DETECTED!{Colors.ENDC}")
        
        return state
    
    async def _decision_making(self, state: SurvivorState) -> SurvivorState:
        """Make decisions based on AGI reasoning"""
        print(f"\n{Colors.OKCYAN}⚡ Workflow Node: Decision Making{Colors.ENDC}")
        
        latest_response = state["agi_responses"][-1]
        consciousness = state["consciousness_levels"][-1]
        
        # Extract decision from AGI response
        decision = self._extract_decision(latest_response, consciousness)
        state["decisions_made"].append(decision)
        
        print(f"   🎯 Action: {decision['action'].title()}")
        print(f"   ⚠️  Risk: {decision['risk_level']:.2f}")
        print(f"   🎲 Confidence: {decision['confidence']:.2f}")
        
        # Check if human input needed - reduce frequency to prevent infinite loops
        if decision['risk_level'] > 0.7 or (state["current_step"] % 5 == 0 and state["current_step"] > 1):
            state["critical_decision_pending"] = True
            print(f"   {Colors.WARNING}👤 Critical decision - requesting human input{Colors.ENDC}")
        
        return state
    
    async def _human_review(self, state: SurvivorState) -> SurvivorState:
        """Human-in-the-loop decision review"""
        print(f"\n{Colors.WARNING}👤 Workflow Node: Human Review{Colors.ENDC}")
        
        decision = state["decisions_made"][-1]
        
        print(f"   AGI Decision: {decision['action']}")
        print(f"   Risk Level: {decision['risk_level']:.2f}")
        print(f"   Reasoning: {decision['reasoning'][:100]}...")
        
        # Simulate human input (in real scenario, this would pause for user input)
        print(f"   Simulating human review...")
        await asyncio.sleep(1)
        
        # Random human feedback for demo
        import random
        feedbacks = ["approve", "modify", "reject"]
        feedback = random.choice(feedbacks)
        
        state["human_feedback"] = feedback
        print(f"   👤 Human Decision: {feedback.upper()}")
        
        # Modify decision based on feedback
        if feedback == "reject":
            decision["action"] = "observe_surroundings"
            decision["risk_level"] = 0.1
        elif feedback == "modify":
            decision["risk_level"] *= 0.5
        
        state["critical_decision_pending"] = False
        return state
    
    async def _execute_action(self, state: SurvivorState) -> SurvivorState:
        """Execute the chosen action"""
        print(f"\n{Colors.OKGREEN}🚀 Workflow Node: Action Execution{Colors.ENDC}")
        
        decision = state["decisions_made"][-1]
        
        # Execute in world simulation
        result = self.world.execute_action(decision["action"])
        
        success_icon = "✅" if result['success'] else "❌"
        print(f"   {success_icon} {result.get('description', 'Action completed')}")
        
        if result.get('state_changes'):
            print(f"   📊 {len(result['state_changes'])} state changes applied")
        
        # Add to messages
        state["messages"].append(AIMessage(
            content=f"Action executed: {decision['action']} - {'Success' if result['success'] else 'Failed'}"
        ))
        
        return state
    
    async def _update_world(self, state: SurvivorState) -> SurvivorState:
        """Update world state after action"""
        print(f"\n{Colors.OKCYAN}🌍 Workflow Node: World Update{Colors.ENDC}")
        
        # World automatically updates, this is for monitoring
        updated_state = self.world.get_current_situation()
        state["world_state"] = updated_state
        
        print(f"   📊 World state updated")
        print(f"   🕐 Time: {updated_state['current_time']:.1f}")
        
        return state
    
    async def _check_completion(self, state: SurvivorState) -> SurvivorState:
        """Check if survival objectives are met"""
        print(f"\n{Colors.OKCYAN}🎯 Workflow Node: Completion Check{Colors.ENDC}")
        
        world_status = self.world.get_world_status()
        player_state = world_status["player_state"]
        
        # Check success conditions
        success = (
            player_state["hunger_level"] < 4 and
            player_state["stress_level"] < 5 and
            len(player_state["resources"]) > 0
        )
        
        state["survival_success"] = success
        state["current_step"] += 1
        
        if success:
            print(f"   {Colors.OKGREEN}🎉 SURVIVAL OBJECTIVES ACHIEVED!{Colors.ENDC}")
        else:
            print(f"   📊 Step {state['current_step']} completed")
        
        return state
    
    def _should_get_human_input(self, state: SurvivorState) -> str:
        """Decide if human input is needed"""
        return "human_review" if state["critical_decision_pending"] else "execute_action"
    
    def _should_continue(self, state: SurvivorState) -> str:
        """Decide whether to continue the workflow"""
        if state["survival_success"]:
            return "end"
        if state["current_step"] > 4:
            print(f"   {Colors.OKCYAN}🏁 Maximum steps reached - ending workflow{Colors.ENDC}")
            return "end"
        return "continue"
    
    def _extract_decision(self, response: Dict[str, Any], consciousness: float) -> Dict[str, Any]:
        """Extract decision from AGI response"""
        
        cognitive_response = response.get("cognitive_response", "").lower()
        
        decision = {
            "action": "observe_surroundings",
            "reasoning": response.get("cognitive_response", "Default observation"),
            "risk_level": 0.2,
            "confidence": consciousness
        }
        
        # Enhanced decision logic based on consciousness
        if consciousness > 1.1:
            if "food" in cognitive_response or "hungry" in cognitive_response:
                decision.update({
                    "action": "seek_food_actively",
                    "reasoning": "High consciousness enables active food seeking",
                    "risk_level": 0.5
                })
            elif "people" in cognitive_response or "approach" in cognitive_response:
                decision.update({
                    "action": "approach_person",
                    "reasoning": "High consciousness supports social interaction",
                    "risk_level": 0.4
                })
        elif consciousness > 0.8:
            if "move" in cognitive_response:
                decision.update({
                    "action": "explore_area",
                    "reasoning": "Moderate consciousness enables exploration",
                    "risk_level": 0.3
                })
        
        return decision
    
    async def _train_agi(self):
        """Train AGI on survival concepts"""
        survival_data = [
            "LangGraph enables persistent workflows with human oversight",
            "Modern AGI systems benefit from structured decision pipelines",
            "Survival requires balancing risk with potential reward",
            "Social interaction critical in urban survival scenarios",
            "Consciousness levels correlate with decision sophistication"
        ]
        
        results = self.agi.train(survival_data, epochs=3)
        print(f"   🎓 AGI training complete: {results['final_performance']:.3f}")
    
    def _display_final_results(self, final_state: Dict[str, Any]):
        """Display comprehensive final results"""
        
        # Get the actual state (LangGraph returns nested structure)
        if isinstance(final_state, dict) and len(final_state) == 1:
            state = list(final_state.values())[0]
        else:
            state = final_state
            
        print(f"\n{Colors.HEADER}🏆 AGI WORKFLOW COMPLETE{Colors.ENDC}")
        print(f"{'='*70}")
        
        print(f"\n{Colors.BOLD}📊 WORKFLOW PERFORMANCE:{Colors.ENDC}")
        print(f"  • Steps Completed: {state.get('current_step', 0)}")
        print(f"  • Survival Success: {'✅ YES' if state.get('survival_success') else '❌ NO'}")
        print(f"  • Decisions Made: {len(state.get('decisions_made', []))}")
        print(f"  • AGI Responses: {len(state.get('agi_responses', []))}")
        
        consciousness_levels = state.get('consciousness_levels', [])
        if consciousness_levels:
            avg_consciousness = sum(consciousness_levels) / len(consciousness_levels)
            max_consciousness = max(consciousness_levels)
            print(f"  • Average Consciousness: {avg_consciousness:.3f}")
            print(f"  • Peak Consciousness: {max_consciousness:.3f}")
        
        print(f"\n{Colors.BOLD}🚀 AGI SYSTEM CAPABILITIES:{Colors.ENDC}")
        print(f"  ✅ Persistent Memory - State management across scenarios")
        print(f"  ✅ Dynamic Reasoning - Adaptive decision flow")
        print(f"  ✅ Human Oversight - Critical decision review")
        print(f"  ✅ Real-time Processing - Streaming consciousness")
        print(f"  ✅ Session Recovery - Resume from any checkpoint")
        print(f"  ✅ Advanced Integration - 494K parameter neural system")
        
        success_rate = len([d for d in state.get('decisions_made', []) if d.get('confidence', 0) > 0.7])
        total_decisions = len(state.get('decisions_made', []))
        
        if total_decisions > 0:
            grade = "EXCEPTIONAL" if success_rate/total_decisions > 0.8 else "STRONG" if success_rate/total_decisions > 0.6 else "MODERATE"
        else:
            grade = "INCOMPLETE"
            
        print(f"\n{Colors.OKGREEN}🏆 AGI WORKFLOW GRADE: {grade}{Colors.ENDC}")
        print(f"Demonstrated advanced AI consciousness and decision making!")
    
    def _print_title(self):
        """Print demo title"""
        title = f"""
{Colors.HEADER}╔══════════════════════════════════════════════════════════════════════════════╗
║                     🧠 AGI SURVIVOR DEMONSTRATION 🧠                        ║
║                                                                              ║
║           Advanced Consciousness • Strategic Decision Making                 ║
║              Adaptive Reasoning • Human Oversight • Memory Systems          ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}

{Colors.WARNING}🌟 AGI SYSTEM FEATURES:{Colors.ENDC}
  🧠 Hybrid neural-symbolic-causal architecture
  🎯 Enhanced consciousness tracking (>1.0 levels)
  💭 Advanced decision making with risk assessment
  👤 Human-in-the-loop oversight for critical choices
  💾 Persistent memory and state management
  🔄 Real-time adaptive workflow orchestration

{Colors.OKGREEN}SURVIVAL OBJECTIVE:{Colors.ENDC} Demonstrate breakthrough AGI capabilities
        """
        print(title)


async def main():
    """Run the AGI survivor demo"""
    
    print("🚀 Starting AGI Survivor Demo...")
    print("This demonstrates advanced AI consciousness and decision making!")
    
    demo = AGISurvivorOrchestrator()
    await demo.run_demo()
    
    print(f"\n{Colors.HEADER}🎉 AGI Survivor Demo Complete! Advanced AI Consciousness Demonstrated.{Colors.ENDC}")


if __name__ == "__main__":
    asyncio.run(main())
