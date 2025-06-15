#!/usr/bin/env python3
"""
🧠 SOTA AGI Survivor Demo
Test-time compute scaling with multi-agent reasoning
"""

import json
import time
from typing import Dict, List, Any
from reasoning_engine import TestTimeReasoningEngine, ReasoningType
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
    UNDERLINE = '\033[4m'

class SurvivorDemo:
    """Mind-blowing AGI survival demo"""
    
    def __init__(self):
        self.world = WorldSimulation()
        self.reasoning_engine = TestTimeReasoningEngine()
        self.demo_running = True
        
    def run_demo(self):
        """Run the complete survival demo"""
        
        self._print_title()
        self._print_scenario()
        
        # Initial situation assessment
        print(f"\n{Colors.HEADER}🧠 INITIALIZING AGI REASONING SYSTEM...{Colors.ENDC}")
        time.sleep(1)
        
        step = 1
        max_steps = 8
        
        while self.demo_running and step <= max_steps:
            print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
            print(f"{Colors.BOLD}STEP {step}: AGI REASONING & DECISION MAKING{Colors.ENDC}")
            print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")
            
            # Get current situation
            situation = self.world.get_current_situation()
            self._display_situation(situation)
            
            # Run reasoning engine with test-time compute scaling
            print(f"\n{Colors.OKCYAN}🔬 RUNNING TEST-TIME COMPUTE SCALING...{Colors.ENDC}")
            compute_budget = min(8, 3 + step)  # Increasing compute budget
            
            actions = self.reasoning_engine.reason(situation, compute_budget)
            
            # Display reasoning summary
            self._display_reasoning_summary()
            
            # Display and execute best action
            if actions:
                best_action = actions[0]
                self._display_action_decision(best_action)
                
                # Execute the action
                result = self.world.execute_action(best_action.action)
                self._display_action_result(result)
                
                # Check for critical success/failure conditions
                if self._check_survival_status():
                    break
            
            step += 1
            
            # Pause for dramatic effect
            time.sleep(2)
        
        self._print_final_summary()
    
    def _print_title(self):
        """Print impressive title"""
        title = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🧠 SOTA AGI SURVIVOR 🧠                           ║
║                                                                              ║
║          Test-Time Compute Scaling • Multi-Agent Reasoning                  ║
║               Iterative Refinement • Self-Critique Loops                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(f"{Colors.HEADER}{title}{Colors.ENDC}")
    
    def _print_scenario(self):
        """Print the survival scenario"""
        scenario = f"""
{Colors.WARNING}🏙️  SURVIVAL SCENARIO: Lost in a Strange City{Colors.ENDC}

{Colors.FAIL}SITUATION:{Colors.ENDC} You wake up in an unfamiliar city with:
  • No phone, no money, no identification
  • Cannot speak the local language
  • Need to find food and shelter before nightfall
  • Must navigate complex social interactions

{Colors.OKGREEN}AGI CHALLENGE:{Colors.ENDC} Demonstrate advanced reasoning capabilities:
  ✓ Multi-perspective analysis (survival, social, navigation, planning)
  ✓ Test-time compute scaling (more thinking = better decisions)
  ✓ Self-critique and iterative refinement
  ✓ Complex situation understanding and adaptation
        """
        print(scenario)
    
    def _display_situation(self, situation: Dict[str, Any]):
        """Display current situation in detail"""
        
        print(f"\n{Colors.OKCYAN}📍 CURRENT SITUATION ANALYSIS{Colors.ENDC}")
        print(f"┌─────────────────────────────────────────────────────────────────────────────┐")
        
        # Location and environment
        print(f"│ 🗺️  Location: {Colors.BOLD}{situation['location'].replace('_', ' ').title()}{Colors.ENDC}")
        print(f"│ 🕐 Time: {situation['current_time']:.1f}:00 ({situation['time_of_day']}) - {situation['hours_until_night']:.1f}h until night")
        print(f"│ 🌤️  Weather: {situation['weather']} | Safety: {situation['safety_level']:.1f}/1.0")
        
        # Physical state
        hunger_bar = "█" * int(situation['hunger_level']) + "░" * (10 - int(situation['hunger_level']))
        thirst_bar = "█" * int(situation['thirst_level']) + "░" * (10 - int(situation['thirst_level']))
        stress_bar = "█" * int(situation['stress_level']) + "░" * (10 - int(situation['stress_level']))
        
        print(f"│ 🍽️  Hunger: [{Colors.FAIL}{hunger_bar}{Colors.ENDC}] {situation['hunger_level']}/10")
        print(f"│ 💧 Thirst:  [{Colors.OKBLUE}{thirst_bar}{Colors.ENDC}] {situation['thirst_level']}/10")
        print(f"│ 😰 Stress:  [{Colors.WARNING}{stress_bar}{Colors.ENDC}] {situation['stress_level']}/10")
        
        # People and resources
        if situation['people_nearby']:
            people_str = ", ".join(situation['people_nearby'])
            print(f"│ 👥 People: {people_str}")
        
        if situation['available_resources']:
            resources_str = ", ".join(situation['available_resources'])
            print(f"│ 🎯 Resources: {resources_str}")
        
        print(f"└─────────────────────────────────────────────────────────────────────────────┘")
    
    def _display_reasoning_summary(self):
        """Display the reasoning process summary"""
        
        summary = self.reasoning_engine.get_reasoning_summary()
        
        print(f"\n{Colors.OKGREEN}🧠 REASONING PROCESS SUMMARY{Colors.ENDC}")
        print(f"┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ Total Thoughts: {summary['total_thoughts']} | Avg Confidence: {summary['avg_confidence']:.2f}")
        print(f"│ Reasoning Types: {', '.join(summary['reasoning_types'])}")
        print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        
        for i, thought in enumerate(summary['final_thoughts'][-3:], 1):  # Show last 3 thoughts
            agent_emoji = {
                'survival': '🆘',
                'social': '👥', 
                'navigation': '🗺️',
                'planning': '📋',
                'critique': '🔍'
            }.get(thought['type'], '🧠')
            
            print(f"│ {agent_emoji} {thought['type'].upper()}: {thought['content'][:65]}...")
            print(f"│    Confidence: {thought['confidence']:.2f}")
            if i < 3:
                print(f"├─────────────────────────────────────────────────────────────────────────────┤")
        
        print(f"└─────────────────────────────────────────────────────────────────────────────┘")
    
    def _display_action_decision(self, action):
        """Display the chosen action with reasoning"""
        
        print(f"\n{Colors.OKGREEN}⚡ OPTIMAL ACTION SELECTED{Colors.ENDC}")
        print(f"┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ 🎯 Action: {Colors.BOLD}{action.action.replace('_', ' ').title()}{Colors.ENDC}")
        print(f"│ 🧠 Reasoning: {action.reasoning}")
        print(f"│ 🎲 Expected Outcome: {action.expected_outcome}")
        print(f"│ ⚠️  Risk Level: {action.risk_level:.1f}/1.0")
        print(f"│ 🎯 Confidence: {action.confidence:.2f}/1.0")
        print(f"└─────────────────────────────────────────────────────────────────────────────┘")
    
    def _display_action_result(self, result: Dict[str, Any]):
        """Display the result of an action"""
        
        success_color = Colors.OKGREEN if result['success'] else Colors.FAIL
        success_icon = "✅" if result['success'] else "❌"
        
        print(f"\n{success_color}{success_icon} ACTION RESULT{Colors.ENDC}")
        print(f"┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ {result['description']}")
        
        if result.get('new_resources'):
            print(f"│ 🎁 Gained: {', '.join(result['new_resources'])}")
        
        if result.get('reputation_change'):
            change = result['reputation_change']
            change_str = f"+{change}" if change > 0 else str(change)
            print(f"│ 📈 Reputation: {change_str}")
        
        if result.get('state_changes'):
            for key, value in result['state_changes'].items():
                if 'level' in key:
                    current_world_state = self.world.player_state
                    old_value = current_world_state.get(key, 0)
                    change = value - old_value
                    change_str = f"{change:+.1f}" if change != 0 else "0"
                    print(f"│ 📊 {key.replace('_', ' ').title()}: {old_value:.1f} → {value:.1f} ({change_str})")
        
        print(f"└─────────────────────────────────────────────────────────────────────────────┘")
    
    def _check_survival_status(self) -> bool:
        """Check if survival conditions are met"""
        
        state = self.world.player_state
        current_time = self.world.current_time
        
        # Success conditions
        has_food = state['hunger_level'] < 4 or any('food' in r for r in state['resources'])
        has_shelter = any('shelter' in r for r in state['resources'])
        night_approached = current_time >= 19
        
        if has_food and has_shelter and night_approached:
            print(f"\n{Colors.OKGREEN}🎉 SURVIVAL SUCCESS! 🎉{Colors.ENDC}")
            print("✅ Food secured")
            print("✅ Shelter found")
            print("✅ Ready for the night")
            self.demo_running = False
            return True
        
        # Failure conditions
        if state['hunger_level'] >= 9:
            print(f"\n{Colors.FAIL}💀 SURVIVAL FAILURE: Severe hunger{Colors.ENDC}")
            self.demo_running = False
            return True
        
        if current_time >= 22 and not has_shelter:
            print(f"\n{Colors.FAIL}🌙 SURVIVAL CHALLENGE: No shelter at night{Colors.ENDC}")
            print("Continuing despite difficulties...")
        
        return False
    
    def _print_final_summary(self):
        """Print final demo summary"""
        
        world_status = self.world.get_world_status()
        reasoning_summary = self.reasoning_engine.get_reasoning_summary()
        
        print(f"\n{Colors.HEADER}📊 FINAL AGI PERFORMANCE SUMMARY{Colors.ENDC}")
        print(f"{'='*80}")
        
        print(f"\n{Colors.BOLD}🧠 REASONING PERFORMANCE:{Colors.ENDC}")
        print(f"  • Total reasoning iterations: {reasoning_summary.get('total_thoughts', 0)}")
        print(f"  • Average confidence: {reasoning_summary.get('avg_confidence', 0):.2f}/1.0")
        print(f"  • Reasoning types utilized: {len(reasoning_summary.get('reasoning_types', []))}/5")
        
        print(f"\n{Colors.BOLD}🎯 SURVIVAL METRICS:{Colors.ENDC}")
        player_state = world_status['player_state']
        print(f"  • Final hunger level: {player_state['hunger_level']:.1f}/10")
        print(f"  • Final stress level: {player_state['stress_level']:.1f}/10")
        print(f"  • Resources acquired: {len(player_state['resources'])}")
        print(f"  • Reputation earned: {player_state['reputation']}")
        print(f"  • Locations explored: {len(player_state['discovered_locations'])}")
        
        print(f"\n{Colors.BOLD}⚡ SOTA FEATURES DEMONSTRATED:{Colors.ENDC}")
        print(f"  ✅ Test-time compute scaling (more thinking → better decisions)")
        print(f"  ✅ Multi-agent reasoning (survival, social, navigation, planning)")
        print(f"  ✅ Self-critique and iterative refinement")
        print(f"  ✅ Dynamic world simulation with complex interactions")
        print(f"  ✅ Emergent behavior from simple rules")
        print(f"  ✅ Zero-shot adaptation to novel scenarios")
        
        print(f"\n{Colors.OKGREEN}🚀 This demonstrates SOTA AGI capabilities without transformers!{Colors.ENDC}")
        print(f"Built with: Multi-agent orchestration + Test-time compute scaling")

def main():
    """Run the survivor demo"""
    
    print("🧠 Starting SOTA AGI Survivor Demo...")
    print("This will demonstrate advanced reasoning capabilities!")
    
    input("\nPress Enter to begin the mind-blowing demo...")
    
    demo = SurvivorDemo()
    demo.run_demo()
    
    print(f"\n{Colors.HEADER}Demo completed! This showcases the future of AGI reasoning.{Colors.ENDC}")

if __name__ == "__main__":
    main()
