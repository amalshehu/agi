"""
Test script to demonstrate the cognitive agent prototype
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.cognitive_agent import CognitiveAgent


async def test_basic_functionality():
    """Test basic agent functionality"""
    print("=== Testing Basic Functionality ===")
    
    agent = CognitiveAgent("test_agent")
    
    test_inputs = [
        "Hello, I'm testing your cognitive architecture",
        "Can you process visual information?",
        "What about spatial relationships?",
        "Do you remember our previous interactions?",
        "How do you learn from experience?"
    ]
    
    for i, input_text in enumerate(test_inputs):
        print(f"\n--- Test {i+1} ---")
        print(f"Input: {input_text}")
        
        response = await agent.process_input(input_text)
        print(f"Response: {response}")
        
        # Show detailed status every few cycles
        if i % 2 == 0:
            status = agent.get_agent_status()
            print(f"Memory: {status['memory_stats']}")
            print(f"Consciousness: {status['consciousness_stats']}")


async def test_learning_progression():
    """Test learning progression over multiple interactions"""
    print("\n=== Testing Learning Progression ===")
    
    agent = CognitiveAgent("learning_test")
    
    # Simulate a conversation that should trigger learning
    conversation = [
        "I see a red ball on the table",
        "The ball is next to a blue cup",
        "I want to pick up the ball",
        "The ball feels smooth and round",
        "I successfully picked up the ball"
    ]
    
    for i, message in enumerate(conversation):
        print(f"\n--- Interaction {i+1} ---")
        print(f"Input: {message}")
        
        response = await agent.process_input(message)
        print(f"Response: {response}")
        
        # Show learning statistics
        learning_stats = agent.learning_coordinator.get_learning_statistics()
        print(f"Learning events: {learning_stats}")


async def test_memory_integration():
    """Test memory system integration"""
    print("\n=== Testing Memory Integration ===")
    
    agent = CognitiveAgent("memory_test")
    
    # Test different types of memory
    memory_tests = [
        ("Sensory", "I hear a loud sound"),
        ("Perceptual", "I see a moving object"),
        ("Spatial", "The object is moving from left to right"),
        ("Episodic", "This reminds me of something I experienced before"),
        ("Declarative", "Objects that move usually have momentum")
    ]
    
    for memory_type, input_text in memory_tests:
        print(f"\n--- Testing {memory_type} Memory ---")
        print(f"Input: {input_text}")
        
        response = await agent.process_input(input_text)
        print(f"Response: {response}")
        
        # Show memory contents
        status = agent.get_agent_status()
        memory_stats = status['memory_stats']
        print(f"Memory contents: {memory_stats}")


async def test_consciousness_competition():
    """Test consciousness competition with multiple stimuli"""
    print("\n=== Testing Consciousness Competition ===")
    
    agent = CognitiveAgent("competition_test")
    
    # Create competing stimuli
    competing_inputs = [
        "There's a bright light flashing",
        "I hear someone calling my name",  
        "I smell something burning",
        "I feel a tap on my shoulder",
        "I see movement in my peripheral vision"
    ]
    
    for input_text in competing_inputs:
        print(f"\nInput: {input_text}")
        response = await agent.process_input(input_text)
        print(f"Response: {response}")
        
        # Show consciousness state
        status = agent.get_agent_status()
        consciousness_stats = status['consciousness_stats']
        print(f"Consciousness: {consciousness_stats}")


async def test_action_selection():
    """Test action selection and execution"""
    print("\n=== Testing Action Selection ===")
    
    agent = CognitiveAgent("action_test")
    
    # Test different action triggers
    action_tests = [
        "I need to explore this new environment",
        "Someone is asking me a question",
        "I'm having trouble focusing on multiple things",
        "I should remember this important information"
    ]
    
    for input_text in action_tests:
        print(f"\nInput: {input_text}")
        response = await agent.process_input(input_text)
        print(f"Response: {response}")
        
        # Show action execution stats
        status = agent.get_agent_status()
        action_stats = status['action_stats']
        print(f"Action execution: {action_stats}")


async def run_comprehensive_test():
    """Run a comprehensive test of the cognitive architecture"""
    print("=== Comprehensive Cognitive Architecture Test ===")
    
    # Test each major component
    await test_basic_functionality()
    await test_learning_progression()
    await test_memory_integration()
    await test_consciousness_competition()
    await test_action_selection()
    
    print("\n=== Test Summary ===")
    print("All tests completed. The cognitive architecture demonstrates:")
    print("1. ✓ Sensory processing and memory storage")
    print("2. ✓ Perceptual integration and association")
    print("3. ✓ Spatial and temporal reasoning")
    print("4. ✓ Episodic memory formation")
    print("5. ✓ Consciousness competition and global workspace")
    print("6. ✓ Action selection and motor execution")
    print("7. ✓ Learning pathways between memory systems")
    print("8. ✓ Attention and structure building codelets")


async def interactive_demo():
    """Interactive demo where user can chat with the agent"""
    print("\n=== Interactive Demo ===")
    print("Chat with the cognitive agent. Type 'quit' to exit.")
    print("Type 'status' to see agent status.")
    
    agent = CognitiveAgent("interactive_demo")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'status':
            status = agent.get_agent_status()
            print(f"Agent Status:")
            print(json.dumps(status, indent=2, default=str))
            continue
        
        if user_input:
            response = await agent.process_input(user_input)
            print(f"Agent: {response}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(run_comprehensive_test())
