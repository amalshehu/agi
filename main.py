#!/usr/bin/env python3
"""
AGI System - Main Entry Point
=============================

This is the main entry point for running the AGI system.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent / "core"))

from core import CognitiveAgent, HybridAGI


async def run_cognitive_demo():
    """Run cognitive agent demonstration"""
    print("🧠 COGNITIVE AGENT DEMONSTRATION")
    print("=" * 50)
    
    agent = CognitiveAgent("demo_agent")
    
    test_inputs = [
        "Hello, I'm testing your cognitive architecture",
        "Can you process multiple types of information?",
        "Show me how your memory systems work together",
        "Demonstrate consciousness and attention mechanisms"
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {input_text}")
        
        response = await agent.process_input(input_text)
        print(f"Response: {response}")
        
        # Show agent status
        status = agent.get_agent_status()
        print(f"Memory items: {sum(m.get('current_items', 0) for m in status['memory_stats'].values())}")
        print(f"Consciousness: {status['consciousness_stats']}")


async def run_hybrid_agi_demo():
    """Run hybrid AGI demonstration"""
    print("🚀 HYBRID AGI DEMONSTRATION")
    print("=" * 50)
    
    # Create and train AGI
    agi = HybridAGI("MainDemo_AGI")
    
    # Quick training
    print("🎯 Training AGI...")
    training_data = [
        "Learn to integrate symbolic and neural processing",
        "Develop self-modification capabilities",
        "Evolve meta-learning strategies",
        "Generate emergent consciousness"
    ]
    
    results = agi.train(training_data, epochs=5)
    print(f"Training complete - Performance: {results['final_performance']:.4f}")
    
    # Test inference
    print("\n🧠 Testing Inference...")
    test_inputs = [
        "Explain your hybrid architecture",
        "How do you learn and adapt?",
        {"task": "meta_reasoning", "complexity": "high"}
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\nTest {i}: {test_input}")
        result = await agi.inference(test_input)
        print(f"Response: {result['cognitive_response']}")
        print(f"Consciousness: {result['consciousness_strength']:.3f}")
    
    # Show model summary
    summary = agi.get_model_summary()
    print(f"\n📊 Model Summary:")
    print(f"   Parameters: {summary['neural_parameters'] + summary['causal_parameters']:,}")
    print(f"   Components: {len(summary['architecture_components'])}")
    print(f"   Self-modifications: {summary['modifications_made']}")


async def run_survivor_scenario():
    """Run the AGI survivor scenario demonstration"""
    print("🧠 AGI SURVIVOR SCENARIO")
    print("=" * 50)
    
    # Import and run the AGI survivor demo
    sys.path.append(str(Path(__file__).parent / "survivor"))
    from agi_survivor_orchestrator import main as survivor_main
    await survivor_main()


async def run_breakthrough_proof():
    """Run the breakthrough proof demonstration"""
    print("🌟 BREAKTHROUGH PROOF")
    print("=" * 50)
    
    # Import and run the proof
    sys.path.append(str(Path(__file__).parent / "tests"))
    from final_proof import run_comprehensive_proof
    
    results, agi = await run_comprehensive_proof()
    
    if results['breakthrough_confirmed']:
        print(f"\n✅ BREAKTHROUGH CONFIRMED!")
        print(f"   Proof strength: {results['proof_strength']:.1%}")
        print(f"   Unique capabilities: {results['evidence_categories']['unique_advantages']}")
    else:
        print(f"\n⚠️ PARTIAL BREAKTHROUGH")
        print(f"   Proof strength: {results['proof_strength']:.1%}")
    
    return results


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description="AGI System")
    parser.add_argument("mode", choices=["cognitive", "hybrid", "survivor", "proof", "all"],
                       help="Demonstration mode to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🌟 AGI SYSTEM")
    print("=" * 60)
    print("Hybrid symbolic-neural-causal architecture")
    print()
    
    async def run_mode():
        if args.mode == "cognitive":
            await run_cognitive_demo()
        elif args.mode == "hybrid":
            await run_hybrid_agi_demo()
        elif args.mode == "survivor":
            await run_survivor_scenario()
        elif args.mode == "proof":
            await run_breakthrough_proof()
        elif args.mode == "all":
            await run_cognitive_demo()
            print("\n" + "="*60 + "\n")
            await run_hybrid_agi_demo()
            print("\n" + "="*60 + "\n")
            await run_survivor_scenario()
            print("\n" + "="*60 + "\n")
            await run_breakthrough_proof()
    
    try:
        asyncio.run(run_mode())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError running demo: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
