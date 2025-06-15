"""
Evaluation Suite for Global Workspace AGI Research
Implements comprehensive testing, baselines, and statistical analysis
"""

import numpy as np
import torch
import torch.nn as nn
import random
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import asyncio

from .survival_simulation import SurvivalEnvironment, SimulationConfig, ScenarioGenerator
from .survival_agi import SurvivalAGI
from .hybrid_agi import HybridAGI

@dataclass
class ExperimentConfig:
    """Configuration for experimental runs"""
    num_episodes: int = 50
    max_steps_per_episode: int = 1000
    num_trials: int = 5
    scenarios: List[str] = None
    ablation_tests: List[str] = None
    
    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = ["easy", "medium", "hard"]
        if self.ablation_tests is None:
            self.ablation_tests = [
                "no_workspace", "no_consciousness", "no_world_model", 
                "no_self_modification", "no_meta_learning"
            ]

class BaselineAgent:
    """Base class for baseline comparison agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.episode_metrics = []
    
    async def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select action given observation"""
        raise NotImplementedError
    
    async def run_episode(self, env: SurvivalEnvironment, max_steps: int = 1000) -> Dict[str, Any]:
        """Run a complete episode"""
        observation = env.reset()
        total_reward = 0.0
        step = 0
        done = False
        
        while not done and step < max_steps:
            action = await self.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
        
        metrics = {
            "episode_length": step,
            "total_reward": total_reward,
            "survival_time": info["metrics"]["survival_time"],
            "resources_found": info["metrics"]["resources_found"],
            "hazards_encountered": info["metrics"]["hazards_encountered"],
            "final_agent_state": info["survival_status"]
        }
        
        self.episode_metrics.append(metrics)
        return metrics

class RandomAgent(BaselineAgent):
    """Random action baseline"""
    
    def __init__(self):
        super().__init__("Random")
        self.action_types = ["move", "collect", "interact", "wait"]
        self.move_directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
    
    async def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Random action selection"""
        action_type = random.choice(self.action_types)
        
        if action_type == "move":
            return {
                "type": "move",
                "direction": random.choice(self.move_directions)
            }
        elif action_type == "collect":
            # Random collection attempt
            map_size = 100  # Assume default map size
            return {
                "type": "collect",
                "target_x": random.randint(0, map_size-1),
                "target_y": random.randint(0, map_size-1)
            }
        elif action_type == "interact":
            # Random interaction attempt
            npcs = observation.get("visible_npcs", [])
            if npcs:
                target_npc = random.choice(npcs)
                return {
                    "type": "interact",
                    "target_id": target_npc["id"]
                }
        
        return {"type": "wait"}

class GreedyAgent(BaselineAgent):
    """Greedy heuristic baseline"""
    
    def __init__(self):
        super().__init__("Greedy")
    
    async def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Greedy action selection based on immediate needs"""
        agent_health = observation.get("agent_health", 100)
        agent_hunger = observation.get("agent_hunger", 0)
        agent_thirst = observation.get("agent_thirst", 0)
        visible_resources = observation.get("visible_resources", [])
        visible_hazards = observation.get("visible_hazards", [])
        
        # Emergency: avoid immediate hazards
        close_hazards = [h for h in visible_hazards if h["distance"] < 5]
        if close_hazards and agent_health < 60:
            # Move away from closest hazard
            closest_hazard = min(close_hazards, key=lambda h: h["distance"])
            agent_pos = observation.get("agent_position", (0, 0))
            hazard_pos = closest_hazard["position"]
            
            dx = agent_pos[0] - hazard_pos[0]
            dy = agent_pos[1] - hazard_pos[1]
            
            if dx != 0 or dy != 0:
                length = np.sqrt(dx*dx + dy*dy)
                dx = int(np.sign(dx / length))
                dy = int(np.sign(dy / length))
            
            return {"type": "move", "direction": (dx, dy)}
        
        # Priority 1: Critical thirst
        if agent_thirst > 85:
            water_resources = [r for r in visible_resources if r["type"] == "water"]
            if water_resources:
                closest_water = min(water_resources, key=lambda r: r["distance"])
                return {
                    "type": "collect",
                    "target_x": closest_water["position"][0],
                    "target_y": closest_water["position"][1]
                }
        
        # Priority 2: Critical hunger
        if agent_hunger > 80:
            food_resources = [r for r in visible_resources if r["type"] == "food"]
            if food_resources:
                closest_food = min(food_resources, key=lambda r: r["distance"])
                return {
                    "type": "collect",
                    "target_x": closest_food["position"][0],
                    "target_y": closest_food["position"][1]
                }
        
        # Priority 3: Any visible resource
        if visible_resources:
            closest_resource = min(visible_resources, key=lambda r: r["distance"])
            return {
                "type": "collect",
                "target_x": closest_resource["position"][0],
                "target_y": closest_resource["position"][1]
            }
        
        # Priority 4: Beneficial NPC interaction
        npcs = observation.get("visible_npcs", [])
        friendly_npcs = [npc for npc in npcs if npc["behavior"] == "friendly" and npc["distance"] < 8]
        if friendly_npcs:
            closest_friendly = min(friendly_npcs, key=lambda npc: npc["distance"])
            return {
                "type": "interact",
                "target_id": closest_friendly["id"]
            }
        
        # Default: Random exploration
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return {"type": "move", "direction": random.choice(directions)}

class SimpleDreamerAgent(BaselineAgent):
    """Simplified Dreamer-V3 baseline without global workspace"""
    
    def __init__(self):
        super().__init__("Simple Dreamer")
        from .dreamer_world_model import DreamerWorldModel, DreamerConfig
        
        config = DreamerConfig(embed_dim=256, deter_dim=256, horizon=5)
        self.world_model = DreamerWorldModel(64, 8, config)
        self.world_model_state = self.world_model.init_state()
        
        self.action_mapping = {
            0: {"type": "move", "direction": (-1, -1)},
            1: {"type": "move", "direction": (-1, 0)},
            2: {"type": "move", "direction": (-1, 1)},
            3: {"type": "move", "direction": (0, -1)},
            4: {"type": "move", "direction": (0, 1)},
            5: {"type": "move", "direction": (1, -1)},
            6: {"type": "move", "direction": (1, 0)},
            7: {"type": "move", "direction": (1, 1)},
        }
    
    async def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Action selection using world model planning only"""
        
        # Update world model state
        self.world_model_state, _ = self.world_model.observe(
            observation, self.world_model_state
        )
        
        # Plan action
        planned_action, _ = self.world_model.plan_action(
            self.world_model_state, observation
        )
        
        # Convert to environment action
        action_idx = torch.argmax(planned_action).item()
        return self.action_mapping.get(action_idx, {"type": "wait"})

class AblatedSurvivalAGI(SurvivalAGI):
    """Ablated version of SurvivalAGI for controlled experiments"""
    
    def __init__(self, ablation_type: str, model_name: str = "ablated_agi"):
        super().__init__(model_name)
        self.ablation_type = ablation_type
        self._apply_ablation()
    
    def _apply_ablation(self):
        """Apply specific ablation to the model"""
        
        if self.ablation_type == "no_workspace":
            # Disable global workspace - use only first codelet
            original_codelets = self.survival_modules.attention_codelets
            self.survival_modules.attention_codelets = {
                "resource_detection": original_codelets["resource_detection"]
            }
        
        elif self.ablation_type == "no_consciousness":
            # Disable consciousness computation - always return low consciousness
            self._original_compute_consciousness = self._compute_consciousness_strength
            self._compute_consciousness_strength = lambda *args: 0.1
        
        elif self.ablation_type == "no_world_model":
            # Disable world model planning
            self.world_model = None
        
        elif self.ablation_type == "no_self_modification":
            # Disable self-modification in the core AGI
            if hasattr(self.agi_core, 'self_modifier'):
                self.agi_core.self_modifier = None
        
        elif self.ablation_type == "no_meta_learning":
            # Disable meta-learning in the core AGI
            if hasattr(self.agi_core, 'meta_learner'):
                self.agi_core.meta_learner = None

class ExperimentRunner:
    """Manages and executes comprehensive experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = defaultdict(list)
        self.baselines = {
            "random": RandomAgent(),
            "greedy": GreedyAgent(),
            "simple_dreamer": SimpleDreamerAgent()
        }
        
    async def run_full_experiment(self) -> Dict[str, Any]:
        """Run complete experimental suite"""
        
        print("🧪 Starting comprehensive AGI evaluation experiment")
        
        # 1. Baseline comparisons
        print("\n📊 Running baseline comparisons...")
        baseline_results = await self._run_baseline_comparisons()
        
        # 2. Main AGI evaluation
        print("\n🧠 Running main AGI evaluation...")
        agi_results = await self._run_agi_evaluation()
        
        # 3. Ablation studies
        print("\n⚗️ Running ablation studies...")
        ablation_results = await self._run_ablation_studies()
        
        # 4. Statistical analysis
        print("\n📈 Running statistical analysis...")
        statistical_results = self._run_statistical_analysis(
            baseline_results, agi_results, ablation_results
        )
        
        # 5. Generate visualizations
        print("\n📊 Generating visualizations...")
        self._generate_visualizations(
            baseline_results, agi_results, ablation_results
        )
        
        final_results = {
            "experiment_config": asdict(self.config),
            "baseline_results": baseline_results,
            "agi_results": agi_results,
            "ablation_results": ablation_results,
            "statistical_analysis": statistical_results,
            "timestamp": time.time()
        }
        
        # Save results
        self._save_results(final_results)
        
        print("✅ Comprehensive experiment completed!")
        return final_results
    
    async def _run_baseline_comparisons(self) -> Dict[str, Any]:
        """Run baseline agent comparisons"""
        
        baseline_results = {}
        
        for scenario_name in self.config.scenarios:
            print(f"  🎯 Testing scenario: {scenario_name}")
            scenario_config = getattr(ScenarioGenerator, f"generate_{scenario_name}_scenario")()
            
            scenario_results = {}
            
            for baseline_name, baseline_agent in self.baselines.items():
                print(f"    🤖 Testing {baseline_name} agent...")
                
                agent_results = []
                for trial in range(self.config.num_trials):
                    trial_results = []
                    
                    for episode in range(self.config.num_episodes):
                        env = SurvivalEnvironment(scenario_config)
                        result = await baseline_agent.run_episode(
                            env, self.config.max_steps_per_episode
                        )
                        trial_results.append(result)
                    
                    agent_results.append(trial_results)
                
                scenario_results[baseline_name] = agent_results
            
            baseline_results[scenario_name] = scenario_results
        
        return baseline_results
    
    async def _run_agi_evaluation(self) -> Dict[str, Any]:
        """Run main AGI evaluation"""
        
        agi_results = {}
        
        for scenario_name in self.config.scenarios:
            print(f"  🧠 Testing AGI on scenario: {scenario_name}")
            scenario_config = getattr(ScenarioGenerator, f"generate_{scenario_name}_scenario")()
            
            scenario_results = []
            
            for trial in range(self.config.num_trials):
                # Create fresh AGI instance for each trial
                agi = SurvivalAGI(f"agi_trial_{trial}")
                
                trial_results = []
                consciousness_events = []
                
                for episode in range(self.config.num_episodes):
                    env = SurvivalEnvironment(scenario_config)
                    result = await agi.run_survival_episode(
                        env, self.config.max_steps_per_episode
                    )
                    
                    trial_results.append(result["metrics"])
                    consciousness_events.extend(result["consciousness_events"])
                
                # Train world model after each trial
                agi.train_world_model(100)
                
                scenario_results.append({
                    "episodes": trial_results,
                    "consciousness_events": consciousness_events,
                    "research_summary": agi.get_research_summary()
                })
            
            agi_results[scenario_name] = scenario_results
        
        return agi_results
    
    async def _run_ablation_studies(self) -> Dict[str, Any]:
        """Run ablation studies"""
        
        ablation_results = {}
        
        for ablation_type in self.config.ablation_tests:
            print(f"  ⚗️ Testing ablation: {ablation_type}")
            
            ablation_results[ablation_type] = {}
            
            for scenario_name in self.config.scenarios:
                scenario_config = getattr(ScenarioGenerator, f"generate_{scenario_name}_scenario")()
                
                scenario_results = []
                
                for trial in range(min(3, self.config.num_trials)):  # Fewer trials for ablations
                    # Create ablated AGI
                    ablated_agi = AblatedSurvivalAGI(ablation_type, f"ablated_{trial}")
                    
                    trial_results = []
                    
                    for episode in range(min(20, self.config.num_episodes)):  # Fewer episodes
                        env = SurvivalEnvironment(scenario_config)
                        result = await ablated_agi.run_survival_episode(
                            env, self.config.max_steps_per_episode
                        )
                        trial_results.append(result["metrics"])
                    
                    scenario_results.append(trial_results)
                
                ablation_results[ablation_type][scenario_name] = scenario_results
        
        return ablation_results
    
    def _run_statistical_analysis(self, baseline_results: Dict, 
                                agi_results: Dict, ablation_results: Dict) -> Dict[str, Any]:
        """Perform statistical analysis on results"""
        
        statistical_results = {}
        
        # Extract performance metrics
        def extract_metric(results, metric_name):
            values = []
            for scenario in results.values():
                if isinstance(scenario, dict):
                    for agent_results in scenario.values():
                        for trial in agent_results:
                            for episode in trial:
                                if isinstance(episode, dict) and metric_name in episode:
                                    values.append(episode[metric_name])
                elif isinstance(scenario, list):
                    for trial in scenario:
                        episodes = trial.get("episodes", trial)
                        for episode in episodes:
                            if isinstance(episode, dict) and metric_name in episode:
                                values.append(episode[metric_name])
            return values
        
        # Compare AGI vs baselines
        agi_survival_times = extract_metric(agi_results, "survival_time")
        baseline_survival_times = {}
        
        for scenario in baseline_results.values():
            for baseline_name, results in scenario.items():
                if baseline_name not in baseline_survival_times:
                    baseline_survival_times[baseline_name] = []
                baseline_survival_times[baseline_name].extend(
                    extract_metric({baseline_name: results}, "survival_time")
                )
        
        # Statistical tests
        comparisons = {}
        for baseline_name, baseline_times in baseline_survival_times.items():
            if len(agi_survival_times) > 0 and len(baseline_times) > 0:
                t_stat, p_value = stats.ttest_ind(agi_survival_times, baseline_times)
                effect_size = (np.mean(agi_survival_times) - np.mean(baseline_times)) / \
                            np.sqrt((np.std(agi_survival_times)**2 + np.std(baseline_times)**2) / 2)
                
                comparisons[f"agi_vs_{baseline_name}"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "effect_size": float(effect_size),
                    "agi_mean": float(np.mean(agi_survival_times)),
                    "baseline_mean": float(np.mean(baseline_times)),
                    "significant": p_value < 0.05
                }
        
        # Ablation analysis
        ablation_comparisons = {}
        for ablation_type, ablation_data in ablation_results.items():
            ablation_times = extract_metric(ablation_data, "survival_time")
            if len(agi_survival_times) > 0 and len(ablation_times) > 0:
                t_stat, p_value = stats.ttest_ind(agi_survival_times, ablation_times)
                effect_size = (np.mean(agi_survival_times) - np.mean(ablation_times)) / \
                            np.sqrt((np.std(agi_survival_times)**2 + np.std(ablation_times)**2) / 2)
                
                ablation_comparisons[ablation_type] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "effect_size": float(effect_size),
                    "full_agi_mean": float(np.mean(agi_survival_times)),
                    "ablated_mean": float(np.mean(ablation_times)),
                    "significant": p_value < 0.05,
                    "performance_drop": float(np.mean(agi_survival_times) - np.mean(ablation_times))
                }
        
        statistical_results = {
            "baseline_comparisons": comparisons,
            "ablation_comparisons": ablation_comparisons,
            "sample_sizes": {
                "agi": len(agi_survival_times),
                "baselines": {name: len(times) for name, times in baseline_survival_times.items()}
            }
        }
        
        return statistical_results
    
    def _generate_visualizations(self, baseline_results: Dict, 
                               agi_results: Dict, ablation_results: Dict):
        """Generate research visualizations"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Global Workspace AGI Survival Experiment Results', fontsize=16)
        
        # 1. Performance comparison
        ax1 = axes[0, 0]
        
        # Extract survival times for plotting
        def extract_survival_times(results):
            times = []
            for scenario in results.values():
                if isinstance(scenario, dict):
                    for agent_results in scenario.values():
                        for trial in agent_results:
                            for episode in trial:
                                if isinstance(episode, dict) and "survival_time" in episode:
                                    times.append(episode["survival_time"])
                elif isinstance(scenario, list):
                    for trial in scenario:
                        episodes = trial.get("episodes", trial)
                        for episode in episodes:
                            if isinstance(episode, dict) and "survival_time" in episode:
                                times.append(episode["survival_time"])
            return times
        
        plot_data = []
        plot_labels = []
        
        # Add baseline data
        for scenario in baseline_results.values():
            for baseline_name, results in scenario.items():
                times = []
                for trial in results:
                    for episode in trial:
                        if "survival_time" in episode:
                            times.append(episode["survival_time"])
                if times:
                    plot_data.append(times)
                    plot_labels.append(baseline_name)
        
        # Add AGI data
        agi_times = extract_survival_times(agi_results)
        if agi_times:
            plot_data.append(agi_times)
            plot_labels.append("Global Workspace AGI")
        
        if plot_data:
            ax1.boxplot(plot_data, labels=plot_labels)
            ax1.set_title('Survival Time Comparison')
            ax1.set_ylabel('Survival Time (steps)')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Consciousness correlation (if AGI data available)
        ax2 = axes[0, 1]
        if agi_results:
            consciousness_scores = []
            performance_scores = []
            
            for scenario in agi_results.values():
                for trial in scenario:
                    episodes = trial.get("episodes", [])
                    for episode in episodes:
                        if ("average_consciousness" in episode and 
                            "total_reward" in episode):
                            consciousness_scores.append(episode["average_consciousness"])
                            performance_scores.append(episode["total_reward"])
            
            if consciousness_scores and performance_scores:
                ax2.scatter(consciousness_scores, performance_scores, alpha=0.6)
                ax2.set_xlabel('Average Consciousness Strength')
                ax2.set_ylabel('Episode Reward')
                ax2.set_title('Consciousness vs Performance')
                
                # Add correlation line
                if len(consciousness_scores) > 1:
                    z = np.polyfit(consciousness_scores, performance_scores, 1)
                    p = np.poly1d(z)
                    ax2.plot(consciousness_scores, p(consciousness_scores), "r--", alpha=0.8)
        
        # 3. Ablation study results
        ax3 = axes[1, 0]
        if ablation_results:
            ablation_means = []
            ablation_names = []
            
            for ablation_type, ablation_data in ablation_results.items():
                times = extract_survival_times(ablation_data)
                if times:
                    ablation_means.append(np.mean(times))
                    ablation_names.append(ablation_type.replace("no_", ""))
            
            if agi_times:
                ablation_means.append(np.mean(agi_times))
                ablation_names.append("Full AGI")
            
            if ablation_means:
                bars = ax3.bar(ablation_names, ablation_means)
                ax3.set_title('Ablation Study Results')
                ax3.set_ylabel('Mean Survival Time')
                ax3.tick_params(axis='x', rotation=45)
                
                # Highlight full AGI
                if len(bars) > 0:
                    bars[-1].set_color('red')
        
        # 4. Learning curves (if available)
        ax4 = axes[1, 1]
        ax4.set_title('Learning Progress')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Survival Time')
        
        # Plot learning curves for each agent type
        colors = ['blue', 'green', 'orange', 'red']
        
        # AGI learning curve
        if agi_results:
            episode_means = []
            for scenario in agi_results.values():
                for trial in scenario:
                    episodes = trial.get("episodes", [])
                    if episodes:
                        trial_times = [ep.get("survival_time", 0) for ep in episodes]
                        if episode_means:
                            episode_means = [(a + b) / 2 for a, b in zip(episode_means, trial_times)]
                        else:
                            episode_means = trial_times
            
            if episode_means:
                ax4.plot(episode_means, label="Global Workspace AGI", color='red', linewidth=2)
        
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('results/agi_experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualizations saved to results/agi_experiment_results.png")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experimental results"""
        
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save full results
        timestamp = int(time.time())
        results_file = results_dir / f"agi_experiment_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            "experiment_timestamp": timestamp,
            "total_episodes_run": sum(
                len(scenario) * len(trial) 
                for scenario in results["baseline_results"].values()
                for trial in scenario.values()
            ),
            "statistical_significance": {
                name: comp["significant"] 
                for name, comp in results["statistical_analysis"]["baseline_comparisons"].items()
            },
            "key_findings": self._extract_key_findings(results)
        }
        
        summary_file = results_dir / f"agi_experiment_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_file}")
        print(f"💾 Summary saved to {summary_file}")
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key experimental findings"""
        
        findings = []
        
        # Check statistical significance
        baseline_comps = results["statistical_analysis"]["baseline_comparisons"]
        significant_improvements = [
            name for name, comp in baseline_comps.items() 
            if comp["significant"] and comp["agi_mean"] > comp["baseline_mean"]
        ]
        
        if significant_improvements:
            findings.append(f"AGI significantly outperformed {len(significant_improvements)} baseline(s)")
        
        # Check ablation results
        ablation_comps = results["statistical_analysis"]["ablation_comparisons"]
        important_modules = [
            ablation_type for ablation_type, comp in ablation_comps.items()
            if comp["significant"] and comp["performance_drop"] > 50  # Significant drop
        ]
        
        if important_modules:
            findings.append(f"Critical modules identified: {', '.join(important_modules)}")
        
        # Check consciousness correlation
        # This would require analyzing the AGI results data structure
        
        return findings

# Example usage function
async def run_comprehensive_evaluation():
    """Run the comprehensive evaluation suite"""
    
    config = ExperimentConfig(
        num_episodes=30,
        max_steps_per_episode=800,
        num_trials=3,
        scenarios=["easy", "medium", "hard"],
        ablation_tests=["no_workspace", "no_consciousness", "no_world_model"]
    )
    
    runner = ExperimentRunner(config)
    results = await runner.run_full_experiment()
    
    return results

if __name__ == "__main__":
    # Run evaluation if script is executed directly
    asyncio.run(run_comprehensive_evaluation())
