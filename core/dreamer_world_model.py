"""
Dreamer-V3 World Model Integration
Implements latent dynamics modeling and imagination-based planning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import math

@dataclass
class DreamerConfig:
    """Configuration for Dreamer-V3 world model"""
    # Model dimensions
    embed_dim: int = 1024
    deter_dim: int = 1024
    stoch_dim: int = 32
    stoch_discrete: int = 32
    hidden_dim: int = 1024
    
    # Training parameters
    kl_scale: float = 1.0
    kl_balance: float = 0.8
    free_nats: float = 1.0
    
    # Planning parameters
    horizon: int = 15
    imagination_steps: int = 50
    
    # Experience replay
    sequence_length: int = 50
    batch_size: int = 16
    
class WorldModelState:
    """Represents the latent state of the world model"""
    
    def __init__(self, deter: torch.Tensor, stoch: torch.Tensor):
        self.deter = deter  # Deterministic state
        self.stoch = stoch  # Stochastic state
        
    def get_feature(self) -> torch.Tensor:
        """Get combined feature representation"""
        return torch.cat([self.deter, self.stoch], dim=-1)
    
    def detach(self) -> 'WorldModelState':
        """Detach gradients for planning"""
        return WorldModelState(self.deter.detach(), self.stoch.detach())

class RepresentationModel(nn.Module):
    """Encode observations into latent representations"""
    
    def __init__(self, obs_dim: int, embed_dim: int, stoch_dim: int, stoch_discrete: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, embed_dim)
        )
        
        # Stochastic state predictor
        self.stoch_predictor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ELU(),
            nn.Linear(512, stoch_dim * stoch_discrete)
        )
    
    def forward(self, obs: torch.Tensor, prev_state: WorldModelState) -> Tuple[WorldModelState, Dict]:
        """Encode observation into latent state"""
        # Encode observation
        embed = self.obs_encoder(obs)
        
        # Predict stochastic state
        stoch_logits = self.stoch_predictor(embed)
        stoch_logits = stoch_logits.view(-1, self.stoch_dim, self.stoch_discrete)
        
        # Sample stochastic state
        stoch_dist = dist.OneHotCategorical(logits=stoch_logits)
        stoch = stoch_dist.sample()
        stoch = stoch.view(-1, self.stoch_dim * self.stoch_discrete)
        
        # Use embed as deterministic state
        new_state = WorldModelState(embed, stoch)
        
        info = {
            "stoch_dist": stoch_dist,
            "embed": embed
        }
        
        return new_state, info

class DynamicsModel(nn.Module):
    """Predict next latent state from current state and action"""
    
    def __init__(self, deter_dim: int, stoch_dim: int, stoch_discrete: int, 
                 action_dim: int, hidden_dim: int):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.action_dim = action_dim
        
        # Recurrent deterministic state update
        self.rnn = nn.GRUCell(
            input_size=stoch_dim * stoch_discrete + action_dim,
            hidden_size=deter_dim
        )
        
        # Stochastic state predictor
        self.stoch_predictor = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_discrete)
        )
    
    def forward(self, prev_state: WorldModelState, action: torch.Tensor) -> Tuple[WorldModelState, Dict]:
        """Predict next state"""
        # Update deterministic state
        rnn_input = torch.cat([prev_state.stoch, action], dim=-1)
        deter = self.rnn(rnn_input, prev_state.deter)
        
        # Predict stochastic state
        stoch_logits = self.stoch_predictor(deter)
        stoch_logits = stoch_logits.view(-1, self.stoch_dim, self.stoch_discrete)
        
        # Sample stochastic state
        stoch_dist = dist.OneHotCategorical(logits=stoch_logits)
        stoch = stoch_dist.sample()
        stoch = stoch.view(-1, self.stoch_dim * self.stoch_discrete)
        
        next_state = WorldModelState(deter, stoch)
        
        info = {
            "stoch_dist": stoch_dist,
            "prior_stoch_dist": stoch_dist
        }
        
        return next_state, info

class RewardModel(nn.Module):
    """Predict reward from latent state"""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.reward_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: WorldModelState) -> torch.Tensor:
        """Predict reward"""
        features = state.get_feature()
        return self.reward_predictor(features)

class ContinueModel(nn.Module):
    """Predict episode continuation from latent state"""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.continue_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: WorldModelState) -> torch.Tensor:
        """Predict continuation probability"""
        features = state.get_feature()
        return torch.sigmoid(self.continue_predictor(features))

class ActorModel(nn.Module):
    """Policy model that outputs actions in latent space"""
    
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.action_dim = action_dim
        
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # For discrete actions
        self.action_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: WorldModelState, sample: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Predict action"""
        features = state.get_feature()
        
        # For survival environment, we use discrete actions
        action_logits = self.action_predictor(features)
        action_dist = dist.Categorical(logits=action_logits)
        
        if sample:
            action = action_dist.sample()
        else:
            action = action_logits.argmax(dim=-1)
        
        # Convert to one-hot for compatibility
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        
        info = {
            "action_dist": action_dist,
            "action_logits": action_logits
        }
        
        return action_onehot, info

class CriticModel(nn.Module):
    """Value function model for planning"""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: WorldModelState) -> torch.Tensor:
        """Predict state value"""
        features = state.get_feature()
        return self.critic(features)

class DreamerWorldModel(nn.Module):
    """Complete Dreamer-V3 world model"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Feature dimension (deterministic + stochastic)
        self.feature_dim = config.deter_dim + config.stoch_dim * config.stoch_discrete
        
        # World model components
        self.representation = RepresentationModel(
            obs_dim, config.deter_dim, config.stoch_dim, config.stoch_discrete
        )
        
        self.dynamics = DynamicsModel(
            config.deter_dim, config.stoch_dim, config.stoch_discrete,
            action_dim, config.hidden_dim
        )
        
        self.reward_model = RewardModel(self.feature_dim, config.hidden_dim)
        self.continue_model = ContinueModel(self.feature_dim, config.hidden_dim)
        
        # Planning components
        self.actor = ActorModel(self.feature_dim, action_dim, config.hidden_dim)
        self.critic = CriticModel(self.feature_dim, config.hidden_dim)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=100000)
        
    def encode_observation(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Convert observation dict to tensor"""
        # Flatten observation for the survival environment
        obs_list = []
        
        # Agent state
        obs_list.extend([
            obs["agent_health"] / 100.0,
            obs["agent_hunger"] / 100.0,
            obs["agent_thirst"] / 100.0,
            obs["agent_shelter"] / 100.0,
            len(obs["inventory"]) / 10.0,  # Normalized inventory size
            obs["lighting_level"],
            obs["weather_severity"],
            obs["step_count"] / 1000.0,  # Normalized time
        ])
        
        # Visible resources (up to 5 closest)
        resources = sorted(obs["visible_resources"], key=lambda x: x["distance"])[:5]
        for i in range(5):
            if i < len(resources):
                res = resources[i]
                obs_list.extend([
                    res["distance"] / 100.0,  # Normalized distance
                    res["value"] / 50.0,      # Normalized value
                    1.0 if res["type"] == "food" else 0.0,
                    1.0 if res["type"] == "water" else 0.0,
                    1.0 if res["type"] == "shelter" else 0.0,
                ])
            else:
                obs_list.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Visible NPCs (up to 3 closest)
        npcs = sorted(obs["visible_npcs"], key=lambda x: x["distance"])[:3]
        for i in range(3):
            if i < len(npcs):
                npc = npcs[i]
                obs_list.extend([
                    npc["distance"] / 100.0,
                    npc["hostility"],
                    1.0 if npc["behavior"] == "friendly" else 0.0,
                    1.0 if npc["behavior"] == "hostile" else 0.0,
                    1.0 if npc["behavior"] == "trader" else 0.0,
                ])
            else:
                obs_list.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Visible hazards (up to 3 closest)
        hazards = sorted(obs["visible_hazards"], key=lambda x: x["distance"])[:3]
        for i in range(3):
            if i < len(hazards):
                hazard = hazards[i]
                obs_list.extend([
                    hazard["distance"] / 100.0,
                    hazard["severity"],
                ])
            else:
                obs_list.extend([0.0, 0.0])
        
        # Pad or truncate to expected observation dimension
        if len(obs_list) < self.obs_dim:
            obs_list.extend([0.0] * (self.obs_dim - len(obs_list)))
        elif len(obs_list) > self.obs_dim:
            obs_list = obs_list[:self.obs_dim]
        
        return torch.FloatTensor(obs_list).unsqueeze(0)
    
    def init_state(self, batch_size: int = 1) -> WorldModelState:
        """Initialize world model state"""
        deter = torch.zeros(batch_size, self.config.deter_dim)
        stoch = torch.zeros(batch_size, self.config.stoch_dim * self.config.stoch_discrete)
        return WorldModelState(deter, stoch)
    
    def observe(self, obs: Dict[str, Any], prev_state: WorldModelState) -> Tuple[WorldModelState, Dict]:
        """Update state with new observation"""
        obs_tensor = self.encode_observation(obs)
        return self.representation(obs_tensor, prev_state)
    
    def imagine(self, init_state: WorldModelState, horizon: int) -> Dict[str, List[torch.Tensor]]:
        """Imagine future trajectories using the actor"""
        
        states = [init_state]
        actions = []
        rewards = []
        continues = []
        
        state = init_state
        
        for step in range(horizon):
            # Sample action from policy
            action, action_info = self.actor(state, sample=True)
            actions.append(action)
            
            # Predict next state
            next_state, dynamics_info = self.dynamics(state, action)
            states.append(next_state)
            
            # Predict reward and continuation
            reward = self.reward_model(state)
            continue_prob = self.continue_model(state)
            
            rewards.append(reward)
            continues.append(continue_prob)
            
            state = next_state
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "continues": continues
        }
    
    def plan_action(self, current_state: WorldModelState, 
                   survival_context: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        """Plan action using model-based imagination"""
        
        # Generate multiple imagination trajectories
        num_trajectories = 8
        trajectories = []
        
        for _ in range(num_trajectories):
            traj = self.imagine(current_state, self.config.horizon)
            trajectories.append(traj)
        
        # Evaluate trajectories
        trajectory_values = []
        for traj in trajectories:
            # Compute trajectory value
            rewards = torch.stack(traj["rewards"])
            continues = torch.stack(traj["continues"])
            
            # Simple value computation (can be enhanced with lambda-return)
            discounts = torch.cumprod(continues * 0.99, dim=0)
            trajectory_value = torch.sum(rewards * discounts)
            trajectory_values.append(trajectory_value)
        
        # Select best trajectory
        best_traj_idx = torch.argmax(torch.stack(trajectory_values))
        best_action = trajectories[best_traj_idx]["actions"][0]
        
        # Survival-specific planning adjustments
        planning_info = self._apply_survival_planning(
            current_state, best_action, survival_context
        )
        
        return best_action, planning_info
    
    def _apply_survival_planning(self, state: WorldModelState, action: torch.Tensor,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply survival-specific planning logic"""
        
        planning_info = {"base_action": action.clone()}
        
        # Emergency override logic
        agent_health = context.get("agent_health", 100)
        agent_hunger = context.get("agent_hunger", 0)
        agent_thirst = context.get("agent_thirst", 0)
        visible_hazards = context.get("visible_hazards", [])
        
        # Critical health - prioritize safety
        if agent_health < 30:
            # Bias toward cautious actions
            planning_info["health_critical"] = True
            planning_info["safety_priority"] = True
        
        # Critical needs - prioritize resources
        if agent_hunger > 80 or agent_thirst > 90:
            planning_info["resource_critical"] = True
            planning_info["resource_priority"] = "food" if agent_hunger > agent_thirst else "water"
        
        # Imminent hazard - prioritize escape
        close_hazards = [h for h in visible_hazards if h["distance"] < 5]
        if close_hazards:
            planning_info["hazard_detected"] = True
            planning_info["escape_priority"] = True
        
        return planning_info
    
    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute world model training losses"""
        
        obs_seq = batch["observations"]  # (batch, seq_len, obs_dim)
        action_seq = batch["actions"]    # (batch, seq_len, action_dim)
        reward_seq = batch["rewards"]    # (batch, seq_len, 1)
        done_seq = batch["dones"]        # (batch, seq_len, 1)
        
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Initialize state
        state = self.init_state(batch_size)
        
        # Forward pass through sequence
        losses = defaultdict(list)
        
        for t in range(seq_len):
            obs = obs_seq[:, t]
            action = action_seq[:, t]
            reward = reward_seq[:, t]
            done = done_seq[:, t]
            
            # Representation loss (posterior vs prior)
            posterior_state, repr_info = self.representation(obs, state)
            prior_state, dyn_info = self.dynamics(state, action)
            
            # KL divergence loss
            posterior_dist = repr_info["stoch_dist"]
            prior_dist = dyn_info["prior_stoch_dist"]
            
            kl_loss = dist.kl_divergence(posterior_dist, prior_dist).sum(-1)
            kl_loss = torch.maximum(kl_loss, torch.full_like(kl_loss, self.config.free_nats))
            losses["kl"].append(kl_loss)
            
            # Reward prediction loss
            pred_reward = self.reward_model(posterior_state)
            reward_loss = F.mse_loss(pred_reward, reward, reduction='none')
            losses["reward"].append(reward_loss)
            
            # Continue prediction loss
            pred_continue = self.continue_model(posterior_state) 
            continue_target = 1.0 - done.float()
            continue_loss = F.binary_cross_entropy(
                pred_continue, continue_target, reduction='none'
            )
            losses["continue"].append(continue_loss)
            
            # Update state for next timestep
            state = posterior_state
        
        # Aggregate losses
        total_losses = {}
        for loss_name, loss_list in losses.items():
            stacked_loss = torch.stack(loss_list, dim=1)  # (batch, seq_len)
            total_losses[loss_name] = stacked_loss.mean()
        
        # Combine losses
        total_losses["world_model"] = (
            total_losses["kl"] * self.config.kl_scale +
            total_losses["reward"] +
            total_losses["continue"]
        )
        
        return total_losses
    
    def add_experience(self, obs: Dict[str, Any], action: torch.Tensor, 
                      reward: float, next_obs: Dict[str, Any], done: bool):
        """Add experience to replay buffer"""
        
        experience = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done
        }
        
        self.experience_buffer.append(experience)
    
    def sample_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Sample training batch from experience buffer"""
        
        if len(self.experience_buffer) < self.config.sequence_length * self.config.batch_size:
            return None
        
        # Sample sequences
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for _ in range(self.config.batch_size):
            # Sample random starting point
            start_idx = np.random.randint(
                0, len(self.experience_buffer) - self.config.sequence_length
            )
            
            # Extract sequence
            seq_obs = []
            seq_actions = []
            seq_rewards = []
            seq_dones = []
            
            for i in range(self.config.sequence_length):
                exp = self.experience_buffer[start_idx + i]
                seq_obs.append(self.encode_observation(exp["obs"]))
                seq_actions.append(exp["action"])
                seq_rewards.append(torch.FloatTensor([exp["reward"]]))
                seq_dones.append(torch.FloatTensor([float(exp["done"])]))
            
            batch_obs.append(torch.stack(seq_obs))
            batch_actions.append(torch.stack(seq_actions))
            batch_rewards.append(torch.stack(seq_rewards))
            batch_dones.append(torch.stack(seq_dones))
        
        return {
            "observations": torch.stack(batch_obs),
            "actions": torch.stack(batch_actions),
            "rewards": torch.stack(batch_rewards),
            "dones": torch.stack(batch_dones)
        }
