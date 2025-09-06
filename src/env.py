import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from src.opponent import RandomOpponent, TitForTatOpponent, AlwaysDefectOpponent, AlwaysCooperateOpponent

class PrisonersDilemmaEnv(gym.Env):
    """
    Prisoner's Dilemma Environment for RL
    
    Actions: 0 = Cooperate, 1 = Defect
    Rewards: Standard payoff matrix
    """
    def __init__(self, opponent_strategy="random", max_steps=100, history_length=5):
        super().__init__()
        
        # Action space: 0 = Cooperate, 1 = Defect
        self.action_space = spaces.Discrete(2)

        # Observation space: history of last N moves for both players
        # Each entry is [my_action, opponent_action] for that round
        # -1 indicates no history available yet
        self.history_length = history_length
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(history_length * 2,), 
            dtype=np.float32
        )

        # Payoff matrix: [my_action][opponent_action] -> (my_reward, opp_reward)
        self.payoff_matrix = {
            (0, 0): (3, 3),  # Both cooperate
            (0, 1): (0, 5),  # I cooperate, opponent defects
            (1, 0): (5, 0),  # I defect, opponent cooperates  
            (1, 1): (1, 1),  # Both defect
        }

        # Environment settings
        self.max_steps = max_steps
        self.opponent_strategy = opponent_strategy

        # State tracking
        self.history = []  # List of (my_action, opp_action) tuples
        self.step_count = 0
        self.total_reward = 0

        # Initialize opponent
        self._init_opponent()


    def _init_opponent(self):
        """Initialize the opponent strategy"""
        if self.opponent_strategy == "random":
            self.opponent = RandomOpponent()
        elif self.opponent_strategy == "tit_for_tat":
            self.opponent = TitForTatOpponent()
        elif self.opponent_strategy == "always_defect":
            self.opponent = AlwaysDefectOpponent()
        elif self.opponent_strategy == "always_cooperate":
            self.opponent = AlwaysCooperateOpponent()
        else:
            raise ValueError(f"Unknown opponent strategy: {self.opponent_strategy}")

    def _get_observation(self) -> np.ndarray:
        """Convert history to observation vector"""
        obs = np.full(self.history_length * 2, -1, dtype=np.float32)
        
        # Fill in available history (most recent first)
        for i, (my_action, opp_action) in enumerate(self.history[-self.history_length:]):
            idx = i * 2
            obs[idx] = my_action      # My action
            obs[idx + 1] = opp_action # Opponent's action
            
        return obs
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.history = []
        self.step_count = 0
        self.total_reward = 0
        self.opponent.reset()
        
        observation = self._get_observation()
        info = {"step": self.step_count, "total_reward": self.total_reward}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.step_count >= self.max_steps:
            raise RuntimeError("Episode is done, call reset()")
        
        # Get opponent's action
        opp_action = self.opponent.get_action(self.history, action)
        
        # Calculate rewards
        my_reward, opp_reward = self.payoff_matrix[(action, opp_action)]
        
        # Update history
        self.history.append((action, opp_action))
        self.step_count += 1
        self.total_reward += my_reward
        
        # Check if episode is done
        terminated = self.step_count >= self.max_steps
        truncated = False
    
        # Get new observation
        observation = self._get_observation()
        
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "opponent_action": opp_action,
            "my_action": action,
            "round_reward": my_reward
        }
        
        return observation, my_reward, terminated, truncated, info
