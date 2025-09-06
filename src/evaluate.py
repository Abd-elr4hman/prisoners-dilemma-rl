import torch
import numpy as np
from src.env import PrisonersDilemmaEnv
from src.policy import PolicyNN

def evaluate_policy(env:PrisonersDilemmaEnv, policy: PolicyNN, num_episodes=100):
    """Evaluate the trained policy"""
    total_rewards = []
    cooperation_rates = []
    
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            actions = []
            
            while True:
                # Use the policy deterministically (take most likely action)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
                
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                actions.append(action)
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            cooperation_rates.append((np.array(actions) == 0).mean())
    
    return total_rewards, cooperation_rates