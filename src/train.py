import torch
import numpy as np
from src.env import PrisonersDilemmaEnv

def select_action_stochastic(policy, state):
    """Select action using the policy network"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_probs = policy(state_tensor)
    
    # Sample action from the probability distribution
    action_dist = torch.distributions.Categorical(action_probs)
    action = action_dist.sample()
    
    # Store log probability for gradient computation
    log_prob = action_dist.log_prob(action)
    
    return action.item(), log_prob


def select_action_deterministic(policy, state):
    """Select action using the policy network"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_probs = policy(state_tensor)
    
    # Sample action from the probability distribution
    action_dist = torch.distributions.Categorical(action_probs)
    action = torch.argmax(action_probs, dim=1).item()
    
    # Store log probability for gradient computation
    log_prob = action_dist.log_prob(action)
    
    return action.item(), log_prob


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns for each time step"""
    returns = []
    R = 0
    
    # Work backwards from the end of the episode
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    
    # Convert to tensor and normalize (optional but often helps)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def train_reinforce(env:PrisonersDilemmaEnv, policy, optimizer:torch.optim.Optimizer, num_episodes=100, gamma=0.99, print_every=10):
    """Train the policy using REINFORCE algorithm"""
    
    episode_rewards = []
    episode_lengths = []
    cooperation_rates = []
    
    for episode in range(num_episodes):
        # Storage for this episode
        log_probs = []
        rewards = []
        actions_taken = []
        
        # Reset environment
        state, _ = env.reset()
        print(f"History {env.history}")
        episode_reward = 0
        
        # Run one episode
        while True:
            # Select action
            action, log_prob = select_action_stochastic(policy, state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store experience
            log_probs.append(log_prob)
            rewards.append(reward)
            actions_taken.append(action)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        
        # Compute policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)  # Negative because we want to maximize
        
        policy_loss = torch.stack(policy_loss).sum() # mine:  I think this should have a calculated mean somewhere, i need to calculate: -(logp * weighted_returns).mean()
        
        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Track statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(len(rewards))
        cooperation_rate = (np.array(actions_taken) == 0).mean()  # 0 = cooperate
        cooperation_rates.append(cooperation_rate)
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            avg_coop = np.mean(cooperation_rates[-print_every:])
            
            print(f"Episode {episode + 1}/{num_episodes}")
            
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Length: {avg_length:.1f}")
            print(f"  Cooperation Rate: {avg_coop:.3f}")
            print(f"  Policy Loss: {policy_loss.item():.4f}")
            print("-" * 40)
    
    return episode_rewards, cooperation_rates