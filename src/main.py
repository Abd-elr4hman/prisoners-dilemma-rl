import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from src.env import PrisonersDilemmaEnv
from src.policy import PolicyNN
from src.train import train_reinforce
from src.evaluate import evaluate_policy


def plot_training_results(episode_rewards, cooperation_rates):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot cooperation rate
    ax2.plot(cooperation_rates)
    ax2.set_title('Cooperation Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Fraction of Cooperative Actions')
    ax2.grid(True)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

    
def main():
    # Create environment and policy
    env = PrisonersDilemmaEnv(
        opponent_strategy="tit_for_tat", 
        max_steps=50, 
        history_length=5
    )

    policy = PolicyNN(input_size=env.history_length*2, hidden_size=32)  # 5 history * 2 = 10 inputs
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    print("Starting training...")
    print(f"Environment: {env.opponent_strategy} opponent")
    print(f"Max steps per episode: {env.max_steps}")
    print(f"Network architecture: {policy}")
    print("=" * 50)

    # Train the policy
    episode_rewards, cooperation_rates = train_reinforce(
        env, policy, optimizer, 
        num_episodes=2000, 
        gamma=0.95,
        print_every=100
    )

    print("\nTraining completed!")

    # Evaluate the trained policy
    print("\nEvaluating trained policy...")
    eval_rewards, eval_cooperation = evaluate_policy(env, policy, num_episodes=100)
    
    print("Evaluation Results:")
    print(f"  Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"  Average Cooperation Rate: {np.mean(eval_cooperation):.3f} ± {np.std(eval_cooperation):.3f}")

    # Plot results
    plot_training_results(episode_rewards, cooperation_rates)
    


if __name__ == "__main__":
    main()
