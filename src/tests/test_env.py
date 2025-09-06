import numpy as np
from src.env import PrisonersDilemmaEnv

def test_env():
    # Test the environment initial observation
    env = PrisonersDilemmaEnv(opponent_strategy="tit_for_tat", max_steps=10)

    obs, info = env.reset()
    print(type(obs))
    print(obs)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")

    assert np.array_equal(obs, np.array([-1.] * (env.history_length * 2))) 

    # Run a few steps
    for step in range(3):
        action = 1
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Observation step {step}: {obs}")

    assert np.array_equal(obs , np.array([ 1. , 0. , 1. , 1. , 1. , 1. ,-1. ,-1. ,-1. ,-1.]))