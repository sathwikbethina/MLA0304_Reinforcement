"""
Experiment 15: PPO and TRPO for a humanoid robot to achieve stable
walking and balance, using the Gymnasium Humanoid-v4 / BipedalWalker-v3
environment with Stable-Baselines3 (PPO) and sb3-contrib (TRPO).
Requires: gymnasium, stable-baselines3, sb3-contrib, box2d-py
"""
import gymnasium as gym
from stable_baselines3 import PPO
try:
    from sb3_contrib import TRPO
    HAS_TRPO = True
except ImportError:
    HAS_TRPO = False

ENV_ID = "BipedalWalker-v3"   # swap for "Humanoid-v4" if MuJoCo is installed

def train_ppo(timesteps=20000):
    env = gym.make(ENV_ID)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("exp15_ppo_walker")
    env.close()
    return model

def train_trpo(timesteps=20000):
    if not HAS_TRPO:
        print("sb3-contrib not installed; skipping TRPO. Install with: pip install sb3-contrib")
        return None
    env = gym.make(ENV_ID)
    model = TRPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("exp15_trpo_walker")
    env.close()
    return model

def evaluate(model, episodes=3):
    env = gym.make(ENV_ID)
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    print("Training PPO...")
    ppo_model = train_ppo(timesteps=5000)   # increase for real training
    evaluate(ppo_model)

    print("Training TRPO...")
    trpo_model = train_trpo(timesteps=5000)
    if trpo_model:
        evaluate(trpo_model)
