"""
Experiment 6: RL model for autonomous robot navigation using OpenAI Gym
(FrozenLake-v1) with a TensorFlow/Keras Q-network (DQN-style).
Requires: gymnasium, tensorflow
"""
import numpy as np
import random
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers, models

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

def build_model():
    model = models.Sequential([
        layers.Input(shape=(n_states,)),
        layers.Dense(24, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(n_actions, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def one_hot(s):
    v = np.zeros(n_states)
    v[s] = 1
    return v.reshape(1, -1)

def train(episodes=300, gamma=0.95, epsilon_start=1.0, epsilon_min=0.05, decay=0.995):
    model = build_model()
    epsilon = epsilon_start
    rewards_log = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(100):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = model.predict(one_hot(state), verbose=0)
                action = np.argmax(q_vals[0])
            next_state, reward, terminated, truncated, _ = env.step(action)
            target = reward
            if not terminated:
                target += gamma * np.max(model.predict(one_hot(next_state), verbose=0)[0])
            q_vals = model.predict(one_hot(state), verbose=0)
            q_vals[0][action] = target
            model.fit(one_hot(state), q_vals, epochs=1, verbose=0)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        epsilon = max(epsilon_min, epsilon * decay)
        rewards_log.append(total_reward)
    return model, rewards_log

if __name__ == "__main__":
    model, rewards_log = train(episodes=50)  # small for demo, increase for real training
    print("Average reward over last 10 episodes:", np.mean(rewards_log[-10:]))
    model.save("exp06_navigation_model.h5")
    print("Model saved to exp06_navigation_model.h5")
