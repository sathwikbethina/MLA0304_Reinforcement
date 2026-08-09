"""
Experiment 10: Deep Q-Network (DQN) for an autonomous drone delivery
system to optimize delivery routes under battery constraints.
Requires: tensorflow
"""
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models

GRID = 6
DEST = (5, 5)
MAX_BATTERY = 20
STATE_DIM = 3   # (row, col, battery)
N_ACTIONS = 4
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def reset():
    return [0, 0, MAX_BATTERY]

def step(s, a_idx):
    r, c, battery = s
    dr, dc = MOVES[ACTIONS[a_idx]]
    nr, nc = max(0, min(GRID - 1, r + dr)), max(0, min(GRID - 1, c + dc))
    battery -= 1
    if (nr, nc) == DEST:
        reward, done = 50, True
    elif battery <= 0:
        reward, done = -20, True
    else:
        reward, done = -1, False
    return [nr, nc, battery], reward, done

def build_model():
    model = models.Sequential([
        layers.Input(shape=(STATE_DIM,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(N_ACTIONS, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train(episodes=100, gamma=0.95, epsilon_start=1.0, epsilon_min=0.05, decay=0.98, batch_size=32):
    model = build_model()
    memory = deque(maxlen=2000)
    epsilon = epsilon_start
    for ep in range(episodes):
        state = reset()
        for _ in range(30):
            if random.random() < epsilon:
                action = random.randrange(N_ACTIONS)
            else:
                q = model.predict(np.array([state]) / [GRID, GRID, MAX_BATTERY], verbose=0)
                action = int(np.argmax(q[0]))
            next_state, reward, done = step(state, action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states = np.array([b[0] for b in batch]) / [GRID, GRID, MAX_BATTERY]
            next_states = np.array([b[3] for b in batch]) / [GRID, GRID, MAX_BATTERY]
            q_targets = model.predict(states, verbose=0)
            q_next = model.predict(next_states, verbose=0)
            for i, (s, a, r, ns, d) in enumerate(batch):
                q_targets[i][a] = r if d else r + gamma * np.max(q_next[i])
            model.fit(states, q_targets, epochs=1, verbose=0)
        epsilon = max(epsilon_min, epsilon * decay)
    return model

if __name__ == "__main__":
    model = train(episodes=30)  # increase for real training
    model.save("exp10_drone_dqn.h5")
    print("Drone delivery DQN model saved to exp10_drone_dqn.h5")
