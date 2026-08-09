"""
Experiment 8: Monte Carlo prediction and control for a robot vacuum
cleaner learning an efficient cleaning policy while minimizing energy use.
"""
import numpy as np
import random
from collections import defaultdict

GRID = 4
DIRT = {(0, 3), (2, 1), (3, 3)}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
MAX_STEPS = 25

def step(state, action, dirt_left):
    r, c = state
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID and 0 <= nc < GRID):
        nr, nc = r, c
    reward = -1  # energy cost per move
    new_dirt = set(dirt_left)
    if (nr, nc) in new_dirt:
        reward += 15
        new_dirt.discard((nr, nc))
    return (nr, nc), reward, new_dirt

def generate_episode(Q, epsilon):
    state = (0, 0)
    dirt_left = set(DIRT)
    episode = []
    for _ in range(MAX_STEPS):
        key = (state, frozenset(dirt_left))
        if key not in Q or random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = max(Q[key], key=Q[key].get)
        next_state, reward, dirt_left = step(state, action, dirt_left)
        episode.append((key, action, reward))
        state = next_state
        if not dirt_left:
            break
    return episode

def mc_control(episodes=3000, gamma=0.95, epsilon=0.2):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for _ in range(episodes):
        episode = generate_episode(Q, epsilon)
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            key, action, reward = episode[t]
            G = gamma * G + reward
            if (key, action) not in visited:
                visited.add((key, action))
                returns_sum[(key, action)] += G
                returns_count[(key, action)] += 1
                Q[key][action] = returns_sum[(key, action)] / returns_count[(key, action)]
    return Q

if __name__ == "__main__":
    Q = mc_control()
    state = (0, 0)
    dirt_left = set(DIRT)
    path = [state]
    for _ in range(MAX_STEPS):
        key = (state, frozenset(dirt_left))
        action = max(Q[key], key=Q[key].get) if key in Q else random.choice(ACTIONS)
        state, _, dirt_left = step(state, action, dirt_left)
        path.append(state)
        if not dirt_left:
            break
    print("Learned cleaning path:", path)
    print("Dirt remaining:", dirt_left)
