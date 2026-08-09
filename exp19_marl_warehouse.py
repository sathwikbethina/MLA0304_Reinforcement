"""
Experiment 19: Multi-Agent Reinforcement Learning (MARL) for a
multi-robot warehouse system to optimize cooperative task allocation
and navigation using independent Q-learning agents.
"""
import numpy as np
import random

GRID = 5
N_AGENTS = 2
TASKS = [(4, 4), (0, 4)]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1), "STAY": (0, 0)}

def init_positions():
    return [(0, 0), (4, 0)]

def move(pos, action):
    dr, dc = MOVES[action]
    return (max(0, min(GRID - 1, pos[0] + dr)), max(0, min(GRID - 1, pos[1] + dc)))

def step(positions, actions, tasks_left):
    new_positions = [move(p, a) for p, a in zip(positions, actions)]
    rewards = [-1] * N_AGENTS
    remaining = set(tasks_left)
    for i, pos in enumerate(new_positions):
        if pos in remaining:
            remaining.discard(pos)
            rewards[i] = 20
    # collision penalty: two agents on same cell
    if new_positions[0] == new_positions[1]:
        rewards = [r - 5 for r in rewards]
    done = len(remaining) == 0
    return new_positions, rewards, remaining, done

def init_Q():
    return {(r, c): {a: 0.0 for a in ACTIONS} for r in range(GRID) for c in range(GRID)}

def train(episodes=1500, alpha=0.1, gamma=0.95, epsilon=0.2):
    Qs = [init_Q() for _ in range(N_AGENTS)]
    for _ in range(episodes):
        positions = init_positions()
        tasks_left = set(TASKS)
        for _ in range(40):
            actions = []
            for i in range(N_AGENTS):
                if random.random() < epsilon:
                    actions.append(random.choice(ACTIONS))
                else:
                    actions.append(max(Qs[i][positions[i]], key=Qs[i][positions[i]].get))
            new_positions, rewards, tasks_left, done = step(positions, actions, tasks_left)
            for i in range(N_AGENTS):
                s, a, r, ns = positions[i], actions[i], rewards[i], new_positions[i]
                Qs[i][s][a] += alpha * (r + gamma * max(Qs[i][ns].values()) - Qs[i][s][a])
            positions = new_positions
            if done:
                break
    return Qs

if __name__ == "__main__":
    Qs = train()
    positions = init_positions()
    tasks_left = set(TASKS)
    paths = [[p] for p in positions]
    for _ in range(20):
        actions = [max(Qs[i][positions[i]], key=Qs[i][positions[i]].get) for i in range(N_AGENTS)]
        positions, rewards, tasks_left, done = step(positions, actions, tasks_left)
        for i in range(N_AGENTS):
            paths[i].append(positions[i])
        if done:
            break
    for i, path in enumerate(paths):
        print(f"Robot {i} path: {path}")
    print("Remaining unassigned tasks:", tasks_left)
