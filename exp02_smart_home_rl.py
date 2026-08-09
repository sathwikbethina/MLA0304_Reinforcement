"""
Experiment 2: RL agent for a smart home robot learning optimal navigation
through agent-environment interaction (Q-learning on a grid house map).
"""
import numpy as np
import random

GRID = 5
GOAL = (4, 4)
OBSTACLES = {(1, 1), (2, 2), (3, 1)}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def step(state, action):
    r, c = state
    dr, dc = MOVES[action]
    ns = (r + dr, c + dc)
    if not (0 <= ns[0] < GRID and 0 <= ns[1] < GRID) or ns in OBSTACLES:
        ns = state
        reward = -5
    elif ns == GOAL:
        reward = 20
    else:
        reward = -1
    done = ns == GOAL
    return ns, reward, done

def train(episodes=2000, alpha=0.1, gamma=0.95, epsilon=0.2):
    Q = {(r, c): {a: 0.0 for a in ACTIONS} for r in range(GRID) for c in range(GRID)}
    for ep in range(episodes):
        state = (0, 0)
        for _ in range(100):
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(Q[state], key=Q[state].get)
            ns, reward, done = step(state, action)
            best_next = max(Q[ns].values())
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
            state = ns
            if done:
                break
    return Q

if __name__ == "__main__":
    Q = train()
    state = (0, 0)
    path = [state]
    for _ in range(30):
        if state == GOAL:
            break
        action = max(Q[state], key=Q[state].get)
        state, _, done = step(state, action)
        path.append(state)
        if done:
            break
    print("Learned navigation path from (0,0) to goal:")
    print(path)
