"""
Experiment 9: TD(0), SARSA, and Q-Learning for a warehouse robot to
optimize navigation and obstacle avoidance. All three algorithms are
implemented and compared on the same grid.
"""
import numpy as np
import random

GRID = 5
GOAL = (4, 4)
OBSTACLES = {(1, 2), (2, 2), (3, 2)}
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
    return ns, reward, ns == GOAL

def choose_action(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

def init_Q():
    return {(r, c): {a: 0.0 for a in ACTIONS} for r in range(GRID) for c in range(GRID)}

def td0_policy_eval(policy, episodes=500, alpha=0.1, gamma=0.95):
    """TD(0) prediction: evaluate a fixed random policy's state-value function."""
    V = {(r, c): 0.0 for r in range(GRID) for c in range(GRID)}
    for _ in range(episodes):
        state = (0, 0)
        for _ in range(100):
            action = random.choice(ACTIONS)
            ns, reward, done = step(state, action)
            V[state] += alpha * (reward + gamma * V[ns] - V[state])
            state = ns
            if done:
                break
    return V

def sarsa(episodes=800, alpha=0.1, gamma=0.95, epsilon=0.2):
    Q = init_Q()
    for _ in range(episodes):
        state = (0, 0)
        action = choose_action(Q, state, epsilon)
        for _ in range(100):
            ns, reward, done = step(state, action)
            na = choose_action(Q, ns, epsilon)
            Q[state][action] += alpha * (reward + gamma * Q[ns][na] - Q[state][action])
            state, action = ns, na
            if done:
                break
    return Q

def q_learning(episodes=800, alpha=0.1, gamma=0.95, epsilon=0.2):
    Q = init_Q()
    for _ in range(episodes):
        state = (0, 0)
        for _ in range(100):
            action = choose_action(Q, state, epsilon)
            ns, reward, done = step(state, action)
            Q[state][action] += alpha * (reward + gamma * max(Q[ns].values()) - Q[state][action])
            state = ns
            if done:
                break
    return Q

def extract_path(Q):
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
    return path

if __name__ == "__main__":
    V = td0_policy_eval(policy=None)
    print("TD(0) value estimate at start state:", round(V[(0, 0)], 2))

    Q_sarsa = sarsa()
    print("SARSA optimal path:", extract_path(Q_sarsa))

    Q_ql = q_learning()
    print("Q-Learning optimal path:", extract_path(Q_ql))
