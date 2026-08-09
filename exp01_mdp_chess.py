"""
Experiment 1: MDP for a simplified chess game (Python)
A tiny chess-like board (3x3) where an agent (King) must reach a goal
square while avoiding an opponent-controlled square. Modeled as an MDP
with states = board positions, actions = moves, transitions, rewards.
Solved with Value Iteration to get the optimal policy.
"""
import numpy as np

GRID = 3
GOAL = (2, 2)
DANGER = (1, 1)          # square attacked by opponent piece
ACTIONS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def in_bounds(s):
    return 0 <= s[0] < GRID and 0 <= s[1] < GRID

def reward(s):
    if s == GOAL:
        return 10
    if s == DANGER:
        return -10
    return -1

def next_state(s, a):
    ns = (s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1])
    return ns if in_bounds(ns) else s

def value_iteration(gamma=0.9, theta=1e-4):
    V = {(r, c): 0.0 for r in range(GRID) for c in range(GRID)}
    policy = {}
    while True:
        delta = 0
        for s in V:
            if s == GOAL:
                continue
            best_val = -np.inf
            best_a = None
            for a in ACTIONS:
                ns = next_state(s, a)
                val = reward(ns) + gamma * V[ns]
                if val > best_val:
                    best_val, best_a = val, a
            delta = max(delta, abs(best_val - V[s]))
            V[s] = best_val
            policy[s] = best_a
        if delta < theta:
            break
    return V, policy

if __name__ == "__main__":
    V, policy = value_iteration()
    print("Optimal Values:")
    for r in range(GRID):
        print([round(V[(r, c)], 2) for c in range(GRID)])
    print("\nOptimal Policy (state -> action):")
    for s, a in policy.items():
        print(f"{s}: {a}")

    # Simulate agent following optimal policy from a start state
    state = (0, 0)
    path = [state]
    for _ in range(10):
        if state == GOAL:
            break
        state = next_state(state, policy[state])
        path.append(state)
    print("\nAgent path to goal:", path)
