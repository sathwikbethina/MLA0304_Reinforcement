"""
Experiment 7: Dynamic Programming (Policy Iteration) for an autonomous
taxi routing system to obtain the optimal driving policy on a city grid.
"""
import numpy as np

GRID = 6
PICKUP = (0, 5)
DROPOFF = (5, 0)
ACTIONS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def in_bounds(s):
    return 0 <= s[0] < GRID and 0 <= s[1] < GRID

def reward(s, has_passenger):
    if not has_passenger and s == PICKUP:
        return 10
    if has_passenger and s == DROPOFF:
        return 20
    return -1

def value_iteration(gamma=0.9, theta=1e-4):
    # state = (position, has_passenger)
    states = [((r, c), p) for r in range(GRID) for c in range(GRID) for p in (0, 1)]
    V = {s: 0.0 for s in states}
    policy = {}
    while True:
        delta = 0
        for (pos, has_pass) in states:
            best_val, best_a = -np.inf, None
            for a, (dr, dc) in ACTIONS.items():
                npos = (pos[0] + dr, pos[1] + dc)
                npos = npos if in_bounds(npos) else pos
                new_has_pass = has_pass
                if not has_pass and npos == PICKUP:
                    new_has_pass = 1
                if has_pass and npos == DROPOFF:
                    new_has_pass = 0
                r = reward(npos, has_pass)
                val = r + gamma * V[(npos, new_has_pass)]
                if val > best_val:
                    best_val, best_a = val, a
            delta = max(delta, abs(best_val - V[(pos, has_pass)]))
            V[(pos, has_pass)] = best_val
            policy[(pos, has_pass)] = best_a
        if delta < theta:
            break
    return V, policy

if __name__ == "__main__":
    V, policy = value_iteration()
    state, has_pass = (5, 5), 0
    path = [state]
    for _ in range(40):
        a = policy[(state, has_pass)]
        dr, dc = ACTIONS[a]
        nstate = (state[0] + dr, state[1] + dc)
        state = nstate if in_bounds(nstate) else state
        if not has_pass and state == PICKUP:
            has_pass = 1
        if has_pass and state == DROPOFF:
            has_pass = 0
            path.append(state)
            break
        path.append(state)
    print("Optimal taxi route (pickup -> dropoff):")
    print(path)
