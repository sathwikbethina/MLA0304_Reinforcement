"""
Experiment 4: Bellman equations for an autonomous delivery robot to find
the minimum-travel-cost path across a grid with varying terrain costs.
"""
import numpy as np

GRID = 5
DEST = (4, 4)
# terrain cost of moving INTO a cell (higher = harder terrain)
COST = np.array([
    [1, 1, 2, 1, 1],
    [1, 3, 3, 3, 1],
    [1, 1, 1, 3, 1],
    [3, 3, 1, 3, 1],
    [1, 1, 1, 1, 1],
])
ACTIONS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def in_bounds(s):
    return 0 <= s[0] < GRID and 0 <= s[1] < GRID

def bellman_value_iteration(gamma=1.0, theta=1e-4):
    V = np.zeros((GRID, GRID))
    policy = {}
    while True:
        delta = 0
        newV = V.copy()
        for r in range(GRID):
            for c in range(GRID):
                if (r, c) == DEST:
                    continue
                best_val = -np.inf
                best_a = None
                for a, (dr, dc) in ACTIONS.items():
                    nr, nc = r + dr, c + dc
                    if not in_bounds((nr, nc)):
                        continue
                    reward = -COST[nr, nc]
                    val = reward + gamma * V[nr, nc]
                    if val > best_val:
                        best_val, best_a = val, a
                newV[r, c] = best_val
                policy[(r, c)] = best_a
                delta = max(delta, abs(newV[r, c] - V[r, c]))
        V = newV
        if delta < theta:
            break
    return V, policy

if __name__ == "__main__":
    V, policy = bellman_value_iteration()
    print("Value function (negative cumulative travel cost):")
    print(np.round(V, 1))

    state = (0, 0)
    path = [state]
    total_cost = 0
    for _ in range(50):
        if state == DEST:
            break
        a = policy[state]
        dr, dc = ACTIONS[a]
        state = (state[0] + dr, state[1] + dc)
        total_cost += COST[state]
        path.append(state)
    print("\nOptimal minimum-cost path:", path)
    print("Total travel cost:", total_cost)
