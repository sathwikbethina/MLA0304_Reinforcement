"""
Experiment 3: MDP for an autonomous warehouse robot.
Defines states (shelf locations), actions (moves/pickup), transition
probabilities (stochastic slip), and rewards, solved with policy iteration.
"""
import numpy as np

GRID = 4
SHELF = (3, 3)          # item pickup location
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
SLIP_PROB = 0.1          # chance robot slips to a random adjacent cell

def in_bounds(s):
    return 0 <= s[0] < GRID and 0 <= s[1] < GRID

def transition(s, a):
    """Returns list of (probability, next_state, reward)."""
    outcomes = []
    intended = (s[0] + MOVES[a][0], s[1] + MOVES[a][1])
    intended = intended if in_bounds(intended) else s
    outcomes.append((1 - SLIP_PROB, intended))
    for other_a in ACTIONS:
        if other_a != a:
            slip_state = (s[0] + MOVES[other_a][0], s[1] + MOVES[other_a][1])
            slip_state = slip_state if in_bounds(slip_state) else s
            outcomes.append((SLIP_PROB / (len(ACTIONS) - 1), slip_state))
    result = []
    for p, ns in outcomes:
        r = 20 if ns == SHELF else -1
        result.append((p, ns, r))
    return result

def policy_iteration(gamma=0.9):
    states = [(r, c) for r in range(GRID) for c in range(GRID)]
    V = {s: 0.0 for s in states}
    policy = {s: np.random.choice(ACTIONS) for s in states}

    stable = False
    while not stable:
        # Policy evaluation
        while True:
            delta = 0
            for s in states:
                if s == SHELF:
                    continue
                v = V[s]
                V[s] = sum(p * (r + gamma * V[ns]) for p, ns, r in transition(s, policy[s]))
                delta = max(delta, abs(v - V[s]))
            if delta < 1e-4:
                break
        # Policy improvement
        stable = True
        for s in states:
            if s == SHELF:
                continue
            old_a = policy[s]
            action_values = {a: sum(p * (r + gamma * V[ns]) for p, ns, r in transition(s, a)) for a in ACTIONS}
            policy[s] = max(action_values, key=action_values.get)
            if policy[s] != old_a:
                stable = False
    return V, policy

if __name__ == "__main__":
    V, policy = policy_iteration()
    print("Optimal Policy for Warehouse Robot:")
    for r in range(GRID):
        print([policy[(r, c)][0] for c in range(GRID)])
    print("\nState Values:")
    for r in range(GRID):
        print([round(V[(r, c)], 1) for c in range(GRID)])
