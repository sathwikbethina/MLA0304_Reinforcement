"""
Experiment 20: Search-and-rescue robot modeled as a Partially Observable
Markov Decision Process (POMDP). The robot maintains a belief
distribution over the survivor's location and updates it with noisy
sensor observations to decide where to search next.
"""
import numpy as np

GRID = 5
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SEARCH"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
SENSOR_ACCURACY = 0.7   # probability sensor correctly detects survivor when co-located

def init_belief():
    b = np.ones((GRID, GRID)) / (GRID * GRID)
    return b

def update_belief(belief, robot_pos, observation):
    """Bayesian belief update given a binary observation ('detected'/'nothing')."""
    new_belief = np.zeros_like(belief)
    for r in range(GRID):
        for c in range(GRID):
            prior = belief[r, c]
            if (r, c) == robot_pos:
                p_obs = SENSOR_ACCURACY if observation == "detected" else (1 - SENSOR_ACCURACY)
            else:
                p_obs = (1 - SENSOR_ACCURACY) if observation == "detected" else SENSOR_ACCURACY
            new_belief[r, c] = prior * p_obs
    total = new_belief.sum()
    return new_belief / total if total > 0 else belief

def simulate_observation(robot_pos, true_survivor_pos):
    if robot_pos == true_survivor_pos:
        return "detected" if np.random.rand() < SENSOR_ACCURACY else "nothing"
    else:
        return "nothing" if np.random.rand() < SENSOR_ACCURACY else "detected"

def choose_action(belief, robot_pos):
    """Move toward the cell with the highest belief probability, then search."""
    target = np.unravel_index(np.argmax(belief), belief.shape)
    if target == robot_pos:
        return "SEARCH"
    dr, dc = target[0] - robot_pos[0], target[1] - robot_pos[1]
    if abs(dr) >= abs(dc):
        return "DOWN" if dr > 0 else "UP"
    return "RIGHT" if dc > 0 else "LEFT"

if __name__ == "__main__":
    np.random.seed(1)
    true_survivor_pos = (3, 4)
    robot_pos = (0, 0)
    belief = init_belief()

    for t in range(15):
        action = choose_action(belief, robot_pos)
        if action == "SEARCH":
            obs = simulate_observation(robot_pos, true_survivor_pos)
            belief = update_belief(belief, robot_pos, obs)
            print(f"Step {t}: SEARCH at {robot_pos}, obs={obs}, belief peak={np.unravel_index(np.argmax(belief), belief.shape)}")
            if robot_pos == true_survivor_pos and obs == "detected":
                print(f"Survivor found at {robot_pos}!")
                break
        else:
            dr, dc = MOVES[action]
            robot_pos = (max(0, min(GRID - 1, robot_pos[0] + dr)), max(0, min(GRID - 1, robot_pos[1] + dc)))
            print(f"Step {t}: MOVE {action} -> {robot_pos}")
