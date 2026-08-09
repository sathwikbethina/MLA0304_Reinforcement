"""
Experiment 17: Hierarchical Reinforcement Learning (HRL) for an
autonomous household robot performing multiple tasks (e.g., "clean
kitchen", "clean living room") using a MAXQ-style task decomposition
(with HAM-style hierarchical machines as sub-controllers).
"""
import numpy as np
import random

ROOMS = ["Kitchen", "LivingRoom"]
GRID = 4

def room_offset(room):
    return (0, 0) if room == "Kitchen" else (0, GRID)

DIRT = {"Kitchen": {(1, 1), (2, 3)}, "LivingRoom": {(3, 0), (0, 2)}}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "CLEAN"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

# --- Low-level (primitive) Q-learning subtask: "go clean this room" ---
def train_room_subtask(room, episodes=800, alpha=0.1, gamma=0.9, epsilon=0.2):
    Q = {}
    dirt_set = DIRT[room]
    for r in range(GRID):
        for c in range(GRID):
            Q[(r, c)] = {a: 0.0 for a in ACTIONS}
    for _ in range(episodes):
        pos = (0, 0)
        remaining = set(dirt_set)
        for _ in range(30):
            state_key = pos
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(Q[state_key], key=Q[state_key].get)
            reward = -1
            if action == "CLEAN":
                if pos in remaining:
                    remaining.discard(pos)
                    reward = 15
            else:
                dr, dc = MOVES[action]
                nr, nc = max(0, min(GRID - 1, pos[0] + dr)), max(0, min(GRID - 1, pos[1] + dc))
                pos = (nr, nc)
            done = len(remaining) == 0
            next_key = pos
            Q[state_key][action] += alpha * (reward + gamma * max(Q[next_key].values()) - Q[state_key][action])
            if done:
                break
    return Q

# --- Top-level MAXQ-style controller: choose which room to clean first ---
def top_level_policy(room_costs):
    """Greedy: clean the room with fewer dirt spots first (lowest estimated cost)."""
    return sorted(ROOMS, key=lambda r: len(DIRT[r]))

if __name__ == "__main__":
    subtask_policies = {room: train_room_subtask(room) for room in ROOMS}
    order = top_level_policy(DIRT)
    print("MAXQ top-level task order:", order)

    for room in order:
        Q = subtask_policies[room]
        pos = (0, 0)
        remaining = set(DIRT[room])
        path = [pos]
        for _ in range(20):
            action = max(Q[pos], key=Q[pos].get)
            if action == "CLEAN" and pos in remaining:
                remaining.discard(pos)
            elif action != "CLEAN":
                dr, dc = MOVES[action]
                pos = (max(0, min(GRID - 1, pos[0] + dr)), max(0, min(GRID - 1, pos[1] + dc)))
                path.append(pos)
            if not remaining:
                break
        print(f"Subtask '{room}': path={path}, remaining dirt={remaining}")
