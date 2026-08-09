"""
Experiment 18: Meta-Reinforcement Learning model for an adaptive
industrial robot to quickly learn new manufacturing tasks. Implements a
simplified Reptile-style meta-learning loop over a family of grid tasks
(different goal positions represent different manufacturing tasks).
"""
import numpy as np
import random

GRID = 5
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def sample_task():
    """Each task = a different goal position (different manufacturing target)."""
    return (random.randint(0, GRID - 1), random.randint(0, GRID - 1))

def step(state, action, goal):
    dr, dc = MOVES[action]
    ns = (max(0, min(GRID - 1, state[0] + dr)), max(0, min(GRID - 1, state[1] + dc)))
    reward = 20 if ns == goal else -1
    return ns, reward, ns == goal

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def featurize(state, goal):
    v = np.zeros(GRID * GRID * 2)
    v[state[0] * GRID + state[1]] = 1
    v[GRID * GRID + goal[0] * GRID + goal[1]] = 1
    return v

def inner_loop_update(theta, goal, steps=15, alpha=0.1, gamma=0.95):
    """Fast adaptation: a few policy-gradient steps on the sampled task."""
    theta = theta.copy()
    for _ in range(steps):
        state = (0, 0)
        traj = []
        for _ in range(20):
            feats = featurize(state, goal)
            probs = softmax(feats @ theta)
            a_idx = np.random.choice(len(ACTIONS), p=probs)
            ns, reward, done = step(state, ACTIONS[a_idx], goal)
            traj.append((feats, a_idx, reward))
            state = ns
            if done:
                break
        G = 0
        for feats, a_idx, reward in reversed(traj):
            G = reward + gamma * G
            probs = softmax(feats @ theta)
            grad = -np.outer(feats, probs)
            grad[:, a_idx] += feats
            theta += alpha * G * grad
    return theta

def reptile_meta_train(meta_iters=200, inner_steps=10, meta_lr=0.1):
    theta = np.zeros((GRID * GRID * 2, len(ACTIONS)))
    for _ in range(meta_iters):
        goal = sample_task()
        adapted_theta = inner_loop_update(theta, goal, steps=inner_steps)
        theta += meta_lr * (adapted_theta - theta)   # Reptile meta-update
    return theta

if __name__ == "__main__":
    meta_theta = reptile_meta_train()
    # Quickly adapt the meta-learned policy to a brand-new manufacturing task
    new_goal = sample_task()
    print("New manufacturing task (goal position):", new_goal)
    fast_theta = inner_loop_update(meta_theta, new_goal, steps=5)

    state = (0, 0)
    path = [state]
    for _ in range(15):
        feats = featurize(state, new_goal)
        probs = softmax(feats @ fast_theta)
        action = ACTIONS[np.argmax(probs)]
        state, reward, done = step(state, action, new_goal)
        path.append(state)
        if done:
            break
    print("Path after fast-adapting to the new task:", path)
