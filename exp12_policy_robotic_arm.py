"""
Experiment 12: Policy-based Reinforcement Learning (REINFORCE with a
softmax policy) for an industrial robotic arm performing efficient
pick-and-place operations on a discretized 1D arm-position problem.
"""
import numpy as np

N_POSITIONS = 8
PICK_POS = 2
PLACE_POS = 6
ACTIONS = ["LEFT", "RIGHT", "GRAB", "RELEASE"]
N_ACTIONS = len(ACTIONS)

def step(pos, holding, action):
    reward = -1
    done = False
    if action == "LEFT":
        pos = max(0, pos - 1)
    elif action == "RIGHT":
        pos = min(N_POSITIONS - 1, pos + 1)
    elif action == "GRAB":
        if pos == PICK_POS and not holding:
            holding = True
            reward = 10
    elif action == "RELEASE":
        if pos == PLACE_POS and holding:
            holding = False
            reward = 30
            done = True
    return pos, holding, reward, done

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def state_features(pos, holding):
    v = np.zeros(N_POSITIONS * 2)
    v[pos + (N_POSITIONS if holding else 0)] = 1
    return v

def reinforce(episodes=2000, alpha=0.05, gamma=0.98):
    theta = np.zeros((N_POSITIONS * 2, N_ACTIONS))
    for ep in range(episodes):
        pos, holding = 0, False
        trajectory = []
        for _ in range(40):
            feats = state_features(pos, holding)
            probs = softmax(feats @ theta)
            action_idx = np.random.choice(N_ACTIONS, p=probs)
            action = ACTIONS[action_idx]
            npos, nholding, reward, done = step(pos, holding, action)
            trajectory.append((feats, action_idx, reward))
            pos, holding = npos, nholding
            if done:
                break
        # policy gradient update
        G = 0
        for feats, action_idx, reward in reversed(trajectory):
            G = reward + gamma * G
            probs = softmax(feats @ theta)
            grad = -np.outer(feats, probs)
            grad[:, action_idx] += feats
            theta += alpha * G * grad
    return theta

if __name__ == "__main__":
    theta = reinforce()
    pos, holding = 0, False
    path = [(pos, holding)]
    for _ in range(20):
        feats = state_features(pos, holding)
        probs = softmax(feats @ theta)
        action = ACTIONS[np.argmax(probs)]
        pos, holding, reward, done = step(pos, holding, action)
        path.append((pos, holding))
        if done:
            break
    print("Learned pick-and-place trajectory (position, holding):")
    print(path)
