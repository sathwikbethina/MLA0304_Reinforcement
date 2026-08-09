"""
Experiment 13: REINFORCE algorithm for an autonomous parking system to
learn optimal parking strategies (align car with parking slot).
"""
import numpy as np

N_POS = 10       # lateral positions
SLOT = 5
ACTIONS = ["LEFT", "RIGHT", "PARK"]
N_ACTIONS = len(ACTIONS)

def step(pos, action):
    reward, done = -1, False
    if action == "LEFT":
        pos = max(0, pos - 1)
    elif action == "RIGHT":
        pos = min(N_POS - 1, pos + 1)
    elif action == "PARK":
        if pos == SLOT:
            reward, done = 50, True
        else:
            reward, done = -20, True
    return pos, reward, done

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def one_hot(pos):
    v = np.zeros(N_POS)
    v[pos] = 1
    return v

def reinforce(episodes=1500, alpha=0.05, gamma=0.99):
    theta = np.zeros((N_POS, N_ACTIONS))
    for ep in range(episodes):
        pos = np.random.randint(N_POS)
        trajectory = []
        for _ in range(20):
            feats = one_hot(pos)
            probs = softmax(feats @ theta)
            a_idx = np.random.choice(N_ACTIONS, p=probs)
            npos, reward, done = step(pos, ACTIONS[a_idx])
            trajectory.append((feats, a_idx, reward))
            pos = npos
            if done:
                break
        G = 0
        for feats, a_idx, reward in reversed(trajectory):
            G = reward + gamma * G
            probs = softmax(feats @ theta)
            grad = -np.outer(feats, probs)
            grad[:, a_idx] += feats
            theta += alpha * G * grad
    return theta

if __name__ == "__main__":
    theta = reinforce()
    for start in [0, 3, 8, 9]:
        pos = start
        actions_taken = []
        for _ in range(15):
            probs = softmax(one_hot(pos) @ theta)
            action = ACTIONS[np.argmax(probs)]
            actions_taken.append(action)
            pos, reward, done = step(pos, action)
            if done:
                break
        print(f"Start pos {start}: {actions_taken} -> final pos {pos} (reward {reward})")
