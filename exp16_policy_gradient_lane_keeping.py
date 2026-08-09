"""
Experiment 16: Comparison of policy gradient algorithms (REINFORCE and
Actor-Critic) for an autonomous lane-keeping system to improve driving
performance and stability, on a simplified 1D lane-offset environment.
"""
import numpy as np

N_STATES = 11     # lane offset discretized: 0 (far left) ... 10 (far right), 5 = centered
CENTER = 5
ACTIONS = ["STEER_LEFT", "STEER_RIGHT", "STRAIGHT"]
N_ACTIONS = len(ACTIONS)

def step(offset, action):
    if action == "STEER_LEFT":
        offset = max(0, offset - 1)
    elif action == "STEER_RIGHT":
        offset = min(N_STATES - 1, offset + 1)
    reward = -abs(offset - CENTER)
    done = abs(offset - CENTER) > 4
    if done:
        reward -= 20
    return offset, reward, done

def one_hot(s):
    v = np.zeros(N_STATES)
    v[s] = 1
    return v

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def reinforce(episodes=1000, alpha=0.05, gamma=0.95):
    theta = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(episodes):
        offset = CENTER + np.random.randint(-3, 4)
        traj = []
        for _ in range(30):
            feats = one_hot(offset)
            probs = softmax(feats @ theta)
            a = np.random.choice(N_ACTIONS, p=probs)
            noffset, reward, done = step(offset, ACTIONS[a])
            traj.append((feats, a, reward))
            offset = noffset
            if done:
                break
        G = 0
        for feats, a, reward in reversed(traj):
            G = reward + gamma * G
            probs = softmax(feats @ theta)
            grad = -np.outer(feats, probs)
            grad[:, a] += feats
            theta += alpha * G * grad
    return theta

def actor_critic(episodes=1000, alpha_a=0.05, alpha_c=0.1, gamma=0.95):
    theta = np.zeros((N_STATES, N_ACTIONS))
    w = np.zeros(N_STATES)
    for _ in range(episodes):
        offset = CENTER + np.random.randint(-3, 4)
        for _ in range(30):
            feats = one_hot(offset)
            probs = softmax(feats @ theta)
            a = np.random.choice(N_ACTIONS, p=probs)
            noffset, reward, done = step(offset, ACTIONS[a])
            nfeats = one_hot(noffset)
            v, v_next = feats @ w, 0 if done else nfeats @ w
            td_error = reward + gamma * v_next - v
            w += alpha_c * td_error * feats
            grad = -np.outer(feats, probs)
            grad[:, a] += feats
            theta += alpha_a * td_error * grad
            offset = noffset
            if done:
                break
    return theta

def evaluate(theta, label):
    offset = CENTER - 3
    total_reward = 0
    for _ in range(20):
        probs = softmax(one_hot(offset) @ theta)
        action = ACTIONS[np.argmax(probs)]
        offset, reward, done = step(offset, action)
        total_reward += reward
        if done:
            break
    print(f"{label}: final offset={offset}, total_reward={total_reward}")

if __name__ == "__main__":
    theta_reinforce = reinforce()
    evaluate(theta_reinforce, "REINFORCE")

    theta_ac = actor_critic()
    evaluate(theta_ac, "Actor-Critic")
