"""
Experiment 14: Actor-Critic (A2C-style; extendable to A3C via multiple
parallel workers) for a smart elevator scheduling system to reduce
passenger waiting time.
"""
import numpy as np

N_FLOORS = 6
ACTIONS = ["UP", "DOWN", "OPEN"]
N_ACTIONS = len(ACTIONS)

def reset():
    elevator_floor = 0
    calls = set(np.random.choice(range(N_FLOORS), size=2, replace=False))
    return elevator_floor, calls

def step(floor, calls, action):
    reward = -1
    if action == "UP":
        floor = min(N_FLOORS - 1, floor + 1)
    elif action == "DOWN":
        floor = max(0, floor - 1)
    elif action == "OPEN" and floor in calls:
        calls = calls - {floor}
        reward = 15
    done = len(calls) == 0
    return floor, calls, reward, done

def featurize(floor, calls):
    v = np.zeros(N_FLOORS * 2)
    v[floor] = 1
    for c in calls:
        v[N_FLOORS + c] = 1
    return v

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def actor_critic(episodes=1500, alpha_actor=0.05, alpha_critic=0.1, gamma=0.97):
    theta = np.zeros((N_FLOORS * 2, N_ACTIONS))   # actor weights
    w = np.zeros(N_FLOORS * 2)                    # critic weights
    for ep in range(episodes):
        floor, calls = reset()
        for _ in range(40):
            feats = featurize(floor, calls)
            probs = softmax(feats @ theta)
            a_idx = np.random.choice(N_ACTIONS, p=probs)
            nfloor, ncalls, reward, done = step(floor, calls, ACTIONS[a_idx])
            nfeats = featurize(nfloor, ncalls)

            v = feats @ w
            v_next = 0 if done else nfeats @ w
            td_error = reward + gamma * v_next - v

            w += alpha_critic * td_error * feats
            grad = -np.outer(feats, probs)
            grad[:, a_idx] += feats
            theta += alpha_actor * td_error * grad[:, a_idx].reshape(-1, 1) * 0 + alpha_actor * td_error * grad

            floor, calls = nfloor, ncalls
            if done:
                break
    return theta, w

if __name__ == "__main__":
    theta, w = actor_critic()
    floor, calls = 0, {2, 5}
    path = [(floor, set(calls))]
    for _ in range(20):
        feats = featurize(floor, calls)
        probs = softmax(feats @ theta)
        action = ACTIONS[np.argmax(probs)]
        floor, calls, reward, done = step(floor, calls, action)
        path.append((floor, set(calls)))
        if done:
            break
    print("Elevator scheduling trajectory (floor, remaining_calls):")
    for p in path:
        print(p)
