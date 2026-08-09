"""
Experiment 11: DQN, Double DQN (DDQN), Dueling DQN, and Prioritized
Experience Replay (PER) compared for smart traffic signal control to
minimize vehicle waiting time. A simple custom traffic environment is used.
Requires: tensorflow
"""
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models

N_ACTIONS = 2   # 0 = keep NS green, 1 = switch to EW green
STATE_DIM = 4   # [NS_queue, EW_queue, current_phase, phase_timer]
MAX_QUEUE = 20

def reset():
    return [random.randint(0, 10), random.randint(0, 10), 0, 0]

def step(s, a):
    ns_q, ew_q, phase, timer = s
    if a == 1:
        phase = 1 - phase
        timer = 0
    else:
        timer += 1
    # arrivals
    ns_q += random.randint(0, 2)
    ew_q += random.randint(0, 2)
    # departures on green phase
    if phase == 0:
        ns_q = max(0, ns_q - 4)
    else:
        ew_q = max(0, ew_q - 4)
    ns_q, ew_q = min(ns_q, MAX_QUEUE), min(ew_q, MAX_QUEUE)
    reward = -(ns_q + ew_q)
    done = False
    return [ns_q, ew_q, phase, timer], reward, done

def build_dqn():
    model = models.Sequential([
        layers.Input(shape=(STATE_DIM,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(N_ACTIONS, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_dueling_dqn():
    inp = layers.Input(shape=(STATE_DIM,))
    x = layers.Dense(32, activation="relu")(inp)
    x = layers.Dense(32, activation="relu")(x)
    value = layers.Dense(1, activation="linear")(x)
    advantage = layers.Dense(N_ACTIONS, activation="linear")(x)
    q = layers.Add()([value, layers.Subtract()([advantage, layers.Lambda(
        lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)])])
    model = models.Model(inp, q)
    model.compile(optimizer="adam", loss="mse")
    return model

class PrioritizedReplay:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, transition, td_error=1.0):
        self.buffer.append(transition)
        self.priorities.append(abs(td_error) + 1e-3)

    def sample(self, batch_size):
        probs = np.array(self.priorities) / sum(self.priorities)
        idx = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probs)
        return [self.buffer[i] for i in idx], idx

def train(algo="dqn", episodes=60, gamma=0.95, batch_size=32):
    model = build_dueling_dqn() if algo == "dueling" else build_dqn()
    target_model = build_dueling_dqn() if algo == "dueling" else build_dqn()
    target_model.set_weights(model.get_weights())
    memory = PrioritizedReplay() if algo == "per" else deque(maxlen=2000)
    epsilon = 1.0
    norm = np.array([MAX_QUEUE, MAX_QUEUE, 1, 10])

    for ep in range(episodes):
        state = reset()
        for t in range(50):
            if random.random() < epsilon:
                action = random.randrange(N_ACTIONS)
            else:
                q = model.predict(np.array([state]) / norm, verbose=0)
                action = int(np.argmax(q[0]))
            next_state, reward, done = step(state, action)
            trans = (state, action, reward, next_state, done)
            if algo == "per":
                memory.add(trans)
            else:
                memory.append(trans)
            state = next_state

        buffer = memory.buffer if algo == "per" else memory
        if len(buffer) >= batch_size:
            if algo == "per":
                batch, _ = memory.sample(batch_size)
            else:
                batch = random.sample(memory, batch_size)
            states = np.array([b[0] for b in batch]) / norm
            next_states = np.array([b[3] for b in batch]) / norm
            q_vals = model.predict(states, verbose=0)
            q_next_online = model.predict(next_states, verbose=0)
            q_next_target = target_model.predict(next_states, verbose=0)
            for i, (s, a, r, ns, d) in enumerate(batch):
                if algo == "ddqn":
                    best_a = np.argmax(q_next_online[i])
                    target = r + gamma * q_next_target[i][best_a]
                else:
                    target = r + gamma * np.max(q_next_target[i])
                q_vals[i][a] = target
            model.fit(states, q_vals, epochs=1, verbose=0)
            if ep % 5 == 0:
                target_model.set_weights(model.get_weights())
        epsilon = max(0.05, epsilon * 0.97)
    return model

if __name__ == "__main__":
    for algo in ["dqn", "ddqn", "dueling", "per"]:
        print(f"Training {algo.upper()} for traffic signal control...")
        model = train(algo=algo, episodes=20)  # small demo run
        model.save(f"exp11_traffic_{algo}.h5")
    print("All 4 variants trained and saved.")
