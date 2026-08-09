"""
Experiment 5: epsilon-greedy Multi-Armed Bandit for an online ad
recommendation system, balancing exploration and exploitation to
maximize user engagement (click-through rate).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N_ADS = 5
TRUE_CTR = np.array([0.05, 0.12, 0.08, 0.20, 0.15])  # unknown to agent
EPSILON = 0.1
ROUNDS = 5000

def epsilon_greedy():
    counts = np.zeros(N_ADS)
    values = np.zeros(N_ADS)
    rewards_history = []
    for t in range(ROUNDS):
        if np.random.rand() < EPSILON:
            ad = np.random.randint(N_ADS)
        else:
            ad = np.argmax(values)
        reward = 1 if np.random.rand() < TRUE_CTR[ad] else 0
        counts[ad] += 1
        values[ad] += (reward - values[ad]) / counts[ad]
        rewards_history.append(reward)
    return values, counts, rewards_history

if __name__ == "__main__":
    values, counts, rewards_history = epsilon_greedy()
    print("Estimated CTR per ad:", np.round(values, 3))
    print("True CTR per ad:     ", TRUE_CTR)
    print("Times each ad was shown:", counts.astype(int))
    print("Best ad chosen:", np.argmax(values), "| Actual best ad:", np.argmax(TRUE_CTR))

    cum_avg = np.cumsum(rewards_history) / (np.arange(ROUNDS) + 1)
    plt.plot(cum_avg)
    plt.xlabel("Round")
    plt.ylabel("Average reward (CTR)")
    plt.title("epsilon-greedy Ad Recommendation Performance")
    plt.savefig("exp05_ctr_plot.png")
    print("Saved plot to exp05_ctr_plot.png")
