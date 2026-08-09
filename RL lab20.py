import random

# Two content options
contents = ["Content A", "Content B"]

# Store rewards
reward = {
    "Content A": 0,
    "Content B": 0
}

# Count selections
count = {
    "Content A": 0,
    "Content B": 0
}

# Exploration probability
epsilon = 0.2

# Simulate user interactions
for i in range(100):

    # Exploration or exploitation
    if random.random() < epsilon:
        choice = random.choice(contents)   # Explore
    else:
        # Select best content
        choice = max(reward, key=reward.get)  # Exploit

    # User feedback
    if choice == "Content A":
        user_reward = random.randint(1, 5)
    else:
        user_reward = random.randint(3, 8)

    # Update values
    count[choice] += 1
    reward[choice] = reward[choice] + (user_reward - reward[choice]) / count[choice]


print("Content Values:")
for c in contents:
    print(c, ":", round(reward[c], 2))

best_content = max(reward, key=reward.get)

print("\nRecommended Content:", best_content)