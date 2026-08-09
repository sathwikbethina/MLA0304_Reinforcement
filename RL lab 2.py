# Runtime Input for Policy Evaluation

states = int(input("Enter number of states: "))

reward = []
print("Enter rewards for each state:")
for i in range(states):
    r = float(input(f"Reward for State {i}: "))
    reward.append(r)

gamma = float(input("Enter discount factor (gamma): "))

value = [0] * states

iterations = int(input("Enter number of iterations: "))

for i in range(iterations):
    for s in range(states - 2, -1, -1):
        value[s] = reward[s] + gamma * value[s + 1]

print("\nState Values")
for i in range(states):
    print("State", i, "=", round(value[i], 2))