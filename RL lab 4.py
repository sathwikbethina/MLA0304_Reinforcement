states = 5

policy = [1,1,1,1,0]

value = [0]*states

reward = [0,0,0,0,10]

gamma = 0.9

for k in range(20):
    for s in range(states-2,-1,-1):
        value[s] = reward[s] + gamma*value[s+1]

print("Optimal Values")

for i in range(states):
    print("State",i,"=",round(value[i],2))