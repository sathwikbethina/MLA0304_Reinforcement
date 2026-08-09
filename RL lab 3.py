import random

price = [10,20,30]

Q = [0,0,0]
count = [0,0,0]

epsilon = 0.2

for i in range(100):

    if random.random()<epsilon:
        arm = random.randint(0,2)
    else:
        arm = Q.index(max(Q))

    reward = random.randint(5,price[arm])

    count[arm]+=1
    Q[arm]=Q[arm]+(reward-Q[arm])/count[arm]

print("Average Rewards")
print(Q)

print("Best Price =",price[Q.index(max(Q))])