import random

# Number of customers
customers = 100

# Track customer retention
retained = 0
churned = 0

# Monte Carlo simulation
for i in range(customers):

    # Policy prediction:
    # 1 = Customer stays, 0 = Customer leaves
    action = random.choice([0, 1])

    # Customer behavior environment
    if action == 1:
        reward = 1      # Retained customer
        retained += 1
    else:
        reward = 0      # Churned customer
        churned += 1

# Policy evaluation
retention_rate = retained / customers
churn_rate = churned / customers

print("Total Customers:", customers)
print("Customers Retained:", retained)
print("Customers Churned:", churned)

print("\nRetention Rate:", round(retention_rate, 2))
print("Churn Rate:", round(churn_rate, 2))