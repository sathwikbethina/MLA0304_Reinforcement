import random
grid = [[0] * 5 for i in range(5)]
d = int(input("Enter number of dirt cells: "))
for i in range(d):
    r, c = map(int, input(f"Enter dirt cell {i+1} (row col): ").split())
    grid[r][c] = 1
o = int(input("Enter number of obstacle cells: "))
for i in range(o):
    r, c = map(int, input(f"Enter obstacle cell {i+1} (row col): ").split())
    grid[r][c] = -1
r, c = map(int, input("Enter robot start position (row col): ").split())
reward = 0
moves = int(input("Enter number of moves: "))
for i in range(moves):
    move = random.choice(["up", "down", "left", "right"])
    if move == "up" and r > 0:
        r -= 1
    elif move == "down" and r < 4:
        r += 1
    elif move == "left" and c > 0:
        c -= 1
    elif move == "right" and c < 4:
        c += 1
    if grid[r][c] == 1:
        reward += 1
        grid[r][c] = 0
        print(f"Move {i+1}: {move} -> Robot: ({r},{c}) Dirt Cleaned! Reward = {reward}")
    elif grid[r][c] == -1:
        reward -= 1
        print(f"Move {i+1}: {move} -> Robot: ({r},{c}) Hit Obstacle! Reward = {reward}")
    else:
        print(f"Move {i+1}: {move} -> Robot: ({r},{c}) Reward = {reward}")
print("\nTotal Reward =", reward)