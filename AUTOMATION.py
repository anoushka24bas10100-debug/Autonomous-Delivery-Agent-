import heapq
import random
from collections import deque

# ------------------------
# Grid Environment
# ------------------------
class Grid:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, pos):
        r, c = pos
        return self.grid[r][c] != 1  # 1 = obstacle

    def cost(self, pos):
        r, c = pos
        return self.grid[r][c] if self.grid[r][c] > 0 else 1

    def neighbors(self, pos):
        r, c = pos
        moves = [(1,0), (-1,0), (0,1), (0,-1)]
        for dr, dc in moves:
            nxt = (r+dr, c+dc)
            if self.in_bounds(nxt) and self.passable(nxt):
                yield nxt

# ------------------------
# BFS (uninformed)
# ------------------------
def bfs(grid, start, goal):
    frontier = deque([start])
    came_from = {start: None}

    while frontier:
        current = frontier.popleft()
        if current == goal:
            break
        for nxt in grid.neighbors(current):
            if nxt not in came_from:
                frontier.append(nxt)
                came_from[nxt] = current

    return reconstruct_path(came_from, start, goal)

# ------------------------
# Uniform Cost Search
# ------------------------
def ucs(grid, start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        cost, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in grid.neighbors(current):
            new_cost = cost_so_far[current] + grid.cost(nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                heapq.heappush(frontier, (new_cost, nxt))
                came_from[nxt] = current

    return reconstruct_path(came_from, start, goal), cost_so_far.get(goal, None)

# ------------------------
# A* Search
# ------------------------
def heuristic(a, b):
    (x1, y1), (x2, y2) = a, b
    return abs(x1 - x2) + abs(y1 - y2)  # Manhattan

def astar(grid, start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in grid.neighbors(current):
            new_cost = cost_so_far[current] + grid.cost(nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(goal, nxt)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    return reconstruct_path(came_from, start, goal), cost_so_far.get(goal, None)

# ------------------------
# Local Search (hill climbing with random restarts)
# ------------------------
def hill_climb(grid, start, goal, max_restarts=10, max_steps=100):
    best_path = None
    best_len = float("inf")

    for _ in range(max_restarts):
        current = start
        path = [current]
        for _ in range(max_steps):
            if current == goal:
                if len(path) < best_len:
                    best_path = path[:]
                    best_len = len(path)
                break
            neighbors = list(grid.neighbors(current))
            if not neighbors:
                break
            current = min(neighbors, key=lambda n: heuristic(n, goal) + random.random())
            path.append(current)

    return best_path

# ------------------------
# Helper: Reconstruct Path
# ------------------------
def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return None
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path

# ------------------------
# Demo Runner
# ------------------------
if __name__ == "__main__":
    # 0 = normal, >1 = terrain cost, 1 = obstacle
    grid_map = [
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,0,2,1,0],
        [0,0,0,0,0]
    ]
    grid = Grid(grid_map)
    start, goal = (0,0), (3,4)

    print("BFS Path:", bfs(grid, start, goal))
    ucs_path, ucs_cost = ucs(grid, start, goal)
    print("UCS Path:", ucs_path, "Cost:", ucs_cost)
    a_path, a_cost = astar(grid, start, goal)
    print("A* Path:", a_path, "Cost:", a_cost)
    print("Hill Climb Path:", hill_climb(grid, start, goal))