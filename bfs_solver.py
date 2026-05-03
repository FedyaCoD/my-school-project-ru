import time
from collections import deque

class BFSSolver:
    def __init__(self, maze):
        self.maze = maze

    def solve(self):
        start = self.maze.get_start()
        goal = self.maze.get_goal()
        start_time = time.time()

        queue = deque()
        queue.append((start, [start]))
        visited = set()
        visited.add(start)

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                elapsed = time.time() - start_time
                return path, elapsed, len(path) - 1
            for nx, ny in self.maze.get_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        elapsed = time.time() - start_time
        return None, elapsed, 0