import numpy as np
import random
from collections import deque

class Maze:
    def __init__(self, size: int = 10, wall_probability: float = 0.3):
        self.size = size
        self.wall_prob = wall_probability
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.grid = None
        self._generate()

    def _generate(self, max_attempts: int = 10) -> None:
        for _ in range(max_attempts):
            grid = np.zeros((self.size, self.size), dtype=int)
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) != self.start and (i, j) != self.goal:
                        if random.random() < self.wall_prob:
                            grid[i, j] = 1
            if self._is_path_exists(grid):
                self.grid = grid
                return
        self.grid = np.zeros((self.size, self.size), dtype=int)
        print("Предупреждение: не удалось сгенерировать лабиринт с заданной сложностью, создан пустой лабиринт.")

    def _is_path_exists(self, grid: np.ndarray) -> bool:
        if grid[self.start] == 1 or grid[self.goal] == 1:
            return False
        visited = np.zeros((self.size, self.size), dtype=bool)
        q = deque()
        q.append(self.start)
        visited[self.start] = True
        while q:
            x, y = q.popleft()
            if (x, y) == self.goal:
                return True
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if not visited[nx, ny] and grid[nx, ny] == 0:
                        visited[nx, ny] = True
                        q.append((nx, ny))
        return False

    def dynamic_update(self, prob: float = 0.1, max_attempts: int = 5) -> bool:
        for _ in range(max_attempts):
            new_grid = self.grid.copy()
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) != self.start and (i, j) != self.goal:
                        if random.random() < prob:
                            new_grid[i, j] = 1 - new_grid[i, j]
            if self._is_path_exists(new_grid):
                self.grid = new_grid
                return True
        return False

    def is_wall(self, x: int, y: int) -> bool:
        return self.grid[x, y] == 1

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size and not self.is_wall(x, y)

    def get_neighbors(self, x: int, y: int):
        neighbors = []
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_start(self):
        return self.start

    def get_goal(self):
        return self.goal