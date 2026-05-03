import time
import heapq

class AStarSolver:
    def __init__(self, maze):
        self.maze = maze

    @staticmethod
    def heuristic(a, b):
        # Манхэттенское расстояние
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        start = self.maze.get_start()
        goal = self.maze.get_goal()
        start_time = time.time()

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # восстановление пути
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                elapsed = time.time() - start_time
                return path, elapsed, len(path) - 1

            for neighbor in self.maze.get_neighbors(current[0], current[1]):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        elapsed = time.time() - start_time
        return None, elapsed, 0