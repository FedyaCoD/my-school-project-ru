import numpy as np
import random

class QLearningAgent:
    def __init__(self, maze,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 reward_exit: float = 100.0,
                 reward_step: float = -1.0,
                 reward_wall: float = -10.0):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_exit = reward_exit
        self.reward_step = reward_step
        self.reward_wall = reward_wall

        self.q_table = np.zeros((maze.size, maze.size, 4))
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _get_action_vector(self, idx: int) -> tuple:
        return self.actions[idx]

    def choose_action(self, state: tuple, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            x, y = state
            return int(np.argmax(self.q_table[x, y]))

    def learn(self, episodes: int = 500, max_steps_per_episode: int = 200,
              dynamic_change_freq: int = None, on_maze_changed=None):
        start = self.maze.get_start()
        goal = self.maze.get_goal()
        steps_history = []

        for ep in range(episodes):
            if dynamic_change_freq and ep > 0 and ep % dynamic_change_freq == 0:
                if on_maze_changed:
                    on_maze_changed()

            state = start
            total_steps = 0
            done = False
            while not done and total_steps < max_steps_per_episode:
                action_idx = self.choose_action(state, self.epsilon)
                dx, dy = self._get_action_vector(action_idx)
                x, y = state
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.maze.size and 0 <= ny < self.maze.size) or self.maze.is_wall(nx, ny):
                    next_state = state
                    reward = self.reward_wall
                else:
                    next_state = (nx, ny)
                    if next_state == goal:
                        reward = self.reward_exit
                        done = True
                    else:
                        reward = self.reward_step

                old_q = self.q_table[x, y, action_idx]
                if not done:
                    next_max_q = np.max(self.q_table[next_state[0], next_state[1]])
                else:
                    next_max_q = 0.0
                new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
                self.q_table[x, y, action_idx] = new_q

                state = next_state
                total_steps += 1

            if done:
                steps_history.append(total_steps)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return steps_history

    def get_path(self):
        start = self.maze.get_start()
        goal = self.maze.get_goal()
        path = [start]
        state = start
        visited = set()
        visited.add(state)
        max_steps = self.maze.size * self.maze.size * 2

        for _ in range(max_steps):
            if state == goal:
                return path, len(path) - 1
            x, y = state
            best_action = np.argmax(self.q_table[x, y])
            dx, dy = self._get_action_vector(int(best_action))
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.maze.size and 0 <= ny < self.maze.size) or self.maze.is_wall(nx, ny):
                break
            next_state = (nx, ny)
            if next_state in visited:
                break
            visited.add(next_state)
            path.append(next_state)
            state = next_state

        return None, 0