import numpy as np
from maze import Maze
from bfs_solver import BFSSolver
from astar_solver import AStarSolver
from q_learning_agent import QLearningAgent
from visualization import plot_results


def run_single_experiment(seed, maze_size=10, wall_prob=0.3, episodes=500,
                          dynamic_change_freq=100, dynamic_change_prob=0.1,
                          max_steps_per_episode=200):
    """
    Выполняет один полный эксперимент:
    - генерирует лабиринт с заданным seed,
    - обучает Q-learning с динамическими изменениями,
    - возвращает длины путей BFS, A* и Q-learning в финальном лабиринте,
    - а также историю шагов Q-learning.
    """
    np.random.seed(seed)
    maze = Maze(size=maze_size, wall_probability=wall_prob)

    # BFS и A* на начальном лабиринте (только для информации, не используется в статистике).
    # Но мы их не сохраняем – важны только финальные после всех изменений.

    # Q-learning агент
    q_agent = QLearningAgent(maze,
                             alpha=0.1,
                             gamma=0.9,
                             epsilon=1.0,
                             epsilon_min=0.01,
                             epsilon_decay=0.995,
                             reward_exit=100.0,
                             reward_step=-1.0,
                             reward_wall=-10.0)

    # Функция изменения лабиринта (вызывается каждые dynamic_change_freq эпизодов).
    # Сохраняем ссылку на лабиринт для возможности изменения внутри агента.
    def on_maze_changed():
        maze.dynamic_update(prob=dynamic_change_prob)

    steps_history = q_agent.learn(episodes=episodes,
                                  max_steps_per_episode=max_steps_per_episode,
                                  dynamic_change_freq=dynamic_change_freq,
                                  on_maze_changed=on_maze_changed)

    # Получаем путь Q-learning после обучения
    q_path, q_steps = q_agent.get_path()

    # Пересчитываем BFS и A* на финальном лабиринте
    bfs_solver = BFSSolver(maze)
    bfs_path, _, bfs_steps = bfs_solver.solve()

    astar_solver = AStarSolver(maze)
    astar_path, _, astar_steps = astar_solver.solve()

    return {
        'bfs_steps': bfs_steps,
        'astar_steps': astar_steps,
        'q_steps': q_steps,
        'steps_history': steps_history,
        'final_maze': maze,  # сохраняем лабиринт для последней визуализации
        'bfs_path': bfs_path,
        'astar_path': astar_path,
        'q_path': q_path
    }


def main():
    # Параметры эксперимента
    num_runs = 10  # количество независимых запусков
    maze_size = 10
    wall_prob = 0.3
    episodes = 500
    dynamic_change_freq = 100
    dynamic_change_prob = 0.1

    print(f"=== Запуск {num_runs} независимых экспериментов ===")
    print(f"Размер лабиринта: {maze_size}x{maze_size}, плотность стен: {wall_prob}")
    print(
        f"Динамические изменения: каждые {dynamic_change_freq} эпизодов, вероятность переключения клетки {dynamic_change_prob}")
    print(f"Эпизодов Q-learning: {episodes}\n")

    results = []
    for run in range(num_runs):
        print(f"Запуск {run + 1}/{num_runs}...")
        res = run_single_experiment(seed=run,
                                    maze_size=maze_size,
                                    wall_prob=wall_prob,
                                    episodes=episodes,
                                    dynamic_change_freq=dynamic_change_freq,
                                    dynamic_change_prob=dynamic_change_prob)
        results.append(res)
        print(
            f"  BFS={res['bfs_steps']}, A*={res['astar_steps']}, Q-learning={res['q_steps']}, успешных эпизодов={len(res['steps_history'])}")

    # Сбор статистики
    bfs_steps_list = [r['bfs_steps'] for r in results]
    astar_steps_list = [r['astar_steps'] for r in results]
    q_steps_list = [r['q_steps'] for r in results if r['q_steps'] is not None]

    bfs_mean, bfs_std = np.mean(bfs_steps_list), np.std(bfs_steps_list)
    astar_mean, astar_std = np.mean(astar_steps_list), np.std(astar_steps_list)
    q_mean, q_std = np.mean(q_steps_list), np.std(q_steps_list)

    print("\n" + "=" * 50)
    print("ИТОГОВАЯ СТАТИСТИКА ПО 10 ЗАПУСКАМ")
    print("=" * 50)
    print(f"A* (финальный лабиринт):        {astar_mean:.1f} ± {astar_std:.1f} шагов")
    print(f"BFS (финальный лабиринт):       {bfs_mean:.1f} ± {bfs_std:.1f} шагов")
    print(f"Q-learning (финальный лабиринт): {q_mean:.1f} ± {q_std:.1f} шагов")
    print(f"Относительная эффективность Q-learning: {q_mean / bfs_mean:.2f} x от BFS")

    # Визуализация последнего запуска (для примера)
    last = results[-1]
    plot_results(last['final_maze'],
                 last['bfs_path'],
                 last['astar_path'],
                 last['q_path'],
                 last['steps_history'],
                 bfs_mean)  # используем средний оптимум для горизонтальной линии

    # Дополнительный вывод: лучший результат Q-learning
    all_best = []
    for r in results:
        if r['steps_history']:
            all_best.append(min(r['steps_history']))
    if all_best:
        print(f"\nЛучшее количество шагов Q-learning за всё обучение (среди всех запусков): {min(all_best)}")


if __name__ == "__main__":
    main()