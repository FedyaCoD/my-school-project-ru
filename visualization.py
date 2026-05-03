import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_results(maze, bfs_path, astar_path, q_path, steps_history, optimal_steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Лабиринт
    cmap = ListedColormap(['white', 'black'])
    ax1.imshow(maze.grid, cmap=cmap, interpolation='none')
    ax1.set_title("Лабиринт и найденные пути")
    ax1.set_xlabel("Столбцы")
    ax1.set_ylabel("Строки")

    start_x, start_y = maze.get_start()
    goal_x, goal_y = maze.get_goal()
    ax1.plot(start_y, start_x, 'go', markersize=12, label='Старт')
    ax1.plot(goal_y, goal_x, 'ro', markersize=12, label='Цель')

    if bfs_path:
        bfs_x = [p[0] for p in bfs_path]
        bfs_y = [p[1] for p in bfs_path]
        ax1.plot(bfs_y, bfs_x, 'g-', linewidth=2, label='BFS (эталон)')

    if astar_path:
        astar_x = [p[0] for p in astar_path]
        astar_y = [p[1] for p in astar_path]
        ax1.plot(astar_y, astar_x, 'b--', linewidth=2, label='A*')

    if q_path:
        q_x = [p[0] for p in q_path]
        q_y = [p[1] for p in q_path]
        ax1.plot(q_y, q_x, 'r-.', linewidth=2, label='Q-learning')

    ax1.legend()
    ax1.grid(False)

    # График сходимости
    if steps_history:
        episodes = range(1, len(steps_history) + 1)
        ax2.plot(episodes, steps_history, 'b-', alpha=0.7, label='Шаги Q-learning (успешные эпизоды)')
        if optimal_steps > 0:
            ax2.axhline(y=optimal_steps, color='g', linestyle='--', label=f'Оптимум (BFS/A*): {optimal_steps} шагов')
        ax2.set_xlabel("Эпизод")
        ax2.set_ylabel("Количество шагов до выхода")
        ax2.set_title("Сходимость Q-learning (с динамическими препятствиями)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Агент не достиг цели ни разу", ha='center', va='center')
        ax2.set_title("Сходимость Q-learning (нет успешных эпизодов)")

    plt.tight_layout()
    plt.show()