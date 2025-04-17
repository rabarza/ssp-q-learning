import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402

from RLib.environments.maze import Maze, generate_maze, render_maze, dijkstra_validate
from RLib.action_selectors import (
    EpsilonGreedyActionSelector,
    BoltzmannSelector,
    UCB1ActionSelector,
)
from RLib.agents.maze import QAgentMaze
from RLib.utils.plots import plot_results_per_episode_comp_matplotlib

if __name__ == "__main__":
    # Parámetros del laberinto
    rows, cols = 30, 30

    # Generar el laberinto
    maze_array, start, goal = generate_maze(rows, cols)

    # Validar que exista un camino entre el inicio y la meta
    if not dijkstra_validate(maze_array, start, goal):
        raise ValueError(
            "El laberinto generado no tiene un camino válido entre inicio y meta."
        )

    # Renderizar el laberinto
    render_maze(maze_array, start, goal)

    # Crear el entorno del laberinto
    environment = Maze(maze_array, start, goal)

    # Crear el selector de acciones y el agente Q-Learning
    eps_selector = EpsilonGreedyActionSelector(epsilon=0.1)
    alpha_expr = "1000 / (1000 + N(s,a))"

    q_agent = QAgentMaze(
        environment=environment, alpha=alpha_expr, action_selector=eps_selector
    )

    # Crear otros selectores de acción
    boltzmann_selector = BoltzmannSelector(0.1)
    ucb1_selector = UCB1ActionSelector(2)

    # Crear agentes con diferentes estrategias
    boltzmann_agent = QAgentMaze(
        environment=environment, alpha=alpha_expr, action_selector=boltzmann_selector
    )
    ucb1_agent = QAgentMaze(
        environment=environment, alpha=alpha_expr, action_selector=ucb1_selector
    )

    # Entrenar agentes
    num_episodes = 1000
    print("Entrenando ε-greedy...")
    q_agent.train(num_episodes=num_episodes)
    print("Entrenando Boltzmann...")
    boltzmann_agent.train(num_episodes=num_episodes)
    print("Entrenando UCB1...")
    ucb1_agent.train(num_episodes=num_episodes)

    # Obtener el mejor camino desde el inicio hasta la meta
    best_path = q_agent.best_path(environment.start_state())

    # Renderizar el laberinto con el mejor camino
    environment.render(path=[environment.position(state) for state in best_path])

    # Graficar los pasos por episodio
    q_agent.plot_steps_per_episode()

    # Comparar resultados de los tres agentes
    agents_list = [q_agent, boltzmann_agent, ucb1_agent]
    plot_results_per_episode_comp_matplotlib(agents_list, criteria="steps", dpi=150)
