import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from typing import Tuple, Union
from RLib.action_selectors import (
    ActionSelector,
    EpsilonGreedyActionSelector,
    BoltzmannSelector,
    UCB1ActionSelector,
)
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from RLib.utils.plots import plot_results_per_episode_comp_matplotlib


class QAgentMaze:
    """
    Clase base para resolver laberintos usando Q-Learning.
    """

    def __init__(
        self,
        environment,
        action_selector: ActionSelector = EpsilonGreedyActionSelector(0.1),
        alpha: Union[str, float] = 0.1,
        gamma: float = 1,
    ):
        """
        Parameters
        ----------
        environment: objeto del entorno del laberinto.

        alpha: Union[str, float]
            fórmula para calcular el valor de alpha. Puede ser cualquier expresión matemática válida que contenga las variables 'N(s,a)', 'N(s)', 't', 'sqrt' y 'log'.

        gamma: float
            factor de descuento. Debe ser un valor entre 0 y 1.

        action_selector: ActionSelector (objeto de la clase ActionSelector)
            selector de acciones.

        game: str
            tipo de laberinto ("fire-walls" o "pit-walls").
        """
        self.env = environment
        self.num_states = np.prod(environment.maze.shape)
        self.num_actions = 4
        self.alpha = alpha
        self.gamma = gamma
        self.action_selector = action_selector
        self.strategy = self.action_selector.strategy
        self.actual_episode = 0
        self.num_episodes = 0

        # Inicializar contadores y tabla Q
        self.visits_actions = np.zeros((self.num_states, self.num_actions))
        self.visits_states = np.zeros(self.num_states)
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.steps = []
        self.scores = []
        self.avg_scores = []
        self.regret = []
        self.average_regret = []

    def eval_alpha(self, state, action):
        """
        Retorna el valor de alpha para el estado y acción indicados en el tiempo de actualización.
        """
        N_sa = self.visits_actions[state, action] + 1
        N_s = self.visits_states[state] + 1
        t = len(self.steps) + 1
        context = {"N_sa": N_sa, "N_s": N_s, "t": t, "sqrt": np.sqrt, "log": np.log}
        try:
            # Reemplazar 'N(s,a)' y 'N(s)' en la expresión de alpha
            alpha_expr = str(self.alpha).replace("N(s,a)", "N_sa").replace("N(s)", "N_s")
            return eval(alpha_expr, context)
        except:
            return float(self.alpha)

    def increment_visits(self, state, action):
        """
        Incrementa los contadores de visitas para el estado y la acción.
        """
        self.visits_states[state] += 1
        self.visits_actions[state, action] += 1

    def select_action(self, state):
        """
        Selecciona la siguiente acción a tomar usando el selector de acciones.
        """
        return self.action_selector.select_action(self, state)

    def argmax_q_table(self, state):
        """
        Retorna la acción con el mayor valor Q(s,a) para un estado dado.
        """
        return np.argmax(self.q_table[state])

    def random_action(self, state):
        """
        Retorna una acción aleatoria para un estado dado.
        """
        return np.random.choice(self.num_actions)

    def action_set(self, state):
        """
        Retorna el conjunto de acciones disponibles para un estado dado.
        """
        return list(range(self.num_actions))

    def train(self, num_episodes):
        """
        Entrena al agente usando el algoritmo Q-Learning.
        """
        self.steps = np.zeros(num_episodes)
        self.scores = np.zeros(num_episodes)
        self.num_episodes = num_episodes  # Actualizar el atributo num_episodes

        for episode in tqdm(range(num_episodes), desc="Entrenando", ncols=100):
            total_score = 0
            state = self.env.start_state()
            self.actual_episode = episode  # Actualizar el atributo actual_episode
            # imprime laberinto y posición inicial
            while not self.env.terminal_state(state):
                action = self.select_action(state)
                next_state, reward = self.env.take_action(state, action)
                # imprimir el estado y la acción en forma de ubicacion
                # action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
                # print(f"Estado: {self.env.position(state)}, Acción: {action_map[action]}, Estado siguiente: {self.env.position(next_state)}")
                # print(self.env.maze)
                # self.env.render()
                # Si el estado es terminal, se asigna una recompensa de 0
                # Actualizar valores Q_table
                alpha = self.eval_alpha(state, action)
                q_old = self.q_table[state, action]
                q_new = q_old * (1 - alpha) + alpha * (
                    reward + self.gamma * np.max(self.q_table[next_state])
                )
                self.q_table[state, action] = q_new

                # Incrementar contadores
                self.increment_visits(state, action)

                # Ir al siguiente estado
                state = next_state

                # Actualizar métricas
                self.steps[episode] += 1
                total_score += reward

            self.scores[episode] = total_score / max(self.steps[episode], 1)

    def best_path(self, state):
        """
        Devuelve el mejor camino desde un estado inicial hasta el estado terminal.
        """
        if isinstance(state, int):
            state = self.env.position(state)
        state = self.env.index(state)
        path = []

        while not self.env.terminal_state(state):
            path.append(state)  # Agregar directamente el índice del estado
            action = np.argmax(self.q_table[state])
            state = self.env.take_action(state, action)[0]

        path.append(state)  # Agregar el estado terminal como índice
        return path

    def plot_steps_per_episode(self):
        """
        Grafica la cantidad de pasos que tardó cada episodio en llegar a un estado terminal.
        """
        plt.figure(dpi=100)
        plt.plot(range(len(self.steps)), self.steps)
        plt.title(self.env.game + "-" + self.strategy)
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.grid()
        plt.show()

    def results(self):
        """
        Devuelve los resultados del agente como un diccionario.
        """
        return {
            "steps": self.steps,
            "scores": self.scores,
            "avg_scores": self.avg_scores,
            "regret": self.regret,
            "average_regret": self.average_regret,
        }


if __name__ == "__main__":
    from RLib.environments.maze import Maze, generate_maze, render_maze, dijkstra_validate

    # Parámetros del laberinto
    rows, cols = 20, 20

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
