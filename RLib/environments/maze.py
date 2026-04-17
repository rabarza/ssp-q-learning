import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import heapq


############ LABERINTO  ############
# Función para renderizar el laberinto
def render_maze(maze: np.array, start: tuple, end: tuple) -> None:
    """
    Renderiza un laberinto con paredes en negro, el punto de inicio en rojo, y la meta en verde.

    Parameters
    ----------
    maze: np.array
        Matriz que representa el laberinto.
    start: tuple
        Coordenadas del punto de inicio (fila, columna).
    end: tuple
        Coordenadas del punto final o meta (fila, columna).
    """
    render_grid = np.zeros_like(maze)

    # Asignar colores (0: fondo, 1: paredes, 2: inicio, 3: meta)
    render_grid[maze == -100] = 1  # Paredes
    render_grid[start] = 2  # Inicio
    render_grid[end] = 3  # Meta

    cmap_colors = [
        (1, 1, 1),  # Blanco para caminos (fondo)
        (0, 0, 0),  # Negro para paredes
        (1, 0, 0),  # Rojo para el inicio
        (0, 1, 0),  # Verde para la meta
    ]
    cmap = ListedColormap(cmap_colors)

    plt.figure(figsize=(maze.shape[1] / 5, maze.shape[0] / 5))
    plt.imshow(render_grid, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.show()


# Función para validar si existe un camino entre inicio y meta
def dijkstra_validate(maze, start, end):
    """Valida si hay un camino entre el nodo de inicio y el nodo de destino usando Dijkstra con paredes de costo infinito."""
    rows, cols = maze.shape
    costs = [[float("inf")] * cols for _ in range(rows)]
    costs[start[0]][start[1]] = 0

    pq = [(0, start)]

    while pq:
        current_cost, (current_row, current_col) = heapq.heappop(pq)

        if (current_row, current_col) == end:
            return True

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = current_row + dr, current_col + dc

            if 0 <= new_row < rows and 0 <= new_col < cols:
                neighbor_cost = float("inf") if maze[new_row][new_col] == -100 else 1

                if current_cost + neighbor_cost < costs[new_row][new_col]:
                    costs[new_row][new_col] = current_cost + neighbor_cost
                    heapq.heappush(pq, (costs[new_row][new_col], (new_row, new_col)))

    return False


def generate_maze(rows, cols):
    start = (0, 0)
    end = (rows - 2, cols - 2)
    maze = [[-100] * cols for _ in range(rows)]

    # Corregir los valores del estado inicial y terminal
    maze[start[0]][start[1]] = -1  # Estado inicial
    maze[end[0]][end[1]] = 500  # Estado terminal
    stack = [start]
    visited = set([start])

    while stack:
        current_row, current_col = stack[-1]

        unvisited_neighbours = []
        if current_row > 1 and (current_row - 2, current_col) not in visited:
            unvisited_neighbours.append((current_row - 2, current_col))
        if current_row < rows - 2 and (current_row + 2, current_col) not in visited:
            unvisited_neighbours.append((current_row + 2, current_col))
        if current_col > 1 and (current_row, current_col - 2) not in visited:
            unvisited_neighbours.append((current_row, current_col - 2))
        if current_col < cols - 2 and (current_row, current_col + 2) not in visited:
            unvisited_neighbours.append((current_row, current_col + 2))

        if unvisited_neighbours:
            next_row, next_col = random.choice(unvisited_neighbours)

            maze[next_row][next_col] = -1
            maze[(current_row + next_row) // 2][(current_col + next_col) // 2] = -1

            visited.add((next_row, next_col))
            stack.append((next_row, next_col))
        else:
            stack.pop()

    if start[0] == 0:
        maze[1][start[1]] = -1
    elif start[0] == rows - 1:
        maze[rows - 2][start[1]] = -1
    elif start[1] == 0:
        maze[start[0]][1] = -1
    elif start[1] == cols - 1:
        maze[start[0]][cols - 2] = -1

    if end[0] == 0:
        maze[1][end[1]] = -1
    elif end[0] == rows - 1:
        maze[rows - 2][end[1]] = -1
    elif end[1] == 0:
        maze[end[0]][1] = -1
    elif end[1] == cols - 1:
        maze[end[0]][cols - 2] = -1

    for row in maze:
        row.insert(0, -100)
        row.append(-100) if rows <= 5 else None
    maze.insert(0, [-100] * len(maze[0]))
    maze.append([-100] * len(maze[0])) if cols <= 5 else None
    start = (1, 1)
    end = end[0] + 1, end[1] + 1

    maze = np.array(maze)

    if not dijkstra_validate(maze, start, end):
        raise ValueError(
            "El laberinto generado no tiene un camino válido entre inicio y meta."
        )

    return maze, start, end


class Maze:
    def __init__(self, maze, start, goal, game="fire-walls"):
        """
        Inicializa el laberinto, el punto de inicio y la meta.
        El laberinto es una matriz de numpy, el punto de inicio y la meta son tuplas (fila, columna).
        El juego puede ser 'fire-walls' o 'pit-walls'.
        """
        assert (
            maze.shape[0] > 0 and maze.shape[1] > 0
        ), "El laberinto no puede estar vacío"
        assert start[0] >= 0 and start[0] < maze.shape[0], "Fila de inicio fuera de rango"
        assert (
            start[1] >= 0 and start[1] < maze.shape[1]
        ), "Columna de inicio fuera de rango"
        assert goal[0] >= 0 and goal[0] < maze.shape[0], "Fila de meta fuera de rango"
        assert goal[1] >= 0 and goal[1] < maze.shape[1], "Columna de meta fuera de rango"
        assert game in [
            "fire-walls",
            "pit-walls",
        ], "El juego debe ser 'fire-walls' o 'pit-walls'"

        self.maze = maze
        self.start = start
        self.goal = goal
        self.game = game

    def start_state(self):
        return self.index(self.start)

    def terminal_state(self, state):
        match self.game:
            case "fire-walls":
                return state == self.index(self.goal)
            case "pit-walls":
                row, col = self.position(state)
                return self.maze[row, col] != -1
            case _:
                raise ValueError("A valid game was expected")

    def take_action(self, state, action):
        row, col = self.position(state)

        if action == 0:  # up
            row = max(row - 1, 0)
        elif action == 1:  # down
            row = min(row + 1, self.maze.shape[0] - 1)
        elif action == 2:  # left
            col = max(col - 1, 0)
        elif action == 3:  # right
            col = min(col + 1, self.maze.shape[1] - 1)
        else:
            raise ValueError("Not a valid action")

        next_state = self.index((row, col))
        reward = self.maze[row, col]
        return next_state, reward

    def index(self, position):
        # índice = fila * tamaño_columnas + columna
        index = position[0] * self.maze.shape[1] + position[1]
        return index

    def position(self, index):
        # fila, columna = divmod(índice, tamaño_columnas)
        # fila = índice // tamaño_columnas
        # columna = índice % tamaño_columnas
        return divmod(index, self.maze.shape[1])

    def render(self, path=[]):

        start_point, end_point = (self.start, self.goal)

        rows = self.maze.shape[0]
        cols = self.maze.shape[1]

        # Crear una figura y un eje
        fig, ax = plt.subplots()

        # Configurar el tamaño de la figura en función del tamaño del laberinto
        fig.set_size_inches(cols, rows)

        # Configurar límites del eje
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)

        # Ocultar ejes
        ax.set_axis_off()

        # Dibujar las paredes
        for row in range(rows):
            for col in range(cols):
                if self.maze[row, col] == -100:
                    rect = Rectangle((col, row), 1, 1, facecolor="black")
                    ax.add_patch(rect)

        # Dibujar el camino
        if path:
            for cell in path:
                path_rect = Rectangle(
                    (cell[1], cell[0]), 1, 1, facecolor="palegreen"
                )
                ax.add_patch(path_rect)

        # Dibujar el punto de inicio
        start_row, start_col = start_point
        start_rect = Rectangle((start_col, start_row), 1, 1, facecolor="red")
        ax.add_patch(start_rect)

        # Dibujar la meta
        end_row, end_col = end_point
        end_rect = Rectangle((end_col, end_row), 1, 1, facecolor="lime")
        ax.add_patch(end_rect)

        # Mostrar la figura
        plt.show()
