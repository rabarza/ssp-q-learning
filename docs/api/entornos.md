# API — Entornos

Los entornos están definidos en `RLib/environments/`. Implementan la interfaz de interacción agente-entorno: `reset`, `action_set`, `take_action` y `check_state`.

---

## `SSPEnv`

Entorno para el problema del Camino Más Corto Estocástico sobre un grafo dirigido.

**Módulo:** `RLib.environments.ssp`

### Constructor

```python
SSPEnv(
    graph: nx.Graph,
    start_state: Any,
    terminal_state: Any,
    costs_distribution: str = "lognormal",
    shortest_path: list = None,
)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `graph` | `nx.Graph` | Grafo dirigido con atributos `length` y opcionalmente `speed_kph` en los arcos |
| `start_state` | `Any` | Nodo de inicio |
| `terminal_state` | `Any` | Nodo terminal. Se le añade automáticamente un arco recurrente de costo 0 |
| `costs_distribution` | `str` | Distribución para muestrear costos: `"lognormal"`, `"normal"`, `"uniform"` |
| `shortest_path` | `list` | Camino óptimo precalculado (opcional, para referencia) |

### Atributos

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `graph` | `nx.Graph` | Copia del grafo con el arco recurrente añadido |
| `start_state` | `Any` | Estado inicial |
| `terminal_state` | `Any` | Estado terminal |
| `current_state` | `Any` | Estado actual del entorno |
| `num_states` | `int` | Número de nodos en el grafo |
| `num_actions` | `dict` | Diccionario `{nodo: número_de_acciones}` |
| `costs_distribution` | `str` | Distribución de costos configurada |

### Métodos

#### `reset()`

Reinicia el entorno: devuelve `current_state` al estado inicial.

```python
env.reset()
```

#### `action_set(state) → list`

Devuelve la lista de nodos vecinos (acciones posibles) desde `state`.

```python
actions = env.action_set(node)
```

#### `check_state(state) → bool`

Devuelve `True` si `state` es el estado terminal.

#### `take_action(state, action) → (next_state, reward, terminated, info)`

Ejecuta la acción `action` (nodo vecino) desde `state`. Muestrea el costo del arco según `costs_distribution`.

| Retorno | Tipo | Descripción |
|---------|------|-------------|
| `next_state` | `Any` | Nodo siguiente |
| `reward` | `float` | Recompensa = −costo del arco |
| `terminated` | `bool` | `True` si se llegó al nodo terminal |
| `info` | `str` | Cadena de texto con información del paso |

### Ejemplo

```python
from RLib.environments.ssp import SSPEnv
import networkx as nx

G = nx.DiGraph()
G.add_edge(0, 1, length=100, speed_kph=30)
G.add_edge(1, 2, length=200, speed_kph=25)

env = SSPEnv(G, start_state=0, terminal_state=2, costs_distribution="uniform")
env.reset()
next_state, reward, done, info = env.take_action(0, 1)
print(next_state, reward, done)
```

---

## `HardSSPEnv`

Variante de `SSPEnv` donde el grafo se modifica dinámicamente durante cada episodio para eliminar arcos que atajen hacia el camino óptimo.

**Módulo:** `RLib.environments.ssp`

### Constructor

```python
HardSSPEnv(
    graph: nx.Graph,
    start_state: Any,
    terminal_state: Any,
    costs_distribution: str = "lognormal",
    shortest_path: list = None,
)
```

Si `shortest_path` es `None`, se calcula automáticamente con `nx.shortest_path`.

### Comportamiento

- En cada `reset()`, se crea una copia del grafo original.
- Los arcos que van desde nodos fuera del camino óptimo hacia nodos intermedios del camino óptimo son **eliminados** del grafo de episodio.
- Esto fuerza al agente a explorar rutas alternativas antes de poder acceder al camino más corto.

### Métodos adicionales

| Método | Descripción |
|--------|-------------|
| `reset()` | Reinicia el entorno y recrea `current_graph` |
| `remove_edges_to_shortest_path()` | Elimina los arcos que apuntan a nodos del SP |
| `ensure_largest_component()` | Retiene solo el mayor componente fuertemente conexo |
| `get_path_edges(path) → list` | Devuelve los arcos del camino `path` como lista de tuplas |

---

## `Maze`

Entorno de laberinto representado como una matriz 2D.

**Módulo:** `RLib.environments.maze`

El laberinto es una cuadrícula donde cada celda puede ser:
- Piso transitable
- Pared
- Fuego (costo adicional)
- Pozo (termina el episodio)

Las acciones son 4 movimientos cardinales: arriba (0), abajo (1), izquierda (2), derecha (3).

### Constructor

```python
Maze(maze_array, start, goal, game="pit-walls")
```

| Parámetro | Descripción |
|-----------|-------------|
| `maze_array` | Matriz numpy con la representación del laberinto |
| `start` | Tupla `(fila, columna)` del estado inicial |
| `goal` | Tupla `(fila, columna)` del estado terminal |
| `game` | Tipo de laberinto: `"pit-walls"` o `"fire-walls"` |

### Funciones auxiliares

```python
from RLib.environments.maze import generate_maze, render_maze, dijkstra_validate

# Generar un laberinto aleatorio
maze_array, start, goal = generate_maze(rows=20, cols=20)

# Validar que exista un camino
valid = dijkstra_validate(maze_array, start, goal)

# Visualizar el laberinto
render_maze(maze_array, start, goal)
```

### Ejemplo completo

```python
from RLib.environments.maze import Maze, generate_maze, render_maze, dijkstra_validate

maze_array, start, goal = generate_maze(15, 15)
assert dijkstra_validate(maze_array, start, goal), "Sin camino válido"
render_maze(maze_array, start, goal)

env = Maze(maze_array, start, goal)
state = env.start_state()
next_state, reward = env.take_action(state, action=0)  # mover arriba
```

---

## Funciones auxiliares en `RLib.environments.ssp`

### `get_edge_cost`

```python
get_edge_cost(
    G, source, target,
    distribution="expectation-lognormal",
    avg_speed=25,
) → float
```

Devuelve el costo del arco `(source, target)`. Si `distribution` comienza con `"expectation-"`, devuelve la esperanza matemática; de lo contrario, muestrea de la distribución indicada.

### `get_cumulative_edges_cost`

```python
get_cumulative_edges_cost(graph, policy, source, target, distribution) → float
```

Calcula el costo acumulado de seguir la política `policy` desde `source` hasta `target`.
