# API — Agentes

Los agentes están definidos en `RLib/agents/`. Todos implementan el loop de Q-Learning e interactúan con un entorno a través de su interfaz `take_action` / `action_set`.

---

## Jerarquía de clases

```
QAgent                        (base para grafos)
└── QAgentSSP                 (Q-Learning para SSP en grafos)
    ├── QAgentSSPSarsa0       (SARSA(0))
    └── QAgentSSPExpectedSarsa0  (Expected-SARSA(0))

QAgentMaze                    (Q-Learning para laberintos)
```

---

## `QAgent`

Clase base para agentes Q-Learning sobre grafos. Define la lógica común: tabla Q, contadores de visitas, tasa de aprendizaje dinámica y actualización Q.

**Módulo:** `RLib.agents.ssp`

### Constructor

```python
QAgent(
    action_selector: ActionSelector = EpsilonGreedyActionSelector(0.1),
    alpha: Union[str, float] = 0.1,
    gamma: float = 1,
)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `action_selector` | `ActionSelector` | Estrategia de selección de acciones |
| `alpha` | `str` o `float` | Tasa de aprendizaje. Puede ser constante (`0.1`) o una expresión dinámica (`"250 / (N(s,a) + 250)"`) |
| `gamma` | `float` | Factor de descuento ∈ [0, 1]. Usar `1` para SSP estándar |

**Variables en expresiones de `alpha`:**

| Variable | Significado |
|----------|-------------|
| `N(s,a)` | Visitas al par (estado, acción) |
| `N(s)` | Visitas al estado |
| `t` | Episodio actual |
| `sqrt`, `log` | Funciones matemáticas |

### Métodos principales

| Método | Descripción |
|--------|-------------|
| `select_action(state)` | Selecciona la acción según el selector configurado |
| `argmax_q_table(state)` | Devuelve la acción con mayor Q(s,·) |
| `update_q_table(state, action, reward, next_state)` | Actualiza Q(s,a) con Q-Learning |
| `eval_alpha(state, action)` | Evalúa la expresión de alpha en el contexto actual |
| `save(path)` | Guarda el agente y sus resultados en disco |
| `results()` | Devuelve un diccionario con las métricas del entrenamiento |

---

## `QAgentSSP`

Agente que resuelve el **Stochastic Shortest Path Problem** sobre un grafo representado como `SSPEnv`.

**Módulo:** `RLib.agents.ssp`

### Constructor

```python
QAgentSSP(
    environment: SSPEnv,
    action_selector: ActionSelector = EpsilonGreedyActionSelector(0.1),
    alpha: Union[str, float] = 0.1,
    gamma: float = 1,
)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `environment` | `SSPEnv` | Entorno de aprendizaje |
| `action_selector` | `ActionSelector` | Estrategia de selección de acciones |
| `alpha` | `str` o `float` | Tasa de aprendizaje |
| `gamma` | `float` | Factor de descuento |

### Método `train`

```python
agent.train(
    num_episodes: int = 100,
    shortest_path: list = None,
    q_star: dict = None,
    verbose: bool = False,
    use_streamlit: bool = False,
)
```

| Parámetro | Descripción |
|-----------|-------------|
| `num_episodes` | Número de episodios de entrenamiento |
| `shortest_path` | Camino óptimo (para calcular error restringido al SP) |
| `q_star` | Tabla Q\* (para calcular norma máxima de error) |
| `verbose` | Imprimir información de cada paso |
| `use_streamlit` | Usar barra de progreso de Streamlit |

### Método `best_path`

```python
path = agent.best_path(state=None)
```

Devuelve el camino greedy desde `state` (o el estado inicial si `None`) hasta el estado terminal.

### Método `results`

Devuelve un diccionario con las siguientes claves:

| Clave | Descripción |
|-------|-------------|
| `steps` | Lista de pasos por episodio |
| `scores` | Lista de scores (costos acumulados negativos) por episodio |
| `avg_scores` | Media acumulada de scores |
| `regret` | Regret acumulado por episodio |
| `average_regret` | Regret promedio por episodio |
| `max_norm_error` | Error de norma máxima ∥Qt − Q\*∥∞ por episodio |
| `max_norm_error_shortest_path` | Error de norma máxima restringido al SP |
| `max_norm_error_normalized` | Error normalizado por el costo óptimo |
| `optimal_cost` | Costo óptimo Q\*(s₀, π\*(s₀)) |
| `optimal_paths` | Número de episodios donde el agente siguió el camino óptimo |

### Ejemplo

```python
from RLib.agents.ssp import QAgentSSP
from RLib.environments.ssp import SSPEnv
from RLib.action_selectors import UCB1ActionSelector

env = SSPEnv(graph, origin_node, target_node, "uniform", shortest_path)
agent = QAgentSSP(
    environment=env,
    action_selector=UCB1ActionSelector(c=1.0),
    alpha="250 / (N(s,a) + 250)",
)
agent.train(num_episodes=5000, shortest_path=shortest_path, q_star=q_star)
print(agent.best_path())
```

---

## `QAgentSSPSarsa0`

Variante de `QAgentSSP` que usa **SARSA(0)** en lugar de Q-Learning para actualizar la tabla Q.

**Módulo:** `RLib.agents.ssp`

La diferencia está en `update_q_table`: en lugar de usar $\max_{a'} Q(s', a')$, usa el valor de la acción *realmente seleccionada* en el siguiente estado:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s,a) \right]$$

---

## `QAgentSSPExpectedSarsa0`

Variante de `QAgentSSP` que usa **Expected-SARSA(0)**: actualiza la tabla Q con el valor esperado (bajo la política actual) en el siguiente estado:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \sum_{a'} \pi(a'|s') \cdot Q(s', a') - Q(s,a) \right]$$

> Requiere que el `action_selector` implemente el método `get_probabilities(agent, state)`.

---

## `QAgentMaze`

Agente Q-Learning para **laberintos** representados como matrices 2D.

**Módulo:** `RLib.agents.maze`

### Constructor

```python
QAgentMaze(
    environment: Maze,
    action_selector: ActionSelector = EpsilonGreedyActionSelector(0.1),
    alpha: Union[str, float] = 0.1,
    gamma: float = 1,
)
```

Las 4 acciones posibles son: `0` (arriba), `1` (abajo), `2` (izquierda), `3` (derecha).

### Método `train`

```python
agent.train(num_episodes: int)
```

### Método `best_path`

```python
path = agent.best_path(state)
```

Devuelve la secuencia de índices de estados desde `state` hasta el estado terminal.

### Ejemplo

```python
from RLib.agents.maze import QAgentMaze
from RLib.environments.maze import Maze, generate_maze
from RLib.action_selectors import EpsilonGreedyActionSelector

maze_array, start, goal = generate_maze(20, 20)
env = Maze(maze_array, start, goal)

agent = QAgentMaze(
    environment=env,
    action_selector=EpsilonGreedyActionSelector(0.1),
    alpha="1000 / (N(s,a) + 1000)",
)
agent.train(num_episodes=2000)
print("Mejor camino:", agent.best_path(env.start_state()))
```
