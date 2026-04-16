# API — Utilidades

El módulo `RLib/utils/` proporciona herramientas de soporte para Dijkstra, visualización, serialización y manejo de tablas Q.

---

## Dijkstra (`RLib.utils.dijkstra`)

### `dijkstra_shortest_path`

Calcula el camino más corto desde `source` hasta `target` usando Dijkstra. Los pesos de los arcos se calculan como la **esperanza** del tiempo de viaje según la distribución indicada.

```python
dijkstra_shortest_path(
    graph: nx.Graph,
    source: Any,
    target: Any,
    avg_speed: float = 25,
    distribution: str = "lognormal",
) → (shortest_distances, parents, shortest_path)
```

| Retorno | Tipo | Descripción |
|---------|------|-------------|
| `shortest_distances` | `dict` | `{nodo: distancia_mínima}` desde el origen |
| `parents` | `dict` | `{nodo: padre}` en el árbol de caminos mínimos |
| `shortest_path` | `list` | Lista de nodos del camino más corto de `source` a `target` |

---

### `get_optimal_policy_and_q_star`

Calcula la **política óptima** y la **tabla Q\*** mediante una búsqueda de Dijkstra desde el nodo de destino en el grafo invertido.

```python
get_optimal_policy_and_q_star(
    graph: nx.MultiDiGraph,
    dest_node: Any,
    distribution: str = "lognormal",
    st: bool = False,
) → (policy, Q_star)
```

| Parámetro | Descripción |
|-----------|-------------|
| `dest_node` | Nodo de destino (estado terminal) |
| `distribution` | Distribución para calcular esperanzas de costo |
| `st` | Si `True`, usa barra de progreso de Streamlit |

| Retorno | Tipo | Descripción |
|---------|------|-------------|
| `policy` | `dict` | `{nodo: acción_óptima}` para todos los nodos |
| `Q_star` | `dict` | `{nodo: {acción: valor_Q*}}` para todos los nodos |

Los valores en `Q_star` son **negativos** (recompensas = −costos).

### Ejemplo

```python
from RLib.utils.dijkstra import get_optimal_policy_and_q_star, get_shortest_path_from_policy

policy, q_star = get_optimal_policy_and_q_star(graph, target_node, "uniform")
shortest_path = get_shortest_path_from_policy(policy, origin_node, target_node)
```

---

### `get_shortest_path_from_policy`

Reconstruye el camino más corto desde `source` hasta `target` siguiendo la política `policy`.

```python
get_shortest_path_from_policy(
    policy: dict,
    source: Any,
    target: Any,
) → List[Any]
```

**Ejemplo:**
```python
>>> policy = {0: 1, 1: 3, 2: 3, 3: 4, 4: 4}
>>> get_shortest_path_from_policy(policy, 0, 4)
[0, 1, 3, 4]
```

---

### `get_q_table_for_path`

Devuelve la tabla Q restringida únicamente a los pares estado-acción del camino `path`.

```python
get_q_table_for_path(
    q_table: dict,
    path: list,
) → dict
```

Útil para calcular el error de la norma máxima restringido al camino más corto.

---

### `get_path_as_stateactions_dict`

Convierte un camino (lista de nodos) en un diccionario de política `{estado: acción_siguiente}`.

```python
get_path_as_stateactions_dict(path: list) → dict
```

**Ejemplo:**
```python
>>> get_path_as_stateactions_dict([0, 1, 3, 4])
{0: 1, 1: 3, 3: 4, 4: 4}
```

---

## Tablas Q (`RLib.utils.tables`)

### Inicialización

```python
from RLib.utils.tables import (
    dict_states_zeros,
    dict_states_actions_zeros,
    dict_states_actions_random,
    dict_states_actions_constant,
)

# {estado: 0}
visits = dict_states_zeros(graph)

# {estado: {acción: 0}}
q_table = dict_states_actions_zeros(graph)

# {estado: {acción: valor_aleatorio ∈ [0,1)}}
q_table = dict_states_actions_random(graph)

# {estado: {acción: constante}}
q_table = dict_states_actions_constant(graph, constant=-100.0)
```

### Norma máxima

```python
from RLib.utils.tables import max_norm

# ||Q - Q*||∞ sobre todos los pares (s,a)
error = max_norm(q_table, q_star)

# ||Q - Q*||∞ restringido a los pares del camino shortest_path
error_sp = max_norm(q_table, q_star, path=shortest_path)
```

### Operaciones sobre tablas

```python
from RLib.utils.tables import max_q_table, argmax_q_table, max_value_in_dict

# max_a Q(s, a)
v = max_q_table(q_table, state)

# argmax_a Q(s, a)
best_action = argmax_q_table(q_table, state)

# valor máximo en todo el diccionario anidado
global_max = max_value_in_dict(q_table)
```

---

## Archivos (`RLib.utils.files`)

### `save_model_results`

Guarda el agente y sus tablas Q en disco.

```python
from RLib.utils.files import save_model_results

save_model_results(agent, results_path="results/my_experiment/")
```

Crea los siguientes archivos en `results_path`:
- `QAgentSSP_<strategy>_<episodes>_<distribution>_<timestamp>.pickle` — agente serializado
- `QTable_*.json` — tabla Q completa
- `QTableSP_*.json` — tabla Q restringida al camino más corto
- `VisitsStateAction_*.json` — tabla de visitas por (estado, acción)

### `load_model_results`

Carga un agente guardado previamente.

```python
from RLib.utils.files import load_model_results

agent = load_model_results("QAgentSSP_e-greedy_5000_uniform_12-30-00_01-01-25.pickle", "results/")
```

### `find_files_by_keyword`

Busca archivos `.pickle` en una carpeta que contengan una palabra clave.

```python
from RLib.utils.files import find_files_by_keyword

files = find_files_by_keyword("UCB", "results/")
```

### `serialize_and_save_table`

Serializa y guarda una tabla Q en formato JSON.

```python
from RLib.utils.files import serialize_and_save_table

serialize_and_save_table(q_star, "results/", "q_star.json")
```

### `download_graph`

Descarga un grafo de OpenStreetMap usando coordenadas de bounding box. Si el archivo ya existe en disco, lo carga directamente.

```python
from RLib.utils.files import download_graph

graph = download_graph(
    north=-33.4283,
    south=-33.6298,
    east=-70.5099,
    west=-70.9051,
    filepath="data/santiago",
    format="graphml",
)
```

### `save_graph_plot_as_pdf`

Genera y guarda una visualización del grafo (con nodos de origen y destino destacados) como archivo PDF.

```python
from RLib.utils.files import save_graph_plot_as_pdf

save_graph_plot_as_pdf(graph, origin_node, target_node, "results/", "Santiago")
```

---

## Visualización (`RLib.utils.plots`)

El módulo `RLib/utils/plots.py` contiene funciones para graficar los resultados de los experimentos, incluyendo:

- Comparación de la evolución del error de norma máxima entre estrategias
- Gráficos de regret acumulado y promedio
- Comparación de scores por episodio entre múltiples agentes

Estas funciones se usan principalmente en los scripts `*_results.py` de la carpeta `scripts/`.
