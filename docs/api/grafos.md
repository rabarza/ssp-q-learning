# API — Grafos

Los generadores de grafos están definidos en `RLib/graphs/`.

---

## `create_perceptron_graph`

Crea un grafo dirigido en capas (estructura de perceptrón multicapa) donde todos los nodos de una capa están conectados a todos los nodos de la capa siguiente.

**Módulo:** `RLib.graphs.perceptron`

```python
create_perceptron_graph(
    nodes_by_layer: List[int] = [1, 1],
    min_length: int = 1,
    max_length: int = 20,
    seed: int = 20,
) → nx.DiGraph
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `nodes_by_layer` | `List[int]` | Número de nodos por capa. Ej: `[1, 5, 5, 1]` crea 4 capas |
| `min_length` | `int` | Longitud mínima de los arcos (en metros) |
| `max_length` | `int` | Longitud máxima de los arcos (en metros) |
| `seed` | `int` | Semilla del generador aleatorio para reproducibilidad |

**Retorna:** `nx.DiGraph` con nodos numerados. El nodo de entrada es `1` y el nodo de salida es `0`.

**Raises:**
- `TypeError` si `nodes_by_layer` no es una lista de enteros.
- `ValueError` si hay menos de 2 capas o alguna capa tiene 0 nodos.

### Ejemplo

```python
from RLib.graphs.perceptron import create_perceptron_graph

# Grafo: 1 nodo entrada → 10 ocultos → 10 ocultos → 1 salida
graph = create_perceptron_graph(
    nodes_by_layer=[1, 10, 10, 1],
    min_length=100,
    max_length=5000,
    seed=42,
)
print(f"Nodos: {graph.number_of_nodes()}, Arcos: {graph.number_of_edges()}")
# Nodos: 22, Arcos: 200

origin = 1
target = 0
```

---

## `create_hard_perceptron_graph`

Versión *hard* del grafo perceptrón: luego de crear el grafo, elimina los arcos que van desde nodos fuera del camino óptimo hacia nodos intermedios del camino óptimo, dificultando el aprendizaje.

**Módulo:** `RLib.graphs.perceptron`

```python
create_hard_perceptron_graph(
    nodes_by_layer: List[int] = [1, 1],
    min_length: int = 1,
    max_length: int = 20,
    costs_distribution: str = None,
    seed: int = 20,
) → nx.DiGraph
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `costs_distribution` | `str` | Distribución usada para calcular el camino más corto (`"uniform"`, `"lognormal"`, etc.) |

### Ejemplo

```python
from RLib.graphs.perceptron import create_hard_perceptron_graph

graph = create_hard_perceptron_graph(
    nodes_by_layer=[1, 5, 5, 1],
    min_length=100,
    max_length=3000,
    costs_distribution="uniform",
)
```

---

## `plot_network_graph`

Visualiza un grafo dirigido con nodos y arcos usando Plotly.

**Módulo:** `RLib.graphs.perceptron`

```python
plot_network_graph(
    graph: nx.DiGraph,
    use_annotations: bool = True,
    label_pos: float = 0.15,
)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `graph` | `nx.DiGraph` | Grafo a visualizar (debe tener atributo `pos` en los nodos) |
| `use_annotations` | `bool` | Mostrar la longitud de los arcos como etiquetas |
| `label_pos` | `float` | Posición relativa de la etiqueta a lo largo del arco (0–1) |

Abre la figura interactiva en el navegador.

### Ejemplo

```python
from RLib.graphs.perceptron import create_perceptron_graph, plot_network_graph

graph = create_perceptron_graph([1, 3, 3, 1])
plot_network_graph(graph, use_annotations=True, label_pos=0.6)
```

---

## `ensure_proper_policy`

Dado un grafo dirigido y un nodo destino, devuelve un subgrafo que contiene únicamente los nodos desde los que existe un camino hacia el destino.

**Módulo:** `RLib.graphs.city`

```python
ensure_proper_policy(
    graph: nx.DiGraph,
    target_node: Any,
) → nx.DiGraph
```

Esta función se usa durante el preprocesamiento de grafos de ciudades reales para garantizar que la política sea *proper* (que desde todo estado se pueda alcanzar el terminal).

### Ejemplo

```python
import osmnx as ox
from RLib.graphs.city import ensure_proper_policy

graph = ox.graph_from_place("Santiago, Chile", network_type="drive")
graph = ensure_proper_policy(graph, target_node=265665010)
```

---

## Grafos de ciudades reales (via `osmnx`)

El módulo `osmnx` permite descargar grafos de calles de OpenStreetMap. Los scripts en `scripts/` muestran el flujo típico de preprocesamiento:

```python
import osmnx as ox
from RLib.graphs.city import ensure_proper_policy

# 1. Descargar grafo
graph = ox.graph_from_place("Santiago, Chile", network_type="drive")

# 2. Añadir velocidades a los arcos
graph = ox.add_edge_speeds(graph)

# 3. Filtrar arcos muy cortos (evita ciclos de costo ≈ 0)
min_length = 3
edges_short = [(u, v) for u, v, d in graph.edges(data=True) if d.get("length", 0) < min_length]
graph.remove_edges_from(edges_short)

# 4. Mantener solo el mayor componente (débilmente conexo)
graph = ox.truncate.largest_component(graph, strongly=False)

# 5. Garantizar política proper
graph = ensure_proper_policy(graph, target_node)

# 6. Añadir arco recurrente en el nodo destino
graph.add_edge(target_node, target_node, length=0, speed_kph=30)
```
