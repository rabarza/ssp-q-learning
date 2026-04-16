# Guía de Inicio Rápido

Esta guía muestra cómo resolver el SSP en un grafo sintético tipo *perceptrón* con un agente Q-Learning en pocos pasos.

---

## 1. Crear el grafo

El módulo `RLib.graphs.perceptron` genera grafos dirigidos en capas donde cada nodo de una capa está conectado a todos los nodos de la siguiente:

```python
from RLib.graphs.perceptron import create_perceptron_graph

# Grafo con 4 capas: 1 → 5 → 5 → 1 nodo
graph = create_perceptron_graph(
    nodes_by_layer=[1, 5, 5, 1],
    min_length=10,
    max_length=500,
    seed=42,
)

origin_node = 1   # primer nodo
target_node = 0   # último nodo (renombrado a 0)
```

---

## 2. Calcular la solución óptima (Q\*)

Antes de entrenar, calculamos la tabla Q\* con Dijkstra para poder medir la convergencia del agente:

```python
from RLib.utils.dijkstra import (
    get_optimal_policy_and_q_star,
    get_shortest_path_from_policy,
    get_q_table_for_path,
)

costs_distribution = "uniform"

optimal_policy, q_star = get_optimal_policy_and_q_star(
    graph, target_node, distribution=costs_distribution
)
shortest_path = get_shortest_path_from_policy(optimal_policy, origin_node, target_node)
q_star_sp = get_q_table_for_path(q_star, shortest_path)

print("Camino más corto:", shortest_path)
```

---

## 3. Crear el entorno

```python
from RLib.environments.ssp import SSPEnv

env = SSPEnv(
    graph=graph,
    start_state=origin_node,
    terminal_state=target_node,
    costs_distribution=costs_distribution,
    shortest_path=shortest_path,
)
```

---

## 4. Crear el agente y entrenarlo

```python
from RLib.agents.ssp import QAgentSSP
from RLib.action_selectors import EpsilonGreedyActionSelector

selector = EpsilonGreedyActionSelector(epsilon=0.1)
alpha = "250 / (N(s,a) + 250)"   # tasa de aprendizaje dinámica

agent = QAgentSSP(environment=env, action_selector=selector, alpha=alpha)

agent.train(
    num_episodes=3000,
    shortest_path=shortest_path,
    q_star=q_star,
)
```

---

## 5. Evaluar los resultados

```python
import numpy as np

results = agent.results()

print(f"Costo óptimo (Q*):   {results['optimal_cost']:.4f}")
print(f"Error final (norma): {results['max_norm_error'][-1]:.4f}")
print(f"Pasos promedio:      {np.mean(results['steps']):.1f}")
print(f"Mejor camino:        {agent.best_path()}")
```

---

## 6. Comparar estrategias

Puedes entrenar múltiples agentes con diferentes selectores y comparar su evolución:

```python
from RLib.action_selectors import UCB1ActionSelector, BoltzmannSelector

agents = []
selectors = [
    EpsilonGreedyActionSelector(epsilon=0.1),
    UCB1ActionSelector(c=1.0),
    BoltzmannSelector(eta="log(n_s) / q_range"),
]

for sel in selectors:
    e = SSPEnv(graph, origin_node, target_node, costs_distribution, shortest_path)
    a = QAgentSSP(environment=e, action_selector=sel, alpha=alpha)
    a.train(num_episodes=3000, shortest_path=shortest_path, q_star=q_star)
    agents.append(a)

# Mostrar error final de cada agente
for a in agents:
    r = a.results()
    print(f"{a.strategy}: error final = {r['max_norm_error'][-1]:.4f}")
```

---

## Próximos pasos

- Consulta la [Guía de Experimentos](experimentos.md) para ejecutar los scripts completos.
- Revisa la [API de Agentes](../api/agentes.md) para opciones avanzadas de configuración.
- Revisa los [Conceptos Teóricos](../conceptos.md) para entender las métricas de evaluación.
