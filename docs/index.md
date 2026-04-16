# Documentación de SSP Q-Learning

Bienvenido a la documentación del proyecto **SSP Q-Learning**, una biblioteca de aprendizaje por refuerzo para resolver el problema del **Camino Más Corto Estocástico** (*Stochastic Shortest Path*, SSP) mediante Q-Learning y variantes.

---

## ¿Qué hace este proyecto?

El proyecto implementa agentes de aprendizaje por refuerzo que:

1. **Aprenden** la política óptima para navegar un grafo desde un nodo de origen hasta un nodo destino, minimizando el costo esperado del trayecto.
2. **Comparan** distintas estrategias de selección de acciones (ε-greedy, Boltzmann, UCB1, etc.) frente a una solución óptima teórica obtenida con Dijkstra.
3. **Evalúan** la convergencia de la tabla Q al óptimo Q\* mediante la norma máxima.

Además, incluye agentes tipo **Multi-Armed Bandit (MAB)**, donde cada camino posible entre el origen y el destino es tratado como un brazo del bandit.

---

## Estructura de la documentación

| Sección | Descripción |
|---------|-------------|
| [Conceptos](conceptos.md) | Fundamentos teóricos: SSP, Q-Learning, MAB y Dijkstra |
| [Instalación](instalacion.md) | Cómo instalar el proyecto y sus dependencias |
| [Inicio Rápido](guias/quickstart.md) | Ejemplo mínimo funcional para comenzar a usar la biblioteca |
| [Guía de Experimentos](guias/experimentos.md) | Cómo ejecutar y personalizar experimentos |
| [API — Agentes](api/agentes.md) | Referencia de los agentes Q-Learning |
| [API — Entornos](api/entornos.md) | Referencia de los entornos SSP y Laberinto |
| [API — Selectores de Acción](api/selectores_accion.md) | Referencia de las estrategias de selección de acciones |
| [API — Bandits](api/bandits.md) | Referencia de los agentes tipo Bandit |
| [API — Grafos](api/grafos.md) | Referencia de los generadores de grafos |
| [API — Utilidades](api/utilidades.md) | Referencia de Dijkstra, plots, serialización y tablas |
| [API — Distribuciones de Costo](api/distribuciones.md) | Referencia de las distribuciones de costo estocástico |

---

## Estructura del módulo `RLib`

```
RLib/
├── agents/
│   ├── ssp.py            # QAgent, QAgentSSP, QAgentSSPSarsa0, QAgentSSPExpectedSarsa0
│   └── maze.py           # QAgentMaze
├── environments/
│   ├── ssp.py            # SSPEnv, HardSSPEnv
│   └── maze.py           # Maze
├── graphs/
│   ├── city.py           # ensure_proper_policy
│   └── perceptron.py     # create_perceptron_graph, create_hard_perceptron_graph
├── bandits/
│   └── ssp.py            # MultiArmedBandit y subclases
├── utils/
│   ├── dijkstra.py       # dijkstra_shortest_path, get_optimal_policy_and_q_star
│   ├── plots.py          # Funciones de visualización
│   ├── tables.py         # Utilidades para tablas Q
│   ├── files.py          # Serialización y guardado de resultados
│   └── serializers.py    # Serializadores auxiliares
├── action_selectors.py   # Estrategias de selección de acciones
└── cost_distributions.py # Distribuciones de costo estocástico
```
