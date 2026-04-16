# Guía de Experimentos

La carpeta `scripts/` contiene los experimentos del proyecto. Cada script configura un entorno, calcula la solución óptima con Dijkstra, entrena varios agentes en paralelo y guarda los resultados.

---

## Scripts disponibles

| Script | Entorno | Descripción |
|--------|---------|-------------|
| `maze.py` | Laberinto generado aleatoriamente | Agentes Q-Learning en un laberinto 2D |
| `perceptron.py` | Grafo perceptrón sintético | Experimentos con grafo tipo red neuronal |
| `hardPerceptron.py` | Grafo perceptrón *hard* | Variante con arcos eliminados hacia el camino óptimo |
| `santiago.py` | Red vial de Santiago, Chile | Experimentos en grafo real (requiere internet) |
| `paine.py` | Red vial de Torres del Paine | Experimentos en grafo real (requiere internet) |
| `constitucion.py` | Red vial de Constitución, Chile | Experimentos en grafo real (requiere internet) |
| `piedmont.py` | Red vial de Piamonte, Italia | Experimentos en grafo real (requiere internet) |

Los scripts terminados en `_results.py` (e.g., `santiago_results.py`) cargan los resultados previamente guardados y generan las visualizaciones.

---

## Estructura de un experimento típico

Todos los scripts de experimento siguen la misma estructura:

```
1. Definir configuración (ciudad, nodos, distribución de costos)
2. Descargar/crear el grafo
3. Preprocesar el grafo (eliminar arcos cortos, garantizar conectividad)
4. Calcular política óptima y Q* con Dijkstra
5. Definir la fórmula de la tasa de aprendizaje (alpha)
6. Crear entornos y agentes con distintos selectores de acción
7. Entrenar agentes (con multiprocessing para paralelismo)
8. Guardar resultados en disco
```

---

## Ejecutar un experimento

### Laberinto (sin internet)

```bash
python scripts/maze.py
```

### Grafo perceptrón sintético (sin internet)

```bash
python scripts/perceptron.py
```

### Ciudad real (requiere internet)

```bash
python scripts/santiago.py
```

> **Nota:** Los scripts de ciudad descargan el grafo de OpenStreetMap la primera vez. Los grafos pueden contener decenas de miles de nodos, por lo que el cálculo de Q\* puede tardar varios minutos.

---

## Personalizar un experimento

### Cambiar la estrategia de selección de acción

En cualquier script, puedes añadir o cambiar el selector de acción:

```python
from RLib.action_selectors import (
    EpsilonGreedyActionSelector,
    EpsilonGreedyDecayActionSelector,
    UCB1ActionSelector,
    AsOptUCBActionSelector,
    BoltzmannSelector,
)

# ε-greedy fijo
selector = EpsilonGreedyActionSelector(epsilon=0.1)

# ε-greedy con decaimiento c/N(s)
selector = EpsilonGreedyDecayActionSelector(constant=1.0)

# UCB1 con parámetro de exploración c
selector = UCB1ActionSelector(c=0.5)

# UCB asintóticamente óptimo
selector = AsOptUCBActionSelector(c=0.01)

# Boltzmann con temperatura dinámica
selector = BoltzmannSelector(eta="log(n_s) / q_range")
```

### Cambiar la tasa de aprendizaje (alpha)

`alpha` puede ser un número flotante constante o una cadena de texto con una expresión matemática que use las variables `N(s,a)`, `N(s)` y `t`:

```python
alpha = 0.1                            # constante
alpha = "1 / N(s,a)"                   # decae con visitas a (s,a)
alpha = "250 / (N(s,a) + 250)"         # decaimiento armonico armónico
alpha = "1000 / (N(s) + 1000)"         # decae con visitas al estado s
alpha = "1 / sqrt(N(s,a))"            # decae con la raíz cuadrada
alpha = "1 / log(N(s,a) + 1)"         # decae con el logaritmo
```

### Cambiar la distribución de costos

```python
costs_distribution = "uniform"      # Distribución uniforme
costs_distribution = "lognormal"    # Distribución log-normal (predeterminada)
costs_distribution = "normal"       # Distribución normal
```

### Cambiar el número de episodios

```python
num_episodes = 20000   # más episodios → mejor convergencia, mayor tiempo
```

---

## Resultados

Los resultados se guardan en `scripts/results/` organizados por tipo de entorno, ciudad y distribución de costos. Cada experimento guarda:

- Archivo `.pkl` con el agente serializado (tabla Q, historial de scores, etc.)
- Archivo `.json` con las métricas del experimento (scores, regret, errores de norma máxima)

Para cargar y visualizar los resultados de un experimento previo:

```bash
python scripts/santiago_results.py
```

---

## Paralelismo

Los scripts de ciudad utilizan `multiprocessing` para entrenar múltiples agentes en paralelo:

```python
import multiprocessing

processes = []
for selector in selectors:
    env = SSPEnv(graph, origin_node, target_node, costs_distribution, shortest_path)
    agent = QAgentSSP(environment=env, action_selector=selector, alpha=alpha)
    p = multiprocessing.Process(target=agent.train, args=(num_episodes,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()
```

Esto aprovecha todos los núcleos disponibles del procesador para reducir el tiempo total de experimentación.
