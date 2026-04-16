# API — Bandits

Los agentes tipo **Multi-Armed Bandit** están definidos en `RLib/bandits/ssp.py`. En este enfoque, cada camino posible entre el nodo origen y el nodo destino es un **brazo** del bandit. El agente no aprende estado por estado, sino que elige directamente qué ruta seguir en cada ronda.

---

## Jerarquía de clases

```
MultiArmedBandit              (clase base)
├── EGreedyMultiArmedBandit   (ε-greedy)
├── UCBMultiArmedBandit       (UCB1)
├── EXP3MultiArmedBandit      (EXP3)
└── BoltzmannBandit           (Boltzmann/Softmax)
```

---

## `MultiArmedBandit`

Clase base abstracta para todos los bandits.

**Módulo:** `RLib.bandits.ssp`

### Constructor

```python
MultiArmedBandit(paths_mean_cost: list)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `paths_mean_cost` | `list[float]` | Lista con el costo medio (negativo de la recompensa media) de cada camino/brazo |

### Atributos

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `num_paths` | `int` | Número de brazos (caminos posibles) |
| `mean_rewards` | `list` | Recompensas medias de cada brazo |
| `optimal_path` | `float` | Recompensa media del brazo óptimo |
| `rewards` | `list` | Historial de recompensas obtenidas |
| `arm_pulls` | `list` | Número de veces que se ha seleccionado cada brazo |
| `arm_rewards` | `list` | Estimación actual de la recompensa media de cada brazo |
| `arms_history` | `list` | Historial del brazo seleccionado en cada ronda |

### Métodos

#### `select_arm() → int`

Selecciona el índice del brazo a usar. **Debe implementarse en subclases.**

#### `pull(chosen_arm, reward)`

Registra la recompensa `reward` obtenida al seleccionar el brazo `chosen_arm` y actualiza la estimación incremental:

$$\hat{\mu}_n(k) = \hat{\mu}_{n-1}(k) + \frac{r - \hat{\mu}_{n-1}(k)}{n}$$

#### `regret(t) → float`

Devuelve el regret acumulado hasta la ronda $t$:

$$R_t = t \cdot \mu^* - \sum_{i=1}^t r_i$$

#### `average_regret(t) → float`

Devuelve el regret promedio: $R_t / t$.

#### `pseudo_regret(t) → float`

Devuelve el pseudo-regret basado en las recompensas medias de los brazos seleccionados.

---

## `EGreedyMultiArmedBandit`

Bandit con estrategia **ε-greedy** fija ($\varepsilon = 0.1$).

```python
EGreedyMultiArmedBandit(paths_mean_cost: list)
```

Selecciona el brazo con mayor estimación de recompensa con probabilidad $0.9$, y un brazo aleatorio con probabilidad $0.1$.

---

## `UCBMultiArmedBandit`

Bandit con estrategia **UCB1**:

$$\text{UCB1}(k) = \hat{\mu}(k) + \sqrt{\frac{c \cdot \ln t}{N_t(k)}}$$

```python
UCBMultiArmedBandit(paths_mean_cost: list, c: float = 2)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `c` | `float` | Parámetro de exploración |

Brazos no visitados se seleccionan primero en orden.

---

## `EXP3MultiArmedBandit`

Bandit con el algoritmo **EXP3** (Exponential-weight algorithm for Exploration and Exploitation), diseñado para entornos adversariales.

```python
EXP3MultiArmedBandit(paths_mean_cost: list, eta: Union[str, float] = 0.1)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `eta` | `str` o `float` | Tasa de aprendizaje. Puede ser expresión dinámica |

### Variables disponibles en `eta`

| Variable | Descripción |
|----------|-------------|
| `t` | Número de ronda actual |
| `k`, `K` | Número de brazos (caminos) |
| `n` | Total de pulls realizados |
| `sqrt`, `log` | Funciones matemáticas |

### Ejemplo

```python
EXP3MultiArmedBandit(paths_mean_cost, eta="sqrt(log(K) / (K * t))")
```

---

## `BoltzmannBandit`

Bandit con distribución de selección **Softmax**:

$$P(k) = \frac{e^{\eta \cdot \hat{\mu}(k)}}{\sum_{k'} e^{\eta \cdot \hat{\mu}(k')}}$$

```python
BoltzmannBandit(paths_mean_cost: list, eta: Union[str, float] = 0.1)
```

Mismos parámetros y variables que `EXP3MultiArmedBandit`.

---

## Función auxiliar `train_bandit`

```python
train_bandit(
    bandit: MultiArmedBandit,
    path_lengths: list,
    num_rounds: int,
    distribution: str = "normal",
)
```

Entrena el bandit durante `num_rounds` rondas, muestreando el costo de cada camino seleccionado según la distribución indicada.

---

## Funciones para obtener los caminos

```python
from RLib.bandits.ssp import find_all_paths, calculate_path_weights

# Encontrar todos los caminos simples entre dos nodos
all_paths = find_all_paths(graph, start_node, end_node)

# Calcular el peso (longitud total) de cada camino
path_lengths = calculate_path_weights(graph, all_paths, weight="length")
```

---

## Ejemplo completo

```python
from RLib.graphs.perceptron import create_perceptron_graph
from RLib.cost_distributions import expected_time
from RLib.bandits.ssp import (
    find_all_paths,
    calculate_path_weights,
    train_bandit,
    EGreedyMultiArmedBandit,
    UCBMultiArmedBandit,
    EXP3MultiArmedBandit,
    BoltzmannBandit,
)

# Crear grafo y obtener caminos
graph = create_perceptron_graph([1, 10, 1], min_length=100, max_length=2000)
start_node, end_node = 1, 0
all_paths = find_all_paths(graph, start_node, end_node)
path_lengths = calculate_path_weights(graph, all_paths)

# Calcular costos medios (negativos para maximizar)
distribution = "uniform"
path_costs = [-expected_time(l, 25, distribution) for l in path_lengths]

# Crear bandits
bandit_eps   = EGreedyMultiArmedBandit(path_costs)
bandit_ucb   = UCBMultiArmedBandit(path_costs, c=2)
bandit_exp3  = EXP3MultiArmedBandit(path_costs, eta="log(t+1)")
bandit_boltz = BoltzmannBandit(path_costs, eta="log(t+1)")

# Entrenar
num_rounds = 1000
for bandit in [bandit_eps, bandit_ucb, bandit_exp3, bandit_boltz]:
    train_bandit(bandit, path_lengths, num_rounds, distribution)
    print(f"{bandit.strategy}: regret final = {bandit.regret(num_rounds):.2f}")
```
