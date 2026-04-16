# API — Selectores de Acción

Los selectores de acción están definidos en `RLib/action_selectors.py`. Todos heredan de `ActionSelector` e implementan el método `select_action(agent, state)`.

---

## Clase base `ActionSelector`

```python
class ActionSelector:
    def select_action(self, agent, state) -> Any:
        raise NotImplementedError()

    def get_label(self) -> str:
        ...
```

Cualquier selector personalizado debe heredar de esta clase e implementar `select_action`.

---

## `EpsilonGreedyActionSelector`

Estrategia **ε-greedy**: explora con probabilidad $\varepsilon$ y explota con probabilidad $1 - \varepsilon$.

```python
EpsilonGreedyActionSelector(epsilon: float = 0.1)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `epsilon` | `float` | Probabilidad de exploración ∈ [0, 1] |

### Ejemplo

```python
from RLib.action_selectors import EpsilonGreedyActionSelector

selector = EpsilonGreedyActionSelector(epsilon=0.1)
```

---

## `EpsilonGreedyDecayActionSelector`

**ε-greedy con decaimiento**: la probabilidad de exploración decrece inversamente con el número de visitas al estado:

$$\varepsilon_t(s) = \frac{c}{N_t(s)}$$

```python
EpsilonGreedyDecayActionSelector(constant: float = 0.99)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `constant` | `float` | Constante $c$ de decaimiento ∈ (0, ∞) |

### Método adicional

```python
selector.get_probabilities(agent, state) → dict
```

Devuelve un diccionario `{acción: probabilidad}` para el estado dado (útil para depuración).

### Ejemplo

```python
from RLib.action_selectors import EpsilonGreedyDecayActionSelector

selector = EpsilonGreedyDecayActionSelector(constant=1.0)
```

---

## `BubeckDecayEpsilonGreedyActionSelector`

Variante de ε-greedy con decaimiento basada en Bubeck (2012). La probabilidad de exploración está acotada por:

$$\varepsilon_t = \min\left\{1,\ \frac{c \cdot |\mathcal{A}(s)|}{d^2 \cdot N_t(s)}\right\}$$

```python
BubeckDecayEpsilonGreedyActionSelector(c: float = 1, d: float = 0.9)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `c` | `float` | Parámetro de exploración |
| `d` | `float` | Parámetro de escala ∈ (0, 1) |

> **Referencia:** Bubeck, S. & Cesa-Bianchi, N. (2012). *Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems*. Foundations and Trends in Machine Learning, §2.4.5.

---

## `UCB1ActionSelector`

Selecciona la acción que maximiza el **Upper Confidence Bound**:

$$\text{UCB1}(s, a) = Q(s, a) + c \sqrt{\frac{\ln N_t(s)}{N_t(s, a)}}$$

Acciones no visitadas se seleccionan primero de forma aleatoria.

```python
UCB1ActionSelector(c: float = 2)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `c` | `float` | Parámetro de exploración > 0 |

### Ejemplo

```python
from RLib.action_selectors import UCB1ActionSelector

selector = UCB1ActionSelector(c=0.5)
```

---

## `AsOptUCBActionSelector`

**UCB asintóticamente óptimo**: usa la función $f(t) = 1 + t \cdot \ln(t)^2$ para controlar la exploración:

$$\text{UCB}(s, a) = Q(s, a) + \sqrt{\frac{c \cdot \ln f(t)}{N_t(s, a)}}$$

donde $t = N_t(s)$.

```python
AsOptUCBActionSelector(c: float = 2)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `c` | `float` | Parámetro de exploración > 0 |

Este selector es asintóticamente óptimo en el entorno de Multi-Armed Bandits: converge a la política óptima a medida que el número de visitas tiende a infinito.

### Ejemplo

```python
from RLib.action_selectors import AsOptUCBActionSelector

selector = AsOptUCBActionSelector(c=0.01)
```

---

## `BoltzmannSelector`

Asigna probabilidades a las acciones usando una distribución Softmax:

$$P(a \mid s) = \frac{e^{\eta \cdot Q(s, a)}}{\sum_{a'} e^{\eta \cdot Q(s, a')}}$$

El parámetro $\eta$ (temperatura inversa) puede ser constante o una expresión dinámica.

```python
BoltzmannSelector(eta: Union[str, float] = "log(n_s) / q_range")
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `eta` | `str` o `float` | Temperatura inversa; puede ser una expresión evaluable |

### Variables disponibles en `eta`

| Variable | Descripción |
|----------|-------------|
| `t` | Episodio actual |
| `T` | Número total de episodios |
| `A` | Número de acciones disponibles en el estado |
| `n_s` | Visitas al estado actual |
| `q_range` | Rango de valores Q en el estado: Q_max − Q_min |
| `sqrt`, `log` | Funciones matemáticas |

### Ejemplos de expresiones de `eta`

```python
BoltzmannSelector(eta=0.1)                           # constante
BoltzmannSelector(eta="sqrt(n_s)")                   # crece con visitas
BoltzmannSelector(eta="log(n_s) / q_range")         # adaptativo al rango Q
BoltzmannSelector(eta="log(n_s * log(n_s+1)) / q_range")
BoltzmannSelector(eta="sqrt(n_s) / q_range")
```

### Método adicional

```python
selector.get_probabilities(agent, state) → dict
```

Devuelve `{acción: probabilidad}` para el último `select_action` llamado.

---

## Resumen comparativo

| Selector | Exploración | Parámetros clave | Ideal para |
|----------|------------|------------------|-----------|
| `EpsilonGreedy` | Constante | `epsilon` | Referencia simple |
| `EpsilonGreedyDecay` | Decrece con $N(s)$ | `constant` | Convergencia garantizada |
| `BubeckDecay` | Decrece con $N(s)$ y $\|\mathcal{A}\|$ | `c`, `d` | Teóricamente motivado |
| `UCB1` | Confianza superior | `c` | Exploración dirigida |
| `AsOptUCB` | UCB asintótico | `c` | Optimalidad asintótica |
| `Boltzmann` | Softmax adaptativo | `eta` | Exploración suave |
