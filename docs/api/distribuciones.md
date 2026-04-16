# API — Distribuciones de Costo

Las distribuciones de costo están definidas en `RLib/cost_distributions.py`. Se usan para modelar el tiempo de viaje estocástico en un arco de longitud $\ell$ metros a una velocidad promedio $v$ km/h.

El tiempo de viaje es:

$$c(\ell, V) = \frac{\ell}{V} \times 3.6$$

donde $V$ es una variable aleatoria cuya distribución puede ser log-normal, normal o uniforme.

---

## `expected_time`

Calcula la **esperanza matemática** del tiempo de viaje para un arco dado.

```python
expected_time(
    arc_length: float,
    avg_speed: float = 25,
    distribution: str = "lognormal",
) → float
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `arc_length` | `float` | Longitud del arco en metros (debe ser > 0) |
| `avg_speed` | `float` | Velocidad promedio en km/h (debe ser > 0) |
| `distribution` | `str` | Distribución: `"lognormal"`, `"normal"` o `"uniform"` |

**Retorna:** tiempo esperado de viaje en segundos.

**Raises:** `ValueError` si la distribución no es soportada o los parámetros son inválidos.

### Ejemplo

```python
from RLib.cost_distributions import expected_time

# Tiempo esperado para un arco de 500 m a 30 km/h con distribución log-normal
t = expected_time(arc_length=500, avg_speed=30, distribution="lognormal")
print(f"Tiempo esperado: {t:.2f} s")
```

---

## `random_time`

Genera una **muestra aleatoria** del tiempo de viaje para un arco dado.

```python
random_time(
    arc_length: float,
    avg_speed: float = 25,
    distribution: str = "lognormal",
) → float
```

Mismos parámetros que `expected_time`.

**Retorna:** una muestra del tiempo de viaje en segundos.

### Ejemplo

```python
from RLib.cost_distributions import random_time

# Muestrear el tiempo de viaje para un arco de 200 m
t_sample = random_time(arc_length=200, avg_speed=25, distribution="uniform")
```

---

## Distribuciones individuales

### Log-normal

La velocidad $V \sim \text{LogNormal}(\mu_V, \sigma_V)$ con $\mu_V = 25$ km/h y $\sigma_V = 6$ km/h. Es la distribución predeterminada y la que mejor modela velocidades en tráfico urbano.

```python
from RLib.cost_distributions import lognormal_expected_time, lognormal_random_time

t_exp = lognormal_expected_time(arc_length=300, avg_speed=25)
t_rnd = lognormal_random_time(arc_length=300, avg_speed=25)
```

**Nota:** $\mathbb{E}[1/V] = e^{-\mu_{\ln V} + \sigma_{\ln V}^2/2}$, donde $\mu_{\ln V}$ y $\sigma_{\ln V}$ se derivan de $\mu_V$ y $\sigma_V$.

### Normal

La velocidad $V \sim \mathcal{N}(\mu_V, \sigma_V)$ con $\sigma_V = 6$ km/h.

```python
from RLib.cost_distributions import normal_expected_time, normal_random_time

t_exp = normal_expected_time(arc_length=300, avg_speed=25)
t_rnd = normal_random_time(arc_length=300, avg_speed=25)
```

> **Nota:** La esperanza usa la aproximación $\mathbb{E}[1/V] \approx 1/\mu_V$, que es solo aproximada para distribuciones normales.

### Uniforme

La velocidad $V \sim \mathcal{U}(\mu_V - 3,\ \mu_V + 3)$.

```python
from RLib.cost_distributions import uniform_expected_time, uniform_random_time

t_exp = uniform_expected_time(arc_length=300, avg_speed=25)
t_rnd = uniform_random_time(arc_length=300, avg_speed=25)
```

La esperanza exacta de $1/V$ para la uniforme es:

$$\mathbb{E}\left[\frac{1}{V}\right] = \frac{\ln(b) - \ln(a)}{b - a}$$

donde $a = \max(\mu_V - 3,\ 0.001)$ y $b = \mu_V + 3$.

---

## Resumen

| Función | Descripción |
|---------|-------------|
| `expected_time(l, v, dist)` | Esperanza del tiempo de viaje |
| `random_time(l, v, dist)` | Muestra aleatoria del tiempo de viaje |
| `lognormal_expected_time(l, v)` | Esperanza con distribución log-normal |
| `lognormal_random_time(l, v)` | Muestra con distribución log-normal |
| `normal_expected_time(l, v)` | Esperanza aproximada con distribución normal |
| `normal_random_time(l, v)` | Muestra con distribución normal |
| `uniform_expected_time(l, v)` | Esperanza exacta con distribución uniforme |
| `uniform_random_time(l, v)` | Muestra con distribución uniforme |

---

## Uso en el entorno SSP

En `SSPEnv`, el parámetro `costs_distribution` acepta los siguientes valores:

| Valor | Comportamiento |
|-------|----------------|
| `"lognormal"` | Muestrea de la distribución log-normal |
| `"normal"` | Muestrea de la distribución normal |
| `"uniform"` | Muestrea de la distribución uniforme |
| `"expectation-lognormal"` | Usa la esperanza log-normal (determinista) |
| `"expectation-normal"` | Usa la esperanza normal (determinista) |
| `"expectation-uniform"` | Usa la esperanza uniforme (determinista) |

El modo `"expectation-*"` es determinista y se usa típicamente para calcular la tabla Q\* óptima con Dijkstra.
