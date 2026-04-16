# Conceptos Teóricos

Esta sección presenta los fundamentos matemáticos y algorítmicos sobre los que se construye el proyecto.

---

## 1. Stochastic Shortest Path (SSP)

El problema del **Camino Más Corto Estocástico** es una variante del Proceso de Decisión de Markov (MDP) con horizonte indefinido. Se define como una tupla $(\mathcal{S}, \mathcal{A}, P, c, s_0, s^*)$:

| Símbolo | Descripción |
|---------|-------------|
| $\mathcal{S}$ | Conjunto finito de estados |
| $\mathcal{A}(s)$ | Conjunto de acciones disponibles en el estado $s$ |
| $P(s' \mid s, a)$ | Probabilidad de transición al estado $s'$ tomando la acción $a$ en $s$ |
| $c(s, a)$ | Costo inmediato (no negativo) de tomar la acción $a$ en el estado $s$ |
| $s_0$ | Estado inicial |
| $s^*$ | Estado terminal absorbente (con costo 0) |

El objetivo es encontrar una **política** $\pi: \mathcal{S} \to \mathcal{A}$ que minimice el costo acumulado esperado hasta alcanzar el estado terminal $s^*$.

### Política *proper*

Una política es **proper** si garantiza que el agente alcanza el estado terminal $s^*$ con probabilidad 1. En el contexto de grafos, esto significa que desde cualquier nodo existe un camino hacia el nodo destino.

---

## 2. Q-Learning para SSP

**Q-Learning** es un algoritmo de aprendizaje por refuerzo libre de modelo (*model-free*) que estima la función de valor de acción $Q(s, a)$, definida como el costo acumulado esperado al tomar la acción $a$ en el estado $s$ y seguir la política óptima a partir de entonces.

### Actualización de la tabla Q

$$Q_{t+1}(s_t, a_t) \leftarrow Q_t(s_t, a_t) + \alpha_t \left[ r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t) \right]$$

donde:

- $\alpha_t \in (0, 1]$ es la **tasa de aprendizaje** (puede ser dinámica, e.g., $\alpha_t = \frac{c}{N_t(s,a) + c}$).
- $\gamma \in [0, 1]$ es el **factor de descuento** (habitualmente $\gamma = 1$ en SSP).
- $r_t$ es la recompensa (negativa del costo) obtenida en el paso $t$.
- $N_t(s, a)$ es la cantidad de veces que se ha tomado la acción $a$ en el estado $s$ hasta el tiempo $t$.

### Convergencia

Bajo condiciones de visita suficiente a cada par $(s, a)$ y una tasa de aprendizaje apropiada (e.g., $\sum_t \alpha_t = \infty$ y $\sum_t \alpha_t^2 < \infty$), Q-Learning converge a $Q^*$, la tabla Q óptima.

### Variantes implementadas

| Clase | Algoritmo |
|-------|-----------|
| `QAgentSSP` | Q-Learning estándar |
| `QAgentSSPSarsa0` | SARSA(0) — usa la acción realmente tomada en el siguiente paso |
| `QAgentSSPExpectedSarsa0` | Expected-SARSA(0) — usa el valor esperado de las acciones del siguiente estado |

---

## 3. Estrategias de Selección de Acciones

La selección de acciones equilibra **exploración** (probar acciones desconocidas) y **explotación** (elegir la mejor acción conocida).

### ε-greedy

Con probabilidad $\varepsilon$ se escoge una acción aleatoria; con probabilidad $1 - \varepsilon$ se escoge la acción codiciosa (greedy):

$$\pi(s) = \begin{cases} \arg\max_{a} Q(s, a) & \text{con probabilidad } 1 - \varepsilon \\ \text{acción aleatoria} & \text{con probabilidad } \varepsilon \end{cases}$$

### ε-greedy con decaimiento

La probabilidad de exploración disminuye con las visitas al estado:

$$\varepsilon_t(s) = \frac{c}{N_t(s)}$$

donde $c$ es una constante y $N_t(s)$ es el número de visitas al estado $s$.

### UCB1

Selecciona la acción que maximiza el intervalo de confianza superior:

$$\text{UCB1}(s, a) = Q(s, a) + c \sqrt{\frac{\ln N_t(s)}{N_t(s, a)}}$$

### AsOpt-UCB (UCB asintóticamente óptimo)

Similar a UCB1, pero con una función de exploración $f(t) = 1 + t \cdot \ln(t)^2$:

$$\text{UCB}(s, a) = Q(s, a) + \sqrt{\frac{c \cdot \ln f(t)}{N_t(s, a)}}$$

### Boltzmann (Softmax)

Asigna probabilidades a cada acción proporcionales a su valor Q:

$$P(a \mid s) = \frac{e^{\eta \cdot Q(s, a)}}{\sum_{a'} e^{\eta \cdot Q(s, a')}}$$

El parámetro $\eta$ (temperatura inversa) puede ser constante o una expresión dinámica (e.g., `"log(n_s) / q_range"`).

---

## 4. Multi-Armed Bandits (MAB) aplicados al SSP

En el enfoque de **bandits**, cada camino posible entre el nodo origen y el nodo destino es un **brazo**. El agente no aprende estado por estado, sino que elige directamente qué ruta seguir en cada episodio.

### Estrategias implementadas

| Clase | Estrategia |
|-------|-----------|
| `EGreedyMultiArmedBandit` | ε-greedy fijo ($\varepsilon = 0.1$) |
| `UCBMultiArmedBandit` | UCB1 |
| `EXP3MultiArmedBandit` | EXP3 (adversarial) |
| `BoltzmannBandit` | Boltzmann/Softmax |

### Regret

El **regret** mide la pérdida acumulada por no haber escogido siempre el brazo óptimo:

$$R_T = T \cdot \mu^* - \sum_{t=1}^T r_t$$

donde $\mu^*$ es la recompensa media del brazo óptimo y $r_t$ la recompensa obtenida en la ronda $t$.

---

## 5. Solución Óptima con Dijkstra

Para evaluar el rendimiento del agente se calcula la tabla $Q^*$ usando el **algoritmo de Dijkstra** sobre el grafo invertido desde el nodo destino. El peso de cada arco $(s, a)$ se toma como la **esperanza del costo** de transitar por ese arco según la distribución elegida.

La función `get_optimal_policy_and_q_star` en `RLib/utils/dijkstra.py` realiza este cálculo y devuelve tanto la política óptima como la tabla $Q^*$ completa.

### Métrica de convergencia

El error de convergencia se mide con la **norma máxima**:

$$\|Q_t - Q^*\|_\infty = \max_{(s,a)} |Q_t(s, a) - Q^*(s, a)|$$

Esta métrica se normaliza dividiendo entre $|Q^*(s_0, \pi^*(s_0))|$ para facilitar la comparación entre distintos experimentos.

---

## 6. Distribuciones de Costo Estocástico

El costo de transitar por un arco de longitud $\ell$ a velocidad media $v$ se modela como el tiempo de viaje:

$$c(s, a) = \frac{\ell}{V}$$

donde $V$ es una variable aleatoria cuya distribución puede ser:

| Distribución | Descripción |
|-------------|-------------|
| **Log-normal** | $V \sim \text{LogNormal}(\mu_v, \sigma_v)$; modela bien velocidades en tráfico urbano |
| **Normal** | $V \sim \mathcal{N}(\mu_v, \sigma_v)$ |
| **Uniforme** | $V \sim \mathcal{U}(\mu_v - 3, \mu_v + 3)$ |

Para evaluar la solución óptima se utiliza la **esperanza** del costo (prefijando `"expectation-"` en el nombre de la distribución). Para el entrenamiento se muestrea de la distribución correspondiente.
