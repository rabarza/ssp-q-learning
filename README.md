# README

## Descripción del Proyecto

Este proyecto implementa **agentes de aprendizaje reforzado** que utilizan el algoritmo **Q-Learning** para resolver el problema de **Stochastic Shortest Path (SSP)**, empleando diversas estrategias de selección de acciones. Las estrategias implementadas se basan en las estrategias de selección de acciones de los **Multi-Armed Bandit Problems (MAB)**. Además, el proyecto incluye **agentes tipo Bandit**, que resuelven el mismo problema pero bajo el enfoque de que cada camino dentro del grafo representa un brazo.

El código está organizado en un módulo principal llamado `RLib`, que contiene la lógica de los **agentes (tanto de Q-Learning como de Bandits)**, la **creación de grafos y entornos**, y la **interacción del agente** con estos. También se implementan distintas **estrategias de selección de acciones** (por ejemplo, \(\epsilon\)-greedy).

Para evaluar el rendimiento del agente, se utiliza el algoritmo de **Dijkstra** para encontrar la tabla \(Q^\*\), donde el peso del arco se considera como la esperanza del costo de transitar por ese arco. Esto permite medir el desempeño del agente mediante la estrategia de selección de acción utilizada frente a una solución óptima teórica, para luego analizar la convergencia de \(Q_t\) a \(Q^\*\).

Finalmente, la carpeta `scripts` contiene los códigos que ejecutan experimentos específicos, utilizando el módulo `RLib` para simular escenarios y analizar los resultados obtenidos por los agentes en distintos entornos y mediante distintas estrategias de selección de acción.

---

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```
código_tesis/
│
├── RLib/
│   ├── agents/               # Implementación de agentes Q-Learning y Bandits
│   ├── environments/         # Definición de entornos (laberintos, grafos, etc.)
│   ├── graphs/               # Generación de grafos para los entornos
│   ├── utils/                # Utilidades como Dijkstra, serialización, etc.
│   └── action_selectors.py   # Estrategias de selección de acciones
│
├── scripts/                  # Scripts para ejecutar experimentos específicos
│
├── results/                  # Resultados de los experimentos
│
└── requirements.txt          # Dependencias del proyecto
```

---

## Instalación

1. **Clonar el repositorio**:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd ssp-qlearning
   ```

2. **Crear un entorno virtual**:

   ```bash
   python -m venv env
   ```

3. **Activar el entorno virtual**:

   - En Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - En Linux/Mac:
     ```bash
     source env/bin/activate
     ```

4. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uso

### Ejecución de Experimentos

Los experimentos se encuentran en la carpeta `scripts`. Cada script ejecuta un experimento específico, utilizando los agentes y entornos definidos en el módulo `RLib`. Por ejemplo:

```bash
python scripts/maze.py
```

### Personalización de Agentes

Puedes personalizar los agentes modificando los parámetros en los scripts, como la estrategia de selección de acciones (
\(\epsilon\)-greedy, Boltzmann, UCB1, etc.) o la fórmula para calcular el valor de \(\alpha\) (tasa de aprendizaje).

---

## Principales Componentes

### 1. **Agentes**

Los agentes implementados en `RLib/agents/` incluyen:

- **QAgentSSP**: Resuelve el problema de SSP aplicado a grafos mediante Q-learning
- **QAgentMaze**: Resuelve el problema de SSP aplicado a laberintos mediante Q-learning
- **Bandits**: Resuelve el problema considerando cada camino posible entre el estado inicial y el terminal como un brazo de un Multi-Armed Bandit.

### 2. **Entornos**

Los entornos definidos en `RLib/environments/` incluyen:

- **Laberintos**: Representados como matrices, donde cada celda tiene un costo asociado.
- **Grafos**: Representados como redes dirigidas, donde los nodos son estados y los arcos tienen costos asociados.

### 3. **Estrategias de Selección de Acciones**

Las estrategias implementadas en `RLib/action_selectors.py` incluyen:

- \(\epsilon\)-greedy
- Boltzmann
- UCB1

Estas estrategias permiten explorar diferentes enfoques para equilibrar la exploración y la explotación.

### 4. **Evaluación**

El rendimiento de los agentes se evalúa comparando sus resultados con la solución óptima obtenida mediante el algoritmo de Dijkstra.

---

## Resultados

Los resultados de los experimentos se almacenan en la carpeta `results/`. Estos incluyen:

- Objetos serializados que contienen información de los experimentos.
- Objetos pickle que contienen los resultados de los experimentos, incluyendo la tabla \(Q\) y el rendimiento del agente a lo largo del tiempo.

---

## Licencia

Copyright (C) 2025 Rolando Ignacio Jesús Abarza Cancino

Este proyecto está bajo la Licencia GNU General Public License v3.0 (GPL-3.0). Esto significa que puedes usar, modificar y redistribuir el código libremente, siempre que cualquier trabajo derivado se distribuya bajo la misma licencia y mantenga el aviso de copyright. Consulta el archivo `LICENSE` para el texto completo.
