# Instalación

## Requisitos

- Python 3.9 o superior
- `pip`

---

## Pasos de instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/rabarza/ssp-q-learning.git
cd ssp-q-learning
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv env
```

Activar el entorno:

- **Linux / macOS**:
  ```bash
  source env/bin/activate
  ```
- **Windows**:
  ```bash
  .\env\Scripts\activate
  ```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

---

## Dependencias principales

| Paquete | Versión | Uso |
|---------|---------|-----|
| `numpy` | ≥ 2.2 | Operaciones numéricas |
| `networkx` | ≥ 3.4 | Representación y manipulación de grafos |
| `osmnx` | ≥ 2.0 | Descarga de grafos de calles desde OpenStreetMap |
| `matplotlib` | ≥ 3.10 | Visualización de resultados |
| `plotly` | ≥ 6.0 | Gráficos interactivos |
| `tqdm` | ≥ 4.67 | Barras de progreso en consola |
| `stqdm` | ≥ 0.0.5 | Barras de progreso en Streamlit |
| `pandas` | ≥ 2.2 | Manejo de tablas y resultados |
| `geopandas` | ≥ 1.0 | Datos geoespaciales |
| `streamlit` | ≥ 1.44 | Interfaz web interactiva (opcional) |

La lista completa se encuentra en [`requirements.txt`](../requirements.txt).

---

## Verificar la instalación

Para comprobar que todo funciona correctamente, ejecuta el script de ejemplo del laberinto:

```bash
python scripts/maze.py
```

Si se instala correctamente, el script entrenará tres agentes en un laberinto generado aleatoriamente y mostrará los resultados.

---

## Notas adicionales

- Los scripts que usan grafos de ciudades reales (e.g., `scripts/santiago.py`) requieren acceso a internet para descargar el grafo desde OpenStreetMap mediante `osmnx`.
- Los resultados de los experimentos se guardan por defecto en la carpeta `scripts/results/`.
