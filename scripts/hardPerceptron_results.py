import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402
from RLib.utils.plots import plot_results_per_episode_comp_plotly
from RLib.utils.files import load_model_results, find_files_by_keyword

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Definir el directorio principal donde se encuentran los resultados
model_name = "q-learning"
# model_name = "sarsa0"

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"results", model_name
)


number_of_layers = 42
number_of_nodes = 402
seed = 20
graph_name = f"HardPerceptron/{number_of_layers}Layers-{number_of_nodes}Nodes-Seed{seed}"
cost_distribution = "lognormal"  # "lognormal"
is_dynamic = True
alpha_type = "dynamic" if is_dynamic else "constant"
ruta_carpeta = os.path.join(
    RESULTS_DIR, graph_name, cost_distribution, f"{alpha_type}_alpha/"
)

# Criterias to plot
# criterias_list = ['max_norm_error_normalized',
#   'max_norm_error_shortest_path_normalized',
#   'average regret']
criterias_list = ["normalized shortest path error"]
# Inicializar una lista para almacenar los DataFrames de resultados


all_results = []


# Función para extraer el número de capas de un nombre de archivo
def get_layers_from_name(name):
    return int(name.split("Layers")[0])


# Iterar sobre cada subcarpeta en RESULTS_DIR
for root, dirs, files in os.walk(ruta_carpeta):
    # Identificar si la carpeta corresponde a un entorno "HardPerceptron"
    if "HardPerceptron" in root:
        # Extraer detalles del entorno a partir del nombre de la carpeta
        # Separar el nombre de la carpeta en partes: ['results', 'HardPerceptron', '42Layers-402Nodes-Seed20', 'uniform', 'dynamic_alpha']
        parts = root.split(os.sep)
        if len(parts) < 3:
            continue
        graph_name = parts[-3]  # Ejemplo: '18Layers-162Nodes-Seed20'
        cost_distribution = parts[-2]  # Ejemplo: 'uniform'
        alpha_type = parts[-1]  # Ejemplo: 'dynamic_alpha' or 'constant_alpha'

        # Cargar los resultados de los agentes en este entorno
        for strategy_dir in dirs:
            strategy_path = os.path.join(root, strategy_dir)
            agent_files = find_files_by_keyword(
                "", strategy_path
            )  # Obtener todos los archivos
            for agent_file in agent_files:
                agent_dir = os.path.join(strategy_path, agent_file)
                agent = load_model_results(agent_file, strategy_path)
                if agent is None:
                    if os.path.isfile(agent_dir):
                        print(f"Eliminando archivo: {agent_dir}")
                        try:
                            os.remove(agent_dir)
                            print(f"Archivo eliminado: {agent_dir}")
                        except Exception as e:
                            print(f"Error al eliminar {agent_dir}: {e}")
                    else:
                        print(f"Archivo no encontrado: {agent_dir}")
                    continue  # No procesar más este archivo

                agent_results = agent.results()
                strategy = agent_results["strategy"]
                if strategy in [
                    "e-greedy",
                    "e-decay",
                    "Boltzmann",
                    "UCB1",
                    "AsOpt-UCB",
                ]:
                    all_results.append(agent)

for criteria in criterias_list:

    fig = plot_results_per_episode_comp_plotly(all_results, criteria, 100, "log")
    fig.show()
