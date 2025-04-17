import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra import get_q_table_for_path
from RLib.utils.plots import plot_results_per_episode_comp_plotly
from RLib.utils.files import load_model_results, find_files_by_keyword
from RLib.utils.serializers import QAgentSSPSerializer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_name = "q-learning"
# model_name = "sarsa0"

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"results", model_name
)


number_of_layers = 62
number_of_nodes = 602
seed = 20
graph_name = f"Perceptron/{number_of_layers}Layers-{number_of_nodes}Nodes-Seed{seed}"
cost_distribution = "lognormal"  # "lognormal"
is_dynamic = True
alpha_type = "dynamic" if is_dynamic else "constant"
ruta_carpeta = os.path.join(
    RESULTS_DIR, graph_name, cost_distribution, f"{alpha_type}_alpha/"
)

# Criterias to plot
criterias_list = [
    "max_norm_error_normalized",
    "max_norm_error_shortest_path_normalized",
    "average regret",
]
# Inicializar una lista para almacenar los DataFrames de resultados
all_results = []


# Función para extraer el número de capas de un nombre de archivo
def get_layers_from_name(name):
    return int(name.split("Layers")[0])


# Iterar sobre cada subcarpeta en RESULTS_DIR
for root, dirs, files in os.walk(ruta_carpeta):
    # Identificar si la carpeta corresponde a un entorno "Perceptron"
    if "Perceptron" in root and not "HardPerceptron" in root:
        # Extraer detalles del entorno a partir del nombre de la carpeta
        # Separar el nombre de la carpeta en partes: ['results', 'Perceptron', '42Layers-402Nodes-Seed20', 'uniform', 'dynamic_alpha']
        print(root)
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
                    try:
                        print(f"Eliminando {agent_dir}")
                        os.remove(agent_dir)
                    except Exception as e:
                        print(f"Error al eliminar {agent_dir}: {e}")
                    continue
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

    fig = plot_results_per_episode_comp_plotly(all_results, criteria, True, "logit")
    fig.show()
