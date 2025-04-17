import os
import sys

# Asegurar que el path al directorio principal esté en el path del sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa: E402

from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra import get_q_table_for_path
from RLib.utils.plots import plot_results_per_episode_comp_plotly
from RLib.utils.files import load_model_results, find_files_by_keyword
from RLib.utils.serializers import QAgentSSPSerializer

# Directorio base y de resultados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "q-learning", "CityGraphs")

# Datos de entrada
city_name = "Paine, Chile" 
orig_node = 8624811405
dest_node = 8956282682
cost_distribution = "lognormal"  # "lognormal"

is_dynamic = True
alpha_type = "dynamic" if is_dynamic else "constant"
ruta_carpeta = os.path.join(RESULTS_DIR, city_name, f"{orig_node}-{dest_node}", cost_distribution , f"{alpha_type}_alpha")

# Criterias to plot
criterias_list = ['max_norm_error_normalized',
                  'max_norm_error_shortest_path_normalized',
                  'average regret']

# Inicializar una lista para almacenar los DataFrames de resultados
all_results = []

# Iterar sobre cada subcarpeta en RESULTS_DIR
for root, dirs, files in os.walk(ruta_carpeta):
    # Identificar si la carpeta corresponde a un entorno "HardPerceptron"
    if "CityGraphs" in root:
        # Extraer detalles del entorno a partir del nombre de la carpeta
        # Separar el nombre de la carpeta en partes: ['results', 'q-learning', 'CityGraphs', 'City_name', 'orig_node-dest_node', 'uniform', 'dynamic_alpha']
        parts = root.split(os.sep)
        if len(parts) < 3:
            continue
        graph_name = parts[-4:-3]  # Ejemplo: ['City_name', 'origin_node-dest_node']
        cost_distribution = parts[-2]  # Ejemplo: 'uniform'
        alpha_type = parts[-1]  # Ejemplo: 'dynamic_alpha' or 'constant_alpha'

        # Cargar los resultados de los agentes en este entorno
        for strategy_dir in dirs:
            strategy_path = os.path.join(root, strategy_dir)
            agent_files = find_files_by_keyword(
                "", strategy_path)  # Obtener todos los archivos
            for agent_file in agent_files:
                agent = load_model_results(agent_file, strategy_path)
                agent_results = agent.results()
                strategy = agent_results["strategy"]
                if strategy in ["e-greedy", "e-decay", "Boltzmann", "UCB1", "AsOpt-UCB"]:
                    all_results.append(agent)

for criteria in criterias_list:

    fig = plot_results_per_episode_comp_plotly(all_results, criteria)
    fig.show()