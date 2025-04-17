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
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Datos de entrada
city_name = "Malloquito, CL - Radio 1600"
costs_distribution = "uniform"
orig_node = 1239125937
dest_node = 7292736884
# city_name = "Malloquito, CL - Radio 1200"
# orig_node = 6133873484
# dest_node = 10923464031
alpha_type = "dynamic"

# Construcción de la ruta de la carpeta
ruta_carpeta = os.path.join(RESULTS_DIR, city_name, f"{orig_node}-{dest_node}", costs_distribution, f"{alpha_type}_alpha")
print(ruta_carpeta)	

# Verificar si la ruta existe antes de listar archivos
if os.path.exists(ruta_carpeta):
    print(os.listdir(ruta_carpeta))
else:
    print(f"La ruta especificada no existe: {ruta_carpeta}")

# Cargar modelos de QAgentSSP
greedy_files = find_files_by_keyword("e-", os.path.join(ruta_carpeta, "e-greedy"))
greedy_agents = list(map(lambda x: load_model_results(x, os.path.join(ruta_carpeta, "e-greedy")), greedy_files))

ucb_files = find_files_by_keyword("UCB1", os.path.join(ruta_carpeta, "UCB1"))
ucb_agents = list(map(lambda x: load_model_results(x, os.path.join(ruta_carpeta, "UCB1")), ucb_files))

exp3_files = find_files_by_keyword("exp3", os.path.join(ruta_carpeta, "exp3"))
exp3_agents = list(map(lambda x: load_model_results(x, os.path.join(ruta_carpeta, "exp3")), exp3_files))

criterias_list = ['error', 'policy error', 'score', 'steps', 'average regret']

for criteria in criterias_list:
    agents = greedy_agents + ucb_agents + exp3_agents
    # agents = ucb_agents
    print(agents)
    fig = plot_results_per_episode_comp_plotly(agents, criteria)
    fig.show()

# if __name__ == "__main__":
#     from RLib.utils.serializers import serialize_table
#     import json

#     for agent in agents:
#         q_table = agent.q_table
#         path = agent.best_path()
#         q_table_for_sp = get_q_table_for_path(q_table, path)
#         serialized_q_table_for_sp = serialize_table(q_table_for_sp)
#         json_q_table_for_sp = json.dumps(serialized_q_table_for_sp, indent=4)
#         with open(os.path.join(RESULTS_DIR, city_name, f"q_star_for_shortest_path_{city_name}_{orig_node}-{dest_node}_{agent.strategy}.json"), "w") as f:
#             f.write(json_q_table_for_sp)
