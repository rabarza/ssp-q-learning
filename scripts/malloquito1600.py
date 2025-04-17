import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402
from RLib.action_selectors import (
    EpsilonGreedyActionSelector,
    UCB1ActionSelector,
    BoltzmannSelector,
)
from RLib.utils.files import save_model_results, serialize_and_save_table
from RLib.utils.dijkstra import (
    get_optimal_policy_and_q_star,
    get_shortest_path_from_policy,
    get_q_table_for_path,
)
from RLib.agents.ssp import QAgentSSP
from RLib.environments.ssp import SSPEnv
import numpy as np
import osmnx as ox
import winsound  # Para hacer sonar un beep al finalizar el entrenamiento

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "q-learning", "CityGraphs")



# Descargar el grafo y obtener el componente conexo más grande
radius = 1600
location_name = f"Malloquito, CL - Radio {radius}"
point = -33.62643, -70.84496

costs_distribution = "uniform"  # "lognormal"
# Descargar el grafo y preprocesarlo
graph = ox.graph_from_point(point, dist=radius, network_type='drive')
# Nodos de origen y destino
origin_node = ox.nearest_nodes(graph, X=-70.86129,Y=-33.61203)
target_node = ox.nearest_nodes(graph, X=-70.84492,Y=-33.62644)
assert origin_node in graph.nodes, "El nodo de origen no está en el grafo."
assert target_node in graph.nodes, "El nodo de destino no está en el grafo."

# Definir la ruta para guardar los resultados
save_path = os.path.join(RESULTS_DIR, f"{location_name}/{origin_node}-{target_node}/{costs_distribution}/")  # noqa: E501


# Preprocesamiento del grafo
# Añadir velocidad a los arcos (speed_kph)
graph = ox.add_edge_speeds(graph, hwy_speeds=30)
graph.to_directed()
# Obtener el componente fuertemente conexo más grande
graph = ox.truncate.largest_component(
    graph, strongly=True
)

# Añadir un arco recurrente de largo 0 en el nodo terminal
outgoing_edges = list(graph.out_edges(target_node))  # arcos salientes del nodo
if outgoing_edges:
    graph.remove_edges_from(outgoing_edges)
graph.add_edge(target_node, target_node, length=0, speed_kph=30)

# Encontrar la política óptima, el camino más corto y la tabla Q*
optimal_policy, optimal_q_table = get_optimal_policy_and_q_star(graph, target_node, costs_distribution)  # noqa: E501

shortest_path = get_shortest_path_from_policy(optimal_policy, origin_node, target_node)  # noqa: E501

# Obtener la tabla Q* para los nodos del camino más corto
optimal_q_table_for_sp = get_q_table_for_path(optimal_q_table, shortest_path)
# Guardar tablas en un archivo .json
serialize_and_save_table(optimal_q_table, save_path, f"optimal_q_table_distr_{costs_distribution}.json")  # noqa: E501
serialize_and_save_table(optimal_q_table_for_sp, save_path, f"optimal_q_table_for_sp_distr_{costs_distribution}.json")  # noqa: E501

NUM_EPISODES = 5000
# Crear un entorno
environment = SSPEnv(graph, origin_node, target_node,
                     costs_distribution, shortest_path)

# Entrenar agentes con diferentes estrategias
agents = []
is_dynamic_alpha = True
alpha_type = "dynamic" if is_dynamic_alpha else "constant"
c_values = np.linspace(0.0001, 0.1, 20)
# c_values = [0.01]
selectors = {
    "e-greedy": EpsilonGreedyActionSelector(epsilon=0.1),
    "UCB1": UCB1ActionSelector,
    "exp3": BoltzmannSelector(eta="sqrt(t)"),
}

for c in c_values:
    for strategy, selector_class in selectors.items():
        selector = selector_class(
            c=c) if strategy == "UCB1" else selector_class
        # Crear y entrenar el agente
        agent = QAgentSSP(
            environment,
            dynamic_alpha=is_dynamic_alpha,
            alpha_formula="40 / (N(s,a) + 40)",
            action_selector=selector
        )
        print(f"Training agent with strategy: {selector.strategy} and cost distribution: {environment.costs_distribution}") # noqa: E501
        agent.train(NUM_EPISODES, shortest_path=shortest_path,
                    q_star=optimal_q_table)
        agents.append(agent)

        results_dir = os.path.join(save_path, f"{alpha_type}_alpha/{selector.strategy}/")
        os.makedirs(results_dir, exist_ok=True)
        save_model_results(agent, results_path=results_dir)
