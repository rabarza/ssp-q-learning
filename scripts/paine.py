import os
import sys
import multiprocessing
import osmnx as ox
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402
from RLib.action_selectors import (
    EpsilonGreedyActionSelector,
    EpsilonGreedyDecayActionSelector,
    UCB1ActionSelector,
    AsOptUCBActionSelector,
    BoltzmannSelector,
)
from RLib.utils.files import serialize_and_save_table,  save_graph_plot_as_pdf
from RLib.utils.dijkstra import (
    get_optimal_policy_and_q_star,
    get_shortest_path_from_policy,
    get_q_table_for_path,
)
from RLib.graphs.city import ensure_proper_policy
from RLib.agents.ssp import QAgentSSP
from RLib.environments.ssp import SSPEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "q-learning", "CityGraphs")

# Configuración de la ciudad y nodos de origen/destino
location_name = "Paine, Chile"
origin_node = 8624811405
target_node = 8956282682
costs_distribution = "lognormal"  # Se puede cambiar a "lognormal"

# Definir la ruta para guardar los resultados
graph_path = os.path.join(RESULTS_DIR, f"{location_name}/{origin_node}-{target_node}/")  # noqa: E501
save_path = os.path.join(RESULTS_DIR, f"{location_name}/{origin_node}-{target_node}/{costs_distribution}/")

# Descargar el grafo y preprocesarlo
print(f"Descargando grafo de {location_name}...")
graph = ox.graph_from_place(location_name, network_type="drive")
print("Grafo descargado exitosamente!")

# Preprocesamiento del grafo
graph = ox.add_edge_speeds(graph)  # Añadir velocidad a los arcos (speed_kph)

# Quitar los arcos con largo demasiado corto (este paso es opcional, previene que el agente se quede en un ciclo por arcos de costo 0)
min_length = 1
edges_to_remove = [(u, v) for u, v, data in graph.edges(data=True) if data.get('length', 0) < min_length]
graph.remove_edges_from(edges_to_remove)

# Obtener el componente más grande (débilmente conexo)
# graph = ox.truncate.largest_component(graph, strongly=False)

# Filtrar el grafo para garantizar la política proper (se crea un subgrafo con los nodos que pueden llegar al nodo objetivo)
graph = ensure_proper_policy(graph, target_node)

# Añadir un arco recurrente de largo 0 en el nodo terminal (target_node)
outgoing_edges = list(graph.out_edges(target_node))  # Arcos salientes del nodo destino
if outgoing_edges:
    graph.remove_edges_from(outgoing_edges)
graph.add_edge(target_node, target_node, length=0, speed_kph=30)

assert origin_node in graph.nodes and target_node in graph.nodes, "Los nodos de origen y destino deben estar en el grafo."

# Guardar el grafo preprocesado en un archivo GraphML (opcional)
# output_graphml_path = os.path.join("paine_graph.graphml")
# ox.save_graphml(graph, output_graphml_path)
save_graph_plot_as_pdf(graph, origin_node, target_node, graph_path, location_name, 5)
print("Grafo preprocesado y guardado exitosamente!")

# Encontrar la política óptima, el camino más corto y la tabla Q*
print("Calculando la política óptima y el camino más corto...")
optimal_policy, optimal_q_table = get_optimal_policy_and_q_star(graph, target_node, costs_distribution)  # noqa: E501
shortest_path = get_shortest_path_from_policy(optimal_policy, origin_node, target_node)
print("Política óptima y camino más corto calculados exitosamente!")

# Obtener la tabla Q* para los nodos del camino más corto
optimal_q_table_for_sp = get_q_table_for_path(optimal_q_table, shortest_path)

# Guardar tablas en un archivo .json
serialize_and_save_table(optimal_q_table, save_path, f"optimal_q_table_distr_{costs_distribution}.json")
serialize_and_save_table(optimal_q_table_for_sp, save_path, f"optimal_q_table_for_sp_distr_{costs_distribution}.json")

# Definir parámetros
num_episodes = 30000
# formula = '5000 / (t + 5000)' # global clock
formula = '5000 / (N(s,a) + 5000)' # local clock

# Entrenamiento de agentes en procesos separados
def train_agent(agent, save_path):
    agent.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)
    agent.save(save_path)

if __name__ == "__main__":
    processes = []
    number_of_trials = 5
    for trial in range(number_of_trials):
        eps_selector = EpsilonGreedyActionSelector(epsilon=0.1)
        environment = SSPEnv(graph, origin_node, target_node, costs_distribution, shortest_path)
        agent = QAgentSSP(environment=environment, alpha=formula, action_selector=eps_selector)
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()

    eta_values = ['log( n_s ) / q_range', 'log(n_s * log(n_s + 1)) / q_range']
    for eta in eta_values:
        exp3_selector = BoltzmannSelector(eta=eta)
        environment = SSPEnv(graph, origin_node, target_node, costs_distribution, shortest_path)
        agent = QAgentSSP(environment=environment, alpha=formula, action_selector=exp3_selector)
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()
        
    c_values_ucb = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4] # noqa: E501
    for c in c_values_ucb:
        ucb_selector = UCB1ActionSelector(c=c)
        environment = SSPEnv(graph, origin_node, target_node, costs_distribution, shortest_path)
        agent = QAgentSSP(environment=environment, alpha=formula, action_selector=ucb_selector)
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()
    
    c_values_asOptUCB = [0.0001, 0.001, 0.01]
    for c in c_values_asOptUCB:
        asOptUCB_selector = AsOptUCBActionSelector(c=c)
        environment = SSPEnv(graph, origin_node, target_node, costs_distribution, shortest_path)
        agent = QAgentSSP(environment=environment, alpha=formula, action_selector=asOptUCB_selector)
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()

    c_values_epsDecay = [0.1, 0.5, 1]
    for c in c_values_epsDecay:
        epsDecay_selector = EpsilonGreedyDecayActionSelector(constant=c)
        environment = SSPEnv(graph, origin_node, target_node, costs_distribution, shortest_path)
        agent = QAgentSSP(environment=environment, alpha=formula, action_selector=epsDecay_selector)
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()

    # Esperar a que todos los procesos terminen
    for p in processes:
        p.join()