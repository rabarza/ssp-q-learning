import os
import sys
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402
from RLib.utils.dijkstra import (
    get_optimal_policy_and_q_star,
    get_shortest_path_from_policy,
    get_q_table_for_path,
)
from RLib.agents.ssp import QAgentSSP
from RLib.environments.ssp import SSPEnv
from RLib.utils.plots import plot_results_per_episode_comp_plotly
from RLib.action_selectors import (
    EpsilonGreedyActionSelector,
    UCB1ActionSelector,
    Exp3ActionSelector,
    AsOptUCBActionSelector,
)
from RLib.utils.files import serialize_and_save_table

# Definir la ubicación de los resultados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "q-learning", "small_graph")


# Crear un grafo simple
G = nx.DiGraph()
G.add_node(0)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
# Añadir arcos
# length es la longitud del arco en metros
# speed_kph es la velocidad media en kilómetros por hora del arco
G.add_edge(0, 1, length=100, speed_kph=50)
G.add_edge(1, 2, length=150, speed_kph=60)
G.add_edge(2, 4, length=200, speed_kph=40)
G.add_edge(0, 3, length=200, speed_kph=40)
G.add_edge(3, 2, length=150, speed_kph=50)
G.add_edge(3, 4, length=100, speed_kph=50)
# G.add_edge(4, 2, length=100, speed_kph=50)

# Definir el nodo de origen y el nodo objetivo
orig_node = 0
dest_node = 4

graph_name = f"small_graph-{G.number_of_nodes()}Nodes"
# Distribución de velocidad de los arcos
distribution = "uniform"  # "lognormal"
# Definir la ruta para guardar los resultados
save_path = os.path.join(RESULTS_DIR, f"{graph_name}/{orig_node}-{dest_node}/{distribution}/")  # noqa: E501

# Encontrar la política óptima, el camino más corto y la tabla Q* mediante Dijkstra
policy, optimal_q_table = get_optimal_policy_and_q_star(G, dest_node, distribution)  # noqa: E501

shortest_path = shortest_path = get_shortest_path_from_policy(policy, orig_node, dest_node) # noqa: E501
optimal_q_table_for_sp = get_q_table_for_path(optimal_q_table, shortest_path)

# Guardar Q* y Q* para los nodos del camino más corto en un archivo .json
serialize_and_save_table(optimal_q_table, save_path, f"optimal_q_table_distr_{distribution}.json")  # noqa: E501
serialize_and_save_table(optimal_q_table_for_sp, save_path, f"optimal_q_table_for_sp_distr_{distribution}.json")  # noqa: E501

# Crear el entorno SSP
env = SSPEnv(G, orig_node, dest_node, distribution, shortest_path)

# Tasa de aprendizaje dinámica
alpha = "10 / (t + 10)" # se evalua dinámicamente usando la variable t := numero de episodio

# Instanciar los agentes de acuerdo a sus estrategias de selección de acción
agent_eps = QAgentSSP(env, action_selector=EpsilonGreedyActionSelector(epsilon=0.1), alpha=alpha)
agent_ucb = QAgentSSP(env, action_selector=UCB1ActionSelector(c=.0001), alpha=alpha)
agent_ao_ucb = QAgentSSP(env, action_selector=AsOptUCBActionSelector(c=2), alpha=alpha)
agent_exp3 = QAgentSSP(env, action_selector=Exp3ActionSelector(eta="sqrt(t)"), alpha=alpha)
agent_exp3_2 = QAgentSSP(env, action_selector=Exp3ActionSelector(eta="log(t+1)"), alpha=alpha)

# Entrenamiento de los agentes
num_episodes = 10000
agent_eps.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)
agent_eps.save(save_path)
agent_ucb.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)
agent_ucb.save(save_path)
agent_ao_ucb.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)
agent_exp3.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)
agent_exp3.save(save_path)
agent_exp3_2.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)

# Graficar los resultados
list_agents = [agent_eps, agent_ucb, agent_exp3, agent_exp3_2]

plot_results_per_episode_comp_plotly(list_agents, "max_norm_error_normalized").show()
plot_results_per_episode_comp_plotly(list_agents, "max_norm_error_shortest_path_normalized").show()
plot_results_per_episode_comp_plotly(list_agents, "average regret").show()
plot_results_per_episode_comp_plotly(list_agents, "optimal paths").show()
plot_results_per_episode_comp_plotly(list_agents, "score").show()
plot_results_per_episode_comp_plotly(list_agents, "steps").show()
