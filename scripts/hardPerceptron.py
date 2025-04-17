import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402
from RLib.action_selectors import (
    EpsilonGreedyActionSelector,
    EpsilonGreedyDecayActionSelector,
    UCB1ActionSelector,
    AsOptUCBActionSelector,
    BoltzmannSelector,
)
from RLib.utils.files import serialize_and_save_table
from RLib.utils.dijkstra import (
    get_optimal_policy_and_q_star,
    get_shortest_path_from_policy,
    get_q_table_for_path,
)
from RLib.graphs.perceptron import create_hard_perceptron_graph, plot_network_graph
from RLib.agents.ssp import QAgentSSP, QAgentSSPSarsa0
from RLib.environments.ssp import SSPEnv
import multiprocessing
import networkx as nx
import winsound

model_name = "q-learning"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", model_name)

# Crear el grafo
nodes_by_layer = [10] * 60
nodes_by_layer.insert(0, 1)
nodes_by_layer.append(1)
seed = 20
# Crear el grafo
costs_distribution = "uniform"  # puede ser "uniform" o "lognormal"
graph = create_hard_perceptron_graph(
    nodes_by_layer,
    min_length=20,
    max_length=60,
    costs_distribution=costs_distribution,
    seed=seed,
)
graph_data = nx.node_link_data(graph)
graph_data["seed"] = seed
graph_data["nodes_by_layer"] = nodes_by_layer
# plot_network_graph(graph, False)

# Definir el nodo de origen y el nodo objetivo
origin_node = 1
target_node = 0
graph_name = f"{len(nodes_by_layer)}Layers-{graph.number_of_nodes()}Nodes-Seed{seed}"
save_path = os.path.join(
    RESULTS_DIR, f"HardPerceptron/{graph_name}/{costs_distribution}/"
)
serialize_and_save_table(
    graph_data,
    os.path.join(RESULTS_DIR, f"HardPerceptron/{graph_name}"),
    f"graph_data_seed{seed}.json",
)

# Encontrar la política óptima, el camino más corto y la tabla Q*
optimal_policy, optimal_q_table = get_optimal_policy_and_q_star(
    graph, target_node, costs_distribution
)
shortest_path = get_shortest_path_from_policy(optimal_policy, origin_node, target_node)
optimal_q_table_for_sp = get_q_table_for_path(optimal_q_table, shortest_path)

# Guardar tablas en un archivo .json
serialize_and_save_table(
    optimal_q_table, save_path, f"optimal_q_table_distr_{costs_distribution}.json"
)
serialize_and_save_table(
    optimal_q_table_for_sp,
    save_path,
    f"optimal_q_table_for_sp_distr_{costs_distribution}.json",
)

# Definir parámetros
num_episodes = 20000
# formula = '1000 / (t + 1000)'  # global clock
# formula = '1000 / (N(s) + 1000)'  # global clock
formula = "1000 / (N(s,a) + 1000)"  # local clock


def train_agent(agent: QAgentSSP | QAgentSSPSarsa0, save_path: str):
    agent.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)
    agent.save(save_path)


if __name__ == "__main__":
    processes = []

    number_of_trials = 1
    for trial in range(number_of_trials):
        eps_selector = EpsilonGreedyActionSelector(epsilon=0.1)
        environment = SSPEnv(
            graph, origin_node, target_node, costs_distribution, shortest_path
        )
        agent = QAgentSSP(
            environment=environment, alpha=formula, action_selector=eps_selector
        )
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()

    c_values_epsDecay = [0.1, 0.5, 1]
    for c in c_values_epsDecay:
        epsDecay_selector = EpsilonGreedyDecayActionSelector(constant=c)
        environment = SSPEnv(
            graph, origin_node, target_node, costs_distribution, shortest_path
        )
        agent = QAgentSSP(
            environment=environment, alpha=formula, action_selector=epsDecay_selector
        )
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()

    eta_values = [
        "log(n_s) / q_range",
        "log(n_s * log(n_s+1)) / q_range",
    ]
    for eta in eta_values:
        exp3_selector = BoltzmannSelector(eta=eta)
        environment = SSPEnv(
            graph, origin_node, target_node, costs_distribution, shortest_path
        )
        agent = QAgentSSP(
            environment=environment, alpha=formula, action_selector=exp3_selector
        )
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()

    c_values_asOptUCB = [0.0001, 0.001, 0.01]
    for c in c_values_asOptUCB:
        asOptUCB_selector = AsOptUCBActionSelector(c=c)
        environment = SSPEnv(
            graph, origin_node, target_node, costs_distribution, shortest_path
        )
        agent = QAgentSSP(
            environment=environment, alpha=formula, action_selector=asOptUCB_selector
        )
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()

    # fmt: off
    c_values_ucb = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4]
    # fmt: on
    for c in c_values_ucb:
        ucb_selector = UCB1ActionSelector(c=c)
        environment = SSPEnv(
            graph, origin_node, target_node, costs_distribution, shortest_path
        )
        agent = QAgentSSP(
            environment=environment, alpha=formula, action_selector=ucb_selector
        )
        p = multiprocessing.Process(target=train_agent, args=(agent, save_path))
        processes.append(p)
        p.start()

    # Esperar a que todos los procesos terminen
    for p in processes:
        p.join()
