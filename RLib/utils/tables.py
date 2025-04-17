import copy
import networkx as nx
import numpy as np

# ======================= Generar tabla =======================


def dict_states_zeros(graph: nx.Graph) -> dict:
    """Retorna un diccionario con los estados con valor 0. Es útil para inicializar la tabla del número de visitas a cada estado, por ejemplo. Tiene la forma {estado: 0, ..., estado: 0}"""
    return {state: 0 for state, actions in nx.to_dict_of_lists(graph).items()}


def dict_states_actions_zeros(graph: nx.Graph) -> dict:
    """Retorna un diccionario con los estados y acciones con valor 0. Es útil para inicializar la tabla Q, o la tabla de la cantidad de veces que se ha visitado cada par estado-acción.
    Tiene la forma {estado: {accion: 0, ..., accion: 0}, ..., estado: {accion: 0, ..., accion: 0}}
    """
    return {
        state: {action: 0 for action in actions}
        for state, actions in nx.to_dict_of_lists(graph).items()
    }


def dict_states_actions_random(graph: nx.Graph) -> dict:
    """Retorna un diccionario con los estados y acciones con valor aleatorio. Es útil para inicializar la tabla Q, o la tabla de la cantidad de veces que se ha visitado cada par estado-acción. Tiene la forma {estado: {accion: valor_aleatorio, ..., accion: valor_aleatorio}, ..., estado: {accion: valor_aleatorio, ..., accion: valor_aleatorio}}"""
    return {
        state: {action: np.random.random() for action in actions}
        for state, actions in nx.to_dict_of_lists(graph).items()
    }


def dict_states_actions_constant(graph: nx.Graph, constant: float) -> dict:
    """Retorna un diccionario con los estados y acciones con valor aleatorio. Es útil para inicializar la tabla Q, o la tabla de la cantidad de veces que se ha visitado cada par estado-acción. Tiene la forma {estado: {accion: valor_aleatorio, ..., accion: valor_aleatorio}, ..., estado: {accion: valor_aleatorio, ..., accion: valor_aleatorio}}"""
    table = {
        state: {action: constant for action in actions}
        for state, actions in nx.to_dict_of_lists(graph).items()
    }
    return table


# ======================= Norma máxima =======================


def max_norm(q_table1: dict, q_table2: dict, path: list = None) -> float:
    """
    Calcula la norma máxima entre dos diccionarios. Es útil para medir el error en la norma infinita de Q := max|Q - Q*|.

    Parameters
    ----------
    q_table1: dict
        Tabla Q, debe ser un diccionario de la forma {estado: {accion: valor, ..., accion: valor}, ..., estado: {accion: valor, ..., accion: valor}}
    q_table2: dict
        Tabla Q* debe ser un diccionario de la forma {estado: {accion: valor, ..., accion: valor}, ..., estado: {accion: valor, ..., accion: valor}}
    path: list
        Lista de nodos que conforman el camino. más corto entre el nodo de inicio y el nodo de destino. Por defecto es None, lo que significa que se calcula la norma máxima para todos los estados y acciones. Si se pasa un camino, se calcula la norma máxima para los estados y acciones del camino.
    """
    assert (
        q_table1.keys() == q_table2.keys()
    ), "Las tablas Q deben tener los mismos estados"
    restas = []
    if path is None:
        # restar para cada estado y acción
        for state in q_table1.keys():
            for action in q_table1[state].keys():
                resta = abs(q_table1[state][action] - q_table2[state][action])
                restas.append(resta)
    else:
        for index in range(len(path) - 1):
            node = path[index]
            next_node = path[index + 1]
            resta = abs(q_table1[node][next_node] - q_table2[node][next_node])
            restas.append(resta)
    return max(restas)


def argmax_q_table(q_table, state):
    """
    Retorna la acción con mayor valor Q(s,a) para un estado s: argmax_a Q(s,a)
    """
    argmax_action = max(q_table[state], key=q_table[state].get)
    return argmax_action


def max_q_table(q_table, state):
    """
    Retorna el valor máximo Q(s,a) para un estado s: max_a Q(s,a)
    """
    if state in q_table and q_table[state]:
        return max(list(q_table[state].values()))
    else:
        raise Exception(
            f"El estado {state} no está en q_table o su diccionario de acciones está vacío."
        )


# ======================= Resta de diccionarios =======================
def resta_diccionarios(diccionario1, diccionario2):
    """
    Resta dos diccionarios. Si los valores de las llaves son diccionarios, se llama recursivamente a la función hasta encontrar el valor máximo.

    Parameters
    ----------
    diccionario1: dict
        diccionario con valores numéricos o diccionarios
    diccionario2: dict
        diccionario con valores numéricos o diccionarios
    """
    diccionario3 = {}
    for key in diccionario1:
        # Verificar si diccionario1[key] es un diccionario
        if isinstance(diccionario1[key], dict) and isinstance(diccionario2[key], dict):
            # restar los diccionarios recursivamente
            diccionario3[key] = resta_diccionarios(diccionario1[key], diccionario2[key])
        # verificar si diccionario1[key] es un número
        elif isinstance(diccionario1[key], (int, float)) and isinstance(
            diccionario2[key], (int, float)
        ):
            # restar los valores
            diccionario3[key] = abs(diccionario1[key] - diccionario2[key])
        # sino, arrojar un error
        else:
            raise ValueError("Los valores no son numéricos o diccionarios.")
    return diccionario3


# ======================= Max value in dict =======================


def max_value_in_dict(diccionario):
    """
    Retorna el valor máximo de un diccionario con valores numéricos. Funciona para diccionarios anidados. Con la forma {key: {key: value, ..., key: value}, ..., key: {key: value, ..., key: value}}

    Parameters
    ----------
    diccionario: dict
        diccionario con valores numéricos
    """
    max_values = [max(diccionario[keys].values()) for keys in diccionario]
    return max(max_values)
