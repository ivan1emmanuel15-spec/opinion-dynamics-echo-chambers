import networkx as nx
import numpy as np


def crear_red(n=1000, m=3, seed=42):
    """
    Construye una red de Barabási-Albert y aplica
    la regla de seguimiento para obtener una red dirigida.
    """

    # Red no dirigida
    #barabasi_albert_graph(n, m, seed) genera un grafo de Barabási-Albert con n nodos y m aristas añadidas desde un nodo nuevo a nodos existentes. El parámetro seed se utiliza para inicializar el generador de números aleatorios y garantizar la reproducibilidad del grafo generado.
    Grafo = nx.barabasi_albert_graph(n,m=m,seed=seed)
    #erdos_renyi_graph(n, p, seed) genera un grafo aleatorio de Erdős-Rényi con n nodos y una probabilidad p de que exista una arista entre cada par de nodos. El parámetro seed se utiliza para inicializar el generador de números aleatorios y garantizar la reproducibilidad del grafo generado.
    #Grafo = nx.erdos_renyi_graph(n, 0.009, directed=False, seed=seed)


    # Red dirigida
    Grafodir = nx.DiGraph()
    Grafodir.add_nodes_from(Grafo.nodes())

    # Grados de los nodos
    degree = dict(Grafo.degree())
    sumadegrados = sum(degree.values())

    # Participación de cada nodo en el total de grados
    grado = {}

    for i in degree:
        grado[i] = degree[i] / sumadegrados

    # Regla de seguimiento
    zeta = min(grado.values())

    for u, v in Grafo.edges():

        if grado[u] <= grado[v] + zeta:
            Grafodir.add_edge(u, v)

        if grado[v] <= grado[u] + zeta:
            Grafodir.add_edge(v, u)

    # Matriz de adyacencia
    nodos = sorted(Grafodir.nodes())

    A = nx.to_numpy_array(
        Grafodir,
        nodelist=nodos
    )

    # Número de seguidores
    followers = dict(Grafodir.in_degree())

    ordenados = sorted(
        followers,
        key=followers.get,
        reverse=True
    )

    # Tres nodos más importantes
    maxim = ordenados[0]
    segundo_maxim = ordenados[1]
    tercer_maxim = ordenados[2]

    return (
        Grafo,
        Grafodir,
        A,
        degree,
        maxim,
        segundo_maxim,
        tercer_maxim
    )