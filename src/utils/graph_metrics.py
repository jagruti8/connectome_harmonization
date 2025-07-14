import networkx as nx
import numpy as np

def local_efficiency_weighted(G_dist, G):
    """
    Compute weighted local efficiency for each node
    :param G_dist: distance graph
    :param G: similarity graph
    :return: efficiency_list
    """
    efficiency_list = [global_efficiency_weighted(G_dist.subgraph(G_dist[v]), G, v, True) for v in G_dist]
    return efficiency_list

def global_efficiency_weighted(G_dist, G, v=0, local=False):
    """
    Compute weighted global efficiency for the entire graph (local=False) or neighbourhood of a node(local=True)
    :param G_dist: distance graph
    :param G: similarity graph
    :param v: Node for which local efficiency is calculated (is of no use for global efficiency)
    :param local: bool
           False : global efficiency is calculated
           True : local efficiency is calculated
    :return: global_efficiency (graph/sub-graph of a node)
    """
    G_node_len = len(G_dist.nodes)
    global_efficiency = 0
    # The node/graph having zero or one neighbour/node has zero local/global efficiency
    if G_node_len < 2:
        return global_efficiency
    G_node = list(sorted(G_dist.nodes))
    A = np.zeros((G_node_len, G_node_len))
    for p in nx.shortest_path_length(G_dist, source=None, target=None, weight="weight", method='dijkstra'):
        for key, value in sorted(p[1].items()):
            if p[0] == key:
                continue
            else:
                if local:
                    A[G_node.index(p[0]), G_node.index(key)] = np.cbrt(
                        1 / value * G.edges[v, p[0]]['weight'] * G.edges[v, key]['weight'])
                else:
                    A[G_node.index(p[0]), G_node.index(key)] = 1 / value
    nodal_efficiency = np.sum(A, axis=0) / (A.shape[0] - 1)
    global_efficiency = np.mean(nodal_efficiency)
    return global_efficiency


def compute_global_efficiency(graphs_inverse_dict, graphs_refined_dict, subject_IDs_valid):
    """
    Compute global efficiency of the graphs
    :param graphs_inverse_dict:
    :param graphs_refined_dict:
    :param subject_IDs_valid:
    :return: metrics_dict
    """
    metrics_dict = {}
    for subject in subject_IDs_valid:
        G_dist = graphs_inverse_dict[subject]
        G_weighted = graphs_refined_dict[subject]
        metrics = global_efficiency_weighted(G_dist, G_weighted)
        metrics_dict[subject] = np.array(metrics)
    return metrics_dict

def compute_local_efficiency(graphs_inverse_dict, graphs_refined_dict, subject_IDs_valid):
    """
    Compute local efficiency of the graphs
    :param graphs_inverse_dict:
    :param graphs_refined_dict:
    :param subject_IDs_valid:
    :return: metrics_dict
    """
    metrics_dict = {}
    for subject in subject_IDs_valid:
        G_dist = graphs_inverse_dict[subject]
        G_weighted = graphs_refined_dict[subject]
        metrics = local_efficiency_weighted(G_dist, G_weighted)
        metrics_dict[subject] = np.array(metrics)
    return metrics_dict

def compute_closeness_centrality(graphs_inverse_dict, subject_IDs_valid):
    """
    Compute closeness centrality of the graphs
    :param graphs_inverse_dict:
    :param subject_IDs_valid:
    :return: metrics_dict
    """
    metrics_dict = {}
    for subject in subject_IDs_valid:
        G_dist = graphs_inverse_dict[subject]
        metrics = nx.closeness_centrality(G_dist, distance='weight')
        metrics_dict[subject] = np.array(list(metrics.values()))
    return metrics_dict

def compute_nodal_strength(matrices_dict, subject_IDs_valid):
    """
    Compute nodal strength of the matrices
    :param matrices_dict:
    :param subject_IDs_valid:
    :return: metrics_dict
    """
    metrics_dict = {}
    for subject in subject_IDs_valid:
        metrics_dict[subject] = np.sum(matrices_dict[subject], axis=-1)
    return metrics_dict

def compute_betweenness_centrality(graphs_inverse_dict, subject_IDs_valid):
    """
    Compute betweenness centrality of the graphs
    :param graphs_inverse_dict:
    :param subject_IDs_valid:
    :return: metrics_dict
    """
    metrics_dict = {}
    for subject in subject_IDs_valid:
        G_dist = graphs_inverse_dict[subject]
        metrics = nx.betweenness_centrality(G_dist, weight='weight')
        metrics_dict[subject] = np.array(list(metrics.values()))
    return metrics_dict

def compute_eigenvector_centrality(graphs_refined_dict, subject_IDs_valid):
    """
    Compute eigenvector centrality of the graphs
    :param graphs_refined_dict:
    :param subject_IDs_valid:
    :return: metrics_dict
    """
    metrics_dict = {}
    for subject in subject_IDs_valid:
        G_weighted = graphs_refined_dict[subject]
        metrics = nx.eigenvector_centrality_numpy(G_weighted, weight='weight')
        metrics_dict[subject] = np.array(list(metrics.values()))
    return metrics_dict

def compute_clustering_coefficient(graphs_refined_dict, matrices_dict, subject_IDs_valid):
    """
    Compute clustering coefficient of the graphs
    :param graphs_refined_dict:
    :param matrices_dict:
    :param subject_IDs_valid:
    :return: metrics_dict
    """
    metrics_dict = {}
    for subject in subject_IDs_valid:
        G_weighted = graphs_refined_dict[subject]
        metrics = nx.clustering(G_weighted, weight='weight')
        metrics_unnormalized = np.array(list(metrics.values())).copy()
        metrics_unnormalized = metrics_unnormalized * (matrices_dict[subject].max())
        metrics_dict[subject] = metrics_unnormalized
    return metrics_dict