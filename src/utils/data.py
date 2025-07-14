"""Module with functions to load data in different formats"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import copy
import csv
import torch

def read_subject_list(list_path):
    """ Reading the text file for the subject_IDs.

    Parameters
    ----------
    list_path : str
        Path to the .txt file contatning the list of subject IDs

    Returns
    -------
    subject_IDs : list of strings
        List of subject_IDs

    """
    # Initiate a list for containing the subject_IDs
    subject_IDs = []
    # Read the text file
    with open(list_path, 'r') as f:
        subjects = f.readlines()
    # Appending subject_IDs to the list
    for subject in subjects:
        subject_IDs.append(subject.replace('\n', ''))

    return subject_IDs


def load_graphs_dict(subject_IDs, folder_path, scale=3, is_hpc=1):
    """ Loading the connectivity matrices in the form networkx graphs for the given subjects,
    parcellation scale and folder_path and only the matrices with the correct parcellation for that scale are considered

    Parameters
    ----------
    subject_IDs : list of strings
        List of subject_IDs
    folder_path : str
        Path of folder containing the connectivity matrices
    scale : int
        Parcellation scale for the connectivity matrix. Default is 3

    Returns
    -------
    graphs_dict : dictionary of networkx graphs
        Dictionary of networkx graphs for the given subjects and scale
    subject_IDs_valid : list of strings
        List of valid subject_IDs which have the correct parcellation
    """

    graphs_dict = {}
    subject_IDs_valid = []
    nroi = None
    #print(folder_path)
    for subject in subject_IDs:
        if is_hpc:
            filepath = os.path.join(folder_path, f"sub-{subject}", f"dwi",
                                f"sub-{subject}_atlas-L2018_res-scale{scale}_conndata-network_connectivity.gpickle")
        else:
            filepath = os.path.join(folder_path, f"sub-{subject}_atlas-L2018_res-scale{scale}_conndata-network_connectivity.gpickle")
        #print(filepath)
        if os.path.exists(filepath):
            #G = nx.readwrite.gpickle.read_gpickle(filepath)
            with open(filepath, 'rb') as f:
                G = pickle.load(f)
            if nroi is None:
                nroi = G.order()
                graphs_dict[subject] = G
                subject_IDs_valid.append(subject)
            elif nroi != G.order():
                print(f"subject {subject} has {G.order()} number of parcellations")
                continue
            else:
                graphs_dict[subject] = G
                subject_IDs_valid.append(subject)


    return graphs_dict, subject_IDs_valid

def graph_matrix_vector(subject_IDs_valid, graphs_dict, weight_value = 'number_of_fibers'):
    """
    Constructing vectors from structural connectivity matrices with edge weights as "number_of_fibers"
    """
    matrices = {}
    SC_vectors = {}
    for i, subject in enumerate(subject_IDs_valid):
        G = copy.deepcopy(graphs_dict[subject])
        for u, v, d in graphs_dict[subject].edges(data=True):
            weight_val = d[weight_value]
            G.remove_edge(u,v)
            if u!=v:
                G.add_edge(u,v)
                G[u][v]['weight'] = weight_val
        nodes = list(G.nodes)
        nodes.sort()  # sort nodes so that the order is preserved
        matrix = nx.adjacency_matrix(G, nodelist=nodes).todense()
        # only the upper triangular matrix without the diagonal is chosen as it is a symmetric matrix and assuming there is no self-loop ,i.e, diagonal elements are zero
        SC_vectors[subject] = matrix[np.triu_indices_from(matrix, k=1)]
        matrices[subject] = np.array(matrix)
    return matrices, SC_vectors

def graph_matrix_mean_vector(subject_IDs_valid, graphs_dict, threshold, weight_value = 'number_of_fibers'):
    """
    Constructing vectors, matrices and mean binary matrices from structural connectivity matrices with edge weights as "number_of_fibers"
    """
    nsub = len(subject_IDs_valid)  # no of subjects
    nroi = graphs_dict[subject_IDs_valid[0]].order() # no of nodes in each matrix
    matrices_mean = np.zeros((nsub, nroi, nroi))
    matrices = {}
    SC_vectors = {}
    for i, subject in enumerate(subject_IDs_valid):
        G = copy.deepcopy(graphs_dict[subject])
        for u, v, d in graphs_dict[subject].edges(data=True):
            weight_val = d[weight_value]
            G.remove_edge(u,v)
            if u!=v:
                G.add_edge(u,v)
                G[u][v]['weight'] = weight_val
        nodes = list(G.nodes)
        nodes.sort()  # sort nodes so that the order is preserved
        matrix = nx.adjacency_matrix(G, nodelist=nodes).todense()
        # only the upper triangular matrix without the diagonal is chosen as it is a symmetric matrix and assuming there is no self-loop ,i.e, diagonal elements are zero
        SC_vectors[subject] = matrix[np.triu_indices_from(matrix, k=1)]
        matrices[subject] = np.array(matrix)
        matrices_mean[i] = matrices[subject].copy()
    print(f"Shape of mean matrices : {matrices_mean.shape}")
    matrices_mean = (matrices_mean > 0) * 1.0
    matrices_mean = np.mean(matrices_mean, axis=0)
    matrices_mean = (matrices_mean > threshold) * 1.0
    return matrices, matrices_mean, SC_vectors



def graph_version1_graph_refined(subject_IDs_valid, graphs_dict, weight_value = None, self_loop=False):
    """
    Refining graph to have no self-loops as even if the self-loops have a value of zero,
    they affect the calculation of betweenness centrality
    :param subject_IDs_valid:
    :param graphs_dict:
    :param weight_value:
    :return: graphs_refined_dict:
    """
    graphs_refined_dict = {}
    for subject in subject_IDs_valid:
        G = copy.deepcopy(graphs_dict[subject])
        for u,v,d in graphs_dict[subject].edges(data=True):
            weight_val = d[weight_value]
            G.remove_edge(u,v)
            if u!=v:
                G.add_edge(u,v)
                G[u][v]['weight'] = weight_val
            if self_loop:
                if u==v:
                    G.add_edge(u, v)
                    G[u][v]['weight'] = 1
        graphs_refined_dict[subject] = G
    return graphs_refined_dict




def graph_version1_graph_inverse(subject_IDs_valid, graphs_dict, weight_value = None):
    """
    Convert similarity to distance graph with self-loops removed
    :param subject_IDs_valid:
    :param graphs_dict:
    :param weight_value:
    :return: graphs_inverse_dict
    """
    graphs_inverse_dict = {}
    for subject in subject_IDs_valid:
        G = copy.deepcopy(graphs_dict[subject])
        for u,v,d in graphs_dict[subject].edges(data=True):
            weight_val = d[weight_value]
            G.remove_edge(u,v)
            if u!=v:
                G.add_edge(u,v)
                G[u][v]['weight'] = 1/weight_val
        graphs_inverse_dict[subject] = G
    return graphs_inverse_dict

def vector_to_matrix(vector,n):
    """
    Convert a vector of size int(n*(n-1)/1) to a symmetric matrix of size n*n
    :param vector:
    :param n:
    :return: matrix
    """
    matrix = np.zeros((n,n))
    triu = np.triu_indices(n,k=1)
    tril = np.tril_indices(n,k=-1)
    matrix[triu] = vector
    matrix[tril] = matrix.T[tril]
    return matrix

def vector_to_matrix_tensor(vector,n,device):
    """
    Convert vector of size (batch,int(n*(n-1)/1)) to a symmetric matrix of size (batch,n*n)
    :param vector:
    :param n:
    :param device:
    :return: matrix
    """
    matrix = torch.zeros((vector.shape[0],n,n)).to(device)
    triu = torch.triu_indices(n,n,offset=1).to(device)
    matrix[:,triu[0],triu[1]] = vector
    matrix = torch.permute(matrix,(0,2,1))
    matrix[:,triu[0],triu[1]] = vector
    return matrix

# convert matrix to graph version1
def matrix_to_graph_version1(matrix, subject_IDs_valid):
    """
    Convert a matrix to graph version 1 (where self-loops may or may not be there)
    :param matrix:
    :param subject_IDs_valid:
    :return: graphs_dict
    """
    graphs_dict = {}
    for i in range(matrix.shape[0]):
        graphs_dict[subject_IDs_valid[i]] = nx.from_numpy_array(matrix[i])
    return graphs_dict

def matrix_to_graph(adj_matrix):
    """Convert an adjacency matrix to a NetworkX graph."""
    graph = nx.from_numpy_array(adj_matrix)
    return graph

def fill_diagonal_with_zero(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i == j:
                matrix[i][j] = 0
    return matrix

def fill_diagonal_with_one(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i == j:
                matrix[i][j] = 1
    return matrix

def read_images(path, file_path):
    images = np.load(os.path.join(path, file_path))
    images = np.squeeze(images)
    images = np.stack([(image + image.T) / 2 for image in images])
    images = np.round(np.where(images < 0, 0, images))
    images = np.stack([fill_diagonal_with_zero(image) for image in images])

    return images

def read_file_numpy(path, file_path):
    file_x = np.load(os.path.join(path,file_path))
    return file_x

def matrix_to_vector(images):
    images_vector = np.stack([a[np.triu_indices_from(a,k=1)] for a in images])
    return images_vector

def matrix_to_vector_one(matrix):
    image_vector = matrix[np.triu_indices_from(matrix, k=1)]
    return image_vector

def save_file(file_array, file_path):
    """
    Save a csv file
    :param file_array:
    :param file_path:
    :return:
    """
    file_list = file_array.tolist()
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(file_list) #change this to writer.writerows(file_list) for saving list of lists


def read_file(file_path):
    """
    Read a csv file
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        file_list = list(reader)

    file_array = []
    for row in file_list:
        nwrow = []
        for r in row:
            nwrow.append(eval(r))
        file_array.append(nwrow)
    return np.array(file_array)

def create_directory(path):
    # os.makedirs creates the directory if it doesn't exist
    # exist_ok=True means it won't raise an error if the directory already exists
    os.makedirs(path, exist_ok=True)
