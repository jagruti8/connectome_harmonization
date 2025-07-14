# import the packages

print("evaluation_autoencoder2D_harmonization_graph_metrics.py starting")

import os
import numpy as np
import pandas as pd
import pickle

from utils.data import read_file_numpy, matrix_to_graph_version1, graph_version1_graph_inverse, graph_version1_graph_refined
from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed, compute_eigenvalue, eigenvalue_difference_laplacian_batch
from utils.graph_metrics import compute_local_efficiency, compute_closeness_centrality, \
    compute_nodal_strength, compute_clustering_coefficient
from create_model_path_autoencoder2D import create_model_path_function

print("All packages imported")

output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"
parameter_file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/batch_scripts/parameters"

def main():

    # Parse arguments
    config = read_configuration_param()

    # Print the argument parameters and their values
    for arg, value in vars(config).items():
        if isinstance(value, bool):
            # If the argument is boolean, check its value
            if value:
                print(f'{arg} is set to True')
            else:
                print(f'{arg} is set to False')
        else:
            # Print non-boolean argument values
            print(f'{arg}: {value}')

    # Setup seeds
    seed = config.seed
    set_seed(seed)

    parameter_file_combination_path = os.path.join(parameter_file_path, config.maindir,
                                                   f"parameters_combinations_{config.parameter_combination_number}.txt")
    saved_models_path = create_model_path_function(output_path, parameter_file_combination_path,
                                                   config.line_number, config.maindir)

    retrieve_path = os.path.join(saved_models_path, f"epoch_{config.model_epoch_chosen+1}", config.evaluation_set)

    print(retrieve_path)

    #graph_metrics_list = ["binary", "eigenvalue", "nodal_strength", "closeness_centrality",
    #                      "clustering_coefficient"]
    graph_metrics_list = ["local_efficiency"]

    metric_array_dict = {}
    metric_dict = {}

    for metric_x in graph_metrics_list:
        metric_array_dict[metric_x], metric_dict[metric_x] = compute_metric(config.evaluation_set, retrieve_path, metric_x)
        save_path = os.path.join(retrieve_path, "metric", metric_x)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"{metric_x}.pkl"), "wb") as f:
            pickle.dump(metric_dict[metric_x], f)
        for i in metric_array_dict[metric_x].keys():
            np.save(os.path.join(save_path, f"{metric_x}_{i}.npy"), metric_array_dict[metric_x][i])

            if metric_x != "binary":
                metric_array_copy = metric_array_dict[metric_x][i].copy()
                print(f"Metric shape is : {metric_array_copy.shape}")
                metric_array_mean = np.mean(metric_array_copy, axis=-1)
                metric_array_mean_df = pd.Series(metric_array_mean)

                print(f"Statistics of {metric_x} for site {i} is : {metric_array_mean_df.describe()}")

    print("Model prediction over")

def compute_metric(set1, retrieve_path, metric_x):

    """load the predicted matrices, convert them to graphs, convert graphs to grahs refined and
    also graphs inverse, and compute the graph metric, also load the label_site and the subject_IDs"""

    predicted_matrices = read_file_numpy(retrieve_path, f"{set1}_predicted_images.npy")
    target_matrices = read_file_numpy(retrieve_path, f"{set1}_target_images.npy")
    label_site = read_file_numpy(retrieve_path, f"{set1}_label_site.npy")
    subject_IDs = read_file_numpy(retrieve_path, f"{set1}_subject_IDs.npy")

    predicted_matrices_dict = {}
    predicted_graphs_dict = {}
    predicted_graphs_refined_dict = {}
    predicted_graphs_inverse_dict = {}
    metric_dict = {}
    metric_array_dict = {}
    for i in np.unique(label_site):
        predicted_matrices_dict[i] = {}
        if i==3:
            predicted_matrices_i = target_matrices[label_site==i]
        else:
            predicted_matrices_i = predicted_matrices[label_site == i]
        subject_IDs_i = subject_IDs[label_site==i]
        for j, subject_x in enumerate(subject_IDs_i):
            predicted_matrices_dict[i][subject_x] = predicted_matrices_i[j]
        if metric_x == "binary":
            metric_dict[i] = {}
            for j, subject_x in enumerate(subject_IDs_i):
                binary_matrix = predicted_matrices_dict[i][subject_x].copy()
                metric_dict[i][subject_x] = (binary_matrix > 0) * 1.0
        elif metric_x == "eigenvalue":
            metric_dict[i] = {}
            for j, subject_x in enumerate(subject_IDs_i):
                metric_dict[i][subject_x] = compute_eigenvalue(predicted_matrices_dict[i][subject_x].copy())
        elif metric_x == "nodal_strength":
            metric_dict[i] = compute_nodal_strength(predicted_matrices_dict[i], subject_IDs_i)
        else:
            predicted_graphs_dict[i] = matrix_to_graph_version1(predicted_matrices_i, subject_IDs_i)
            if metric_x == "closeness_centrality":
                predicted_graphs_inverse_dict[i] = graph_version1_graph_inverse(subject_IDs_i, predicted_graphs_dict[i],
                                                                             weight_value="weight")
                metric_dict[i] = compute_closeness_centrality(predicted_graphs_inverse_dict[i], subject_IDs_i)
            elif metric_x == "clustering_coefficient":
                predicted_graphs_refined_dict[i] = graph_version1_graph_refined(subject_IDs_i, predicted_graphs_dict[i],
                                                                             weight_value="weight")
                metric_dict[i] = compute_clustering_coefficient(predicted_graphs_refined_dict[i], predicted_matrices_dict[i],
                                                             subject_IDs_i)
            elif metric_x == "local_efficiency":
                predicted_graphs_inverse_dict[i] = graph_version1_graph_inverse(subject_IDs_i, predicted_graphs_dict[i],
                                                                             weight_value="weight")
                predicted_graphs_refined_dict[i] = graph_version1_graph_refined(subject_IDs_i, predicted_graphs_dict[i],
                                                                             weight_value="weight")
                metric_dict[i] = compute_local_efficiency(predicted_graphs_inverse_dict[i], predicted_graphs_refined_dict[i],
                                                       subject_IDs_i)

        if metric_x == "binary":
            metric_array_dict[i] = np.zeros((len(subject_IDs_i), *metric_dict[i][subject_IDs_i[0]].shape))
        else:
            metric_array_dict[i] = np.zeros((len(subject_IDs_i), metric_dict[i][subject_IDs_i[0]].shape[0]))
        for j, subject_x in enumerate(subject_IDs_i):
            metric_array_dict[i][j] = metric_dict[i][subject_x]

    print(f"Evaluating {metric_x}:")
    print(f"Shape of {metric_x} in site 3 is : {metric_array_dict[3].shape}")
    for i in range(0,3):
        print(f"Site : {i}")
        print(f"Shape of {metric_x} in site {i} is : {metric_array_dict[i].shape}")
        if metric_x == "eigenvalue":
            diff_l1, diff_l2 = eigenvalue_difference_laplacian_batch(metric_array_dict[3],
                                                                     metric_array_dict[i])
        else:
            diff_l1 = np.mean(np.abs(metric_array_dict[3]-metric_array_dict[i]))
            diff_l2 = np.mean(np.square(metric_array_dict[3] - metric_array_dict[i]))

        print(f"Mean absolute error between target and predicted {metric_x} values is : {diff_l1}")
        print(f"Mean square error between target and predicted {metric_x} values is : {diff_l2}")

    return metric_array_dict, metric_dict

if __name__ == "__main__":
    main()


