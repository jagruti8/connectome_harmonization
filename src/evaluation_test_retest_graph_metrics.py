# import the packages

# To calculate the graph metrics of test-retest matrices for the highest b-value and highest resolution

print("evaluation_test_retest_graph_metrics.py starting")

import os
import numpy as np
import pandas as pd
import pickle

from utils.data import load_graphs_dict, graph_matrix_vector, graph_version1_graph_inverse, graph_version1_graph_refined
from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed, compute_eigenvalue, eigenvalue_difference_laplacian_batch
from utils.graph_metrics import compute_local_efficiency, compute_closeness_centrality, \
    compute_nodal_strength, compute_clustering_coefficient

print("All packages imported")

data_path = "/data/hagmann_group/jagruti/dataset_1065"
file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/dataset_creation"
output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"

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

    save_path = os.path.join(output_path, "models", config.maindir, "test_retest")
    datasets = ['HCP_DWI','HCP_DWI_Retest']
    evaluation_set = 'test_retest'
    #graph_metrics_list = ["binary", "eigenvalue", "nodal_strength", "closeness_centrality",
    #                      "clustering_coefficient"]
    graph_metrics_list = ["local_efficiency"]

    metric_array_dict = {}
    metric_dict = {}

    for metric_x in graph_metrics_list:
        metric_array_dict[metric_x], metric_dict[metric_x] = compute_metric(config, evaluation_set, metric_x, datasets)
        save_path_x = os.path.join(save_path, "metric", metric_x)
        os.makedirs(save_path_x, exist_ok=True)
        with open(os.path.join(save_path_x, f"{metric_x}.pkl"), "wb") as f:
            pickle.dump(metric_dict[metric_x], f)
        for dataset_x in metric_array_dict[metric_x].keys():
            save_path_xd = os.path.join(save_path_x, dataset_x)
            os.makedirs(save_path_xd, exist_ok=True)
            np.save(os.path.join(save_path_xd, f"{metric_x}.npy"), metric_array_dict[metric_x][dataset_x])

            if metric_x != "binary":
                metric_array_copy = metric_array_dict[metric_x][dataset_x].copy()
                print(f"Metric shape is : {metric_array_copy.shape}")
                metric_array_mean = np.mean(metric_array_copy, axis=-1)
                metric_array_mean_df = pd.Series(metric_array_mean)

                print(f"Statistics of {metric_x} for dataset {dataset_x} is : {metric_array_mean_df.describe()}")

    print("Model prediction over")

def compute_metric(config, set1, metric_x, datasets):

    """load the graphs and also matrices for test-retest, convert graphs to grahs refined and
    also graphs inverse, and compute the graph metric"""

    bvalue = 3000
    resolution_path = "1_25"
    subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])
    metric_dict = {}
    metric_array_dict = {}

    for dataset_x in datasets:
        graphs_dict, subject_IDs_valid = load_graphs_dict(subject_IDs,
                                                         os.path.join(data_path, dataset_x, "ds_HCP_bval_" + str(
                                                                          bvalue) + "_anat_0_7_dwi_" + resolution_path),
                                                                          config.scale, 0)

        if metric_x == "binary":
            matrices_number_of_fibers_dict, _ = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                    weight_value='number_of_fibers')
            metric_dict[dataset_x] = {}
            for j, subject_x in enumerate(subject_IDs_valid):
                binary_matrix = matrices_number_of_fibers_dict[subject_x].copy()
                metric_dict[dataset_x][subject_x] = (binary_matrix > 0) * 1.0
        elif metric_x == "eigenvalue":
            matrices_number_of_fibers_dict, _ = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                    weight_value='number_of_fibers')
            metric_dict[dataset_x] = {}
            for j, subject_x in enumerate(subject_IDs_valid):
                metric_dict[dataset_x][subject_x] = compute_eigenvalue(matrices_number_of_fibers_dict[subject_x].copy())
        elif metric_x == "nodal_strength":
            matrices_number_of_fibers_dict, _ = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                      weight_value='number_of_fibers')
            metric_dict[dataset_x] = compute_nodal_strength(matrices_number_of_fibers_dict, subject_IDs_valid)
        elif metric_x == "closeness_centrality":
            graphs_inverse_dict = graph_version1_graph_inverse(subject_IDs_valid, graphs_dict,
                                                                         weight_value='number_of_fibers')
            metric_dict[dataset_x] = compute_closeness_centrality(graphs_inverse_dict, subject_IDs_valid)
        elif metric_x == "clustering_coefficient":
            matrices_number_of_fibers_dict, _ = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                    weight_value='number_of_fibers')
            graphs_refined_dict = graph_version1_graph_refined(subject_IDs_valid, graphs_dict,
                                                               weight_value='number_of_fibers')
            metric_dict[dataset_x] = compute_clustering_coefficient(graphs_refined_dict, matrices_number_of_fibers_dict,
                                                     subject_IDs_valid)
        elif metric_x == "local_efficiency":
            graphs_inverse_dict = graph_version1_graph_inverse(subject_IDs_valid, graphs_dict,
                                                                     weight_value='number_of_fibers')
            graphs_refined_dict = graph_version1_graph_refined(subject_IDs_valid, graphs_dict,
                                                                     weight_value='number_of_fibers')
            metric_dict[dataset_x] = compute_local_efficiency(graphs_inverse_dict, graphs_refined_dict,
                                               subject_IDs_valid)

        if metric_x == "binary":
            metric_array_dict[dataset_x] = np.zeros((len(subject_IDs_valid), *metric_dict[dataset_x][subject_IDs_valid[0]].shape))
        else:
            metric_array_dict[dataset_x] = np.zeros((len(subject_IDs_valid), metric_dict[dataset_x][subject_IDs_valid[0]].shape[0]))
        for j, subject_x in enumerate(subject_IDs_valid):
            metric_array_dict[dataset_x][j] = metric_dict[dataset_x][subject_x]

    print(f"Evaluating {metric_x}:")
    print(f"Shape of {metric_x} in dataset {datasets[0]} is : {metric_array_dict[datasets[0]].shape}")
    print(f"Shape of {metric_x} in dataset {datasets[1]} is : {metric_array_dict[datasets[1]].shape}")

    if set(metric_dict[datasets[0]].keys()) != set(metric_dict[datasets[1]].keys()):
        print("Dictionaries have different subjects")
        return metric_array_dict, metric_dict

    print("Subjects match!")

    if metric_x == "eigenvalue":
        diff_l1, diff_l2 = eigenvalue_difference_laplacian_batch(metric_array_dict[datasets[0]],
                                                                 metric_array_dict[datasets[1]])
    else:
        diff_l1 = np.mean(np.abs(metric_array_dict[datasets[0]] - metric_array_dict[datasets[1]]))
        diff_l2 = np.mean(np.square(metric_array_dict[datasets[0]] - metric_array_dict[datasets[1]]))

    print(f"Mean absolute error between target and predicted {metric_x} values is : {diff_l1}")
    print(f"Mean square error between target and predicted {metric_x} values is : {diff_l2}")

    return metric_array_dict, metric_dict

if __name__ == "__main__":
    main()


