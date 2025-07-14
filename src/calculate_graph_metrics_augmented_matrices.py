# import the packages

# To calculate different graph metrics

print("calculate_graph_metrics_augmented_matrices.py starting")

import os
import numpy as np
import pandas as pd
import pickle

from utils.data import read_file_numpy, matrix_to_graph_version1, graph_version1_graph_inverse, graph_version1_graph_refined
from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed
from utils.graph_metrics import compute_local_efficiency, compute_closeness_centrality, \
    compute_nodal_strength, compute_clustering_coefficient
from create_model_path import create_model_path_function

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

    retrieve_path = os.path.join(saved_models_path, "mixed_train_subjects")

    print(config.graph_metric)

    metric_array, metric_dict = compute_metric(config, retrieve_path)

    save_path = os.path.join(retrieve_path, "metric")
    os.makedirs(save_path, exist_ok=True)

    metric_array_copy = metric_array.copy()
    print(f"Metric shape is : {metric_array_copy.shape}")
    metric_array_mean = np.mean(metric_array_copy, axis=-1)
    metric_array_mean_df = pd.Series(metric_array_mean)

    if config.graph_metric == "local_efficiency" and config.bvalue == "3000":
        print(f"Statistics of {config.graph_metric} of augmented graphs for part {config.part_number+1} for b-value {config.bvalue} "
              f"and spatial resolution {config.resolution} : {metric_array_mean_df.describe()}")
        with open(os.path.join(save_path, f"{config.graph_metric}_{config.bvalue}_{config.resolution}_{config.part_number+1}.pkl"), "wb") as f:
            pickle.dump(metric_dict, f)
        np.save(os.path.join(save_path, f"{config.graph_metric}_{config.bvalue}_{config.resolution}_{config.part_number+1}.npy"), metric_array)
    else:
        print(f"Statistics of {config.graph_metric} of augmented graphs for b-value {config.bvalue} "
              f"and spatial resolution {config.resolution} : {metric_array_mean_df.describe()}")
        with open(os.path.join(save_path,f"{config.graph_metric}_{config.bvalue}_{config.resolution}.pkl"), "wb") as f:
            pickle.dump(metric_dict, f)
        np.save(os.path.join(save_path,f"{config.graph_metric}_{config.bvalue}_{config.resolution}.npy"), metric_array)

    print("Model training over")


def compute_metric(config, retrieve_path):

    """load the augmented matrices for the given bvalue and spatial resolution, convert them to graphs, convert graphs to grahs refined and
    also graphs inverse, and compute the graph metric"""

    augmented_matrices = read_file_numpy(retrieve_path, f"{config.bvalue}_{config.resolution}_augmented_matrices.npy")
    subject_IDs_valid = np.arange(0, len(augmented_matrices))
    if config.graph_metric == "local_efficiency" and config.bvalue == "3000":
        augmented_matrices = augmented_matrices[config.len_part*config.part_number : config.len_part*(config.part_number+1)]
        subject_IDs_valid = subject_IDs_valid[config.len_part*config.part_number : config.len_part*(config.part_number+1)]
    augmented_matrices_dict = {}
    for i, subject_x in enumerate(subject_IDs_valid):
        augmented_matrices_dict[subject_x] = augmented_matrices[i]
    if config.graph_metric == "nodal_strength":
        metric_dict = compute_nodal_strength(augmented_matrices_dict, subject_IDs_valid)
    else:
        augmented_graphs_dict = matrix_to_graph_version1(augmented_matrices, subject_IDs_valid)
        if config.graph_metric == "closeness_centrality":
            augmented_graphs_inverse_dict = graph_version1_graph_inverse(subject_IDs_valid, augmented_graphs_dict,
                                                                         weight_value="weight")
            metric_dict = compute_closeness_centrality(augmented_graphs_inverse_dict, subject_IDs_valid)
        elif config.graph_metric == "clustering_coefficient":
            augmented_graphs_refined_dict = graph_version1_graph_refined(subject_IDs_valid, augmented_graphs_dict,
                                                                         weight_value="weight")
            metric_dict = compute_clustering_coefficient(augmented_graphs_refined_dict, augmented_matrices_dict,
                                                         subject_IDs_valid)
        elif config.graph_metric == "local_efficiency":
            augmented_graphs_inverse_dict = graph_version1_graph_inverse(subject_IDs_valid, augmented_graphs_dict,
                                                                         weight_value="weight")
            augmented_graphs_refined_dict = graph_version1_graph_refined(subject_IDs_valid, augmented_graphs_dict,
                                                                         weight_value="weight")
            metric_dict = compute_local_efficiency(augmented_graphs_inverse_dict, augmented_graphs_refined_dict,
                                                   subject_IDs_valid)

    metric_array = np.zeros((len(subject_IDs_valid), metric_dict[subject_IDs_valid[0]].shape[0]))
    for i, subject_x in enumerate(subject_IDs_valid):
        metric_array[i] = metric_dict[subject_x]

    return metric_array, metric_dict

if __name__ == "__main__":
    main()


