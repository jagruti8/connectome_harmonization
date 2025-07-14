# import the packages

# To calculate different graph metrics

print("calculate_graph_metrics_actual_matrices.py starting")

import os
import numpy as np
import pandas as pd
import pickle

from utils.data import load_graphs_dict, graph_matrix_vector, graph_version1_graph_inverse, graph_version1_graph_refined
from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed
from utils.graph_metrics import compute_local_efficiency, compute_closeness_centrality, \
    compute_nodal_strength, compute_clustering_coefficient

data_path = "/data/hagmann_group/jagruti/dataset_1065/HCP_DWI"
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

    save_path = os.path.join(output_path, "models", config.maindir)
    #datasets = ['train','val','test']
    datasets = ['train']
    print(f"Computing : {config.graph_metric}")

    for set1 in datasets:
        save_setx_path = os.path.join(save_path, set1)
        os.makedirs(save_setx_path, exist_ok=True)
        compute_metric(config, set1, save_setx_path)

    print("Model training over")


def compute_metric(config, set1, save_setx_path):

    """load the graphs and also matrices, convert graphs to grahs refined and
    also graphs inverse, and compute the graph metric"""

    #bvalues = [1000, 3000]
    bvalues = [3000]
    #resolutions_paths = ["2_3", "1_25"]
    resolutions_paths = ["1_25"]

    subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])

    for j, bval in enumerate(bvalues):
        for k, res in enumerate(resolutions_paths):
            graphs_dict, subject_IDs_valid = load_graphs_dict(subject_IDs,
                                                                          os.path.join(data_path, "ds_HCP_bval_" + str(
                                                                              bval) + "_anat_0_7_dwi_" + res),
                                                                          config.scale, 0)

            if config.graph_metric == "nodal_strength":
                matrices_number_of_fibers_dict, _ = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                          weight_value='number_of_fibers')
                metric_dict = compute_nodal_strength(matrices_number_of_fibers_dict, subject_IDs_valid)
            elif config.graph_metric == "closeness_centrality":
                graphs_inverse_dict = graph_version1_graph_inverse(subject_IDs_valid, graphs_dict,
                                                                             weight_value='number_of_fibers')
                metric_dict = compute_closeness_centrality(graphs_inverse_dict, subject_IDs_valid)
            elif config.graph_metric == "clustering_coefficient":
                matrices_number_of_fibers_dict, _ = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                        weight_value='number_of_fibers')
                graphs_refined_dict = graph_version1_graph_refined(subject_IDs_valid, graphs_dict,
                                                                   weight_value='number_of_fibers')
                metric_dict = compute_clustering_coefficient(graphs_refined_dict, matrices_number_of_fibers_dict,
                                                         subject_IDs_valid)
            elif config.graph_metric == "local_efficiency":
                graphs_inverse_dict = graph_version1_graph_inverse(subject_IDs_valid, graphs_dict,
                                                                         weight_value='number_of_fibers')
                graphs_refined_dict = graph_version1_graph_refined(subject_IDs_valid, graphs_dict,
                                                                         weight_value='number_of_fibers')
                metric_dict = compute_local_efficiency(graphs_inverse_dict, graphs_refined_dict,
                                                   subject_IDs_valid)

            metric_array = np.zeros((len(subject_IDs_valid), metric_dict[subject_IDs_valid[0]].shape[0]))
            for i_x, subject_x in enumerate(subject_IDs_valid):
                metric_array[i_x] = metric_dict[subject_x]

            metric_array_copy = metric_array.copy()
            print(f"Metric shape is : {metric_array_copy.shape}")
            metric_array_mean = np.mean(metric_array_copy, axis=-1)
            metric_array_mean_df = pd.Series(metric_array_mean)

            print(f"Statistics of {config.graph_metric} of graphs for b-value {bval} "
                  f"and spatial resolution {res} for {set1} data : {metric_array_mean_df.describe()}")

            with open(os.path.join(save_setx_path, f"{config.graph_metric}_{bval}_{res}_{set1}.pkl"), "wb") as f:
                pickle.dump(metric_dict, f)
            np.save(os.path.join(save_setx_path, f"{config.graph_metric}_{bval}_{res}_{set1}.npy"), metric_array)

    return

if __name__ == "__main__":
    main()


