# import the packages

# To calculate and compare the graph metrics of input test matrices

print("evaluation_test_input_graph_metrics.py starting")

import os
import numpy as np
import pandas as pd
import pickle

from utils.data import load_graphs_dict, graph_matrix_vector
from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed, compute_eigenvalue, eigenvalue_difference_laplacian_batch

print("All packages imported")

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

    evaluation_set = 'test'
    retrieve_path = os.path.join(output_path, "models", config.maindir, evaluation_set)
    graph_metrics_list = ["binary", "eigenvalue", "nodal_strength", "closeness_centrality",
                          "clustering_coefficient", "local_efficiency"]

    for metric_x in graph_metrics_list:
        compute_metric(config, retrieve_path, evaluation_set, metric_x)

    print("Model prediction over")

def compute_metric(config, retrieve_path, set1, metric_x):

    """load the graphs and also matrices, convert graphs to grahs refined and
    also graphs inverse, and compute the graph metric"""

    bvalues = [1000, 3000]
    resolutions_paths = ["2_3", "1_25"]
    subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])
    metric_dict = {}
    metric_array_dict = {}

    for j, bval in enumerate(bvalues):
        for k, res in enumerate(resolutions_paths):
            graphs_dict, subject_IDs_valid = load_graphs_dict(subject_IDs,
                                                                          os.path.join(data_path, "ds_HCP_bval_" + str(
                                                                              bval) + "_anat_0_7_dwi_" + res),
                                                                          config.scale, 0)

            if metric_x == "binary":
                matrices_number_of_fibers_dict, _ = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                        weight_value='number_of_fibers')
                metric_dict[j*2+k] = {}
                for i, subject_x in enumerate(subject_IDs_valid):
                    binary_matrix = matrices_number_of_fibers_dict[subject_x].copy()
                    metric_dict[j*2+k][subject_x] = (binary_matrix > 0) * 1.0
            elif metric_x == "eigenvalue":
                matrices_number_of_fibers_dict, _ = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                        weight_value='number_of_fibers')
                metric_dict[j*2+k] = {}
                for i, subject_x in enumerate(subject_IDs_valid):
                    metric_dict[j*2+k][subject_x] = compute_eigenvalue(
                        matrices_number_of_fibers_dict[subject_x].copy())
            else:
                with open(os.path.join(retrieve_path, f"{metric_x}_{bval}_{res}_{set1}.pkl"), "rb") as f:
                    metric_dict_values = pickle.load(f)
                metric_dict[j * 2 + k] = {}
                for i, subject_x in enumerate(subject_IDs_valid):
                    metric_dict[j * 2 + k][subject_x] = metric_dict_values[subject_x]

            if metric_x == "binary":
                metric_array_dict[j * 2 + k] = np.zeros(
                    (len(subject_IDs_valid), *metric_dict[j * 2 + k][subject_IDs_valid[0]].shape))
            else:
                metric_array_dict[j * 2 + k] = np.zeros(
                    (len(subject_IDs_valid), metric_dict[j * 2 + k][subject_IDs_valid[0]].shape[0]))
            for i, subject_x in enumerate(subject_IDs_valid):
                metric_array_dict[j * 2 + k][i] = metric_dict[j * 2 + k][subject_x]

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

    if metric_x == "eigenvalue":
        save_path_x = os.path.join(retrieve_path, "metric", metric_x)
        os.makedirs(save_path_x, exist_ok=True)
        with open(os.path.join(save_path_x, f"{metric_x}.pkl"), "wb") as f:
            pickle.dump(metric_dict, f)
        for i in metric_array_dict.keys():
            np.save(os.path.join(save_path_x, f"{metric_x}_{i}.npy"), metric_array_dict[i])

            metric_array_copy = metric_array_dict[i].copy()
            print(f"Metric shape is : {metric_array_copy.shape}")
            metric_array_mean = np.mean(metric_array_copy, axis=-1)
            metric_array_mean_df = pd.Series(metric_array_mean)

            print(f"Statistics of {metric_x} for dataset {i} is : {metric_array_mean_df.describe()}")

    return

if __name__ == "__main__":
    main()


