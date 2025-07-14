# import the packages

# To calculate the mae, pearson correlation, fingerprinting accuracy, Idiff
# of test-retest matrices for the highest b-value and highest resolution

print("evaluation_test_retest_all_metrics.py starting")

import os
import numpy as np
import pandas as pd

from utils.data import load_graphs_dict, graph_matrix_vector
from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed
from scipy.stats import pearsonr

print("All packages imported")

data_path = "/data/hagmann_group/jagruti/dataset_1065"
file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/dataset_creation"

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

    datasets = ['HCP_DWI','HCP_DWI_Retest']
    evaluation_set = 'test_retest'

    compute_metrics(config, evaluation_set, datasets)

    print("Model prediction over")

def compute_metrics(config, set1, datasets):

    """load the graphs and also matrices for test-retest, convert graphs to grahs refined and
    also graphs inverse, and compute the metrics"""

    bvalue = 3000
    resolution_path = "1_25"
    subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])
    vectors_number_of_fibers_array_dict = {}

    for dataset_x in datasets:
        graphs_dict, subject_IDs_valid = load_graphs_dict(subject_IDs,
                                                         os.path.join(data_path, dataset_x, "ds_HCP_bval_" + str(
                                                                          bvalue) + "_anat_0_7_dwi_" + resolution_path),
                                                                          config.scale, 0)
        _, vectors_number_of_fibers_dict = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                weight_value='number_of_fibers')

        vectors_number_of_fibers_array_dict[dataset_x] = []

        for i, subject_x in enumerate(subject_IDs_valid):
            vectors_number_of_fibers_array_dict[dataset_x].append(vectors_number_of_fibers_dict[subject_x])

        vectors_number_of_fibers_array_dict[dataset_x] = np.stack(vectors_number_of_fibers_array_dict[dataset_x])

    print(f"Shape of matrices in dataset {datasets[0]} is : {vectors_number_of_fibers_array_dict[datasets[0]].shape}")
    print(f"Shape of matrices in dataset {datasets[1]} is : {vectors_number_of_fibers_array_dict[datasets[1]].shape}")

    difference_matrix_s1_s2 = np.zeros((len(vectors_number_of_fibers_array_dict[datasets[0]]),
                                        len(vectors_number_of_fibers_array_dict[datasets[0]])))

    for j in range(len(vectors_number_of_fibers_array_dict[datasets[0]])):
        for k in range(len(vectors_number_of_fibers_array_dict[datasets[0]])):
            difference_matrix_s1_s2[j, k] = np.mean(np.abs(vectors_number_of_fibers_array_dict[datasets[0]][j] -
                                                           vectors_number_of_fibers_array_dict[datasets[1]][k]))

    pearsonr_all = []
    for j in range(len(vectors_number_of_fibers_array_dict[datasets[0]])):
        pearsonr_all.append(pearsonr(vectors_number_of_fibers_array_dict[datasets[0]][j], vectors_number_of_fibers_array_dict[datasets[1]][j])[0])

    pearsonr_all_mean = np.mean(np.array(pearsonr_all))
    pearsonr_all_std = np.std(np.array(pearsonr_all))

    mae_all_mean = np.mean(np.abs(vectors_number_of_fibers_array_dict[datasets[0]] -
                                  vectors_number_of_fibers_array_dict[datasets[1]]))
    mae_all_std = np.std(np.mean(np.abs(vectors_number_of_fibers_array_dict[datasets[0]] -
                                  vectors_number_of_fibers_array_dict[datasets[1]]),axis=1))

    accuracy_all = find_accuracy(difference_matrix_s1_s2)

    Idiff_all = find_Idiff(difference_matrix_s1_s2)

    print(f"MAE: \n {mae_all_mean} ± {mae_all_std}")
    print(f"Accuracy across axis 0: \n {accuracy_all[0]}")
    print(f"Accuracy across axis 1: \n {accuracy_all[1]}")
    print(f"Pearsonr: \n {pearsonr_all_mean} ± {pearsonr_all_std}")
    print(f"Idiff: \n {Idiff_all}")

    return


def find_Idiff(difference_matrix):
    avg_diagonal = np.mean(difference_matrix[np.diag_indices_from(difference_matrix)])
    off_diagonal = np.mean(np.concatenate((difference_matrix[np.triu_indices_from(difference_matrix,k=1)],
                                           difference_matrix[np.tril_indices_from(difference_matrix,k=-1)])))
    Idiff = off_diagonal - avg_diagonal
    return Idiff

def find_accuracy(difference_matrix):
    true_values = np.arange(0,difference_matrix.shape[0],1)
    predicted_values_axis_0 = np.argmin(difference_matrix,axis=0)
    predicted_values_axis_1 = np.argmin(difference_matrix,axis=1)
    accuracy_axis_0 = np.sum(true_values==predicted_values_axis_0)/len(true_values)
    accuracy_axis_1 = np.sum(true_values==predicted_values_axis_1)/len(true_values)
    return np.array([accuracy_axis_0, accuracy_axis_1])


if __name__ == "__main__":
    main()


