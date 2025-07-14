# import the packages

print("linear_mixed_travelling_harmonization.py starting")

import os
import random
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge

from utils.data import load_graphs_dict, graph_matrix_vector, vector_to_matrix
from utils.config_advanced import read_configuration_param
from utils.functions_basic import check_unique_and_no_intersection, \
    combine_vectors, combine_vectors_equal, combine_vectors_linear, find_matrix_shape_from_upper_triangular_numpy

print("All packages imported")

data_path = "/data/hagmann_group/jagruti/dataset_1065/HCP_DWI"
file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/dataset_creation"
output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"

def main():

    # Parse arguments
    config = read_configuration_param()

    # Setup seeds
    seed = config.seed
    set_seed(seed)

    if config.mixed:
        directory_name1 = "MIXED"
    else:
        directory_name1 = "ALLTRAVELLING"

    if config.augment:
        directory_name2 ="AUGMENT_"+str(config.num_pairs)
        if config.linear_augment:
            directory_name2 = os.path.join(directory_name2,
                                           "LINEAR_" + (str(config.linear_augment_lambda).replace(".", "_")))
        else:
            if config.random_augment:
                directory_name2 = os.path.join(directory_name2,"RANDOM")
            else:
                directory_name2 = os.path.join(directory_name2, "EQUAL")
    else:
        directory_name2 = "NO_AUGMENT"

    save_path = os.path.join(output_path, "models", config.maindir, directory_name1,
                             directory_name2, f"SEED_{config.seed}", config.ckdir)
    os.makedirs(save_path, exist_ok=True)

    train_label_eval, train_dataset_eval, train_subject_IDs_eval = create_dataset(config.train_set, 'eval', save_path, config)
    if config.mixed:
        train_label_train, train_dataset_train = create_dataset(config.train_set, 'train', save_path, config)
    else:
        train_label_train, train_dataset_train = train_label_eval, train_dataset_eval
    val_label_eval, val_dataset_eval, val_subject_IDs_eval = create_dataset(config.val_set, 'eval', save_path, config)
    test_label_eval, test_dataset_eval, test_subject_IDs_eval = create_dataset(config.test_set, 'eval', save_path, config)

    # diffusion MRI parameters used
    bvalues = [1000, 3000]
    resolutions = [2.3, 1.25]

    X_train = train_dataset_train.copy()
    X_label = train_label_train.copy()
    X_eval_dict = {}
    X_eval_dict['subject_IDs'] = {}
    X_eval_dict['subject_IDs']['train'] = train_subject_IDs_eval.copy()
    X_eval_dict['subject_IDs']['val'] = val_subject_IDs_eval.copy()
    X_eval_dict['subject_IDs']['test'] = test_subject_IDs_eval.copy()
    X_eval_dict['features'] = {}
    X_eval_dict['features']['train'] = train_dataset_eval.copy()
    X_eval_dict['features']['val'] = val_dataset_eval.copy()
    X_eval_dict['features']['test'] = test_dataset_eval.copy()
    X_eval_dict['label'] = {}
    X_eval_dict['label']['train'] = train_label_eval.copy()
    X_eval_dict['label']['val'] = val_label_eval.copy()
    X_eval_dict['label']['test'] = test_label_eval.copy()

    print("X_train shape for training : {}".format(X_train.shape))
    print("X_train shape for evaluation : {}".format(X_eval_dict['features']['train'].shape))
    print("X_val shape for evaluation : {}".format(X_eval_dict['features']['val'].shape))
    print("X_test shape for evaluation : {}".format(X_eval_dict['features']['test'].shape))

    # Applying linear regression to each component of the original data across the training subjects
    n_connections = X_train.shape[1]
    A_sc_number_of_fibers_coefficients_complete = np.zeros((n_connections, len(bvalues) * len(resolutions)))
    for i in range(n_connections):
        A_sc_number_of_fibers_coefficients_complete[i,:] = linear_regression_model_complete(X_train[:,i], X_label, bvalues, resolutions)

    sets = ['train', 'val', 'test']
    unique_labels = [0, 1, 2, 3]
    subject_IDs_dict = {}
    labels_dict = {}
    A_sc_number_of_fibers_dict = {}
    A_sc_number_of_fibers_harmonized_dict = {}
    beta_0 = np.expand_dims(A_sc_number_of_fibers_coefficients_complete[:, 0], axis=0) #shape is now then 1x(n_connections)
    beta_1 = np.expand_dims(A_sc_number_of_fibers_coefficients_complete[:, 1], axis=0) #shape is now then 1x(n_connections)
    beta_2 = np.expand_dims(A_sc_number_of_fibers_coefficients_complete[:, 2], axis=0) #shape is now then 1x(n_connections)
    beta_3 = np.expand_dims(A_sc_number_of_fibers_coefficients_complete[:, 3], axis=0) #shape is now then 1x(n_connections)
    for set_x in sets:
        subject_IDs_dict[set_x] = []
        labels_dict[set_x] = []
        A_sc_number_of_fibers_dict[set_x] = []
        A_sc_number_of_fibers_harmonized_dict[set_x] = []
        for label_x in unique_labels:
            ID_x = np.where(X_eval_dict['label'][set_x] == label_x)
            if label_x == 3:
                A_sc_number_of_fibers_harmonized_x = X_eval_dict['features'][set_x][ID_x]
            else:
                bvalue_x = bvalues[int(label_x/2)]
                resolution_x = resolutions[int(label_x % 2)]
                A_sc_number_of_fibers_harmonized_x = X_eval_dict['features'][set_x][ID_x] \
                                                                        + beta_1/1000*(3000-bvalue_x) \
                                                                        + beta_2*(1.25-resolution_x) \
                                                                        + beta_3/1000*(3000*1.25 - bvalue_x*resolution_x)
            subject_IDs_dict[set_x].append(X_eval_dict['subject_IDs'][set_x][ID_x])
            labels_dict[set_x].append(X_eval_dict['label'][set_x][ID_x])
            A_sc_number_of_fibers_dict[set_x].append(X_eval_dict['features'][set_x][ID_x])
            A_sc_number_of_fibers_harmonized_dict[set_x].append(
                A_sc_number_of_fibers_harmonized_x)
        A_sc_number_of_fibers_harmonized_dict[set_x] = np.stack(A_sc_number_of_fibers_harmonized_dict[set_x], axis=1)
        A_sc_number_of_fibers_harmonized_dict[set_x] = np.round(
            np.where(A_sc_number_of_fibers_harmonized_dict[set_x] < 0,
                     0,
                     A_sc_number_of_fibers_harmonized_dict[set_x]))
        subject_IDs_dict[set_x] = np.stack(subject_IDs_dict[set_x], axis=1)
        labels_dict[set_x] = np.stack(labels_dict[set_x], axis=1)
        A_sc_number_of_fibers_dict[set_x] = np.stack(A_sc_number_of_fibers_dict[set_x], axis=1)

        np.save(os.path.join(save_path, set_x + "_subject_IDs.npy"), subject_IDs_dict[set_x])

        np.save(os.path.join(save_path, set_x+"_labels.npy"), labels_dict[set_x])

        np.save(os.path.join(save_path, set_x+"_input_images.npy"), A_sc_number_of_fibers_dict[set_x])

        np.save(os.path.join(save_path, set_x+"_predicted_images.npy"), A_sc_number_of_fibers_harmonized_dict[set_x])

    print("Model prediction over")

def set_seed(seed):

    # Optionally set PYTHONHASHSEED to ensure consistent hash behavior
    #os.environ['PYTHONHASHSEED'] = str(seed)

    # Set seeds for Python random, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU

    # Ensure reproducibility with CUDA
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False
    #torch.backends.cudnn.deterministic = True  # Ensures that the same inputs yield the same outputs
    #torch.backends.cudnn.benchmark = False     # Disables some optimizations for reproducibility

    # Fill uninitialized memory with a known pattern for added determinism
    #torch.utils.deterministic.fill_uninitialized_memory=True

    # If you're on a GPU, synchronize to ensure order of operations
    #if torch.cuda.is_available():
    #    torch.cuda.synchronize()


# linear regression model : Y = b0 + b1*b-value/1000 + b2*dwi + b3*b-value*dwi/1000 where Y is the number of fibers
def linear_regression_model_complete(a, y_label_train, bvalues, resolutions):
    bval_train = np.array([bvalues[int(x/2)] for x in y_label_train])
    res_train = np.array([resolutions[int(x%2)] for x in y_label_train])
    bval_res_train = np.multiply(bval_train, res_train)
    number_of_fibers = a.copy()
    X = np.vstack((bval_train/1000, res_train, bval_res_train/1000)).T
    y = number_of_fibers
    
    # define model
    model = LinearRegression()
    # fit model
    model.fit(X, y)
    return np.append(model.intercept_, model.coef_)

def create_dataset(set1, set2, path, config):
    """load the graphs, the matrices and the sites for the subject set"""

    bvalues = [1000, 3000]
    resolutions_paths = ["2_3", "1_25"]

    # load list of subjects
    if config.mixed == True and set2=="train":
        path_subjects = os.path.join(path, "mixed_train_subjects")
        print("Mixed training with independent data points")
        if os.path.exists(path_subjects):
            print(f"{path_subjects} exists. Downloading the subjects from there")
            mixed_subjects_dict = {}
            for j, bval in enumerate(bvalues):
                for k, res in enumerate(resolutions_paths):
                    mixed_subjects_dict[str(bval) + "_" + res] = list(pd.read_csv(os.path.join(path_subjects, str(bval) + "_" + res + "_subjects.csv"), header=None, index_col=None)[0])
        else:
            print(f"{path_subjects} does not exist. Will create the required directories and then save the files")
            os.makedirs(path_subjects, exist_ok=True)
            # load list of subjects
            subject_IDs = list(
                pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])
            no_of_subjects = len(subject_IDs)
            random.shuffle(subject_IDs)
            IDs_start = [0, int(no_of_subjects / 4), int(no_of_subjects / 2), int(3 * no_of_subjects / 4)]
            IDs_end = [int(no_of_subjects / 4), int(no_of_subjects / 2), int(3 * no_of_subjects / 4), no_of_subjects]
            mixed_subjects_dict = {}
            m_x = 0
            for j, bval in enumerate(bvalues):
                for k, res in enumerate(resolutions_paths):
                    mixed_subjects_dict[str(bval) + "_" + res] = subject_IDs[IDs_start[m_x]:IDs_end[m_x]]
                    pd.DataFrame(list(mixed_subjects_dict[str(bval) + "_" + res])).to_csv(
                        os.path.join(path_subjects, str(bval) + "_" + res + "_subjects.csv"), index=False, header=False)
                    m_x += 1

            # check if the subject lists have any intersection or not and if they comprise all the subjects in the training set
            list_subject_IDs = []
            total_subject_IDs = []
            for key_x, value_x in mixed_subjects_dict.items():
                list_subject_IDs.append(value_x)
                total_subject_IDs.extend(value_x)
            total_subject_IDs = np.unique(total_subject_IDs)
            assert len(total_subject_IDs) == len(subject_IDs)
            print("All subject IDs from training are included")
            if check_unique_and_no_intersection(list_subject_IDs):
                print("All subject IDs in the training set are used just once")
            else:
                print("All subject IDs in the training set are not used just once")
    else:
        print(f"Mixed_{config.mixed}_set_{set1}_for_{set2}")
        subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])
        graphs_dict = {}
        subject_IDs_valid_common = None
        for j, bval in enumerate(bvalues):
            for k, res in enumerate(resolutions_paths):
                graphs_dict[str(bval) + "_" + res], subject_IDs_valid = load_graphs_dict(subject_IDs,
                                                                  os.path.join(data_path, "ds_HCP_bval_" + str(bval) + "_anat_0_7_dwi_" + res),
                                                                  config.scale, 0)

                print(f"Length of valid subject IDs : {len(subject_IDs_valid)}")

                if subject_IDs_valid_common is None:
                    subject_IDs_valid_common = subject_IDs_valid
                else:
                    subject_IDs_valid_common = np.intersect1d(subject_IDs_valid_common, subject_IDs_valid)

        print(f"Length of common valid subject IDs : {len(subject_IDs_valid_common)}")

    track_subjects_IDs = []
    y_label_all = []
    vectors_number_of_fibers_all = []

    nroi = None

    for j, bval in enumerate(bvalues):
        for k, res in enumerate(resolutions_paths):

            # loading the .gpickle files for the chosen subjects
            if config.mixed == True and set2 == "train":
                print("Mixed training with independent data points")
                graphs_dict, subject_IDs_valid = load_graphs_dict(
                    mixed_subjects_dict[str(bval) + "_" + res],
                    os.path.join(data_path, "ds_HCP_bval_" + str(bval) + "_anat_0_7_dwi_" + res),
                    config.scale, 0)
                _, vectors_number_of_fibers = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                  weight_value='number_of_fibers')
            else:
                print(f"Mixed_{config.mixed}_set_{set1}_for_{set2}")
                _, vectors_number_of_fibers = graph_matrix_vector(subject_IDs_valid_common, graphs_dict[str(bval) + "_" + res],
                                                               weight_value='number_of_fibers')

            for subject_x, vectors_number_of_fibers_x in vectors_number_of_fibers.items():
                track_subjects_IDs.append(subject_x)
                vectors_number_of_fibers_all.append(vectors_number_of_fibers_x)
                y_label_all.append(j*2+k)

            if config.augment == True and config.mixed == True and set2 == "train":
                augmented_matrices = []

                print(
                    f"Inside the loop for augmented matrices, number of subjects from which pairs are formed : {len(subject_IDs_valid)}")

                # Step 1: Generate unique random pairs of subjects as given by config.num_pairs
                pairs = set()
                while len(pairs) < config.num_pairs:
                    subject1, subject2 = random.sample(list(subject_IDs_valid), 2)
                    pair = tuple(sorted((subject1, subject2)))  # Store pairs in sorted order to avoid duplicates
                    pairs.add(pair)

                for subject1, subject2 in pairs:
                    vector1 = vectors_number_of_fibers[subject1]
                    vector2 = vectors_number_of_fibers[subject2]

                    # Step 2a: Create the combined vector
                    if config.linear_augment:
                        combined_vector = combine_vectors_linear(vector1, vector2, config.linear_augment_lambda)
                    else:
                        if config.random_augment:
                            combined_vector = combine_vectors(vector1, vector2)
                        else:
                            combined_vector = combine_vectors_equal(vector1, vector2)

                    if nroi is None:
                        nroi = find_matrix_shape_from_upper_triangular_numpy(vector1)
                        print(f"No of nodes is : {nroi}")

                    combined_matrix = vector_to_matrix(combined_vector, nroi)
                    track_subjects_IDs.append((subject1, subject2))
                    vectors_number_of_fibers_all.append(combined_vector)
                    y_label_all.append(j * 2 + k)

                    augmented_matrices.append(combined_matrix)

                augmented_matrices = np.stack(augmented_matrices)
                np.save(os.path.join(path_subjects, str(bval) + "_" + res + "_augmented_matrices.npy"),
                        augmented_matrices)

    print(f"No of subjects in {set1} for {set2} is {len(track_subjects_IDs)}")
    print(f"No of unique subjects in {set1} for {set2} is {len(set(track_subjects_IDs))}")

    y_label_all = np.stack(y_label_all)
    vectors_number_of_fibers_all = np.stack(vectors_number_of_fibers_all)

    if set2=="eval":
        subjects_IDs_all = np.stack(track_subjects_IDs)
        return y_label_all, vectors_number_of_fibers_all, subjects_IDs_all

    return y_label_all, vectors_number_of_fibers_all

if __name__ == "__main__":
    main()
