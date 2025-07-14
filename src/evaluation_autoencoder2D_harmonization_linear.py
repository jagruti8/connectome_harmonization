# import the packages

print("evaluation_autoencoder2D_harmonization_linear.py starting")

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

from utils.data import read_file_numpy, matrix_to_vector_one
from utils.functions_basic import set_seed
from create_model_path_autoencoder2D import create_model_path_function

print("All packages imported")

data_behaviour_path = "/data/hagmann_group/behaviour_prediction/data_Urblauna/HCP_behavioral"
output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"
parameter_file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/batch_scripts/parameters"
MAINDIR = "AUTOENCODER2D"
CKDIR="checkpoints1"
evalSet="test"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--epochs", default=500, type=int, help="Number of epochs")

    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size")

    parser.add_argument(
        "--maindir", default="checkpoints1", type=str, help="Main directory")

    parser.add_argument(
        "--ckdir", default="checkpoints2", type=str, help="Checkpoint directory")

    parser.add_argument(
        "--seed", default=42, type=int, help="Random Seed Generator")

    parser.add_argument(
        "--parameter_combination_number", default=1, type=int, help="parameters_combinations file number")

    parser.add_argument(
        "--line_number", default=1, type=int, help="Line number of the parameters_combinations file")

    parser.add_argument(
        "--model_epoch_chosen", default=999, type=int, help="model epoch from which to take the latent embedding")

    parser.add_argument(
        "--graph_metric", default="closeness_centrality", type=str, help="name of the graph metrics to be calculated")

    parser.add_argument(
        "--bvalue", default="3000", type=str, help="b-value of diffusion imaging MRI")

    parser.add_argument(
        "--resolution", default="1_25", type=str, help="spatial resolution of diffusion imaging MRI")

    parser.add_argument(
        "--train_set", default="train", type=str, help="Name of training set")

    parser.add_argument(
        "--val_set", default="val", type=str, help="Name of validation set")

    parser.add_argument(
        "--test_set", default="test", type=str, help="Name of test set")

    parser.add_argument(
        "--evaluation_set", default="test", type=str, help="Name of evaluation set")

    parser.add_argument(
        "--part_number", default=0, type=int,
        help="choose for which part of the augmented matrices, the graph metric needs to be computed")

    parser.add_argument(
        "--len_part", default=1000, type=int,
        help="Number of matrices in each part")

    parser.add_argument(
        "--best", default="", const="", nargs='?', type=str, help="Model Selection")

    parser.add_argument(
        "--task", default="age", type=str, help="Task to be performed")

    parser.add_argument(
        "--feature_latent", default="graph", type=str,
        help="latent space is from graph neural networks or 2D convolutional neural networks")

    parser.add_argument(
        "--model_task", default=1, type=int,
        help="choose which type of mlp model to use for the specific task")

    parser.add_argument(
        "--base_layer", default="cheb", type=str, help="Default base graph convolutional layer")

    parser.add_argument(
        "--hidden_layer_size", default=256, type=int,
        help="Number of features per node in the hidden layer next to input layer")

    parser.add_argument(
        "--normalization_g", default="batch", type=str,
        help="Normalization layer for graph convolutional layers")

    parser.add_argument(
        "--normalization_l", default="batch", type=str,
        help="Normalization layer for fully connected part")

    parser.add_argument(
        "--activation_g", default="lrelu", type=str,
        help="Activation layer for graph convolutional layers")

    parser.add_argument(
        "--activation_l", default="lrelu", type=str,
        help="Activation layer for fully connected part")

    parser.add_argument(
        "--instance_track", default=False, type=str2bool, nargs='?',
        help="To keep track of running mean and variance for model if it uses instance normalization")

    parser.add_argument(
        "--filter_size", default=2, type=int, help="Hyperparameter for setting the Chebyshev filter size")

    parser.add_argument(
        "--heads", default=2, type=int,
        help="Number of multi-head-attentions for GATv2Conv graph convolutional layer")

    parser.add_argument(
        "--concat_gat", default=False, type=str2bool, nargs='?',
        help="To concat the multi-head outputs or not for GATv2Conv graph convolutional layer")

    parser.add_argument(
        "--dropout_gat", default=0.2, type=float,
        help="Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training for GATv2Conv graph convolutional layer")

    parser.add_argument(
        "--dropout", default=True, type=str2bool, nargs='?', help="To include dropout or not")

    # parser.add_argument(
    #    "--dropout_rate", default=0.1, type=float,
    #    help="Dropout rate if dropout is True")   #TODO, also as a hyperparameter

    parser.add_argument(
        "--scale", default=3, type=int, help="Scale of parcellation used")

    parser.add_argument(
        "--scaler_input", default=0, type=int, help="To scale or not the input weights and which scaler to use")

    parser.add_argument(
        "--scaler_input_param", default=10000.0, type=float, help="Maximum weight for min-max scaler for input weights")

    parser.add_argument(
        "--self_loop_add_input", default=True, type=str2bool, nargs='?', help="Add self-loops to the input graphs")

    parser.add_argument(
        "--lr", default=0.01, type=float,
        help="Initial learning rate of the Adam optimizer in general")  # TODO - keep lr in the main program

    parser.add_argument(
        "--lr_patience", default=5, type=int,
        help="Patience for the learning rate scheduler")  # TODO - Keep this as a hyperparameter

    parser.add_argument(
        "--lr_factor", default=0.91, type=float, help="Factor for the learning rate scheduler")

    parser.add_argument(
        "--lr_threshold", default=0.00001, type=float,
        help="Threshold for the learning rate scheduler")  # TODO - Keep this as a hyperparameter

    # parameter_combination_file_numbers = np.arange(11,14)
    parameter_combination_file_numbers = np.array([1])
    parameter_combination_file_names = [f"parameters_combinations_prediction_new_{i}.txt" for i in parameter_combination_file_numbers]

    parameter_combinations = 0
    model_all_present = 0
    model_not_present = 0
    for file_name_x in parameter_combination_file_names:
        with open(os.path.join(parameter_file_path, MAINDIR, file_name_x), "r") as f:
            args_from_file = f.readlines()  # Read lines
            args_params = [x.strip().split() for x in args_from_file]  # Read and split arguments
        for j, args_params_x in enumerate(args_params):
            print(file_name_x)
            print(j + 1)
            config = parser.parse_args(args_params_x)
            present_flag = run_prediction(config)
            if present_flag:
                model_all_present += 1
            else:
                model_not_present += 1
            parameter_combinations += 1
    print(f"Total number of parameter combinations is : {parameter_combinations}")
    print(f"Total number of parameter combinations having models is : {model_all_present}")
    print(f"Models not present : {model_not_present}")

    print("Model prediction over")

def run_prediction(config):

    # Setup seeds
    seed = config.seed
    set_seed(seed)

    saved_models_path = os.path.join(output_path, "models", "EVALUATION", "LINEAR",
                                     f"SEED_{config.seed}", CKDIR)

    if not os.path.exists(saved_models_path):
        print(f"{saved_models_path} not present")
        return False

    parameter_file_combination_path = os.path.join(parameter_file_path, MAINDIR,
                                                   f"parameters_combinations_{config.parameter_combination_number}.txt")
    saved_matrices_path = create_model_path_function(output_path, parameter_file_combination_path,
                                                     config.line_number, MAINDIR)

    retrieve_path = os.path.join(saved_matrices_path, f"epoch_{config.model_epoch_chosen + 1}", evalSet)
    print(retrieve_path)

    if not os.path.exists(retrieve_path):
        print(f"{retrieve_path} not present")
        return False

    behavioral_restricted_dataset = pd.read_csv(os.path.join(data_behaviour_path, "hcp_behavioral_RESTRICTED.csv"),
                                                index_col=0)
    behavioral_dataset = pd.read_csv(os.path.join(data_behaviour_path, "hcp_behavioral.csv"), index_col=0)

    # adding the 'Gender' and 'Age' columns of the behavioral dataset and 'Age_in_Yrs' column of the restricted behavioral dataset
    behavioral_all_dataset = pd.concat(
        (behavioral_dataset[['Gender', 'Age']], behavioral_restricted_dataset['Age_in_Yrs']), axis=1)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the 'Gender' column
    behavioral_all_dataset['Gender_Numeric'] = label_encoder.fit_transform(behavioral_all_dataset['Gender'])

    vectors_numbers_of_fibers = {}
    y_label_site = {}
    y_label_age = {}
    y_label_gender = {}
    set_x = {}
    sets = [evalSet]

    for set1 in sets:
        vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_age[set1], y_label_gender[set1], \
        set_x[set1]= create_dataset(set1, retrieve_path, behavioral_all_dataset)

        print(f"Number of samples in the {set1} dataset : {len(vectors_numbers_of_fibers[set1])}")

    models = {}
    tasks = ['age', 'gender']
    models['age'] = joblib.load(os.path.join(saved_models_path, "model_age_Regression.pkl"))
    #print("SVM model for age prediction loaded - 'model_age_Regression.pkl'")
    models['gender'] = joblib.load(os.path.join(saved_models_path, "model_gender_SVM.pkl"))
    #print("SVM model for gender prediction loaded - 'model_gender_SVM.pkl'")

    for task in tasks:
        print(task)
        for set1 in sets:
            print(set1)
            if task == "age":
                predict_age(models[task], vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_age[set1], set_x[set1])
            elif task == "gender":
                predict_gender(models[task], vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_gender[set1], set_x[set1])

    return True

def create_dataset(set1, retrieve_path, behavioral_all_dataset):

    input_matrices = read_file_numpy(retrieve_path, f"{set1}_input_images.npy")
    predicted_matrices = read_file_numpy(retrieve_path, f"{set1}_predicted_images.npy")
    target_matrices = read_file_numpy(retrieve_path, f"{set1}_target_images.npy")
    label_site = read_file_numpy(retrieve_path, f"{set1}_label_site.npy")
    subject_IDs = read_file_numpy(retrieve_path, f"{set1}_subject_IDs.npy")

    track_subjects_IDs = []
    y_label_site_all = []
    y_label_age_all = []
    y_label_gender_all = []
    vectors_number_of_fibers_all = []
    set_x_all = []

    for i, subject in enumerate(subject_IDs):
        track_subjects_IDs.append(subject)
        vectors_number_of_fibers_all.append(matrix_to_vector_one(input_matrices[i]))
        y_label_age = behavioral_all_dataset.loc[subject]['Age_in_Yrs']
        y_label_gender = behavioral_all_dataset.loc[subject]['Gender_Numeric']
        y_label_site_all.append(label_site[i])
        y_label_age_all.append(y_label_age)
        y_label_gender_all.append(y_label_gender)
        set_x_all.append(0)

    for i, subject in enumerate(subject_IDs):
        track_subjects_IDs.append(subject)
        if label_site[i] ==  3:
            vectors_number_of_fibers_all.append(matrix_to_vector_one(target_matrices[i]))
        else:
            vectors_number_of_fibers_all.append(matrix_to_vector_one(predicted_matrices[i]))
        y_label_age = behavioral_all_dataset.loc[subject]['Age_in_Yrs']
        y_label_gender = behavioral_all_dataset.loc[subject]['Gender_Numeric']
        y_label_site_all.append(label_site[i])
        y_label_age_all.append(y_label_age)
        y_label_gender_all.append(y_label_gender)
        set_x_all.append(1)

    print(f"No of subjects in {set1} is {len(track_subjects_IDs)}")
    print(f"No of unique subjects in {set1} is {len(set(track_subjects_IDs))}")

    vectors_number_of_fibers_all = np.stack(vectors_number_of_fibers_all)
    y_label_site_all = np.stack(y_label_site_all)
    y_label_age_all = np.stack(y_label_age_all)
    y_label_gender_all = np.stack(y_label_gender_all)
    set_x_all = np.stack(set_x_all)

    return vectors_number_of_fibers_all, y_label_site_all, y_label_age_all, y_label_gender_all, set_x_all

def predict_age(model, vectors_number_of_fibers, y_label_site, y_label_age, set_x):
    X_test = vectors_number_of_fibers
    y_test = y_label_age
    y_pred = model.predict(X_test)
    y_pred_integer = y_pred.copy()
    y_pred_integer = np.round(y_pred_integer)
    print(f"Lengths : {len(y_test)}, {len(y_pred)}")
    # Evaluate the model
    for i in np.unique(set_x):
        if i == 0:
            print("Unharmonized set")
        else:
            print("Harmonized set")
        # Evaluate the model
        y_test_i = y_test[set_x==i]
        y_pred_i = y_pred[set_x==i]
        y_pred_integer_i = y_pred_integer[set_x==i]
        y_label_site_i = y_label_site[set_x==i]
        validation_loss = np.mean(np.abs(y_test_i - y_pred_i))
        validation_loss_integer = np.mean(np.abs(y_test_i - y_pred_integer_i))
        cod_score = r2_score(y_test_i, y_pred_i)
        cod_score_integer = r2_score(y_test_i, y_pred_integer_i)
        correlation = pearsonr(y_test_i, y_pred_i)
        correlation_integer = pearsonr(y_test_i, y_pred_integer_i)
        print(f"MAE is {validation_loss}, r2 score is {cod_score} and pearson correlation is {correlation}")
        print(f"For predicted values rounded to nearest integer")
        print(f"MAE is {validation_loss_integer}, r2 score is {cod_score_integer} and "
              f"pearson correlation is {correlation_integer}")
        for j in np.unique(y_label_site_i):
            print(f"Site : {j}")
            y_test_i_j = y_test_i[y_label_site_i == j]
            y_pred_i_j = y_pred_i[y_label_site_i == j]
            y_pred_integer_i_j = y_pred_integer_i[y_label_site_i == j]
            validation_loss = np.mean(np.abs(y_test_i_j - y_pred_i_j))
            validation_loss_integer = np.mean(np.abs(y_test_i_j - y_pred_integer_i_j))
            cod_score = r2_score(y_test_i_j, y_pred_i_j)
            cod_score_integer = r2_score(y_test_i_j, y_pred_integer_i_j)
            correlation = pearsonr(y_test_i_j, y_pred_i_j)
            correlation_integer = pearsonr(y_test_i_j, y_pred_integer_i_j)
            print(f"MAE is {validation_loss}, r2 score is {cod_score} and pearson correlation is {correlation}")
            print(f"For predicted values rounded to nearest integer")
            print(f"MAE is {validation_loss_integer}, r2 score is {cod_score_integer} and "
                  f"pearson correlation is {correlation_integer}")

def predict_gender(model, vectors_number_of_fibers, y_label_site, y_label_gender, set_x):
    X_test = vectors_number_of_fibers
    y_test = y_label_gender
    y_pred = model.predict(X_test)
    print(f"Lengths : {len(y_test)}, {len(y_pred)}")
    # Evaluate the model
    for i in np.unique(set_x):
        if i == 0:
            print("Unharmonized set")
        else:
            print("Harmonized set")
        y_test_i = y_test[set_x == i]
        y_pred_i = y_pred[set_x == i]
        y_label_site_i = y_label_site[set_x == i]
        accuracy = accuracy_score(y_test_i, y_pred_i)
        print(f"Accuracy: {accuracy * 100}")
        #print("Classification Report:\n", classification_report(y_test_i, y_pred_i))
        for j in np.unique(y_label_site_i):
            print(f"Site : {j}")
            y_test_i_j = y_test_i[y_label_site_i == j]
            y_pred_i_j = y_pred_i[y_label_site_i == j]
            # Evaluate the model
            accuracy = accuracy_score(y_test_i_j, y_pred_i_j)
            print(f"Accuracy: {accuracy * 100}")
            #print("Classification Report:\n", classification_report(y_test_i_j, y_pred_i_j))

if __name__ == "__main__":
    main()


