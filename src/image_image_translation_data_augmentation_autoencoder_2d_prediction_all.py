# import the packages

import os
import glob
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix

from utils.data import load_graphs_dict, graph_matrix_vector
from utils.autoencoder_model_2d import ResNetAutoEncoder2D
from utils.functions_basic import entropy_of_confusion_matrix ,set_seed

data_path = "/data/hagmann_group/jagruti/dataset_1065/HCP_DWI"
file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/dataset_creation"
output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"
MAINDIR = "AUTOENCODER2D"
VAL_SET = "val"

def run_prediction(config, device):

    """
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
    """

    model_directory_parameters = ["normalization_l_domain", "activation_l_domain", "alpha_param", "dropout",
                                  "dropout_rate",
                                  "conditional_vector_size", "num_layers_mapping_network_latent",
                                  "scaler_input", "scaler_output", "symmetric",
                                  "correlation", "corr_param", "siamese", "sia_param", "nodStr", "nodStr_param",
                                  "eigenValue", "eigenValue_param",
                                  "weight", "weight_param0", "weight_param1", "weight_param2", "weight_param3",
                                  "lr", "domClass_lr", "mapNet_lr", "lr_patience", "lr_factor", "lr_threshold"]

    model_directory_arguments = ["dom", "",
                                 "a", "dp", "",
                                 "map", "",
                                 "scIO", "", "sym",
                                 "cor", "", "sia", "", "nod", "", "eig", "",
                                 "w", "", "", "", "",
                                 "lr", "", "", "p", "f", "t"]

    # Initialize an empty list to store the model directory name parts
    model_directory_name_parts = []

    # Process each argument and replace values accordingly
    for i, param in enumerate(model_directory_parameters):

        value = getattr(config, param)  # Retrieve the argument dynamically

        # Check the type and apply transformation
        if isinstance(value, str):
            if len(value) > 2:
                value = value[:2]
            else:
                value = value
        elif isinstance(value, float):  # If it's a float, replace '.' with '_'
            value = str(value)
            if '.' in value or 'e-' in value:
                # Replace '.' with '_' and 'e-' with 'e_'
                value = value.replace('.', '_').replace('e-', 'e_')
        elif value is True:  # If it's True, replace with 'T'
            value = "T"
        elif value is False:  # If it's False, replace with 'F'
            value = "F"
        elif isinstance(value, int):
            value = str(value)

        model_directory_name_parts.append(f"{model_directory_arguments[i]}_{value}")

    model_directory_name = "_".join(model_directory_name_parts)
    model_directory_name = model_directory_name.replace("__", "_")

    if config.linkPred:
        model_directory_name = model_directory_name + "_lP_T_" + (str(config.linkPred_param).replace(".","_"))
        if config.linkPred_weighted:
            model_directory_name = model_directory_name + "_" + (str(config.linkPred_weight).replace(".","_"))

    #print(f"Length of model directory name {len(model_directory_name)}")

    #print(f"Model Directory Name : {model_directory_name}")

    # Setup seeds
    seed = config.seed
    set_seed(seed)

    if config.mixed:
        directory_name1 = "MIXED"
    else:
        directory_name1 = "ALLTRAVELLING"

    if config.augment:
        directory_name2 = "AUGMENT_" + str(config.num_pairs)
        if config.symmetric_augment:
            directory_name2 = os.path.join(directory_name2, "SYMMETRIC")
        if config.linear_augment:
            directory_name2 = os.path.join(directory_name2,
                                           "LINEAR_" + (str(config.linear_augment_lambda).replace(".", "_")))
        else:
            if config.random_augment:
                directory_name2 = os.path.join(directory_name2, "RANDOM")
            else:
                directory_name2 = os.path.join(directory_name2, "EQUAL")
    else:
        directory_name2 = "NO_AUGMENT"

    if config.alpha_change_100:
        directory_name3 = "ALPHA_CHANGE_100"
    else:
        directory_name3 = "NO_ALPHA_CHANGE_100"

    if config.weighted_ce:
        directory_name4 = "WEIGHTED_CE"
    else:
        directory_name4 = "NO_WEIGHTED_CE"

    if config.scaler_output != 0:
        if config.scaler_output == 1:
            directory_name5 = "SCALER_" + (str(config.scaler_input_param).replace(".", "_")) + "_" \
                              + (str(config.scaler_output_param).replace(".", "_"))
            save_path = os.path.join(output_path, "models", MAINDIR, directory_name1,
                                     directory_name2, directory_name3, directory_name4,
                                     f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                     f"CONSTRAINT_{config.constraint}", directory_name5,
                                     model_directory_name)
        else:
            save_path = os.path.join(output_path, "models", MAINDIR, directory_name1,
                                     directory_name2, directory_name3, directory_name4,
                                     f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                     f"CONSTRAINT_{config.constraint}", model_directory_name)
    else:
        save_path = os.path.join(output_path, "models", MAINDIR, directory_name1,
                                 directory_name2, directory_name3, directory_name4,
                                 f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                 model_directory_name)

    print(save_path)

    # Use glob to list all files that match the pattern
    file_list_paths = glob.glob(os.path.join(save_path, "train_model_epoch_*.pth"))
    file_list = [os.path.basename(file) for file in file_list_paths if os.path.isfile(file)]
    if len(file_list) > 0:
        model_names = [(name_x.split('_')[3].split('.')[0]) for name_x in file_list]
        print(model_names)
    else:
        print("No matching files found.")
        return False, save_path

    val_dataset = create_dataset(VAL_SET, save_path, config)

    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=8)

    # Number of features and outputs
    n_features, n_outputs = val_dataset.input_tensors[0].shape[0], config.output

    #print("Number of samples in the train dataset: ", len(train_dataset))
    #print("Number of samples in the val dataset: ", len(val_dataset))
    #print("Number of features per node: ", n_features)
    #print("Number of outputs per graph: ", n_outputs)

    if config.dropout:
        dp_rate = config.dropout_rate
    else:
        dp_rate = 0

    ### DEFINE THE MODEL
    model = ResNetAutoEncoder2D(in_channels=n_features, n_fc=config.num_layers_mapping_network_latent,
                                dim_latent=config.conditional_vector_size, out_channels=n_features, num_sites=n_outputs,
                                model_domain=config.model_domain, normalization_l_domain=config.normalization_l_domain,
                                activation_l_domain=config.activation_l_domain, p=dp_rate).to(device)
    #print(model)
    results = {}

    for model_x in model_names:
        # loading a saved model
        model_number = "train_model_epoch_{}.pth".format(model_x)
        model_path = os.path.join(save_path, model_number)
        #print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint["epoch"] + 1
        #print("epoch_loaded", epoch)

        mae, accuracy, entropy, mae_all, accuracy_all_site = predict(model, device, val_dataloader, config)
        if accuracy_all_site[0][0] > 0.9 and accuracy_all_site[0][1] > 0.9:
            results[epoch] = {}
            results[epoch]['accuracy_site'] = accuracy
            results[epoch]['entropy_site'] = entropy
            results[epoch]['mae'] = mae_all
            results[epoch]['accuracy'] = accuracy_all_site

    with open(os.path.join(save_path, "results_all.pkl"), "wb") as f:
        pickle.dump(results, f)

    results_dictionary = pd.DataFrame(columns=['accuracy_x_0', 'accuracy_y_0', 'accuracy_x_1', 'accuracy_y_1',
                                               'accuracy_x_2', 'accuracy_y_2', 'accuracy_x_3', 'accuracy_y_3',
                                               'mae_0', 'mae_1', 'mae_2', 'mae_3', 'entropy_site', 'accuracy_site'])
    for key_1, value_1 in results.items():
        for key_2, value_2 in value_1.items():
            if key_2 == "accuracy":
                for key_3, value_3 in value_2.items():
                    results_dictionary.loc[key_1, f"{key_2}_x_{key_3}"] = value_3[0]
                    results_dictionary.loc[key_1, f"{key_2}_y_{key_3}"] = value_3[1]
            elif key_2 == "mae":
                for key_3, value_3 in value_2.items():
                    results_dictionary.loc[key_1, f"{key_2}_{key_3}"] = value_3
            else:
                results_dictionary.loc[key_1, f"{key_2}"] = value_2

    for col_x in results_dictionary.columns:
        results_dictionary[col_x] = pd.to_numeric(results_dictionary[col_x])

    results_dictionary_sorted_acc = results_dictionary.sort_values(by=['accuracy_x_0', 'accuracy_x_1', 'mae_0'],
                                                                   ascending=[False, False, True])
    results_dictionary_sorted_mae = results_dictionary.sort_values(by=['mae_0', 'accuracy_x_0', 'accuracy_x_1'],
                                                                   ascending=[True, False, False])

    columns_to_print = ['accuracy_x_0', 'accuracy_x_1', 'mae_0', 'mae_1', 'mae_2', 'mae_3', 'accuracy_site']

    print("Top ten performing epochs as per accuracy of fingerprinting when translating from site 0 to site 3")
    print(results_dictionary_sorted_acc[:10][columns_to_print])

    print(
        "Top ten performing epochs as per mean absolute error between predicted and target when translating from site 0 to site 3")
    print(results_dictionary_sorted_mae[:10][columns_to_print])

    return True, save_path

def create_dataset(set1, path, config):

    """load the graphs, the matrices and the sites for the subject set"""

    bvalues = [1000, 3000]
    resolutions_paths = ["2_3", "1_25"]

    num_sites = len(bvalues) * len(resolutions_paths)
    conditional_variable_dict = {}  # input code for the corresponding site
    num_vectors_per_site = int(config.conditional_vector_size / num_sites)
    #print(f"Number of vectors per site : {num_vectors_per_site}")
    for site_x in range(num_sites):
        z = torch.zeros([1, config.conditional_vector_size], dtype=torch.float)
        z[:, num_vectors_per_site * site_x:num_vectors_per_site * (site_x + 1)] = 1
        conditional_variable_dict[site_x] = z

    #print(conditional_variable_dict)

    input_tensors = []
    site_outputs = []
    target_tensors = []
    conditional_tensors = []

    # load list of subjects
    subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])

    # loading the .gpickle files for the chosen subjects for output SCs
    graphs_dict_output, subject_IDs_valid_output = load_graphs_dict(subject_IDs,
                                                                    os.path.join(data_path, "ds_HCP_bval_3000_anat_0_7_dwi_1_25"),
                                                                    config.scale, 0)

    track_subjects_IDs = []

    for j, bval in enumerate(bvalues):
        for k, res in enumerate(resolutions_paths):

            # loading the .gpickle files for the chosen subjects for input SCs
            graphs_dict_input, subject_IDs_valid_input = load_graphs_dict(subject_IDs, os.path.join(data_path,"ds_HCP_bval_"+str(bval)+"_anat_0_7_dwi_"+res),config.scale,0)

            subject_IDs_valid_common = np.intersect1d(np.array(subject_IDs_valid_input),
                                                      np.array(subject_IDs_valid_output))

            #print(f"Length of common valid subject IDs : {len(subject_IDs_valid_common)}")

            # getting the matrices for edge attribute "normalized_fiber_density"
            matrices_number_of_fibers_input, vectors_number_of_fibers_input = graph_matrix_vector(
                subject_IDs_valid_common, graphs_dict_input,
                weight_value='number_of_fibers')
            matrices_number_of_fibers_input1 = matrices_number_of_fibers_input.copy()
            matrices_number_of_fibers_output, _ = graph_matrix_vector(subject_IDs_valid_common, graphs_dict_output,
                                                                      weight_value='number_of_fibers')

            for i, subject in enumerate(subject_IDs_valid_common):
                x = torch.tensor(matrices_number_of_fibers_input1[subject], dtype=torch.float)
                x = x.unsqueeze(0)
                y1 = torch.tensor(j * 2 + k, dtype=torch.long)
                y2 = torch.tensor(matrices_number_of_fibers_output[subject], dtype=torch.float)
                y2 = y2.unsqueeze(0)
                y3 = conditional_variable_dict[3]
                # if config.scaler_input is 0, means no transformation, if 1 then normalizing the whole weights by a given
                # maximum value and if 2 then log transforming it
                if config.scaler_input == 1:
                    x = x / config.scaler_input_param
                elif config.scaler_input == 2:
                    x = torch.log1p(x)
                input_tensors.append(x)
                site_outputs.append(y1)
                target_tensors.append(y2)
                conditional_tensors.append(y3)
                track_subjects_IDs.append(subject)

    print(f"No of subjects in {set1} is {len(track_subjects_IDs)}")
    print(f"No of unique subjects in {set1} is {len(set(track_subjects_IDs))}")

    image_dataset = ImageDataset(input_tensors, site_outputs, target_tensors, conditional_tensors)

    return image_dataset

def predict(model, device, dataloader, config):

    "model prediction"

    score_mae_average = []

    label_all_site = []
    output_all_site = []
    y_all = {}
    x_hat_all = {}
    accuracy_all_site = {}

    model.eval()

    with torch.inference_mode():

        for i, batch in enumerate(dataloader):

            x, y1, y2, y3 = batch
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            y3 = y3.squeeze(1)

            _, site_output, x_hat_2 = model(x, y3, 1.0)

            if config.scaler_output == 1 and config.constraint == True:
                x_hat_2 = torch.sigmoid(x_hat_2)
            elif config.scaler_output == 2 and config.constraint == True:
                x_hat_2 = 9.21 * torch.sigmoid(x_hat_2)

            x_hat_2 = x_hat_2.squeeze()

            nroi = x_hat_2.size()[-1]

            x_hat = 0.5 * (torch.permute(x_hat_2, (0, 2, 1)) + x_hat_2)
            x_hat = x_hat * (1 - torch.eye(nroi, nroi).repeat(len(x), 1, 1)).to(device)  # TODO - check this logic again

            if config.scaler_output == 1:
                x_hat = x_hat * config.scaler_output_param
            elif config.scaler_output == 2:
                x_hat = torch.special.expm1(x_hat)

            x_hat = torch.round(torch.where(x_hat < 0, 0, x_hat))
            x_hat = x_hat.detach().cpu().numpy()

            site_output1 = torch.argmax(site_output, 1)
            site_output1 = site_output1.detach().cpu().numpy()

            y2_reshape = y2.squeeze()
            y2_reshape = y2_reshape.cpu().numpy()

            y1 = y1.cpu().numpy()

            label_all_site.extend(y1)
            output_all_site.extend(site_output1)

            score_mae = np.mean(np.abs(y2_reshape - x_hat), axis=(1, 2))
            score_mae_average.extend(score_mae)

            for j in np.unique(y1):
                if j not in y_all.keys():
                    y_all[j] = []
                if j not in x_hat_all.keys():
                    x_hat_all[j] = []
                y_all[j].extend(y2_reshape[y1 == j])
                x_hat_all[j].extend(x_hat[y1 == j])

    mae_all = {}

    for i in np.unique(label_all_site):
        #print("Site number", i)
        difference_matrix = np.zeros((len(y_all[i]), len(y_all[i])))
        for j in range(len(y_all[i])):
            for k in range(len(y_all[i])):
                # diff_y = np.mean(np.abs(y_all[i][j] - y_all[i][k]))
                # diff_x_hat = np.mean(np.abs(x_hat_all[i][j] - x_hat_all[i][k]))
                difference_matrix[j, k] = np.mean(np.abs(y_all[i][j] - x_hat_all[i][k]))
        mae_all[i] = np.mean(np.abs(np.array(y_all[i]) - np.array(x_hat_all[i])))
        #print("MAE", mae_all[i])
        true_values = np.arange(0, difference_matrix.shape[0], 1)
        predicted_values_axis_0 = np.argmin(difference_matrix, axis=0)
        predicted_values_axis_1 = np.argmin(difference_matrix, axis=1)
        #print(predicted_values_axis_0)
        #print(predicted_values_axis_1)
        accuracy_axis_0 = np.sum(true_values == predicted_values_axis_0) / len(true_values)
        accuracy_axis_1 = np.sum(true_values == predicted_values_axis_1) / len(true_values)
        #print(accuracy_axis_0, accuracy_axis_1)
        accuracy_all_site[i] = np.array([accuracy_axis_0, accuracy_axis_1])

    score_mae_average = np.mean(np.array(score_mae_average))  # check if this logic is correct or not
    label_all_site = np.stack(label_all_site)
    output_all_site = np.stack(output_all_site)

    score_accuracy = accuracy_score(label_all_site, output_all_site)
    confusion_matrix_x = confusion_matrix(label_all_site, output_all_site)
    entropy = entropy_of_confusion_matrix(confusion_matrix_x)

    return score_mae_average, score_accuracy, entropy, mae_all, accuracy_all_site

class ImageDataset(Dataset):
    def __init__(self, input_tensors, site_outputs, target_tensors, conditional_tensors):
        """
        Args:
            input_tensors (list): List of input tensors
            site_outputs (list): List of sites
            target_tensors (list): List of target tensors
            conditional_tensors (list): List of conditional tensors
        """
        self.input_tensors = input_tensors
        self.site_outputs = site_outputs
        self.target_tensors = target_tensors
        self.conditional_tensors = conditional_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        # Load input, target and conditional tensors and the site value
        input_tensor = self.input_tensors[idx]
        site_output = self.site_outputs[idx]
        target_tensor = self.target_tensors[idx]
        conditional_tensor = self.conditional_tensors[idx]

        return input_tensor, site_output, target_tensor, conditional_tensor