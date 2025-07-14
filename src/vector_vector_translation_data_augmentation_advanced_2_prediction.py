# import the packages

print("vector_vector_translation_data_augmentation_advanced_2_prediction.py starting")

import os
import glob
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix

from utils.data import load_graphs_dict, graph_matrix_vector, vector_to_matrix_tensor, create_directory
from utils.autoencoder_model_1d_advanced import VectorToVectorTranslationAE
from utils.config_advanced import read_configuration_param
from utils.functions_basic import entropy_of_confusion_matrix ,set_seed, \
find_matrix_shape_from_upper_triangular

print("All packages imported")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print("PATH:", os.environ.get('PATH'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
if torch.cuda.is_available():
    print("Available GPUs:", torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

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

    model_directory_parameters = ["hidden_layer_size", "bottleneck_size", "normalization_l_encoder",
                                  "activation_l_encoder", "dropout_encoder", "dropout_rate_encoder",
                                  "normalization_l_domain", "activation_l_domain", "alpha_param", "dropout",
                                  "dropout_rate",
                                  "conditional_vector_size", "latent_vector_size", "hidden_layer_size_mapping_network",
                                  "num_layers_mapping_network_latent", "activation_l_mapping",
                                  "normalization_l1_decoder", "activation_l1_decoder", "dropout_decoder1",
                                  "dropout_rate_decoder1",
                                  "scaler_input", "scaler_output",
                                  "correlation", "corr_param", "siamese", "sia_param", "nodStr", "nodStr_param",
                                  "eigenValue", "eigenValue_param",
                                  "weight", "weight_param0", "weight_param1", "weight_param2", "weight_param3",
                                  "lr", "domClass_lr", "mapNet_lr", "lr_patience", "lr_factor", "lr_threshold"]

    model_directory_arguments = ["enc", "", "", "", "", "",
                                 "dom", "",
                                 "a", "dp", "",
                                 "map", "", "", "", "",
                                 "dec", "", "", "",
                                 "scIO", "",
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

    print(f"Length of model directory name {len(model_directory_name)}")

    print(f"Model Directory Name : {model_directory_name}")

    # Setup seeds
    seed = config.seed
    set_seed(seed)

    ### DEVICE GPU OR CPU : will select GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice: ", device)

    if config.mixed:
        directory_name1 = "MIXED"
    else:
        directory_name1 = "ALLTRAVELLING"

    if config.augment:
        directory_name2 = "AUGMENT_" + str(config.num_pairs)
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
            save_path = os.path.join(output_path, "models", config.maindir, directory_name1,
                                     directory_name2, directory_name3, directory_name4,
                                     f"ENCODER_{config.model_encoder}", f"DOMAIN_{config.model_domain}",
                                     f"DECODER_{config.model_decoder}", f"BATCHSIZE_{config.batch_size}",
                                     f"SEED_{config.seed}", f"CONSTRAINT_{config.constraint}",
                                     directory_name5, model_directory_name)
        else:
            save_path = os.path.join(output_path, "models", config.maindir, directory_name1,
                                     directory_name2, directory_name3, directory_name4,
                                     f"ENCODER_{config.model_encoder}", f"DOMAIN_{config.model_domain}",
                                     f"DECODER_{config.model_decoder}",
                                     f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                     f"CONSTRAINT_{config.constraint}", model_directory_name)
    else:
        save_path = os.path.join(output_path, "models", config.maindir, directory_name1,
                                 directory_name2, directory_name3, directory_name4,
                                 f"ENCODER_{config.model_encoder}", f"DOMAIN_{config.model_domain}",
                                 f"DECODER_{config.model_decoder}", f"BATCHSIZE_{config.batch_size}",
                                 f"SEED_{config.seed}", model_directory_name)

    # Use glob to list all files that match the pattern
    file_list_paths = glob.glob(os.path.join(save_path, "train_model_epoch_*.pth"))
    file_list = [os.path.basename(file) for file in file_list_paths if os.path.isfile(file)]
    if len(file_list) > 0:
        model_names = [(name_x.split('_')[3].split('.')[0]) for name_x in file_list]
        numbers = np.array([int(name_x) for name_x in model_names if name_x.isdigit()])
        if len(numbers) > 0:
            print(numbers)
            max_number = numbers.max()
            print(f"Largest epoch number: {max_number}")
        else:
            print("Only latest epoch model found")
    else:
        print("No matching files found.")
        return

    #train_dataset = create_dataset(config.train_set, save_path, config)
    val_dataset = create_dataset(config.val_set, save_path, config)
    test_dataset = create_dataset(config.test_set, save_path, config)

    #train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=8)

    for i, d in enumerate(test_dataloader):
        # print(d)
        print(f"Length of test batch device : {len(d[0])}")

    # Input and output size
    n_inputs, n_outputs = test_dataset.input_tensors[0].shape[0], config.output

    #print("Number of samples in the train dataset: ", len(train_dataset))
    print("Number of samples in the val dataset: ", len(val_dataset))
    print("Number of samples in the test dataset: ", len(test_dataset))
    print("Output of one sample from the test dataset: ", test_dataset[0])
    print("Input size: ", n_inputs)
    print("Output size of the domain classifier: ", n_outputs)

    if config.dropout_encoder:
        dp_rate_encoder = config.dropout_rate_encoder
    else:
        dp_rate_encoder = 0

    if config.dropout:
        dp_rate = config.dropout_rate
    else:
        dp_rate = 0

    if config.dropout_decoder1:
        dp_rate_decoder1 = config.dropout_rate_decoder1
    else:
        dp_rate_decoder1 = 0

    ### DEFINE THE MODEL
    model = VectorToVectorTranslationAE(input_size=n_inputs, hidden_size=config.hidden_layer_size,
                                        bottleneck_size=config.bottleneck_size,
                                        mapping_input_size=config.conditional_vector_size,
                                        mapping_hidden_size=config.hidden_layer_size_mapping_network,
                                        mapping_latent_size=config.latent_vector_size,
                                        num_layers_latent=config.num_layers_mapping_network_latent,
                                        output_size=n_outputs,
                                        model_encoder=config.model_encoder,
                                        normalization_l_encoder=config.normalization_l_encoder,
                                        activation_l_encoder=config.activation_l_encoder, p_encoder=dp_rate_encoder,
                                        model_domain=config.model_domain,
                                        normalization_l_domain=config.normalization_l_domain,
                                        activation_l_domain=config.activation_l_domain, p_domain=dp_rate,
                                        normalization_l_mapping=config.normalization_l_mapping,
                                        activation_l_mapping=config.activation_l_mapping,
                                        model_decoder=config.model_decoder,
                                        normalization_l1_decoder=config.normalization_l1_decoder,
                                        activation_l1_decoder=config.activation_l1_decoder,
                                        p_decoder1=dp_rate_decoder1).to(device)
    print(model)

    # loading a saved model
    model_number = "train_model_epoch_{}.pth".format(config.best)
    model_path = os.path.join(save_path, model_number)
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    epoch = checkpoint["epoch"] + 1
    print("epoch_loaded", epoch)

    ### predict from the model
    #train_subject_IDs, train_input_images, train_target_images, train_predicted_images, train_invariant_features, train_reversed_features, \
    #train_label_site, train_output_site, train_mae, train_accuracy, train_entropy, train_confusion_matrix = predict(model, device, train_dataloader, config, save_path)
    val_subject_IDs, val_input_images, val_target_images, val_predicted_images, val_invariant_features, val_reversed_features, \
    val_label_site, val_output_site, val_mae, val_accuracy, val_entropy, val_confusion_matrix = predict(model, device, val_dataloader, config, save_path)
    test_subject_IDs, test_input_images, test_target_images, test_predicted_images, test_invariant_features, test_reversed_features, \
    test_label_site, test_output_site, test_mae, test_accuracy, test_entropy, test_confusion_matrix = predict(model, device, test_dataloader, config, save_path)

    #print("Training Set:")
    #print("MAE : {}".format(train_mae))
    #print("Accuracy : {}".format(train_accuracy))
    #print("Entropy : {}".format(train_entropy))
    #print("Maximum edge : {}".format(train_predicted_images.max()))
    #print("Minimum edge : {}".format(train_predicted_images.min()))
    #print("Invariant features equals reversed features : {}".format(np.array_equal(train_invariant_features, train_reversed_features)))
    print("Validation Set:")
    print("MAE : {}".format(val_mae))
    print("Accuracy : {}".format(val_accuracy))
    print("Entropy : {}".format(val_entropy))
    print("Confusion Matrix : {}".format(val_confusion_matrix))
    print("Maximum edge : {}".format(val_predicted_images.max()))
    print("Minimum edge : {}".format(val_predicted_images.min()))
    print("Invariant features equals reversed features : {}".format(np.array_equal(val_invariant_features, val_reversed_features)))
    print("Test Set:")
    print("MAE : {}".format(test_mae))
    print("Accuracy : {}".format(test_accuracy))
    print("Entropy : {}".format(test_entropy))
    print("Confusion Matrix : {}".format(test_confusion_matrix))
    print("Maximum edge : {}".format(test_predicted_images.max()))
    print("Minimum edge : {}".format(test_predicted_images.min()))
    print("Invariant features equals reversed features : {}".format(np.array_equal(test_invariant_features, test_reversed_features)))

    save_results_path = os.path.join(save_path, "epoch_"+str(epoch))
    create_directory(save_results_path)
    print(f"Directory '{save_results_path}' created successfully (if it didn't exist).")

    #save_results_path_train = os.path.join(save_results_path, "train")
    #create_directory(save_results_path_train)
    #print(f"Directory '{save_results_path_train}' created successfully (if it didn't exist).")

    save_results_path_val = os.path.join(save_results_path, "val")
    create_directory(save_results_path_val)
    print(f"Directory '{save_results_path_val}' created successfully (if it didn't exist).")

    save_results_path_test = os.path.join(save_results_path, "test")
    create_directory(save_results_path_test)
    print(f"Directory '{save_results_path_test}' created successfully (if it didn't exist).")

    #np.save(os.path.join(save_results_path_train, "train_input_images.npy"), train_input_images)
    np.save(os.path.join(save_results_path_val, "val_input_images.npy"), val_input_images)
    np.save(os.path.join(save_results_path_test, "test_input_images.npy"), test_input_images)

    #np.save(os.path.join(save_results_path_train, "train_predicted_images.npy"), train_predicted_images)
    np.save(os.path.join(save_results_path_val,"val_predicted_images.npy"), val_predicted_images)
    np.save(os.path.join(save_results_path_test,"test_predicted_images.npy"), test_predicted_images)

    #np.save(os.path.join(save_results_path_train, "train_target_images.npy"), train_target_images)
    np.save(os.path.join(save_results_path_val,"val_target_images.npy"), val_target_images)
    np.save(os.path.join(save_results_path_test,"test_target_images.npy"), test_target_images)

    #np.save(os.path.join(save_results_path_train, "train_invariant_features.npy"), train_invariant_features)
    np.save(os.path.join(save_results_path_val, "val_invariant_features.npy"), val_invariant_features)
    np.save(os.path.join(save_results_path_test, "test_invariant_features.npy"), test_invariant_features)

    #np.save(os.path.join(save_results_path_train, "train_reversed_features.npy"), train_reversed_features)
    #np.save(os.path.join(save_results_path_val, "val_reversed_features.npy"), val_reversed_features)
    #np.save(os.path.join(save_results_path_test, "test_reversed_features.npy"), test_reversed_features)

    #np.save(os.path.join(save_results_path_train, "train_label_site.npy"), train_label_site)
    np.save(os.path.join(save_results_path_val, "val_label_site.npy"), val_label_site)
    np.save(os.path.join(save_results_path_test, "test_label_site.npy"), test_label_site)

    #np.save(os.path.join(save_results_path_train, "train_output_site.npy"), train_output_site)
    np.save(os.path.join(save_results_path_val, "val_output_site.npy"), val_output_site)
    np.save(os.path.join(save_results_path_test, "test_output_site.npy"), test_output_site)

    #np.save(os.path.join(save_results_path_train, "train_subject_IDs.npy"), train_subject_IDs)
    np.save(os.path.join(save_results_path_val, "val_subject_IDs.npy"), val_subject_IDs)
    np.save(os.path.join(save_results_path_test, "test_subject_IDs.npy"), test_subject_IDs)

    print("Model prediction over")

def create_dataset(set1, path, config):

    """load the vectors and the sites for the subject set"""

    bvalues = [1000, 3000]
    resolutions_paths = ["2_3", "1_25"]

    num_sites = len(bvalues) * len(resolutions_paths)
    conditional_variable_dict = {}  # input code for the corresponding site
    num_vectors_per_site = int(config.latent_vector_size / num_sites)
    print(f"Number of vectors per site : {num_vectors_per_site}")
    for site_x in range(num_sites):
        z = torch.zeros([1, config.latent_vector_size], dtype=torch.float)
        z[:, num_vectors_per_site * site_x:num_vectors_per_site * (site_x + 1)] = 1
        conditional_variable_dict[site_x] = z

    print(conditional_variable_dict)

    input_tensors = []
    input_original_tensors = []
    site_outputs = []
    target_tensors = []
    conditional_tensors = []
    subject_ID_tensors = []

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

            print(f"Length of common valid subject IDs : {len(subject_IDs_valid_common)}")

            # getting the vectors for edge attribute "normalized_fiber_density"
            matrices_number_of_fibers_input, vectors_number_of_fibers_input = graph_matrix_vector(subject_IDs_valid_common, graphs_dict_input,
                                                                    weight_value='number_of_fibers')
            vectors_number_of_fibers_input1 = vectors_number_of_fibers_input.copy()
            _, vectors_number_of_fibers_output = graph_matrix_vector(subject_IDs_valid_common, graphs_dict_output,
                                                                     weight_value='number_of_fibers')

            for i, subject in enumerate(subject_IDs_valid_common):
                x = torch.tensor(vectors_number_of_fibers_input1[subject], dtype=torch.float)
                x1 = torch.tensor(matrices_number_of_fibers_input[subject], dtype=torch.float)
                y1 = torch.tensor(j * 2 + k, dtype=torch.long)
                y2 = torch.tensor(vectors_number_of_fibers_output[subject], dtype=torch.float)
                y3 = conditional_variable_dict[3]
                # if config.scaler_input is 0, means no transformation, if 1 then normalizing the whole weights by a given
                # maximum value and if 2 then log transforming it
                if config.scaler_input == 1:
                    x = x / config.scaler_input_param
                elif config.scaler_input == 2:
                    x = torch.log1p(x)
                ID_x = torch.tensor(subject)

                input_tensors.append(x)
                input_original_tensors.append(x1)
                site_outputs.append(y1)
                target_tensors.append(y2)
                conditional_tensors.append(y3)
                subject_ID_tensors.append(ID_x)
                track_subjects_IDs.append(subject)

    print(f"No of subjects in {set1} is {len(track_subjects_IDs)}")
    print(f"No of unique subjects in {set1} is {len(set(track_subjects_IDs))}")

    input_tensors_copy = torch.stack(input_tensors)
    target_tensors_copy = torch.stack(target_tensors)
    if config.scaler_input == 3:
        mean_input = np.load(os.path.join(path, "mean_input.npy"))
        std_input = np.load(os.path.join(path, "std_input.npy"))
        mean_input = torch.from_numpy(mean_input)
        std_input = torch.from_numpy(std_input)
        input_tensors_copy = input_tensors_copy - mean_input
        input_tensors_copy = input_tensors_copy / std_input

    vector_dataset = VectorDataset(input_tensors_copy, input_original_tensors,
                                   site_outputs, target_tensors_copy,
                                   conditional_tensors, subject_ID_tensors)

    return vector_dataset

def predict(model, device, dataloader, config, path):

    "model prediction"

    score_mae_average = []
    input_images = []
    target_images = []
    predicted_images = []
    label_all_site = []
    output_all_site = []
    invariant_features_all = []
    reversed_features_all = []
    subject_IDs_all = []

    nroi = None

    model.eval()

    with torch.inference_mode():
        for i, batch in enumerate(dataloader):

            x, x1, y1, y2, y3, ID_x = batch
            x = x.to(device)
            x1 = x1.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)
            ID_x = ID_x.to(device)

            y3 = y3.squeeze(1)

            invariant_features, reversed_features, site_output, x_hat_2 = model(x, 1.0, y3)

            if config.scaler_output == 1 and config.constraint == True:
                x_hat_2 = torch.sigmoid(x_hat_2)
            elif config.scaler_output == 2 and config.constraint == True:
                x_hat_2 = 9.21 * torch.sigmoid(x_hat_2)

            if config.scaler_output == 3:
                mean_target = np.load(os.path.join(path, "mean_target.npy"))
                std_target = np.load(os.path.join(path, "std_target.npy"))
                mean_target = (torch.from_numpy(mean_target)).to(device)
                std_target = (torch.from_numpy(std_target)).to(device)
                x_hat_2 = x_hat_2 * std_target
                x_hat_2 = x_hat_2 + mean_target

            if nroi is None:
                nroi = find_matrix_shape_from_upper_triangular(y2[0])
                print(f"No of nodes is : {nroi}")

            x_hat = vector_to_matrix_tensor(x_hat_2, nroi, device)

            if config.scaler_output == 1:
                x_hat = x_hat * config.scaler_output_param
            elif config.scaler_output == 2:
                x_hat = torch.special.expm1(x_hat)

            x_hat = torch.round(torch.where(x_hat < 0, 0, x_hat))
            x_hat = x_hat.detach().cpu().numpy()

            x_reshape = x1
            x_reshape = x_reshape.cpu().numpy()

            site_output1 = torch.argmax(site_output, 1)
            site_output1 = site_output1.detach().cpu().numpy()

            y2_reshape = vector_to_matrix_tensor(y2, nroi, device)
            y2_reshape = y2_reshape.cpu().numpy()

            y1 = y1.cpu().numpy()

            if i == 0:
                print(f"Shape of invariant features before reshaping: {invariant_features.size()}")
            invariant_features = torch.reshape(invariant_features, (len(x), -1))
            if i == 0:
                print(f"Shape of invariant features after reshaping: {invariant_features.size()}")
            invariant_features = invariant_features.detach().cpu().numpy()

            reversed_features = torch.reshape(reversed_features, (len(x), -1))
            reversed_features = reversed_features.detach().cpu().numpy()

            ID_x = ID_x.cpu().numpy()
            print(ID_x)

            score_mae = np.mean(np.abs(y2_reshape - x_hat), axis=(1, 2))
            score_mae_average.extend(score_mae)

            input_images.extend(x_reshape)
            target_images.extend(y2_reshape)
            predicted_images.extend(x_hat)
            invariant_features_all.extend(invariant_features)
            reversed_features_all.extend(reversed_features)
            label_all_site.extend(y1)
            output_all_site.extend(site_output1)
            subject_IDs_all.extend(ID_x)

    score_mae_average = np.mean(np.array(score_mae_average))  # check if this logic is correct or not
    input_images = np.stack(input_images)
    target_images = np.stack(target_images)
    predicted_images = np.stack(predicted_images)
    invariant_features_all = np.stack(invariant_features_all)
    reversed_features_all = np.stack(reversed_features_all)
    label_all_site = np.stack(label_all_site)
    output_all_site = np.stack(output_all_site)
    subject_IDs_all = np.stack(subject_IDs_all)

    score_accuracy = accuracy_score(label_all_site, output_all_site)
    confusion_matrix_x = confusion_matrix(label_all_site, output_all_site)
    entropy = entropy_of_confusion_matrix(confusion_matrix_x)

    return subject_IDs_all, input_images, target_images, predicted_images, invariant_features_all, reversed_features_all, label_all_site, \
           output_all_site, score_mae_average, score_accuracy, entropy, confusion_matrix_x

class VectorDataset(Dataset):
    def __init__(self, input_tensors, input_original_tensors, site_outputs,
                 target_tensors, conditional_tensors, subject_ID_tensors):
        """
        Args:
            input_tensors (list): Array of input tensors
            input_original_tensors (list): List of input original tensors
            site_outputs (list): List of sites
            target_tensors (list): Array of target tensors
            conditional_tensors (list): List of conditional tensors
            subject_ID_tensors (list): List of subject_ID tensors
        """
        self.input_tensors = input_tensors
        self.input_original_tensors = input_original_tensors
        self.site_outputs = site_outputs
        self.target_tensors = target_tensors
        self.conditional_tensors = conditional_tensors
        self.subject_ID_tensors = subject_ID_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        # Load input, target and conditional tensors and the site value
        input_tensor = self.input_tensors[idx]
        input_original_tensor = self.input_original_tensors[idx]
        site_output = self.site_outputs[idx]
        target_tensor = self.target_tensors[idx]
        conditional_tensor = self.conditional_tensors[idx]
        subject_ID_tensor = self.subject_ID_tensors[idx]

        return input_tensor, input_original_tensor, site_output, \
            target_tensor, conditional_tensor, subject_ID_tensor

if __name__ == "__main__":
    main()
