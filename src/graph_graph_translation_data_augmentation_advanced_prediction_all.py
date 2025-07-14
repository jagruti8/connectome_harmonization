# import the packages

import os
import glob
import copy
import numpy as np
import pandas as pd
import pickle

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

from sklearn.metrics import accuracy_score, confusion_matrix

from utils.data import load_graphs_dict, graph_matrix_vector, graph_matrix_mean_vector, graph_version1_graph_refined, create_directory
from utils.ml_model_domain_discriminator_advanced_harmonization import GraphToGraphTranslationAE
from utils.functions_basic import entropy_of_confusion_matrix ,set_seed, matrix_to_graph

data_path = "/data/hagmann_group/jagruti/dataset_1065/HCP_DWI"
file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/dataset_creation"
output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"
MAINDIR = "GRAPHTOGRAPHADVANCED2"
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

    model_directory_parameters = ["hidden_layer_size", "normalization_g_encoder", "activation_g_encoder",
                                  "instance_track_encoder", "filter_size_encoder",
                                  "normalization_l_domain", "activation_l_domain", "instance_track_domain",
                                  "alpha_param", "dropout", "dropout_rate",
                                  "conditional_vector_size", "latent_vector_size", "style1_vector_size",
                                  "style2_vector_size",
                                  "hidden_layer_size_mapping_network", "num_layers_mapping_network_latent",
                                  "num_layers_mapping_network_style1",
                                  "num_layers_mapping_network_style2", "activation_l_mapping",
                                  "normalization_l1_decoder", "normalization_l2_decoder", "normalization_g_decoder",
                                  "activation_l1_decoder",
                                  "activation_l2_decoder", "activation_g_decoder", "instance_track_decoder",
                                  "filter_size_decoder",
                                  "scaler_input", "scaler_output", "self_loop_add_input", "self_loop_add_output",
                                  "symmetric",
                                  "correlation", "corr_param", "siamese", "sia_param", "nodStr", "nodStr_param",
                                  "eigenValue", "eigenValue_param",
                                  "weight", "weight_param0", "weight_param1", "weight_param2", "weight_param3",
                                  "lr", "domClass_lr", "mapNet_lr", "lr_patience", "lr_factor", "lr_threshold"]

    model_directory_arguments = ["enc", "", "", "", "",
                                 "dom", "", "",
                                 "a", "dp", "",
                                 "map", "", "", "", "", "", "", "", "",
                                 "dec", "", "", "", "", "", "", "",
                                 "scIO", "", "loopIO", "", "sym",
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

    if config.use_mean:
        directory_name5 = "MEAN_" + (str(config.mean_binary_threshold).replace(".", "_"))
    else:
        directory_name5 = "SPECIFIC"

    if config.type_input_domain == "linear":
        directory_name6 = "DOMAIN_LINEAR_" + str(config.model_domain)
    elif config.type_input_domain == "graph":
        directory_name6 = "DOMAIN_GRAPH_" + str(config.model_domain) + "_" + str(config.model_domain_graph) + "_" + str(
            config.base_layer_domain) + "_" \
                          + str(config.normalization_g_domain) + "_" + str(config.activation_g_domain) + "_" + str(
            config.filter_size_domain)

    if config.model_decoder == 9 or config.model_decoder == 10 or config.model_decoder == 12 or config.model_decoder == 13\
            or config.model_decoder==18 or config.model_decoder==20:
        directory_name7 = "DECODER_" + str(config.model_decoder) + "_" + str(config.base_layer_decoder) + "_" \
                          + str(config.model_edgePred) + "_" + (str(config.matrix_threshold).replace(".", "_")) + "_" \
                          + (str(config.weight_param_edge).replace(".", "_")) + "_" \
                          + (str(config.edgePred_lr).replace(".", "_")) + "_" \
                          + (str(config.weightPred_lr).replace(".", "_"))
    else:
        directory_name7 = "DECODER_" + str(config.model_decoder) + "_" + str(config.base_layer_decoder)

    if config.scaler_output != 0:
        if config.scaler_output == 1:
            directory_name8 = "SCALER_" + (str(config.scaler_input_param).replace(".", "_")) + "_" \
                              + (str(config.scaler_output_param).replace(".", "_"))
            save_path = os.path.join(output_path, "models", MAINDIR, directory_name1,
                                     directory_name2, directory_name3, directory_name4, directory_name5,
                                     f"ENCODER_{config.model_encoder}_{config.base_layer_encoder}", directory_name6,
                                     directory_name7,
                                     f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                     f"CONSTRAINT_{config.constraint}", directory_name8,
                                     model_directory_name)
        else:
            save_path = os.path.join(output_path, "models", MAINDIR, directory_name1,
                                     directory_name2, directory_name3, directory_name4, directory_name5,
                                     f"ENCODER_{config.model_encoder}_{config.base_layer_encoder}", directory_name6,
                                     directory_name7,
                                     f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                     f"CONSTRAINT_{config.constraint}", model_directory_name)
    else:
        save_path = os.path.join(output_path, "models", MAINDIR, directory_name1,
                                 directory_name2, directory_name3, directory_name4, directory_name5,
                                 f"ENCODER_{config.model_encoder}_{config.base_layer_encoder}", directory_name6,
                                 directory_name7,
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
    n_features, n_outputs = val_dataset[0].x.shape[1], config.output

    #print("Number of samples in the train dataset: ", len(train_dataset))
    #print("Number of samples in the val dataset: ", len(val_dataset))
    #print("Number of features per node: ", n_features)
    #print("Number of outputs per graph: ", n_outputs)

    if config.dropout:
        dp_rate = config.dropout_rate
    else:
        dp_rate = 0

    if config.dropout_decoder1:
        dp_rate_decoder1 = config.dropout_rate_decoder1
    else:
        dp_rate_decoder1 = 0

    if config.dropout_decoder2:
        dp_rate_decoder2 = config.dropout_rate_decoder2
    else:
        dp_rate_decoder2 = 0

    ### DEFINE THE MODEL
    model = GraphToGraphTranslationAE(input_size=n_features, hidden_size=config.hidden_layer_size,
                                      mapping_input_size=config.conditional_vector_size,
                                      mapping_hidden_size=config.hidden_layer_size_mapping_network,
                                      mapping_latent_size=config.latent_vector_size,
                                      style_dim1=config.style1_vector_size, style_dim2=config.style2_vector_size,
                                      num_layers_latent=config.num_layers_mapping_network_latent,
                                      num_layers_style1=config.num_layers_mapping_network_style1,
                                      num_layers_style2=config.num_layers_mapping_network_style2,
                                      output_size=n_outputs, model_encoder=config.model_encoder,
                                      base_layer_encoder=config.base_layer_encoder,
                                      normalization_g_encoder=config.normalization_g_encoder,
                                      activation_g_encoder=config.activation_g_encoder,
                                      instance_track_encoder=config.instance_track_encoder,
                                      filter_size_encoder=config.filter_size_encoder,
                                      heads_encoder=config.heads_encoder,
                                      concat_gat_encoder=config.concat_gat_encoder,
                                      dropout_gat_encoder=config.dropout_gat_encoder,
                                      type_input_domain=config.type_input_domain,
                                      model_domain=config.model_domain,
                                      normalization_l_domain=config.normalization_l_domain,
                                      activation_l_domain=config.activation_l_domain,
                                      p=dp_rate, model_domain_graph=config.model_domain_graph,
                                      base_layer_domain=config.base_layer_domain,
                                      normalization_g_domain=config.normalization_g_domain,
                                      activation_g_domain=config.activation_g_domain,
                                      instance_track_domain=config.instance_track_domain,
                                      filter_size_domain=config.filter_size_domain, heads_domain=config.heads_domain,
                                      concat_gat_domain=config.concat_gat_domain,
                                      dropout_gat_domain=config.dropout_gat_domain,
                                      normalization_l_mapping=config.normalization_l_mapping,
                                      activation_l_mapping=config.activation_l_mapping,
                                      model_decoder=config.model_decoder,
                                      model_edgePred=config.model_edgePred,
                                      base_layer_decoder=config.base_layer_decoder,
                                      normalization_l1_decoder=config.normalization_l1_decoder,
                                      normalization_l2_decoder=config.normalization_l2_decoder,
                                      normalization_g_decoder=config.normalization_g_decoder,
                                      activation_l1_decoder=config.activation_l1_decoder,
                                      activation_l2_decoder=config.activation_l2_decoder,
                                      activation_g_decoder=config.activation_g_decoder,
                                      instance_track_decoder=config.instance_track_decoder,
                                      filter_size_decoder=config.filter_size_decoder,
                                      heads_decoder=config.heads_decoder, concat_gat_decoder=config.concat_gat_decoder,
                                      dropout_gat_decoder=config.dropout_gat_decoder, p_decoder1=dp_rate_decoder1,
                                      p_decoder2=dp_rate_decoder2).to(device)
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

    columns_to_print = ['accuracy_x_0', 'accuracy_x_1', 'mae_0', 'mae_1', 'mae_2', 'mae_3']

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

    # load list of subjects
    subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])

    # loading the .gpickle files for the chosen subjects for output SCs
    graphs_dict_output, subject_IDs_valid_output = load_graphs_dict(subject_IDs,
                                                                    os.path.join(data_path, "ds_HCP_bval_3000_anat_0_7_dwi_1_25"),
                                                                    config.scale, 0)

    nroi = None
    torch_graphs = []
    track_subjects_IDs = []

    for j, bval in enumerate(bvalues):
        for k, res in enumerate(resolutions_paths):

            # loading the .gpickle files for the chosen subjects for input SCs
            graphs_dict_input, subject_IDs_valid_input = load_graphs_dict(subject_IDs, os.path.join(data_path,"ds_HCP_bval_"+str(bval)+"_anat_0_7_dwi_"+res),config.scale,0)

            subject_IDs_valid_common = np.intersect1d(np.array(subject_IDs_valid_input),
                                                      np.array(subject_IDs_valid_output))

            #print(f"Length of common valid subject IDs : {len(subject_IDs_valid_common)}")

            # getting the graphs for edge attribute "normalized_fiber_density"
            graphs_number_of_fibers_input = graph_version1_graph_refined(subject_IDs_valid_common, graphs_dict_input,
                                                                   weight_value='number_of_fibers', self_loop=config.self_loop_add_input)
            matrices_number_of_fibers_output, _ = graph_matrix_vector(subject_IDs_valid_common, graphs_dict_output,
                                                                      weight_value='number_of_fibers')

            if config.use_mean:
                matrices_number_of_fibers_output_mean = np.load(
                    os.path.join(path, "3000_1_25" + "_mean_binary_matrix.npy"))
                print(
                    f"Number of connections in the mean structural matrix obtained with bvalue - {bval}, resolution - {res} is :{np.sum(matrices_number_of_fibers_output_mean)} and its shape is {matrices_number_of_fibers_output_mean.shape}")

                matrices_number_of_fibers_output_mean_1 = matrices_number_of_fibers_output_mean.copy()

                if config.self_loop_add_input:
                    np.fill_diagonal(matrices_number_of_fibers_output_mean_1, 1)
                mean_graph = matrix_to_graph(matrices_number_of_fibers_output_mean_1)
                mean_G = copy.deepcopy(mean_graph)
                mean_data_1 = from_networkx(mean_G, group_edge_attrs=list(next(iter(mean_G.edges(data=True)))[-1].keys()))
                mean_edge_index = mean_data_1.edge_index
                mean_edge_attr = mean_data_1.edge_attr.to(torch.float)

            if nroi is None:
                nroi = graphs_number_of_fibers_input[subject_IDs_valid_common[0]].order()
            for i, subject in enumerate(subject_IDs_valid_common):
                x_features = np.identity(nroi)
                # loading the graph
                G = copy.deepcopy(graphs_number_of_fibers_input[subject])
                # converting graph from networkx to tensor format
                data_1 = from_networkx(G, group_edge_attrs=list(next(iter(G.edges(data=True)))[-1].keys()))
                # edge indices
                edge_index = data_1.edge_index
                # edge weights
                edge_attr = data_1.edge_attr.to(torch.float)
                # if config.scaler_input is 0, means no transformation, if 1 then normalizing the whole weights by a given
                # maximum value and if 2 then log transforming it
                if config.scaler_input == 1:
                    edge_attr = edge_attr / config.scaler_input_param
                elif config.scaler_input == 2:
                    edge_attr = torch.log1p(edge_attr)
                # print(edge_attr.dtype)
                # node features - identity matrix
                x = torch.tensor(x_features, dtype=torch.float)
                y1 = torch.tensor(j*2+k, dtype=torch.long)
                y2 = torch.tensor(matrices_number_of_fibers_output[subject], dtype=torch.float)
                y3 = conditional_variable_dict[3]
                y_all = [y1, y2, y3]
                #print(len(y))
                if config.use_mean:
                    data_2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mean_edge_index=mean_edge_index,
                                  mean_edge_attr=mean_edge_attr, y=y_all)
                else:
                    data_2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_all)

                torch_graphs.append(data_2)
                track_subjects_IDs.append(subject)

    print(f"No of subjects in {set1} is {len(track_subjects_IDs)}")
    print(f"No of unique subjects in {set1} is {len(set(track_subjects_IDs))}")

    return torch_graphs

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
            batch = batch.to(device)
            nroi = int(batch.x.size()[0] / len(batch))

            y1, y2, y3 = batch.y

            if config.use_mean:
                _, _, site_output, x_hat_2, _ = model(batch.x, batch.edge_index,
                                                      batch.edge_attr, len(batch),
                                                      nroi, config.use_mean, batch.batch, config.matrix_threshold,
                                                      1.0, y3, batch.mean_edge_index,
                                                      batch.mean_edge_attr)
            else:
                _, _, site_output, x_hat_2, _ = model(batch.x, batch.edge_index,
                                                      batch.edge_attr, len(batch),
                                                      nroi, config.use_mean, batch.batch, config.matrix_threshold,
                                                      1.0, y3)

            if config.scaler_output == 1 and config.constraint == True:
                x_hat_2 = torch.sigmoid(x_hat_2)
            elif config.scaler_output == 2 and config.constraint == True:
                x_hat_2 = 9.21 * torch.sigmoid(x_hat_2)

            x_hat = 0.5 * (torch.permute(x_hat_2, (0, 2, 1)) + x_hat_2)
            x_hat = x_hat * (1 - torch.eye(nroi, nroi).repeat(len(batch), 1, 1)).to(device)  # TODO - check this logic again

            if config.scaler_output == 1:
                x_hat = x_hat * config.scaler_output_param
            elif config.scaler_output == 2:
                x_hat = torch.special.expm1(x_hat)

            x_hat = torch.round(torch.where(x_hat < 0, 0, x_hat))
            x_hat = x_hat.detach().cpu().numpy()

            site_output1 = torch.argmax(site_output, 1)
            site_output1 = site_output1.detach().cpu().numpy()

            y1 = y1.cpu().numpy()
            y2_reshape = torch.reshape(y2, (len(batch), nroi, nroi))
            y2_reshape = y2_reshape.cpu().numpy()

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

