import os
import argparse
import glob
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import pearsonr

from utils.data import load_graphs_dict, graph_matrix_vector, matrix_to_vector, create_directory
from utils.autoencoder_model_2d import ResNetAutoEncoder2D
from utils.functions_basic import entropy_of_confusion_matrix ,set_seed

data_path = "/data/hagmann_group/jagruti/dataset_1065/HCP_DWI"
file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/dataset_creation"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

def autoencoder2D_evaluation_function(output_path, parameter_file_combination_path, line_number, epoch_number, MAINDIR, EVAL_SET):

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
        "--train_set", default="train", type=str, help="Name of training set")

    parser.add_argument(
        "--val_set", default="val", type=str, help="Name of validation set")

    parser.add_argument(
        "--test_set", default="test", type=str, help="Name of test set")

    parser.add_argument(
        "--best", default="", const="", nargs='?', type=str, help="Model Selection")

    parser.add_argument(
        "--mixed", default=True, type=str2bool, nargs='?', help="To have the training data as mixed and independent or have 4 travelling subjects per data point")

    parser.add_argument(
        "--augment", default=False, type=str2bool, nargs='?', help="To augment data or not for mixed independent training")

    parser.add_argument(
        "--linear_augment", default=False, type=str2bool, nargs='?',
        help="To augment data for mixed independent training by mixing two matrices in a linear interpolation way")

    parser.add_argument(
        "--linear_augment_lambda", default=0.5, type=float, help="lambda value for mixing two matrices in a linear interpolation way")

    parser.add_argument(
        "--random_augment", default=False, type=str2bool, nargs='?',
        help="To augment data for mixed independent training with random percentage of contribution or 50-50 contribution from two matrices")

    parser.add_argument(
        "--symmetric_augment", default=False, type=str2bool, nargs='?', help="To keep the augmented data symmetric or not")

    parser.add_argument(
        "--num_pairs", default=1000, type=int, help="Hyperparameter for setting the number of pairs to augment for each site")

    parser.add_argument(
        "--alpha_change_100", default=True, type=str2bool, nargs='?', help="To keep changing the alpha parameter after epoch 100 or not")

    parser.add_argument(
        "--alpha_param", default=10.0, type=float, help="Hyperparameter for setting the alpha value for the gradient reversal layer")

    parser.add_argument(
        "--use_mean", default=False, type=str2bool, nargs='?', help="To use site-specific mean of matrices at the decoder stage instead of individual matrices")

    parser.add_argument(
        "--mean_binary_threshold", default=0.35, type=float, help="Threshold for creating the mean binary matrix")

    parser.add_argument(
        "--model_name", default="AE", type=str, help="To choose auto-encoder or variational auto-encoder")

    parser.add_argument(
        "--model_encoder", default=0, type=int, help="choose which type of encoder model to use")

    parser.add_argument(
        "--base_layer_encoder", default="cheb", type=str, help="Base graph convolutional layer for encoder")

    parser.add_argument(
        "--hidden_layer_size", default=256, type=int,
        help="Number of features per node in the hidden layer next to input layer incase of a graph neural network or number of hidden units next to input units incase of a fully connected network")

    parser.add_argument(
        "--bottleneck_size", default=64, type=int,
        help="Number of features per node in the embedding space incase of a graph neural network or number of features in the embedding space incase of a fully connected network")

    parser.add_argument(
        "--normalization_g_encoder", default="batch", type=str, help="Normalization layer for graph convolutional layers of encoder")

    parser.add_argument(
        "--normalization_l_encoder", default="batch", type=str, help="Normalization layer for fully connected part of encoder")

    parser.add_argument(
        "--activation_g_encoder", default="lrelu", type=str, help="Activation layer for graph convolutional layers of encoder")

    parser.add_argument(
        "--activation_l_encoder", default="lrelu", type=str, help="Activation layer for fully connected part of encoder")

    parser.add_argument(
        "--instance_track_encoder", default=False, type=str2bool, nargs='?', help="To keep track of running mean and variance for encoder if it uses instance normalization")

    parser.add_argument(
        "--filter_size_encoder", default=2, type=int, help="Hyperparameter for setting the Chebyshev filter size for encoder")

    parser.add_argument(
        "--heads_encoder", default=2, type=int, help="Number of multi-head-attentions for GATv2Conv graph convolutional layer for encoder")

    parser.add_argument(
        "--concat_gat_encoder", default=False, type=str2bool, nargs='?', help="To concat the multi-head outputs or not for GATv2Conv graph convolutional layer for encoder")

    parser.add_argument(
        "--dropout_gat_encoder", default=0.2, type=float,
        help="Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training for GATv2Conv graph convolutional layer for encoder")

    parser.add_argument(
        "--dropout_encoder", default=False, type=str2bool, nargs='?', help="To include dropout or not if there is a fully connected part in the encoder")

    parser.add_argument(
        "--dropout_rate_encoder", default=0.1, type=float, help="Dropout rate if dropout for the fully connected part of the encoder is True")

    parser.add_argument(
        "--type_input_domain", default="linear", type=str, help="choose if the input to the domain classifier is linear or graph")

    parser.add_argument(
        "--model_domain", default=1, type=int, help="choose which type of domain classifier model to use for the linear part")

    parser.add_argument(
        "--model_domain_graph", default=0, type=int, help="choose which type of domain classifier model to use for the graph part")

    parser.add_argument(
        "--base_layer_domain", default="cheb", type=str, help="Base graph convolutional layer for domain classifier")

    parser.add_argument(
        "--normalization_l_domain", default="batch", type=str, help="Normalization layer for linear part of domain classifier")

    parser.add_argument(
        "--activation_l_domain", default="lrelu", type=str, help="Activation layer for linear part of domain classifier")

    parser.add_argument(
        "--normalization_g_domain", default="batch", type=str, help="Normalization layer for graph convolutional layers of domain classifier")

    parser.add_argument(
        "--activation_g_domain", default="lrelu", type=str, help="Activation layer for graph convolutional layers of domain classifier")

    parser.add_argument(
        "--instance_track_domain", default=False, type=str2bool, nargs='?', help="To keep track of running mean and variance for domain classifier if it uses instance normalization")

    parser.add_argument(
        "--filter_size_domain", default=2, type=int, help="Hyperparameter for setting the Chebyshev filter size for domain classifier")

    parser.add_argument(
        "--heads_domain", default=2, type=int, help="Number of multi-head-attentions for GATv2Conv graph convolutional layer for domain classifier")

    parser.add_argument(
        "--concat_gat_domain", default=False, type=str2bool, nargs='?', help="To concat the multi-head outputs or not for GATv2Conv graph convolutional layer for domain classifier")

    parser.add_argument(
        "--dropout_gat_domain", default=0.2, type=float,
        help="Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training for GATv2Conv graph convolutional layer for domain classifier")

    parser.add_argument(
        "--dropout", default=True, type=str2bool, nargs='?', help="To include dropout or not in the domain classifier or GAN")

    parser.add_argument(
        "--dropout_rate", default=0.1, type=float, help="Dropout rate if dropout is True")

    parser.add_argument(
        "--conditional_decoder", default=True, type=str2bool, nargs='?', help="To have the decoder conditioned on a latent variable")

    parser.add_argument(
        "--conditional_vector_size", default=64, type=int, help="Number of dimensions of the conditional variable")

    parser.add_argument(
        "--latent_vector_size", default=64, type=int, help="Number of dimensions of the conditional latent variable")

    parser.add_argument(
        "--style1_vector_size", default=64, type=int, help="Number of dimensions of the style1 variable")

    parser.add_argument(
        "--style2_vector_size", default=64, type=int, help="Number of dimensions of the style2 variable")

    parser.add_argument(
        "--hidden_layer_size_mapping_network", default=64, type=int, help="Number of features for the hidden layers of the mapping network")

    parser.add_argument(
        "--num_layers_mapping_network_latent", default=5, type=int, help="Number of layers in the mapping network for the latent variable")

    parser.add_argument(
        "--num_layers_mapping_network_style1", default=5, type=int, help="Number of layers in the mapping network for the style1 vector")

    parser.add_argument(
        "--num_layers_mapping_network_style2", default=5, type=int, help="Number of layers in the mapping network for the style2 vector")

    parser.add_argument(
        "--normalization_l_mapping", default=None, type=str, help="Normalization layer for mapping network")

    parser.add_argument(
        "--activation_l_mapping", default="lrelu", type=str, help="Activation layer for mapping network")

    parser.add_argument(
        "--model_decoder", default=0, type=int, help="choose which type of decoder model to use")

    parser.add_argument(
        "--model_edgePred", default=0, type=int, help="choose which type of edge prediction model to use")

    parser.add_argument(
        "--matrix_threshold", default=0.5, type=float, help="Threshold for creating the binary matrix")

    parser.add_argument(
        "--base_layer_decoder", default="cheb", type=str, help="Base graph convolutional layer for decoder")

    parser.add_argument(
        "--normalization_l1_decoder", default="instance", type=str,
        help="Normalization layer for linear part of decoder")

    parser.add_argument(
        "--normalization_l2_decoder", default="instance", type=str, help="Normalization layer for linear part of decoder for only edge prediction")

    parser.add_argument(
        "--normalization_g_decoder", default="batch", type=str, help="Normalization layer for graph convolutional layers of decoder")

    parser.add_argument(
        "--activation_l1_decoder", default="lrelu", type=str, help="Activation layer for linear part of decoder")

    parser.add_argument(
        "--activation_l2_decoder", default="lrelu", type=str, help="Activation layer for linear part of decoder for only edge prediction")

    parser.add_argument(
        "--activation_g_decoder", default="lrelu", type=str, help="Activation layer for graph convolutional layers of decoder")

    parser.add_argument(
        "--instance_track_decoder", default=False, type=str2bool, nargs='?', help="To keep track of running mean and variance for decoder if it uses instance normalization")

    parser.add_argument(
        "--filter_size_decoder", default=2, type=int, help="Hyperparameter for setting the Chebyshev filter size for decoder")

    parser.add_argument(
        "--heads_decoder", default=4, type=int, help="Number of multi-head-attentions for GATv2Conv graph convolutional layer for decoder")

    parser.add_argument(
        "--concat_gat_decoder", default=True, type=str2bool, nargs='?', help="To concat the multi-head outputs or not for GATv2Conv graph convolutional layer for decoder")

    parser.add_argument(
        "--dropout_gat_decoder", default=0.3, type=float,
        help="Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training for GATv2Conv graph convolutional layer for decoder")

    parser.add_argument(
        "--dropout_decoder1", default=False, type=str2bool, nargs='?', help="To include dropout or not in the fully connected part of the decoder")

    parser.add_argument(
        "--dropout_rate_decoder1", default=0.1, type=float, help="Dropout rate if dropout for the fully connected part of the decoder is True")

    parser.add_argument(
        "--dropout_decoder2", default=False, type=str2bool, nargs='?', help="To include dropout or not in the fully connected part of the decoder for only edge prediction")

    parser.add_argument(
        "--dropout_rate_decoder2", default=0.1, type=float, help="Dropout rate if dropout for the fully connected part of the decoder for only edge prediction is True")

    parser.add_argument(
        "--GAN", default=False, type=str2bool, nargs='?', help="whether to use GAN or not")

    parser.add_argument(
        "--model_GAN", default=1, type=int, help="choose which type of GAN model to use")

    parser.add_argument(
        "--output", default=4, type=int, help="Number of output channels for the domain classifier")

    parser.add_argument(
        "--regression_variable", default=0, type=int, help="Acquisition parameter to predict")

    parser.add_argument(
        "--scale", default=3, type=int, help="Scale of parcellation used")

    parser.add_argument(
        "--scaler_input", default=0, type=int, help="To scale or not the input weights and which scaler to use")

    parser.add_argument(
        "--scaler_input_param", default=10000.0, type=float, help="Maximum weight for min-max scaler for input weights")

    parser.add_argument(
        "--scaler_output", default=0, type=int, help="To scale or not the outputs and which scaler to use")

    parser.add_argument(
        "--scaler_output_param", default=10000.0, type=float, help="Maximum weight for min-max scaler for outputs")

    parser.add_argument(
        "--constraint", default=False, type=str2bool, nargs='?', help="To add constraints to the reconstruction output when the scaler_output is non-zero")

    parser.add_argument(
        "--symmetric", default=False, type=str2bool, nargs='?', help="To have the output matrices symmetric or not")

    parser.add_argument(
        "--self_loop_add_input", default=False, type=str2bool, nargs='?', help="Add self-loops to the input graphs")

    parser.add_argument(
        "--self_loop_add_output", default=True, type=str2bool, nargs='?', help="Add self-loops to the output matrices")

    parser.add_argument(
        "--regularize", default=False, type=str2bool, nargs='?', help="To regularize the reconstruction loss or not")

    parser.add_argument(
        "--reg_param", default=0.5, type=float, help="Amount of regularization loss compared to reconstruction loss")

    parser.add_argument(
        "--correlation", default=False, type=str2bool, nargs='?', help="To use correlation loss in the loss or not")

    parser.add_argument(
        "--corr_param", default=0.5, type=float, help="Amount of correlation loss compared to reconstruction loss")

    parser.add_argument(
        "--siamese", default=False, type=str2bool, nargs='?', help="To use siamese loss in the loss or not")

    parser.add_argument(
        "--sia_param", default=0.5, type=float, help="Amount of siamese loss compared to reconstruction loss")

    parser.add_argument(
        "--nodStr", default=False, type=str2bool, nargs='?', help="To use nodal strength loss in the loss or not")

    parser.add_argument(
        "--nodStr_param", default=0.2, type=float, help="Amount of nodal strength loss compared to reconstruction loss")

    parser.add_argument(
        "--klDiv", default=False, type=str2bool, nargs='?', help="To use kl divergence loss in the loss or not")

    parser.add_argument(
        "--klDiv_param", default=0.1, type=float, help="Amount of kl divergence loss compared to reconstruction loss")

    parser.add_argument(
        "--eigenValue", default=False, type=str2bool, nargs='?', help="To use eigen value loss in the loss or not")

    parser.add_argument(
        "--eigenValue_param", default=0.2, type=float, help="Amount of eigen value loss compared to reconstruction loss")

    parser.add_argument(
        "--linkPred", default=False, type=str2bool, nargs='?', help="To predict link not")

    parser.add_argument(
        "--linkPred_param", default=10.0, type=float,
        help="Amount of weight given to link prediction compared to reconstruction loss")

    parser.add_argument(
        "--linkPred_weighted", default=False, type=str2bool, nargs='?',
        help="To weigh the positive vs zero labels in link prediction")

    parser.add_argument(
        "--linkPred_weight", default=2.5, type=float,
        help="How much to weigh the positive vs zero labels in link prediction")

    parser.add_argument(
        "--weighted_ce", default=True, type=str2bool, nargs='?', help="To weigh the cross entropy predictions or not as per the target pairs")

    parser.add_argument(
        "--weight", default=True, type=str2bool, nargs='?', help="To weigh the zero vs non-zero connections separately")

    parser.add_argument(
        "--weight_param0", default=2.0, type=float, help="Amount of weight for misclassifying non-zero edges for site0")

    parser.add_argument(
        "--weight_param1", default=2.2, type=float, help="Amount of weight for misclassifying non-zero edges for site1")

    parser.add_argument(
        "--weight_param2", default=2.3, type=float, help="Amount of weight for misclassifying non-zero edges for site2")

    parser.add_argument(
        "--weight_param3", default=2.5, type=float, help="Amount of weight for misclassifying non-zero edges for site3")

    parser.add_argument(
        "--weight_param_edge", default=0.1, type=float, help="Amount of weight for only edge prediction network compared to entire graph prediction network")

    parser.add_argument(
        "--reconstruction_param", default=100.0, type=float, help="Amount of weight for reconstruction loss in a GAN network")

    parser.add_argument(
        "--lr", default=1.0, type=float, help="Initial learning rate of the Adam optimizer in general")

    parser.add_argument(
        "--lr_patience", default=5, type=int, help="Patience for the learning rate scheduler")

    parser.add_argument(
        "--lr_factor", default=0.91, type=float, help="Factor for the learning rate scheduler")

    parser.add_argument(
        "--lr_threshold", default=0.00001, type=float, help="Threshold for the learning rate scheduler")

    parser.add_argument(
        "--dis_lr", default=0.01, type=float, help="Initial learning rate of the discriminator")

    parser.add_argument(
        "--domClass_lr", default=0.001, type=float, help="Initial learning rate of the domain classifier")

    parser.add_argument(
        "--mapNet_lr", default=0.001, type=float, help="Initial learning rate of the mapping network")

    parser.add_argument(
        "--edgePred_lr", default=0.001, type=float, help="Initial learning rate of the only edge prediction network")

    parser.add_argument(
        "--weightPred_lr", default=1.0, type=float, help="Initial learning rate of the weight prediction network")

    with open(parameter_file_combination_path, "r") as f:
        args_from_file = f.readlines() # Read lines
        args_params = args_from_file[line_number].strip().split()  # Read and split arguments of the specific line number

    print(parameter_file_combination_path)
    print(f"Line number: {line_number+1}")

    config = parser.parse_args(args_params)

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
        model_directory_name = model_directory_name + "_lP_T_" + (str(config.linkPred_param).replace(".", "_"))
        if config.linkPred_weighted:
            model_directory_name = model_directory_name + "_" + (str(config.linkPred_weight).replace(".", "_"))

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
        return

    eval_dataset = create_dataset(EVAL_SET, save_path, config)

    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, num_workers=8)

    for i, d in enumerate(eval_dataloader):
        print(d)
        print(f"Length of test batch device : {len(d[0])}")

    # Number of features and outputs
    n_features, n_outputs = eval_dataset.input_tensors[0].shape[0], config.output

    print("Number of samples in the evaluation dataset: ", len(eval_dataset))
    print("Output of one sample from the evaluation dataset: ", eval_dataset[0])
    print("Number of features per node: ", n_features)
    print("Number of outputs per graph: ", n_outputs)

    if config.dropout:
        dp_rate = config.dropout_rate
    else:
        dp_rate = 0

    ### DEFINE THE MODEL
    model = ResNetAutoEncoder2D(in_channels=n_features, n_fc=config.num_layers_mapping_network_latent,
                       dim_latent=config.conditional_vector_size, out_channels=n_features, num_sites=n_outputs,
                       model_domain=config.model_domain, normalization_l_domain=config.normalization_l_domain,
                       activation_l_domain=config.activation_l_domain, p=dp_rate).to(device)

    print(model)

    # loading a saved model
    model_number = "train_model_epoch_{}.pth".format(epoch_number)
    model_path = os.path.join(save_path, model_number)
    if os.path.exists(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint["epoch"] + 1
        print("epoch_loaded", epoch)
    else:
        print(f"{config.best} model not present")
        return

    ### predict from the model

    eval_subject_IDs, eval_input_images, eval_target_images, eval_predicted_images, eval_reversed_features, eval_label_site, \
    eval_output_site, eval_mae_mean_site, eval_accuracy_site, eval_confusion_matrix, eval_entropy, \
    eval_mae, eval_accuracy, eval_pearsonr, eval_Idiff = predict(model, device, eval_dataloader, config)

    eval_mae_df = pd.DataFrame(columns=['input','predicted'])
    eval_accuracy_x_df = pd.DataFrame(columns=['input', 'predicted'])
    eval_accuracy_y_df = pd.DataFrame(columns=['input', 'predicted'])
    eval_pearsonr_df = pd.DataFrame(columns=['input', 'predicted'])
    eval_Idiff_df = pd.DataFrame(columns=['input', 'predicted'])

    for i_x in np.unique(eval_label_site):
        for col_x in eval_mae_df.columns:
            eval_mae_df.loc[i_x, col_x] = eval_mae[i_x][col_x]
            eval_accuracy_x_df.loc[i_x, col_x] = eval_accuracy[i_x][col_x][0]
            eval_accuracy_y_df.loc[i_x, col_x] = eval_accuracy[i_x][col_x][1]
            eval_pearsonr_df.loc[i_x, col_x] = eval_pearsonr[i_x][col_x]
            eval_Idiff_df.loc[i_x, col_x] = eval_Idiff[i_x][col_x]

    print(f"Epoch : {epoch}")
    print(f"{EVAL_SET} set:")
    print("MAE of all sites: {}".format(eval_mae_mean_site))
    print("Accuracy of site prediction: {}".format(eval_accuracy_site))
    print("Confusion Matrix : {}".format(eval_confusion_matrix))
    print("Entropy : {}".format(eval_entropy))
    print("Maximum edge : {}".format(eval_predicted_images.max()))
    print("Minimum edge : {}".format(eval_predicted_images.min()))
    print(f"MAE for individual sites: \n {eval_mae_df}")
    print(f"Accuracy for individual sites across axis 0: \n {eval_accuracy_x_df}")
    print(f"Accuracy for individual sites across axis 1: \n {eval_accuracy_y_df}")
    print(f"Pearsonr for individual sites: \n {eval_pearsonr_df}")
    print(f"Idiff for individual sites: \n {eval_Idiff_df}")
    print(f"Size of {EVAL_SET} input matrices: {eval_input_images.shape}")
    print(f"Size of {EVAL_SET} target matrices: {eval_target_images.shape}")
    print(f"Size of {EVAL_SET} predicted matrices: {eval_predicted_images.shape}")

    save_results_path = os.path.join(save_path, "epoch_" + str(epoch))
    create_directory(save_results_path)
    print(f"Directory '{save_results_path}' created successfully (if it didn't exist).")

    save_results_path_eval = os.path.join(save_results_path, EVAL_SET)
    create_directory(save_results_path_eval)
    print(f"Directory '{save_results_path_eval}' created successfully (if it didn't exist).")

    np.save(os.path.join(save_results_path_eval, f"{EVAL_SET}_input_images.npy"), eval_input_images)

    np.save(os.path.join(save_results_path_eval, f"{EVAL_SET}_predicted_images.npy"), eval_predicted_images)

    np.save(os.path.join(save_results_path_eval, f"{EVAL_SET}_target_images.npy"), eval_target_images)

    np.save(os.path.join(save_results_path_eval, f"{EVAL_SET}_reversed_features.npy"), eval_reversed_features)

    np.save(os.path.join(save_results_path_eval, f"{EVAL_SET}_label_site.npy"), eval_label_site)

    np.save(os.path.join(save_results_path_eval, f"{EVAL_SET}_output_site.npy"), eval_output_site)

    np.save(os.path.join(save_results_path_eval, f"{EVAL_SET}_subject_IDs.npy"), eval_subject_IDs)

    print("Model prediction over")

    return

def create_dataset(set1, path, config):

    """load the graphs, the matrices and the sites for the subject set"""

    bvalues = [1000, 3000]
    resolutions_paths = ["2_3", "1_25"]

    num_sites = len(bvalues) * len(resolutions_paths)
    conditional_variable_dict = {}  # input code for the corresponding site
    num_vectors_per_site = int(config.conditional_vector_size / num_sites)
    print(f"Number of vectors per site : {num_vectors_per_site}")
    for site_x in range(num_sites):
        z = torch.zeros([1, config.conditional_vector_size], dtype=torch.float)
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

            # getting the graphs for edge attribute "normalized_fiber_density"
            matrices_number_of_fibers_input, _ = graph_matrix_vector(subject_IDs_valid_common, graphs_dict_input,
                                                                     weight_value='number_of_fibers')
            matrices_number_of_fibers_input1 = matrices_number_of_fibers_input.copy()
            matrices_number_of_fibers_output, _ = graph_matrix_vector(subject_IDs_valid_common, graphs_dict_output,
                                                                      weight_value='number_of_fibers')

            for i, subject in enumerate(subject_IDs_valid_common):
                x = torch.tensor(matrices_number_of_fibers_input1[subject], dtype=torch.float)
                x = x.unsqueeze(0)
                x1 = torch.tensor(matrices_number_of_fibers_input[subject], dtype=torch.float)
                y1 = torch.tensor(j*2+k, dtype=torch.long)
                y2 = torch.tensor(matrices_number_of_fibers_output[subject], dtype=torch.float)
                y2 = y2.unsqueeze(0)
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

    image_dataset = ImageDataset(input_tensors, input_original_tensors, site_outputs, target_tensors, conditional_tensors, subject_ID_tensors)

    return image_dataset
def predict(model, device, dataloader, config):
    "model prediction"

    score_mae_average = []
    x_all = {}
    y_all = {}
    x_hat_all = {}
    input_images = []
    target_images = []
    predicted_images = []
    label_all_site = []
    output_all_site = []
    reversed_features_all = []
    subject_IDs_all = []

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

            reversed_features, site_output, x_hat_2 = model(x, y3, 1.0)

            if i == 0:
                print(f"Shape of invariant features: {reversed_features.size()}")
            reversed_features = reversed_features.detach().cpu().numpy()

            site_output1 = torch.argmax(site_output, 1)
            site_output1 = site_output1.detach().cpu().numpy()

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

            x_reshape = x1
            x_reshape = x_reshape.cpu().numpy()
            y1 = y1.cpu().numpy()
            y2_reshape = y2.squeeze()
            y2_reshape = y2_reshape.cpu().numpy()

            #ID_x = ID_x.squeeze()
            ID_x = ID_x.cpu().numpy()
            print(ID_x)

            score_mae = np.mean(np.abs(y2_reshape - x_hat), axis=(1, 2))
            score_mae_average.extend(score_mae)

            input_images.extend(x_reshape)
            target_images.extend(y2_reshape)
            predicted_images.extend(x_hat)
            reversed_features_all.extend(reversed_features)
            label_all_site.extend(y1)
            output_all_site.extend(site_output1)
            subject_IDs_all.extend(ID_x)

            for j in np.unique(y1):
                if j not in x_all.keys():
                    x_all[j] = []
                if j not in y_all.keys():
                    y_all[j] = []
                if j not in x_hat_all.keys():
                    x_hat_all[j] = []
                x_all[j].extend(x_reshape[y1 == j])
                y_all[j].extend(y2_reshape[y1 == j])
                x_hat_all[j].extend(x_hat[y1 == j])

    mae_all = {}
    accuracy_all_site = {}
    pearsonr_all = {}
    Idiff_all = {}

    for i in np.unique(label_all_site):
        print("Site number", i)
        input_vectors_site = matrix_to_vector(x_all[i])
        target_vectors_site = matrix_to_vector(y_all[i])
        predicted_vectors_site = matrix_to_vector(x_hat_all[i])
        difference_matrix_input = np.zeros((len(target_vectors_site), len(target_vectors_site)))
        difference_matrix_predicted = np.zeros((len(target_vectors_site), len(target_vectors_site)))
        mae_all[i] = {}
        accuracy_all_site[i] = {}
        pearsonr_all[i] = {}
        Idiff_all[i] = {}

        for j in range(len(target_vectors_site)):
            for k in range(len(target_vectors_site)):
                difference_matrix_input[j, k] = np.mean(np.abs(target_vectors_site[j] - input_vectors_site[k]))
                difference_matrix_predicted[j, k] = np.mean(np.abs(target_vectors_site[j] - predicted_vectors_site[k]))

        pearsonr_all[i]['input'] = []
        pearsonr_all[i]['predicted'] = []
        for j in range(len(target_vectors_site)):
            pearsonr_all[i]['input'].append(pearsonr(target_vectors_site[j], input_vectors_site[j])[0])
            pearsonr_all[i]['predicted'].append(pearsonr(target_vectors_site[j], predicted_vectors_site[j])[0])

        pearsonr_all[i]['input'] = np.mean(np.array(pearsonr_all[i]['input']))
        pearsonr_all[i]['predicted'] = np.mean(np.array(pearsonr_all[i]['predicted']))

        mae_all[i]['input'] = np.mean(np.abs(target_vectors_site - input_vectors_site))
        mae_all[i]['predicted'] = np.mean(np.abs(target_vectors_site - predicted_vectors_site))

        accuracy_all_site[i]['input'] = find_accuracy(difference_matrix_input)
        accuracy_all_site[i]['predicted'] = find_accuracy(difference_matrix_predicted)

        Idiff_all[i]['input'] = find_Idiff(difference_matrix_input)
        Idiff_all[i]['predicted'] = find_Idiff(difference_matrix_predicted)

    score_mae_average = np.mean(np.array(score_mae_average))  # check if this logic is correct or not
    input_images = np.stack(input_images)
    target_images = np.stack(target_images)
    predicted_images = np.stack(predicted_images)
    label_all_site = np.stack(label_all_site)
    output_all_site = np.stack(output_all_site)
    reversed_features_all = np.stack(reversed_features_all)
    subject_IDs_all = np.stack(subject_IDs_all)

    score_accuracy = accuracy_score(label_all_site, output_all_site)
    confusion_matrix_x = confusion_matrix(label_all_site, output_all_site)
    entropy = entropy_of_confusion_matrix(confusion_matrix_x)

    return subject_IDs_all, input_images, target_images, predicted_images, reversed_features_all, label_all_site, \
           output_all_site, score_mae_average, score_accuracy, confusion_matrix_x, entropy, \
           mae_all, accuracy_all_site, pearsonr_all, Idiff_all

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

class ImageDataset(Dataset):
    def __init__(self, input_tensors, input_original_tensors, site_outputs, target_tensors, conditional_tensors, subject_ID_tensors):
        """
        Args:
            input_tensors (list): List of input tensors
            input_original_tensors (list): List of input original tensors
            site_outputs (list): List of sites
            target_tensors (list): List of target tensors
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

        return input_tensor, input_original_tensor, site_output, target_tensor, conditional_tensor, subject_ID_tensor











