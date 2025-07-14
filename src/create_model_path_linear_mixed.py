import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

def create_model_path_function(output_path, parameter_file_combination_path, line_number, MAINDIR):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--epochs", default=500, type=int, help="Number of epochs")

    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size")

    parser.add_argument(
        "--maindir", default="checkpoints1", type=str, help="Main directory")

    parser.add_argument(
        "--ckdir", default="checkpoints", type=str, help="Checkpoint directory")

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
    #for arg, value in vars(config).items():
    #    if isinstance(value, bool):
    #        # If the argument is boolean, check its value
    #        if value:
    #            print(f'{arg} is set to True')
    #        else:
    #            print(f'{arg} is set to False')
    #    else:
    #        # Print non-boolean argument values
    #        print(f'{arg}: {value}')

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

    save_path = os.path.join(output_path, "models", MAINDIR, directory_name1,
                             directory_name2, f"SEED_{config.seed}", config.ckdir)

    #print(save_path)

    return save_path










