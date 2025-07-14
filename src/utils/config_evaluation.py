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

def read_configuration_param():
    """Returns configuration parameters"""

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
        "--task", default="site", type=str, help="Task to be performed")

    parser.add_argument(
        "--feature_latent", default="graph", type=str,
        help="latent space is from graph neural networks or 2D convolutional neural networks")

    parser.add_argument(
        "--model_task", default=1, type=int,
        help="choose which type of mlp model to use for the specific task")

    parser.add_argument(
        "--base_layer", default="cheb", type=str, help="Default base graph convolutional layer")

    parser.add_argument(
        "--hidden_layer_size", default=256, type=int, help="Number of features per node in the hidden layer next to input layer")

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

    #parser.add_argument(
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
        "--lr", default=0.01, type=float, help="Initial learning rate of the Adam optimizer in general") #TODO - keep lr in the main program

    parser.add_argument(
        "--lr_patience", default=5, type=int, help="Patience for the learning rate scheduler") #TODO - Keep this as a hyperparameter

    parser.add_argument(
        "--lr_factor", default=0.91, type=float, help="Factor for the learning rate scheduler")

    parser.add_argument(
        "--lr_threshold", default=0.00001, type=float, help="Threshold for the learning rate scheduler") #TODO - Keep this as a hyperparameter

    config = parser.parse_args()

    return config

