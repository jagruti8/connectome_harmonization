# import the packages

print("image_image_translation_data_augmentation_autoencoder_2d.py starting")

import os
import glob
import random
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix

from utils.data import load_graphs_dict, graph_matrix_vector, graph_matrix_mean_vector, graph_version1_graph_refined, save_file, vector_to_matrix
from utils.autoencoder_model_2d import ResNetAutoEncoder2D
from utils.config_advanced import read_configuration_param
from utils.functions_basic import entropy_of_confusion_matrix, check_unique_and_no_intersection, set_seed, custom_cross_entropy_loss,\
pearson_correlation, combine_matrices, combine_matrices_equal, combine_matrices_linear, matrix_to_graph, nodal_strength_loss, \
kl_divergence_nodal_strength, eigenvalue_difference_batch, combine_vectors, combine_vectors_equal, combine_vectors_linear

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

    model_directory_parameters = ["normalization_l_domain", "activation_l_domain", "alpha_param", "dropout", "dropout_rate",
                                  "conditional_vector_size", "num_layers_mapping_network_latent",
                                  "scaler_input", "scaler_output", "symmetric",
                                  "correlation", "corr_param", "siamese", "sia_param", "nodStr", "nodStr_param", "eigenValue", "eigenValue_param",
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
            if len(value)>2:
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
        directory_name2 ="AUGMENT_"+str(config.num_pairs)
        if config.symmetric_augment:
            directory_name2 = os.path.join(directory_name2, "SYMMETRIC")
        if config.linear_augment:
            directory_name2 = os.path.join(directory_name2,"LINEAR_"+(str(config.linear_augment_lambda).replace(".","_")))
        else:
            if config.random_augment:
                directory_name2 = os.path.join(directory_name2,"RANDOM")
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

    if config.scaler_output!=0:
        if config.scaler_output==1:
            directory_name5 = "SCALER_"+(str(config.scaler_input_param).replace(".","_"))+"_"\
                              +(str(config.scaler_output_param).replace(".","_"))
            save_path = os.path.join(output_path, "models", config.maindir, directory_name1,
                                     directory_name2, directory_name3, directory_name4,
                                     f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                     f"CONSTRAINT_{config.constraint}", directory_name5,
                                     model_directory_name)
        else:
            save_path = os.path.join(output_path, "models", config.maindir, directory_name1,
                                 directory_name2, directory_name3, directory_name4,
                                 f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                 f"CONSTRAINT_{config.constraint}", model_directory_name)
    else:
        save_path = os.path.join(output_path, "models", config.maindir, directory_name1,
                                 directory_name2, directory_name3, directory_name4,
                                 f"BATCHSIZE_{config.batch_size}", f"SEED_{config.seed}",
                                 model_directory_name)

    os.makedirs(save_path, exist_ok=True)

    # Use glob to list all files that match the pattern
    file_list_paths = glob.glob(os.path.join(save_path, "train_model_epoch_*.pth"))
    file_list = [os.path.basename(file) for file in file_list_paths if os.path.isfile(file)]
    if len(file_list)>0:
        model_names = [(name_x.split('_')[3].split('.')[0]) for name_x in file_list]
        numbers = np.array([int(name_x) for name_x in model_names if name_x.isdigit()])
        if len(numbers) > 0:
            print(numbers)
            max_number = numbers.max()
            print(f"Largest epoch number: {max_number}")
            if (max_number+1) == config.epochs:
                print("Model training already over")
                return
            else:
                print("Model still needs to be trained")
        else:
            print("Only latest epoch model found")
            model_number = "train_model_epoch_latest.pth"
            model_path = os.path.join(save_path, model_number)
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            if start_epoch == config.epochs:
                print("Model training already over")
                return
            else:
                print("Model still needs to be trained")
    else:
        print("No matching files found.")



    train_dataset = create_dataset(config.train_set, "train", save_path, config)
    val_dataset = create_dataset(config.val_set, "val", save_path, config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=8)

    for i, d in enumerate(train_dataloader):
        #print(d)
        print(f"Length of train batch device : {len(d[0])}")

    # Number of features and outputs
    n_features, n_outputs = train_dataset.input_tensors[0].shape[0], config.output

    print("Number of samples in the train dataset: ", len(train_dataset))
    print("Number of samples in the val dataset: ", len(val_dataset))
    print("Output of one sample from the train dataset: ", train_dataset[0])
    print("Number of features per node: ", n_features)
    print("Number of classes per graph: ", n_outputs)

    ### Max number of epochs
    max_epochs = config.epochs

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

    ### DEFINE LOSS FUNCTION
    loss_task = nn.L1Loss().to(device)
    loss_site = nn.CrossEntropyLoss().to(device)
    if config.linkPred_weighted:
        pos_weight = torch.tensor([config.linkPred_weight])
        loss_edge = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        loss_edge = nn.BCEWithLogitsLoss().to(device)

    # Define a custom cost matrix
    # For 4 classes (0, 1, 2, 3)
    # Lower penalty for misclassifying within (0,1) or (2,3) pairs
    cost_matrix = torch.tensor([
        [0, 1, 3, 3],  # Misclassification cost for true class 0
        [1, 0, 3, 3],  # Misclassification cost for true class 1
        [3, 3, 0, 1],  # Misclassification cost for true class 2
        [3, 3, 1, 0]  # Misclassification cost for true class 3
    ], dtype=torch.float32)

    ### DEFINE OPTIMIZER

    optimizer = torch.optim.Adam([
        {'params': model.encoder_conv1.parameters(), 'lr':config.lr},
        {'params': model.encoder_bn1.parameters(), 'lr': config.lr},
        {'params': model.encoder_relu.parameters(), 'lr': config.lr},
        {'params': model.encoder_maxpool.parameters(), 'lr': config.lr},
        {'params': model.encoder_layer1.parameters(), 'lr': config.lr},
        {'params': model.encoder_layer2.parameters(), 'lr': config.lr},
        {'params': model.encoder_layer3.parameters(), 'lr': config.lr},
        {'params': model.encoder_layer4.parameters(), 'lr': config.lr},
        {'params': model.upconv4.parameters(), 'lr': config.lr},
        {'params': model.upconv3.parameters(), 'lr': config.lr},
        {'params': model.upconv2.parameters(), 'lr': config.lr},
        {'params': model.upconv1.parameters(), 'lr': config.lr},
        {'params': model.upconv0.parameters(), 'lr': config.lr},
        {'params': model.decoder3.parameters(), 'lr': config.lr},
        {'params': model.decoder2.parameters(), 'lr': config.lr},
        {'params': model.decoder1.parameters(), 'lr': config.lr},
        {'params': model.decoder0.parameters(), 'lr': config.lr},
        {'params': model.conv_original_size2.parameters(), 'lr': config.lr},
        {'params': model.final_conv.parameters(), 'lr': config.lr},
        {'params': model.site_classifier.parameters(), 'lr': config.domClass_lr},
        {'params': model.mapping_network.parameters(), 'lr': config.mapNet_lr}
            ])


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                                                           patience=config.lr_patience, threshold=config.lr_threshold)

    ### TRAIN THE MODEL
    epoch_list, mae_list, accuracy_list, entropy_list, results = train(model, loss_task, loss_site, loss_edge, cost_matrix, device, optimizer, scheduler, max_epochs,
                                                              train_dataloader, val_dataloader, config, save_path)

    print(epoch_list)
    print(mae_list)
    print(accuracy_list)
    print(entropy_list)
    print(results)

    save_file(np.array(epoch_list), os.path.join(save_path, "epoch_list"))
    save_file(np.array(mae_list),  os.path.join(save_path,"mae_list"))
    save_file(np.array(accuracy_list), os.path.join(save_path, "accuracy_list"))
    save_file(np.array(entropy_list), os.path.join(save_path, "entropy_list"))
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

    print("Model training over")

def train(model, loss_task, loss_site, loss_edge, cost_matrix, device, optimizer, scheduler, max_epochs, train_dataloader, val_dataloader, config, save_path):

    """train function"""

    epoch_list = []
    train_loss_list = []
    mae_list = []
    accuracy_list = []
    entropy_list = []
    alpha_list = []
    results = {}

    start_epoch = 0
    best_mae = None
    best_epoch = 0

    # TODO - find the latest model in the directory
    # loading a saved model
    if config.best:
        model_number = "train_model_epoch_{}.pth".format(config.best)
        model_path = os.path.join(save_path, model_number)
        if os.path.exists(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_mae = checkpoint["best_mae"]
            best_epoch = checkpoint["best_epoch"]
            epoch_list = checkpoint["epoch_list"]
            mae_list = checkpoint["mae_list"]
            accuracy_list = checkpoint["accuracy_list"]
            entropy_list = checkpoint["entropy_list"]
            train_loss_list = checkpoint["train_loss_list"]
            alpha_list = checkpoint["alpha_list"]
            results = checkpoint["results"]
            print("epoch_loaded", start_epoch)

    # loop over epochs
    for epoch in range(start_epoch, max_epochs):
        model.train()

        losses = []
        alpha_s = []

        if epoch < 100:
            up_epochs = 100
        if config.alpha_change_100:
            if epoch >= 100 and epoch < 200:
                up_epochs = 200
            elif epoch >= 200 and epoch < 600:
                up_epochs = 600
            elif epoch >= 600 and epoch < 700:
                up_epochs = 700
            elif epoch >= 700 and epoch < 800:
                up_epochs = 800
            elif epoch >= 800 and epoch < 900:
                up_epochs = 900
            elif epoch >= 900 and epoch < 1000:
                up_epochs = 1000
            elif epoch >= 1000 and epoch < 1100:
                up_epochs = 1100
            elif epoch >= 1100 and epoch < 1200:
                up_epochs = 1200
            elif epoch >= 1200 and epoch < 1300:
                up_epochs = 1300
            elif epoch >= 1300 and epoch < 1400:
                up_epochs = 1400
            elif epoch >= 1400 and epoch < 1500:
                up_epochs = 1500
            elif epoch >= 1500 and epoch < 1600:
                up_epochs = 1600
            elif epoch >= 1600 and epoch < 1700:
                up_epochs = 1700
            elif epoch >= 1700 and epoch < 1800:
                up_epochs = 1800
            elif epoch >= 1800 and epoch < 1900:
                up_epochs = 1900
            elif epoch >= 1900 and epoch < 2000:
                up_epochs = 2000
            else:
                up_epochs = max_epochs

        if epoch < 100 or config.alpha_change_100:
            start_steps = epoch * len(train_dataloader)
            total_steps = up_epochs * len(train_dataloader)
        # loop over batches
        for i, train_batch in enumerate(train_dataloader):

            # setup hyperparameters
            if epoch < 100 or config.alpha_change_100:
                p_alpha = float(i + start_steps) / total_steps
                alpha = 2. / (1. + np.exp(-config.alpha_param * p_alpha)) - 1   #Normally the constant inside exponential is 10
                                                               #but I have feel it should be 1 since I believe first the discriminator should learn better
                                                               #and then the feature extractor should start fooling it. I will train with both and see
            else:
                alpha=1

            alpha_s.append(alpha)
            #alpha = -1
            #alpha = float(epoch/max_epochs)
            optimizer.zero_grad()

            x, y1, y2, y3 = train_batch
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            y3 = y3.squeeze(1)

            print(f"Shape of input :{x.size()}")

            _, site_output, x_hat_2 = model(x, y3, alpha)

            if config.linkPred:
                x_hat_2_bce = x_hat_2.clone()

            if config.scaler_output == 1 and config.constraint == True:
                x_hat_2 = torch.sigmoid(x_hat_2)
            elif config.scaler_output == 2 and config.constraint == True:
                x_hat_2 = 9.21 * torch.sigmoid(x_hat_2)

            x_hat_2 = x_hat_2.squeeze()

            if config.symmetric:
                x_hat = 0.5*(torch.permute(x_hat_2,(0,2,1))+x_hat_2)
            else:
                x_hat = x_hat_2

            # compute the loss
            y2_target = y2.squeeze()

            link_pred_loss = None

            if config.linkPred:
                x_hat_2_bce = x_hat_2_bce.squeeze()
                if config.symmetric:
                    x_hat_bce = 0.5 * (torch.permute(x_hat_2_bce, (0, 2, 1)) + x_hat_2_bce)
                else:
                    x_hat_bce = x_hat_2_bce
                y2_target_binary = y2_target.clone()
                y2_target_binary = (y2_target_binary > 0) * 1.0
                link_pred_loss = loss_edge(x_hat_bce, y2_target_binary)

            if config.scaler_output == 1:
                y2_target = y2_target / config.scaler_output_param
            elif config.scaler_output == 2:
                y2_target = torch.log1p(y2_target)

            if config.weighted_ce:
                site_pred_loss = custom_cross_entropy_loss(site_output, y1, cost_matrix.to(device))
            else:
                site_pred_loss = loss_site(site_output, y1)

            y2_target_dict = {}
            x_hat_dict = {}
            for site_x in torch.unique(y1):
                y2_target_dict[site_x.item()] = y2_target[y1==site_x]
                x_hat_dict[site_x.item()] = x_hat[y1==site_x]

            if config.weight:
                weight_dict = {}
                weight_dict[0] = config.weight_param0
                weight_dict[1] = config.weight_param1
                weight_dict[2] = config.weight_param2
                weight_dict[3] = config.weight_param3
                reconstruction_loss_total = []
                for site_x in torch.unique(y1):
                    reconstruction_loss_site = torch.abs(y2_target_dict[site_x.item()] - x_hat_dict[site_x.item()])
                    reconstruction_loss_non_zero = torch.sum(torch.where(y2_target_dict[site_x.item()] != 0, reconstruction_loss_site * weight_dict[site_x.item()], 0))
                    reconstruction_loss_zero = torch.sum(torch.where(y2_target_dict[site_x.item()] == 0, reconstruction_loss_site, 0))
                    non_zero_elements = torch.sum(torch.where(y2_target_dict[site_x.item()] != 0, 1, 0))
                    zero_elements = torch.sum(torch.where(y2_target_dict[site_x.item()] == 0, 1, 0))
                    reconstruction_loss = (reconstruction_loss_non_zero / (non_zero_elements+1e-12)) + (reconstruction_loss_zero / (zero_elements+1e-12)) #TODO check if this logic is correct
                    reconstruction_loss = reconstruction_loss / 2
                    reconstruction_loss_total.append(reconstruction_loss)
                reconstruction_loss_total = torch.mean(torch.stack(reconstruction_loss_total))
            else:
                reconstruction_loss_total = loss_task(y2_target, x_hat)

            loss = site_pred_loss + reconstruction_loss_total
            print(f"Site prediction loss : {site_pred_loss}")
            print(f"Reconstruction loss : {reconstruction_loss_total}")

            if config.correlation:
                correlation_loss = []
                for j in range(y2_target.shape[0]):
                    correlation_loss.append(torch.abs(1-pearson_correlation(torch.flatten(y2_target[j]), torch.flatten(x_hat[j]))))
                correlation_loss = torch.mean(torch.stack(correlation_loss))
                loss += (config.corr_param*correlation_loss)

            if config.siamese:
                siamese_loss = []
                for j in range(y2_target.shape[0]):
                    for k in range(j + 1, y2_target.shape[0]):
                        siamese_loss.append(torch.abs(
                            pearson_correlation(torch.flatten(y2_target[j]), torch.flatten(y2_target[k]))
                            - pearson_correlation(torch.flatten(x_hat[j]), torch.flatten(x_hat[k]))))
                        siamese_loss.append(torch.abs(torch.mean(torch.abs(y2_target[j] - y2_target[k]))
                                                                - torch.mean(torch.abs(x_hat[j] - x_hat[k]))))
                siamese_loss = torch.mean(torch.stack(siamese_loss))
                loss += (config.sia_param*siamese_loss)

            if config.nodStr:
                nodal_strength_loss_x = nodal_strength_loss(y2_target, x_hat)
                loss += (config.nodStr_param * nodal_strength_loss_x)

            if config.klDiv:
                kl_divergence_loss_x = kl_divergence_nodal_strength(y2_target, x_hat)
                loss += (config.klDiv_param * kl_divergence_loss_x)

            if config.eigenValue:
                eigenvalue_loss_x = eigenvalue_difference_batch(y2_target, x_hat)
                loss += (config.eigenValue_param * eigenvalue_loss_x)

            if link_pred_loss is not None:
                link_pred_weight = y2_target.mean() / config.linkPred_param
                print(f"link prediction weight : {link_pred_weight}")
                loss += (link_pred_weight * link_pred_loss)
                print(f"Link prediction loss : {link_pred_loss}")

            # TODO add regularization - later, mmd loss - later, something coral alignment loss, GAN part later - later

            # optimizer step
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))
        train_loss_list.append(loss_data)
        alpha_list.extend(alpha_s)

        scheduler.step(loss_data)

        print(f"Encoder learning rate {optimizer.param_groups[0]['lr']}")
        print(f"Domain classifier learning rate {optimizer.param_groups[19]['lr']}")

        if ((epoch + 1) % 5 == 0):
            # evaluate the model on the validation set
            # compute the mae, accuracy and entropy
            mae, accuracy, entropy, mae_all, accuracy_all_site = evaluate(model, device, val_dataloader, config)
            print("Mean absolute error : {}".format(mae))
            print("Mean accuracy : {}".format(accuracy))
            print("Entropy : {}".format(entropy))

            mae_list.append(mae)
            accuracy_list.append(accuracy)
            entropy_list.append(entropy)
            epoch_list.append(epoch+1)

            if best_mae is None:
                best_mae = mae_all[0]
                best_epoch = epoch + 1
            elif mae_all[0] < best_mae:
                best_mae = mae_all[0]
                best_epoch = epoch + 1

            if accuracy_all_site[0][0]>0.9 and accuracy_all_site[0][1]>0.9:

                results[epoch+1] = {}
                results[epoch+1]['accuracy_site'] = accuracy
                results[epoch+1]['entropy_site'] = entropy
                results[epoch+1]['mae'] = mae_all
                results[epoch+1]['accuracy'] = accuracy_all_site

                #TODO Save these results and also print the best epoch like in jupyter notebook


            state = {"epoch": epoch,
                     "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(),
                     "best_epoch": best_epoch,
                     "best_mae" : best_mae,
                     "epoch_list": epoch_list,
                     "mae_list": mae_list,
                     "accuracy_list": accuracy_list,
                     "entropy_list": entropy_list,
                     "train_loss_list": train_loss_list,
                     "alpha_list": alpha_list,
                     "results": results
                     }
            if accuracy_all_site[0].mean() >= 0.92:
                model_name_1 = "train_model_epoch_{}.pth".format(epoch)
                model_path_1 = os.path.join(save_path, model_name_1)
                torch.save(state, model_path_1)

            model_name_2 = "train_model_epoch_latest.pth"
            model_path_2 = os.path.join(save_path, model_name_2)
            torch.save(state, model_path_2)

            save_file(np.array(mae_list), os.path.join(save_path, "mae_list"))
            save_file(np.array(accuracy_list), os.path.join(save_path, "accuracy_list"))
            save_file(np.array(entropy_list), os.path.join(save_path, "entropy_list"))
            save_file(np.array(train_loss_list), os.path.join(save_path, "train_loss_list"))
            save_file(np.array(alpha_list), os.path.join(save_path, "alpha_list"))
            with open(os.path.join(save_path, "results_all.pkl"), "wb") as f:
                pickle.dump(results, f)

    print("Epoch where the validation mean absolute error for translating from site 0 to site 3 was lowest: {}".format(
        best_epoch))

    return epoch_list, mae_list, accuracy_list, entropy_list, results

def create_dataset(set1, set2, path, config):

    """load the graphs, the matrices and the sites for the subject set"""

    bvalues = [1000, 3000]
    resolutions_paths = ["2_3", "1_25"]

    num_sites = len(bvalues) * len(resolutions_paths)
    conditional_variable_dict = {}  # input code for the corresponding site
    num_vectors_per_site = int(config.conditional_vector_size / num_sites)
    print(f"Number of vectors per site : {num_vectors_per_site}")
    for site_x in range(num_sites):
        z = torch.zeros([1, config.conditional_vector_size], dtype=torch.float)
        z[:,num_vectors_per_site*site_x:num_vectors_per_site*(site_x+1)] = 1
        conditional_variable_dict[site_x] = z

    print(conditional_variable_dict)

    input_tensors = []
    site_outputs = []
    target_tensors = []
    conditional_tensors = []

    nroi = None

    if config.mixed == True and set2 == "train":
        print("Mixed training with independent data points")
        path_subjects = os.path.join(path, "mixed_train_subjects")  # TODO check if it exists when config.mixed is False
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
            m = 0
            for j, bval in enumerate(bvalues):
                for k, res in enumerate(resolutions_paths):
                    mixed_subjects_dict[str(bval) + "_" + res] = subject_IDs[IDs_start[m]:IDs_end[m]]
                    pd.DataFrame(list(mixed_subjects_dict[str(bval) + "_" + res])).to_csv(os.path.join(path_subjects, str(bval) + "_" + res + "_subjects.csv"), index=False, header=False)
                    m += 1

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
        print(f"Mixed_{config.mixed}_set_{set2}")
        subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])
        graphs_dict_output, subject_IDs_valid_output = load_graphs_dict(subject_IDs,
                                                                        os.path.join(data_path, "ds_HCP_bval_3000_anat_0_7_dwi_1_25"),
                                                                        config.scale, 0)

    track_subjects_IDs = []

    for j, bval in enumerate(bvalues):
        for k, res in enumerate(resolutions_paths):
            # loading the .gpickle files for the chosen subjects
            if config.mixed == True and set2 == "train":
                print("Mixed training with independent data points")
                graphs_dict_input, subject_IDs_valid_input = load_graphs_dict(mixed_subjects_dict[str(bval) + "_" + res],
                                                                  os.path.join(data_path, "ds_HCP_bval_" + str(bval) + "_anat_0_7_dwi_" + res),
                                                                  config.scale, 0)
                graphs_dict_output, subject_IDs_valid_output = graphs_dict_input, subject_IDs_valid_input
            else:
                print(f"Mixed_{config.mixed}_set_{set2}")
                graphs_dict_input, subject_IDs_valid_input = load_graphs_dict(subject_IDs,
                                                                  os.path.join(data_path, "ds_HCP_bval_" + str(bval) + "_anat_0_7_dwi_" + res),
                                                                  config.scale, 0)

            subject_IDs_valid_common = np.intersect1d(np.array(subject_IDs_valid_input),
                                                      np.array(subject_IDs_valid_output))

            print(f"Length of common valid subject IDs : {len(subject_IDs_valid_common)}")

            # getting the matrices for edge attribute "normalized_fiber_density"
            matrices_number_of_fibers_input, vectors_number_of_fibers_input = graph_matrix_vector(subject_IDs_valid_common, graphs_dict_input,
                                                                      weight_value='number_of_fibers')
            matrices_number_of_fibers_input1 = matrices_number_of_fibers_input.copy()
            matrices_number_of_fibers_output, _ = graph_matrix_vector(subject_IDs_valid_common, graphs_dict_output,
                                                                      weight_value='number_of_fibers')

            if nroi is None:
                nroi = matrices_number_of_fibers_input[subject_IDs_valid_common[0]].shape[-1]
            for i, subject in enumerate(subject_IDs_valid_common):
                x = torch.tensor(matrices_number_of_fibers_input1[subject], dtype=torch.float)
                x = x.unsqueeze(0)
                y1 = torch.tensor(j * 2 + k, dtype=torch.long)
                y2 = torch.tensor(matrices_number_of_fibers_output[subject], dtype=torch.float)
                y2 = y2.unsqueeze(0)
                if config.mixed == True and set2 == "train":
                    y3 = conditional_variable_dict[y1.item()]
                else:
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

            if config.augment == True and config.mixed == True and set2 == "train":
                augmented_matrices = []

                print(
                    f"Inside the loop for augmented matrices, number of subjects from which pairs are formed : {len(subject_IDs_valid_common)}")

                # Step 1: Generate unique random pairs of subjects as given by config.num_pairs
                pairs = set()
                while len(pairs) < config.num_pairs:
                    subject1, subject2 = random.sample(list(subject_IDs_valid_common), 2)
                    pair = tuple(sorted((subject1, subject2)))  # Store pairs in sorted order to avoid duplicates
                    pairs.add(pair)

                for subject1, subject2 in pairs:
                    # Step 2a: Create the combined matrix
                    if config.symmetric_augment:
                        vector1 = vectors_number_of_fibers_input[subject1]
                        vector2 = vectors_number_of_fibers_input[subject2]
                        if config.linear_augment:
                            combined_vector = combine_vectors_linear(vector1, vector2, config.linear_augment_lambda)
                        else:
                            if config.random_augment:
                                combined_vector = combine_vectors(vector1, vector2)
                            else:
                                combined_vector = combine_vectors_equal(vector1, vector2)
                        combined_matrix = vector_to_matrix(combined_vector, nroi)
                    else:
                        matrix1 = matrices_number_of_fibers_input[subject1]
                        matrix2 = matrices_number_of_fibers_input[subject2]
                        if config.linear_augment:
                            combined_matrix = combine_matrices_linear(matrix1, matrix2, config.linear_augment_lambda)
                        else:
                            if config.random_augment:
                                combined_matrix = combine_matrices(matrix1, matrix2)
                            else:
                                combined_matrix = combine_matrices_equal(matrix1, matrix2)

                    combined_matrix_copy1 = combined_matrix.copy()
                    np.fill_diagonal(combined_matrix_copy1, 0)

                    combined_matrix_copy2 = combined_matrix.copy()
                    np.fill_diagonal(combined_matrix_copy2, 0)

                    x = torch.tensor(combined_matrix_copy1, dtype=torch.float)
                    x = x.unsqueeze(0)
                    y1 = torch.tensor(j * 2 + k, dtype=torch.long)
                    y2 = torch.tensor(combined_matrix_copy2, dtype=torch.float)
                    y2 = y2.unsqueeze(0)
                    if config.mixed == True and set2 == "train":
                        y3 = conditional_variable_dict[y1.item()]
                    else:
                        y3 = conditional_variable_dict[3]
                    # if config.scaler_input is 0, means no transformation, if 1 then normalizing the whole weights by a given
                    # maximum value and if 2 then log transforming it
                    if config.scaler_input == 1:
                        x = x / config.scaler_input_param
                    elif config.scaler_input == 2:
                        x = torch.log1p(x)
                    # print(edge_attr.dtype)
                    input_tensors.append(x)
                    site_outputs.append(y1)
                    target_tensors.append(y2)
                    conditional_tensors.append(y3)

                    track_subjects_IDs.append((subject1, subject2))

                    augmented_matrices.append(combined_matrix_copy2)

                augmented_matrices = np.stack(augmented_matrices)
                np.save(os.path.join(path_subjects, str(bval) + "_" + res + "_augmented_matrices.npy"), augmented_matrices)

    print(f"No of subjects in {set2} is {len(track_subjects_IDs)}")
    print(f"No of unique subjects in {set2} is {len(set(track_subjects_IDs))}")

    image_dataset = ImageDataset(input_tensors, site_outputs, target_tensors, conditional_tensors)

    return image_dataset

def evaluate(model, device, dataloader, config):

    "model evaluation"

    score_mae_average = []

    label_all_site = []
    output_all_site = []
    y_all = {}
    x_hat_all = {}
    accuracy_all_site = {}

    model.eval()

    # Disable gradient calculations
    with torch.no_grad():

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
            x_hat = x_hat * (1 - torch.eye(nroi, nroi).repeat(len(x), 1, 1)).to(device) # TODO - check this logic again

            if config.scaler_output == 1:
                x_hat = x_hat * config.scaler_output_param
            elif config.scaler_output == 2:
                x_hat = torch.special.expm1(x_hat)

            x_hat = torch.round(torch.where(x_hat<0,0,x_hat))
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
                y_all[j].extend(y2_reshape[y1==j])
                x_hat_all[j].extend(x_hat[y1==j])

    mae_all = {}

    for i in np.unique(label_all_site):
        print("Site number", i)
        difference_matrix = np.zeros((len(y_all[i]), len(y_all[i])))
        for j in range(len(y_all[i])):
            for k in range(len(y_all[i])):
                #diff_y = np.mean(np.abs(y_all[i][j] - y_all[i][k]))
                #diff_x_hat = np.mean(np.abs(x_hat_all[i][j] - x_hat_all[i][k]))
                difference_matrix[j, k] = np.mean(np.abs(y_all[i][j] - x_hat_all[i][k]))
        mae_all[i] = np.mean(np.abs(np.array(y_all[i]) - np.array(x_hat_all[i])))
        print("MAE", mae_all[i])
        true_values = np.arange(0, difference_matrix.shape[0], 1)
        predicted_values_axis_0 = np.argmin(difference_matrix, axis=0)
        predicted_values_axis_1 = np.argmin(difference_matrix, axis=1)
        print(predicted_values_axis_0)
        print(predicted_values_axis_1)
        accuracy_axis_0 = np.sum(true_values == predicted_values_axis_0) / len(true_values)
        accuracy_axis_1 = np.sum(true_values == predicted_values_axis_1) / len(true_values)
        print(accuracy_axis_0, accuracy_axis_1)
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


if __name__ == "__main__":
    main()
