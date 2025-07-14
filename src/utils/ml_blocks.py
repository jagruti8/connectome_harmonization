import torch
import torch.nn as nn
import torch_geometric.nn as graphnn
from .ml_unit_models import MLP_Unit, GCN_Unit
from .ml_linear_models import MLP_1NormalizedSmall, MLP_2NormalizedMedium, MLP_3NormalizedMedium, MLP_4NormalizedLarge

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, model_encoder=0, base_layer="cheb", normalization_g="batch", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=2, concat_gat=False, dropout_gat=0.2):
        super().__init__()

        if model_encoder == 0:
            hidden_size = hidden_size
        elif model_encoder == 1 or model_encoder == 2:
            hidden_size = hidden_size * 2

        self.layers = nn.ModuleList()

        # if hidden_size=256, 274->256, 256->128, 128->64
        while (hidden_size >= 64):
            self.layers.append(GCN_Unit(input_size, hidden_size, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat))
            if model_encoder == 2 and hidden_size <= 256 and hidden_size > 64:
                self.layers.append(GCN_Unit(hidden_size, hidden_size, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat))
            input_size = hidden_size
            hidden_size = hidden_size // 2

        self.layers.append(GCN_Unit(hidden_size * 2, hidden_size * 2, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat))  # 64->64

    def forward(self, x, edge_index, edge_attr, nroi, batch_idx=None):

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, nroi, batch_idx)

        return x

class SiteClassifierLinear(nn.Module):

    def __init__(self, input_size, output_size, model_domain=1, normalization_l="batch", activation_l="lrelu", p=0):
        super().__init__()

        if model_domain == 0:
            mlp = MLP_1NormalizedSmall
        elif model_domain == 1:
            mlp = MLP_2NormalizedMedium
        elif model_domain == 2:
            mlp = MLP_3NormalizedMedium
        elif model_domain == 3:
            mlp = MLP_4NormalizedLarge

        self.fc1 = mlp(input_size*2, output_size, normalization_l, activation_l, p)

    def forward(self, x, batch_idx=None):

        x1 = graphnn.global_max_pool(x, batch_idx)
        x2 = graphnn.global_mean_pool(x, batch_idx)

        xx = torch.cat((x1, x2), dim=1)

        site_output = self.fc1(xx)

        return site_output

class SiteClassifierGraph(nn.Module):
    def __init__(self, input_size, output_size, model_domain_graph=0, model_domain=1, base_layer="cheb",
                 normalization_g="batch", normalization_l="batch", activation_g="lrelu", activation_l="lrelu",
                 instance_track=False, filter_size=2, heads=2, concat_gat=False, dropout_gat=0.2, p=0):
        super().__init__()

        if model_domain_graph == 0:
            hidden_size = input_size
        elif model_domain_graph == 1:
            hidden_size = input_size * 2

        self.layers = nn.ModuleList()

        # if hidden_size=64, 64->64, 64->32
        while (hidden_size >= 32):
            self.layers.append(GCN_Unit(input_size, hidden_size, base_layer, normalization_g, activation_g,
                                        instance_track, filter_size, heads, concat_gat, dropout_gat))
            input_size = hidden_size
            hidden_size = hidden_size // 2

        if model_domain == 0:
            mlp = MLP_1NormalizedSmall
        elif model_domain == 1:
            mlp = MLP_2NormalizedMedium
        elif model_domain == 2:
            mlp = MLP_3NormalizedMedium
        elif model_domain == 3:
            mlp = MLP_4NormalizedLarge

        self.fc1 = mlp(hidden_size * 4, output_size, normalization_l, activation_l, p)

    def forward(self, x, edge_index, edge_attr, nroi, batch_idx=None):

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, nroi, batch_idx)

        x1 = graphnn.global_max_pool(x, batch_idx)
        x2 = graphnn.global_mean_pool(x, batch_idx)

        xx = torch.cat((x1, x2), dim=1)

        site_output = self.fc1(xx)

        return site_output

class MappingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers, normalization_l=None, activation_l="lrelu"):
        super().__init__()

        # List to store the fully connected layers
        layers = []

        # Input layer: first layer from latent_size to hidden_size
        layers.append(MLP_Unit(input_size, hidden_size, normalization_l, activation_l))

        # Hidden layers: hidden_size -> hidden_size, repeated num_layers times
        for _ in range(num_layers - 2):
            layers.append(MLP_Unit(hidden_size, hidden_size, normalization_l, activation_l))

        # Output layer: from hidden_size back to latent_size to ensure same output dimension
        layers.append(MLP_Unit(hidden_size, latent_size, normalization_l, activation_l))

        # Combine layers into a sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        z = self.network(z)
        return z


