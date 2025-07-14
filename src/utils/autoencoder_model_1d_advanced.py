import torch
import numpy as np
import torch.nn as nn
from .ml_unit_models import MLP_Unit, GradientReversalLayer
from .ml_blocks import MappingNetwork
from .ml_linear_models import MLP_1NormalizedSmall, MLP_2NormalizedMedium, MLP_3NormalizedMedium, MLP_4NormalizedLarge

class VectorToVectorTranslationAE(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck_size,
                 mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, output_size,
                 model_encoder=0, normalization_l_encoder = "batch", activation_l_encoder = "lrelu", p_encoder=0,
                 model_domain=1, normalization_l_domain="batch", activation_l_domain="lrelu", p_domain=0,
                 normalization_l_mapping=None, activation_l_mapping="lrelu",
                 model_decoder=0, normalization_l1_decoder="batch", activation_l1_decoder="lrelu", p_decoder1=0):
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size, bottleneck_size, model_encoder, normalization_l_encoder, activation_l_encoder, p_encoder)
        self.site_classifier = SiteClassifier(bottleneck_size, output_size, model_domain, normalization_l_domain, activation_l_domain, p_domain)
        self.mapping_network = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size,
                                              num_layers_latent, normalization_l_mapping, activation_l_mapping)
        self.decoder = Decoder(bottleneck_size, hidden_size, input_size, mapping_latent_size, model_decoder,
                                normalization_l1_decoder, activation_l1_decoder, p_decoder1)

    def forward(self, x, alpha=1.0, conditional_vector=None):

        invariant_features = self.encoder(x)
        reversed_features = GradientReversalLayer(alpha)(invariant_features)
        site_output = self.site_classifier(reversed_features)
        latent_vector = self.mapping_network(conditional_vector)
        reconstructed_output = self.decoder(invariant_features, latent_vector)

        return invariant_features, reversed_features, site_output, reconstructed_output

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, bottleneck_size, model_encoder=0, normalization_l="batch", activation_l="lrelu",
                 p_encoder=0):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(MLP_Unit(input_size, hidden_size, normalization_l, activation_l, p_encoder))
        if model_encoder == 1:
            self.layers.append(MLP_Unit(hidden_size, hidden_size, normalization_l, activation_l, p_encoder))

        while (hidden_size >= (2*bottleneck_size)):
            self.layers.append(MLP_Unit(hidden_size, hidden_size // 2, normalization_l, activation_l, p_encoder))
            if model_encoder == 1:
                self.layers.append(MLP_Unit(hidden_size // 2, hidden_size // 2, normalization_l, activation_l, p_encoder))
            hidden_size = hidden_size // 2

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

class SiteClassifier(nn.Module):

    def __init__(self, input_size, output_size, model_domain=1, normalization_l="batch", activation_l="lrelu", p_domain=0):
        super().__init__()

        if model_domain == 0:
            mlp = MLP_1NormalizedSmall
        elif model_domain == 1:
            mlp = MLP_2NormalizedMedium
        elif model_domain == 2:
            mlp = MLP_3NormalizedMedium
        elif model_domain == 3:
            mlp = MLP_4NormalizedLarge

        self.fc1 = mlp(input_size, output_size, normalization_l, activation_l, p_domain)

    def forward(self, x):

        site_output = self.fc1(x)

        return site_output

class Decoder(nn.Module):

    def __init__(self, bottleneck_size, hidden_size, output_size, latent_size, model_decoder=0, normalization_l="batch",
                 activation_l="lrelu", p_decoder=0):
        super().__init__()

        feature_size = bottleneck_size + latent_size

        self.layers = nn.ModuleList()

        if feature_size < output_size:
            self.layers.append(MLP_Unit(feature_size, feature_size, normalization_l, activation_l, p_decoder))

        pow_2 = np.log2(feature_size)
        pow_2_int = np.ceil(pow_2)
        if pow_2 != pow_2_int:
            feature_size1 = int(np.power(2, pow_2_int))
            if (2*feature_size1) < output_size :
                self.layers.append(MLP_Unit(feature_size, feature_size1, normalization_l, activation_l, p_decoder))
                if model_decoder == 1:
                    self.layers.append(MLP_Unit(feature_size1, feature_size1, normalization_l, activation_l, p_decoder))
                feature_size = feature_size1

        while ((feature_size)*2 <= hidden_size):
            self.layers.append(MLP_Unit(feature_size, feature_size * 2, normalization_l, activation_l, p_decoder))
            if model_decoder==1:
                self.layers.append(MLP_Unit(feature_size*2, feature_size*2, normalization_l, activation_l, p_decoder))
            feature_size = feature_size * 2

        self.layers.append(nn.Linear(feature_size, output_size))

    def forward(self, x, latent_vector):

        x = torch.cat((x, latent_vector), dim=-1)

        for layer in self.layers:
            x = layer(x)

        return x

