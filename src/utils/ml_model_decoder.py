import torch
import torch.nn as nn
from .ml_unit_models import GCN_Unit, MLP_Graph_Unit

# Decoder0 : Model with adaptive instance normalization
class Decoder0(nn.Module):

    def __init__(self, input_size, output_size, style_dim, base_layer="cheb", normalization_l="instance", normalization_g="batch",
                 activation_l="lrelu", activation_g="lrelu", instance_track=False, filter_size=2, heads=4, concat_gat=True,
                 dropout_gat=0.3, p_decoder=0):
        super().__init__()

        self.fc1 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc2 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc3 = MLP_Graph_Unit(input_size * 2, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc4 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(input_size * 4, input_size * 4, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(input_size * 4 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(input_size * 4, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)

    def forward(self, x, style_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):
        x = self.fc1(x, nroi, batch_idx, style_vector)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.fc4(x, nroi, batch_idx, style_vector)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder1 : Model with adaptive instance normalization even for the graph convolution layers
class Decoder1(nn.Module):

    def __init__(self, input_size, output_size, style_dim, base_layer="cheb", normalization_l="instance", normalization_g="instance",
                 activation_l="lrelu", activation_g="lrelu", instance_track=False, filter_size=2, heads=4, concat_gat=True,
                 dropout_gat=0.3, p_decoder=0):
        super().__init__()

        self.fc1 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc2 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc3 = MLP_Graph_Unit(input_size * 2, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc4 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(input_size * 4, input_size * 4, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat, style_dim)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(input_size * 4 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0, style_dim)
        else:
            self.graphconvblock2 = GCN_Unit(input_size * 4, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0, style_dim)

    def forward(self, x, style_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):
        x = self.fc1(x, nroi, batch_idx, style_vector)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.fc4(x, nroi, batch_idx, style_vector)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx, style_vector)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx, style_vector)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder2 : Model with adaptive instance normalization and then inner product decoder
class Decoder2(nn.Module):

    def __init__(self, input_size, style_dim, normalization_l="instance", activation_l="lrelu", instance_track=False, p_decoder=0):
        super().__init__()

        self.fc11 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc12 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc13 = MLP_Graph_Unit(input_size*2, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc14 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc15 = MLP_Graph_Unit(input_size * 4, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc16 = MLP_Graph_Unit(input_size * 4, input_size * 8, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc21 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc22 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc23 = MLP_Graph_Unit(input_size * 2, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc24 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc25 = MLP_Graph_Unit(input_size * 4, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc26 = MLP_Graph_Unit(input_size * 4, input_size * 8, normalization_l, activation_l, instance_track, p_decoder, style_dim)

    def forward(self, x, style_vector1, style_vector2, batch_size, nroi, batch_idx=None):
        x1 = self.fc11(x, nroi, batch_idx, style_vector1)
        x1 = self.fc12(x1, nroi, batch_idx, style_vector1)
        x1 = self.fc13(x1, nroi, batch_idx, style_vector1)
        x1 = self.fc14(x1, nroi, batch_idx, style_vector1)
        x1 = self.fc15(x1, nroi, batch_idx, style_vector1)
        x1 = self.fc16(x1, nroi, batch_idx, style_vector1)
        x2 = self.fc21(x, nroi, batch_idx, style_vector2)
        x2 = self.fc22(x2, nroi, batch_idx, style_vector2)
        x2 = self.fc23(x2, nroi, batch_idx, style_vector2)
        x2 = self.fc24(x2, nroi, batch_idx, style_vector2)
        x2 = self.fc25(x2, nroi, batch_idx, style_vector2)
        x2 = self.fc26(x2, nroi, batch_idx, style_vector2)

        # Generate new adjacency matrix based on node embeddings
        x1_reshape = torch.reshape(x1, (batch_size, nroi, x1.size()[1]))
        x2_reshape = torch.reshape(x2, (batch_size, nroi, x2.size()[1]))
        A_matrix = torch.matmul(x1_reshape, torch.permute(x2_reshape, (0, 2, 1)))

        return A_matrix


# Decoder21 : Model with adaptive instance normalization and then inner product decoder but a bit simpler than Decoder2
class Decoder21(nn.Module):
    def __init__(self, input_size, style_dim, normalization_l="instance", activation_l="lrelu", instance_track=False,
                 p_decoder=0):
        super().__init__()

        self.fc11 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc12 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc13 = MLP_Graph_Unit(input_size * 4, input_size * 8, normalization_l, activation_l, instance_track, p_decoder, style_dim)

        self.fc21 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc22 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc23 = MLP_Graph_Unit(input_size * 4, input_size * 8, normalization_l, activation_l, instance_track,p_decoder, style_dim)

    def forward(self, x, style_vector1, style_vector2, batch_size, nroi, batch_idx=None):

        x1 = self.fc11(x, nroi, batch_idx, style_vector1)
        x1 = self.fc12(x1, nroi, batch_idx, style_vector1)
        x1 = self.fc13(x1, nroi, batch_idx, style_vector1)

        x2 = self.fc21(x, nroi, batch_idx, style_vector2)
        x2 = self.fc22(x2, nroi, batch_idx, style_vector2)
        x2 = self.fc23(x2, nroi, batch_idx, style_vector2)

        # Generate new adjacency matrix based on node embeddings
        x1_reshape = torch.reshape(x1, (batch_size, nroi, x1.size()[1]))
        x2_reshape = torch.reshape(x2, (batch_size, nroi, x2.size()[1]))
        A_matrix = torch.matmul(x1_reshape, torch.permute(x2_reshape, (0, 2, 1)))
        return A_matrix

# Decoder3 : Model with condition appended to the decoder at the latent feature stage
class Decoder3(nn.Module):

    def __init__(self, input_size, output_size, latent_size, base_layer="cheb", normalization_l="batch", normalization_g="batch",
                 activation_l="lrelu", activation_g="lrelu", instance_track=False,
                 filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder=0):
        super().__init__()

        feature_size = input_size+latent_size
        self.fc1 = MLP_Graph_Unit(feature_size, feature_size, normalization_l, activation_l, instance_track, p_decoder)
        self.fc2 = MLP_Graph_Unit(feature_size, feature_size, normalization_l, activation_l, instance_track, p_decoder)
        self.fc3 = MLP_Graph_Unit(feature_size, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder)
        self.fc4 = MLP_Graph_Unit(feature_size * 2, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder)
        self.graphconvblock1 = GCN_Unit(feature_size * 2, feature_size * 2, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(feature_size * 2 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(feature_size * 2, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)

    def forward(self, x, latent_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):

        latent_vector = latent_vector.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector), dim=-1)
        x = self.fc1(x, nroi, batch_idx)
        x = self.fc2(x, nroi, batch_idx)
        x = self.fc3(x, nroi, batch_idx)
        x = self.fc4(x, nroi, batch_idx)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder4 : Model with condition appended to the decoder at the latent feature stage and then inner product decoder
class Decoder4(nn.Module):

    def __init__(self, input_size, latent_size, normalization_l="batch",
                 activation_l="lrelu", instance_track=False, p_decoder=0):
        super().__init__()

        feature_size = input_size+latent_size
        self.fc11 = MLP_Graph_Unit(feature_size, feature_size, normalization_l, activation_l, instance_track, p_decoder)
        self.fc12 = MLP_Graph_Unit(feature_size, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder)
        self.fc13 = MLP_Graph_Unit(feature_size * 2, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder)
        self.fc14 = MLP_Graph_Unit(feature_size * 2, feature_size * 4, normalization_l, activation_l, instance_track, p_decoder)
        self.fc21 = MLP_Graph_Unit(feature_size, feature_size, normalization_l, activation_l, instance_track, p_decoder)
        self.fc22 = MLP_Graph_Unit(feature_size, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder)
        self.fc23 = MLP_Graph_Unit(feature_size * 2, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder)
        self.fc24 = MLP_Graph_Unit(feature_size * 2, feature_size * 4, normalization_l, activation_l, instance_track, p_decoder)

    def forward(self, x, latent_vector1, latent_vector2, batch_size, nroi, batch_idx=None):

        latent_vector1 = latent_vector1.repeat_interleave(nroi, dim=0)
        x1 = torch.cat((x, latent_vector1), dim=-1)
        x1 = self.fc11(x1, nroi, batch_idx)
        x1 = self.fc12(x1, nroi, batch_idx)
        x1 = self.fc13(x1, nroi, batch_idx)
        x1 = self.fc14(x1, nroi, batch_idx)

        latent_vector2 = latent_vector2.repeat_interleave(nroi, dim=0)
        x2 = torch.cat((x, latent_vector2), dim=-1)
        x2 = self.fc21(x2, nroi, batch_idx)
        x2 = self.fc22(x2, nroi, batch_idx)
        x2 = self.fc23(x2, nroi, batch_idx)
        x2 = self.fc24(x2, nroi, batch_idx)

        # Generate new adjacency matrix based on node embeddings
        x1_reshape = torch.reshape(x1, (batch_size, nroi, x1.size()[1]))
        x2_reshape = torch.reshape(x2, (batch_size, nroi, x2.size()[1]))
        A_matrix = torch.matmul(x1_reshape, torch.permute(x2_reshape, (0, 2, 1)))

        return A_matrix

# Decoder5 : Model with both adaptive instance normalization and condition appended to the decoder(64 features+64 = 128 - 128 - AdaIn,
# 128 - 128 - Adain, 128-256-Adain, 256-256- Adain)
class Decoder5(nn.Module):

    def __init__(self, input_size, output_size, latent_size, style_dim, base_layer="cheb",
                 normalization_l="instance", normalization_g="batch", activation_l="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder=0):
        super().__init__()

        feature_size = input_size+latent_size
        self.fc1 = MLP_Graph_Unit(feature_size, feature_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc2 = MLP_Graph_Unit(feature_size, feature_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc3 = MLP_Graph_Unit(feature_size, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc4 = MLP_Graph_Unit(feature_size * 2, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(feature_size * 2, feature_size * 2, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(feature_size * 2 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(feature_size * 2, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)

    def forward(self, x, latent_vector, style_vector, edge_index, edge_attr,  batch_size, nroi, batch_idx=None):

        latent_vector = latent_vector.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector), dim=-1)
        x = self.fc1(x, nroi, batch_idx, style_vector)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.fc4(x, nroi, batch_idx, style_vector)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder6 : Model with both adaptive instance normalization also for graph convolutional layers and condition appended to the decoder(64 features+64 = 128 - 128 - AdaIn,
# 128 - 128 - Adain, 128-256-Adain, 256-256- Adain)
class Decoder6(nn.Module):

    def __init__(self, input_size, output_size, latent_size, style_dim, base_layer="cheb",
                 normalization_l="instance", normalization_g="instance", activation_l="lrelu",  activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder=0):
        super().__init__()

        feature_size = input_size+latent_size
        self.fc1 = MLP_Graph_Unit(feature_size, feature_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc2 = MLP_Graph_Unit(feature_size, feature_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc3 = MLP_Graph_Unit(feature_size, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc4 = MLP_Graph_Unit(feature_size * 2, feature_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(feature_size * 2, feature_size * 2, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat, style_dim)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(feature_size * 2 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0, style_dim)
        else:
            self.graphconvblock2 = GCN_Unit(feature_size * 2, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0, style_dim)

    def forward(self, x, latent_vector, style_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):

        latent_vector = latent_vector.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector), dim=-1)
        x = self.fc1(x, nroi, batch_idx, style_vector)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.fc4(x, nroi, batch_idx, style_vector)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx, style_vector)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx, style_vector)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder7 : Model with both adaptive instance normalization and condition appended to the decoder(64 features - 64 - AdaIn,
# 64 + 64 = 128 - 128 - AdaIn, 128+128 = 256 -256 - AdaIn)
class Decoder7(nn.Module):

    def __init__(self, input_size, output_size, latent_size, style_dim, base_layer="cheb",
                 normalization_l="instance", normalization_g="batch", activation_l="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder=0):
        super().__init__()

        self.fc1 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        feature_size1 = input_size + latent_size
        self.fc2 = MLP_Graph_Unit(feature_size1, feature_size1, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        feature_size2 = feature_size1 + latent_size * 2
        self.fc3 = MLP_Graph_Unit(feature_size2, feature_size2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(feature_size2, feature_size2, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(feature_size2 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(feature_size2, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)

    def forward(self, x, latent_vector1, latent_vector2, style_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):

        x = self.fc1(x, nroi, batch_idx, style_vector)
        latent_vector1 = latent_vector1.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector1), dim=-1)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        latent_vector2 = latent_vector2.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector2), dim=-1)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder8 : Model with both adaptive instance normalization also for graph convolutional layers and condition appended to the decoder(64 features - 64 - AdaIn,
# 64 + 64 = 128 - 128 - AdaIn, 128+128 = 256 -256 - AdaIn)
class Decoder8(nn.Module):

    def __init__(self, input_size, output_size, latent_size, style_dim, base_layer="cheb",
                 normalization_l="instance", normalization_g="instance", activation_l="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder=0):
        super().__init__()

        self.fc1 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        feature_size1 = input_size + latent_size
        self.fc2 = MLP_Graph_Unit(feature_size1, feature_size1, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        feature_size2 = feature_size1 + latent_size * 2
        self.fc3 = MLP_Graph_Unit(feature_size2, feature_size2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(feature_size2, feature_size2, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat, style_dim)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(feature_size2 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0, style_dim)
        else:
            self.graphconvblock2 = GCN_Unit(feature_size2, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0, style_dim)

    def forward(self, x, latent_vector1, latent_vector2, style_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):

        x = self.fc1(x, nroi, batch_idx, style_vector)
        latent_vector1 = latent_vector1.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector1), dim=-1)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        latent_vector2 = latent_vector2.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector2), dim=-1)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx, style_vector)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx, style_vector)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder9 : Model with two units, one for edge prediction and the other for weight prediction
class Decoder9(nn.Module):

    def __init__(self, input_size, output_size, style_dim1, style_dim2, model_edgePred = 0, base_layer="cheb",
                 normalization_l1="instance", normalization_l2="instance", normalization_g="batch",
                 activation_l1="lrelu", activation_l2="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder1=0, p_decoder2=0):
        super().__init__()

        if model_edgePred == 0:
            self.edge_prediction = Decoder21(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)
        elif model_edgePred == 1:
            self.edge_prediction = Decoder2(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)

        self.weight_prediction = Decoder0(input_size, output_size, style_dim1, base_layer, normalization_l1, normalization_g,
                 activation_l1, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat, p_decoder1)

    def forward(self, x, style_vector11, style_vector12, style_vector2, batch_size, nroi, batch_idx=None, matrix_threshold=0.5):

        A_matrix = self.edge_prediction(x, style_vector11, style_vector12, batch_size, nroi, batch_idx)
        A_out = torch.sigmoid(A_matrix)
        A_out1 = (A_out > matrix_threshold) * 1.0

        # Convert A_out to edge index and edge weights
        offset_g, row_g, col_g = (A_out > matrix_threshold).nonzero().t()
        edge_attr_g = A_out1[offset_g, row_g, col_g]
        edge_attr_g = torch.unsqueeze(edge_attr_g, dim=1)
        row_g = row_g + offset_g * nroi
        col_g = col_g + offset_g * nroi
        edge_index_g = torch.stack([row_g, col_g], dim=0)

        x_reshape = self.weight_prediction(x, style_vector2, edge_index_g, edge_attr_g, batch_size, nroi, batch_idx)

        return A_matrix, x_reshape

# Decoder10 : Model with two units, one for edge prediction and the other for weight prediction and adaptive instance normalization even for the graph convolution layers
class Decoder10(nn.Module):

    def __init__(self, input_size, output_size, style_dim1, style_dim2, model_edgePred = 0, base_layer="cheb",
                 normalization_l1="instance", normalization_l2="instance", normalization_g="instance",
                 activation_l1="lrelu", activation_l2="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder1=0, p_decoder2=0):
        super().__init__()

        if model_edgePred == 0:
            self.edge_prediction = Decoder21(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)
        elif model_edgePred == 1:
            self.edge_prediction = Decoder2(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)

        self.weight_prediction = Decoder1(input_size, output_size, style_dim1, base_layer, normalization_l1, normalization_g,
                 activation_l1, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat, p_decoder1)

    def forward(self, x, style_vector11, style_vector12, style_vector2, batch_size, nroi, batch_idx=None, matrix_threshold=0.5):

        A_matrix = self.edge_prediction(x, style_vector11, style_vector12, batch_size, nroi, batch_idx)
        A_out = torch.sigmoid(A_matrix)
        A_out1 = (A_out > matrix_threshold) * 1.0

        # Convert A_out to edge index and edge weights
        offset_g, row_g, col_g = (A_out > matrix_threshold).nonzero().t()
        edge_attr_g = A_out1[offset_g, row_g, col_g]
        edge_attr_g = torch.unsqueeze(edge_attr_g, dim=1)
        row_g = row_g + offset_g * nroi
        col_g = col_g + offset_g * nroi
        edge_index_g = torch.stack([row_g, col_g], dim=0)

        x_reshape = self.weight_prediction(x, style_vector2, edge_index_g, edge_attr_g, batch_size, nroi, batch_idx)

        return A_matrix, x_reshape

# Decoder11 : Model with condition appended to the decoder(64 + 64 = 128 - 128 + 128 = 256 - 256,
# 64 + 64 = 128 - 128 - AdaIn, 128+128 = 256 -256 - AdaIn)
class Decoder11(nn.Module):

    def __init__(self, input_size, output_size, latent_size, base_layer="cheb",
                 normalization_l="batch", normalization_g="batch", activation_l="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder=0):
        super().__init__()

        feature_size1 = input_size + latent_size
        self.fc1 = MLP_Graph_Unit(feature_size1, feature_size1, normalization_l, activation_l, instance_track, p_decoder)
        self.fc2 = MLP_Graph_Unit(feature_size1, feature_size1, normalization_l, activation_l, instance_track, p_decoder)
        feature_size2 = feature_size1 + latent_size * 2
        self.fc3 = MLP_Graph_Unit(feature_size2, feature_size2, normalization_l, activation_l, instance_track, p_decoder)
        self.fc4 = MLP_Graph_Unit(feature_size2, feature_size2, normalization_l, activation_l, instance_track, p_decoder)
        self.graphconvblock1 = GCN_Unit(feature_size2, feature_size2, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(feature_size2 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(feature_size2, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)

    def forward(self, x, latent_vector1, latent_vector2, edge_index, edge_attr, batch_size, nroi, batch_idx=None):

        latent_vector1 = latent_vector1.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector1), dim=-1)
        x = self.fc1(x, nroi, batch_idx)
        x = self.fc2(x, nroi, batch_idx)
        latent_vector2 = latent_vector2.repeat_interleave(nroi, dim=0)
        x = torch.cat((x, latent_vector2), dim=-1)
        x = self.fc3(x, nroi, batch_idx)
        x = self.fc4(x, nroi, batch_idx)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder12 : Model with two units, one for edge prediction and the other for weight prediction, but instead of binary, weighted matrices are given
# for decoder graph convolutional layers
class Decoder12(nn.Module):

    def __init__(self, input_size, output_size, style_dim1, style_dim2, model_edgePred = 0, base_layer="cheb",
                 normalization_l1="instance", normalization_l2="instance", normalization_g="batch",
                 activation_l1="lrelu", activation_l2="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder1=0, p_decoder2=0):
        super().__init__()

        if model_edgePred == 0:
            self.edge_prediction = Decoder21(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)
        elif model_edgePred == 1:
            self.edge_prediction = Decoder2(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)

        self.weight_prediction = Decoder0(input_size, output_size, style_dim1, base_layer, normalization_l1, normalization_g,
                 activation_l1, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat, p_decoder1)

    def forward(self, x, style_vector11, style_vector12, style_vector2, batch_size, nroi, batch_idx=None, matrix_threshold=0.5):

        A_matrix = self.edge_prediction(x, style_vector11, style_vector12, batch_size, nroi, batch_idx)
        A_out = torch.sigmoid(A_matrix)

        # Convert A_out to edge index and edge weights
        offset_g, row_g, col_g = (A_out > matrix_threshold).nonzero().t()
        edge_attr_g = A_matrix[offset_g, row_g, col_g]
        edge_attr_g = torch.unsqueeze(edge_attr_g, dim=1)
        row_g = row_g + offset_g * nroi
        col_g = col_g + offset_g * nroi
        edge_index_g = torch.stack([row_g, col_g], dim=0)

        x_reshape = self.weight_prediction(x, style_vector2, edge_index_g, edge_attr_g, batch_size, nroi, batch_idx)

        return A_matrix, x_reshape

# Decoder13 : Model with two units, one for edge prediction and the other for weight prediction and adaptive instance normalization even for the graph convolution layers
# but instead of binary, weighted matrices are given for decoder graph convolutional layers
class Decoder13(nn.Module):

    def __init__(self, input_size, output_size, style_dim1, style_dim2, model_edgePred = 0, base_layer="cheb",
                 normalization_l1="instance", normalization_l2="instance", normalization_g="instance",
                 activation_l1="lrelu", activation_l2="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder1=0, p_decoder2=0):
        super().__init__()

        if model_edgePred == 0:
            self.edge_prediction = Decoder21(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)
        elif model_edgePred == 1:
            self.edge_prediction = Decoder2(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)

        self.weight_prediction = Decoder1(input_size, output_size, style_dim1, base_layer, normalization_l1, normalization_g,
                 activation_l1, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat, p_decoder1)

    def forward(self, x, style_vector11, style_vector12, style_vector2, batch_size, nroi, batch_idx=None, matrix_threshold=0.5):

        A_matrix = self.edge_prediction(x, style_vector11, style_vector12, batch_size, nroi, batch_idx)
        A_out = torch.sigmoid(A_matrix)

        # Convert A_out to edge index and edge weights
        offset_g, row_g, col_g = (A_out > matrix_threshold).nonzero().t()
        edge_attr_g = A_matrix[offset_g, row_g, col_g]
        edge_attr_g = torch.unsqueeze(edge_attr_g, dim=1)
        row_g = row_g + offset_g * nroi
        col_g = col_g + offset_g * nroi
        edge_index_g = torch.stack([row_g, col_g], dim=0)

        x_reshape = self.weight_prediction(x, style_vector2, edge_index_g, edge_attr_g, batch_size, nroi, batch_idx)

        return A_matrix, x_reshape

# Decoder14 : Model with adaptive instance normalization and using mean binary indices but adaptive edge weights
class Decoder14(nn.Module):

    def __init__(self, input_size, output_size, style_dim, base_layer="cheb", normalization_l="instance", normalization_g="batch",
                 activation_l="lrelu", activation_g="lrelu", instance_track=False, filter_size=2, heads=4, concat_gat=True,
                 dropout_gat=0.3, p_decoder=0):
        super().__init__()

        self.fc1 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc2 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc3 = MLP_Graph_Unit(input_size * 2, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc4 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(input_size * 4, input_size * 4, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(input_size * 4 * heads, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(input_size * 4, output_size, base_layer, normalization_g, activation_g, instance_track, filter_size, 1, False, 0.0)

    def forward(self, x, style_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):
        x = self.fc1(x, nroi, batch_idx, style_vector)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.fc4(x, nroi, batch_idx, style_vector)

        # Generate new adjacency matrix based on node embeddings
        x_reshape1 = torch.reshape(x, (batch_size, nroi, x.size()[1]))
        A_matrix = torch.matmul(x_reshape1, torch.permute(x_reshape1, (0, 2, 1)))

        # Extract the edge weights corresponding to the mean edge indices
        row_g, col_g = edge_index
        composite_matrix = torch.block_diag(*A_matrix)
        edge_attr_g = composite_matrix[row_g, col_g]
        edge_attr_g = torch.unsqueeze(edge_attr_g, dim=1)

        x = self.graphconvblock1(x, edge_index, edge_attr_g, nroi, batch_idx)
        x = self.graphconvblock2(x, edge_index, edge_attr_g, nroi, batch_idx)
        x_reshape2 = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape2

# Decoder15 : Model with adaptive instance normalization and last layer has only graph convolution
class Decoder15(nn.Module):

    def __init__(self, input_size, output_size, style_dim, base_layer="cheb", normalization_l="instance", normalization_g="batch",
                 activation_l="lrelu", activation_g="lrelu", instance_track=False, filter_size=2, heads=4, concat_gat=True,
                 dropout_gat=0.3, p_decoder=0):
        super().__init__()

        self.fc1 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc2 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc3 = MLP_Graph_Unit(input_size * 2, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc4 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(input_size * 4, input_size * 4, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(input_size * 4 * heads, output_size, base_layer, None, None, instance_track, filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(input_size * 4, output_size, base_layer, None, None, instance_track, filter_size, 1, False, 0.0)

    def forward(self, x, style_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):
        x = self.fc1(x, nroi, batch_idx, style_vector)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.fc4(x, nroi, batch_idx, style_vector)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape

# Decoder16 : Model with adaptive instance normalization even for the graph convolution layers and last layer has only graph convolution
class Decoder16(nn.Module):

    def __init__(self, input_size, output_size, style_dim, base_layer="cheb", normalization_l="instance", normalization_g="instance",
                 activation_l="lrelu", activation_g="lrelu", instance_track=False, filter_size=2, heads=4, concat_gat=True,
                 dropout_gat=0.3, p_decoder=0):
        super().__init__()

        self.fc1 = MLP_Graph_Unit(input_size, input_size, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc2 = MLP_Graph_Unit(input_size, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc3 = MLP_Graph_Unit(input_size * 2, input_size * 2, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.fc4 = MLP_Graph_Unit(input_size * 2, input_size * 4, normalization_l, activation_l, instance_track, p_decoder, style_dim)
        self.graphconvblock1 = GCN_Unit(input_size * 4, input_size * 4, base_layer, normalization_g, activation_g, instance_track, filter_size, heads, concat_gat, dropout_gat, style_dim)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(input_size * 4 * heads, output_size, base_layer, None, None, instance_track, filter_size, 1, False, 0.0, style_dim)
        else:
            self.graphconvblock2 = GCN_Unit(input_size * 4, output_size, base_layer, None, None, instance_track, filter_size, 1, False, 0.0, style_dim)

    def forward(self, x, style_vector, edge_index, edge_attr, batch_size, nroi, batch_idx=None):
        x = self.fc1(x, nroi, batch_idx, style_vector)
        x = self.fc2(x, nroi, batch_idx, style_vector)
        x = self.fc3(x, nroi, batch_idx, style_vector)
        x = self.fc4(x, nroi, batch_idx, style_vector)
        x = self.graphconvblock1(x, edge_index, edge_attr, nroi, batch_idx, style_vector)
        x = self.graphconvblock2(x, edge_index, edge_attr, nroi, batch_idx, style_vector)
        x_reshape = torch.reshape(x, (batch_size, nroi, x.size()[1]))

        return x_reshape


# Decoder17 : Model where the adjacency matrix is first predicted using inner product decoder and then this matrix
# is used a feature matrix for two layers of graph neural networks and the last layer is only a graph neural network with no normalization or activation
class Decoder17(nn.Module):

    def __init__(self, input_size, output_size, style_dim1, base_layer="cheb",
                 normalization_l1="instance", normalization_g="batch", activation_l1="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder1=0):
        super().__init__()

        self.weight_prediction = Decoder2(input_size, style_dim1, normalization_l1, activation_l1, instance_track, p_decoder1)

        self.graphconvblock1 = GCN_Unit(output_size, output_size, base_layer, normalization_g, activation_g,
                                        instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(output_size * heads, output_size, base_layer, None, None, instance_track,
                                            filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(output_size, output_size, base_layer, None, None, instance_track,
                                            filter_size, 1, False, 0.0)

    def forward(self, x, style_vector1, style_vector2, edge_index, edge_attr, batch_size, nroi, batch_idx=None):

        A_matrix = self.weight_prediction(x, style_vector1, style_vector2, batch_size, nroi, batch_idx)
        x_reshape1 = torch.reshape(A_matrix, (-1, A_matrix.size()[-1]))
        x_reshape1 = self.graphconvblock1(x_reshape1, edge_index, edge_attr, nroi, batch_idx)
        x_reshape1 = self.graphconvblock2(x_reshape1, edge_index, edge_attr, nroi, batch_idx)
        x_reshape2 = torch.reshape(x_reshape1, (batch_size, nroi, x_reshape1.size()[1]))

        return x_reshape2

# Decoder18 : Model with two units, one for edge prediction and the other for weight prediction, but instead of binary, weighted matrices are given
# for decoder graph convolutional layers and the last layer is only a graph neural network with no normalization or activation
class Decoder18(nn.Module):

    def __init__(self, input_size, output_size, style_dim1, style_dim2, model_edgePred = 0, base_layer="cheb",
                 normalization_l1="instance", normalization_l2="instance", normalization_g="batch",
                 activation_l1="lrelu", activation_l2="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder1=0, p_decoder2=0):
        super().__init__()

        if model_edgePred == 0:
            self.edge_prediction = Decoder21(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)
        elif model_edgePred == 1:
            self.edge_prediction = Decoder2(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)

        self.weight_prediction = Decoder2(input_size, style_dim1, normalization_l1, activation_l1, instance_track, p_decoder1)

        self.graphconvblock1 = GCN_Unit(output_size, output_size, base_layer, normalization_g, activation_g,
                                        instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(output_size * heads, output_size, base_layer, None, None, instance_track,
                                            filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(output_size, output_size, base_layer, None, None, instance_track,
                                            filter_size, 1, False, 0.0)

    def forward(self, x, style_vector11, style_vector12, style_vector21, style_vector22, batch_size, nroi, batch_idx=None, matrix_threshold=0.5):

        A_matrix1 = self.edge_prediction(x, style_vector11, style_vector12, batch_size, nroi, batch_idx)
        A_out = torch.sigmoid(A_matrix1)

        # Convert A_out to edge index and edge weights
        offset_g, row_g, col_g = (A_out > matrix_threshold).nonzero().t()
        edge_attr_g = A_matrix1[offset_g, row_g, col_g]
        edge_attr_g = torch.unsqueeze(edge_attr_g, dim=1)
        row_g = row_g + offset_g * nroi
        col_g = col_g + offset_g * nroi
        edge_index_g = torch.stack([row_g, col_g], dim=0)

        A_matrix2 = self.weight_prediction(x, style_vector21, style_vector22, batch_size, nroi, batch_idx)
        x_reshape1 = torch.reshape(A_matrix2, (-1, A_matrix2.size()[-1]))
        x_reshape1 = self.graphconvblock1(x_reshape1, edge_index_g, edge_attr_g, nroi, batch_idx)
        x_reshape1 = self.graphconvblock2(x_reshape1, edge_index_g, edge_attr_g, nroi, batch_idx)
        x_reshape2 = torch.reshape(x_reshape1, (batch_size, nroi, x_reshape1.size()[1]))

        return A_matrix1, x_reshape2

# Decoder19 : Model where the adjacency matrix is first predicted using inner product decoder and then this matrix
# is used a feature matrix for two layers of graph neural networks
class Decoder19(nn.Module):

    def __init__(self, input_size, output_size, style_dim1, base_layer="cheb",
                 normalization_l1="instance", normalization_g="batch", activation_l1="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder1=0):
        super().__init__()

        self.weight_prediction = Decoder2(input_size, style_dim1, normalization_l1, activation_l1, instance_track, p_decoder1)

        self.graphconvblock1 = GCN_Unit(output_size, output_size, base_layer, normalization_g, activation_g,
                                        instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(output_size * heads, output_size, base_layer, normalization_g, activation_g, instance_track,
                                            filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(output_size, output_size, base_layer, normalization_g, activation_g, instance_track,
                                            filter_size, 1, False, 0.0)

    def forward(self, x, style_vector1, style_vector2, edge_index, edge_attr, batch_size, nroi, batch_idx=None):

        A_matrix = self.weight_prediction(x, style_vector1, style_vector2, batch_size, nroi, batch_idx)
        x_reshape1 = torch.reshape(A_matrix, (-1, A_matrix.size()[-1]))
        x_reshape1 = self.graphconvblock1(x_reshape1, edge_index, edge_attr, nroi, batch_idx)
        x_reshape1 = self.graphconvblock2(x_reshape1, edge_index, edge_attr, nroi, batch_idx)
        x_reshape2 = torch.reshape(x_reshape1, (batch_size, nroi, x_reshape1.size()[1]))

        return x_reshape2

# Decoder20 : Model with two units, one for edge prediction and the other for weight prediction, but instead of binary, weighted matrices are given
# for decoder graph convolutional layers
class Decoder20(nn.Module):

    def __init__(self, input_size, output_size, style_dim1, style_dim2, model_edgePred = 0, base_layer="cheb",
                 normalization_l1="instance", normalization_l2="instance", normalization_g="batch",
                 activation_l1="lrelu", activation_l2="lrelu", activation_g="lrelu",
                 instance_track=False, filter_size=2, heads=4, concat_gat=True, dropout_gat=0.3, p_decoder1=0, p_decoder2=0):
        super().__init__()

        if model_edgePred == 0:
            self.edge_prediction = Decoder21(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)
        elif model_edgePred == 1:
            self.edge_prediction = Decoder2(input_size, style_dim2, normalization_l2, activation_l2, instance_track, p_decoder2)

        self.weight_prediction = Decoder2(input_size, style_dim1, normalization_l1, activation_l1, instance_track, p_decoder1)

        self.graphconvblock1 = GCN_Unit(output_size, output_size, base_layer, normalization_g, activation_g,
                                        instance_track, filter_size, heads, concat_gat, dropout_gat)
        if base_layer == "gat" and concat_gat:
            self.graphconvblock2 = GCN_Unit(output_size * heads, output_size, base_layer, normalization_g, activation_g, instance_track,
                                            filter_size, 1, False, 0.0)
        else:
            self.graphconvblock2 = GCN_Unit(output_size, output_size, base_layer, normalization_g, activation_g, instance_track,
                                            filter_size, 1, False, 0.0)

    def forward(self, x, style_vector11, style_vector12, style_vector21, style_vector22, batch_size, nroi, batch_idx=None, matrix_threshold=0.5):

        A_matrix1 = self.edge_prediction(x, style_vector11, style_vector12, batch_size, nroi, batch_idx)
        A_out = torch.sigmoid(A_matrix1)

        # Convert A_out to edge index and edge weights
        offset_g, row_g, col_g = (A_out > matrix_threshold).nonzero().t()
        edge_attr_g = A_matrix1[offset_g, row_g, col_g]
        edge_attr_g = torch.unsqueeze(edge_attr_g, dim=1)
        row_g = row_g + offset_g * nroi
        col_g = col_g + offset_g * nroi
        edge_index_g = torch.stack([row_g, col_g], dim=0)

        A_matrix2 = self.weight_prediction(x, style_vector21, style_vector22, batch_size, nroi, batch_idx)
        x_reshape1 = torch.reshape(A_matrix2, (-1, A_matrix2.size()[-1]))
        x_reshape1 = self.graphconvblock1(x_reshape1, edge_index_g, edge_attr_g, nroi, batch_idx)
        x_reshape1 = self.graphconvblock2(x_reshape1, edge_index_g, edge_attr_g, nroi, batch_idx)
        x_reshape2 = torch.reshape(x_reshape1, (batch_size, nroi, x_reshape1.size()[1]))

        return A_matrix1, x_reshape2