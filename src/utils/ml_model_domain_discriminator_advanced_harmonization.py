import torch.nn as nn
from .ml_unit_models import GradientReversalLayer
from .ml_blocks import Encoder, SiteClassifierLinear, SiteClassifierGraph, MappingNetwork
from .ml_model_decoder import Decoder0, Decoder1, Decoder2, Decoder3, Decoder4, Decoder5, Decoder6, Decoder7, Decoder8, Decoder9, Decoder10, Decoder11,\
Decoder12, Decoder13, Decoder14, Decoder15, Decoder16, Decoder17, Decoder18, Decoder19, Decoder20

class GraphToGraphTranslationAE(nn.Module):
    def __init__(self, input_size, hidden_size, mapping_input_size, mapping_hidden_size, mapping_latent_size,
                 style_dim1, style_dim2, num_layers_latent, num_layers_style1, num_layers_style2, output_size, model_encoder=0, base_layer_encoder="cheb",
                 normalization_g_encoder = "batch", activation_g_encoder = "lrelu", instance_track_encoder=False,
                 filter_size_encoder=2, heads_encoder=2, concat_gat_encoder=False, dropout_gat_encoder=0.2,
                 type_input_domain="linear", model_domain=1, normalization_l_domain="batch", activation_l_domain="lrelu", p=0,
                 model_domain_graph=0, base_layer_domain="cheb", normalization_g_domain="batch", activation_g_domain="lrelu",
                 instance_track_domain = False, filter_size_domain=2, heads_domain=2, concat_gat_domain=False, dropout_gat_domain=0.2,
                 normalization_l_mapping=None, activation_l_mapping="lrelu",
                 model_decoder=0, model_edgePred = 0, base_layer_decoder="cheb", normalization_l1_decoder="instance", normalization_l2_decoder="instance",
                 normalization_g_decoder="batch", activation_l1_decoder="lrelu", activation_l2_decoder="lrelu", activation_g_decoder="lrelu",
                 instance_track_decoder=False, filter_size_decoder=2, heads_decoder=4, concat_gat_decoder=True, dropout_gat_decoder=0.3,
                 p_decoder1=0, p_decoder2=0):
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size, model_encoder, base_layer_encoder, normalization_g_encoder, activation_g_encoder,
                 instance_track_encoder, filter_size_encoder, heads_encoder, concat_gat_encoder, dropout_gat_encoder)

        self.type_input_domain = type_input_domain

        if self.type_input_domain=="linear":
            self.site_classifier = SiteClassifierLinear(hidden_size // 4, output_size, model_domain, normalization_l_domain, activation_l_domain, p)

        elif self.type_input_domain=="graph":
            self.site_classifier = SiteClassifierGraph(hidden_size // 4, output_size, model_domain_graph, model_domain, base_layer_domain,
                 normalization_g_domain, normalization_l_domain, activation_g_domain, activation_l_domain,
                 instance_track_domain, filter_size_domain, heads_domain, concat_gat_domain, dropout_gat_domain, p)

        self.model_decoder=model_decoder

        if self.model_decoder==0:
            self.mapping_network = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder0(hidden_size // 4, input_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder==1:
            self.mapping_network = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder1(hidden_size // 4, input_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder == 2:
            self.mapping_network1 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder2(hidden_size // 4, style_dim1, normalization_l1_decoder, activation_l1_decoder, instance_track_decoder, p_decoder1)

        elif self.model_decoder==3:
            self.mapping_network = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder3(hidden_size // 4, input_size, mapping_latent_size, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder == 4:
            self.mapping_network1 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder4(hidden_size // 4, mapping_latent_size, normalization_l1_decoder, activation_l1_decoder, instance_track_decoder, p_decoder1)

        elif self.model_decoder == 5:
            self.mapping_network1 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder5(hidden_size // 4, input_size, mapping_latent_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder==6:
            self.mapping_network1 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder6(hidden_size // 4, input_size, mapping_latent_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder,  activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder==7:
            self.mapping_network11 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.mapping_network12 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size * 2, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder7(hidden_size // 4, input_size, mapping_latent_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder == 8:
            self.mapping_network11 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.mapping_network12 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size * 2, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder8(hidden_size // 4, input_size, mapping_latent_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder == 9:
            self.mapping_network11 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network12 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder9(hidden_size // 4, input_size, style_dim1, style_dim2, model_edgePred, base_layer_decoder,
                 normalization_l1_decoder, normalization_l2_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_l2_decoder, activation_g_decoder,
                 instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1, p_decoder2)

        elif self.model_decoder == 10:
            self.mapping_network11 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network12 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder10(hidden_size // 4, input_size, style_dim1, style_dim2, model_edgePred, base_layer_decoder,
                 normalization_l1_decoder, normalization_l2_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_l2_decoder, activation_g_decoder,
                 instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1, p_decoder2)

        elif self.model_decoder==11:
            self.mapping_network1 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, mapping_latent_size * 2, num_layers_latent, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder11(hidden_size // 4, input_size, mapping_latent_size, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder == 12:
            self.mapping_network11 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network12 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder12(hidden_size // 4, input_size, style_dim1, style_dim2, model_edgePred, base_layer_decoder,
                 normalization_l1_decoder, normalization_l2_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_l2_decoder, activation_g_decoder,
                 instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1, p_decoder2)

        elif self.model_decoder == 13:
            self.mapping_network11 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network12 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder13(hidden_size // 4, input_size, style_dim1, style_dim2, model_edgePred, base_layer_decoder,
                 normalization_l1_decoder, normalization_l2_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_l2_decoder, activation_g_decoder,
                 instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1, p_decoder2)

        elif self.model_decoder == 14:
            self.mapping_network = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder14(hidden_size // 4, input_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder==15:
            self.mapping_network = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder15(hidden_size // 4, input_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder==16:
            self.mapping_network = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder16(hidden_size // 4, input_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder == 17:
            self.mapping_network1 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder17(hidden_size // 4, input_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder == 18:
            self.mapping_network11 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network12 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network21 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.mapping_network22 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder18(hidden_size // 4, input_size, style_dim1, style_dim2, model_edgePred, base_layer_decoder,
                 normalization_l1_decoder, normalization_l2_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_l2_decoder, activation_g_decoder,
                 instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1, p_decoder2)

        elif self.model_decoder == 19:
            self.mapping_network1 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.mapping_network2 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder19(hidden_size // 4, input_size, style_dim1, base_layer_decoder, normalization_l1_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_g_decoder, instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1)

        elif self.model_decoder == 20:
            self.mapping_network11 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network12 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim2, num_layers_style2, normalization_l_mapping, activation_l_mapping)
            self.mapping_network21 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.mapping_network22 = MappingNetwork(mapping_input_size, mapping_hidden_size, style_dim1, num_layers_style1, normalization_l_mapping, activation_l_mapping)
            self.decoder = Decoder20(hidden_size // 4, input_size, style_dim1, style_dim2, model_edgePred, base_layer_decoder,
                 normalization_l1_decoder, normalization_l2_decoder, normalization_g_decoder,
                 activation_l1_decoder, activation_l2_decoder, activation_g_decoder,
                 instance_track_decoder, filter_size_decoder, heads_decoder, concat_gat_decoder, dropout_gat_decoder, p_decoder1, p_decoder2)

    def forward(self, x, edge_index, edge_attr, batch_size, nroi, use_mean, batch_idx=None, matrix_threshold=0.5, alpha=1.0, conditional_vector=None,
                mean_edge_index = None, mean_edge_attr = None):

        invariant_features = self.encoder(x, edge_index, edge_attr, nroi, batch_idx)
        reversed_features = GradientReversalLayer(alpha)(invariant_features)

        if self.type_input_domain == "linear":
            site_output = self.site_classifier(reversed_features, batch_idx)

        elif self.type_input_domain == "graph":
            site_output = self.site_classifier(reversed_features, edge_index, edge_attr, nroi, batch_idx)

        if use_mean:
            edge_index_decoder = mean_edge_index.clone()
            edge_attr_decoder = mean_edge_attr.clone()
        else:
            edge_index_decoder = edge_index.clone()
            edge_attr_decoder = edge_attr.clone()

        A_matrix = None

        if self.model_decoder == 0 or self.model_decoder == 1 or self.model_decoder == 14 or self.model_decoder == 15 or self.model_decoder == 16:
            style_vector = self.mapping_network(conditional_vector)
            reconstructed_output = self.decoder(invariant_features, style_vector, edge_index_decoder, edge_attr_decoder, batch_size, nroi, batch_idx)

        elif self.model_decoder == 2:
            style_vector1 = self.mapping_network1(conditional_vector)
            style_vector2 = self.mapping_network2(conditional_vector)
            reconstructed_output = self.decoder(invariant_features, style_vector1, style_vector2, batch_size, nroi, batch_idx)

        elif self.model_decoder == 3:
            latent_vector = self.mapping_network(conditional_vector)
            reconstructed_output = self.decoder(invariant_features, latent_vector, edge_index_decoder, edge_attr_decoder, batch_size, nroi, batch_idx)

        elif self.model_decoder == 4:
            latent_vector1 = self.mapping_network1(conditional_vector)
            latent_vector2 = self.mapping_network2(conditional_vector)
            reconstructed_output = self.decoder(invariant_features, latent_vector1, latent_vector2, batch_size, nroi, batch_idx)

        elif self.model_decoder == 5 or self.model_decoder == 6:
            latent_vector = self.mapping_network1(conditional_vector)
            style_vector = self.mapping_network2(conditional_vector)
            reconstructed_output = self.decoder(invariant_features, latent_vector, style_vector, edge_index_decoder, edge_attr_decoder,  batch_size, nroi, batch_idx)

        elif self.model_decoder == 7 or self.model_decoder == 8:
            latent_vector1 = self.mapping_network11(conditional_vector)
            latent_vector2 = self.mapping_network12(conditional_vector)
            style_vector = self.mapping_network2(conditional_vector)
            reconstructed_output = self.decoder(invariant_features, latent_vector1, latent_vector2, style_vector, edge_index_decoder, edge_attr_decoder, batch_size, nroi, batch_idx)

        elif self.model_decoder == 9 or self.model_decoder == 10 or self.model_decoder == 12 or self.model_decoder == 13:
            style_vector11 = self.mapping_network11(conditional_vector)
            style_vector12 = self.mapping_network12(conditional_vector)
            style_vector2 = self.mapping_network2(conditional_vector)
            A_matrix, reconstructed_output = self.decoder(invariant_features, style_vector11, style_vector12, style_vector2, batch_size, nroi, batch_idx, matrix_threshold)

        elif self.model_decoder == 11:
            latent_vector1 = self.mapping_network1(conditional_vector)
            latent_vector2 = self.mapping_network2(conditional_vector)
            reconstructed_output = self.decoder(invariant_features, latent_vector1, latent_vector2, edge_index_decoder, edge_attr_decoder, batch_size, nroi, batch_idx)

        elif self.model_decoder == 17 or self.model_decoder == 19:
            style_vector1 = self.mapping_network1(conditional_vector)
            style_vector2 = self.mapping_network2(conditional_vector)
            reconstructed_output = self.decoder(invariant_features, style_vector1, style_vector2,
                                                edge_index_decoder, edge_attr_decoder, batch_size, nroi, batch_idx)

        elif self.model_decoder == 18 or self.model_decoder == 20:
            style_vector11 = self.mapping_network11(conditional_vector)
            style_vector12 = self.mapping_network12(conditional_vector)
            style_vector21 = self.mapping_network21(conditional_vector)
            style_vector22 = self.mapping_network22(conditional_vector)
            A_matrix, reconstructed_output = self.decoder(invariant_features, style_vector11, style_vector12, style_vector21, style_vector22,
                                                          batch_size, nroi, batch_idx, matrix_threshold)

        return invariant_features, reversed_features, site_output, reconstructed_output, A_matrix









