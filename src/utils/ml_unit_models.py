import torch
import torch.nn as nn
import torch_geometric.nn as graphnn
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class GCN_Unit(nn.Module):
    def __init__(self, input_size, output_size, base_layer="cheb", normalization_f=None, activation_f="lrelu",
                 instance_track=False, filter_size=2, heads=2, concat_gat=False, dropout_gat=0.3, style_dim = None):
        super().__init__()
        if base_layer == "cheb":
            self.graphconv = graphnn.ChebConv(input_size, output_size, K=filter_size)
            output_size1 = output_size
        elif base_layer == "gcn":
            self.graphconv = graphnn.GCNConv(input_size, output_size)
            output_size1 = output_size
        elif base_layer == "gat":
            self.graphconv = graphnn.GATv2Conv(input_size, output_size, heads=heads, concat=concat_gat,
                                               dropout=dropout_gat, edge_dim=1)
            if concat_gat:
                output_size1 = heads * output_size
            else:
                output_size1 = output_size
        self.normalization_f = normalization_f
        if self.normalization_f == "batch":
            self.norm_f = graphnn.norm.BatchNorm(output_size1)
        elif self.normalization_f == "instance":
            if style_dim:
                self.norm_f = AdaIN(output_size, style_dim, instance_track)
            else:
                self.norm_f = graphnn.norm.InstanceNorm(output_size1, affine=True, track_running_stats=instance_track)
        elif self.normalization_f == "layer":
            self.norm_f = graphnn.norm.LayerNorm(output_size1)
        elif self.normalization_f == "graph":
            self.norm_f = graphnn.norm.GraphNorm(output_size1)
        else:
            self.norm_f = None
        if activation_f=="lrelu":
            self.act_f = nn.LeakyReLU()
        elif activation_f=="elu":
            self.act_f = nn.ELU()
        else:
            self.act_f = None

    def forward(self, x, edge_index, edge_attr, nroi, batch_idx=None, style_vector=None):
        output = self.graphconv(x, edge_index, edge_attr)
        if self.norm_f is not None:
            if self.normalization_f == "batch":
                output = self.norm_f(output)
            else:
                if self.normalization_f == "instance":
                    if style_vector is not None:
                        output = self.norm_f(output, style_vector, nroi, batch_idx)
                    else:
                        output = self.norm_f(output, batch_idx)
                else:
                    output = self.norm_f(output, batch_idx)
        if self.act_f is not None:
            output = self.act_f(output)

        return output

class MLP_Unit(nn.Module):
    def __init__(self, input_size, output_size, normalization_f=None, activation_f="lrelu", p=0):
        super().__init__()
        self.l1 = nn.Linear(input_size, output_size)
        if normalization_f=="batch":
            self.norm_f = nn.BatchNorm1d(output_size)
        elif normalization_f=="layer":
            self.norm_f = nn.LayerNorm(output_size)
        else:
            self.norm_f = None
        if p:
            self.dropout = nn.Dropout(p=p)
        else:
            self.dropout = None
        if activation_f=="lrelu":
            self.act_f = nn.LeakyReLU()
        elif activation_f=="elu":
            self.act_f = nn.ELU()
        else:
            self.act_f = None

    def forward(self, x):
        output = self.l1(x)
        if self.norm_f is not None:
            output = self.norm_f(output)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.act_f is not None:
            output = self.act_f(output)

        return output

class MLP_AdaLayer_Unit(nn.Module):
    def __init__(self, input_size, output_size, activation_f="lrelu", p=0, style_dim = None):
        super().__init__()
        self.l1 = nn.Linear(input_size, output_size)
        self.norm_f = AdaLA(output_size, style_dim)
        if p:
            self.dropout = nn.Dropout(p=p)
        else:
            self.dropout = None
        if activation_f=="lrelu":
            self.act_f = nn.LeakyReLU()
        elif activation_f=="elu":
            self.act_f = nn.ELU()
        else:
            self.act_f = None

    def forward(self, x, style_vector=None):
        output = self.l1(x)
        output = self.norm_f(output, style_vector)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.act_f is not None:
            output = self.act_f(output)

        return output

class MLP_Graph_Unit(nn.Module):
    def __init__(self, input_size, output_size, normalization_f=None, activation_f="lrelu", instance_track=False, p=0, style_dim = None):
        super().__init__()
        self.l1 = nn.Linear(input_size, output_size)
        self.normalization_f = normalization_f
        if self.normalization_f == "batch":
            self.norm_f = nn.BatchNorm1d(output_size)
        elif self.normalization_f == "batch2":
            self.norm_f = graphnn.norm.BatchNorm(output_size)
        elif self.normalization_f == "layer":
            self.norm_f = nn.LayerNorm(output_size)
        elif self.normalization_f == "layer2":
            self.norm_f = graphnn.norm.LayerNorm(output_size)
        elif self.normalization_f == "instance":
            if style_dim:
                self.norm_f = AdaIN(output_size, style_dim, instance_track)
            else:
                self.norm_f = graphnn.norm.InstanceNorm(output_size, affine=True, track_running_stats=instance_track)
        elif self.normalization_f == "graph":
            self.norm_f = graphnn.norm.GraphNorm(output_size)
        else:
            self.norm_f = None
        if p:
            self.dropout = nn.Dropout(p=p)
        else:
            self.dropout = None
        if activation_f=="lrelu":
            self.act_f = nn.LeakyReLU()
        elif activation_f=="elu":
            self.act_f = nn.ELU()
        else:
            self.act_f = None

    def forward(self, x, nroi, batch_idx=None, style_vector=None):
        output = self.l1(x)
        if self.norm_f is not None:
            if self.normalization_f == "batch" or self.normalization_f == "batch2" or self.normalization_f == "layer":
                output = self.norm_f(output)
            else:
                if self.normalization_f == "instance":
                    if style_vector is not None:
                        output = self.norm_f(output, style_vector, nroi, batch_idx)
                    else:
                        output = self.norm_f(output, batch_idx)
                else:
                    output = self.norm_f(output, batch_idx)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.act_f is not None:
            output = self.act_f(output)

        return output

class AdaIN(nn.Module):
    def __init__(self, num_features, style_dim, instance_track=False):
        super(AdaIN, self).__init__()
        # Linear layers to produce global scale and shift parameters based on style vector
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)
        self.norm = graphnn.norm.InstanceNorm(num_features, affine=False, track_running_stats=instance_track) # Graph-aware instance norm

    def forward(self, x, style_vector, nroi, batch_idx=None):
        # x has shape [total_nodes, num_features]
        # style_vector has shape [batch_size, style_dim] (one style vector per graph in batch)
        # nroi is the number of nodes per graph
        # batch_idx has shape [total_nodes], mapping nodes to their respective graph in the batch

        # Compute the global scale and shift for each graph
        scale = self.style_scale(style_vector)  # Shape: [batch_size, num_features]
        shift = self.style_shift(style_vector)  # Shape: [batch_size, num_features]

        # Apply graph-aware InstanceNorm
        x = self.norm(x, batch_idx)  # Uses the batch index to apply graph-level normalization

        # Broadcast scale and shift across all nodes in each graph in the batch
        scale = scale.repeat_interleave(nroi, dim=0)  # Shape: [total_nodes, num_features]
        shift = shift.repeat_interleave(nroi, dim=0)  # Shape: [total_nodes, num_features]

        # Apply adaptive scaling and shifting
        x = scale * x + shift

        return x

class AdaLA(nn.Module):
    def __init__(self, num_features, style_dim):
        super(AdaLA, self).__init__()
        # Linear layers to produce global scale and shift parameters based on style vector
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False, bias=False) # Layernorm without the learnable scale and bias parameters

    def forward(self, x, style_vector):

        # x has shape [batch_size, num_features]
        # style_vector has shape [batch_size, style_dim] (one style vector per example in batch)

        # Compute the global scale and shift for each example
        scale = self.style_scale(style_vector)  # Shape: [batch_size, num_features]
        shift = self.style_shift(style_vector)  # Shape: [batch_size, num_features]

        # Apply LayerNorm (normalize to zero mean and unit variance)
        x = self.norm(x)

        # Apply adaptive scaling and shifting
        x = scale * x + shift

        return x


