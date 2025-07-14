import torch
from .ml_unit_models import GradientReversalLayer
from .ml_linear_models import MLP_1NormalizedSmall, MLP_2NormalizedMedium, MLP_3NormalizedMedium, MLP_4NormalizedLarge
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
import math


import random

# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module

class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)

# Normalization on every element of input vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    intermediate vector w into a style vector
    '''

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result

class convrelu(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaIn(out_channel)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)

        return result


class Intermediate_Generator(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''

    def __init__(self, n_fc, dim_latent):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_fc):
            layers.append(SLinear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        latent_w = self.mapping(latent_z)
        return latent_w

class ResNetAutoEncoder2D(nn.Module):

    def __init__(self, in_channels=1, n_fc=8, dim_latent=512, out_channels=1, num_sites=4, model_domain=1,
                 normalization_l_domain="batch", activation_l_domain="lrelu", p=0):
        super().__init__()

        resnet = models.resnet18(weights=None)
        self.mapping_network = Intermediate_Generator(n_fc, dim_latent)

        # Modify the first convolutional layer to handle in_channels
        if in_channels != 3:
            self.encoder_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.encoder_conv1 = resnet.conv1

        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        # Encoder layers
        self.encoder_layer1 = resnet.layer1  # Output: 64 features
        self.encoder_layer2 = resnet.layer2  # Output: 128 features
        self.encoder_layer3 = resnet.layer3  # Output: 256 features
        self.encoder_layer4 = resnet.layer4  # Output: 512 features

        # site classifier
        self.site_classifier = SiteClassifier(512, num_sites, model_domain, normalization_l_domain, activation_l_domain, p)

        # skip connection layers
        #self.layer0_1x1 = convrelu(64, 64, 1, 0, dim_latent)
        #self.layer1_1x1 = convrelu(64, 64, 1, 0, dim_latent)
        #self.layer2_1x1 = convrelu(128, 128, 1, 0, dim_latent)
        #self.layer3_1x1 = convrelu(256, 256, 1, 0, dim_latent)
        #self.layer4_1x1 = convrelu(512, 512, 1, 0, dim_latent)

        # Decoder layers (Upsampling in U-Net fashion)
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.decoder3 = convrelu(512, 512, 3, 1, dim_latent)

        self.upconv3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.decoder2 = convrelu(512, 256, 3, 1, dim_latent)

        self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.decoder1 = convrelu(256, 256, 3, 1, dim_latent)

        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.decoder0 = convrelu(256, 128, 3, 1, dim_latent)

        self.upconv0 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        #self.conv_original_size0 = convrelu(1, 64, 3, 1, dim_latent)
        #self.conv_original_size1 = convrelu(64, 64, 3, 1, dim_latent)
        self.conv_original_size2 = convrelu(128, 64, 3, 1, dim_latent)

        # Final convolution to restore the spatial resolution to 274x274
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, input, latent_z, alpha=1.0):

        # input is the input image and latent_z is the 512-d input code for the corresponding site
        if type(latent_z) != type([]):
            #print('You should use list to package your latent_z')
            latent_z = [latent_z]

        # latent_w as well as current_latent is the intermediate vector
        latent_w = [self.mapping_network(latent) for latent in latent_z]
        current_latent1 = latent_w
        current_latent = current_latent1[0]

        #x_original = self.conv_original_size0(input, current_latent)
        #x_original = self.conv_original_size1(x_original, current_latent)

        layer0 = self.encoder_conv1(input)
        layer0 = self.encoder_bn1(layer0)
        layer0 = self.encoder_relu(layer0)
        layer1 = self.encoder_maxpool(layer0)
        layer1 = self.encoder_layer1(layer1)
        layer2 = self.encoder_layer2(layer1)
        layer3 = self.encoder_layer3(layer2)
        layer4 = self.encoder_layer4(layer3)

        reversed_features = GradientReversalLayer(alpha)(layer4)
        site_output = self.site_classifier(reversed_features)

        #layer4 = self.layer4_1x1(layer4, current_latent)
        x = self.upconv4(layer4)
        #layer3 = self.layer3_1x1(layer3, current_latent)
        #x = torch.cat([x, layer3], dim=1)
        x = self.decoder3(x, current_latent)

        x = self.upconv3(x)
        #layer2 = self.layer2_1x1(layer2, current_latent)
        #x = torch.cat([x, layer2], dim=1)
        x = self.decoder2(x, current_latent)

        x = self.upconv2(x)
        #layer1 = self.layer1_1x1(layer1, current_latent)
        #x = torch.cat([x, layer1], dim=1)
        x = self.decoder1(x, current_latent)

        x = self.upconv1(x)
        #layer0 = self.layer0_1x1(layer0, current_latent)
        #x = torch.cat([x, layer0], dim=1)
        x = self.decoder0(x, current_latent)

        x = self.upconv0(x)
        #x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x,current_latent)

        reconstructed_output = self.final_conv(x)

        return reversed_features, site_output, reconstructed_output

class SiteClassifier(nn.Module):

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

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = mlp(input_size, output_size, normalization_l, activation_l, p)

    def forward(self, x):

        x = self.global_avg_pool(x)  # Output: [batch_size, 512, 1, 1]
        x = x.view(x.size(0), -1)

        site_output = self.fc1(x)

        return site_output




