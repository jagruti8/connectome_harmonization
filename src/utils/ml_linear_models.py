import torch.nn as nn
from .ml_unit_models import MLP_Unit

class MLP_1NormalizedSmall(nn.Module):

    def __init__(self, input_size, output_size, normalization_l="batch", activation_l="lrelu", p=0):
        super().__init__()
        self.block1 = MLP_Unit(input_size, input_size // 2, normalization_l, activation_l, p)
        self.block2 = MLP_Unit(input_size // 2, input_size // 4, normalization_l, activation_l, p)
        self.last_layer = nn.Linear(input_size // 4, output_size)

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.last_layer(output)

        return output

class MLP_2NormalizedMedium(nn.Module):

    def __init__(self, input_size, output_size, normalization_l="batch", activation_l="lrelu", p=0):
        super().__init__()
        self.block1 = MLP_Unit(input_size, input_size // 2, normalization_l, activation_l, p)
        self.block2 = MLP_Unit(input_size // 2, input_size // 2, normalization_l, activation_l, p)
        self.block3 = MLP_Unit(input_size // 2, input_size // 4, normalization_l, activation_l, p)
        self.block4 = MLP_Unit(input_size // 4, input_size // 4, normalization_l, activation_l, p)
        self.last_layer = nn.Linear(input_size // 4, output_size)

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.last_layer(output)

        return output

class MLP_3NormalizedMedium(nn.Module):

    def __init__(self, input_size, output_size, normalization_l="batch", activation_l="lrelu", p=0):
        super().__init__()
        self.block1 = MLP_Unit(input_size, input_size // 2, normalization_l, activation_l, p)
        self.block2 = MLP_Unit(input_size // 2, input_size // 4, normalization_l, activation_l, p)
        self.block3 = MLP_Unit(input_size // 4, input_size // 8, normalization_l, activation_l, p)
        self.block4 = MLP_Unit(input_size // 8, input_size // 16, normalization_l, activation_l, p)
        self.last_layer = nn.Linear(input_size // 16, output_size)

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.last_layer(output)

        return output

class MLP_4NormalizedLarge(nn.Module):

    def __init__(self, input_size, output_size, normalization_l="batch", activation_l="lrelu", p=0):
        super().__init__()
        self.block1 = MLP_Unit(input_size, input_size // 2, normalization_l, activation_l, p)
        self.block2 = MLP_Unit(input_size // 2, input_size // 2, normalization_l, activation_l, p)
        self.block3 = MLP_Unit(input_size // 2, input_size // 4, normalization_l, activation_l, p)
        self.block4 = MLP_Unit(input_size // 4, input_size // 4, normalization_l, activation_l, p)
        self.block5 = MLP_Unit(input_size // 4, input_size // 8, normalization_l, activation_l, p)
        self.block6 = MLP_Unit(input_size // 8, input_size // 8, normalization_l, activation_l, p)
        self.block7 = MLP_Unit(input_size // 8, input_size // 16, normalization_l, activation_l, p)
        self.block8 = MLP_Unit(input_size // 16, input_size // 16, normalization_l, activation_l, p)
        self.last_layer = nn.Linear(input_size // 16, output_size)

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.block6(output)
        output = self.block7(output)
        output = self.block8(output)
        output = self.last_layer(output)

        return output