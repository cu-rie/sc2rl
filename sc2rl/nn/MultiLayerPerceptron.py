import torch
import torch.nn.functional as F
from sc2rl.config.nn_configs import VERY_SMALL_NUMBER
from sc2rl.nn.SpectralNorm import SNLinear


def swish(x):
    return x * torch.nn.functional.sigmoid(x)


def mish(x):
    return x * torch.tanh(F.softplus(x))


def softplus(x):
    return F.softplus(x)


class Linear(torch.nn.Linear):

    def __init__(self, norm=False, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.norm = norm

    def forward(self, inputs):
        if self.norm:
            weight = self.weight
            weight_mean = weight.mean().detach()
            weight_std = weight.std().detach() + VERY_SMALL_NUMBER
            weight = (weight - weight_mean) / weight_std
        else:
            weight = self.weight
        return F.linear(inputs, weight, self.bias)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self,
                 input_dimension,
                 output_dimension,
                 num_neurons=[64, 64],
                 spectral_norm=False,
                 input_normalization=0,
                 hidden_activation='mish',
                 out_activation=None,
                 drop_probability=0.0,
                 init=None,
                 weight_standardization=False):
        """
        :param num_neurons: number of neurons for each layer
        :param out_activation: output layer's activation unit
        :param input_normalization: input normalization behavior flag
        0: Do not normalize, 1: Batch normalization, 2: Layer normalization
        :param hidden_activation: hidden layer activation units. supports 'relu','SELU','leaky_relu','sigmoid', 'tanh'
        :param init: hidden layer initialization. supports 'kaiming_normal'
        """

        super(MultiLayerPerceptron, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.out_activation = out_activation
        self.input_normalization = input_normalization
        self.hidden_activation = hidden_activation
        self.drop_probability = drop_probability
        self.init = init
        self.weight_standardization = weight_standardization
        self.spectral_norm = spectral_norm
        ws = self.weight_standardization

        # infer normalization layers
        if self.input_normalization == 0:
            pass
        else:
            if self.input_normalization == 1:
                norm_layer = torch.nn.BatchNorm1d(self.input_dimension)
            elif self.input_normalization == 2:
                norm_layer = torch.nn.LayerNorm(self.input_dimension)
            self.layers.append(norm_layer)

        # input -> hidden 1
        if self.spectral_norm:
            input_layer = SNLinear(in_features=self.input_dimension, out_features=num_neurons[0])
        else:
            input_layer = Linear(norm=ws, in_features=self.input_dimension, out_features=num_neurons[0])
        self.apply_weight_init(input_layer, self.init)
        self.layers.append(input_layer)
        for i, num_neuron in enumerate(num_neurons[:-1]):
            if self.spectral_norm:
                hidden_layer = SNLinear(in_features=num_neuron, out_features=num_neurons[i + 1])
            else:
                hidden_layer = Linear(norm=ws, in_features=num_neuron, out_features=num_neurons[i + 1])
            self.apply_weight_init(hidden_layer, self.init)
            self.layers.append(hidden_layer)
        if self.spectral_norm:
            last_layer = SNLinear(in_features=num_neurons[-1], out_features=self.output_dimension)
        else:
            last_layer = Linear(norm=ws, in_features=num_neurons[-1], out_features=self.output_dimension)
        self.apply_weight_init(last_layer, self.init)
        self.layers.append(last_layer)  # hidden_n -> output

    def forward(self, x):
        if self.input_normalization != 0:  # The first layer is not normalization layer
            out = self.layers[0](x)
            for layer in self.layers[1:-1]:  # Linear layer starts from layers[1]
                out = layer(out)
                if self.drop_probability > 0.0:
                    out = self.infer_dropout(self.drop_probability)(out)  # Apply dropout
                out = self.infer_activation(self.hidden_activation)(out)

            out = self.layers[-1](out)  # The last linear layer
            if self.out_activation is None:
                pass
            else:
                out = self.infer_activation(self.out_activation)(out)
        else:
            out = x
            for layer in self.layers[:-1]:
                out = layer(out)
                if self.drop_probability > 0.0:
                    out = self.infer_dropout(self.drop_probability)(out)
                out = self.infer_activation(self.hidden_activation)(out)

            out = self.layers[-1](out)
            # infer output activation units
            if self.out_activation is None:
                pass
            else:
                out = self.infer_activation(self.out_activation)(out)
        return out

    def apply_weight_init(self, tensor, init_method=None):
        if init_method is None:
            pass  # do not apply weight init
        elif init_method == "normal":
            torch.nn.init.normal_(tensor.weight, std=0.3)
            torch.nn.init.constant_(tensor.bias, 0.0)
        elif init_method == "kaiming_normal":
            if self.hidden_activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
                torch.nn.init.kaiming_normal_(tensor.weight, nonlinearity=self.hidden_activation)
                torch.nn.init.constant_(tensor.bias, 0.0)
            else:
                pass
        elif init_method == "xavier":
            torch.nn.init.xavier_uniform_(tensor.weight)
        else:
            raise NotImplementedError("MLP initializer {} is not supported".format(init_method))

    def infer_activation(self, activation):
        if activation == 'relu':
            ret = torch.nn.ReLU()
        elif activation == 'swish':
            ret = swish
        elif activation == 'sigmoid':
            ret = torch.nn.Sigmoid()
        elif activation == 'SELU':
            ret = torch.nn.SELU()
        elif activation == 'leaky_relu':
            ret = torch.nn.LeakyReLU()
        elif activation == 'tanh':
            ret = torch.nn.Tanh()
        elif activation == 'mish':
            ret = mish
        elif activation == 'softplus':
            ret = softplus
        else:
            raise RuntimeError("Given {} activation is not supported".format(self.out_activation))
        return ret

    @staticmethod
    def infer_dropout(p):
        if p >= 0.0:
            ret = torch.nn.Dropout(p=p)
        return ret
