from collections import OrderedDict

import torch


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.ReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layer_dict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return out


class MLPBlock(torch.nn.Module):
    def __init__(self, hidden_size=64, num_layers=3, dropout_prob=0.5):
        super(MLPBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout_prob))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NCLNetwork(torch.nn.Module):
    def __init__(self, input_size=2, output_size=1, num_blocks=10, hidden_size=64, num_layers=3, dropout_prob=0.5):
        super(NCLNetwork, self).__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(MLPBlock(hidden_size, num_layers, dropout_prob))
            blocks.append(torch.nn.BatchNorm1d(hidden_size))
            blocks.append(torch.nn.Dropout(p=dropout_prob))
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.blocks = torch.nn.Sequential(*blocks)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

