import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRelu(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_layers (list): List of hidden layer sizes.
            output_dim (int): Number of output features.
        """
        super(MLPRelu, self).__init__()

        assert (
            isinstance(hidden_layers, list) and len(hidden_layers) > 0
        ), "hidden_layers must be a non-empty list"

        layers = []
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)