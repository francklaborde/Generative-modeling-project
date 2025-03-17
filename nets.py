import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRelu(nn.Module):
    def __init__(self, d_in, hidden_layers, d_out):
        super().__init__()
        self.layer_dims = [d_in] + hidden_layers + [d_out]
        self.num_layers = len(self.layer_dims) - 1

        self.weights = nn.ModuleList()
        self.biases = nn.ParameterList()

        for n in range(1, len(self.layer_dims)):
            weight_n = nn.ParameterList()
            for i in range(n):
                W = nn.Parameter(
                    torch.randn(self.layer_dims[n], self.layer_dims[i])
                    * (2 / self.layer_dims[i]) ** 0.5  # He initialization
                )
                weight_n.append(W)
            self.weights.append(weight_n)

            b_n = nn.Parameter(torch.randn(self.layer_dims[n]))
            self.biases.append(b_n)

    def forward(self, x):
        h = [x]

        for n in range(self.num_layers):
            out = (
                sum(torch.matmul(h_i, W_ni.T) for W_ni, h_i in zip(self.weights[n], h))
                + self.biases[n]
            )
            if n < self.num_layers - 1:
                out = F.relu(out)
            h.append(out)

        return h[-1]


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_features=64):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
        """
        super(CNN, self).__init__()

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_dim,
                out_channels=n_features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            # nn.BatchNorm2d(n_features * 8),
            nn.ReLU(True),
            # state size. (n_features*8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels=n_features * 8,
                out_channels=n_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # nn.BatchNorm2d(n_features * 4),
            nn.ReLU(True),
            # state size. (n_features*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=n_features * 4,
                out_channels=n_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # nn.BatchNorm2d(n_features * 2),
            nn.ReLU(True),
            # state size. (n_features*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=n_features * 2,
                out_channels=n_features,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # nn.BatchNorm2d(n_features),
            nn.ReLU(True),
            # state size. (n_features) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=n_features,
                out_channels=output_dim,
                kernel_size=1,
                stride=1,
                padding=2,
            ),
            nn.Tanh(),
            # output size. 1 x 28 x 28
        )

    def forward(self, x):
        x.unsqueeze_(2)
        x.unsqueeze_(3)
        x = self.conv_transpose(x)
        return x.view(x.size(0), -1)


class ResCNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_features=64):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
        """
        super(ResCNN, self).__init__()

        self.conv_transpose1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_dim,
                out_channels=n_features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
            ))
        self.conv1 = nn.Conv2d(
            in_channels=n_features * 8,
            out_channels=n_features * 8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_transpose2 = nn.ConvTranspose2d(  # state size. (n_features*8) x 4 x 4
                in_channels=n_features * 8,
                out_channels=n_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        self.conv2 = nn.Conv2d(
            in_channels=n_features * 4,
            out_channels=n_features * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_transpose3 = nn.ConvTranspose2d( # state size. (n_features*4) x 8 x 8
                in_channels=n_features * 4,
                out_channels=n_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        self.conv3 = nn.Conv2d(
            in_channels=n_features * 2,
            out_channels=n_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_transpose4 = nn.Sequential(
            # state size. (n_features*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=n_features * 2,
                out_channels=n_features,
                kernel_size=4,
                stride=2,
                padding=1,
            ))
        self.conv4 = nn.Conv2d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_transpose5 = nn.Sequential(
            # state size. (n_features) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=n_features,
                out_channels=output_dim,
                kernel_size=1,
                stride=1,
                padding=2,
            )
            # output size. 1 x 28 x 28
        )
        self.conv5 = nn.Conv2d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x.unsqueeze_(2)
        x.unsqueeze_(3)

        x = F.relu(self.conv1(self.conv_transpose1(x)) + self.conv_transpose1(x))
        x = F.relu(self.conv2(self.conv_transpose2(x)) + self.conv_transpose2(x))
        x = F.relu(self.conv3(self.conv_transpose3(x)) + self.conv_transpose3(x))
        x = F.relu(self.conv4(self.conv_transpose4(x)) + self.conv_transpose4(x))
        x = F.tanh(self.conv5(self.conv_transpose5(x)) + self.conv_transpose5(x))
        
        return x.view(x.size(0), -1)


def make_model(name, input_dim, output_dim, hidden_layers=None):
    """
    Factory function for creating a model.

    Args:
        name (str): Name of the model. Options: "mlp", "cnn".
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden_layers (list): List of hidden layer sizes (Require for "mlp").
    """

    if name == "mlp":
        assert hidden_layers is not None, "hidden_layers must be provided for MLP"
        return MLPRelu(input_dim, hidden_layers, output_dim)
    elif name == "cnn":
        return CNN(input_dim, output_dim)
    elif name == "res_cnn":
        return ResCNN(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model name: {name}")
