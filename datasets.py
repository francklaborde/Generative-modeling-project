import numpy as np
import torch
import torchvision
from sklearn.datasets import make_moons, make_swiss_roll
from torch.utils.data import Dataset


class TwoMoonsDataset(Dataset):
    """
    PyTorch Dataset for the Two Moons distribution.
    """

    def __init__(self, num_samples, noise=0.1, scale=1.0):
        """
        Args:
            num_samples (int): Number of samples to generate.
            noise (float): Standard deviation of Gaussian noise added to the two moons.
            scale (float): Scaling factor for the two moons.
        """
        self.num_samples = num_samples
        self.noise = noise
        self.scale = scale
        X, _ = make_moons(n_samples=num_samples, noise=noise)
        self.data = torch.tensor(X, dtype=torch.float32) * scale

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class SwissRollDataset(Dataset):
    """
    PyTorch Dataset for the Swiss Roll distribution.
    """

    def __init__(self, num_samples, noise=0.0, scale=1.0):
        """
        Args:
            num_samples (int): Number of samples to generate.
            noise (float): Standard deviation of noise added to the swiss roll.
            scale (float): Scaling factor for the swiss roll.
        """
        self.num_samples = num_samples
        self.noise = noise
        X, _ = make_swiss_roll(n_samples=num_samples, noise=noise)
        X = X[:, [0, 2]]
        self.data = torch.tensor(X, dtype=torch.float32) * scale

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class UniformDataset(Dataset):
    """
    PyTorch Dataset for the Uniform distribution.
    """

    def __init__(self, num_samples, low=0.0, high=1.0, dim=2):
        """
        Args:
            num_samples (int): Number of samples to generate.
            low (float or array-like): Lower bound of the Uniform distribution.
            high (float or array-like): Upper bound of the Uniform distribution.
            dim (int): Dimensionality of the samples.
        """
        self.num_samples = num_samples
        self.low = low
        self.high = high

        if np.isscalar(low):
            low_arr = np.full((dim,), low)
        else:
            low_arr = np.array(low)
        if np.isscalar(high):
            high_arr = np.full((dim,), high)
        else:
            high_arr = np.array(high)

        X = np.random.uniform(low=low_arr, high=high_arr, size=(num_samples, dim))
        self.data = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class GaussianDataset(Dataset):
    """
    PyTorch Dataset for the Gaussian distribution.
    """

    def __init__(self, num_samples, mu=0.0, sigma=1.0, dim=2):
        """
        Args:
            num_samples (int): Number of samples to generate.
            mu (float or array-like): Mean of the Gaussian distribution.
            sigma (float or array-like): Standard deviation of the Gaussian.
            dim (int): Dimensionality of the samples.
        """
        self.num_samples = num_samples
        self.mu = mu
        self.sigma = sigma
        self.dim = dim

        # Ensure mu and sigma are arrays of the correct shape.
        if np.isscalar(mu):
            mu_arr = np.full((dim,), mu)
        else:
            mu_arr = np.array(mu)
        if np.isscalar(sigma):
            sigma_arr = np.full((dim,), sigma)
        else:
            sigma_arr = np.array(sigma)

        X = np.random.normal(loc=mu_arr, scale=sigma_arr, size=(num_samples, dim))
        self.data = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class DiscretePointsDataset(Dataset):
    """
    PyTorch Dataset for a set of discrete points arranged in a multi-dimensional grid with added noise.
    """

    def __init__(self, num_samples, low=0.0, high=1.0, dim=2, noise_factor=0.1):
        """
        Args:
            num_samples (int): Total number of points to generate (must be a perfect power of dim).
            low (float): Lower bound of the grid.
            high (float): Upper bound of the grid.
            dim (int): Dimensionality of the grid.
            noise_factor (float): Fraction of grid spacing to use as noise magnitude.
        """
        num_rows = int(num_samples ** (1 / dim))
        if num_rows**dim != num_samples:
            raise ValueError("num_samples must be a perfect power of dim.")

        self.num_rows = num_rows
        self.low = float(low)
        self.high = float(high)
        self.dim = dim
        self.noise_factor = noise_factor

        axes = [np.linspace(self.low, self.high, num_rows) for _ in range(dim)]
        grid = np.meshgrid(*axes, indexing="ij")
        grid_points = np.vstack([axis.ravel() for axis in grid]).T

        grid_spacing = (self.high - self.low) / (num_rows - 1) if num_rows > 1 else 1.0

        noise = np.random.uniform(
            -self.noise_factor * grid_spacing,
            self.noise_factor * grid_spacing,
            grid_points.shape,
        )
        grid_points += noise

        self.data = torch.tensor(grid_points, dtype=torch.float32)

    def __len__(self):
        return self.num_rows**self.dim

    def __getitem__(self, idx):
        return self.data[idx]


class FashionMNISTDataset(Dataset):
    """
    PyTorch Dataset for the Fashion MNIST dataset.
    """

    def __init__(self, num_samples=None, data_path="./data"):
        """
        Args:
            num_samples (int): Number of samples to generate.
            data_path (str): Path to the Fashion MNIST dataset.
            transform (callable): Optional transform to apply to the samples. If not provided, the samples are converted to tensors and normalized.
        """
        self.num_samples = num_samples

        # Load Fashion MNIST dataset using torchvision.
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        dataset = torchvision.datasets.FashionMNIST(
            root=data_path, train=True, download=True, transform=transform
        )

        images = [dataset[i][0] for i in range(len(dataset))]

        assert num_samples <= len(
            dataset
        ), "num_samples must be less than or equal to the dataset size."
        if num_samples is not None:
            self.data = torch.utils.data.Subset(
                images, np.random.choice(len(dataset), num_samples)
            )
        else:
            self.data = images

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def make_dataset(name, num_samples=1000, **kwargs):
    """
    Factory function for creating a dataset.

    Args:
        name (str): Name of the dataset. Options: "two_moons", "swiss_roll", "gaussian".
        num_samples (int): Number of samples to generate.
        **kwargs: Additional parameters for the dataset creation.
            - For "two_moons": noise (default 0.1)
            - For "swiss_roll": noise (default 0.0)
            - For "gaussian": mu (default 0.0), sigma (default 1.0), dim (default 2)
            - For "fashion_mnist": data_path (default "./data"), transform (default None)

    Returns:
        torch.utils.data.Dataset: An instance of the requested dataset.
    """
    name = name.lower()
    if name == "two_moons":
        noise = kwargs.get("noise", 0.1)
        scale = kwargs.get("scale", 1.0)
        return TwoMoonsDataset(num_samples, noise, scale)
    elif name == "swiss_roll":
        noise = kwargs.get("noise", 0.0)
        scale = kwargs.get("scale", 1.0)
        return SwissRollDataset(num_samples, noise, scale)
    elif name == "gaussian":
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 1.0)
        dim = kwargs.get("dim", 2)
        return GaussianDataset(num_samples, mu, sigma, dim)
    elif name == "fashion_mnist":
        data_path = kwargs.get("data_path", "./data")
        return FashionMNISTDataset(num_samples, data_path=data_path)
    elif name == "uniform":
        low = kwargs.get("low", 0.0)
        high = kwargs.get("high", 1.0)
        dim = kwargs.get("dim", 2)
        return UniformDataset(num_samples, low, high, dim)
    elif name == "discrete_points":
        low = kwargs.get("low", 0.0)
        high = kwargs.get("high", 1.0)
        dim = kwargs.get("dim", 2)
        return DiscretePointsDataset(num_samples, low, high, dim)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, target_dataset):
        assert len(source_dataset) == len(
            target_dataset
        ), "Source and target datasets must have the same number of points"
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        return self.source_dataset[idx], self.target_dataset[idx]
