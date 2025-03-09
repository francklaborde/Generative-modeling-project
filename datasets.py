import numpy as np
import torch
from sklearn.datasets import make_moons, make_swiss_roll
from torch.utils.data import Dataset


class TwoMoonsDataset(Dataset):
    """
    PyTorch Dataset for the Two Moons distribution.
    """

    def __init__(self, num_samples, noise=0.1):
        """
        Args:
            num_samples (int): Number of samples to generate.
            noise (float): Standard deviation of Gaussian noise added to the two moons.
        """
        self.num_samples = num_samples
        self.noise = noise
        # Generate data using sklearn's make_moons function.
        X, _ = make_moons(n_samples=num_samples, noise=noise)
        self.data = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class SwissRollDataset(Dataset):
    """
    PyTorch Dataset for the Swiss Roll distribution.
    """

    def __init__(self, num_samples, noise=0.0):
        """
        Args:
            num_samples (int): Number of samples to generate.
            noise (float): Standard deviation of noise added to the swiss roll.
        """
        self.num_samples = num_samples
        self.noise = noise
        # Generate data using sklearn's make_swiss_roll function.
        X, _ = make_swiss_roll(n_samples=num_samples, noise=noise)
        # Swiss roll is typically 3D. If desired, one could slice dimensions.
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

    Returns:
        torch.utils.data.Dataset: An instance of the requested dataset.
    """
    name = name.lower()
    if name == "two_moons":
        noise = kwargs.get("noise", 0.1)
        return TwoMoonsDataset(num_samples, noise)
    elif name == "swiss_roll":
        noise = kwargs.get("noise", 0.0)
        return SwissRollDataset(num_samples, noise)
    elif name == "gaussian":
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 1.0)
        dim = kwargs.get("dim", 2)
        return GaussianDataset(num_samples, mu, sigma, dim)
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
