import argparse

import torch
from torch.utils.data import DataLoader

from datasets import PairedDataset, make_dataset
from loss import sw_loss
from nets import MLPRelu
from train import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a generator to map a source distribution to a target distribution."
    )
    # Dataset parameters
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="gaussian",
        choices=["two_moons", "swiss_roll", "gaussian"],
        help="Name of the source dataset.",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="two_moons",
        choices=["two_moons", "swiss_roll", "gaussian"],
        help="Name of the target dataset.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1000,
        help="Number of points in each of the training and validation datasets.",
    )
    parser.add_argument(
        "--source_mu", 
        type=float,
        default=0,
        help="Mean of the source Gaussian distribution (applies to gaussian).",
    )
    parser.add_argument(
        "--target_mu", 
        type=float,
        default=0,
        help="Mean of the target Gaussian distribution (applies to gaussian).",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Dimensionality of the dataset (applies to gaussian).",
    )
    parser.add_argument(
        "--source_noise",
        type=float,
        default=0.1,
        help="Noise level for the source dataset (if applicable).",
    )
    parser.add_argument(
        "--target_noise",
        type=float,
        default=0.1,
        help="Noise level for the target dataset (if applicable).",
    )

    # Model hyperparameters
    parser.add_argument(
        "--hidden_layers",
        type=str,
        default="64,64",
        help="Comma separated list of hidden layer sizes for the generator.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "npsgd"],
        help="Optimizer to use: 'sgd' or 'npsgd' (npsgd is a placeholder for now).",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Interval (in batches) at which to log training progress.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints and training history.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hidden_layers = [
        int(x.strip()) for x in args.hidden_layers.split(",") if x.strip().isdigit()
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.source_dataset == "gaussian":
        source_dataset = make_dataset(
            args.source_dataset,
            num_samples=args.num_points,
            mu=args.source_mu,
            sigma=args.source_noise,
            dim=args.dimension,
        )
    else:
        source_dataset = make_dataset(
            args.source_dataset, num_samples=args.num_points, noise=args.source_noise
        )

    if args.target_dataset == "gaussian":
        target_dataset = make_dataset(
            args.target_dataset,
            num_samples=args.num_points,
            mu=args.target_mu,
            sigma=args.target_noise,
            dim=args.dimension,
        )
    else:
        target_dataset = make_dataset(
            args.target_dataset, num_samples=args.num_points, noise=args.target_noise
        )

    paired_dataset = PairedDataset(source_dataset, target_dataset)

    train_loader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=False)

    sample_source = source_dataset[0]
    sample_target = target_dataset[0]
    input_dim = (
        sample_source.shape[-1]
        if isinstance(sample_source, torch.Tensor)
        else len(sample_source)
    )
    output_dim = (
        sample_target.shape[-1]
        if isinstance(sample_target, torch.Tensor)
        else len(sample_target)
    )

    model = MLPRelu(
        input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim
    ).to(device)

    criterion = sw_loss

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "npsgd":
        print("npsgd selected but not implemented; falling back to standard SGD.")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    print("Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    history = train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        num_epochs=args.epochs,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
