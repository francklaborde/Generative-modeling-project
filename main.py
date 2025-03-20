import argparse

import torch
from torch.utils.data import DataLoader

from datasets import PairedDataset, make_dataset
from loss import SWDLoss
from nets import make_model
from optim import NoisedProjectedSGD
from plot import plot_loss, plot_model_results
from train import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a generator to map a source distribution to a target distribution."
    )
    # Dataset parameters
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "res_cnn"],
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="gaussian",
        choices=["two_moons", "swiss_roll", "gaussian", "uniform", "discrete_points"],
        help="Name of the source dataset.",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="two_moons",
        choices=[
            "two_moons",
            "swiss_roll",
            "gaussian",
            "fashion_mnist",
            "uniform",
            "discrete_points",
        ],
        help="Name of the target dataset.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1000,
        help="Number of points in each of the training and validation datasets. Must be a power of input dimension for DiscretePointsDataset.",
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
        "--source_scale",
        type=float,
        default=1.0,
        help="Scale of the source distribution.",
    )
    parser.add_argument(
        "--target_scale",
        type=float,
        default=1.0,
        help="Scale of the target distribution.",
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
        default=1.0,
        help="Noise level for the source dataset (if applicable).",
    )
    parser.add_argument(
        "--target_noise",
        type=float,
        default=0.1,
        help="Noise level for the target dataset (if applicable).",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=0.0,
        help="Lower bound for the uniform or discrete distribution.",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=1.0,
        help="Upper bound for the uniform or discrete distribution.",
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
        "--momentum",
        type=float,
        default=0.0,
        help="Momentum of SGD.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Radius of the projection ball for NoisedProjectedSGD.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1,
        help="Scale of the noise for NoisedProjectedSGD.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints and training history.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save the final model in addition to the best model.",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.5,
        help="Factor by which to reduce learning rate on plateau.",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement after which to reduce learning rate.",
    )
    parser.add_argument(
        "--scheduler_min_lr",
        type=float,
        default=1e-8,
        help="Minimum learning rate for the scheduler.",
    )
    parser.add_argument(
        "--plot_loss",
        action="store_true",
        default=False,
        help="Plot the training and validation loss after training.",
    )
    parser.add_argument(
        "--loss_projections",
        type=int,
        default=100,
        help="Number of random projections for the SWD loss.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to the directory where the dataset will be stored.",
    )
    parser.add_argument(
        "--use_notebook",
        action="store_true",
        default=False,
        help="Use the notebook backend for matplotlib.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hidden_layers = [
        int(x.strip()) for x in args.hidden_layers.split(",") if x.strip().isdigit()
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.source_dataset == "gaussian":
        source_dataset = make_dataset(
            args.source_dataset,
            num_samples=args.num_points,
            mu=args.source_mu,
            sigma=args.source_noise,
            dim=args.dimension,
        )
    elif args.target_dataset == "fashion_mnist":
        source_dataset = make_dataset(
            "gaussian",
            num_samples=args.num_points,
            mu=args.source_mu,
            sigma=args.source_noise,
            dim=args.dimension,
        )
    elif args.source_dataset == "uniform" or args.source_dataset == "discrete_points":
        source_dataset = make_dataset(
            args.source_dataset,
            num_samples=args.num_points,
            low=args.low,
            high=args.high,
            dim=args.dimension,
        )
    else:
        source_dataset = make_dataset(
            args.source_dataset,
            num_samples=args.num_points,
            noise=args.source_noise,
            scale=args.source_scale,
        )

    if args.target_dataset == "gaussian":
        target_dataset = make_dataset(
            args.target_dataset,
            num_samples=args.num_points,
            mu=args.target_mu,
            sigma=args.target_noise,
            dim=args.dimension,
        )
    elif args.source_dataset == "uniform" or args.source_dataset == "discrete_points":
        target_dataset = make_dataset(
            args.target_dataset,
            num_samples=args.num_points,
            low=args.low,
            high=args.high,
            dim=args.dimension,
        )
    elif args.target_dataset == "fashion_mnist":
        target_dataset = make_dataset(
            args.target_dataset, num_samples=args.num_points, data_path=args.data_path
        )
    else:
        target_dataset = make_dataset(
            args.target_dataset,
            num_samples=args.num_points,
            noise=args.target_noise,
            scale=args.target_scale,
        )

    paired_dataset = PairedDataset(source_dataset, target_dataset)

    train_loader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=False)
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
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    if args.target_dataset == "fashion_mnist":
        assert args.model in [
            "cnn",
            "res_cnn",
        ], "Only CNN and generator models are supported for Fashion MNIST"
        model = make_model(args.model, input_dim, 1)
        mnist = True
    else:
        model = make_model("mlp", input_dim, output_dim, hidden_layers=hidden_layers)
        mnist = False
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    model.to(device)

    criterion = SWDLoss(num_projections=args.loss_projections)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "npsgd":
        optimizer = NoisedProjectedSGD(
            model.parameters(),
            lr=args.lr,
            radius=args.radius,
            noise_scale=args.noise_scale,
        )
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
        save_dir=args.save_dir,
        save_model=args.save_model,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        mnist=mnist,
        use_notebook=args.use_notebook,
    )

    plot_model_results(
        model,
        source_dataset,
        target_dataset,
        device,
        title="Distribution Comparison",
        filename=f"{args.save_dir}/results_{args.source_dataset}_to_{args.target_dataset}_{args.dimension}d.png",
        mnist=mnist,
    )

    if args.plot_loss:
        plot_loss(
            f"{args.save_dir}/training_history.json",
            filename=f"{args.save_dir}/loss.png",
        )


if __name__ == "__main__":
    main()
