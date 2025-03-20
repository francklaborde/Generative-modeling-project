# Generative-modeling-project

Implementation of the paper [Convergence of SGD for Training Neural Networks with Sliced Wasserstein Losses](https://arxiv.org/abs/2307.11714)
on the generation of simple 2D distributions and FashionMNIST images.

## Usage

First install dependencies, for example with pyenv and poetry:

```bash
pyenv virtualenv 3.12 swd
pyenv activate swd
pip install poetry
poetry install
```

Then you can train a model, using the following command

```bash
usage: main.py [-h] [--model {mlp,cnn,res_cnn}] [--source_dataset {two_moons,swiss_roll,gaussian,uniform,discrete_points}]
               [--target_dataset {two_moons,swiss_roll,gaussian,fashion_mnist,uniform,discrete_points}] [--num_points NUM_POINTS] [--source_mu SOURCE_MU] [--target_mu TARGET_MU]
               [--source_scale SOURCE_SCALE] [--target_scale TARGET_SCALE] [--dimension DIMENSION] [--source_noise SOURCE_NOISE] [--target_noise TARGET_NOISE] [--low LOW] [--high HIGH]
               [--hidden_layers HIDDEN_LAYERS] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--optimizer {sgd,npsgd}] [--momentum MOMENTUM] [--radius RADIUS]
               [--noise_scale NOISE_SCALE] [--save_dir SAVE_DIR] [--save_model] [--scheduler_factor SCHEDULER_FACTOR] [--scheduler_patience SCHEDULER_PATIENCE]
               [--scheduler_min_lr SCHEDULER_MIN_LR] [--plot_loss] [--loss_projections LOSS_PROJECTIONS] [--data_path DATA_PATH] [--use_notebook]

Train a generator to map a source distribution to a target distribution.

options:
  -h, --help            show this help message and exit
  --model {mlp,cnn,res_cnn}
                        Name of the model to use.
  --source_dataset {two_moons,swiss_roll,gaussian,uniform,discrete_points}
                        Name of the source dataset.
  --target_dataset {two_moons,swiss_roll,gaussian,fashion_mnist,uniform,discrete_points}
                        Name of the target dataset.
  --num_points NUM_POINTS
                        Number of points in each of the training and validation datasets. Must be a power of input dimension for DiscretePointsDataset.
  --source_mu SOURCE_MU
                        Mean of the source Gaussian distribution (applies to gaussian).
  --target_mu TARGET_MU
                        Mean of the target Gaussian distribution (applies to gaussian).
  --source_scale SOURCE_SCALE
                        Scale of the source distribution.
  --target_scale TARGET_SCALE
                        Scale of the target distribution.
  --dimension DIMENSION
                        Dimensionality of the dataset (applies to gaussian).
  --source_noise SOURCE_NOISE
                        Noise level for the source dataset (if applicable).
  --target_noise TARGET_NOISE
                        Noise level for the target dataset (if applicable).
  --low LOW             Lower bound for the uniform or discrete distribution.
  --high HIGH           Upper bound for the uniform or discrete distribution.
  --hidden_layers HIDDEN_LAYERS
                        Comma separated list of hidden layer sizes for the generator.
  --epochs EPOCHS       Number of training epochs.
  --batch_size BATCH_SIZE
                        Batch size for training.
  --lr LR               Learning rate.
  --optimizer {sgd,npsgd}
                        Optimizer to use: 'sgd' or 'npsgd' (npsgd is a placeholder for now).
  --momentum MOMENTUM   Momentum of SGD.
  --radius RADIUS       Radius of the projection ball for NoisedProjectedSGD.
  --noise_scale NOISE_SCALE
                        Scale of the noise for NoisedProjectedSGD.
  --save_dir SAVE_DIR   Directory to save model checkpoints and training history.
  --save_model          Save the final model in addition to the best model.
  --scheduler_factor SCHEDULER_FACTOR
                        Factor by which to reduce learning rate on plateau.
  --scheduler_patience SCHEDULER_PATIENCE
                        Number of epochs with no improvement after which to reduce learning rate.
  --scheduler_min_lr SCHEDULER_MIN_LR
                        Minimum learning rate for the scheduler.
  --plot_loss           Plot the training and validation loss after training.
  --loss_projections LOSS_PROJECTIONS
                        Number of random projections for the SWD loss.
  --data_path DATA_PATH
                        Path to the directory where the dataset will be stored.
  --use_notebook        Use the notebook backend for matplotlib.
  ```
