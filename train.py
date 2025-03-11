import json
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from loss import KLDiv


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, log_interval=10, mnist=False
):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        criterion (callable): Loss function (e.g., sliced Wasserstein loss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., SGD or its variants).
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number (used for display/logging).
        log_interval (int): Number of batches between progress updates.

    Returns:
        avg_loss (float): Average training loss over the epoch.
        batch_losses (list): List of losses for each batch.
    """
    model.train()
    running_loss = 0.0
    running_KL = 0.0
    batch_losses = []
    batch_KL = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.view(targets.size(0), -1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_losses.append(loss.item())
        kl = KLDiv(outputs, targets)
        running_KL += kl.item()
        batch_KL.append(kl.item())
    avg_loss = running_loss / len(train_loader)
    avg_KL = running_KL / len(train_loader)
    return avg_loss, batch_losses, avg_KL, batch_KL


def evaluate(model, valid_loader, criterion, device, mnist=False):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network model.
        valid_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion (callable): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        avg_loss (float): Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    running_KL = 0.0

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if mnist:
                outputs = outputs.view(outputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            running_KL += KLDiv(outputs, targets).item()

    avg_loss = running_loss / len(valid_loader)
    avg_KL = running_KL / len(valid_loader)
    return avg_loss, avg_KL


def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    num_epochs=50,
    log_interval=10,
    save_dir="checkpoints",
    save_model=False,
    scheduler_factor=0.5,
    scheduler_patience=5,
    scheduler_min_lr=1e-8,
    mnist=False,
):
    """
    Trains the model for a given number of epochs, evaluates on a validation set,
    and saves the best model (based on validation loss) and training history.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        valid_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion (callable): Loss function (e.g., sliced Wasserstein loss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., SGD, Adam, etc.).
        device (torch.device): Device to run training on.
        num_epochs (int): Number of training epochs.
        log_interval (int): Number of batches between progress updates.
        save_dir (str): Directory where checkpoints and history will be saved.

    Returns:
        history (dict): Dictionary containing training and validation losses.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_kl": [],
        "valid_kl": [],
        "lr": [],
    }
    best_valid_loss = float("inf")
    pbar = tqdm(total=num_epochs, desc="Training")
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )
    for epoch in range(1, num_epochs + 1):
        train_loss, _, train_kl, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, log_interval, mnist=mnist
        )
        valid_loss, valid_kl = evaluate(model, valid_loader, criterion, device, mnist=mnist)


        history["train_loss"].append(train_loss)
        history["train_kl"].append(train_kl)
        history["valid_loss"].append(valid_loss)
        history["valid_kl"].append(valid_kl)
        lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(lr)
        pbar.set_description(
            f"Train Loss: {train_loss:.4f} - Train KL: {train_kl:.4f} - Valid Loss: {valid_loss:.4f} - Valid KL: {valid_kl:.4f} - lr: {lr}"
        )
        if save_model:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint_name = f"best_model_epoch{epoch}_valloss{valid_loss:.4f}.pth"
                checkpoint_path = os.path.join(save_dir, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_path)

        scheduler.step(valid_loss)

        pbar.update(1)
    pbar.close()
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    return history
