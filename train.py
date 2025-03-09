import json
import os

import torch
from tqdm import tqdm


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, log_interval=10
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
    batch_losses = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_losses.append(loss.item())

    avg_loss = running_loss / len(train_loader)
    return avg_loss, batch_losses


def evaluate(model, valid_loader, criterion, device):
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

    with torch.no_grad():
        for inputs, targets in tqdm(valid_loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    avg_loss = running_loss / len(valid_loader)
    return avg_loss


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

    history = {"train_loss": [], "valid_loss": []}
    best_valid_loss = float("inf")
    pbar = tqdm(total=num_epochs, desc="Training")
    for epoch in range(1, num_epochs + 1):
        train_loss, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, log_interval
        )
        valid_loss = evaluate(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        pbar.set_description(
            f"Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}"
        )
        if save_model:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint_name = f"best_model_epoch{epoch}_valloss{valid_loss:.4f}.pth"
                checkpoint_path = os.path.join(save_dir, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")

        pbar.update()
    pbar.close()
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    return history
