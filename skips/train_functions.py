import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_utilities import EarlyStopping, BestCheckpointSaver
from tqdm import tqdm

def train_epoch(model: nn.Module, train_loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, criterion: nn.Module, scaler: torch.amp.GradScaler, config: dict, rigl_scheduler = None) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    device : torch.device
        The device to run the training on.
    optimizer : torch.optim.Optimizer
        The optimizer for updating model parameters.
    criterion : nn.Module
        The loss function.
    scaler : torch.amp.GradScaler
        The GradScaler for AMP.
    config : dict
        Configuration dictionary.

    Returns
    -------
    avg_loss : float
        The average loss over the epoch.
    """

    loss_sum = 0.0

    model.train()
    #use tqdm
    for (data, target) in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device, non_blocking=config["non_blocking"]), target.to(device, non_blocking=config["non_blocking"])
        
        optimizer.zero_grad(set_to_none=config["none_gradients"])

        with torch.amp.autocast(device_type=config["device"], enabled=config["amp"]):
            output = model(data)
            loss = criterion(output, target)

        if config["amp"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if rigl_scheduler is not None and rigl_scheduler():
                optimizer.step()

        loss_sum += loss.item()

    return loss_sum / len(train_loader)

def test_model(model: nn.Module, test_loader: DataLoader, device: torch.device, criterion: nn.Module, config: dict) -> float:
    """
    Test the model on the test (or val) dataset.

    Parameters
    ----------
    model : nn.Module
        The neural network model to test.
    test_loader : DataLoader
        DataLoader for the test (or val) dataset.
    device : torch.device
        The device to run the testing on.
    criterion : nn.Module
        The loss function.
    config : dict
        Configuration dictionary.

    Returns
    -------
    avg_loss : float
        The average loss on the test (or val) dataset.
    """

    loss_sum = 0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, non_blocking=config["non_blocking"]), target.to(device, non_blocking=config["non_blocking"])

            with torch.amp.autocast(device_type=config["device"], enabled=config["amp"]):
                output = model(data)
                loss = criterion(output, target)

            loss_sum += loss.item()

    return loss_sum / len(test_loader)

def train_model(model, 
                train_loader, 
                optimizer, 
                criterion, 
                config, 
                val_loader=None, 
                early_stopping: EarlyStopping=None,
                best_checkpoint_saver: BestCheckpointSaver=None, 
                scheduler=None,
                rigl_scheduler=None,
                save_path=None):
    
    device = torch.device(config["device"])
    scaler = torch.amp.GradScaler(device=config["device"], enabled=config["amp"])
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    print(f"Using device: {device}")

    for epoch in range(config["epochs"]):

        train_loss = train_epoch(model, 
                                 train_loader, 
                                 device, 
                                 optimizer, 
                                 criterion, 
                                 scaler, 
                                 config, 
                                 rigl_scheduler=rigl_scheduler)

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            val_loss = test_model(model, val_loader, device, criterion, config)
            print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if early_stopping is not None and early_stopping(model, val_loss):
                print(f"Early stopping at epoch {epoch+1}, restoring best model...")
                best_state = early_stopping.get_best_state()
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

            if best_checkpoint_saver is not None:
                best_checkpoint_saver(model, val_loss)
        else:
            print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}")

    print("Training complete!")

    if best_checkpoint_saver is not None and best_checkpoint_saver.restore_best:
        print("Restoring best model from checkpoint...")
        model.load_state_dict(best_checkpoint_saver.best_state)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model