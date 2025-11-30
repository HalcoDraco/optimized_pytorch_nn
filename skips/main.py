import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataloaders import get_data_loader
from train_functions import train_model
from pruned_model import PruningMLP
from train_utilities import EarlyStopping, BestCheckpointSaver
from rigl_torch.RigL import RigLScheduler

def main(config):

    # Deactivate AMP if not supported by the device
    if config["amp"] and not torch.amp.autocast_mode.is_autocast_available(config["device"]):
        print(f"Automatic Mixed Precision (AMP) is not available on device {config['device']}. Disabling AMP.")
        config["amp"] = False

    if config["use_val"]:
        train_loader, val_loader = get_data_loader(config, train=True)
    else:
        train_loader = get_data_loader(config, train=True)
        val_loader = None

    print("Size of training dataset:", len(train_loader.dataset))
    if val_loader:
        print("Size of validation dataset:", len(val_loader.dataset))

    model = PruningMLP().to(torch.device(config["device"]))
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    # early_stopping = EarlyStopping(patience=5, min_delta=0.01) if config["use_val"] else None
    early_stopping = None

    best_checkpoint_saver = BestCheckpointSaver(save_path="best_cifar_model.pth", min_delta=0.01) if config["use_val"] else None

    warmup_epochs = 10
    scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=config["epochs"] - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    epochs = config["epochs"]
    total_iters = epochs * len(train_loader)
    T_end = int(0.75 * total_iters)

    pruner = RigLScheduler(model,
                           optimizer,
                           dense_allocation=0.1,
                           sparsity_distribution='uniform',
                           T_end=T_end,
                           delta=30,
                           alpha=0.3,
                           grad_accumulation_n=4,
                           static_topo=False,
                           ignore_linear_layers=False,
                           state_dict=None)

    train_model(model, 
                train_loader, 
                optimizer, 
                criterion, 
                config, 
                val_loader=val_loader, 
                early_stopping=early_stopping,
                best_checkpoint_saver=best_checkpoint_saver,
                scheduler=scheduler,
                rigl_scheduler=pruner,
                save_path="final_cifar_model.pth")
    
    del train_loader._iterator
    del val_loader._iterator

if __name__ == '__main__':

    CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 256,
        "num_workers": 4,
        "pin_memory": True,
        "non_blocking": True,
        "persistent_workers": True,
        "none_gradients": True,
        "cudnn_benchmark": True,
        "amp": False,
        "learning_rate": 1e-2,
        "epochs": 50,
        "use_val": True,
        "benchmark": False,
    }

    main(CONFIG)