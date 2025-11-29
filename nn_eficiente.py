import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from time import time
import torch.nn.functional as F
import numpy as np

CIFAR10_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

class Bmk:
    """
    Benchmarking utility class.
    
    Methods
    -------
    set(work: bool)
        Configure whether benchmarking is active.
    __call__(msg: str = "")
        Record a timestamp with an optional message.
    report()
        Print the benchmark report.
    reset()
        Reset the recorded timestamps.
    """

    def __init__(self):
        """Initialize the benchmark utility."""
        self.__work: bool = None
        self.__times: list[tuple[float, str]] = []
    
    def set(self, work: bool):
        if self.__work is None:
            self.__work = work
        else:
            raise RuntimeError("Benchmark has already been configured.")
    
    def __call__(self, msg: str = ""):
        if self.__work is None:
            raise RuntimeError("Benchmark has not been configured yet.")
        elif self.__work:
            self.__times.append((time(), msg))

    def report(self):
        if self.__work:
            print("Benchmark report:")
            for i in range(1, len(self.__times)):
                start_time, start_msg = self.__times[i-1]
                end_time, end_msg = self.__times[i]
                elapsed = end_time - start_time
                print(f"  {start_msg} -> {end_msg}: {elapsed:.4f} seconds")
            print("")
            print(f"Total time: {self.__times[-1][0] - self.__times[0][0]:.4f} seconds")
    
    def reset(self):
        self.__times = []


bmk = Bmk()

class SimpleMnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)
    
class SimpleCifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.network(x)
    
class SimpleCifarMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)
    
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_state = None
        self.counter = 0

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def get_best_state(self):
        return self.best_state
    
class BestCheckpointSaver:
    def __init__(self, save_path: str, min_delta: float = 0.0, restore_best: bool = True):
        self.restore_best = restore_best
        self.save_path = save_path
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_state = None

    def __call__(self, model: nn.Module, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            torch.save(self.best_state, self.save_path)
    
def train_epoch(model: nn.Module, train_loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, criterion: nn.Module, scaler: torch.amp.GradScaler, config: dict) -> float:
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
    for batch_idx, (data, target) in enumerate(train_loader):
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

def get_data_loader(config: dict, train: bool, batch_size: int = None, shuffle: bool = None) -> DataLoader:
    """
    Get DataLoader for CIFAR dataset.

    Parameters
    ----------
    config : dict
        Configuration dictionary with DataLoader parameters.
    train : bool
        Whether to get the training or test DataLoader.
    batch_size : int, optional
        The batch size for the DataLoader.
    shuffle : bool, optional
        Whether to shuffle the dataset.
    
    Returns
    -------
    data_loader : DataLoader
        The DataLoader for the CIFAR dataset.
    """

    if batch_size is None:
        batch_size = config["batch_size"]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR10_NORMALIZATION)
    ])

    train_transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR10_NORMALIZATION)
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                                  download=True, transform=train_transform)
    full_val_dataset   = datasets.CIFAR10(root='./data', train=True, 
                                                    download=True, transform=test_transform)
    test_dataset       = datasets.CIFAR10(root='./data', train=False, 
                                                    download=True, transform=test_transform)
    
    if not train:
        dataset = test_dataset
    elif train and config["use_val"]:
        val_proportion = 0.15
        dataset_size = len(full_train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_proportion * dataset_size))
        
        # Shuffle indices manually with a fixed seed
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]

        # Create Subsets using the specific transforms
        dataset = Subset(full_train_dataset, train_indices)
        val_ds  = Subset(full_val_dataset, val_indices)
    else:
        dataset = full_train_dataset

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if shuffle is not None else train,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"]
    )

    if train and config["use_val"]:
        val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            persistent_workers=config["persistent_workers"]
        )

        return data_loader, val_dataloader

    return data_loader

def train_model(model, 
                train_loader, 
                optimizer, 
                criterion, 
                config, 
                val_loader=None, 
                early_stopping: EarlyStopping=None,
                best_checkpoint_saver: BestCheckpointSaver=None, 
                scheduler=None,
                save_path=None):
    
    device = torch.device(config["device"])
    scaler = torch.amp.GradScaler(device=config["device"], enabled=config["amp"])
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    print(f"Using device: {device}")

    bmk("Starting training loop")

    for epoch in range(config["epochs"]):

        train_loss = train_epoch(model, train_loader, device, optimizer, criterion, scaler, config)

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

        bmk(f"Completed epoch {epoch+1}")

    print("Training complete!")

    if best_checkpoint_saver is not None and best_checkpoint_saver.restore_best:
        print("Restoring best model from checkpoint...")
        model.load_state_dict(best_checkpoint_saver.best_state)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model

def main(config):

    # Deactivate AMP if not supported by the device
    if config["amp"] and not torch.amp.autocast_mode.is_autocast_available(config["device"]):
        print(f"Automatic Mixed Precision (AMP) is not available on device {config['device']}. Disabling AMP.")
        config["amp"] = False

    bmk("Start")

    if config["use_val"]:
        train_loader, val_loader = get_data_loader(config, train=True)
    else:
        train_loader = get_data_loader(config, train=True)
        val_loader = None

    print("Size of training dataset:", len(train_loader.dataset))
    if val_loader:
        print("Size of validation dataset:", len(val_loader.dataset))

    bmk("DataLoader prepared.")
    model = SimpleCifarCNN().to(torch.device(config["device"]))
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=5, min_delta=0.01) if config["use_val"] else None

    best_checkpoint_saver = BestCheckpointSaver(save_path="best_cifar_model.pth", min_delta=0.01) if config["use_val"] else None

    warmup_epochs = 10
    scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=config["epochs"] - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    train_model(model, 
                train_loader, 
                optimizer, 
                criterion, 
                config, 
                val_loader=val_loader, 
                early_stopping=early_stopping,
                best_checkpoint_saver=best_checkpoint_saver,
                scheduler=scheduler,
                save_path="final_cifar_model.pth")
    
    bmk("Training complete.")
    
    bmk("Deleting DataLoaders.")
    del train_loader._iterator
    del val_loader._iterator
    bmk("DataLoaders deleted.")

if __name__ == '__main__':

    # configs_gridsearch = {
    #     "device": ["cuda"],
    #     "batch_size": [128],
    #     "num_workers": [8],
    #     "pin_memory": [False, True],
    #     "non_blocking": [False, True],
    #     "persistent_workers": [True],
    #     "none_gradients": [False, True],
    #     "cudnn_benchmark": [False, True],
    #     "amp": [False, True],
    #     "learning_rate": [1e-4],
    #     "epochs": [20],
    #     "use_val": [True],
    #     "benchmark": [False],
    # }

    # from itertools import product

    # cartesian_product_dicts = [dict(zip(configs_gridsearch.keys(), values)) for values in product(*configs_gridsearch.values())]
    # times = {}
    # bmk.set(False)
    # reps = 3

    # print("Warming up...")
    # main(cartesian_product_dicts[-1])  # Warm-up run
    # print("Starting grid search...")

    # for config in cartesian_product_dicts:
    #     print(f"Running configuration: {config}")
    #     times[str(config)] = []
    #     for i in range(reps):
    #         init_time = time()
    #         main(config)
    #         end_time = time()
    #         times[str(config)].append(end_time - init_time)
    #         print(end_time - init_time)
    
    # for config in times:
    #     avg_time = sum(times[config]) / len(times[config])
    #     print(f"Config: {config}, Times: {times[config]}, Average Time: {avg_time:.4f}")


    CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 256,
        "num_workers": 8,
        "pin_memory": True,
        "non_blocking": True,
        "persistent_workers": True,
        "none_gradients": True,
        "cudnn_benchmark": True,
        "amp": False,
        "learning_rate": 1e-4,
        "epochs": 50,
        "use_val": True,
        "benchmark": True,
    }

    bmk.set(CONFIG["benchmark"])
    main(CONFIG)
    bmk.report()