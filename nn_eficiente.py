import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time
import torch.nn.functional as F

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

def get_data_loader(config: dict, train: bool) -> DataLoader:
    """
    Get DataLoader for MNIST dataset.

    Parameters
    ----------
    config : dict
        Configuration dictionary with DataLoader parameters.
    train : bool
        Whether to load the training or validation dataset.
    
    Returns
    -------
    data_loader : DataLoader
        The DataLoader for the MNIST dataset.
    """
    
    # transform_mnist = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    transfrom_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform_mnist)
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transfrom_cifar10)

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=train,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"]
    )
    return data_loader

def train_model(train_loader, config, val_loader=None, early_stopping=None):
    
    device = torch.device(config["device"])
    scaler = torch.amp.GradScaler(device=config["device"], enabled=config["amp"])
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    print(f"Using device: {device}")

    model = SimpleCifarCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    bmk("Starting training loop")

    for epoch in range(config["epochs"]):

        train_loss = train_epoch(model, train_loader, device, optimizer, criterion, scaler, config)

        if val_loader is not None:
            val_loss = test_model(model, val_loader, device, criterion, config)
            print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if early_stopping is not None and early_stopping(model, val_loss):
                print(f"Early stopping at epoch {epoch+1}, restoring best model...")
                best_state = early_stopping.get_best_state()
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        else:
            print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}")

        bmk(f"Completed epoch {epoch+1}")

    print("Training complete!")

def main(config):

    # Deactivate AMP if not supported by the device
    if config["amp"] and not torch.amp.autocast_mode.is_autocast_available(config["device"]):
        print(f"Automatic Mixed Precision (AMP) is not available on device {config['device']}. Disabling AMP.")
        config["amp"] = False

    bmk("Start")

    train_loader = get_data_loader(config, train=True)
    val_loader = get_data_loader(config, train=False) if config["use_val"] else None

    print("Size of training dataset:", len(train_loader.dataset))
    if val_loader:
        print("Size of validation dataset:", len(val_loader.dataset))

    bmk("DataLoader prepared.")
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    train_model(train_loader, config, val_loader, early_stopping=early_stopping)
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