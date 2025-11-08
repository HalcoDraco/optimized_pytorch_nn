import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time

class Bmk:
    def __init__(self):
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

class SimpleCNN(nn.Module):
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
    
def train_epoch(model, train_loader, device, optimizer, criterion):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

    return loss

def get_data_loader(config, train):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=train,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=True
    )
    return data_loader

def train_model(train_loader, config):
    
    device = torch.device(config["device"])
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    bmk("Starting training loop")

    for epoch in range(config["epochs"]):

        loss = train_epoch(model, train_loader, device, optimizer, criterion)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.4f}")
        bmk(f"Completed epoch {epoch+1}")

    print("Training complete!")

def main(config):
    bmk("Start")
    train_loader = get_data_loader(config, train=True)
    bmk("DataLoader prepared.")
    train_model(train_loader, config)
    bmk("Training complete.")

if __name__ == '__main__':
    CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 256,
        "num_workers": 8,
        "pin_memory": False,
        "learning_rate": 1e-4,
        "epochs": 15,
        "benchmark": True
    }
    bmk.set(CONFIG["benchmark"])
    main(CONFIG)
    bmk.report()