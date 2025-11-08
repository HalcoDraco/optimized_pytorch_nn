import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timeit import timeit

# -- 2. DEFINICIÓN DEL MODELO --
# Una CNN simple y eficiente para MNIST
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
    
def train_epoch(model, train_loader, device, optimizer, criterion, scaler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # Limpiar gradientes (optimización menor)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass con autocast para AMP
        with torch.amp.autocast("cuda", enabled=(device.type == 'cuda')):
            output = model(data)
            loss = criterion(output, target)

        # Backward pass usando el scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

# -- 3. FUNCIÓN DE ENTRENAMIENTO --
def train_model(train_loader, config):
    """Función principal que encapsula todo el proceso."""
    
    # -- OPTIMIZACIONES DE CUDA --
    device = torch.device(config["device"])
    # Permite a cuDNN encontrar los mejores algoritmos para tu hardware
    torch.backends.cudnn.benchmark = True
    print(f"Usando dispositivo: {device}")

    # -- INICIALIZACIÓN DE COMPONENTES --
    model = SimpleCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # -- OPTIMIZACIÓN: Automatic Mixed Precision (AMP) --
    # GradScaler ayuda a evitar problemas de underflow con gradientes pequeños en float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == 'cuda'))

    # -- BUCLE DE ENTRENAMIENTO --
    for epoch in range(config["epochs"]):

        loss = train_epoch(model, train_loader, device, optimizer, criterion, scaler)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.4f}")

    print("¡Entrenamiento completado!")

def main(config):
    train_loader = get_data_loader(config, train=True)
    train_model(train_loader, config)

if __name__ == '__main__':
    CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 256,
        "num_workers": 8,  # Ajusta según los núcleos de tu CPU
        "pin_memory": True,
        "learning_rate": 1e-4,
        "epochs": 15,
    }

    execution_time = timeit(lambda: main(CONFIG), number=2)
    print(f"Tiempo de ejecución total: {execution_time:.4f} segundos")