import torch
import torch.nn as nn

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