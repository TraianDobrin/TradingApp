from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    @abstractmethod
    def _step(self, batch, train=True):
        """Run one forward/backward pass. Must be implemented in subclass."""
        pass


