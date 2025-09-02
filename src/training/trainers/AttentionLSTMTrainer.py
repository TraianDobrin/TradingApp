import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.training.trainers.Trainer import Trainer  # abstract base class
import joblib
class AttentionLSTMTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, X, Y, Z,  symbol,  device="cpu", test_size=0.2, batch_size=64):
        super().__init__(model, optimizer, criterion, device)
        self.test_size = test_size
        self.batch_size = batch_size
        self.scaler = joblib.load("../../data/scalers/" + symbol + ".pkl")

        # These will be set after split()
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.Z_train = None
        self.Z_test = None
        self.split(X, Y, Z)

    def split(self, X, Y, Z):
        """
        Split datasets into training and testing sets.
        Randomly sampled across all time windows.
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.Z_train, self.Z_test = train_test_split(
            X, Y, Z, test_size=self.test_size, shuffle=True
        )

    def _step(self, batch, train=True):
        """
        Perform a single step (forward + optional backward) on a batch.
        """
        target_x, other_x, y = batch
        target_x, other_x, y = target_x.to(self.device), other_x.to(self.device), y.to(self.device)

        preds, attn_weights = self.model(target_x, other_x)
        loss = self.criterion(preds, y)

        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss, preds

    def _get_loader(self, train=True):
        """
        Create DataLoader for train or test set.
        """
        if train:
            dataset = TensorDataset(self.X_train, self.Z_train, self.Y_train)
        else:
            dataset = TensorDataset(self.X_test, self.Z_test, self.Y_test)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

    def train_epoch(self):
        self.model.train()
        loader = self._get_loader(train=True)  # get train DataLoader
        epoch_loss = 0.0
        for batch in loader:
            loss = self._step(batch, train=True)
            epoch_loss += loss[0].item() * batch[0].size(0)
        return epoch_loss / len(loader.dataset)

    def validate(self):
        """
        Evaluate the model on the test set.
        """
        self.model.eval()
        loader = self._get_loader(train=False)
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in loader:
                loss, preds = self._step(batch, train=False)
                total_loss += loss.item() * batch[0].size(0)
                all_preds.append(preds.cpu())
                all_targets.append(batch[2].cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        print(self.scaler)
        return np.mean(np.abs(self.scaler.inverse_transform(all_preds) - self.scaler.inverse_transform(all_targets))), all_preds, all_targets

    def fit(self, epochs=100, verbose = False, overfit_check = 0.0):
        overfit_check *= 1.0
        """
        Full training loop.
        """
        val_loss = 0.0
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            if (overfit_check * epochs) == int(overfit_check * epochs):
                val_loss, _, _ = self.validate()
            if verbose:
                print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
