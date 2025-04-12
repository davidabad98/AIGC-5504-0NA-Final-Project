import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class VAETrainer:
    """
    Trainer class to handle the training of the Variational Autoencoder.

    Args:
        model (nn.Module): The VAE model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_dataset (TensorDataset): Training dataset.
        val_dataset (TensorDataset, optional): Validation dataset for tracking performance.
        device (torch.device): Device to perform training on.
        config (dict): Dictionary containing training parameters.
    """
    def __init__(self, model, optimizer, train_dataset, val_dataset=None, device=None, config=None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device if device else torch.device("cpu")
        self.config = config if config else {}
        self.epochs = self.config.get("epochs", 50)
        self.batch_size = self.config.get("batch_size", 64)
        self.checkpoint_path = self.config.get("checkpoint_path", "vae_checkpoint.pth")

        self.model.to(self.device)

    def train(self):
        """
        Executes the training loop over the dataset.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        best_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                # VAE operates in an unsupervised manner; we only need the features.
                x = batch[0].to(self.device)
                self.optimizer.zero_grad()

                x_recon, mu, logvar = self.model(x)
                loss_dict = self.model.loss_function(x_recon, x, mu, logvar)
                loss = loss_dict["total_loss"]
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch}/{self.epochs}], Train Loss: {avg_train_loss:.4f}")

            # Validate if a validation dataset is provided.
            if self.val_dataset is not None:
                val_loss = self.validate()
                print(f"Epoch [{epoch}/{self.epochs}], Validation Loss: {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(epoch, best_loss)

    def validate(self):
        """
        Evaluates model performance on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                x_recon, mu, logvar = self.model(x)
                loss_dict = self.model.loss_function(x_recon, x, mu, logvar)
                loss = loss_dict["total_loss"]
                val_loss += loss.item()

        return val_loss / len(val_loader.dataset)

    def save_checkpoint(self, epoch, loss):
        """
        Saves the model checkpoint to disk.

        Args:
            epoch (int): Epoch number.
            loss (float): Validation loss at checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")
