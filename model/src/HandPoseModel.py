import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
import json
import os
import src.helpers as helpers

BONE_PAIRS = helpers.read_hand_keypoints()


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, path: str = None):
        """Implements early stopping to prevent overfitting.

        Args:
            patience (int, optional): Number of epochs to wait before stopping if no improvement. Defaults to 10.
            min_delta (float, optional): Minimum change in validation loss to be considered an improvement. Defaults to 0.001.
            path (str, optional): Path to save the best model. Defaults to None.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(
        self, val_loss: float, model: torch.nn.Module, epoch: int, tuning: bool
    ) -> bool:
        """
        Call this function at the end of each epoch to check if training should stop.

        Args:
            val_loss (float): Current epoch validation loss
            model (torch.nn.Module): The model being trained

        Returns:
            bool: Whether training should stop.
        """

        if val_loss < self.best_loss - self.min_delta:
            print(
                f"Model improved with val loss {val_loss} vs. previous {self.best_loss}. Saving new version."
            )
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(self.path, "best_model.pth"))
            with open(os.path.join(self.path, "checkpoint_stats.json"), "w") as f:
                json.dump(
                    {
                        "val_loss": self.best_loss,
                        "epoch": epoch,
                        "plateu": self.counter,
                    },
                    f,
                )

        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"No model improvement for {self.counter} epochs. Stopping training.")
            return True
        return False


class HandPoseModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        lr: float = 0.001,
        batch_size: int = 32,
        test_split: float = 0.1,
        val_split: float = 0.1,
        train_loss="MSE",
        eval_loss="R2",  # Allows use of a consistent evaluation loss (e.g. if comparing different losses in training)
        train_transforms: transforms.Compose | None = None,
        inference_transforms: transforms.Compose | None = None,
        early_stopping: EarlyStopping = EarlyStopping,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0,
        early_stopping_path: str | None = None,
    ):
        """
        Wrapper for PyTorch model with integrated data management
        for hand landmark detection.

        Args:
            model (nn.Module): _description_
            dataset (torch.utils.data.Dataset): _description_
            lr (float, optional): _description_. Defaults to 0.001.
            batch_size (int, optional): _description_. Defaults to 32.
            test_split (float, optional): _description_. Defaults to 0.1.
            val_split (float, optional): _description_. Defaults to 0.1.
        """

        super(HandPoseModel, self).__init__()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6
        )
        self.train_transforms = train_transforms
        self.inference_transforms = inference_transforms
        self.start_epoch = 0

        self.scaler = torch.amp.GradScaler(
            self.device.type
        )  # torch.amp for mixed precision training. Might not be supported on older GPUs

        self.batch_size = batch_size
        self.early_stopping = early_stopping(
            early_stopping_patience, early_stopping_delta, early_stopping_path
        )

        # Allows for testing several loss functions.
        LOSS_FUNCTIONS = {"MSE": self.mse_loss, "R2": self.r2_score}

        self.train_loss_func, self.train_loss_name = (
            LOSS_FUNCTIONS.get(train_loss),
            train_loss,
        )
        self.eval_loss_func, self.eval_loss_name = (
            LOSS_FUNCTIONS.get(eval_loss),
            eval_loss,
        )

        # Split dataset
        total_size = len(dataset)
        test_size = int(test_split * total_size)
        val_size = int(val_split * total_size)
        train_size = total_size - test_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        self.set_loaders()

        self.model.to(self.device)

        if self.train_transforms:
            self.train_transforms.to(self.device)

        if self.inference_transforms:
            self.inference_transforms.to(self.device)

    def set_loaders(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def fit(self, num_epochs=10):
        """
        Trains the model on its given dataset.

        Args:
            num_epochs (int, optional): Number of epochs. Defaults to 10.
        """

        writer = SummaryWriter(log_dir="./logs", purge_step=self.start_epoch)

        for epoch in range(1 + self.start_epoch, num_epochs + self.start_epoch + 1):
            epoch_progress = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch}/{num_epochs + self.start_epoch}",
                leave=True,
            )
            self.model.train()
            train_loss = 0
            train_eval = 0

            for batch_idx, batch in epoch_progress:
                images, landmarks = batch["image"].to(self.device), batch[
                    "landmarks"
                ].to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(self.device.type):
                    images = self.train_transforms(images)
                    outputs = self.model(images)  # Prediction

                    loss = self.train_loss_func(outputs, landmarks)
                    eval = self.eval_loss_func(outputs, landmarks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()
                train_eval += eval.item()

                # batch progress
                epoch_progress.set_postfix(batch=batch_idx + 1, loss=loss.item())

            # Compute validation loss
            val_loss, val_eval = self.evaluate(self.val_loader)
            epoch_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step(val_loss)
            train_loss /= len(self.train_loader)
            train_eval /= len(self.train_loader)

            writer.add_scalar(
                "Training/learning_rate",
                epoch_lr,
                epoch,
            )
            writer.add_scalars(
                f"Loss/{self.train_loss_name}",
                {"train": train_loss, "val": val_loss},
                epoch,
            )

            writer.add_scalars(
                f"Loss/{self.eval_loss_name}",
                {"train": train_eval, "val": val_eval},
                epoch,
            )

            print(
                f"Epoch {epoch}/{num_epochs + self.start_epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if self.early_stopping(val_loss, self.model, epoch):
                break

    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluates the model for a given dataloader.

        Args:
            data_loader (DataLoader): Validation or test data

        Returns:
            tuple[float]: Computed loss and eval loss
        """

        self.model.eval()
        total_loss = 0
        eval_loss = 0

        # Disable dataset augmentation
        self.change_augmentation_flag(to_val=False)

        with torch.no_grad():
            eval_progress = tqdm(
                data_loader,
                total=len(data_loader),
                desc=f"Evaluating model...",
                leave=True,
            )
            for batch in eval_progress:
                images, landmarks = batch["image"].to(self.device), batch[
                    "landmarks"
                ].to(self.device)

                images = self.inference_transforms(images)
                outputs = self.model(images)
                landmarks = landmarks.view(-1, 63)

                loss = self.train_loss_func(outputs, landmarks)
                eval = self.eval_loss_func(outputs, landmarks)

                total_loss += loss.item()
                eval_loss += eval.item()

        # Enable augmentation again
        self.change_augmentation_flag(to_val=True)

        return (
            total_loss / len(data_loader),
            eval_loss / len(data_loader),
        )

    def evaluate_per_landmark(self, data_loader: DataLoader) -> torch.Tensor:
        """Computes loss per landmark over the entire dataset for a given loss function."""

        self.model.eval()
        self.change_augmentation_flag(to_val=False)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            eval_progress = tqdm(
                data_loader,
                total=len(data_loader),
                desc=f"Evaluating model...",
                leave=True,
            )
            for batch in eval_progress:
                images = batch["image"].to(self.device)
                landmarks = batch["landmarks"].to(self.device)

                images = self.inference_transforms(images)
                outputs = self.model(images)

                all_preds.append(outputs.view(-1, 21, 3))
                all_targets.append(landmarks.view(-1, 21, 3))

        self.change_augmentation_flag(to_val=True)

        # Concatenate all predictions and targets
        preds = torch.cat(all_preds, dim=0)  # Shape: (N, 21, 3)
        targets = torch.cat(all_targets, dim=0)

        return (
            self.mse_per_landmark(preds, targets),
            self.r2_score_per_landmark(preds, targets),
        )

    def test(self):
        """
        Evaluates the model on the test set
        """
        test_loss, eval_test = self.evaluate(self.test_loader)
        return test_loss, eval_test

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Runs inference on new images

        Args:
            images (torch.Tensor): Input images

        Returns:
            torch.Tensor: Predicted hand landmark coordinates
        """

        self.model.eval()
        images = images.to(self.device)
        with torch.no_grad():
            images = self.inference_transforms(images)
            outputs = self.model(images)
        return outputs.cpu().numpy()

    def forward(self, x):
        x = self.model(x)
        return x

    def change_augmentation_flag(self, to_val=False):
        """Changes the dataset's augmentation flag, allowing to switch augmentation on and off.

        Args:
            to_val (bool, optional): Bool indicating if augmentation should be set to True or False. Defaults to False.
        """
        ds = self.train_dataset
        while isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        ds.apply_augmentation = to_val

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def mse_per_landmark(self, pred, target):
        pred = pred.view(-1, 21, 3)
        target = target.view(-1, 21, 3)

        mse = ((pred - target) ** 2).mean(dim=(0, 2))  # Mean over batch and coords
        return mse  # Shape: (21,)

    def r2_score(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes R² for a given tensor of predictions and targets

        Args:
            pred (torch.Tensor): A tensor with the predicted values
            target (torch.Tensor): A tensor with the target values

        Returns:
            torch.Tensor: The R² value contained in a tensor
        """
        pred = pred.view(-1, 21, 3)
        target = target.view(-1, 21, 3)
        ss_res = ((target - pred) ** 2).sum(dim=0)
        ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0)

        r2 = 1 - ss_res / (ss_tot + 1e-8)
        r2_mean = r2.mean()

        return r2_mean

    def r2_score_per_landmark(self, pred, target):
        """Computes R² for a given tensor of predictions and targets
        for each landmark

        Args:
            pred (torch.Tensor): A tensor with the predicted values
            target (torch.Tensor): A tensor with the target values

        Returns:
            torch.Tensor: The R² value contained in a tensor
        """

        pred = pred.view(-1, 21, 3)
        target = target.view(-1, 21, 3)

        # Sum over batch and coordinate dims
        ss_res = ((target - pred) ** 2).sum(
            dim=(0, 2)
        )  # Residual sum of squares per landmark
        ss_tot = ((target - target.mean(dim=0)) ** 2).sum(
            dim=(0, 2)
        )  # Total sum of squares per landmark

        r2_per_landmark = 1 - ss_res / (ss_tot + 1e-8)  # Avoid div by zero
        return r2_per_landmark  # Shape: (21,)

    def set_head(self, head: torch.nn.Sequential, layer_name="fc"):
        if not hasattr(self.model, layer_name):
            raise AttributeError(f"Model has no layer named '{layer_name}'")
        setattr(self.model, layer_name, head.to(self.device))
