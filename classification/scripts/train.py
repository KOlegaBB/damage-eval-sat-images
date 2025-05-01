import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score


# --------- Utility Functions ---------
def calculate_metrics(y_true, y_pred):
    """
    Calculate classification evaluation metrics: precision, recall, F1 score, and accuracy.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.

    Returns:
        tuple: A tuple containing:
            - precision (float): Weighted average precision score.
            - recall (float): Weighted average recall score.
            - f1 (float): Weighted average F1 score.
            - accuracy (float): Overall accuracy score.
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


# --------- Siamese ResNet Training ---------
def train_siamese_model(
        model, train_loader, valid_loader, loss_fn, optimizer,
        lr_scheduler=None,
        epochs=12, device="cuda", mode='classification', start_save_epoch=0,
        save_path="", save_interval=10
):
    """
    Train a Siamese network using a given dataset and compute training/validation metrics.

    Args:
        model (torch.nn.Module): Siamese model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        loss_fn (callable): Loss function for training (e.g., CrossEntropyLoss or MSELoss).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        epochs (int): Number of training epochs. Default is 12.
        device (str): Device to train the model on ('cuda' or 'cpu'). Default to 'cuda'.
        mode (str): Mode of operation, either 'classification' or 'regression'. Default to 'classification'.
        start_save_epoch (int): Epoch offset used when saving checkpoints. Default is 0.
        save_path (str): Path to save model checkpoints. If empty, checkpoints are not saved.
        save_interval (int): Interval (in epochs) for saving model checkpoints. Default is 10.

    Returns:
        tuple: Two lists containing per-batch training and validation losses:
            - train_logs_list (list of float): Training loss values.
            - val_logs_list (list of float): Validation loss values.
    """
    train_logs_list = []
    val_logs_list = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()

        train_losses = []
        train_true_labels, train_pred_labels = [], []

        # Training loop
        for pre_images, post_images, labels in tqdm(train_loader,
                                                    desc="Training"):
            pre_images, post_images, labels = pre_images.to(
                device), post_images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(pre_images, post_images)

            # Handle classification or regression mode
            if mode == 'regression':
                labels = labels.float()
                loss = loss_fn(outputs.squeeze(), labels)
                preds = outputs.squeeze().round().clamp(0, 3).long()
            else:  # classification
                loss = loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            # Logging results
            train_losses.append(loss.item())
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(preds.cpu().numpy())

        # Aggregate training metrics
        train_loss = np.mean(train_losses)
        train_precision, train_recall, train_f1, train_accuracy = calculate_metrics(
            train_true_labels, train_pred_labels)

        # Validation loop
        model.eval()
        val_losses, val_true_labels, val_pred_labels = [], [], []

        with torch.no_grad():
            for pre_images, post_images, labels in tqdm(valid_loader,
                                                        desc="Validation"):
                pre_images, post_images, labels = pre_images.to(
                    device), post_images.to(device), labels.to(device)

                outputs = model(pre_images, post_images)

                if mode == 'regression':
                    labels = labels.float()
                    loss = loss_fn(outputs.squeeze(), labels)
                    preds = outputs.squeeze().round().clamp(0, 3).long()
                else:  # classification
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                val_losses.append(loss.item())
                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(preds.cpu().numpy())

        # Aggregate validation metrics
        val_loss = np.mean(val_losses)
        val_precision, val_recall, val_f1, val_accuracy = calculate_metrics(
            val_true_labels, val_pred_labels)

        train_logs_list.extend(train_losses)
        val_logs_list.extend(val_losses)

        print(
            f"Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save checkpoint
        if save_path and (epoch + 1 + start_save_epoch) % save_interval == 0:
            checkpoint = {
                "epoch": epoch + 1 + start_save_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            if lr_scheduler:
                checkpoint["scheduler_state"] = lr_scheduler.state_dict()
            torch.save(checkpoint,
                       f"{save_path}/model_SiameseResNet_{mode}_epoch{epoch + 1 + start_save_epoch}.pt")

        # Adjust learning rate if scheduler is provided
        if lr_scheduler:
            lr_scheduler.step(val_loss)

    return train_logs_list, val_logs_list


# --------- DamageNet Training ---------
def train_damagenet_model(
        model, train_loader, valid_loader, loss_fn, optimizer,
        lr_scheduler=None,
        epochs=12, device="cuda", mode='classification',
        start_save_epoch=0, save_path="", save_interval=10
):
    """
    Train a DamageNet model on a given dataset.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for the training dataset.
        valid_loader: DataLoader for the validation dataset.
        loss_fn: Loss function for model training.
        optimizer: Optimizer used for model training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        epochs: Number of training epochs. Default is 12.
        device: Device to run the training on (e.g., 'cuda', 'cpu'). 'cuda'
        mode: Mode of the task ('classification', 'regression', 'regression_sigmoid'). Default is 'classification'
        start_save_epoch: Epoch from which to start saving the model. Default is 0
        save_path: Directory to save model checkpoints.
        save_interval: Interval (in epochs) at which to save model checkpoints. Default is 10

    Returns:
        train_logs_list: List of training losses over epochs.
        val_logs_list: List of validation losses over epochs.
    """
    train_logs_list, val_logs_list = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        train_losses, train_true_labels, train_pred_labels = [], [], []

        # --- Training Loop ---
        for pre_images, post_images, mask_images, labels in tqdm(train_loader,
                                                                 desc="Training"):
            # Move inputs to the specified device
            pre_images, post_images, mask_images, labels = (
                pre_images.to(device), post_images.to(device),
                mask_images.unsqueeze(1).to(device), labels.to(device)
            )

            optimizer.zero_grad()  # Clear previous gradients

            # Forward pass through the model
            outputs = model(pre_images, post_images, mask_images)

            # Calculate loss and predictions based on mode
            if mode == 'regression':
                labels = labels.float()
                loss = loss_fn(outputs, labels)
                preds = outputs.round().clamp(0, 3).long()
            elif mode == 'regression_sigmoid':
                labels = labels.float()
                outputs = torch.sigmoid(outputs) * 3
                loss = loss_fn(outputs, labels)
                preds = outputs.round().clamp(0, 3).long()
            else:  # classification
                loss = loss_fn(outputs, labels)
                preds = outputs.argmax(1)

            loss.backward()  # Backpropagation
            optimizer.step()  # Optimizer step

            # Track training metrics
            train_losses.append(loss.item())
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(preds.cpu().numpy())

        # Compute epoch-level training metrics
        train_loss = np.mean(train_losses)
        train_precision, train_recall, train_f1, train_accuracy = calculate_metrics(
            train_true_labels, train_pred_labels)

        # --- Validation Loop ---
        model.eval()
        val_losses, val_true_labels, val_pred_labels = [], [], []

        with torch.no_grad():  # Disable gradient computation for validation
            for pre_images, post_images, mask_images, labels in tqdm(
                    valid_loader, desc="Validation"):
                pre_images, post_images, mask_images, labels = (
                    pre_images.to(device), post_images.to(device),
                    mask_images.unsqueeze(1).to(device), labels.to(device)
                )

                # Forward pass
                outputs = model(pre_images, post_images, mask_images)

                # Compute loss and predictions based on mode
                if mode == 'regression':
                    labels = labels.float()
                    loss = loss_fn(outputs, labels)
                    preds = outputs.round().clamp(0, 3).long()
                elif mode == 'regression_sigmoid':
                    labels = labels.float()
                    outputs = torch.sigmoid(outputs) * 3
                    loss = loss_fn(outputs, labels)
                    preds = outputs.round().clamp(0, 3).long()
                else:
                    loss = loss_fn(outputs, labels)
                    preds = outputs.argmax(1)

                # Store validation metrics
                val_losses.append(loss.item())
                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(preds.cpu().numpy())

        # Compute epoch-level validation metrics
        val_loss = np.mean(val_losses)
        val_precision, val_recall, val_f1, val_accuracy = calculate_metrics(
            val_true_labels, val_pred_labels)

        # Store losses for logging
        train_logs_list.extend(train_losses)
        val_logs_list.extend(val_losses)

        # Print epoch summary
        print(
            f"Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save model checkpoint if applicable
        if save_path and (epoch + 1 + start_save_epoch) % save_interval == 0:
            checkpoint = {
                "epoch": epoch + 1 + start_save_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            if lr_scheduler:
                checkpoint["scheduler_state"] = lr_scheduler.state_dict()
            torch.save(checkpoint,
                       f"{save_path}/model_DamageNet_{mode}_epoch{epoch + 1 + start_save_epoch}.pt")

        # Step the learning rate scheduler if provided
        if lr_scheduler:
            lr_scheduler.step(val_loss)

    return train_logs_list, val_logs_list
