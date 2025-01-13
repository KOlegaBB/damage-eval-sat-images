import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from .evaluation import calculate_metrics

def train_model(model, train_loader, valid_loader, loss_fn, optimizer, lr_scheduler, epochs=12, device="cuda", exp_num=1):
    """
    Trains a model for a specified number of epochs and evaluates it on a validation set.
    This function also logs various metrics such as loss, precision, recall, F1 score, and IoU to TensorBoard.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        loss_fn (torch.nn.Module): The loss function to compute the loss during training and validation.
        optimizer (torch.optim.Optimizer): The optimizer to update the model's weights.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler to adjust the learning rate during training.
        epochs (int, optional): The number of epochs to train the model. Default is 12.
        device (str, optional): The device to run the training on ('cuda' or 'cpu'). Default is 'cuda'.
        exp_num (int, optional): The experiment number used to organize TensorBoard logs. Default is 1.

    Returns:
        tuple: A tuple containing two lists:
            - train_logs_list: A list of training losses for each batch across all epochs.
            - val_logs_list: A list of validation losses for each batch across all epochs.

    Logs:
        - Logs scalar values for training and validation loss, precision, recall, F1 score, and IoU at each epoch.
        - Saves training and validation losses to TensorBoard using the SummaryWriter.
    """
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/experiment_{exp_num}")

    train_logs_list = []
    val_logs_list = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()

        train_losses = []
        train_cm = np.zeros((2, 2))  # For confusion matrix
        train_precisions, train_recalls, train_f1s, train_ious = [], [], [], []
        for step, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            writer.add_scalar(f"ItemLoss/Train", loss.item(), epoch * len(train_loader) + step)

            pred_masks = outputs.argmax(dim=1).cpu().numpy()
            true_masks = masks.argmax(dim=1).cpu().numpy()

            # Calculate and accumulate metrics
            cm, precision, recall, f1, iou = calculate_metrics(true_masks, pred_masks)
            train_cm += cm
            train_precisions.append(precision)
            train_recalls.append(recall)
            train_f1s.append(f1)
            train_ious.append(iou)

        # Calculate average training metrics for the epoch
        train_loss = np.mean(train_losses)
        train_precision, train_recall = np.mean(train_precisions), np.mean(train_recalls)
        train_f1, train_iou = np.mean(train_f1s), np.mean(train_ious)

        # Log training metrics to TensorBoard
        writer.add_scalar(f"TrainPrecision/Epoch", train_precision, epoch)
        writer.add_scalar(f"TrainRecall/Epoch", train_recall, epoch)
        writer.add_scalar(f"TrainF1/Epoch", train_f1, epoch)
        writer.add_scalar(f"TrainIoU/Epoch", train_iou, epoch)

        # Validation
        model.eval()
        val_losses, val_precisions, val_recalls, val_f1s, val_ious = [], [], [], [], []
        val_cm = np.zeros((2, 2))  # For confusion matrix
        with torch.no_grad():
            for step, (images, masks) in enumerate(tqdm(valid_loader)):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)

                val_losses.append(loss.item())
                writer.add_scalar(f"ItemLoss/Val", loss.item(), len(valid_loader) * epoch + step)

                pred_masks = outputs.argmax(dim=1).cpu().numpy()
                true_masks = masks.argmax(dim=1).cpu().numpy()

                cm, precision, recall, f1, iou = calculate_metrics(true_masks, pred_masks)
                val_cm += cm
                val_precisions.append(precision)
                val_recalls.append(recall)
                val_f1s.append(f1)
                val_ious.append(iou)

        # Calculate average validation metrics for the epoch
        val_loss = np.mean(val_losses)
        val_precision, val_recall = np.mean(val_precisions), np.mean(val_recalls)
        val_f1, val_iou = np.mean(val_f1s), np.mean(val_ious)

        # Store losses
        train_logs_list.extend(train_losses)
        val_logs_list.extend(val_losses)

        # Print metrics for this epoch
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, IoU: {val_iou:.4f}")

        lr_scheduler.step()

    # Close TensorBoard writer
    writer.close()

    return train_logs_list, val_logs_list