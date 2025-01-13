from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score
import numpy as np
from tqdm import tqdm
import torch
import csv
import os
from .utils import label_to_rgb


def calculate_metrics(true_mask, pred_mask, class_idx=1):
    """
    Calculate evaluation metrics for a specific class in binary segmentation tasks.

    This function computes precision, recall, F1-score, and Intersection over Union (IoU)
    for the specified class by comparing the true and predicted segmentation masks.

    Args:
        true_mask (numpy.ndarray): Ground truth binary mask with values representing classes.
        pred_mask (numpy.ndarray): Predicted binary mask with values representing classes.
        class_idx (int, optional): Index of the class to evaluate. Default is 1.

    Returns:
        tuple: A tuple containing:
            - cm (numpy.ndarray): Confusion matrix for the evaluated class.
            - precision (float): Precision metric for the class.
            - recall (float): Recall metric for the class.
            - f1 (float): F1-score for the class.
            - iou (float): Intersection over Union (IoU) for the class.
    """
    # Flatten the masks to 1D arrays for evaluation
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()

    # Calculate precision, recall, F1-score and confusion matrix
    precision = precision_score(true_flat, pred_flat, pos_label=class_idx,
                                zero_division=0)
    recall = recall_score(true_flat, pred_flat, pos_label=class_idx,
                          zero_division=0)
    f1 = f1_score(true_flat, pred_flat, pos_label=class_idx, zero_division=0)
    cm = confusion_matrix(true_flat, pred_flat, labels=[0, class_idx])

    # Calculate Intersection over Union
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    iou = tp / (tp + fp + fn + 1e-7)

    return cm, precision, recall, f1, iou


def evaluate_model(model, dataloader, loss_fn, device, class_rgb_values,
                   num_pred=10):
    """
    Evaluate a segmentation model on a given dataset and compute performance metrics.

    Args:
        model (torch.nn.Module): The segmentation model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of images and masks.
        loss_fn (callable): Loss function used for evaluation (e.g., CrossEntropyLoss).
        device (torch.device): Device to run the evaluation on (e.g., 'cpu' or 'cuda').
        class_rgb_values (list): List of RGB values corresponding to each class label.
        num_pred (int): Number of predictions to store.

    Returns:
        tuple: A tuple containing:
            - predictions (list): List of tuples `(image, true_mask_rgb, pred_mask_rgb)`.
            - epoch_loss (float): Average loss over the dataset.
            - metrics (dict): Dictionary with aggregated evaluation metrics:
                - "confusion_matrix" (numpy.ndarray): 2x2 confusion matrix.
                - "precision" (float): Average precision across batches.
                - "recall" (float): Average recall across batches.
                - "f1_score" (float): Average F1-score across batches.
                - "iou" (float): Average Intersection over Union (IoU) across batches.
    """
    model.eval()
    predictions = []
    running_loss = 0.0
    total_cm = np.zeros((2, 2))
    total_precision = []
    total_recall = []
    total_f1 = []
    total_iou = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            pred_masks = outputs.argmax(dim=1).cpu().numpy()
            true_masks = masks.argmax(dim=1).cpu().numpy()

            loss = loss_fn(outputs, masks)
            running_loss += loss.item()

            # Compute evaluation metrics for the batch
            running_cm, running_precision, running_recall, running_f1, running_iou = calculate_metrics(
                true_masks, pred_masks)
            total_cm += running_cm
            total_precision.append(running_precision)
            total_recall.append(running_recall)
            total_f1.append(running_f1)
            total_iou.append(running_iou)

            # Store sample predictions
            if len(predictions) < num_pred:
                mask_rgb = label_to_rgb(masks.squeeze(0).cpu().numpy(),
                                        class_rgb_values)
                pred_rgb = label_to_rgb(outputs.squeeze(0).cpu().numpy(),
                                        class_rgb_values)
                predictions.append((images.squeeze(0).cpu().permute(1, 2,
                                                                    0).numpy() / 256,
                                    mask_rgb, pred_rgb))

    # Compute average loss and metrics
    epoch_loss = running_loss / len(dataloader)
    avg_precision = np.mean(total_precision)
    avg_recall = np.mean(total_recall)
    avg_f1 = np.mean(total_f1)
    avg_iou = np.mean(total_iou)

    return predictions, epoch_loss, {
        "confusion_matrix": total_cm,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1,
        "iou": avg_iou
    }


def update_experiment_results_csv(
        csv_path, model_name, learning_rate, scheduler, optimizer,
        num_epochs, test_loss, test_metrics
):
    """
    Append experiment results to a CSV file. If the file doesn't exist, create it and add a header.

    Args:
        csv_path (str): Path to the CSV file where results will be stored.
        model_name (str): Name of the model used in the experiment.
        learning_rate (float): Learning rate used during training.
        scheduler (str): Name of the learning rate scheduler used.
        optimizer (str): Name of the optimizer used.
        num_epochs (int): Number of epochs the model was trained for.
        test_loss (float): Loss value computed on the test set.
        test_metrics (dict): Dictionary of test performance metrics containing:
            - "precision" (float): Precision value.
            - "recall" (float): Recall value.
            - "f1_score" (float): F1-score value.
            - "iou" (float): Intersection over Union (IoU) value.

    Returns:
        None
    """
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_path)

    # Define the header for the CSV file
    header = [
        "Model", "Learning Rate", "Scheduler", "Optimizer",
        "Num Epochs", "Test Loss", "Precision", "Recall", "F1 Score", "IoU"
    ]

    # Create a row with the experiment results
    row = [
        model_name, learning_rate, scheduler, optimizer,
        num_epochs, test_loss,
        test_metrics["precision"], test_metrics["recall"],
        test_metrics["f1_score"], test_metrics["iou"]
    ]

    # Open the CSV file in append mode and write the data
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:  # Add a header if the file doesn't exist
            writer.writerow(header)
        writer.writerow(row)
