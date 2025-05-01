from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score
import numpy as np
from tqdm import tqdm
import torch
import csv
import os
import torch.nn.functional as F

from .instance_segm import instance_post_processing_statistics
from .regularize import regularize
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


def evaluate_model_with_regularization(model, encoder, generator, dataloader,
                                       device, class_rgb_values, num_pred=10,
                                       iou_threshold=0.5):
    """
    Evaluate a segmentation model before and after applying regularization,
    computing both pixel-level and instance-level metrics.

    Args:
        model (torch.nn.Module): The main segmentation model to be evaluated.
        encoder (torch.nn.Module): Encoder model used for feature extraction in regularization.
        generator (torch.nn.Module): Generator model used in regularization to refine outputs.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of images and masks.
        device (torch.device): Device to perform computation on (e.g., 'cpu' or 'cuda').
        class_rgb_values (list): List of RGB values corresponding to each class label.
        num_pred (int): Number of sample predictions to store for visualization.
        iou_threshold (float): IoU threshold for considering instance matches.

    Returns:
        tuple: A tuple containing:
            - predictions (list): List of tuples
              (image, true_mask_rgb, pred_mask_rgb, reg_pred_mask_rgb).
            - metrics (dict): Dictionary containing evaluation metrics both
              before and after regularization:
                - "confusion_matrix_before", "confusion_matrix_after"
                - "precision_before", "recall_before", "f1_score_before", "iou_before"
                - "precision_after", "recall_after", "f1_score_after", "iou_after"
                - "average_iou_per_building_before", "average_iou_per_building_after"
                - "total_target_buildings_before", "total_predicted_buildings_before", "total_successful_buildings_before"
                - "total_target_buildings_after", "total_predicted_buildings_after", "total_successful_buildings_after"
    """
    model.eval()
    encoder.eval()
    generator.eval()
    predictions = []

    # Initialize pixel-wise metrics
    total_cm_before = np.zeros((2, 2))
    total_cm_after = np.zeros((2, 2))
    total_precision_before, total_recall_before, total_f1_before, total_iou_before = [], [], [], []
    total_precision_after, total_recall_after, total_f1_after, total_iou_after = [], [], [], []

    # Initialize instance-level metrics
    total_target_buildings_before = 0
    total_predicted_buildings_before = 0
    total_successful_buildings_before = 0
    total_iou_sum_before = 0
    total_iou_count_before = 0

    total_target_buildings_after = 0
    total_predicted_buildings_after = 0
    total_successful_buildings_after = 0
    total_iou_sum_after = 0
    total_iou_count_after = 0

    i = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # Model forward pass
            outputs = model(images)
            reg_outputs = regularize(images, outputs, encoder, generator)

            # Get predictions before and after regularization
            pred_masks = outputs.argmax(dim=1).cpu().numpy()
            reg_pred_masks = F.softmax(reg_outputs, dim=1).argmax(
                dim=1).cpu().numpy()
            true_masks = masks.argmax(dim=1).cpu().numpy()

            # Compute metrics before regularization
            cm_before, precision_before, recall_before, f1_before, iou_before = calculate_metrics(
                true_masks, pred_masks)
            total_cm_before += cm_before
            total_precision_before.append(precision_before)
            total_recall_before.append(recall_before)
            total_f1_before.append(f1_before)
            total_iou_before.append(iou_before)

            # Compute metrics after regularization
            cm_after, precision_after, recall_after, f1_after, iou_after = calculate_metrics(
                true_masks, reg_pred_masks)
            total_cm_after += cm_after
            total_precision_after.append(precision_after)
            total_recall_after.append(recall_after)
            total_f1_after.append(f1_after)
            total_iou_after.append(iou_after)

            # Convert predictions to RGB for instance-level analysis
            true_mask_rgb = label_to_rgb(masks.squeeze(0).cpu().numpy(),
                                         class_rgb_values)
            pred_mask_rgb = label_to_rgb(outputs.squeeze(0).cpu().numpy(),
                                         class_rgb_values)
            reg_pred_mask_rgb = label_to_rgb(
                reg_outputs.squeeze(0).cpu().numpy(), class_rgb_values)

            # Instance-level statistics before regularization
            inst_stats_before = instance_post_processing_statistics(
                true_mask_rgb, pred_mask_rgb, target_rgb=(255, 255, 255),
                iou_threshold=iou_threshold)

            total_target_buildings_before += inst_stats_before[
                'total_target_buildings']
            total_predicted_buildings_before += inst_stats_before[
                'total_predicted_buildings']
            total_successful_buildings_before += inst_stats_before[
                'total_successful_buildings']
            total_iou_sum_before += sum(inst_stats_before['iou_per_building'])
            total_iou_count_before += len(
                inst_stats_before['iou_per_building'])

            # Instance-level statistics after regularization
            inst_stats_after = instance_post_processing_statistics(
                true_mask_rgb, reg_pred_mask_rgb, target_rgb=(255, 255, 255),
                iou_threshold=iou_threshold)

            total_target_buildings_after += inst_stats_after[
                'total_target_buildings']
            total_predicted_buildings_after += inst_stats_after[
                'total_predicted_buildings']
            total_successful_buildings_after += inst_stats_after[
                'total_successful_buildings']
            total_iou_sum_after += sum(inst_stats_after['iou_per_building'])
            total_iou_count_after += len(inst_stats_after['iou_per_building'])

            # Store predictions periodically
            if len(predictions) < num_pred and i % 10 == 0:
                predictions.append(
                    (images.squeeze(0).cpu().permute(1, 2, 0).numpy() / 256,
                     true_mask_rgb, pred_mask_rgb, reg_pred_mask_rgb)
                )
            i += 1

    # Aggregate average metrics before and after regularization
    avg_precision_before = np.mean(total_precision_before)
    avg_recall_before = np.mean(total_recall_before)
    avg_f1_before = np.mean(total_f1_before)
    avg_iou_before = np.mean(total_iou_before)

    avg_precision_after = np.mean(total_precision_after)
    avg_recall_after = np.mean(total_recall_after)
    avg_f1_after = np.mean(total_f1_after)
    avg_iou_after = np.mean(total_iou_after)

    avg_iou_per_building_before = total_iou_sum_before / total_iou_count_before if total_iou_count_before > 0 else 0
    avg_iou_per_building_after = total_iou_sum_after / total_iou_count_after if total_iou_count_after > 0 else 0

    return predictions, {
        "confusion_matrix_before": total_cm_before,
        "confusion_matrix_after": total_cm_after,
        "precision_before": avg_precision_before,
        "recall_before": avg_recall_before,
        "f1_score_before": avg_f1_before,
        "iou_before": avg_iou_before,
        "precision_after": avg_precision_after,
        "recall_after": avg_recall_after,
        "f1_score_after": avg_f1_after,
        "iou_after": avg_iou_after,
        "average_iou_per_building_before": avg_iou_per_building_before,
        "average_iou_per_building_after": avg_iou_per_building_after,
        "total_target_buildings_before": total_target_buildings_before,
        "total_predicted_buildings_before": total_predicted_buildings_before,
        "total_successful_buildings_before": total_successful_buildings_before,
        "total_target_buildings_after": total_target_buildings_after,
        "total_predicted_buildings_after": total_predicted_buildings_after,
        "total_successful_buildings_after": total_successful_buildings_after,
    }


def print_regularization_metrics(metrics):
    """
    Print a summary of segmentation performance metrics before and after a regularization step.

    This function reports standard classification metrics (precision, recall, F1 score, IoU)
    and instance-level metrics (e.g., number of target/predicted/successfully matched buildings,
    success rate, and error rate) both before and after a regularization step in a segmentation pipeline.

    Parameters:
        metrics (dict): A dictionary containing various evaluation metrics with the following keys:
            - 'confusion_matrix_before', 'confusion_matrix_after' (np.ndarray or str): Confusion matrices.
            - 'precision_before', 'precision_after' (float): Precision scores.
            - 'recall_before', 'recall_after' (float): Recall scores.
            - 'f1_score_before', 'f1_score_after' (float): F1 scores.
            - 'iou_before', 'iou_after' (float): Intersection-over-Union scores.
            - 'average_iou_per_building_before', 'average_iou_per_building_after' (float): Average IoU per instance.
            - 'total_target_buildings_before', 'total_target_buildings_after' (int): Ground truth building counts.
            - 'total_predicted_buildings_before', 'total_predicted_buildings_after' (int): Predicted building counts.
            - 'total_successful_buildings_before', 'total_successful_buildings_after' (int): Correctly predicted buildings.

    Prints:
        A detailed breakdown of metrics before and after regularization, including confusion matrix,
        precision, recall, F1, IoU, average IoU per building, total building counts,
        and success/error rates.
    """
    # Compute success and error rates
    success_rate_before = metrics['total_successful_buildings_before'] / \
                          metrics['total_target_buildings_before'] * 100 if \
        metrics['total_target_buildings_before'] > 0 else 0
    error_rate_before = (metrics['total_predicted_buildings_before'] - metrics[
        'total_successful_buildings_before']) / metrics[
                            'total_target_buildings_before'] * 100 if metrics[
                                                                          'total_target_buildings_before'] > 0 else 0

    success_rate_after = metrics['total_successful_buildings_after'] / metrics[
        'total_target_buildings_after'] * 100 if metrics[
                                                     'total_target_buildings_after'] > 0 else 0
    error_rate_after = (metrics['total_predicted_buildings_after'] - metrics[
        'total_successful_buildings_after']) / metrics[
                           'total_target_buildings_after'] * 100 if metrics[
                                                                        'total_target_buildings_after'] > 0 else 0

    print("=== Before Regularization ===")
    print(f"Confusion matrix:\n{metrics['confusion_matrix_before']}")
    print(f"Precision: {metrics['precision_before']:.4f}")
    print(f"Recall: {metrics['recall_before']:.4f}")
    print(f"F1 score: {metrics['f1_score_before']:.4f}")
    print(f"IOU: {metrics['iou_before']:.4f}")
    print()

    print("=== After Regularization ===")
    print(f"Confusion matrix:\n{metrics['confusion_matrix_after']}")
    print(f"Precision: {metrics['precision_after']:.4f}")
    print(f"Recall: {metrics['recall_after']:.4f}")
    print(f"F1 score: {metrics['f1_score_after']:.4f}")
    print(f"IOU: {metrics['iou_after']:.4f}")
    print()

    print("=== Instance-Level Metrics ===\n")

    print("=== Before Regularization ===")
    print(
        f"Average IoU per building: {metrics['average_iou_per_building_before']:.4f}")
    print(
        f"Total target buildings: {metrics['total_target_buildings_before']}")
    print(
        f"Total predicted buildings: {metrics['total_predicted_buildings_before']}")
    print(
        f"Total successful buildings: {metrics['total_successful_buildings_before']}")
    print(f"Success Rate: {success_rate_before:.2f}%")
    print(f"Error Rate: {error_rate_before:.2f}%\n")

    print("=== After Regularization ===")
    print(
        f"Average IoU per building: {metrics['average_iou_per_building_after']:.4f}")
    print(f"Total target buildings: {metrics['total_target_buildings_after']}")
    print(
        f"Total predicted buildings: {metrics['total_predicted_buildings_after']}")
    print(
        f"Total successful buildings: {metrics['total_successful_buildings_after']}")
    print(f"Success Rate: {success_rate_after:.2f}%")
    print(f"Error Rate: {error_rate_after:.2f}%")


def update_experiment_results_csv(
        csv_path, model_name, learning_rate, scheduler, optimizer,
        num_epochs, test_loss, test_metrics, building_metrics
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
        building_metrics (dict): Dictionary of building segmentation metrics containing:
            - "total_target_buildings" (int): Total number of target buildings.
            - "total_predicted_buildings" (int): Total number of predicted buildings.
            - "total_successfully_identified_buildings" (int): Total successfully identified buildings.
            - "average_iou_per_building" (float): Average IoU per building.
            - "overall_success_rate" (float): Overall success rate (%).
            - "overall_error_rate" (float): Overall error rate (%).

    Returns:
        None
    """
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_path)

    # Define the header for the CSV file
    header = [
        "Model", "Learning Rate", "Scheduler", "Optimizer",
        "Num Epochs", "Test Loss", "Precision", "Recall", "F1 Score", "IoU",
        "Total Target Buildings", "Total Predicted Buildings",
        "Total Successfully Identified Buildings", "Average IoU Per Building",
        "Overall Success Building Rate (%)", "Overall Error Building Rate (%)"
    ]

    # Create a row with the experiment results
    row = [
        model_name, learning_rate, scheduler, optimizer,
        num_epochs, test_loss,
        test_metrics["precision"], test_metrics["recall"],
        test_metrics["f1_score"], test_metrics["iou"],
        building_metrics["total_target_buildings"],
        building_metrics["total_predicted_buildings"],
        building_metrics["total_successful_buildings"],
        building_metrics["average_iou_per_building"],
        building_metrics["overall_success_rate"],
        building_metrics["overall_error_rate"]
    ]

    # Open the CSV file in append mode and write the data
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:  # Add a header if the file doesn't exist
            writer.writerow(header)
        writer.writerow(row)
