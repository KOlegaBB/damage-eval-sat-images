from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import torch
from .utils import label_to_rgb

def calculate_metrics(true_mask, pred_mask, class_idx=1):
    """
    Calculate precision, recall, F1-score, and IoU for the specified class.
    """
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()


    precision = precision_score(true_flat, pred_flat, pos_label=class_idx, zero_division=0)
    recall = recall_score(true_flat, pred_flat, pos_label=class_idx, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, pos_label=class_idx, zero_division=0)

    cm = confusion_matrix(true_flat, pred_flat, labels=[0, class_idx])
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    iou = tp / (tp + fp + fn + 1e-7)

    return cm, precision, recall, f1, iou

def evaluate_model(model, dataloader, loss_fn, device, class_rgb_values, num_pred=10):
    model.eval()
    predictions = []
    running_loss = 0.0
    total_cm = np.zeros((2, 2))  # For confusion matrix
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
            running_cm, running_precision, running_recall, running_f1, running_iou = calculate_metrics(true_masks, pred_masks)
            total_cm += running_cm
            total_precision.append(running_precision)
            total_recall.append(running_recall)
            total_f1.append(running_f1)
            total_iou.append(running_iou)

            if len(predictions) < num_pred:
                mask_rgb = label_to_rgb(masks.squeeze(0).cpu().numpy(), class_rgb_values)
                pred_rgb = label_to_rgb(outputs.squeeze(0).cpu().numpy(), class_rgb_values)
                predictions.append((images.squeeze(0).cpu().permute(1, 2, 0).numpy() / 256, mask_rgb, pred_rgb))
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
