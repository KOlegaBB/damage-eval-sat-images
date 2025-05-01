from tqdm import tqdm
import cv2
import numpy as np


def iou(true_mask, pred_mask):
    """
    Calculate Intersection over Union (IoU) between two binary masks.

    Args:
        true_mask (numpy.ndarray): True binary mask.
        pred_mask (numpy.ndarray): Predicted binary mask.

    Returns:
        float: IoU value between 0 and 1.
    """
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    return np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0.0


def semantic_to_instance_mask(mask_rgb, target_rgb=(255, 255, 255)):
    """
    Convert an RGB mask to an instance segmentation mask.

    Args:
        mask_rgb (numpy.ndarray): RGB mask (H, W, 3).
        target_rgb (tuple): Target RGB value to isolate (e.g., (255, 255, 255)).

    Returns:
        instance_mask (numpy.ndarray): Instance segmentation mask with unique labels for each instance.
    """
    # Create a binary mask for the target class
    binary_mask = np.all(mask_rgb == np.array(target_rgb), axis=-1).astype(
        np.uint8)  # Shape: (H, W)

    # Initialize the instance mask
    instance_mask = np.zeros_like(binary_mask, dtype=np.int32)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Assign unique IDs to each contour
    for instance_id, contour in enumerate(contours, start=1):
        cv2.drawContours(instance_mask, [contour], -1, instance_id,
                         thickness=cv2.FILLED)

    return instance_mask, contours


def get_contour_bounding_boxes(contours):
    """
    Get bounding boxes for a list of contours.

    Args:
        contours (list): List of contours from cv2.findContours.

    Returns:
        list: List of bounding boxes [(x, y, w, h)].
    """
    return [cv2.boundingRect(contour) for contour in contours]


def filter_contours_by_bbox(true_contours, pred_contours, true_bboxes,
                            pred_bboxes):
    """
    Filter contours by checking bounding box intersections.

    Args:
        true_contours (list): True contours.
        pred_contours (list): Predicted contours.
        true_bboxes (list): Bounding boxes for true contours.
        pred_bboxes (list): Bounding boxes for predicted contours.

    Returns:
        list: Filtered true and predicted contour pairs.
    """
    filtered_pairs = []
    for i, true_bbox in enumerate(true_bboxes):
        filtered_buildings = []
        for j, pred_bbox in enumerate(pred_bboxes):
            # Check if bounding boxes overlap
            if (
                    true_bbox[0] < pred_bbox[0] + pred_bbox[2] and
                    true_bbox[0] + true_bbox[2] > pred_bbox[0] and
                    true_bbox[1] < pred_bbox[1] + pred_bbox[3] and
                    true_bbox[1] + true_bbox[3] > pred_bbox[1]
            ):
                filtered_buildings.append((true_contours[i], pred_contours[j]))
        filtered_pairs.append(filtered_buildings)
    return filtered_pairs


def instance_post_processing_statistics(true_mask_rgb, pred_mask_rgb,
                                      target_rgb=(255, 255, 255),
                                      iou_threshold=0.5):
    """
    Compute instance-level evaluation statistics for building segmentation masks.

    This function evaluates the performance of an instance segmentation prediction by:
    1. Extracting building instances from both ground truth and predicted RGB masks.
    2. Matching predicted and ground truth instances using bounding box filtering.
    3. Calculating the Intersection-over-Union (IoU) for matched instances.
    4. Reporting statistics such as total targets, predictions, successful detections,
      mean IoU, and per-instance IoUs.

    Parameters:
       true_mask_rgb (np.ndarray): Ground truth semantic segmentation mask in RGB format.
       pred_mask_rgb (np.ndarray): Predicted semantic segmentation mask in RGB format.
       target_rgb (tuple): RGB value representing the target class (default is white `(255, 255, 255)`).
       iou_threshold (float): IoU threshold for considering a predicted instance as a successful match (default: 0.5).

    Returns:
       dict: A dictionary with the following metrics:
           - 'total_target_buildings' (int): Number of building instances in the ground truth.
           - 'total_predicted_buildings' (int): Number of building instances in the prediction.
           - 'total_successful_buildings' (int): Number of predicted buildings matched successfully with ground truth (IoU ≥ threshold).
           - 'mean_iou' (float): Average IoU across all matched building instances.
           - 'iou_per_building' (List[float]): List of maximum IoU values for each predicted building instance.
    """
    # Step 1: Convert semantic masks to instance masks and get contours
    true_mask, true_contours = semantic_to_instance_mask(true_mask_rgb,
                                                         target_rgb=target_rgb)
    pred_mask, pred_contours = semantic_to_instance_mask(pred_mask_rgb,
                                                         target_rgb=target_rgb)

    # Step 2: Get bounding boxes for contours
    true_bboxes = get_contour_bounding_boxes(true_contours)
    pred_bboxes = get_contour_bounding_boxes(pred_contours)

    # Step 3: Count true and predicted buildings
    target_buildings_true = len(true_contours)
    predicted_buildings_pred = len(pred_contours)

    # Initialize variables to track metrics
    successful_buildings_pred = 0
    iou_per_building_pred = []

    # Step 4: Calculate IoU for filtered pairs of true and predicted buildings
    filtered_pairs_true_pred = filter_contours_by_bbox(true_contours,
                                                       pred_contours,
                                                       true_bboxes,
                                                       pred_bboxes)
    for filtered_buildings in filtered_pairs_true_pred:
        ious = []
        for true_contour, pred_contour in filtered_buildings:
            # Create masks for the filtered contours
            true_building_mask = np.zeros_like(true_mask, dtype=np.uint8)
            pred_building_mask = np.zeros_like(pred_mask, dtype=np.uint8)

            cv2.drawContours(true_building_mask, [true_contour], -1, 255,
                             thickness=cv2.FILLED)
            cv2.drawContours(pred_building_mask, [pred_contour], -1, 255,
                             thickness=cv2.FILLED)

            # Calculate IoU
            iou_value = iou(true_building_mask, pred_building_mask)
            ious.append(iou_value)

        if ious:
            max_iou = max(ious)
            iou_per_building_pred.append(max_iou)
            if max_iou >= iou_threshold:
                successful_buildings_pred += 1

    # Step 5: Aggregate metrics
    total_target_buildings_before = target_buildings_true
    total_predicted_buildings_before = predicted_buildings_pred
    total_successful_buildings_before = successful_buildings_pred
    total_iou_sum_before = sum(iou_per_building_pred)
    total_iou_count_before = len(iou_per_building_pred)

    # Returning metrics
    return {
        'total_target_buildings': total_target_buildings_before,
        'total_predicted_buildings': total_predicted_buildings_before,
        'total_successful_buildings': total_successful_buildings_before,
        'mean_iou': total_iou_sum_before / total_iou_count_before if total_iou_count_before > 0 else 0,
        'iou_per_building': iou_per_building_pred
    }
